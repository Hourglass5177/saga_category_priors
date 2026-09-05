from __future__ import annotations

"""Fused/reference all-alpha backends and content-addressed sparse cache."""

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import RefinementConfig
from .evidence import AlphaMass, alpha_mass_from_gradients, build_alpha_objective, iter_three_channel_masks
from .runtime_io import json_atomic


FORMULA_VERSION = "normalized-alpha-t-prev-v1"
CACHE_SCHEMA = "saga-alpha-evidence-v1"
KERNEL_VERSION = "alpha-mass-fused-v3-sparse-double"


def _update_array(digest: Any, value: Any) -> None:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu().numpy()
    except ImportError:
        pass
    array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def array_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_array(digest, value)
    return digest.hexdigest()


def mask_sha256(mask: Any) -> str:
    array = np.asarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(array.reshape(-1), bitorder="little").tobytes())
    return digest.hexdigest()


def gaussian_render_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("xyz", model.get_xyz), ("scaling", model.get_scaling),
        ("rotation", model.get_rotation), ("opacity", model.get_opacity),
    ):
        digest.update(name.encode())
        _update_array(digest, value)
    return digest.hexdigest()


def camera_render_sha256(camera: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(camera.image_height)).encode())
    digest.update(str(int(camera.image_width)).encode())
    digest.update(repr(float(camera.FoVx)).encode())
    digest.update(repr(float(camera.FoVy)).encode())
    for value in (camera.world_view_transform, camera.full_proj_transform, camera.camera_center):
        _update_array(digest, value)
    return digest.hexdigest()


def pack_mask_bits(masks: Sequence[np.ndarray]) -> np.ndarray:
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 3 or len(array) > 32:
        raise ValueError("one fused mask bitset requires MxHxW with M<=32")
    result = np.zeros(array.shape[1:], dtype=np.uint32)
    for index, mask in enumerate(array):
        result |= mask.astype(np.uint32) << np.uint32(index)
    return result.view(np.int32)


def expected_fused_package() -> Path:
    configured = os.environ.get("SAGA_ALPHA_MASS_PACKAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2] / "submodules"
        / "diff-gaussian-rasterization-alpha-mass"
        / "diff_gaussian_rasterization_alpha_mass"
    ).resolve()


def validate_fused_extension() -> Any:
    import diff_gaussian_rasterization_alpha_mass as module
    package = expected_fused_package()
    if Path(module.__file__).resolve() != (package / "__init__.py").resolve():
        raise RuntimeError("wrong fused alpha-mass Python package imported")
    if Path(module._C.__file__).resolve().parent != package:
        raise RuntimeError("wrong fused alpha-mass CUDA binary imported")
    if module.KERNEL_VERSION != KERNEL_VERSION:
        raise RuntimeError("fused alpha-mass Python/kernel version mismatch")
    return module


def render_fused(
    camera: Any, model: Any, pipeline: Any, masks: Sequence[np.ndarray],
    *, config: RefinementConfig,
) -> AlphaMass:
    import torch
    module = validate_fused_extension()
    mask_array = np.asarray(masks, dtype=bool)
    shape = (int(camera.image_height), int(camera.image_width))
    if mask_array.ndim != 3 or mask_array.shape[1:] != shape:
        raise ValueError("alpha masks must share camera image shape")
    if not len(mask_array):
        return AlphaMass(np.zeros((0, len(model.get_xyz))), np.zeros(len(model.get_xyz)), 0)
    if len(mask_array) > 32:
        raise ValueError("render_fused accepts at most 32 masks per pass")
    if pipeline.compute_cov3D_python:
        covariance = model.get_covariance(1.0)
        scales = torch.empty(0, device="cuda")
        rotations = torch.empty(0, device="cuda")
    else:
        covariance = torch.empty(0, device="cuda")
        scales = model.get_scaling
        rotations = model.get_rotation
    packed = torch.as_tensor(pack_mask_bits(mask_array), dtype=torch.int32, device="cuda")
    visible, inside, valid = module.accumulate_alpha_mass(
        model.get_xyz, model.get_opacity, scales, rotations, 1.0, covariance,
        camera.world_view_transform, camera.full_proj_transform,
        math.tan(float(camera.FoVx) * 0.5), math.tan(float(camera.FoVy) * 0.5),
        int(camera.image_height), int(camera.image_width), camera.camera_center,
        packed, len(mask_array), float(config.alpha_opacity_min), False, bool(pipeline.debug),
    )
    return AlphaMass(
        inside.detach().cpu().numpy().astype(np.float64),
        visible.detach().cpu().numpy().astype(np.float64), int(valid.item()),
    )


def render_gradient_reference(
    camera: Any, model: Any, pipeline: Any, background: Any,
    masks: Sequence[np.ndarray], *, config: RefinementConfig,
) -> AlphaMass:
    """One forward graph per camera; gradients are replayed for mask batches."""
    import torch
    from gaussian_renderer import render_mask
    mask_array = np.asarray(masks, dtype=np.float32)
    shape = (int(camera.image_height), int(camera.image_width))
    if mask_array.ndim != 3 or mask_array.shape[1:] != shape:
        raise ValueError("alpha masks must share camera image shape")
    point_count = int(model.get_xyz.shape[0])
    probe = torch.ones((point_count, 3), dtype=model.get_xyz.dtype, device=model.get_xyz.device, requires_grad=True)
    image = render_mask(camera, model, pipeline, torch.zeros_like(background), precomputed_mask=probe)["mask"]
    opacity = image.detach()[0].float().cpu().numpy()
    batches = iter_three_channel_masks(mask_array)
    base = build_alpha_objective((), np.zeros((3, *shape)), opacity, min_opacity=config.alpha_opacity_min)
    visible_coeff = torch.as_tensor(base.visible_coefficient, dtype=image.dtype, device=image.device)
    visible_gradient = torch.autograd.grad(torch.sum(image[0] * visible_coeff), probe, retain_graph=bool(batches))[0].detach().cpu().numpy()
    inside_batches = []
    for batch_index, (indices, targets) in enumerate(batches):
        objective = build_alpha_objective(indices, targets, opacity, min_opacity=config.alpha_opacity_min)
        coefficients = torch.as_tensor(objective.inside_coefficients, dtype=image.dtype, device=image.device)
        gradient = torch.autograd.grad(
            torch.sum(image * coefficients), probe,
            retain_graph=batch_index + 1 < len(batches),
        )[0].detach().cpu().numpy()
        inside_batches.append((indices, gradient))
    return alpha_mass_from_gradients(visible_gradient, inside_batches, len(mask_array), int(base.valid_pixels.sum()))


def render_gradient_original(
    camera: Any, model: Any, pipeline: Any, background: Any,
    masks: Sequence[np.ndarray], *, config: RefinementConfig,
) -> AlphaMass:
    """Frozen pre-optimization implementation, retained only for benchmarks."""
    import torch
    from gaussian_renderer import render_mask
    mask_array = np.asarray(masks, dtype=np.float32)
    point_count = int(model.get_xyz.shape[0])
    visible_gradient = None
    inside_batches = []
    valid_count = 0
    for batch_number, (indices, targets) in enumerate(iter_three_channel_masks(mask_array)):
        probe = torch.ones((point_count, 3), dtype=model.get_xyz.dtype, device=model.get_xyz.device, requires_grad=True)
        image = render_mask(camera, model, pipeline, torch.zeros_like(background), precomputed_mask=probe)["mask"]
        opacity = image.detach()[0].float().cpu().numpy()
        objective = build_alpha_objective(indices, targets, opacity, min_opacity=config.alpha_opacity_min)
        coefficients = torch.as_tensor(objective.inside_coefficients, dtype=image.dtype, device=image.device)
        if batch_number == 0:
            visible_coeff = torch.as_tensor(objective.visible_coefficient, dtype=image.dtype, device=image.device)
            visible_gradient = torch.autograd.grad(
                torch.sum(image[0] * visible_coeff), probe, retain_graph=bool(indices)
            )[0].detach().cpu().numpy()
            valid_count = int(objective.valid_pixels.sum())
        gradient = torch.autograd.grad(torch.sum(image * coefficients), probe)[0].detach().cpu().numpy()
        inside_batches.append((indices, gradient))
    if visible_gradient is None:
        raise RuntimeError("alpha mass rendering requires at least one mask")
    return alpha_mass_from_gradients(visible_gradient, inside_batches, len(mask_array), valid_count)


@dataclass
class AlphaCacheStats:
    camera_hits: int = 0
    mask_hits: int = 0
    kernel_calls: int = 0
    completed_cameras: int = 0


class AlphaEvidenceCache:
    def __init__(self, root: str | Path, *, mode: str, gaussian_identity: str):
        if mode not in {"readwrite", "readonly", "off"}:
            raise ValueError("invalid alpha cache mode")
        self.root = Path(root)
        self.mode = mode
        self.gaussian_identity = gaussian_identity
        self.stats = AlphaCacheStats()
        self._last_result_key: tuple[int, str, tuple[str, ...]] | None = None
        self._last_result: AlphaMass | None = None

    def _identity(self, camera: Any, config: RefinementConfig) -> Mapping[str, Any]:
        return {
            "schema": CACHE_SCHEMA,
            "formula": FORMULA_VERSION,
            "kernel": KERNEL_VERSION,
            "gaussian_render_sha256": self.gaussian_identity,
            "camera_sha256": camera_render_sha256(camera),
            "opacity_min": float(config.alpha_opacity_min),
            "alpha_skip": "1/255",
            "transmittance_stop": 0.0001,
        }

    @staticmethod
    def _load_sparse(path: Path, point_count: int, identity: str) -> np.ndarray | None:
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["identity"].item()) != identity:
                    return None
                ids = data["ids"].astype(np.int64, copy=False)
                values = data["mass"].astype(np.float64, copy=False)
            if ids.ndim != 1 or values.shape != ids.shape or len(np.unique(ids)) != len(ids):
                return None
            if np.any(ids < 0) or np.any(ids >= point_count) or not np.isfinite(values).all() or np.any(values < 0):
                return None
            dense = np.zeros(point_count, dtype=np.float64)
            dense[ids] = values
            return dense
        except (OSError, ValueError, KeyError):
            return None

    @staticmethod
    def _save_sparse(path: Path, values: np.ndarray, identity: str) -> None:
        ids = np.flatnonzero(values > 0).astype(np.int64)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.part")
        with temporary.open("wb") as handle:
            np.savez(handle, identity=np.asarray(identity), ids=ids, mass=np.asarray(values[ids], dtype=np.float64))
        os.replace(temporary, path)

    def _cached_sparse(self, path: Path, point_count: int, identity: str) -> np.ndarray | None:
        # Never retain per-mask dense arrays across cameras. A ScanNet scene can
        # contain thousands of masks over more than a million Gaussians; keeping
        # those reconstructed arrays in a process-wide dictionary defeats the
        # sparse on-disk representation and can exceed the 90 GiB cgroup.
        return self._load_sparse(path, point_count, identity)

    def get(
        self, camera_index: int, camera: Any, model: Any, pipeline: Any,
        background: Any, masks: Sequence[np.ndarray], *, backend: str,
        config: RefinementConfig,
    ) -> AlphaMass:
        point_count = int(model.get_xyz.shape[0])
        if self.mode == "off":
            self.stats.kernel_calls += 1
            return _render_backend(backend, camera, model, pipeline, background, masks, config)
        camera_root = self.root / f"camera{camera_index:05d}"
        identity_payload = self._identity(camera, config)
        identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
        mask_hashes = [mask_sha256(mask) for mask in masks]
        result_key = (camera_index, identity, tuple(mask_hashes))
        if result_key == self._last_result_key and self._last_result is not None:
            self.stats.camera_hits += 1
            self.stats.mask_hits += len(mask_hashes)
            return self._last_result
        unique_hashes = list(dict.fromkeys(mask_hashes))
        valid_pixels = -1
        visible_path = camera_root / "visible.npz"
        manifest_path = camera_root / "manifest.json"
        old_manifest: dict[str, Any] = {}
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old_manifest.get("identity") != identity:
                old_manifest = {}
        except (OSError, ValueError, TypeError):
            old_manifest = {}
        visible = self._cached_sparse(visible_path, point_count, identity)
        loaded: dict[str, np.ndarray] = {}
        for digest in unique_hashes:
            path = camera_root / "masks" / f"{digest}.npz"
            value = self._cached_sparse(path, point_count, identity + ":" + digest)
            if value is not None:
                loaded[digest] = value
                self.stats.mask_hits += 1
        if visible is not None:
            self.stats.camera_hits += 1
            valid_pixels = int(old_manifest.get("valid_pixel_count", -1))
        missing = [digest for digest in unique_hashes if digest not in loaded]
        if visible is None or missing:
            if self.mode == "readonly":
                raise RuntimeError(f"alpha cache miss in readonly mode: camera {camera_index}")
            masks_by_hash = {mask_sha256(mask): np.asarray(mask, dtype=bool) for mask in masks}
            compute_hashes = missing if visible is not None else unique_hashes
            # Current production data has <=22 masks/camera. Deterministic chunks
            # retain correctness for uncommon cameras with more than 32 masks.
            for start in range(0, len(compute_hashes), 32):
                chunk_hashes = compute_hashes[start:start + 32]
                mass = _render_backend(
                    backend, camera, model, pipeline, background,
                    [masks_by_hash[digest] for digest in chunk_hashes], config,
                )
                self.stats.kernel_calls += 1
                if visible is None:
                    visible = mass.visible_mass
                    valid_pixels = mass.valid_pixel_count
                    self._save_sparse(visible_path, visible, identity)
                elif not np.allclose(visible, mass.visible_mass, rtol=5e-5, atol=5e-5):
                    raise RuntimeError("camera visible mass changed across deterministic mask chunks")
                for index, digest in enumerate(chunk_hashes):
                    loaded[digest] = mass.inside_mass[index]
                    path = camera_root / "masks" / f"{digest}.npz"
                    self._save_sparse(path, loaded[digest], identity + ":" + digest)
            json_atomic(manifest_path, identity_payload | {
                "identity": identity,
                "mask_hashes": sorted(set(old_manifest.get("mask_hashes", ())) | set(unique_hashes)),
                "point_count": point_count,
                "valid_pixel_count": int(valid_pixels),
            })
        assert visible is not None
        inside = np.stack([loaded[digest] for digest in mask_hashes]) if mask_hashes else np.zeros((0, point_count))
        tolerance = 5e-5 * np.maximum(visible[None, :], 1.0)
        if inside.size and np.any(inside - visible[None, :] > tolerance):
            raise RuntimeError("cached inside mass exceeds visible mass")
        self.stats.completed_cameras += 1
        result = AlphaMass(inside, visible, valid_pixels)
        # Keep only the latest dense batch. Retaining every camera would defeat
        # the sparse disk cache on million-Gaussian scenes.
        self._last_result_key = result_key
        self._last_result = result
        return result


def _render_backend(backend: str, camera: Any, model: Any, pipeline: Any, background: Any, masks: Sequence[np.ndarray], config: RefinementConfig) -> AlphaMass:
    if backend == "fused":
        return render_fused(camera, model, pipeline, masks, config=config)
    if backend == "gradient-reference":
        return render_gradient_reference(camera, model, pipeline, background, masks, config=config)
    raise ValueError(f"unknown alpha backend: {backend}")


__all__ = [
    "AlphaEvidenceCache", "array_sha256", "camera_render_sha256",
    "expected_fused_package", "gaussian_render_sha256", "mask_sha256",
    "pack_mask_bits", "render_fused", "render_gradient_reference",
    "render_gradient_original",
    "validate_fused_extension",
]
