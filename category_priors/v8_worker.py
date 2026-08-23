from __future__ import annotations

"""GPU worker for V8 mask/lifting frame artifacts.

This module deliberately stops at per-frame fragments.  It never reads ground
truth, associates objects across views, or changes an existing B0/B1 output.
M1 and AM share the same attribution-to-fragment implementation in
``v8_lifting``.  AM obtains all-contributor alpha mass from gradients of the
existing differentiable mask renderer and never materialises a dense
pixel-by-Gaussian contributor cache.
"""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .v8_lifting import (
    AttributionFragment,
    AttributionMass,
    V8FragmentConfig,
    attribution_from_am_gradients,
    build_am_objective_targets,
    fragments_from_attribution,
    iter_three_channel_mask_batches,
    mass_from_max_contributor,
)


DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


@dataclass(frozen=True)
class FrameMaskPayload:
    masks: np.ndarray | None
    labels: np.ndarray
    abstained: bool

    @property
    def mask_count(self) -> int:
        return 0 if self.masks is None else int(len(self.masks))


@dataclass(frozen=True)
class FrameLiftRecord:
    frame_id: int
    image_name: str
    fragments: tuple[AttributionFragment, ...]
    labels: np.ndarray
    visible_ids: np.ndarray
    visible_mass: np.ndarray
    abstained: bool
    mask_count: int


def sparse_frame_lift_record(
    frame_id: int,
    image_name: str,
    attribution: AttributionMass,
    fragments: tuple[AttributionFragment, ...],
    labels: np.ndarray,
    *,
    retain_visibility: bool,
) -> FrameLiftRecord:
    """Discard dense per-mask attribution after extracting frame evidence."""
    if retain_visibility:
        visible_ids = np.flatnonzero(attribution.visible_mass > 0).astype(np.int32)
        visible_mass = attribution.visible_mass[visible_ids].astype(np.float32)
    else:
        visible_ids = np.empty(0, dtype=np.int32)
        visible_mass = np.empty(0, dtype=np.float32)
    return FrameLiftRecord(
        frame_id=int(frame_id),
        image_name=str(image_name),
        fragments=tuple(fragments),
        labels=np.asarray(labels, dtype=np.int16),
        visible_ids=visible_ids,
        visible_mass=visible_mass,
        abstained=bool(attribution.abstained),
        mask_count=int(attribution.mask_count),
    )


def _git_commit(repo: Path) -> str:
    marker = repo / "GIT_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted-preflight"


def _scene_paths(base_path: Path) -> dict[str, Path]:
    rgb_candidates = (
        base_path / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
        base_path / "output_models/point_cloud/iteration_30000/point_cloud.ply",
    )
    return {
        "rgb_ply": next(
            (path for path in rgb_candidates if path.is_file()), rgb_candidates[0]
        ),
        "feature_ply": base_path / "saga/contrastive_feature_point_cloud.ply",
        "sparse": base_path / "fastRecon/dense/sparse/0",
        "images": base_path / "fastRecon/dense/sparse/0/images",
        "masks": base_path / "saga/masks",
        "labels": base_path / "saga/labels",
    }


def _load_cameras(paths: Mapping[str, Path]) -> list[Any]:
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import cameraList_from_camInfos

    sparse = paths["sparse"]
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except (FileNotFoundError, OSError):
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
    infos = readColmapCameras(
        extrinsics,
        intrinsics,
        str(paths["images"]),
        masks_folder=str(paths["masks"]),
        labels_folder=str(paths["labels"]),
    )
    args = SimpleNamespace(resolution=1, data_device="cuda")
    return cameraList_from_camInfos(infos, 1, args)


def _resize_masks_array(
    masks: Any,
    height: int,
    width: int,
) -> np.ndarray:
    tensor = torch.as_tensor(masks).detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"mask tensor must be MxHxW, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor.float().unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    return tensor.bool().numpy()


def normalize_grounded_payload(
    masks: Any | None,
    labels: Any | None,
    height: int,
    width: int,
) -> FrameMaskPayload:
    """Treat a missing detector result as abstention, never as background."""
    if masks is None and labels is None:
        return FrameMaskPayload(None, np.empty(0, dtype=np.int16), True)
    if masks is None or labels is None:
        raise ValueError("Grounded-SAM masks and labels must both exist or both be absent")
    mask_array = _resize_masks_array(masks, height, width)
    label_array = np.asarray(torch.as_tensor(labels).detach().cpu()).reshape(-1)
    if len(mask_array) != len(label_array):
        raise ValueError(
            f"Grounded-SAM has {len(mask_array)} masks but {len(label_array)} labels"
        )
    if not np.issubdtype(label_array.dtype, np.integer):
        if not np.all(np.equal(label_array, np.floor(label_array))):
            raise ValueError("Grounded-SAM labels must be integer class IDs")
    return FrameMaskPayload(
        mask_array,
        label_array.astype(np.int16, copy=False),
        False,
    )


def load_segment_everything_payload(
    root: Path,
    image_name: str,
    height: int,
    width: int,
) -> FrameMaskPayload:
    path = root / f"{image_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing SAM-everything frame output: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            packed = np.asarray(payload["packed"], dtype=np.uint8)
            count = int(np.asarray(payload["count"]).item())
            stored_height = int(np.asarray(payload["height"]).item())
            stored_width = int(np.asarray(payload["width"]).item())
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"invalid SAM-everything frame output: {path}") from error
    if (stored_height, stored_width) != (height, width):
        raise ValueError(
            f"SAM-everything shape {(stored_height, stored_width)} does not match "
            f"camera {(height, width)}"
        )
    expected = (count, (height * width + 7) // 8)
    if count < 0 or packed.shape != expected:
        raise ValueError(f"invalid packed SAM-everything array: {path}")
    mask_array = np.unpackbits(
        packed, axis=1, count=height * width
    ).reshape(count, height, width).astype(np.bool_, copy=False)
    return FrameMaskPayload(
        mask_array,
        np.full(len(mask_array), -1, dtype=np.int16),
        False,
    )


def _validated_rendered_mask(
    rendered: Mapping[str, Any],
    height: int,
    width: int,
) -> torch.Tensor:
    image = rendered.get("mask")
    if not isinstance(image, torch.Tensor) or image.shape != (3, height, width):
        shape = None if not isinstance(image, torch.Tensor) else tuple(image.shape)
        raise ValueError(f"mask renderer returned unexpected shape {shape}")
    if not image.requires_grad:
        raise RuntimeError("differentiable mask renderer returned a detached image")
    return image


def render_alpha_mass_attribution(
    camera: Any,
    rgb_gaussians: Any,
    pipeline: Any,
    background: torch.Tensor,
    masks: np.ndarray | None,
    point_count: int,
    *,
    render_mask_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> AttributionMass:
    """Accumulate all normalized ``alpha*T`` contributors via autograd.

    Each rasterizer invocation carries at most three mask objectives.  The
    returned value is per-Gaussian mass only; no per-pixel contributor tensor
    is retained or serialized.
    """
    if render_mask_fn is None:
        from gaussian_renderer import render_mask as render_mask_fn

    height = int(camera.image_height)
    width = int(camera.image_width)
    abstained = masks is None
    mask_array = (
        np.zeros((0, height, width), dtype=bool)
        if masks is None else np.asarray(masks, dtype=bool)
    )
    if mask_array.ndim != 3 or mask_array.shape[1:] != (height, width):
        raise ValueError("AM masks must match the rendered image shape")
    if point_count <= 0:
        raise ValueError("point_count must be positive")

    xyz = rgb_gaussians.get_xyz
    if int(xyz.shape[0]) != int(point_count):
        raise ValueError("RGB Gaussian count does not match point_count")
    batches = list(iter_three_channel_mask_batches(mask_array.astype(np.float32)))
    if not batches:
        # A render is still needed for shared per-frame visibility.  This is an
        # empty mask source (or abstention), not background evidence.
        from .v8_lifting import AMChannelBatch

        batches = [
            AMChannelBatch(
                mask_indices=(),
                targets=np.zeros((3, height, width), dtype=np.float32),
            )
        ]

    visible_gradient: np.ndarray | None = None
    valid_pixel_count = 0
    inside_batches: list[tuple[Any, np.ndarray]] = []
    for batch_number, batch in enumerate(batches):
        probe = torch.ones(
            (point_count, 3),
            dtype=xyz.dtype,
            device=xyz.device,
            requires_grad=True,
        )
        rendered = render_mask_fn(
            camera,
            rgb_gaussians,
            pipeline,
            background,
            precomputed_mask=probe,
        )
        image = _validated_rendered_mask(rendered, height, width)
        opacity = image.detach()[0].float().cpu().numpy()
        objectives = build_am_objective_targets(batch, opacity)
        coefficients = torch.as_tensor(
            objectives.inside_coefficients,
            dtype=image.dtype,
            device=image.device,
        )

        if batch_number == 0:
            visible_coeff = torch.as_tensor(
                objectives.visible_coefficient,
                dtype=image.dtype,
                device=image.device,
            )
            visible_objective = torch.sum(image[0] * visible_coeff)
            visible_tensor = torch.autograd.grad(
                visible_objective,
                probe,
                retain_graph=bool(batch.active_channels),
                create_graph=False,
            )[0]
            visible_gradient = visible_tensor.detach().cpu().numpy()
            valid_pixel_count = int(np.count_nonzero(objectives.valid_pixels))

        if batch.active_channels:
            inside_objective = torch.sum(image * coefficients)
            gradient = torch.autograd.grad(
                inside_objective,
                probe,
                retain_graph=False,
                create_graph=False,
            )[0]
            inside_batches.append((batch, gradient.detach().cpu().numpy()))

    if visible_gradient is None:
        raise RuntimeError("AM visibility gradient was not produced")
    return attribution_from_am_gradients(
        visible_gradient,
        inside_batches,
        len(mask_array),
        valid_pixel_count,
        abstained=abstained,
    )


def _slice_attribution(
    attribution: AttributionMass,
    start: int,
    stop: int,
    *,
    abstained: bool,
) -> AttributionMass:
    return AttributionMass(
        source=attribution.source,
        inside_mass=attribution.inside_mass[start:stop].copy(),
        visible_mass=attribution.visible_mass.copy(),
        valid_pixel_count=attribution.valid_pixel_count,
        abstained=abstained,
    )


def stable_fragments(
    attribution: AttributionMass,
    frame_id: int,
    stable_mask_offset: int,
    *,
    config: V8FragmentConfig = V8FragmentConfig(),
) -> tuple[AttributionFragment, ...]:
    """Assign IDs from source mask identity, independent of arm survival."""
    fragments = fragments_from_attribution(attribution, frame_id, config=config)
    return tuple(
        replace(
            fragment,
            fragment_id=int(stable_mask_offset + fragment.mask_index),
        )
        for fragment in fragments
    )


def compare_contributor_images(
    historical_id: np.ndarray,
    historical_weight: np.ndarray,
    fixed_id: np.ndarray,
    fixed_weight: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return aggregate and changed-pixel evidence for the Stage-0 audit."""
    old_id = np.asarray(historical_id, dtype=np.int64)
    new_id = np.asarray(fixed_id, dtype=np.int64)
    old_weight = np.asarray(historical_weight, dtype=np.float64)
    new_weight = np.asarray(fixed_weight, dtype=np.float64)
    if old_id.shape != new_id.shape or old_weight.shape != old_id.shape or new_weight.shape != old_id.shape:
        raise ValueError("historical and fixed contributor images must share one HxW shape")
    changed = (old_id != new_id) | ~np.isclose(old_weight, new_weight, rtol=1e-6, atol=1e-8)
    changed_flat = np.flatnonzero(changed.reshape(-1))
    abs_weight = np.abs(old_weight - new_weight)
    summary = {
        "pixel_count": int(old_id.size),
        "changed_pixel_count": int(len(changed_flat)),
        "changed_pixel_fraction": float(len(changed_flat) / old_id.size) if old_id.size else 0.0,
        "id_changed_pixel_count": int(np.count_nonzero(old_id != new_id)),
        "historical_invalid_fixed_valid": int(np.count_nonzero((old_id < 0) & (new_id >= 0))),
        "historical_valid_fixed_invalid": int(np.count_nonzero((old_id >= 0) & (new_id < 0))),
        "mean_absolute_weight_difference": float(abs_weight.mean()) if abs_weight.size else 0.0,
        "max_absolute_weight_difference": float(abs_weight.max()) if abs_weight.size else 0.0,
    }
    evidence = {
        "flat_pixel": changed_flat.astype(np.int32),
        "historical_id": old_id.reshape(-1)[changed_flat].astype(np.int32),
        "fixed_id": new_id.reshape(-1)[changed_flat].astype(np.int32),
        "historical_weight": old_weight.reshape(-1)[changed_flat].astype(np.float32),
        "fixed_weight": new_weight.reshape(-1)[changed_flat].astype(np.float32),
    }
    return summary, evidence


def _load_historical_contributor(
    root: Path,
    image_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = root / f"{image_name}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as arrays:
        id_key = "max_id" if "max_id" in arrays else "max_contributor"
        weight_key = "max_weight" if "max_weight" in arrays else "max_contribute"
        if id_key not in arrays or weight_key not in arrays:
            raise ValueError(f"historical contributor file lacks ID/weight: {path}")
        return np.asarray(arrays[id_key]), np.asarray(arrays[weight_key])


def _ragged(
    rows: Sequence[np.ndarray],
    dtype: np.dtype[Any],
) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=dtype) for row in rows])
        if int(indptr[-1]) else np.empty(0, dtype=dtype)
    )
    return indptr, values


def _serialize_records(
    output_dir: Path,
    xyz_m: np.ndarray,
    opacity: np.ndarray,
    affinity: np.ndarray,
    semantic: np.ndarray,
    label_features: np.ndarray,
    geometry_records: Sequence[FrameLiftRecord],
    semantic_records: Sequence[FrameLiftRecord],
    contributor_diff_rows: Sequence[Mapping[str, np.ndarray]],
) -> None:
    fragments = [fragment for record in geometry_records for fragment in record.fragments]
    semantic_fragments = [
        fragment for record in semantic_records for fragment in record.fragments
    ]
    full_indptr, full_ids = _ragged([item.full_ids for item in fragments], np.int32)
    _, full_mass = _ragged([item.full_inside_mass for item in fragments], np.float32)
    core_indptr, core_ids = _ragged([item.core_ids for item in fragments], np.int32)
    _, core_mass = _ragged([item.core_inside_mass for item in fragments], np.float32)
    _, core_ratio = _ragged([item.core_inside_ratio for item in fragments], np.float32)
    visible_ids_rows = [record.visible_ids for record in geometry_records]
    visible_mass_rows = [record.visible_mass for record in geometry_records]
    visible_indptr, visible_ids = _ragged(visible_ids_rows, np.int32)
    _, visible_mass = _ragged(visible_mass_rows, np.float32)

    semantic_full_indptr, semantic_full_ids = _ragged(
        [item.full_ids for item in semantic_fragments], np.int32
    )
    _, semantic_full_mass = _ragged(
        [item.full_inside_mass for item in semantic_fragments], np.float32
    )
    semantic_core_indptr, semantic_core_ids = _ragged(
        [item.core_ids for item in semantic_fragments], np.int32
    )
    _, semantic_core_mass = _ragged(
        [item.core_inside_mass for item in semantic_fragments], np.float32
    )

    diff_indptr, diff_pixels = _ragged(
        [np.asarray(row["flat_pixel"]) for row in contributor_diff_rows], np.int32
    )
    diff_arrays: dict[str, np.ndarray] = {}
    for key, dtype in (
        ("historical_id", np.int32),
        ("fixed_id", np.int32),
        ("historical_weight", np.float32),
        ("fixed_weight", np.float32),
    ):
        _, diff_arrays[key] = _ragged(
            [np.asarray(row[key]) for row in contributor_diff_rows], dtype
        )

    geometry_label_by_fragment: list[int] = []
    for record in geometry_records:
        geometry_label_by_fragment.extend(
            int(record.labels[item.mask_index]) for item in record.fragments
        )
    semantic_label_by_fragment: list[int] = []
    for record in semantic_records:
        semantic_label_by_fragment.extend(
            int(record.labels[item.mask_index]) for item in record.fragments
        )

    np.savez_compressed(
        output_dir / "lifting_bank.npz",
        xyz_m=np.asarray(xyz_m, dtype=np.float32),
        opacity=np.asarray(opacity, dtype=np.float32),
        affinity=np.asarray(affinity, dtype=np.float32),
        semantic=np.asarray(semantic, dtype=np.float32),
        label_features=np.asarray(label_features, dtype=np.float32),
        fragment_full_indptr=full_indptr,
        fragment_full_ids=full_ids,
        fragment_full_mass=full_mass,
        fragment_core_indptr=core_indptr,
        fragment_core_ids=core_ids,
        fragment_core_mass=core_mass,
        fragment_core_ratio=core_ratio,
        fragment_id=np.asarray([item.fragment_id for item in fragments], dtype=np.int64),
        fragment_frame=np.asarray([item.frame_id for item in fragments], dtype=np.int32),
        fragment_mask_index=np.asarray([item.mask_index for item in fragments], dtype=np.int32),
        fragment_source_class=np.asarray(geometry_label_by_fragment, dtype=np.int16),
        frame_visible_indptr=visible_indptr,
        frame_visible_ids=visible_ids,
        frame_visible_mass=visible_mass,
        frame_geometry_abstained=np.asarray(
            [record.abstained for record in geometry_records], dtype=np.bool_
        ),
        frame_grounded_missing=np.asarray(
            [record.abstained for record in semantic_records], dtype=np.bool_
        ),
        semantic_fragment_full_indptr=semantic_full_indptr,
        semantic_fragment_full_ids=semantic_full_ids,
        semantic_fragment_full_mass=semantic_full_mass,
        semantic_fragment_core_indptr=semantic_core_indptr,
        semantic_fragment_core_ids=semantic_core_ids,
        semantic_fragment_core_mass=semantic_core_mass,
        semantic_fragment_id=np.asarray(
            [item.fragment_id for item in semantic_fragments], dtype=np.int64
        ),
        semantic_fragment_frame=np.asarray(
            [item.frame_id for item in semantic_fragments], dtype=np.int32
        ),
        semantic_fragment_mask_index=np.asarray(
            [item.mask_index for item in semantic_fragments], dtype=np.int32
        ),
        semantic_fragment_class=np.asarray(semantic_label_by_fragment, dtype=np.int16),
        contributor_diff_frame_indptr=diff_indptr,
        contributor_diff_flat_pixel=diff_pixels,
        contributor_diff_historical_id=diff_arrays["historical_id"],
        contributor_diff_fixed_id=diff_arrays["fixed_id"],
        contributor_diff_historical_weight=diff_arrays["historical_weight"],
        contributor_diff_fixed_weight=diff_arrays["fixed_weight"],
    )


def lifting_bank_is_complete(
    output_dir: Path,
    *,
    expected_scene_id: str | None = None,
    expected_mask_source: str | None = None,
    expected_lifting_source: str | None = None,
    expected_contributor_audit: bool | None = None,
) -> bool:
    # Keep the standalone worker and the parent resumable runner on one
    # completeness contract.  Otherwise a partially written bank can be
    # rejected by the parent but incorrectly skipped by the worker.
    from .v8_runner import lifting_bank_is_complete as validate

    return validate(
        Path(output_dir),
        expected_scene_id=expected_scene_id,
        expected_mask_source=expected_mask_source,
        expected_lifting_source=expected_lifting_source,
        expected_contributor_audit=expected_contributor_audit,
    )


def run_v8_lifting_bank(
    scene_id: str,
    base_path: Path,
    output_dir: Path,
    scene_scale_m_per_unit: float,
    *,
    mask_source: str,
    lifting_source: str,
    segment_everything_root: Path | None = None,
    historical_contributor_root: Path | None = None,
    contributor_audit: bool = False,
    label_features_path: Path | None = None,
    feature_ply_path: Path | None = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Render one deterministic scene into a V8 frame-fragment artifact."""
    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    if mask_source not in {"G", "S"}:
        raise ValueError("mask_source must be G or S")
    if lifting_source not in {"M1", "AM"}:
        raise ValueError("lifting_source must be M1 or AM")
    if mask_source == "S" and segment_everything_root is None:
        raise ValueError("segment_everything_root is required for mask source S")

    from gaussian_renderer import render_with_max_contributor
    from scene import FeatureGaussianModel, GaussianModel

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    paths = _scene_paths(base_path)
    required = tuple(paths.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing scene assets: {missing}")

    rgb = GaussianModel(0)
    rgb.load_ply(str(paths["rgb_ply"]))
    feature = FeatureGaussianModel(32, 32)
    selected_feature_ply = feature_ply_path or paths["feature_ply"]
    if not selected_feature_ply.is_file():
        raise FileNotFoundError(f"feature PLY not found: {selected_feature_ply}")
    feature.load_ply(str(selected_feature_ply))
    rgb_xyz = rgb.get_xyz.detach().cpu().numpy()
    feature_xyz = feature.get_xyz.detach().cpu().numpy()
    if rgb_xyz.shape != feature_xyz.shape or not np.allclose(
        rgb_xyz, feature_xyz, rtol=0.0, atol=1e-6
    ):
        raise ValueError("RGB and feature Gaussian XYZ/order do not match")
    point_count = int(len(rgb_xyz))
    xyz_m = feature_xyz.astype(np.float64) * float(scene_scale_m_per_unit)
    opacity = rgb.get_opacity.detach().cpu().numpy().reshape(-1)
    affinity = feature.get_point_features.detach().cpu().numpy()
    affinity /= np.maximum(np.linalg.norm(affinity, axis=1, keepdims=True), 1e-12)
    semantic = feature.get_point_semantic_features.detach().cpu().numpy()
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
    if label_features_path is None:
        label_features = np.empty((0, semantic.shape[1]), dtype=np.float32)
    else:
        if not label_features_path.is_file():
            raise FileNotFoundError(f"label feature codebook not found: {label_features_path}")
        loaded_labels = torch.load(label_features_path, map_location="cpu")
        if not isinstance(loaded_labels, torch.Tensor) or loaded_labels.ndim != 2:
            raise ValueError("label feature codebook must be a two-dimensional tensor")
        label_features = loaded_labels.detach().cpu().numpy().astype(np.float32)
        if label_features.shape != (len(classes), semantic.shape[1]):
            raise ValueError(
                "label feature codebook must match the 32 classes and semantic dimension"
            )
        label_features /= np.maximum(
            np.linalg.norm(label_features, axis=1, keepdims=True), 1e-12
        )

    cameras = sorted(_load_cameras(paths), key=lambda item: item.image_name)
    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    geometry_records: list[FrameLiftRecord] = []
    semantic_records: list[FrameLiftRecord] = []
    geometry_offset = 0
    semantic_offset = 0
    grounded_abstentions = 0
    contributor_summaries: list[dict[str, Any]] = []
    contributor_diff_rows: list[dict[str, np.ndarray]] = []

    for frame_id, camera in enumerate(cameras):
        grounded = normalize_grounded_payload(
            camera.original_masks,
            camera.labels,
            int(camera.image_height),
            int(camera.image_width),
        )
        grounded_abstentions += int(grounded.abstained)
        geometry = (
            grounded if mask_source == "G" else load_segment_everything_payload(
                Path(segment_everything_root),
                str(camera.image_name),
                int(camera.image_height),
                int(camera.image_width),
            )
        )

        fixed_ids: np.ndarray | None = None
        fixed_weights: np.ndarray | None = None
        if lifting_source == "M1" or contributor_audit or historical_contributor_root is not None:
            with torch.no_grad():
                rendered = render_with_max_contributor(
                    camera, rgb, pipeline, background
                )
            fixed_ids = rendered["max_contributor"].detach().cpu().numpy().astype(np.int64)
            fixed_weights = rendered["max_contribute"].detach().cpu().numpy().astype(np.float64)

        if lifting_source == "M1":
            assert fixed_ids is not None and fixed_weights is not None
            geometry_attr = mass_from_max_contributor(
                fixed_ids, fixed_weights, geometry.masks, point_count
            )
            semantic_attr = (
                geometry_attr if mask_source == "G" else mass_from_max_contributor(
                    fixed_ids, fixed_weights, grounded.masks, point_count
                )
            )
        else:
            if mask_source == "G":
                geometry_attr = render_alpha_mass_attribution(
                    camera, rgb, pipeline, background, geometry.masks, point_count
                )
                semantic_attr = geometry_attr
            else:
                geometry_masks = geometry.masks
                assert geometry_masks is not None
                semantic_masks = (
                    np.zeros((0, *geometry_masks.shape[1:]), dtype=bool)
                    if grounded.masks is None else grounded.masks
                )
                combined_masks = np.concatenate((geometry_masks, semantic_masks), axis=0)
                combined = render_alpha_mass_attribution(
                    camera, rgb, pipeline, background, combined_masks, point_count
                )
                boundary = len(geometry_masks)
                geometry_attr = _slice_attribution(
                    combined, 0, boundary, abstained=False
                )
                semantic_attr = _slice_attribution(
                    combined,
                    boundary,
                    boundary + len(semantic_masks),
                    abstained=grounded.abstained,
                )

        geometry_fragments = stable_fragments(
            geometry_attr, frame_id, geometry_offset
        )
        # Late MV-label classification compares tracks with every Grounded
        # label mask that has any lifted full support.  It must not inherit the
        # geometry-fragment minimum-size gate, which would silently censor
        # small semantic evidence before the registered weighted-IoU vote.
        semantic_fragments = stable_fragments(
            semantic_attr,
            frame_id,
            semantic_offset,
            config=V8FragmentConfig(
                fragment_min_core=0,
                fragment_min_full=1,
            ),
        )
        geometry_records.append(sparse_frame_lift_record(
            frame_id,
            str(camera.image_name),
            geometry_attr,
            geometry_fragments,
            geometry.labels,
            retain_visibility=True,
        ))
        semantic_records.append(sparse_frame_lift_record(
            frame_id,
            str(camera.image_name),
            semantic_attr,
            semantic_fragments,
            grounded.labels,
            retain_visibility=False,
        ))
        geometry_offset += geometry.mask_count
        semantic_offset += grounded.mask_count

        if contributor_audit or historical_contributor_root is not None:
            historical: tuple[np.ndarray, np.ndarray] | None = None
            if contributor_audit and "historical_max_contributor" in rendered:
                historical = (
                    rendered["historical_max_contributor"].detach().cpu().numpy(),
                    rendered["historical_max_contribute"].detach().cpu().numpy(),
                )
            elif historical_contributor_root is not None:
                historical = _load_historical_contributor(
                    historical_contributor_root, str(camera.image_name)
                )
            if historical is None:
                contributor_summaries.append({
                    "frame_id": frame_id,
                    "image_name": str(camera.image_name),
                    "status": "historical-missing",
                })
                contributor_diff_rows.append({
                    "flat_pixel": np.empty(0, dtype=np.int32),
                    "historical_id": np.empty(0, dtype=np.int32),
                    "fixed_id": np.empty(0, dtype=np.int32),
                    "historical_weight": np.empty(0, dtype=np.float32),
                    "fixed_weight": np.empty(0, dtype=np.float32),
                })
            else:
                assert fixed_ids is not None and fixed_weights is not None
                summary, evidence = compare_contributor_images(
                    historical[0], historical[1], fixed_ids, fixed_weights
                )
                contributor_summaries.append({
                    "frame_id": frame_id,
                    "image_name": str(camera.image_name),
                    "status": "compared",
                    **summary,
                })
                contributor_diff_rows.append(evidence)

    _serialize_records(
        output_dir,
        xyz_m,
        opacity,
        affinity,
        semantic,
        label_features,
        geometry_records,
        semantic_records,
        contributor_diff_rows,
    )
    fragment_count = sum(len(record.fragments) for record in geometry_records)
    semantic_fragment_count = sum(len(record.fragments) for record in semantic_records)
    bank = {
        "schema": "saga-v8-lifting-bank-v1",
        "scene_id": scene_id,
        "git_commit": _git_commit(repo_root or Path(__file__).resolve().parents[1]),
        "mask_source": mask_source,
        "lifting_source": lifting_source,
        "point_count": point_count,
        "frame_count": len(geometry_records),
        "frame_image_names": [record.image_name for record in geometry_records],
        "grounded_abstention_frame_count": grounded_abstentions,
        "mask_count": int(sum(record.mask_count for record in geometry_records)),
        "fragment_count": fragment_count,
        "semantic_fragment_count": semantic_fragment_count,
        "classes": list(classes),
        "config": V8FragmentConfig().__dict__,
        "arrays_npz": "lifting_bank.npz",
        "contributor_audit_requested": bool(contributor_audit),
        "historical_contributor_root": (
            None if historical_contributor_root is None else str(historical_contributor_root)
        ),
        "contributor_comparisons": contributor_summaries,
        "runtime_seconds": float(time.monotonic() - started),
    }
    (output_dir / "lifting_bank.json").write_text(
        json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not lifting_bank_is_complete(output_dir):
        raise RuntimeError("serialized V8 lifting bank failed validation")
    return bank


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-scale-m-per-unit", type=float, default=1.0)
    parser.add_argument("--mask-source", choices=("G", "S"), required=True)
    parser.add_argument("--lifting-source", choices=("M1", "AM"), required=True)
    parser.add_argument("--segment-everything-root", type=Path)
    parser.add_argument("--sam-mask-root", type=Path, dest="segment_everything_root")
    parser.add_argument("--label-features", type=Path)
    parser.add_argument("--feature-ply", type=Path)
    parser.add_argument("--historical-contributor-root", type=Path)
    parser.add_argument("--contributor-audit", action="store_true")
    args = parser.parse_args(argv)
    if lifting_bank_is_complete(
        args.output_dir,
        expected_scene_id=args.scene_id,
        expected_mask_source=args.mask_source,
        expected_lifting_source=args.lifting_source,
        expected_contributor_audit=True if args.contributor_audit else None,
    ):
        print(f"complete lifting bank exists, skipping: {args.output_dir}")
        return 0
    bank = run_v8_lifting_bank(
        args.scene_id,
        args.base_path,
        args.output_dir,
        args.scene_scale_m_per_unit,
        mask_source=args.mask_source,
        lifting_source=args.lifting_source,
        segment_everything_root=args.segment_everything_root,
        historical_contributor_root=args.historical_contributor_root,
        contributor_audit=args.contributor_audit,
        label_features_path=args.label_features,
        feature_ply_path=args.feature_ply,
    )
    print(json.dumps({
        key: bank[key] for key in (
            "scene_id", "mask_source", "lifting_source", "fragment_count",
            "semantic_fragment_count", "runtime_seconds",
        )
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
