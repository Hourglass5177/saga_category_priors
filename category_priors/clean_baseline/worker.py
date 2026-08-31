from __future__ import annotations

"""Scene worker for the clean alpha-mask evidence bank.

This module is deliberately independent of the historical V3--V10 runners.
It uses the differentiable Gaussian renderer only to reduce per-pixel
``alpha * T_prev`` contributions into per-frame, per-mask sparse Gaussian
support.  No pixel-by-Gaussian cache is materialised or written to disk.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


DEFAULT_CLASSES: tuple[str, ...] = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


@dataclass(frozen=True)
class CleanSceneInputs:
    base_path: Path
    rgb_ply: Path
    sparse: Path
    images: Path
    sam_masks: Path
    grounded_masks: Path
    grounded_labels: Path


@dataclass(frozen=True)
class RenderedMaskSupport:
    mask_index: int
    gaussian_ids: np.ndarray
    inside_mass: np.ndarray
    inside_ratio: np.ndarray
    ambiguous_ids: np.ndarray
    class_probabilities: np.ndarray


@dataclass(frozen=True)
class RenderedFrameEvidence:
    frame_id: int
    image_name: str
    visible_ids: np.ndarray
    visible_mass: np.ndarray
    masks: tuple[RenderedMaskSupport, ...]
    grounded_abstained: bool
    valid_pixel_count: int


@dataclass(frozen=True)
class _MaskBatch:
    indices: tuple[int, ...]
    targets: np.ndarray


def _resolve_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str | Path
) -> Path:
    base = Path(str(scene["base_path"])).resolve()
    value: Any = default
    for key in keys:
        if scene.get(key) not in (None, ""):
            value = scene[key]
            break
    path = Path(str(value))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_clean_scene_inputs(
    scene: Mapping[str, Any], *, sam_masks: Path | None = None,
    require_exists: bool = True,
) -> CleanSceneInputs:
    base = Path(str(scene["base_path"])).resolve()
    rgb_candidates = (
        base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
    )
    rgb = (
        _resolve_path(scene, ("point_cloud_path",), "")
        if scene.get("point_cloud_path")
        else next((path for path in rgb_candidates if path.is_file()), rgb_candidates[0])
    )
    if sam_masks is None:
        sam_value = next(
            (
                scene[key]
                for key in (
                    "segment_everything_root",
                    "sam_everything_packed_path",
                    "sam_everything_root",
                )
                if scene.get(key) not in (None, "")
            ),
            None,
        )
        if sam_value is None:
            raise ValueError("runtime scene lacks a SAM-everything mask root")
        sam_path = Path(str(sam_value))
        sam_masks = (
            (base / sam_path).resolve()
            if not sam_path.is_absolute()
            else sam_path.resolve()
        )
    result = CleanSceneInputs(
        base_path=base,
        rgb_ply=Path(rgb).resolve(),
        sparse=_resolve_path(scene, ("sparse_path",), "fastRecon/dense/sparse/0"),
        images=_resolve_path(scene, ("images_path",), "fastRecon/dense/sparse/0/images"),
        sam_masks=Path(sam_masks).resolve(),
        grounded_masks=_resolve_path(
            scene, ("grounded_masks_path", "masks_path"), "saga/masks"
        ),
        grounded_labels=_resolve_path(
            scene, ("grounded_labels_path", "labels_path"), "saga/labels"
        ),
    )
    if require_exists:
        missing = [
            str(path)
            for path in (
                result.rgb_ply,
                result.sparse,
                result.images,
                result.sam_masks,
            )
            if not path.exists()
        ]
        # A frame without a Grounded-SAM detection is an abstention.  The two
        # directories may therefore both be absent, but a one-sided source is
        # malformed because it can silently pair masks with the wrong labels.
        grounded_exists = (
            result.grounded_masks.exists(), result.grounded_labels.exists()
        )
        if grounded_exists[0] != grounded_exists[1]:
            missing.append(
                "Grounded-SAM mask/label roots are one-sided: "
                f"{result.grounded_masks}, {result.grounded_labels}"
            )
        if missing:
            raise FileNotFoundError(
                f"missing clean-baseline scene inputs: {missing}"
            )
    return result


def iter_mask_batches(masks: np.ndarray) -> Iterator[_MaskBatch]:
    # Keep the scene-sized mask stack in its compact source dtype (normally
    # bool/uint8).  Converting it here used to materialise a second float32
    # copy of every mask in the frame before the first render.
    array = np.asarray(masks)
    if array.ndim != 3:
        raise ValueError("masks must have shape MxHxW")
    for start in range(0, len(array), 3):
        stop = min(start + 3, len(array))
        targets = np.zeros((3, *array.shape[1:]), dtype=np.float32)
        targets[: stop - start] = np.asarray(
            array[start:stop], dtype=np.float32
        )
        yield _MaskBatch(tuple(range(start, stop)), targets)


def normalized_alpha_objectives(
    targets: np.ndarray,
    opacity: np.ndarray,
    *,
    min_opacity: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inside/visible coefficients for normalized alpha mass.

    If ``rendered = sum_g(alpha*T_prev*feature_g)``, differentiating the
    coefficient-weighted image with respect to ``feature_g`` yields the sum of
    normalized per-pixel contributor mass for Gaussian ``g``.
    """

    target = np.asarray(targets, dtype=np.float64)
    alpha = np.asarray(opacity, dtype=np.float64)
    if target.ndim != 3 or target.shape[0] != 3:
        raise ValueError("targets must be 3xHxW")
    if alpha.shape != target.shape[1:]:
        raise ValueError("opacity must match target image shape")
    valid = np.isfinite(alpha) & (alpha > float(min_opacity))
    inv = np.zeros_like(alpha, dtype=np.float64)
    inv[valid] = 1.0 / alpha[valid]
    inside = target * inv[None, :, :]
    visible = inv
    return inside.astype(np.float32), visible.astype(np.float32), valid


def _validate_mass(
    inside_mass: np.ndarray, visible_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inside = np.asarray(inside_mass, dtype=np.float64)
    visible = np.asarray(visible_mass, dtype=np.float64).reshape(-1)
    if inside.ndim != 2 or inside.shape[1] != len(visible):
        raise ValueError("inside mass must be MxN and visible mass must be N")
    if np.any(~np.isfinite(inside)) or np.any(~np.isfinite(visible)):
        raise ValueError("alpha mass must be finite")
    if np.any(inside < -1e-7) or np.any(visible < -1e-7):
        raise ValueError("alpha mass must be non-negative")
    inside = np.maximum(inside, 0.0)
    visible = np.maximum(visible, 0.0)
    tolerance = 5e-5 * np.maximum(visible[None, :], 1.0)
    if inside.size and np.any(inside - visible[None, :] > tolerance):
        raise ValueError("inside mass cannot exceed visible mass")
    return np.minimum(inside, visible[None, :]), visible


def sparse_support_from_mass(
    inside_mass: np.ndarray,
    visible_mass: np.ndarray,
    *,
    inside_threshold: float = 0.5,
    ratio_threshold: float = 0.5,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    inside, visible = _validate_mass(inside_mass, visible_mass)
    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row in inside:
        ratio = np.divide(row, visible, out=np.zeros_like(row), where=visible > 0)
        keep = (row >= float(inside_threshold)) & (ratio >= float(ratio_threshold))
        ids = np.flatnonzero(keep).astype(np.int32)
        result.append(
            (
                ids,
                row[ids].astype(np.float32),
                ratio[ids].astype(np.float32),
            )
        )
    return tuple(result)


def mark_same_frame_ambiguity(
    supports: Sequence[np.ndarray], point_count: int
) -> tuple[np.ndarray, ...]:
    """Return per-mask ambiguous IDs without deleting the original support."""

    counts = np.zeros(int(point_count), dtype=np.int32)
    rows = tuple(np.unique(np.asarray(row, dtype=np.int64)) for row in supports)
    for row in rows:
        if np.any(row < 0) or np.any(row >= point_count):
            raise ValueError("support contains out-of-range Gaussian ID")
        counts[row] += 1
    return tuple(row[counts[row] > 1].astype(np.int32) for row in rows)


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    lookup = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1).astype(np.uint8)
    return lookup[np.asarray(packed, dtype=np.uint8)].sum(axis=-1, dtype=np.int64)


def mask_class_probabilities(
    sam_masks: np.ndarray,
    grounded_masks: np.ndarray | None,
    grounded_labels: np.ndarray | None,
    *,
    class_count: int = 32,
    iou_threshold: float = 0.25,
    chunk_size: int = 8,
) -> tuple[np.ndarray, bool]:
    """Match full SAM masks to same-frame labelled masks using exact 2D IoU."""

    sam = np.asarray(sam_masks, dtype=bool)
    if sam.ndim != 3:
        raise ValueError("SAM masks must be MxHxW")
    result = np.zeros((len(sam), int(class_count)), dtype=np.float32)
    if grounded_masks is None and grounded_labels is None:
        return result, True
    if grounded_masks is None or grounded_labels is None:
        raise ValueError("Grounded masks and labels must both exist or both abstain")
    grounded = np.asarray(grounded_masks, dtype=bool)
    labels = np.asarray(grounded_labels).reshape(-1)
    if grounded.ndim != 3 or grounded.shape[1:] != sam.shape[1:]:
        raise ValueError("Grounded masks must match SAM mask image shape")
    if len(grounded) != len(labels):
        raise ValueError("Grounded masks and labels have different lengths")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("Grounded class ID is outside the complete class vocabulary")
    pixels = int(np.prod(sam.shape[1:]))
    sam_packed = np.packbits(sam.reshape(len(sam), pixels), axis=1)
    grounded_packed = np.packbits(grounded.reshape(len(grounded), pixels), axis=1)
    sam_area = _popcount_rows(sam_packed)
    grounded_area = _popcount_rows(grounded_packed)
    for start in range(0, len(sam), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(sam))
        for local, packed in enumerate(sam_packed[start:stop]):
            intersection = _popcount_rows(
                np.bitwise_and(grounded_packed, packed[None, :])
            )
            union = sam_area[start + local] + grounded_area - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection, dtype=np.float64),
                where=union > 0,
            )
            for class_id in np.unique(labels):
                class_iou = iou[labels == class_id]
                if class_iou.size:
                    score = float(np.max(class_iou))
                    if score >= float(iou_threshold):
                        result[start + local, int(class_id)] = score
    normalizer = result.sum(axis=1, keepdims=True)
    np.divide(result, normalizer, out=result, where=normalizer > 0)
    return result, False


def _resize_masks(value: Any, height: int, width: int) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"mask tensor must be MxHxW, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor.float().unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    return tensor.bool().numpy()


def load_sam_masks(root: Path, image_name: str, height: int, width: int) -> np.ndarray:
    path = Path(root) / f"{image_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing SAM-everything mask frame: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            packed = np.asarray(payload["packed"], dtype=np.uint8)
            count = int(np.asarray(payload["count"]).item())
            stored_height = int(np.asarray(payload["height"]).item())
            stored_width = int(np.asarray(payload["width"]).item())
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"invalid packed SAM-everything frame: {path}") from error
    if (stored_height, stored_width) != (height, width):
        raise ValueError("SAM-everything frame and rendered camera shapes differ")
    expected = (count, (height * width + 7) // 8)
    if count < 0 or packed.shape != expected:
        raise ValueError(f"invalid packed SAM-everything payload: {path}")
    return np.unpackbits(packed, axis=1, count=height * width).reshape(
        count, height, width
    ).astype(bool, copy=False)


def _load_grounded(camera: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    masks = getattr(camera, "original_masks", None)
    labels = getattr(camera, "labels", None)
    if masks is None and labels is None:
        return None, None
    if masks is None or labels is None:
        raise ValueError("Grounded-SAM masks and labels must both exist or abstain")
    resized = _resize_masks(masks, int(camera.image_height), int(camera.image_width))
    label_array = np.asarray(labels.detach().cpu() if hasattr(labels, "detach") else labels).reshape(-1)
    if len(resized) != len(label_array):
        raise ValueError("Grounded-SAM masks and labels have different lengths")
    if not np.issubdtype(label_array.dtype, np.integer):
        if not np.all(label_array == np.floor(label_array)):
            raise ValueError("Grounded-SAM labels must be integral class IDs")
    return resized, label_array.astype(np.int16, copy=False)


def _load_cameras(inputs: CleanSceneInputs) -> list[Any]:
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import cameraList_from_camInfos

    try:
        extrinsics = read_extrinsics_binary(str(inputs.sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(inputs.sparse / "cameras.bin"))
    except (FileNotFoundError, OSError):
        extrinsics = read_extrinsics_text(str(inputs.sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(inputs.sparse / "cameras.txt"))
    infos = readColmapCameras(
        extrinsics,
        intrinsics,
        str(inputs.images),
        masks_folder=str(inputs.grounded_masks),
        labels_folder=str(inputs.grounded_labels),
    )
    args = SimpleNamespace(resolution=1, data_device="cuda")
    return sorted(
        cameraList_from_camInfos(infos, 1, args), key=lambda item: item.image_name
    )


def render_frame_evidence(
    camera: Any,
    gaussians: Any,
    pipeline: Any,
    background: Any,
    sam_masks: np.ndarray,
    *,
    render_mask_fn: Callable[..., Mapping[str, Any]] | None = None,
    class_count: int = 32,
) -> RenderedFrameEvidence:
    """Render one frame and immediately reduce it to sparse mask evidence."""

    import torch

    if render_mask_fn is None:
        from gaussian_renderer import render_mask as render_mask_fn
    height, width = int(camera.image_height), int(camera.image_width)
    masks = np.asarray(sam_masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[1:] != (height, width):
        raise ValueError("SAM masks must match the rendered camera")
    point_count = int(gaussians.get_xyz.shape[0])
    batches: Iterator[_MaskBatch]
    if len(masks):
        batches = iter_mask_batches(masks)
    else:
        batches = iter(
            (_MaskBatch((), np.zeros((3, height, width), dtype=np.float32)),)
        )
    visible_mass: np.ndarray | None = None
    valid_pixel_count = 0
    support_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for batch_number, batch in enumerate(batches):
        probe = torch.ones(
            (point_count, 3),
            dtype=gaussians.get_xyz.dtype,
            device=gaussians.get_xyz.device,
            requires_grad=True,
        )
        rendered = render_mask_fn(
            camera, gaussians, pipeline, background, precomputed_mask=probe
        )
        image = rendered.get("mask")
        if (
            not isinstance(image, torch.Tensor)
            or image.shape != (3, height, width)
            or not image.requires_grad
        ):
            raise RuntimeError("differentiable mask renderer returned invalid output")
        inside_coeff, visible_coeff, valid = normalized_alpha_objectives(
            batch.targets, image.detach()[0].float().cpu().numpy()
        )
        if batch_number == 0:
            coefficient = torch.as_tensor(
                visible_coeff, dtype=image.dtype, device=image.device
            )
            gradient = torch.autograd.grad(
                torch.sum(image[0] * coefficient),
                probe,
                retain_graph=bool(batch.indices),
            )[0]
            visible_mass = gradient[:, 0].detach().cpu().numpy().astype(np.float64)
            valid_pixel_count = int(np.count_nonzero(valid))
        if batch.indices:
            coefficient = torch.as_tensor(
                inside_coeff, dtype=image.dtype, device=image.device
            )
            gradient = torch.autograd.grad(torch.sum(image * coefficient), probe)[0]
            inside = np.stack(
                [
                    gradient[:, channel].detach().cpu().numpy()
                    for channel in range(len(batch.indices))
                ],
                axis=0,
            )
            assert visible_mass is not None
            support_rows.extend(sparse_support_from_mass(inside, visible_mass))
    if visible_mass is None:
        raise RuntimeError("alpha-mass visibility gradient was not produced")
    grounded_masks, grounded_labels = _load_grounded(camera)
    class_probs, abstained = mask_class_probabilities(
        masks,
        grounded_masks,
        grounded_labels,
        class_count=class_count,
    )
    ambiguous = mark_same_frame_ambiguity(
        [row[0] for row in support_rows], point_count
    )
    supports = tuple(
        RenderedMaskSupport(
            mask_index=index,
            gaussian_ids=row[0],
            inside_mass=row[1],
            inside_ratio=row[2],
            ambiguous_ids=ambiguous[index],
            class_probabilities=class_probs[index],
        )
        for index, row in enumerate(support_rows)
    )
    visible_ids = np.flatnonzero(visible_mass >= 0.5).astype(np.int32)
    return RenderedFrameEvidence(
        frame_id=int(getattr(camera, "uid", 0)),
        image_name=str(camera.image_name),
        visible_ids=visible_ids,
        visible_mass=visible_mass[visible_ids].astype(np.float32),
        masks=supports,
        grounded_abstained=abstained,
        valid_pixel_count=valid_pixel_count,
    )


def render_scene_frames(
    inputs: CleanSceneInputs,
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> tuple[np.ndarray, tuple[RenderedFrameEvidence, ...]]:
    """Load one immutable 30k scene and render every frame sequentially."""

    import torch
    from scene import GaussianModel

    gaussians = GaussianModel(0)
    gaussians.load_ply(str(inputs.rgb_ply))
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    records: list[RenderedFrameEvidence] = []
    for stable_frame_id, camera in enumerate(_load_cameras(inputs)):
        masks = load_sam_masks(
            inputs.sam_masks,
            str(camera.image_name),
            int(camera.image_height),
            int(camera.image_width),
        )
        record = render_frame_evidence(
            camera,
            gaussians,
            pipeline,
            background,
            masks,
            class_count=len(classes),
        )
        records.append(
            RenderedFrameEvidence(
                frame_id=stable_frame_id,
                image_name=record.image_name,
                visible_ids=record.visible_ids,
                visible_mass=record.visible_mass,
                masks=record.masks,
                grounded_abstained=record.grounded_abstained,
                valid_pixel_count=record.valid_pixel_count,
            )
        )
    return xyz, tuple(records)
