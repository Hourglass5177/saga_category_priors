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

from .sam_inputs import PackedMaskFrame, colmap_frame_specs, load_packed_mask_frame


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


@dataclass(frozen=True)
class _GroundedSemanticIndex:
    packed_masks: np.ndarray
    mask_area: np.ndarray
    labels: np.ndarray
    height: int
    width: int
    class_count: int


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


def iter_mask_batches(masks: np.ndarray | PackedMaskFrame) -> Iterator[_MaskBatch]:
    # Keep the scene-sized mask stack in its compact source dtype (normally
    # bool/uint8).  Converting it here used to materialise a second float32
    # copy of every mask in the frame before the first render.
    if isinstance(masks, PackedMaskFrame):
        count, height, width = masks.count, masks.height, masks.width
        dense_batch = masks.dense_batch
    else:
        array = np.asarray(masks)
        if array.ndim != 3:
            raise ValueError("masks must have shape MxHxW")
        count, height, width = array.shape
        dense_batch = lambda start, stop: array[start:stop]
    for start in range(0, count, 3):
        stop = min(start + 3, count)
        targets = np.zeros((3, height, width), dtype=np.float32)
        targets[: stop - start] = np.asarray(
            dense_batch(start, stop), dtype=np.float32
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


def _prepare_grounded_semantics(
    grounded_masks: np.ndarray | None,
    grounded_labels: np.ndarray | None,
    *,
    height: int,
    width: int,
    class_count: int,
) -> _GroundedSemanticIndex | None:
    if grounded_masks is None and grounded_labels is None:
        return None
    if grounded_masks is None or grounded_labels is None:
        raise ValueError("Grounded masks and labels must both exist or both abstain")
    grounded = np.asarray(grounded_masks, dtype=bool)
    raw_labels = np.asarray(grounded_labels).reshape(-1)
    if grounded.ndim != 3 or grounded.shape[1:] != (int(height), int(width)):
        raise ValueError("Grounded masks must match SAM mask image shape")
    if len(grounded) != len(raw_labels):
        raise ValueError("Grounded masks and labels have different lengths")
    if np.issubdtype(raw_labels.dtype, np.bool_) or not np.issubdtype(
        raw_labels.dtype, np.integer
    ):
        raise ValueError("Grounded labels must use an integer dtype")
    labels = raw_labels.astype(np.int32, copy=False)
    if np.any(labels < 0) or np.any(labels >= int(class_count)):
        raise ValueError("Grounded class ID is outside the complete class vocabulary")
    pixels = int(height) * int(width)
    packed = np.packbits(grounded.reshape(len(grounded), pixels), axis=1)
    return _GroundedSemanticIndex(
        packed_masks=packed,
        mask_area=_popcount_rows(packed),
        labels=labels,
        height=int(height),
        width=int(width),
        class_count=int(class_count),
    )


def _class_probabilities_against_index(
    sam_masks: np.ndarray,
    grounded: _GroundedSemanticIndex | None,
    *,
    iou_threshold: float,
    chunk_size: int,
    class_count: int,
) -> tuple[np.ndarray, bool]:
    sam = np.asarray(sam_masks, dtype=bool)
    if sam.ndim != 3:
        raise ValueError("SAM masks must be MxHxW")
    result = np.zeros((len(sam), int(class_count)), dtype=np.float32)
    if grounded is None:
        return result, True
    if (
        sam.shape[1:] != (grounded.height, grounded.width)
        or int(class_count) != grounded.class_count
    ):
        raise ValueError("SAM and Grounded semantic geometry/classes disagree")
    pixels = grounded.height * grounded.width
    sam_packed = np.packbits(sam.reshape(len(sam), pixels), axis=1)
    sam_area = _popcount_rows(sam_packed)
    for start in range(0, len(sam), max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), len(sam))
        for local, packed in enumerate(sam_packed[start:stop]):
            intersection = _popcount_rows(
                np.bitwise_and(grounded.packed_masks, packed[None, :])
            )
            union = sam_area[start + local] + grounded.mask_area - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection, dtype=np.float64),
                where=union > 0,
            )
            for class_id in np.unique(grounded.labels):
                class_iou = iou[grounded.labels == class_id]
                if class_iou.size:
                    score = float(np.max(class_iou))
                    if score >= float(iou_threshold):
                        result[start + local, int(class_id)] = score
    normalizer = result.sum(axis=1, keepdims=True)
    np.divide(result, normalizer, out=result, where=normalizer > 0)
    return result, False


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
    index = _prepare_grounded_semantics(
        grounded_masks,
        grounded_labels,
        height=sam.shape[1],
        width=sam.shape[2],
        class_count=int(class_count),
    )
    return _class_probabilities_against_index(
        sam,
        index,
        iou_threshold=float(iou_threshold),
        chunk_size=int(chunk_size),
        class_count=int(class_count),
    )


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
    """Compatibility helper that expands one complete frame.

    Production rendering uses :func:`load_packed_sam_frame` and never calls
    this eager helper.
    """

    frame = load_packed_sam_frame(root, image_name, height, width)
    return frame.dense_batch(0, frame.count)


def load_packed_sam_frame(
    root: Path, image_name: str, height: int, width: int
) -> PackedMaskFrame:
    path = Path(root) / f"{image_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing SAM-everything mask frame: {path}")
    return load_packed_mask_frame(path, height=height, width=width)


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


def _load_cameras(
    inputs: CleanSceneInputs,
    *,
    frame_ids: frozenset[int] | None = None,
) -> Iterator[Any]:
    """Yield exactly one registered camera at a time.

    The historical helper materialised every RGB image and every Grounded-SAM
    tensor before rendering the first frame.  On a long ScanNet sequence that
    can exhaust both host and GPU memory.  This iterator keeps only COLMAP's
    small pose dictionaries resident and loads one image/mask/label triple per
    yield.
    """

    from PIL import Image
    from scene.colmap_loader import (
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from scene.dataset_readers import readColmapCameras
    from utils.camera_utils import loadCam

    try:
        extrinsics = read_extrinsics_binary(str(inputs.sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(inputs.sparse / "cameras.bin"))
    except (FileNotFoundError, OSError):
        extrinsics = read_extrinsics_text(str(inputs.sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(inputs.sparse / "cameras.txt"))
    expected = colmap_frame_specs(inputs.sparse)
    expected_names = tuple(frame.image_name for frame in expected)
    ordered_extrinsics = sorted(
        extrinsics.items(),
        key=lambda item: str(
            Path(str(item[1].name).replace("\\", "/")).with_suffix("")
        ).replace("\\", "/"),
    )
    actual_names = tuple(
        str(
            Path(str(extrinsic.name).replace("\\", "/")).with_suffix("")
        ).replace("\\", "/")
        for _, extrinsic in ordered_extrinsics
    )
    if actual_names != expected_names or len(ordered_extrinsics) != len(expected):
        raise ValueError(
            "rendered camera order/names differ from the COLMAP registration"
        )
    args = SimpleNamespace(resolution=1, data_device="cuda")
    for stable_id, ((key, extrinsic), frame) in enumerate(
        zip(ordered_extrinsics, expected, strict=True)
    ):
        if frame_ids is not None and stable_id not in frame_ids:
            continue
        image_path = inputs.images / frame.relative_image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"missing exact COLMAP image: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (frame.width, frame.height):
                raise ValueError(
                    f"{frame.image_name}: image dimensions differ from COLMAP"
                )
        infos = readColmapCameras(
            {key: extrinsic},
            intrinsics,
            str(inputs.images),
            masks_folder=str(inputs.grounded_masks),
            labels_folder=str(inputs.grounded_labels),
        )
        if len(infos) != 1:
            raise RuntimeError(f"{frame.image_name}: camera loader did not return one frame")
        info = infos[0]
        try:
            camera = loadCam(args, stable_id, info, 1)
        finally:
            if getattr(info, "image", None) is not None:
                info.image.close()
        if (int(camera.image_height), int(camera.image_width)) != (
            frame.height,
            frame.width,
        ) or str(camera.image_name) != frame.image_name:
            raise ValueError(f"{frame.image_name}: loaded camera disagrees with COLMAP")
        yield camera


def render_frame_evidence(
    camera: Any,
    gaussians: Any,
    pipeline: Any,
    background: Any,
    sam_masks: np.ndarray | PackedMaskFrame,
    *,
    render_mask_fn: Callable[..., Mapping[str, Any]] | None = None,
    class_count: int = 32,
    mask_observation_mode: str = "hierarchy",
) -> RenderedFrameEvidence:
    """Render one frame and immediately reduce it to sparse mask evidence."""

    import torch

    observation_mode = str(mask_observation_mode)
    if observation_mode not in {"hierarchy", "flat-highest-quality"}:
        raise ValueError(
            "mask_observation_mode must be hierarchy or flat-highest-quality"
        )

    if render_mask_fn is None:
        from gaussian_renderer import render_mask as render_mask_fn
    height, width = int(camera.image_height), int(camera.image_width)
    if isinstance(sam_masks, PackedMaskFrame):
        if (sam_masks.height, sam_masks.width) != (height, width):
            raise ValueError("SAM masks must match the rendered camera")
        mask_count = sam_masks.count
        masks: np.ndarray | PackedMaskFrame = sam_masks
    else:
        dense_masks = np.asarray(sam_masks, dtype=bool)
        if dense_masks.ndim != 3 or dense_masks.shape[1:] != (height, width):
            raise ValueError("SAM masks must match the rendered camera")
        mask_count = len(dense_masks)
        masks = dense_masks
    point_count = int(gaussians.get_xyz.shape[0])
    batches: Iterator[_MaskBatch]
    if mask_count:
        batches = iter_mask_batches(masks)
    else:
        batches = iter(
            (_MaskBatch((), np.zeros((3, height, width), dtype=np.float32)),)
        )
    visible_mass: np.ndarray | None = None
    valid_pixel_count = 0
    support_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    semantic_rows: list[np.ndarray] = []
    grounded_masks, grounded_labels = _load_grounded(camera)
    grounded_index = _prepare_grounded_semantics(
        grounded_masks,
        grounded_labels,
        height=height,
        width=width,
        class_count=int(class_count),
    )
    grounded_abstained = grounded_index is None
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
            class_rows, batch_abstained = _class_probabilities_against_index(
                batch.targets[: len(batch.indices)] > 0.5,
                grounded_index,
                iou_threshold=0.25,
                chunk_size=8,
                class_count=class_count,
            )
            if bool(batch_abstained) != grounded_abstained:
                raise RuntimeError("Grounded-SAM abstention changed within one frame")
            semantic_rows.extend(class_rows)
    if visible_mass is None:
        raise RuntimeError("alpha-mass visibility gradient was not produced")
    if len(support_rows) != mask_count or len(semantic_rows) != mask_count:
        raise RuntimeError("mask streaming lost or duplicated an observation row")
    if observation_mode == "flat-highest-quality" and support_rows:
        # P-flat masks are ordered from highest to lowest frozen SAM quality.
        # A Gaussian footprint may nevertheless cross multiple disjoint pixel
        # masks.  Resolve that residual overlap by the largest measured
        # inside/visible ratio; an exact tie keeps the earlier (better-quality)
        # mask row.  This is streaming/sparse and never materialises MxN mass.
        from .mask_contract import make_sparse_support_exclusive

        support_rows = list(
            make_sparse_support_exclusive(
                [row[0] for row in support_rows],
                [row[1] for row in support_rows],
                [row[2] for row in support_rows],
                point_count=point_count,
            )
        )
        ambiguous = tuple(np.empty(0, dtype=np.int32) for _ in support_rows)
    else:
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
            class_probabilities=np.asarray(semantic_rows[index], dtype=np.float32),
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
        grounded_abstained=grounded_abstained,
        valid_pixel_count=valid_pixel_count,
    )


def render_scene_frames(
    inputs: CleanSceneInputs,
    *,
    classes: Sequence[str] = DEFAULT_CLASSES,
    mask_observation_mode: str = "hierarchy",
    frame_ids: Sequence[int] | None = None,
    frame_callback: Callable[[np.ndarray, RenderedFrameEvidence], None] | None = None,
) -> tuple[np.ndarray, tuple[RenderedFrameEvidence, ...]]:
    """Load one immutable 30k scene and render selected frames sequentially.

    ``frame_ids`` preserves the stable COLMAP row IDs; it never repacks a
    resumed subset.  ``frame_callback`` runs immediately after each sparse
    frame has been reduced, before the next image is loaded.  Callers can
    therefore persist a durable frame checkpoint without ever writing a
    pixel-by-Gaussian contributor cache.  Omitting both arguments retains the
    original all-frame API and return value.
    """

    import torch
    from scene import GaussianModel

    gaussians = GaussianModel(0)
    gaussians.load_ply(str(inputs.rgb_ply))
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    frame_count = len(colmap_frame_specs(inputs.sparse))
    if frame_ids is None:
        selected: tuple[int, ...] | None = None
        selected_set: frozenset[int] | None = None
    else:
        raw_ids = tuple(frame_ids)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_ids
        ):
            raise TypeError("frame_ids must contain only integer frame IDs")
        selected = tuple(int(value) for value in raw_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("frame_ids must be unique")
        if any(value < 0 or value >= frame_count for value in selected):
            raise ValueError("frame_ids contain an out-of-range COLMAP row ID")
        selected = tuple(sorted(selected))
        selected_set = frozenset(selected)
    records: list[RenderedFrameEvidence] = []
    for camera in _load_cameras(inputs, frame_ids=selected_set):
        stable_frame_id = int(camera.uid)
        masks = load_packed_sam_frame(
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
            mask_observation_mode=mask_observation_mode,
        )
        stable_record = RenderedFrameEvidence(
            frame_id=stable_frame_id,
            image_name=record.image_name,
            visible_ids=record.visible_ids,
            visible_mass=record.visible_mass,
            masks=record.masks,
            grounded_abstained=record.grounded_abstained,
            valid_pixel_count=record.valid_pixel_count,
        )
        if frame_callback is not None:
            frame_callback(xyz, stable_record)
        records.append(stable_record)
    expected_ids = tuple(range(frame_count)) if selected is None else selected
    actual_ids = tuple(record.frame_id for record in records)
    if actual_ids != expected_ids:
        raise RuntimeError(
            "scene worker did not render exactly the requested stable frame IDs"
        )
    return xyz, tuple(records)
