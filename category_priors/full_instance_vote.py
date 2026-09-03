from __future__ import annotations

"""Reusable, GT-free 2-D votes for frozen Gaussian instance partitions.

The historical postprocessor renders the scene again every time it needs to
vote on a different instance partition.  This module renders each labelled
frame once and reduces the corrected maximum-contributor images to sparse
per-Gaussian counts for the 32 foreground classes plus background.  Any
partition over the same Gaussian ordering can then be voted on with NumPy.

This is deliberately *not* a pixel contributor cache: no pixel ID, weight,
mask, or image is persisted.  The only durable payload is a compressed CSR
table ``Gaussian x vote-channel``.
"""

import io
import json
import os
import subprocess
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .io import sha256_file


VOTE_EVIDENCE_SCHEMA = "saga-full-instance-gaussian-votes-v1"
DEFAULT_CLASSES_32: tuple[str, ...] = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


@dataclass(frozen=True)
class GaussianVoteAssets:
    scene_id: str
    base_path: Path
    rgb_ply: Path
    feature_ply: Path
    sparse: Path
    images: Path
    masks: Path
    labels: Path
    resolution: int
    white_background: bool


@dataclass(frozen=True)
class GaussianVoteEvidence:
    """Sparse CSR counts with one row per Gaussian and 33 vote channels."""

    row_offsets: np.ndarray
    channels: np.ndarray
    counts: np.ndarray
    metadata: Mapping[str, Any]

    @property
    def point_count(self) -> int:
        return int(self.metadata["point_count"])

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.metadata["class_names"])

    @property
    def channel_count(self) -> int:
        return len(self.class_names) + 1

    @property
    def background_index(self) -> int:
        return len(self.class_names)


def _resolve_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str | Path
) -> Path:
    if "base_path" not in scene:
        raise ValueError("runtime scene is missing base_path")
    base = Path(str(scene["base_path"])).resolve()
    value: Any = default
    for key in keys:
        if scene.get(key) not in (None, ""):
            value = scene[key]
            break
    path = Path(str(value))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _default_rgb_ply(base: Path) -> Path:
    candidates = (
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_vote_assets(
    scene_id: str, scene_mapping: Mapping[str, Any], *, require_exists: bool = True
) -> GaussianVoteAssets:
    """Resolve exactly the assets used by the current 32-class T1 path."""

    if not scene_id or Path(scene_id).name != scene_id:
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    mapped_scene_id = scene_mapping.get("scene_id")
    if mapped_scene_id not in (None, scene_id):
        raise ValueError(
            f"scene mapping identifies {mapped_scene_id!r}, expected {scene_id!r}"
        )
    base = Path(str(scene_mapping["base_path"])).resolve()
    rgb = (
        _resolve_path(scene_mapping, ("point_cloud_path", "gaussian_ply"), "")
        if scene_mapping.get("point_cloud_path") or scene_mapping.get("gaussian_ply")
        else _default_rgb_ply(base).resolve()
    )
    result = GaussianVoteAssets(
        scene_id=scene_id,
        base_path=base,
        rgb_ply=rgb,
        feature_ply=_resolve_path(
            scene_mapping,
            (
                "contrastive_feature_point_cloud_path",
                "feature_point_cloud_path",
                "feature_ply_path",
                "feature_ply",
            ),
            "saga/contrastive_feature_point_cloud.ply",
        ),
        sparse=_resolve_path(
            scene_mapping, ("sparse_path",), "fastRecon/dense/sparse/0"
        ),
        images=_resolve_path(
            scene_mapping,
            ("images_path",),
            "fastRecon/dense/sparse/0/images",
        ),
        masks=_resolve_path(
            scene_mapping,
            ("grounded_masks_path", "masks_path"),
            "saga/masks",
        ),
        labels=_resolve_path(
            scene_mapping,
            ("grounded_labels_path", "labels_path"),
            "saga/labels",
        ),
        resolution=int(scene_mapping.get("resolution", -1)),
        white_background=bool(scene_mapping.get("white_background", False)),
    )
    if require_exists:
        missing = [
            str(path)
            for path in (
                result.rgb_ply,
                result.feature_ply,
                result.sparse,
                result.images,
                result.masks,
                result.labels,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"{scene_id}: missing vote assets: {missing}")
        if not result.rgb_ply.is_file() or not result.feature_ply.is_file():
            raise ValueError(f"{scene_id}: Gaussian assets must be PLY files")
        for path in (result.sparse, result.images, result.masks, result.labels):
            if not path.is_dir():
                raise ValueError(f"{scene_id}: expected directory: {path}")
    return result


def validate_rgb_feature_order(
    rgb_xyz: np.ndarray, feature_xyz: np.ndarray, *, atol: float = 1e-6
) -> None:
    """Reject a vote table that cannot index the frozen feature partition."""

    rgb = np.asarray(rgb_xyz, dtype=np.float64)
    feature = np.asarray(feature_xyz, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1:] != (3,):
        raise ValueError("RGB Gaussian XYZ must have shape (N, 3)")
    if feature.ndim != 2 or feature.shape[1:] != (3,):
        raise ValueError("feature Gaussian XYZ must have shape (N, 3)")
    if rgb.shape != feature.shape:
        raise ValueError(
            f"RGB/feature Gaussian counts differ: {len(rgb)} != {len(feature)}"
        )
    if not np.isfinite(rgb).all() or not np.isfinite(feature).all():
        raise ValueError("Gaussian XYZ contains non-finite values")
    if not np.allclose(rgb, feature, rtol=0.0, atol=float(atol)):
        difference = float(np.max(np.abs(rgb - feature)))
        raise ValueError(
            "RGB and feature Gaussian XYZ/order differ "
            f"(maximum absolute difference {difference:.9g})"
        )


def _normalise_frame_inputs(
    contributor_ids: np.ndarray,
    contribution_weights: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(contributor_ids)
    weights = np.asarray(contribution_weights, dtype=np.float64)
    mask_array = np.asarray(masks, dtype=bool)
    label_array = np.asarray(labels).reshape(-1)
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("contributor IDs and weights must be matching HxW arrays")
    if not np.issubdtype(ids.dtype, np.integer):
        if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError("contributor IDs must be integral")
        ids = ids.astype(np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    if mask_array.ndim != 3 or mask_array.shape[1:] != ids.shape:
        raise ValueError("masks must have shape (M, H, W) matching contributors")
    if len(mask_array) != len(label_array):
        raise ValueError(
            f"mask/label length mismatch: {len(mask_array)} != {len(label_array)}"
        )
    return ids, weights, mask_array, label_array


def frame_vote_updates(
    contributor_ids: np.ndarray,
    contribution_weights: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    point_count: int,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return reduced flat-index updates for one frame.

    Overlapping masks vote independently, exactly as in the teacher
    postprocessor.  Background is a single vote for pixels outside *every*
    mask, including masks whose label is outside the 32-class vocabulary.
    Invalid IDs and zero/non-finite contributor weights never vote.
    """

    if int(point_count) <= 0 or int(class_count) <= 0:
        raise ValueError("point_count and class_count must be positive")
    ids, weights, mask_array, label_array = _normalise_frame_inputs(
        contributor_ids, contribution_weights, masks, labels
    )
    valid = (
        (ids >= 0)
        & (ids < int(point_count))
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    background = ~np.any(mask_array, axis=0) if len(mask_array) else np.ones_like(valid)
    channel_count = int(class_count) + 1
    packed_parts: list[np.ndarray] = []
    foreground_pixel_votes = 0
    ignored_label_masks = 0
    for mask, raw_label in zip(mask_array, label_array):
        try:
            numeric_label = float(raw_label)
            label = int(numeric_label)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"mask label is not an integer: {raw_label!r}") from exc
        if not np.isfinite(numeric_label) or numeric_label != float(label):
            raise ValueError(f"mask label is not an integer: {raw_label!r}")
        if label < 0 or label >= int(class_count):
            ignored_label_masks += 1
            continue
        selected = ids[mask & valid]
        if selected.size:
            foreground_pixel_votes += int(selected.size)
            packed_parts.append(selected * channel_count + label)
    selected_background = ids[background & valid]
    if selected_background.size:
        packed_parts.append(
            selected_background * channel_count + int(class_count)
        )
    if packed_parts:
        packed = np.concatenate(packed_parts).astype(np.int64, copy=False)
        unique, counts = np.unique(packed, return_counts=True)
        update_counts = counts.astype(np.uint64, copy=False)
    else:
        unique = np.empty(0, dtype=np.int64)
        update_counts = np.empty(0, dtype=np.uint64)
    return unique, update_counts, {
        "pixel_count": int(ids.size),
        "valid_contributor_pixel_count": int(np.count_nonzero(valid)),
        "foreground_pixel_votes": foreground_pixel_votes,
        "background_pixel_votes": int(selected_background.size),
        "total_vote_count": foreground_pixel_votes + int(selected_background.size),
        "mask_count": int(len(mask_array)),
        "ignored_label_mask_count": ignored_label_masks,
    }


def _checked_add_updates(
    accumulator: np.ndarray, indices: np.ndarray, counts: np.ndarray
) -> None:
    target = np.asarray(accumulator)
    update_indices = np.asarray(indices, dtype=np.int64)
    update_counts = np.asarray(counts, dtype=np.uint64)
    if target.ndim != 1 or target.dtype != np.uint64:
        raise ValueError("vote accumulator must be a flat uint64 array")
    if update_indices.shape != update_counts.shape:
        raise ValueError("update indices and counts differ in shape")
    if update_indices.size == 0:
        return
    if np.any(update_indices < 0) or np.any(update_indices >= len(target)):
        raise IndexError("vote update index is out of range")
    if len(np.unique(update_indices)) != len(update_indices):
        raise ValueError("vote updates must already be reduced to unique indices")
    maximum = np.iinfo(np.uint64).max
    if np.any(update_counts > maximum - target[update_indices]):
        raise OverflowError("uint64 Gaussian vote counter would overflow")
    target[update_indices] += update_counts


def accumulate_frame_votes(
    accumulator: np.ndarray,
    contributor_ids: np.ndarray,
    contribution_weights: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    *,
    point_count: int,
    class_count: int,
) -> dict[str, int]:
    """Apply one frame to a flat ``point_count * (class_count + 1)`` table."""

    expected = int(point_count) * (int(class_count) + 1)
    if len(accumulator) != expected:
        raise ValueError(f"vote accumulator has {len(accumulator)} cells, expected {expected}")
    indices, counts, diagnostics = frame_vote_updates(
        contributor_ids,
        contribution_weights,
        masks,
        labels,
        point_count=point_count,
        class_count=class_count,
    )
    _checked_add_updates(accumulator, indices, counts)
    return diagnostics


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _directory_payload_identity(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(path.glob("*.pt"), key=lambda candidate: candidate.name):
        identity = _file_identity(item)
        identity["name"] = item.name
        identity.pop("path", None)
        rows.append(identity)
    return rows


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _vote_implementation_identity() -> dict[str, Any]:
    """Bind cached votes to the Python, CUDA source, and loaded binary.

    A stale max-contributor extension can import successfully while still
    implementing the historical ``alpha*T_new`` bug.  Content identities make
    that impossible to confuse with evidence produced by the corrected source.
    The deployment smoke test separately checks the numerical winner.
    """

    import diff_gaussian_rasterization_max_contributor as contributor_module
    import gaussian_renderer

    root = Path(__file__).resolve().parents[1]
    extension_root = (
        root / "submodules" / "diff-gaussian-rasterization-max-contributor"
    )
    source_paths = (
        Path(__file__).resolve(),
        Path(str(gaussian_renderer.__file__)).resolve(),
        extension_root / "diff_gaussian_rasterization_max_contributor" / "__init__.py",
        extension_root / "rasterize_points.cu",
        extension_root / "cuda_rasterizer" / "forward.cu",
    )
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"vote implementation sources are missing: {missing}")
    binary = Path(str(contributor_module._C.__file__)).resolve()  # type: ignore[attr-defined]
    if not binary.is_file():
        raise FileNotFoundError(f"max-contributor extension binary is missing: {binary}")
    return {
        "contract": "corrected-max-alpha-times-t-prev-v1",
        "repository_git_commit": _git_head(root),
        "sources": [_file_identity(path) for path in source_paths],
        "loaded_extension_binary": _file_identity(binary),
    }


def _evidence_from_dense(
    dense_flat: np.ndarray,
    *,
    point_count: int,
    class_names: Sequence[str],
    metadata: Mapping[str, Any],
) -> GaussianVoteEvidence:
    class_tuple = tuple(str(value) for value in class_names)
    channel_count = len(class_tuple) + 1
    flat = np.asarray(dense_flat)
    if flat.dtype != np.uint64 or flat.shape != (int(point_count) * channel_count,):
        raise ValueError("dense vote table has the wrong shape or dtype")
    nonzero = np.flatnonzero(flat).astype(np.int64, copy=False)
    rows = nonzero // channel_count
    row_sizes = np.bincount(rows, minlength=int(point_count)).astype(np.int64)
    row_offsets = np.empty(int(point_count) + 1, dtype=np.int64)
    row_offsets[0] = 0
    np.cumsum(row_sizes, out=row_offsets[1:])
    channels = (nonzero % channel_count).astype(np.uint8)
    counts = flat[nonzero].astype(np.uint64, copy=True)
    result = GaussianVoteEvidence(
        row_offsets=row_offsets,
        channels=channels,
        counts=counts,
        metadata=dict(metadata),
    )
    validate_vote_evidence(result)
    return result


def validate_vote_evidence(evidence: GaussianVoteEvidence) -> None:
    metadata = evidence.metadata
    if metadata.get("schema") != VOTE_EVIDENCE_SCHEMA:
        raise ValueError("unknown Gaussian vote evidence schema")
    point_count = int(metadata.get("point_count", -1))
    class_names = metadata.get("class_names")
    if point_count <= 0 or not isinstance(class_names, list) or not class_names:
        raise ValueError("vote metadata lacks point_count/class_names")
    channel_count = len(class_names) + 1
    if int(metadata.get("channel_count", -1)) != channel_count:
        raise ValueError("vote metadata has an inconsistent channel count")
    if int(metadata.get("background_index", -1)) != len(class_names):
        raise ValueError("vote metadata has an inconsistent background channel")
    offsets = np.asarray(evidence.row_offsets)
    channels = np.asarray(evidence.channels)
    counts = np.asarray(evidence.counts)
    if offsets.dtype != np.int64 or offsets.shape != (point_count + 1,):
        raise ValueError("CSR row_offsets has the wrong shape or dtype")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] != len(counts):
        raise ValueError("CSR row_offsets is invalid")
    if channels.dtype != np.uint8 or channels.shape != counts.shape:
        raise ValueError("CSR channels has the wrong shape or dtype")
    if counts.dtype != np.uint64 or np.any(counts == 0):
        raise ValueError("CSR counts must be positive uint64 values")
    if len(channels) and int(channels.max()) >= channel_count:
        raise ValueError("CSR vote channel is out of range")
    for start, stop in zip(offsets[:-1], offsets[1:]):
        if stop - start > 1 and np.any(np.diff(channels[start:stop].astype(np.int16)) <= 0):
            raise ValueError("CSR channels must be strictly ordered within each row")
    total = sum(int(value) for value in counts)
    if total != int(metadata.get("total_vote_count", -1)):
        raise ValueError("vote count total does not match metadata")
    if total > np.iinfo(np.uint64).max:
        raise OverflowError("total Gaussian votes exceed uint64")


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(
        target, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(
                payload, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return target.getvalue()


def save_gaussian_vote_evidence(
    path: str | Path, evidence: GaussianVoteEvidence
) -> Path:
    """Atomically write byte-deterministic compressed evidence."""

    validate_vote_evidence(evidence)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "channels": np.asarray(evidence.channels, dtype=np.uint8),
        "counts": np.asarray(evidence.counts, dtype=np.uint64),
        "metadata_json": np.asarray(_canonical_json(dict(evidence.metadata))),
        "row_offsets": np.asarray(evidence.row_offsets, dtype=np.int64),
    }
    payload = _deterministic_npz_bytes(arrays)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_gaussian_vote_evidence(
    path: str | Path, *, expected_identity: Mapping[str, Any] | None = None
) -> GaussianVoteEvidence:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as payload:
        expected_keys = {"channels", "counts", "metadata_json", "row_offsets"}
        if set(payload.files) != expected_keys:
            raise ValueError(f"vote evidence has unexpected arrays: {payload.files}")
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
        evidence = GaussianVoteEvidence(
            row_offsets=np.asarray(payload["row_offsets"]).copy(),
            channels=np.asarray(payload["channels"]).copy(),
            counts=np.asarray(payload["counts"]).copy(),
            metadata=metadata,
        )
    validate_vote_evidence(evidence)
    if expected_identity is not None and _canonical_json(
        dict(evidence.metadata.get("input_identity", {}))
    ) != _canonical_json(dict(expected_identity)):
        raise ValueError("vote evidence was produced from different runtime inputs")
    return evidence


def vote_evidence_is_complete(
    path: str | Path, *, expected_identity: Mapping[str, Any] | None = None
) -> bool:
    try:
        load_gaussian_vote_evidence(path, expected_identity=expected_identity)
    except (
        EOFError,
        FileNotFoundError,
        OSError,
        OverflowError,
        KeyError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return False
    return True


def aggregate_instance_votes(
    labels: np.ndarray, evidence: GaussianVoteEvidence
) -> dict[int, np.ndarray]:
    """Aggregate the 33-channel evidence for any frozen instance partition."""

    validate_vote_evidence(evidence)
    point_labels = np.asarray(labels)
    if point_labels.ndim != 1 or len(point_labels) != evidence.point_count:
        raise ValueError(
            f"point labels must have shape ({evidence.point_count},), got {point_labels.shape}"
        )
    if not np.issubdtype(point_labels.dtype, np.integer):
        if not np.isfinite(point_labels).all() or not np.equal(
            point_labels, np.floor(point_labels)
        ).all():
            raise ValueError("point labels must be integral")
    point_labels = point_labels.astype(np.int64, copy=False)
    instance_ids = np.unique(point_labels[point_labels >= 0])
    if not len(instance_ids):
        return {}
    total = sum(int(value) for value in evidence.counts)
    if total > np.iinfo(np.uint64).max:
        raise OverflowError("vote evidence cannot be aggregated without overflow")
    row_ids = np.repeat(
        np.arange(evidence.point_count, dtype=np.int64),
        np.diff(evidence.row_offsets),
    )
    entry_instances = point_labels[row_ids]
    keep = entry_instances >= 0
    result = np.zeros(
        (len(instance_ids), evidence.channel_count), dtype=np.uint64
    )
    if np.any(keep):
        result_rows = np.searchsorted(instance_ids, entry_instances[keep])
        np.add.at(
            result,
            (result_rows, evidence.channels[keep].astype(np.int64)),
            evidence.counts[keep],
        )
    return {
        int(instance_id): result[index].copy()
        for index, instance_id in enumerate(instance_ids)
    }


def _load_torch_payload(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old cloud torch
        return torch.load(path, map_location="cpu")


def _load_cameras(assets: GaussianVoteAssets) -> list[Any]:
    from scene.dataset_readers import (
        readColmapCameras,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from utils.camera_utils import cameraList_from_camInfos

    try:
        infos = readColmapCameras(
            read_extrinsics_binary(str(assets.sparse / "images.bin")),
            read_intrinsics_binary(str(assets.sparse / "cameras.bin")),
            str(assets.images),
        )
    except (FileNotFoundError, OSError, ValueError):
        infos = readColmapCameras(
            read_extrinsics_text(str(assets.sparse / "images.txt")),
            read_intrinsics_text(str(assets.sparse / "cameras.txt")),
            str(assets.images),
        )
    args = SimpleNamespace(resolution=assets.resolution, data_device="cuda")
    return cameraList_from_camInfos(infos, 1, args)


def _resize_masks(masks: Any, height: int, width: int) -> np.ndarray:
    import torch
    import torch.nn.functional as torch_functional

    tensor = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)
    if tensor.ndim != 3:
        raise ValueError(f"mask tensor must have shape (M,H,W), got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (int(height), int(width)):
        tensor = torch_functional.interpolate(
            tensor.float().unsqueeze(1),
            mode="bilinear",
            size=(int(height), int(width)),
            align_corners=False,
        ).squeeze(1) > 0.5
    else:
        tensor = tensor.bool()
    return tensor.detach().cpu().numpy().astype(bool, copy=False)


def _runtime_input_identity(
    assets: GaussianVoteAssets,
    cameras: Sequence[Any],
    class_names: Sequence[str],
    point_count: int,
    implementation_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": assets.scene_id,
        "rgb_ply": _file_identity(assets.rgb_ply),
        "feature_ply": _file_identity(assets.feature_ply),
        "masks_root": str(assets.masks.resolve()),
        "labels_root": str(assets.labels.resolve()),
        "mask_files": _directory_payload_identity(assets.masks),
        "label_files": _directory_payload_identity(assets.labels),
        "classes": [str(value) for value in class_names],
        "point_count": int(point_count),
        "resolution": int(assets.resolution),
        "white_background": bool(assets.white_background),
        "cameras": [
            {
                "image_name": str(camera.image_name),
                "height": int(camera.image_height),
                "width": int(camera.image_width),
                "world_view_transform": camera.world_view_transform.detach().cpu().tolist(),
                "full_proj_transform": camera.full_proj_transform.detach().cpu().tolist(),
            }
            for camera in cameras
        ],
        "implementation": dict(implementation_identity),
    }


def build_gaussian_vote_evidence(
    scene_id: str,
    scene_mapping: Mapping[str, Any],
    output_path: str | Path,
    class_names: Sequence[str] = DEFAULT_CLASSES_32,
    *,
    force: bool = False,
) -> GaussianVoteEvidence:
    """Render and atomically persist one scene's reusable 33-way vote table."""

    import torch
    from gaussian_renderer import render_with_max_contributor
    from scene import GaussianModel

    classes = tuple(str(value) for value in class_names)
    if len(classes) != 32 or len(set(classes)) != 32:
        raise ValueError("full-instance voting requires 32 unique foreground classes")
    assets = resolve_vote_assets(scene_id, scene_mapping)
    from .evaluator import load_ply_xyz

    rgb_xyz = load_ply_xyz(assets.rgb_ply)
    feature_xyz = load_ply_xyz(assets.feature_ply)
    validate_rgb_feature_order(rgb_xyz, feature_xyz)
    point_count = int(len(rgb_xyz))
    cameras = _load_cameras(assets)
    identity = _runtime_input_identity(
        assets,
        cameras,
        classes,
        point_count,
        _vote_implementation_identity(),
    )
    destination = Path(output_path).resolve()
    if not force and vote_evidence_is_complete(destination, expected_identity=identity):
        return load_gaussian_vote_evidence(destination, expected_identity=identity)

    model = GaussianModel(0)
    model.load_ply(str(assets.rgb_ply))
    model_xyz = model.get_xyz.detach().cpu().numpy()
    validate_rgb_feature_order(rgb_xyz, model_xyz)
    pipeline = SimpleNamespace(
        compute_cov3D_python=False, convert_SHs_python=False, debug=False
    )
    background = torch.tensor(
        [1.0, 1.0, 1.0] if assets.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    channel_count = len(classes) + 1
    accumulator = np.zeros(point_count * channel_count, dtype=np.uint64)
    totals = {
        "camera_count": int(len(cameras)),
        "rendered_frame_count": 0,
        "abstained_frame_count": 0,
        "pixel_count": 0,
        "valid_contributor_pixel_count": 0,
        "foreground_pixel_votes": 0,
        "background_pixel_votes": 0,
        "total_vote_count": 0,
        "mask_count": 0,
        "ignored_label_mask_count": 0,
    }
    with torch.no_grad():
        for camera in cameras:
            mask_path = assets.masks / f"{camera.image_name}.pt"
            label_path = assets.labels / f"{camera.image_name}.pt"
            if mask_path.is_file() != label_path.is_file():
                raise ValueError(
                    f"{scene_id}/{camera.image_name}: one-sided mask/label payload"
                )
            if not mask_path.is_file():
                totals["abstained_frame_count"] += 1
                continue
            masks = _resize_masks(
                _load_torch_payload(mask_path), camera.image_height, camera.image_width
            )
            raw_labels = _load_torch_payload(label_path)
            if isinstance(raw_labels, torch.Tensor):
                raw_labels = raw_labels.detach().cpu().numpy()
            labels = np.asarray(raw_labels).reshape(-1)
            if len(masks) != len(labels):
                raise ValueError(
                    f"{scene_id}/{camera.image_name}: "
                    f"{len(masks)} masks != {len(labels)} labels"
                )
            rendered = render_with_max_contributor(
                camera, model, pipeline, background
            )
            if "max_contributor" not in rendered or "max_contribute" not in rendered:
                raise ValueError("renderer lacks corrected contributor ID/weight outputs")
            ids = rendered["max_contributor"].detach().cpu().numpy().reshape(
                camera.image_height, camera.image_width
            )
            weights = rendered["max_contribute"].detach().cpu().numpy().reshape(
                camera.image_height, camera.image_width
            )
            frame = accumulate_frame_votes(
                accumulator,
                ids,
                weights,
                masks,
                labels,
                point_count=point_count,
                class_count=len(classes),
            )
            totals["rendered_frame_count"] += 1
            for key, value in frame.items():
                totals[key] += int(value)
    if sum(int(value) for value in accumulator) != totals["total_vote_count"]:
        raise RuntimeError("streamed vote total disagrees with accumulated table")
    metadata: dict[str, Any] = {
        "schema": VOTE_EVIDENCE_SCHEMA,
        "status": "complete",
        "scene_id": scene_id,
        "point_count": point_count,
        "class_names": list(classes),
        "channel_count": channel_count,
        "background_index": len(classes),
        "total_vote_count": int(totals["total_vote_count"]),
        "diagnostics": totals,
        "input_identity": identity,
        "contract": {
            "contributor": "max(alpha*T_prev)",
            "valid_pixel": "id>=0 && id<point_count && finite(weight) && weight>0",
            "foreground": "each overlapping mask votes independently",
            "background": "valid contributor pixel outside union(all masks)",
            "pixel_contributor_cache_persisted": False,
        },
    }
    evidence = _evidence_from_dense(
        accumulator,
        point_count=point_count,
        class_names=classes,
        metadata=metadata,
    )
    save_gaussian_vote_evidence(destination, evidence)
    return evidence
