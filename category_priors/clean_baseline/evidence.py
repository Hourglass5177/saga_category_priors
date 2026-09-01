from __future__ import annotations

"""Pure NumPy alpha-mask evidence construction and strict persistence."""

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .models import (
    AlphaMaskEvidenceBank,
    AlphaMassFrame,
    DIAGNOSTICS_SCHEMA,
    EVIDENCE_SCHEMA,
    EvidenceThresholds,
    FrameEvidence,
    FrameMetadata,
    MaskMetadata,
    MaskSupportCSR,
    PackedIndexRows,
    PackedVisibility,
)


EVIDENCE_ARRAY_FILE = "evidence.npz"
EVIDENCE_METADATA_FILE = "masks.json"
EVIDENCE_DIAGNOSTICS_FILE = "diagnostics.json"

_DECISION_CANONICAL_VALUE_CONTRACT = "thresholded-membership-indicator"
_VALUE_CONTRACT_SOURCE_KEYS = {
    "evidence_values",
    "continuous_alpha_mass_persisted",
}

# Durable resume data lives beside, never inside, the three-file evidence
# bank.  The final files above remain the only completion contract consumed by
# downstream code.  Checkpoints contain only already-reduced sparse evidence;
# per-pixel contributors are neither serialised nor reconstructable from them.
_CHECKPOINT_DIRECTORY_SUFFIX = ".frame-evidence"
_SCENE_CHECKPOINT_SCHEMA = "saga-clean-scene-evidence-checkpoint-v1"
_FRAME_CHECKPOINT_SCHEMA = "saga-clean-frame-evidence-checkpoint-v1"
_CHECKPOINT_HEADER_KEY = "header_json"
_SCENE_CHECKPOINT_KEYS = {"xyz", _CHECKPOINT_HEADER_KEY}
_FRAME_CHECKPOINT_PAYLOAD_KEYS = {
    "mask_index",
    "support_indptr",
    "support_gaussian_ids",
    "support_inside_mass",
    "support_inside_ratio",
    "support_ambiguous",
    "visible_indptr",
    "visible_gaussian_ids",
    "visible_mass",
    "ambiguous_gaussian_ids",
    "semantic_posteriors",
    "semantic_abstained",
}
_FRAME_CHECKPOINT_KEYS = _FRAME_CHECKPOINT_PAYLOAD_KEYS | {
    _CHECKPOINT_HEADER_KEY
}

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024

_ARRAY_KEYS = {
    "xyz_m",
    "frame_id",
    "frame_valid_pixel_count",
    "frame_geometry_abstained",
    "frame_semantic_abstained",
    "frame_visible_indptr",
    "frame_visible_gaussian_ids",
    "frame_visible_mass",
    "frame_ambiguous_indptr",
    "frame_ambiguous_gaussian_ids",
    "mask_global_id",
    "mask_frame_id",
    "mask_index",
    "mask_support_indptr",
    "mask_support_gaussian_ids",
    "mask_support_inside_mass",
    "mask_support_inside_ratio",
    "mask_support_ambiguous",
    "mask_semantic_posteriors",
    "mask_semantic_abstained",
}


def _integer_array(value: Any, dtype: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.size == 0:
        return np.asarray(raw, dtype=dtype)
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError(f"{name} must use an integer dtype")
    limits = np.iinfo(np.dtype(dtype))
    if np.any(raw < limits.min) or np.any(raw > limits.max):
        raise ValueError(f"{name} cannot be represented as {np.dtype(dtype)}")
    return np.asarray(raw, dtype=dtype)


def _stream_file_digest(path: Path) -> tuple[int, str]:
    """Hash one immutable producer input without creating a sidecar file."""

    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"evidence input is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise RuntimeError(f"evidence input changed while hashing: {resolved}")
    return int(after.st_size), digest.hexdigest()


def _path_content_identity(path: Path) -> dict[str, Any]:
    """Return a deterministic path/list/content identity for a file tree.

    Only the aggregate digest is persisted in ``masks.json``.  It covers each
    relative path, byte size, and complete file SHA-256, so a changed file or
    a changed camera/frame inventory invalidates resume.  No separate hash or
    contributor-cache artifact is written.
    """

    root = path.resolve(strict=True)
    if root.is_file():
        files = (root,)
        relative_paths = (root.name,)
        kind = "file"
    elif root.is_dir():
        files = tuple(
            sorted(
                (candidate for candidate in root.rglob("*") if candidate.is_file()),
                key=lambda candidate: candidate.relative_to(root).as_posix(),
            )
        )
        relative_paths = tuple(
            candidate.relative_to(root).as_posix() for candidate in files
        )
        kind = "directory"
    else:
        raise ValueError(f"unsupported evidence input path: {root}")
    manifest = hashlib.sha256()
    total_bytes = 0
    for relative, candidate in zip(relative_paths, files, strict=True):
        size, content_digest = _stream_file_digest(candidate)
        encoded = relative.encode("utf-8")
        manifest.update(len(encoded).to_bytes(8, "big"))
        manifest.update(encoded)
        manifest.update(size.to_bytes(8, "big"))
        manifest.update(bytes.fromhex(content_digest))
        total_bytes += size
    if root.is_dir():
        after_paths = tuple(
            candidate.relative_to(root).as_posix()
            for candidate in sorted(
                (candidate for candidate in root.rglob("*") if candidate.is_file()),
                key=lambda candidate: candidate.relative_to(root).as_posix(),
            )
        )
        if after_paths != relative_paths:
            raise RuntimeError(f"evidence input inventory changed while hashing: {root}")
    return {
        "path": str(root),
        "kind": kind,
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "relative_paths": list(relative_paths),
        "manifest_sha256": manifest.hexdigest(),
    }


def _colmap_camera_content_identity(sparse: Path) -> dict[str, Any]:
    """Bind the exact camera files selected by the worker's COLMAP loader."""

    root = sparse.resolve(strict=True)
    binary = (root / "images.bin", root / "cameras.bin")
    text = (root / "images.txt", root / "cameras.txt")
    selected = binary if all(path.is_file() for path in binary) else text
    if not all(path.is_file() for path in selected):
        raise FileNotFoundError(
            f"missing complete COLMAP images/cameras pair under {root}"
        )
    manifest = hashlib.sha256()
    total_bytes = 0
    relative_paths: list[str] = []
    for path in selected:
        relative = path.relative_to(root).as_posix()
        size, content_digest = _stream_file_digest(path)
        encoded = relative.encode("utf-8")
        manifest.update(len(encoded).to_bytes(8, "big"))
        manifest.update(encoded)
        manifest.update(size.to_bytes(8, "big"))
        manifest.update(bytes.fromhex(content_digest))
        total_bytes += size
        relative_paths.append(relative)
    return {
        "path": str(root),
        "kind": "colmap-camera-files",
        "file_count": len(selected),
        "total_bytes": int(total_bytes),
        "relative_paths": relative_paths,
        "manifest_sha256": manifest.hexdigest(),
    }


def _selected_tree_content_identity(
    root: Path,
    relative_paths: Sequence[str],
    *,
    kind: str,
    allow_missing: bool,
) -> dict[str, Any]:
    """Bind only the registered files that the worker can actually consume.

    Hashing an entire directory used to include stale ``.part`` files and
    unrelated exports while still failing to describe which COLMAP frames
    were absent.  This selected manifest records every expected relative path
    and an explicit present/absent marker, then hashes present files in full.
    """

    base = root.resolve()
    if base.exists() and not base.is_dir():
        raise ValueError(f"evidence input root is not a directory: {base}")
    normalized: list[str] = []
    for value in relative_paths:
        relative = Path(str(value))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"evidence input path must be relative: {value}")
        canonical = relative.as_posix()
        if not canonical or canonical in normalized:
            raise ValueError("selected evidence paths must be non-empty and unique")
        normalized.append(canonical)
    manifest = hashlib.sha256()
    present: list[str] = []
    missing: list[str] = []
    total_bytes = 0
    for relative in normalized:
        encoded = relative.encode("utf-8")
        manifest.update(len(encoded).to_bytes(8, "big"))
        manifest.update(encoded)
        candidate = base / Path(relative)
        if not candidate.exists():
            if not allow_missing:
                raise FileNotFoundError(f"missing registered evidence input: {candidate}")
            manifest.update(b"\x00")
            missing.append(relative)
            continue
        if not candidate.is_file():
            raise ValueError(f"registered evidence input is not a file: {candidate}")
        size, content_digest = _stream_file_digest(candidate)
        manifest.update(b"\x01")
        manifest.update(size.to_bytes(8, "big"))
        manifest.update(bytes.fromhex(content_digest))
        present.append(relative)
        total_bytes += size
    return {
        "path": str(base),
        "kind": kind,
        "expected_file_count": len(normalized),
        "file_count": len(present),
        "missing_file_count": len(missing),
        "total_bytes": int(total_bytes),
        "relative_paths": present,
        "missing_relative_paths": missing,
        "manifest_sha256": manifest.hexdigest(),
    }


def _producer_input_identity(inputs: Any) -> dict[str, Any]:
    from .sam_inputs import colmap_frame_specs

    frames = colmap_frame_specs(Path(inputs.sparse))
    image_paths = [frame.relative_image_path for frame in frames]
    sam_paths = [f"{frame.image_name}.npz" for frame in frames]
    grounded_paths = [f"{frame.image_name}.pt" for frame in frames]
    images = _selected_tree_content_identity(
        Path(inputs.images), image_paths, kind="registered-images", allow_missing=False
    )
    sam = _selected_tree_content_identity(
        Path(inputs.sam_masks), sam_paths, kind="registered-sam-masks", allow_missing=False
    )
    grounded_masks = _selected_tree_content_identity(
        Path(inputs.grounded_masks),
        grounded_paths,
        kind="registered-grounded-masks",
        allow_missing=True,
    )
    grounded_labels = _selected_tree_content_identity(
        Path(inputs.grounded_labels),
        grounded_paths,
        kind="registered-grounded-labels",
        allow_missing=True,
    )
    if grounded_masks["missing_relative_paths"] != grounded_labels["missing_relative_paths"]:
        raise ValueError(
            "Grounded-SAM mask/label inputs must both exist or both abstain per frame"
        )
    return {
        "schema": "saga-clean-evidence-input-content-v2",
        "gaussian_ply": _path_content_identity(Path(inputs.rgb_ply)),
        "colmap_cameras": _colmap_camera_content_identity(Path(inputs.sparse)),
        "image_inputs": images,
        "sam_everything_masks": sam,
        "grounded_masks": grounded_masks,
        "grounded_labels": grounded_labels,
    }


def accumulate_alpha_mass_from_contributors(
    alpha: np.ndarray,
    transmittance_prev: np.ndarray,
    gaussian_ids: np.ndarray,
    masks: np.ndarray | None,
    point_count: int,
    *,
    valid_pixels: np.ndarray | None = None,
) -> AlphaMassFrame:
    """Accumulate normalized ``alpha * T_prev`` mass with pure NumPy.

    The final dimension enumerates actual contributors for each pixel.  Any
    leading dimensions are pixel dimensions (normally ``H, W``).  Positive
    mass with an invalid Gaussian ID is rejected.  A zero-mass ``-1`` entry is
    the canonical empty contributor and is ignored.

    ``masks=None`` is an explicit geometry abstention: visibility is retained,
    while no foreground or background mask evidence is created.
    """

    alpha_array = np.asarray(alpha, dtype=np.float64)
    transmittance = np.asarray(transmittance_prev, dtype=np.float64)
    ids = np.asarray(gaussian_ids)
    points = int(point_count)
    if alpha_array.ndim < 2 or alpha_array.shape[-1] <= 0:
        raise ValueError("contributors must have pixel dimensions plus K")
    if transmittance.shape != alpha_array.shape or ids.shape != alpha_array.shape:
        raise ValueError("alpha, transmittance_prev, and gaussian_ids must match")
    if points <= 0:
        raise ValueError("point_count must be positive")
    if np.issubdtype(ids.dtype, np.bool_) or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("gaussian_ids must use an integer dtype")
    if (
        np.any(~np.isfinite(alpha_array))
        or np.any(~np.isfinite(transmittance))
        or np.any(alpha_array < 0)
        or np.any(transmittance < 0)
        or np.any(alpha_array > 1 + 1e-7)
        or np.any(transmittance > 1 + 1e-7)
    ):
        raise ValueError("alpha and T_prev must be finite values in [0, 1]")
    pixel_shape = alpha_array.shape[:-1]
    allowed = (
        np.ones(pixel_shape, dtype=bool)
        if valid_pixels is None
        else np.asarray(valid_pixels, dtype=bool)
    )
    if allowed.shape != pixel_shape:
        raise ValueError("valid_pixels must match the contributor pixel dimensions")
    if masks is None:
        mask_array = np.zeros((0, *pixel_shape), dtype=bool)
        abstained = True
    else:
        mask_array = np.asarray(masks, dtype=bool)
        if mask_array.ndim != len(pixel_shape) + 1 or mask_array.shape[1:] != pixel_shape:
            raise ValueError("masks must be mask_count followed by the pixel dimensions")
        abstained = False

    raw_weight = alpha_array * transmittance
    valid_id = (ids >= 0) & (ids < points)
    if np.any((raw_weight > 0) & ~valid_id & allowed[..., None]):
        raise ValueError("positive contributor mass has an invalid Gaussian ID")
    raw_weight = np.where(valid_id & allowed[..., None], raw_weight, 0.0)
    denominator = raw_weight.sum(axis=-1)
    normalized = np.divide(
        raw_weight,
        denominator[..., None],
        out=np.zeros_like(raw_weight),
        where=denominator[..., None] > 0,
    )
    flat_ids = ids.reshape(-1, ids.shape[-1]).astype(np.int64, copy=False)
    flat_mass = normalized.reshape(-1, normalized.shape[-1])
    flat_valid = flat_mass > 0
    visible = np.bincount(
        flat_ids[flat_valid], weights=flat_mass[flat_valid], minlength=points
    ).astype(np.float64, copy=False)
    inside = np.zeros((len(mask_array), points), dtype=np.float64)
    for mask_index, mask in enumerate(mask_array):
        selected = flat_valid & mask.reshape(-1, 1)
        inside[mask_index] = np.bincount(
            flat_ids[selected], weights=flat_mass[selected], minlength=points
        )
    return AlphaMassFrame(
        inside_mass=inside,
        visible_mass=visible,
        valid_pixel_count=int(np.count_nonzero(denominator > 0)),
        geometry_abstained=abstained,
    )


def build_frame_evidence(
    *,
    frame_id: int,
    image_name: str,
    alpha_mass: AlphaMassFrame,
    global_mask_id_start: int = 0,
    global_mask_ids: Sequence[int] | None = None,
    mask_indices: Sequence[int] | None = None,
    semantic_posteriors: np.ndarray | None = None,
    semantic_abstained: Sequence[bool] | np.ndarray | None = None,
    class_count: int = 32,
    thresholds: EvidenceThresholds = EvidenceThresholds(),
) -> FrameEvidence:
    """Threshold one dense accumulation into sparse, ambiguity-aware evidence."""

    mask_count = alpha_mass.mask_count
    classes = int(class_count)
    if classes <= 0:
        raise ValueError("class_count must be positive")
    if global_mask_ids is not None:
        ids = _integer_array(global_mask_ids, np.int64, name="global_mask_ids")
        if ids.shape != (mask_count,):
            raise ValueError("global_mask_ids must have one value per mask")
    else:
        start = int(global_mask_id_start)
        if start < 0:
            raise ValueError("global_mask_id_start must be non-negative")
        ids = np.arange(start, start + mask_count, dtype=np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("global_mask_ids must be unique non-negative integers")
    local = (
        np.arange(mask_count, dtype=np.int32)
        if mask_indices is None
        else _integer_array(mask_indices, np.int32, name="mask_indices")
    )
    if local.shape != (mask_count,) or np.any(local < 0):
        raise ValueError("mask_indices must have one non-negative value per mask")
    if len(np.unique(local)) != len(local) or np.any(np.diff(local) <= 0):
        raise ValueError("mask_indices must be sorted and unique")

    visible_ids = np.flatnonzero(
        alpha_mass.visible_mass >= thresholds.visible_min_mass
    ).astype(np.int32)
    visible_mass = alpha_mass.visible_mass[visible_ids].astype(np.float32)
    ratios = np.divide(
        alpha_mass.inside_mass,
        alpha_mass.visible_mass[None, :],
        out=np.zeros_like(alpha_mass.inside_mass),
        where=alpha_mass.visible_mass[None, :] > 0,
    )
    rows: list[np.ndarray] = []
    row_mass: list[np.ndarray] = []
    row_ratio: list[np.ndarray] = []
    membership_count = np.zeros(alpha_mass.point_count, dtype=np.int32)
    for mask_index in range(mask_count):
        support = np.flatnonzero(
            (alpha_mass.visible_mass >= thresholds.visible_min_mass)
            & (alpha_mass.inside_mass[mask_index] >= thresholds.inside_min_mass)
            & (ratios[mask_index] >= thresholds.inside_min_ratio)
        ).astype(np.int32)
        rows.append(support)
        row_mass.append(alpha_mass.inside_mass[mask_index, support].astype(np.float32))
        row_ratio.append(ratios[mask_index, support].astype(np.float32))
        membership_count[support] += 1
    ambiguous_ids = np.flatnonzero(membership_count > 1).astype(np.int32)
    row_ambiguous = [np.isin(row, ambiguous_ids) for row in rows]
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))

    def concat(values: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
        return (
            np.concatenate([np.asarray(value, dtype=dtype) for value in values])
            if values and int(indptr[-1])
            else np.empty(0, dtype=dtype)
        )

    if semantic_posteriors is None:
        posterior = np.zeros((mask_count, classes), dtype=np.float32)
        abstained = np.ones(mask_count, dtype=bool)
    else:
        posterior = np.asarray(semantic_posteriors, dtype=np.float32)
        if posterior.shape != (mask_count, classes):
            raise ValueError("semantic_posteriors must be mask_count x class_count")
        abstained = (
            posterior.sum(axis=1, dtype=np.float64) <= 1e-7
            if semantic_abstained is None
            else np.asarray(semantic_abstained, dtype=bool)
        )
    if abstained.shape != (mask_count,):
        raise ValueError("semantic_abstained must have one value per mask")
    if semantic_posteriors is None and semantic_abstained is not None:
        supplied = np.asarray(semantic_abstained, dtype=bool)
        if supplied.shape != (mask_count,) or np.any(~supplied):
            raise ValueError("missing semantic posteriors require abstention")
    mask_meta = tuple(
        MaskMetadata(
            global_mask_id=int(ids[index]),
            frame_id=int(frame_id),
            image_name=str(image_name),
            mask_index=int(local[index]),
        )
        for index in range(mask_count)
    )
    metadata = FrameMetadata(
        frame_id=int(frame_id),
        image_name=str(image_name),
        valid_pixel_count=alpha_mass.valid_pixel_count,
        geometry_abstained=alpha_mass.geometry_abstained,
        semantic_abstained=bool(mask_count == 0 or np.all(abstained)),
    )
    return FrameEvidence(
        metadata=metadata,
        masks=mask_meta,
        support=MaskSupportCSR(
            indptr,
            concat(rows, np.int32),
            concat(row_mass, np.float32),
            concat(row_ratio, np.float32),
            concat(row_ambiguous, np.bool_),
            mask_count,
            alpha_mass.point_count,
        ),
        visibility=PackedVisibility(
            np.asarray([0, len(visible_ids)], dtype=np.int64),
            visible_ids,
            visible_mass,
            1,
            alpha_mass.point_count,
        ),
        ambiguous_gaussians=ambiguous_ids,
        semantic_posteriors=posterior,
        semantic_abstained=abstained,
    )


def build_sparse_frame_evidence(
    *,
    frame_id: int,
    image_name: str,
    point_count: int,
    visible_ids: np.ndarray,
    visible_mass: np.ndarray,
    mask_gaussian_ids: Sequence[np.ndarray],
    mask_inside_mass: Sequence[np.ndarray],
    mask_inside_ratio: Sequence[np.ndarray],
    ambiguous_ids: np.ndarray | Sequence[np.ndarray] | None = None,
    semantic_posteriors: np.ndarray | None = None,
    semantic_abstained: Sequence[bool] | np.ndarray | None = None,
    global_mask_id_start: int = 0,
    global_mask_ids: Sequence[int] | None = None,
    mask_indices: Sequence[int] | None = None,
    valid_pixel_count: int = 0,
    geometry_abstained: bool = False,
    class_count: int = 32,
    thresholds: EvidenceThresholds = EvidenceThresholds(),
) -> FrameEvidence:
    """Build a frame from already reduced sparse GPU-worker evidence.

    This is the production counterpart to :func:`build_frame_evidence`.  It
    deliberately accepts no dense ``mask_count x point_count`` array, so a
    renderer can reduce each three-channel mask batch and release it
    immediately.  The same thresholds remain part of the serialized contract;
    sparse rows below those thresholds are rejected rather than silently
    thresholded a second time.
    """

    points = int(point_count)
    if points <= 0:
        raise ValueError("point_count must be positive")
    visible = _integer_array(visible_ids, np.int32, name="visible_ids")
    visible_values = np.asarray(visible_mass, dtype=np.float32)
    if (
        visible.ndim != 1
        or visible_values.shape != visible.shape
        or np.any(visible < 0)
        or np.any(visible >= points)
        or (len(visible) and np.any(np.diff(visible) <= 0))
        or np.any(~np.isfinite(visible_values))
        or np.any(visible_values < thresholds.visible_min_mass)
    ):
        raise ValueError("visible IDs/mass violate the packed visibility contract")
    rows = tuple(
        _integer_array(row, np.int32, name="mask_gaussian_ids")
        for row in mask_gaussian_ids
    )
    masses = tuple(np.asarray(row, dtype=np.float32) for row in mask_inside_mass)
    ratios = tuple(np.asarray(row, dtype=np.float32) for row in mask_inside_ratio)
    count = len(rows)
    if len(masses) != count or len(ratios) != count:
        raise ValueError("sparse mask support arrays must have the same row count")
    membership_count = np.zeros(points, dtype=np.int32)
    for row, mass, ratio in zip(rows, masses, ratios):
        if (
            row.ndim != 1
            or mass.shape != row.shape
            or ratio.shape != row.shape
            or np.any(row < 0)
            or np.any(row >= points)
            or (len(row) and np.any(np.diff(row) <= 0))
            or np.any(~np.isfinite(mass))
            or np.any(~np.isfinite(ratio))
            or np.any(mass < thresholds.inside_min_mass)
            or np.any(ratio < thresholds.inside_min_ratio)
            or np.any(ratio > 1 + 1e-6)
        ):
            raise ValueError("sparse mask row violates support thresholds")
        positions = np.searchsorted(visible, row)
        if len(row) and (
            np.any(positions >= len(visible)) or np.any(visible[positions] != row)
        ):
            raise ValueError("sparse mask support must be visible in its frame")
        if len(row):
            frame_mass = visible_values[positions]
            if np.any(mass - frame_mass > 5e-5 * np.maximum(frame_mass, 1.0)):
                raise ValueError("sparse inside mass exceeds visible mass")
            if not np.allclose(ratio, mass / frame_mass, atol=2e-5, rtol=2e-5):
                raise ValueError("sparse inside ratio disagrees with mass/visibility")
        membership_count[row] += 1
    expected_ambiguous = np.flatnonzero(membership_count > 1).astype(np.int32)
    if ambiguous_ids is not None:
        if isinstance(ambiguous_ids, np.ndarray) and ambiguous_ids.ndim == 1:
            supplied_ambiguous = _integer_array(
                ambiguous_ids, np.int32, name="ambiguous_ids"
            )
        else:
            ambiguity_rows = tuple(
                _integer_array(row, np.int32, name="ambiguous_ids")
                for row in ambiguous_ids
            )
            if len(ambiguity_rows) != count:
                raise ValueError("per-mask ambiguous IDs must match mask rows")
            for row, supplied in zip(rows, ambiguity_rows):
                expected_row = row[np.isin(row, expected_ambiguous)]
                if not np.array_equal(supplied, expected_row):
                    raise ValueError("per-mask ambiguous IDs disagree with support overlap")
            supplied_ambiguous = (
                np.unique(np.concatenate(ambiguity_rows)).astype(np.int32)
                if ambiguity_rows and any(len(row) for row in ambiguity_rows)
                else np.empty(0, dtype=np.int32)
            )
        if not np.array_equal(supplied_ambiguous, expected_ambiguous):
            raise ValueError("ambiguous IDs disagree with same-frame multi-mask support")
    row_flags = [np.isin(row, expected_ambiguous) for row in rows]
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))

    def concat(values: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
        return (
            np.concatenate([np.asarray(value, dtype=dtype) for value in values])
            if values and int(indptr[-1])
            else np.empty(0, dtype=dtype)
        )

    if global_mask_ids is None:
        start = int(global_mask_id_start)
        if start < 0:
            raise ValueError("global_mask_id_start must be non-negative")
        global_ids = np.arange(start, start + count, dtype=np.int64)
    else:
        global_ids = _integer_array(
            global_mask_ids, np.int64, name="global_mask_ids"
        )
    if global_ids.shape != (count,) or np.any(global_ids < 0) or len(np.unique(global_ids)) != count:
        raise ValueError("global mask IDs must be unique, non-negative, and match rows")
    local = (
        np.arange(count, dtype=np.int32)
        if mask_indices is None
        else _integer_array(mask_indices, np.int32, name="mask_indices")
    )
    if (
        local.shape != (count,)
        or np.any(local < 0)
        or len(np.unique(local)) != count
        or (len(local) and np.any(np.diff(local) <= 0))
    ):
        raise ValueError("mask_indices must be sorted unique non-negative values")
    classes = int(class_count)
    if classes <= 0:
        raise ValueError("class_count must be positive")
    if semantic_posteriors is None:
        posterior = np.zeros((count, classes), dtype=np.float32)
        abstained = np.ones(count, dtype=bool)
    else:
        posterior = np.asarray(semantic_posteriors, dtype=np.float32)
        if posterior.shape != (count, classes):
            raise ValueError("semantic_posteriors must be mask_count x class_count")
        abstained = (
            posterior.sum(axis=1, dtype=np.float64) <= 1e-7
            if semantic_abstained is None
            else np.asarray(semantic_abstained, dtype=bool)
        )
    if abstained.shape != (count,):
        raise ValueError("semantic_abstained must have one value per mask")
    if semantic_posteriors is None and semantic_abstained is not None:
        supplied = np.asarray(semantic_abstained, dtype=bool)
        if supplied.shape != (count,) or np.any(~supplied):
            raise ValueError("missing semantic posteriors require abstention")
    if bool(geometry_abstained) and count:
        raise ValueError("an abstained geometry frame cannot contain masks")
    metadata = FrameMetadata(
        frame_id=int(frame_id),
        image_name=str(image_name),
        valid_pixel_count=int(valid_pixel_count),
        geometry_abstained=bool(geometry_abstained),
        semantic_abstained=bool(count == 0 or np.all(abstained)),
    )
    mask_meta = tuple(
        MaskMetadata(
            global_mask_id=int(global_ids[index]),
            frame_id=metadata.frame_id,
            image_name=metadata.image_name,
            mask_index=int(local[index]),
        )
        for index in range(count)
    )
    return FrameEvidence(
        metadata=metadata,
        masks=mask_meta,
        support=MaskSupportCSR(
            indptr,
            concat(rows, np.int32),
            concat(masses, np.float32),
            concat(ratios, np.float32),
            concat(row_flags, np.bool_),
            count,
            points,
        ),
        visibility=PackedVisibility(
            np.asarray([0, len(visible)], dtype=np.int64),
            visible,
            visible_values,
            1,
            points,
        ),
        ambiguous_gaussians=expected_ambiguous,
        semantic_posteriors=posterior,
        semantic_abstained=abstained,
    )


def _metadata(bank: AlphaMaskEvidenceBank) -> dict[str, Any]:
    return {
        "schema": EVIDENCE_SCHEMA,
        "scene_id": bank.scene_id,
        "point_count": bank.point_count,
        "class_names": list(bank.class_names),
        "thresholds": bank.thresholds.to_dict(),
        "source": dict(bank.source),
        "files": {
            "arrays": EVIDENCE_ARRAY_FILE,
            "diagnostics": EVIDENCE_DIAGNOSTICS_FILE,
        },
        "frames": [
            {
                "frame_id": frame.frame_id,
                "image_name": frame.image_name,
                "valid_pixel_count": frame.valid_pixel_count,
                "geometry_abstained": frame.geometry_abstained,
                "semantic_abstained": frame.semantic_abstained,
            }
            for frame in bank.frames
        ],
        "masks": [
            {
                "global_mask_id": mask.global_mask_id,
                "frame_id": mask.frame_id,
                "image_name": mask.image_name,
                "mask_index": mask.mask_index,
            }
            for mask in bank.masks
        ],
    }


def _diagnostics(bank: AlphaMaskEvidenceBank) -> dict[str, Any]:
    result = {
        "schema": DIAGNOSTICS_SCHEMA,
        "scene_id": bank.scene_id,
        "point_count": bank.point_count,
        "frame_count": bank.frame_count,
        "mask_count": bank.mask_count,
        "support_entry_count": int(len(bank.mask_support.gaussian_ids)),
        "positive_support_entry_count": int(
            np.count_nonzero(~bank.mask_support.ambiguous)
        ),
        "ambiguous_support_entry_count": int(
            np.count_nonzero(bank.mask_support.ambiguous)
        ),
        "ambiguous_frame_gaussian_count": int(len(bank.frame_ambiguity.ids)),
        "visible_frame_gaussian_count": int(len(bank.frame_visibility.gaussian_ids)),
        "geometry_abstention_frame_count": int(
            sum(frame.geometry_abstained for frame in bank.frames)
        ),
        "semantic_abstention_frame_count": int(
            sum(frame.semantic_abstained for frame in bank.frames)
        ),
        "semantic_abstention_mask_count": int(np.count_nonzero(bank.semantic_abstained)),
        "valid_pixel_count": int(sum(frame.valid_pixel_count for frame in bank.frames)),
    }
    # Legacy hierarchy banks predate the decision-canonical persistence
    # contract and must remain byte-valid read-only inputs.  New banks declare
    # the contract in their source identity and repeat it in diagnostics so a
    # stored indicator can never be mistaken for measured alpha mass.
    if "evidence_values" in bank.source:
        result["evidence_values"] = str(bank.source["evidence_values"])
        result["continuous_alpha_mass_persisted"] = bool(
            bank.source.get("continuous_alpha_mass_persisted", True)
        )
    return result


def _validate_evidence_value_contract(bank: AlphaMaskEvidenceBank) -> None:
    """Keep legacy mass banks and decision-canonical banks unambiguous.

    Historical hierarchy evidence stores measured continuous alpha mass and
    predates an explicit value-contract discriminator.  New evidence instead
    persists only the already-thresholded membership decisions, represented by
    exact one-valued entries.  A partial or forged discriminator would let one
    representation be interpreted as the other, so it is rejected both before
    persistence and after loading.
    """

    present = _VALUE_CONTRACT_SOURCE_KEYS.intersection(bank.source)
    if not present:
        return
    if present != _VALUE_CONTRACT_SOURCE_KEYS:
        missing = sorted(_VALUE_CONTRACT_SOURCE_KEYS - present)
        raise ValueError(
            "evidence value contract is incomplete; missing source fields: "
            + ", ".join(missing)
        )
    if bank.source["evidence_values"] != _DECISION_CANONICAL_VALUE_CONTRACT:
        raise ValueError("unsupported evidence value contract")
    if bank.source["continuous_alpha_mass_persisted"] is not False:
        raise ValueError(
            "decision-canonical evidence must declare "
            "continuous_alpha_mass_persisted=false"
        )
    indicator_arrays = {
        "frame_visible_mass": bank.frame_visibility.visible_mass,
        "mask_support_inside_mass": bank.mask_support.inside_mass,
        "mask_support_inside_ratio": bank.mask_support.inside_ratio,
    }
    for name, values in indicator_arrays.items():
        if not np.all(np.asarray(values) == 1.0):
            raise ValueError(
                f"{name} must contain exact one-valued membership indicators"
            )


def _arrays(bank: AlphaMaskEvidenceBank) -> dict[str, np.ndarray]:
    return {
        "xyz_m": bank.xyz_m,
        "frame_id": np.asarray([frame.frame_id for frame in bank.frames], dtype=np.int64),
        "frame_valid_pixel_count": np.asarray(
            [frame.valid_pixel_count for frame in bank.frames], dtype=np.int64
        ),
        "frame_geometry_abstained": np.asarray(
            [frame.geometry_abstained for frame in bank.frames], dtype=np.bool_
        ),
        "frame_semantic_abstained": np.asarray(
            [frame.semantic_abstained for frame in bank.frames], dtype=np.bool_
        ),
        "frame_visible_indptr": bank.frame_visibility.indptr,
        "frame_visible_gaussian_ids": bank.frame_visibility.gaussian_ids,
        "frame_visible_mass": bank.frame_visibility.visible_mass,
        "frame_ambiguous_indptr": bank.frame_ambiguity.indptr,
        "frame_ambiguous_gaussian_ids": bank.frame_ambiguity.ids,
        "mask_global_id": np.asarray(
            [mask.global_mask_id for mask in bank.masks], dtype=np.int64
        ),
        "mask_frame_id": np.asarray([mask.frame_id for mask in bank.masks], dtype=np.int64),
        "mask_index": np.asarray([mask.mask_index for mask in bank.masks], dtype=np.int32),
        "mask_support_indptr": bank.mask_support.indptr,
        "mask_support_gaussian_ids": bank.mask_support.gaussian_ids,
        "mask_support_inside_mass": bank.mask_support.inside_mass,
        "mask_support_inside_ratio": bank.mask_support.inside_ratio,
        "mask_support_ambiguous": bank.mask_support.ambiguous,
        "mask_semantic_posteriors": bank.semantic_posteriors,
        "mask_semantic_abstained": bank.semantic_abstained,
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _source_identity_digest(source: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(source)).hexdigest()


def _checkpoint_payload_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash exact reduced arrays inside one checkpoint.

    The digest is embedded in the NPZ header rather than written as a SHA
    sidecar.  It catches a syntactically valid but silently modified NPZ, while
    normal ZIP/NPY corruption is already rejected by ``numpy.load``.
    """

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        encoded_name = name.encode("utf-8")
        encoded_dtype = array.dtype.str.encode("ascii")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(8, "big"))
        digest.update(encoded_dtype)
        digest.update(array.ndim.to_bytes(8, "big"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "big", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encode_checkpoint_header(value: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(_canonical_json_bytes(value), dtype=np.uint8).copy()


def _decode_checkpoint_header(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 1 or len(array) == 0:
        raise ValueError("checkpoint header must be a non-empty uint8 vector")
    decoded = json.loads(array.tobytes().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("checkpoint header must decode to a JSON object")
    return decoded


def _checkpoint_root(output_dir: Path) -> Path:
    return output_dir.with_name(output_dir.name + _CHECKPOINT_DIRECTORY_SUFFIX)


def _read_checkpoint_arrays(
    path: Path, *, expected_keys: set[str]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != expected_keys:
            raise ValueError("checkpoint has missing or unexpected arrays")
        header = _decode_checkpoint_header(loaded[_CHECKPOINT_HEADER_KEY])
        arrays = {
            name: np.asarray(loaded[name]).copy()
            for name in loaded.files
            if name != _CHECKPOINT_HEADER_KEY
        }
    expected_digest = str(header.get("payload_sha256", ""))
    if not expected_digest or _checkpoint_payload_digest(arrays) != expected_digest:
        raise ValueError("checkpoint payload digest does not match its header")
    return header, arrays


def _validate_raw_xyz(value: Any) -> np.ndarray:
    xyz = np.asarray(value, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) <= 0:
        raise ValueError("scene worker returned an invalid Gaussian XYZ array")
    if np.any(~np.isfinite(xyz)):
        raise ValueError("scene worker returned non-finite Gaussian XYZ")
    return xyz


def _save_scene_checkpoint(
    path: Path,
    *,
    scene_id: str,
    source_sha256: str,
    xyz: np.ndarray,
) -> None:
    payload = {"xyz": _validate_raw_xyz(xyz)}
    header = {
        "schema": _SCENE_CHECKPOINT_SCHEMA,
        "scene_id": str(scene_id),
        "source_sha256": str(source_sha256),
        "point_count": int(len(payload["xyz"])),
        "payload_sha256": _checkpoint_payload_digest(payload),
    }
    _write_npz_atomic(
        path, {**payload, _CHECKPOINT_HEADER_KEY: _encode_checkpoint_header(header)}
    )


def _load_scene_checkpoint(
    path: Path,
    *,
    scene_id: str,
    source_sha256: str,
) -> np.ndarray:
    header, arrays = _read_checkpoint_arrays(
        path, expected_keys=_SCENE_CHECKPOINT_KEYS
    )
    if set(header) != {
        "schema",
        "scene_id",
        "source_sha256",
        "point_count",
        "payload_sha256",
    } or type(header.get("point_count")) is not int:
        raise ValueError("scene checkpoint header contract is invalid")
    expected_header = {
        "schema": _SCENE_CHECKPOINT_SCHEMA,
        "scene_id": str(scene_id),
        "source_sha256": str(source_sha256),
        "point_count": int(np.asarray(arrays["xyz"]).shape[0]),
        "payload_sha256": _checkpoint_payload_digest(arrays),
    }
    if header != expected_header:
        raise ValueError("scene checkpoint identity does not match this request")
    return _validate_raw_xyz(arrays["xyz"])


def _frame_checkpoint_payload(frame: FrameEvidence) -> dict[str, np.ndarray]:
    return {
        "mask_index": np.asarray(
            [mask.mask_index for mask in frame.masks], dtype=np.int32
        ),
        "support_indptr": frame.support.indptr,
        "support_gaussian_ids": frame.support.gaussian_ids,
        "support_inside_mass": frame.support.inside_mass,
        "support_inside_ratio": frame.support.inside_ratio,
        "support_ambiguous": frame.support.ambiguous,
        "visible_indptr": frame.visibility.indptr,
        "visible_gaussian_ids": frame.visibility.gaussian_ids,
        "visible_mass": frame.visibility.visible_mass,
        "ambiguous_gaussian_ids": frame.ambiguous_gaussians,
        "semantic_posteriors": frame.semantic_posteriors,
        "semantic_abstained": frame.semantic_abstained,
    }


def _save_frame_checkpoint(
    path: Path,
    *,
    scene_id: str,
    source_sha256: str,
    point_count: int,
    class_count: int,
    frame: FrameEvidence,
) -> None:
    payload = _frame_checkpoint_payload(frame)
    header = {
        "schema": _FRAME_CHECKPOINT_SCHEMA,
        "scene_id": str(scene_id),
        "source_sha256": str(source_sha256),
        "frame_id": int(frame.metadata.frame_id),
        "image_name": str(frame.metadata.image_name),
        "point_count": int(point_count),
        "class_count": int(class_count),
        "mask_count": len(frame.masks),
        "valid_pixel_count": int(frame.metadata.valid_pixel_count),
        "geometry_abstained": bool(frame.metadata.geometry_abstained),
        "semantic_abstained": bool(frame.metadata.semantic_abstained),
        "payload_sha256": _checkpoint_payload_digest(payload),
    }
    _write_npz_atomic(
        path, {**payload, _CHECKPOINT_HEADER_KEY: _encode_checkpoint_header(header)}
    )


def _load_frame_checkpoint(
    path: Path,
    *,
    scene_id: str,
    source_sha256: str,
    frame_id: int,
    image_name: str,
    point_count: int,
    class_count: int,
) -> FrameEvidence:
    header, arrays = _read_checkpoint_arrays(
        path, expected_keys=_FRAME_CHECKPOINT_KEYS
    )
    if (
        set(header)
        != {
            "schema",
            "scene_id",
            "source_sha256",
            "frame_id",
            "image_name",
            "point_count",
            "class_count",
            "mask_count",
            "valid_pixel_count",
            "geometry_abstained",
            "semantic_abstained",
            "payload_sha256",
        }
        or any(
            type(header.get(name)) is not int
            for name in (
                "frame_id",
                "point_count",
                "class_count",
                "mask_count",
                "valid_pixel_count",
            )
        )
        or type(header.get("geometry_abstained")) is not bool
        or type(header.get("semantic_abstained")) is not bool
    ):
        raise ValueError("frame checkpoint header contract is invalid")
    mask_indices = _integer_array(
        arrays["mask_index"], np.int32, name="checkpoint.mask_index"
    )
    mask_count = len(mask_indices)
    expected_identity = {
        "schema": _FRAME_CHECKPOINT_SCHEMA,
        "scene_id": str(scene_id),
        "source_sha256": str(source_sha256),
        "frame_id": int(frame_id),
        "image_name": str(image_name),
        "point_count": int(point_count),
        "class_count": int(class_count),
        "mask_count": mask_count,
        "valid_pixel_count": int(header.get("valid_pixel_count", -1)),
        "geometry_abstained": bool(header.get("geometry_abstained")),
        "semantic_abstained": bool(header.get("semantic_abstained")),
        "payload_sha256": _checkpoint_payload_digest(arrays),
    }
    if header != expected_identity:
        raise ValueError("frame checkpoint identity does not match this request")
    masks = tuple(
        MaskMetadata(
            global_mask_id=index,
            frame_id=int(frame_id),
            image_name=str(image_name),
            mask_index=int(mask_indices[index]),
        )
        for index in range(mask_count)
    )
    frame = FrameEvidence(
        metadata=FrameMetadata(
            frame_id=int(frame_id),
            image_name=str(image_name),
            valid_pixel_count=int(header["valid_pixel_count"]),
            geometry_abstained=bool(header["geometry_abstained"]),
            semantic_abstained=bool(header["semantic_abstained"]),
        ),
        masks=masks,
        support=MaskSupportCSR(
            arrays["support_indptr"],
            arrays["support_gaussian_ids"],
            arrays["support_inside_mass"],
            arrays["support_inside_ratio"],
            arrays["support_ambiguous"],
            mask_count,
            int(point_count),
        ),
        visibility=PackedVisibility(
            arrays["visible_indptr"],
            arrays["visible_gaussian_ids"],
            arrays["visible_mass"],
            1,
            int(point_count),
        ),
        ambiguous_gaussians=arrays["ambiguous_gaussian_ids"],
        semantic_posteriors=arrays["semantic_posteriors"],
        semantic_abstained=arrays["semantic_abstained"],
    )
    if frame.metadata.semantic_abstained != bool(header["semantic_abstained"]):
        raise ValueError("frame checkpoint semantic abstention is inconsistent")
    if frame.semantic_posteriors.shape[1] != int(class_count):
        raise ValueError("frame checkpoint has the wrong class dimension")
    return frame


def _rendered_record_to_frame(
    record: Any,
    *,
    point_count: int,
    class_count: int,
    image_name: str | None = None,
) -> FrameEvidence:
    supports = tuple(record.masks)
    if supports:
        posterior = np.stack(
            [
                np.asarray(mask.class_probabilities, dtype=np.float32)
                for mask in supports
            ]
        )
        semantic_abstention = (
            np.ones(len(supports), dtype=bool)
            if bool(record.grounded_abstained)
            else posterior.sum(axis=1, dtype=np.float64) <= 1e-7
        )
    else:
        posterior = np.empty((0, int(class_count)), dtype=np.float32)
        semantic_abstention = np.empty(0, dtype=bool)
    return build_sparse_frame_evidence(
        frame_id=int(record.frame_id),
        image_name=str(record.image_name if image_name is None else image_name),
        point_count=int(point_count),
        visible_ids=np.asarray(record.visible_ids),
        visible_mass=np.asarray(record.visible_mass),
        mask_gaussian_ids=[np.asarray(mask.gaussian_ids) for mask in supports],
        mask_inside_mass=[np.asarray(mask.inside_mass) for mask in supports],
        mask_inside_ratio=[np.asarray(mask.inside_ratio) for mask in supports],
        ambiguous_ids=[np.asarray(mask.ambiguous_ids) for mask in supports],
        semantic_posteriors=posterior,
        semantic_abstained=semantic_abstention,
        global_mask_id_start=0,
        mask_indices=[int(mask.mask_index) for mask in supports],
        valid_pixel_count=int(record.valid_pixel_count),
        geometry_abstained=not supports,
        class_count=int(class_count),
    )


def _rebase_frame_masks(frame: FrameEvidence, start: int) -> FrameEvidence:
    masks = tuple(
        MaskMetadata(
            global_mask_id=int(start) + index,
            frame_id=mask.frame_id,
            image_name=mask.image_name,
            mask_index=mask.mask_index,
        )
        for index, mask in enumerate(frame.masks)
    )
    return FrameEvidence(
        metadata=frame.metadata,
        masks=masks,
        support=frame.support,
        visibility=frame.visibility,
        ambiguous_gaussians=frame.ambiguous_gaussians,
        semantic_posteriors=frame.semantic_posteriors,
        semantic_abstained=frame.semantic_abstained,
    )


def save_evidence_bank(
    bank: AlphaMaskEvidenceBank,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Serialize one bank without hashes, pickle, or hidden compatibility paths."""

    _validate_evidence_value_contract(bank)

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    files = (
        target / EVIDENCE_ARRAY_FILE,
        target / EVIDENCE_METADATA_FILE,
        target / EVIDENCE_DIAGNOSTICS_FILE,
    )
    if not overwrite and any(path.exists() for path in files):
        raise FileExistsError(f"evidence target is occupied: {target}")
    if overwrite:
        # masks.json is the completion marker.  Invalidate it *before* any
        # constituent file is replaced so an interruption can never expose a
        # new NPZ together with stale metadata from the previous bank.  The
        # runner will treat the directory as incomplete and safely rebuild it.
        files[1].unlink(missing_ok=True)
    _write_npz_atomic(files[0], _arrays(bank))
    _write_json_atomic(files[2], _diagnostics(bank))
    # Metadata is written last and therefore acts as the completion marker.
    _write_json_atomic(files[1], _metadata(bank))


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_evidence_bank(
    directory: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_point_count: int | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> AlphaMaskEvidenceBank:
    """Load and cross-validate JSON metadata, NPZ arrays, and diagnostics."""

    root = Path(directory)
    metadata = _read_json_object(root / EVIDENCE_METADATA_FILE)
    diagnostics = _read_json_object(root / EVIDENCE_DIAGNOSTICS_FILE)
    if metadata.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported or missing evidence schema")
    if metadata.get("files") != {
        "arrays": EVIDENCE_ARRAY_FILE,
        "diagnostics": EVIDENCE_DIAGNOSTICS_FILE,
    }:
        raise ValueError("evidence file declaration is invalid")
    with np.load(root / EVIDENCE_ARRAY_FILE, allow_pickle=False) as loaded:
        if set(loaded.files) != _ARRAY_KEYS:
            raise ValueError("evidence.npz has missing or unexpected arrays")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    frame_rows = metadata.get("frames")
    mask_rows = metadata.get("masks")
    if not isinstance(frame_rows, list) or not isinstance(mask_rows, list):
        raise ValueError("frames and masks metadata must be lists")
    frames = tuple(
        FrameMetadata(
            frame_id=row["frame_id"],
            image_name=row["image_name"],
            valid_pixel_count=row["valid_pixel_count"],
            geometry_abstained=row["geometry_abstained"],
            semantic_abstained=row["semantic_abstained"],
        )
        for row in frame_rows
    )
    masks = tuple(
        MaskMetadata(
            global_mask_id=row["global_mask_id"],
            frame_id=row["frame_id"],
            image_name=row["image_name"],
            mask_index=row["mask_index"],
        )
        for row in mask_rows
    )
    points = int(metadata["point_count"])
    bank = AlphaMaskEvidenceBank(
        scene_id=str(metadata["scene_id"]),
        point_count=points,
        xyz_m=arrays["xyz_m"],
        class_names=tuple(metadata["class_names"]),
        thresholds=EvidenceThresholds.from_dict(metadata["thresholds"]),
        frames=frames,
        masks=masks,
        mask_support=MaskSupportCSR(
            arrays["mask_support_indptr"],
            arrays["mask_support_gaussian_ids"],
            arrays["mask_support_inside_mass"],
            arrays["mask_support_inside_ratio"],
            arrays["mask_support_ambiguous"],
            len(masks),
            points,
        ),
        frame_visibility=PackedVisibility(
            arrays["frame_visible_indptr"],
            arrays["frame_visible_gaussian_ids"],
            arrays["frame_visible_mass"],
            len(frames),
            points,
        ),
        frame_ambiguity=PackedIndexRows(
            arrays["frame_ambiguous_indptr"],
            arrays["frame_ambiguous_gaussian_ids"],
            len(frames),
            points,
            "frame_ambiguity",
        ),
        semantic_posteriors=arrays["mask_semantic_posteriors"],
        semantic_abstained=arrays["mask_semantic_abstained"],
        source=metadata.get("source", {}),
    )
    _validate_evidence_value_contract(bank)
    if not np.array_equal(
        arrays["frame_id"], np.asarray([frame.frame_id for frame in frames], dtype=np.int64)
    ) or not np.array_equal(
        arrays["frame_valid_pixel_count"],
        np.asarray([frame.valid_pixel_count for frame in frames], dtype=np.int64),
    ) or not np.array_equal(
        arrays["frame_geometry_abstained"],
        np.asarray([frame.geometry_abstained for frame in frames], dtype=np.bool_),
    ) or not np.array_equal(
        arrays["frame_semantic_abstained"],
        np.asarray([frame.semantic_abstained for frame in frames], dtype=np.bool_),
    ):
        raise ValueError("frame JSON and NPZ metadata disagree")
    if not np.array_equal(
        arrays["mask_global_id"],
        np.asarray([mask.global_mask_id for mask in masks], dtype=np.int64),
    ) or not np.array_equal(
        arrays["mask_frame_id"],
        np.asarray([mask.frame_id for mask in masks], dtype=np.int64),
    ) or not np.array_equal(
        arrays["mask_index"],
        np.asarray([mask.mask_index for mask in masks], dtype=np.int32),
    ):
        raise ValueError("mask JSON and NPZ metadata disagree")
    expected_diagnostics = _diagnostics(bank)
    if diagnostics != expected_diagnostics:
        raise ValueError("diagnostics.json does not match the evidence bank")
    if expected_scene_id is not None and bank.scene_id != str(expected_scene_id):
        raise ValueError("evidence scene_id does not match the expected scene")
    if expected_point_count is not None and bank.point_count != int(expected_point_count):
        raise ValueError("evidence point_count does not match the expected scene")
    if expected_source is not None and dict(bank.source) != dict(expected_source):
        raise ValueError("evidence source identity does not match")
    return bank


def evidence_bank_is_complete(
    directory: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_point_count: int | None = None,
    expected_source: Mapping[str, Any] | None = None,
) -> bool:
    try:
        load_evidence_bank(
            directory,
            expected_scene_id=expected_scene_id,
            expected_point_count=expected_point_count,
            expected_source=expected_source,
        )
        return True
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _resolve_evidence_request(
    scene_id: str, request: Mapping[str, Any]
) -> tuple[dict[str, Any], tuple[str, ...], Any, dict[str, Any]]:
    """Resolve exactly the inputs that can affect evidence production.

    This function imports the worker lazily, but performs no rendering and
    initializes no CUDA state.  Keeping one resolver for both the completion
    check and the producer prevents a scene-only cache hit from silently
    reusing a bank built from different masks, coordinates, labels, classes,
    or metric scale.
    """

    from .worker import (
        CONTINUOUS_ALPHA_MASS_PERSISTED,
        DEFAULT_CLASSES,
        EVIDENCE_VALUE_CONTRACT,
        resolve_clean_scene_inputs,
    )

    if not isinstance(request, Mapping):
        raise TypeError("evidence request must be a mapping")
    scene_value = request.get(
        "scene", request.get("runtime", request.get("runtime_item", request))
    )
    if not isinstance(scene_value, Mapping):
        raise TypeError("evidence request.scene must be a mapping")
    scene = dict(scene_value)
    requested_scene = str(scene.get("scene_id", scene_id))
    if requested_scene != str(scene_id):
        raise ValueError("request scene_id differs from the CLI scene_id")
    scene["scene_id"] = str(scene_id)
    classes_value = request.get("classes", DEFAULT_CLASSES)
    if not isinstance(classes_value, Sequence) or isinstance(
        classes_value, (str, bytes)
    ):
        raise TypeError("request.classes must be a class-name sequence")
    classes = tuple(str(value).strip() for value in classes_value)
    if classes != tuple(DEFAULT_CLASSES):
        raise ValueError(
            "request.classes must exactly match the registered 32-class "
            "Grounded-SAM/codebook order"
        )
    sam_override: Path | None = None
    if request.get("sam_masks") not in (None, ""):
        sam_override = Path(str(request["sam_masks"]))
        if not sam_override.is_absolute():
            sam_override = Path(str(scene["base_path"])) / sam_override
    inputs = resolve_clean_scene_inputs(scene, sam_masks=sam_override)
    scale_m_per_unit = float(scene.get("scene_scale_m_per_unit", 1.0))
    if not np.isfinite(scale_m_per_unit) or scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    producer = request.get(
        "producer_commit",
        request.get(
            "code_commit",
            request.get(
                "commit",
                scene.get(
                    "producer_commit", scene.get("code_commit", scene.get("commit"))
                ),
            ),
        ),
    )
    producer_text = str(producer or "")
    if _FULL_COMMIT.fullmatch(producer_text) is None:
        raise ValueError(
            "evidence request producer_commit must be an exact 40-character "
            "lowercase Git commit"
        )
    mask_observation_mode = str(
        request.get("mask_observation_mode", "hierarchy")
    )
    if mask_observation_mode not in {"hierarchy", "flat-highest-quality"}:
        raise ValueError(
            "request.mask_observation_mode must be hierarchy or "
            "flat-highest-quality"
        )
    scene["mask_observation_mode"] = mask_observation_mode
    source: dict[str, Any] = {
        "worker": "category_priors.clean_baseline.worker:render_scene_frames",
        "evidence_schema": EVIDENCE_SCHEMA,
        "thresholds": EvidenceThresholds().to_dict(),
        "base_path": str(inputs.base_path),
        "rgb_ply": str(inputs.rgb_ply),
        "sparse": str(inputs.sparse),
        "images": str(inputs.images),
        "sam_masks": str(inputs.sam_masks),
        "grounded_masks": str(inputs.grounded_masks),
        "grounded_labels": str(inputs.grounded_labels),
        "class_names": list(classes),
        "xyz_units": "meters",
        "scene_scale_m_per_unit": scale_m_per_unit,
        "producer_inputs": _producer_input_identity(inputs),
        "producer_commit": producer_text,
        "mask_observation_mode": mask_observation_mode,
        "evidence_values": EVIDENCE_VALUE_CONTRACT,
        "continuous_alpha_mass_persisted": CONTINUOUS_ALPHA_MASS_PERSISTED,
    }
    # The producer revision and the actual producer input contents jointly
    # define the resume boundary.  This identity lives inside masks.json; no
    # separate SHA file is generated.
    return scene, classes, inputs, source


def evidence_request_source(
    *, scene_id: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the canonical, pure-CPU cache identity for one request."""

    return _resolve_evidence_request(scene_id, request)[3]


def build_alpha_mask_evidence(
    *,
    scene_id: str,
    request: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Thin scene-worker adapter used by ``build-alpha-mask-evidence``.

    Rendering stays in :mod:`category_priors.clean_baseline.worker`; this
    adapter only resolves the registered request, converts each already-sparse
    frame, and persists the strict bank.  The import is deliberately lazy so
    importing the CPU evidence API never imports Torch or CUDA code.
    """

    from .worker import render_scene_frames

    scene, classes, inputs, source = _resolve_evidence_request(scene_id, request)
    output = Path(output_dir)
    if evidence_bank_is_complete(
        output,
        expected_scene_id=str(scene_id),
        expected_source=source,
    ):
        return _diagnostics(load_evidence_bank(output))
    from .sam_inputs import colmap_frame_specs

    frame_specs = colmap_frame_specs(inputs.sparse)
    source_sha256 = _source_identity_digest(source)
    checkpoint_root = _checkpoint_root(output)
    scene_checkpoint = checkpoint_root / "scene.npz"
    frames_root = checkpoint_root / "frames"
    xyz_array: np.ndarray | None
    try:
        xyz_array = _load_scene_checkpoint(
            scene_checkpoint,
            scene_id=str(scene_id),
            source_sha256=source_sha256,
        )
    except Exception:
        # A missing/corrupt scene checkpoint invalidates frame checkpoints as
        # a set because their point IDs cannot be interpreted safely.  They
        # remain on disk for forensics and are atomically replaced as frames
        # are rendered again.
        xyz_array = None

    local_frames: dict[int, FrameEvidence] = {}
    if xyz_array is not None:
        for frame_id, spec in enumerate(frame_specs):
            try:
                local_frames[frame_id] = _load_frame_checkpoint(
                    frames_root / f"{frame_id:08d}.npz",
                    scene_id=str(scene_id),
                    source_sha256=source_sha256,
                    frame_id=frame_id,
                    image_name=spec.image_name,
                    point_count=len(xyz_array),
                    class_count=len(classes),
                )
            except Exception:
                # Frame validity is independent once the exact scene/source
                # identity is established.  Only this row is rendered again.
                continue

    pending = tuple(
        frame_id for frame_id in range(len(frame_specs)) if frame_id not in local_frames
    )
    rendered_ids: set[int] = set()

    def persist_rendered_frame(raw_xyz: np.ndarray, record: Any) -> None:
        nonlocal xyz_array
        current_xyz = _validate_raw_xyz(raw_xyz)
        if xyz_array is None:
            xyz_array = current_xyz.copy()
            _save_scene_checkpoint(
                scene_checkpoint,
                scene_id=str(scene_id),
                source_sha256=source_sha256,
                xyz=xyz_array,
            )
        elif not np.array_equal(current_xyz, xyz_array):
            raise RuntimeError(
                "renderer Gaussian XYZ differs from the strict scene checkpoint"
            )
        frame_id = int(record.frame_id)
        if frame_id not in pending or frame_id in rendered_ids:
            raise RuntimeError("renderer returned an unexpected or duplicate frame")
        expected_name = frame_specs[frame_id].image_name
        actual_name = str(Path(str(record.image_name).replace("\\", "/")).with_suffix(""))
        actual_name = actual_name.replace("\\", "/")
        if actual_name != expected_name:
            raise RuntimeError("rendered frame image name differs from COLMAP order")
        frame = _rendered_record_to_frame(
            record,
            point_count=len(xyz_array),
            class_count=len(classes),
            image_name=expected_name,
        )
        _save_frame_checkpoint(
            frames_root / f"{frame_id:08d}.npz",
            scene_id=str(scene_id),
            source_sha256=source_sha256,
            point_count=len(xyz_array),
            class_count=len(classes),
            frame=frame,
        )
        local_frames[frame_id] = frame
        rendered_ids.add(frame_id)

    if pending:
        rendered_xyz, records = render_scene_frames(
            inputs,
            classes=classes,
            mask_observation_mode=str(scene["mask_observation_mode"]),
            frame_ids=pending,
            frame_callback=persist_rendered_frame,
        )
        # Compatibility with simple test/dry-run workers that return sparse
        # records but do not invoke the callback.  The production worker calls
        # it before advancing to the next frame, which is what makes an
        # interrupted run durable.
        for record in records:
            if int(record.frame_id) not in rendered_ids:
                persist_rendered_frame(rendered_xyz, record)
        if rendered_ids != set(pending):
            raise RuntimeError("renderer did not produce every requested frame")
    if xyz_array is None:
        raise RuntimeError("scene has no valid Gaussian checkpoint")
    if set(local_frames) != set(range(len(frame_specs))):
        raise RuntimeError("frame checkpoint set is incomplete after rendering")

    scale_m_per_unit = float(source["scene_scale_m_per_unit"])
    xyz_m = np.asarray(xyz_array * scale_m_per_unit, dtype=np.float32)
    frames: list[FrameEvidence] = []
    next_global_mask_id = 0
    for frame_id in range(len(frame_specs)):
        frame = _rebase_frame_masks(local_frames[frame_id], next_global_mask_id)
        frames.append(frame)
        next_global_mask_id += len(frame.masks)
    bank = AlphaMaskEvidenceBank.from_frames(
        scene_id=str(scene_id),
        point_count=len(xyz_array),
        xyz_m=xyz_m,
        class_names=classes,
        frames=frames,
        source=source,
    )
    save_evidence_bank(bank, output, overwrite=True)
    return _diagnostics(bank)
