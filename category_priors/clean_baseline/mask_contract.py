from __future__ import annotations

"""Pure contracts for the paired hierarchy/flat SAM-mask control.

The clean-baseline control must derive both conditions from one immutable SAM
generation.  This module deliberately contains no renderer, class label, GT,
or consensus code.  It validates that immutable source stack, deterministically
turns overlapping hierarchy masks into exclusive object observations, and
performs the equivalent exclusivity decision after alpha lifting.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SAM_METADATA_KEYS = frozenset(
    {
        "packed",
        "count",
        "height",
        "width",
        "predicted_iou",
        "stability_score",
        "area",
    }
)


def _integer_scalar(value: Any, *, name: str) -> int:
    raw = np.asarray(value)
    if (
        raw.size != 1
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(f"{name} must be one integer scalar")
    return int(raw.reshape(()).item())


def _readonly(array: Any, *, dtype: Any, ndim: int, name: str) -> np.ndarray:
    raw = np.asarray(array)
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    result = np.ascontiguousarray(raw, dtype=dtype)
    result.setflags(write=False)
    return result


def _quality_vector(value: Any, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{name} must be a one-dimensional floating array")
    result = np.ascontiguousarray(raw, dtype=np.float32)
    result.setflags(write=False)
    return result


def _popcount_rows(packed: np.ndarray) -> np.ndarray:
    lookup = np.unpackbits(
        np.arange(256, dtype=np.uint8)[:, None], axis=1
    ).sum(axis=1, dtype=np.int64)
    return lookup[np.asarray(packed, dtype=np.uint8)].sum(axis=1, dtype=np.int64)


@dataclass(frozen=True)
class SamMaskMetadataFrame:
    """One metadata-rich, immutable SAM-everything frame.

    Rows retain the generator's original order.  Consequently a row index is
    also the original mask ID used by the final tie breaker.
    """

    packed: np.ndarray
    count: int
    height: int
    width: int
    predicted_iou: np.ndarray
    stability_score: np.ndarray
    area: np.ndarray

    def __post_init__(self) -> None:
        count = _integer_scalar(self.count, name="count")
        height = _integer_scalar(self.height, name="height")
        width = _integer_scalar(self.width, name="width")
        if count < 0 or height <= 0 or width <= 0:
            raise ValueError("count/height/width are invalid")

        raw_packed = np.asarray(self.packed)
        if raw_packed.dtype != np.uint8 or raw_packed.ndim != 2:
            raise ValueError("packed must be a two-dimensional uint8 array")
        byte_count = (height * width + 7) // 8
        if raw_packed.shape != (count, byte_count):
            raise ValueError("packed shape disagrees with count/height/width")
        remainder = (height * width) % 8
        if remainder and count:
            unused_bits = (1 << (8 - remainder)) - 1
            if np.any(np.bitwise_and(raw_packed[:, -1], unused_bits)):
                raise ValueError("packed padding bits must be zero")
        packed = np.ascontiguousarray(raw_packed)

        predicted = _quality_vector(self.predicted_iou, name="predicted_iou")
        stability = _quality_vector(self.stability_score, name="stability_score")
        raw_area = np.asarray(self.area)
        if (
            raw_area.ndim != 1
            or np.issubdtype(raw_area.dtype, np.bool_)
            or not np.issubdtype(raw_area.dtype, np.integer)
        ):
            raise ValueError("area must be a one-dimensional integer array")
        area = _readonly(raw_area, dtype=np.int64, ndim=1, name="area")
        if predicted.shape != (count,) or stability.shape != (count,) or area.shape != (count,):
            raise ValueError("SAM metadata lengths must equal count")
        if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(stability)):
            raise ValueError("SAM quality metadata must be finite")
        # SAM's IoU head is a linear MLP score (no sigmoid in the published
        # implementation), so real finite outputs can be slightly above 1.
        # Preserve the raw score because H'/P use it only for deterministic
        # quality ordering; clipping would silently change that ordering.
        if np.any((stability < 0) | (stability > 1)):
            raise ValueError("stability_score must be in [0, 1]")
        if np.any(area <= 0):
            raise ValueError("source SAM masks must have positive area")
        measured_area = _popcount_rows(packed)
        if not np.array_equal(area, measured_area):
            raise ValueError("area disagrees with packed segmentation")

        packed.setflags(write=False)
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "predicted_iou", predicted)
        object.__setattr__(self, "stability_score", stability)
        object.__setattr__(self, "area", area)

    @property
    def pixel_count(self) -> int:
        return self.height * self.width

    def dense_batch(self, start: int, stop: int) -> np.ndarray:
        first, last = int(start), int(stop)
        if not 0 <= first <= last <= self.count:
            raise IndexError("mask batch is out of range")
        return np.unpackbits(
            self.packed[first:last], axis=1, count=self.pixel_count
        ).reshape(last - first, self.height, self.width).astype(np.bool_, copy=False)

    def dense(self) -> np.ndarray:
        return self.dense_batch(0, self.count)


@dataclass(frozen=True)
class FlatMaskFrame:
    """Exclusive masks derived from one :class:`SamMaskMetadataFrame`."""

    frame: SamMaskMetadataFrame
    source_mask_ids: np.ndarray
    pixel_owner_source_id: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_mask_ids)
        if (
            source.ndim != 1
            or np.issubdtype(source.dtype, np.bool_)
            or not np.issubdtype(source.dtype, np.integer)
        ):
            raise ValueError("source_mask_ids must be a one-dimensional integer array")
        source = np.ascontiguousarray(source, dtype=np.int32)
        if source.shape != (self.frame.count,):
            raise ValueError("source_mask_ids length must equal flat mask count")
        if np.any(source < 0) or (len(source) and len(np.unique(source)) != len(source)):
            raise ValueError("source_mask_ids must be unique and non-negative")
        owner = np.asarray(self.pixel_owner_source_id)
        if (
            owner.dtype != np.int32
            or owner.shape != (self.frame.height, self.frame.width)
        ):
            raise ValueError("pixel_owner_source_id must be int32 HxW")
        allowed = np.concatenate((np.asarray([-1], dtype=np.int32), source))
        if np.any(~np.isin(owner, allowed)):
            raise ValueError("pixel owner refers to an absent source mask")
        dense = self.frame.dense()
        if dense.size and np.any(dense.sum(axis=0, dtype=np.int32) > 1):
            raise ValueError("flat masks must not overlap")
        reconstructed = np.full(owner.shape, -1, dtype=np.int32)
        for row, source_id in enumerate(source):
            reconstructed[dense[row]] = source_id
        if not np.array_equal(reconstructed, owner):
            raise ValueError("flat masks disagree with pixel ownership")
        source.setflags(write=False)
        owner = np.ascontiguousarray(owner)
        owner.setflags(write=False)
        object.__setattr__(self, "source_mask_ids", source)
        object.__setattr__(self, "pixel_owner_source_id", owner)


@dataclass(frozen=True)
class GaussianMaskAssignment:
    """One exclusive mask owner for every Gaussian in one physical frame."""

    owner_mask_index: np.ndarray
    owner_source_mask_id: np.ndarray
    owner_ratio: np.ndarray
    qualifying_mask_count: np.ndarray

    def __post_init__(self) -> None:
        index = _readonly(
            self.owner_mask_index, dtype=np.int32, ndim=1, name="owner_mask_index"
        )
        source = _readonly(
            self.owner_source_mask_id,
            dtype=np.int32,
            ndim=1,
            name="owner_source_mask_id",
        )
        ratio = _readonly(
            self.owner_ratio, dtype=np.float32, ndim=1, name="owner_ratio"
        )
        count = _readonly(
            self.qualifying_mask_count,
            dtype=np.int32,
            ndim=1,
            name="qualifying_mask_count",
        )
        if not (index.shape == source.shape == ratio.shape == count.shape):
            raise ValueError("Gaussian assignment arrays must have equal lengths")
        if np.any(count < 0) or np.any((ratio < 0) | (ratio > 1)):
            raise ValueError("Gaussian assignment counts/ratios are invalid")
        assigned = index >= 0
        if np.any((source >= 0) != assigned) or np.any((count > 0) != assigned):
            raise ValueError("Gaussian assignment sentinel fields disagree")
        if np.any(ratio[~assigned] != 0):
            raise ValueError("unassigned Gaussian ratios must be zero")
        object.__setattr__(self, "owner_mask_index", index)
        object.__setattr__(self, "owner_source_mask_id", source)
        object.__setattr__(self, "owner_ratio", ratio)
        object.__setattr__(self, "qualifying_mask_count", count)


def metadata_frame_from_sam_rows(
    rows: Sequence[Mapping[str, Any]], *, height: int, width: int
) -> SamMaskMetadataFrame:
    """Validate a raw SAM generator result without dropping quality metadata."""

    height = _integer_scalar(height, name="height")
    width = _integer_scalar(width, name="width")
    if height <= 0 or width <= 0:
        raise ValueError("height/width must be positive")
    masks: list[np.ndarray] = []
    predicted: list[float] = []
    stability: list[float] = []
    areas: list[int] = []
    for mask_id, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"SAM row {mask_id} must be a mapping")
        required = {"segmentation", "predicted_iou", "stability_score"}
        if not required.issubset(row):
            raise ValueError(f"SAM row {mask_id} lacks required metadata")
        raw_mask = np.asarray(row["segmentation"])
        if raw_mask.dtype != np.bool_ or raw_mask.shape != (height, width):
            raise ValueError(f"SAM row {mask_id} segmentation must be bool HxW")
        area = int(np.count_nonzero(raw_mask))
        if area <= 0:
            raise ValueError(f"SAM row {mask_id} has an empty segmentation")
        if "area" in row:
            supplied = _integer_scalar(row["area"], name=f"SAM row {mask_id} area")
            if supplied != area:
                raise ValueError(f"SAM row {mask_id} area disagrees with segmentation")
        try:
            predicted_value = float(row["predicted_iou"])
            stability_value = float(row["stability_score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SAM row {mask_id} quality values are invalid") from exc
        masks.append(raw_mask)
        predicted.append(predicted_value)
        stability.append(stability_value)
        areas.append(area)
    dense = (
        np.stack(masks)
        if masks
        else np.zeros((0, height, width), dtype=np.bool_)
    )
    return SamMaskMetadataFrame(
        packed=np.packbits(dense.reshape(len(dense), height * width), axis=1),
        count=len(dense),
        height=height,
        width=width,
        predicted_iou=np.asarray(predicted, dtype=np.float32),
        stability_score=np.asarray(stability, dtype=np.float32),
        area=np.asarray(areas, dtype=np.int64),
    )


def save_sam_mask_metadata(path: str | Path, frame: SamMaskMetadataFrame) -> None:
    """Atomically persist the exact metadata contract used by both arms."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed=frame.packed,
            count=np.asarray(frame.count, dtype=np.int32),
            height=np.asarray(frame.height, dtype=np.int32),
            width=np.asarray(frame.width, dtype=np.int32),
            predicted_iou=frame.predicted_iou,
            stability_score=frame.stability_score,
            area=frame.area,
        )
    os.replace(temporary, target)


def load_sam_mask_metadata(path: str | Path) -> SamMaskMetadataFrame:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            if set(payload.files) != SAM_METADATA_KEYS:
                raise ValueError("SAM metadata payload has missing or unexpected arrays")
            packed = np.asarray(payload["packed"])
            if packed.dtype != np.uint8:
                raise ValueError("packed must use uint8 without coercion")
            frame = SamMaskMetadataFrame(
                packed=packed,
                count=_integer_scalar(payload["count"], name="count"),
                height=_integer_scalar(payload["height"], name="height"),
                width=_integer_scalar(payload["width"], name="width"),
                predicted_iou=np.asarray(payload["predicted_iou"]),
                stability_score=np.asarray(payload["stability_score"]),
                area=np.asarray(payload["area"]),
            )
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise ValueError(f"invalid SAM metadata frame: {source}") from exc
    return frame


def mask_priority_order(
    predicted_iou: np.ndarray,
    stability_score: np.ndarray,
    source_mask_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Return best-to-worst row order for the frozen three-level tie break."""

    predicted = np.asarray(predicted_iou, dtype=np.float64)
    stability = np.asarray(stability_score, dtype=np.float64)
    if predicted.ndim != 1 or stability.shape != predicted.shape:
        raise ValueError("mask quality arrays must be equal-length vectors")
    if np.any(~np.isfinite(predicted)) or np.any(~np.isfinite(stability)):
        raise ValueError("mask quality arrays must be finite")
    if source_mask_ids is None:
        source = np.arange(len(predicted), dtype=np.int64)
    else:
        raw_source = np.asarray(source_mask_ids)
        if (
            raw_source.shape != predicted.shape
            or np.issubdtype(raw_source.dtype, np.bool_)
            or not np.issubdtype(raw_source.dtype, np.integer)
        ):
            raise ValueError("source_mask_ids must be an equal-length integer vector")
        source = raw_source.astype(np.int64, copy=False)
        if np.any(source < 0) or len(np.unique(source)) != len(source):
            raise ValueError("source_mask_ids must be unique and non-negative")
    # np.lexsort uses its last key as primary: descending quality, then the
    # smaller original source mask ID.
    return np.lexsort((source, -stability, -predicted)).astype(np.int32)


def _flat_frame_from_owner(
    hierarchy: SamMaskMetadataFrame, owner: np.ndarray
) -> FlatMaskFrame:
    flat_owner = np.asarray(owner, dtype=np.int32).reshape(-1)
    present = np.unique(flat_owner[flat_owner >= 0]).astype(np.int32)
    priority = mask_priority_order(
        hierarchy.predicted_iou, hierarchy.stability_score
    )
    # Flat rows themselves are best-to-worst.  The lifting worker can then use
    # the smaller local row index for exact ratio ties without loading a second
    # priority table.  ``source_mask_ids`` still preserves original identity.
    retained = priority[np.isin(priority, present)].astype(np.int32)
    masks = np.stack(
        [(flat_owner == mask_id).reshape(hierarchy.height, hierarchy.width) for mask_id in retained]
    ) if len(retained) else np.zeros((0, hierarchy.height, hierarchy.width), dtype=np.bool_)
    areas = (
        masks.reshape(len(masks), hierarchy.pixel_count).sum(axis=1, dtype=np.int64)
        if len(masks)
        else np.zeros(0, dtype=np.int64)
    )
    flat = SamMaskMetadataFrame(
        packed=np.packbits(masks.reshape(len(masks), hierarchy.pixel_count), axis=1),
        count=len(masks),
        height=hierarchy.height,
        width=hierarchy.width,
        predicted_iou=hierarchy.predicted_iou[retained],
        stability_score=hierarchy.stability_score[retained],
        area=areas,
    )
    return FlatMaskFrame(
        frame=flat,
        source_mask_ids=retained,
        pixel_owner_source_id=flat_owner.reshape(hierarchy.height, hierarchy.width),
    )


def flatten_mask_stack(hierarchy: SamMaskMetadataFrame) -> FlatMaskFrame:
    """Streaming deterministic flattening without an MxHxW materialization."""

    owner = np.full(hierarchy.pixel_count, -1, dtype=np.int32)
    for mask_id in mask_priority_order(
        hierarchy.predicted_iou, hierarchy.stability_score
    ):
        support = np.unpackbits(
            hierarchy.packed[int(mask_id)], count=hierarchy.pixel_count
        ).astype(np.bool_, copy=False)
        selected = support & (owner < 0)
        owner[selected] = int(mask_id)
    return _flat_frame_from_owner(hierarchy, owner)


def flatten_mask_stack_dense_reference(
    hierarchy: SamMaskMetadataFrame,
) -> FlatMaskFrame:
    """Small dense reference used to test the production streaming path."""

    dense = hierarchy.dense().reshape(hierarchy.count, hierarchy.pixel_count)
    priority = mask_priority_order(
        hierarchy.predicted_iou, hierarchy.stability_score
    )
    rank = np.empty(hierarchy.count, dtype=np.int32)
    rank[priority] = np.arange(hierarchy.count, dtype=np.int32)
    ranked = np.where(dense, rank[:, None], hierarchy.count)
    best_rows = np.argmin(ranked, axis=0) if hierarchy.count else np.zeros(hierarchy.pixel_count, dtype=np.int64)
    any_mask = dense.any(axis=0) if hierarchy.count else np.zeros(hierarchy.pixel_count, dtype=np.bool_)
    owner = np.full(hierarchy.pixel_count, -1, dtype=np.int32)
    if hierarchy.count:
        owner[any_mask] = best_rows[any_mask].astype(np.int32)
    return _flat_frame_from_owner(hierarchy, owner)


def audit_flat_mask_contract(
    hierarchy: SamMaskMetadataFrame, flat: FlatMaskFrame
) -> dict[str, Any]:
    """Return the mechanical invariants required before a scientific run."""

    if (hierarchy.height, hierarchy.width) != (flat.frame.height, flat.frame.width):
        raise ValueError("hierarchy and flat frames have different image shapes")
    hierarchy_union = hierarchy.dense().any(axis=0)
    flat_dense = flat.frame.dense()
    flat_union = flat_dense.any(axis=0)
    overlap_pixels = int(
        np.count_nonzero(flat_dense.sum(axis=0, dtype=np.int32) > 1)
    )
    union_changed = int(np.count_nonzero(hierarchy_union != flat_union))
    return {
        "hierarchy_mask_count": hierarchy.count,
        "flat_mask_count": flat.frame.count,
        "empty_remnant_count": hierarchy.count - flat.frame.count,
        "hierarchy_union_pixel_count": int(np.count_nonzero(hierarchy_union)),
        "flat_union_pixel_count": int(np.count_nonzero(flat_union)),
        "union_changed_pixel_count": union_changed,
        "flat_overlap_pixel_count": overlap_pixels,
        "union_exact": union_changed == 0,
        "flat_overlap_rate": (
            float(overlap_pixels / max(1, np.count_nonzero(flat_union)))
        ),
        "mechanical_contract_pass": union_changed == 0 and overlap_pixels == 0,
    }


def assign_gaussians_to_flat_masks(
    inside_mass: np.ndarray,
    visible_mass: np.ndarray,
    *,
    predicted_iou: np.ndarray,
    stability_score: np.ndarray,
    source_mask_ids: np.ndarray | None = None,
    inside_min_mass: float = 0.5,
    inside_min_ratio: float = 0.5,
) -> GaussianMaskAssignment:
    """Assign every qualifying frame/Gaussian observation to one mask.

    The primary comparison is the measured ``inside/visible`` ratio.  Exact
    ratio ties use predicted IoU, then stability score, then the smaller
    original mask ID.  No semantic or GT value is accepted by this API.
    """

    inside = np.asarray(inside_mass, dtype=np.float64)
    visible = np.asarray(visible_mass, dtype=np.float64)
    if inside.ndim != 2 or visible.ndim != 1 or inside.shape[1] != len(visible):
        raise ValueError("inside_mass must be MxN and visible_mass must be N")
    if np.any(~np.isfinite(inside)) or np.any(~np.isfinite(visible)):
        raise ValueError("alpha masses must be finite")
    if np.any(inside < 0) or np.any(visible < 0):
        raise ValueError("alpha masses must be non-negative")
    tolerance = 5e-6 * np.maximum(visible[None, :], 1.0)
    if inside.size and np.any(inside - visible[None, :] > tolerance):
        raise ValueError("inside mass cannot exceed visible mass")
    if not np.isfinite(inside_min_mass) or float(inside_min_mass) <= 0:
        raise ValueError("inside_min_mass must be positive and finite")
    if not np.isfinite(inside_min_ratio) or not 0 < float(inside_min_ratio) <= 1:
        raise ValueError("inside_min_ratio must be in (0, 1]")

    predicted = np.asarray(predicted_iou, dtype=np.float64)
    stability = np.asarray(stability_score, dtype=np.float64)
    if predicted.shape != (inside.shape[0],) or stability.shape != predicted.shape:
        raise ValueError("mask quality metadata must match inside_mass rows")
    if source_mask_ids is None:
        source = np.arange(inside.shape[0], dtype=np.int32)
    else:
        raw_source = np.asarray(source_mask_ids)
        if (
            raw_source.shape != predicted.shape
            or np.issubdtype(raw_source.dtype, np.bool_)
            or not np.issubdtype(raw_source.dtype, np.integer)
        ):
            raise ValueError("source_mask_ids must match inside_mass rows")
        if np.any(raw_source < 0) or np.any(raw_source > np.iinfo(np.int32).max):
            raise ValueError("source_mask_ids are outside int32 range")
        source = raw_source.astype(np.int32, copy=False)
    if (
        np.any(~np.isfinite(predicted))
        or np.any(~np.isfinite(stability))
        or np.any((predicted < 0) | (predicted > 1))
        or np.any((stability < 0) | (stability > 1))
    ):
        raise ValueError("mask quality metadata must be finite and in [0, 1]")
    order = mask_priority_order(predicted, stability, source)
    ratio = np.divide(
        inside,
        visible[None, :],
        out=np.zeros_like(inside),
        where=visible[None, :] > 0,
    )
    ratio = np.clip(ratio, 0.0, 1.0)
    qualifies = (
        (inside >= float(inside_min_mass))
        & (ratio >= float(inside_min_ratio))
    )
    qualifying_count = qualifies.sum(axis=0, dtype=np.int32)
    owner = np.full(len(visible), -1, dtype=np.int32)
    owner_ratio = np.zeros(len(visible), dtype=np.float64)
    # Priority-best rows run first.  A strict ratio improvement replaces the
    # owner; an exact tie therefore preserves the better mask priority.
    for row in order:
        row = int(row)
        select = qualifies[row] & ((owner < 0) | (ratio[row] > owner_ratio))
        owner[select] = row
        owner_ratio[select] = ratio[row, select]
    owner_source = np.full(len(visible), -1, dtype=np.int32)
    assigned = owner >= 0
    owner_source[assigned] = source[owner[assigned]]
    owner_ratio[~assigned] = 0.0
    return GaussianMaskAssignment(
        owner_mask_index=owner,
        owner_source_mask_id=owner_source,
        owner_ratio=owner_ratio.astype(np.float32),
        qualifying_mask_count=qualifying_count,
    )


def make_sparse_support_exclusive(
    mask_gaussian_ids: Sequence[np.ndarray],
    mask_inside_mass: Sequence[np.ndarray],
    mask_inside_ratio: Sequence[np.ndarray],
    *,
    point_count: int,
    predicted_iou: np.ndarray | None = None,
    stability_score: np.ndarray | None = None,
    source_mask_ids: np.ndarray | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    """Filter thresholded sparse lifting support to one mask per Gaussian.

    The inputs are the per-mask ``(Gaussian ID, inside mass, inside/visible
    ratio)`` rows already produced by the renderer's frozen thresholds.  The
    highest ratio wins.  Exact ties use predicted IoU, stability, and original
    mask ID.  Every returned row remains sorted by Gaussian ID and all losing
    observations are removed.
    """

    raw_point_count = np.asarray(point_count)
    if (
        raw_point_count.size != 1
        or np.issubdtype(raw_point_count.dtype, np.bool_)
        or not np.issubdtype(raw_point_count.dtype, np.integer)
    ):
        raise ValueError("point_count must be one integer scalar")
    total = int(raw_point_count.reshape(()).item())
    if total < 0:
        raise ValueError("point_count must be non-negative")
    row_count = len(mask_gaussian_ids)
    if len(mask_inside_mass) != row_count or len(mask_inside_ratio) != row_count:
        raise ValueError("sparse support sequences must have equal row counts")
    if (predicted_iou is None) != (stability_score is None):
        raise ValueError("predicted_iou and stability_score must be supplied together")
    if predicted_iou is None:
        # P rows are already serialized in best-to-worst mask priority.  This
        # is the normal worker path and needs no extra metadata allocation.
        predicted = np.zeros(row_count, dtype=np.float64)
        stability = np.zeros(row_count, dtype=np.float64)
    else:
        predicted = np.asarray(predicted_iou, dtype=np.float64)
        stability = np.asarray(stability_score, dtype=np.float64)
        if predicted.shape != (row_count,) or stability.shape != (row_count,):
            raise ValueError("mask quality metadata must match sparse support rows")
    if source_mask_ids is None:
        source = np.arange(row_count, dtype=np.int32)
    else:
        raw_source = np.asarray(source_mask_ids)
        if (
            raw_source.shape != (row_count,)
            or np.issubdtype(raw_source.dtype, np.bool_)
            or not np.issubdtype(raw_source.dtype, np.integer)
        ):
            raise ValueError("source_mask_ids must match sparse support rows")
        if np.any(raw_source < 0) or np.any(raw_source > np.iinfo(np.int32).max):
            raise ValueError("source_mask_ids are outside int32 range")
        source = raw_source.astype(np.int32, copy=False)
    priority = (
        np.arange(row_count, dtype=np.int32)
        if predicted_iou is None
        else mask_priority_order(predicted, stability, source)
    )

    ids_rows: list[np.ndarray] = []
    mass_rows: list[np.ndarray] = []
    ratio_rows: list[np.ndarray] = []
    for row_index, (raw_ids, raw_mass, raw_ratio) in enumerate(
        zip(mask_gaussian_ids, mask_inside_mass, mask_inside_ratio)
    ):
        ids = np.asarray(raw_ids)
        if (
            ids.ndim != 1
            or np.issubdtype(ids.dtype, np.bool_)
            or not np.issubdtype(ids.dtype, np.integer)
        ):
            raise ValueError(f"mask row {row_index} Gaussian IDs must be integers")
        ids = ids.astype(np.int32, copy=False)
        mass = np.asarray(raw_mass, dtype=np.float64)
        ratio = np.asarray(raw_ratio, dtype=np.float64)
        if mass.shape != ids.shape or ratio.shape != ids.shape:
            raise ValueError(f"mask row {row_index} sparse arrays have different lengths")
        if (
            np.any(ids < 0)
            or np.any(ids >= total)
            or (len(ids) and np.any(np.diff(ids) <= 0))
        ):
            raise ValueError(f"mask row {row_index} IDs must be sorted unique and in range")
        if np.any(~np.isfinite(mass)) or np.any(mass < 0):
            raise ValueError(f"mask row {row_index} inside mass is invalid")
        if np.any(~np.isfinite(ratio)) or np.any((ratio < 0) | (ratio > 1)):
            raise ValueError(f"mask row {row_index} inside ratio is invalid")
        ids_rows.append(np.ascontiguousarray(ids))
        mass_rows.append(np.ascontiguousarray(mass, dtype=np.float32))
        ratio_rows.append(np.ascontiguousarray(ratio, dtype=np.float64))

    owner = np.full(total, -1, dtype=np.int32)
    best_ratio = np.zeros(total, dtype=np.float64)
    for row in priority:
        row = int(row)
        ids = ids_rows[row]
        ratios = ratio_rows[row]
        select = (owner[ids] < 0) | (ratios > best_ratio[ids])
        selected_ids = ids[select]
        owner[selected_ids] = row
        best_ratio[selected_ids] = ratios[select]

    result: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for row, (ids, mass, ratio) in enumerate(zip(ids_rows, mass_rows, ratio_rows)):
        keep = owner[ids] == row
        result.append(
            (
                np.ascontiguousarray(ids[keep], dtype=np.int32),
                np.ascontiguousarray(mass[keep], dtype=np.float32),
                np.ascontiguousarray(ratio[keep], dtype=np.float32),
            )
        )
    return tuple(result)
