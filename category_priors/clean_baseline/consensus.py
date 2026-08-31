"""Class-agnostic multi-view mask consensus for the clean SAGA baseline.

The module deliberately operates on thresholded, compact Gaussian evidence.
It has no renderer, semantic, category-prior, ground-truth, or legacy
post-processing dependency.  A caller may supply a merge veto, but the raw
observer/supporter graph and its scores remain identical across conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


MergeVeto = Callable[[tuple[int, ...], np.ndarray], bool]
ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _readonly_unique_ids(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative IDs")
    result = np.unique(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MaskObservation:
    """One complete 2D mask lifted to a set of Gaussian IDs.

    ``ambiguous_ids`` are same-frame hierarchical-mask overlaps.  They remain
    available for final full-mask reconstruction, but never provide positive
    association, supporter, or conflict evidence.
    """

    mask_id: int
    frame_id: int
    gaussian_ids: np.ndarray
    ambiguous_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    _association_ids: np.ndarray = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if int(self.mask_id) < 0 or int(self.frame_id) < 0:
            raise ValueError("mask_id and frame_id must be non-negative")
        full = _readonly_unique_ids(self.gaussian_ids, name="gaussian_ids")
        ambiguous = _readonly_unique_ids(
            self.ambiguous_ids, name="ambiguous_ids"
        )
        if np.any(~np.isin(ambiguous, full)):
            raise ValueError("ambiguous_ids must be a subset of gaussian_ids")
        association = np.setdiff1d(full, ambiguous, assume_unique=True)
        association.setflags(write=False)
        object.__setattr__(self, "mask_id", int(self.mask_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "gaussian_ids", full)
        object.__setattr__(self, "ambiguous_ids", ambiguous)
        object.__setattr__(self, "_association_ids", association)

    @property
    def association_ids(self) -> np.ndarray:
        # This set is immutable.  Computing it on every access made the
        # scene-scale consensus path repeat the same ``setdiff1d`` millions of
        # times without changing a single result.
        return self._association_ids


@dataclass(frozen=True)
class ConsensusConfig:
    """Frozen ScanNet settings from the clean-baseline plan."""

    mask_visible_threshold: float = 0.30
    undersegment_filter_threshold: float = 0.30
    view_consensus_threshold: float = 0.90
    contained_threshold: float = 0.80
    point_filter_threshold: float = 0.50
    dbscan_eps_m: float = 0.10
    dbscan_min_samples: int = 4
    min_views: int = 2

    def validate(self) -> None:
        unit_values = (
            self.mask_visible_threshold,
            self.undersegment_filter_threshold,
            self.view_consensus_threshold,
            self.contained_threshold,
            self.point_filter_threshold,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in unit_values):
            raise ValueError("consensus thresholds must be finite values in [0, 1]")
        if not np.isfinite(self.dbscan_eps_m) or self.dbscan_eps_m <= 0.0:
            raise ValueError("dbscan_eps_m must be finite and positive")
        if self.dbscan_min_samples < 1 or self.min_views < 1:
            raise ValueError("DBSCAN support and min_views must be positive")


@dataclass(frozen=True)
class PairConsensus:
    left_mask_ids: tuple[int, ...]
    right_mask_ids: tuple[int, ...]
    observer_count: int
    supporter_count: int
    consensus: float


@dataclass(frozen=True)
class ConsensusEdge:
    left_mask_ids: tuple[int, ...]
    right_mask_ids: tuple[int, ...]
    observer_count: int
    supporter_count: int
    consensus: float
    observer_level: int


@dataclass(frozen=True)
class ConsensusObject:
    object_id: int
    mask_ids: tuple[int, ...]
    frame_ids: tuple[int, ...]
    gaussian_ids: np.ndarray
    mean_view_consensus: float
    mean_detection_ratio: float
    geometric_quality: float

    def __post_init__(self) -> None:
        ids = _readonly_unique_ids(self.gaussian_ids, name="gaussian_ids")
        object.__setattr__(self, "gaussian_ids", ids)


@dataclass(frozen=True)
class ConsensusResult:
    objects: tuple[ConsensusObject, ...]
    accepted_edges: tuple[ConsensusEdge, ...]
    rejected_undersegmented_mask_ids: tuple[int, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _FrameSupportIndex:
    """Exact sparse point-to-mask index for one physical frame.

    Most lifted association supports are disjoint after hierarchical
    ambiguity is removed.  The single-membership arrays keep that common path
    vectorised; the explicit multi-membership rows preserve the legacy result
    for malformed or synthetic overlapping inputs.
    """

    mask_count: int
    single_ids: np.ndarray
    single_labels: np.ndarray
    multi_ids: np.ndarray
    multi_labels: tuple[np.ndarray, ...]

    @classmethod
    def from_supports(
        cls, supports: Sequence[np.ndarray]
    ) -> "_FrameSupportIndex":
        mask_count = len(supports)
        nonempty = [
            (label, np.asarray(ids, dtype=np.int64))
            for label, ids in enumerate(supports)
            if len(ids)
        ]
        if not nonempty:
            empty = np.empty(0, dtype=np.int64)
            empty.setflags(write=False)
            return cls(mask_count, empty, empty, empty, ())

        ids = np.concatenate([row for _, row in nonempty])
        labels = np.concatenate(
            [
                np.full(len(row), label, dtype=np.int64)
                for label, row in nonempty
            ]
        )
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        sorted_labels = labels[order]
        unique_ids, starts, counts = np.unique(
            sorted_ids, return_index=True, return_counts=True
        )
        single = counts == 1
        single_ids = np.asarray(unique_ids[single], dtype=np.int64)
        single_labels = np.asarray(sorted_labels[starts[single]], dtype=np.int64)
        multi_ids = np.asarray(unique_ids[~single], dtype=np.int64)
        multi_labels = tuple(
            np.asarray(
                sorted_labels[start : start + count], dtype=np.int64
            )
            for start, count in zip(starts[~single], counts[~single])
        )
        for array in (single_ids, single_labels, multi_ids, *multi_labels):
            array.setflags(write=False)
        return cls(
            mask_count,
            single_ids,
            single_labels,
            multi_ids,
            multi_labels,
        )

    def counts(self, gaussian_ids: np.ndarray) -> np.ndarray:
        result = np.zeros(self.mask_count, dtype=np.int64)
        query = np.asarray(gaussian_ids, dtype=np.int64)
        if query.size == 0 or self.mask_count == 0:
            return result

        if self.single_ids.size:
            positions = np.searchsorted(self.single_ids, query)
            in_bounds = positions < self.single_ids.size
            matched = np.zeros(query.size, dtype=bool)
            matched[in_bounds] = (
                self.single_ids[positions[in_bounds]] == query[in_bounds]
            )
            if np.any(matched):
                result += np.bincount(
                    self.single_labels[positions[matched]],
                    minlength=self.mask_count,
                )

        if self.multi_ids.size:
            positions = np.searchsorted(self.multi_ids, query)
            in_bounds = positions < self.multi_ids.size
            safe_positions = np.minimum(positions, self.multi_ids.size - 1)
            matched = in_bounds & (
                self.multi_ids[safe_positions] == query
            )
            matched_positions = np.unique(positions[matched])
            for position in matched_positions:
                result += np.bincount(
                    self.multi_labels[int(position)],
                    minlength=self.mask_count,
                )
        return result


@dataclass(frozen=True)
class _PairFrameContext:
    ambiguity_ids: np.ndarray
    active_index: _FrameSupportIndex
    rejected_index: _FrameSupportIndex


@dataclass(frozen=True)
class _PairConsensusContext:
    visible: np.ndarray
    frames: tuple[_PairFrameContext, ...]
    config: ConsensusConfig


@dataclass(frozen=True)
class _ComponentPairState:
    observer_frames: int
    qualifier_bits_by_frame: tuple[int, ...]


def _prepare_pair_consensus_context(
    by_frame: Mapping[int, Sequence[MaskObservation]],
    abstain_masks_by_frame: Mapping[int, Sequence[np.ndarray]],
    visible: np.ndarray,
    config: ConsensusConfig,
    ambiguous_by_frame: Mapping[int, np.ndarray],
) -> _PairConsensusContext:
    """Build the immutable scene index used by every component pair.

    This is an exact index of the existing observer/supporter definition.  It
    changes neither the candidate graph nor any threshold; it only avoids
    rescanning every same-frame mask for every component pair.
    """

    empty_ids = np.empty(0, dtype=np.int64)
    empty_ids.setflags(write=False)
    frames: list[_PairFrameContext] = []
    for frame_id in range(visible.shape[0]):
        active_supports = tuple(
            item.association_ids for item in by_frame.get(frame_id, ())
        )
        rejected_supports = tuple(
            np.asarray(ids, dtype=np.int64)
            for ids in abstain_masks_by_frame.get(frame_id, ())
        )
        ambiguity = np.asarray(
            ambiguous_by_frame.get(frame_id, empty_ids), dtype=np.int64
        )
        ambiguity.setflags(write=False)
        frames.append(
            _PairFrameContext(
                ambiguity_ids=ambiguity,
                active_index=_FrameSupportIndex.from_supports(
                    active_supports
                ),
                rejected_index=_FrameSupportIndex.from_supports(
                    rejected_supports
                ),
            )
        )
    return _PairConsensusContext(
        visible=np.asarray(visible, dtype=bool),
        frames=tuple(frames),
        config=config,
    )


def _build_component_pair_state(
    support: np.ndarray,
    context: _PairConsensusContext,
) -> _ComponentPairState:
    """Materialise target-specific observer and containing-mask evidence.

    A component's observer eligibility and the masks that contain it depend
    only on that component, not on the other side of a proposed pair.  The
    legacy implementation recomputed those two facts for both sides of every
    pair.  Storing them once is mathematically identical, including negative
    observer frames with no active mask evidence and target-specific abstention
    caused by a rejected mask.
    """

    ids = np.asarray(support, dtype=np.int64)
    if ids.size == 0:
        return _ComponentPairState(
            observer_frames=0,
            qualifier_bits_by_frame=(0,) * len(context.frames),
        )

    observer_frames = 0
    qualifiers = [0] * len(context.frames)
    for frame_id, frame in enumerate(context.frames):
        visible_ids = ids[context.visible[frame_id, ids]]
        if frame.ambiguity_ids.size:
            visible_ids = np.setdiff1d(
                visible_ids,
                frame.ambiguity_ids,
                assume_unique=True,
            )
        if (
            visible_ids.size / ids.size
            < context.config.mask_visible_threshold
        ):
            continue

        rejected_counts = frame.rejected_index.counts(visible_ids)
        if rejected_counts.size and np.any(
            rejected_counts / visible_ids.size
            >= context.config.contained_threshold
        ):
            continue

        observer_frames |= 1 << frame_id
        active_counts = frame.active_index.counts(visible_ids)
        containing = np.flatnonzero(
            active_counts / visible_ids.size
            >= context.config.contained_threshold
        )
        qualifier_bits = 0
        for local_mask_index in containing:
            qualifier_bits |= 1 << int(local_mask_index)
        qualifiers[frame_id] = qualifier_bits

    return _ComponentPairState(
        observer_frames=observer_frames,
        qualifier_bits_by_frame=tuple(qualifiers),
    )


def _pair_consensus_from_states(
    left_state: _ComponentPairState,
    right_state: _ComponentPairState,
    left_masks: tuple[int, ...],
    right_masks: tuple[int, ...],
) -> PairConsensus:
    common_observers = (
        left_state.observer_frames & right_state.observer_frames
    )
    observers = int(common_observers.bit_count())
    supporters = 0
    remaining = common_observers
    while remaining:
        frame_bit = remaining & -remaining
        frame_id = frame_bit.bit_length() - 1
        if (
            left_state.qualifier_bits_by_frame[frame_id]
            & right_state.qualifier_bits_by_frame[frame_id]
        ):
            supporters += 1
        remaining ^= frame_bit
    consensus = supporters / observers if observers else 0.0
    return PairConsensus(
        left_masks,
        right_masks,
        observers,
        supporters,
        consensus,
    )


class _DeterministicUnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: int, right: int) -> int:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        keep, drop = sorted((left_root, right_root))
        self.parent[drop] = keep
        return keep


def _validate_inputs(
    observations: Sequence[MaskObservation],
    visibility: object,
    xyz_m: object | None = None,
) -> tuple[tuple[MaskObservation, ...], np.ndarray, np.ndarray | None]:
    items = tuple(observations)
    mask_ids = [item.mask_id for item in items]
    if len(mask_ids) != len(set(mask_ids)):
        raise ValueError("mask_id must be unique")
    visible = np.asarray(visibility)
    if visible.ndim != 2:
        raise ValueError("visibility must have shape [frames, gaussians]")
    if not np.issubdtype(visible.dtype, np.bool_):
        visible = np.asarray(visible, dtype=np.float64)
        if not np.isfinite(visible).all() or np.any(visible < 0.0):
            raise ValueError("visibility must be finite and non-negative")
    point_count = visible.shape[1]
    for item in items:
        if item.frame_id >= visible.shape[0]:
            raise ValueError("observation frame_id is outside visibility")
        if item.gaussian_ids.size and int(item.gaussian_ids[-1]) >= point_count:
            raise ValueError("observation references an unknown Gaussian")
    xyz = None
    if xyz_m is not None:
        xyz = np.asarray(xyz_m, dtype=np.float64)
        if xyz.shape != (point_count, 3):
            raise ValueError("xyz_m must have shape [gaussians, 3]")
        if not np.isfinite(xyz).all():
            raise ValueError("xyz_m must be finite")
    return items, visible, xyz


def _visible_boolean(visibility: np.ndarray, threshold: float) -> np.ndarray:
    if np.issubdtype(visibility.dtype, np.bool_):
        return np.asarray(visibility, dtype=bool)
    return np.asarray(visibility >= threshold, dtype=bool)


def detect_undersegmented_masks(
    observations: Sequence[MaskObservation],
    visibility: object,
    *,
    config: ConsensusConfig = ConsensusConfig(),
    progress_callback: ProgressCallback | None = None,
) -> tuple[int, ...]:
    """Find masks that are diversely partitioned in too many observable views.

    MaskClustering defines under-segmentation as a *frequency*: for every frame
    in which the target is sufficiently visible, form the frame's mask-ID
    distribution over that visible support.  A frame is diverse when no one
    assigned mask ID approximately contains the target.  The target is rejected
    only when the fraction of diverse observer frames is strictly greater than
    ``undersegment_filter_threshold``.

    Same-frame ambiguous Gaussians abstain from the distribution.  A frame with
    no remaining mask evidence is an abstention, not a background observation
    and not a spuriously diverse observation.
    """

    config.validate()
    items, visible_raw, _ = _validate_inputs(observations, visibility)
    visible = _visible_boolean(visible_raw, config.mask_visible_threshold)
    rejected: list[int] = []
    by_frame: dict[int, list[MaskObservation]] = {}
    ambiguous_by_frame: dict[int, np.ndarray] = {}
    for item in items:
        by_frame.setdefault(item.frame_id, []).append(item)
    for frame_id, frame_items in by_frame.items():
        rows = [item.ambiguous_ids for item in frame_items if item.ambiguous_ids.size]
        ambiguous_by_frame[frame_id] = (
            np.unique(np.concatenate(rows))
            if rows
            else np.empty(0, dtype=np.int64)
        )
    # For each physical frame, collapse all non-ambiguous mask assignments to
    # one sorted ID/label row.  The old implementation intersected the target
    # separately with every same-frame mask, giving target x frame x mask
    # complexity.  This index preserves the same deterministic "later sorted
    # mask wins" behavior for malformed overlapping inputs while requiring one
    # intersection per target/frame.
    assignments_by_frame: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame_id, frame_items in by_frame.items():
        assigned_ids: list[np.ndarray] = []
        assigned_labels: list[np.ndarray] = []
        for local_label, candidate in enumerate(
            sorted(frame_items, key=lambda item: item.mask_id)
        ):
            ids = candidate.association_ids
            if ids.size == 0:
                continue
            assigned_ids.append(ids)
            assigned_labels.append(
                np.full(ids.size, local_label, dtype=np.int64)
            )
        if not assigned_ids:
            continue
        concatenated_ids = np.concatenate(assigned_ids)
        concatenated_labels = np.concatenate(assigned_labels)
        order = np.argsort(concatenated_ids, kind="stable")
        sorted_ids = concatenated_ids[order]
        sorted_labels = concatenated_labels[order]
        keep_last = np.ones(sorted_ids.size, dtype=bool)
        keep_last[:-1] = sorted_ids[:-1] != sorted_ids[1:]
        assignments_by_frame[frame_id] = (
            sorted_ids[keep_last],
            sorted_labels[keep_last],
        )

    ordered_target_indices = sorted(
        range(len(items)), key=lambda index: items[index].mask_id
    )
    for target_offset, target_index in enumerate(ordered_target_indices, start=1):
        target = items[target_index]
        target_ids = target.association_ids
        if target_ids.size == 0:
            continue
        observable_frames = 0
        diverse_frames = 0
        # Frames without any mask assignment are geometry abstentions for this
        # statistic, so they can be skipped entirely.
        for frame_id, (assigned_ids, assigned_labels) in sorted(
            assignments_by_frame.items()
        ):
            visible_target = target_ids[visible[frame_id, target_ids]]
            frame_ambiguous = ambiguous_by_frame.get(
                frame_id, np.empty(0, dtype=np.int64)
            )
            if frame_ambiguous.size:
                visible_target = np.setdiff1d(
                    visible_target, frame_ambiguous, assume_unique=True
                )
            if (
                visible_target.size == 0
                or visible_target.size / target_ids.size
                < config.mask_visible_threshold
            ):
                continue
            # -1 is the explicit unassigned/background mask ID.  MaskClustering
            # only treats a frame as an observer after some *mask* evidence is
            # present, and computes diversity over assigned mask IDs rather
            # than letting uncovered/background support form a fake class.
            assigned_positions = np.searchsorted(assigned_ids, visible_target)
            in_bounds = assigned_positions < assigned_ids.size
            matched = np.zeros(visible_target.size, dtype=bool)
            matched[in_bounds] = (
                assigned_ids[assigned_positions[in_bounds]]
                == visible_target[in_bounds]
            )
            if not np.any(matched):
                continue
            observable_frames += 1
            _, counts = np.unique(
                assigned_labels[assigned_positions[matched]], return_counts=True
            )
            dominant_fraction = float(counts.max() / np.count_nonzero(matched))
            if dominant_fraction < config.contained_threshold:
                diverse_frames += 1

        if (
            observable_frames > 0
            and diverse_frames / observable_frames
            > config.undersegment_filter_threshold
        ):
            rejected.append(target.mask_id)
        if (
            progress_callback is not None
            and (
                target_offset == len(ordered_target_indices)
                or target_offset % 250 == 0
            )
        ):
            progress_callback(
                "undersegmentation-progress",
                {
                    "completed_mask_count": target_offset,
                    "mask_count": len(ordered_target_indices),
                    "rejected_mask_count": len(rejected),
                },
            )
    return tuple(sorted(rejected))


def _component_support(
    component: Iterable[int], observations: Sequence[MaskObservation]
) -> np.ndarray:
    arrays = [observations[index].association_ids for index in component]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def _component_full(
    component: Iterable[int], observations: Sequence[MaskObservation]
) -> np.ndarray:
    arrays = [observations[index].gaussian_ids for index in component]
    if not arrays:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays))


def _ambiguity_by_frame(
    by_frame: Mapping[int, Sequence[MaskObservation]],
) -> dict[int, np.ndarray]:
    """Materialise immutable same-frame ambiguity once per scene context."""

    result: dict[int, np.ndarray] = {}
    for frame_id, frame_masks in by_frame.items():
        rows = [mask.ambiguous_ids for mask in frame_masks if mask.ambiguous_ids.size]
        ambiguous = (
            np.unique(np.concatenate(rows))
            if rows
            else np.empty(0, dtype=np.int64)
        )
        ambiguous.setflags(write=False)
        result[int(frame_id)] = ambiguous
    return result


def compute_pair_consensus(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    observations: Sequence[MaskObservation],
    visibility: object,
    *,
    config: ConsensusConfig = ConsensusConfig(),
    active_indices: Sequence[int] | None = None,
    rejected_mask_ids: Sequence[int] = (),
) -> PairConsensus:
    """Compute observer/supporter view consensus between two components."""

    config.validate()
    items, visible_raw, _ = _validate_inputs(observations, visibility)
    visible = _visible_boolean(visible_raw, config.mask_visible_threshold)
    rejected_set = {int(value) for value in rejected_mask_ids}
    known_mask_ids = {item.mask_id for item in items}
    if not rejected_set.issubset(known_mask_ids):
        raise ValueError("rejected_mask_ids contains an unknown mask")
    active_raw = (
        tuple(range(len(items)))
        if active_indices is None
        else tuple(int(index) for index in active_indices)
    )
    left = tuple(sorted(set(int(value) for value in left_indices)))
    right = tuple(sorted(set(int(value) for value in right_indices)))
    if not left or not right or set(left) & set(right):
        raise ValueError("pair components must be non-empty and disjoint")
    if any(index < 0 or index >= len(items) for index in left + right + active_raw):
        raise ValueError("component index is outside observations")
    active = tuple(
        index for index in active_raw if items[index].mask_id not in rejected_set
    )
    if any(items[index].mask_id in rejected_set for index in left + right):
        raise ValueError("a rejected mask cannot form a pair component")

    left_support = _component_support(left, items)
    right_support = _component_support(right, items)
    left_masks = tuple(sorted(items[index].mask_id for index in left))
    right_masks = tuple(sorted(items[index].mask_id for index in right))
    by_frame: dict[int, list[MaskObservation]] = {}
    for index in active:
        by_frame.setdefault(items[index].frame_id, []).append(items[index])
    rejected_support_by_frame: dict[int, list[np.ndarray]] = {}
    for item in items:
        if item.mask_id in rejected_set and item.association_ids.size:
            rejected_support_by_frame.setdefault(item.frame_id, []).append(
                item.association_ids
            )
    ambiguous_by_frame = _ambiguity_by_frame(by_frame)
    return _pair_consensus_from_support(
        left_support,
        right_support,
        left_masks,
        right_masks,
        by_frame,
        visible,
        config,
        rejected_support_by_frame,
        ambiguous_by_frame,
    )


def _pair_consensus_from_support(
    left_support: np.ndarray,
    right_support: np.ndarray,
    left_masks: tuple[int, ...],
    right_masks: tuple[int, ...],
    by_frame: dict[int, list[MaskObservation]],
    visible: np.ndarray,
    config: ConsensusConfig,
    abstain_masks_by_frame: Mapping[int, Sequence[np.ndarray]] | None = None,
    ambiguous_by_frame: Mapping[int, np.ndarray] | None = None,
    left_state: _ComponentPairState | None = None,
    right_state: _ComponentPairState | None = None,
) -> PairConsensus:
    if (left_state is None) != (right_state is None):
        raise ValueError("prepared pair states must be provided together")
    if left_state is not None and right_state is not None:
        return _pair_consensus_from_states(
            left_state,
            right_state,
            left_masks,
            right_masks,
        )
    if left_support.size == 0 or right_support.size == 0:
        return PairConsensus(left_masks, right_masks, 0, 0, 0.0)
    observers = 0
    supporters = 0
    frame_ambiguity = (
        _ambiguity_by_frame(by_frame)
        if ambiguous_by_frame is None
        else ambiguous_by_frame
    )
    for frame_id in range(visible.shape[0]):
        frame_masks = by_frame.get(frame_id, ())
        left_visible = left_support[visible[frame_id, left_support]]
        right_visible = right_support[visible[frame_id, right_support]]
        frame_ambiguous = frame_ambiguity.get(
            frame_id, np.empty(0, dtype=np.int64)
        )
        if frame_ambiguous.size:
            left_visible = np.setdiff1d(
                left_visible, frame_ambiguous, assume_unique=True
            )
            right_visible = np.setdiff1d(
                right_visible, frame_ambiguous, assume_unique=True
            )
        if (
            left_visible.size / left_support.size < config.mask_visible_threshold
            or right_visible.size / right_support.size < config.mask_visible_threshold
        ):
            continue
        # Filtering an under-segmented mask must only withdraw the target/frame
        # visibility entries for which that rejected mask was actually the
        # containing witness.  Clearing its whole physical frame globally
        # would erase valid negative evidence for unrelated objects.
        abstainers = (
            ()
            if abstain_masks_by_frame is None
            else abstain_masks_by_frame.get(frame_id, ())
        )
        target_abstains = False
        for rejected_support in abstainers:
            left_coverage = (
                np.intersect1d(
                    left_visible, rejected_support, assume_unique=True
                ).size
                / left_visible.size
            )
            right_coverage = (
                np.intersect1d(
                    right_visible, rejected_support, assume_unique=True
                ).size
                / right_visible.size
            )
            if (
                left_coverage >= config.contained_threshold
                or right_coverage >= config.contained_threshold
            ):
                target_abstains = True
                break
        if target_abstains:
            continue
        observers += 1
        for candidate in frame_masks:
            support = candidate.association_ids
            left_coverage = (
                np.intersect1d(left_visible, support, assume_unique=True).size
                / left_visible.size
            )
            right_coverage = (
                np.intersect1d(right_visible, support, assume_unique=True).size
                / right_visible.size
            )
            if (
                left_coverage >= config.contained_threshold
                and right_coverage >= config.contained_threshold
            ):
                supporters += 1
                break
    consensus = supporters / observers if observers else 0.0
    return PairConsensus(left_masks, right_masks, observers, supporters, consensus)


def _sparse_raw_candidate_pairs(
    observations: Sequence[MaskObservation],
    active_indices: Sequence[int],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[tuple[int, int], ...]:
    """Generate only pairs that could share at least one supporter mask.

    A supporter must overlap both sides, so it is sufficient to enumerate all
    observations that co-occur in the inverted Gaussian support list of one
    active mask.  The result is a conservative superset of qualifying pairs.
    """

    point_to_masks: dict[int, list[int]] = {}
    active_count = len(active_indices)
    for active_offset, index in enumerate(active_indices, start=1):
        for gaussian_id in observations[index].association_ids:
            point_to_masks.setdefault(int(gaussian_id), []).append(int(index))
        if (
            progress_callback is not None
            and (active_offset == active_count or active_offset % 500 == 0)
        ):
            progress_callback(
                "raw-candidate-index-progress",
                {
                    "completed_mask_count": active_offset,
                    "active_mask_count": active_count,
                    "indexed_gaussian_count": len(point_to_masks),
                },
            )
    pairs: set[tuple[int, int]] = set()
    for supporter_offset, supporter_index in enumerate(active_indices, start=1):
        neighbours: set[int] = set()
        for gaussian_id in observations[supporter_index].association_ids:
            neighbours.update(point_to_masks.get(int(gaussian_id), ()))
        for left, right in combinations(sorted(neighbours), 2):
            if observations[left].frame_id == observations[right].frame_id:
                continue
            pairs.add((left, right))
        if (
            progress_callback is not None
            and (supporter_offset == active_count or supporter_offset % 250 == 0)
        ):
            progress_callback(
                "raw-candidate-pair-progress",
                {
                    "completed_supporter_count": supporter_offset,
                    "active_mask_count": active_count,
                    "candidate_pair_count": len(pairs),
                },
            )
    return tuple(sorted(pairs))


def _dbscan_labels(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Small deterministic DBSCAN implementation using SciPy's radius query."""

    from scipy.spatial import cKDTree

    count = len(points)
    labels = np.full(count, -1, dtype=np.int64)
    if count == 0:
        return labels
    tree = cKDTree(points)
    neighborhoods = [
        np.asarray(sorted(tree.query_ball_point(points[index], eps)), dtype=np.int64)
        for index in range(count)
    ]
    core = np.asarray(
        [len(neighbors) >= min_samples for neighbors in neighborhoods], dtype=bool
    )
    cluster_id = 0
    visited_core = np.zeros(count, dtype=bool)
    for seed in range(count):
        if not core[seed] or visited_core[seed]:
            continue
        queue = [seed]
        visited_core[seed] = True
        labels[seed] = cluster_id
        cursor = 0
        while cursor < len(queue):
            current = queue[cursor]
            cursor += 1
            for neighbor in neighborhoods[current]:
                neighbor_id = int(neighbor)
                if labels[neighbor_id] < 0:
                    labels[neighbor_id] = cluster_id
                if core[neighbor_id] and not visited_core[neighbor_id]:
                    visited_core[neighbor_id] = True
                    queue.append(neighbor_id)
        cluster_id += 1
    return labels


def split_disconnected_support(
    gaussian_ids: object,
    xyz_m: object,
    *,
    eps_m: float = 0.10,
    min_samples: int = 4,
) -> tuple[np.ndarray, ...]:
    """Split support into deterministic physical DBSCAN components."""

    ids = _readonly_unique_ids(gaussian_ids, name="gaussian_ids")
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not np.isfinite(xyz).all():
        raise ValueError("xyz_m must be a finite [gaussians, 3] array")
    if ids.size and int(ids[-1]) >= len(xyz):
        raise ValueError("gaussian_ids references an unknown coordinate")
    if not np.isfinite(eps_m) or eps_m <= 0.0 or min_samples < 1:
        raise ValueError("DBSCAN settings must be positive")
    labels = _dbscan_labels(xyz[ids], float(eps_m), int(min_samples))
    parts: list[np.ndarray] = []
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        part = np.asarray(ids[labels == label], dtype=np.int64)
        part.setflags(write=False)
        parts.append(part)
    return tuple(parts)


def remove_contained_objects(
    objects: Sequence[ConsensusObject],
    *,
    contained_threshold: float = 0.80,
) -> tuple[ConsensusObject, ...]:
    """Drop lower-quality duplicate objects using point-set containment."""

    if not np.isfinite(contained_threshold) or not 0.0 <= contained_threshold <= 1.0:
        raise ValueError("contained_threshold must be in [0, 1]")
    ranked = sorted(
        objects,
        key=lambda item: (
            -item.geometric_quality,
            -len(item.gaussian_ids),
            item.mask_ids,
        ),
    )
    kept: list[ConsensusObject] = []
    for candidate in ranked:
        duplicate = False
        for existing in kept:
            intersection = np.intersect1d(
                candidate.gaussian_ids,
                existing.gaussian_ids,
                assume_unique=True,
            ).size
            denominator = min(len(candidate.gaussian_ids), len(existing.gaussian_ids))
            containment = intersection / denominator if denominator else 0.0
            if containment >= contained_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return tuple(
        sorted(
            kept,
            key=lambda item: (
                item.mask_ids,
                int(item.gaussian_ids[0]) if item.gaussian_ids.size else -1,
            ),
        )
    )


def run_mask_consensus(
    observations: Sequence[MaskObservation],
    visibility: object,
    xyz_m: object,
    *,
    config: ConsensusConfig = ConsensusConfig(),
    merge_veto: MergeVeto | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ConsensusResult:
    """Build deterministic objects from complete lifted masks.

    The optional veto sees only stable mask IDs and the proposed full Gaussian
    union.  It cannot change raw observer/supporter evidence or AP scoring.
    """

    config.validate()
    items, visible_raw, xyz = _validate_inputs(observations, visibility, xyz_m)
    assert xyz is not None

    def emit_progress(stage: str, **payload: object) -> None:
        if progress_callback is not None:
            progress_callback(stage, payload)

    emit_progress(
        "validated-inputs",
        mask_count=len(items),
        frame_count=int(visible_raw.shape[0]),
        gaussian_count=int(visible_raw.shape[1]),
    )
    visible = _visible_boolean(visible_raw, config.mask_visible_threshold)
    emit_progress("undersegmentation-start", mask_count=len(items))
    rejected = detect_undersegmented_masks(
        items,
        visible_raw,
        config=config,
        progress_callback=(
            None
            if progress_callback is None
            else lambda stage, payload: progress_callback(stage, payload)
        ),
    )
    rejected_set = set(rejected)
    emit_progress(
        "undersegmentation-complete",
        rejected_mask_count=len(rejected),
    )
    active = tuple(
        index for index, item in enumerate(items) if item.mask_id not in rejected_set
    )
    raw_candidate_pairs = _sparse_raw_candidate_pairs(
        items,
        active,
        progress_callback=(
            None
            if progress_callback is None
            else lambda stage, payload: progress_callback(stage, payload)
        ),
    )
    by_frame: dict[int, list[MaskObservation]] = {}
    for index in active:
        by_frame.setdefault(items[index].frame_id, []).append(items[index])
    ambiguity_by_frame_active = _ambiguity_by_frame(by_frame)
    all_by_frame: dict[int, list[MaskObservation]] = {}
    for item in items:
        all_by_frame.setdefault(item.frame_id, []).append(item)
    ambiguity_by_frame_all = _ambiguity_by_frame(all_by_frame)
    # Rejected masks are removed as graph nodes/supporters.  Their effect on an
    # observer denominator is target-specific: only a target that was actually
    # contained by that rejected mask abstains in its source frame.
    rejected_support_by_frame: dict[int, list[np.ndarray]] = {}
    for item in items:
        if item.mask_id in rejected_set and item.association_ids.size:
            rejected_support_by_frame.setdefault(item.frame_id, []).append(
                item.association_ids
            )
    uf = _DeterministicUnionFind(len(items))
    components: dict[int, set[int]] = {index: {index} for index in active}
    component_supports = {
        index: items[index].association_ids for index in active
    }
    component_full = {index: items[index].gaussian_ids for index in active}
    component_mask_ids = {index: (items[index].mask_id,) for index in active}
    component_frame_ids = {index: {items[index].frame_id} for index in active}
    component_revisions = {index: 0 for index in active}
    pair_context = _prepare_pair_consensus_context(
        by_frame,
        rejected_support_by_frame,
        visible,
        config,
        ambiguity_by_frame_active,
    )
    component_pair_states: dict[int, _ComponentPairState] = {}
    component_state_build_count = 0
    emit_progress(
        "component-pair-state-start",
        component_count=len(component_supports),
    )
    for state_offset, index in enumerate(sorted(component_supports), start=1):
        component_pair_states[index] = _build_component_pair_state(
            component_supports[index], pair_context
        )
        component_state_build_count += 1
        if state_offset == len(component_supports) or state_offset % 250 == 0:
            emit_progress(
                "component-pair-state-progress",
                completed_component_count=state_offset,
                component_count=len(component_supports),
            )
    emit_progress(
        "component-pair-state-complete",
        component_count=len(component_pair_states),
        component_state_build_count=component_state_build_count,
    )
    accepted_edges: list[ConsensusEdge] = []
    pair_cache: dict[tuple[int, int], PairConsensus] = {}
    pair_evaluation_count = 0

    def evaluate_pair(left_root: int, right_root: int) -> PairConsensus:
        nonlocal pair_evaluation_count
        pair_evaluation_count += 1
        return _pair_consensus_from_support(
            component_supports[left_root],
            component_supports[right_root],
            component_mask_ids[left_root],
            component_mask_ids[right_root],
            by_frame,
            visible,
            config,
            rejected_support_by_frame,
            ambiguity_by_frame_active,
            component_pair_states[left_root],
            component_pair_states[right_root],
        )

    def rebuild_pair_cache(
        previous_cache: Mapping[tuple[int, int], PairConsensus] | None = None,
        previous_revisions: Mapping[tuple[int, int], tuple[int, int]] | None = None,
    ) -> tuple[
        dict[tuple[int, int], PairConsensus],
        dict[tuple[int, int], tuple[int, int]],
    ]:
        current_pairs: set[tuple[int, int]] = set()
        for left_index, right_index in raw_candidate_pairs:
            left_root = uf.find(left_index)
            right_root = uf.find(right_index)
            if left_root == right_root:
                continue
            left_root, right_root = sorted((left_root, right_root))
            if left_root in components and right_root in components:
                current_pairs.add((left_root, right_root))
        cache: dict[tuple[int, int], PairConsensus] = {}
        revisions: dict[tuple[int, int], tuple[int, int]] = {}
        for pair in sorted(current_pairs):
            revision = (
                component_revisions[pair[0]],
                component_revisions[pair[1]],
            )
            if (
                previous_cache is not None
                and previous_revisions is not None
                and pair in previous_cache
                and previous_revisions.get(pair) == revision
            ):
                cache[pair] = previous_cache[pair]
            else:
                cache[pair] = evaluate_pair(*pair)
            revisions[pair] = revision
        return cache, revisions

    emit_progress(
        "raw-candidate-pairs-complete",
        active_mask_count=len(active),
        sparse_candidate_pair_count=len(raw_candidate_pairs),
    )
    pair_cache, pair_cache_revisions = rebuild_pair_cache()
    emit_progress(
        "initial-pair-consensus-complete",
        pair_count=len(pair_cache),
        pair_evaluation_count=pair_evaluation_count,
    )
    initial_pair_rows = [
        {
            "left_mask_ids": list(evidence.left_mask_ids),
            "right_mask_ids": list(evidence.right_mask_ids),
            "observer_count": int(evidence.observer_count),
            "supporter_count": int(evidence.supporter_count),
        }
        for _, evidence in sorted(pair_cache.items())
    ]
    raw_graph_identity = hashlib.sha256(
        json.dumps(
            initial_pair_rows,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    positive_observer_counts = sorted(
        evidence.observer_count
        for evidence in pair_cache.values()
        if evidence.observer_count > 0
    )
    observer_schedule: list[tuple[int, int]] = []
    if positive_observer_counts:
        count = len(positive_observer_counts)
        for top_percent in range(5, 100, 5):
            # Keep the top p% by requiring n >= this deterministic order
            # statistic.  Repeated numeric thresholds remain separate rounds:
            # graph contraction can expose new edges after recomputation.
            selected_count = (top_percent * count + 99) // 100
            rank = max(0, count - selected_count)
            observer_schedule.append(
                (top_percent, int(positive_observer_counts[rank]))
            )

    rejected_batch_keys: set[tuple[int, ...]] = set()
    completed_schedule: list[dict[str, int]] = []
    for top_percent, observer_threshold in observer_schedule:
        qualified: list[tuple[int, int, PairConsensus]] = []
        for (left_root, right_root), evidence in pair_cache.items():
            if left_root not in components or right_root not in components:
                continue
            if component_frame_ids[left_root] & component_frame_ids[right_root]:
                continue
            if (
                evidence.observer_count < observer_threshold
                or evidence.consensus < config.view_consensus_threshold
            ):
                continue
            qualified.append((left_root, right_root, evidence))
        qualified.sort(
            key=lambda row: (
                -row[2].consensus,
                -row[2].observer_count,
                -row[2].supporter_count,
                row[2].left_mask_ids,
                row[2].right_mask_ids,
            )
        )

        # Build the round graph in consensus-descending order.  Edges that
        # would put two alternative masks from the same physical frame in one
        # component are omitted.  No production component is mutated here;
        # every resulting connected component is committed simultaneously.
        batch_parent = {root: root for root in components}
        batch_frames = {
            root: set(component_frame_ids[root]) for root in components
        }

        def batch_find(root: int) -> int:
            while batch_parent[root] != root:
                root = batch_parent[root]
            return root

        selected_edges: list[tuple[int, int, PairConsensus]] = []
        for left_root, right_root, evidence in qualified:
            left_batch = batch_find(left_root)
            right_batch = batch_find(right_root)
            if left_batch == right_batch:
                continue
            if batch_frames[left_batch] & batch_frames[right_batch]:
                continue
            keep, drop = sorted((left_batch, right_batch))
            batch_parent[drop] = keep
            batch_frames[keep] |= batch_frames.pop(drop)
            selected_edges.append((left_root, right_root, evidence))

        batch_groups: dict[int, list[int]] = {}
        for root in sorted(components):
            batch_groups.setdefault(batch_find(root), []).append(root)

        accepted_groups: list[tuple[list[int], tuple[int, ...], np.ndarray]] = []
        for roots in sorted(
            batch_groups.values(),
            key=lambda values: tuple(
                sorted(
                    mask_id
                    for root in values
                    for mask_id in component_mask_ids[root]
                )
            ),
        ):
            if len(roots) < 2:
                continue
            mask_ids = tuple(
                sorted(
                    mask_id
                    for root in roots
                    for mask_id in component_mask_ids[root]
                )
            )
            full_ids = np.unique(
                np.concatenate([component_full[root] for root in roots])
            )
            if mask_ids in rejected_batch_keys:
                continue
            if merge_veto is not None and not bool(merge_veto(mask_ids, full_ids)):
                # Whole-component all-or-nothing rejection.  No subset is
                # silently merged, and an unchanged component is not submitted
                # to the veto again at every repeated percentile threshold.
                rejected_batch_keys.add(mask_ids)
                continue
            accepted_groups.append((roots, mask_ids, full_ids))

        for roots, mask_ids, full_ids in accepted_groups:
            component_state_build_count += 1
            merged_revision = 1 + max(component_revisions[root] for root in roots)
            merged_indices = set().union(*(components[root] for root in roots))
            merged_support = np.unique(
                np.concatenate([component_supports[root] for root in roots])
            )
            merged_frames = set().union(
                *(component_frame_ids[root] for root in roots)
            )
            for root in roots:
                components.pop(root)
                component_supports.pop(root)
                component_full.pop(root)
                component_mask_ids.pop(root)
                component_frame_ids.pop(root)
                component_revisions.pop(root)
                component_pair_states.pop(root)
            root = roots[0]
            for other in roots[1:]:
                root = uf.union(root, other)
            components[root] = merged_indices
            component_supports[root] = merged_support
            component_full[root] = full_ids
            component_mask_ids[root] = mask_ids
            component_frame_ids[root] = merged_frames
            component_revisions[root] = merged_revision
            component_pair_states[root] = _build_component_pair_state(
                merged_support, pair_context
            )

            root_set = set(roots)
            for left_root, right_root, evidence in selected_edges:
                if left_root in root_set and right_root in root_set:
                    accepted_edges.append(
                        ConsensusEdge(
                            evidence.left_mask_ids,
                            evidence.right_mask_ids,
                            evidence.observer_count,
                            evidence.supporter_count,
                            evidence.consensus,
                            observer_threshold,
                        )
                    )

        completed_schedule.append(
            {
                "top_percent": top_percent,
                "observer_threshold": observer_threshold,
                "qualified_edge_count": len(qualified),
                "accepted_component_count": len(accepted_groups),
            }
        )
        if accepted_groups:
            pair_cache, pair_cache_revisions = rebuild_pair_cache(
                pair_cache, pair_cache_revisions
            )
        emit_progress(
            "observer-round-complete",
            top_percent=top_percent,
            observer_threshold=observer_threshold,
            qualified_edge_count=len(qualified),
            accepted_component_count=len(accepted_groups),
            remaining_component_count=len(components),
            pair_count=len(pair_cache),
            pair_evaluation_count=pair_evaluation_count,
            component_state_build_count=component_state_build_count,
        )

    provisional: list[ConsensusObject] = []
    dropped_by_views = 0
    dropped_by_detection = 0
    dropped_by_connectivity = 0
    for component in sorted(
        components.values(),
        key=lambda members: tuple(sorted(items[index].mask_id for index in members)),
    ):
        component_frame_ids_final = tuple(
            sorted({items[index].frame_id for index in component})
        )
        if len(component_frame_ids_final) < config.min_views:
            dropped_by_views += 1
            continue
        full_ids = _component_full(component, items)
        if full_ids.size == 0:
            dropped_by_detection += 1
            continue
        # Accumulate one frame at a time.  Materialising visible[:, full_ids]
        # used memory proportional to frames x component points, even though
        # only the two per-point counts survive this loop.
        visible_counts = np.zeros(full_ids.size, dtype=np.int64)
        detection_counts = np.zeros(full_ids.size, dtype=np.int64)
        member_indices_by_frame: dict[int, list[int]] = {}
        for index in component:
            member_indices_by_frame.setdefault(items[index].frame_id, []).append(index)
        for frame_id in range(visible.shape[0]):
            eligible_in_frame = visible[frame_id, full_ids].copy()
            ambiguous_ids = ambiguity_by_frame_all.get(
                frame_id, np.empty(0, dtype=np.int64)
            )
            if ambiguous_ids.size:
                eligible_in_frame &= ~np.isin(
                    full_ids, ambiguous_ids, assume_unique=True
                )
            visible_counts += eligible_in_frame.astype(np.int64)
            frame_indices = member_indices_by_frame.get(frame_id, ())
            if not frame_indices:
                continue
            detected_in_frame = np.zeros(full_ids.size, dtype=bool)
            for index in frame_indices:
                detected_in_frame |= np.isin(
                    full_ids, items[index].association_ids, assume_unique=True
                )
            # Detection frequency is conditioned on the same eligible frame
            # evidence as its denominator.  In particular, invisible and
            # same-frame ambiguous points can neither help nor hurt the ratio.
            detection_counts += (
                detected_in_frame & eligible_in_frame
            ).astype(np.int64)
        ratios = np.divide(
            detection_counts,
            visible_counts,
            out=np.zeros(full_ids.size, dtype=np.float64),
            where=visible_counts > 0,
        )
        keep = ratios >= config.point_filter_threshold
        filtered_ids = full_ids[keep]
        filtered_ratios = ratios[keep]
        if filtered_ids.size == 0:
            dropped_by_detection += 1
            continue
        physical_parts = split_disconnected_support(
            filtered_ids,
            xyz,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        )
        if not physical_parts:
            dropped_by_connectivity += 1
            continue
        for part in physical_parts:
            # Physical splitting creates distinct output objects.  Each part
            # retains only complete masks that actually support that part;
            # otherwise a one-view fragment inherits unrelated parent-track
            # views and can masquerade as a valid multi-view object.
            part_indices = tuple(
                index
                for index in sorted(
                    component, key=lambda value: items[value].mask_id
                )
                if np.intersect1d(
                    part, items[index].association_ids, assume_unique=True
                ).size
                > 0
            )
            part_frame_ids = tuple(
                sorted({items[index].frame_id for index in part_indices})
            )
            if len(part_frame_ids) < config.min_views:
                dropped_by_views += 1
                continue
            part_mask_ids = tuple(
                sorted(items[index].mask_id for index in part_indices)
            )
            component_edges = [
                edge
                for edge in accepted_edges
                if set(edge.left_mask_ids + edge.right_mask_ids).issubset(
                    part_mask_ids
                )
            ]
            mean_consensus = (
                float(np.mean([edge.consensus for edge in component_edges]))
                if component_edges
                else 0.0
            )
            positions = np.searchsorted(filtered_ids, part)
            mean_detection = float(np.mean(filtered_ratios[positions]))
            geometric_quality = float(
                np.sqrt(max(0.0, mean_consensus) * max(0.0, mean_detection))
            )
            provisional.append(
                ConsensusObject(
                    object_id=-1,
                    mask_ids=part_mask_ids,
                    frame_ids=part_frame_ids,
                    gaussian_ids=part,
                    mean_view_consensus=mean_consensus,
                    mean_detection_ratio=mean_detection,
                    geometric_quality=geometric_quality,
                )
            )

    deduplicated = remove_contained_objects(
        provisional, contained_threshold=config.contained_threshold
    )
    objects = tuple(
        ConsensusObject(
            object_id=index,
            mask_ids=item.mask_ids,
            frame_ids=item.frame_ids,
            gaussian_ids=item.gaussian_ids,
            mean_view_consensus=item.mean_view_consensus,
            mean_detection_ratio=item.mean_detection_ratio,
            geometric_quality=item.geometric_quality,
        )
        for index, item in enumerate(deduplicated)
    )
    diagnostics: dict[str, object] = {
        "observation_count": len(items),
        "active_observation_count": len(active),
        "total_possible_cross_frame_pair_count": int(
            len(active) * (len(active) - 1) // 2
            - sum(
                len(frame_items) * (len(frame_items) - 1) // 2
                for frame_items in by_frame.values()
            )
        ),
        "sparse_candidate_pair_count": len(raw_candidate_pairs),
        "raw_graph_identity": raw_graph_identity,
        "raw_pair_evidence_count": len(initial_pair_rows),
        "pair_evaluation_count": pair_evaluation_count,
        "component_state_build_count": component_state_build_count,
        "undersegmented_mask_count": len(rejected),
        "undersegmented_source_frame_count": len(rejected_support_by_frame),
        "accepted_edge_count": len(accepted_edges),
        "observer_schedule": completed_schedule,
        "component_count_before_output_filters": len(components),
        "dropped_by_min_views": dropped_by_views,
        "dropped_by_detection_ratio": dropped_by_detection,
        "dropped_by_physical_connectivity": dropped_by_connectivity,
        "contained_duplicate_count": len(provisional) - len(deduplicated),
        "object_count": len(objects),
    }
    return ConsensusResult(objects, tuple(accepted_edges), rejected, diagnostics)


__all__ = [
    "ConsensusConfig",
    "ConsensusEdge",
    "ConsensusObject",
    "ConsensusResult",
    "MaskObservation",
    "MergeVeto",
    "PairConsensus",
    "compute_pair_consensus",
    "detect_undersegmented_masks",
    "remove_contained_objects",
    "run_mask_consensus",
    "split_disconnected_support",
]
