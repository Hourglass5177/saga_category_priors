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
from typing import Callable, Iterable, Sequence

import numpy as np


MergeVeto = Callable[[tuple[int, ...], np.ndarray], bool]


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

    def __post_init__(self) -> None:
        if int(self.mask_id) < 0 or int(self.frame_id) < 0:
            raise ValueError("mask_id and frame_id must be non-negative")
        full = _readonly_unique_ids(self.gaussian_ids, name="gaussian_ids")
        ambiguous = _readonly_unique_ids(
            self.ambiguous_ids, name="ambiguous_ids"
        )
        if np.any(~np.isin(ambiguous, full)):
            raise ValueError("ambiguous_ids must be a subset of gaussian_ids")
        object.__setattr__(self, "mask_id", int(self.mask_id))
        object.__setattr__(self, "frame_id", int(self.frame_id))
        object.__setattr__(self, "gaussian_ids", full)
        object.__setattr__(self, "ambiguous_ids", ambiguous)

    @property
    def association_ids(self) -> np.ndarray:
        result = np.setdiff1d(
            self.gaussian_ids, self.ambiguous_ids, assume_unique=True
        )
        result.setflags(write=False)
        return result


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
) -> tuple[int, ...]:
    """Find masks that are diversely partitioned in too many observable views.

    MaskClustering defines under-segmentation as a *frequency*: for every frame
    in which the target is sufficiently visible, form the frame's mask-ID
    distribution over that visible support.  A frame is diverse when no one
    mask ID (including the unassigned/background ID) approximately contains the
    target.  The target is rejected only when the fraction of diverse observer
    frames is strictly greater than ``undersegment_filter_threshold``.

    Same-frame ambiguous Gaussians abstain from the distribution.  A frame with
    no remaining mask evidence is background-concentrated, not spuriously
    diverse.
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

    for target_index in sorted(range(len(items)), key=lambda index: items[index].mask_id):
        target = items[target_index]
        target_ids = target.association_ids
        if target_ids.size == 0:
            continue
        observable_frames = 0
        diverse_frames = 0
        for frame_id in range(visible.shape[0]):
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
            observable_frames += 1

            # -1 is the explicit unassigned/background mask ID.  Ambiguous
            # points were removed above, so every remaining point contributes
            # exactly one distribution vote or abstains as background.
            labels = np.full(visible_target.size, -1, dtype=np.int64)
            evidence_found = False
            for local_label, candidate in enumerate(
                sorted(by_frame.get(frame_id, ()), key=lambda item: item.mask_id)
            ):
                overlap = np.intersect1d(
                    visible_target, candidate.association_ids, assume_unique=True
                )
                if overlap.size == 0:
                    continue
                evidence_found = True
                positions = np.searchsorted(visible_target, overlap)
                labels[positions] = local_label
            if not evidence_found:
                continue
            _, counts = np.unique(labels, return_counts=True)
            dominant_fraction = float(counts.max() / visible_target.size)
            if dominant_fraction < config.contained_threshold:
                diverse_frames += 1

        if (
            observable_frames > 0
            and diverse_frames / observable_frames
            > config.undersegment_filter_threshold
        ):
            rejected.append(target.mask_id)
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


def compute_pair_consensus(
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    observations: Sequence[MaskObservation],
    visibility: object,
    *,
    config: ConsensusConfig = ConsensusConfig(),
    active_indices: Sequence[int] | None = None,
) -> PairConsensus:
    """Compute observer/supporter view consensus between two components."""

    config.validate()
    items, visible_raw, _ = _validate_inputs(observations, visibility)
    visible = _visible_boolean(visible_raw, config.mask_visible_threshold)
    active = tuple(range(len(items))) if active_indices is None else tuple(active_indices)
    left = tuple(sorted(set(int(value) for value in left_indices)))
    right = tuple(sorted(set(int(value) for value in right_indices)))
    if not left or not right or set(left) & set(right):
        raise ValueError("pair components must be non-empty and disjoint")
    if any(index < 0 or index >= len(items) for index in left + right + active):
        raise ValueError("component index is outside observations")

    left_support = _component_support(left, items)
    right_support = _component_support(right, items)
    left_masks = tuple(sorted(items[index].mask_id for index in left))
    right_masks = tuple(sorted(items[index].mask_id for index in right))
    by_frame: dict[int, list[MaskObservation]] = {}
    for index in active:
        by_frame.setdefault(items[index].frame_id, []).append(items[index])
    return _pair_consensus_from_support(
        left_support,
        right_support,
        left_masks,
        right_masks,
        by_frame,
        visible,
        config,
    )


def _pair_consensus_from_support(
    left_support: np.ndarray,
    right_support: np.ndarray,
    left_masks: tuple[int, ...],
    right_masks: tuple[int, ...],
    by_frame: dict[int, list[MaskObservation]],
    visible: np.ndarray,
    config: ConsensusConfig,
) -> PairConsensus:
    if left_support.size == 0 or right_support.size == 0:
        return PairConsensus(left_masks, right_masks, 0, 0, 0.0)
    observers = 0
    supporters = 0
    ambiguous_by_frame: dict[int, np.ndarray] = {}
    for frame_id, frame_masks in by_frame.items():
        rows = [mask.ambiguous_ids for mask in frame_masks if mask.ambiguous_ids.size]
        ambiguous_by_frame[frame_id] = (
            np.unique(np.concatenate(rows))
            if rows
            else np.empty(0, dtype=np.int64)
        )
    for frame_id in range(visible.shape[0]):
        frame_masks = by_frame.get(frame_id, ())
        left_visible = left_support[visible[frame_id, left_support]]
        right_visible = right_support[visible[frame_id, right_support]]
        frame_ambiguous = ambiguous_by_frame.get(
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
    observations: Sequence[MaskObservation], active_indices: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Generate only pairs that could share at least one supporter mask.

    A supporter must overlap both sides, so it is sufficient to enumerate all
    observations that co-occur in the inverted Gaussian support list of one
    active mask.  The result is a conservative superset of qualifying pairs.
    """

    point_to_masks: dict[int, list[int]] = {}
    for index in active_indices:
        for gaussian_id in observations[index].association_ids:
            point_to_masks.setdefault(int(gaussian_id), []).append(int(index))
    pairs: set[tuple[int, int]] = set()
    for supporter_index in active_indices:
        neighbours: set[int] = set()
        for gaussian_id in observations[supporter_index].association_ids:
            neighbours.update(point_to_masks.get(int(gaussian_id), ()))
        for left, right in combinations(sorted(neighbours), 2):
            if observations[left].frame_id == observations[right].frame_id:
                continue
            pairs.add((left, right))
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
) -> ConsensusResult:
    """Build deterministic objects from complete lifted masks.

    The optional veto sees only stable mask IDs and the proposed full Gaussian
    union.  It cannot change raw observer/supporter evidence or AP scoring.
    """

    config.validate()
    items, visible_raw, xyz = _validate_inputs(observations, visibility, xyz_m)
    assert xyz is not None
    visible = _visible_boolean(visible_raw, config.mask_visible_threshold)
    rejected = detect_undersegmented_masks(items, visible_raw, config=config)
    rejected_set = set(rejected)
    active = tuple(
        index for index, item in enumerate(items) if item.mask_id not in rejected_set
    )
    raw_candidate_pairs = _sparse_raw_candidate_pairs(items, active)
    by_frame: dict[int, list[MaskObservation]] = {}
    for index in active:
        by_frame.setdefault(items[index].frame_id, []).append(items[index])
    ambiguity_by_frame_all: dict[int, np.ndarray] = {}
    for frame_id in range(visible.shape[0]):
        rows = [
            item.ambiguous_ids
            for item in items
            if item.frame_id == frame_id and item.ambiguous_ids.size
        ]
        ambiguity_by_frame_all[frame_id] = (
            np.unique(np.concatenate(rows))
            if rows
            else np.empty(0, dtype=np.int64)
        )
    # A rejected under-segmented mask is removed both as a graph node and as
    # evidence supplied by its source frame.  This mirrors MaskClustering's
    # removal of that frame from F/M sets instead of letting a rejected mask
    # inflate or depress another pair's consensus.
    rejected_frames = {
        item.frame_id for item in items if item.mask_id in rejected_set
    }
    if rejected_frames:
        visible = visible.copy()
        visible[np.asarray(sorted(rejected_frames), dtype=np.int64), :] = False
    uf = _DeterministicUnionFind(len(items))
    components: dict[int, set[int]] = {index: {index} for index in active}
    component_supports = {
        index: items[index].association_ids for index in active
    }
    component_full = {index: items[index].gaussian_ids for index in active}
    component_mask_ids = {index: (items[index].mask_id,) for index in active}
    component_frame_ids = {index: {items[index].frame_id} for index in active}
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
        )

    def rebuild_pair_cache() -> dict[tuple[int, int], PairConsensus]:
        current_pairs: set[tuple[int, int]] = set()
        for left_index, right_index in raw_candidate_pairs:
            left_root = uf.find(left_index)
            right_root = uf.find(right_index)
            if left_root == right_root:
                continue
            left_root, right_root = sorted((left_root, right_root))
            if left_root in components and right_root in components:
                current_pairs.add((left_root, right_root))
        return {
            pair: evaluate_pair(*pair)
            for pair in sorted(current_pairs)
        }

    pair_cache = rebuild_pair_cache()
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
            root = roots[0]
            for other in roots[1:]:
                root = uf.union(root, other)
            components[root] = merged_indices
            component_supports[root] = merged_support
            component_full[root] = full_ids
            component_mask_ids[root] = mask_ids
            component_frame_ids[root] = merged_frames

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
            pair_cache = rebuild_pair_cache()

    provisional: list[ConsensusObject] = []
    dropped_by_views = 0
    dropped_by_detection = 0
    dropped_by_connectivity = 0
    for component in sorted(
        components.values(),
        key=lambda members: tuple(sorted(items[index].mask_id for index in members)),
    ):
        frame_ids = tuple(sorted({items[index].frame_id for index in component}))
        if len(frame_ids) < config.min_views:
            dropped_by_views += 1
            continue
        full_ids = _component_full(component, items)
        if full_ids.size == 0:
            dropped_by_detection += 1
            continue
        eligible_visibility = visible[:, full_ids].copy()
        for frame_id, ambiguous_ids in ambiguity_by_frame_all.items():
            if ambiguous_ids.size == 0:
                continue
            ambiguous_positions = np.isin(
                full_ids, ambiguous_ids, assume_unique=True
            )
            eligible_visibility[frame_id, ambiguous_positions] = False
        detection_counts = np.zeros(full_ids.size, dtype=np.int64)
        member_indices_by_frame: dict[int, list[int]] = {}
        for index in component:
            member_indices_by_frame.setdefault(items[index].frame_id, []).append(index)
        for frame_id, frame_indices in member_indices_by_frame.items():
            detected_in_frame = np.zeros(full_ids.size, dtype=bool)
            for index in frame_indices:
                detected_in_frame |= np.isin(
                    full_ids, items[index].association_ids, assume_unique=True
                )
            # Detection frequency is conditioned on the same eligible frame
            # evidence as its denominator.  In particular, invisible and
            # same-frame ambiguous points can neither help nor hurt the ratio.
            detection_counts += (
                detected_in_frame & eligible_visibility[frame_id]
            ).astype(np.int64)
        visible_counts = eligible_visibility.sum(axis=0)
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
        mask_ids = tuple(sorted(items[index].mask_id for index in component))
        component_edges = [
            edge
            for edge in accepted_edges
            if set(edge.left_mask_ids + edge.right_mask_ids).issubset(mask_ids)
        ]
        mean_consensus = (
            float(np.mean([edge.consensus for edge in component_edges]))
            if component_edges
            else 0.0
        )
        for part in split_disconnected_support(
            filtered_ids,
            xyz,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        ):
            positions = np.searchsorted(filtered_ids, part)
            mean_detection = float(np.mean(filtered_ratios[positions]))
            geometric_quality = float(
                np.sqrt(max(0.0, mean_consensus) * max(0.0, mean_detection))
            )
            provisional.append(
                ConsensusObject(
                    object_id=-1,
                    mask_ids=mask_ids,
                    frame_ids=frame_ids,
                    gaussian_ids=part,
                    mean_view_consensus=mean_consensus,
                    mean_detection_ratio=mean_detection,
                    geometric_quality=geometric_quality,
                )
            )
        if not any(
            set(candidate.mask_ids) == set(mask_ids) for candidate in provisional
        ):
            dropped_by_connectivity += 1

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
        "undersegmented_mask_count": len(rejected),
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
