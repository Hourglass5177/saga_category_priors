from __future__ import annotations

"""Pure deterministic object construction for the V9 Clean ObjectBank.

The module contains no renderer, ground truth, filesystem, or legacy
post-processing dependency.  Geometry and affinity form objects first;
semantic evidence is attached only after tracks and unique cores are frozen.
"""

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


SAGA20 = frozenset(
    {
        "chair",
        "table",
        "plant",
        "tv",
        "painting",
        "sofa",
        "cabinet",
        "bed",
        "socket",
        "book",
        "switch",
        "door",
        "window",
        "lamp",
        "speaker",
        "fan",
        "refrigerator",
        "cup",
        "phone",
        "trash can",
    }
)

AssociationMode = Literal["A0", "A1", "A2", "A3"]
_NEIGHBOR_QUERY_CHUNK = 8192


@dataclass(frozen=True)
class V9Config:
    direct_min_shared_core: int = 3
    direct_min_overlap: float = 0.25
    sequential_min_margin: float = 0.10
    bridge_radius_m: float = 0.05
    bridge_min_affinity: float = 0.95
    bridge_max_conflict_ratio: float = 0.25
    bridge_min_pairs: int = 3
    graph_physical_neighbors: int = 24
    graph_affinity_neighbors: int = 4
    core_min_positive_views: int = 2
    core_min_positive_ratio: float = 0.60
    core_max_conflict_ratio: float = 0.25
    core_min_points: int = 3
    attach_radius_m: float = 0.05
    attach_min_anchors: int = 3
    attach_min_affinity: float = 0.95
    attach_min_margin: float = 0.02
    local_density_neighbors: int = 16
    boundary_radius_m: float = 0.05

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_sparse(
    ids: np.ndarray | Sequence[int],
    values: np.ndarray | Sequence[float],
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    point_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    weights = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(point_ids) != len(weights):
        raise ValueError(f"{name} ids and values must have the same length")
    if np.any(point_ids < 0):
        raise ValueError(f"{name} ids must be non-negative")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError(f"{name} values must be finite and non-negative")
    if not len(point_ids):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    unique, inverse = np.unique(point_ids, return_inverse=True)
    totals = np.zeros(len(unique), dtype=np.float64)
    np.add.at(totals, inverse, weights)
    return unique.astype(np.int32), totals.astype(np.float32)


def _max_union(
    left_ids: np.ndarray,
    left_values: np.ndarray,
    right_ids: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.union1d(left_ids, right_ids).astype(np.int32)
    values = np.zeros(len(ids), dtype=np.float32)
    if len(left_ids):
        pos = np.searchsorted(ids, left_ids)
        values[pos] = np.maximum(values[pos], left_values)
    if len(right_ids):
        pos = np.searchsorted(ids, right_ids)
        values[pos] = np.maximum(values[pos], right_values)
    return ids, values


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    frame_id: int
    mask_index: int
    full_ids: np.ndarray
    core_ids: np.ndarray
    full_mass: np.ndarray
    core_mass: np.ndarray
    # ``None`` means that the lifting artifact did not record same-view
    # conflict evidence.  A2 must reject such an artifact instead of silently
    # treating missing evidence as the safest possible value (zero).
    conflict_ratio: float | None = None

    def __post_init__(self) -> None:
        full_ids, full_mass = _canonical_sparse(
            self.full_ids, self.full_mass, name="fragment full"
        )
        core_ids, core_mass = _canonical_sparse(
            self.core_ids, self.core_mass, name="fragment core"
        )
        if not np.all(np.isin(core_ids, full_ids, assume_unique=True)):
            raise ValueError("fragment core_ids must be a subset of full_ids")
        if self.conflict_ratio is not None and (
            not np.isfinite(self.conflict_ratio)
            or not 0 <= self.conflict_ratio <= 1
        ):
            raise ValueError("fragment conflict_ratio must be in [0, 1] when present")
        for array in (full_ids, full_mass, core_ids, core_mass):
            array.setflags(write=False)
        object.__setattr__(self, "full_ids", full_ids)
        object.__setattr__(self, "full_mass", full_mass)
        object.__setattr__(self, "core_ids", core_ids)
        object.__setattr__(self, "core_mass", core_mass)


@dataclass(frozen=True)
class FrameEvidence:
    frame_id: int
    fragments: tuple[Fragment, ...]
    visible_ids: np.ndarray
    visible_mass: np.ndarray
    abstain: bool = False

    def __post_init__(self) -> None:
        ids, mass = _canonical_sparse(
            self.visible_ids, self.visible_mass, name="frame visible"
        )
        fragments = tuple(self.fragments)
        if any(int(fragment.frame_id) != int(self.frame_id) for fragment in fragments):
            raise ValueError("frame fragments must have the same frame_id")
        if any(not np.all(np.isin(fragment.full_ids, ids)) for fragment in fragments):
            raise ValueError("fragment support must be visible in its frame")
        ids.setflags(write=False)
        mass.setflags(write=False)
        object.__setattr__(self, "fragments", fragments)
        object.__setattr__(self, "visible_ids", ids)
        object.__setattr__(self, "visible_mass", mass)


@dataclass(frozen=True)
class AssociationEdge:
    left_fragment_id: int
    right_fragment_id: int
    kind: str
    score: float
    support: int


@dataclass(frozen=True)
class ObjectTrack:
    track_id: int
    fragment_ids: tuple[int, ...]
    frame_ids: tuple[int, ...]
    merge_scores: tuple[float, ...]
    association_mode: AssociationMode


@dataclass(frozen=True)
class AssociationResult:
    mode: AssociationMode
    tracks: tuple[ObjectTrack, ...]
    accepted_edges: tuple[AssociationEdge, ...]
    graph_edge_count: int = 0


@dataclass(frozen=True)
class SparseCounts:
    ids: np.ndarray
    values: np.ndarray
    point_count: int

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values, dtype=np.int32).reshape(-1)
        if len(ids) != len(values):
            raise ValueError("sparse ids and values must have equal length")
        if np.any(ids < 0) or np.any(ids >= int(self.point_count)):
            raise ValueError("sparse id is outside point_count")
        if len(ids) and np.any(np.diff(ids) <= 0):
            raise ValueError("sparse ids must be sorted and unique")
        ids = ids.astype(np.int32)
        ids.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "values", values)

    @classmethod
    def from_counter(cls, counts: Counter[int], point_count: int) -> "SparseCounts":
        ids = np.asarray(sorted(counts), dtype=np.int32)
        values = np.asarray([counts[int(point_id)] for point_id in ids], dtype=np.int32)
        return cls(ids, values, point_count)

    def take(self, point_ids: np.ndarray | Sequence[int]) -> np.ndarray:
        query = np.asarray(point_ids, dtype=np.int64)
        flat = query.reshape(-1)
        output = np.zeros(len(flat), dtype=np.int32)
        if len(self.ids) and len(flat):
            positions = np.searchsorted(self.ids, flat)
            valid = positions < len(self.ids)
            matched = np.zeros(len(flat), dtype=bool)
            matched[valid] = self.ids[positions[valid]] == flat[valid]
            output[matched] = self.values[positions[matched]]
        return output.reshape(query.shape)


@dataclass(frozen=True)
class ConsensusResult:
    core_track_id: np.ndarray
    visible_views: np.ndarray
    assignment_margin: np.ndarray
    valid_track_ids: tuple[int, ...]
    positive_views: Mapping[int, SparseCounts]
    conflict_views: Mapping[int, SparseCounts]
    core_ids: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class MultiviewClassVote:
    frame_id: int
    class_id: int
    weight: float


@dataclass(frozen=True)
class TrackClassification:
    track_id: int
    class_id: int
    class_name: str
    semantic_ratio: float
    semantic_margin: float
    effective_view_count: int
    source: str
    eligible: bool


@dataclass(frozen=True)
class CandidateBank:
    point_count: int
    association_mode: AssociationMode
    core_candidate_id: np.ndarray
    full_ids: tuple[np.ndarray, ...]
    core_ids: tuple[np.ndarray, ...]
    candidates: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        labels = np.asarray(self.core_candidate_id, dtype=np.int32).copy()
        rows = tuple(dict(row) for row in self.candidates)
        full = tuple(np.unique(np.asarray(ids, dtype=np.int32)) for ids in self.full_ids)
        core = tuple(np.unique(np.asarray(ids, dtype=np.int32)) for ids in self.core_ids)
        if self.point_count <= 0 or labels.shape != (int(self.point_count),):
            raise ValueError("bank labels must match a positive point_count")
        if len(rows) != len(full) or len(rows) != len(core):
            raise ValueError("bank rows and ragged masks must have equal length")
        expected = set(range(len(rows)))
        actual = {int(row.get("candidate_id", -1)) for row in rows}
        if actual != expected:
            raise ValueError("candidate_id values must be contiguous from zero")
        if np.any(labels < -1) or np.any(labels >= len(rows)):
            raise ValueError("core_candidate_id contains an invalid candidate id")
        for candidate_id, (full_ids, core_ids) in enumerate(zip(full, core)):
            if np.any(full_ids < 0) or np.any(full_ids >= self.point_count):
                raise ValueError("bank full mask contains an out-of-range id")
            if not np.all(np.isin(core_ids, full_ids, assume_unique=True)):
                raise ValueError("bank core mask must be a subset of full mask")
            if not np.array_equal(np.flatnonzero(labels == candidate_id), core_ids):
                raise ValueError("bank core labels and ragged core mask disagree")
            full_ids.setflags(write=False)
            core_ids.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "core_candidate_id", labels)
        object.__setattr__(self, "full_ids", full)
        object.__setattr__(self, "core_ids", core)
        object.__setattr__(self, "candidates", rows)


def _normalise_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 1e-12)


def _weighted_core_overlap_arrays(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> tuple[int, float]:
    shared, li, ri = np.intersect1d(
        left_ids,
        right_ids,
        assume_unique=True,
        return_indices=True,
    )
    denominator = min(float(left_mass.sum()), float(right_mass.sum()))
    if denominator <= 0:
        return int(len(shared)), 0.0
    numerator = float(np.minimum(left_mass[li], right_mass[ri]).sum())
    return int(len(shared)), float(np.clip(numerator / denominator, 0.0, 1.0))


def weighted_core_overlap(left: Fragment, right: Fragment) -> tuple[int, float]:
    return _weighted_core_overlap_arrays(
        left.core_ids,
        left.core_mass,
        right.core_ids,
        right.core_mass,
    )


def _shared_core_candidate_pairs(
    fragments: Sequence[Fragment], minimum_shared: int
) -> np.ndarray:
    """Return exact fragment-position pairs sharing enough unique core IDs.

    Direct association used to call ``intersect1d`` for every cross-frame
    fragment pair.  The two Stage-2 scenes contain more than 57 million such
    pairs, although only pairs co-occurring in at least three point postings
    can pass the registered gate.  This inverted index materializes one int64
    code per actual posting co-occurrence, sorts it, and retains the exact
    qualifying pair set.  It changes neither overlap weights nor edge order.
    """

    count = len(fragments)
    if count < 2:
        return np.empty((0, 2), dtype=np.int32)
    lengths = np.fromiter(
        (len(fragment.core_ids) for fragment in fragments),
        dtype=np.int64,
        count=count,
    )
    membership_count = int(lengths.sum())
    if not membership_count:
        return np.empty((0, 2), dtype=np.int32)
    point_ids = np.concatenate(
        [np.asarray(fragment.core_ids, dtype=np.int32) for fragment in fragments]
    )
    fragment_positions = np.repeat(np.arange(count, dtype=np.int32), lengths)
    order = np.argsort(point_ids, kind="stable")
    point_ids = point_ids[order]
    fragment_positions = fragment_positions[order]
    starts = np.flatnonzero(
        np.r_[True, point_ids[1:] != point_ids[:-1]]
    )
    stops = np.r_[starts[1:], len(point_ids)]
    degrees = stops - starts
    pair_occurrences = int(np.sum(degrees * (degrees - 1) // 2, dtype=np.int64))
    if not pair_occurrences:
        return np.empty((0, 2), dtype=np.int32)
    codes = np.empty(pair_occurrences, dtype=np.int64)
    offset = 0
    for start, stop in zip(starts, stops):
        postings = fragment_positions[start:stop]
        degree = len(postings)
        if degree < 2:
            continue
        left, right = np.triu_indices(degree, 1)
        size = len(left)
        codes[offset : offset + size] = (
            postings[left].astype(np.int64) * count + postings[right]
        )
        offset += size
    if offset != pair_occurrences:
        raise RuntimeError("fragment posting pair count is inconsistent")
    codes.sort()
    run_starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    run_stops = np.r_[run_starts[1:], len(codes)]
    shared = run_stops - run_starts
    qualified = codes[run_starts[shared >= int(minimum_shared)]]
    if not len(qualified):
        return np.empty((0, 2), dtype=np.int32)
    return np.column_stack((qualified // count, qualified % count)).astype(
        np.int32
    )


@dataclass
class _Component:
    fragment_ids: set[int]
    frame_ids: set[int]
    merge_scores: list[float] = field(default_factory=list)
    core_ids: np.ndarray | None = field(default=None, repr=False)
    core_mass: np.ndarray | None = field(default=None, repr=False)


def _component_core(
    component: _Component, fragment_by_id: Mapping[int, Fragment]
) -> tuple[np.ndarray, np.ndarray]:
    if component.core_ids is not None and component.core_mass is not None:
        return component.core_ids, component.core_mass
    ids = np.empty(0, dtype=np.int32)
    mass = np.empty(0, dtype=np.float32)
    for fragment_id in sorted(component.fragment_ids):
        fragment = fragment_by_id[fragment_id]
        ids, mass = _max_union(ids, mass, fragment.core_ids, fragment.core_mass)
    component.core_ids = ids
    component.core_mass = mass
    return ids, mass


def _merge_component_core(
    target: _Component,
    source: _Component,
    fragment_by_id: Mapping[int, Fragment],
) -> None:
    target_ids, target_mass = _component_core(target, fragment_by_id)
    source_ids, source_mass = _component_core(source, fragment_by_id)
    target.core_ids, target.core_mass = _max_union(
        target_ids, target_mass, source_ids, source_mass
    )


def _tracks_from_components(
    components: Sequence[_Component], mode: AssociationMode
) -> tuple[ObjectTrack, ...]:
    ordered = sorted(components, key=lambda component: min(component.fragment_ids))
    return tuple(
        ObjectTrack(
            track_id=index,
            fragment_ids=tuple(sorted(component.fragment_ids)),
            frame_ids=tuple(sorted(component.frame_ids)),
            merge_scores=tuple(component.merge_scores),
            association_mode=mode,
        )
        for index, component in enumerate(ordered)
    )


def _direct_edges(
    fragments: Sequence[Fragment], config: V9Config
) -> list[AssociationEdge]:
    output: list[AssociationEdge] = []
    ordered = sorted(fragments, key=lambda fragment: int(fragment.fragment_id))
    candidates = _shared_core_candidate_pairs(
        ordered, int(config.direct_min_shared_core)
    )
    for left_position, right_position in candidates:
        left = ordered[int(left_position)]
        right = ordered[int(right_position)]
        if int(left.frame_id) == int(right.frame_id):
            continue
        shared, overlap = weighted_core_overlap(left, right)
        if overlap >= float(config.direct_min_overlap):
            output.append(
                AssociationEdge(
                    int(left.fragment_id),
                    int(right.fragment_id),
                    "direct",
                    overlap,
                    shared,
                )
            )
    return output


def _constrained_components(
    fragments: Sequence[Fragment], edges: Sequence[AssociationEdge]
) -> tuple[list[_Component], list[AssociationEdge]]:
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    components = {
        fragment_id: _Component({fragment_id}, {int(fragment.frame_id)})
        for fragment_id, fragment in fragment_by_id.items()
    }
    owner = {fragment_id: fragment_id for fragment_id in fragment_by_id}
    accepted: list[AssociationEdge] = []
    kind_order = {"direct": 0, "graph": 1, "affinity": 2}
    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            kind_order.get(edge.kind, 9),
            -float(edge.score),
            -int(edge.support),
            int(edge.left_fragment_id),
            int(edge.right_fragment_id),
        ),
    )
    for edge in ordered_edges:
        left_root = owner[int(edge.left_fragment_id)]
        right_root = owner[int(edge.right_fragment_id)]
        if left_root == right_root:
            continue
        left = components[left_root]
        right = components[right_root]
        if left.frame_ids.intersection(right.frame_ids):
            continue
        keep, remove = sorted((left_root, right_root))
        target = components[keep]
        source = components[remove]
        _merge_component_core(target, source, fragment_by_id)
        target.fragment_ids.update(source.fragment_ids)
        target.frame_ids.update(source.frame_ids)
        target.merge_scores.extend(source.merge_scores)
        target.merge_scores.append(float(edge.score))
        for fragment_id in source.fragment_ids:
            owner[fragment_id] = keep
        del components[remove]
        accepted.append(edge)
    return list(components.values()), accepted


def _associate_a0(
    fragments: Sequence[Fragment], config: V9Config
) -> AssociationResult:
    fragment_ids = [int(fragment.fragment_id) for fragment in fragments]
    if len(fragment_ids) != len(set(fragment_ids)):
        raise ValueError("fragment_id values must be unique")
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    components: list[_Component] = []
    accepted: list[AssociationEdge] = []
    for fragment in sorted(
        fragments,
        key=lambda item: (int(item.frame_id), int(item.mask_index), int(item.fragment_id)),
    ):
        scored: list[tuple[float, int, int]] = []
        for component_index, component in enumerate(components):
            if int(fragment.frame_id) in component.frame_ids:
                continue
            core_ids, core_mass = _component_core(component, fragment_by_id)
            shared, overlap = _weighted_core_overlap_arrays(
                fragment.core_ids,
                fragment.core_mass,
                core_ids,
                core_mass,
            )
            if (
                shared >= int(config.direct_min_shared_core)
                and overlap >= float(config.direct_min_overlap)
            ):
                scored.append((overlap, shared, component_index))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected: int | None = None
        if scored:
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if scored[0][0] - runner_up >= float(config.sequential_min_margin):
                selected = scored[0][2]
        if selected is None:
            components.append(_Component({int(fragment.fragment_id)}, {int(fragment.frame_id)}))
            continue
        component = components[selected]
        target_fragment = min(component.fragment_ids)
        edge = AssociationEdge(
            min(target_fragment, int(fragment.fragment_id)),
            max(target_fragment, int(fragment.fragment_id)),
            "direct",
            float(scored[0][0]),
            int(scored[0][1]),
        )
        core_ids, core_mass = _component_core(component, fragment_by_id)
        component.core_ids, component.core_mass = _max_union(
            core_ids, core_mass, fragment.core_ids, fragment.core_mass
        )
        component.fragment_ids.add(int(fragment.fragment_id))
        component.frame_ids.add(int(fragment.frame_id))
        component.merge_scores.append(float(scored[0][0]))
        accepted.append(edge)
    return AssociationResult("A0", _tracks_from_components(components, "A0"), tuple(accepted))


def _mutual_spatial_affinity(
    left_ids: np.ndarray,
    right_ids: np.ndarray,
    xyz: np.ndarray,
    features: np.ndarray,
    radius_m: float,
) -> tuple[int, float]:
    if not len(left_ids) or not len(right_ids):
        return 0, 0.0

    def best(source: np.ndarray, target: np.ndarray) -> dict[int, tuple[int, float]]:
        tree = cKDTree(xyz[target])
        neighborhoods = tree.query_ball_point(xyz[source], r=float(radius_m))
        output: dict[int, tuple[int, float]] = {}
        for source_id, positions in zip(source, neighborhoods):
            if not positions:
                continue
            target_ids = target[np.asarray(positions, dtype=np.int64)]
            similarities = features[target_ids] @ features[int(source_id)]
            order = np.lexsort((target_ids, -similarities))
            target_id = int(target_ids[order[0]])
            output[int(source_id)] = (target_id, float(similarities[order[0]]))
        return output

    left_best = best(left_ids, right_ids)
    right_best = best(right_ids, left_ids)
    similarities = [
        similarity
        for left_id, (right_id, similarity) in left_best.items()
        if right_best.get(right_id, (-1, -1.0))[0] == left_id
    ]
    if not similarities:
        return 0, 0.0
    return len(similarities), float(np.median(similarities))


@dataclass(frozen=True)
class AffinityGraph:
    component_id: np.ndarray
    edges: np.ndarray

    def __post_init__(self) -> None:
        component_id = np.asarray(self.component_id, dtype=np.int32).reshape(-1)
        edges = np.asarray(self.edges, dtype=np.int32)
        if edges.shape == (0,):
            edges = edges.reshape(0, 2)
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("affinity graph edges must have shape E x 2")
        component_id.setflags(write=False)
        edges.setflags(write=False)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "edges", edges)


def build_mutual_affinity_graph(
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    config: V9Config = V9Config(),
) -> AffinityGraph:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity, name="affinity")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(features):
        raise ValueError("xyz_m and affinity must have matching point rows")
    point_count = len(xyz)
    if not point_count:
        return AffinityGraph(
            np.empty(0, dtype=np.int32), np.empty((0, 2), dtype=np.int32)
        )
    physical_k = min(int(config.graph_physical_neighbors), point_count - 1)
    if physical_k <= 0:
        return AffinityGraph(
            np.zeros(point_count, dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
        )
    distances, neighbors = cKDTree(xyz).query(xyz, k=physical_k + 1)
    del distances
    neighbors = np.asarray(neighbors, dtype=np.int32)
    affinity_k = min(int(config.graph_affinity_neighbors), physical_k)
    selected = np.empty((point_count, affinity_k), dtype=np.int32)
    # Chunking avoids materializing N x K x feature_dim for million-point scenes.
    chunk_size = 8192
    for start in range(0, point_count, chunk_size):
        stop = min(start + chunk_size, point_count)
        candidates = neighbors[start:stop]
        similarities = np.einsum(
            "nd,nkd->nk",
            features[start:stop],
            features[candidates],
            optimize=True,
        )
        row_ids = np.arange(start, stop, dtype=np.int64)[:, None]
        similarities = similarities.copy()
        similarities[candidates == row_ids] = -np.inf
        # cKDTree neighbor order is deterministic; stable sort therefore gives
        # a deterministic tie break without a Python loop.
        order = np.argsort(-similarities, axis=1, kind="stable")[:, :affinity_k]
        selected[start:stop] = np.take_along_axis(candidates, order, axis=1)

    sources = np.repeat(np.arange(point_count, dtype=np.int64), affinity_k)
    targets = selected.reshape(-1).astype(np.int64, copy=False)
    codes = sources * point_count + targets
    reverse = targets * point_count + sources
    sorted_codes = np.sort(codes)
    positions = np.searchsorted(sorted_codes, reverse)
    mutual = positions < len(sorted_codes)
    mutual[mutual] &= sorted_codes[positions[mutual]] == reverse[mutual]
    unique_direction = mutual & (sources < targets)
    edges = np.column_stack((sources[unique_direction], targets[unique_direction])).astype(
        np.int32
    )
    if len(edges):
        adjacency = coo_matrix(
            (np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
            shape=(point_count, point_count),
        )
        _, labels = connected_components(adjacency, directed=False)
        labels = labels.astype(np.int32)
    else:
        labels = np.arange(point_count, dtype=np.int32)
    return AffinityGraph(labels, edges)


def _graph_fragment_edge(
    left: Fragment,
    right: Fragment,
    component_id: np.ndarray,
    config: V9Config,
) -> AssociationEdge | None:
    left_components, left_counts = np.unique(component_id[left.core_ids], return_counts=True)
    right_components, right_counts = np.unique(component_id[right.core_ids], return_counts=True)
    shared, li, ri = np.intersect1d(
        left_components,
        right_components,
        assume_unique=True,
        return_indices=True,
    )
    support = int(np.minimum(left_counts[li], right_counts[ri]).sum()) if len(shared) else 0
    overlap = support / max(min(len(left.core_ids), len(right.core_ids)), 1)
    if (
        support < int(config.direct_min_shared_core)
        or overlap < float(config.direct_min_overlap)
    ):
        return None
    return AssociationEdge(
        min(int(left.fragment_id), int(right.fragment_id)),
        max(int(left.fragment_id), int(right.fragment_id)),
        "graph",
        float(overlap),
        support,
    )


def associate_fragments(
    fragments: Sequence[Fragment],
    mode: AssociationMode,
    *,
    xyz_m: np.ndarray | None = None,
    affinity: np.ndarray | None = None,
    config: V9Config = V9Config(),
) -> AssociationResult:
    """Associate fragments using one registered V9 structure.

    ``A0`` is the V8 sequential negative control.  ``A1`` globally merges
    direct overlap edges under same-frame cannot-link constraints.  ``A2``
    adds one mutual-best affinity attachment from singleton components to
    already-established components.  ``A3`` adds a physical mutual-top-k
    affinity graph before applying the same global constraints.
    """
    fragments = tuple(fragments)
    fragment_ids = [int(fragment.fragment_id) for fragment in fragments]
    if len(fragment_ids) != len(set(fragment_ids)):
        raise ValueError("fragment_id values must be unique")
    if mode == "A0":
        return _associate_a0(fragments, config)
    if mode not in {"A1", "A2", "A3"}:
        raise ValueError(f"unknown V9 association mode: {mode}")
    direct = _direct_edges(fragments, config)
    graph_edge_count = 0
    edges: list[AssociationEdge] = list(direct)
    if mode == "A3":
        if xyz_m is None or affinity is None:
            raise ValueError("A3 requires xyz_m and affinity")
        graph = build_mutual_affinity_graph(xyz_m, affinity, config)
        graph_edge_count = len(graph.edges)
        for index, left in enumerate(fragments):
            for right in fragments[index + 1 :]:
                if int(left.frame_id) == int(right.frame_id):
                    continue
                edge = _graph_fragment_edge(left, right, graph.component_id, config)
                if edge is not None:
                    edges.append(edge)
    components, accepted = _constrained_components(fragments, edges)
    if mode != "A2":
        return AssociationResult(
            mode,
            _tracks_from_components(components, mode),
            tuple(accepted),
            graph_edge_count,
        )

    if xyz_m is None or affinity is None:
        raise ValueError("A2 requires xyz_m and affinity")
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity, name="affinity")
    if xyz.shape != (len(features), 3):
        raise ValueError("xyz_m and affinity must have matching point rows")
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    singletons = [component for component in components if len(component.fragment_ids) == 1]
    established = [component for component in components if len(component.fragment_ids) >= 2]
    proposals: list[tuple[float, int, int, _Component, _Component]] = []
    for source in singletons:
        source_fragment = fragment_by_id[next(iter(source.fragment_ids))]
        if source_fragment.conflict_ratio is None:
            raise ValueError("A2 requires per-view fragment conflict evidence")
        if source_fragment.conflict_ratio > float(config.bridge_max_conflict_ratio):
            continue
        for target in established:
            if source.frame_ids.intersection(target.frame_ids):
                continue
            target_fragments = [fragment_by_id[item] for item in target.fragment_ids]
            if any(fragment.conflict_ratio is None for fragment in target_fragments):
                raise ValueError("A2 requires per-view fragment conflict evidence")
            if max(float(fragment.conflict_ratio) for fragment in target_fragments) > float(
                config.bridge_max_conflict_ratio
            ):
                continue
            target_ids, _ = _component_core(target, fragment_by_id)
            support, score = _mutual_spatial_affinity(
                source_fragment.core_ids,
                target_ids,
                xyz,
                features,
                float(config.bridge_radius_m),
            )
            if (
                support >= int(config.bridge_min_pairs)
                and score >= float(config.bridge_min_affinity)
            ):
                proposals.append((score, support, min(target.fragment_ids), source, target))
    source_best: dict[int, tuple[float, int, int, _Component, _Component]] = {}
    target_best: dict[int, tuple[float, int, int, _Component, _Component]] = {}
    for proposal in proposals:
        source_id = min(proposal[3].fragment_ids)
        target_id = min(proposal[4].fragment_ids)
        key = (-proposal[0], -proposal[1], proposal[2])
        previous = source_best.get(source_id)
        if previous is None or key < (-previous[0], -previous[1], previous[2]):
            source_best[source_id] = proposal
        previous = target_best.get(target_id)
        target_key = (-proposal[0], -proposal[1], source_id)
        if previous is None or target_key < (
            -previous[0],
            -previous[1],
            min(previous[3].fragment_ids),
        ):
            target_best[target_id] = proposal
    attached_sources: set[int] = set()
    for source_id in sorted(source_best):
        proposal = source_best[source_id]
        target_id = min(proposal[4].fragment_ids)
        if target_best.get(target_id) is not proposal:
            continue
        source, target = proposal[3], proposal[4]
        _merge_component_core(target, source, fragment_by_id)
        target.fragment_ids.update(source.fragment_ids)
        target.frame_ids.update(source.frame_ids)
        target.merge_scores.extend(source.merge_scores)
        target.merge_scores.append(float(proposal[0]))
        attached_sources.add(source_id)
        accepted.append(
            AssociationEdge(
                min(source_id, target_id),
                max(source_id, target_id),
                "affinity",
                float(proposal[0]),
                int(proposal[1]),
            )
        )
    components = [
        component
        for component in components
        if not (
            len(component.fragment_ids) == 1
            and min(component.fragment_ids) in attached_sources
        )
    ]
    return AssociationResult("A2", _tracks_from_components(components, "A2"), tuple(accepted))


def build_consensus_core(
    association: AssociationResult,
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    point_count: int,
    config: V9Config = V9Config(),
) -> ConsensusResult:
    """Build a unique core while counting each physical view at most once."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    frame_by_id = {int(frame.frame_id): frame for frame in frames}
    if len(fragment_by_id) != len(fragments) or len(frame_by_id) != len(frames):
        raise ValueError("fragment and frame IDs must be unique")
    if any(np.any(frame.visible_ids >= point_count) for frame in frames):
        raise ValueError("frame evidence exceeds point_count")
    track_by_fragment = {
        fragment_id: int(track.track_id)
        for track in association.tracks
        for fragment_id in track.fragment_ids
    }
    if set(track_by_fragment) != set(fragment_by_id):
        raise ValueError("association must contain every fragment exactly once")

    visible_views = np.zeros(point_count, dtype=np.int32)
    positive_counters = {int(track.track_id): Counter() for track in association.tracks}
    # A conflict is queried only for points that have positive support for the
    # same track.  The former implementation nevertheless copied every
    # foreground point in every frame into every other track's Counter.  With
    # thousands of lifted fragments that is O(tracks x visible foreground)
    # Python objects and can exceed the 90 GiB cgroup before the first bank is
    # written.  Count the equivalent sufficient statistics instead:
    #
    #   conflict(t, p) = frames_with_any_track(p)
    #                    - frames_where_only_track_t_contains(p)
    #
    # Per-frame ``by_track`` rows are already unique, so a point that occurs
    # once is exclusive to that track and a point that occurs more than once
    # has an other-track conflict for every owning track.  Tracks absent from
    # the frame are handled by the global presence count.  This preserves the
    # registered per-physical-view semantics while bounding stored evidence by
    # the positive support instead of by all track/point combinations.
    foreground_views = np.zeros(point_count, dtype=np.int32)
    exclusive_counters = {
        int(track.track_id): Counter() for track in association.tracks
    }
    for frame in frames:
        if frame.abstain:
            continue
        visible_views[frame.visible_ids] += 1
        by_track: dict[int, np.ndarray] = {}
        for fragment in frame.fragments:
            track_id = track_by_fragment[int(fragment.fragment_id)]
            previous = by_track.get(track_id, np.empty(0, dtype=np.int32))
            by_track[track_id] = np.union1d(previous, fragment.core_ids).astype(np.int32)
        for track_id, own_ids in by_track.items():
            positive_counters[track_id].update(map(int, own_ids))
        if by_track:
            track_chunks = [
                np.full(len(ids), track_id, dtype=np.int32)
                for track_id, ids in by_track.items()
            ]
            point_chunks = list(by_track.values())
            all_points = np.concatenate(point_chunks)
            all_tracks = np.concatenate(track_chunks)
            unique_points, first, owner_count = np.unique(
                all_points, return_index=True, return_counts=True
            )
            foreground_views[unique_points] += 1
            exclusive = owner_count == 1
            if np.any(exclusive):
                exclusive_points = unique_points[exclusive]
                exclusive_tracks = all_tracks[first[exclusive]]
                for track_id in np.unique(exclusive_tracks):
                    ids = exclusive_points[exclusive_tracks == track_id]
                    exclusive_counters[int(track_id)].update(map(int, ids))

    positive = {
        track_id: SparseCounts.from_counter(counter, point_count)
        for track_id, counter in positive_counters.items()
    }
    conflict: dict[int, SparseCounts] = {}
    for track_id, support in positive.items():
        ids = support.ids
        exclusive = SparseCounts.from_counter(
            exclusive_counters[track_id], point_count
        ).take(ids)
        values = foreground_views[ids] - exclusive
        nonzero = values > 0
        conflict[track_id] = SparseCounts(
            ids[nonzero], values[nonzero], point_count
        )
    eligible: dict[int, np.ndarray] = {}
    for track in association.tracks:
        track_id = int(track.track_id)
        if len(track.frame_ids) < int(config.core_min_positive_views):
            continue
        ids = positive[track_id].ids
        counts = positive[track_id].values
        visible = np.maximum(visible_views[ids], 1)
        conflict_count = conflict[track_id].take(ids)
        keep = (
            (counts >= int(config.core_min_positive_views))
            & (counts / visible >= float(config.core_min_positive_ratio))
            & (conflict_count / visible <= float(config.core_max_conflict_ratio))
        )
        eligible[track_id] = ids[keep]

    active = {track_id for track_id, ids in eligible.items() if len(ids) >= config.core_min_points}
    labels = np.full(point_count, -1, dtype=np.int32)
    margins = np.zeros(point_count, dtype=np.float32)
    while active:
        labels.fill(-1)
        margins.fill(0)
        choices: dict[int, list[tuple[float, int, int]]] = {}
        for track_id in sorted(active):
            ids = eligible[track_id]
            counts = positive[track_id].take(ids)
            ratios = counts / np.maximum(visible_views[ids], 1)
            for point_id, ratio, count in zip(ids, ratios, counts):
                choices.setdefault(int(point_id), []).append(
                    (float(ratio), int(count), track_id)
                )
        for point_id, point_choices in choices.items():
            point_choices.sort(key=lambda item: (-item[0], -item[1], item[2]))
            labels[point_id] = point_choices[0][2]
            runner_up = point_choices[1][0] if len(point_choices) > 1 else 0.0
            margins[point_id] = float(point_choices[0][0] - runner_up)
        surviving = {
            track_id
            for track_id in active
            if np.count_nonzero(labels == track_id) >= int(config.core_min_points)
        }
        if surviving == active:
            break
        active = surviving
    if not active:
        labels.fill(-1)
        margins.fill(0)
    labels.setflags(write=False)
    visible_views.setflags(write=False)
    margins.setflags(write=False)
    core_ids = {
        track_id: np.flatnonzero(labels == track_id).astype(np.int32)
        for track_id in sorted(active)
    }
    for ids in core_ids.values():
        ids.setflags(write=False)
    return ConsensusResult(
        labels,
        visible_views,
        margins,
        tuple(sorted(active)),
        positive,
        conflict,
        core_ids,
    )


def attach_local_halo(
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    consensus: ConsensusResult,
    config: V9Config = V9Config(),
) -> np.ndarray:
    """Attach unassigned Gaussians once to a unique nearby affinity core."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity, name="affinity")
    labels = np.asarray(consensus.core_track_id, dtype=np.int32)
    if xyz.shape != (len(labels), 3) or len(features) != len(labels):
        raise ValueError("xyz, affinity, and consensus must have matching rows")
    output = labels.copy()
    anchors = np.flatnonzero(labels >= 0)
    queries = np.flatnonzero(labels < 0)
    if len(anchors) < int(config.attach_min_anchors) or not len(queries):
        return output
    # Query every anchor inside the registered physical radius.  A fixed-k
    # truncation made the result depend on unrelated nearby tracks and could
    # hide the third anchor of the correct object in dense regions.
    tree = cKDTree(xyz[anchors])
    for start in range(0, len(queries), _NEIGHBOR_QUERY_CHUNK):
        query_ids = queries[start : start + _NEIGHBOR_QUERY_CHUNK]
        neighborhoods = tree.query_ball_point(
            xyz[query_ids], r=float(config.attach_radius_m)
        )
        for point_id, positions in zip(query_ids, neighborhoods):
            neighbor_ids = anchors[np.asarray(positions, dtype=np.int64)]
            choices: list[tuple[float, int]] = []
            for track_id in np.unique(labels[neighbor_ids]):
                same = neighbor_ids[labels[neighbor_ids] == track_id]
                if len(same) < int(config.attach_min_anchors):
                    continue
                prototype = features[same].mean(axis=0)
                norm = float(np.linalg.norm(prototype))
                similarity = float(
                    features[point_id] @ (prototype / max(norm, 1e-12))
                )
                if similarity >= float(config.attach_min_affinity):
                    choices.append((similarity, int(track_id)))
            choices.sort(key=lambda item: (-item[0], item[1]))
            if choices:
                runner_up = choices[1][0] if len(choices) > 1 else -1.0
                if choices[0][0] - runner_up >= float(config.attach_min_margin):
                    output[point_id] = choices[0][1]
    return output


def classify_tracks_multiview(
    association: AssociationResult,
    votes_by_track: Mapping[int, Sequence[MultiviewClassVote]],
    class_names: Sequence[str],
) -> dict[int, TrackClassification]:
    """Attach semantics after tracking using equally weighted physical views."""
    output: dict[int, TrackClassification] = {}
    for track in association.tracks:
        per_frame: dict[int, np.ndarray] = {}
        for vote in votes_by_track.get(int(track.track_id), ()):
            if not 0 <= int(vote.class_id) < len(class_names):
                continue
            if not np.isfinite(vote.weight) or vote.weight <= 0:
                continue
            values = per_frame.setdefault(
                int(vote.frame_id), np.zeros(len(class_names), dtype=np.float64)
            )
            values[int(vote.class_id)] += float(vote.weight)
        distribution = np.zeros(len(class_names), dtype=np.float64)
        effective_views = 0
        for values in per_frame.values():
            total = float(values.sum())
            if total > 0:
                distribution += values / total
                effective_views += 1
        total = float(distribution.sum())
        if total <= 0:
            output[int(track.track_id)] = TrackClassification(
                int(track.track_id), -1, "", 0.0, 0.0, 0, "mv-label", False
            )
            continue
        order = np.argsort(-distribution, kind="stable")
        winner = int(order[0])
        runner_up = float(distribution[order[1]]) if len(order) > 1 else 0.0
        ratio = float(distribution[winner] / total)
        margin = float((distribution[winner] - runner_up) / total)
        class_name = str(class_names[winner])
        output[int(track.track_id)] = TrackClassification(
            int(track.track_id),
            winner,
            class_name,
            ratio,
            margin,
            effective_views,
            "mv-label",
            class_name in SAGA20,
        )
    return output


def classify_tracks_codebook(
    association: AssociationResult,
    consensus: ConsensusResult,
    semantic_features: np.ndarray,
    label_embeddings: np.ndarray,
    class_names: Sequence[str],
) -> dict[int, TrackClassification]:
    """Attach semantics after tracking using the complete normalized codebook."""
    features = _normalise_rows(semantic_features, name="semantic_features")
    embeddings = _normalise_rows(label_embeddings, name="label_embeddings")
    if len(features) != len(consensus.core_track_id):
        raise ValueError("semantic_features must have one row per Gaussian")
    if embeddings.shape != (len(class_names), features.shape[1]):
        raise ValueError("label embeddings and class names must match feature dimensions")
    output: dict[int, TrackClassification] = {}
    track_by_id = {int(track.track_id): track for track in association.tracks}
    for track_id in consensus.valid_track_ids:
        prototype = features[consensus.core_ids[track_id]].mean(axis=0)
        norm = float(np.linalg.norm(prototype))
        if norm <= 1e-12:
            output[track_id] = TrackClassification(
                track_id, -1, "", 0.0, 0.0, 0, "codebook", False
            )
            continue
        similarities = embeddings @ (prototype / norm)
        order = np.argsort(-similarities, kind="stable")
        winner = int(order[0])
        runner_up = float(similarities[order[1]]) if len(order) > 1 else -1.0
        top = float(similarities[winner])
        class_name = str(class_names[winner])
        output[track_id] = TrackClassification(
            track_id,
            winner,
            class_name,
            float(np.clip((top + 1.0) * 0.5, 0.0, 1.0)),
            float(np.clip((top - runner_up) * 0.5, 0.0, 1.0)),
            len(track_by_id[track_id].frame_ids),
            "codebook",
            class_name in SAGA20,
        )
    return output


def _local_density(xyz: np.ndarray, core_ids: np.ndarray, neighbors: int) -> float:
    k = min(int(neighbors), max(len(core_ids) - 1, 0))
    if k <= 0:
        return 0.0
    distances, _ = cKDTree(xyz[core_ids]).query(xyz[core_ids], k=k + 1)
    radii = np.maximum(np.asarray(distances)[:, -1], 1e-6)
    return float(np.median(k / (np.pi * radii * radii)))


def _boundary_ratio(
    xyz: np.ndarray, member_ids: np.ndarray, tree: cKDTree, radius_m: float
) -> float:
    if not len(member_ids):
        return 0.0
    inside = np.zeros(len(xyz), dtype=bool)
    inside[member_ids] = True
    boundary_edges = 0
    total_edges = 0
    for start in range(0, len(member_ids), _NEIGHBOR_QUERY_CHUNK):
        query_ids = member_ids[start : start + _NEIGHBOR_QUERY_CHUNK]
        neighborhoods = tree.query_ball_point(
            xyz[query_ids], r=float(radius_m)
        )
        for point_id, neighbors in zip(query_ids, neighborhoods):
            neighbor_ids = np.asarray(neighbors, dtype=np.int64)
            neighbor_ids = neighbor_ids[neighbor_ids != int(point_id)]
            total_edges += len(neighbor_ids)
            boundary_edges += int(np.count_nonzero(~inside[neighbor_ids]))
    return float(boundary_edges / total_edges) if total_edges else 0.0


def _internal_affinity(features: np.ndarray, core_ids: np.ndarray) -> float:
    prototype = features[core_ids].mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-12:
        return 0.0
    similarities = features[core_ids] @ (prototype / norm)
    return float(np.clip(np.mean(similarities), 0.0, 1.0))


def materialize_candidate_bank(
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    association: AssociationResult,
    consensus: ConsensusResult,
    final_track_id: np.ndarray,
    classifications: Mapping[int, TrackClassification],
    config: V9Config = V9Config(),
) -> CandidateBank:
    """Freeze candidate geometry, score evidence, and masks for CPU replay."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity, name="affinity")
    final_tracks = np.asarray(final_track_id, dtype=np.int32)
    if xyz.shape != (len(consensus.core_track_id), 3):
        raise ValueError("xyz_m and consensus must have matching rows")
    if len(features) != len(xyz) or final_tracks.shape != (len(xyz),):
        raise ValueError("affinity and final_track_id must match xyz_m")
    track_by_id = {int(track.track_id): track for track in association.tracks}
    tree = cKDTree(xyz)
    core_labels = np.full(len(xyz), -1, dtype=np.int32)
    full_masks: list[np.ndarray] = []
    core_masks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for track_id in consensus.valid_track_ids:
        classification = classifications.get(track_id)
        if classification is None:
            # Geometry is the bank identity.  Late classification is merely
            # an attribute on that immutable track; it may not censor the
            # candidate pool used to choose A0--A3.
            classification = TrackClassification(
                track_id,
                -1,
                "__unknown__",
                0.0,
                0.0,
                0,
                "missing",
                False,
            )
        core_ids = consensus.core_ids[track_id]
        full_ids = np.flatnonzero(final_tracks == track_id).astype(np.int32)
        if len(core_ids) < int(config.core_min_points) or not len(full_ids):
            continue
        if not np.all(np.isin(core_ids, full_ids)):
            raise ValueError("final track mask must retain its unique core")
        positive = consensus.positive_views[track_id].take(core_ids)
        visible = np.maximum(consensus.visible_views[core_ids], 1)
        conflict = consensus.conflict_views[track_id].take(core_ids)
        positive_ratio = float(np.mean(positive / visible))
        conflict_ratio = float(np.mean(conflict / visible))
        track = track_by_id[track_id]
        overlap = float(np.median(track.merge_scores)) if track.merge_scores else 0.0
        view_term = min(len(track.frame_ids), 5) / 5.0
        affinity_term = _internal_affinity(features, core_ids)
        terms = np.asarray(
            [
                classification.semantic_ratio,
                positive_ratio,
                affinity_term,
                overlap,
                view_term,
                max(0.0, 1.0 - conflict_ratio),
            ],
            dtype=np.float64,
        )
        q = float(np.prod(np.clip(terms, 0.0, 1.0)) ** (1.0 / len(terms)))
        candidate_id = len(rows)
        core_labels[core_ids] = candidate_id
        core_masks.append(core_ids.copy())
        full_masks.append(full_ids.copy())
        extents = np.ptp(xyz[full_ids], axis=0)
        rows.append(
            {
                "candidate_id": candidate_id,
                "track_id": int(track_id),
                "association_mode": association.mode,
                "branch_class": classification.class_name or "__unknown__",
                "class_id": int(classification.class_id),
                "classification_source": classification.source,
                "classification_eligible": bool(classification.eligible),
                "full_point_count": int(len(full_ids)),
                "core_point_count": int(len(core_ids)),
                "halo_point_count": int(len(full_ids) - len(core_ids)),
                "effective_view_count": int(len(track.frame_ids)),
                "semantic_ratio": float(classification.semantic_ratio),
                "semantic_margin": float(classification.semantic_margin),
                "mean_core_positive_ratio": positive_ratio,
                "conflict_ratio": conflict_ratio,
                "internal_affinity": affinity_term,
                "median_track_overlap": overlap,
                "base_score": q,
                "metric_extents_m": [float(value) for value in np.sort(extents)],
                "local_surface_density": _local_density(
                    xyz, core_ids, int(config.local_density_neighbors)
                ),
                "boundary_ratio_5cm": _boundary_ratio(
                    xyz, full_ids, tree, float(config.boundary_radius_m)
                ),
            }
        )
    return CandidateBank(
        point_count=len(xyz),
        association_mode=association.mode,
        core_candidate_id=core_labels,
        full_ids=tuple(full_masks),
        core_ids=tuple(core_masks),
        candidates=tuple(rows),
    )
