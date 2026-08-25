from __future__ import annotations

"""Deterministic V10 fragment association and evidence-preserving reconstruction.

V10 consumes the immutable V9 lifting structures but owns a separate track and
candidate-bank contract.  Semantics are attached only after geometry is frozen.
The module is pure NumPy/SciPy: it has no filesystem, renderer, or ground-truth
dependency.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .v9_objectbank import (
    ConsensusResult,
    Fragment,
    FrameEvidence,
    MultiviewClassVote,
    TrackClassification,
    V9Config,
    attach_local_halo,
    build_consensus_core,
    classify_tracks_codebook,
    classify_tracks_multiview,
    weighted_core_overlap,
)


PairMode = Literal["P0", "P1"]
ReconstructionMode = Literal["R0", "R1"]


@dataclass(frozen=True)
class V10Config:
    frame_top_k: int = 8
    min_shared: int = 3
    p0_min_overlap: float = 0.25
    p1_min_coverage: float = 0.25
    strong_min_coverage: float = 0.80
    match_min_margin: float = 0.10
    component_min_consensus: float = 0.80
    full_membership: float = 0.40
    core_membership: float = 0.60
    min_positive_views: int = 2
    ownership_min_margin: float = 0.10
    min_core_points: int = 3
    r0_attach_radius_m: float = 0.05
    r0_attach_min_anchors: int = 3
    r0_attach_min_affinity: float = 0.95
    r0_attach_min_margin: float = 0.02


@dataclass(frozen=True)
class V10PairEvidence:
    left_fragment_id: int
    right_fragment_id: int
    left_frame_id: int
    right_frame_id: int
    shared_core: int
    shared_full: int
    p0_overlap: float
    left_coverage: float
    right_coverage: float
    p1_score: float
    weighted_jaccard: float
    p0_eligible: bool
    p1_eligible: bool
    strong: bool

    def score(self, mode: PairMode) -> float:
        if mode == "P0":
            return self.p0_overlap if self.p0_eligible else 0.0
        if mode == "P1":
            return self.p1_score if self.p1_eligible else 0.0
        raise ValueError(f"unknown V10 pair mode: {mode}")


@dataclass(frozen=True)
class V10MatchEdge:
    left_fragment_id: int
    right_fragment_id: int
    left_frame_id: int
    right_frame_id: int
    score: float
    shared: int
    strong: bool
    cycle_supported: bool
    frame_weighted_jaccard: float
    p0_overlap: float = 0.0
    left_coverage: float = 0.0
    right_coverage: float = 0.0
    row_margin: float = 0.0
    column_margin: float = 0.0
    component_support_ratio: float = 0.0


@dataclass(frozen=True)
class V10Track:
    track_id: int
    fragment_ids: tuple[int, ...]
    frame_ids: tuple[int, ...]
    component_consensus: float
    pair_mode: PairMode


@dataclass(frozen=True)
class V10Association:
    pair_mode: PairMode
    frame_pairs: tuple[tuple[int, int], ...]
    tentative_edges: tuple[V10MatchEdge, ...]
    accepted_edges: tuple[V10MatchEdge, ...]
    tracks: tuple[V10Track, ...]


@dataclass(frozen=True)
class V10CandidateBank:
    point_count: int
    pair_mode: PairMode
    reconstruction_mode: ReconstructionMode
    point_candidate_id: np.ndarray
    full_ids: tuple[np.ndarray, ...]
    core_ids: tuple[np.ndarray, ...]
    candidates: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        labels = np.asarray(self.point_candidate_id, dtype=np.int32).copy()
        full = tuple(np.unique(np.asarray(row, dtype=np.int32)) for row in self.full_ids)
        core = tuple(np.unique(np.asarray(row, dtype=np.int32)) for row in self.core_ids)
        rows = tuple(dict(row) for row in self.candidates)
        if int(self.point_count) <= 0 or labels.shape != (int(self.point_count),):
            raise ValueError("V10 bank labels must match point_count")
        if not (len(rows) == len(full) == len(core)):
            raise ValueError("V10 bank rows and masks must align")
        if [int(row.get("candidate_id", -1)) for row in rows] != list(range(len(rows))):
            raise ValueError("V10 candidate IDs must be contiguous")
        if np.any(labels < -1) or np.any(labels >= len(rows)):
            raise ValueError("V10 bank contains an invalid candidate ID")
        for candidate_id, (full_row, core_row) in enumerate(zip(full, core)):
            if np.any(full_row < 0) or np.any(full_row >= int(self.point_count)):
                raise ValueError("V10 full mask contains an out-of-range point")
            if not np.all(np.isin(core_row, full_row, assume_unique=True)):
                raise ValueError("V10 core must be a subset of full")
            if not np.array_equal(np.flatnonzero(labels == candidate_id), full_row):
                raise ValueError("V10 labels and full masks disagree")
            full_row.setflags(write=False)
            core_row.setflags(write=False)
        labels.setflags(write=False)
        object.__setattr__(self, "point_candidate_id", labels)
        object.__setattr__(self, "full_ids", full)
        object.__setattr__(self, "core_ids", core)
        object.__setattr__(self, "candidates", rows)


_FUNNEL_STAGES = (
    "single_full",
    "single_core",
    "component_full_union",
    "component_core_union",
    "pre_conflict",
    "post_conflict",
    "unique_ownership",
    "final_candidate",
)


def _sparse_take(ids: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.int64)
    output = np.zeros(len(query), dtype=np.float64)
    if not len(ids) or not len(query):
        return output
    positions = np.searchsorted(ids, query)
    valid = positions < len(ids)
    matched = np.zeros(len(query), dtype=bool)
    matched[valid] = ids[positions[valid]] == query[valid]
    output[matched] = values[positions[matched]]
    return output


def _fragment_membership(fragment: Fragment, frame: FrameEvidence) -> tuple[np.ndarray, np.ndarray]:
    visible = _sparse_take(frame.visible_ids, frame.visible_mass, fragment.full_ids)
    if np.any(visible <= 0):
        raise ValueError("fragment full support must have positive frame visibility")
    membership = np.divide(
        np.asarray(fragment.full_mass, dtype=np.float64),
        visible,
        out=np.zeros(len(fragment.full_ids), dtype=np.float64),
        where=visible > 0,
    )
    return fragment.full_ids, np.clip(membership, 0.0, 1.0)


def _weighted_sparse_jaccard(
    left_ids: np.ndarray,
    left_values: np.ndarray,
    right_ids: np.ndarray,
    right_values: np.ndarray,
) -> float:
    union = np.union1d(left_ids, right_ids)
    if not len(union):
        return 0.0
    left = _sparse_take(left_ids, np.asarray(left_values, dtype=np.float64), union)
    right = _sparse_take(right_ids, np.asarray(right_values, dtype=np.float64), union)
    denominator = float(np.maximum(left, right).sum())
    return float(np.minimum(left, right).sum() / denominator) if denominator else 0.0


def frame_weighted_jaccard(left: FrameEvidence, right: FrameEvidence) -> float:
    return _weighted_sparse_jaccard(
        left.visible_ids,
        left.visible_mass,
        right.visible_ids,
        right.visible_mass,
    )


def select_covisible_frame_pairs(
    frames: Sequence[FrameEvidence], config: V10Config = V10Config()
) -> tuple[tuple[int, int], ...]:
    """Return the symmetric union of each frame's positive top-k neighbours."""

    ordered = tuple(sorted(frames, key=lambda row: int(row.frame_id)))
    if len({int(row.frame_id) for row in ordered}) != len(ordered):
        raise ValueError("V10 frame IDs must be unique")
    scores: dict[tuple[int, int], float] = {}
    by_id = {int(row.frame_id): row for row in ordered}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            score = frame_weighted_jaccard(left, right)
            if score > 0:
                scores[(int(left.frame_id), int(right.frame_id))] = score
    selected: set[tuple[int, int]] = set()
    top_k = max(int(config.frame_top_k), 0)
    for frame_id in sorted(by_id):
        neighbors = [
            (score, right if left == frame_id else left)
            for (left, right), score in scores.items()
            if left == frame_id or right == frame_id
        ]
        neighbors.sort(key=lambda row: (-row[0], row[1]))
        for _, neighbor in neighbors[:top_k]:
            selected.add(tuple(sorted((frame_id, neighbor))))
    return tuple(sorted(selected))


def pair_evidence(
    left: Fragment,
    right: Fragment,
    frames_by_id: Mapping[int, FrameEvidence],
    config: V10Config = V10Config(),
) -> V10PairEvidence:
    if int(left.frame_id) == int(right.frame_id):
        raise ValueError("V10 pair evidence requires different frames")
    shared_core, p0_overlap = weighted_core_overlap(left, right)
    left_ids, left_membership = _fragment_membership(left, frames_by_id[int(left.frame_id)])
    right_ids, right_membership = _fragment_membership(right, frames_by_id[int(right.frame_id)])
    shared, li, ri = np.intersect1d(
        left_ids,
        right_ids,
        assume_unique=True,
        return_indices=True,
    )
    union_ids = np.union1d(left_ids, right_ids)
    left_probability = _sparse_take(left_ids, left_membership, union_ids)
    right_probability = _sparse_take(right_ids, right_membership, union_ids)
    left_visibility = _sparse_take(
        frames_by_id[int(left.frame_id)].visible_ids,
        frames_by_id[int(left.frame_id)].visible_mass,
        union_ids,
    )
    right_visibility = _sparse_take(
        frames_by_id[int(right.frame_id)].visible_ids,
        frames_by_id[int(right.frame_id)].visible_mass,
        union_ids,
    )
    common_probability = left_probability * right_probability
    left_denominator = float((right_visibility * left_probability).sum())
    right_denominator = float((left_visibility * right_probability).sum())
    left_coverage = (
        float((right_visibility * common_probability).sum()) / left_denominator
        if left_denominator
        else 0.0
    )
    right_coverage = (
        float((left_visibility * common_probability).sum()) / right_denominator
        if right_denominator
        else 0.0
    )
    jaccard = _weighted_sparse_jaccard(
        left_ids, left_membership, right_ids, right_membership
    )
    p1_score = float(np.sqrt(max(left_coverage * right_coverage, 0.0)))
    p0_eligible = (
        shared_core >= int(config.min_shared)
        and p0_overlap >= float(config.p0_min_overlap)
    )
    p1_eligible = (
        len(shared) >= int(config.min_shared)
        and left_coverage >= float(config.p1_min_coverage)
        and right_coverage >= float(config.p1_min_coverage)
    )
    strong = (
        len(shared) >= int(config.min_shared)
        and left_coverage >= float(config.strong_min_coverage)
        and right_coverage >= float(config.strong_min_coverage)
    )
    return V10PairEvidence(
        int(left.fragment_id),
        int(right.fragment_id),
        int(left.frame_id),
        int(right.frame_id),
        int(shared_core),
        int(len(shared)),
        float(p0_overlap),
        float(left_coverage),
        float(right_coverage),
        float(p1_score),
        float(jaccard),
        bool(p0_eligible),
        bool(p1_eligible),
        bool(strong),
    )


def _runner_up_margin(values: np.ndarray, selected: int) -> float:
    best = float(values[selected])
    if len(values) <= 1:
        return best
    remaining = np.delete(values, selected)
    return best - float(np.max(remaining))


def _match_frame_pair(
    left_fragments: Sequence[Fragment],
    right_fragments: Sequence[Fragment],
    frames_by_id: Mapping[int, FrameEvidence],
    pair_mode: PairMode,
    frame_jaccard: float,
    config: V10Config,
) -> tuple[V10MatchEdge, ...]:
    left_rows = tuple(sorted(left_fragments, key=lambda row: int(row.fragment_id)))
    right_rows = tuple(sorted(right_fragments, key=lambda row: int(row.fragment_id)))
    if not left_rows or not right_rows:
        return ()
    evidence: list[list[V10PairEvidence]] = []
    scores = np.zeros((len(left_rows), len(right_rows)), dtype=np.float64)
    for left_index, left in enumerate(left_rows):
        row: list[V10PairEvidence] = []
        for right_index, right in enumerate(right_rows):
            current = pair_evidence(left, right, frames_by_id, config)
            row.append(current)
            scores[left_index, right_index] = current.score(pair_mode)
        evidence.append(row)
    if not np.any(scores > 0):
        return ()
    # Rows/columns are stable-ID sorted, so scipy's deterministic tie behavior
    # yields stable assignments.  Mutual-best below rejects ambiguous ties.
    row_indices, column_indices = linear_sum_assignment(-scores)
    output: list[V10MatchEdge] = []
    for left_index, right_index in zip(row_indices, column_indices):
        score = float(scores[left_index, right_index])
        if score <= 0:
            continue
        if right_index != int(np.argmax(scores[left_index])):
            continue
        if left_index != int(np.argmax(scores[:, right_index])):
            continue
        row_margin = _runner_up_margin(scores[left_index], int(right_index))
        column_margin = _runner_up_margin(scores[:, right_index], int(left_index))
        if row_margin < float(config.match_min_margin):
            continue
        if column_margin < float(config.match_min_margin):
            continue
        current = evidence[int(left_index)][int(right_index)]
        output.append(
            V10MatchEdge(
                current.left_fragment_id,
                current.right_fragment_id,
                current.left_frame_id,
                current.right_frame_id,
                score,
                current.shared_full if pair_mode == "P1" else current.shared_core,
                bool(current.strong) if pair_mode == "P1" else False,
                False,
                float(frame_jaccard),
                float(current.p0_overlap),
                float(current.left_coverage),
                float(current.right_coverage),
                float(row_margin),
                float(column_margin),
            )
        )
    return tuple(output)


def _mark_three_view_cycles(
    edges: Sequence[V10MatchEdge], fragments_by_id: Mapping[int, Fragment]
) -> tuple[V10MatchEdge, ...]:
    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left_fragment_id, set()).add(edge.right_fragment_id)
        adjacency.setdefault(edge.right_fragment_id, set()).add(edge.left_fragment_id)
    output: list[V10MatchEdge] = []
    for edge in edges:
        cycle = False
        common = adjacency.get(edge.left_fragment_id, set()).intersection(
            adjacency.get(edge.right_fragment_id, set())
        )
        for third in sorted(common):
            frames = {
                int(fragments_by_id[edge.left_fragment_id].frame_id),
                int(fragments_by_id[edge.right_fragment_id].frame_id),
                int(fragments_by_id[third].frame_id),
            }
            if len(frames) == 3:
                cycle = True
                break
        output.append(
            V10MatchEdge(
                edge.left_fragment_id,
                edge.right_fragment_id,
                edge.left_frame_id,
                edge.right_frame_id,
                edge.score,
                edge.shared,
                edge.strong,
                cycle,
                edge.frame_weighted_jaccard,
                edge.p0_overlap,
                edge.left_coverage,
                edge.right_coverage,
                edge.row_margin,
                edge.column_margin,
                edge.component_support_ratio,
            )
        )
    return tuple(output)


@dataclass
class _Component:
    fragment_ids: set[int]
    frame_ids: set[int]
    consensus_values: list[float]


def _component_support_ratio(
    left: _Component,
    right: _Component,
    selected_frame_pairs: set[tuple[int, int]],
    tentative_pairs: set[tuple[int, int]],
    fragments_by_id: Mapping[int, Fragment],
) -> float:
    comparable = 0
    supported = 0
    for left_id in sorted(left.fragment_ids):
        for right_id in sorted(right.fragment_ids):
            left_fragment = fragments_by_id[left_id]
            right_fragment = fragments_by_id[right_id]
            if int(left_fragment.frame_id) == int(right_fragment.frame_id):
                return 0.0
            frame_pair = tuple(sorted((int(left_fragment.frame_id), int(right_fragment.frame_id))))
            if frame_pair not in selected_frame_pairs:
                continue
            comparable += 1
            if tuple(sorted((left_id, right_id))) in tentative_pairs:
                supported += 1
    return supported / comparable if comparable else 0.0


def associate_fragments_v10(
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    pair_mode: PairMode,
    config: V10Config = V10Config(),
    *,
    view_consensus: bool = False,
) -> V10Association:
    if pair_mode not in {"P0", "P1"}:
        raise ValueError(f"unknown V10 pair mode: {pair_mode}")
    ordered_fragments = tuple(sorted(fragments, key=lambda row: int(row.fragment_id)))
    fragments_by_id = {int(row.fragment_id): row for row in ordered_fragments}
    if len(fragments_by_id) != len(ordered_fragments):
        raise ValueError("V10 fragment IDs must be unique")
    frames_by_id = {int(row.frame_id): row for row in frames}
    if len(frames_by_id) != len(frames):
        raise ValueError("V10 frame IDs must be unique")
    if any(int(row.frame_id) not in frames_by_id for row in ordered_fragments):
        raise ValueError("every V10 fragment must reference a frame")
    fragments_by_frame: dict[int, list[Fragment]] = {}
    for fragment in ordered_fragments:
        fragments_by_frame.setdefault(int(fragment.frame_id), []).append(fragment)
    frame_pairs = select_covisible_frame_pairs(tuple(frames_by_id.values()), config)
    tentative: list[V10MatchEdge] = []
    for left_frame, right_frame in frame_pairs:
        tentative.extend(
            _match_frame_pair(
                fragments_by_frame.get(left_frame, ()),
                fragments_by_frame.get(right_frame, ()),
                frames_by_id,
                pair_mode,
                frame_weighted_jaccard(frames_by_id[left_frame], frames_by_id[right_frame]),
                config,
            )
        )
    if view_consensus:
        if pair_mode != "P1":
            raise ValueError("V10 view consensus is defined only for P1 evidence")
        tentative = list(_mark_three_view_cycles(tentative, fragments_by_id))
        eligible = [edge for edge in tentative if edge.strong or edge.cycle_supported]
        eligible.sort(
            key=lambda edge: (
                not edge.strong,
                not edge.cycle_supported,
                -float(edge.score),
                -int(edge.shared),
                int(edge.left_fragment_id),
                int(edge.right_fragment_id),
            )
        )
    else:
        eligible = sorted(
            tentative,
            key=lambda edge: (
                -float(edge.score),
                -int(edge.shared),
                int(edge.left_fragment_id),
                int(edge.right_fragment_id),
            ),
        )
    components = {
        int(fragment.fragment_id): _Component(
            {int(fragment.fragment_id)}, {int(fragment.frame_id)}, []
        )
        for fragment in ordered_fragments
    }
    owner = {fragment_id: fragment_id for fragment_id in fragments_by_id}
    accepted: list[V10MatchEdge] = []
    selected_set = set(frame_pairs)
    tentative_pairs = {
        tuple(sorted((edge.left_fragment_id, edge.right_fragment_id)))
        for edge in tentative
    }
    for edge in eligible:
        left_root = owner[edge.left_fragment_id]
        right_root = owner[edge.right_fragment_id]
        if left_root == right_root:
            continue
        left_component = components[left_root]
        right_component = components[right_root]
        if left_component.frame_ids.intersection(right_component.frame_ids):
            continue
        # A cycle-supported weak edge may attach a singleton hypothesis to an
        # established component, but it may never become a one-edge bridge
        # between two already established components.  Such a bridge is the
        # exact transitive-collapse failure that VC1 is designed to exclude.
        if (
            view_consensus
            and not edge.strong
            and len(left_component.fragment_ids) > 1
            and len(right_component.fragment_ids) > 1
        ):
            continue
        if view_consensus:
            consensus = _component_support_ratio(
                left_component,
                right_component,
                selected_set,
                tentative_pairs,
                fragments_by_id,
            )
            if consensus < float(config.component_min_consensus):
                continue
        else:
            consensus = float(edge.score)
        keep, remove = sorted((left_root, right_root))
        target = components[keep]
        source = components[remove]
        target.fragment_ids.update(source.fragment_ids)
        target.frame_ids.update(source.frame_ids)
        target.consensus_values.extend(source.consensus_values)
        target.consensus_values.append(float(consensus))
        for fragment_id in source.fragment_ids:
            owner[fragment_id] = keep
        del components[remove]
        accepted.append(
            replace(
                edge,
                component_support_ratio=(
                    float(consensus) if view_consensus else 1.0
                ),
            )
        )
    ordered_components = sorted(components.values(), key=lambda row: min(row.fragment_ids))
    tracks = tuple(
        V10Track(
            track_id=index,
            fragment_ids=tuple(sorted(component.fragment_ids)),
            frame_ids=tuple(sorted(component.frame_ids)),
            component_consensus=(
                min(component.consensus_values) if component.consensus_values else 1.0
            ),
            pair_mode=pair_mode,
        )
        for index, component in enumerate(ordered_components)
    )
    return V10Association(
        pair_mode,
        tuple(frame_pairs),
        tuple(
            sorted(
                tentative,
                key=lambda edge: (edge.left_fragment_id, edge.right_fragment_id),
            )
        ),
        tuple(accepted),
        tracks,
    )


@dataclass(frozen=True)
class _TrackMembership:
    track_id: int
    full_ids: np.ndarray
    full_scores: np.ndarray
    core_ids: np.ndarray
    core_scores: np.ndarray


def _track_membership(
    track: V10Track,
    fragments_by_id: Mapping[int, Fragment],
    frames_by_id: Mapping[int, FrameEvidence],
    config: V10Config,
    *,
    core_from_full: bool,
) -> _TrackMembership:
    members = [fragments_by_id[item] for item in track.fragment_ids]
    union = np.unique(np.concatenate([row.full_ids for row in members])).astype(np.int32)
    membership_sum = np.zeros(len(union), dtype=np.float64)
    visible_views = np.zeros(len(union), dtype=np.int32)
    full_positive = np.zeros(len(union), dtype=np.int32)
    core_positive = np.zeros(len(union), dtype=np.int32)
    for fragment in members:
        frame = frames_by_id[int(fragment.frame_id)]
        visible = _sparse_take(frame.visible_ids, frame.visible_mass, union)
        visible_views += visible > 0
        ids, membership = _fragment_membership(fragment, frame)
        positions = np.searchsorted(union, ids)
        membership_sum[positions] += membership
        full_positive[positions] += membership >= float(config.full_membership)
        if core_from_full:
            core_positive[positions] += membership >= float(config.core_membership)
        else:
            core_positions = np.searchsorted(ids, fragment.core_ids)
            core_membership = membership[core_positions]
            union_core_positions = np.searchsorted(union, fragment.core_ids)
            core_positive[union_core_positions] += core_membership >= float(
                config.core_membership
            )
    scores = np.divide(
        membership_sum,
        np.maximum(visible_views, 1),
        out=np.zeros_like(membership_sum),
        where=visible_views > 0,
    )
    full_keep = (
        (full_positive >= 1)
        & (scores >= float(config.full_membership))
    )
    core_keep = (
        (core_positive >= int(config.min_positive_views))
        & (scores >= float(config.core_membership))
    )
    return _TrackMembership(
        int(track.track_id),
        union[full_keep],
        scores[full_keep].astype(np.float32),
        union[core_keep],
        scores[core_keep].astype(np.float32),
    )


def _unique_ownership(
    rows: Sequence[_TrackMembership],
    point_count: int,
    *,
    core: bool,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    choices: dict[int, list[tuple[float, int]]] = {}
    for row in rows:
        ids = row.core_ids if core else row.full_ids
        scores = row.core_scores if core else row.full_scores
        for point_id, score in zip(ids, scores):
            choices.setdefault(int(point_id), []).append((float(score), int(row.track_id)))
    owner = np.full(point_count, -1, dtype=np.int32)
    score_output = np.zeros(point_count, dtype=np.float32)
    for point_id, point_choices in choices.items():
        point_choices.sort(key=lambda item: (-item[0], item[1]))
        runner_up = point_choices[1][0] if len(point_choices) > 1 else 0.0
        if point_choices[0][0] - runner_up >= float(margin):
            score_output[point_id] = point_choices[0][0]
            owner[point_id] = point_choices[0][1]
    return owner, score_output


def _v9_config(config: V10Config) -> V9Config:
    return V9Config(
        core_min_positive_views=int(config.min_positive_views),
        core_min_positive_ratio=float(config.core_membership),
        core_min_points=int(config.min_core_points),
        attach_radius_m=float(config.r0_attach_radius_m),
        attach_min_anchors=int(config.r0_attach_min_anchors),
        attach_min_affinity=float(config.r0_attach_min_affinity),
        attach_min_margin=float(config.r0_attach_min_margin),
    )


def _v9_consensus(
    association: V10Association,
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    point_count: int,
    config: V10Config,
) -> ConsensusResult:
    return build_consensus_core(  # type: ignore[arg-type]
        association,
        fragments,
        frames,
        point_count,
        _v9_config(config),
    )


def _r0_consensus(
    association: V10Association,
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    point_count: int,
    config: V10Config,
) -> ConsensusResult:
    # R0 is the exact V9 reconstruction negative control.  In particular it
    # must not inherit V10's 0.10 ambiguous-ownership rejection; that margin is
    # one of the registered R1 semantics.
    return _v9_consensus(association, fragments, frames, point_count, config)


def _normalise_feature_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 1e-12)


def _local_surface_density(xyz: np.ndarray, core_ids: np.ndarray) -> float:
    k = min(16, max(len(core_ids) - 1, 0))
    if k <= 0:
        return 0.0
    distances, _ = cKDTree(xyz[core_ids]).query(xyz[core_ids], k=k + 1)
    radii = np.maximum(np.asarray(distances)[:, -1], 1e-6)
    return float(np.median(k / (np.pi * radii * radii)))


def _boundary_ratio_5cm(
    xyz: np.ndarray, member_ids: np.ndarray, tree: cKDTree
) -> float:
    if not len(member_ids):
        return 0.0
    inside = np.zeros(len(xyz), dtype=bool)
    inside[member_ids] = True
    boundary_edges = 0
    total_edges = 0
    for start in range(0, len(member_ids), 8192):
        query_ids = member_ids[start : start + 8192]
        neighborhoods = tree.query_ball_point(xyz[query_ids], r=0.05)
        for point_id, neighbors in zip(query_ids, neighborhoods):
            neighbor_ids = np.asarray(neighbors, dtype=np.int64)
            neighbor_ids = neighbor_ids[neighbor_ids != int(point_id)]
            total_edges += len(neighbor_ids)
            boundary_edges += int(np.count_nonzero(~inside[neighbor_ids]))
    return float(boundary_edges / total_edges) if total_edges else 0.0


def _internal_affinity(features: np.ndarray, core_ids: np.ndarray) -> float:
    if not len(core_ids):
        return 0.0
    prototype = features[core_ids].mean(axis=0)
    norm = float(np.linalg.norm(prototype))
    if norm <= 1e-12:
        return 0.0
    return float(
        np.clip(np.mean(features[core_ids] @ (prototype / norm)), 0.0, 1.0)
    )


def build_v10_candidate_bank(
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    point_count: int,
    *,
    pair_mode: PairMode,
    reconstruction_mode: ReconstructionMode,
    classifications: Mapping[int, TrackClassification] | None = None,
    xyz_m: np.ndarray | None = None,
    affinity: np.ndarray | None = None,
    config: V10Config = V10Config(),
    view_consensus: bool = False,
    frozen_association: V10Association | None = None,
) -> tuple[V10Association, V10CandidateBank]:
    """Associate once and deterministically reconstruct an independent V10 bank."""

    if reconstruction_mode not in {"R0", "R1"}:
        raise ValueError(f"unknown V10 reconstruction mode: {reconstruction_mode}")
    if int(point_count) <= 0:
        raise ValueError("V10 point_count must be positive")
    xyz = None if xyz_m is None else np.asarray(xyz_m, dtype=np.float64)
    features = None if affinity is None else _normalise_feature_rows(
        np.asarray(affinity), name="affinity"
    )
    if xyz is not None and xyz.shape != (int(point_count), 3):
        raise ValueError("V10 xyz_m must match point_count")
    if features is not None and len(features) != int(point_count):
        raise ValueError("V10 affinity must match point_count")
    if (xyz is None) != (features is None):
        raise ValueError("V10 xyz_m and affinity must be supplied together")
    association = frozen_association or associate_fragments_v10(
        fragments,
        frames,
        pair_mode,
        config,
        view_consensus=view_consensus,
    )
    if association.pair_mode != pair_mode:
        raise ValueError("frozen V10 association has the wrong pair mode")
    fragments_by_id = {int(row.fragment_id): row for row in fragments}
    frames_by_id = {int(row.frame_id): row for row in frames}
    if reconstruction_mode == "R0":
        if xyz is None or features is None:
            raise ValueError("V10 R0 requires xyz_m and affinity for the V9 halo")
        consensus = _r0_consensus(
            association, fragments, frames, int(point_count), config
        )
        core_owner = np.asarray(consensus.core_track_id, dtype=np.int32).copy()
        core_score = np.zeros(int(point_count), dtype=np.float32)
        for track_id in consensus.valid_track_ids:
            ids = consensus.core_ids[track_id]
            positive = consensus.positive_views[track_id].take(ids)
            core_score[ids] = positive / np.maximum(consensus.visible_views[ids], 1)
        full_owner = attach_local_halo(
            xyz,
            features,
            consensus,
            _v9_config(config),
        )
        full_score = core_score.copy()
    else:
        memberships = tuple(
            _track_membership(
                track,
                fragments_by_id,
                frames_by_id,
                config,
                core_from_full=True,
            )
            for track in association.tracks
            if len(track.frame_ids) >= int(config.min_positive_views)
        )
        core_owner, core_score = _unique_ownership(
            memberships,
            int(point_count),
            core=True,
            margin=float(config.ownership_min_margin),
        )
        full_owner, full_score = _unique_ownership(
            memberships,
            int(point_count),
            core=False,
            margin=float(config.ownership_min_margin),
        )
        fixed_core = core_owner >= 0
        full_owner[fixed_core] = core_owner[fixed_core]
        full_score[fixed_core] = core_score[fixed_core]
    track_by_id = {int(row.track_id): row for row in association.tracks}
    valid_tracks = [
        track_id
        for track_id in sorted(track_by_id)
        if np.count_nonzero(core_owner == track_id) >= int(config.min_core_points)
    ]
    labels = np.full(int(point_count), -1, dtype=np.int32)
    full_masks: list[np.ndarray] = []
    core_masks: list[np.ndarray] = []
    candidates: list[dict[str, Any]] = []
    classification_rows = classifications or {}
    tree = cKDTree(xyz) if xyz is not None else None
    for track_id in valid_tracks:
        core_ids = np.flatnonzero(core_owner == track_id).astype(np.int32)
        full_ids = np.flatnonzero(full_owner == track_id).astype(np.int32)
        full_ids = np.union1d(full_ids, core_ids).astype(np.int32)
        candidate_id = len(candidates)
        labels[full_ids] = candidate_id
        track = track_by_id[track_id]
        classification = classification_rows.get(track_id)
        semantic_ratio = float(classification.semantic_ratio) if classification else 0.0
        mean_core = float(np.mean(core_score[core_ids])) if len(core_ids) else 0.0
        view_term = min(len(track.frame_ids), 5) / 5.0
        component_fragments = [fragments_by_id[item] for item in track.fragment_ids]
        conflict_values = [
            float(row.conflict_ratio)
            for row in component_fragments
            if row.conflict_ratio is not None
        ]
        conflict_ratio = float(np.mean(conflict_values)) if conflict_values else 0.0
        affinity_term = (
            _internal_affinity(features, core_ids) if features is not None else 1.0
        )
        base_score = float(
            np.prod(
                np.clip(
                    [
                        semantic_ratio,
                        track.component_consensus,
                        mean_core,
                        affinity_term,
                        view_term,
                        1.0 - conflict_ratio,
                    ],
                    0.0,
                    1.0,
                )
            )
            ** (1.0 / 6.0)
        )
        extents = (
            np.sort(np.ptp(xyz[full_ids], axis=0))
            if xyz is not None and len(full_ids)
            else np.zeros(3, dtype=np.float64)
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "track_id": int(track_id),
                "pair_mode": pair_mode,
                "reconstruction_mode": reconstruction_mode,
                "fragment_ids": list(track.fragment_ids),
                "frame_ids": list(track.frame_ids),
                "component_consensus": float(track.component_consensus),
                "full_point_count": int(len(full_ids)),
                "core_point_count": int(len(core_ids)),
                "mean_core_membership": mean_core,
                "conflict_ratio": conflict_ratio,
                "internal_affinity": affinity_term,
                "base_score": base_score,
                "branch_class": (
                    classification.class_name
                    if classification and classification.class_name
                    else "__unknown__"
                ),
                "class_id": int(classification.class_id) if classification else -1,
                "semantic_ratio": semantic_ratio,
                "classification_source": (
                    classification.source if classification else "missing"
                ),
                "classification_eligible": (
                    bool(classification.eligible) if classification else False
                ),
                "metric_extents_m": [float(value) for value in extents],
                "local_surface_density": (
                    _local_surface_density(xyz, core_ids) if xyz is not None else 0.0
                ),
                "boundary_ratio_5cm": (
                    _boundary_ratio_5cm(xyz, full_ids, tree)
                    if xyz is not None and tree is not None
                    else 0.0
                ),
            }
        )
        full_masks.append(full_ids)
        core_masks.append(core_ids)
    return association, V10CandidateBank(
        int(point_count),
        pair_mode,
        reconstruction_mode,
        labels,
        tuple(full_masks),
        tuple(core_masks),
        tuple(candidates),
    )


def _ragged_row(
    indptr: np.ndarray, values: np.ndarray, index: int, *, dtype: Any
) -> np.ndarray:
    start, stop = int(indptr[index]), int(indptr[index + 1])
    return np.asarray(values[start:stop], dtype=dtype)


def _lifting_fragments(arrays: Mapping[str, np.ndarray]) -> tuple[Fragment, ...]:
    full_indptr = np.asarray(arrays["fragment_full_indptr"])
    core_indptr = np.asarray(arrays["fragment_core_indptr"])
    fragment_ids = np.asarray(arrays["fragment_id"])
    return tuple(
        Fragment(
            fragment_id=int(fragment_id),
            frame_id=int(arrays["fragment_frame"][index]),
            mask_index=int(arrays["fragment_mask_index"][index]),
            full_ids=_ragged_row(
                full_indptr, arrays["fragment_full_ids"], index, dtype=np.int32
            ),
            core_ids=_ragged_row(
                core_indptr, arrays["fragment_core_ids"], index, dtype=np.int32
            ),
            full_mass=_ragged_row(
                full_indptr, arrays["fragment_full_mass"], index, dtype=np.float32
            ),
            core_mass=_ragged_row(
                core_indptr, arrays["fragment_core_mass"], index, dtype=np.float32
            ),
            conflict_ratio=float(arrays["fragment_conflict_ratio"][index]),
        )
        for index, fragment_id in enumerate(fragment_ids)
    )


def _lifting_frames(
    arrays: Mapping[str, np.ndarray],
    fragments: Sequence[Fragment],
    frame_count: int,
) -> tuple[FrameEvidence, ...]:
    visible_indptr = np.asarray(arrays["frame_visible_indptr"])
    abstained = np.asarray(
        arrays.get("frame_geometry_abstained", np.zeros(frame_count, dtype=bool)),
        dtype=bool,
    )
    by_frame: dict[int, list[Fragment]] = {}
    for fragment in fragments:
        by_frame.setdefault(int(fragment.frame_id), []).append(fragment)
    return tuple(
        FrameEvidence(
            frame_id=frame_id,
            fragments=tuple(
                sorted(
                    by_frame.get(frame_id, ()),
                    key=lambda row: int(row.fragment_id),
                )
            ),
            visible_ids=_ragged_row(
                visible_indptr,
                arrays["frame_visible_ids"],
                frame_id,
                dtype=np.int32,
            ),
            visible_mass=_ragged_row(
                visible_indptr,
                arrays["frame_visible_mass"],
                frame_id,
                dtype=np.float32,
            ),
            abstain=bool(abstained[frame_id]),
        )
        for frame_id in range(frame_count)
    )


def _weighted_iou(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> float:
    return _weighted_sparse_jaccard(left_ids, left_mass, right_ids, right_mass)


def _semantic_rows(
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[int, int, np.ndarray, np.ndarray], ...]:
    indptr = np.asarray(arrays["semantic_fragment_full_indptr"])
    classes = np.asarray(arrays["semantic_fragment_class"])
    return tuple(
        (
            int(arrays["semantic_fragment_frame"][index]),
            int(class_id),
            _ragged_row(
                indptr,
                arrays["semantic_fragment_full_ids"],
                index,
                dtype=np.int32,
            ),
            _ragged_row(
                indptr,
                arrays["semantic_fragment_full_mass"],
                index,
                dtype=np.float32,
            ),
        )
        for index, class_id in enumerate(classes)
    )


def _multiview_classifications(
    association: V10Association,
    fragments: Sequence[Fragment],
    arrays: Mapping[str, np.ndarray],
    classes: Sequence[str],
) -> dict[int, TrackClassification]:
    fragment_by_id = {int(row.fragment_id): row for row in fragments}
    semantic_by_frame: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for frame_id, class_id, ids, mass in _semantic_rows(arrays):
        semantic_by_frame.setdefault(frame_id, []).append((class_id, ids, mass))
    votes: dict[int, list[MultiviewClassVote]] = {}
    for track in association.tracks:
        for fragment_id in track.fragment_ids:
            fragment = fragment_by_id[int(fragment_id)]
            best_by_class: dict[int, float] = {}
            for class_id, ids, mass in semantic_by_frame.get(
                int(fragment.frame_id), ()
            ):
                score = _weighted_iou(
                    fragment.full_ids,
                    fragment.full_mass,
                    ids,
                    mass,
                )
                best_by_class[class_id] = max(best_by_class.get(class_id, 0.0), score)
            if not best_by_class:
                continue
            # MV-label is a late *per-view* classifier: one physical view can
            # cast at most one vote.  Stable class ID breaks exact IoU ties.
            class_id, score = min(
                best_by_class.items(), key=lambda item: (-item[1], item[0])
            )
            if score < 0.25:
                continue
            votes.setdefault(int(track.track_id), []).append(
                MultiviewClassVote(
                    frame_id=int(fragment.frame_id),
                    class_id=int(class_id),
                    weight=float(score),
                )
            )
    # The V9 classifier intentionally accepts a duck-typed association: V10
    # owns its track schema, while track_id/fragment_ids/frame_ids are the only
    # fields the late semantic classifier consumes.
    return classify_tracks_multiview(association, votes, classes)  # type: ignore[arg-type]


def _codebook_classifications(
    association: V10Association,
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    point_count: int,
    semantic: np.ndarray,
    label_features: np.ndarray,
    classes: Sequence[str],
    config: V10Config,
    *,
    core_from_full: bool,
) -> dict[int, TrackClassification]:
    fragments_by_id = {int(row.fragment_id): row for row in fragments}
    frames_by_id = {int(row.frame_id): row for row in frames}
    if not core_from_full:
        consensus = _r0_consensus(
            association, fragments, frames, point_count, config
        )
    else:
        memberships = tuple(
            _track_membership(
                track,
                fragments_by_id,
                frames_by_id,
                config,
                core_from_full=True,
            )
            for track in association.tracks
            if len(track.frame_ids) >= int(config.min_positive_views)
        )
        core_owner, _ = _unique_ownership(
            memberships,
            point_count,
            core=True,
            margin=float(config.ownership_min_margin),
        )
        valid_track_ids = tuple(
            track_id
            for track_id in sorted(set(core_owner[core_owner >= 0].tolist()))
            if np.count_nonzero(core_owner == track_id) >= int(config.min_core_points)
        )
        consensus = ConsensusResult(
            core_track_id=core_owner,
            visible_views=np.zeros(point_count, dtype=np.int32),
            assignment_margin=np.zeros(point_count, dtype=np.float32),
            valid_track_ids=valid_track_ids,
            positive_views={},
            conflict_views={},
            core_ids={
                track_id: np.flatnonzero(core_owner == track_id).astype(np.int32)
                for track_id in valid_track_ids
            },
        )
    return classify_tracks_codebook(  # type: ignore[arg-type]
        association,
        consensus,
        semantic,
        label_features,
        classes,
    )


def _edge_row(edge: V10MatchEdge) -> dict[str, Any]:
    return {
        "left_fragment_id": int(edge.left_fragment_id),
        "right_fragment_id": int(edge.right_fragment_id),
        "left_frame_id": int(edge.left_frame_id),
        "right_frame_id": int(edge.right_frame_id),
        "kind": "strong" if edge.strong else ("cycle" if edge.cycle_supported else "pair"),
        "score": float(edge.score),
        "shared": int(edge.shared),
        "strong": bool(edge.strong),
        "cycle_supported": bool(edge.cycle_supported),
        "frame_weighted_jaccard": float(edge.frame_weighted_jaccard),
        "p0_overlap": float(edge.p0_overlap),
        "left_coverage": float(edge.left_coverage),
        "right_coverage": float(edge.right_coverage),
        "row_margin": float(edge.row_margin),
        "column_margin": float(edge.column_margin),
        "component_support_ratio": float(edge.component_support_ratio),
    }


def _support_row(
    candidate_id: int,
    gaussian_ids: np.ndarray,
    class_name: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "candidate_id": int(candidate_id),
        "class_name": class_name,
        "gaussian_ids": np.unique(np.asarray(gaussian_ids, dtype=np.int32)),
        **metadata,
    }


def _stage_supports(
    association: V10Association,
    bank: V10CandidateBank,
    fragments: Sequence[Fragment],
    frames: Sequence[FrameEvidence],
    config: V10Config,
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    fragments_by_id = {int(row.fragment_id): row for row in fragments}
    frames_by_id = {int(row.frame_id): row for row in frames}
    if bank.reconstruction_mode == "R0":
        consensus = _r0_consensus(
            association, fragments, frames, bank.point_count, config
        )
        core_owner = np.asarray(consensus.core_track_id, dtype=np.int32)
        pre_supports: list[tuple[int, np.ndarray]] = []
        post_supports: list[tuple[int, np.ndarray]] = []
        for track in association.tracks:
            track_id = int(track.track_id)
            positive = consensus.positive_views[track_id]
            ids = positive.ids
            counts = positive.values
            visible = np.maximum(consensus.visible_views[ids], 1)
            pre_keep = (
                (counts >= int(config.min_positive_views))
                & (counts / visible >= float(config.core_membership))
            )
            conflict = consensus.conflict_views[track_id].take(ids)
            post_keep = pre_keep & (
                conflict / visible <= float(_v9_config(config).core_max_conflict_ratio)
            )
            if np.any(pre_keep):
                pre_supports.append((track_id, ids[pre_keep]))
            if np.any(post_keep):
                # This is deliberately before winner-take-all ownership.
                # `unique_ownership` below records the subsequent V9 choice.
                post_supports.append((track_id, ids[post_keep]))
    else:
        memberships = tuple(
            _track_membership(
                track,
                fragments_by_id,
                frames_by_id,
                config,
                core_from_full=True,
            )
            for track in association.tracks
            if len(track.frame_ids) >= int(config.min_positive_views)
        )
        core_owner, _ = _unique_ownership(
            memberships,
            bank.point_count,
            core=True,
            margin=float(config.ownership_min_margin),
        )
        # R1 has no separate conflict filter: same-frame hierarchical masks
        # are alternative hypotheses.  The pre/post stages must therefore be
        # identical core evidence; 0.40 full membership is already represented
        # by component_full_union and is not mislabeled as conflict removal.
        pre_supports = [
            (int(row.track_id), row.core_ids) for row in memberships if len(row.core_ids)
        ]
        post_supports = list(pre_supports)
    track_by_id = {int(row.track_id): row for row in association.tracks}
    component_full_supports: list[tuple[int, np.ndarray]] = []
    component_core_supports: list[tuple[int, np.ndarray]] = []
    for track in association.tracks:
        full = np.unique(
            np.concatenate(
                [fragments_by_id[item].full_ids for item in track.fragment_ids]
            )
        )
        core = np.unique(
            np.concatenate(
                [fragments_by_id[item].core_ids for item in track.fragment_ids]
            )
        )
        if len(full):
            component_full_supports.append((int(track.track_id), full))
        if len(core):
            component_core_supports.append((int(track.track_id), core))
    component_full = tuple(
        _support_row(index, ids, track_id=track_id)
        for index, (track_id, ids) in enumerate(component_full_supports)
    )
    component_core = tuple(
        _support_row(index, ids, track_id=track_id)
        for index, (track_id, ids) in enumerate(component_core_supports)
    )
    pre_conflict = tuple(
        _support_row(index, ids, track_id=track_id)
        for index, (track_id, ids) in enumerate(pre_supports)
    )
    post_conflict = tuple(
        _support_row(index, ids, track_id=track_id)
        for index, (track_id, ids) in enumerate(post_supports)
    )
    unique_tracks = [
        track_id for track_id in sorted(track_by_id) if np.any(core_owner == track_id)
    ]
    unique_rows = tuple(
        _support_row(
            index,
            np.flatnonzero(core_owner == track_id),
            track_id=int(track_id),
        )
        for index, track_id in enumerate(unique_tracks)
    )
    final_rows = tuple(
        _support_row(
            candidate_id,
            bank.full_ids[candidate_id],
        )
        for candidate_id in range(len(bank.candidates))
    )
    output: dict[str, tuple[Mapping[str, Any], ...]] = {
        "single_full": tuple(
            _support_row(index, row.full_ids, fragment_id=int(row.fragment_id))
            for index, row in enumerate([row for row in fragments if len(row.full_ids)])
        ),
        "single_core": tuple(
            _support_row(index, row.core_ids, fragment_id=int(row.fragment_id))
            for index, row in enumerate([row for row in fragments if len(row.core_ids)])
        ),
        "component_full_union": component_full,
        "component_core_union": component_core,
        "pre_conflict": pre_conflict,
        "post_conflict": post_conflict,
        "unique_ownership": unique_rows,
        "final_candidate": final_rows,
    }
    if tuple(output) != _FUNNEL_STAGES:
        raise AssertionError("V10 funnel stage order changed")
    return output


def build_v10_object_bank(
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    condition: str,
) -> dict[str, Any]:
    """Adapt one immutable V9 lifting bank to the V10 runner protocol."""

    condition_map: dict[str, tuple[PairMode, ReconstructionMode, bool]] = {
        "P0R0": ("P0", "R0", False),
        "P1R0": ("P1", "R0", False),
        "P0R1": ("P0", "R1", False),
        "P1R1": ("P1", "R1", False),
        "VC1": ("P1", "R1", True),
    }
    if condition not in condition_map:
        raise ValueError(f"unknown V10 structure condition: {condition}")
    pair_mode, reconstruction_mode, view_consensus = condition_map[condition]
    point_count = int(metadata["point_count"])
    frame_count = int(metadata["frame_count"])
    classes = tuple(map(str, metadata["classes"]))
    fragments = _lifting_fragments(arrays)
    frames = _lifting_frames(arrays, fragments, frame_count)
    config = V10Config()
    association = associate_fragments_v10(
        fragments,
        frames,
        pair_mode,
        config,
        view_consensus=view_consensus,
    )
    multiview = _multiview_classifications(
        association, fragments, arrays, classes
    )
    codebook = _codebook_classifications(
        association,
        fragments,
        frames,
        point_count,
        np.asarray(arrays["semantic"]),
        np.asarray(arrays["label_features"]),
        classes,
        config,
        core_from_full=reconstruction_mode == "R1",
    )
    association, bank = build_v10_candidate_bank(
        fragments,
        frames,
        point_count,
        pair_mode=pair_mode,
        reconstruction_mode=reconstruction_mode,
        classifications=multiview,
        xyz_m=np.asarray(arrays["xyz_m"]),
        affinity=np.asarray(arrays["affinity"]),
        config=config,
        view_consensus=view_consensus,
        frozen_association=association,
    )
    candidate_rows: list[dict[str, Any]] = []
    for row in bank.candidates:
        enriched = dict(row)
        enriched["structure_condition"] = condition
        selected = multiview.get(int(row["track_id"]))
        alternative = codebook.get(int(row["track_id"]))
        geometry_product = float(
            np.prod(
                np.clip(
                    [
                        float(row["component_consensus"]),
                        float(row["mean_core_membership"]),
                        float(row["internal_affinity"]),
                        min(len(row["frame_ids"]), 5) / 5.0,
                        1.0 - float(row["conflict_ratio"]),
                    ],
                    0.0,
                    1.0,
                )
            )
        )

        def classifier_payload(
            classification: TrackClassification | None,
        ) -> dict[str, Any]:
            semantic_ratio = (
                float(classification.semantic_ratio) if classification else 0.0
            )
            return {
                "branch_class": (
                    classification.class_name
                    if classification is not None and classification.class_name
                    else "__unknown__"
                ),
                "class_id": int(classification.class_id) if classification else -1,
                "semantic_ratio": semantic_ratio,
                "semantic_margin": (
                    float(classification.semantic_margin) if classification else 0.0
                ),
                "effective_views": (
                    int(classification.effective_view_count) if classification else 0
                ),
                "classification_eligible": (
                    bool(classification.eligible) if classification else False
                ),
                "base_score": float(
                    max(geometry_product * semantic_ratio, 0.0) ** (1.0 / 6.0)
                ),
            }

        enriched.update(
            {
                "codebook_class": (
                    alternative.class_name
                    if alternative is not None and alternative.class_name
                    else "__unknown__"
                ),
                "codebook_class_id": int(alternative.class_id) if alternative else -1,
                "codebook_semantic_ratio": (
                    float(alternative.semantic_ratio) if alternative else 0.0
                ),
                "codebook_eligible": bool(alternative.eligible) if alternative else False,
                "classifiers": {
                    "mv-label": classifier_payload(selected),
                    "codebook": classifier_payload(alternative),
                },
            }
        )
        candidate_rows.append(enriched)
    bank = V10CandidateBank(
        bank.point_count,
        bank.pair_mode,
        bank.reconstruction_mode,
        bank.point_candidate_id,
        bank.full_ids,
        bank.core_ids,
        tuple(candidate_rows),
    )
    accepted_edges = tuple(_edge_row(edge) for edge in association.accepted_edges)
    stages = _stage_supports(
        association,
        bank,
        fragments,
        frames,
        config,
    )
    return {
        "point_count": point_count,
        "fragments": tuple(
            {
                "fragment_id": int(row.fragment_id),
                "frame_id": int(row.frame_id),
                "mask_index": int(row.mask_index),
                "full_point_count": int(len(row.full_ids)),
                "core_point_count": int(len(row.core_ids)),
                "conflict_ratio": (
                    None if row.conflict_ratio is None else float(row.conflict_ratio)
                ),
            }
            for row in fragments
        ),
        "tracks": tuple(
            {
                "track_id": int(row.track_id),
                "fragment_ids": list(row.fragment_ids),
                "frame_ids": list(row.frame_ids),
                "component_consensus": float(row.component_consensus),
                "pair_mode": row.pair_mode,
            }
            for row in association.tracks
        ),
        "candidates": bank.candidates,
        "full_ids": bank.full_ids,
        "core_ids": bank.core_ids,
        "accepted_edges": accepted_edges,
        "stage_supports": stages,
        "diagnostics": {
            "condition": condition,
            "pair_mode": pair_mode,
            "reconstruction_mode": reconstruction_mode,
            "view_consensus": view_consensus,
            "frame_pair_count": len(association.frame_pairs),
            "tentative_edge_count": len(association.tentative_edges),
            "accepted_edge_count": len(accepted_edges),
            "track_count": len(association.tracks),
            "candidate_count": len(bank.candidates),
            "selected_classifier": "unselected",
            "stage_counts": {
                stage: len(rows) for stage, rows in stages.items()
            },
        },
    }


__all__ = [
    "PairMode",
    "ReconstructionMode",
    "V10Association",
    "V10CandidateBank",
    "V10Config",
    "V10MatchEdge",
    "V10PairEvidence",
    "V10Track",
    "associate_fragments_v10",
    "build_v10_candidate_bank",
    "build_v10_object_bank",
    "frame_weighted_jaccard",
    "pair_evidence",
    "select_covisible_frame_pairs",
]
