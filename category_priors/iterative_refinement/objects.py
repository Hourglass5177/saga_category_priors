from __future__ import annotations

"""Non-transitive object merge/split decisions and auditable lineage."""

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from ..geometry import pca_sorted_extents_m
from .contracts import (
    CandidateSeed,
    GaussianEvidence,
    LineageRecord,
    MaskHypothesis,
    ObjectState,
    RefinementConfig,
)
from .local_refine import SizePrior, object_components
from .views import observations_are_independent


def mask_iou(first: MaskHypothesis, second: MaskHypothesis) -> float:
    if first.mask_shape != second.mask_shape:
        raise ValueError("same-camera masks must share image geometry")
    a, b = first.unpack_mask(), second.unpack_mask()
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def _minimum_distance(left: np.ndarray, right: np.ndarray, xyz: np.ndarray) -> float:
    if not len(left) or not len(right):
        return float("inf")
    tree = cKDTree(xyz[left])
    distance, _ = tree.query(xyz[right], k=1)
    return float(np.min(distance))


def _state_lookup(state: ObjectState) -> tuple[dict[int, float], dict[int, float]]:
    hard = {int(point): float(value) for point, value in zip(state.point_ids, state.hard_positive_counts)}
    margin = {int(point): float(value) for point, value in zip(state.point_ids, state.evidence_margin)}
    return hard, margin


def combine_states(left: ObjectState, right: ObjectState, *, round_index: int) -> ObjectState:
    points = np.union1d(left.point_ids, right.point_ids)
    left_hard, left_margin = _state_lookup(left)
    right_hard, right_margin = _state_lookup(right)
    hard_counts = np.asarray(
        [max(left_hard.get(int(point), 0.0), right_hard.get(int(point), 0.0)) for point in points],
        dtype=np.float64,
    )
    margins = np.asarray(
        [max(left_margin.get(int(point), -np.inf), right_margin.get(int(point), -np.inf)) for point in points],
        dtype=np.float64,
    )
    margins[~np.isfinite(margins)] = 0.0
    reliable = left.reliable_review_class and right.reliable_review_class
    review_class = left.review_class if reliable and left.review_class == right.review_class else None
    return ObjectState(
        object_id=min(left.object_id, right.object_id),
        parent_candidate_ids=tuple(sorted(set(left.parent_candidate_ids) | set(right.parent_candidate_ids))),
        point_ids=points,
        anchor_ids=np.union1d(left.anchor_ids, right.anchor_ids),
        hard_positive_ids=points[hard_counts >= 2],
        hard_positive_counts=hard_counts,
        evidence_margin=margins,
        review_class=review_class,
        reliable_review_class=bool(review_class),
        round_index=int(round_index),
        changed=True,
    )


def _merge_support(
    left_id: int,
    right_id: int,
    hypotheses: Mapping[int, Sequence[MaskHypothesis]],
    independent_pairs: Mapping[tuple[int, int], bool],
    config: RefinementConfig,
) -> tuple[int, float]:
    # The two masks must agree *in the same camera*.  The evidence becomes
    # multi-view only when this occurs in at least two independent cameras.
    matches: list[tuple[int, float]] = []
    left_by_camera = {row.camera_index: row for row in hypotheses.get(left_id, ())}
    right_by_camera = {row.camera_index: row for row in hypotheses.get(right_id, ())}
    for camera in sorted(set(left_by_camera) & set(right_by_camera)):
        a, b = left_by_camera[camera], right_by_camera[camera]
        value = mask_iou(a, b)
        if (
            value >= config.merge_mask_iou_min
            and a.seed_coverage >= config.merge_seed_coverage_min
            and b.seed_coverage >= config.merge_seed_coverage_min
        ):
            matches.append((camera, value))
    if len(matches) < 2:
        return 0, 0.0
    independent = False
    for index, (left_camera, _) in enumerate(matches):
        for right_camera, _ in matches[index + 1 :]:
            if independent_pairs.get(tuple(sorted((left_camera, right_camera))), False):
                independent = True
                break
    return (len(matches), float(np.mean([row[1] for row in matches]))) if independent else (0, 0.0)


def merge_objects_once(
    states: Sequence[ObjectState],
    *,
    hypotheses: Mapping[int, Sequence[MaskHypothesis]],
    evidence: Mapping[int, GaussianEvidence],
    independent_pairs: Mapping[tuple[int, int], bool],
    xyz_m: np.ndarray,
    prior_by_object: Mapping[int, SizePrior],
    round_index: int,
    config: RefinementConfig = RefinementConfig(),
) -> tuple[tuple[ObjectState, ...], tuple[LineageRecord, ...]]:
    """Merge mutual-best pairs once; never apply a transitive closure."""

    xyz = np.asarray(xyz_m, dtype=np.float64)
    by_id = {row.object_id: row for row in states}
    scores: dict[tuple[int, int], tuple[int, float]] = {}
    for left_index, left in enumerate(sorted(states, key=lambda row: row.object_id)):
        for right in sorted(states, key=lambda row: row.object_id)[left_index + 1 :]:
            if (
                left.reliable_review_class
                and right.reliable_review_class
                and left.review_class != right.review_class
            ):
                continue
            support = _merge_support(left.object_id, right.object_id, hypotheses, independent_pairs, config)
            if support[0] < 2 or _minimum_distance(left.point_ids, right.point_ids, xyz) > config.merge_distance_m:
                continue
            # A two-view hard exclusion of the other object's points vetoes.
            left_ev, right_ev = evidence.get(left.object_id), evidence.get(right.object_id)
            veto = False
            if left_ev is not None:
                veto |= bool(np.any((left_ev.hard_negative_views >= 2) & np.isin(left_ev.point_ids, right.point_ids)))
            if right_ev is not None:
                veto |= bool(np.any((right_ev.hard_negative_views >= 2) & np.isin(right_ev.point_ids, left.point_ids)))
            if veto:
                continue
            merged_points = np.union1d(left.point_ids, right.point_ids)
            limit = prior_by_object[left.object_id].extents_q95_m
            extents = pca_sorted_extents_m(xyz[merged_points], 1.0)
            if np.count_nonzero(extents > np.asarray(limit)) >= 2 or extents[-1] > 1.25 * limit[-1]:
                continue
            scores[(left.object_id, right.object_id)] = support
    best: dict[int, int] = {}
    for object_id in sorted(by_id):
        choices = []
        for pair, value in scores.items():
            if object_id in pair:
                other = pair[1] if pair[0] == object_id else pair[0]
                choices.append((value[0], value[1], -other, other))
        if choices:
            best[object_id] = max(choices)[3]
    used: set[int] = set()
    output: list[ObjectState] = []
    lineage: list[LineageRecord] = []
    for object_id in sorted(by_id):
        if object_id in used:
            continue
        other = best.get(object_id)
        if other is not None and best.get(other) == object_id and other not in used:
            combined = combine_states(by_id[object_id], by_id[other], round_index=round_index)
            output.append(combined)
            used.update((object_id, other))
            lineage.append(
                LineageRecord(
                    node_id=f"r{round_index}:merge:{combined.object_id}",
                    parent_node_ids=(f"r{round_index}:refine:{object_id}", f"r{round_index}:refine:{other}"),
                    candidate_ids=combined.parent_candidate_ids,
                    affected_b0_ids=(),
                    round_index=round_index,
                    operation="merge",
                    added_point_ids=(),
                    removed_point_ids=(),
                    hypothesis_ids=tuple(
                        sorted(
                            [row.hypothesis_id for row in hypotheses.get(object_id, ())]
                            + [row.hypothesis_id for row in hypotheses.get(other, ())]
                        )
                    ),
                )
            )
        else:
            output.append(by_id[object_id])
            used.add(object_id)
    return tuple(sorted(output, key=lambda row: row.object_id)), tuple(lineage)


def split_disconnected_objects(
    states: Sequence[ObjectState],
    *,
    seed_by_parent: Mapping[int, CandidateSeed],
    xyz_m: np.ndarray,
    round_index: int,
    config: RefinementConfig = RefinementConfig(),
) -> tuple[tuple[ObjectState, ...], tuple[LineageRecord, ...]]:
    output: list[ObjectState] = []
    lineage: list[LineageRecord] = []
    next_id = max((row.object_id for row in states), default=-1) + 1
    for state in sorted(states, key=lambda row: row.object_id):
        components = object_components(state, xyz_m, config)
        if len(components) <= 1:
            output.append(state)
            continue
        anchor_rows = [
            seed_by_parent[parent].seed_anchor
            for parent in state.parent_candidate_ids
            if parent in seed_by_parent and len(seed_by_parent[parent].seed_anchor)
        ]
        anchors = np.unique(np.concatenate(anchor_rows)) if anchor_rows else np.empty(0, dtype=np.int64)
        main_index = max(
            range(len(components)),
            key=lambda index: (int(np.count_nonzero(np.isin(components[index], anchors))), len(components[index]), -int(components[index].min())),
        )
        hard_map, margin_map = _state_lookup(state)
        kept: list[ObjectState] = []
        for index, component in enumerate(components):
            hard_counts = np.asarray([hard_map[int(point)] for point in component])
            if index != main_index and not np.any(hard_counts >= 2):
                continue
            object_id = state.object_id if index == main_index else next_id
            if index != main_index:
                next_id += 1
            margins = np.asarray([margin_map[int(point)] for point in component])
            kept.append(
                replace(
                    state,
                    object_id=object_id,
                    point_ids=component,
                    anchor_ids=np.intersect1d(state.anchor_ids, component, assume_unique=True),
                    hard_positive_ids=component[hard_counts >= 2],
                    hard_positive_counts=hard_counts,
                    evidence_margin=margins,
                    changed=True,
                )
            )
        output.extend(kept)
        lineage.append(
            LineageRecord(
                node_id=f"r{round_index}:split:{state.object_id}",
                parent_node_ids=(f"r{round_index}:refine:{state.object_id}",),
                candidate_ids=state.parent_candidate_ids,
                affected_b0_ids=(),
                round_index=round_index,
                operation="split",
                added_point_ids=(),
                removed_point_ids=tuple(
                    sorted(set(state.point_ids.tolist()) - set(np.concatenate([row.point_ids for row in kept]).tolist()))
                ),
                hypothesis_ids=(),
            )
        )
    return tuple(sorted(output, key=lambda row: row.object_id)), tuple(lineage)


__all__ = [
    "combine_states",
    "mask_iou",
    "merge_objects_once",
    "split_disconnected_objects",
]
