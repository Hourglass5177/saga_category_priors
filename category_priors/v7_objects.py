from __future__ import annotations

"""Deterministic cross-view object construction for the V7 experiment.

This module deliberately has no renderer, ground-truth, or legacy-postprocess
dependency.  A frame is lifted into sparse Gaussian fragments, fragments are
associated without semantic routing, and semantic labels are attached only
after a unique multiview core has been formed.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree


SAGA20 = frozenset(
    {
        "chair", "table", "plant", "tv", "painting", "sofa", "cabinet",
        "bed", "socket", "book", "switch", "door", "window", "lamp",
        "speaker", "fan", "refrigerator", "cup", "phone", "trash can",
    }
)


@dataclass(frozen=True)
class V7Config:
    fragment_min_core: int = 5
    fragment_min_full: int = 10
    fragment_core_pixels: int = 2
    fragment_core_visible_fraction: float = 0.50
    track_min_shared_core: int = 3
    track_min_overlap: float = 0.25
    track_min_margin: float = 0.10
    track_min_frames: int = 2
    core_min_positive_views: int = 2
    core_min_positive_ratio: float = 0.60
    core_max_conflict_ratio: float = 0.25
    core_min_points: int = 10
    halo_min_positive_views: int = 1
    halo_min_positive_ratio: float = 0.40
    halo_max_conflict_ratio: float = 0.25
    halo_radius_m: float = 0.05
    halo_min_anchors: int = 3
    halo_min_affinity: float = 0.95
    halo_min_margin: float = 0.02
    semantic_min_views: int = 2
    semantic_min_ratio: float = 0.60
    semantic_min_margin: float = 0.10

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    frame_id: int
    mask_index: int
    class_id: int
    full_ids: np.ndarray
    core_ids: np.ndarray
    full_support: np.ndarray
    core_support: np.ndarray


@dataclass(frozen=True)
class FrameEvidence:
    frame_id: int
    fragments: tuple[Fragment, ...]
    visible_ids: np.ndarray
    background_ids: np.ndarray


@dataclass
class Track:
    track_id: int
    fragment_ids: list[int] = field(default_factory=list)
    frame_ids: set[int] = field(default_factory=set)
    core_union: set[int] = field(default_factory=set)
    full_union: set[int] = field(default_factory=set)
    overlaps: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class CoreAssignment:
    core_track_id: np.ndarray
    positive_views: np.ndarray
    visible_views: np.ndarray
    background_views: np.ndarray
    conflict_views: np.ndarray
    assignment_margin: np.ndarray
    valid_track_ids: tuple[int, ...]
    track_positive: dict[int, np.ndarray]
    track_conflict: dict[int, np.ndarray]


def _flat_image(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.reshape(-1)


def lift_frame(
    contributor_id: np.ndarray,
    contributor_weight: np.ndarray,
    masks: np.ndarray,
    class_ids: Sequence[int],
    frame_id: int,
    point_count: int,
    *,
    fragment_id_start: int = 0,
    config: V7Config = V7Config(),
) -> FrameEvidence:
    """Lift one frame's masks into sparse Gaussian fragments.

    Repeated contributor IDs are accumulated with ``bincount``.  Pixels with
    zero contribution or invalid IDs never enter either positive or background
    evidence.
    """
    ids = np.asarray(contributor_id)
    weights = np.asarray(contributor_weight, dtype=np.float64)
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("contributor_id and contributor_weight must be H x W")
    mask_array = np.asarray(masks, dtype=bool)
    if mask_array.ndim != 3 or mask_array.shape[1:] != ids.shape:
        raise ValueError("masks must be M x H x W and match contributor images")
    classes = np.asarray(class_ids, dtype=np.int64)
    if len(classes) != len(mask_array):
        raise ValueError("class_ids must have one entry per mask")
    if point_count <= 0:
        raise ValueError("point_count must be positive")

    flat_ids = ids.reshape(-1).astype(np.int64, copy=False)
    flat_weights = weights.reshape(-1)
    valid = (
        (flat_ids >= 0)
        & (flat_ids < int(point_count))
        & np.isfinite(flat_weights)
        & (flat_weights > 0)
    )
    valid_ids = flat_ids[valid]
    visible_pixels = np.bincount(valid_ids, minlength=point_count)
    visible_ids = np.flatnonzero(visible_pixels).astype(np.int32)
    union = np.any(mask_array, axis=0).reshape(-1) if len(mask_array) else np.zeros(ids.size, dtype=bool)
    background_pixels = np.bincount(
        flat_ids[valid & ~union], minlength=point_count
    )
    background_ids = np.flatnonzero(background_pixels).astype(np.int32)

    fragments: list[Fragment] = []
    next_id = int(fragment_id_start)
    for mask_index, (mask, class_id) in enumerate(zip(mask_array, classes)):
        selected = valid & mask.reshape(-1)
        if not np.any(selected):
            continue
        selected_ids = flat_ids[selected]
        selected_weights = flat_weights[selected]
        pixel_counts = np.bincount(selected_ids, minlength=point_count)
        weight_sums = np.bincount(
            selected_ids, weights=selected_weights, minlength=point_count
        )
        full_ids = np.flatnonzero(pixel_counts).astype(np.int32)
        if len(full_ids) < int(config.fragment_min_full):
            continue
        full_fraction = pixel_counts[full_ids] / np.maximum(visible_pixels[full_ids], 1)
        core_keep = (
            (pixel_counts[full_ids] >= int(config.fragment_core_pixels))
            & (full_fraction >= float(config.fragment_core_visible_fraction))
        )
        core_ids = full_ids[core_keep]
        if len(core_ids) < int(config.fragment_min_core):
            continue
        fragments.append(
            Fragment(
                fragment_id=next_id,
                frame_id=int(frame_id),
                mask_index=int(mask_index),
                class_id=int(class_id),
                full_ids=full_ids,
                core_ids=core_ids,
                full_support=weight_sums[full_ids].astype(np.float32),
                core_support=weight_sums[core_ids].astype(np.float32),
            )
        )
        next_id += 1
    return FrameEvidence(
        frame_id=int(frame_id),
        fragments=tuple(fragments),
        visible_ids=visible_ids,
        background_ids=background_ids,
    )


def _overlap_coefficient(left: set[int], right: set[int]) -> tuple[int, float]:
    if not left or not right:
        return 0, 0.0
    shared = len(left.intersection(right))
    return shared, shared / min(len(left), len(right))


def associate_fragments(
    fragments: Sequence[Fragment],
    config: V7Config = V7Config(),
) -> list[Track]:
    """Associate fragments without using their semantic labels."""
    tracks: list[Track] = []
    for fragment in sorted(
        fragments, key=lambda item: (item.frame_id, item.mask_index, item.fragment_id)
    ):
        fragment_core = set(map(int, fragment.core_ids))
        scored: list[tuple[float, int, int]] = []
        for track in tracks:
            if fragment.frame_id in track.frame_ids:
                continue
            shared, overlap = _overlap_coefficient(fragment_core, track.core_union)
            if (
                shared >= int(config.track_min_shared_core)
                and overlap >= float(config.track_min_overlap)
            ):
                scored.append((float(overlap), int(shared), track.track_id))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        best_track: Track | None = None
        if scored:
            second = scored[1][0] if len(scored) > 1 else 0.0
            if scored[0][0] - second >= float(config.track_min_margin):
                best_track = tracks[scored[0][2]]
        if best_track is None:
            best_track = Track(track_id=len(tracks))
            tracks.append(best_track)
        else:
            best_track.overlaps.append(float(scored[0][0]))
        best_track.fragment_ids.append(int(fragment.fragment_id))
        best_track.frame_ids.add(int(fragment.frame_id))
        best_track.core_union.update(fragment_core)
        best_track.full_union.update(map(int, fragment.full_ids))
    return tracks


def build_consensus_core(
    tracks: Sequence[Track],
    fragments: Sequence[Fragment],
    frame_evidence: Sequence[FrameEvidence],
    point_count: int,
    config: V7Config = V7Config(),
) -> CoreAssignment:
    """Build a unique multiview core for every retained track."""
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    visible = np.zeros(point_count, dtype=np.int16)
    background = np.zeros(point_count, dtype=np.int16)
    for frame in frame_evidence:
        visible[frame.visible_ids] += 1
        background[frame.background_ids] += 1

    point_track_memberships: list[list[tuple[int, int]]] = [list() for _ in range(point_count)]
    track_positive: dict[int, np.ndarray] = {}
    for track in tracks:
        positive = np.zeros(point_count, dtype=np.int16)
        for fragment_id in track.fragment_ids:
            fragment = fragment_by_id[fragment_id]
            positive[fragment.full_ids] += 1
        track_positive[track.track_id] = positive
        for point_id in np.flatnonzero(positive):
            point_track_memberships[int(point_id)].append(
                (track.track_id, int(positive[point_id]))
            )

    track_conflict: dict[int, np.ndarray] = {}
    total_fragment_membership = np.zeros(point_count, dtype=np.int16)
    for positive in track_positive.values():
        total_fragment_membership += positive
    for track_id, positive in track_positive.items():
        track_conflict[track_id] = np.maximum(
            total_fragment_membership - positive, 0
        ).astype(np.int16)

    candidates: dict[int, list[tuple[float, int, int]]] = {}
    for track in tracks:
        if len(track.frame_ids) < int(config.track_min_frames):
            continue
        positive = track_positive[track.track_id]
        conflict = track_conflict[track.track_id]
        members = np.flatnonzero(positive)
        if not len(members):
            continue
        positive_ratio = positive[members] / np.maximum(visible[members], 1)
        conflict_ratio = conflict[members] / np.maximum(visible[members], 1)
        keep = (
            (positive[members] >= int(config.core_min_positive_views))
            & (positive_ratio >= float(config.core_min_positive_ratio))
            & (conflict_ratio <= float(config.core_max_conflict_ratio))
        )
        for point_id, ratio, count in zip(
            members[keep], positive_ratio[keep], positive[members][keep]
        ):
            candidates.setdefault(int(point_id), []).append(
                (float(ratio), int(count), track.track_id)
            )

    labels = np.full(point_count, -1, dtype=np.int32)
    margins = np.zeros(point_count, dtype=np.float32)
    for point_id, choices in candidates.items():
        choices.sort(key=lambda item: (-item[0], -item[1], item[2]))
        labels[point_id] = choices[0][2]
        runner_up = choices[1][0] if len(choices) > 1 else 0.0
        margins[point_id] = float(choices[0][0] - runner_up)

    valid_tracks: list[int] = []
    for track in tracks:
        if np.count_nonzero(labels == track.track_id) >= int(config.core_min_points):
            valid_tracks.append(track.track_id)
        else:
            labels[labels == track.track_id] = -1
    assigned_positive = np.zeros(point_count, dtype=np.int16)
    assigned_conflict = np.zeros(point_count, dtype=np.int16)
    for track_id in valid_tracks:
        mask = labels == track_id
        assigned_positive[mask] = track_positive[track_id][mask]
        assigned_conflict[mask] = track_conflict[track_id][mask]
    return CoreAssignment(
        core_track_id=labels,
        positive_views=assigned_positive,
        visible_views=visible,
        background_views=background,
        conflict_views=assigned_conflict,
        assignment_margin=margins,
        valid_track_ids=tuple(valid_tracks),
        track_positive=track_positive,
        track_conflict=track_conflict,
    )


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("features must be a matrix")
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def attach_unique_halo(
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    tracks: Sequence[Track],
    core: CoreAssignment,
    config: V7Config = V7Config(),
) -> np.ndarray:
    """Attach a one-shot local halo to unique high-confidence cores."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity)
    if xyz.shape != (len(core.core_track_id), 3) or len(features) != len(xyz):
        raise ValueError("xyz, affinity, and core assignment must have matching rows")
    result = np.asarray(core.core_track_id, dtype=np.int32).copy()
    track_by_id = {track.track_id: track for track in tracks}
    candidate_scores: dict[int, list[tuple[float, int]]] = {}
    for track_id in core.valid_track_ids:
        anchor_ids = np.flatnonzero(core.core_track_id == track_id)
        if len(anchor_ids) < int(config.halo_min_anchors):
            continue
        positive = core.track_positive[track_id]
        conflict = core.track_conflict[track_id]
        union = np.fromiter(track_by_id[track_id].full_union, dtype=np.int64)
        union = union[result[union] < 0]
        if not len(union):
            continue
        visible = np.maximum(core.visible_views[union], 1)
        keep = (
            (positive[union] >= int(config.halo_min_positive_views))
            & (positive[union] / visible >= float(config.halo_min_positive_ratio))
            & (conflict[union] / visible <= float(config.halo_max_conflict_ratio))
        )
        union = union[keep]
        if not len(union):
            continue
        tree = cKDTree(xyz[anchor_ids])
        k = min(int(config.halo_min_anchors), len(anchor_ids))
        distances, local_indices = tree.query(
            xyz[union], k=k, distance_upper_bound=float(config.halo_radius_m)
        )
        distances = np.asarray(distances)
        local_indices = np.asarray(local_indices)
        if k == 1:
            distances = distances[:, None]
            local_indices = local_indices[:, None]
        spatial = np.all(np.isfinite(distances), axis=1)
        for point_id, neighbor_rows in zip(union[spatial], local_indices[spatial]):
            neighbors = anchor_ids[np.asarray(neighbor_rows, dtype=np.int64)]
            prototype = features[neighbors].mean(axis=0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            similarity = float(features[point_id] @ prototype)
            if similarity >= float(config.halo_min_affinity):
                candidate_scores.setdefault(int(point_id), []).append(
                    (similarity, track_id)
                )
    for point_id, choices in candidate_scores.items():
        choices.sort(key=lambda item: (-item[0], item[1]))
        runner_up = choices[1][0] if len(choices) > 1 else -1.0
        if choices[0][0] - runner_up >= float(config.halo_min_margin):
            result[point_id] = choices[0][1]
    return result


def attach_local_labels(
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    labels: np.ndarray,
    *,
    radius_m: float = 0.05,
    min_anchors: int = 3,
    affinity_threshold: float = 0.95,
    margin: float = 0.02,
) -> np.ndarray:
    """One-shot local attach used only by the registered L3 causal ablation."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = _normalise_rows(affinity)
    result = np.asarray(labels, dtype=np.int32).copy()
    anchors = np.flatnonzero(result >= 0)
    queries = np.flatnonzero(result < 0)
    if len(anchors) < min_anchors or not len(queries):
        return result
    k = min(12, len(anchors))
    distances, positions = cKDTree(xyz[anchors]).query(
        xyz[queries], k=k, distance_upper_bound=float(radius_m)
    )
    if k == 1:
        distances = distances[:, None]
        positions = positions[:, None]
    for point_id, point_distances, point_positions in zip(queries, distances, positions):
        valid = np.isfinite(point_distances) & (point_positions < len(anchors))
        neighbor_ids = anchors[np.asarray(point_positions[valid], dtype=np.int64)]
        choices: list[tuple[float, int]] = []
        for label in np.unique(result[neighbor_ids]):
            same = neighbor_ids[result[neighbor_ids] == label]
            if len(same) < int(min_anchors):
                continue
            prototype = features[same].mean(axis=0)
            prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
            similarity = float(features[point_id] @ prototype)
            if similarity >= float(affinity_threshold):
                choices.append((similarity, int(label)))
        choices.sort(key=lambda item: (-item[0], item[1]))
        if choices:
            runner_up = choices[1][0] if len(choices) > 1 else -1.0
            if choices[0][0] - runner_up >= float(margin):
                result[point_id] = choices[0][1]
    return result


def materialize_instances(
    xyz_m: np.ndarray,
    tracks: Sequence[Track],
    fragments: Sequence[Fragment],
    core: CoreAssignment,
    final_track_id: np.ndarray,
    class_names: Sequence[str],
    config: V7Config = V7Config(),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Attach semantics after tracking and return dense candidate labels."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    final_tracks = np.asarray(final_track_id, dtype=np.int32)
    fragment_by_id = {fragment.fragment_id: fragment for fragment in fragments}
    output = np.full(len(final_tracks), -1, dtype=np.int32)
    candidates: list[dict[str, Any]] = []
    for track in tracks:
        if track.track_id not in core.valid_track_ids:
            continue
        votes = np.zeros(len(class_names), dtype=np.int32)
        for fragment_id in track.fragment_ids:
            class_id = int(fragment_by_id[fragment_id].class_id)
            if 0 <= class_id < len(votes):
                votes[class_id] += 1
        total = int(votes.sum())
        if total < int(config.semantic_min_views):
            continue
        order = np.argsort(-votes, kind="stable")
        winner = int(order[0])
        runner_up = int(votes[order[1]]) if len(order) > 1 else 0
        ratio = float(votes[winner] / total)
        margin = float((votes[winner] - runner_up) / total)
        class_name = str(class_names[winner])
        if (
            class_name not in SAGA20
            or ratio < float(config.semantic_min_ratio)
            or margin < float(config.semantic_min_margin)
        ):
            continue
        member_ids = np.flatnonzero(final_tracks == track.track_id)
        core_ids = np.flatnonzero(core.core_track_id == track.track_id)
        if not len(member_ids) or len(core_ids) < int(config.core_min_points):
            continue
        positive_ratio = core.positive_views[core_ids] / np.maximum(
            core.visible_views[core_ids], 1
        )
        conflict_ratio = core.conflict_views[core_ids] / np.maximum(
            core.visible_views[core_ids], 1
        )
        median_overlap = float(np.median(track.overlaps)) if track.overlaps else 0.0
        view_term = min(len(track.frame_ids), 5) / 5.0
        q = float(
            ratio
            * float(np.mean(positive_ratio))
            * (0.5 + 0.5 * median_overlap)
            * (0.5 + 0.5 * view_term)
            * (1.0 - float(np.mean(conflict_ratio)))
        )
        extents = np.ptp(xyz[member_ids], axis=0)
        local_k = min(16, max(len(xyz) - 1, 0))
        if local_k:
            distances, _ = cKDTree(xyz).query(xyz[core_ids], k=local_k + 1)
            radii = np.maximum(np.asarray(distances)[:, -1], 1e-6)
            density = float(np.median(local_k / (np.pi * radii * radii)))
        else:
            density = 0.0
        candidate_id = len(candidates)
        output[member_ids] = candidate_id
        candidates.append(
            {
                "candidate_id": candidate_id,
                "track_id": int(track.track_id),
                "branch_class": class_name,
                "class_id": winner,
                "full_point_count": int(len(member_ids)),
                "core_point_count": int(len(core_ids)),
                "halo_point_count": int(len(member_ids) - len(core_ids)),
                "effective_view_count": int(len(track.frame_ids)),
                "semantic_ratio": ratio,
                "semantic_margin": margin,
                "mean_core_positive_ratio": float(np.mean(positive_ratio)),
                "conflict_ratio": float(np.mean(conflict_ratio)),
                "median_track_overlap": median_overlap,
                "base_score": float(np.clip(q, 0.0, 1.0)),
                "metric_extent_xyz_m": [float(value) for value in extents],
                "metric_extents_m": [float(value) for value in np.sort(extents)],
                "local_surface_density": density,
            }
        )
    return output, candidates
