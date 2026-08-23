from __future__ import annotations

"""Pure, deterministic object construction for the V8 experiment.

The module intentionally has no renderer, ground-truth, or legacy postprocess
dependency.  Fragments are associated without semantic labels, physical views
are counted at most once, and category labels are attached only after a unique
multiview core/full assignment has been frozen.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
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


@dataclass(frozen=True)
class V8Config:
    track_min_shared_core: int = 3
    track_min_overlap: float = 0.25
    track_min_margin: float = 0.10
    track_min_frames: int = 2
    core_min_positive_views: int = 2
    core_min_positive_ratio: float = 0.60
    core_max_conflict_ratio: float = 0.25
    core_min_points: int = 10
    full_min_positive_mass_ratio: float = 0.40
    mv_label_min_iou: float = 0.25
    local_density_neighbors: int = 16

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_sparse(
    ids: np.ndarray,
    mass: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    point_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    if len(point_ids) != len(values):
        raise ValueError(f"{name} ids and mass must have the same length")
    if np.any(point_ids < 0):
        raise ValueError(f"{name} ids must be non-negative")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{name} mass must be finite and non-negative")
    if not len(point_ids):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    unique, inverse = np.unique(point_ids, return_inverse=True)
    accumulated = np.zeros(len(unique), dtype=np.float64)
    np.add.at(accumulated, inverse, values)
    return unique.astype(np.int32), accumulated.astype(np.float32)


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    frame_id: int
    mask_index: int
    full_ids: np.ndarray
    core_ids: np.ndarray
    full_mass: np.ndarray
    core_mass: np.ndarray

    def __post_init__(self) -> None:
        full_ids, full_mass = _canonical_sparse(
            self.full_ids, self.full_mass, name="fragment full"
        )
        core_ids, core_mass = _canonical_sparse(
            self.core_ids, self.core_mass, name="fragment core"
        )
        if not np.all(np.isin(core_ids, full_ids, assume_unique=True)):
            raise ValueError("fragment core_ids must be a subset of full_ids")
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
    grounded_missing: bool = False

    def __post_init__(self) -> None:
        visible_ids, visible_mass = _canonical_sparse(
            self.visible_ids, self.visible_mass, name="frame visible"
        )
        for fragment in self.fragments:
            if int(fragment.frame_id) != int(self.frame_id):
                raise ValueError("all frame fragments must share FrameEvidence.frame_id")
        object.__setattr__(self, "fragments", tuple(self.fragments))
        object.__setattr__(self, "visible_ids", visible_ids)
        object.__setattr__(self, "visible_mass", visible_mass)


def _sparse_max_union(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(left_ids):
        return np.asarray(right_ids, dtype=np.int32).copy(), np.asarray(
            right_mass, dtype=np.float32
        ).copy()
    if not len(right_ids):
        return np.asarray(left_ids, dtype=np.int32).copy(), np.asarray(
            left_mass, dtype=np.float32
        ).copy()
    ids = np.union1d(left_ids, right_ids).astype(np.int32)
    mass = np.zeros(len(ids), dtype=np.float32)
    left_pos = np.searchsorted(ids, left_ids)
    right_pos = np.searchsorted(ids, right_ids)
    mass[left_pos] = np.maximum(mass[left_pos], left_mass)
    mass[right_pos] = np.maximum(mass[right_pos], right_mass)
    return ids, mass


@dataclass
class Track:
    track_id: int
    fragment_ids: list[int] = field(default_factory=list)
    frame_ids: set[int] = field(default_factory=set)
    core_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    core_mass: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    full_ids: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )
    full_mass: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float32)
    )
    overlaps: list[float] = field(default_factory=list)

    def add_fragment(self, fragment: Fragment, overlap: float | None = None) -> None:
        if int(fragment.frame_id) in self.frame_ids:
            raise ValueError("a track may contain at most one fragment per frame")
        self.fragment_ids.append(int(fragment.fragment_id))
        self.frame_ids.add(int(fragment.frame_id))
        self.core_ids, self.core_mass = _sparse_max_union(
            self.core_ids, self.core_mass, fragment.core_ids, fragment.core_mass
        )
        self.full_ids, self.full_mass = _sparse_max_union(
            self.full_ids, self.full_mass, fragment.full_ids, fragment.full_mass
        )
        if overlap is not None:
            self.overlaps.append(float(overlap))


@dataclass(frozen=True)
class SparseEvidence:
    """A zero-filled sparse vector indexed by Gaussian id."""

    ids: np.ndarray
    values: np.ndarray
    point_count: int

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids, dtype=np.int64).reshape(-1)
        values = np.asarray(self.values).reshape(-1)
        if len(ids) != len(values):
            raise ValueError("sparse evidence ids and values must have equal length")
        if np.any(ids < 0) or np.any(ids >= int(self.point_count)):
            raise ValueError("sparse evidence id is outside point_count")
        if len(ids) and np.any(np.diff(ids) <= 0):
            raise ValueError("sparse evidence ids must be sorted and unique")
        object.__setattr__(self, "ids", ids.astype(np.int32))
        object.__setattr__(self, "values", values)

    def __getitem__(self, point_ids: int | slice | np.ndarray) -> Any:
        if isinstance(point_ids, slice):
            query = np.arange(self.point_count, dtype=np.int64)[point_ids]
        else:
            raw = np.asarray(point_ids)
            if raw.dtype == bool:
                if raw.shape != (self.point_count,):
                    raise IndexError("boolean sparse evidence index has wrong length")
                query = np.flatnonzero(raw)
            else:
                query = raw.astype(np.int64, copy=False)
        flat = query.reshape(-1)
        result = np.zeros(len(flat), dtype=self.values.dtype)
        if len(self.ids) and len(flat):
            positions = np.searchsorted(self.ids, flat)
            valid = positions < len(self.ids)
            matched = np.zeros(len(flat), dtype=bool)
            matched[valid] = self.ids[positions[valid]] == flat[valid]
            result[matched] = self.values[positions[matched]]
        shaped = result.reshape(query.shape)
        return shaped.item() if query.ndim == 0 else shaped


def _sum_sparse_chunks(
    id_chunks: Sequence[np.ndarray],
    value_chunks: Sequence[np.ndarray],
    point_count: int,
    *,
    dtype: np.dtype[Any],
) -> SparseEvidence:
    if len(id_chunks) != len(value_chunks):
        raise ValueError("sparse chunk ids and values must have equal length")
    if not id_chunks:
        return SparseEvidence(
            np.empty(0, dtype=np.int32), np.empty(0, dtype=dtype), point_count
        )
    ids = np.concatenate([np.asarray(chunk, dtype=np.int64) for chunk in id_chunks])
    values = np.concatenate([np.asarray(chunk, dtype=dtype) for chunk in value_chunks])
    unique, inverse = np.unique(ids, return_inverse=True)
    totals = np.zeros(len(unique), dtype=dtype)
    np.add.at(totals, inverse, values)
    return SparseEvidence(unique.astype(np.int32), totals, point_count)


@dataclass(frozen=True)
class ConsensusAssignment:
    core_track_id: np.ndarray
    visible_views: np.ndarray
    visible_mass: np.ndarray
    assignment_margin: np.ndarray
    valid_track_ids: tuple[int, ...]
    track_positive_views: dict[int, SparseEvidence]
    track_conflict_views: dict[int, SparseEvidence]
    track_positive_mass: dict[int, SparseEvidence]
    track_core_ids: dict[int, np.ndarray]
    track_full_ids: dict[int, np.ndarray]


@dataclass(frozen=True)
class MultiViewLabelVote:
    frame_id: int
    class_id: int
    weighted_iou: float


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
    core_candidate_id: np.ndarray
    full_ids: tuple[np.ndarray, ...]
    core_ids: tuple[np.ndarray, ...]
    candidates: tuple[dict[str, Any], ...]


def weighted_core_overlap(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> tuple[int, float]:
    """Return shared points and the weighted overlap coefficient.

    Support is intersected with ``min(left_mass, right_mass)`` and normalized
    by the smaller total mass.  The coefficient is therefore symmetric and in
    ``[0, 1]``.  Track support uses the per-Gaussian maximum across views so a
    long track does not win merely because its mass has been repeatedly added.
    """
    left_ids, left_mass = _canonical_sparse(left_ids, left_mass, name="left core")
    right_ids, right_mass = _canonical_sparse(right_ids, right_mass, name="right core")
    if not len(left_ids) or not len(right_ids):
        return 0, 0.0
    shared, left_index, right_index = np.intersect1d(
        left_ids, right_ids, assume_unique=True, return_indices=True
    )
    denominator = min(float(left_mass.sum()), float(right_mass.sum()))
    if denominator <= 0:
        return int(len(shared)), 0.0
    numerator = float(np.minimum(left_mass[left_index], right_mass[right_index]).sum())
    return int(len(shared)), float(np.clip(numerator / denominator, 0.0, 1.0))


def associate_fragments(
    fragments: Sequence[Fragment],
    config: V8Config = V8Config(),
) -> list[Track]:
    """Deterministically associate fragments without reading semantic labels."""
    fragment_ids = [int(fragment.fragment_id) for fragment in fragments]
    if len(set(fragment_ids)) != len(fragment_ids):
        raise ValueError("fragment_id values must be unique")
    tracks: list[Track] = []
    for fragment in sorted(
        fragments,
        key=lambda item: (int(item.frame_id), int(item.mask_index), int(item.fragment_id)),
    ):
        scored: list[tuple[float, int, int]] = []
        for track in tracks:
            if int(fragment.frame_id) in track.frame_ids:
                continue
            shared, overlap = weighted_core_overlap(
                fragment.core_ids,
                fragment.core_mass,
                track.core_ids,
                track.core_mass,
            )
            if (
                shared >= int(config.track_min_shared_core)
                and overlap >= float(config.track_min_overlap)
            ):
                scored.append((overlap, shared, int(track.track_id)))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected: Track | None = None
        overlap: float | None = None
        if scored:
            runner_up = scored[1][0] if len(scored) > 1 else 0.0
            if scored[0][0] - runner_up >= float(config.track_min_margin):
                selected = tracks[scored[0][2]]
                overlap = scored[0][0]
        if selected is None:
            selected = Track(track_id=len(tracks))
            tracks.append(selected)
        selected.add_fragment(fragment, overlap)
    return tracks


def _frame_sparse_max(
    fragments: Sequence[Fragment],
    *,
    core: bool,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty(0, dtype=np.int32)
    mass = np.empty(0, dtype=np.float32)
    for fragment in fragments:
        fragment_ids = fragment.core_ids if core else fragment.full_ids
        fragment_mass = fragment.core_mass if core else fragment.full_mass
        ids, mass = _sparse_max_union(ids, mass, fragment_ids, fragment_mass)
    return ids, mass


def _assign_unique_core(
    eligible: Mapping[int, np.ndarray],
    positive: Mapping[int, SparseEvidence],
    visible_views: np.ndarray,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    point_count = len(visible_views)
    active = {track_id for track_id, ids in eligible.items() if len(ids) >= min_points}
    labels = np.full(point_count, -1, dtype=np.int32)
    margins = np.zeros(point_count, dtype=np.float32)
    while active:
        labels.fill(-1)
        margins.fill(0.0)
        choices: dict[int, list[tuple[float, int, int]]] = {}
        for track_id in sorted(active):
            for point_id in eligible[track_id]:
                count = int(positive[track_id][point_id])
                ratio = count / max(int(visible_views[point_id]), 1)
                choices.setdefault(int(point_id), []).append((ratio, count, track_id))
        for point_id, point_choices in choices.items():
            point_choices.sort(key=lambda item: (-item[0], -item[1], item[2]))
            labels[point_id] = point_choices[0][2]
            runner_up = point_choices[1][0] if len(point_choices) > 1 else 0.0
            margins[point_id] = float(point_choices[0][0] - runner_up)
        surviving = {
            track_id
            for track_id in active
            if np.count_nonzero(labels == track_id) >= int(min_points)
        }
        if surviving == active:
            break
        active = surviving
    if not active:
        labels.fill(-1)
        margins.fill(0.0)
    return labels, margins, tuple(sorted(active))


def build_consensus_assignment(
    tracks: Sequence[Track],
    fragments: Sequence[Fragment],
    frame_evidence: Sequence[FrameEvidence],
    point_count: int,
    config: V8Config = V8Config(),
) -> ConsensusAssignment:
    """Build unique core/full masks with every physical view counted once.

    A ``grounded_missing`` frame is an abstention: its visibility is not added
    to the support denominator.  Core positive evidence comes exclusively from
    ``Fragment.core_ids``.  Overlapping fragments in one frame contribute at
    most one positive/conflict count for that physical view.
    """
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    fragment_by_id = {int(fragment.fragment_id): fragment for fragment in fragments}
    if len(fragment_by_id) != len(fragments):
        raise ValueError("fragment_id values must be unique")
    frame_by_id = {int(frame.frame_id): frame for frame in frame_evidence}
    if len(frame_by_id) != len(frame_evidence):
        raise ValueError("FrameEvidence.frame_id values must be unique")
    if any(int(fragment.frame_id) not in frame_by_id for fragment in fragments):
        raise ValueError("every fragment must have matching FrameEvidence")
    if any(
        np.any(ids >= point_count)
        for ids in (
            [frame.visible_ids for frame in frame_evidence]
            + [fragment.full_ids for fragment in fragments]
            + [fragment.core_ids for fragment in fragments]
        )
    ):
        raise ValueError("point id exceeds point_count")

    visible_views = np.zeros(point_count, dtype=np.int32)
    visible_mass = np.zeros(point_count, dtype=np.float64)
    for frame in frame_evidence:
        if frame.grounded_missing:
            continue
        visible_views[frame.visible_ids] += 1
        visible_mass[frame.visible_ids] += frame.visible_mass

    track_positive_views: dict[int, SparseEvidence] = {}
    track_positive_mass: dict[int, SparseEvidence] = {}
    track_fragments_by_frame: dict[int, dict[int, list[Fragment]]] = {}
    for track in tracks:
        by_frame: dict[int, list[Fragment]] = {}
        for fragment_id in track.fragment_ids:
            fragment = fragment_by_id[int(fragment_id)]
            by_frame.setdefault(int(fragment.frame_id), []).append(fragment)
        track_fragments_by_frame[int(track.track_id)] = by_frame
        positive_ids: list[np.ndarray] = []
        positive_values: list[np.ndarray] = []
        mass_ids: list[np.ndarray] = []
        mass_values: list[np.ndarray] = []
        for frame_id, frame_fragments in by_frame.items():
            if frame_by_id[frame_id].grounded_missing:
                continue
            core_ids, _ = _frame_sparse_max(frame_fragments, core=True)
            full_ids, full_mass = _frame_sparse_max(frame_fragments, core=False)
            positive_ids.append(core_ids)
            positive_values.append(np.ones(len(core_ids), dtype=np.int32))
            mass_ids.append(full_ids)
            mass_values.append(full_mass.astype(np.float64))
        track_positive_views[int(track.track_id)] = _sum_sparse_chunks(
            positive_ids,
            positive_values,
            point_count,
            dtype=np.dtype(np.int32),
        )
        track_positive_mass[int(track.track_id)] = _sum_sparse_chunks(
            mass_ids,
            mass_values,
            point_count,
            dtype=np.dtype(np.float64),
        )

    conflict_ids: dict[int, list[np.ndarray]] = {
        int(track.track_id): [] for track in tracks
    }
    for frame in frame_evidence:
        if frame.grounded_missing:
            continue
        per_track_core: dict[int, np.ndarray] = {}
        for track in tracks:
            frame_fragments = track_fragments_by_frame[int(track.track_id)].get(
                int(frame.frame_id), []
            )
            if frame_fragments:
                per_track_core[int(track.track_id)], _ = _frame_sparse_max(
                    frame_fragments, core=True
                )
        for track in tracks:
            other = [
                ids
                for track_id, ids in per_track_core.items()
                if track_id != int(track.track_id) and len(ids)
            ]
            if other:
                conflicting = np.intersect1d(
                    track.core_ids,
                    np.unique(np.concatenate(other)),
                    assume_unique=True,
                ).astype(np.int32)
                if len(conflicting):
                    conflict_ids[int(track.track_id)].append(conflicting)
    track_conflict_views: dict[int, SparseEvidence] = {}
    for track in tracks:
        track_id = int(track.track_id)
        chunks = conflict_ids[track_id]
        track_conflict_views[track_id] = _sum_sparse_chunks(
            chunks,
            [np.ones(len(ids), dtype=np.int32) for ids in chunks],
            point_count,
            dtype=np.dtype(np.int32),
        )

    eligible: dict[int, np.ndarray] = {}
    for track in tracks:
        track_id = int(track.track_id)
        if len(track.frame_ids) < int(config.track_min_frames):
            continue
        positive = track_positive_views[track_id]
        conflict = track_conflict_views[track_id]
        members = positive.ids
        denominator = visible_views[members]
        keep = (
            (denominator > 0)
            & (positive[members] >= int(config.core_min_positive_views))
            & (
                positive[members] / np.maximum(denominator, 1)
                >= float(config.core_min_positive_ratio)
            )
            & (
                conflict[members] / np.maximum(denominator, 1)
                <= float(config.core_max_conflict_ratio)
            )
        )
        eligible[track_id] = members[keep].astype(np.int32)

    core_labels, margins, valid_track_ids = _assign_unique_core(
        eligible,
        track_positive_views,
        visible_views,
        int(config.core_min_points),
    )
    track_full_ids: dict[int, np.ndarray] = {}
    track_by_id = {int(track.track_id): track for track in tracks}
    for track_id in valid_track_ids:
        positive_mass = track_positive_mass[track_id]
        union = track_by_id[track_id].full_ids
        denominator = visible_mass[union]
        ratio = np.divide(
            positive_mass[union],
            denominator,
            out=np.zeros(len(union), dtype=np.float64),
            where=denominator > 0,
        )
        keep = ratio >= float(config.full_min_positive_mass_ratio)
        # Full masks intentionally remain overlapping until prior replay.  The
        # only pre-replay ownership decision is the unique consensus core.
        track_full_ids[track_id] = np.union1d(
            union[keep], np.flatnonzero(core_labels == track_id)
        ).astype(np.int32)

    track_core_ids = {
        track_id: np.flatnonzero(core_labels == track_id).astype(np.int32)
        for track_id in valid_track_ids
    }
    return ConsensusAssignment(
        core_track_id=core_labels,
        visible_views=visible_views,
        visible_mass=visible_mass.astype(np.float32),
        assignment_margin=margins,
        valid_track_ids=valid_track_ids,
        track_positive_views=track_positive_views,
        track_conflict_views=track_conflict_views,
        track_positive_mass=track_positive_mass,
        track_core_ids=track_core_ids,
        track_full_ids=track_full_ids,
    )


def classify_tracks_multiview(
    tracks: Sequence[Track],
    votes_by_track: Mapping[int, Sequence[MultiViewLabelVote]],
    class_names: Sequence[str],
    config: V8Config = V8Config(),
) -> dict[int, TrackClassification]:
    """Classify tracks after association using at most one label per view."""
    output: dict[int, TrackClassification] = {}
    for track in sorted(tracks, key=lambda item: int(item.track_id)):
        per_frame: dict[int, MultiViewLabelVote] = {}
        for vote in votes_by_track.get(int(track.track_id), ()):
            if not np.isfinite(vote.weighted_iou):
                continue
            if vote.weighted_iou < float(config.mv_label_min_iou):
                continue
            if not 0 <= int(vote.class_id) < len(class_names):
                continue
            previous = per_frame.get(int(vote.frame_id))
            key = (float(vote.weighted_iou), -int(vote.class_id))
            if previous is None or key > (
                float(previous.weighted_iou),
                -int(previous.class_id),
            ):
                per_frame[int(vote.frame_id)] = vote
        counts = np.zeros(len(class_names), dtype=np.int32)
        for vote in per_frame.values():
            counts[int(vote.class_id)] += 1
        total = int(counts.sum())
        if not total:
            output[int(track.track_id)] = TrackClassification(
                int(track.track_id), -1, "", 0.0, 0.0, 0, "mv-label", False
            )
            continue
        order = np.argsort(-counts, kind="stable")
        winner = int(order[0])
        runner_up = int(counts[order[1]]) if len(order) > 1 else 0
        ratio = float(counts[winner] / total)
        margin = float((counts[winner] - runner_up) / total)
        class_name = str(class_names[winner])
        output[int(track.track_id)] = TrackClassification(
            track_id=int(track.track_id),
            class_id=winner,
            class_name=class_name,
            semantic_ratio=ratio,
            semantic_margin=margin,
            effective_view_count=total,
            source="mv-label",
            eligible=class_name in SAGA20,
        )
    return output


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("features must be a matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 1e-12)


def classify_tracks_codebook(
    tracks: Sequence[Track],
    consensus: ConsensusAssignment,
    semantic_features: np.ndarray,
    label_embeddings: np.ndarray,
    class_names: Sequence[str],
) -> dict[int, TrackClassification]:
    """Classify a frozen core using the complete normalized class codebook."""
    features = _normalise_rows(semantic_features)
    embeddings = _normalise_rows(label_embeddings)
    if len(features) != len(consensus.core_track_id):
        raise ValueError("semantic_features must have one row per Gaussian")
    if len(embeddings) != len(class_names) or features.shape[1] != embeddings.shape[1]:
        raise ValueError("label embeddings and class names must match feature dimensions")
    track_by_id = {int(track.track_id): track for track in tracks}
    output: dict[int, TrackClassification] = {}
    for track_id in consensus.valid_track_ids:
        core_ids = consensus.track_core_ids[track_id]
        prototype = features[core_ids].mean(axis=0)
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
            track_id=track_id,
            class_id=winner,
            class_name=class_name,
            semantic_ratio=float(np.clip((top + 1.0) * 0.5, 0.0, 1.0)),
            semantic_margin=float(np.clip((top - runner_up) * 0.5, 0.0, 1.0)),
            effective_view_count=len(track_by_id[track_id].frame_ids),
            source="codebook",
            eligible=class_name in SAGA20,
        )
    return output


def _local_surface_density(
    xyz: np.ndarray,
    core_ids: np.ndarray,
    neighbors: int,
) -> float:
    # The registered support prior is candidate-local.  Querying the whole
    # scene would let a nearby object inflate this candidate's density.
    k = min(int(neighbors), max(len(core_ids) - 1, 0))
    if not k or not len(core_ids):
        return 0.0
    candidate_xyz = xyz[core_ids]
    distances, _ = cKDTree(candidate_xyz).query(candidate_xyz, k=k + 1)
    radii = np.maximum(np.asarray(distances)[:, -1], 1e-6)
    return float(np.median(k / (np.pi * radii * radii)))


def materialize_candidates(
    xyz_m: np.ndarray,
    tracks: Sequence[Track],
    consensus: ConsensusAssignment,
    classifications: Mapping[int, TrackClassification],
    config: V8Config = V8Config(),
) -> CandidateBank:
    """Materialize immutable, uniquely owned candidate core/full masks."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.shape != (len(consensus.core_track_id), 3):
        raise ValueError("xyz_m and consensus assignment must have matching rows")
    track_by_id = {int(track.track_id): track for track in tracks}
    core_candidates = np.full(len(xyz), -1, dtype=np.int32)
    candidate_full_ids: list[np.ndarray] = []
    candidate_core_ids: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for track_id in consensus.valid_track_ids:
        classification = classifications.get(track_id)
        if classification is None or not classification.eligible:
            continue
        core_ids = consensus.track_core_ids[track_id]
        full_ids = consensus.track_full_ids[track_id]
        if len(core_ids) < int(config.core_min_points) or not len(full_ids):
            continue
        positive = consensus.track_positive_views[track_id][core_ids]
        conflict = consensus.track_conflict_views[track_id][core_ids]
        visible = np.maximum(consensus.visible_views[core_ids], 1)
        positive_ratio = float(np.mean(positive / visible))
        conflict_ratio = float(np.mean(conflict / visible))
        track = track_by_id[track_id]
        overlap = float(np.median(track.overlaps)) if track.overlaps else 0.0
        view_count = len(track.frame_ids)
        q = float(
            np.clip(
                classification.semantic_ratio
                * positive_ratio
                * (0.5 + 0.5 * overlap)
                * (0.5 + 0.5 * min(view_count, 5) / 5.0)
                * (1.0 - conflict_ratio),
                0.0,
                1.0,
            )
        )
        candidate_id = len(rows)
        core_candidates[core_ids] = candidate_id
        candidate_full_ids.append(full_ids.copy())
        candidate_core_ids.append(core_ids.copy())
        extents = np.ptp(xyz[full_ids], axis=0)
        rows.append(
            {
                "candidate_id": candidate_id,
                "track_id": int(track_id),
                "branch_class": classification.class_name,
                "class_id": int(classification.class_id),
                "classification_source": classification.source,
                "full_point_count": int(len(full_ids)),
                "core_point_count": int(len(core_ids)),
                "effective_view_count": int(view_count),
                "semantic_ratio": float(classification.semantic_ratio),
                "semantic_margin": float(classification.semantic_margin),
                "mean_core_positive_ratio": positive_ratio,
                "conflict_ratio": conflict_ratio,
                "median_track_overlap": overlap,
                "base_score": q,
                "metric_extent_xyz_m": [float(value) for value in extents],
                "metric_extents_m": [float(value) for value in np.sort(extents)],
                "local_surface_density": _local_surface_density(
                    xyz, core_ids, int(config.local_density_neighbors)
                ),
            }
        )
    return CandidateBank(
        point_count=len(xyz),
        core_candidate_id=core_candidates,
        full_ids=tuple(candidate_full_ids),
        core_ids=tuple(candidate_core_ids),
        candidates=tuple(rows),
    )
