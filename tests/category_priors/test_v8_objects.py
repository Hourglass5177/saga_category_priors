from __future__ import annotations

import numpy as np

from category_priors.v8_objects import (
    Fragment,
    FrameEvidence,
    MultiViewLabelVote,
    Track,
    TrackClassification,
    V8Config,
    associate_fragments,
    build_consensus_assignment,
    classify_tracks_codebook,
    classify_tracks_multiview,
    materialize_candidates,
    weighted_core_overlap,
)


def _fragment(
    fragment_id: int,
    frame_id: int,
    core: list[int],
    full: list[int] | None = None,
    *,
    mask_index: int = 0,
    full_mass: float = 0.6,
) -> Fragment:
    full_ids = np.asarray(core if full is None else full, dtype=np.int32)
    core_ids = np.asarray(core, dtype=np.int32)
    return Fragment(
        fragment_id=fragment_id,
        frame_id=frame_id,
        mask_index=mask_index,
        full_ids=full_ids,
        core_ids=core_ids,
        full_mass=np.full(len(full_ids), full_mass, dtype=np.float32),
        core_mass=np.ones(len(core_ids), dtype=np.float32),
    )


def _frame(frame_id: int, fragments: list[Fragment], count: int = 20) -> FrameEvidence:
    return FrameEvidence(
        frame_id=frame_id,
        fragments=tuple(fragments),
        visible_ids=np.arange(count, dtype=np.int32),
        visible_mass=np.ones(count, dtype=np.float32),
    )


def test_weighted_core_overlap_uses_mass_not_only_point_count() -> None:
    shared, overlap = weighted_core_overlap(
        np.array([0, 1, 2]),
        np.array([1.0, 2.0, 1.0]),
        np.array([1, 2, 3]),
        np.array([1.0, 2.0, 1.0]),
    )
    assert shared == 2
    assert np.isclose(overlap, 0.5)


def test_fragment_canonicalizes_duplicate_support() -> None:
    fragment = Fragment(
        0,
        0,
        0,
        np.array([2, 1, 2]),
        np.array([2, 2]),
        np.array([0.2, 0.3, 0.4]),
        np.array([0.5, 0.7]),
    )
    assert fragment.full_ids.tolist() == [1, 2]
    assert np.allclose(fragment.full_mass, [0.3, 0.6])
    assert fragment.core_ids.tolist() == [2]
    assert np.allclose(fragment.core_mass, [1.2])


def test_association_is_semantic_free_and_same_frame_fragments_never_merge() -> None:
    first = _fragment(0, 0, list(range(10)), mask_index=0)
    same_frame = _fragment(1, 0, list(range(10, 20)), mask_index=1)
    next_frame = _fragment(2, 1, list(range(10)))
    tracks = associate_fragments([first, same_frame, next_frame])
    assert len(tracks) == 2
    assert tracks[0].fragment_ids == [0, 2]
    assert tracks[1].fragment_ids == [1]


def test_ambiguous_bridge_starts_a_new_track() -> None:
    left = _fragment(0, 0, list(range(10)), mask_index=0)
    right = _fragment(1, 0, list(range(10, 20)), mask_index=1)
    bridge = _fragment(2, 1, list(range(5)) + list(range(10, 15)))
    tracks = associate_fragments([left, right, bridge])
    assert len(tracks) == 3
    assert tracks[2].fragment_ids == [2]


def test_consensus_core_uses_core_ids_and_full_uses_mass_ratio() -> None:
    first = _fragment(0, 0, list(range(10)), list(range(12)))
    second = _fragment(1, 1, list(range(10)), list(range(12)))
    track = Track(track_id=0)
    track.add_fragment(first)
    track.add_fragment(second, 1.0)
    consensus = build_consensus_assignment(
        [track],
        [first, second],
        [_frame(0, [first], 12), _frame(1, [second], 12)],
        12,
    )
    assert np.all(consensus.core_track_id[:10] == 0)
    assert np.all(consensus.core_track_id[10:] == -1)
    assert consensus.track_full_ids[0].tolist() == list(range(12))
    assert np.all(consensus.track_positive_views[0][10:] == 0)


def test_conflicts_are_counted_once_per_physical_view() -> None:
    own0 = _fragment(0, 0, list(range(10)))
    own1 = _fragment(1, 1, list(range(10)))
    other0a = _fragment(2, 0, list(range(5, 15)), mask_index=1)
    other0b = _fragment(3, 0, list(range(5, 15)), mask_index=2)
    other1 = _fragment(4, 1, list(range(5, 15)), mask_index=1)
    track0 = Track(0)
    track0.add_fragment(own0)
    track0.add_fragment(own1, 1.0)
    # Deliberately construct duplicate same-frame fragments to verify that the
    # consensus layer itself deduplicates views, even for malformed input.
    track1 = Track(
        track_id=1,
        fragment_ids=[2, 3, 4],
        frame_ids={0, 1},
        core_ids=np.arange(5, 15, dtype=np.int32),
        core_mass=np.ones(10, dtype=np.float32),
        full_ids=np.arange(5, 15, dtype=np.int32),
        full_mass=np.ones(10, dtype=np.float32),
    )
    frames = [
        _frame(0, [own0, other0a, other0b], 15),
        _frame(1, [own1, other1], 15),
    ]
    consensus = build_consensus_assignment(
        [track0, track1],
        [own0, own1, other0a, other0b, other1],
        frames,
        15,
        V8Config(core_max_conflict_ratio=1.0, core_min_points=3),
    )
    assert consensus.track_conflict_views[0][6] == 2
    assert consensus.track_conflict_views[1][6] == 2


def test_grounded_missing_frame_is_abstention_not_negative_visibility() -> None:
    first = _fragment(0, 0, list(range(10)))
    track = Track(0)
    track.add_fragment(first)
    missing = FrameEvidence(
        1,
        (),
        np.arange(10),
        np.ones(10),
        grounded_missing=True,
    )
    consensus = build_consensus_assignment(
        [track],
        [first],
        [_frame(0, [first], 10), missing],
        10,
        V8Config(track_min_frames=1, core_min_positive_views=1),
    )
    assert np.all(consensus.visible_views == 1)
    assert np.all(consensus.core_track_id == 0)


def test_multiview_label_votes_are_deduplicated_per_frame() -> None:
    track = Track(track_id=0, frame_ids={0, 1})
    votes = {
        0: [
            MultiViewLabelVote(0, 0, 0.30),
            MultiViewLabelVote(0, 1, 0.90),
            MultiViewLabelVote(1, 0, 0.80),
        ]
    }
    result = classify_tracks_multiview([track], votes, ["chair", "table"])[0]
    assert result.effective_view_count == 2
    assert result.class_name == "chair"  # stable class-id tie break
    assert result.semantic_ratio == 0.5
    assert result.eligible


def test_codebook_classification_is_late_and_uses_consensus_core() -> None:
    track = Track(track_id=0, frame_ids={0, 1})
    consensus = build_consensus_assignment(
        [
            Track(
                0,
                fragment_ids=[0, 1],
                frame_ids={0, 1},
                core_ids=np.arange(10, dtype=np.int32),
                core_mass=np.ones(10, dtype=np.float32),
                full_ids=np.arange(10, dtype=np.int32),
                full_mass=np.ones(10, dtype=np.float32),
            )
        ],
        [_fragment(0, 0, list(range(10))), _fragment(1, 1, list(range(10)))],
        [
            _frame(0, [_fragment(0, 0, list(range(10)))], 10),
            _frame(1, [_fragment(1, 1, list(range(10)))], 10),
        ],
        10,
    )
    features = np.tile(np.array([[1.0, 0.0]]), (10, 1))
    result = classify_tracks_codebook(
        [track], consensus, features, np.eye(2), ["chair", "wall"]
    )[0]
    assert result.class_name == "chair"
    assert result.eligible


def test_materialized_bank_keeps_separate_core_full_and_required_score_fields() -> None:
    fragments = [
        _fragment(0, 0, list(range(10)), list(range(12))),
        _fragment(1, 1, list(range(10)), list(range(12))),
    ]
    track = Track(0)
    track.add_fragment(fragments[0])
    track.add_fragment(fragments[1], 0.75)
    consensus = build_consensus_assignment(
        [track], fragments, [_frame(0, [fragments[0]], 12), _frame(1, [fragments[1]], 12)], 12
    )
    classification = TrackClassification(0, 0, "chair", 0.8, 0.6, 2, "mv-label", True)
    xyz = np.column_stack((np.arange(12) * 0.01, np.zeros((12, 2))))
    bank = materialize_candidates(xyz, [track], consensus, {0: classification})
    assert np.count_nonzero(bank.core_candidate_id == 0) == 10
    assert bank.full_ids[0].tolist() == list(range(12))
    assert bank.core_ids[0].tolist() == list(range(10))
    row = bank.candidates[0]
    assert row["branch_class"] == "chair"
    assert row["core_point_count"] == 10
    assert row["full_point_count"] == 12
    assert 0.0 <= row["base_score"] <= 1.0
    assert len(row["metric_extents_m"]) == 3
    assert row["local_surface_density"] >= 0.0


def test_bank_preserves_overlapping_full_masks_for_prior_replay() -> None:
    fragments = [
        _fragment(0, 0, list(range(10)), list(range(15)), mask_index=0),
        _fragment(1, 1, list(range(10)), list(range(15)), mask_index=0),
        _fragment(2, 0, list(range(15, 25)), list(range(10, 25)), mask_index=1),
        _fragment(3, 1, list(range(15, 25)), list(range(10, 25)), mask_index=1),
    ]
    tracks = [Track(0), Track(1)]
    tracks[0].add_fragment(fragments[0])
    tracks[0].add_fragment(fragments[1], 1.0)
    tracks[1].add_fragment(fragments[2])
    tracks[1].add_fragment(fragments[3], 1.0)
    frames = [
        _frame(0, [fragments[0], fragments[2]], 25),
        _frame(1, [fragments[1], fragments[3]], 25),
    ]
    consensus = build_consensus_assignment(tracks, fragments, frames, 25)
    classifications = {
        0: TrackClassification(0, 0, "chair", 0.8, 0.5, 2, "mv-label", True),
        1: TrackClassification(1, 1, "table", 0.8, 0.5, 2, "mv-label", True),
    }
    xyz = np.column_stack((np.arange(25) * 0.01, np.zeros((25, 2))))
    bank = materialize_candidates(xyz, tracks, consensus, classifications)
    assert np.intersect1d(bank.full_ids[0], bank.full_ids[1]).tolist() == list(
        range(10, 15)
    )
    assert len(np.intersect1d(bank.core_ids[0], bank.core_ids[1])) == 0
