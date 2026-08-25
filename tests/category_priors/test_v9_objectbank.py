from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

import category_priors.v9_objectbank as v9_objectbank
from category_priors.v9_objectbank import (
    AssociationResult,
    ConsensusResult,
    Fragment,
    FrameEvidence,
    MultiviewClassVote,
    ObjectTrack,
    SparseCounts,
    TrackClassification,
    V9Config,
    _boundary_ratio,
    _direct_edges,
    associate_fragments,
    attach_local_halo,
    build_consensus_core,
    classify_tracks_codebook,
    classify_tracks_multiview,
    materialize_candidate_bank,
    weighted_core_overlap,
)


def _fragment(
    fragment_id: int,
    frame_id: int,
    core: list[int],
    full: list[int] | None = None,
    *,
    mask_index: int = 0,
) -> Fragment:
    full_ids = np.asarray(core if full is None else full, dtype=np.int32)
    core_ids = np.asarray(core, dtype=np.int32)
    return Fragment(
        fragment_id,
        frame_id,
        mask_index,
        full_ids,
        core_ids,
        np.ones(len(full_ids), dtype=np.float32),
        np.ones(len(core_ids), dtype=np.float32),
        0.0,
    )


def _frame(frame_id: int, fragments: list[Fragment], point_count: int) -> FrameEvidence:
    return FrameEvidence(
        frame_id,
        tuple(fragments),
        np.arange(point_count, dtype=np.int32),
        np.ones(point_count, dtype=np.float32),
    )


def test_fragment_canonicalizes_duplicate_mass() -> None:
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


def test_a1_global_merge_respects_same_frame_cannot_link() -> None:
    left = _fragment(0, 0, [0, 1, 2])
    next_view = _fragment(1, 1, [0, 1, 2])
    same_frame = _fragment(2, 0, [0, 1, 2], mask_index=1)
    result = associate_fragments([left, next_view, same_frame], "A1")
    assert len(result.tracks) == 2
    assert result.tracks[0].fragment_ids == (0, 1)
    assert result.tracks[1].fragment_ids == (2,)


def test_a2_mutual_affinity_attaches_only_singleton_to_established_component() -> None:
    fragments = [
        _fragment(0, 0, [0, 1, 2]),
        _fragment(1, 1, [0, 1, 2]),
        _fragment(2, 2, [3, 4, 5]),
    ]
    xyz = np.asarray(
        [[0.00, 0, 0], [0.10, 0, 0], [0.20, 0, 0],
         [0.01, 0, 0], [0.11, 0, 0], [0.21, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    result = associate_fragments(
        fragments,
        "A2",
        xyz_m=xyz,
        affinity=affinity,
        config=V9Config(bridge_radius_m=0.02),
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].fragment_ids == (0, 1, 2)
    assert [edge.kind for edge in result.accepted_edges].count("affinity") == 1


def test_a2_never_uses_affinity_to_bridge_two_established_components() -> None:
    fragments = [
        _fragment(0, 0, [0, 1, 2]),
        _fragment(1, 1, [0, 1, 2]),
        _fragment(2, 2, [3, 4, 5]),
        _fragment(3, 3, [3, 4, 5]),
    ]
    xyz = np.asarray(
        [[0.00, 0, 0], [0.10, 0, 0], [0.20, 0, 0],
         [0.01, 0, 0], [0.11, 0, 0], [0.21, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    result = associate_fragments(
        fragments,
        "A2",
        xyz_m=xyz,
        affinity=affinity,
        config=V9Config(bridge_radius_m=0.02),
    )
    assert [track.fragment_ids for track in result.tracks] == [(0, 1), (2, 3)]
    assert all(edge.kind != "affinity" for edge in result.accepted_edges)


def test_a3_mutual_topk_graph_can_link_disjoint_fragment_support() -> None:
    fragments = [
        _fragment(0, 0, [0, 1, 2]),
        _fragment(1, 1, [3, 4, 5]),
    ]
    xyz = np.asarray(
        [[0.00, 0, 0], [0.10, 0, 0], [0.20, 0, 0],
         [0.01, 0, 0], [0.11, 0, 0], [0.21, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    result = associate_fragments(
        fragments,
        "A3",
        xyz_m=xyz,
        affinity=affinity,
        config=V9Config(graph_physical_neighbors=5, graph_affinity_neighbors=1),
    )
    assert len(result.tracks) == 1
    assert result.graph_edge_count == 3
    assert any(edge.kind == "graph" for edge in result.accepted_edges)


def test_inverted_direct_edges_match_brute_force_order_and_scores() -> None:
    rng = np.random.default_rng(20260825)
    fragments = []
    for fragment_id in range(18):
        core = np.sort(rng.choice(24, size=7, replace=False)).astype(np.int32)
        mass = rng.uniform(0.5, 3.0, size=len(core)).astype(np.float32)
        fragments.append(
            Fragment(
                fragment_id,
                fragment_id % 4,
                0,
                core,
                core,
                mass,
                mass,
                0.0,
            )
        )
    config = V9Config(direct_min_shared_core=3, direct_min_overlap=0.18)
    expected = []
    ordered = sorted(fragments, key=lambda row: row.fragment_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.frame_id == right.frame_id:
                continue
            shared, overlap = weighted_core_overlap(left, right)
            if shared >= config.direct_min_shared_core and overlap >= config.direct_min_overlap:
                expected.append(
                    (left.fragment_id, right.fragment_id, overlap, shared)
                )
    actual = [
        (edge.left_fragment_id, edge.right_fragment_id, edge.score, edge.support)
        for edge in _direct_edges(fragments, config)
    ]
    assert actual == expected


def test_consensus_counts_conflict_once_per_view_and_assigns_unique_core() -> None:
    fragments = [
        _fragment(0, 0, [0, 1, 2, 3]),
        _fragment(1, 1, [0, 1, 2, 3]),
        _fragment(2, 0, [2, 3, 4, 5], mask_index=1),
        _fragment(3, 1, [2, 3, 4, 5], mask_index=1),
    ]
    association = AssociationResult(
        "A1",
        (
            ObjectTrack(0, (0, 1), (0, 1), (1.0,), "A1"),
            ObjectTrack(1, (2, 3), (0, 1), (1.0,), "A1"),
        ),
        (),
    )
    frames = [
        _frame(0, [fragments[0], fragments[2]], 6),
        _frame(1, [fragments[1], fragments[3]], 6),
    ]
    consensus = build_consensus_core(
        association,
        fragments,
        frames,
        6,
        V9Config(core_max_conflict_ratio=1.0, core_min_points=2),
    )
    assert consensus.conflict_views[0].take([2]).item() == 2
    assert consensus.conflict_views[1].take([2]).item() == 2
    assert consensus.core_track_id[2] == 0
    assert not np.intersect1d(consensus.core_ids[0], consensus.core_ids[1]).size


def test_consensus_conflict_is_exact_but_stored_only_on_positive_support() -> None:
    fragments = [
        _fragment(0, 0, [0, 1, 2]),
        _fragment(1, 1, [0, 1, 2]),
        _fragment(2, 1, [1, 3, 4], mask_index=1),
        _fragment(3, 2, [0, 3, 4], mask_index=1),
    ]
    association = AssociationResult(
        "A1",
        (
            ObjectTrack(0, (0, 1), (0, 1), (1.0,), "A1"),
            ObjectTrack(1, (2, 3), (1, 2), (1.0,), "A1"),
        ),
        (),
    )
    frames = [
        _frame(0, [fragments[0]], 5),
        _frame(1, [fragments[1], fragments[2]], 5),
        _frame(2, [fragments[3]], 5),
    ]
    consensus = build_consensus_core(
        association,
        fragments,
        frames,
        5,
        V9Config(
            core_min_positive_views=1,
            core_max_conflict_ratio=1.0,
            core_min_points=1,
        ),
    )

    # Track 0 sees point 0 in two positive views.  Frame 2 contains the same
    # point only in track 1, so the absent-track conflict still counts once.
    # Point 1 is shared by both tracks in frame 1 and also counts once.
    assert consensus.conflict_views[0].take([0, 1, 2]).tolist() == [1, 1, 0]
    # Track 1 conflicts on points 0/1 in both frames 0 and 1: it is absent in
    # frame 0 and overlaps track 0 in frame 1.  Points 3/4 remain exclusive.
    assert consensus.conflict_views[1].take([0, 1, 3, 4]).tolist() == [2, 2, 0, 0]
    for track_id, evidence in consensus.conflict_views.items():
        positive_ids = consensus.positive_views[track_id].ids
        assert np.all(np.isin(evidence.ids, positive_ids, assume_unique=True))


def test_sparse_conflict_matches_registered_brute_force_semantics() -> None:
    rng = np.random.default_rng(901)
    point_count = 50
    fragments = []
    track_fragments: list[list[int]] = [[] for _ in range(12)]
    track_frames: list[list[int]] = [[] for _ in range(12)]
    frame_fragments: list[list[Fragment]] = [[] for _ in range(6)]
    for track_id in range(12):
        selected_frames = np.sort(rng.choice(6, size=3, replace=False))
        for frame_id in selected_frames:
            fragment_id = len(fragments)
            core = np.sort(rng.choice(point_count, size=8, replace=False)).tolist()
            fragment = _fragment(fragment_id, int(frame_id), core)
            fragments.append(fragment)
            frame_fragments[int(frame_id)].append(fragment)
            track_fragments[track_id].append(fragment_id)
            track_frames[track_id].append(int(frame_id))
    tracks = tuple(
        ObjectTrack(
            track_id,
            tuple(track_fragments[track_id]),
            tuple(sorted(track_frames[track_id])),
            (1.0,),
            "A1",
        )
        for track_id in range(12)
    )
    association = AssociationResult("A1", tracks, ())
    frames = [
        FrameEvidence(
            frame_id,
            tuple(frame_fragments[frame_id]),
            np.arange(point_count, dtype=np.int32),
            np.ones(point_count, dtype=np.float32),
            abstain=frame_id == 5,
        )
        for frame_id in range(6)
    ]
    track_by_fragment = {
        fragment_id: track.track_id
        for track in tracks
        for fragment_id in track.fragment_ids
    }
    brute = {track.track_id: np.zeros(point_count, dtype=np.int32) for track in tracks}
    for frame in frames:
        if frame.abstain:
            continue
        by_track: dict[int, np.ndarray] = {}
        for fragment in frame.fragments:
            track_id = track_by_fragment[fragment.fragment_id]
            by_track[track_id] = np.union1d(
                by_track.get(track_id, np.empty(0, dtype=np.int32)),
                fragment.core_ids,
            ).astype(np.int32)
        for track in tracks:
            other = [ids for key, ids in by_track.items() if key != track.track_id]
            if other:
                brute[track.track_id][np.unique(np.concatenate(other))] += 1

    consensus = build_consensus_core(
        association,
        fragments,
        frames,
        point_count,
        V9Config(
            core_min_positive_views=1,
            core_max_conflict_ratio=1.0,
            core_min_points=1,
        ),
    )
    for track_id, positive in consensus.positive_views.items():
        assert np.array_equal(
            consensus.conflict_views[track_id].take(positive.ids),
            brute[track_id][positive.ids],
        )
        assert len(consensus.conflict_views[track_id].ids) <= len(positive.ids)


def test_local_halo_is_one_shot_and_never_overwrites_core() -> None:
    labels = np.asarray([0, 0, 0, -1, -1], dtype=np.int32)
    consensus = ConsensusResult(
        labels,
        np.ones(5, dtype=np.int32),
        np.zeros(5, dtype=np.float32),
        (0,),
        {0: SparseCounts(np.array([0, 1, 2]), np.array([2, 2, 2]), 5)},
        {0: SparseCounts(np.array([], dtype=np.int32), np.array([], dtype=np.int32), 5)},
        {0: np.array([0, 1, 2], dtype=np.int32)},
    )
    xyz = np.asarray([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [0.049, 0, 0], [0.098, 0, 0]])
    affinity = np.ones((5, 2), dtype=np.float64)
    attached = attach_local_halo(xyz, affinity, consensus)
    assert attached.tolist() == [0, 0, 0, 0, -1]


def test_neighbor_query_chunking_is_exact_for_halo_and_boundary(monkeypatch) -> None:
    labels = np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int32)
    consensus = ConsensusResult(
        labels,
        np.ones(6, dtype=np.int32),
        np.zeros(6, dtype=np.float32),
        (0,),
        {0: SparseCounts(np.array([0, 1, 2]), np.array([2, 2, 2]), 6)},
        {0: SparseCounts(np.array([], dtype=np.int32), np.array([], dtype=np.int32), 6)},
        {0: np.array([0, 1, 2], dtype=np.int32)},
    )
    xyz = np.asarray(
        [[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0],
         [0.020, 0, 0], [0.040, 0, 0], [0.090, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.ones((6, 2), dtype=np.float64)
    tree = cKDTree(xyz)
    members = np.asarray([0, 1, 2, 3], dtype=np.int32)

    monkeypatch.setattr(v9_objectbank, "_NEIGHBOR_QUERY_CHUNK", 100)
    halo_reference = attach_local_halo(xyz, affinity, consensus)
    boundary_reference = _boundary_ratio(xyz, members, tree, 0.05)
    monkeypatch.setattr(v9_objectbank, "_NEIGHBOR_QUERY_CHUNK", 1)
    assert np.array_equal(attach_local_halo(xyz, affinity, consensus), halo_reference)
    assert _boundary_ratio(xyz, members, tree, 0.05) == boundary_reference


def test_late_multiview_and_codebook_classifiers_do_not_change_tracks() -> None:
    association = AssociationResult(
        "A1", (ObjectTrack(0, (0, 1), (0, 1), (0.8,), "A1"),), ()
    )
    votes = {
        0: [
            MultiviewClassVote(0, 0, 0.6),
            MultiviewClassVote(0, 1, 0.4),
            MultiviewClassVote(1, 0, 1.0),
        ]
    }
    mv = classify_tracks_multiview(association, votes, ["chair", "wall"])[0]
    assert mv.class_name == "chair"
    assert np.isclose(mv.semantic_ratio, 0.8)
    consensus = ConsensusResult(
        np.array([0, 0, 0], dtype=np.int32),
        np.ones(3, dtype=np.int32),
        np.zeros(3, dtype=np.float32),
        (0,),
        {0: SparseCounts(np.arange(3), np.ones(3, dtype=np.int32), 3)},
        {0: SparseCounts(np.array([], dtype=np.int32), np.array([], dtype=np.int32), 3)},
        {0: np.arange(3, dtype=np.int32)},
    )
    codebook = classify_tracks_codebook(
        association,
        consensus,
        np.tile([[1.0, 0.0]], (3, 1)),
        np.eye(2),
        ["chair", "wall"],
    )[0]
    assert codebook.class_name == "chair"
    assert association.tracks[0].fragment_ids == (0, 1)


def test_materialized_bank_has_geometric_mean_score_and_prior_features() -> None:
    fragment0 = _fragment(0, 0, [0, 1, 2], [0, 1, 2, 3])
    fragment1 = _fragment(1, 1, [0, 1, 2], [0, 1, 2, 3])
    association = AssociationResult(
        "A1", (ObjectTrack(0, (0, 1), (0, 1), (0.8,), "A1"),), ()
    )
    consensus = build_consensus_core(
        association,
        [fragment0, fragment1],
        [_frame(0, [fragment0], 4), _frame(1, [fragment1], 4)],
        4,
    )
    xyz = np.asarray([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]])
    affinity = np.ones((4, 2), dtype=np.float64)
    classification = TrackClassification(0, 0, "chair", 0.8, 0.5, 2, "mv-label", True)
    bank = materialize_candidate_bank(
        xyz,
        affinity,
        association,
        consensus,
        np.array([0, 0, 0, 0], dtype=np.int32),
        {0: classification},
    )
    row = bank.candidates[0]
    expected = (0.8 * 1.0 * 1.0 * 0.8 * 0.4 * 1.0) ** (1 / 6)
    assert np.isclose(row["base_score"], expected)
    assert row["local_surface_density"] > 0
    assert 0 <= row["boundary_ratio_5cm"] <= 1
    assert bank.core_ids[0].tolist() == [0, 1, 2]
    assert bank.full_ids[0].tolist() == [0, 1, 2, 3]
