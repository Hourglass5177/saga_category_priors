from __future__ import annotations

import numpy as np

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
    associate_fragments,
    attach_local_halo,
    build_consensus_core,
    classify_tracks_codebook,
    classify_tracks_multiview,
    materialize_candidate_bank,
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
