from __future__ import annotations

import numpy as np

import category_priors.v10_objectbank as v10_objectbank
import category_priors.v10_runner as v10_runner
from category_priors.v10_objectbank import (
    V10Association,
    V10Config,
    V10Track,
    _multiview_classifications,
    _r0_consensus,
    _v9_consensus,
    associate_fragments_v10,
    build_v10_candidate_bank,
    build_v10_object_bank,
    pair_evidence,
    select_covisible_frame_pairs,
)
from category_priors.v9_objectbank import (
    ConsensusResult,
    Fragment,
    FrameEvidence,
    SparseCounts,
)


def _fragment(
    fragment_id: int,
    frame_id: int,
    full: np.ndarray | list[int],
    core: np.ndarray | list[int],
    *,
    membership: float = 1.0,
    mask_index: int = 0,
) -> Fragment:
    full_ids = np.asarray(full, dtype=np.int32)
    core_ids = np.asarray(core, dtype=np.int32)
    return Fragment(
        fragment_id,
        frame_id,
        mask_index,
        full_ids,
        core_ids,
        np.full(len(full_ids), membership, dtype=np.float32),
        np.ones(len(core_ids), dtype=np.float32),
        0.0,
    )


def _frame(frame_id: int, fragments: list[Fragment], visible: np.ndarray) -> FrameEvidence:
    ids = np.asarray(visible, dtype=np.int32)
    return FrameEvidence(
        frame_id,
        tuple(fragments),
        ids,
        np.ones(len(ids), dtype=np.float32),
    )


def _frame_with_mass(
    frame_id: int,
    fragments: list[Fragment],
    visible: np.ndarray,
    mass: np.ndarray,
) -> FrameEvidence:
    return FrameEvidence(
        frame_id,
        tuple(fragments),
        np.asarray(visible, dtype=np.int32),
        np.asarray(mass, dtype=np.float32),
    )


def test_r1_rebuilds_100_point_full_and_core_from_soft_full_membership() -> None:
    full = np.arange(100, dtype=np.int32)
    core = np.arange(3, dtype=np.int32)
    fragments = [
        _fragment(0, 0, full, core),
        _fragment(1, 1, full, core),
    ]
    frames = [
        _frame(0, [fragments[0]], full),
        _frame(1, [fragments[1]], full),
    ]
    xyz = np.column_stack(
        (np.arange(100, dtype=np.float64), np.zeros(100), np.zeros(100))
    )
    affinity = np.ones((100, 1), dtype=np.float64)

    association_r0, bank_r0 = build_v10_candidate_bank(
        fragments,
        frames,
        100,
        pair_mode="P1",
        reconstruction_mode="R0",
        xyz_m=xyz,
        affinity=affinity,
    )
    association_r1, bank_r1 = build_v10_candidate_bank(
        fragments,
        frames,
        100,
        pair_mode="P1",
        reconstruction_mode="R1",
    )

    assert association_r0.tracks[0].fragment_ids == (0, 1)
    assert association_r1.tracks[0].fragment_ids == (0, 1)
    assert bank_r0.core_ids[0].tolist() == [0, 1, 2]
    assert bank_r0.full_ids[0].tolist() == [0, 1, 2]
    assert bank_r1.core_ids[0].tolist() == full.tolist()
    assert bank_r1.full_ids[0].tolist() == full.tolist()


def test_p1_rejects_three_to_one_hundred_containment_that_p0_calls_perfect() -> None:
    small = _fragment(0, 0, np.arange(3), np.arange(3))
    large = _fragment(1, 1, np.arange(100), np.arange(100))
    frames = {
        0: _frame(0, [small], np.arange(100)),
        1: _frame(1, [large], np.arange(100)),
    }

    evidence = pair_evidence(small, large, frames)

    assert evidence.p0_overlap == 1.0
    assert evidence.p0_eligible
    assert evidence.left_coverage == 1.0
    assert evidence.right_coverage == 0.03
    assert not evidence.p1_eligible
    assert not evidence.strong

    p0 = associate_fragments_v10([small, large], list(frames.values()), "P0")
    p1 = associate_fragments_v10([small, large], list(frames.values()), "P1")
    assert [track.fragment_ids for track in p0.tracks] == [(0, 1)]
    assert [track.fragment_ids for track in p1.tracks] == [(0,), (1,)]


def test_p1_uses_cross_frame_visibility_and_geometric_mean_not_minimum() -> None:
    ids = np.arange(4, dtype=np.int32)
    left_visible = np.asarray([1.0, 1.0, 1.0, 10.0])
    right_visible = np.asarray([10.0, 1.0, 1.0, 1.0])
    left_probability = np.asarray([1.0, 1.0, 1.0, 0.5])
    right_probability = np.asarray([1.0, 1.0, 0.5, 0.5])
    left = Fragment(
        0,
        0,
        0,
        ids,
        ids,
        left_visible * left_probability,
        left_visible * left_probability,
        0.0,
    )
    right = Fragment(
        1,
        1,
        0,
        ids,
        ids,
        right_visible * right_probability,
        right_visible * right_probability,
        0.0,
    )
    frames = {
        0: _frame_with_mass(0, [left], ids, left_visible),
        1: _frame_with_mass(1, [right], ids, right_visible),
    }

    evidence = pair_evidence(left, right, frames)
    expected_left = 11.75 / 12.5
    expected_right = 5.0 / 7.5

    assert np.isclose(evidence.left_coverage, expected_left)
    assert np.isclose(evidence.right_coverage, expected_right)
    assert np.isclose(
        evidence.p1_score, np.sqrt(expected_left * expected_right)
    )
    assert not np.isclose(
        evidence.p1_score, min(expected_left, expected_right)
    )


def test_weak_a_b_c_chain_cannot_bridge_without_three_view_cycle() -> None:
    # A overlaps the first half of B and C overlaps the second half.  Both
    # directional matches pass the ordinary P1 gate but neither is strong;
    # A and C have no edge, so the weak chain must not become one object.
    a = _fragment(0, 0, np.arange(0, 10), np.arange(0, 10))
    b = _fragment(1, 1, np.arange(0, 20), np.arange(0, 20))
    c = _fragment(2, 2, np.arange(10, 20), np.arange(10, 20))
    frames = [
        _frame(0, [a], np.arange(0, 20)),
        _frame(1, [b], np.arange(0, 20)),
        _frame(2, [c], np.arange(0, 20)),
    ]

    result = associate_fragments_v10(
        [a, b, c], frames, "P1", view_consensus=True
    )

    assert len(result.tentative_edges) == 2
    assert not result.accepted_edges
    assert [track.fragment_ids for track in result.tracks] == [(0,), (1,), (2,)]


def test_three_distinct_views_can_confirm_a_weak_cycle() -> None:
    ids = np.arange(12, dtype=np.int32)
    fragments = [
        _fragment(0, 0, ids, ids, membership=1.0),
        _fragment(1, 1, ids, ids, membership=0.5),
        _fragment(2, 2, ids, ids, membership=0.25),
    ]
    frames = [
        _frame(index, [fragment], ids)
        for index, fragment in enumerate(fragments)
    ]

    result = associate_fragments_v10(
        fragments, frames, "P1", view_consensus=True
    )

    assert len(result.tentative_edges) == 3
    assert all(edge.cycle_supported for edge in result.tentative_edges)
    assert not any(edge.strong for edge in result.tentative_edges)
    assert len(result.tracks) == 1
    assert result.tracks[0].fragment_ids == (0, 1, 2)


def test_same_frame_alternative_hypothesis_never_enters_the_same_component() -> None:
    ids = np.arange(12, dtype=np.int32)
    primary = _fragment(0, 0, ids, ids, membership=1.0, mask_index=0)
    alternative = _fragment(1, 0, ids, ids, membership=0.7, mask_index=1)
    second = _fragment(2, 1, ids, ids)
    third = _fragment(3, 2, ids, ids)
    fragments = [primary, alternative, second, third]
    frames = [
        _frame(0, [primary, alternative], ids),
        _frame(1, [second], ids),
        _frame(2, [third], ids),
    ]

    result = associate_fragments_v10(
        fragments, frames, "P1", view_consensus=True
    )

    assert all(len(track.frame_ids) == len(set(track.frame_ids)) for track in result.tracks)
    assert not any({0, 1}.issubset(track.fragment_ids) for track in result.tracks)
    assert any(track.fragment_ids == (0, 2, 3) for track in result.tracks)
    assert any(track.fragment_ids == (1,) for track in result.tracks)


def test_cycle_supported_weak_edge_cannot_bridge_two_established_components() -> None:
    shared = np.arange(0, 5, dtype=np.int32)
    left_only = np.arange(5, 10, dtype=np.int32)
    right_only = np.arange(10, 15, dtype=np.int32)
    left_support = np.concatenate((shared, left_only))
    right_support = np.concatenate((shared, right_only))
    visible = np.arange(15, dtype=np.int32)
    fragments = [
        _fragment(0, 0, left_support, left_support),
        _fragment(1, 1, left_support, left_support),
        _fragment(2, 2, right_support, right_support),
        _fragment(3, 3, right_support, right_support),
    ]
    frames = [
        _frame(index, [fragment], visible)
        for index, fragment in enumerate(fragments)
    ]

    result = associate_fragments_v10(
        fragments, frames, "P1", view_consensus=True
    )

    assert any(edge.strong for edge in result.accepted_edges)
    assert any(
        edge.cycle_supported and not edge.strong for edge in result.tentative_edges
    )
    assert [track.fragment_ids for track in result.tracks] == [(0, 1), (2, 3)]


def test_r0_halo_can_attach_outside_member_union_but_r1_never_can() -> None:
    union = np.arange(3, dtype=np.int32)
    fragments = [
        _fragment(0, 0, union, union),
        _fragment(1, 1, union, union),
    ]
    frames = [
        _frame(0, [fragments[0]], union),
        _frame(1, [fragments[1]], union),
    ]
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]],
        dtype=np.float64,
    )
    affinity = np.ones((4, 1), dtype=np.float64)

    _, r0 = build_v10_candidate_bank(
        fragments,
        frames,
        4,
        pair_mode="P1",
        reconstruction_mode="R0",
        xyz_m=xyz,
        affinity=affinity,
    )
    _, r1 = build_v10_candidate_bank(
        fragments,
        frames,
        4,
        pair_mode="P1",
        reconstruction_mode="R1",
        xyz_m=xyz,
        affinity=affinity,
    )

    assert r0.full_ids[0].tolist() == [0, 1, 2, 3]
    assert r1.full_ids[0].tolist() == [0, 1, 2]


def test_r1_keeps_single_view_full_support_but_requires_two_views_for_core() -> None:
    common = np.arange(3, dtype=np.int32)
    first_full = np.arange(4, dtype=np.int32)
    first = _fragment(0, 0, first_full, common)
    second = _fragment(1, 1, common, common)
    frames = [
        _frame(0, [first], first_full),
        _frame(1, [second], first_full),
    ]

    _, bank = build_v10_candidate_bank(
        [first, second],
        frames,
        4,
        pair_mode="P1",
        reconstruction_mode="R1",
    )

    assert bank.core_ids[0].tolist() == [0, 1, 2]
    assert bank.full_ids[0].tolist() == [0, 1, 2, 3]


def test_r0_is_exact_v9_consensus_even_when_v10_margin_would_reject(
    monkeypatch,
) -> None:
    support = np.arange(3, dtype=np.int32)
    counts = SparseCounts(support, np.full(3, 2), 3)
    expected = ConsensusResult(
        core_track_id=np.zeros(3, dtype=np.int32),
        visible_views=np.full(3, 2, dtype=np.int32),
        assignment_margin=np.zeros(3, dtype=np.float32),
        valid_track_ids=(0,),
        positive_views={0: counts},
        conflict_views={0: SparseCounts(np.empty(0), np.empty(0), 3)},
        core_ids={0: support},
    )
    monkeypatch.setattr(v10_objectbank, "_v9_consensus", lambda *_args: expected)

    actual = _r0_consensus(
        V10Association("P0", (), (), (), ()), (), (), 3, V10Config()
    )

    assert actual is expected
    assert actual.core_track_id.tolist() == [0, 0, 0]
    assert np.all(actual.assignment_margin == 0.0)


def test_mv_label_casts_at_most_one_vote_per_view_and_abstains_below_iou_025() -> None:
    support = np.arange(3, dtype=np.int32)
    fragments = [
        _fragment(0, 0, support, support),
        _fragment(1, 1, support, support),
    ]
    association = V10Association(
        "P1",
        (),
        (),
        (),
        (V10Track(0, (0, 1), (0, 1), 1.0, "P1"),),
    )

    def arrays(primary: float, secondary: float) -> dict[str, np.ndarray]:
        return {
            "semantic_fragment_full_indptr": np.asarray(
                [0, 3, 6, 9, 12], dtype=np.int64
            ),
            "semantic_fragment_full_ids": np.tile(support, 4),
            "semantic_fragment_full_mass": np.concatenate(
                [
                    np.full(3, primary, dtype=np.float32),
                    np.full(3, secondary, dtype=np.float32),
                    np.full(3, primary, dtype=np.float32),
                    np.full(3, secondary, dtype=np.float32),
                ]
            ),
            "semantic_fragment_frame": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "semantic_fragment_class": np.asarray([0, 1, 0, 1], dtype=np.int32),
        }

    abstained = _multiview_classifications(
        association, fragments, arrays(0.24, 0.23), ("chair", "table")
    )[0]
    accepted = _multiview_classifications(
        association, fragments, arrays(0.30, 0.29), ("chair", "table")
    )[0]

    assert not abstained.eligible
    assert abstained.effective_view_count == 0
    assert accepted.eligible
    assert accepted.class_name == "chair"
    assert accepted.effective_view_count == 2


def test_frame_top8_uses_symmetric_union_and_positive_visibility_only() -> None:
    frames = []
    for frame_id in range(10):
        # Frame 9 has no co-visible Gaussian with the remaining frames.
        visible = np.asarray([100] if frame_id == 9 else [0, frame_id + 1], dtype=np.int32)
        frames.append(_frame(frame_id, [], visible))

    pairs = select_covisible_frame_pairs(frames, V10Config(frame_top_k=8))

    assert all(9 not in pair for pair in pairs)
    assert all(left < right for left, right in pairs)
    # The common Gaussian 0 makes every pair among frames 0..8 co-visible.
    assert (0, 8) in pairs


def test_association_and_r1_bank_are_identical_under_reversed_input_order() -> None:
    full = np.arange(20, dtype=np.int32)
    core = np.arange(5, dtype=np.int32)
    fragments = [
        _fragment(10, 0, full, core),
        _fragment(20, 1, full, core),
        _fragment(30, 2, full, core),
    ]
    frames = [
        _frame(index, [fragment], full)
        for index, fragment in enumerate(fragments)
    ]

    first_association, first_bank = build_v10_candidate_bank(
        fragments,
        frames,
        20,
        pair_mode="P1",
        reconstruction_mode="R1",
    )
    second_association, second_bank = build_v10_candidate_bank(
        list(reversed(fragments)),
        list(reversed(frames)),
        20,
        pair_mode="P1",
        reconstruction_mode="R1",
    )

    assert first_association == second_association
    assert first_bank.candidates == second_bank.candidates
    assert np.array_equal(first_bank.point_candidate_id, second_bank.point_candidate_id)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(first_bank.full_ids, second_bank.full_ids)
    )
    assert all(
        np.array_equal(left, right)
        for left, right in zip(first_bank.core_ids, second_bank.core_ids)
    )


def test_top_level_builder_preserves_two_classifiers_and_all_funnel_stages() -> None:
    support = np.arange(3, dtype=np.int32)
    visible = np.arange(4, dtype=np.int32)
    metadata = {
        "point_count": 4,
        "frame_count": 2,
        "classes": ["chair", "wall"],
    }
    arrays = {
        "fragment_id": np.asarray([0, 1], dtype=np.int32),
        "fragment_frame": np.asarray([0, 1], dtype=np.int32),
        "fragment_mask_index": np.asarray([0, 0], dtype=np.int32),
        "fragment_conflict_ratio": np.zeros(2, dtype=np.float32),
        "fragment_full_indptr": np.asarray([0, 3, 6], dtype=np.int64),
        "fragment_full_ids": np.tile(support, 2),
        "fragment_full_mass": np.ones(6, dtype=np.float32),
        "fragment_core_indptr": np.asarray([0, 3, 6], dtype=np.int64),
        "fragment_core_ids": np.tile(support, 2),
        "fragment_core_mass": np.ones(6, dtype=np.float32),
        "frame_visible_indptr": np.asarray([0, 4, 8], dtype=np.int64),
        "frame_visible_ids": np.tile(visible, 2),
        "frame_visible_mass": np.ones(8, dtype=np.float32),
        "frame_geometry_abstained": np.zeros(2, dtype=bool),
        "semantic_fragment_full_indptr": np.asarray([0, 3, 6], dtype=np.int64),
        "semantic_fragment_full_ids": np.tile(support, 2),
        "semantic_fragment_full_mass": np.ones(6, dtype=np.float32),
        "semantic_fragment_frame": np.asarray([0, 1], dtype=np.int32),
        "semantic_fragment_class": np.asarray([0, 0], dtype=np.int32),
        "xyz_m": np.asarray(
            [[0.00, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]],
            dtype=np.float32,
        ),
        "affinity": np.ones((4, 2), dtype=np.float32),
        "semantic": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (4, 1)),
        "label_features": np.eye(2, dtype=np.float32),
    }

    r0 = build_v10_object_bank(metadata, arrays, condition="P0R0")
    vc1 = build_v10_object_bank(metadata, arrays, condition="VC1")

    assert r0["full_ids"][0].tolist() == [0, 1, 2, 3]
    assert vc1["full_ids"][0].tolist() == [0, 1, 2]
    assert set(vc1["candidates"][0]["classifiers"]) == {"mv-label", "codebook"}
    assert vc1["candidates"][0]["branch_class"] == "chair"
    assert vc1["candidates"][0]["classifiers"]["mv-label"][
        "classification_eligible"
    ]
    assert tuple(vc1["stage_supports"]) == (
        "single_full",
        "single_core",
        "component_full_union",
        "component_core_union",
        "pre_conflict",
        "post_conflict",
        "unique_ownership",
        "final_candidate",
    )
    assert vc1["stage_supports"]["final_candidate"][0][
        "gaussian_ids"
    ].tolist() == [0, 1, 2]
    assert np.array_equal(
        vc1["stage_supports"]["pre_conflict"][0]["gaussian_ids"],
        vc1["stage_supports"]["post_conflict"][0]["gaussian_ids"],
    )
    assert vc1["accepted_edges"][0]["left_fragment_id"] == 0
    assert vc1["accepted_edges"][0]["right_fragment_id"] == 1
    assert vc1["accepted_edges"][0]["left_coverage"] == 1.0
    assert vc1["accepted_edges"][0]["right_coverage"] == 1.0
    assert vc1["accepted_edges"][0]["row_margin"] >= 0.10
    assert vc1["accepted_edges"][0]["column_margin"] >= 0.10
    assert vc1["accepted_edges"][0]["component_support_ratio"] == 1.0
    assert vc1["diagnostics"]["selected_classifier"] == "unselected"
    normalized, packed = v10_runner._normalise_builder_payload(
        vc1, point_count=4, condition="VC1"
    )
    assert normalized["candidates"][0]["classifiers"]["codebook"][
        "branch_class"
    ] == "chair"
    assert packed["stage_final_candidate_ids"].tolist() == [0, 1, 2]
