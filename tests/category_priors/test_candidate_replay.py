from __future__ import annotations

import inspect

import numpy as np
import pytest

from category_priors.candidate_replay import (
    candidate_bank_to_replay_candidates,
    candidate_export_ids,
    LegacyReplayCandidate,
    replay_candidates_through_legacy,
)
from category_priors.candidate_bank import CandidateBank
from category_priors.legacy_candidate_replay import legacy_knn_filter
from category_priors.prediction_contract import normalize_prediction


def _xyz(count: int) -> np.ndarray:
    return np.column_stack(
        (np.arange(count, dtype=np.float64), np.zeros(count), np.zeros(count))
    )


def _candidate(
    candidate_id: int,
    full: list[int],
    core: list[int],
    *,
    q: float = 0.8,
    branch_class: str = "chair",
) -> LegacyReplayCandidate:
    return LegacyReplayCandidate(
        candidate_id=candidate_id,
        branch_class=branch_class,
        q_score=q,
        full_point_indices=np.asarray(full, dtype=np.int64),
        trusted_core_indices=np.asarray(core, dtype=np.int64),
    )


def test_zero_accepted_candidates_is_pointwise_exact_b0() -> None:
    xyz = _xyz(7)
    global_labels = np.asarray([4, 4, 4, -1, 9, 9, -1], dtype=np.int64)
    candidates = (_candidate(8, [0, 1, 2], [0, 1]),)
    baseline = legacy_knn_filter(xyz, global_labels, k=3, min_count=2)

    result = replay_candidates_through_legacy(
        xyz_scene=xyz,
        global_pre_knn=global_labels,
        candidates=candidates,
        accepted_candidate_ids=(),
        k=3,
        min_count=2,
    )

    np.testing.assert_array_equal(result.source_labels, global_labels)
    np.testing.assert_array_equal(result.after_knn, baseline.after_knn)
    np.testing.assert_array_equal(result.after_filter, baseline.after_filter)
    assert result.diagnostics["zero_candidate_b0_pointwise_exact"] is True
    assert result.diagnostics["protected_or_reinserted_point_count"] == 0
    row = result.candidates[0]
    assert not row.accepted
    assert row.raw_label is None
    assert row.pre_knn_owned_count == row.post_knn_total_count == 0


def test_conflicts_use_q_then_total_core_count_then_stable_candidate_id() -> None:
    # Points 0/1: candidate 10 wins despite candidate 3 being core at point 1,
    # because Q is the primary key.  Point 3 is an all-else tie won by the
    # smaller stable ID 5.  Point 4 is a Q tie won by candidate 9 because its
    # candidate-level trusted core contains more points, even though point 4
    # itself is core only for candidate 8.
    candidates = (
        _candidate(10, [0, 1], [0], q=0.8, branch_class="chair"),
        _candidate(3, [0, 1, 2], [1, 2], q=0.7, branch_class="table"),
        _candidate(7, [3], [], q=0.8, branch_class="sofa"),
        _candidate(5, [3], [], q=0.8, branch_class="cabinet"),
        _candidate(8, [4], [4], q=0.8, branch_class="door"),
        _candidate(9, [4, 5, 6], [5, 6], q=0.8, branch_class="window"),
    )

    result = replay_candidates_through_legacy(
        xyz_scene=_xyz(7),
        global_pre_knn=np.full(7, -1, dtype=np.int64),
        candidates=candidates,
        accepted_candidate_ids=(10, 3, 7, 5, 8, 9),
        k=1,
        min_count=1,
    )

    # Candidate raw labels are allocated above global labels in sorted stable
    # candidate-ID order: 3->0, 5->1, 7->2, 8->3, 9->4, 10->5.
    assert dict(result.candidate_raw_labels) == {
        3: 0,
        5: 1,
        7: 2,
        8: 3,
        9: 4,
        10: 5,
    }
    np.testing.assert_array_equal(result.source_labels, [5, 5, 0, 1, 4, 4, 4])
    np.testing.assert_array_equal(result.after_filter, result.source_labels)
    by_id = {row.candidate_id: row for row in result.candidates}
    assert by_id[10].pre_knn_owned_count == 2
    assert by_id[3].pre_knn_conflict_lost_count == 2
    assert by_id[7].pre_knn_conflict_lost_count == 1
    assert by_id[8].pre_knn_conflict_lost_count == 1
    assert by_id[9].pre_knn_owned_trusted_core_count == 2


def test_accepted_candidates_are_not_protected_or_reinserted() -> None:
    xyz = _xyz(10)
    global_labels = np.asarray([7, 7, 7] + [-1] * 7, dtype=np.int64)
    candidate = _candidate(0, [0, 1, 2], [0, 1, 2], q=0.4)

    result = replay_candidates_through_legacy(
        xyz_scene=xyz,
        global_pre_knn=global_labels,
        candidates=(candidate,),
        accepted_candidate_ids=(0,),
        k=10,
        min_count=1,
    )

    raw_label = result.candidate_raw_labels[0]
    expected = legacy_knn_filter(xyz, result.source_labels, k=10, min_count=1)
    np.testing.assert_array_equal(result.after_knn, expected.after_knn)
    np.testing.assert_array_equal(result.after_filter, expected.after_filter)
    assert not np.any(result.after_filter == raw_label)
    row = result.candidates[0]
    assert row.pre_knn_owned_count == 3
    assert row.post_knn_total_count == 0
    assert not row.survived_post_filter
    assert row.post_filter_raw_label is None


def test_surviving_branch_carries_only_branch_hint_before_final_vote() -> None:
    candidate = _candidate(
        4, [0, 1, 2], [0, 1], q=0.625, branch_class="toilet"
    )
    result = replay_candidates_through_legacy(
        xyz_scene=_xyz(4),
        global_pre_knn=np.asarray([-1, -1, -1, 12], dtype=np.int64),
        candidates=(candidate,),
        accepted_candidate_ids=(4,),
        k=1,
        min_count=1,
    )

    raw_label = result.candidate_raw_labels[4]
    assert result.candidate_branch_class_by_raw_label[raw_label] == "toilet"
    assert result.candidate_q_by_raw_label[raw_label] == pytest.approx(0.625)
    row = result.candidates[0]
    assert row.post_filter_raw_label == raw_label
    assert row.branch_class_hint == "toilet"
    assert result.diagnostics["secondary_class_vote_applied"] is False


def test_survival_diagnostics_separate_retention_from_knn_growth() -> None:
    candidate = _candidate(0, [0, 1], [0], q=0.9)
    result = replay_candidates_through_legacy(
        xyz_scene=_xyz(3),
        global_pre_knn=np.full(3, -1, dtype=np.int64),
        candidates=(candidate,),
        accepted_candidate_ids=(0,),
        k=3,
        min_count=1,
    )

    row = result.candidates[0]
    assert row.pre_knn_owned_count == 2
    assert row.post_knn_retained_owned_count == 2
    assert row.post_knn_gained_outside_count == 1
    assert row.post_filter_total_count == 3
    assert row.survived_post_filter


def test_candidate_contract_rejects_core_outside_full_and_unknown_acceptance() -> None:
    invalid = _candidate(0, [0, 1], [2])
    with pytest.raises(ValueError, match="trusted core is not a subset"):
        replay_candidates_through_legacy(
            xyz_scene=_xyz(3),
            global_pre_knn=np.full(3, -1),
            candidates=(invalid,),
            accepted_candidate_ids=(0,),
            k=1,
            min_count=1,
        )

    valid = _candidate(0, [0, 1], [0])
    with pytest.raises(ValueError, match="unknown IDs"):
        replay_candidates_through_legacy(
            xyz_scene=_xyz(3),
            global_pre_knn=np.full(3, -1),
            candidates=(valid,),
            accepted_candidate_ids=(99,),
            k=1,
            min_count=1,
        )


def test_replay_is_deterministic_read_only_and_has_no_gt_or_prior_interface() -> None:
    xyz = _xyz(4)
    global_labels = np.asarray([-1, -1, 2, 2], dtype=np.int64)
    candidate = _candidate(2, [0, 1], [0], q=0.8)
    kwargs = dict(
        xyz_scene=xyz,
        global_pre_knn=global_labels,
        candidates=(candidate,),
        accepted_candidate_ids=(2,),
        k=2,
        min_count=1,
    )

    first = replay_candidates_through_legacy(**kwargs)
    second = replay_candidates_through_legacy(**kwargs)

    for name in ("source_labels", "after_knn", "after_filter"):
        left = getattr(first, name)
        right = getattr(second, name)
        np.testing.assert_array_equal(left, right)
        assert left.flags.writeable is False
    assert first.candidates == second.candidates
    assert dict(first.diagnostics) == dict(second.diagnostics)
    assert xyz.flags.writeable and global_labels.flags.writeable

    parameter_names = set(
        inspect.signature(replay_candidates_through_legacy).parameters
    )
    assert not any(
        forbidden in name.lower()
        for name in parameter_names
        for forbidden in ("gt", "prior", "semantic", "vote", "protected")
    )


def _bank_with_reassigned_raw_core() -> CandidateBank:
    class_names = ("chair",) + tuple(f"unused-{index}" for index in range(31))
    return CandidateBank(
        class_names=class_names,
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.full(6, -1, dtype=np.int64),
        semantic_top1=np.zeros(6, dtype=np.int64),
        semantic_top1_score=np.full(6, 0.8),
        branch_full_labels=np.asarray([0, 0, 0, 1, 1, 1]),
        branch_core_labels=np.asarray([0, 0, 1, 1, 1, 0]),
        assignment_confidence=np.full(6, 0.7),
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "branch_class_index": 0,
                "full_point_count": 3,
                "core_point_count": 3,
                "base_score": 0.6,
            },
            {
                "candidate_id": 1,
                "branch_class": "chair",
                "branch_class_index": 0,
                "full_point_count": 3,
                "core_point_count": 3,
                "base_score": 0.5,
            },
        ),
        diagnostics={},
    )


def test_bank_adapter_intersects_raw_core_with_full_and_records_loss() -> None:
    converted = candidate_bank_to_replay_candidates(_bank_with_reassigned_raw_core())
    assert len(converted.candidates) == 2
    np.testing.assert_array_equal(converted.candidates[0].full_point_indices, [0, 1, 2])
    np.testing.assert_array_equal(converted.candidates[0].trusted_core_indices, [0, 1])
    np.testing.assert_array_equal(converted.candidates[1].trusted_core_indices, [3, 4])
    assert converted.diagnostics["raw_core_outside_full_count"] == 2
    assert converted.diagnostics["candidates_with_raw_core_outside_full"] == 2
    assert converted.candidates[0].q_score == pytest.approx(0.6)
    assert not converted.candidates[0].full_point_indices.flags.writeable


def test_bank_adapter_requires_frozen_vote_score() -> None:
    bank = _bank_with_reassigned_raw_core()
    rows = [dict(row) for row in bank.candidates]
    rows[0].pop("base_score")
    invalid = CandidateBank(**{**bank.__dict__, "candidates": tuple(rows)})
    with pytest.raises(ValueError, match="attach 2D vote evidence"):
        candidate_bank_to_replay_candidates(invalid)


def test_candidate_to_raw_to_export_chain_is_explicit() -> None:
    converted = candidate_bank_to_replay_candidates(_bank_with_reassigned_raw_core())
    replay = replay_candidates_through_legacy(
        xyz_scene=_xyz(6),
        global_pre_knn=np.full(6, -1, dtype=np.int64),
        candidates=converted.candidates,
        accepted_candidate_ids=(1,),
        k=1,
        min_count=1,
    )
    raw_id = replay.candidate_raw_labels[1]
    contracted = normalize_prediction(
        replay.after_filter,
        {raw_id: {"class": "chair", "score": 0.75}},
    )
    assert dict(candidate_export_ids(replay, contracted)) == {1: 0}
