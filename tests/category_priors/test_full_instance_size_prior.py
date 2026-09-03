from __future__ import annotations

import copy
import inspect
import math

import numpy as np
import pytest

import category_priors.full_instance_size_prior as module
from category_priors.full_instance_size_prior import (
    FullInstanceCandidate,
    build_full_instance_candidates,
    evaluate_pre_vote,
    restore_selected_instances,
    score_full_instance_candidates,
    score_same_bank_size_priors,
    verify_same_bank_size_scores,
)
from category_priors.category_denoise import pca_sorted_extents_m
from category_priors.teacher_prior import SAGA20_CLASSES


CLASSES_32 = (
    "chair",
    "table",
    "plant",
    "flower",
    "foliage",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "wall",
    "floor",
    "ceiling",
    "person",
    "socket",
    "remote",
    "key",
    "book",
    "lighting",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "computer",
    "fan",
    "refrigerator",
    "robot",
    "cup",
    "vase",
    "phone",
    "trash can",
)


def _node(
    q25: tuple[float, float, float],
    q50: tuple[float, float, float],
    q75: tuple[float, float, float],
) -> dict[str, object]:
    geometry: dict[str, object] = {}
    for field, low, middle, high in zip(
        (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        ),
        q25,
        q50,
        q75,
    ):
        geometry[field] = {
            "q25": math.log(low),
            "q50": math.log(middle),
            "q75": math.log(high),
        }
    return {"shrunk": {"geometry": geometry}}


def _priors() -> dict[str, object]:
    return {
        "global": _node((0.5,) * 3, (1.0,) * 3, (2.0,) * 3),
        "categories": {
            "chair": _node((0.125,) * 3, (0.25,) * 3, (0.5,) * 3),
        },
    }


def _votes(index: int, count: float, background: float = 0.0) -> np.ndarray:
    votes = np.zeros(33, dtype=np.float64)
    votes[index] = count
    votes[32] = background
    return votes


def _candidate_row(
    raw_id: int,
    *,
    class_name: str = "chair",
    eligible: bool = True,
    extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
    members: tuple[int, ...] = (0, 1),
) -> dict[str, object]:
    class_index = CLASSES_32.index(class_name)
    histogram = _votes(class_index, 8.0, 2.0)
    return {
        "scene_id": "scene0000_00",
        "candidate_id": raw_id,
        "raw_instance_id": raw_id,
        "source": "global",
        "member_indices": np.asarray(members, dtype=np.int64),
        "point_count": len(members),
        "metric_extents_m": extents,
        "vote_histogram": tuple(histogram),
        "predicted_class_index": class_index,
        "predicted_class": class_name,
        "winner_ratio": 0.8,
        "background_ratio": 0.2,
        "eligible": eligible,
        "eligibility_reason": "eligible" if eligible else "background_not_lower",
        "Q": 0.8,
    }


def test_pre_vote_requires_unique_saga20_winner_and_all_channel_ratio() -> None:
    decision = evaluate_pre_vote(_votes(0, 8.0, 2.0), CLASSES_32)
    assert decision.eligible
    assert decision.predicted_class == "chair"
    assert decision.predicted_class_index == 0
    assert decision.Q == pytest.approx(0.8)
    assert decision.background_ratio == pytest.approx(0.2)

    exact_threshold = _votes(0, 3, 2)
    exact_threshold[1:4] = (2, 2, 1)
    exact = evaluate_pre_vote(exact_threshold, CLASSES_32)
    assert exact.eligible
    assert exact.Q == pytest.approx(0.30)

    tie = _votes(0, 4.0, 1.0)
    tie[1] = 4.0
    tied = evaluate_pre_vote(tie, CLASSES_32)
    assert not tied.eligible
    assert tied.predicted_class is None
    assert tied.eligibility_reason == "foreground_tie"

    outside = evaluate_pre_vote(_votes(CLASSES_32.index("flower"), 9, 1), CLASSES_32)
    assert not outside.eligible
    assert outside.eligibility_reason == "winner_not_saga20"

    low = evaluate_pre_vote(_votes(0, 2, 5), CLASSES_32)
    assert not low.eligible
    assert low.eligibility_reason == "winner_ratio_below_threshold"

    background_tie = _votes(0, 4, 4)
    background_tie[1] = 2
    background_tied = evaluate_pre_vote(background_tie, CLASSES_32)
    assert not background_tied.eligible
    assert background_tied.Q == pytest.approx(0.4)
    assert background_tied.eligibility_reason == "background_not_lower"


def test_pre_vote_rejects_invalid_histograms_and_zero_votes() -> None:
    with pytest.raises(ValueError, match="33 channels"):
        evaluate_pre_vote(np.zeros(32), CLASSES_32)
    invalid = np.zeros(33)
    invalid[0] = -1
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_pre_vote(invalid, CLASSES_32)
    empty = evaluate_pre_vote(np.zeros(33), CLASSES_32)
    assert not empty.eligible
    assert empty.eligibility_reason == "no_votes"
    assert empty.Q == 0.0


def test_snapshot_materializes_every_raw_instance_as_disjoint_full_mask() -> None:
    labels = np.asarray([-1, 2, 2, 7, 7, 7], dtype=np.int64)
    xyz = np.asarray(
        [
            [10.0, 10.0, 10.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    snapshot = build_full_instance_candidates(
        labels,
        xyz,
        0.5,
        {2: _votes(0, 9, 1), 7: _votes(3, 9, 1)},
        CLASSES_32,
        scene_id="scene0001_00",
        branch_instance_classes={7: "cup", 99: "unused"},
    )

    assert [candidate.raw_instance_id for candidate in snapshot.candidates] == [2, 7]
    first, second = snapshot.candidates
    assert isinstance(first, FullInstanceCandidate)
    assert first.source == "global"
    assert second.source == "other_classes"
    assert first.member_indices.tolist() == [1, 2]
    assert second.member_indices.tolist() == [3, 4, 5]
    assert not first.member_indices.flags.writeable
    assert first.metric_extents_m == pytest.approx((0.0, 0.0, 0.5))
    assert second.metric_extents_m == pytest.approx(
        pca_sorted_extents_m(xyz[[3, 4, 5]], 0.5)
    )
    assert first.eligible
    assert not second.eligible
    assert snapshot.diagnostics["foreground_point_count"] == 5
    assert snapshot.diagnostics["candidate_count"] == 2
    assert snapshot.diagnostics["unused_branch_instance_ids"] == (99,)
    assert "member_indices" not in snapshot.rows()[0]
    assert snapshot.rows(include_members=True)[0]["member_indices"].tolist() == [1, 2]


def test_snapshot_validates_partition_xyz_and_complete_vote_rows() -> None:
    labels = np.asarray([0, 0], dtype=np.int64)
    xyz = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="missing raw instance"):
        build_full_instance_candidates(
            labels, xyz, 1.0, {}, CLASSES_32, scene_id="scene"
        )
    with pytest.raises(TypeError, match="integer labels"):
        build_full_instance_candidates(
            labels.astype(np.float64),
            xyz,
            1.0,
            {0: _votes(0, 1)},
            CLASSES_32,
            scene_id="scene",
        )
    with pytest.raises(ValueError, match="shape"):
        build_full_instance_candidates(
            labels,
            np.zeros((3, 3)),
            1.0,
            {0: _votes(0, 1)},
            CLASSES_32,
            scene_id="scene",
        )


def test_q_global_and_class_scores_share_exact_candidate_bank() -> None:
    candidate = _candidate_row(4)
    original = copy.deepcopy(candidate)
    scores = score_same_bank_size_priors((candidate,), _priors())

    assert scores.q_only[0]["G"] == 1.0
    assert scores.q_only[0]["S"] == pytest.approx(0.8)
    assert scores.global_size[0]["G"] == pytest.approx(1.0)
    assert scores.global_size[0]["S"] == pytest.approx(0.8)
    assert scores.class_size[0]["G"] == pytest.approx(math.exp(-0.5))
    assert scores.class_size[0]["S"] == pytest.approx(0.8 * math.exp(-0.5))
    assert scores.class_size[0]["size_lookup_class"] == "chair"
    assert not scores.class_size[0]["size_fallback_global"]
    assert scores.identity.raw_instance_ids == (4,)
    assert scores.identity.member_point_count == 2
    assert np.array_equal(candidate["member_indices"], original["member_indices"])
    assert set(candidate) == set(original)


def test_missing_class_falls_back_global_but_malformed_class_does_not() -> None:
    missing = _candidate_row(1, class_name="book")
    scored = score_full_instance_candidates((missing,), _priors(), "class-size")[0]
    assert scored["G"] == pytest.approx(1.0)
    assert scored["size_lookup_class"] == "global"
    assert scored["size_fallback_global"]

    malformed = _priors()
    malformed["categories"]["chair"] = {}  # type: ignore[index]
    with pytest.raises(TypeError, match="has no shrunk node"):
        score_full_instance_candidates(
            (_candidate_row(2),), malformed, "class-size"
        )


def test_ineligible_candidate_is_retained_but_prior_cannot_change_it() -> None:
    candidate = _candidate_row(3, eligible=False)
    scores = score_same_bank_size_priors((candidate,), _priors())
    for row in (scores.q_only[0], scores.global_size[0], scores.class_size[0]):
        assert not row["eligible"]
        assert not row["prior_applied"]
        assert row["G"] == 1.0
        assert row["S"] == pytest.approx(row["Q"])
        assert row["size_lookup_class"] is None


def test_same_bank_audit_detects_q_mask_or_base_field_changes() -> None:
    candidates = (_candidate_row(1), _candidate_row(2, members=(2, 3)))
    left = score_full_instance_candidates(candidates, _priors(), "global-size")
    right = list(score_full_instance_candidates(candidates, _priors(), "class-size"))
    assert verify_same_bank_size_scores(left, right).bank_identity_equal

    changed_q = copy.deepcopy(right)
    changed_q[0]["Q"] = 0.7
    with pytest.raises(ValueError, match="bank identity differs"):
        verify_same_bank_size_scores(left, changed_q)

    changed_mask = copy.deepcopy(right)
    changed_mask[0]["member_indices"] = np.asarray([0, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="bank identity differs"):
        verify_same_bank_size_scores(left, changed_mask)


def test_restore_selected_full_masks_is_order_independent_and_audited() -> None:
    post_filter = np.asarray([-1, 9, 9, 2, -1, 8], dtype=np.int64)
    candidates = (
        _candidate_row(2, members=(0, 1)),
        _candidate_row(7, members=(3, 4)),
        _candidate_row(8, eligible=False, members=(5,)),
    )
    forward = restore_selected_instances(post_filter, candidates, (2, 7))
    reverse = restore_selected_instances(post_filter, candidates, (7, 2))

    assert forward.point_labels.tolist() == [2, 2, 9, 7, 7, 8]
    assert np.array_equal(forward.point_labels, reverse.point_labels)
    assert not forward.point_labels.flags.writeable
    assert np.array_equal(post_filter, [-1, 9, 9, 2, -1, 8])
    assert forward.diagnostics["selected_raw_instance_ids"] == (2, 7)
    assert forward.diagnostics["changed_point_count"] == 4
    assert forward.diagnostics["restored_from_background_count"] == 2
    assert forward.diagnostics["overwritten_other_foreground_count"] == 2
    assert forward.diagnostics["outside_selected_changed_count"] == 0
    assert forward.diagnostics["candidate_members_mutually_exclusive"]
    assert forward.diagnostics["selection_order_independent"]

    empty = restore_selected_instances(post_filter, candidates, ())
    assert np.array_equal(empty.point_labels, post_filter)
    assert empty.diagnostics["changed_point_count"] == 0


def test_restore_rejects_ineligible_unknown_overlapping_and_invalid_masks() -> None:
    labels = np.asarray([-1, -1, -1], dtype=np.int64)
    eligible = _candidate_row(1, members=(0, 1))
    ineligible = _candidate_row(2, eligible=False, members=(2,))
    with pytest.raises(ValueError, match="ineligible"):
        restore_selected_instances(labels, (eligible, ineligible), (2,))
    with pytest.raises(ValueError, match="absent from bank"):
        restore_selected_instances(labels, (eligible,), (99,))

    overlapping = _candidate_row(2, members=(1, 2))
    with pytest.raises(ValueError, match="overlaps"):
        restore_selected_instances(labels, (eligible, overlapping), (1,))

    duplicate = _candidate_row(1, members=(0, 0))
    with pytest.raises(ValueError, match="duplicate member"):
        restore_selected_instances(labels, (duplicate,), (1,))


def test_core_module_has_no_annotation_or_branch_confidence_input() -> None:
    for function in (
        build_full_instance_candidates,
        score_full_instance_candidates,
        restore_selected_instances,
    ):
        names = set(inspect.signature(function).parameters)
        assert not any(name.startswith("gt") for name in names)
        assert "assignment_confidence" not in names
    assert "assignment_confidence" not in inspect.getsource(module)
    assert set(SAGA20_CLASSES).issubset(CLASSES_32)
