from __future__ import annotations

from dataclasses import replace

import pytest

from category_priors.category_candidate_prior_evaluation import (
    DEV2_SCENE_IDS,
    DEV8_SCENE_IDS,
    CandidatePriorExample,
    OfficialCandidateGroundTruth,
    binary_average_precision,
    candidate_prior_mechanical_effect,
    evaluate_candidate_prior_dev8,
    join_candidate_prior_rows,
    select_uniform_threshold_dev2,
)


def _example(
    scene_id: str,
    candidate_id: int,
    *,
    branch_class: str = "chair",
    uniform_score: float = 0.3,
    class_score: float = 0.3,
    uniform_support_pass: bool = True,
    class_support_pass: bool = True,
    iou: float = 0.5,
    instance_id: int | None = 10,
) -> CandidatePriorExample:
    return CandidatePriorExample(
        scene_id=scene_id,
        candidate_id=candidate_id,
        branch_class=branch_class,
        core_point_count=5,
        uniform_score=uniform_score,
        class_score=class_score,
        uniform_support_pass=uniform_support_pass,
        class_support_pass=class_support_pass,
        same_class_iou=iou,
        matched_gt_class=branch_class if instance_id is not None else None,
        matched_gt_instance_id=instance_id,
        matched_gt_size_bin="tiny" if instance_id is not None else None,
        q_value=0.4,
    )


def test_join_requires_same_frozen_candidate_q_and_derives_support_pass() -> None:
    candidate = {
        "scene_id": "scene0001_00",
        "candidate_id": 0,
        "branch_class": "chair",
        "branch_class_index": 0,
        "core_point_count": 4,
        "full_point_count": 8,
        "Q": 0.4,
    }
    uniform = {
        "scene_id": "scene0001_00",
        "candidate_id": 0,
        "branch_class": "chair",
        "branch_class_index": 0,
        "core_point_count": 4,
        "full_point_count": 8,
        "Q": 0.4,
        "score": 0.2,
        "support_threshold": 5,
    }
    per_class = {
        **uniform,
        "score": 0.3,
        "support_threshold": 3,
    }
    label = {
        "scene_id": "scene0001_00",
        "candidate_id": 0,
        "full_best_same_class_iou": 0.5,
        "full_best_same_class_gt_class": "chair",
        "full_best_same_class_gt_instance": 10,
        "matched_gt_size_bin": "tiny",
    }

    rows = join_candidate_prior_rows(
        candidate_rows=(candidate,),
        uniform_score_rows=(uniform,),
        class_score_rows=(per_class,),
        label_rows=(label,),
    )

    assert len(rows) == 1
    assert rows[0].uniform_support_pass is False
    assert rows[0].class_support_pass is True
    assert rows[0].uniform_score == 0.2
    assert rows[0].class_score == 0.3
    assert rows[0].matched_gt_key == ("scene0001_00", "chair", 10)

    changed_q = {**per_class, "Q": 0.400002}
    with pytest.raises(ValueError, match="frozen candidate Q"):
        join_candidate_prior_rows(
            candidate_rows=(candidate,),
            uniform_score_rows=(uniform,),
            class_score_rows=(changed_q,),
            label_rows=(label,),
        )

    changed_core = {**per_class, "core_point_count": 5}
    with pytest.raises(ValueError, match="core point count"):
        join_candidate_prior_rows(
            candidate_rows=(candidate,),
            uniform_score_rows=(uniform,),
            class_score_rows=(changed_core,),
            label_rows=(label,),
        )


def test_binary_ap_groups_score_ties_without_candidate_order_artifacts() -> None:
    first = binary_average_precision([0.9, 0.8, 0.8], [True, True, False])
    reversed_tie = binary_average_precision(
        [0.9, 0.8, 0.8], [True, False, True]
    )

    assert first == pytest.approx(5 / 6)
    assert reversed_tie == pytest.approx(first)
    assert binary_average_precision([0.5, 0.4], [False, False]) == 0.0


def test_binary_ap_support_gate_keeps_rejected_positive_in_denominator() -> None:
    # A hard support gate makes the first positive unrecoverable; it must not
    # quietly shrink the positive universe and inflate candidate AP.
    value = binary_average_precision(
        [0.9, 0.8, 0.7],
        [True, True, False],
        eligible=[False, True, True],
    )
    assert value == pytest.approx(0.5)


def test_dev2_threshold_uses_uniform_only_and_exact_tie_chooses_higher() -> None:
    rows = tuple(
        _example(
            scene_id,
            0,
            uniform_score=0.3,
            class_score=0.0 if index == 0 else 1.0,
        )
        for index, scene_id in enumerate(DEV2_SCENE_IDS)
    )

    selection = select_uniform_threshold_dev2(rows)

    assert selection.selected_threshold == 0.25
    assert selection.score_source == "uniform"
    assert all(
        row["scene_equal_candidate_f1_025"] == 1.0
        for row in selection.grid_rows
    )


def test_mechanical_support_alternative_requires_count_class_and_scene() -> None:
    rows = (
        _example(
            "scene-a",
            0,
            branch_class="chair",
            uniform_support_pass=False,
            class_support_pass=True,
        ),
        _example(
            "scene-a",
            1,
            branch_class="table",
            uniform_support_pass=False,
            class_support_pass=True,
        ),
        _example(
            "scene-b",
            0,
            branch_class="chair",
            uniform_support_pass=False,
            class_support_pass=True,
        ),
        _example(
            "scene-b",
            1,
            branch_class="table",
            uniform_support_pass=False,
            class_support_pass=True,
        ),
        _example(
            "scene-b",
            2,
            branch_class="chair",
            uniform_support_pass=False,
            class_support_pass=True,
        ),
    )

    result = candidate_prior_mechanical_effect(rows)

    assert result["score_change_gate_passed"] is False
    assert result["support_changed_candidate_count"] == 5
    assert result["support_changed_class_count"] == 2
    assert result["support_changed_scene_count"] == 2
    assert result["support_change_gate_passed"] is True
    assert result["mechanically_effective"] is True

    one_class = tuple(
        replace(row, branch_class="chair", matched_gt_class="chair") for row in rows
    )
    assert candidate_prior_mechanical_effect(one_class)[
        "support_change_gate_passed"
    ] is False


def _passing_dev8() -> tuple[
    tuple[CandidatePriorExample, ...],
    tuple[OfficialCandidateGroundTruth, ...],
]:
    examples: list[CandidatePriorExample] = []
    ground_truth: list[OfficialCandidateGroundTruth] = []
    for index, scene_id in enumerate(DEV8_SCENE_IDS):
        improved = index < 5
        examples.append(
            _example(
                scene_id,
                0,
                uniform_score=0.4 if improved else 0.8,
                class_score=0.8,
                iou=0.5,
                instance_id=10,
            )
        )
        examples.append(
            _example(
                scene_id,
                1,
                uniform_score=0.6 if improved else 0.2,
                class_score=0.2,
                iou=0.0,
                instance_id=None,
            )
        )
        ground_truth.append(
            OfficialCandidateGroundTruth(
                scene_id=scene_id,
                class_name="chair",
                instance_id=10,
                size_bin="tiny",
            )
        )
    # A duplicate positive candidate must increase candidate TP, but it must
    # not recover the first scene's sole GT object twice.
    examples.append(
        _example(
            DEV8_SCENE_IDS[0],
            2,
            uniform_score=0.35,
            class_score=0.7,
            iou=0.5,
            instance_id=10,
        )
    )
    return tuple(examples), tuple(ground_truth)


def test_dev8_evaluation_is_scene_equal_paired_and_threshold_free() -> None:
    examples, ground_truth = _passing_dev8()

    result = evaluate_candidate_prior_dev8(
        examples=examples,
        official_gt=ground_truth,
    )
    analysis = result.analysis

    assert analysis["scene_count"] == 8
    assert analysis["candidate_count"] == 17
    assert analysis["candidate_ap"]["aggregation"] == "scene_equal"
    assert analysis["candidate_ap"]["delta_025"] > 0.002
    assert analysis["candidate_ap"]["delta_050"] > -0.002
    assert analysis["positive_scene_count_ap_025"] == 5
    assert analysis["mechanical_effect"]["score_change_gate_passed"] is True
    assert analysis["support_only_acceptance"]["uniform_025"][
        "true_positive_count"
    ] == 9
    assert analysis["tiny_small_recall"]["uniform_025"][
        "recovered_gt_count"
    ] == 8
    assert analysis["tiny_small_recall"]["class_025"][
        "recovered_gt_count"
    ] == 8
    assert analysis["fp_tp_gate"]["passed"] is True
    assert analysis["passed"] is True
    assert analysis["conclusion_boundary"]["DEV8_used_for_threshold_selection"] is False


def test_threshold_free_acceptance_and_tiny_recall_require_support() -> None:
    examples, ground_truth = _passing_dev8()
    unsupported = tuple(
        replace(row, class_support_pass=False)
        if row.scene_id == DEV8_SCENE_IDS[0]
        and row.matched_gt_instance_id == 10
        else row
        for row in examples
    )

    analysis = evaluate_candidate_prior_dev8(
        examples=unsupported,
        official_gt=ground_truth,
    ).analysis

    assert analysis["support_only_acceptance"]["class_025"][
        "true_positive_count"
    ] == 7
    assert analysis["tiny_small_recall"]["class_025"][
        "recovered_gt_count"
    ] == 7
    assert analysis["gates"]["tiny_small_recall_025_non_degradation"] is False


def test_dev8_has_no_threshold_argument_or_threshold_in_analysis() -> None:
    examples, ground_truth = _passing_dev8()
    analysis = evaluate_candidate_prior_dev8(
        examples=examples,
        official_gt=ground_truth,
    ).analysis
    assert analysis["acceptance_threshold"] is None
    assert analysis["conclusion_boundary"]["threshold_selected_before_this_gate"] is False
