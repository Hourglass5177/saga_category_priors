from __future__ import annotations

import numpy as np
import pytest

from category_priors.evaluation_strata import load_evaluation_strata
from category_priors.recheck_evaluation import (
    BranchPrediction,
    GroundTruthObject,
    aggregate_rescue_scenes,
    evaluate_rescue_scene,
    match_one_to_one,
)


def _mask(size: int, *indices: int) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    result[list(indices)] = True
    return result


def _gt(
    gt_id: int,
    class_name: str,
    indices: tuple[int, ...],
    *,
    size: int = 32,
    diagonal_m: float = 0.4,
) -> GroundTruthObject:
    return GroundTruthObject(
        gt_id=gt_id,
        class_name=class_name,
        mask=_mask(size, *indices),
        bbox_diagonal_m=diagonal_m,
    )


def _prediction(
    prediction_id: int,
    class_name: str,
    indices: tuple[int, ...],
    *,
    size: int = 32,
    score: float = 0.8,
    source_candidate_id: int | None = None,
) -> BranchPrediction:
    return BranchPrediction(
        prediction_id=prediction_id,
        class_name=class_name,
        score=score,
        mask=_mask(size, *indices),
        source_candidate_id=(
            prediction_id if source_candidate_id is None else source_candidate_id
        ),
    )


def _evaluate(
    *,
    scene_id: str = "scene0000_00",
    b0: tuple[BranchPrediction, ...] = (),
    branch: tuple[BranchPrediction, ...] = (),
    ground_truth: tuple[GroundTruthObject, ...],
    threshold: float = 0.25,
):
    return evaluate_rescue_scene(
        scene_id=scene_id,
        b0_predictions=b0,
        branch_predictions=branch,
        ground_truth=ground_truth,
        strata=load_evaluation_strata(),
        iou_threshold=threshold,
    )


def test_one_to_one_matching_maximizes_cardinality_before_total_iou() -> None:
    # P1 has the single strongest edge to G1.  Greedily taking it would leave
    # P2 unmatched.  The required solution instead uses P2-G1 and P1-G2 so
    # that two objects, rather than one, are matched.
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3)),
        _gt(2, "chair", (4, 5, 6, 7)),
    )
    predictions = (
        _prediction(10, "chair", (0, 1, 2, 3, 4, 5, 6)),
        _prediction(11, "chair", (0, 1, 8, 9)),
    )

    result = match_one_to_one(predictions, ground_truth, iou_threshold=0.25)

    assert {(row.prediction_id, row.gt_id) for row in result.matches} == {
        (10, 2),
        (11, 1),
    }
    assert result.unmatched_prediction_ids == ()
    assert result.unmatched_gt_ids == ()


def test_one_to_one_matching_maximizes_total_iou_after_cardinality() -> None:
    ground_truth = (_gt(1, "chair", (0, 1, 2, 3)),)
    predictions = (
        _prediction(10, "chair", (0, 1, 2, 3)),
        _prediction(11, "chair", (0, 1, 2, 4)),
    )

    result = match_one_to_one(predictions, ground_truth, iou_threshold=0.25)

    assert len(result.matches) == 1
    assert result.matches[0].prediction_id == 10
    assert result.matches[0].gt_id == 1
    assert result.matches[0].iou == pytest.approx(1.0)
    assert result.unmatched_prediction_ids == (11,)


def test_matching_uses_strictly_greater_than_iou_threshold() -> None:
    ground_truth = (_gt(1, "chair", (0, 1)),)
    predictions = (_prediction(10, "chair", (0, 1, 2, 3)),)

    at_threshold = match_one_to_one(predictions, ground_truth, iou_threshold=0.50)
    below_threshold = match_one_to_one(predictions, ground_truth, iou_threshold=0.499)

    assert at_threshold.matches == ()
    assert at_threshold.unmatched_prediction_ids == (10,)
    assert at_threshold.unmatched_gt_ids == (1,)
    assert [(row.prediction_id, row.gt_id) for row in below_threshold.matches] == [
        (10, 1)
    ]


def test_duplicate_branch_predictions_can_rescue_only_one_gt() -> None:
    ground_truth = (_gt(1, "chair", (0, 1, 2, 3)),)
    branch = (
        _prediction(10, "chair", (0, 1, 2, 3), score=0.9),
        _prediction(11, "chair", (0, 1, 2, 3), score=0.8),
    )

    result = _evaluate(ground_truth=ground_truth, branch=branch)
    overall = result["strata"]["overall"]

    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 0)
    assert overall["precision"] == pytest.approx(0.5)
    assert overall["recall"] == pytest.approx(1.0)
    assert overall["f1"] == pytest.approx(2.0 / 3.0)


def test_branch_prediction_of_b0_hit_is_classified_as_duplicate_b0() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3)),
        _gt(2, "chair", (4, 5, 6, 7)),
    )
    b0 = (_prediction(1, "chair", (0, 1, 2, 3)),)
    branch = (_prediction(10, "chair", (0, 1, 2, 3)),)

    result = _evaluate(ground_truth=ground_truth, b0=b0, branch=branch)
    overall = result["strata"]["overall"]

    assert (overall["tp"], overall["fp"], overall["fn"]) == (0, 1, 1)
    assert overall["duplicate_b0"] == 1
    assert overall["ignored"] == 0


def test_small_stratum_ignores_correct_hit_on_medium_or_large_object() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3), diagonal_m=0.4),
        _gt(2, "chair", (4, 5, 6, 7), diagonal_m=1.4),
    )
    branch = (_prediction(10, "chair", (4, 5, 6, 7)),)

    result = _evaluate(ground_truth=ground_truth, branch=branch)

    assert (
        result["strata"]["overall"]["tp"],
        result["strata"]["overall"]["fn"],
    ) == (1, 1)
    small = result["strata"]["small"]
    assert (small["tp"], small["fp"], small["fn"], small["ignored"]) == (
        0,
        0,
        1,
        1,
    )


def test_small_stratum_ignores_only_one_duplicate_large_prediction() -> None:
    ground_truth = (
        _gt(1, "chair", (0, 1, 2, 3), diagonal_m=0.4),
        _gt(2, "chair", (4, 5, 6, 7), diagonal_m=1.4),
    )
    branch = (
        _prediction(10, "chair", (4, 5, 6, 7)),
        _prediction(11, "chair", (4, 5, 6, 7)),
    )

    small = _evaluate(ground_truth=ground_truth, branch=branch)["strata"]["small"]

    assert (small["tp"], small["fp"], small["fn"], small["ignored"]) == (
        0,
        1,
        1,
        1,
    )
    assert small["other_fp"] == 1


def test_zero_denominators_are_reported_as_undefined() -> None:
    result = _evaluate(ground_truth=(), branch=())
    overall = result["strata"]["overall"]

    assert overall["precision"] is None
    assert overall["recall"] is None
    assert overall["f1"] is None


def test_tail_and_small_tail_strata_use_class_and_instance_definitions() -> None:
    ground_truth = (
        _gt(1, "socket", (0, 1, 2, 3), diagonal_m=0.3),
        _gt(2, "socket", (4, 5, 6, 7), diagonal_m=1.2),
        _gt(3, "chair", (8, 9, 10, 11), diagonal_m=0.3),
        _gt(4, "chair", (12, 13, 14, 15), diagonal_m=1.2),
    )
    branch = (
        _prediction(10, "socket", (0, 1, 2, 3)),
        _prediction(11, "socket", (4, 5, 6, 7)),
        _prediction(12, "chair", (8, 9, 10, 11)),
    )

    result = _evaluate(ground_truth=ground_truth, branch=branch)

    assert (
        result["strata"]["overall"]["tp"],
        result["strata"]["overall"]["fn"],
    ) == (3, 1)
    assert (
        result["strata"]["small"]["tp"],
        result["strata"]["small"]["fp"],
        result["strata"]["small"]["fn"],
    ) == (2, 0, 0)
    assert (
        result["strata"]["tail"]["tp"],
        result["strata"]["tail"]["fp"],
        result["strata"]["tail"]["fn"],
    ) == (2, 0, 0)
    assert (
        result["strata"]["small_tail"]["tp"],
        result["strata"]["small_tail"]["fp"],
        result["strata"]["small_tail"]["fn"],
    ) == (1, 0, 0)
    assert result["strata"]["small_tail"]["ignored"] == 2


def test_scene_aggregation_reports_pooled_counts_and_equal_scene_means() -> None:
    scene_one = _evaluate(
        scene_id="scene0001_00",
        ground_truth=(
            _gt(1, "chair", (0, 1, 2, 3)),
            _gt(2, "chair", (4, 5, 6, 7)),
        ),
        branch=(_prediction(10, "chair", (0, 1, 2, 3)),),
    )
    scene_two = _evaluate(
        scene_id="scene0002_00",
        ground_truth=(_gt(1, "chair", (0, 1, 2, 3)),),
        branch=(
            _prediction(20, "chair", (0, 1, 2, 3), score=0.9),
            _prediction(21, "chair", (0, 1, 2, 3), score=0.8),
        ),
    )

    aggregate = aggregate_rescue_scenes((scene_one, scene_two))
    overall = aggregate["strata"]["overall"]

    assert overall["pooled_counts"]["tp"] == 2
    assert overall["pooled_counts"]["fp"] == 1
    assert overall["pooled_counts"]["fn"] == 1
    assert overall["pooled_metrics"]["precision"] == pytest.approx(2.0 / 3.0)
    assert overall["pooled_metrics"]["recall"] == pytest.approx(2.0 / 3.0)
    assert overall["pooled_metrics"]["f1"] == pytest.approx(2.0 / 3.0)
    assert overall["scene_equal_mean"]["precision"] == pytest.approx(0.75)
    assert overall["scene_equal_mean"]["recall"] == pytest.approx(0.75)
    assert overall["scene_equal_mean"]["f1"] == pytest.approx(2.0 / 3.0)
