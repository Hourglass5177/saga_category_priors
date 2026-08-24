from __future__ import annotations

import numpy as np
import pytest

from category_priors.baseline_closure_evaluation import (
    HISTORICAL_OVERLAPS,
    SCANNET_OFFICIAL_OVERLAPS,
    adapt_prediction_scores,
    evaluate_baseline_closure,
)
from category_priors.evaluator import GroundTruthScene, PredictedInstance


def _prediction(
    instance_id: int,
    class_id: int,
    score: float,
    indices: list[int],
    *,
    size: int = 6,
) -> PredictedInstance:
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return PredictedInstance("scene", instance_id, class_id, score, mask)


def _ground_truth() -> GroundTruthScene:
    return GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 1, 1, 1]),
        instance=np.asarray([10, 10, 10, 20, 20, 20]),
    )


def test_protocol_thresholds_are_explicit_and_distinct() -> None:
    assert SCANNET_OFFICIAL_OVERLAPS == (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    )
    assert HISTORICAL_OVERLAPS == (*SCANNET_OFFICIAL_OVERLAPS, 0.95)


def test_score_adapters_are_read_only_and_oracle_is_diagnostic() -> None:
    predictions = (
        _prediction(1, 0, 0.2, [0, 1, 2]),
        _prediction(2, 0, 0.8, [0, 1, 3]),
        _prediction(3, 1, 0.6, [0, 1, 2]),
    )

    unit = adapt_prediction_scores(predictions, "unit", min_region_size=1)
    assert [prediction.score for prediction in unit.predictions] == [1.0, 1.0, 1.0]
    assert [prediction.score for prediction in predictions] == [0.2, 0.8, 0.6]
    assert unit.diagnostic_only is False

    vote = adapt_prediction_scores(
        predictions,
        "final_vote",
        final_vote_scores={
            ("scene", 1): 0.7,
            ("scene", 2): 0.4,
            ("scene", 3): 0.1,
        },
        min_region_size=1,
    )
    assert [prediction.score for prediction in vote.predictions] == [0.7, 0.4, 0.1]

    oracle = adapt_prediction_scores(
        predictions,
        "gt_oracle",
        ground_truth=[_ground_truth()],
        min_region_size=1,
    )
    assert oracle.diagnostic_only is True
    assert [prediction.score for prediction in oracle.predictions] == [1.0, 0.5, 0.0]


def test_final_vote_requires_every_instance_score() -> None:
    with pytest.raises(ValueError, match="Missing final-vote score"):
        adapt_prediction_scores(
            [_prediction(1, 0, 0.2, [0, 1, 2])],
            "final_vote",
            final_vote_scores={},
            min_region_size=1,
        )


def test_dual_protocol_and_predictable_class_view() -> None:
    predictions = (
        _prediction(1, 0, 0.9, [0, 1, 2]),
        # A wrong cup prediction leaves the full-view cup AP at zero.
        _prediction(2, 1, 0.8, [0, 1, 2]),
    )
    result = evaluate_baseline_closure(
        [_ground_truth()],
        predictions,
        ["chair", "cup"],
        predictable_classes=["chair"],
        score_mode="unit",
        min_region_size=1,
    )

    official = result["protocols"]["scannet_official_9"]
    historical = result["protocols"]["historical_10"]
    assert official["overlaps"][-1] == 0.90
    assert historical["overlaps"][-1] == 0.95
    assert (
        official["full_saga20"]["protocol_version"] == "scannet-official-instance-9-v1"
    )
    assert "map_50_95" not in official["full_saga20"]["aggregate"]
    assert official["full_saga20"]["aggregate"]["map_0.25"] == 0.5
    assert official["full_saga20"]["aggregate"]["map_50_90"] == 0.5
    assert official["predictable_intersection"]["aggregate"]["map_50_90"] == 1.0
    assert historical["full_saga20"]["aggregate"]["map_50_95"] == 0.5
    assert (
        historical["full_saga20"]["protocol_version"]
        == "saga20-historical-instance-10-v1"
    )
    assert result["score_adapter"] == {
        "mode": "unit",
        "diagnostic_only": False,
    }


def test_gt_oracle_closure_is_marked_diagnostic_only() -> None:
    result = evaluate_baseline_closure(
        [_ground_truth()],
        [_prediction(1, 0, 0.3, [0, 1, 2])],
        ["chair", "cup"],
        predictable_classes=["chair"],
        score_mode="gt_oracle",
        min_region_size=1,
    )
    assert result["score_adapter"]["diagnostic_only"] is True
