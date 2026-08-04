from __future__ import annotations

import numpy as np

from category_priors.evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
)


def prediction(
    instance_id: int, class_id: int, score: float, indices: list[int], size: int = 8
) -> PredictedInstance:
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return PredictedInstance("scene", instance_id, class_id, score, mask)


def ground_truth() -> GroundTruthScene:
    return GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0, 0, 0, -1, -1]),
        instance=np.asarray([1, 1, 1, 2, 2, 2, -1, -1]),
    )


def test_perfect_predictions_have_unit_ap() -> None:
    result = evaluate_instances(
        [ground_truth()],
        [prediction(10, 0, 0.9, [0, 1, 2]), prediction(11, 0, 0.8, [3, 4, 5])],
        ["chair", "cup"],
        min_region_size=1,
    )
    assert result["per_class"]["chair"]["ap_0.50"] == 1.0
    assert result["per_class"]["chair"]["ap_50_95"] == 1.0


def test_merge_and_wrong_class_are_penalized() -> None:
    merged = evaluate_instances(
        [ground_truth()],
        [prediction(10, 0, 0.9, [0, 1, 2, 3, 4, 5])],
        ["chair", "cup"],
        min_region_size=1,
    )
    wrong = evaluate_instances(
        [ground_truth()],
        [prediction(10, 1, 0.9, [0, 1, 2])],
        ["chair", "cup"],
        min_region_size=1,
    )
    assert merged["per_class"]["chair"]["ap_0.50"] == 0.5
    assert merged["per_class"]["chair"]["ap_0.55"] == 0.0
    assert wrong["per_class"]["chair"]["ap_0.50"] == 0.0


def test_void_dominated_prediction_is_ignored() -> None:
    result = evaluate_instances(
        [ground_truth()],
        [
            prediction(9, 0, 0.99, [6, 7]),
            prediction(10, 0, 0.9, [0, 1, 2]),
            prediction(11, 0, 0.8, [3, 4, 5]),
        ],
        ["chair"],
        min_region_size=1,
    )
    assert result["per_class"]["chair"]["ap_0.50"] == 1.0
