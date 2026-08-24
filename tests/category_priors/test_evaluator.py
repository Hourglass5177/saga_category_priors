from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import category_priors.evaluator as evaluator
from category_priors.evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
)
from category_priors.taxonomy import load_taxonomy


def prediction(
    instance_id: int, class_id: int, score: float, indices: list[int], size: int = 8
) -> PredictedInstance:
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return PredictedInstance("scene", instance_id, class_id, score, mask)


def ground_truth() -> GroundTruthScene:
    return GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0, 0, 0, -1, 99]),
        instance=np.asarray([1, 1, 1, 2, 2, 2, -1, -1]),
    )


def test_perfect_predictions_have_unit_ap() -> None:
    result = evaluate_instances(
        [ground_truth()],
        [prediction(10, 0, 0.9, [0, 1, 2]), prediction(11, 0, 0.8, [3, 4, 5])],
        ["chair", "cup"],
        min_region_size=1,
    )
    assert result["aggregate"]["map_0.25"] == 1.0
    assert result["aggregate"]["map_0.50"] == 1.0
    assert result["aggregate"]["map_0.95"] == 1.0
    assert result["aggregate"]["map_50_95"] == 1.0
    assert result["per_class"]["chair"]["ap_0.50"] == 1.0
    assert result["per_class"]["chair"]["ap_0.95"] == 1.0
    assert result["per_class"]["chair"]["ap_50_95"] == 1.0


def test_official_matching_uses_strict_iou_comparison() -> None:
    merged = evaluate_instances(
        [ground_truth()],
        [prediction(10, 0, 0.9, [0, 1, 2, 3, 4, 5])],
        ["chair", "cup"],
        min_region_size=1,
    )
    assert merged["per_class"]["chair"]["ap_0.25"] == 0.5
    assert merged["per_class"]["chair"]["ap_0.50"] == 0.0
    assert merged["per_class"]["chair"]["ap_0.55"] == 0.0


def test_wrong_class_is_penalized_and_missing_class_is_undefined() -> None:
    wrong = evaluate_instances(
        [ground_truth()],
        [prediction(10, 1, 0.9, [0, 1, 2])],
        ["chair", "cup"],
        min_region_size=1,
    )
    assert wrong["per_class"]["chair"]["ap_0.50"] == 0.0
    assert wrong["per_class"]["cup"]["ap_0.50"] is None


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


def test_void_fraction_equal_to_threshold_is_a_false_positive() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, -1, -1, 1, 1]),
        instance=np.asarray([1, 1, -1, -1, 2, 2]),
    )
    result = evaluate_instances(
        [scene],
        [
            prediction(9, 0, 0.99, [2, 3, 4, 5], size=6),
            prediction(10, 0, 0.9, [0, 1], size=6),
        ],
        ["chair", "table"],
        min_region_size=1,
    )
    # This is the official ScanNet PR integration result for one high-score FP
    # followed by one TP.  Predictions are ignored only when void_fraction > IoU.
    assert result["per_class"]["chair"]["ap_0.50"] == 0.25


def test_small_ground_truth_overlap_is_ignored() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0, 0, 1]),
        instance=np.asarray([1, 1, 2, 2, 2, 3]),
    )
    result = evaluate_instances(
        [scene],
        [
            prediction(9, 0, 0.99, [0, 1, 5], size=6),
            prediction(10, 0, 0.9, [2, 3, 4], size=6),
        ],
        ["chair", "table"],
        min_region_size=3,
    )
    assert result["per_class"]["chair"]["gt_instances"] == 1
    assert result["per_class"]["chair"]["ap_0.50"] == 1.0


def test_duplicate_predictions_follow_official_score_assignment() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0]),
        instance=np.asarray([1, 1, 1]),
    )
    result = evaluate_instances(
        [scene],
        [
            prediction(10, 0, 0.8, [0, 1, 2], size=3),
            prediction(11, 0, 0.9, [0, 1, 2], size=3),
        ],
        ["chair"],
        min_region_size=1,
    )
    # ScanNet assigns the larger score to the TP and the smaller to the duplicate
    # FP, so the duplicate is below the TP on the PR curve.
    assert result["per_class"]["chair"]["ap_0.50"] == 1.0


def test_prediction_smaller_than_min_region_size_is_skipped() -> None:
    scene = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0]),
        instance=np.asarray([1, 1, 1, 1]),
    )
    result = evaluate_instances(
        [scene],
        [prediction(10, 0, 0.9, [0, 1], size=4)],
        ["chair"],
        min_region_size=3,
    )
    assert result["per_class"]["chair"]["ap_0.50"] == 0.0


def test_scene_adapter_projects_orphans_without_changing_declared_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "point_labels": [4, 9, 4],
                "instances": {"4": {"class": "chair"}},
            }
        ),
        encoding="utf-8",
    )
    gaussian_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float64,
    )
    monkeypatch.setattr(evaluator, "load_ply_xyz", lambda _path: gaussian_xyz)

    predictions, diagnostics = evaluator.saga_scene_predictions(
        scene_id="scene",
        gt_coords=gaussian_xyz,
        output_json=output_path,
        gaussian_ply=tmp_path / "unused.ply",
        taxonomy=load_taxonomy(),
        metadata_json=None,
        transform=np.eye(4),
        require_scores=False,
    )

    assert len(predictions) == 1
    assert predictions[0].instance_id == 4
    assert predictions[0].mask.tolist() == [True, False, True]
    assert diagnostics["mapped_fraction"] == 1.0
    assert diagnostics["gt_nearest_declared_fraction"] == 2 / 3
    assert diagnostics["orphan_instance_count"] == 1.0
    assert diagnostics["orphan_gaussian_count"] == 1.0
