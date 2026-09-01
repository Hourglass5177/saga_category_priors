from __future__ import annotations

import numpy as np
import pytest

from category_priors.clean_baseline.evaluation import (
    CleanCandidate,
    GroundTruthObject,
)
from category_priors.clean_baseline.metric_reaudit import (
    HISTORICAL_OVERLAPS,
    SCANNET_OFFICIAL_OVERLAPS,
    build_bidirectional_nearest,
    evaluate_candidate_set_three_spaces,
    evaluate_dual_protocols,
    evaluate_gt_as_prediction_dual_protocols,
    formal_gt_point_mask,
    support_coverage_ceiling,
)
from category_priors.evaluator import GroundTruthScene, PredictedInstance


def _candidate(
    object_id: int,
    gaussian_ids: list[int] | np.ndarray,
    class_id: str = "chair",
) -> CleanCandidate:
    return CleanCandidate(
        object_id=object_id,
        gaussian_ids=np.asarray(gaussian_ids, dtype=np.int64),
        class_id=class_id,
        winner_probability=1.0,
        view_consensus=1.0,
        detection_ratio=1.0,
    )


def test_redundant_correct_gaussians_do_not_lower_formal_gt_point_iou() -> None:
    gt_xyz = np.asarray([[0.0, 0.0, 0.0]])
    gaussian_xyz = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
    result = evaluate_candidate_set_three_spaces(
        candidates=[_candidate(7, [0, 1])],
        gt_objects=[GroundTruthObject(1, "chair", np.asarray([0]))],
        nearest=nearest,
        min_region_size=1,
    )

    row = result["candidate_rows"][0]
    assert row["formal_same_class_iou_5cm"] == 1.0
    assert row["formal_gt_point_count_5cm"] == 1
    assert row["radii"]["0.05"]["gaussian_to_gt_target_precision"] == 1.0
    assert result["formal_metric_space"]["synthetic_false_positive_sentinels"] is False


def test_far_gaussian_lowers_only_gaussian_precision_not_formal_iou() -> None:
    gt_xyz = np.asarray([[0.0, 0.0, 0.0]])
    gaussian_xyz = np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
    result = evaluate_candidate_set_three_spaces(
        candidates=[_candidate(7, [0, 1])],
        gt_objects=[GroundTruthObject(1, "chair", np.asarray([0]))],
        nearest=nearest,
        min_region_size=1,
    )

    row = result["candidate_rows"][0]
    assert row["formal_same_class_iou_5cm"] == 1.0
    radius = row["radii"]["0.05"]
    assert radius["gaussian_to_gt_target_precision"] == pytest.approx(0.5)
    assert radius["gaussian_unsupported_count"] == 1
    assert radius["gt_to_gaussian_candidate_recall"] == 1.0


def test_gaussian_directional_bins_are_mutually_exclusive() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        ),
    )
    result = evaluate_candidate_set_three_spaces(
        candidates=[_candidate(7, [0, 1, 2, 3], "chair")],
        gt_objects=[
            GroundTruthObject(1, "chair", np.asarray([0])),
            GroundTruthObject(2, "chair", np.asarray([1])),
            GroundTruthObject(3, "table", np.asarray([2])),
        ],
        nearest=nearest,
        min_region_size=1,
    )

    radius = result["candidate_rows"][0]["radii"]["0.05"]
    assert radius["gaussian_correct_target_instance_count"] == 1
    assert radius["gaussian_same_class_wrong_instance_count"] == 1
    assert radius["gaussian_wrong_class_count"] == 1
    assert radius["gaussian_unsupported_count"] == 1


def test_duplicate_candidates_are_one_tp_and_one_fp_after_one_to_one_matching() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0]]),
    )
    result = evaluate_candidate_set_three_spaces(
        candidates=[_candidate(20, [0]), _candidate(10, [0])],
        gt_objects=[GroundTruthObject(1, "chair", np.asarray([0]))],
        nearest=nearest,
        min_region_size=1,
    )

    match = result["subsets"]["all"]["matching"]["same_class"]["0.50"]
    assert match["true_positive_count"] == 1
    assert match["false_positive_count"] == 1
    assert match["false_negative_count"] == 0
    assert match["precision"] == pytest.approx(0.5)
    assert match["matches"][0]["candidate_id"] == 10


def test_tiny_small_denominator_includes_unmapped_official_gt() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0]]),
    )
    result = evaluate_candidate_set_three_spaces(
        candidates=[_candidate(1, [0])],
        gt_objects=[
            GroundTruthObject(
                1, "chair", np.asarray([0]), official_valid=True, is_tiny_small=True
            ),
            GroundTruthObject(
                2, "chair", np.asarray([1]), official_valid=True, is_tiny_small=True
            ),
        ],
        nearest=nearest,
        min_region_size=1,
    )

    assert result["official_tiny_small_gt_count"] == 2
    assert result["gt_to_gaussian_scene_coverage"]["0.05"]["mapped_fraction"] == 0.5
    match = result["subsets"]["all"]["matching"]["same_class"]["0.25"]
    assert match["tiny_small_recall"] == pytest.approx(0.5)


def test_all_and_official_evaluable_candidate_subsets_are_both_reported() -> None:
    count = 120
    xyz = np.column_stack(
        (np.arange(count, dtype=np.float64) * 0.001, np.zeros(count), np.zeros(count))
    )
    nearest = build_bidirectional_nearest(xyz, xyz.copy())
    result = evaluate_candidate_set_three_spaces(
        candidates=[
            _candidate(1, np.arange(120)),
            _candidate(2, np.arange(50)),
        ],
        gt_objects=[GroundTruthObject(1, "chair", np.arange(120))],
        nearest=nearest,
        min_region_size=100,
    )

    assert result["subsets"]["all"]["candidate_count"] == 2
    assert result["subsets"]["official_evaluable"]["candidate_count"] == 1


def test_support_coverage_ceiling_allows_reuse_and_is_not_instance_bound() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0]]),
    )
    result = support_coverage_ceiling(
        mask_gaussian_ids=[np.asarray([0])],
        gt_objects=[
            GroundTruthObject(1, "chair", np.asarray([0])),
            GroundTruthObject(2, "chair", np.asarray([1])),
        ],
        nearest=nearest,
    )

    assert [row["support_coverage_ceiling"] for row in result["rows"]] == [1.0, 1.0]
    assert result["evidence_may_be_reused_across_gt_objects"] is True
    assert result["joint_instance_upper_bound"] is False
    assert "perfect_trim" not in str(result)


def test_formal_projection_never_appends_synthetic_points() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]]),
    )
    mask = formal_gt_point_mask([0, 1], nearest)
    assert mask.shape == (1,)
    assert mask.tolist() == [True]


def test_official_and_historical_protocol_endpoints_and_gt_parity() -> None:
    assert SCANNET_OFFICIAL_OVERLAPS[0] == 0.50
    assert SCANNET_OFFICIAL_OVERLAPS[-1] == 0.90
    assert HISTORICAL_OVERLAPS == (*SCANNET_OFFICIAL_OVERLAPS, 0.95)
    ground_truth = [
        GroundTruthScene(
            "scene",
            semantic=np.asarray([0, 0, 1, 1]),
            instance=np.asarray([1, 1, 2, 2]),
        )
    ]
    result = evaluate_gt_as_prediction_dual_protocols(
        ground_truth, ["chair", "table"], min_region_size=1
    )

    assert result["gt_as_prediction_parity"] is True
    assert result["official_9"]["overlaps"][-1] == 0.90
    assert result["historical_10"]["overlaps"][-1] == 0.95
    assert result["official_9"]["aggregate"]["map_50_90"] == 1.0
    assert result["official_9"]["aggregate"]["map_0.25"] == 1.0
    assert result["historical_10"]["aggregate"]["map_50_95"] == 1.0


def test_dual_protocols_keep_ap25_and_use_distinct_main_averages() -> None:
    ground_truth = [
        GroundTruthScene(
            "scene",
            semantic=np.asarray([0, 0]),
            instance=np.asarray([1, 1]),
        )
    ]
    predictions = [
        PredictedInstance(
            "scene", 9, 0, 1.0, np.asarray([True, True], dtype=bool)
        )
    ]
    result = evaluate_dual_protocols(
        ground_truth, predictions, ["chair"], min_region_size=1
    )

    assert "map_0.25" in result["official_9"]["aggregate"]
    assert "map_50_90" in result["official_9"]["aggregate"]
    assert "map_50_95" not in result["official_9"]["aggregate"]
    assert "map_50_95" in result["historical_10"]["aggregate"]
