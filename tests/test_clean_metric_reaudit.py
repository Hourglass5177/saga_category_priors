from __future__ import annotations

import numpy as np
import pytest

from category_priors.clean_baseline import metric_reaudit
from category_priors.clean_baseline.evaluation import (
    CleanCandidate,
    GroundTruthObject,
)
from category_priors.clean_baseline.metric_reaudit import (
    HISTORICAL_OVERLAPS,
    SCANNET_OFFICIAL_OVERLAPS,
    BidirectionalNearest,
    build_bidirectional_nearest,
    deterministic_one_to_one_matches,
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
    assert radius["gaussian_geometry_target_instance_count"] == 1
    assert radius["gaussian_mapped_other_instance_count"] == 2
    assert radius["gaussian_to_gt_geometry_target_precision"] == pytest.approx(0.25)
    assert radius["gaussian_to_gt_geometry_pollution_fraction"] == pytest.approx(0.75)


def test_classless_funnel_candidate_has_geometry_only_pollution_diagnostic() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
    )
    result = evaluate_candidate_set_three_spaces(
        candidates=[
            {
                "object_id": "pre-semantic-stage",
                "gaussian_ids": np.asarray([0, 1, 2]),
                "class_id": None,
                "winner_probability": 1.0,
                "view_consensus": 1.0,
                "detection_ratio": 1.0,
            }
        ],
        gt_objects=[
            GroundTruthObject(1, "chair", np.asarray([0])),
            GroundTruthObject(2, "table", np.asarray([1])),
        ],
        nearest=nearest,
        min_region_size=1,
    )

    row = result["candidate_rows"][0]
    radius = row["radii"]["0.05"]
    assert row["geometric_precision_target_gt_instance_id"] == 1
    assert radius["gaussian_geometry_target_instance_count"] == 1
    assert radius["gaussian_mapped_other_instance_count"] == 1
    assert radius["gaussian_unsupported_count"] == 1
    assert radius["gaussian_to_gt_geometry_target_precision"] == pytest.approx(1 / 3)
    # There is intentionally no predicted semantic class at this stage.
    assert radius["gaussian_to_gt_target_precision"] == 0.0


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


def test_sparse_formal_evaluation_matches_dense_reference_fixture() -> None:
    xyz = np.column_stack(
        (np.arange(8, dtype=np.float64), np.zeros(8), np.zeros(8))
    )
    nearest = build_bidirectional_nearest(xyz, xyz.copy())
    ground_truth = [
        GroundTruthObject(1, "chair", np.asarray([0, 1, 2])),
        GroundTruthObject(2, "table", np.asarray([3, 4])),
    ]
    candidates = [
        _candidate(10, [0, 1, 2], "chair"),
        _candidate(20, [2, 3], "chair"),
        _candidate(30, [3, 4], "table"),
        _candidate(40, [0, 1, 2], "chair"),
    ]
    dense_masks = [
        formal_gt_point_mask(candidate.gaussian_ids, nearest)
        for candidate in candidates
    ]
    dense_iou = np.zeros((len(candidates), len(ground_truth)), dtype=np.float64)
    for candidate_index, mask in enumerate(dense_masks):
        predicted_count = int(np.count_nonzero(mask))
        for gt_index, gt in enumerate(ground_truth):
            intersection = int(np.count_nonzero(mask[gt.point_ids]))
            union = predicted_count + len(gt.point_ids) - intersection
            dense_iou[candidate_index, gt_index] = intersection / union

    result = evaluate_candidate_set_three_spaces(
        candidates=candidates,
        gt_objects=ground_truth,
        nearest=nearest,
        radii_m=(0.05,),
        min_region_size=1,
    )

    for candidate_index, row in enumerate(result["candidate_rows"]):
        assert row["formal_gt_point_count_5cm"] == int(
            np.count_nonzero(dense_masks[candidate_index])
        )
        assert row["formal_geometry_iou_5cm"] == pytest.approx(
            float(dense_iou[candidate_index].max())
        )
        compatible = [
            dense_iou[candidate_index, gt_index]
            for gt_index, gt in enumerate(ground_truth)
            if str(gt.class_id) == str(candidates[candidate_index].class_id)
        ]
        assert row["formal_same_class_iou_5cm"] == pytest.approx(
            max(compatible, default=0.0)
        )
    dense_matches = deterministic_one_to_one_matches(
        candidate_ids=[candidate.object_id for candidate in candidates],
        candidate_class_ids=[candidate.class_id for candidate in candidates],
        gt_objects=ground_truth,
        iou_matrix=dense_iou,
        threshold=0.50,
        same_class=False,
    )
    assert result["subsets"]["all"]["matching"]["geometry"]["0.50"][
        "matches"
    ] == dense_matches
    assert result["formal_metric_space"] == {
        "domain": "real_gt_points",
        "radius_m": 0.05,
        "synthetic_false_positive_sentinels": False,
    }


def test_sparse_formal_evaluation_scales_without_dense_candidate_gt_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dense candidate-by-GT-point representation would require 800 million
    # boolean cells here.  The sparse representation stores only two formal
    # point assignments per candidate, plus one linear inverse index.
    gt_count = 400_000
    candidate_count = 2_000
    point_ids = np.arange(gt_count, dtype=np.int64)
    nearest = BidirectionalNearest(
        point_ids,
        np.zeros(gt_count, dtype=np.float64),
        point_ids,
        np.zeros(gt_count, dtype=np.float64),
    )
    candidates = [
        _candidate(
            candidate_id,
            [candidate_id * 2, candidate_id * 2 + 1],
            "not-a-gt-class",
        )
        for candidate_id in range(candidate_count)
    ]
    ground_truth = [GroundTruthObject(1, "chair", point_ids)]

    def dense_projection_forbidden(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("dense formal GT mask helper must not be called")

    monkeypatch.setattr(
        metric_reaudit, "formal_gt_point_mask", dense_projection_forbidden
    )
    result = evaluate_candidate_set_three_spaces(
        candidates=candidates,
        gt_objects=ground_truth,
        nearest=nearest,
        radii_m=(0.05,),
        min_region_size=1,
    )

    assert len(result["candidate_rows"]) == candidate_count
    assert sum(
        row["formal_gt_point_count_5cm"] for row in result["candidate_rows"]
    ) == candidate_count * 2


def test_sparse_projection_rejects_out_of_range_gaussian_before_indexing() -> None:
    nearest = build_bidirectional_nearest(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="out-of-range"):
        evaluate_candidate_set_three_spaces(
            candidates=[_candidate(1, [1])],
            gt_objects=[GroundTruthObject(1, "chair", np.asarray([0]))],
            nearest=nearest,
            min_region_size=1,
        )


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
