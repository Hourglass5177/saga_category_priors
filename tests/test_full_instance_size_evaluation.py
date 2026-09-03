from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import category_priors.clean_baseline.evaluation as clean_evaluation
import category_priors.full_instance_size_evaluation as evaluation_module
from category_priors.evaluator import GroundTruthScene, PredictedInstance
from category_priors.full_instance_size_evaluation import (
    HISTORICAL_10_OVERLAPS,
    OFFICIAL_9_OVERLAPS,
    CandidatePrediction,
    analyze_candidate_ranking,
    candidate_average_precision,
    candidate_predictions_from_rows,
    choose_global_threshold,
    evaluate_candidate_rankings,
    evaluate_candidate_scenes,
    evaluate_official_protocols,
    matched_recall_summary,
    maximum_cardinality_iou_matching,
    oracle_class_diagnostics,
    paired_bootstrap,
    paired_physical_scene_bootstrap,
    ranked_candidate_matches,
)


def _candidate(
    scene: str,
    candidate_id: int,
    class_id: int,
    score: float,
    members: list[int],
) -> CandidatePrediction:
    return CandidatePrediction(
        scene_id=scene,
        candidate_id=candidate_id,
        class_id=class_id,
        score=score,
        member_indices=np.asarray(members, dtype=np.int64),
    )


def test_registered_protocol_thresholds_do_not_conflate_nine_and_ten() -> None:
    assert OFFICIAL_9_OVERLAPS == (
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
    assert HISTORICAL_10_OVERLAPS == (*OFFICIAL_9_OVERLAPS, 0.95)


def test_capacity_matching_maximizes_cardinality_before_total_iou() -> None:
    # Greedily taking (candidate 0, GT 0) at .90 would leave only one match.
    # The registered capacity objective must instead retain two matches.
    matrix = np.asarray([[0.90, 0.80], [0.85, 0.00]])
    matches = maximum_cardinality_iou_matching(
        matrix,
        0.25,
        candidate_classes=[0, 0],
        gt_classes=[0, 0],
    )
    assert {(row.candidate_index, row.gt_index) for row in matches} == {
        (0, 1),
        (1, 0),
    }
    assert sum(row.iou for row in matches) == pytest.approx(1.65)

    # Among equal-cardinality matchings, total IoU is the second objective.
    matrix = np.asarray([[0.90, 0.70], [0.60, 0.80]])
    matches = maximum_cardinality_iou_matching(matrix, 0.25)
    assert {(row.candidate_index, row.gt_index) for row in matches} == {
        (0, 0),
        (1, 1),
    }


def test_iou_threshold_is_strictly_greater_than() -> None:
    matches = maximum_cardinality_iou_matching(
        np.asarray([[0.25, 0.250001]]), 0.25
    )
    assert len(matches) == 1
    assert matches[0].gt_index == 1


def test_score_order_marks_duplicate_prediction_as_false_positive() -> None:
    ranked = ranked_candidate_matches(
        np.asarray([[1.0], [1.0]]),
        scores=[0.9, 0.8],
        candidate_classes=[0, 0],
        gt_classes=[0],
        threshold=0.50,
        candidate_ids=["first", "duplicate"],
    )
    assert [row["true_positive"] for row in ranked] == [True, False]
    assert ranked[1]["matched_gt_index"] is None

    result = candidate_average_precision(
        np.asarray([[1.0], [1.0]]),
        scores=[0.9, 0.8],
        candidate_classes=[0, 0],
        gt_classes=[0],
        threshold=0.50,
        candidate_ids=["first", "duplicate"],
    )
    assert result["true_positive_count"] == 1
    assert result["false_positive_count"] == 1
    assert result["ap"] == pytest.approx(1.0)


def test_equal_scores_use_stable_candidate_identity() -> None:
    ranked = ranked_candidate_matches(
        np.asarray([[1.0], [1.0]]),
        scores=[0.5, 0.5],
        candidate_classes=[0, 0],
        gt_classes=[0],
        threshold=0.5,
        candidate_ids=["b", "a"],
    )
    assert [row["candidate_id"] for row in ranked] == ["a", "b"]
    assert [row["true_positive"] for row in ranked] == [True, False]


def test_scene_equal_candidate_ap_and_tiny_small_denominator() -> None:
    ground_truth = [
        GroundTruthScene(
            "scene-a",
            semantic=np.asarray([0, 0]),
            instance=np.asarray([1, 1]),
        ),
        GroundTruthScene(
            "scene-b",
            semantic=np.asarray([0, 0]),
            instance=np.asarray([2, 2]),
        ),
    ]
    result = evaluate_candidate_rankings(
        [_candidate("scene-a", 10, 0, 1.0, [0, 1])],
        ground_truth,
        {
            "scene-a": np.asarray([0, 1]),
            "scene-b": np.asarray([0, 1]),
        },
        min_region_size=1,
        tiny_small_instance_ids={"scene-a": {1}, "scene-b": {2}},
    )
    aggregate = result["views"]["all"]["aggregate"]["thresholds"]
    assert aggregate["025"]["match_count"] == 1
    assert aggregate["025"]["scene_equal_candidate_ap"] == pytest.approx(0.5)
    assert aggregate["025"]["tiny_small_gt_count"] == 2
    assert aggregate["025"]["tiny_small_recall"] == pytest.approx(0.5)
    assert aggregate["025"]["scene_equal_tiny_small_candidate_ap"] == pytest.approx(
        0.5
    )
    assert aggregate["050"]["scene_equal_candidate_ap"] == pytest.approx(0.5)


def test_tiny_small_candidate_ap_ignores_correct_non_tiny_same_class() -> None:
    ground_truth = [
        GroundTruthScene(
            "scene",
            semantic=np.asarray([0, 0, 0, 0]),
            instance=np.asarray([1, 1, 2, 2]),
        )
    ]
    # The medium/large object deliberately has the higher score.  In a tiny
    # view it is a correct out-of-stratum prediction, not a tiny false positive.
    result = evaluate_candidate_rankings(
        [
            _candidate("scene", 20, 0, 0.9, [2, 3]),
            _candidate("scene", 10, 0, 0.8, [0, 1]),
        ],
        ground_truth,
        {"scene": np.arange(4, dtype=np.int64)},
        min_region_size=1,
        tiny_small_instance_ids={"scene": {1}},
    )
    threshold = result["views"]["all"]["aggregate"]["thresholds"]["050"]
    assert threshold["tiny_small_match_count"] == 1
    assert threshold["scene_equal_tiny_small_candidate_ap"] == pytest.approx(1.0)
    assert threshold["tiny_small_ap_ignored_non_target_match_count"] == 1


def test_all_and_official_100_candidate_views_are_separate() -> None:
    ground_truth = [
        GroundTruthScene(
            "scene",
            semantic=np.zeros(100, dtype=np.int64),
            instance=np.ones(100, dtype=np.int64),
        )
    ]
    result = evaluate_candidate_rankings(
        [_candidate("scene", 1, 0, 1.0, list(range(99)))],
        ground_truth,
        {"scene": np.arange(100, dtype=np.int64)},
        min_region_size=100,
    )
    assert result["views"]["all"]["aggregate"]["candidate_count"] == 1
    assert result["views"]["official_100"]["aggregate"]["candidate_count"] == 0


def test_mapping_adapter_keeps_gt_out_and_respects_eligibility() -> None:
    rows = [
        {
            "scene_id": "scene",
            "raw_instance_id": 5,
            "predicted_class_name": "chair",
            "member_indices": [0, 1],
            "S": 0.7,
            "eligible": True,
        },
        {
            "scene_id": "scene",
            "raw_instance_id": 6,
            "predicted_class_name": "chair",
            "member_indices": [2],
            "S": 0.1,
            "eligible": False,
        },
    ]
    normalized = candidate_predictions_from_rows(
        rows,
        score_key="S",
        class_names=["chair"],
        eligible_only=True,
    )
    assert len(normalized) == 1
    assert normalized[0].candidate_id == 5
    assert normalized[0].score == pytest.approx(0.7)

    named = candidate_predictions_from_rows(
        [
            {
                "scene_id": "scene",
                "candidate_id": 7,
                "predicted_class": "chair",
                "member_indices": [3],
                "Q": 0.6,
            }
        ],
        score_key="Q",
        class_names=["chair"],
    )
    assert named[0].class_id == 0


def test_prediction_of_class_absent_from_scene_is_still_a_false_positive() -> None:
    result = candidate_average_precision(
        np.asarray([[1.0], [0.0]]),
        scores=[0.9, 0.8],
        candidate_classes=[0, 1],
        gt_classes=[0],
        threshold=0.5,
        candidate_ids=["correct", "absent-class"],
    )
    assert result["ap"] == pytest.approx(1.0)
    assert result["true_positive_count"] == 1
    assert result["false_positive_count"] == 1
    absent = next(
        row for row in result["ranked_rows"] if row["candidate_id"] == "absent-class"
    )
    assert absent["ap_class_has_gt"] is False


def test_oracle_class_is_evaluation_only_and_rejects_cross_class_tie() -> None:
    gt = [
        GroundTruthScene(
            "scene",
            semantic=np.asarray([0, 1]),
            instance=np.asarray([1, 2]),
        )
    ]
    candidates = [_candidate("scene", 8, 0, 0.5, [0, 1])]
    diagnostic = oracle_class_diagnostics(
        candidates,
        gt,
        {"scene": np.asarray([0, 1])},
        min_region_size=1,
    )[0]
    assert diagnostic["best_geometric_iou"] == pytest.approx(0.5)
    assert diagnostic["oracle_class_id"] is None
    assert diagnostic["ambiguous"] is True
    assert diagnostic["diagnostic_only"] is True


def test_oracle_class_requires_strict_iou_above_quarter() -> None:
    gt = [
        GroundTruthScene(
            "scene",
            semantic=np.asarray([0, 0, 0, 0]),
            instance=np.asarray([1, 1, 1, 1]),
        )
    ]
    mapping = {"scene": np.arange(4, dtype=np.int64)}
    at_threshold = oracle_class_diagnostics(
        [_candidate("scene", 1, 0, 0.5, [0])],
        gt,
        mapping,
        min_region_size=1,
    )[0]
    above_threshold = oracle_class_diagnostics(
        [_candidate("scene", 2, 0, 0.5, [0, 1])],
        gt,
        mapping,
        min_region_size=1,
    )[0]
    assert at_threshold["best_geometric_iou"] == pytest.approx(0.25)
    assert at_threshold["oracle_class_id"] is None
    assert at_threshold["below_or_at_oracle_support_threshold"] is True
    assert above_threshold["best_geometric_iou"] == pytest.approx(0.5)
    assert above_threshold["oracle_class_id"] == 0
    assert above_threshold["oracle_semantics"] == (
        "classification-or-eligibility-upper-bound"
    )


def test_official_protocol_wrapper_reports_perfect_gt_prediction() -> None:
    gt = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0]),
        instance=np.asarray([1, 1]),
    )
    prediction = PredictedInstance(
        "scene", 0, 0, 1.0, np.asarray([True, True])
    )
    result = evaluate_official_protocols(
        [gt], [prediction], ["chair"], min_region_size=1
    )
    official = result["protocols"]["official_9"]
    history = result["protocols"]["historical_10"]
    assert official["overlaps"][-1] == 0.90
    assert history["overlaps"][-1] == 0.95
    assert official["aggregate"]["official_map_50_90"] == pytest.approx(1.0)
    assert official["aggregate"]["map_0.25"] == pytest.approx(1.0)
    assert history["aggregate"]["historical_map_50_95"] == pytest.approx(1.0)


def test_paired_bootstrap_groups_repeated_scans_before_resampling() -> None:
    control = {"a0": 0.0, "a1": 0.0, "b0": 0.0}
    treatment = {"a0": 0.1, "a1": 0.3, "b0": -0.1}
    groups = {"a0": "a", "a1": "a", "b0": "b"}
    first = paired_physical_scene_bootstrap(
        control,
        treatment,
        physical_scene_by_scan=groups,
        samples=1000,
        seed=20260804,
    )
    second = paired_physical_scene_bootstrap(
        control,
        treatment,
        physical_scene_by_scan=groups,
        samples=1000,
        seed=20260804,
    )
    assert first == second
    assert first["scan_count"] == 3
    assert first["physical_scene_count"] == 2
    assert first["difference"] == pytest.approx(0.05)

    with pytest.raises(ValueError, match="cover exactly"):
        paired_physical_scene_bootstrap(
            control,
            treatment,
            physical_scene_by_scan={"a0": "a"},
            samples=10,
        )


def test_matched_recall_summary_is_strict_and_counts_duplicate_fp() -> None:
    gt = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0]),
        instance=np.asarray([1, 1]),
    )
    predictions = [
        PredictedInstance("scene", 1, 0, 0.9, np.asarray([True, True])),
        PredictedInstance("scene", 2, 0, 0.8, np.asarray([True, True])),
    ]
    result = matched_recall_summary(
        gt,
        predictions,
        size_by_gt={(0, 1): "tiny"},
        thresholds=(0.50,),
        min_region_size=1,
    )
    assert result["thresholds"]["050"]["true_positive_count"] == 1
    assert result["thresholds"]["050"]["false_positive_count"] == 1
    assert result["fp_tp_ratio_050"] == pytest.approx(1.0)
    assert result["tiny_small_recall_050"] == pytest.approx(1.0)

    exact_half = [
        PredictedInstance("scene", 3, 0, 0.9, np.asarray([True, False]))
    ]
    strict = matched_recall_summary(
        gt, exact_half, thresholds=(0.50,), min_region_size=1
    )
    assert strict["recall_050"] == 0.0


def _aggregate_threshold(
    ap: float,
    *,
    matches: int = 20,
    scenes: int = 6,
    classes: int = 4,
    tiny_matches: int = 4,
) -> dict[str, object]:
    return {
        "match_count": matches,
        "matched_scene_count": scenes,
        "matched_class_count": classes,
        "scene_equal_candidate_ap": ap,
        "scene_equal_tiny_small_candidate_ap": ap,
        "tiny_small_match_count": tiny_matches,
        "ap_true_positive_count": 20,
        "ap_false_positive_count": 10,
    }


def _mode_result(ap025: float, ap050: float) -> dict[str, object]:
    return {
        "views": {
            "all": {
                "aggregate": {
                    "thresholds": {
                        "025": _aggregate_threshold(ap025),
                        "050": _aggregate_threshold(
                            ap050, matches=12, scenes=4, classes=4
                        ),
                    }
                },
                "per_scene": {
                    f"scene-{index}": {
                        "thresholds": {"025": {"candidate_ap": ap025}}
                    }
                    for index in range(8)
                },
            }
        }
    }


def test_analyze_candidate_ranking_applies_registered_scene_equal_gates() -> None:
    evaluation = {
        "mode_results": {
            "q-only": _mode_result(0.100, 0.080),
            "global-g-only": _mode_result(0.090, 0.070),
            "class-g-only": _mode_result(0.095, 0.071),
            "global-size": _mode_result(0.100, 0.080),
            "class-size": _mode_result(0.103, 0.079),
            "oracle-prior-lookup-only": _mode_result(0.104, 0.081),
            "oracle-class-global-size": _mode_result(0.100, 0.080),
            "oracle-class-size": _mode_result(0.106, 0.082),
        }
    }
    # Give the class arm a positive direction in every physical scene.
    class_scenes = evaluation["mode_results"]["class-size"]["views"]["all"][
        "per_scene"
    ]
    for row in class_scenes.values():
        row["thresholds"]["025"]["candidate_ap"] = 0.103
    scored = [
        {
            "scene_id": f"scene-{index % 4}",
            "predicted_class": ("chair", "table", "book")[index % 3],
            "eligible": True,
            "class_prior_fallback": False,
            "G_global": 0.8,
            "G_class": 0.9,
            "S_global": 0.4,
            "S_class": 0.45,
        }
        for index in range(10)
    ]
    result = analyze_candidate_ranking(evaluation, scored)
    assert result["capacity"]["match_025"] == 20
    assert result["mechanical_effect"]["passed"] is True
    assert result["ranking_gate"]["passed"] is True
    assert result["oracle_gate"]["oracle_better_than_global"] is True
    assert result["ranking_gate"]["global_g_only_candidate_ap_025"] == pytest.approx(0.090)
    assert result["ranking_gate"]["class_g_only_candidate_ap_025"] == pytest.approx(0.095)
    assert result["oracle_gate"]["lookup_only_candidate_ap_025"] == pytest.approx(0.104)
    assert result["oracle_gate"]["oracle_size_delta_candidate_ap_025"] == pytest.approx(0.006)
    # The predicted arm already clears the registered AP25 effect threshold, so
    # a still-larger oracle delta is not evidence that classification blocked it.
    assert result["oracle_gate"]["oracle_better_than_predicted"] is False


def test_stage2_fp_tp_ratio_is_reported_but_not_an_unregistered_gate() -> None:
    evaluation = {
        "mode_results": {
            "q-only": _mode_result(0.100, 0.080),
            "global-g-only": _mode_result(0.090, 0.070),
            "class-g-only": _mode_result(0.095, 0.071),
            "global-size": _mode_result(0.100, 0.080),
            "class-size": _mode_result(0.103, 0.079),
            "oracle-prior-lookup-only": _mode_result(0.104, 0.081),
            "oracle-class-global-size": _mode_result(0.100, 0.080),
            "oracle-class-size": _mode_result(0.106, 0.082),
        }
    }
    for row in evaluation["mode_results"]["class-size"]["views"]["all"][
        "per_scene"
    ].values():
        row["thresholds"]["025"]["candidate_ap"] = 0.103
    class_025 = evaluation["mode_results"]["class-size"]["views"]["all"][
        "aggregate"
    ]["thresholds"]["025"]
    class_025["ap_true_positive_count"] = 1
    class_025["ap_false_positive_count"] = 100
    scored = [
        {
            "scene_id": f"scene-{index % 4}",
            "predicted_class": ("chair", "table", "book")[index % 3],
            "eligible": True,
            "class_prior_fallback": False,
            "G_global": 0.8,
            "G_class": 0.9,
            "S_global": 0.4,
            "S_class": 0.45,
        }
        for index in range(10)
    ]

    result = analyze_candidate_ranking(evaluation, scored)

    assert result["ranking_gate"]["fp_tp_guard_passed"] is False
    assert result["ranking_gate"]["fp_tp_guard_is_diagnostic_only"] is True
    assert result["ranking_gate"]["passed"] is True


def test_oracle_relabelling_gain_is_not_counted_as_size_discrimination() -> None:
    evaluation = {
        "mode_results": {
            "q-only": _mode_result(0.100, 0.080),
            "global-g-only": _mode_result(0.090, 0.070),
            "class-g-only": _mode_result(0.090, 0.070),
            "global-size": _mode_result(0.100, 0.080),
            "class-size": _mode_result(0.100, 0.080),
            "oracle-prior-lookup-only": _mode_result(0.100, 0.080),
            # Perfect classes greatly improve both oracle arms, but the class
            # size lookup itself adds nothing over the oracle-global arm.
            "oracle-class-global-size": _mode_result(0.800, 0.700),
            "oracle-class-size": _mode_result(0.800, 0.700),
        }
    }
    scored = [
        {
            "scene_id": f"scene-{index % 4}",
            "predicted_class": ("chair", "table", "book")[index % 3],
            "eligible": True,
            "class_prior_fallback": False,
            "G_global": 0.8,
            "G_class": 0.9,
            "S_global": 0.4,
            "S_class": 0.45,
        }
        for index in range(10)
    ]

    result = analyze_candidate_ranking(evaluation, scored)
    assert result["oracle_gate"]["oracle_size_delta_candidate_ap_025"] == 0.0
    assert result["oracle_gate"]["oracle_size_has_discrimination_value"] is False
    assert result["oracle_gate"]["oracle_better_than_global"] is False


def test_oracle_size_delta_can_identify_a_predicted_class_bottleneck() -> None:
    evaluation = {
        "mode_results": {
            "q-only": _mode_result(0.100, 0.080),
            "global-g-only": _mode_result(0.090, 0.070),
            "class-g-only": _mode_result(0.090, 0.070),
            "global-size": _mode_result(0.100, 0.080),
            "class-size": _mode_result(0.099, 0.080),
            "oracle-prior-lookup-only": _mode_result(0.100, 0.080),
            "oracle-class-global-size": _mode_result(0.300, 0.200),
            "oracle-class-size": _mode_result(0.304, 0.202),
        }
    }
    scored = [
        {
            "scene_id": f"scene-{index % 4}",
            "predicted_class": ("chair", "table", "book")[index % 3],
            "eligible": True,
            "class_prior_fallback": False,
            "G_global": 0.8,
            "G_class": 0.9,
            "S_global": 0.4,
            "S_class": 0.45,
        }
        for index in range(10)
    ]

    result = analyze_candidate_ranking(evaluation, scored)
    assert result["oracle_gate"]["oracle_size_has_discrimination_value"] is True
    assert result["oracle_gate"]["oracle_better_than_predicted"] is True


def test_choose_global_threshold_uses_dev_scenes_and_breaks_tie_high() -> None:
    candidate_rows = [
        {
            "raw_instance_id": 7,
            "class_id": 0,
            "eligible": True,
            "S_global": 0.9,
        }
    ]
    payload = {
        "iou": np.asarray([[0.75]]),
        "gt_classes": np.asarray([0]),
        "candidate_rows": candidate_rows,
    }
    evaluation = {"_scene_payloads": {"a": payload, "b": payload}}
    result = choose_global_threshold(evaluation, ["a", "b"], [0.5, 0.8])
    assert result["threshold"] == pytest.approx(0.8)
    assert result["scene_equal_candidate_f1_025"] == pytest.approx(1.0)
    assert result["retained_true_positives_025"] == 2


def test_choose_global_threshold_scores_the_retained_set_with_maximum_matching() -> None:
    candidate_rows = [
        {
            "raw_instance_id": 10,
            "class_id": 0,
            "eligible": True,
            "S_global": 0.9,
        },
        {
            "raw_instance_id": 11,
            "class_id": 0,
            "eligible": True,
            "S_global": 0.8,
        },
    ]
    # A score-greedy match would assign candidate 10 to GT 0 and strand
    # candidate 11.  Maximum-cardinality matching instead assigns 10 -> 1 and
    # 11 -> 0, correctly measuring the quality of the accepted candidate set.
    payload = {
        "iou": np.asarray([[0.8, 0.7], [0.7, 0.0]]),
        "gt_classes": np.asarray([0, 0]),
        "candidate_rows": candidate_rows,
    }
    evaluation = {"_scene_payloads": {"a": payload}}

    result = choose_global_threshold(evaluation, ["a"], [0.5])

    assert result["scene_equal_candidate_f1_025"] == pytest.approx(1.0)
    assert result["retained_true_positives_025"] == 2
    assert result["false_positive_count_025"] == 0
    assert result["false_negative_count_025"] == 0


def test_paired_bootstrap_vector_api_is_seeded_and_strictly_finite() -> None:
    first = paired_bootstrap([0.01, 0.02, -0.01], samples=1000, seed=17)
    second = paired_bootstrap([0.01, 0.02, -0.01], samples=1000, seed=17)
    assert first == second
    assert first["physical_scene_count"] == 3
    assert first["ci95_low"] == pytest.approx(first["paired_bootstrap_ci95"][0])


def test_evaluate_candidate_scenes_reconstructs_members_in_gt_only_adapter(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.npz"
    np.savez(trace, merged_partition=np.asarray([0, 0], dtype=np.int64))
    gt = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0], dtype=np.int64),
        instance=np.asarray([1, 1], dtype=np.int64),
    )
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    monkeypatch.setattr(
        evaluation_module,
        "load_ground_truth_npz",
        lambda *_args, **_kwargs: (xyz, gt),
    )
    monkeypatch.setattr(evaluation_module, "load_ply_xyz", lambda *_args: xyz)
    monkeypatch.setattr(
        clean_evaluation,
        "gt_point_to_gaussian_mapping",
        lambda *_args, **_kwargs: (
            np.asarray([0, 1], dtype=np.int64),
            {"mapped_fraction": 1.0},
        ),
    )
    node = {
        "shrunk": {
            "geometry": {
                name: {
                    "q25": math.log(0.05),
                    "q50": math.log(0.10),
                    "q75": math.log(0.20),
                }
                for name in (
                    "log_extent_short_m",
                    "log_extent_mid_m",
                    "log_extent_long_m",
                )
            }
        }
    }
    row = {
        "scene_id": "scene",
        "candidate_id": 0,
        "raw_instance_id": 0,
        "source": "global",
        "point_count": 2,
        "metric_extents_m": [0.05, 0.10, 0.20],
        # This is the 32-class index and must not be compared to SAGA20 GT IDs.
        "predicted_class_index": 17,
        "predicted_class": "chair",
        "Q": 0.9,
        "eligible": True,
        "G_global": 1.0,
        "S_global": 0.9,
        "G_class": 1.0,
        "S_class": 0.9,
    }
    result = evaluate_candidate_scenes(
        scene_ids=["scene"],
        scenes={"scene": {"base_path": str(tmp_path)}},
        gt_dir=tmp_path,
        snapshots={
            "scene": {"baseline_trace": str(trace), "rows": [row]}
        },
        taxonomy=SimpleNamespace(canonical_classes=("chair",)),
        size_spec={
            "boundaries_m": {
                "tiny_max_m": 0.2,
                "small_max_m": 0.5,
                "medium_max_m": 1.0,
            }
        },
        radius_m=0.05,
        min_region_size=1,
        priors={"global": node, "categories": {"chair": node}},
    )
    aggregate = result["mode_results"]["global-size"]["views"]["all"][
        "aggregate"
    ]["thresholds"]["050"]
    assert aggregate["match_count"] == 1
    assert result["rows"][0]["best_same_class_iou"] == pytest.approx(1.0)
    assert "member_indices" not in result["rows"][0]
    assert {
        "q-only",
        "global-g-only",
        "class-g-only",
        "global-size",
        "class-size",
        "oracle-prior-lookup-only",
        "oracle-class-global-size",
        "oracle-class-size",
    } == set(result["mode_results"])


def test_full_oracle_relabels_only_its_diagnostic_view(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.npz"
    np.savez(trace, merged_partition=np.asarray([0, 0], dtype=np.int64))
    gt = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0], dtype=np.int64),
        instance=np.asarray([1, 1], dtype=np.int64),
    )
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    monkeypatch.setattr(
        evaluation_module,
        "load_ground_truth_npz",
        lambda *_args, **_kwargs: (xyz, gt),
    )
    monkeypatch.setattr(evaluation_module, "load_ply_xyz", lambda *_args: xyz)
    monkeypatch.setattr(
        clean_evaluation,
        "gt_point_to_gaussian_mapping",
        lambda *_args, **_kwargs: (
            np.asarray([0, 1], dtype=np.int64),
            {"mapped_fraction": 1.0},
        ),
    )
    geometry = {
        name: {"q25": -3.0, "q50": -2.0, "q75": 0.0}
        for name in (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        )
    }
    node = {"shrunk": {"geometry": geometry}}
    row = {
        "scene_id": "scene",
        "candidate_id": 0,
        "raw_instance_id": 0,
        "source": "global",
        "point_count": 2,
        "metric_extents_m": [0.05, 0.10, 0.20],
        "predicted_class": "table",
        "Q": 0.9,
        "eligible": True,
        "G_global": 1.0,
        "S_global": 0.9,
        "G_class": 1.0,
        "S_class": 0.9,
    }
    result = evaluate_candidate_scenes(
        scene_ids=["scene"],
        scenes={"scene": {"base_path": str(tmp_path)}},
        gt_dir=tmp_path,
        snapshots={"scene": {"baseline_trace": str(trace), "rows": [row]}},
        taxonomy=SimpleNamespace(canonical_classes=("chair", "table")),
        size_spec={
            "boundaries_m": {
                "tiny_max_m": 0.2,
                "small_max_m": 0.5,
                "medium_max_m": 1.0,
            }
        },
        radius_m=0.05,
        min_region_size=1,
        priors={"global": node, "categories": {"chair": node, "table": node}},
    )

    automatic = result["mode_results"]["oracle-prior-lookup-only"]["views"][
        "all"
    ]["aggregate"]["thresholds"]["050"]
    full_global = result["mode_results"]["oracle-class-global-size"]["views"][
        "all"
    ]["aggregate"]["thresholds"]["050"]
    full_class = result["mode_results"]["oracle-class-size"]["views"]["all"][
        "aggregate"
    ]["thresholds"]["050"]
    assert automatic["match_count"] == 0
    assert full_global["match_count"] == 1
    assert full_class["match_count"] == 1
    # Relabelling alone cannot masquerade as evidence for a useful size prior:
    # both full-oracle arms use the identical GT-derived evaluation class.
    assert full_class["scene_equal_candidate_ap"] == pytest.approx(
        full_global["scene_equal_candidate_ap"]
    )
    assert result["rows"][0]["class_id"] == 1
    assert result["rows"][0]["oracle_class_id"] == 0
    assert result["rows"][0]["full_oracle_eligible"] is True


def test_full_oracle_does_not_inject_class_at_iou_quarter(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = tmp_path / "trace.npz"
    np.savez(trace, merged_partition=np.asarray([0, -1, -1, -1], dtype=np.int64))
    gt = GroundTruthScene(
        "scene",
        semantic=np.asarray([0, 0, 0, 0], dtype=np.int64),
        instance=np.asarray([1, 1, 1, 1], dtype=np.int64),
    )
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    monkeypatch.setattr(
        evaluation_module,
        "load_ground_truth_npz",
        lambda *_args, **_kwargs: (xyz, gt),
    )
    monkeypatch.setattr(evaluation_module, "load_ply_xyz", lambda *_args: xyz)
    monkeypatch.setattr(
        clean_evaluation,
        "gt_point_to_gaussian_mapping",
        lambda *_args, **_kwargs: (
            np.arange(4, dtype=np.int64),
            {"mapped_fraction": 1.0},
        ),
    )
    geometry = {
        name: {"q25": -3.0, "q50": -2.0, "q75": 0.0}
        for name in (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        )
    }
    node = {"shrunk": {"geometry": geometry}}
    result = evaluate_candidate_scenes(
        scene_ids=["scene"],
        scenes={"scene": {"base_path": str(tmp_path)}},
        gt_dir=tmp_path,
        snapshots={
            "scene": {
                "baseline_trace": str(trace),
                "rows": [
                    {
                        "scene_id": "scene",
                        "candidate_id": 0,
                        "raw_instance_id": 0,
                        "source": "global",
                        "point_count": 1,
                        "metric_extents_m": [0.05, 0.10, 0.20],
                        "predicted_class": "table",
                        "Q": 0.9,
                        "eligible": False,
                        "G_global": 1.0,
                        "S_global": 0.9,
                        "G_class": 1.0,
                        "S_class": 0.9,
                    }
                ],
            }
        },
        taxonomy=SimpleNamespace(canonical_classes=("chair", "table")),
        size_spec={
            "boundaries_m": {
                "tiny_max_m": 0.2,
                "small_max_m": 0.5,
                "medium_max_m": 1.0,
            }
        },
        radius_m=0.05,
        min_region_size=1,
        priors={"global": node, "categories": {"chair": node, "table": node}},
    )
    row = result["rows"][0]
    assert row["best_geometric_iou"] == pytest.approx(0.25)
    assert row["oracle_class_id"] is None
    assert row["full_oracle_supported"] is False
    assert row["full_oracle_eligible"] is False
    assert (
        result["mode_results"]["oracle-class-size"]["views"]["all"][
            "aggregate"
        ]["candidate_count"]
        == 0
    )
