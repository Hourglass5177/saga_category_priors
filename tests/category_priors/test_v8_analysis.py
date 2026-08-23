from __future__ import annotations

import numpy as np
import pytest

from category_priors.evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
)
from category_priors.v8_analysis import (
    V8_LIFTING_ARMS,
    aggregate_late_classifier_results,
    aggregate_lifting_factorial_rows,
    evaluate_loaded_object_bank_classifiers,
    gaussian_sets_to_gt_point_ids,
    paired_scannet_scene_bootstrap,
    pooled_scannet_metrics_from_scene_weights,
    precompute_scannet_scene_ap_events,
    select_late_classifier,
    select_stage1_combination,
    stage1_combination_gate,
    stage2_bank_health_gate,
)


def _arm_row(
    combination: str,
    match_050: int,
    tiny_recall: float,
    precision: float,
    *,
    masks: int = 20,
    seconds: float = 10.0,
) -> dict[str, object]:
    return {
        "combination": combination,
        "geometric_greedy_match_050_count": match_050,
        "tiny_small_recall_025": tiny_recall,
        "fragment_precision_025": precision,
        "mask_count": masks,
        "runtime_seconds": seconds,
    }


def test_gaussian_sets_map_to_shared_gt_point_indices() -> None:
    mapping = np.array([2, 0, -1, 2, 1, 0])
    memberships = gaussian_sets_to_gt_point_ids(
        [np.array([0, 2]), np.array([1])], mapping, gaussian_count=3
    )
    assert memberships[0].tolist() == [0, 1, 3, 5]
    assert memberships[1].tolist() == [4]


def test_stage1_absolute_gate_requires_both_registered_thresholds() -> None:
    assert stage1_combination_gate(_arm_row("G-M1", 6, 0.20, 0.1))["passed"]
    assert not stage1_combination_gate(_arm_row("G-M1", 5, 0.20, 0.1))["passed"]
    assert not stage1_combination_gate(_arm_row("G-M1", 6, 0.199, 0.1))["passed"]


def test_stage1_selection_uses_registered_order_and_reports_factor_effects() -> None:
    rows = [
        _arm_row("G-M1", 6, 0.20, 0.30, masks=10, seconds=8),
        _arm_row("G-AM", 8, 0.22, 0.25, masks=10, seconds=12),
        _arm_row("S-M1", 8, 0.22, 0.25, masks=30, seconds=9),
        _arm_row("S-AM", 8, 0.22, 0.25, masks=30, seconds=14),
    ]
    result = select_stage1_combination(rows)

    assert result["passed"]
    assert result["selected_combination"] == "G-AM"
    assert result["factor_effects"]["lifting_at_G"]["substantive"]
    assert result["factor_effects"]["mask_at_AM"]["substantive"] is False


def test_stage1_exact_tie_prefers_m1_then_grounded_masks() -> None:
    rows = [_arm_row(name, 7, 0.25, 0.2) for name in V8_LIFTING_ARMS]
    result = select_stage1_combination(rows)
    assert result["selected_combination"] == "G-M1"


def test_factorial_aggregation_uses_counts_not_mean_of_scene_recalls() -> None:
    scene_rows = []
    for arm in V8_LIFTING_ARMS:
        scene_rows.extend(
            [
                {
                    "combination": arm,
                    "official_gt_count": 10,
                    "geometric_greedy_match_050_count": 5,
                    "semantic_greedy_match_050_count": 4,
                    "tiny_small_gt_count": 2,
                    "tiny_small_geometric_match_025_count": 1,
                    "tiny_small_semantic_match_025_count": 1,
                    "fragment_count": 4,
                    "fragment_match_025_count": 2,
                },
                {
                    "combination": arm,
                    "official_gt_count": 30,
                    "geometric_greedy_match_050_count": 3,
                    "semantic_greedy_match_050_count": 2,
                    "tiny_small_gt_count": 8,
                    "tiny_small_geometric_match_025_count": 1,
                    "tiny_small_semantic_match_025_count": 0,
                    "fragment_count": 6,
                    "fragment_match_025_count": 1,
                },
            ]
        )
    rows = aggregate_lifting_factorial_rows(scene_rows)
    assert rows[0]["geometric_greedy_recall_050"] == pytest.approx(8 / 40)
    assert rows[0]["tiny_small_recall_025"] == pytest.approx(2 / 10)
    assert rows[0]["fragment_precision_025"] == pytest.approx(3 / 10)


def test_late_classifier_prefers_mv_within_two_percentage_points() -> None:
    candidates = [np.array([0, 1]), np.array([2, 3])]
    gt = [np.array([0, 1]), np.array([2, 3])]
    tied = select_late_classifier(
        candidates, [1, 0], [1, 0], gt, [1, 1]
    )
    assert tied["mv_label_accuracy"] == tied["codebook_accuracy"] == 0.5
    assert tied["selected_classifier"] == "MV-label"

    codebook_wins = select_late_classifier(
        candidates, [0, 0], [1, 1], gt, [1, 1]
    )
    assert codebook_wins["selected_classifier"] == "codebook"


def test_late_classifier_aggregation_is_candidate_weighted() -> None:
    result = aggregate_late_classifier_results(
        [
            {
                "eligible_candidate_count": 1,
                "mv_correct_count": 1,
                "codebook_correct_count": 0,
            },
            {
                "eligible_candidate_count": 9,
                "mv_correct_count": 4,
                "codebook_correct_count": 9,
            },
        ]
    )
    assert result["mv_label_accuracy"] == 0.5
    assert result["codebook_accuracy"] == 0.9
    assert result["selected_classifier"] == "codebook"


def test_object_bank_classifier_comparison_aligns_by_track_id() -> None:
    metadata = {
        "point_count": 4,
        "classifiers": {
            "mv-label": {
                "candidates": [
                    {"candidate_id": 0, "track_id": 7, "class_id": 1},
                ]
            },
            "codebook": {
                "candidates": [
                    {"candidate_id": 0, "track_id": 7, "class_id": 2},
                ]
            },
        },
    }
    arrays = {
        "valid_track_ids": np.array([7]),
        "track_full_indptr": np.array([0, 2]),
        "track_full_ids": np.array([0, 1]),
        "full_candidate_indptr_mv": np.array([0, 2]),
        "full_candidate_ids_mv": np.array([0, 1]),
        "full_candidate_indptr_codebook": np.array([0, 2]),
        "full_candidate_ids_codebook": np.array([0, 1]),
    }
    result = evaluate_loaded_object_bank_classifiers(
        metadata=metadata,
        arrays=arrays,
        gt_nearest_gaussian=np.array([0, 1, 2, 3]),
        gt_memberships={
            "point_ids": [np.array([0, 1])],
            "class_ids": [1],
            "official_valid": [True],
        },
    )
    selection = result["classifier_selection"]
    assert result["track_ids"] == [7]
    assert selection["mv_label_accuracy"] == 1.0
    assert selection["codebook_accuracy"] == 0.0
    assert selection["selected_classifier"] == "MV-label"


def test_late_classifier_counts_double_abstention_as_two_errors() -> None:
    metadata = {
        "point_count": 4,
        "classifiers": {
            "mv-label": {
                "candidates": [
                    {"candidate_id": 0, "track_id": 7, "class_id": 1},
                ]
            },
            "codebook": {
                "candidates": [
                    {"candidate_id": 0, "track_id": 7, "class_id": 1},
                ]
            },
        },
    }
    arrays = {
        "valid_track_ids": np.array([7, 8]),
        "track_full_indptr": np.array([0, 2, 4]),
        "track_full_ids": np.array([0, 1, 2, 3]),
        "full_candidate_indptr_mv": np.array([0, 2]),
        "full_candidate_ids_mv": np.array([0, 1]),
        "full_candidate_indptr_codebook": np.array([0, 2]),
        "full_candidate_ids_codebook": np.array([0, 1]),
    }
    result = evaluate_loaded_object_bank_classifiers(
        metadata=metadata,
        arrays=arrays,
        gt_nearest_gaussian=np.array([0, 1, 2, 3]),
        gt_memberships={
            "point_ids": [np.array([0, 1]), np.array([2, 3])],
            "class_ids": [1, 1],
            "official_valid": [True, True],
        },
    )
    selection = result["classifier_selection"]
    assert result["track_ids"] == [7, 8]
    assert selection["eligible_candidate_count"] == 2
    assert selection["mv_label_accuracy"] == pytest.approx(0.5)
    assert selection["codebook_accuracy"] == pytest.approx(0.5)


def _bootstrap_fixture() -> tuple[
    list[GroundTruthScene], list[PredictedInstance], tuple[str, ...]
]:
    classes = ("chair", "table")
    scenes = [
        GroundTruthScene(
            "scene-a",
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        ),
        GroundTruthScene(
            "scene-b",
            np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([3, 3, 3, 3, 4, 4, 4, 4]),
        ),
    ]
    predictions = [
        PredictedInstance(
            "scene-a", 0, 0, 0.9,
            np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        ),
        # A lower-score duplicate exercises ScanNet's matched-GT duplicate
        # event rather than only the one-to-one happy path.
        PredictedInstance(
            "scene-a", 2, 0, 0.4,
            np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        ),
        PredictedInstance(
            "scene-a", 1, 1, 0.8,
            np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool),
        ),
        PredictedInstance(
            "scene-b", 0, 0, 0.7,
            np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool),
        ),
        PredictedInstance(
            "scene-b", 1, 1, 0.6,
            np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool),
        ),
    ]
    return scenes, predictions, classes


def test_scene_ap_events_reproduce_official_pooled_metrics_and_weights() -> None:
    scenes, predictions, classes = _bootstrap_fixture()
    overlaps = (0.50, 0.75)
    events = precompute_scannet_scene_ap_events(
        scenes, predictions, classes, overlaps=overlaps, min_region_size=2
    )
    direct = evaluate_instances(
        scenes, predictions, classes, overlaps=overlaps, min_region_size=2
    )
    pooled = pooled_scannet_metrics_from_scene_weights(events)
    for key, value in pooled["aggregate"].items():
        assert value == pytest.approx(direct["aggregate"][key])
    for class_name in classes:
        for key, value in pooled["per_class"][class_name].items():
            assert value == pytest.approx(direct["per_class"][class_name][key])

    # A weight of two is exactly equivalent to cloning that physical scene
    # under a fresh ID before running the official pooled evaluator.
    weighted = pooled_scannet_metrics_from_scene_weights(events, np.array([2, 1]))
    duplicated_scenes: list[GroundTruthScene] = []
    duplicated_predictions: list[PredictedInstance] = []
    for scene, multiplicity in zip(scenes, (2, 1)):
        source_predictions = [
            prediction for prediction in predictions
            if prediction.scene_id == scene.scene_id
        ]
        for copy_index in range(multiplicity):
            clone_id = f"{scene.scene_id}-copy-{copy_index}"
            duplicated_scenes.append(
                GroundTruthScene(clone_id, scene.semantic.copy(), scene.instance.copy())
            )
            duplicated_predictions.extend(
                PredictedInstance(
                    clone_id,
                    prediction.instance_id,
                    prediction.class_id,
                    prediction.score,
                    prediction.mask.copy(),
                )
                for prediction in source_predictions
            )
    direct_weighted = evaluate_instances(
        duplicated_scenes,
        duplicated_predictions,
        classes,
        overlaps=overlaps,
        min_region_size=2,
    )
    for key, value in weighted["aggregate"].items():
        assert value == pytest.approx(direct_weighted["aggregate"][key])


def test_paired_scene_bootstrap_uses_same_multinomial_draws() -> None:
    scenes, predictions, classes = _bootstrap_fixture()
    events = precompute_scannet_scene_ap_events(
        scenes, predictions, classes, overlaps=(0.50, 0.75), min_region_size=2
    )
    result = paired_scannet_scene_bootstrap(events, events, samples=128, seed=17)
    assert result["delta_map_50_95"] == pytest.approx(0.0)
    assert result["paired_bootstrap_ci95"] == pytest.approx([0.0, 0.0])
    assert result["finite_sample_count"] == 128


def _healthy_bank() -> dict[str, float | int]:
    return {
        "geometric_match_050_count": 16,
        "geometric_match_050_scene_count": 4,
        "same_class_match_050_count": 12,
        "same_class_match_050_scene_count": 4,
        "same_class_candidate_precision_025": 0.10,
        "tiny_small_recall_025": 0.20,
        "gaussian_micro_precision": 0.45,
        "unsupported_instance_fraction": 0.30,
        "gt_recall": 0.46,
        "map_50_95": 0.059,
        "ap50": 0.198,
        "predicted_instance_count": 25,
        "score_iou_spearman": 0.20,
    }


def _b1_fixed() -> dict[str, float | int]:
    return {
        "gaussian_micro_precision": 0.40,
        "unsupported_instance_fraction": 0.40,
        "gt_recall": 0.50,
        "map_50_95": 0.060,
        "ap50": 0.200,
        "predicted_instance_count": 20,
    }


def test_stage2_bank_health_requires_every_registered_check() -> None:
    healthy = stage2_bank_health_gate(_healthy_bank(), _b1_fixed())
    assert healthy["passed"]

    bad = _healthy_bank()
    bad["score_iou_spearman"] = 0.199
    failed = stage2_bank_health_gate(bad, _b1_fixed())
    assert not failed["passed"]
    assert not failed["checks"]["score_iou_spearman_at_least_020"]


def test_stage2_precision_or_unsupported_is_an_or_gate() -> None:
    bank = _healthy_bank()
    bank["gaussian_micro_precision"] = 0.41
    assert stage2_bank_health_gate(bank, _b1_fixed())["passed"]

    bank["unsupported_instance_fraction"] = 0.35
    assert not stage2_bank_health_gate(bank, _b1_fixed())["passed"]
