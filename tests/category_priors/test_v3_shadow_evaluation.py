from __future__ import annotations

import numpy as np

from category_priors.v3_shadow_evaluation import (
    _affinity_instance_metrics,
    evaluate_shadow_scene_arrays,
)


def test_shadow_oracle_marks_incremental_small_candidate_and_death_stage() -> None:
    gt_mask = np.asarray([True, True, True, True, False, False])
    gt = [{
        "class_id": 0,
        "canonical_class": "chair",
        "gt_instance_id": 7,
        "mask": gt_mask,
        "point_count": 4,
        "bbox_diag_m": 0.4,
        "physical_size_bin": "small",
        "below_official_min_region_size": True,
    }]
    candidates = [{
        "candidate_id": 0,
        "branch_class": "chair",
        "branch_class_index": 0,
        "active_branch_points": 4,
        "after_knn_points": 0,
        "after_filter_points": 0,
        "vote": {"winner_matches_branch": True, "branch_class_ratio": 0.8, "winner": "chair", "winner_ratio": 0.8, "background_ratio": 0.2},
        "global_pre_overlap": {"fraction": 0.0},
        "global_final_overlap": {"fraction": 0.0},
    }]
    rows, global_best = evaluate_shadow_scene_arrays(
        scene_id="scene0000_00",
        mode="exclusive",
        gt_instances=gt,
        mapped_branch_labels=np.asarray([0, 0, 0, 0, -1, -1]),
        candidates=candidates,
        final_predictions=[],
    )
    assert global_best[(0, 7)] == 0.0
    assert rows[0]["same_class_best_iou"] == 1.0
    assert rows[0]["new_oracle_match_025"]
    assert rows[0]["death_stage"] == "global_knn"


def test_shadow_oracle_uses_class_name_not_codebook_row() -> None:
    gt = [{
        "class_id": 0,
        "canonical_class": "chair",
        "gt_instance_id": 1,
        "mask": np.asarray([True, True]),
        "point_count": 2,
        "bbox_diag_m": 0.2,
        "physical_size_bin": "tiny",
        "below_official_min_region_size": True,
    }]
    rows, _ = evaluate_shadow_scene_arrays(
        scene_id="scene0000_00",
        mode="exact",
        gt_instances=gt,
        mapped_branch_labels=np.asarray([3, 3]),
        candidates=[{
            "candidate_id": 3,
            "branch_class": "chair",
            "branch_class_index": 19,
            "active_branch_points": 2,
            "after_knn_points": 2,
            "after_filter_points": 2,
            "vote": {"winner_matches_branch": True},
            "global_pre_overlap": {},
            "global_final_overlap": {},
        }],
        final_predictions=[],
    )
    assert rows[0]["same_class_best_iou"] == 1.0


def test_affinity_margin_compares_only_same_class_instances() -> None:
    gt = [
        {"class_id": 0, "gt_instance_id": 1, "mask": np.asarray([True, True, False, False])},
        {"class_id": 0, "gt_instance_id": 2, "mask": np.asarray([False, False, True, False])},
        {"class_id": 1, "gt_instance_id": 3, "mask": np.asarray([False, False, False, True])},
    ]
    features = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [1.0, 0.0]])
    result = _affinity_instance_metrics(gt, np.arange(4), features)
    assert result[(0, 1)]["intra_cosine"] == 1.0
    assert result[(0, 1)]["nearest_same_class_cosine"] == 0.8
    assert np.isclose(result[(0, 1)]["same_class_margin"], 0.2)
