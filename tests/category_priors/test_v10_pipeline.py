from __future__ import annotations

from pathlib import Path

import pytest

from category_priors.v10_pipeline import (
    HOLDOUT5,
    final48_gate,
    holdout5_gate,
    physical_scene_macro_gate,
    select_best_prior_condition,
    select_late_classifier,
    select_pair_reconstruction_arm,
    select_uniform_threshold,
    stage1_structure_gate,
    stage2_uniform_health_gate,
    stage3_prior_gate,
    write_v10b_identity_training_proposal,
)


def test_stage1_gate_requires_real_association_precision_and_bounded_count() -> None:
    metrics = {
        "geometric_match_050_count": 6,
        "geometric_candidate_precision_025": 0.10,
        "geometric_tiny_small_recall_025": 0.20,
        "identifiable_association_precision": 0.50,
        "candidate_count": 15,
    }
    assert stage1_structure_gate(metrics, p0r0_candidate_count=10)["passed"]
    metrics["identifiable_association_precision"] = 0.49
    result = stage1_structure_gate(metrics, p0r0_candidate_count=10)
    assert not result["passed"]
    assert not result["checks"]["identifiable_association_precision_at_least_050"]


def test_pair_arm_selection_uses_registered_lexicographic_order() -> None:
    rows = []
    for arm in ("P0R0", "P1R0", "P0R1", "P1R1"):
        rows.append(
            {
                "condition": arm,
                "geometric_match_050_count": 6,
                "geometric_tiny_small_recall_025": 0.2,
                "geometric_candidate_precision_025": 0.1,
            }
        )
    assert select_pair_reconstruction_arm(rows)["selected"] == "P0R0"
    rows[-1]["geometric_match_050_count"] = 7
    selection = select_pair_reconstruction_arm(rows)
    assert selection["selected"] == "P1R1"
    assert selection["registered_final_structure"] == "VC1"


def test_late_classifier_uses_accuracy_and_registered_close_tie() -> None:
    close = {
        "mv-label": {
            "geometric_candidate_match_025_count": 100,
            "late_classifier_correct_025_count": 60,
        },
        "codebook": {
            "geometric_candidate_match_025_count": 100,
            "late_classifier_correct_025_count": 61,
        },
    }
    assert select_late_classifier(close)["selected"] == "mv-label"
    close["codebook"]["late_classifier_correct_025_count"] = 70
    assert select_late_classifier(close)["selected"] == "codebook"


def test_uniform_health_gate_uses_b1_fixed_as_the_structure_baseline() -> None:
    b1 = {
        "gaussian_micro_precision": 0.30,
        "unsupported_instance_fraction": 0.40,
        "gt_recall": 0.60,
        "map_50_95": 0.050,
        "ap50": 0.100,
        "predicted_instance_count": 20,
    }
    bank = {
        "geometric_match_050_count": 16,
        "geometric_match_050_scene_count": 4,
        "same_class_match_050_count": 12,
        "same_class_match_050_scene_count": 4,
        "same_class_candidate_precision_025": 0.10,
        "tiny_small_recall_025": 0.20,
        "gaussian_micro_precision": 0.35,
        "unsupported_instance_fraction": 0.40,
        "gt_recall": 0.55,
        "map_50_95": 0.049,
        "ap50": 0.098,
        "predicted_instance_count": 25,
        "score_iou_spearman": 0.20,
        "orphan_gaussian_count": 0,
        "negative_metadata_count": 0,
    }
    result = stage2_uniform_health_gate(bank, b1_fixed=b1)
    assert result["passed"]
    bank["orphan_gaussian_count"] = 1
    assert not stage2_uniform_health_gate(bank, b1_fixed=b1)["passed"]


def test_holdout_gate_requires_exact_five_scenes_and_three_positive() -> None:
    uniform = [
        {"scene_id": scene, "map_50_95": 0.1, "tiny_small_recall_050": 0.2}
        for scene in HOLDOUT5
    ]
    deltas = (0.01, 0.01, 0.01, -0.001, -0.001)
    data = [
        {
            "scene_id": scene,
            "map_50_95": 0.1 + delta,
            "tiny_small_recall_050": 0.21,
        }
        for scene, delta in zip(HOLDOUT5, deltas, strict=True)
    ]
    assert holdout5_gate(uniform, data)["passed"]
    with pytest.raises(ValueError, match="exactly"):
        holdout5_gate(uniform[:-1], data)


def test_uniform_threshold_requires_structure_and_breaks_map_tie_upward() -> None:
    rows = [
        {
            "acceptance_threshold": threshold,
            "map_50_95": 0.2 if threshold in {0.15, 0.20} else 0.1,
            "structure_passed": threshold != 0.25,
        }
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.25)
    ]
    result = select_uniform_threshold(rows)
    assert result["passed"]
    assert result["selected_threshold"] == 0.20


def test_prior_gate_requires_mechanical_effect_benefit_and_scene_direction() -> None:
    uniform = [
        {
            "scene_id": str(index),
            "map_50_95": 0.1,
            "tiny_small_recall_050": 0.2,
            "false_positive_count": 10,
            "true_positive_count": 10,
        }
        for index in range(5)
    ]
    deltas = (0.004, 0.004, 0.004, -0.0005, -0.0005)
    data = [
        {
            "scene_id": str(index),
            "map_50_95": 0.1 + delta,
            "tiny_small_recall_050": 0.21,
            "false_positive_count": 11,
            "true_positive_count": 10,
        }
        for index, delta in enumerate(deltas)
    ]
    result = stage3_prior_gate(
        uniform, data, candidate_score_deltas=[0.02] + [0.0] * 9
    )
    assert result["passed"]
    assert result["mean_map_delta"] == pytest.approx(0.0022)
    assert not stage3_prior_gate(uniform, data)["passed"]


def test_best_prior_selection_uses_registered_metric_order() -> None:
    gates = {"D100": {"passed": True}, "D111": {"passed": True}}
    metrics = {
        "D100": {"map_50_95": 0.11, "tiny_small_recall_050": 0.2, "ap50": 0.3},
        "D111": {"map_50_95": 0.11, "tiny_small_recall_050": 0.2, "ap50": 0.3},
    }
    assert select_best_prior_condition(gates, metrics)["selected"] == "D100"


def test_physical_macro_and_final_bootstrap_gates() -> None:
    uniform = [
        {"scene_id": "scene1_00", "physical_scene_id": "scene1", "map_50_95": 0.1},
        {"scene_id": "scene1_01", "physical_scene_id": "scene1", "map_50_95": 0.2},
        {"scene_id": "scene2_00", "physical_scene_id": "scene2", "map_50_95": 0.1},
    ]
    data = [{**row, "map_50_95": row["map_50_95"] + 0.003} for row in uniform]
    macro = physical_scene_macro_gate(uniform, data)
    assert macro["passed"]
    assert macro["physical_scene_count"] == 2
    final = final48_gate(
        {"delta_map_50_95": 0.002, "paired_bootstrap_ci95": [0.0001, 0.004]}
    )
    assert final["passed"]


def test_v10b_proposal_requests_approval_and_is_not_silent(tmp_path: Path) -> None:
    path = write_v10b_identity_training_proposal(
        tmp_path / "V10B_IDENTITY_TRAINING_PROPOSAL.md",
        failed_stage="stage1",
        diagnosis={"candidate_precision": 0.03},
    )
    text = path.read_text("utf-8")
    assert "请批准或拒绝" in text
    assert "类别先验 replay 之前" in text
    assert "scene0474_01" in text
