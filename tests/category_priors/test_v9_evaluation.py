from __future__ import annotations

import numpy as np
import pytest

from category_priors.v9_evaluation import (
    CandidateSupport,
    GroundTruthSupport,
    SceneMethodMetrics,
    evaluate_object_candidates,
    score_iou_spearman,
    stage2_oracle_gate,
    stage3_uniform_health_gate,
    stage4_prior_gate,
)


def _candidate(
    scene: str,
    candidate_id: int,
    ids: list[int],
    class_name: str,
    score: float,
) -> CandidateSupport:
    return CandidateSupport(scene, candidate_id, np.asarray(ids), class_name, score)


def _gt(
    scene: str,
    instance_id: int,
    ids: list[int],
    class_name: str,
    size_bin: str,
    *,
    valid: bool = True,
) -> GroundTruthSupport:
    return GroundTruthSupport(
        scene,
        instance_id,
        np.asarray(ids),
        class_name,
        size_bin=size_bin,
        support_count=100,
        official_valid=valid,
    )


def test_candidate_evaluation_separates_geometry_class_and_gt_recall() -> None:
    gt = [
        _gt("a", 0, [0, 1, 2, 3], "chair", "tiny"),
        _gt("a", 1, [10, 11, 12, 13], "table", "medium"),
        _gt("b", 0, [0, 1, 2, 3], "book", "small"),
        _gt("b", 1, [20, 21], "chair", "tiny", valid=False),
    ]
    candidates = [
        _candidate("a", 0, [0, 1, 2, 3], "chair", 0.9),
        _candidate("a", 1, [10, 11, 12, 13], "chair", 0.2),
        _candidate("a", 2, [30, 31], "chair", 0.1),
        _candidate("b", 0, [0, 1, 2, 3, 4, 5], "book", 0.8),
    ]

    result = evaluate_object_candidates(candidates, gt)

    assert result["official_valid_gt_count"] == 3
    assert result["geometric"]["candidate_match_050_count"] == 3
    assert result["geometric"]["candidate_precision_050"] == pytest.approx(0.75)
    assert result["geometric"]["recall_050"] == 1.0
    assert result["same_class"]["candidate_match_050_count"] == 2
    assert result["same_class_candidate_precision_025"] == pytest.approx(0.5)
    assert result["same_class"]["recall_050"] == pytest.approx(2 / 3)
    assert result["tiny_small"]["official_valid_gt_count"] == 2
    assert result["geometric_tiny_small_recall_025"] == 1.0
    assert result["tiny_small_recall_050"] == 1.0
    assert result["geometric_match_050_scene_count"] == 2
    assert result["same_class_match_050_scene_count"] == 2
    assert result["score_iou_spearman"] > 0.8
    wrong_class = result["per_candidate"][1]
    assert wrong_class["geometric_best_iou"] == 1.0
    assert wrong_class["same_class_best_iou"] == 0.0


def test_spearman_uses_average_ties_and_constant_input_is_zero() -> None:
    assert score_iou_spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert score_iou_spearman([1, 1, 1], [0, 0.5, 1]) == 0.0
    with pytest.raises(ValueError, match="equal length"):
        score_iou_spearman([1], [1, 2])


def test_stage2_gate_requires_geometry_and_tiny_small_oracle() -> None:
    metrics = {
        "geometric_match_050_count": 6,
        "geometric_tiny_small_recall_025": 0.20,
    }
    assert stage2_oracle_gate(metrics)["passed"]
    metrics["geometric_match_050_count"] = 5
    result = stage2_oracle_gate(metrics)
    assert not result["passed"]
    assert not result["checks"]["geometric_match_050_at_least_6"]


def _healthy_stage3() -> dict[str, float | int]:
    return {
        "geometric_match_050_count": 16,
        "geometric_match_050_scene_count": 4,
        "same_class_match_050_count": 12,
        "same_class_match_050_scene_count": 4,
        "same_class_candidate_precision_025": 0.10,
        "tiny_small_recall_025": 0.20,
        "gaussian_micro_precision": 0.35,
        "unsupported_instance_fraction": 0.30,
        "gt_recall": 0.56,
        "map_50_95": 0.049,
        "ap50": 0.098,
        "predicted_instance_count": 25,
        "score_iou_spearman": 0.20,
        "orphan_gaussian_count": 0,
        "negative_metadata_count": 0,
    }


def test_stage3_gate_uses_two_distinct_baselines_and_output_contract() -> None:
    t1 = {
        "gaussian_micro_precision": 0.30,
        "unsupported_instance_fraction": 0.39,
        "gt_recall": 0.60,
    }
    f10k = {"map_50_95": 0.05, "ap50": 0.10, "predicted_instance_count": 20}
    bank = _healthy_stage3()
    healthy = stage3_uniform_health_gate(bank, t1_b1=t1, f10k_b0=f10k)
    assert healthy["passed"]
    assert (
        healthy["gt_recall_semantics"]
        == "unique_official_gt_instance_macro_coverage"
    )
    assert healthy["checks"][
        "unique_official_gt_instance_recall_drop_at_most_005"
    ]

    bank["orphan_gaussian_count"] = 1
    failed = stage3_uniform_health_gate(bank, t1_b1=t1, f10k_b0=f10k)
    assert not failed["passed"]
    assert not failed["checks"]["orphan_gaussian_count_zero"]


def _scene_metrics(
    scene_id: str,
    value: float,
    *,
    tiny_matches: int,
    fp: int = 10,
    tp: int = 10,
) -> SceneMethodMetrics:
    return SceneMethodMetrics(
        scene_id,
        value,
        tiny_small_match_050_count=tiny_matches,
        tiny_small_gt_count=10,
        false_positive_count=fp,
        true_positive_count=tp,
    )


def test_stage4_prior_gate_requires_mechanical_effect_and_registered_benefit() -> None:
    uniform = [
        _scene_metrics(str(index), 0.10, tiny_matches=2) for index in range(5)
    ]
    deltas = [0.004, 0.004, 0.004, -0.0005, -0.0005]
    data = [
        _scene_metrics(str(index), 0.10 + delta, tiny_matches=3, fp=11)
        for index, delta in enumerate(deltas)
    ]
    result = stage4_prior_gate(
        uniform,
        data,
        candidate_score_deltas=[0.02] + [0.0] * 9,
    )
    assert result["passed"]
    assert result["mean_map_delta"] == pytest.approx(0.0022)
    assert result["positive_scene_count"] == 3
    assert result["negative_scene_count"] == 2
    assert result["tiny_small_recall_050_delta"] == pytest.approx(0.10)
    assert result["intervention_fraction"] == pytest.approx(0.10)

    no_effect = stage4_prior_gate(uniform, data)
    assert not no_effect["passed"]
    assert not no_effect["checks"]["prior_mechanically_effective"]


def test_stage4_prior_gate_accepts_tiny_small_alternative_without_map_loss() -> None:
    uniform = [
        _scene_metrics(str(index), 0.10, tiny_matches=2) for index in range(5)
    ]
    deltas = [0.0002, 0.0002, 0.0002, -0.0002, -0.0002]
    data = [
        _scene_metrics(str(index), 0.10 + delta, tiny_matches=3)
        for index, delta in enumerate(deltas)
    ]
    result = stage4_prior_gate(
        uniform, data, accepted_or_ownership_changed=True
    )
    assert result["passed"]
    assert result["mean_map_delta"] == pytest.approx(0.00004)


def test_stage4_gate_rejects_duplicate_scenes_and_changed_gt_denominator() -> None:
    row = _scene_metrics("a", 0.1, tiny_matches=2)
    with pytest.raises(ValueError, match="unique"):
        stage4_prior_gate([row, row], [row], accepted_or_ownership_changed=True)
    changed_gt = SceneMethodMetrics(
        "a",
        0.1,
        tiny_small_match_050_count=2,
        tiny_small_gt_count=11,
        true_positive_count=10,
    )
    with pytest.raises(ValueError, match="denominators"):
        stage4_prior_gate([row], [changed_gt], accepted_or_ownership_changed=True)
