from __future__ import annotations

"""Preregistered gates and failure hand-off for SAGA V10.

This module is deliberately independent of rendering, association and ground-
truth loading.  Evaluators pass compact metric dictionaries into these gates;
the runtime worker therefore cannot accidentally consume validation labels.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_EPSILON = 1e-12
DEV2 = ("scene0645_00", "scene0025_01")
DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)
PAIR_RECONSTRUCTION_ARMS = ("P0R0", "P1R0", "P0R1", "P1R1")
VIEW_CONSENSUS_ARM = "VC1"
PRIOR_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)


def _finite(metrics: Mapping[str, Any], key: str) -> float:
    value = float(metrics[key])
    if not np.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def stage1_structure_gate(
    metrics: Mapping[str, Any],
    *,
    p0r0_candidate_count: int,
) -> dict[str, Any]:
    """Apply the fixed two-scene V10 structure gate."""

    reference_count = int(p0r0_candidate_count)
    if reference_count < 0:
        raise ValueError("p0r0_candidate_count must be non-negative")
    candidate_count = int(metrics["candidate_count"])
    checks = {
        "geometric_match_050_at_least_6": int(
            metrics["geometric_match_050_count"]
        )
        >= 6,
        "candidate_precision_025_at_least_010": _finite(
            metrics, "geometric_candidate_precision_025"
        )
        >= 0.10,
        "tiny_small_recall_025_at_least_020": _finite(
            metrics, "geometric_tiny_small_recall_025"
        )
        >= 0.20,
        "identifiable_association_precision_at_least_050": _finite(
            metrics, "identifiable_association_precision"
        )
        >= 0.50,
        "candidate_count_at_most_1_5x_p0r0": candidate_count
        <= 1.5 * reference_count,
    }
    return {
        "schema": "saga-v10-stage1-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "geometric_match_050_count": int(
                metrics["geometric_match_050_count"]
            ),
            "geometric_candidate_precision_025": _finite(
                metrics, "geometric_candidate_precision_025"
            ),
            "geometric_tiny_small_recall_025": _finite(
                metrics, "geometric_tiny_small_recall_025"
            ),
            "identifiable_association_precision": _finite(
                metrics, "identifiable_association_precision"
            ),
            "candidate_count": candidate_count,
            "p0r0_candidate_count": reference_count,
        },
    }


def select_pair_reconstruction_arm(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Rank the frozen 2x2 arms without inventing a composite score."""

    by_arm = {str(row["condition"]): row for row in rows}
    missing = [arm for arm in PAIR_RECONSTRUCTION_ARMS if arm not in by_arm]
    if missing:
        raise ValueError(f"missing pair/reconstruction arms: {missing}")

    def rank(arm: str) -> tuple[float, float, float, int, str]:
        row = by_arm[arm]
        # Registered order: geometric matches, tiny/small recall, precision,
        # then simpler structure.  P0/R0 each count as one simpler choice.
        complexity = int(arm[1]) + int(arm[3])
        return (
            -float(row["geometric_match_050_count"]),
            -float(row["geometric_tiny_small_recall_025"]),
            -float(row["geometric_candidate_precision_025"]),
            complexity,
            arm,
        )

    selected = min(PAIR_RECONSTRUCTION_ARMS, key=rank)
    return {
        "schema": "saga-v10-pair-reconstruction-selection-v1",
        "selected": selected,
        "use": "causal_audit_only",
        "registered_final_structure": VIEW_CONSENSUS_ARM,
        "registered_order": [
            "geometric_match_050_count",
            "geometric_tiny_small_recall_025",
            "geometric_candidate_precision_025",
            "lower_complexity",
        ],
        "rows": [dict(by_arm[arm]) for arm in PAIR_RECONSTRUCTION_ARMS],
    }


def select_late_classifier(
    classifier_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the one late classifier from geometric IoU>=.25 candidates."""

    expected = {"mv-label", "codebook"}
    if set(classifier_metrics) != expected:
        raise ValueError("late-classifier metrics must contain mv-label and codebook")
    accuracy: dict[str, float] = {}
    for classifier in sorted(expected):
        row = classifier_metrics[classifier]
        denominator = int(row["geometric_candidate_match_025_count"])
        numerator = int(row["late_classifier_correct_025_count"])
        if numerator > denominator:
            raise ValueError("same-class matches cannot exceed geometric matches")
        accuracy[classifier] = numerator / denominator if denominator else 0.0
    if abs(accuracy["mv-label"] - accuracy["codebook"]) <= 0.02 + _EPSILON:
        selected = "mv-label"
        reason = "accuracy difference <= 0.02; registered tie preference"
    else:
        selected = max(accuracy, key=lambda key: (accuracy[key], key == "mv-label"))
        reason = "higher geometric-candidate class accuracy"
    return {
        "schema": "saga-v10-late-classifier-selection-v1",
        "selected": selected,
        "accuracy_at_geometric_iou_025": accuracy,
        "reason": reason,
    }


def stage2_uniform_health_gate(
    bank: Mapping[str, Any],
    *,
    b1_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply all eight-scene V10-U health checks."""

    precision_gain = _finite(bank, "gaussian_micro_precision") - _finite(
        b1_fixed, "gaussian_micro_precision"
    )
    unsupported_drop = _finite(
        b1_fixed, "unsupported_instance_fraction"
    ) - _finite(bank, "unsupported_instance_fraction")
    recall_drop = _finite(b1_fixed, "gt_recall") - _finite(bank, "gt_recall")
    b1_instances = int(b1_fixed["predicted_instance_count"])
    checks = {
        "geometric_match_050_at_least_16": int(
            bank["geometric_match_050_count"]
        )
        >= 16,
        "geometric_match_050_scene_count_at_least_4": int(
            bank["geometric_match_050_scene_count"]
        )
        >= 4,
        "same_class_match_050_at_least_12": int(
            bank["same_class_match_050_count"]
        )
        >= 12,
        "same_class_match_050_scene_count_at_least_4": int(
            bank["same_class_match_050_scene_count"]
        )
        >= 4,
        "same_class_candidate_precision_025_at_least_010": _finite(
            bank, "same_class_candidate_precision_025"
        )
        >= 0.10,
        "tiny_small_recall_025_at_least_020": _finite(
            bank, "tiny_small_recall_025"
        )
        >= 0.20,
        "precision_gain_or_unsupported_drop": precision_gain >= 0.05 - _EPSILON
        or unsupported_drop >= 0.10 - _EPSILON,
        "gt_recall_drop_at_most_005": recall_drop <= 0.05 + _EPSILON,
        "map_not_below_b1_by_0001": _finite(bank, "map_50_95")
        >= _finite(b1_fixed, "map_50_95") - 0.001 - _EPSILON,
        "ap50_not_below_b1_by_0002": _finite(bank, "ap50")
        >= _finite(b1_fixed, "ap50") - 0.002 - _EPSILON,
        "instance_count_at_most_1_25x_b1": int(bank["predicted_instance_count"])
        <= 1.25 * b1_instances,
        "score_iou_spearman_at_least_020": _finite(
            bank, "score_iou_spearman"
        )
        >= 0.20,
        "orphan_zero": int(bank["orphan_gaussian_count"]) == 0,
        "negative_metadata_zero": int(bank["negative_metadata_count"]) == 0,
    }
    return {
        "schema": "saga-v10-stage2-uniform-health-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "diagnostic_deltas": {
            "gaussian_micro_precision": precision_gain,
            "unsupported_instance_fraction": unsupported_drop,
            "gt_recall_drop": recall_drop,
        },
    }


def holdout5_gate(
    uniform_rows: Sequence[Mapping[str, Any]],
    data_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the canonical five-scene holdout gate."""

    uniform = {str(row["scene_id"]): row for row in uniform_rows}
    data = {str(row["scene_id"]): row for row in data_rows}
    expected = set(HOLDOUT5)
    if set(uniform) != expected or set(data) != expected:
        raise ValueError("holdout rows must contain exactly the five canonical scenes")
    deltas = np.asarray(
        [
            float(data[scene]["map_50_95"])
            - float(uniform[scene]["map_50_95"])
            for scene in HOLDOUT5
        ],
        dtype=np.float64,
    )
    tiny_u = sum(float(uniform[scene]["tiny_small_recall_050"]) for scene in HOLDOUT5)
    tiny_d = sum(float(data[scene]["tiny_small_recall_050"]) for scene in HOLDOUT5)
    checks = {
        "mean_delta_map_positive": float(np.mean(deltas)) > 0.0,
        "at_least_3_of_5_positive": int(np.count_nonzero(deltas > 0.0)) >= 3,
        "tiny_small_recall_050_positive": tiny_d - tiny_u > 0.0,
    }
    return {
        "schema": "saga-v10-holdout5-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "mean_map_delta": float(np.mean(deltas)),
        "positive_scene_count": int(np.count_nonzero(deltas > 0.0)),
        "tiny_small_recall_050_delta": float((tiny_d - tiny_u) / len(HOLDOUT5)),
        "per_scene_delta": {
            scene: float(delta) for scene, delta in zip(HOLDOUT5, deltas, strict=True)
        },
    }


def select_uniform_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the single registered U000 threshold on DEV2 only."""

    by_threshold: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        threshold = float(row["acceptance_threshold"])
        if threshold not in PRIOR_THRESHOLDS:
            raise ValueError(f"unregistered U000 threshold: {threshold}")
        if threshold in by_threshold:
            raise ValueError(f"duplicate U000 threshold: {threshold}")
        by_threshold[threshold] = row
    if set(by_threshold) != set(PRIOR_THRESHOLDS):
        raise ValueError("threshold rows must contain the five registered values")
    eligible = [
        threshold
        for threshold, row in by_threshold.items()
        if bool(row.get("structure_passed", False))
    ]
    if not eligible:
        return {
            "schema": "saga-v10-u000-threshold-selection-v1",
            "passed": False,
            "selected_threshold": None,
            "reason": "no threshold passed the frozen structure gate",
        }
    selected = min(
        eligible,
        key=lambda threshold: (
            -float(by_threshold[threshold]["map_50_95"]),
            -threshold,
        ),
    )
    return {
        "schema": "saga-v10-u000-threshold-selection-v1",
        "passed": True,
        "selected_threshold": float(selected),
        "selected_map_50_95": float(by_threshold[selected]["map_50_95"]),
        "rows": [dict(by_threshold[value]) for value in PRIOR_THRESHOLDS],
    }


def stage3_prior_gate(
    uniform_rows: Sequence[Mapping[str, Any]],
    data_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_score_deltas: Sequence[float] = (),
    accepted_or_ownership_changed: bool = False,
) -> dict[str, Any]:
    """Apply the frozen DEV8 category-prior gate to one D condition."""

    uniform = {str(row["scene_id"]): row for row in uniform_rows}
    data = {str(row["scene_id"]): row for row in data_rows}
    if len(uniform) != len(uniform_rows) or len(data) != len(data_rows):
        raise ValueError("prior-gate scene IDs must be unique")
    if set(uniform) != set(data) or not uniform:
        raise ValueError("uniform and data rows must contain identical scenes")
    ordered = sorted(uniform)
    map_deltas = np.asarray(
        [
            float(data[scene]["map_50_95"])
            - float(uniform[scene]["map_50_95"])
            for scene in ordered
        ],
        dtype=np.float64,
    )
    tiny_deltas = np.asarray(
        [
            float(data[scene]["tiny_small_recall_050"])
            - float(uniform[scene]["tiny_small_recall_050"])
            for scene in ordered
        ],
        dtype=np.float64,
    )
    fp_u = sum(int(uniform[scene]["false_positive_count"]) for scene in ordered)
    tp_u = sum(int(uniform[scene]["true_positive_count"]) for scene in ordered)
    fp_d = sum(int(data[scene]["false_positive_count"]) for scene in ordered)
    tp_d = sum(int(data[scene]["true_positive_count"]) for scene in ordered)
    ratio_u = fp_u / max(tp_u, 1)
    ratio_d = fp_d / max(tp_d, 1)
    intervention_fraction = (
        float(np.mean(np.abs(np.asarray(candidate_score_deltas, dtype=np.float64)) >= 0.01))
        if len(candidate_score_deltas)
        else 0.0
    )
    mechanical = intervention_fraction >= 0.10 - _EPSILON or bool(
        accepted_or_ownership_changed
    )
    mean_map = float(np.mean(map_deltas))
    mean_tiny = float(np.mean(tiny_deltas))
    benefit = mean_map >= 0.002 - _EPSILON or (
        mean_tiny >= 0.01 - _EPSILON and mean_map >= -0.0005 - _EPSILON
    )
    checks = {
        "prior_mechanically_effective": mechanical,
        "registered_benefit": benefit,
        "positive_scenes_more_than_negative": int(np.count_nonzero(map_deltas > 0))
        > int(np.count_nonzero(map_deltas < 0)),
        "fp_tp_not_worse_than_20_percent": ratio_d <= 1.2 * ratio_u + _EPSILON,
    }
    return {
        "schema": "saga-v10-stage3-prior-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "mean_map_delta": mean_map,
        "tiny_small_recall_050_delta": mean_tiny,
        "positive_scene_count": int(np.count_nonzero(map_deltas > 0)),
        "negative_scene_count": int(np.count_nonzero(map_deltas < 0)),
        "fp_tp_ratio_uniform": float(ratio_u),
        "fp_tp_ratio_data": float(ratio_d),
        "intervention_fraction": intervention_fraction,
    }


def select_best_prior_condition(
    gates: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select at most one passing D condition using the registered order."""

    data_conditions = tuple(sorted(key for key in gates if key != "U000"))
    passing = [condition for condition in data_conditions if bool(gates[condition]["passed"])]
    if not passing:
        return {
            "schema": "saga-v10-prior-selection-v1",
            "passed": False,
            "selected": None,
            "reason": "no data-driven condition passed the DEV8 gate",
        }

    def factor_count(condition: str) -> int:
        suffix = condition[1:]
        return sum(character == "1" for character in suffix)

    selected = min(
        passing,
        key=lambda condition: (
            -float(metrics[condition]["map_50_95"]),
            -float(metrics[condition]["tiny_small_recall_050"]),
            -float(metrics[condition]["ap50"]),
            factor_count(condition),
            condition,
        ),
    )
    return {
        "schema": "saga-v10-prior-selection-v1",
        "passed": True,
        "selected": selected,
        "passing_conditions": passing,
        "registered_order": [
            "map_50_95",
            "tiny_small_recall_050",
            "ap50",
            "fewer_factors",
        ],
    }


def physical_scene_macro_gate(
    uniform_rows: Sequence[Mapping[str, Any]],
    data_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Average scans within physical scenes before the tune24 decision."""

    def grouped(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
        output: dict[str, list[float]] = {}
        seen: set[str] = set()
        for row in rows:
            scene_id = str(row["scene_id"])
            if scene_id in seen:
                raise ValueError("scan rows must be unique")
            seen.add(scene_id)
            physical = str(row.get("physical_scene_id", scene_id.split("_")[0]))
            output.setdefault(physical, []).append(float(row["map_50_95"]))
        return output

    uniform = grouped(uniform_rows)
    data = grouped(data_rows)
    if set(uniform) != set(data):
        raise ValueError("uniform and data scans must cover identical physical scenes")
    deltas = {
        physical: float(np.mean(data[physical]) - np.mean(uniform[physical]))
        for physical in sorted(uniform)
    }
    macro = float(np.mean(list(deltas.values()))) if deltas else 0.0
    return {
        "schema": "saga-v10-physical-scene-macro-gate-v1",
        "passed": macro >= 0.002 - _EPSILON,
        "physical_scene_count": len(deltas),
        "macro_mean_map_delta": macro,
        "per_physical_scene_delta": deltas,
    }


def final48_gate(bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the one preregistered final significance decision."""

    delta = float(bootstrap["delta_map_50_95"])
    interval = bootstrap["paired_bootstrap_ci95"]
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise ValueError("paired bootstrap must provide a two-value CI")
    lower, upper = map(float, interval)
    checks = {
        "delta_map_at_least_0002": delta >= 0.002 - _EPSILON,
        "ci95_lower_above_zero": lower > 0.0,
    }
    return {
        "schema": "saga-v10-final48-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "delta_map_50_95": delta,
        "paired_bootstrap_ci95": [lower, upper],
    }
def write_v10b_identity_training_proposal(
    path: Path,
    *,
    failed_stage: str,
    diagnosis: Mapping[str, Any],
) -> Path:
    """Write the mandatory next-step proposal after a V10 structure stop."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_lines = "\n".join(
        f"- `{key}`: `{value}`" for key, value in sorted(diagnosis.items())
    ) or "- 暂无可解析诊断值。"
    text = f"""# SAGA V10B：跨视角 Identity Head 训练提案

## 触发原因

V10 在 `{failed_stage}` 未通过预注册结构门槛。该停止发生在类别先验 replay 之前，不能据此判定类别先验无效。

{diagnostic_lines}

## 训练目标

在冻结的 30k 3DGS 上新增独立跨视角 identity head。它只学习同一物理对象在不同视角中的一致身份，不读取类别 ID，不替代现有 affinity/semantic feature，也不重训 3DGS。

## 高置信伪轨迹监督

1. 从 V10 双向可见性匹配中选择 mutual-best 且两个方向 coverage 均不低于 0.80 的 fragment pair。
2. 仅保留三视图一致 cycle，或有至少两个独立共视邻居支持的 pair，形成高置信 pseudo-track。
3. pseudo-track 内跨视角 core Gaussian 构成正样本；同帧不同 fragment、共视帧中互斥 fragment以及 5 cm 内不同轨迹 Gaussian 构成 hard negative。
4. GT 只用于训练完成后的离线正控评价，绝不生成 pseudo-track 或样本标签。

## 模型与损失

- 在冻结 Gaussian 上训练 16 维 L2-normalized identity embedding；不输入 class ID。
- 损失为 supervised contrastive loss 加 hard-negative margin loss；同一 pseudo-track 为正，不同轨迹为负。
- affinity 与 semantic head 保持冻结，identity head 输出写入独立目录。
- 训练 seed 固定为 42；不搜索 embedding 维度、阈值或损失权重。

## 三场景正控

固定 `scene0645_00`、`scene0025_01`、`scene0474_01`：

- identity pair AUROC 相对现有 affinity 至少提高 0.05；
- 自动同类 IoU≥0.50 候选合计至少新增 2 个；
- 任一场景不得损失超过 1 个原有匹配；
- 候选数不得超过 V10-U 的 1.5 倍。

正控通过后再提交是否扩展到开发 8 场景的授权申请；不自动训练 24/48 场景。

## 资源预算

- 3 个场景，每场约 10k identity-head iterations；
- 单 GPU、单训练，预计 2–4 GPU 小时；
- 复用图像、相机、30k 3DGS、S-AM lifting 和伪轨迹，不下载数据或权重；
- 独立输出预计低于 10 GB，不覆盖任何历史资产。

## 所需批准

请批准或拒绝上述 V10B 三场景 identity-head 正控。未获批准前，流程保持停止，不会把 V10 的结构失败解释为类别先验失败。
"""
    target.write_text(text, encoding="utf-8")
    return target
