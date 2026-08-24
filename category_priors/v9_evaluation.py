from __future__ import annotations

"""Pure offline candidate evaluation and preregistered V9 stage gates.

Candidate and ground-truth supports must already be expressed in the same
point-index space.  This module has no filesystem, renderer, worker, training,
or runtime dependency; ground truth can therefore only enter after an
ObjectBank has been frozen.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_GATE_EPSILON = 1e-12
_TINY_SMALL = frozenset({"tiny", "small"})


def _support_ids(values: np.ndarray | Sequence[int], *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    try:
        ids = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must contain integers") from exc
    if not np.array_equal(raw, ids) or np.any(ids < 0):
        raise TypeError(f"{name} must contain non-negative integers")
    ids = np.unique(ids)
    ids.setflags(write=False)
    return ids


@dataclass(frozen=True)
class CandidateSupport:
    scene_id: str
    candidate_id: int
    support_ids: np.ndarray
    class_name: str
    score: float

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        class_name = str(self.class_name).strip()
        score = float(self.score)
        if not scene_id or not class_name:
            raise ValueError("candidate scene_id and class_name must be non-empty")
        if int(self.candidate_id) < 0:
            raise ValueError("candidate_id must be non-negative")
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("candidate score must be finite and in [0, 1]")
        support = _support_ids(self.support_ids, name="candidate support_ids")
        if not len(support):
            raise ValueError("candidate support must be non-empty")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "candidate_id", int(self.candidate_id))
        object.__setattr__(self, "support_ids", support)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class GroundTruthSupport:
    scene_id: str
    instance_id: int
    support_ids: np.ndarray
    class_name: str
    size_bin: str | None = None
    support_count: int | None = None
    official_valid: bool = True

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        class_name = str(self.class_name).strip()
        size_bin = None if self.size_bin is None else str(self.size_bin).strip().lower()
        if not scene_id or not class_name:
            raise ValueError("GT scene_id and class_name must be non-empty")
        if int(self.instance_id) < 0:
            raise ValueError("GT instance_id must be non-negative")
        support = _support_ids(self.support_ids, name="GT support_ids")
        if not len(support):
            raise ValueError("GT support must be non-empty")
        support_count = len(support) if self.support_count is None else int(
            self.support_count
        )
        if support_count < 0:
            raise ValueError("GT support_count must be non-negative")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "instance_id", int(self.instance_id))
        object.__setattr__(self, "support_ids", support)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "size_bin", size_bin)
        object.__setattr__(self, "support_count", support_count)
        object.__setattr__(self, "official_valid", bool(self.official_valid))

    @property
    def tiny_small(self) -> bool:
        return self.size_bin in _TINY_SMALL


@dataclass(frozen=True)
class SceneMethodMetrics:
    scene_id: str
    map_50_95: float
    tiny_small_match_050_count: int = 0
    tiny_small_gt_count: int = 0
    false_positive_count: int = 0
    true_positive_count: int = 0

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        value = float(self.map_50_95)
        if not scene_id or not np.isfinite(value):
            raise ValueError("scene_id must be non-empty and mAP finite")
        integer_fields = (
            "tiny_small_match_050_count",
            "tiny_small_gt_count",
            "false_positive_count",
            "true_positive_count",
        )
        for field in integer_fields:
            raw = getattr(self, field)
            value_int = int(raw)
            if value_int != raw or value_int < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, value_int)
        if self.tiny_small_match_050_count > self.tiny_small_gt_count:
            raise ValueError("tiny/small matches cannot exceed the GT denominator")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "map_50_95", value)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return float(intersection / union) if union else 0.0


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def score_iou_spearman(scores: Sequence[float], ious: Sequence[float]) -> float:
    """Compute Spearman correlation with average ties and safe constant input."""

    if len(scores) != len(ious):
        raise ValueError("scores and ious must have equal length")
    if len(scores) < 2:
        return 0.0
    score_array = np.asarray(scores, dtype=np.float64)
    iou_array = np.asarray(ious, dtype=np.float64)
    if np.any(~np.isfinite(score_array)) or np.any(~np.isfinite(iou_array)):
        raise ValueError("scores and ious must be finite")
    score_ranks = _average_ranks(score_array)
    iou_ranks = _average_ranks(iou_array)
    if np.ptp(score_ranks) == 0 or np.ptp(iou_ranks) == 0:
        return 0.0
    value = float(np.corrcoef(score_ranks, iou_ranks)[0, 1])
    return value if np.isfinite(value) else 0.0


def _metric_block(
    candidate_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    candidate_ious = np.asarray(
        [float(row[f"{prefix}_best_iou"]) for row in candidate_rows],
        dtype=np.float64,
    )
    gt_ious = np.asarray(
        [float(row[f"{prefix}_best_iou"]) for row in gt_rows], dtype=np.float64
    )
    result: dict[str, Any] = {}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        candidate_matches = int(np.count_nonzero(candidate_ious >= threshold))
        gt_matches = int(np.count_nonzero(gt_ious >= threshold))
        result[f"candidate_match_{suffix}_count"] = candidate_matches
        result[f"candidate_precision_{suffix}"] = (
            candidate_matches / len(candidate_rows) if candidate_rows else 0.0
        )
        result[f"gt_match_{suffix}_count"] = gt_matches
        result[f"recall_{suffix}"] = gt_matches / len(gt_rows) if gt_rows else 0.0
        result[f"match_{suffix}_scene_count"] = len(
            {
                str(row["scene_id"])
                for row in candidate_rows
                if float(row[f"{prefix}_best_iou"]) >= threshold
            }
        )
    return result


def _recall_block(
    gt_rows: Sequence[Mapping[str, Any]], *, prefix: str
) -> dict[str, Any]:
    ious = np.asarray(
        [float(row[f"{prefix}_best_iou"]) for row in gt_rows], dtype=np.float64
    )
    result: dict[str, Any] = {}
    for threshold, suffix in ((0.25, "025"), (0.50, "050")):
        matches = int(np.count_nonzero(ious >= threshold))
        result[f"gt_match_{suffix}_count"] = matches
        result[f"recall_{suffix}"] = matches / len(gt_rows) if gt_rows else 0.0
    return result


def evaluate_object_candidates(
    candidates: Sequence[CandidateSupport],
    ground_truth: Sequence[GroundTruthSupport],
) -> dict[str, Any]:
    """Evaluate a frozen candidate bank against offline ground truth supports."""

    ordered_candidates = tuple(
        sorted(candidates, key=lambda row: (row.scene_id, row.candidate_id))
    )
    ordered_gt = tuple(
        sorted(
            (row for row in ground_truth if row.official_valid),
            key=lambda row: (row.scene_id, row.instance_id),
        )
    )
    candidate_keys = [(row.scene_id, row.candidate_id) for row in ordered_candidates]
    gt_keys = [(row.scene_id, row.instance_id) for row in ordered_gt]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate IDs must be unique within each scene")
    if len(gt_keys) != len(set(gt_keys)):
        raise ValueError("GT instance IDs must be unique within each scene")

    gt_by_scene: dict[str, list[GroundTruthSupport]] = {}
    for row in ordered_gt:
        gt_by_scene.setdefault(row.scene_id, []).append(row)
    candidates_by_scene: dict[str, list[CandidateSupport]] = {}
    for row in ordered_candidates:
        candidates_by_scene.setdefault(row.scene_id, []).append(row)

    candidate_rows: list[dict[str, Any]] = []
    gt_best: dict[tuple[str, int], dict[str, Any]] = {
        (row.scene_id, row.instance_id): {
            "scene_id": row.scene_id,
            "gt_instance_id": row.instance_id,
            "gt_class": row.class_name,
            "size_bin": row.size_bin,
            "support_count": row.support_count,
            "geometric_best_iou": 0.0,
            "same_class_best_iou": 0.0,
            "geometric_best_candidate_id": None,
            "same_class_best_candidate_id": None,
        }
        for row in ordered_gt
    }
    for candidate in ordered_candidates:
        scene_gt = gt_by_scene.get(candidate.scene_id, [])
        geometric = [
            (_iou(candidate.support_ids, gt.support_ids), gt) for gt in scene_gt
        ]
        same_class = [
            (iou, gt)
            for iou, gt in geometric
            if candidate.class_name == gt.class_name
        ]
        geometric.sort(key=lambda item: (-item[0], item[1].instance_id))
        same_class.sort(key=lambda item: (-item[0], item[1].instance_id))
        geometric_iou, geometric_gt = (
            geometric[0] if geometric else (0.0, None)
        )
        same_iou, same_gt = same_class[0] if same_class else (0.0, None)
        candidate_rows.append(
            {
                "scene_id": candidate.scene_id,
                "candidate_id": candidate.candidate_id,
                "class": candidate.class_name,
                "score": candidate.score,
                "support_count": len(candidate.support_ids),
                "geometric_best_iou": float(geometric_iou),
                "geometric_best_gt_instance_id": (
                    geometric_gt.instance_id if geometric_gt is not None else None
                ),
                "geometric_best_gt_class": (
                    geometric_gt.class_name if geometric_gt is not None else None
                ),
                "same_class_best_iou": float(same_iou),
                "same_class_best_gt_instance_id": (
                    same_gt.instance_id if same_gt is not None else None
                ),
                "geometric_match_025": bool(geometric_iou >= 0.25),
                "geometric_match_050": bool(geometric_iou >= 0.50),
                "same_class_match_025": bool(same_iou >= 0.25),
                "same_class_match_050": bool(same_iou >= 0.50),
            }
        )
        for iou, gt in geometric:
            target = gt_best[(gt.scene_id, gt.instance_id)]
            current = float(target["geometric_best_iou"])
            current_id = target["geometric_best_candidate_id"]
            if iou > current or (
                iou == current
                and iou > 0
                and (current_id is None or candidate.candidate_id < int(current_id))
            ):
                target["geometric_best_iou"] = float(iou)
                target["geometric_best_candidate_id"] = candidate.candidate_id
            if candidate.class_name == gt.class_name:
                current = float(target["same_class_best_iou"])
                current_id = target["same_class_best_candidate_id"]
                if iou > current or (
                    iou == current
                    and iou > 0
                    and (
                        current_id is None
                        or candidate.candidate_id < int(current_id)
                    )
                ):
                    target["same_class_best_iou"] = float(iou)
                    target["same_class_best_candidate_id"] = candidate.candidate_id

    gt_rows = [gt_best[key] for key in sorted(gt_best)]
    geometric = _metric_block(candidate_rows, gt_rows, prefix="geometric")
    same_class = _metric_block(candidate_rows, gt_rows, prefix="same_class")
    tiny_gt_rows = [row for row in gt_rows if row["size_bin"] in _TINY_SMALL]
    tiny_small = {
        "official_valid_gt_count": len(tiny_gt_rows),
        "geometric": _recall_block(tiny_gt_rows, prefix="geometric"),
        "same_class": _recall_block(tiny_gt_rows, prefix="same_class"),
    }
    scores = [float(row["score"]) for row in candidate_rows]
    same_ious = [float(row["same_class_best_iou"]) for row in candidate_rows]

    scene_ids = sorted(set(candidates_by_scene) | set(gt_by_scene))
    per_scene: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_candidate_rows = [
            row for row in candidate_rows if row["scene_id"] == scene_id
        ]
        scene_gt_rows = [row for row in gt_rows if row["scene_id"] == scene_id]
        scene_tiny = [row for row in scene_gt_rows if row["size_bin"] in _TINY_SMALL]
        per_scene.append(
            {
                "scene_id": scene_id,
                "candidate_count": len(scene_candidate_rows),
                "official_valid_gt_count": len(scene_gt_rows),
                "geometric": _metric_block(
                    scene_candidate_rows, scene_gt_rows, prefix="geometric"
                ),
                "same_class": _metric_block(
                    scene_candidate_rows, scene_gt_rows, prefix="same_class"
                ),
                "geometric_tiny_small_recall_025": _recall_block(
                    scene_tiny, prefix="geometric"
                )["recall_025"],
                "same_class_tiny_small_recall_025": _recall_block(
                    scene_tiny, prefix="same_class"
                )["recall_025"],
            }
        )

    result = {
        "schema": "saga-v9-candidate-evaluation-v1",
        "scene_count": len(scene_ids),
        "candidate_count": len(candidate_rows),
        "official_valid_gt_count": len(gt_rows),
        "geometric": geometric,
        "same_class": same_class,
        "tiny_small": tiny_small,
        "score_iou_spearman": score_iou_spearman(scores, same_ious),
        "per_candidate": candidate_rows,
        "per_gt": gt_rows,
        "per_scene": per_scene,
    }
    # Stable aliases used directly by the V9 preregistered gates.
    result.update(
        {
            "geometric_match_050_count": geometric[
                "candidate_match_050_count"
            ],
            "geometric_match_050_scene_count": geometric[
                "match_050_scene_count"
            ],
            "same_class_match_050_count": same_class[
                "candidate_match_050_count"
            ],
            "same_class_match_050_scene_count": same_class[
                "match_050_scene_count"
            ],
            "same_class_candidate_precision_025": same_class[
                "candidate_precision_025"
            ],
            "geometric_tiny_small_recall_025": tiny_small["geometric"][
                "recall_025"
            ],
            "tiny_small_recall_025": tiny_small["same_class"]["recall_025"],
            "tiny_small_recall_050": tiny_small["same_class"]["recall_050"],
        }
    )
    return result


def stage2_oracle_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen two-scene geometric feasibility gate."""

    checks = {
        "geometric_match_050_at_least_6": (
            int(metrics["geometric_match_050_count"]) >= 6
        ),
        "geometric_tiny_small_recall_025_at_least_020": (
            float(metrics["geometric_tiny_small_recall_025"])
            >= 0.20 - _GATE_EPSILON
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def stage3_uniform_health_gate(
    bank: Mapping[str, Any],
    *,
    t1_b1: Mapping[str, Any],
    f10k_b0: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every frozen eight-scene uniform ObjectBank health threshold."""

    precision_gain = float(bank["gaussian_micro_precision"]) - float(
        t1_b1["gaussian_micro_precision"]
    )
    unsupported_reduction = float(t1_b1["unsupported_instance_fraction"]) - float(
        bank["unsupported_instance_fraction"]
    )
    instance_limit = 1.25 * int(f10k_b0["predicted_instance_count"])
    checks = {
        "geometric_match_050_at_least_16": (
            int(bank["geometric_match_050_count"]) >= 16
        ),
        "geometric_match_050_scenes_at_least_4": (
            int(bank["geometric_match_050_scene_count"]) >= 4
        ),
        "same_class_match_050_at_least_12": (
            int(bank["same_class_match_050_count"]) >= 12
        ),
        "same_class_match_050_scenes_at_least_4": (
            int(bank["same_class_match_050_scene_count"]) >= 4
        ),
        "same_class_precision_025_at_least_010": (
            float(bank["same_class_candidate_precision_025"])
            >= 0.10 - _GATE_EPSILON
        ),
        "tiny_small_recall_025_at_least_020": (
            float(bank["tiny_small_recall_025"]) >= 0.20 - _GATE_EPSILON
        ),
        "precision_or_unsupported_improved": (
            precision_gain >= 0.05 - _GATE_EPSILON
            or unsupported_reduction >= 0.10 - _GATE_EPSILON
        ),
        # ``gt_recall`` is the evaluator's compatibility alias for unique,
        # official-valid GT-instance macro coverage.  Missed GTs contribute
        # zero and duplicate predictions contribute only their best overlap.
        "unique_official_gt_instance_recall_drop_at_most_005": (
            float(bank["gt_recall"])
            >= float(t1_b1["gt_recall"]) - 0.05 - _GATE_EPSILON
        ),
        "n0_map_drop_at_most_0001": (
            float(bank["map_50_95"])
            >= float(f10k_b0["map_50_95"]) - 0.001 - _GATE_EPSILON
        ),
        "n0_ap50_drop_at_most_0002": (
            float(bank["ap50"])
            >= float(f10k_b0["ap50"]) - 0.002 - _GATE_EPSILON
        ),
        "instance_count_at_most_1_25x": (
            int(bank["predicted_instance_count"]) <= instance_limit
        ),
        "score_iou_spearman_at_least_020": (
            float(bank["score_iou_spearman"]) >= 0.20 - _GATE_EPSILON
        ),
        "orphan_gaussian_count_zero": int(bank["orphan_gaussian_count"]) == 0,
        "negative_metadata_count_zero": int(bank["negative_metadata_count"]) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "precision_gain": precision_gain,
        "unsupported_reduction": unsupported_reduction,
        "instance_limit": instance_limit,
        "gt_recall_semantics": "unique_official_gt_instance_macro_coverage",
    }


def _scene_metric(value: SceneMethodMetrics | Mapping[str, Any]) -> SceneMethodMetrics:
    if isinstance(value, SceneMethodMetrics):
        return value
    return SceneMethodMetrics(
        scene_id=str(value["scene_id"]),
        map_50_95=float(value["map_50_95"]),
        tiny_small_match_050_count=int(
            value.get("tiny_small_match_050_count", 0)
        ),
        tiny_small_gt_count=int(value.get("tiny_small_gt_count", 0)),
        false_positive_count=int(value.get("false_positive_count", 0)),
        true_positive_count=int(value.get("true_positive_count", 0)),
    )


def _fp_tp_ratio(rows: Sequence[SceneMethodMetrics]) -> float:
    false_positives = sum(row.false_positive_count for row in rows)
    true_positives = sum(row.true_positive_count for row in rows)
    if true_positives == 0:
        return 0.0 if false_positives == 0 else float("inf")
    return false_positives / true_positives


def stage4_prior_gate(
    uniform_scene_metrics: Sequence[SceneMethodMetrics | Mapping[str, Any]],
    data_scene_metrics: Sequence[SceneMethodMetrics | Mapping[str, Any]],
    *,
    candidate_score_deltas: Sequence[float] = (),
    accepted_or_ownership_changed: bool = False,
) -> dict[str, Any]:
    """Apply the V9 mechanical-effect and eight-scene prior-benefit gate."""

    uniform_rows = tuple(_scene_metric(row) for row in uniform_scene_metrics)
    data_rows = tuple(_scene_metric(row) for row in data_scene_metrics)
    uniform = {row.scene_id: row for row in uniform_rows}
    data = {row.scene_id: row for row in data_rows}
    if len(uniform) != len(uniform_rows) or len(data) != len(data_rows):
        raise ValueError("scene IDs must be unique within each condition")
    if not uniform or set(uniform) != set(data):
        raise ValueError(
            "uniform and data metrics must cover the same non-empty scenes"
        )
    ordered_scenes = sorted(uniform)
    map_deltas = np.asarray(
        [
            data[scene].map_50_95 - uniform[scene].map_50_95
            for scene in ordered_scenes
        ],
        dtype=np.float64,
    )
    mean_map_delta = float(np.mean(map_deltas))
    positive_scenes = int(np.count_nonzero(map_deltas > _GATE_EPSILON))
    negative_scenes = int(np.count_nonzero(map_deltas < -_GATE_EPSILON))
    if any(
        uniform[scene].tiny_small_gt_count != data[scene].tiny_small_gt_count
        for scene in ordered_scenes
    ):
        raise ValueError("uniform and data tiny/small GT denominators differ")
    uniform_tiny_gt = sum(row.tiny_small_gt_count for row in uniform.values())
    data_tiny_gt = sum(row.tiny_small_gt_count for row in data.values())
    uniform_tiny_recall = (
        sum(row.tiny_small_match_050_count for row in uniform.values())
        / uniform_tiny_gt
        if uniform_tiny_gt
        else 0.0
    )
    data_tiny_recall = (
        sum(row.tiny_small_match_050_count for row in data.values()) / data_tiny_gt
        if data_tiny_gt
        else 0.0
    )
    tiny_recall_delta = data_tiny_recall - uniform_tiny_recall
    uniform_fp_tp = _fp_tp_ratio(tuple(uniform.values()))
    data_fp_tp = _fp_tp_ratio(tuple(data.values()))
    if np.isinf(uniform_fp_tp):
        fp_tp_relative_change = 0.0 if np.isinf(data_fp_tp) else -1.0
    elif uniform_fp_tp == 0.0:
        fp_tp_relative_change = 0.0 if data_fp_tp == 0.0 else float("inf")
    else:
        fp_tp_relative_change = data_fp_tp / uniform_fp_tp - 1.0
    deltas = np.abs(np.asarray(candidate_score_deltas, dtype=np.float64))
    if np.any(~np.isfinite(deltas)):
        raise ValueError("candidate_score_deltas must be finite")
    intervention_fraction = (
        float(np.mean(deltas >= 0.01 - _GATE_EPSILON)) if len(deltas) else 0.0
    )
    mechanically_effective = (
        intervention_fraction >= 0.10 - _GATE_EPSILON
        or bool(accepted_or_ownership_changed)
    )
    checks = {
        "prior_mechanically_effective": mechanically_effective,
        "registered_benefit_threshold": (
            mean_map_delta >= 0.002 - _GATE_EPSILON
            or (
                tiny_recall_delta >= 0.01 - _GATE_EPSILON
                and mean_map_delta >= -0.0005 - _GATE_EPSILON
            )
        ),
        "positive_scenes_more_than_negative": positive_scenes > negative_scenes,
        "fp_tp_degradation_at_most_020": (
            fp_tp_relative_change <= 0.20 + _GATE_EPSILON
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scene_count": len(ordered_scenes),
        "mean_map_delta": mean_map_delta,
        "positive_scene_count": positive_scenes,
        "negative_scene_count": negative_scenes,
        "tiny_small_recall_050_delta": tiny_recall_delta,
        "uniform_fp_tp_ratio": uniform_fp_tp,
        "data_fp_tp_ratio": data_fp_tp,
        "fp_tp_relative_change": fp_tp_relative_change,
        "intervention_fraction": intervention_fraction,
        "accepted_or_ownership_changed": bool(accepted_or_ownership_changed),
    }
