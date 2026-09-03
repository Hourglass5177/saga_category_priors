from __future__ import annotations

"""Strict, GT-only evaluation for the full-instance size-prior experiment.

The production snapshot/scoring/replay path must not import this module.  It is
the only new module in that experiment which accepts ground truth.  Candidate
capacity uses an overlap-optimal one-to-one matching; candidate AP instead
uses the registered score order, so duplicate predictions for one GT object
remain false positives.
"""

import math
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    GroundTruthScene,
    PredictedInstance,
    apply_transform,
    evaluate_instances,
    load_ground_truth_npz,
    load_ply_xyz,
)


OFFICIAL_9_OVERLAPS = tuple(
    np.arange(0.50, 0.95, 0.05).round(2).tolist()
)
HISTORICAL_10_OVERLAPS = tuple(
    np.arange(0.50, 0.96, 0.05).round(2).tolist()
)
ORACLE_CLASS_MIN_GEOMETRIC_IOU = 0.25


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _integer_ids(value: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype == np.bool_:
        result = np.flatnonzero(raw).astype(np.int64, copy=False)
    else:
        try:
            result = raw.astype(np.int64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must contain integers") from exc
        if not np.array_equal(raw, result):
            raise TypeError(f"{name} must contain integers")
        result = np.unique(result)
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative values")
    result = np.asarray(result, dtype=np.int64)
    result.setflags(write=False)
    return result


def _finite_unit(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class CandidatePrediction:
    scene_id: str
    candidate_id: Any
    class_id: int
    score: float
    member_indices: np.ndarray

    def __post_init__(self) -> None:
        if not str(self.scene_id):
            raise ValueError("scene_id cannot be empty")
        if isinstance(self.class_id, (bool, np.bool_)):
            raise TypeError("class_id must be an integer")
        class_id = int(self.class_id)
        if class_id != self.class_id or class_id < 0:
            raise ValueError("class_id must be a non-negative integer")
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(
            self, "score", _finite_unit(self.score, name="candidate score")
        )
        members = _integer_ids(self.member_indices, name="member_indices")
        if len(members) == 0:
            raise ValueError("candidate member_indices cannot be empty")
        object.__setattr__(self, "member_indices", members)


@dataclass(frozen=True)
class GroundTruthInstance:
    scene_id: str
    instance_id: int
    class_id: int
    point_indices: np.ndarray
    is_tiny_small: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.instance_id, (bool, np.bool_)):
            raise TypeError("GT instance_id must be an integer")
        if isinstance(self.class_id, (bool, np.bool_)):
            raise TypeError("GT class_id must be an integer")
        instance_id = int(self.instance_id)
        class_id = int(self.class_id)
        if instance_id != self.instance_id or instance_id < 0:
            raise ValueError("GT instance_id must be a non-negative integer")
        if class_id != self.class_id or class_id < 0:
            raise ValueError("GT class_id must be a non-negative integer")
        points = _integer_ids(self.point_indices, name="GT point_indices")
        if len(points) == 0:
            raise ValueError("GT instance cannot be empty")
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "point_indices", points)


@dataclass(frozen=True)
class CandidateMatch:
    candidate_index: int
    gt_index: int
    iou: float


def candidate_predictions_from_rows(
    rows: Sequence[CandidatePrediction | Mapping[str, Any]],
    *,
    score_key: str,
    class_names: Sequence[str] | None = None,
    eligible_only: bool = False,
) -> tuple[CandidatePrediction, ...]:
    """Adapt snapshot/scoring rows without coupling to the runtime module."""

    class_to_id = (
        None
        if class_names is None
        else {str(name).strip().lower(): index for index, name in enumerate(class_names)}
    )
    result: list[CandidatePrediction] = []
    seen: set[tuple[str, tuple[str, str]]] = set()
    for row in rows:
        if isinstance(row, CandidatePrediction):
            candidate = row
        else:
            if eligible_only and not bool(row.get("eligible", True)):
                continue
            scene_id = str(row["scene_id"])
            candidate_id = row.get(
                "candidate_id", row.get("raw_instance_id", row.get("object_id"))
            )
            if candidate_id is None:
                raise KeyError("candidate row is missing its stable ID")
            class_value: Any = row.get(
                "class_id", row.get("predicted_class_index")
            )
            if class_value is None:
                name = row.get(
                    "predicted_class_name",
                    row.get("predicted_class", row.get("class_name")),
                )
                if class_to_id is None or name is None:
                    raise KeyError("candidate row is missing its predicted class")
                normalized = str(name).strip().lower()
                if normalized not in class_to_id:
                    raise ValueError(f"unknown predicted class: {name}")
                class_value = class_to_id[normalized]
            members = row.get(
                "member_indices",
                row.get("gaussian_ids", row.get("full_ids")),
            )
            if members is None:
                raise KeyError("candidate row is missing member_indices")
            if score_key not in row:
                raise KeyError(f"candidate row is missing score {score_key!r}")
            candidate = CandidatePrediction(
                scene_id=scene_id,
                candidate_id=candidate_id,
                class_id=int(class_value),
                score=row[score_key],
                member_indices=np.asarray(members),
            )
        identity = (candidate.scene_id, _stable_key(candidate.candidate_id))
        if identity in seen:
            raise ValueError(f"duplicate candidate identity: {identity}")
        seen.add(identity)
        result.append(candidate)
    return tuple(
        sorted(result, key=lambda row: (row.scene_id, _stable_key(row.candidate_id)))
    )


def ground_truth_instances(
    scenes: Sequence[GroundTruthScene],
    *,
    min_region_size: int = 100,
    tiny_small_instance_ids: Mapping[
        str, Collection[int | tuple[int, int]]
    ]
    | None = None,
) -> tuple[GroundTruthInstance, ...]:
    if isinstance(min_region_size, bool) or int(min_region_size) <= 0:
        raise ValueError("min_region_size must be a positive integer")
    seen_scenes: set[str] = set()
    result: list[GroundTruthInstance] = []
    for scene in scenes:
        if scene.scene_id in seen_scenes:
            raise ValueError(f"duplicate GT scene_id: {scene.scene_id}")
        seen_scenes.add(scene.scene_id)
        semantic = np.asarray(scene.semantic, dtype=np.int64)
        instance = np.asarray(scene.instance, dtype=np.int64)
        if semantic.ndim != 1 or semantic.shape != instance.shape:
            raise ValueError(f"{scene.scene_id}: invalid GT arrays")
        tiny_spec = set(
            ()
            if tiny_small_instance_ids is None
            else tiny_small_instance_ids.get(scene.scene_id, ())
        )
        for class_id in sorted(int(value) for value in np.unique(semantic) if value >= 0):
            class_mask = semantic == class_id
            for instance_id in sorted(
                int(value) for value in np.unique(instance[class_mask]) if value >= 0
            ):
                mask = class_mask & (instance == instance_id)
                point_indices = np.flatnonzero(mask)
                if len(point_indices) < int(min_region_size):
                    continue
                is_tiny = (
                    instance_id in tiny_spec or (class_id, instance_id) in tiny_spec
                )
                result.append(
                    GroundTruthInstance(
                        scene_id=scene.scene_id,
                        instance_id=instance_id,
                        class_id=class_id,
                        point_indices=point_indices,
                        is_tiny_small=is_tiny,
                    )
                )
    return tuple(result)


def project_candidate_to_gt_points(
    candidate: CandidatePrediction,
    gt_point_to_gaussian: Sequence[int] | np.ndarray,
) -> np.ndarray:
    mapping = np.asarray(gt_point_to_gaussian)
    if mapping.ndim != 1:
        raise ValueError("gt_point_to_gaussian must be one-dimensional")
    try:
        mapping = mapping.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("gt_point_to_gaussian must contain integers") from exc
    if np.any(mapping < -1):
        raise ValueError("gt_point_to_gaussian may only use -1 as its sentinel")
    result = np.isin(mapping, candidate.member_indices, assume_unique=False)
    result = np.asarray(result, dtype=bool)
    result.setflags(write=False)
    return result


def pairwise_iou(
    candidate_masks: Sequence[np.ndarray],
    gt_instances: Sequence[GroundTruthInstance],
) -> np.ndarray:
    if not candidate_masks:
        point_count = 0 if not gt_instances else int(gt_instances[0].point_indices.max() + 1)
        return np.zeros((0, len(gt_instances)), dtype=np.float64)
    point_count = len(np.asarray(candidate_masks[0]))
    masks: list[np.ndarray] = []
    for mask in candidate_masks:
        normalized = np.asarray(mask, dtype=bool)
        if normalized.shape != (point_count,):
            raise ValueError("candidate masks must be aligned vectors")
        masks.append(normalized)
    result = np.zeros((len(masks), len(gt_instances)), dtype=np.float64)
    for gt_index, gt in enumerate(gt_instances):
        if np.any(gt.point_indices >= point_count):
            raise ValueError("GT point index exceeds candidate mask domain")
        gt_mask = np.zeros(point_count, dtype=bool)
        gt_mask[gt.point_indices] = True
        gt_count = len(gt.point_indices)
        for candidate_index, mask in enumerate(masks):
            intersection = int(np.count_nonzero(mask & gt_mask))
            union = int(np.count_nonzero(mask)) + gt_count - intersection
            result[candidate_index, gt_index] = (
                intersection / union if union else 0.0
            )
    return result


def _hungarian_maximum(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free rectangular Hungarian maximization.

    Rows are made the smaller side.  Stable input ordering and strict updates
    provide deterministic tie handling without changing the numeric objective.
    """

    value = np.asarray(weights, dtype=np.float64)
    if value.ndim != 2 or np.any(~np.isfinite(value)):
        raise ValueError("assignment weights must be a finite matrix")
    original_rows, original_columns = value.shape
    if not original_rows or not original_columns:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    transposed = original_rows > original_columns
    if transposed:
        value = value.T
    row_count, column_count = value.shape
    cost = -value
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    p = np.zeros(column_count + 1, dtype=np.int64)
    way = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        p[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column0 = 0
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = math.inf
            column1 = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = float(minimum[column])
                    column1 = column
            if not math.isfinite(delta):
                raise AssertionError("assignment unexpectedly has no augmenting column")
            for column in range(column_count + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(int(p[column]) - 1, column - 1) for column in range(1, column_count + 1) if p[column]]
    row_indices = np.asarray([row for row, _ in pairs], dtype=np.int64)
    column_indices = np.asarray([column for _, column in pairs], dtype=np.int64)
    if transposed:
        return column_indices, row_indices
    return row_indices, column_indices


def maximum_cardinality_iou_matching(
    iou_matrix: np.ndarray,
    threshold: float,
    *,
    candidate_classes: Sequence[int] | np.ndarray | None = None,
    gt_classes: Sequence[int] | np.ndarray | None = None,
) -> tuple[CandidateMatch, ...]:
    """First maximize strict-threshold matches, then their total IoU."""

    iou = np.asarray(iou_matrix, dtype=np.float64)
    if iou.ndim != 2 or np.any(~np.isfinite(iou)):
        raise ValueError("iou_matrix must be a finite matrix")
    if np.any((iou < 0.0) | (iou > 1.0)):
        raise ValueError("iou_matrix values must be in [0, 1]")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    valid = iou > threshold_value
    if (candidate_classes is None) != (gt_classes is None):
        raise ValueError("candidate_classes and gt_classes must be supplied together")
    if candidate_classes is not None:
        left = np.asarray(candidate_classes, dtype=np.int64)
        right = np.asarray(gt_classes, dtype=np.int64)
        if left.shape != (iou.shape[0],) or right.shape != (iou.shape[1],):
            raise ValueError("class vectors do not match the IoU matrix")
        valid &= left[:, None] == right[None, :]
    cardinality_bonus = min(iou.shape) + 1.0
    weights = np.where(valid, cardinality_bonus + iou, 0.0)
    rows, columns = _hungarian_maximum(weights)
    matches = [
        CandidateMatch(int(row), int(column), float(iou[row, column]))
        for row, column in zip(rows, columns)
        if valid[row, column]
    ]
    return tuple(sorted(matches, key=lambda item: (item.candidate_index, item.gt_index)))


def _precision_envelope_ap(true_positive: np.ndarray, gt_count: int) -> float:
    if gt_count <= 0:
        raise ValueError("gt_count must be positive")
    if len(true_positive) == 0:
        return 0.0
    tp = np.asarray(true_positive, dtype=np.float64)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / float(gt_count)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1.0)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def ranked_candidate_matches(
    iou_matrix: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    candidate_classes: Sequence[int] | np.ndarray,
    gt_classes: Sequence[int] | np.ndarray,
    threshold: float,
    *,
    candidate_ids: Sequence[Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Greedily lock same-class GT in score order; duplicates remain FP."""

    iou = np.asarray(iou_matrix, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    left_class = np.asarray(candidate_classes, dtype=np.int64)
    right_class = np.asarray(gt_classes, dtype=np.int64)
    if iou.ndim != 2 or score.shape != (iou.shape[0],):
        raise ValueError("scores do not match the IoU matrix")
    if left_class.shape != (iou.shape[0],) or right_class.shape != (iou.shape[1],):
        raise ValueError("class vectors do not match the IoU matrix")
    if np.any(~np.isfinite(iou)) or np.any(~np.isfinite(score)):
        raise ValueError("IoUs and scores must be finite")
    if np.any((iou < 0.0) | (iou > 1.0)):
        raise ValueError("IoUs must be in [0, 1]")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("scores must be in [0, 1]")
    threshold_value = float(threshold)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    ids = list(range(iou.shape[0])) if candidate_ids is None else list(candidate_ids)
    if len(ids) != iou.shape[0]:
        raise ValueError("candidate_ids do not match the IoU matrix")
    stable_ids = [_stable_key(value) for value in ids]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("candidate_ids must be unique")
    order = sorted(range(len(ids)), key=lambda index: (-score[index], _stable_key(ids[index])))
    matched_gt: set[int] = set()
    rows: list[dict[str, Any]] = []
    for candidate_index in order:
        compatible = [
            gt_index
            for gt_index in range(iou.shape[1])
            if gt_index not in matched_gt
            and left_class[candidate_index] == right_class[gt_index]
            and iou[candidate_index, gt_index] > threshold_value
        ]
        best_gt = (
            min(compatible, key=lambda gt_index: (-iou[candidate_index, gt_index], gt_index))
            if compatible
            else None
        )
        if best_gt is not None:
            matched_gt.add(best_gt)
        rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": ids[candidate_index],
                "score": float(score[candidate_index]),
                "true_positive": best_gt is not None,
                "matched_gt_index": best_gt,
                "matched_iou": (
                    None if best_gt is None else float(iou[candidate_index, best_gt])
                ),
            }
        )
    return tuple(rows)


def candidate_average_precision(
    iou_matrix: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    candidate_classes: Sequence[int] | np.ndarray,
    gt_classes: Sequence[int] | np.ndarray,
    threshold: float,
    *,
    candidate_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return macro AP across GT-supported classes in one physical scene."""

    iou = np.asarray(iou_matrix, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    left_class = np.asarray(candidate_classes, dtype=np.int64)
    right_class = np.asarray(gt_classes, dtype=np.int64)
    ids = list(range(iou.shape[0])) if candidate_ids is None else list(candidate_ids)
    if iou.ndim != 2 or np.any(~np.isfinite(iou)):
        raise ValueError("iou_matrix must be a finite matrix")
    if np.any((iou < 0.0) | (iou > 1.0)):
        raise ValueError("iou_matrix values must be in [0, 1]")
    if np.any(~np.isfinite(score)) or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("scores must be finite and in [0, 1]")
    if score.shape != (iou.shape[0],) or left_class.shape != (iou.shape[0],):
        raise ValueError("candidate vectors do not match the IoU matrix")
    if right_class.shape != (iou.shape[1],) or len(ids) != iou.shape[0]:
        raise ValueError("GT/classes/IDs do not match the IoU matrix")
    stable_ids = [_stable_key(value) for value in ids]
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("candidate_ids must be unique")
    per_class: dict[int, float] = {}
    all_rows: list[dict[str, Any]] = []
    for class_id in sorted(int(value) for value in np.unique(right_class)):
        candidate_index = np.flatnonzero(left_class == class_id)
        gt_index = np.flatnonzero(right_class == class_id)
        local_iou = iou[np.ix_(candidate_index, gt_index)]
        ranked = ranked_candidate_matches(
            local_iou,
            score[candidate_index],
            left_class[candidate_index],
            right_class[gt_index],
            threshold,
            candidate_ids=[ids[index] for index in candidate_index],
        )
        flags = np.asarray([row["true_positive"] for row in ranked], dtype=bool)
        per_class[class_id] = _precision_envelope_ap(flags, len(gt_index))
        for row in ranked:
            global_candidate = int(candidate_index[int(row["candidate_index"])])
            local_gt = row["matched_gt_index"]
            all_rows.append(
                {
                    **row,
                    "candidate_index": global_candidate,
                    "class_id": class_id,
                    "matched_gt_index": (
                        None if local_gt is None else int(gt_index[int(local_gt)])
                    ),
                    "ap_class_has_gt": True,
                }
            )
    gt_supported_classes = set(int(value) for value in np.unique(right_class))
    for candidate_index in range(iou.shape[0]):
        class_id = int(left_class[candidate_index])
        if class_id in gt_supported_classes:
            continue
        all_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_id": ids[candidate_index],
                "score": float(score[candidate_index]),
                "true_positive": False,
                "matched_gt_index": None,
                "matched_iou": None,
                "class_id": class_id,
                "ap_class_has_gt": False,
            }
        )
    return {
        "ap": float(np.mean(list(per_class.values()))) if per_class else None,
        "per_class": per_class,
        "ranked_rows": tuple(
            sorted(all_rows, key=lambda row: (-row["score"], _stable_key(row["candidate_id"])))
        ),
        "true_positive_count": sum(bool(row["true_positive"]) for row in all_rows),
        "false_positive_count": sum(not bool(row["true_positive"]) for row in all_rows),
        "gt_count": len(right_class),
    }


def _candidate_average_precision_for_gt_subset(
    iou_matrix: np.ndarray,
    scores: Sequence[float] | np.ndarray,
    candidate_classes: Sequence[int] | np.ndarray,
    gt_classes: Sequence[int] | np.ndarray,
    target_gt_indices: Sequence[int] | np.ndarray,
    threshold: float,
    *,
    candidate_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an object-size subset without penalizing valid other sizes.

    A prediction which cannot match a target GT object, but *can* strictly
    match a same-class non-target GT object, is a correct prediction outside
    the requested size stratum.  It is ignored rather than counted as a false
    positive in the subset AP.  Predictions which can match a target remain
    eligible for the target AP, and genuinely unmatched predictions remain
    false positives.
    """

    iou = np.asarray(iou_matrix, dtype=np.float64)
    score = np.asarray(scores, dtype=np.float64)
    left_class = np.asarray(candidate_classes, dtype=np.int64)
    right_class = np.asarray(gt_classes, dtype=np.int64)
    target = np.asarray(target_gt_indices, dtype=np.int64)
    ids = list(range(iou.shape[0])) if candidate_ids is None else list(candidate_ids)
    if iou.ndim != 2 or iou.shape != (len(score), len(right_class)):
        raise ValueError("candidate/GT arrays do not match the IoU matrix")
    if left_class.shape != (iou.shape[0],) or len(ids) != iou.shape[0]:
        raise ValueError("candidate classes/IDs do not match the IoU matrix")
    if target.ndim != 1 or len(np.unique(target)) != len(target):
        raise ValueError("target_gt_indices must be a unique one-dimensional vector")
    if np.any(target < 0) or np.any(target >= iou.shape[1]):
        raise ValueError("target_gt_indices contain an out-of-range index")
    threshold_value = float(threshold)
    target_mask = np.zeros(iou.shape[1], dtype=bool)
    target_mask[target] = True
    non_target = np.flatnonzero(~target_mask)

    def can_match(gt_indices: np.ndarray) -> np.ndarray:
        if len(gt_indices) == 0:
            return np.zeros(iou.shape[0], dtype=bool)
        compatible = left_class[:, None] == right_class[gt_indices][None, :]
        return np.any((iou[:, gt_indices] > threshold_value) & compatible, axis=1)

    target_match = can_match(target)
    non_target_match = can_match(non_target)
    ignored = (~target_match) & non_target_match
    keep = np.flatnonzero(~ignored)
    result = candidate_average_precision(
        iou[np.ix_(keep, target)],
        score[keep],
        left_class[keep],
        right_class[target],
        threshold_value,
        candidate_ids=[ids[int(index)] for index in keep],
    )
    result["ignored_non_target_match_count"] = int(np.count_nonzero(ignored))
    result["ignored_non_target_candidate_ids"] = tuple(
        ids[int(index)] for index in np.flatnonzero(ignored)
    )
    return result


def _scene_view_metrics(
    candidates: Sequence[CandidatePrediction],
    candidate_masks: Sequence[np.ndarray],
    gt: Sequence[GroundTruthInstance],
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    iou = pairwise_iou(candidate_masks, gt)
    candidate_classes = np.asarray([row.class_id for row in candidates], dtype=np.int64)
    gt_classes = np.asarray([row.class_id for row in gt], dtype=np.int64)
    scores = np.asarray([row.score for row in candidates], dtype=np.float64)
    ids = [row.candidate_id for row in candidates]
    result: dict[str, Any] = {
        "candidate_count": len(candidates),
        "gt_count": len(gt),
        "thresholds": {},
    }
    for threshold in thresholds:
        key = f"{int(round(float(threshold) * 100)):03d}"
        matches = maximum_cardinality_iou_matching(
            iou,
            threshold,
            candidate_classes=candidate_classes,
            gt_classes=gt_classes,
        )
        ap = candidate_average_precision(
            iou,
            scores,
            candidate_classes,
            gt_classes,
            threshold,
            candidate_ids=ids,
        )
        tiny_gt_index = np.asarray(
            [index for index, row in enumerate(gt) if row.is_tiny_small],
            dtype=np.int64,
        )
        tiny_matches = maximum_cardinality_iou_matching(
            iou[:, tiny_gt_index],
            threshold,
            candidate_classes=candidate_classes,
            gt_classes=gt_classes[tiny_gt_index],
        )
        tiny_ap = (
            _candidate_average_precision_for_gt_subset(
                iou,
                scores,
                candidate_classes,
                gt_classes,
                tiny_gt_index,
                threshold,
                candidate_ids=ids,
            )
            if len(tiny_gt_index)
            else None
        )
        matched_classes = sorted({gt[match.gt_index].class_id for match in matches})
        result["thresholds"][key] = {
            "threshold": float(threshold),
            "match_count": len(matches),
            "precision": len(matches) / len(candidates) if candidates else 0.0,
            "recall": len(matches) / len(gt) if gt else 0.0,
            "matched_class_ids": matched_classes,
            "total_matched_iou": float(sum(match.iou for match in matches)),
            "candidate_ap": ap["ap"],
            "ap_true_positive_count": ap["true_positive_count"],
            "ap_false_positive_count": ap["false_positive_count"],
            "tiny_small_gt_count": len(tiny_gt_index),
            "tiny_small_match_count": len(tiny_matches),
            "tiny_small_recall": (
                len(tiny_matches) / len(tiny_gt_index) if len(tiny_gt_index) else None
            ),
            "tiny_small_candidate_ap": None if tiny_ap is None else tiny_ap["ap"],
            "tiny_small_ap_ignored_non_target_match_count": (
                0
                if tiny_ap is None
                else int(tiny_ap["ignored_non_target_match_count"])
            ),
            "matches": [
                {
                    "candidate_id": candidates[match.candidate_index].candidate_id,
                    "gt_instance_id": gt[match.gt_index].instance_id,
                    "gt_class_id": gt[match.gt_index].class_id,
                    "iou": match.iou,
                }
                for match in matches
            ],
        }
    return result


def evaluate_candidate_rankings(
    candidates: Sequence[CandidatePrediction | Mapping[str, Any]],
    ground_truth: Sequence[GroundTruthScene],
    gt_point_to_gaussian_by_scene: Mapping[str, Sequence[int] | np.ndarray],
    *,
    score_key: str = "score",
    class_names: Sequence[str] | None = None,
    eligible_only: bool = False,
    min_region_size: int = 100,
    tiny_small_instance_ids: Mapping[
        str, Collection[int | tuple[int, int]]
    ]
    | None = None,
    thresholds: Sequence[float] = (0.25, 0.50),
) -> dict[str, Any]:
    """Evaluate all and official-100 candidate views with scene-equal AP."""

    normalized = candidate_predictions_from_rows(
        candidates,
        score_key=score_key,
        class_names=class_names,
        eligible_only=eligible_only,
    )
    gt_instances = ground_truth_instances(
        ground_truth,
        min_region_size=min_region_size,
        tiny_small_instance_ids=tiny_small_instance_ids,
    )
    gt_scene_by_id = {scene.scene_id: scene for scene in ground_truth}
    if len(gt_scene_by_id) != len(ground_truth):
        raise ValueError("ground-truth scene IDs must be unique")
    if set(gt_scene_by_id) != set(gt_point_to_gaussian_by_scene):
        raise ValueError("GT scenes and point-to-Gaussian mappings must have identical keys")
    candidate_by_scene: dict[str, list[CandidatePrediction]] = defaultdict(list)
    mask_by_scene: dict[str, list[np.ndarray]] = defaultdict(list)
    candidate_rows: list[dict[str, Any]] = []
    for candidate in normalized:
        if candidate.scene_id not in gt_scene_by_id:
            raise ValueError(f"candidate references unknown scene {candidate.scene_id}")
        mapping = np.asarray(gt_point_to_gaussian_by_scene[candidate.scene_id])
        if mapping.shape != np.asarray(gt_scene_by_id[candidate.scene_id].semantic).shape:
            raise ValueError(f"{candidate.scene_id}: mapping does not match GT point domain")
        mask = project_candidate_to_gt_points(candidate, mapping)
        candidate_by_scene[candidate.scene_id].append(candidate)
        mask_by_scene[candidate.scene_id].append(mask)
        candidate_rows.append(
            {
                "scene_id": candidate.scene_id,
                "candidate_id": candidate.candidate_id,
                "class_id": candidate.class_id,
                "score": candidate.score,
                "gaussian_count": len(candidate.member_indices),
                "projected_gt_point_count": int(np.count_nonzero(mask)),
                "official_100_candidate": int(np.count_nonzero(mask)) >= min_region_size,
            }
        )
    gt_by_scene: dict[str, list[GroundTruthInstance]] = defaultdict(list)
    for row in gt_instances:
        gt_by_scene[row.scene_id].append(row)

    views: dict[str, Any] = {}
    for view_name, require_official_candidate in (("all", False), ("official_100", True)):
        per_scene: dict[str, Any] = {}
        for scene_id in sorted(gt_scene_by_id):
            rows = candidate_by_scene.get(scene_id, [])
            masks = mask_by_scene.get(scene_id, [])
            if require_official_candidate:
                keep = [
                    index
                    for index, mask in enumerate(masks)
                    if int(np.count_nonzero(mask)) >= min_region_size
                ]
                rows = [rows[index] for index in keep]
                masks = [masks[index] for index in keep]
            per_scene[scene_id] = _scene_view_metrics(
                rows,
                masks,
                gt_by_scene.get(scene_id, []),
                thresholds=thresholds,
            )
        aggregate: dict[str, Any] = {
            "scene_count": len(per_scene),
            "candidate_count": sum(row["candidate_count"] for row in per_scene.values()),
            "gt_count": sum(row["gt_count"] for row in per_scene.values()),
            "thresholds": {},
        }
        for threshold in thresholds:
            key = f"{int(round(float(threshold) * 100)):03d}"
            scene_rows = [row["thresholds"][key] for row in per_scene.values()]
            match_count = sum(int(row["match_count"]) for row in scene_rows)
            candidate_count = aggregate["candidate_count"]
            gt_count = aggregate["gt_count"]
            ap_values = [float(row["candidate_ap"]) for row in scene_rows if row["candidate_ap"] is not None]
            tiny_gt = sum(int(row["tiny_small_gt_count"]) for row in scene_rows)
            tiny_match = sum(int(row["tiny_small_match_count"]) for row in scene_rows)
            tiny_ap_values = [
                float(row["tiny_small_candidate_ap"])
                for row in scene_rows
                if row["tiny_small_candidate_ap"] is not None
            ]
            aggregate["thresholds"][key] = {
                "threshold": float(threshold),
                "match_count": match_count,
                "matched_scene_count": sum(int(row["match_count"] > 0) for row in scene_rows),
                "matched_class_count": len(
                    {class_id for row in scene_rows for class_id in row["matched_class_ids"]}
                ),
                "precision": match_count / candidate_count if candidate_count else 0.0,
                "recall": match_count / gt_count if gt_count else 0.0,
                "scene_equal_candidate_ap": float(np.mean(ap_values)) if ap_values else None,
                "scene_equal_precision": float(np.mean([row["precision"] for row in scene_rows])) if scene_rows else None,
                "tiny_small_gt_count": tiny_gt,
                "tiny_small_match_count": tiny_match,
                "tiny_small_recall": tiny_match / tiny_gt if tiny_gt else None,
                "scene_equal_tiny_small_candidate_ap": (
                    float(np.mean(tiny_ap_values)) if tiny_ap_values else None
                ),
                "tiny_small_ap_ignored_non_target_match_count": sum(
                    int(row["tiny_small_ap_ignored_non_target_match_count"])
                    for row in scene_rows
                ),
                "ap_true_positive_count": sum(int(row["ap_true_positive_count"]) for row in scene_rows),
                "ap_false_positive_count": sum(int(row["ap_false_positive_count"]) for row in scene_rows),
            }
        views[view_name] = {"per_scene": per_scene, "aggregate": aggregate}
    return {
        "schema": "saga-full-instance-size-candidate-evaluation-v2",
        "score_key": score_key,
        "eligible_only": eligible_only,
        "strict_iou_comparison": ">",
        "min_region_size": min_region_size,
        "candidate_rows": candidate_rows,
        "views": views,
    }


def oracle_class_diagnostics(
    candidates: Sequence[CandidatePrediction | Mapping[str, Any]],
    ground_truth: Sequence[GroundTruthScene],
    gt_point_to_gaussian_by_scene: Mapping[str, Sequence[int] | np.ndarray],
    *,
    score_key: str = "score",
    class_names: Sequence[str] | None = None,
    min_region_size: int = 100,
) -> tuple[dict[str, Any], ...]:
    """Classification-or-eligibility upper bound for clear GT overlap only.

    The oracle is diagnostic-only.  A candidate must have geometric IoU
    strictly greater than 0.25 with its best GT object before a GT class can be
    injected.  This deliberately does not claim that classification alone is
    the bottleneck: the full oracle may also repair automatic eligibility.
    """

    normalized = candidate_predictions_from_rows(
        candidates, score_key=score_key, class_names=class_names
    )
    gt = ground_truth_instances(ground_truth, min_region_size=min_region_size)
    gt_by_scene: dict[str, list[GroundTruthInstance]] = defaultdict(list)
    for row in gt:
        gt_by_scene[row.scene_id].append(row)
    result: list[dict[str, Any]] = []
    for candidate in normalized:
        scene_gt = gt_by_scene.get(candidate.scene_id, [])
        mapping = gt_point_to_gaussian_by_scene[candidate.scene_id]
        mask = project_candidate_to_gt_points(candidate, mapping)
        iou = pairwise_iou([mask], scene_gt)[0] if scene_gt else np.empty(0)
        best = float(iou.max()) if len(iou) else 0.0
        supported = best > ORACLE_CLASS_MIN_GEOMETRIC_IOU
        tied = (
            np.flatnonzero(np.isclose(iou, best, atol=1e-12, rtol=0.0))
            if supported
            else np.empty(0, dtype=np.int64)
        )
        tied_classes = sorted({scene_gt[int(index)].class_id for index in tied})
        unambiguous = supported and len(tied_classes) == 1
        result.append(
            {
                "scene_id": candidate.scene_id,
                "candidate_id": candidate.candidate_id,
                "predicted_class_id": candidate.class_id,
                "oracle_class_id": tied_classes[0] if unambiguous else None,
                "best_geometric_iou": best,
                "oracle_support_iou_threshold": ORACLE_CLASS_MIN_GEOMETRIC_IOU,
                "oracle_support_iou_comparison": ">",
                "below_or_at_oracle_support_threshold": not supported,
                "ambiguous": supported and not unambiguous,
                "oracle_semantics": "classification-or-eligibility-upper-bound",
                "diagnostic_only": True,
            }
        )
    return tuple(result)


def matched_recall_summary(
    ground_truth: GroundTruthScene,
    predictions: Sequence[PredictedInstance],
    *,
    size_by_gt: Mapping[tuple[int, int], str | None] | None = None,
    thresholds: Sequence[float] = (0.25, 0.50),
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Summarize endpoint recall with strict, one-to-one same-class matches.

    This is deliberately a diagnostic alongside the official evaluator.  It
    never lets two predictions claim one GT object, and it applies the same
    minimum predicted-region size as the official ScanNet evaluator.
    """

    tiny_ids: set[int | tuple[int, int]] = set()
    if size_by_gt is not None:
        tiny_ids = {
            (int(class_id), int(instance_id))
            for (class_id, instance_id), value in size_by_gt.items()
            if value in {"tiny", "small"}
        }
    gt = ground_truth_instances(
        [ground_truth],
        min_region_size=min_region_size,
        tiny_small_instance_ids={ground_truth.scene_id: tiny_ids},
    )
    kept_predictions: list[PredictedInstance] = []
    candidate_masks: list[np.ndarray] = []
    point_count = len(np.asarray(ground_truth.semantic))
    for prediction in predictions:
        if prediction.scene_id != ground_truth.scene_id:
            raise ValueError("predictions must belong to the supplied GT scene")
        mask = np.asarray(prediction.mask, dtype=bool)
        if mask.shape != (point_count,):
            raise ValueError("prediction mask does not match the GT point domain")
        if int(np.count_nonzero(mask)) < int(min_region_size):
            continue
        kept_predictions.append(prediction)
        candidate_masks.append(mask)
    iou = pairwise_iou(candidate_masks, gt)
    candidate_classes = np.asarray(
        [row.class_id for row in kept_predictions], dtype=np.int64
    )
    gt_classes = np.asarray([row.class_id for row in gt], dtype=np.int64)
    result: dict[str, Any] = {
        "scene_id": ground_truth.scene_id,
        "prediction_count": len(kept_predictions),
        "gt_count": len(gt),
        "tiny_small_gt_count": sum(int(row.is_tiny_small) for row in gt),
        "strict_iou_comparison": ">",
        "thresholds": {},
    }
    for threshold in thresholds:
        threshold_value = float(threshold)
        matches = maximum_cardinality_iou_matching(
            iou,
            threshold_value,
            candidate_classes=candidate_classes,
            gt_classes=gt_classes,
        )
        tiny_gt_indices = np.asarray(
            [index for index, row in enumerate(gt) if row.is_tiny_small],
            dtype=np.int64,
        )
        tiny_matches = maximum_cardinality_iou_matching(
            iou[:, tiny_gt_indices],
            threshold_value,
            candidate_classes=candidate_classes,
            gt_classes=gt_classes[tiny_gt_indices],
        )
        true_positive = len(matches)
        false_positive = len(kept_predictions) - true_positive
        false_negative = len(gt) - true_positive
        key = f"{int(round(threshold_value * 100)):03d}"
        row = {
            "threshold": threshold_value,
            "true_positive_count": true_positive,
            "false_positive_count": false_positive,
            "false_negative_count": false_negative,
            "precision": (
                true_positive / len(kept_predictions)
                if kept_predictions
                else 0.0
            ),
            "recall": true_positive / len(gt) if gt else 0.0,
            "fp_tp_ratio": false_positive / max(true_positive, 1),
            "tiny_small_match_count": len(tiny_matches),
            "tiny_small_recall": (
                len(tiny_matches) / len(tiny_gt_indices)
                if len(tiny_gt_indices)
                else None
            ),
        }
        result["thresholds"][key] = row
        suffix = f"{int(round(threshold_value * 100)):03d}"
        result[f"recall_{suffix}"] = row["recall"]
        result[f"fp_tp_ratio_{suffix}"] = row["fp_tp_ratio"]
        result[f"tiny_small_recall_{suffix}"] = row["tiny_small_recall"]
    return result


def _resolve_runtime_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str
) -> Path:
    value: Any = None
    for key in keys:
        if scene.get(key) not in {None, ""}:
            value = scene[key]
            break
    path = Path(str(default if value is None else value))
    if not path.is_absolute():
        path = Path(str(scene["base_path"])) / path
    return path.resolve()


def _runtime_point_cloud(scene: Mapping[str, Any]) -> Path:
    if scene.get("point_cloud_path") not in {None, ""}:
        return _resolve_runtime_path(scene, ("point_cloud_path",), "")
    base = Path(str(scene["base_path"])).resolve()
    standard = base / "output_models/point_cloud/iteration_30000/point_cloud.ply"
    alternate = base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
    return alternate if not standard.is_file() and alternate.is_file() else standard


def _runtime_gt_path(
    scene_id: str, scene: Mapping[str, Any], gt_dir: str | Path
) -> Path:
    if scene.get("gt_npz") not in {None, ""}:
        return _resolve_runtime_path(scene, ("gt_npz",), "")
    return (Path(gt_dir) / f"{scene_id}.npz").resolve()


def _runtime_transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _tiny_small_spec(
    gt_xyz: np.ndarray,
    gt: GroundTruthScene,
    size_spec: Mapping[str, Any] | None,
    min_region_size: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], str | None]]:
    if size_spec is None:
        return set(), {}
    limits = size_spec.get("boundaries_m", size_spec)
    tiny_max = float(limits["tiny_max_m"])
    small_max = float(limits["small_max_m"])
    medium_max = float(limits["medium_max_m"])
    if not 0.0 < tiny_max <= small_max <= medium_max:
        raise ValueError("size-bin boundaries must be positive and ordered")
    tiny_small: set[tuple[int, int]] = set()
    bins: dict[tuple[int, int], str | None] = {}
    semantic = np.asarray(gt.semantic, dtype=np.int64)
    instance = np.asarray(gt.instance, dtype=np.int64)
    for class_id in sorted(int(value) for value in np.unique(semantic) if value >= 0):
        class_mask = semantic == class_id
        for instance_id in sorted(
            int(value) for value in np.unique(instance[class_mask]) if value >= 0
        ):
            mask = class_mask & (instance == instance_id)
            if int(np.count_nonzero(mask)) < int(min_region_size):
                continue
            diagonal = float(np.linalg.norm(np.ptp(gt_xyz[mask], axis=0)))
            if diagonal <= tiny_max:
                size_bin = "tiny"
            elif diagonal <= small_max:
                size_bin = "small"
            elif diagonal <= medium_max:
                size_bin = "medium"
            else:
                size_bin = "large"
            bins[(class_id, instance_id)] = size_bin
            if size_bin in {"tiny", "small"}:
                tiny_small.add((class_id, instance_id))
    return tiny_small, bins


def _global_prior_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise TypeError("category priors are missing global/shrunk statistics")
    return node


def _class_prior_node(
    priors: Mapping[str, Any], class_name: str
) -> tuple[Mapping[str, Any], bool]:
    categories = priors.get("categories")
    if not isinstance(categories, Mapping) or class_name not in categories:
        return _global_prior_node(priors), True
    node = categories[class_name]
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise TypeError(f"category prior {class_name!r} lacks shrunk statistics")
    return node, False


def evaluate_candidate_scenes(
    *,
    scene_ids: Sequence[str],
    scenes: Mapping[str, Mapping[str, Any]],
    gt_dir: str | Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    taxonomy: Any,
    size_spec: Mapping[str, Any] | None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    priors: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen pre-KNN candidates without leaking GT into replay.

    The returned ``rows`` are scalar audit records suitable for Parquet.  The
    private in-memory ``_scene_payloads`` retain IoU matrices solely so the
    registered DEV2 threshold can be chosen without remapping the scene.
    """

    from .category_candidate_prior_v2 import size_platform_compatibility
    from .clean_baseline.evaluation import gt_point_to_gaussian_mapping

    ids = tuple(str(value) for value in scene_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("scene_ids must be unique")
    class_names = tuple(str(value) for value in taxonomy.canonical_classes)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    gt_scenes: list[GroundTruthScene] = []
    mapping_by_scene: dict[str, np.ndarray] = {}
    tiny_by_scene: dict[str, set[tuple[int, int]]] = {}
    scored_by_scene: dict[str, list[dict[str, Any]]] = {}
    full_oracle_by_scene: dict[str, list[dict[str, Any]]] = {}
    mapping_diagnostics: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    scene_payloads: dict[str, dict[str, Any]] = {}
    _global_prior_node(priors)

    for scene_id in ids:
        if scene_id not in scenes or scene_id not in snapshots:
            raise KeyError(f"missing runtime scene or snapshot for {scene_id}")
        scene = scenes[scene_id]
        snapshot = snapshots[scene_id]
        gt_xyz, gt = load_ground_truth_npz(
            _runtime_gt_path(scene_id, scene, gt_dir), scene_id
        )
        gaussian_xyz = apply_transform(
            load_ply_xyz(_runtime_point_cloud(scene)), _runtime_transform(scene)
        )
        mapping, diagnostics = gt_point_to_gaussian_mapping(
            gt_xyz, gaussian_xyz, radius_m=float(radius_m)
        )
        gt_scenes.append(gt)
        mapping_by_scene[scene_id] = mapping
        mapping_diagnostics[scene_id] = diagnostics
        tiny_small, size_bins = _tiny_small_spec(
            gt_xyz, gt, size_spec, min_region_size
        )
        tiny_by_scene[scene_id] = tiny_small

        trace_path = Path(str(snapshot["baseline_trace"]))
        with np.load(trace_path, allow_pickle=False) as trace:
            merged = np.asarray(trace["merged_partition"], dtype=np.int64)
        if merged.shape != (len(gaussian_xyz),):
            raise ValueError(f"{scene_id}: merged_partition/Gaussian mismatch")
        scene_gt = ground_truth_instances(
            [gt],
            min_region_size=min_region_size,
            tiny_small_instance_ids={scene_id: tiny_small},
        )
        candidate_rows: list[dict[str, Any]] = []
        candidate_masks: list[np.ndarray] = []
        for raw in snapshot.get("rows", []):
            row = dict(raw)
            raw_id = int(row["raw_instance_id"])
            members = np.flatnonzero(merged == raw_id).astype(np.int64, copy=False)
            if len(members) != int(row["point_count"]):
                raise ValueError(f"{scene_id}/{raw_id}: snapshot membership mismatch")
            predicted_name = row.get("predicted_class")
            row["class_id"] = (
                class_to_id[str(predicted_name)]
                if predicted_name in class_to_id
                else -1
            )
            row["member_indices"] = members
            candidate_rows.append(row)
            mask_candidate = CandidatePrediction(
                scene_id=scene_id,
                candidate_id=raw_id,
                class_id=max(int(row["class_id"]), 0),
                score=float(row["Q"]),
                member_indices=members,
            )
            candidate_masks.append(project_candidate_to_gt_points(mask_candidate, mapping))

        iou = pairwise_iou(candidate_masks, scene_gt)
        for index, row in enumerate(candidate_rows):
            best_any = float(iou[index].max()) if iou.shape[1] else 0.0
            oracle_supported_by_iou = best_any > ORACLE_CLASS_MIN_GEOMETRIC_IOU
            tied = (
                np.flatnonzero(np.isclose(iou[index], best_any, atol=1e-12, rtol=0.0))
                if oracle_supported_by_iou
                else np.empty(0, dtype=np.int64)
            )
            tied_classes = sorted({scene_gt[int(value)].class_id for value in tied})
            oracle_class_id = tied_classes[0] if len(tied_classes) == 1 else None
            oracle_class = (
                class_names[oracle_class_id]
                if oracle_class_id is not None and 0 <= oracle_class_id < len(class_names)
                else None
            )
            if oracle_class is not None:
                oracle_node, oracle_fallback = _class_prior_node(priors, oracle_class)
                full_oracle_g = float(size_platform_compatibility(row, oracle_node))
            else:
                oracle_fallback = True
                full_oracle_g = float(row.get("G_global", 1.0))

            # This first oracle changes only the prior lookup.  It intentionally
            # preserves the automatic eligibility and predicted class, so it
            # cannot by itself diagnose a classification failure.
            lookup_oracle_g = (
                full_oracle_g
                if bool(row.get("eligible", False)) and oracle_class is not None
                else float(row.get("G_global", 1.0))
            )
            row["G_oracle_lookup_only"] = lookup_oracle_g
            row["S_oracle_lookup_only"] = float(row["Q"]) * lookup_oracle_g
            row["oracle_lookup_prior_fallback"] = bool(
                oracle_class is None or oracle_fallback
            )

            # The full oracle is a separate, evaluation-only upper bound.  For
            # candidates with an unambiguous geometric GT class it replaces the
            # evaluation class and unlocks eligibility, but never changes the
            # geometry or Q.  Its global/class pair uses identical oracle rows;
            # their AP difference therefore isolates the size lookup rather
            # than crediting the class-label correction to the size prior.
            full_oracle_supported = oracle_class_id is not None
            row["G_oracle_class"] = full_oracle_g
            row["S_oracle_class"] = float(row["Q"]) * full_oracle_g
            row["S_oracle_global"] = float(row["Q"]) * float(
                row.get("G_global", 1.0)
            )
            row["full_oracle_supported"] = full_oracle_supported
            row["full_oracle_support_iou_threshold"] = (
                ORACLE_CLASS_MIN_GEOMETRIC_IOU
            )
            row["full_oracle_support_iou_comparison"] = ">"
            row["full_oracle_semantics"] = (
                "classification-or-eligibility-upper-bound"
            )
            row["full_oracle_eligible"] = bool(
                row.get("eligible", False) or full_oracle_supported
            )
            row["oracle_class_prior_fallback"] = oracle_fallback
            row["oracle_class_id"] = oracle_class_id
            row["oracle_class"] = oracle_class
            row["best_geometric_iou"] = best_any
            compatible = [
                gt_index
                for gt_index, gt_row in enumerate(scene_gt)
                if gt_row.class_id == row["class_id"]
            ]
            row["best_same_class_iou"] = (
                max(float(iou[index, gt_index]) for gt_index in compatible)
                if compatible
                else 0.0
            )
            row["best_gt_size_bin"] = (
                size_bins.get(
                    (
                        scene_gt[int(tied[0])].class_id,
                        scene_gt[int(tied[0])].instance_id,
                    )
                )
                if len(tied) == 1
                else None
            )
            flat_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "member_indices"
                }
            )
        scored_by_scene[scene_id] = candidate_rows
        oracle_rows: list[dict[str, Any]] = []
        for row in candidate_rows:
            oracle_row = dict(row)
            if bool(row["full_oracle_supported"]):
                oracle_row["class_id"] = int(row["oracle_class_id"])
            oracle_row["eligible"] = bool(row["full_oracle_eligible"])
            oracle_rows.append(oracle_row)
        full_oracle_by_scene[scene_id] = oracle_rows
        scene_payloads[scene_id] = {
            "iou": iou,
            "gt_classes": np.asarray(
                [row.class_id for row in scene_gt], dtype=np.int64
            ),
            "candidate_rows": candidate_rows,
        }

    modes = {
        "q-only": "Q",
        "global-g-only": "G_global",
        "class-g-only": "G_class",
        "global-size": "S_global",
        "class-size": "S_class",
        "oracle-prior-lookup-only": "S_oracle_lookup_only",
    }
    mode_results: dict[str, Any] = {}
    all_scored_rows = [
        row for scene_id in ids for row in scored_by_scene[scene_id]
    ]
    for mode, score_key in modes.items():
        mode_results[mode] = evaluate_candidate_rankings(
            all_scored_rows,
            gt_scenes,
            mapping_by_scene,
            score_key=score_key,
            eligible_only=True,
            min_region_size=min_region_size,
            tiny_small_instance_ids=tiny_by_scene,
            thresholds=(0.25, 0.50),
        )
    all_oracle_rows = [
        row for scene_id in ids for row in full_oracle_by_scene[scene_id]
    ]
    for mode, score_key in (
        ("oracle-class-global-size", "S_oracle_global"),
        ("oracle-class-size", "S_oracle_class"),
    ):
        mode_results[mode] = evaluate_candidate_rankings(
            all_oracle_rows,
            gt_scenes,
            mapping_by_scene,
            score_key=score_key,
            eligible_only=True,
            min_region_size=min_region_size,
            tiny_small_instance_ids=tiny_by_scene,
            thresholds=(0.25, 0.50),
        )
    return {
        "schema": "saga-full-instance-size-candidate-scenes-v3",
        "scene_ids": list(ids),
        "rows": flat_rows,
        "mode_results": mode_results,
        "mapping_diagnostics": mapping_diagnostics,
        "oracle_diagnostics": {
            "minimum_geometric_iou": ORACLE_CLASS_MIN_GEOMETRIC_IOU,
            "strict_iou_comparison": ">",
            "semantics": "classification-or-eligibility-upper-bound",
            "diagnostic_only": True,
        },
        "_scene_payloads": scene_payloads,
    }


def _mode_threshold(
    candidate_evaluation: Mapping[str, Any], mode: str, threshold: float
) -> Mapping[str, Any]:
    key = f"{int(round(float(threshold) * 100)):03d}"
    try:
        return candidate_evaluation["mode_results"][mode]["views"]["all"][
            "aggregate"
        ]["thresholds"][key]
    except KeyError as exc:
        raise KeyError(f"candidate evaluation lacks {mode}/{key}") from exc


def _finite_metric(value: Any) -> float:
    return 0.0 if value is None else float(value)


def _fp_tp_ratio(row: Mapping[str, Any]) -> float:
    true_positive = int(row["ap_true_positive_count"])
    false_positive = int(row["ap_false_positive_count"])
    return false_positive / max(true_positive, 1)


def analyze_candidate_ranking(
    candidate_evaluation: Mapping[str, Any],
    scored_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the registered DEV8 capacity, intervention, and ranking gates."""

    global_025 = _mode_threshold(candidate_evaluation, "global-size", 0.25)
    global_050 = _mode_threshold(candidate_evaluation, "global-size", 0.50)
    class_025 = _mode_threshold(candidate_evaluation, "class-size", 0.25)
    class_050 = _mode_threshold(candidate_evaluation, "class-size", 0.50)
    q_025 = _mode_threshold(candidate_evaluation, "q-only", 0.25)
    global_g_025 = _mode_threshold(candidate_evaluation, "global-g-only", 0.25)
    class_g_025 = _mode_threshold(candidate_evaluation, "class-g-only", 0.25)
    oracle_lookup_025 = _mode_threshold(
        candidate_evaluation, "oracle-prior-lookup-only", 0.25
    )
    oracle_global_025 = _mode_threshold(
        candidate_evaluation, "oracle-class-global-size", 0.25
    )
    oracle_class_025 = _mode_threshold(
        candidate_evaluation, "oracle-class-size", 0.25
    )

    eligible_rows = [row for row in scored_rows if bool(row.get("eligible", False))]
    prior_available = [
        row
        for row in eligible_rows
        if not bool(
            row.get(
                "class_prior_fallback",
                row.get("size_fallback_global", row.get("prior_fallback", False)),
            )
        )
    ]
    changed = [
        row
        for row in prior_available
        if abs(float(row["G_class"]) - float(row["G_global"])) >= 0.05
        or abs(float(row["S_class"]) - float(row["S_global"])) >= 0.01
    ]
    changed_scenes = {str(row["scene_id"]) for row in changed}
    changed_classes = {
        str(row["predicted_class"])
        for row in changed
        if row.get("predicted_class") is not None
    }
    required_changed = max(10, math.ceil(0.10 * len(prior_available)))
    mechanical_passed = bool(
        len(changed) >= required_changed
        and len(changed_scenes) >= 4
        and len(changed_classes) >= 3
    )

    mode_results = candidate_evaluation["mode_results"]
    global_scenes = mode_results["global-size"]["views"]["all"]["per_scene"]
    class_scenes = mode_results["class-size"]["views"]["all"]["per_scene"]
    positive_scenes = 0
    per_scene_deltas: list[dict[str, Any]] = []
    for scene_id in sorted(global_scenes):
        global_ap = _finite_metric(global_scenes[scene_id]["thresholds"]["025"]["candidate_ap"])
        class_ap = _finite_metric(class_scenes[scene_id]["thresholds"]["025"]["candidate_ap"])
        delta = class_ap - global_ap
        positive_scenes += int(delta > 0.0)
        per_scene_deltas.append(
            {
                "scene_id": scene_id,
                "global_candidate_ap_025": global_ap,
                "class_candidate_ap_025": class_ap,
                "delta_candidate_ap_025": delta,
            }
        )

    global_ap_025 = _finite_metric(global_025["scene_equal_candidate_ap"])
    class_ap_025 = _finite_metric(class_025["scene_equal_candidate_ap"])
    global_ap_050 = _finite_metric(global_050["scene_equal_candidate_ap"])
    class_ap_050 = _finite_metric(class_050["scene_equal_candidate_ap"])
    q_ap_025 = _finite_metric(q_025["scene_equal_candidate_ap"])
    global_g_ap_025 = _finite_metric(global_g_025["scene_equal_candidate_ap"])
    class_g_ap_025 = _finite_metric(class_g_025["scene_equal_candidate_ap"])
    oracle_lookup_ap_025 = _finite_metric(
        oracle_lookup_025["scene_equal_candidate_ap"]
    )
    oracle_global_ap_025 = _finite_metric(
        oracle_global_025["scene_equal_candidate_ap"]
    )
    oracle_class_ap_025 = _finite_metric(
        oracle_class_025["scene_equal_candidate_ap"]
    )
    predicted_size_delta = class_ap_025 - global_ap_025
    oracle_size_delta = oracle_class_ap_025 - oracle_global_ap_025
    oracle_size_has_value = oracle_size_delta >= 0.002
    oracle_advantage_over_predicted = (
        oracle_size_has_value
        and predicted_size_delta < 0.002
        and oracle_size_delta - predicted_size_delta >= 0.002
    )
    tiny_capacity = int(global_025["tiny_small_match_count"])
    tiny_guard_applied = tiny_capacity >= 5
    global_tiny_ap = global_025.get("scene_equal_tiny_small_candidate_ap")
    class_tiny_ap = class_025.get("scene_equal_tiny_small_candidate_ap")
    tiny_guard = bool(
        not tiny_guard_applied
        or (
            global_tiny_ap is not None
            and class_tiny_ap is not None
            and float(class_tiny_ap) >= float(global_tiny_ap)
        )
    )
    global_ratio = _fp_tp_ratio(global_025)
    class_ratio = _fp_tp_ratio(class_025)
    fp_guard = class_ratio <= 1.20 * max(global_ratio, 1e-12)
    ranking_passed = bool(
        class_ap_025 - global_ap_025 >= 0.002
        and class_ap_050 - global_ap_050 >= -0.002
        and class_ap_025 >= q_ap_025
        and positive_scenes >= 5
        and tiny_guard
    )

    return {
        "schema": "saga-full-instance-size-ranking-analysis-v2",
        "capacity": {
            "match_025": int(global_025["match_count"]),
            "match_025_scene_count": int(global_025["matched_scene_count"]),
            "match_025_class_count": int(global_025["matched_class_count"]),
            "match_050": int(global_050["match_count"]),
            "match_050_scene_count": int(global_050["matched_scene_count"]),
            "match_050_class_count": int(global_050["matched_class_count"]),
            "tiny_small_match_025": tiny_capacity,
            "tiny_small_capacity_sufficient_for_gate": tiny_guard_applied,
        },
        "mechanical_effect": {
            "passed": mechanical_passed,
            "prior_available_candidate_count": len(prior_available),
            "required_changed_candidate_count": required_changed,
            "changed_candidate_count": len(changed),
            "changed_scene_count": len(changed_scenes),
            "changed_class_count": len(changed_classes),
        },
        "ranking_gate": {
            "passed": ranking_passed,
            "global_candidate_ap_025": global_ap_025,
            "class_candidate_ap_025": class_ap_025,
            "delta_candidate_ap_025": class_ap_025 - global_ap_025,
            "global_candidate_ap_050": global_ap_050,
            "class_candidate_ap_050": class_ap_050,
            "delta_candidate_ap_050": class_ap_050 - global_ap_050,
            "q_only_candidate_ap_025": q_ap_025,
            "global_g_only_candidate_ap_025": global_g_ap_025,
            "class_g_only_candidate_ap_025": class_g_ap_025,
            "class_not_below_q_only": class_ap_025 >= q_ap_025,
            "positive_scene_count": positive_scenes,
            "tiny_small_guard_applied": tiny_guard_applied,
            "tiny_small_guard_passed": tiny_guard,
            "global_tiny_small_candidate_ap_025": global_tiny_ap,
            "class_tiny_small_candidate_ap_025": class_tiny_ap,
            "global_fp_tp_ratio_025": global_ratio,
            "class_fp_tp_ratio_025": class_ratio,
            "fp_tp_guard_passed": fp_guard,
            "fp_tp_guard_is_diagnostic_only": True,
            "per_scene": per_scene_deltas,
        },
        "oracle_gate": {
            "global_candidate_ap_025": global_ap_025,
            "predicted_candidate_ap_025": class_ap_025,
            "predicted_size_delta_candidate_ap_025": predicted_size_delta,
            "lookup_only_candidate_ap_025": oracle_lookup_ap_025,
            "oracle_class_global_size_candidate_ap_025": oracle_global_ap_025,
            "oracle_class_size_candidate_ap_025": oracle_class_ap_025,
            "oracle_size_delta_candidate_ap_025": oracle_size_delta,
            "oracle_size_has_discrimination_value": oracle_size_has_value,
            "oracle_size_advantage_over_predicted": oracle_advantage_over_predicted,
            # Compatibility keys consumed by the Stage-2 dispatcher.  Unlike
            # the former implementation, both are now based on within-arm
            # size deltas and cannot be triggered by oracle relabelling alone.
            "oracle_better_than_global": oracle_size_has_value,
            "oracle_better_than_predicted": oracle_advantage_over_predicted,
            "interpretation": "classification-or-eligibility-upper-bound",
            "diagnostic_only": True,
        },
    }


def choose_global_threshold(
    candidate_evaluation: Mapping[str, Any],
    scene_ids: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    """Choose the DEV2 global threshold by scene-equal F1@0.25.

    Exact F1 ties select the higher threshold, as preregistered.  Selection is
    based only on the global-size scores and never inspects class-size scores.
    """

    payloads = candidate_evaluation.get("_scene_payloads")
    if not isinstance(payloads, Mapping):
        raise ValueError("candidate evaluation lacks in-memory scene payloads")
    requested = tuple(str(value) for value in scene_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("scene_ids must be non-empty and unique")
    grid = sorted({_finite_unit(value, name="threshold") for value in thresholds})
    if not grid:
        raise ValueError("threshold grid cannot be empty")
    results: list[dict[str, Any]] = []
    for threshold in grid:
        scene_f1: list[float] = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        per_scene: list[dict[str, Any]] = []
        for scene_id in requested:
            if scene_id not in payloads:
                raise KeyError(f"candidate evaluation lacks scene {scene_id}")
            payload = payloads[scene_id]
            rows = [
                row
                for row in payload["candidate_rows"]
                if bool(row.get("eligible", False))
                and float(row["S_global"]) >= threshold
            ]
            all_rows = payload["candidate_rows"]
            positions = {
                int(row["raw_instance_id"]): index
                for index, row in enumerate(all_rows)
            }
            selected = [positions[int(row["raw_instance_id"])] for row in rows]
            iou = np.asarray(payload["iou"], dtype=np.float64)[selected, :]
            candidate_classes = np.asarray(
                [int(row["class_id"]) for row in rows], dtype=np.int64
            )
            gt_classes = np.asarray(payload["gt_classes"], dtype=np.int64)
            matches = maximum_cardinality_iou_matching(
                iou,
                0.25,
                candidate_classes=candidate_classes,
                gt_classes=gt_classes,
            )
            # Threshold selection evaluates the retained *set*, not its AP
            # ranking.  Use the preregistered maximum-cardinality, then
            # maximum-total-IoU assignment so an early high-score candidate
            # cannot consume the only GT available to another candidate and
            # artificially reduce F1.  Score-ordered matching remains the AP
            # contract elsewhere in this module.
            tp = len(matches)
            fp = len(rows) - tp
            fn = len(gt_classes) - tp
            denominator = 2 * tp + fp + fn
            f1 = 2 * tp / denominator if denominator else 0.0
            scene_f1.append(f1)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            per_scene.append(
                {
                    "scene_id": scene_id,
                    "candidate_f1_025": f1,
                    "true_positive_count": tp,
                    "false_positive_count": fp,
                    "false_negative_count": fn,
                }
            )
        results.append(
            {
                "threshold": threshold,
                "scene_equal_candidate_f1_025": float(np.mean(scene_f1)),
                "retained_true_positives_025": total_tp,
                "false_positive_count_025": total_fp,
                "false_negative_count_025": total_fn,
                "per_scene": per_scene,
            }
        )
    best = max(
        results,
        key=lambda row: (row["scene_equal_candidate_f1_025"], row["threshold"]),
    )
    return {
        "schema": "saga-full-instance-size-threshold-v1",
        **best,
        "selection_rule": "max_scene_equal_candidate_f1_025_then_higher_threshold",
        "grid": results,
    }


def paired_bootstrap(
    differences: Sequence[float] | np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Bootstrap already physical-scene-level paired differences."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("differences must be a non-empty finite vector")
    if isinstance(samples, bool) or int(samples) != samples or int(samples) <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    replicates = np.mean(values[indices], axis=1)
    low, high = np.quantile(replicates, (0.025, 0.975))
    return {
        "schema": "saga-full-instance-size-paired-bootstrap-v1",
        "unit": "physical_scene",
        "physical_scene_count": len(values),
        "difference": float(np.mean(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_bootstrap_ci95": [float(low), float(high)],
        "positive_physical_scenes": int(np.count_nonzero(values > 0.0)),
        "negative_physical_scenes": int(np.count_nonzero(values < 0.0)),
        "zero_physical_scenes": int(np.count_nonzero(values == 0.0)),
        "samples": int(samples),
        "seed": int(seed),
    }


def evaluate_official_protocols(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    *,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Keep ScanNet's nine-threshold primary metric separate from history."""

    protocols: dict[str, Any] = {}
    for name, overlaps, primary in (
        ("official_9", OFFICIAL_9_OVERLAPS, "official_map_50_90"),
        ("historical_10", HISTORICAL_10_OVERLAPS, "historical_map_50_95"),
    ):
        result = evaluate_instances(
            ground_truth,
            predictions,
            class_names,
            overlaps=overlaps,
            min_region_size=min_region_size,
        )
        result["aggregate"][primary] = result["aggregate"].pop("map_50_95")
        for values in result["per_class"].values():
            values[primary.replace("map", "ap")] = values.pop("ap_50_95")
        result["primary_metric"] = primary
        result["protocol"] = name
        protocols[name] = result
    return {
        "schema": "saga-full-instance-size-official-evaluation-v1",
        "protocols": protocols,
    }


def paired_physical_scene_bootstrap(
    control_by_scan: Mapping[str, float],
    treatment_by_scan: Mapping[str, float],
    *,
    physical_scene_by_scan: Mapping[str, str] | None = None,
    samples: int = 10_000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Bootstrap paired physical-scene mean differences, grouping scans first."""

    if set(control_by_scan) != set(treatment_by_scan):
        raise ValueError("control and treatment scan identities must match")
    if not control_by_scan:
        raise ValueError("paired bootstrap requires at least one scan")
    if physical_scene_by_scan is not None and set(physical_scene_by_scan) != set(
        control_by_scan
    ):
        raise ValueError("physical_scene_by_scan must cover exactly the paired scans")
    if isinstance(samples, bool) or int(samples) != samples or int(samples) <= 0:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("seed must be an integer")
    groups: dict[str, list[str]] = defaultdict(list)
    for scan_id in sorted(control_by_scan):
        physical_id = (
            scan_id
            if physical_scene_by_scan is None
            else str(physical_scene_by_scan[scan_id])
        )
        groups[physical_id].append(scan_id)
    physical_ids = sorted(groups)
    control = np.asarray(
        [np.mean([float(control_by_scan[scan]) for scan in groups[group]]) for group in physical_ids],
        dtype=np.float64,
    )
    treatment = np.asarray(
        [np.mean([float(treatment_by_scan[scan]) for scan in groups[group]]) for group in physical_ids],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(control)) or np.any(~np.isfinite(treatment)):
        raise ValueError("paired metrics must be finite")
    deltas = treatment - control
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(deltas), size=(int(samples), len(deltas)))
    replicates = np.mean(deltas[draws], axis=1)
    low, high = np.quantile(replicates, (0.025, 0.975))
    return {
        "schema": "saga-full-instance-size-paired-bootstrap-v1",
        "unit": "physical_scene",
        "scan_count": len(control_by_scan),
        "physical_scene_count": len(physical_ids),
        "physical_scene_ids": physical_ids,
        "control_mean": float(np.mean(control)),
        "treatment_mean": float(np.mean(treatment)),
        "difference": float(np.mean(deltas)),
        "ci95": [float(low), float(high)],
        "positive_physical_scenes": int(np.count_nonzero(deltas > 0.0)),
        "negative_physical_scenes": int(np.count_nonzero(deltas < 0.0)),
        "zero_physical_scenes": int(np.count_nonzero(deltas == 0.0)),
        "samples": int(samples),
        "seed": int(seed),
    }


__all__ = [
    "CandidateMatch",
    "CandidatePrediction",
    "GroundTruthInstance",
    "HISTORICAL_10_OVERLAPS",
    "OFFICIAL_9_OVERLAPS",
    "analyze_candidate_ranking",
    "candidate_average_precision",
    "candidate_predictions_from_rows",
    "choose_global_threshold",
    "evaluate_candidate_rankings",
    "evaluate_candidate_scenes",
    "evaluate_official_protocols",
    "ground_truth_instances",
    "matched_recall_summary",
    "maximum_cardinality_iou_matching",
    "oracle_class_diagnostics",
    "paired_bootstrap",
    "paired_physical_scene_bootstrap",
    "pairwise_iou",
    "project_candidate_to_gt_points",
    "ranked_candidate_matches",
]
