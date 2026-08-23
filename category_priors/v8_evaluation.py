from __future__ import annotations

"""Ground-truth-only oracle diagnostics for V8 lifted fragments.

The functions in this module operate on sparse point memberships that have
already been mapped into one shared ground-truth point-index space.  They have
no renderer, tracking, or runtime dependency: ground truth enters only here,
after candidate construction has finished.
"""

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np


_METRIC_NAMES = (
    "geometric_single",
    "semantic_single",
    "geometric_greedy_upper_bound",
    "semantic_greedy_upper_bound",
    "geometric_perfect_trim_support_ceiling",
    "semantic_perfect_trim_support_ceiling",
)


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _point_ids(values: np.ndarray | Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype == np.bool_:
        return np.flatnonzero(array).astype(np.int64, copy=False)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer point IDs or booleans")
    result = np.unique(array.astype(np.int64, copy=False))
    if len(result) and result[0] < 0:
        raise ValueError(f"{name} contains a negative point ID")
    return result


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return float(intersection / union) if union else 0.0


def _single_best(
    candidates: Sequence[np.ndarray], gt: np.ndarray
) -> tuple[float, int | None]:
    best_score = 0.0
    best_index: int | None = None
    for index, candidate in enumerate(candidates):
        score = _iou(candidate, gt)
        if score > best_score:
            best_score = score
            best_index = index
    return best_score, best_index


def _greedy_upper_bound(
    candidates: Sequence[np.ndarray], gt: np.ndarray
) -> tuple[float, tuple[int, ...]]:
    """Return a deterministic, monotonically improving GT-guided union.

    This is an oracle association upper-bound diagnostic, not a deployable
    association rule.  A fragment is accepted only when its union strictly
    increases IoU; ties are resolved by the original fragment order.
    """
    current = np.empty(0, dtype=np.int64)
    score = 0.0
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining:
        best_index: int | None = None
        best_union: np.ndarray | None = None
        best_score = score
        for index in remaining:
            combined = np.union1d(current, candidates[index])
            candidate_score = _iou(combined, gt)
            if candidate_score > best_score + 1e-12:
                best_index = index
                best_union = combined
                best_score = candidate_score
        if best_index is None or best_union is None:
            break
        current = best_union
        score = best_score
        selected.append(best_index)
        remaining.remove(best_index)
    return score, tuple(selected)


def _perfect_trim_ceiling(
    candidates: Sequence[np.ndarray], gt: np.ndarray
) -> float:
    """Recall ceiling when an oracle may remove every false-positive point."""
    if not len(gt) or not candidates:
        return 0.0
    support = np.empty(0, dtype=np.int64)
    for candidate in candidates:
        support = np.union1d(support, candidate)
    covered = len(np.intersect1d(support, gt, assume_unique=True))
    return float(covered / len(gt))


def _aggregate(
    rows: Sequence[dict[str, Any]],
    selected: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    included = [row for row, keep in zip(rows, selected) if bool(keep)]
    result: dict[str, Any] = {"gt_count": len(included)}
    for metric_name in _METRIC_NAMES:
        values = np.asarray(
            [float(row[metric_name]) for row in included], dtype=np.float64
        )
        metric: dict[str, Any] = {
            "mean_iou": float(values.mean()) if len(values) else 0.0,
        }
        for threshold in thresholds:
            suffix = f"{int(round(float(threshold) * 100)):03d}"
            count = int(np.count_nonzero(values >= float(threshold)))
            metric[f"match_{suffix}_count"] = count
            metric[f"recall_{suffix}"] = count / len(values) if len(values) else 0.0
        result[metric_name] = metric
    return result


def evaluate_fragment_oracles(
    fragment_point_ids: Sequence[np.ndarray | Sequence[int]],
    fragment_class_ids: Sequence[Hashable],
    gt_point_ids: Sequence[np.ndarray | Sequence[int]],
    gt_class_ids: Sequence[Hashable],
    *,
    fragment_ids: Sequence[Hashable] | None = None,
    gt_instance_ids: Sequence[Hashable] | None = None,
    gt_valid: Sequence[bool] | None = None,
    gt_is_tiny_small: Sequence[bool] | None = None,
    thresholds: Sequence[float] = (0.25, 0.50),
) -> dict[str, Any]:
    """Evaluate geometry and semantics without feeding GT into construction.

    Each fragment and GT instance is represented by point IDs in one shared
    mapped-GT universe.  Geometric metrics ignore fragment labels.  Semantic
    metrics admit only fragments whose class exactly equals the GT class.
    Perfect-trim ceilings remove all fragment points outside the target GT and
    therefore measure support coverage rather than deployable IoU.
    """
    fragment_count = len(fragment_point_ids)
    gt_count = len(gt_point_ids)
    if len(fragment_class_ids) != fragment_count:
        raise ValueError("fragment_class_ids must have one entry per fragment")
    if len(gt_class_ids) != gt_count:
        raise ValueError("gt_class_ids must have one entry per GT instance")
    if fragment_ids is None:
        fragment_ids = tuple(range(fragment_count))
    if gt_instance_ids is None:
        gt_instance_ids = tuple(range(gt_count))
    if len(fragment_ids) != fragment_count:
        raise ValueError("fragment_ids must have one entry per fragment")
    if len(gt_instance_ids) != gt_count:
        raise ValueError("gt_instance_ids must have one entry per GT instance")

    valid = (
        np.ones(gt_count, dtype=bool)
        if gt_valid is None
        else np.asarray(gt_valid, dtype=bool)
    )
    tiny_small = (
        np.zeros(gt_count, dtype=bool)
        if gt_is_tiny_small is None
        else np.asarray(gt_is_tiny_small, dtype=bool)
    )
    if valid.shape != (gt_count,):
        raise ValueError("gt_valid must have one entry per GT instance")
    if tiny_small.shape != (gt_count,):
        raise ValueError("gt_is_tiny_small must have one entry per GT instance")
    normalized_thresholds = tuple(float(value) for value in thresholds)
    if any(value < 0.0 or value > 1.0 for value in normalized_thresholds):
        raise ValueError("thresholds must lie in [0, 1]")

    fragments = [
        _point_ids(values, f"fragment_point_ids[{index}]")
        for index, values in enumerate(fragment_point_ids)
    ]
    ground_truth = [
        _point_ids(values, f"gt_point_ids[{index}]")
        for index, values in enumerate(gt_point_ids)
    ]
    rows: list[dict[str, Any]] = []
    for gt_index, gt in enumerate(ground_truth):
        gt_class = gt_class_ids[gt_index]
        same_class_indices = [
            index
            for index, fragment_class in enumerate(fragment_class_ids)
            if fragment_class == gt_class
        ]
        same_class_fragments = [fragments[index] for index in same_class_indices]

        geometric_single, geometric_single_index = _single_best(fragments, gt)
        semantic_single, semantic_local_index = _single_best(same_class_fragments, gt)
        semantic_single_index = (
            same_class_indices[semantic_local_index]
            if semantic_local_index is not None
            else None
        )
        geometric_greedy, geometric_selected = _greedy_upper_bound(fragments, gt)
        semantic_greedy, semantic_selected_local = _greedy_upper_bound(
            same_class_fragments, gt
        )
        semantic_selected = tuple(
            same_class_indices[index] for index in semantic_selected_local
        )
        rows.append(
            {
                "gt_index": gt_index,
                "gt_instance_id": _json_scalar(gt_instance_ids[gt_index]),
                "gt_class_id": _json_scalar(gt_class),
                "gt_point_count": int(len(gt)),
                "official_valid": bool(valid[gt_index]),
                "tiny_small": bool(tiny_small[gt_index]),
                "geometric_single": geometric_single,
                "geometric_single_fragment_id": (
                    _json_scalar(fragment_ids[geometric_single_index])
                    if geometric_single_index is not None
                    else None
                ),
                "semantic_single": semantic_single,
                "semantic_single_fragment_id": (
                    _json_scalar(fragment_ids[semantic_single_index])
                    if semantic_single_index is not None
                    else None
                ),
                "geometric_greedy_upper_bound": geometric_greedy,
                "geometric_greedy_fragment_ids": [
                    _json_scalar(fragment_ids[index]) for index in geometric_selected
                ],
                "semantic_greedy_upper_bound": semantic_greedy,
                "semantic_greedy_fragment_ids": [
                    _json_scalar(fragment_ids[index]) for index in semantic_selected
                ],
                "geometric_perfect_trim_support_ceiling": _perfect_trim_ceiling(
                    fragments, gt
                ),
                "semantic_perfect_trim_support_ceiling": _perfect_trim_ceiling(
                    same_class_fragments, gt
                ),
            }
        )

    all_rows = np.ones(gt_count, dtype=bool)
    return {
        "schema": "saga-v8-fragment-oracles-v1",
        "fragment_count": fragment_count,
        "gt_count": gt_count,
        "per_gt": rows,
        "aggregate": {
            "all": _aggregate(rows, all_rows, normalized_thresholds),
            "official_valid": _aggregate(rows, valid, normalized_thresholds),
            "tiny_small_official_valid": _aggregate(
                rows, valid & tiny_small, normalized_thresholds
            ),
        },
    }


def evaluate_v8_replays(**kwargs: Any) -> dict[str, Any]:
    """Run the shared official/Gaussian audit under an explicit V8 schema."""
    from .io import write_json
    from .v7_evaluation import evaluate_v7_replays

    result = evaluate_v7_replays(**kwargs)
    result["schema"] = "saga-v8-replay-analysis-v1"
    write_json(kwargs["analysis_output"], result)
    return result
