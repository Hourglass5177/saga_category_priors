"""Minimal fixed-partition replay for the final category-prior check.

The module deliberately does one thing: given an existing instance partition,
remove instances whose Gaussian count is below a threshold.  The uniform and
category-aware conditions therefore share exactly the same input candidates.
No ground truth, evaluator, clustering, or feature code is used here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MIN_POINTS = 10
MIN_CLASS_THRESHOLD = 3
MAX_CLASS_THRESHOLD = 10


def _labels_array(labels: np.ndarray | Collection[int]) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise ValueError("point labels must be a one-dimensional array")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("point labels must use an integer dtype")
    return array


def _threshold(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def replay_filter(
    labels: np.ndarray | Collection[int],
    threshold_by_id: Mapping[int | str, int] | None = None,
    *,
    default: int = DEFAULT_MIN_POINTS,
) -> np.ndarray:
    """Remove exactly the non-negative instance IDs smaller than their threshold.

    Counts equal to the threshold are retained.  Existing IDs are never
    renumbered, and negative labels are left unchanged.  String keys are
    accepted so a threshold table can be loaded directly from JSON.
    """

    source = _labels_array(labels)
    default_threshold = _threshold(default, name="default threshold")
    thresholds = {
        int(instance_id): _threshold(value, name=f"threshold for {instance_id}")
        for instance_id, value in (threshold_by_id or {}).items()
    }

    output = source.copy()
    instance_ids, counts = np.unique(source[source >= 0], return_counts=True)
    for instance_id, count in zip(instance_ids.tolist(), counts.tolist()):
        required = thresholds.get(int(instance_id), default_threshold)
        if int(count) < required:
            output[source == instance_id] = -1
    return output


def replay_u10(labels: np.ndarray | Collection[int]) -> np.ndarray:
    """Replay the historical uniform ``filter_num(10)`` rule."""

    return replay_filter(labels, default=DEFAULT_MIN_POINTS)


def u10_parity(
    labels: np.ndarray | Collection[int],
    expected_labels: np.ndarray | Collection[int],
) -> bool:
    """Return whether a uniform replay exactly matches a frozen U10 output."""

    expected = _labels_array(expected_labels)
    actual = replay_u10(labels)
    return actual.shape == expected.shape and bool(np.array_equal(actual, expected))


def load_category_priors(path: str | Path) -> dict[str, Any]:
    """Load the frozen train-only category-prior JSON table."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("category prior JSON must contain an object")
    return payload


def _log_area_q50(node: Mapping[str, Any]) -> float:
    return float(
        node["shrunk"]["geometry"]["log_surface_area_m2"]["q50"]
    )


def class_threshold(
    priors: Mapping[str, Any],
    class_name: str,
) -> int:
    """Derive one class threshold from frozen, shrunk train statistics.

    ``m_c = clip(round(10 * sqrt(A_c / A_global)), 3, 10)``.  Missing or
    malformed class statistics use the uniform threshold, as required by the
    final comparison.
    """

    categories = priors.get("categories", {})
    class_node = categories.get(class_name) if isinstance(categories, Mapping) else None
    try:
        global_log_area = _log_area_q50(priors["global"])
        if not isinstance(class_node, Mapping):
            return DEFAULT_MIN_POINTS
        class_log_area = _log_area_q50(class_node)
        ratio = math.exp(class_log_area - global_log_area)
        value = round(DEFAULT_MIN_POINTS * math.sqrt(ratio))
    except (KeyError, TypeError, ValueError, OverflowError):
        return DEFAULT_MIN_POINTS
    return int(np.clip(value, MIN_CLASS_THRESHOLD, MAX_CLASS_THRESHOLD))


def derive_thresholds_by_id(
    *,
    instance_classes: Mapping[int | str, str],
    branch_instance_ids: Collection[int],
    priors: Mapping[str, Any],
) -> dict[int, int]:
    """Build D-condition thresholds without changing non-branch instances."""

    branch_ids = {int(instance_id) for instance_id in branch_instance_ids}
    classes = {int(instance_id): str(name) for instance_id, name in instance_classes.items()}
    thresholds: dict[int, int] = {}
    for instance_id, class_name in classes.items():
        thresholds[instance_id] = (
            class_threshold(priors, class_name)
            if instance_id in branch_ids
            else DEFAULT_MIN_POINTS
        )
    return thresholds


def replay_class_prior(
    labels: np.ndarray | Collection[int],
    *,
    instance_classes: Mapping[int | str, str],
    branch_instance_ids: Collection[int],
    priors: Mapping[str, Any],
) -> tuple[np.ndarray, dict[int, int]]:
    """Replay the class-aware threshold on a frozen partition."""

    thresholds = derive_thresholds_by_id(
        instance_classes=instance_classes,
        branch_instance_ids=branch_instance_ids,
        priors=priors,
    )
    return replay_filter(labels, thresholds, default=DEFAULT_MIN_POINTS), thresholds


def summarize_replay(
    before: np.ndarray | Collection[int],
    after: np.ndarray | Collection[int],
    *,
    threshold_by_id: Mapping[int | str, int] | None = None,
    branch_instance_ids: Collection[int] = (),
) -> dict[str, Any]:
    """Return a compact, JSON-serializable mechanical-change summary."""

    source = _labels_array(before)
    replayed = _labels_array(after)
    if source.shape != replayed.shape:
        raise ValueError("before and after labels must have the same shape")

    before_ids, before_counts = np.unique(source[source >= 0], return_counts=True)
    after_ids, after_counts = np.unique(replayed[replayed >= 0], return_counts=True)
    before_count_by_id = {
        int(instance_id): int(count)
        for instance_id, count in zip(before_ids.tolist(), before_counts.tolist())
    }
    after_count_by_id = {
        int(instance_id): int(count)
        for instance_id, count in zip(after_ids.tolist(), after_counts.tolist())
    }
    removed_ids = sorted(set(before_count_by_id) - set(after_count_by_id))
    branch_ids = {int(instance_id) for instance_id in branch_instance_ids}
    before_id_set = set(before_count_by_id)
    after_id_set = set(after_count_by_id)
    changed = source != replayed
    labeled_before = source >= 0

    return {
        "point_count": int(source.size),
        "candidate_count_before": len(before_count_by_id),
        "candidate_count_after": len(after_count_by_id),
        "candidate_count_removed": len(removed_ids),
        "branch_candidate_count_before": len(before_id_set & branch_ids),
        "branch_candidate_count_after": len(after_id_set & branch_ids),
        "branch_candidate_count_removed": len(set(removed_ids) & branch_ids),
        "nonbranch_candidate_count_before": len(before_id_set - branch_ids),
        "nonbranch_candidate_count_after": len(after_id_set - branch_ids),
        "candidate_points_before": int(np.count_nonzero(labeled_before)),
        "candidate_points_after": int(np.count_nonzero(replayed >= 0)),
        "changed_point_count": int(np.count_nonzero(changed)),
        "changed_point_fraction_all": float(np.mean(changed)) if source.size else 0.0,
        "changed_point_fraction_candidates": (
            float(np.count_nonzero(changed & labeled_before) / np.count_nonzero(labeled_before))
            if np.any(labeled_before)
            else 0.0
        ),
        "removed_instance_ids": removed_ids,
        "removed_branch_instance_ids": [
            instance_id for instance_id in removed_ids if instance_id in branch_ids
        ],
        "removed_nonbranch_instance_ids": [
            instance_id for instance_id in removed_ids if instance_id not in branch_ids
        ],
        "candidate_point_counts_before": {
            str(instance_id): count for instance_id, count in before_count_by_id.items()
        },
        "candidate_point_counts_after": {
            str(instance_id): count for instance_id, count in after_count_by_id.items()
        },
        "threshold_by_id": {
            str(int(instance_id)): int(value)
            for instance_id, value in (threshold_by_id or {}).items()
        },
    }
