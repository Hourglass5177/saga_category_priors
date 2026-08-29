from __future__ import annotations

"""Pure same-bank category-prior scoring for experiment-plan section 30.

The scoring path deliberately has no ground-truth inputs.  It consumes one
frozen sequence of candidate rows, applies either the global (``uniform``) or
class-shrunk (``class``) statistics, and adds only the five arm-dependent
fields listed in :data:`DERIVED_SCORE_FIELDS`.  All other fields are preserved
so that the two arms can be audited as the same candidate bank.

Ground-truth-derived binary labels enter only the candidate-level evaluation
helpers at the bottom of this module.  Those helpers are diagnostics for
ranking and DEV2 threshold selection; they are not ScanNet instance AP.
"""

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


EXTENT_FIELDS = (
    "log_extent_short_m",
    "log_extent_mid_m",
    "log_extent_long_m",
)
DEV2_THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)
DERIVED_SCORE_FIELDS = frozenset(
    {"mode", "G", "S", "support_threshold", "support_pass"}
)


@dataclass(frozen=True)
class SameBankIdentityAudit:
    """Successful exact comparison of the non-arm fields in U and D rows."""

    candidate_ids: tuple[int, ...]
    q_values: tuple[float, ...]
    compared_row_count: int
    bank_identity_equal: bool = True
    q_unchanged: bool = True


@dataclass(frozen=True)
class SameBankCandidatePriorScores:
    """Uniform and class-shrunk scores materialized from one candidate bank."""

    uniform: tuple[dict[str, Any], ...]
    class_shrunk: tuple[dict[str, Any], ...]
    identity: SameBankIdentityAudit


@dataclass(frozen=True)
class ThresholdGridPoint:
    threshold: float
    f1: float


@dataclass(frozen=True)
class DEV2ThresholdSelection:
    """One frozen threshold selected on U from the registered DEV2 grid."""

    selected_threshold: float
    selected_f1: float
    grid: tuple[ThresholdGridPoint, ...]


def _shrunk_node(node: Any) -> Mapping[str, Any] | None:
    if not isinstance(node, Mapping):
        return None
    shrunk = node.get("shrunk")
    return node if isinstance(shrunk, Mapping) else None


def _global_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = _shrunk_node(priors.get("global"))
    if node is None:
        raise TypeError("category priors are missing a global shrunk node")
    return node


def _class_or_global_node(
    priors: Mapping[str, Any], class_name: str
) -> Mapping[str, Any]:
    categories = priors.get("categories")
    node = categories.get(class_name) if isinstance(categories, Mapping) else None
    return _shrunk_node(node) or _global_node(priors)


def _summary(
    node: Mapping[str, Any], section: str, field: str
) -> Mapping[str, Any]:
    shrunk = node.get("shrunk")
    subsection = shrunk.get(section) if isinstance(shrunk, Mapping) else None
    summary = subsection.get(field) if isinstance(subsection, Mapping) else None
    if not isinstance(summary, Mapping):
        raise TypeError(f"prior node is missing shrunk.{section}.{field}")
    return summary


def _finite_quantiles(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        q25 = float(summary["q25"])
        q50 = float(summary["q50"])
        q75 = float(summary["q75"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("extent summaries require numeric q25/q50/q75") from exc
    if not np.isfinite((q25, q50, q75)).all():
        raise ValueError("extent q25/q50/q75 must be finite")
    if q25 > q50 or q50 > q75:
        raise ValueError("extent quantiles must satisfy q25 <= q50 <= q75")
    return q25, q50, q75


def size_platform_compatibility(
    candidate: Mapping[str, Any], node: Mapping[str, Any]
) -> float:
    """Return the registered two-sided sorted-PCA log-extent compatibility G.

    Each sorted log extent has zero penalty on the inclusive ``[q25, q75]``
    platform.  Distance below or above that platform is normalized by its own
    half-IQR (``q50-q25`` or ``q75-q50``), squared, and capped at 25 before the
    three axes are averaged.
    """

    extents = np.asarray(candidate["metric_extents_m"], dtype=np.float64)
    if extents.shape != (3,):
        raise ValueError("candidate metric_extents_m must contain three values")
    if not np.isfinite(extents).all() or np.any(extents < 0.0):
        raise ValueError("candidate metric_extents_m must be finite and non-negative")
    log_extents = np.log(np.maximum(np.sort(extents), 1e-9))

    squared_z: list[float] = []
    for value, field in zip(log_extents, EXTENT_FIELDS):
        q25, q50, q75 = _finite_quantiles(_summary(node, "geometry", field))
        if value < q25:
            z_value = (q25 - float(value)) / max(q50 - q25, 1e-6)
        elif value > q75:
            z_value = (float(value) - q75) / max(q75 - q50, 1e-6)
        else:
            z_value = 0.0
        squared_z.append(min(z_value * z_value, 25.0))
    return float(math.exp(-0.5 * float(np.mean(squared_z))))


def _area_log_q50(node: Mapping[str, Any]) -> float:
    summary = _summary(node, "geometry", "log_surface_area_m2")
    try:
        value = float(summary["q50"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("surface-area summaries require numeric q50") from exc
    if not np.isfinite(value):
        raise ValueError("surface-area q50 must be finite")
    return value


def trusted_core_support_threshold(
    priors: Mapping[str, Any], class_name: str, mode: str
) -> int:
    """Return the hard trusted-core threshold for U or D.

    ``uniform`` is fixed at five.  ``class`` uses
    ``clip(round(5*sqrt(A_c/A_global)), 3, 10)`` where both areas are the
    exponentiated shrunk median log surface areas.  A missing class falls back
    to the global shrunk node and therefore returns five.
    """

    if mode == "uniform":
        return 5
    if mode != "class":
        raise ValueError("mode must be 'uniform' or 'class'")
    global_node = _global_node(priors)
    class_node = _class_or_global_node(priors, str(class_name))
    half_log_ratio = 0.5 * (
        _area_log_q50(class_node) - _area_log_q50(global_node)
    )
    # Values outside this range are already far beyond the final [3, 10]
    # clipping limits.  Clamping only prevents avoidable exp overflow.
    scale = math.exp(float(np.clip(half_log_ratio, -50.0, 50.0)))
    return int(np.clip(round(5.0 * scale), 3, 10))


def _candidate_q(candidate: Mapping[str, Any]) -> float:
    has_q = "Q" in candidate
    has_base = "base_score" in candidate
    if not has_q and not has_base:
        raise ValueError("candidate requires Q or base_score")
    try:
        q_value = float(candidate["Q"] if has_q else candidate["base_score"])
    except (TypeError, ValueError) as exc:
        raise TypeError("candidate Q must be numeric") from exc
    if has_q and has_base:
        try:
            base_value = float(candidate["base_score"])
        except (TypeError, ValueError) as exc:
            raise TypeError("candidate base_score must be numeric") from exc
        if q_value != base_value:
            raise ValueError("candidate Q and base_score disagree")
    if not np.isfinite(q_value) or not 0.0 <= q_value <= 1.0:
        raise ValueError("candidate Q must be finite and in [0, 1]")
    return q_value


def _trusted_core_count(candidate: Mapping[str, Any]) -> int:
    if "trusted_core_point_count" not in candidate:
        raise ValueError("candidate requires trusted_core_point_count")
    value = candidate["trusted_core_point_count"]
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("trusted_core_point_count must be an integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("trusted_core_point_count must be an integer") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("trusted_core_point_count must be an integer") from exc
    if not np.isfinite(exact) or exact != float(count) or count < 0:
        raise ValueError("trusted_core_point_count must be a non-negative integer")
    return count


def score_candidate_prior_v2(
    candidates: Sequence[Mapping[str, Any]],
    priors: Mapping[str, Any],
    mode: str,
) -> tuple[dict[str, Any], ...]:
    """Score a frozen candidate sequence without GT or a score threshold.

    The input rows are never mutated.  Every input field is copied into each
    output row, ``Q`` is materialized from ``base_score`` when needed, and only
    ``mode``, ``G``, ``S``, ``support_threshold`` and ``support_pass`` may
    differ between U and D.  Smoothness and neighborhood statistics are not
    read.
    """

    if mode not in {"uniform", "class"}:
        raise ValueError("mode must be 'uniform' or 'class'")
    global_node = _global_node(priors)
    rows: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("candidate rows must be mappings")
        collisions = DERIVED_SCORE_FIELDS.intersection(candidate)
        if collisions:
            raise ValueError(
                "candidate already contains derived score fields: "
                + ", ".join(sorted(collisions))
            )
        try:
            candidate_id = int(candidate["candidate_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TypeError("candidate_id must be an integer") from exc
        if candidate_id < 0 or candidate_id in observed_ids:
            raise ValueError("candidate_id values must be unique and non-negative")
        observed_ids.add(candidate_id)
        class_name = str(candidate.get("branch_class", ""))
        if not class_name:
            raise ValueError(f"candidate {candidate_id} requires branch_class")

        q_value = _candidate_q(candidate)
        trusted_count = _trusted_core_count(candidate)
        node = (
            global_node
            if mode == "uniform"
            else _class_or_global_node(priors, class_name)
        )
        g_value = size_platform_compatibility(candidate, node)
        support = trusted_core_support_threshold(priors, class_name, mode)

        row = copy.deepcopy(dict(candidate))
        row.setdefault("Q", q_value)
        row.update(
            {
                "mode": mode,
                "G": g_value,
                "S": q_value * g_value,
                "support_threshold": support,
                "support_pass": bool(trusted_count >= support),
            }
        )
        rows.append(row)
    return tuple(rows)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        if not isinstance(left, np.generic) or not isinstance(right, np.generic):
            return False
        if left.dtype != right.dtype:
            return False
        if (
            np.issubdtype(left.dtype, np.floating)
            and np.isnan(left)
            and np.isnan(right)
        ):
            return True
        return bool(left == right)
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, float) or not isinstance(right, float):
            return False
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right
    try:
        return bool(type(left) is type(right) and left == right)
    except (TypeError, ValueError):
        return False


def verify_same_bank_scores(
    uniform_rows: Sequence[Mapping[str, Any]],
    class_rows: Sequence[Mapping[str, Any]],
) -> SameBankIdentityAudit:
    """Verify exact U/D identity after removing only arm-derived fields."""

    if len(uniform_rows) != len(class_rows):
        raise ValueError("U/D candidate row counts differ")
    candidate_ids: list[int] = []
    q_values: list[float] = []
    observed_ids: set[int] = set()
    for index, (uniform, class_shrunk) in enumerate(zip(uniform_rows, class_rows)):
        if not isinstance(uniform, Mapping) or not isinstance(class_shrunk, Mapping):
            raise TypeError("U/D score rows must be mappings")
        uniform_complete = DERIVED_SCORE_FIELDS.issubset(uniform)
        class_complete = DERIVED_SCORE_FIELDS.issubset(class_shrunk)
        if not uniform_complete or not class_complete:
            raise ValueError("U/D rows are missing derived score fields")
        if uniform.get("mode") != "uniform" or class_shrunk.get("mode") != "class":
            raise ValueError("U/D score rows have invalid modes")
        left = {
            key: value
            for key, value in uniform.items()
            if key not in DERIVED_SCORE_FIELDS
        }
        right = {
            key: value
            for key, value in class_shrunk.items()
            if key not in DERIVED_SCORE_FIELDS
        }
        if not _values_equal(left, right):
            differing = sorted(
                key
                for key in set(left).union(right)
                if key not in left
                or key not in right
                or not _values_equal(left.get(key), right.get(key))
            )
            raise ValueError(
                f"U/D bank identity differs at row {index}: {differing}"
            )
        if "Q" not in left:
            raise ValueError(f"U/D row {index} does not materialize Q")
        candidate_id = int(left["candidate_id"])
        if candidate_id in observed_ids:
            raise ValueError("U/D score rows contain duplicate candidate IDs")
        observed_ids.add(candidate_id)
        candidate_ids.append(candidate_id)
        q_values.append(_candidate_q(left))
    return SameBankIdentityAudit(
        candidate_ids=tuple(candidate_ids),
        q_values=tuple(q_values),
        compared_row_count=len(candidate_ids),
    )


def score_same_bank_candidate_priors(
    candidates: Sequence[Mapping[str, Any]], priors: Mapping[str, Any]
) -> SameBankCandidatePriorScores:
    """Materialize U/D from the same input and immediately verify identity."""

    uniform = score_candidate_prior_v2(candidates, priors, "uniform")
    class_shrunk = score_candidate_prior_v2(candidates, priors, "class")
    identity = verify_same_bank_scores(uniform, class_shrunk)
    return SameBankCandidatePriorScores(
        uniform=uniform,
        class_shrunk=class_shrunk,
        identity=identity,
    )


def _evaluation_arrays(
    scores: Sequence[float],
    positives: Sequence[bool | int],
    eligible: Sequence[bool | int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score_array = np.asarray(scores, dtype=np.float64)
    positive_raw = np.asarray(positives)
    if score_array.ndim != 1 or positive_raw.ndim != 1:
        raise ValueError("scores and positives must be one-dimensional")
    if len(score_array) != len(positive_raw):
        raise ValueError("scores and positives must have equal length")
    if not np.isfinite(score_array).all() or np.any(
        (score_array < 0.0) | (score_array > 1.0)
    ):
        raise ValueError("candidate scores must be finite and in [0, 1]")
    if not np.all(np.isin(positive_raw, (False, True, 0, 1))):
        raise ValueError("positives must contain only binary values")
    positive_array = positive_raw.astype(bool, copy=False)
    if eligible is None:
        eligible_array = np.ones(len(score_array), dtype=bool)
    else:
        eligible_raw = np.asarray(eligible)
        if eligible_raw.ndim != 1 or len(eligible_raw) != len(score_array):
            raise ValueError("eligible must match scores")
        if not np.all(np.isin(eligible_raw, (False, True, 0, 1))):
            raise ValueError("eligible must contain only binary values")
        eligible_array = eligible_raw.astype(bool, copy=False)
    return score_array, positive_array, eligible_array


def _scene_partitions(
    length: int, scene_ids: Sequence[str] | None
) -> tuple[np.ndarray, ...]:
    if scene_ids is None:
        return (np.arange(length, dtype=np.int64),)
    scenes = np.asarray(scene_ids)
    if scenes.ndim != 1 or len(scenes) != length:
        raise ValueError("scene_ids must match scores")
    ordered: list[Any] = []
    for value in scenes.tolist():
        if not any(value == prior for prior in ordered):
            ordered.append(value)
    return tuple(np.flatnonzero(scenes == value) for value in ordered)


def _average_precision_one(
    scores: np.ndarray, positives: np.ndarray, eligible: np.ndarray
) -> float:
    positive_count = int(np.count_nonzero(positives))
    if positive_count == 0 or not np.any(eligible):
        return 0.0
    selected_scores = scores[eligible]
    selected_positive = positives[eligible]
    order = np.argsort(-selected_scores, kind="mergesort")
    sorted_scores = selected_scores[order]
    sorted_positive = selected_positive[order]

    ap = 0.0
    true_positive = 0
    detection_count = 0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        group_positive = int(np.count_nonzero(sorted_positive[start:stop]))
        true_positive += group_positive
        detection_count += stop - start
        if group_positive:
            ap += (group_positive / positive_count) * (
                true_positive / detection_count
            )
        start = stop
    return float(ap)


def candidate_average_precision(
    scores: Sequence[float],
    positives: Sequence[bool | int],
    *,
    eligible: Sequence[bool | int] | None = None,
    scene_ids: Sequence[str] | None = None,
) -> float:
    """Return threshold-free binary candidate AP, optionally scene-equal.

    Equal scores enter as one threshold group, so candidate order cannot break
    ties.  Support-failing candidates can be supplied through ``eligible``;
    they are absent from the ranked detections but still count in the positive
    denominator.  With ``scene_ids``, AP is computed per scene and then averaged
    with equal scene weight.  A partition with no positive candidate has AP 0.
    """

    score_array, positive_array, eligible_array = _evaluation_arrays(
        scores, positives, eligible
    )
    partitions = _scene_partitions(len(score_array), scene_ids)
    if not partitions:
        return 0.0
    return float(
        np.mean(
            [
                _average_precision_one(
                    score_array[index],
                    positive_array[index],
                    eligible_array[index],
                )
                for index in partitions
            ]
        )
    )


def _f1_one(
    scores: np.ndarray,
    positives: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> float:
    predicted = eligible & (scores >= threshold)
    true_positive = int(np.count_nonzero(predicted & positives))
    false_positive = int(np.count_nonzero(predicted & ~positives))
    false_negative = int(np.count_nonzero(~predicted & positives))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else float(2 * true_positive / denominator)


def candidate_f1_at_threshold(
    scores: Sequence[float],
    positives: Sequence[bool | int],
    threshold: float,
    *,
    eligible: Sequence[bool | int] | None = None,
    scene_ids: Sequence[str] | None = None,
) -> float:
    """Return binary candidate F1 at an inclusive S threshold."""

    threshold_value = float(threshold)
    if not np.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    score_array, positive_array, eligible_array = _evaluation_arrays(
        scores, positives, eligible
    )
    partitions = _scene_partitions(len(score_array), scene_ids)
    if not partitions:
        return 0.0
    return float(
        np.mean(
            [
                _f1_one(
                    score_array[index],
                    positive_array[index],
                    eligible_array[index],
                    threshold_value,
                )
                for index in partitions
            ]
        )
    )


def select_dev2_threshold(
    scores: Sequence[float],
    positives: Sequence[bool | int],
    thresholds: Sequence[float] = DEV2_THRESHOLD_GRID,
    *,
    eligible: Sequence[bool | int] | None = None,
    scene_ids: Sequence[str] | None = None,
) -> DEV2ThresholdSelection:
    """Select once on U using only the registered DEV2 grid.

    Passing a different grid is rejected rather than silently opening another
    tuning dimension.  Exact F1 ties select the higher threshold, after which
    the caller must reuse that threshold for D.
    """

    grid_values = tuple(float(value) for value in thresholds)
    if grid_values != DEV2_THRESHOLD_GRID:
        raise ValueError(f"thresholds must equal fixed DEV2 grid {DEV2_THRESHOLD_GRID}")
    grid = tuple(
        ThresholdGridPoint(
            threshold=value,
            f1=candidate_f1_at_threshold(
                scores,
                positives,
                value,
                eligible=eligible,
                scene_ids=scene_ids,
            ),
        )
        for value in DEV2_THRESHOLD_GRID
    )
    selected = max(grid, key=lambda point: (point.f1, point.threshold))
    return DEV2ThresholdSelection(
        selected_threshold=selected.threshold,
        selected_f1=selected.f1,
        grid=grid,
    )


__all__ = [
    "DERIVED_SCORE_FIELDS",
    "DEV2_THRESHOLD_GRID",
    "DEV2ThresholdSelection",
    "EXTENT_FIELDS",
    "SameBankCandidatePriorScores",
    "SameBankIdentityAudit",
    "ThresholdGridPoint",
    "candidate_average_precision",
    "candidate_f1_at_threshold",
    "score_candidate_prior_v2",
    "score_same_bank_candidate_priors",
    "select_dev2_threshold",
    "size_platform_compatibility",
    "trusted_core_support_threshold",
    "verify_same_bank_scores",
]
