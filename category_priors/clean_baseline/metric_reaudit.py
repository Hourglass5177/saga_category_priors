from __future__ import annotations

"""Corrected, read-only metrics for the clean alpha-mask baseline.

The historical clean-baseline diagnostics mixed three different coordinate
spaces.  This module keeps them deliberately separate:

* formal instance IoU is computed only on real ScanNet GT points;
* Gaussian-to-GT precision is a directional nearest-neighbour diagnostic;
* GT-to-Gaussian coverage/recall is the opposite directional diagnostic.

In particular, a Gaussian that is not the unique nearest Gaussian of any GT
point never creates a synthetic point in the formal IoU union.  It is still
counted as unsupported in the Gaussian-to-GT diagnostic.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from ..baseline_closure_evaluation import (
    HISTORICAL_OVERLAPS,
    SCANNET_OFFICIAL_OVERLAPS,
)
from ..evaluator import GroundTruthScene, PredictedInstance, evaluate_instances
from .evaluation import CleanCandidate, GroundTruthObject


REAUDIT_SCHEMA = "saga-clean-baseline-metric-reaudit-v1"
DEFAULT_RADII_M = (0.02, 0.05, 0.10)
FORMAL_RADIUS_M = 0.05
MATCH_THRESHOLDS = (0.25, 0.50)


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _normalize_ids(
    values: Sequence[int] | np.ndarray,
    *,
    upper_bound: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype == np.bool_:
        if len(raw) != int(upper_bound):
            raise ValueError(f"{name} boolean mask has the wrong length")
        result = np.flatnonzero(raw).astype(np.int64, copy=False)
    else:
        try:
            result = raw.astype(np.int64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must contain integer IDs") from exc
        if not np.array_equal(raw, result):
            raise TypeError(f"{name} must contain integer IDs")
        result = np.unique(result)
    if np.any(result < 0) or np.any(result >= int(upper_bound)):
        raise ValueError(f"{name} contains an out-of-range ID")
    return np.asarray(result, dtype=np.int64)


def _normalize_radii(radii_m: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted({float(value) for value in radii_m}))
    if not result or any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("radii_m must contain positive finite values")
    if not any(math.isclose(value, FORMAL_RADIUS_M) for value in result):
        raise ValueError("radii_m must include the formal 0.05 m radius")
    return result


@dataclass(frozen=True)
class BidirectionalNearest:
    """Unthresholded nearest-neighbour queries in both directions."""

    gt_to_gaussian_index: np.ndarray
    gt_to_gaussian_distance_m: np.ndarray
    gaussian_to_gt_index: np.ndarray
    gaussian_to_gt_distance_m: np.ndarray

    def __post_init__(self) -> None:
        gt_index = np.asarray(self.gt_to_gaussian_index, dtype=np.int64).copy()
        gt_distance = np.asarray(
            self.gt_to_gaussian_distance_m, dtype=np.float64
        ).copy()
        gaussian_index = np.asarray(
            self.gaussian_to_gt_index, dtype=np.int64
        ).copy()
        gaussian_distance = np.asarray(
            self.gaussian_to_gt_distance_m, dtype=np.float64
        ).copy()
        if gt_index.ndim != 1 or gt_index.shape != gt_distance.shape:
            raise ValueError("GT-to-Gaussian nearest arrays must be aligned vectors")
        if gaussian_index.ndim != 1 or gaussian_index.shape != gaussian_distance.shape:
            raise ValueError("Gaussian-to-GT nearest arrays must be aligned vectors")
        if np.any(gt_index < 0) or np.any(gt_index >= len(gaussian_index)):
            raise ValueError("GT-to-Gaussian index is out of range")
        if np.any(gaussian_index < 0) or np.any(gaussian_index >= len(gt_index)):
            raise ValueError("Gaussian-to-GT index is out of range")
        if not np.all(np.isfinite(gt_distance)) or np.any(gt_distance < 0):
            raise ValueError("GT-to-Gaussian distances must be finite and non-negative")
        if not np.all(np.isfinite(gaussian_distance)) or np.any(gaussian_distance < 0):
            raise ValueError("Gaussian-to-GT distances must be finite and non-negative")
        for array in (gt_index, gt_distance, gaussian_index, gaussian_distance):
            array.setflags(write=False)
        object.__setattr__(self, "gt_to_gaussian_index", gt_index)
        object.__setattr__(self, "gt_to_gaussian_distance_m", gt_distance)
        object.__setattr__(self, "gaussian_to_gt_index", gaussian_index)
        object.__setattr__(self, "gaussian_to_gt_distance_m", gaussian_distance)

    @property
    def gt_count(self) -> int:
        return int(len(self.gt_to_gaussian_index))

    @property
    def gaussian_count(self) -> int:
        return int(len(self.gaussian_to_gt_index))


def build_bidirectional_nearest(
    gt_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
) -> BidirectionalNearest:
    """Build nearest-neighbour queries once, without applying a radius."""

    gt = np.asarray(gt_xyz, dtype=np.float64)
    gaussian = np.asarray(gaussian_xyz, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,):
        raise ValueError("gt_xyz must have shape (N, 3)")
    if gaussian.ndim != 2 or gaussian.shape[1:] != (3,):
        raise ValueError("gaussian_xyz must have shape (M, 3)")
    if not len(gt) or not len(gaussian):
        raise ValueError("GT and Gaussian point sets must both be non-empty")
    gt_distance, gt_index = cKDTree(gaussian).query(gt, k=1, workers=-1)
    gaussian_distance, gaussian_index = cKDTree(gt).query(
        gaussian, k=1, workers=-1
    )
    return BidirectionalNearest(
        gt_index,
        gt_distance,
        gaussian_index,
        gaussian_distance,
    )


def formal_gt_point_mask(
    gaussian_ids: Sequence[int] | np.ndarray,
    nearest: BidirectionalNearest,
    *,
    radius_m: float = FORMAL_RADIUS_M,
) -> np.ndarray:
    """Project a Gaussian set into the real GT-point domain.

    The output length is exactly ``nearest.gt_count``.  No synthetic point is
    appended for a Gaussian that is not selected by the GT-to-Gaussian query.
    """

    ids = _normalize_ids(
        gaussian_ids,
        upper_bound=nearest.gaussian_count,
        name="candidate Gaussian IDs",
    )
    lookup = np.zeros(nearest.gaussian_count, dtype=bool)
    lookup[ids] = True
    valid = nearest.gt_to_gaussian_distance_m <= float(radius_m)
    result = np.zeros(nearest.gt_count, dtype=bool)
    result[valid] = lookup[nearest.gt_to_gaussian_index[valid]]
    return result


@dataclass(frozen=True)
class _FormalProjectionIndex:
    """Sparse inverse of the formal GT-to-Gaussian nearest mapping.

    A stage can contain thousands of mask/object hypotheses while a ScanNet
    scene contains millions of GT points.  Materialising one dense boolean GT
    vector per hypothesis therefore scales as ``candidate_count * gt_count``.
    This index stores the radius-valid inverse mapping once and projects each
    candidate to only the GT point IDs that it actually covers.
    """

    gaussian_offsets: np.ndarray
    gt_point_ids_by_gaussian: np.ndarray

    @classmethod
    def build(
        cls,
        nearest: BidirectionalNearest,
        *,
        radius_m: float = FORMAL_RADIUS_M,
    ) -> "_FormalProjectionIndex":
        valid_gt_ids = np.flatnonzero(
            nearest.gt_to_gaussian_distance_m <= float(radius_m)
        ).astype(np.int64, copy=False)
        mapped_gaussian_ids = nearest.gt_to_gaussian_index[valid_gt_ids]
        order = np.argsort(mapped_gaussian_ids, kind="stable")
        sorted_gaussian_ids = mapped_gaussian_ids[order]
        sorted_gt_ids = np.asarray(valid_gt_ids[order], dtype=np.int64)
        counts = np.bincount(
            sorted_gaussian_ids,
            minlength=nearest.gaussian_count,
        )
        offsets = np.empty(nearest.gaussian_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        offsets.setflags(write=False)
        sorted_gt_ids.setflags(write=False)
        return cls(offsets, sorted_gt_ids)

    def project(self, gaussian_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(gaussian_ids, dtype=np.int64)
        starts = self.gaussian_offsets[ids]
        stops = self.gaussian_offsets[ids + 1]
        populated = np.flatnonzero(stops > starts)
        if not populated.size:
            result = np.empty(0, dtype=np.int64)
        elif populated.size == 1:
            index = int(populated[0])
            result = self.gt_point_ids_by_gaussian[
                starts[index] : stops[index]
            ].copy()
        else:
            result = np.concatenate(
                [
                    self.gt_point_ids_by_gaussian[starts[index] : stops[index]]
                    for index in populated
                ]
            )
            result.sort()
        result = np.asarray(result, dtype=np.int64)
        result.setflags(write=False)
        return result


def _as_candidate(value: CleanCandidate | Mapping[str, Any] | Any) -> CleanCandidate:
    if isinstance(value, CleanCandidate):
        return value

    def field(name: str, *aliases: str, default: Any = None) -> Any:
        names = (name, *aliases)
        if isinstance(value, Mapping):
            for key in names:
                if key in value:
                    return value[key]
        else:
            for key in names:
                if hasattr(value, key):
                    return getattr(value, key)
        return default

    object_id = field("object_id", "candidate_id")
    gaussian_ids = field("gaussian_ids")
    if object_id is None or gaussian_ids is None:
        raise TypeError("candidate requires object_id/candidate_id and gaussian_ids")
    return CleanCandidate(
        object_id=object_id,
        gaussian_ids=np.asarray(gaussian_ids),
        class_id=field("class_id", "class_name"),
        winner_probability=float(field("winner_probability", default=0.0)),
        view_consensus=float(
            field("view_consensus", "mean_view_consensus", default=0.0)
        ),
        detection_ratio=float(
            field("detection_ratio", "mean_detection_ratio", default=0.0)
        ),
    )


def _validate_gt_objects(
    gt_objects: Sequence[GroundTruthObject],
    *,
    point_count: int,
) -> tuple[GroundTruthObject, ...]:
    result = tuple(gt_objects)
    object_ids: set[int] = set()
    occupied = np.zeros(int(point_count), dtype=bool)
    for gt in result:
        if int(gt.object_id) in object_ids:
            raise ValueError(f"duplicate GT object_id: {gt.object_id}")
        object_ids.add(int(gt.object_id))
        if np.any(gt.point_ids >= int(point_count)):
            raise ValueError("GT point ID outside nearest-neighbour domain")
        if np.any(occupied[gt.point_ids]):
            raise ValueError("GT objects must have disjoint point membership")
        occupied[gt.point_ids] = True
    return result


def _iou(predicted: np.ndarray, gt_point_ids: np.ndarray) -> float:
    predicted_count = int(np.count_nonzero(predicted))
    intersection = int(np.count_nonzero(predicted[gt_point_ids]))
    union = predicted_count + len(gt_point_ids) - intersection
    return float(intersection / union) if union else 0.0


def _iou_matrix(
    predicted_masks: Sequence[np.ndarray],
    gt_objects: Sequence[GroundTruthObject],
) -> np.ndarray:
    result = np.zeros((len(predicted_masks), len(gt_objects)), dtype=np.float64)
    for candidate_index, mask in enumerate(predicted_masks):
        for gt_index, gt in enumerate(gt_objects):
            result[candidate_index, gt_index] = _iou(mask, gt.point_ids)
    return result


def _sparse_iou_matrices(
    predicted_point_ids: Sequence[np.ndarray],
    gt_objects: Sequence[GroundTruthObject],
    *,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return IoU/intersection matrices from sparse formal point sets.

    Only the small ``candidate_count * gt_object_count`` result matrices are
    dense.  No ``candidate_count * gt_point_count`` mask matrix is created.
    GT instances are disjoint by contract, so a single point-to-object lookup
    is sufficient to count every intersection exactly.
    """

    candidate_count = len(predicted_point_ids)
    gt_count = len(gt_objects)
    intersections = np.zeros((candidate_count, gt_count), dtype=np.int64)
    if not candidate_count or not gt_count:
        return (
            np.zeros((candidate_count, gt_count), dtype=np.float64),
            intersections,
        )
    point_to_gt = np.full(int(point_count), -1, dtype=np.int32)
    gt_sizes = np.empty(gt_count, dtype=np.int64)
    for gt_index, gt in enumerate(gt_objects):
        point_to_gt[gt.point_ids] = int(gt_index)
        gt_sizes[gt_index] = len(gt.point_ids)
    predicted_sizes = np.empty(candidate_count, dtype=np.int64)
    for candidate_index, point_ids in enumerate(predicted_point_ids):
        ids = np.asarray(point_ids, dtype=np.int64)
        predicted_sizes[candidate_index] = len(ids)
        if not len(ids):
            continue
        rows = point_to_gt[ids]
        rows = rows[rows >= 0]
        if rows.size:
            intersections[candidate_index] = np.bincount(
                rows, minlength=gt_count
            )
    unions = predicted_sizes[:, None] + gt_sizes[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros((candidate_count, gt_count), dtype=np.float64),
        where=unions > 0,
    )
    return iou, intersections


def _best_match_from_row(
    iou_row: np.ndarray,
    intersection_row: np.ndarray,
    gt_objects: Sequence[GroundTruthObject],
    *,
    class_id: str | int | None = None,
) -> tuple[float, GroundTruthObject | None, int]:
    scored: list[tuple[float, int, GroundTruthObject]] = []
    for gt_index, gt in enumerate(gt_objects):
        if class_id is not None and str(class_id) != str(gt.class_id):
            continue
        scored.append(
            (
                float(iou_row[gt_index]),
                int(intersection_row[gt_index]),
                gt,
            )
        )
    if not scored:
        return 0.0, None, 0
    value, intersection, gt = min(
        scored,
        key=lambda row: (-row[0], -row[1], int(row[2].object_id)),
    )
    if intersection == 0:
        return 0.0, None, 0
    return value, gt, intersection


def _sorted_contains(sorted_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorised membership in a sorted unique integer ID vector."""

    query = np.asarray(values, dtype=np.int64)
    if not len(sorted_ids):
        return np.zeros(query.shape, dtype=bool)
    positions = np.searchsorted(sorted_ids, query)
    within = positions < len(sorted_ids)
    result = np.zeros(query.shape, dtype=bool)
    result[within] = sorted_ids[positions[within]] == query[within]
    return result


def deterministic_one_to_one_matches(
    *,
    candidate_ids: Sequence[Any],
    candidate_class_ids: Sequence[str | int | None],
    gt_objects: Sequence[GroundTruthObject],
    iou_matrix: np.ndarray,
    threshold: float,
    same_class: bool,
) -> list[dict[str, Any]]:
    """Lexicographic maximum one-to-one matching above one IoU threshold.

    The objective is, in order: maximum number of valid matches, maximum total
    IoU, then a deterministic stable-ID-derived tie score.  Candidate and GT
    inputs are sorted before assignment, so insertion order is irrelevant.
    """

    candidate_count = len(candidate_ids)
    gt_count = len(gt_objects)
    matrix = np.asarray(iou_matrix, dtype=np.float64)
    if matrix.shape != (candidate_count, gt_count):
        raise ValueError("iou_matrix shape does not match candidates and GT")
    if not math.isfinite(float(threshold)) or not 0 <= float(threshold) <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")
    if not candidate_count or not gt_count:
        return []

    candidate_order = sorted(
        range(candidate_count), key=lambda index: _stable_key(candidate_ids[index])
    )
    gt_order = sorted(
        range(gt_count), key=lambda index: int(gt_objects[index].object_id)
    )
    sorted_iou = matrix[np.ix_(candidate_order, gt_order)]
    valid = sorted_iou >= float(threshold)
    if same_class:
        class_compatible = np.asarray(
            [
                [
                    candidate_class_ids[candidate_index] is not None
                    and str(candidate_class_ids[candidate_index]) == str(gt.class_id)
                    for gt in (gt_objects[gt_index] for gt_index in gt_order)
                ]
                for candidate_index in candidate_order
            ],
            dtype=bool,
        )
        valid &= class_compatible
    maximum_pairs = min(candidate_count, gt_count)
    cardinality_bonus = float(maximum_pairs + 1)
    # Treat IoU differences below 1e-12 as exact ties.  The aggregate tie term
    # is kept below that tolerance and depends on both stable ranks.
    row_rank = np.arange(candidate_count, 0, -1, dtype=np.float64)[:, None]
    col_rank = np.arange(gt_count, 0, -1, dtype=np.float64)[None, :]
    tie = (row_rank * col_rank) / ((candidate_count + 1) * (gt_count + 1))
    tie *= 1e-12 / (maximum_pairs + 1)
    weights = np.where(valid, cardinality_bonus + sorted_iou + tie, 0.0)
    row_indices, column_indices = linear_sum_assignment(-weights)
    matches: list[dict[str, Any]] = []
    for row_index, column_index in zip(row_indices, column_indices):
        if not valid[row_index, column_index]:
            continue
        original_candidate = candidate_order[int(row_index)]
        original_gt = gt_order[int(column_index)]
        matches.append(
            {
                "candidate_id": candidate_ids[original_candidate],
                "gt_instance_id": int(gt_objects[original_gt].object_id),
                "gt_class_id": gt_objects[original_gt].class_id,
                "iou": float(matrix[original_candidate, original_gt]),
            }
        )
    return sorted(
        matches,
        key=lambda row: (
            _stable_key(row["candidate_id"]),
            int(row["gt_instance_id"]),
        ),
    )


def _best_match(
    mask: np.ndarray,
    gt_objects: Sequence[GroundTruthObject],
    *,
    class_id: str | int | None = None,
) -> tuple[float, GroundTruthObject | None, int]:
    scored: list[tuple[float, int, GroundTruthObject]] = []
    for gt in gt_objects:
        if class_id is not None and str(class_id) != str(gt.class_id):
            continue
        intersection = int(np.count_nonzero(mask[gt.point_ids]))
        scored.append((_iou(mask, gt.point_ids), intersection, gt))
    if not scored:
        return 0.0, None, 0
    value, intersection, gt = min(
        scored,
        key=lambda row: (-row[0], -row[1], int(row[2].object_id)),
    )
    if intersection == 0:
        return 0.0, None, 0
    return float(value), gt, int(intersection)


def _point_gt_lookup(
    gt_objects: Sequence[GroundTruthObject], point_count: int
) -> tuple[np.ndarray, np.ndarray]:
    instance = np.full(int(point_count), -1, dtype=np.int64)
    class_id = np.empty(int(point_count), dtype=object)
    class_id[:] = None
    for gt in gt_objects:
        instance[gt.point_ids] = int(gt.object_id)
        class_id[gt.point_ids] = gt.class_id
    return instance, class_id


def _matching_summary(
    *,
    candidates: Sequence[CleanCandidate],
    gt_objects: Sequence[GroundTruthObject],
    iou_matrix: np.ndarray,
    subset_indices: Sequence[int],
) -> dict[str, Any]:
    subset = list(subset_indices)
    candidate_ids = [candidates[index].object_id for index in subset]
    class_ids = [candidates[index].class_id for index in subset]
    matrix = np.asarray(iou_matrix, dtype=np.float64)[
        np.asarray(subset, dtype=np.int64)
    ]
    tiny_gt = [gt for gt in gt_objects if gt.is_tiny_small]
    result: dict[str, Any] = {
        "candidate_count": len(subset),
        "gt_count": len(gt_objects),
        "tiny_small_gt_count": len(tiny_gt),
        "matching": {},
    }
    for label, same_class in (("geometry", False), ("same_class", True)):
        threshold_rows: dict[str, Any] = {}
        for threshold in MATCH_THRESHOLDS:
            matches = deterministic_one_to_one_matches(
                candidate_ids=candidate_ids,
                candidate_class_ids=class_ids,
                gt_objects=gt_objects,
                iou_matrix=matrix,
                threshold=threshold,
                same_class=same_class,
            )
            matched_gt = {int(row["gt_instance_id"]) for row in matches}
            tiny_matched = sum(int(gt.object_id) in matched_gt for gt in tiny_gt)
            true_positive = len(matches)
            false_positive = len(subset) - true_positive
            false_negative = len(gt_objects) - true_positive
            threshold_rows[f"{threshold:.2f}"] = {
                "matches": matches,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
                "precision": (
                    float(true_positive / len(subset)) if subset else 0.0
                ),
                "recall": (
                    float(true_positive / len(gt_objects)) if gt_objects else 0.0
                ),
                "tiny_small_recall": (
                    float(tiny_matched / len(tiny_gt)) if tiny_gt else 0.0
                ),
                "total_matched_iou": float(
                    sum(float(row["iou"]) for row in matches)
                ),
            }
        result["matching"][label] = threshold_rows
    return result


def evaluate_candidate_set_three_spaces(
    *,
    candidates: Sequence[CleanCandidate | Mapping[str, Any] | Any],
    gt_objects: Sequence[GroundTruthObject],
    nearest: BidirectionalNearest,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    min_region_size: int = 100,
    _projection_index: _FormalProjectionIndex | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate set without mixing the three metric spaces."""

    if int(min_region_size) <= 0:
        raise ValueError("min_region_size must be positive")
    radii = _normalize_radii(radii_m)
    ground_truth = tuple(
        sorted(
            _validate_gt_objects(gt_objects, point_count=nearest.gt_count),
            key=lambda value: int(value.object_id),
        )
    )
    official_gt = tuple(gt for gt in ground_truth if gt.official_valid)
    normalized = tuple(
        sorted(
            (_as_candidate(value) for value in candidates),
            key=lambda value: _stable_key(value.object_id),
        )
    )
    if len({str(value.object_id) for value in normalized}) != len(normalized):
        raise ValueError("candidate IDs must be unique")

    candidate_gaussian_ids = [
        _normalize_ids(
            value.gaussian_ids,
            upper_bound=nearest.gaussian_count,
            name="candidate Gaussian IDs",
        )
        for value in normalized
    ]
    projection_index = (
        _FormalProjectionIndex.build(nearest)
        if _projection_index is None
        else _projection_index
    )
    if (
        len(projection_index.gaussian_offsets) != nearest.gaussian_count + 1
        or int(projection_index.gaussian_offsets[-1])
        != int(
            np.count_nonzero(
                nearest.gt_to_gaussian_distance_m <= FORMAL_RADIUS_M
            )
        )
    ):
        raise ValueError("formal projection index is incompatible with nearest mapping")
    formal_point_ids = [
        projection_index.project(ids) for ids in candidate_gaussian_ids
    ]
    all_iou, all_intersections = _sparse_iou_matrices(
        formal_point_ids,
        ground_truth,
        point_count=nearest.gt_count,
    )
    official_gt_positions = np.asarray(
        [
            index
            for index, gt in enumerate(ground_truth)
            if bool(gt.official_valid)
        ],
        dtype=np.int64,
    )
    if np.array_equal(
        official_gt_positions, np.arange(len(ground_truth), dtype=np.int64)
    ):
        official_iou = all_iou
        official_intersections = all_intersections
    else:
        official_iou = all_iou[:, official_gt_positions]
        official_intersections = all_intersections[:, official_gt_positions]
    point_instance, point_class = _point_gt_lookup(ground_truth, nearest.gt_count)
    gt_by_instance = {int(gt.object_id): gt for gt in ground_truth}
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(normalized):
        formal_ids = formal_point_ids[candidate_index]
        geometry_iou, geometry_gt, _ = _best_match_from_row(
            official_iou[candidate_index],
            official_intersections[candidate_index],
            official_gt,
        )
        same_iou, same_gt, _ = _best_match_from_row(
            official_iou[candidate_index],
            official_intersections[candidate_index],
            official_gt,
            class_id=candidate.class_id,
        )
        _, geometric_target, _ = _best_match_from_row(
            all_iou[candidate_index],
            all_intersections[candidate_index],
            ground_truth,
        )
        _, target, _ = _best_match_from_row(
            all_iou[candidate_index],
            all_intersections[candidate_index],
            ground_truth,
            class_id=candidate.class_id,
        )
        ids = candidate_gaussian_ids[candidate_index]
        gaussian_gt_points = nearest.gaussian_to_gt_index[ids]
        if geometric_target is None:
            # Early, class-agnostic funnel stages deliberately have no
            # semantic label.  Give them a geometry-only pollution target by
            # taking the deterministic majority GT instance among mapped
            # Gaussians.  This is an offline directional diagnostic only; it
            # never enters formal IoU, association, filtering, or scoring.
            valid_5cm = nearest.gaussian_to_gt_distance_m[ids] <= FORMAL_RADIUS_M
            supported_instances = point_instance[gaussian_gt_points]
            compatible = valid_5cm & (supported_instances >= 0)
            if np.any(compatible):
                identities, counts = np.unique(
                    supported_instances[compatible], return_counts=True
                )
                maximum = int(counts.max())
                target_id = int(identities[counts == maximum].min())
                geometric_target = gt_by_instance.get(target_id)
        if target is None and candidate.class_id is not None:
            # The formal GT-point projection can be empty when these Gaussians
            # are not the unique nearest representatives of the object.  Keep
            # that fact out of formal IoU, but choose a deterministic 5 cm
            # Gaussian-majority target so the directional precision diagnostic
            # still describes the candidate rather than an arbitrary GT.
            valid_5cm = nearest.gaussian_to_gt_distance_m[ids] <= FORMAL_RADIUS_M
            supported_instances = point_instance[gaussian_gt_points]
            supported_classes = point_class[gaussian_gt_points]
            compatible = valid_5cm & (supported_instances >= 0) & np.asarray(
                [str(value) == str(candidate.class_id) for value in supported_classes],
                dtype=bool,
            )
            if np.any(compatible):
                identities, counts = np.unique(
                    supported_instances[compatible], return_counts=True
                )
                maximum = int(counts.max())
                target_id = int(identities[counts == maximum].min())
                target = gt_by_instance.get(target_id)
        per_radius: dict[str, Any] = {}
        for radius in radii:
            within_radius = nearest.gaussian_to_gt_distance_m[ids] <= radius
            nearest_instances = point_instance[gaussian_gt_points]
            nearest_classes = point_class[gaussian_gt_points]
            evaluable = within_radius & (nearest_instances >= 0)
            same_class_support = evaluable & np.asarray(
                [
                    candidate.class_id is not None
                    and str(value) == str(candidate.class_id)
                    for value in nearest_classes
                ],
                dtype=bool,
            )
            correct = np.zeros(len(ids), dtype=bool)
            if target is not None:
                correct = same_class_support & (
                    nearest_instances == int(target.object_id)
                )
            same_class_wrong = same_class_support & ~correct
            wrong_class = evaluable & ~same_class_support
            unsupported = ~evaluable
            geometric_correct = np.zeros(len(ids), dtype=bool)
            if geometric_target is not None:
                geometric_correct = evaluable & (
                    nearest_instances == int(geometric_target.object_id)
                )
            geometric_other = evaluable & ~geometric_correct

            target_recalled = 0
            target_asset_covered = 0
            target_point_count = len(target.point_ids) if target is not None else 0
            if target is not None:
                target_valid = (
                    nearest.gt_to_gaussian_distance_m[target.point_ids] <= radius
                )
                target_asset_covered = int(np.count_nonzero(target_valid))
                target_nearest = nearest.gt_to_gaussian_index[target.point_ids]
                target_recalled = int(
                    np.count_nonzero(
                        target_valid & _sorted_contains(ids, target_nearest)
                    )
                )
            per_radius[f"{radius:.2f}"] = {
                "radius_m": float(radius),
                "gaussian_correct_target_instance_count": int(
                    np.count_nonzero(correct)
                ),
                "gaussian_same_class_wrong_instance_count": int(
                    np.count_nonzero(same_class_wrong)
                ),
                "gaussian_wrong_class_count": int(np.count_nonzero(wrong_class)),
                "gaussian_unsupported_count": int(np.count_nonzero(unsupported)),
                "gaussian_geometry_target_instance_count": int(
                    np.count_nonzero(geometric_correct)
                ),
                "gaussian_mapped_other_instance_count": int(
                    np.count_nonzero(geometric_other)
                ),
                "gaussian_to_gt_geometry_target_precision": float(
                    np.count_nonzero(geometric_correct) / len(ids)
                ),
                "gaussian_to_gt_geometry_pollution_fraction": float(
                    (np.count_nonzero(geometric_other) + np.count_nonzero(unsupported))
                    / len(ids)
                ),
                "gaussian_to_gt_target_precision": float(
                    np.count_nonzero(correct) / len(ids)
                ),
                "gaussian_to_gt_semantic_precision": float(
                    np.count_nonzero(same_class_support) / len(ids)
                ),
                "gaussian_to_gt_unsupported_fraction": float(
                    np.count_nonzero(unsupported) / len(ids)
                ),
                "target_gt_point_count": int(target_point_count),
                "target_gt_recalled_point_count": int(target_recalled),
                "target_gt_asset_covered_point_count": int(target_asset_covered),
                "gt_to_gaussian_candidate_recall": float(
                    target_recalled / target_point_count
                )
                if target_point_count
                else 0.0,
                "gt_to_gaussian_asset_coverage": float(
                    target_asset_covered / target_point_count
                )
                if target_point_count
                else 0.0,
            }
        candidate_rows.append(
            {
                "candidate_id": candidate.object_id,
                "candidate_class_id": candidate.class_id,
                "candidate_score": candidate.score,
                "candidate_gaussian_count": len(ids),
                "formal_gt_point_count_5cm": len(formal_ids),
                "official_evaluable": len(formal_ids) >= int(min_region_size),
                "formal_geometry_iou_5cm": float(geometry_iou),
                "formal_geometry_gt_instance_id": (
                    None if geometry_gt is None else int(geometry_gt.object_id)
                ),
                "formal_same_class_iou_5cm": float(same_iou),
                "formal_same_class_gt_instance_id": (
                    None if same_gt is None else int(same_gt.object_id)
                ),
                "precision_target_gt_instance_id": (
                    None if target is None else int(target.object_id)
                ),
                "geometric_precision_target_gt_instance_id": (
                    None
                    if geometric_target is None
                    else int(geometric_target.object_id)
                ),
                "radii": per_radius,
            }
        )

    all_indices = list(range(len(normalized)))
    official_indices = [
        index for index, row in enumerate(candidate_rows) if row["official_evaluable"]
    ]
    scene_coverage = {
        f"{radius:.2f}": {
            "radius_m": float(radius),
            "mapped_gt_point_count": int(
                np.count_nonzero(nearest.gt_to_gaussian_distance_m <= radius)
            ),
            "gt_point_count": nearest.gt_count,
            "mapped_fraction": float(
                np.mean(nearest.gt_to_gaussian_distance_m <= radius)
            ),
        }
        for radius in radii
    }
    return {
        "schema": REAUDIT_SCHEMA,
        "formal_metric_space": {
            "domain": "real_gt_points",
            "radius_m": FORMAL_RADIUS_M,
            "synthetic_false_positive_sentinels": False,
        },
        "directional_diagnostics_are_not_formal_iou": True,
        "candidate_rows": candidate_rows,
        "subsets": {
            "all": _matching_summary(
                candidates=normalized,
                gt_objects=official_gt,
                iou_matrix=official_iou,
                subset_indices=all_indices,
            ),
            "official_evaluable": _matching_summary(
                candidates=normalized,
                gt_objects=official_gt,
                iou_matrix=official_iou,
                subset_indices=official_indices,
            ),
        },
        "gt_to_gaussian_scene_coverage": scene_coverage,
        "official_gt_count": len(official_gt),
        "official_tiny_small_gt_count": sum(
            bool(gt.is_tiny_small) for gt in official_gt
        ),
    }


def support_coverage_ceiling(
    *,
    mask_gaussian_ids: Sequence[Sequence[int] | np.ndarray],
    gt_objects: Sequence[GroundTruthObject],
    nearest: BidirectionalNearest,
    mask_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Per-GT coverage after perfect FP trimming, not an instance upper bound."""

    ground_truth = _validate_gt_objects(gt_objects, point_count=nearest.gt_count)
    identifiers = (
        list(range(len(mask_gaussian_ids))) if mask_ids is None else list(mask_ids)
    )
    if len(identifiers) != len(mask_gaussian_ids):
        raise ValueError("mask_ids must contain one identity per support")
    projected = [
        formal_gt_point_mask(values, nearest) for values in mask_gaussian_ids
    ]
    rows: list[dict[str, Any]] = []
    for gt in ground_truth:
        covered = np.zeros(nearest.gt_count, dtype=bool)
        contributors: list[Any] = []
        for mask_id, mask in zip(identifiers, projected):
            overlap = mask[gt.point_ids]
            if np.any(overlap):
                covered[gt.point_ids[overlap]] = True
                contributors.append(mask_id)
        value = float(np.count_nonzero(covered[gt.point_ids]) / len(gt.point_ids))
        rows.append(
            {
                "gt_instance_id": int(gt.object_id),
                "gt_class_id": gt.class_id,
                "official_valid": bool(gt.official_valid),
                "is_tiny_small": bool(gt.is_tiny_small),
                "support_coverage_ceiling": value,
                "contributing_mask_ids": contributors,
            }
        )
    return {
        "schema": "saga-clean-baseline-support-coverage-ceiling-v1",
        "joint_instance_upper_bound": False,
        "false_positives_perfectly_removed": True,
        "evidence_may_be_reused_across_gt_objects": True,
        "rows": rows,
    }


def _protocol_result(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    *,
    overlaps: Sequence[float],
    primary_metric: str,
    protocol: str,
    min_region_size: int,
) -> dict[str, Any]:
    result = evaluate_instances(
        ground_truth,
        predictions,
        class_names,
        overlaps=overlaps,
        min_region_size=int(min_region_size),
    )
    raw_primary = result["aggregate"].pop("map_50_95")
    result["aggregate"][primary_metric] = raw_primary
    class_primary = primary_metric.replace("map_", "ap_")
    for values in result["per_class"].values():
        values[class_primary] = values.pop("ap_50_95")
    result["protocol"] = protocol
    result["primary_metric"] = primary_metric
    return result


def evaluate_dual_protocols(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    *,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Report ScanNet official nine thresholds and historical ten thresholds."""

    return {
        "schema": "saga-clean-baseline-dual-protocol-v1",
        "min_region_size": int(min_region_size),
        "official_9": _protocol_result(
            ground_truth,
            predictions,
            class_names,
            overlaps=SCANNET_OFFICIAL_OVERLAPS,
            primary_metric="map_50_90",
            protocol="ScanNet-official-instance-9-threshold",
            min_region_size=min_region_size,
        ),
        "historical_10": _protocol_result(
            ground_truth,
            predictions,
            class_names,
            overlaps=HISTORICAL_OVERLAPS,
            primary_metric="map_50_95",
            protocol="SAGA-historical-instance-10-threshold",
            min_region_size=min_region_size,
        ),
    }


def evaluate_gt_as_prediction_dual_protocols(
    ground_truth: Sequence[GroundTruthScene],
    class_names: Sequence[str],
    *,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Construct GT-as-prediction and require unit AP under both protocols."""

    predictions: list[PredictedInstance] = []
    export_id = 0
    for scene in ground_truth:
        for class_id in range(len(class_names)):
            class_mask = scene.semantic == class_id
            for instance_id in np.unique(scene.instance[class_mask]):
                if int(instance_id) < 0:
                    continue
                mask = class_mask & (scene.instance == int(instance_id))
                if int(np.count_nonzero(mask)) < int(min_region_size):
                    continue
                predictions.append(
                    PredictedInstance(
                        scene_id=scene.scene_id,
                        instance_id=export_id,
                        class_id=class_id,
                        score=1.0,
                        mask=mask,
                    )
                )
                export_id += 1
    result = evaluate_dual_protocols(
        ground_truth,
        predictions,
        class_names,
        min_region_size=min_region_size,
    )
    for protocol_name, primary in (
        ("official_9", "map_50_90"),
        ("historical_10", "map_50_95"),
    ):
        aggregate = result[protocol_name]["aggregate"]
        if not math.isclose(float(aggregate[primary]), 1.0, abs_tol=1e-12):
            raise AssertionError(f"GT-as-prediction {protocol_name} primary AP failed")
        if not math.isclose(float(aggregate["map_0.25"]), 1.0, abs_tol=1e-12):
            raise AssertionError(f"GT-as-prediction {protocol_name} AP25 failed")
    result["gt_as_prediction_parity"] = True
    return result
