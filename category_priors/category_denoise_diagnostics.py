from __future__ import annotations

"""Pure CPU diagnostics for the frozen category-denoising candidate bank.

This module deliberately stops before command-line parsing, filesystem I/O,
prediction construction, and KNN replay.  Ground truth is used only to audit
the immutable bank.  In particular, ``branch_core_labels`` is described as a
*retained* raw core: bank construction only persists a HDBSCAN core when its
corresponding full-assignment candidate has at least three points.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .category_denoise import (
    ASSIGNMENT_THRESHOLD,
    SCORE_THRESHOLD,
    SEMANTIC_THRESHOLD,
    CandidateBank,
    score_bank_candidates,
)
from .evaluator import GroundTruthScene, map_gaussians_to_gt
from .scannet import physical_scene_id
from .taxonomy import Taxonomy
from .v9_metrics import _bbox_diagonal, _size_bin

FLOOR_EXP_NEG_12_5 = math.exp(-12.5)
RANK_TOP_K = (10, 25, 50, 100, 200, 400)
SCORE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

# The five predicates intentionally remain available as independent columns.
# This precedence only supplies the requested single descriptive status.
FUNNEL_STATUS_PRECEDENCE = (
    "semantic_unreachable",
    "core_formation_failed",
    "full_assignment_helpful",
    "full_assignment_harmful",
    "usable_full_candidate",
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _xyz(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    return radius


@dataclass(frozen=True)
class NearestPointMapping:
    """Nearest-reference indices for one directed point-set mapping.

    ``-1`` and ``inf`` jointly denote a query with no reference within the
    frozen radius.  Arrays are read-only to prevent an evaluator from silently
    changing the mapping shared by several candidate stages.
    """

    indices: np.ndarray
    distances_m: np.ndarray
    radius_m: float
    diagnostics: dict[str, Any]

    @property
    def supported_mask(self) -> np.ndarray:
        return np.asarray(self.indices) >= 0


@dataclass(frozen=True)
class BidirectionalPointMapping:
    """Complementary official-projection and Gaussian-precision mappings."""

    gt_to_gaussian: NearestPointMapping
    gaussian_to_gt: NearestPointMapping


@dataclass(frozen=True)
class OfficialGroundTruthObject:
    """One official-valid SAGA20 GT instance in the GT point domain."""

    scene_id: str
    physical_scene_id: str
    class_id: int
    class_name: str
    instance_id: int
    point_indices: np.ndarray
    point_count: int
    bbox_diagonal_m: float
    size_bin: str | None


@dataclass(frozen=True)
class CandidateFunnelSceneResult:
    """Tables and aggregate diagnostics for one immutable scene bank."""

    scene_id: str
    candidate_rows: tuple[dict[str, Any], ...]
    gt_rows: tuple[dict[str, Any], ...]
    analysis: dict[str, Any]


def gt_to_gaussian_mapping(
    gt_xyz: Any,
    gaussian_xyz: Any,
    radius_m: float = 0.05,
) -> NearestPointMapping:
    """Map every GT point to its nearest Gaussian using the official helper."""

    gt = _xyz(gt_xyz, "gt_xyz")
    gaussian = _xyz(gaussian_xyz, "gaussian_xyz")
    radius = _radius(radius_m)
    if not len(gaussian):
        indices = np.full(len(gt), -1, dtype=np.int64)
        distances = np.full(len(gt), math.inf, dtype=np.float64)
        diagnostics: dict[str, Any] = {
            "mapped_fraction": 0.0,
            "median_nn_distance_m": None,
            "p95_nn_distance_m": None,
        }
    else:
        # Passing stable Gaussian indices through the repository helper exposes
        # its exact nearest-neighbour semantics once for every later label map.
        indices, raw_diagnostics = map_gaussians_to_gt(
            gt,
            gaussian,
            np.arange(len(gaussian), dtype=np.int64),
            radius,
        )
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.full(len(gt), math.inf, dtype=np.float64)
        supported = indices >= 0
        if np.any(supported):
            distances[supported] = np.linalg.norm(
                gt[supported] - gaussian[indices[supported]], axis=1
            )
        diagnostics = dict(raw_diagnostics)
        for key in ("median_nn_distance_m", "p95_nn_distance_m"):
            if not np.isfinite(float(diagnostics[key])):
                diagnostics[key] = None
    return NearestPointMapping(
        indices=_readonly(indices),
        distances_m=_readonly(distances),
        radius_m=radius,
        diagnostics=diagnostics,
    )


def gaussian_to_gt_mapping(
    gaussian_xyz: Any,
    gt_xyz: Any,
    radius_m: float = 0.05,
) -> NearestPointMapping:
    """Map every Gaussian to its nearest GT point for strict precision audit."""

    from scipy.spatial import cKDTree

    gaussian = _xyz(gaussian_xyz, "gaussian_xyz")
    gt = _xyz(gt_xyz, "gt_xyz")
    radius = _radius(radius_m)
    indices = np.full(len(gaussian), -1, dtype=np.int64)
    distances = np.full(len(gaussian), math.inf, dtype=np.float64)
    if len(gaussian) and len(gt):
        raw_distances, raw_indices = cKDTree(gt).query(
            gaussian,
            k=1,
            distance_upper_bound=radius,
            workers=-1,
        )
        supported = np.isfinite(raw_distances) & (raw_indices < len(gt))
        indices[supported] = np.asarray(raw_indices[supported], dtype=np.int64)
        distances[supported] = np.asarray(raw_distances[supported], dtype=np.float64)
    supported = indices >= 0
    diagnostics: dict[str, Any] = {
        "mapped_fraction": float(supported.mean()) if len(supported) else 0.0,
        "median_nn_distance_m": float(np.median(distances[supported]))
        if np.any(supported)
        else None,
        "p95_nn_distance_m": float(np.quantile(distances[supported], 0.95))
        if np.any(supported)
        else None,
    }
    return NearestPointMapping(
        indices=_readonly(indices),
        distances_m=_readonly(distances),
        radius_m=radius,
        diagnostics=diagnostics,
    )


def build_bidirectional_mapping(
    gt_xyz: Any,
    gaussian_xyz: Any,
    radius_m: float = 0.05,
) -> BidirectionalPointMapping:
    """Build both frozen nearest-neighbour directions for one scene."""

    return BidirectionalPointMapping(
        gt_to_gaussian=gt_to_gaussian_mapping(gt_xyz, gaussian_xyz, radius_m),
        gaussian_to_gt=gaussian_to_gt_mapping(gaussian_xyz, gt_xyz, radius_m),
    )


def build_official_gt_objects(
    scene_id: str,
    gt_xyz: Any,
    ground_truth: GroundTruthScene,
    taxonomy: Taxonomy,
    *,
    min_region_size: int = 100,
    size_spec: Mapping[str, Any] | None = None,
) -> tuple[OfficialGroundTruthObject, ...]:
    """Return sorted official-valid SAGA20 objects without materializing masks."""

    xyz = _xyz(gt_xyz, "gt_xyz")
    semantic = np.asarray(ground_truth.semantic, dtype=np.int64)
    instance = np.asarray(ground_truth.instance, dtype=np.int64)
    if semantic.shape != (len(xyz),) or instance.shape != (len(xyz),):
        raise ValueError(
            "GT coordinates, semantic and instance arrays differ in length"
        )
    if ground_truth.scene_id != str(scene_id):
        raise ValueError("ground_truth.scene_id does not match scene_id")
    if isinstance(min_region_size, bool) or int(min_region_size) <= 0:
        raise ValueError("min_region_size must be a positive integer")
    class_count = len(taxonomy.canonical_classes)
    unknown = semantic[(semantic >= class_count)]
    if len(unknown):
        raise ValueError("GT semantic contains a non-canonical non-negative class id")
    valid = (semantic >= 0) & (semantic < class_count) & (instance >= 0)
    pairs = sorted(
        set(zip(semantic[valid].tolist(), instance[valid].tolist())),
        key=lambda item: (int(item[0]), int(item[1])),
    )
    rows: list[OfficialGroundTruthObject] = []
    for raw_class_id, raw_instance_id in pairs:
        class_id = int(raw_class_id)
        instance_id = int(raw_instance_id)
        indices = np.flatnonzero(
            valid & (semantic == class_id) & (instance == instance_id)
        )
        if len(indices) < int(min_region_size):
            continue
        diagonal = float(_bbox_diagonal(xyz[indices]))
        rows.append(
            OfficialGroundTruthObject(
                scene_id=str(scene_id),
                physical_scene_id=physical_scene_id(str(scene_id)),
                class_id=class_id,
                class_name=str(taxonomy.canonical_classes[class_id]),
                instance_id=instance_id,
                point_indices=_readonly(indices.astype(np.int64, copy=False)),
                point_count=len(indices),
                bbox_diagonal_m=diagonal,
                size_bin=_size_bin(diagonal, size_spec),
            )
        )
    return tuple(rows)


def _project_gaussian_labels(
    gaussian_labels: np.ndarray,
    gt_to_gaussian: NearestPointMapping,
) -> np.ndarray:
    labels = np.asarray(gaussian_labels, dtype=np.int64)
    nearest = np.asarray(gt_to_gaussian.indices, dtype=np.int64)
    output = np.full(len(nearest), -1, dtype=np.int64)
    supported = nearest >= 0
    output[supported] = labels[nearest[supported]]
    return output


def _object_index_by_gt_point(
    point_count: int, objects: Sequence[OfficialGroundTruthObject]
) -> np.ndarray:
    output = np.full(point_count, -1, dtype=np.int64)
    for object_index, item in enumerate(objects):
        if np.any(output[item.point_indices] >= 0):
            raise ValueError("official GT objects overlap in point space")
        output[item.point_indices] = object_index
    return output


def _stage_iou(
    projected_labels: np.ndarray,
    candidate_count: int,
    point_object_index: np.ndarray,
    object_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(projected_labels, dtype=np.int64)
    if labels.shape != point_object_index.shape:
        raise ValueError("projected labels and GT object index differ in length")
    if np.any((labels < -1) | (labels >= candidate_count)):
        raise ValueError("projected candidate labels contain an invalid id")
    predicted_counts = np.bincount(
        labels[labels >= 0], minlength=candidate_count
    ).astype(np.int64, copy=False)
    intersections = np.zeros((candidate_count, len(object_counts)), dtype=np.int64)
    joint = (labels >= 0) & (point_object_index >= 0)
    if np.any(joint) and candidate_count and len(object_counts):
        encoded = labels[joint] * len(object_counts) + point_object_index[joint]
        intersections = np.bincount(
            encoded, minlength=candidate_count * len(object_counts)
        ).reshape(candidate_count, len(object_counts))
    unions = predicted_counts[:, None] + object_counts[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(unions, dtype=np.float64),
        where=unions > 0,
    )
    return iou, intersections, predicted_counts


def _best_index(
    values: np.ndarray, eligible_indices: np.ndarray
) -> tuple[int | None, float]:
    if not len(eligible_indices):
        return None, 0.0
    local = np.asarray(values, dtype=np.float64)[eligible_indices]
    selected_local = int(np.argmax(local))
    selected_value = float(local[selected_local])
    if selected_value <= 0.0:
        return None, 0.0
    return int(eligible_indices[selected_local]), selected_value


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _reverse_arrays(
    mapping: NearestPointMapping,
    ground_truth: GroundTruthScene,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nearest = np.asarray(mapping.indices, dtype=np.int64)
    semantic = np.full(len(nearest), -1, dtype=np.int64)
    instance = np.full(len(nearest), -1, dtype=np.int64)
    spatially_supported = nearest >= 0
    if np.any(spatially_supported):
        selected = nearest[spatially_supported]
        semantic[spatially_supported] = ground_truth.semantic[selected]
        instance[spatially_supported] = ground_truth.instance[selected]
    evaluable = (
        spatially_supported
        & (semantic >= 0)
        & (semantic < class_count)
        & (instance >= 0)
    )
    return semantic, instance, evaluable


def _gaussian_metrics(
    mask: np.ndarray,
    target: OfficialGroundTruthObject | None,
    reverse_semantic: np.ndarray,
    reverse_instance: np.ndarray,
    reverse_evaluable: np.ndarray,
) -> dict[str, float | int]:
    selected = np.asarray(mask, dtype=bool)
    total = int(np.count_nonzero(selected))
    unsupported = int(np.count_nonzero(selected & ~reverse_evaluable))
    correct = 0
    same_class = 0
    if target is not None:
        same_class = int(
            np.count_nonzero(
                selected
                & reverse_evaluable
                & (reverse_semantic == target.class_id)
            )
        )
        correct = int(
            np.count_nonzero(
                selected
                & reverse_evaluable
                & (reverse_semantic == target.class_id)
                & (reverse_instance == target.instance_id)
            )
        )
    return {
        "point_count": total,
        "correct_count": correct,
        "same_class_count": same_class,
        "purity": _fraction(correct, total),
        "semantic_precision": _fraction(same_class, total),
        "unsupported_count": unsupported,
        "unsupported_fraction": _fraction(unsupported, total),
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {f"q{int(round(q * 100)):02d}": None for q in SCORE_QUANTILES}
    result = np.quantile(array, SCORE_QUANTILES)
    return {
        f"q{int(round(q * 100)):02d}": float(value)
        for q, value in zip(SCORE_QUANTILES, result)
    }


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if len(left) < 2:
        return {"n": len(left), "rho": None, "reason": "insufficient_n"}
    if np.all(left == left[0]) or np.all(right == right[0]):
        return {"n": len(left), "rho": None, "reason": "constant_input"}
    from scipy.stats import spearmanr

    rho = float(spearmanr(left, right).statistic)
    if not np.isfinite(rho):
        return {"n": len(left), "rho": None, "reason": "non_finite_result"}
    return {"n": len(left), "rho": rho, "reason": None}


def _candidate_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    positive_025 = sum(float(row["full_best_same_class_iou"]) >= 0.25 for row in rows)
    positive_050 = sum(float(row["full_best_same_class_iou"]) >= 0.50 for row in rows)
    return {
        "candidate_count": count,
        "same_class_iou_025_count": int(positive_025),
        "same_class_iou_050_count": int(positive_050),
        "candidate_precision_025": _fraction(positive_025, count),
        "candidate_precision_050": _fraction(positive_050, count),
    }


def _stage_recall(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    fields = {
        "S0_semantic_reachable_025": "S0_semantic_coverage",
        "S1_raw_core_025": "S1_best_same_iou_raw_core",
        "S1_raw_core_050": "S1_best_same_iou_raw_core",
        "S1_core_intersection_full_025": "S1_best_same_iou_core_intersection_full",
        "S1_core_intersection_full_050": "S1_best_same_iou_core_intersection_full",
        "S2_full_025": "S2_best_same_iou",
        "S2_full_050": "S2_best_same_iou",
    }
    output: dict[str, Any] = {}
    for name, field in fields.items():
        threshold = 0.50 if name.endswith("050") else 0.25
        matched = sum(float(row[field]) >= threshold for row in rows)
        output[name] = {
            "matched_gt_count": int(matched),
            "official_valid_gt_count": total,
            "recall": _fraction(matched, total),
        }
    return output


def _score_domain(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "Q",
        "U_G_size",
        "U_B_smooth",
        "U_score",
        "D_G_size",
        "D_B_smooth",
        "D_score",
    )
    output: dict[str, Any] = {
        field: _quantiles([float(row[field]) for row in rows]) for field in fields
    }
    for mode in ("U", "D"):
        for factor in ("G_size", "B_smooth"):
            values = np.asarray(
                [float(row[f"{mode}_{factor}"]) for row in rows], dtype=np.float64
            )
            at_floor = int(
                np.count_nonzero(
                    np.isclose(
                        values,
                        FLOOR_EXP_NEG_12_5,
                        rtol=0.0,
                        atol=np.spacing(FLOOR_EXP_NEG_12_5) * 4.0,
                    )
                )
            )
            output[f"{mode}_{factor}_floor"] = {
                "count": at_floor,
                "fraction": _fraction(at_floor, len(values)),
                "floor_value": FLOOR_EXP_NEG_12_5,
            }
        scores = [float(row[f"{mode}_score"]) for row in rows]
        maximum = max(scores) if scores else None
        output[f"{mode}_threshold_reachability"] = {
            "threshold": SCORE_THRESHOLD,
            "observed_max": maximum,
            "observed_reachable": bool(
                maximum is not None and maximum >= SCORE_THRESHOLD
            ),
            "scope": "frozen_bank_and_current_priors",
        }
    return output


def _rank_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {
        "Q": [float(row["Q"]) for row in rows],
        "U_QxG": [float(row["Q"]) * float(row["U_G_size"]) for row in rows],
        "D_QxG": [float(row["Q"]) * float(row["D_G_size"]) for row in rows],
        "U_QxB": [float(row["Q"]) * float(row["U_B_smooth"]) for row in rows],
        "D_QxB": [float(row["Q"]) * float(row["D_B_smooth"]) for row in rows],
        "U_QxGxB": [float(row["U_score"]) for row in rows],
        "D_QxGxB": [float(row["D_score"]) for row in rows],
    }
    iou = [float(row["full_best_same_class_iou"]) for row in rows]
    output: dict[str, Any] = {}
    for name, values in scores.items():
        order = sorted(
            range(len(rows)),
            key=lambda index: (
                -values[index],
                str(rows[index]["scene_id"]),
                int(rows[index]["candidate_id"]),
            ),
        )
        top_k: dict[str, Any] = {}
        for requested in RANK_TOP_K:
            chosen = order[: min(requested, len(order))]
            top_k[str(requested)] = {
                "requested_k": requested,
                "actual_k": len(chosen),
                "same_class_iou_025_count": int(
                    sum(iou[index] >= 0.25 for index in chosen)
                ),
                "same_class_iou_050_count": int(
                    sum(iou[index] >= 0.50 for index in chosen)
                ),
            }
        output[name] = {
            "spearman_same_class_iou": _safe_spearman(values, iou),
            "top_k": top_k,
        }
    return output


def _counterfactual_summary(
    rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    full_iou: np.ndarray,
    candidate_class_ids: np.ndarray,
    objects: Sequence[OfficialGroundTruthObject],
) -> dict[str, Any]:
    layer_fields = (
        "P0_vote",
        "P0_U_support",
        "P0_D_support",
        "P1_U",
        "P1_D",
        "P2_U_size",
        "P2_D_size",
        "P3_U_smooth",
        "P3_D_smooth",
        "P4_U",
        "P4_D",
    )
    candidate_iou = np.asarray(
        [float(row["full_best_same_class_iou"]) for row in rows], dtype=np.float64
    )
    tiny_small = np.asarray(
        [item.size_bin in {"tiny", "small"} for item in objects], dtype=bool
    )
    output: dict[str, Any] = {}
    for field in layer_fields:
        retained = np.asarray([bool(row[field]) for row in rows], dtype=bool)
        positive_025 = int(np.count_nonzero(retained & (candidate_iou >= 0.25)))
        positive_050 = int(np.count_nonzero(retained & (candidate_iou >= 0.50)))
        best = np.zeros(len(objects), dtype=np.float64)
        for object_index, item in enumerate(objects):
            eligible = retained & (candidate_class_ids == item.class_id)
            if np.any(eligible):
                best[object_index] = float(np.max(full_iou[eligible, object_index]))
        retained_count = int(np.count_nonzero(retained))
        gt_025 = int(np.count_nonzero(best >= 0.25))
        gt_050 = int(np.count_nonzero(best >= 0.50))
        tiny_count = int(np.count_nonzero(tiny_small))
        tiny_025 = int(np.count_nonzero((best >= 0.25) & tiny_small))
        tiny_050 = int(np.count_nonzero((best >= 0.50) & tiny_small))
        output[field] = {
            "retained_candidate_count": retained_count,
            "same_class_iou_025_candidate_count": positive_025,
            "same_class_iou_050_candidate_count": positive_050,
            "candidate_precision_025": _fraction(positive_025, retained_count),
            "candidate_precision_050": _fraction(positive_050, retained_count),
            "official_valid_gt_count": len(gt_rows),
            "gt_recall_025_count": gt_025,
            "gt_recall_050_count": gt_050,
            "gt_recall_025": _fraction(gt_025, len(gt_rows)),
            "gt_recall_050": _fraction(gt_050, len(gt_rows)),
            "tiny_small_gt_count": tiny_count,
            "tiny_small_recall_025_count": tiny_025,
            "tiny_small_recall_050_count": tiny_050,
            "tiny_small_recall_025": _fraction(tiny_025, tiny_count),
            "tiny_small_recall_050": _fraction(tiny_050, tiny_count),
        }
    return output


def _tiny_small(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("size_bin") in {"tiny", "small"}]
    result = _stage_recall(selected)
    result["gt_count"] = len(selected)
    return result


def _per_class(
    candidate_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = sorted(
        {str(row["branch_class"]) for row in candidate_rows}
        | {str(row["gt_class"]) for row in gt_rows}
    )
    output: dict[str, Any] = {}
    for name in names:
        candidates = [row for row in candidate_rows if row["branch_class"] == name]
        ground_truth = [row for row in gt_rows if row["gt_class"] == name]
        output[name] = {
            "candidate_quality": _candidate_quality(candidates),
            "stage_recall": _stage_recall(ground_truth),
            "funnel_status_counts": dict(
                sorted(Counter(row["funnel_status"] for row in ground_truth).items())
            ),
            "tiny_small": _tiny_small(ground_truth),
        }
    return output


def _scene_analysis(
    scene_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    gt_rows: Sequence[Mapping[str, Any]],
    full_iou: np.ndarray,
    candidate_class_ids: np.ndarray,
    objects: Sequence[OfficialGroundTruthObject],
) -> dict[str, Any]:
    violations = sum(not bool(row["core_subset_full"]) for row in candidate_rows)
    flags = {
        name: int(sum(bool(row[f"funnel_flag_{name}"]) for row in gt_rows))
        for name in FUNNEL_STATUS_PRECEDENCE
    }
    return {
        "schema": "saga-category-denoise-funnel-scene-v1",
        "scene_id": scene_id,
        "candidate_count": len(candidate_rows),
        "official_valid_gt_count": len(gt_rows),
        "core_subset_violation_count": int(violations),
        "core_subset_violation_fraction": _fraction(violations, len(candidate_rows)),
        "stage_recall": _stage_recall(gt_rows),
        "funnel_status_counts": dict(
            sorted(Counter(row["funnel_status"] for row in gt_rows).items())
        ),
        "funnel_flag_counts": flags,
        "candidate_quality": _candidate_quality(candidate_rows),
        "score_domain": _score_domain(candidate_rows),
        "counterfactual_filters": _counterfactual_summary(
            candidate_rows,
            gt_rows,
            full_iou,
            candidate_class_ids,
            objects,
        ),
        "rank_diagnostics": _rank_diagnostics(candidate_rows),
        "per_class": _per_class(candidate_rows, gt_rows),
        "tiny_small": _tiny_small(gt_rows),
        "conclusion_boundary": {
            "S1_scope": "retained_raw_core_only",
            "discarded_raw_hdbscan_cores_recoverable_from_bank": False,
            "status_is_strict_causal_proof": False,
            "stage_best_candidates_are_not_guaranteed_identical": True,
            "S0_is_scene_wide_same_class_reachability_not_an_instance_candidate": True,
        },
    }


def diagnose_candidate_funnel_scene(
    *,
    scene_id: str,
    bank: CandidateBank,
    gaussian_xyz: Any,
    gt_xyz: Any,
    ground_truth: GroundTruthScene,
    taxonomy: Taxonomy,
    category_priors: Mapping[str, Any],
    size_spec: Mapping[str, Any] | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> CandidateFunnelSceneResult:
    """Audit S0/S1/S2 and U/D score domains for one frozen scene bank."""

    gaussian = _xyz(gaussian_xyz, "gaussian_xyz")
    gt = _xyz(gt_xyz, "gt_xyz")
    if len(gaussian) != bank.point_count:
        raise ValueError("Gaussian coordinates do not match the candidate bank")
    if ground_truth.semantic.shape != (len(gt),) or ground_truth.instance.shape != (
        len(gt),
    ):
        raise ValueError(
            "GT coordinates, semantic and instance arrays differ in length"
        )
    objects = build_official_gt_objects(
        scene_id,
        gt,
        ground_truth,
        taxonomy,
        min_region_size=min_region_size,
        size_spec=size_spec,
    )
    mapping = build_bidirectional_mapping(gt, gaussian, radius_m)
    candidate_count = len(bank.candidates)
    class_to_gt_id = {
        str(name): index for index, name in enumerate(taxonomy.canonical_classes)
    }
    missing_bank_classes = sorted(set(bank.saga20_names) - set(class_to_gt_id))
    if missing_bank_classes:
        raise ValueError(
            f"bank SAGA20 classes are absent from taxonomy: {missing_bank_classes}"
        )
    candidate_class_ids = np.asarray(
        [class_to_gt_id[str(row["branch_class"])] for row in bank.candidates],
        dtype=np.int64,
    )

    decisions_u = score_bank_candidates(bank, category_priors, "uniform")
    decisions_d = score_bank_candidates(bank, category_priors, "class")
    by_id_u = {int(row["candidate_id"]): row for row in decisions_u}
    by_id_d = {int(row["candidate_id"]): row for row in decisions_d}

    point_object_index = _object_index_by_gt_point(len(gt), objects)
    object_counts = np.asarray([item.point_count for item in objects], dtype=np.int64)
    projected_core = _project_gaussian_labels(
        bank.branch_core_labels, mapping.gt_to_gaussian
    )
    projected_full = _project_gaussian_labels(
        bank.branch_full_labels, mapping.gt_to_gaussian
    )
    same_id_intersection_labels = np.where(
        (bank.branch_core_labels >= 0)
        & (bank.branch_core_labels == bank.branch_full_labels),
        bank.branch_core_labels,
        -1,
    )
    projected_core_intersection = _project_gaussian_labels(
        same_id_intersection_labels, mapping.gt_to_gaussian
    )
    core_iou, core_intersections, _ = _stage_iou(
        projected_core, candidate_count, point_object_index, object_counts
    )
    core_intersection_iou, _, _ = _stage_iou(
        projected_core_intersection,
        candidate_count,
        point_object_index,
        object_counts,
    )
    full_iou, full_intersections, _ = _stage_iou(
        projected_full, candidate_count, point_object_index, object_counts
    )

    reverse_semantic, reverse_instance, reverse_evaluable = _reverse_arrays(
        mapping.gaussian_to_gt, ground_truth, len(taxonomy.canonical_classes)
    )
    object_indices_by_class: dict[int, np.ndarray] = {
        class_id: np.asarray(
            [index for index, item in enumerate(objects) if item.class_id == class_id],
            dtype=np.int64,
        )
        for class_id in range(len(taxonomy.canonical_classes))
    }
    all_object_indices = np.arange(len(objects), dtype=np.int64)

    candidate_rows: list[dict[str, Any]] = []
    for candidate in bank.candidates:
        candidate_id = int(candidate["candidate_id"])
        branch_class = str(candidate["branch_class"])
        branch_gt_class_id = class_to_gt_id[branch_class]
        same_objects = object_indices_by_class[branch_gt_class_id]
        core_same_index, core_same_iou = _best_index(
            core_iou[candidate_id], same_objects
        )
        core_any_index, core_any_iou = _best_index(
            core_iou[candidate_id], all_object_indices
        )
        full_same_index, full_same_iou = _best_index(
            full_iou[candidate_id], same_objects
        )
        full_any_index, full_any_iou = _best_index(
            full_iou[candidate_id], all_object_indices
        )
        core_target = objects[core_same_index] if core_same_index is not None else None
        full_target = objects[full_same_index] if full_same_index is not None else None
        core_mask = np.asarray(bank.branch_core_labels) == candidate_id
        full_mask = np.asarray(bank.branch_full_labels) == candidate_id
        intersection_mask = core_mask & full_mask
        core_metrics = _gaussian_metrics(
            core_mask,
            core_target,
            reverse_semantic,
            reverse_instance,
            reverse_evaluable,
        )
        full_metrics = _gaussian_metrics(
            full_mask,
            full_target,
            reverse_semantic,
            reverse_instance,
            reverse_evaluable,
        )
        decision_u = by_id_u[candidate_id]
        decision_d = by_id_d[candidate_id]
        if not math.isclose(
            float(decision_u["Q"]), float(decision_d["Q"]), rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError("uniform/class replay changed the frozen Q evidence")
        q_value = float(decision_u["Q"])
        vote_gate = bool(decision_u["vote_matches_branch"]) and float(
            candidate["branch_vote_ratio"]
        ) >= ASSIGNMENT_THRESHOLD
        core_actual = int(np.count_nonzero(core_mask))
        p0_u = vote_gate and core_actual >= int(decision_u["support_threshold"])
        p0_d = vote_gate and core_actual >= int(decision_d["support_threshold"])
        p4_u = p0_u and q_value * float(decision_u["G_size"]) * float(
            decision_u["B_smooth"]
        ) >= SCORE_THRESHOLD
        p4_d = p0_d and q_value * float(decision_d["G_size"]) * float(
            decision_d["B_smooth"]
        ) >= SCORE_THRESHOLD
        if p4_u != bool(decision_u["accepted"]) or p4_d != bool(
            decision_d["accepted"]
        ):
            raise RuntimeError(
                "counterfactual P4 does not reproduce candidate acceptance"
            )
        extents = np.sort(
            np.asarray(candidate["metric_extents_m"], dtype=np.float64)
        )
        if extents.shape != (3,) or not np.isfinite(extents).all():
            raise ValueError(f"candidate {candidate_id} has invalid metric extents")
        same_class_mask = np.asarray(
            [item.class_id == branch_gt_class_id for item in objects], dtype=bool
        )
        core_best_key = (
            (core_target.class_name, core_target.instance_id)
            if core_target is not None
            else (None, None)
        )
        full_best_key = (
            (full_target.class_name, full_target.instance_id)
            if full_target is not None
            else (None, None)
        )
        candidate_rows.append(
            {
                "scene_id": str(scene_id),
                "candidate_id": candidate_id,
                "branch_class": branch_class,
                "branch_class_index": int(candidate["branch_class_index"]),
                "core_point_count_recorded": int(candidate["core_point_count"]),
                "full_point_count_recorded": int(candidate["full_point_count"]),
                "core_actual_count": core_actual,
                "full_actual_count": int(np.count_nonzero(full_mask)),
                "core_intersection_full_count": int(
                    np.count_nonzero(intersection_mask)
                ),
                "core_only_count": int(np.count_nonzero(core_mask & ~full_mask)),
                "full_only_count": int(np.count_nonzero(full_mask & ~core_mask)),
                "core_in_full_fraction": _fraction(
                    np.count_nonzero(intersection_mask), core_actual
                ),
                "core_subset_full": bool(np.all(~core_mask | full_mask)),
                "metric_extent_short_m": float(extents[0]),
                "metric_extent_mid_m": float(extents[1]),
                "metric_extent_long_m": float(extents[2]),
                "boundary_ratio_5cm": float(candidate["boundary_ratio_5cm"]),
                "assignment_confidence_mean": float(
                    candidate["assignment_confidence_mean"]
                ),
                "assignment_confidence_actual_mean": float(
                    np.mean(bank.assignment_confidence[full_mask])
                )
                if np.any(full_mask)
                else 0.0,
                "branch_vote_ratio": float(candidate["branch_vote_ratio"]),
                "background_vote_ratio": float(candidate["background_vote_ratio"]),
                "vote_winner": candidate.get("vote_winner"),
                "vote_winner_unique": bool(candidate.get("vote_winner_unique", False)),
                "vote_matches_branch": bool(decision_u["vote_matches_branch"]),
                "Q": q_value,
                "U_G_size": float(decision_u["G_size"]),
                "U_B_smooth": float(decision_u["B_smooth"]),
                "U_support_threshold": int(decision_u["support_threshold"]),
                "U_score": float(decision_u["score"]),
                "U_accepted": bool(decision_u["accepted"]),
                "D_G_size": float(decision_d["G_size"]),
                "D_B_smooth": float(decision_d["B_smooth"]),
                "D_support_threshold": int(decision_d["support_threshold"]),
                "D_score": float(decision_d["score"]),
                "D_accepted": bool(decision_d["accepted"]),
                "P0_vote": vote_gate,
                "P0_U_support": p0_u,
                "P0_D_support": p0_d,
                "P1_U": p0_u and q_value >= SCORE_THRESHOLD,
                "P1_D": p0_d and q_value >= SCORE_THRESHOLD,
                "P2_U_size": p0_u
                and q_value * float(decision_u["G_size"]) >= SCORE_THRESHOLD,
                "P2_D_size": p0_d
                and q_value * float(decision_d["G_size"]) >= SCORE_THRESHOLD,
                "P3_U_smooth": p0_u
                and q_value * float(decision_u["B_smooth"]) >= SCORE_THRESHOLD,
                "P3_D_smooth": p0_d
                and q_value * float(decision_d["B_smooth"]) >= SCORE_THRESHOLD,
                "P4_U": p4_u,
                "P4_D": p4_d,
                "core_best_same_class_iou": core_same_iou,
                "core_best_any_class_iou": core_any_iou,
                "full_best_same_class_iou": full_same_iou,
                "full_best_any_class_iou": full_any_iou,
                "core_best_same_class_gt_class": core_best_key[0],
                "core_best_same_class_gt_instance": core_best_key[1],
                "full_best_same_class_gt_class": full_best_key[0],
                "full_best_same_class_gt_instance": full_best_key[1],
                "core_best_any_class_gt_class": objects[core_any_index].class_name
                if core_any_index is not None
                else None,
                "core_best_any_class_gt_instance": objects[core_any_index].instance_id
                if core_any_index is not None
                else None,
                "full_best_any_class_gt_class": objects[full_any_index].class_name
                if full_any_index is not None
                else None,
                "full_best_any_class_gt_instance": objects[full_any_index].instance_id
                if full_any_index is not None
                else None,
                "core_gaussian_purity_5cm": float(core_metrics["purity"]),
                "core_gaussian_semantic_precision_5cm": float(
                    core_metrics["semantic_precision"]
                ),
                "core_gt_coverage_5cm": (
                    _fraction(
                        core_intersections[candidate_id, core_same_index],
                        core_target.point_count,
                    )
                    if core_target is not None
                    else 0.0
                ),
                "core_unsupported_fraction_5cm": float(
                    core_metrics["unsupported_fraction"]
                ),
                "full_gaussian_purity_5cm": float(full_metrics["purity"]),
                "full_gaussian_semantic_precision_5cm": float(
                    full_metrics["semantic_precision"]
                ),
                "full_gt_coverage_5cm": (
                    _fraction(
                        full_intersections[candidate_id, full_same_index],
                        full_target.point_count,
                    )
                    if full_target is not None
                    else 0.0
                ),
                "full_unsupported_fraction_5cm": float(
                    full_metrics["unsupported_fraction"]
                ),
                "full_overlapping_gt_count_any": int(
                    np.count_nonzero(full_intersections[candidate_id] > 0)
                ),
                "full_overlapping_gt_count_iou_005": int(
                    np.count_nonzero(full_iou[candidate_id] >= 0.05)
                ),
                "full_overlapping_same_class_gt_count_iou_005": int(
                    np.count_nonzero(
                        (full_iou[candidate_id] >= 0.05) & same_class_mask
                    )
                ),
                "full_vs_core_best_same_iou_delta": full_same_iou - core_same_iou,
                "full_best_same_gt_matches_core_best": core_best_key == full_best_key
                and core_target is not None,
                "full_assignment_expanded": bool(np.any(full_mask & ~core_mask)),
                "full_assignment_lost_raw_core": bool(np.any(core_mask & ~full_mask)),
            }
        )

    gt_rows: list[dict[str, Any]] = []
    gt_to_gaussian_indices = np.asarray(mapping.gt_to_gaussian.indices, dtype=np.int64)
    for object_index, item in enumerate(objects):
        same_candidates = np.flatnonzero(candidate_class_ids == item.class_id)
        core_same_candidate, core_same_value = _best_index(
            core_iou[:, object_index], same_candidates
        )
        core_any_candidate, core_any_value = _best_index(
            core_iou[:, object_index], np.arange(candidate_count, dtype=np.int64)
        )
        full_same_candidate, full_same_value = _best_index(
            full_iou[:, object_index], same_candidates
        )
        full_any_candidate, full_any_value = _best_index(
            full_iou[:, object_index], np.arange(candidate_count, dtype=np.int64)
        )
        s0_class_index = bank.class_names.index(item.class_name)
        s0_gaussian = (
            (bank.semantic_top1 == s0_class_index)
            & (bank.semantic_top1_score >= SEMANTIC_THRESHOLD)
        )
        object_nearest = gt_to_gaussian_indices[item.point_indices]
        object_supported = object_nearest >= 0
        s0_covered = np.zeros(item.point_count, dtype=bool)
        s0_covered[object_supported] = s0_gaussian[object_nearest[object_supported]]
        s0_metrics = _gaussian_metrics(
            s0_gaussian,
            item,
            reverse_semantic,
            reverse_instance,
            reverse_evaluable,
        )
        gaussian_object_count = int(
            np.count_nonzero(
                reverse_evaluable
                & (reverse_semantic == item.class_id)
                & (reverse_instance == item.instance_id)
            )
        )
        s1_intersection_value = (
            float(core_intersection_iou[core_same_candidate, object_index])
            if core_same_candidate is not None
            else 0.0
        )
        s2_own_core_value = (
            float(core_iou[full_same_candidate, object_index])
            if full_same_candidate is not None
            else 0.0
        )
        s0_coverage = float(s0_covered.mean()) if len(s0_covered) else 0.0
        flags = {
            "semantic_unreachable": s0_coverage < 0.25,
            "core_formation_failed": s0_coverage >= 0.25
            and core_same_value < 0.25,
            "full_assignment_helpful": full_same_value - core_same_value >= 0.05,
            "full_assignment_harmful": core_same_value >= 0.25
            and full_same_value <= core_same_value - 0.05,
            "usable_full_candidate": full_same_value >= 0.25,
        }
        status = next(
            (name for name in FUNNEL_STATUS_PRECEDENCE if flags[name]), "unresolved"
        )
        same_full_intersections = (
            full_intersections[same_candidates, object_index]
            if len(same_candidates)
            else np.asarray([], dtype=np.int64)
        )
        same_full_iou = (
            full_iou[same_candidates, object_index]
            if len(same_candidates)
            else np.asarray([], dtype=np.float64)
        )
        row: dict[str, Any] = {
            "scene_id": str(scene_id),
            "physical_scene_id": item.physical_scene_id,
            "gt_class": item.class_name,
            "gt_class_id": item.class_id,
            "gt_instance_id": item.instance_id,
            "gt_point_count": item.point_count,
            "gt_bbox_diagonal_m": item.bbox_diagonal_m,
            "size_bin": item.size_bin,
            "gaussian_object_count_5cm": gaussian_object_count,
            "S0_semantic_coverage": s0_coverage,
            "S0_gaussian_purity": float(s0_metrics["purity"]),
            "S0_gaussian_same_class_precision": float(
                s0_metrics["semantic_precision"]
            ),
            "S0_unsupported_fraction": float(s0_metrics["unsupported_fraction"]),
            "S1_best_same_candidate_id": core_same_candidate,
            "S1_best_same_iou_raw_core": core_same_value,
            "S1_best_same_iou_core_intersection_full": s1_intersection_value,
            "S1_best_any_candidate_id": core_any_candidate,
            "S1_best_any_iou": core_any_value,
            "S2_best_same_candidate_id": full_same_candidate,
            "S2_best_same_iou": full_same_value,
            "S2_best_any_candidate_id": full_any_candidate,
            "S2_best_any_iou": full_any_value,
            "S2_best_same_raw_core_iou": s2_own_core_value,
            "S2_best_same_vs_own_core_delta": full_same_value - s2_own_core_value,
            "stage_best_same_iou_delta": full_same_value - core_same_value,
            "S1_same_recall_025": core_same_value >= 0.25,
            "S1_same_recall_050": core_same_value >= 0.50,
            "S2_same_recall_025": full_same_value >= 0.25,
            "S2_same_recall_050": full_same_value >= 0.50,
            "candidate_split_count_positive_intersection": int(
                np.count_nonzero(same_full_intersections > 0)
            ),
            "candidate_split_count_iou_005": int(
                np.count_nonzero(same_full_iou >= 0.05)
            ),
            "candidate_split_count_positive_intersection_any_class": int(
                np.count_nonzero(full_intersections[:, object_index] > 0)
            ),
            "candidate_split_count_iou_005_any_class": int(
                np.count_nonzero(full_iou[:, object_index] >= 0.05)
            ),
            "funnel_status": status,
        }
        row.update(
            {
                f"funnel_flag_{name}": bool(value)
                for name, value in flags.items()
            }
        )
        gt_rows.append(row)

    analysis = _scene_analysis(
        str(scene_id),
        candidate_rows,
        gt_rows,
        full_iou,
        candidate_class_ids,
        objects,
    )
    return CandidateFunnelSceneResult(
        scene_id=str(scene_id),
        candidate_rows=tuple(candidate_rows),
        gt_rows=tuple(gt_rows),
        analysis=analysis,
    )


def _merge_stage_recall(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not analyses:
        return {}
    names = tuple(analyses[0]["stage_recall"])
    output: dict[str, Any] = {}
    for name in names:
        records = [analysis["stage_recall"][name] for analysis in analyses]
        total = int(sum(int(row["official_valid_gt_count"]) for row in records))
        matched = int(sum(int(row["matched_gt_count"]) for row in records))
        output[name] = {
            "matched_gt_count": matched,
            "official_valid_gt_count": total,
            "recall": _fraction(matched, total),
        }
    return output


def _merge_counterfactual(
    analyses: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not analyses:
        return {}
    layers = tuple(analyses[0]["counterfactual_filters"])
    output: dict[str, Any] = {}
    for layer in layers:
        records = [analysis["counterfactual_filters"][layer] for analysis in analyses]
        retained = int(sum(int(row["retained_candidate_count"]) for row in records))
        positive_025 = int(
            sum(int(row["same_class_iou_025_candidate_count"]) for row in records)
        )
        positive_050 = int(
            sum(int(row["same_class_iou_050_candidate_count"]) for row in records)
        )
        gt_total = int(sum(int(row["official_valid_gt_count"]) for row in records))
        gt_025 = int(sum(int(row["gt_recall_025_count"]) for row in records))
        gt_050 = int(sum(int(row["gt_recall_050_count"]) for row in records))
        tiny_total = int(sum(int(row["tiny_small_gt_count"]) for row in records))
        tiny_025 = int(
            sum(int(row["tiny_small_recall_025_count"]) for row in records)
        )
        tiny_050 = int(
            sum(int(row["tiny_small_recall_050_count"]) for row in records)
        )
        output[layer] = {
            "retained_candidate_count": retained,
            "same_class_iou_025_candidate_count": positive_025,
            "same_class_iou_050_candidate_count": positive_050,
            "candidate_precision_025": _fraction(positive_025, retained),
            "candidate_precision_050": _fraction(positive_050, retained),
            "official_valid_gt_count": gt_total,
            "gt_recall_025_count": gt_025,
            "gt_recall_050_count": gt_050,
            "gt_recall_025": _fraction(gt_025, gt_total),
            "gt_recall_050": _fraction(gt_050, gt_total),
            "tiny_small_gt_count": tiny_total,
            "tiny_small_recall_025_count": tiny_025,
            "tiny_small_recall_050_count": tiny_050,
            "tiny_small_recall_025": _fraction(tiny_025, tiny_total),
            "tiny_small_recall_050": _fraction(tiny_050, tiny_total),
        }
    return output


def _scene_equal_summary(
    results: Sequence[CandidateFunnelSceneResult],
) -> dict[str, Any]:
    def mean(values: Sequence[float]) -> float | None:
        return float(np.mean(values)) if values else None

    stage_names = tuple(results[0].analysis["stage_recall"]) if results else ()
    layer_names = (
        tuple(results[0].analysis["counterfactual_filters"]) if results else ()
    )
    return {
        "candidate_precision_025": mean(
            [
                float(
                    result.analysis["candidate_quality"][
                        "candidate_precision_025"
                    ]
                )
                for result in results
            ]
        ),
        "stage_recall": {
            name: mean(
                [
                    float(result.analysis["stage_recall"][name]["recall"])
                    for result in results
                ]
            )
            for name in stage_names
        },
        "counterfactual_filters": {
            layer: {
                "candidate_precision_025": mean(
                    [
                        float(
                            result.analysis["counterfactual_filters"][layer][
                                "candidate_precision_025"
                            ]
                        )
                        for result in results
                    ]
                ),
                "gt_recall_025": mean(
                    [
                        float(
                            result.analysis["counterfactual_filters"][layer][
                                "gt_recall_025"
                            ]
                        )
                        for result in results
                    ]
                ),
                "tiny_small_recall_025": mean(
                    [
                        float(
                            result.analysis["counterfactual_filters"][layer][
                                "tiny_small_recall_025"
                            ]
                        )
                        for result in results
                    ]
                ),
            }
            for layer in layer_names
        },
    }


def aggregate_candidate_funnel_results(
    results: Sequence[CandidateFunnelSceneResult],
) -> dict[str, Any]:
    """Aggregate scene results without treating candidates as replications."""

    normalized = tuple(results)
    if not normalized:
        raise ValueError("at least one scene result is required")
    scene_ids = [str(result.scene_id) for result in normalized]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("scene results contain duplicate scene IDs")
    candidate_rows = [row for result in normalized for row in result.candidate_rows]
    gt_rows = [row for result in normalized for row in result.gt_rows]
    analyses = [result.analysis for result in normalized]
    violations = sum(
        int(analysis["core_subset_violation_count"]) for analysis in analyses
    )
    status_counts = dict(
        sorted(Counter(str(row["funnel_status"]) for row in gt_rows).items())
    )
    flag_counts = {
        name: int(sum(bool(row[f"funnel_flag_{name}"]) for row in gt_rows))
        for name in FUNNEL_STATUS_PRECEDENCE
    }
    per_class_names = sorted(
        {str(row["branch_class"]) for row in candidate_rows}
        | {str(row["gt_class"]) for row in gt_rows}
    )
    per_class = {
        name: {
            "candidate_quality": _candidate_quality(
                [row for row in candidate_rows if row["branch_class"] == name]
            ),
            "stage_recall": _stage_recall(
                [row for row in gt_rows if row["gt_class"] == name]
            ),
            "funnel_status_counts": dict(
                sorted(
                    Counter(
                        row["funnel_status"]
                        for row in gt_rows
                        if row["gt_class"] == name
                    ).items()
                )
            ),
            "tiny_small": _tiny_small(
                [row for row in gt_rows if row["gt_class"] == name]
            ),
        }
        for name in per_class_names
    }
    return {
        "schema": "saga-category-denoise-funnel-analysis-v1",
        "scene_count": len(normalized),
        "scene_ids": sorted(scene_ids),
        "candidate_count": len(candidate_rows),
        "official_valid_gt_count": len(gt_rows),
        "core_subset_violation_count": violations,
        "core_subset_violation_fraction": _fraction(violations, len(candidate_rows)),
        "stage_recall": _merge_stage_recall(analyses),
        "funnel_status_counts": status_counts,
        "funnel_flag_counts": flag_counts,
        "candidate_quality": _candidate_quality(candidate_rows),
        "score_domain": _score_domain(candidate_rows),
        "counterfactual_filters": _merge_counterfactual(analyses),
        "rank_diagnostics": _rank_diagnostics(candidate_rows),
        "scene_equal": _scene_equal_summary(normalized),
        "per_scene": [
            {
                "scene_id": result.scene_id,
                "candidate_count": result.analysis["candidate_count"],
                "official_valid_gt_count": result.analysis["official_valid_gt_count"],
                "core_subset_violation_count": result.analysis[
                    "core_subset_violation_count"
                ],
                "stage_recall": result.analysis["stage_recall"],
                "candidate_quality": result.analysis["candidate_quality"],
                "counterfactual_filters": result.analysis["counterfactual_filters"],
            }
            for result in sorted(normalized, key=lambda item: item.scene_id)
        ],
        "per_class": per_class,
        "tiny_small": _tiny_small(gt_rows),
        "conclusion_boundary": {
            "S1_scope": "retained_raw_core_only",
            "discarded_raw_hdbscan_cores_recoverable_from_bank": False,
            "status_is_strict_causal_proof": False,
            "candidate_rows_are_independent_replicates": False,
            "independent_experimental_unit": "physical_scene",
            "stage_best_candidates_are_not_guaranteed_identical": True,
            "S0_is_scene_wide_same_class_reachability_not_an_instance_candidate": True,
        },
    }


__all__ = [
    "BidirectionalPointMapping",
    "CandidateFunnelSceneResult",
    "FLOOR_EXP_NEG_12_5",
    "FUNNEL_STATUS_PRECEDENCE",
    "NearestPointMapping",
    "OfficialGroundTruthObject",
    "aggregate_candidate_funnel_results",
    "build_bidirectional_mapping",
    "build_official_gt_objects",
    "diagnose_candidate_funnel_scene",
    "gaussian_to_gt_mapping",
    "gt_to_gaussian_mapping",
]
