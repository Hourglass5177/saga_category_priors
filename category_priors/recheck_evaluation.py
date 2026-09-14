from __future__ import annotations

"""Evaluate whether the 2D recheck branch rescues objects missed by B0.

This module is deliberately evaluation-only.  It consumes finalized predictions
and ground truth masks on the ScanNet point domain; it is not imported by the
candidate builder, 2D reviewer, or replay runtime.
"""

import argparse
import hashlib
import json
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation_strata import EvaluationStrata, load_evaluation_strata
from .evaluator import (
    HISTORICAL_10_OVERLAPS,
    SCANNET_OFFICIAL_OVERLAPS,
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .geometry import pca_sorted_extents_m
from .io import hash_json, load_json, sha256_file, write_json
from .taxonomy import Taxonomy, load_taxonomy


@dataclass(frozen=True)
class GroundTruthObject:
    gt_id: Hashable
    class_name: str
    mask: np.ndarray
    bbox_diagonal_m: float


@dataclass(frozen=True)
class BranchPrediction:
    prediction_id: Hashable
    class_name: str
    score: float
    mask: np.ndarray
    source_candidate_id: int | None = None
    source_candidate_ids: tuple[int, ...] = ()
    # Exact Gaussian-domain membership, rather than the lossy GT projection.
    # Pure unit callers without a separate domain may leave this unset.
    member_sha256: str | None = None


@dataclass(frozen=True)
class IoUMatch:
    prediction_id: Hashable
    gt_id: Hashable
    iou: float


@dataclass(frozen=True)
class OneToOneMatchResult:
    matches: tuple[IoUMatch, ...]
    unmatched_prediction_ids: tuple[Hashable, ...]
    unmatched_gt_ids: tuple[Hashable, ...]


def _stable_key(value: Hashable) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def mask_content_sha256(mask: np.ndarray) -> str:
    value = _validated_mask(mask, owner="content identity")
    digest = hashlib.sha256()
    digest.update(int(len(value)).to_bytes(8, "little"))
    digest.update(np.packbits(value, bitorder="little").tobytes())
    return digest.hexdigest()


def _content_key(row: BranchPrediction | GroundTruthObject) -> tuple[str, str, str]:
    return (
        row.class_name,
        mask_content_sha256(row.mask),
        str(getattr(row, "member_sha256", None) or ""),
    )


def _member_key(row: BranchPrediction) -> tuple[str, str]:
    return row.class_name, row.member_sha256 or mask_content_sha256(row.mask)


def _nonnegative_id(value: Any) -> int:
    if isinstance(value, bool) or not (
        isinstance(value, (int, np.integer))
        or isinstance(value, str) and value.isascii() and value.isdigit()
    ):
        raise ValueError(f"instance and candidate IDs must be nonnegative integers: {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError("instance and candidate IDs must be nonnegative integers")
    return result


def _validated_mask(value: Any, *, owner: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 1:
        raise ValueError(f"{owner}: mask must be one-dimensional")
    return mask


def _intersection_over_union(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return float(intersection / union) if union else 0.0


def match_one_to_one(
    predictions: Sequence[BranchPrediction],
    ground_truth: Sequence[GroundTruthObject],
    iou_threshold: float,
) -> OneToOneMatchResult:
    """Maximum-cardinality, then maximum-total-IoU same-class matching.

    The strict ``>`` comparison intentionally follows the repository's ScanNet
    evaluator. Content ordering makes exact ties invariant to label renumbering.
    Exact duplicate masks are interchangeable; IDs break only those residual ties.
    """

    threshold = float(iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    ordered_predictions = sorted(
        predictions, key=lambda row: (_content_key(row), _stable_key(row.prediction_id))
    )
    ordered_gt = sorted(
        ground_truth, key=lambda row: (_content_key(row), _stable_key(row.gt_id))
    )
    prediction_ids = [row.prediction_id for row in ordered_predictions]
    gt_ids = [row.gt_id for row in ordered_gt]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError("prediction ids must be unique")
    if len(set(gt_ids)) != len(gt_ids):
        raise ValueError("ground-truth ids must be unique")

    masks: list[np.ndarray] = []
    for row in ordered_predictions:
        masks.append(_validated_mask(row.mask, owner=f"prediction {row.prediction_id}"))
    gt_masks: list[np.ndarray] = []
    for row in ordered_gt:
        gt_masks.append(_validated_mask(row.mask, owner=f"GT {row.gt_id}"))
    lengths = {len(mask) for mask in (*masks, *gt_masks)}
    if len(lengths) > 1:
        raise ValueError("prediction and ground-truth masks must share a point domain")
    if not ordered_predictions or not ordered_gt:
        return OneToOneMatchResult(
            matches=(),
            unmatched_prediction_ids=tuple(prediction_ids),
            unmatched_gt_ids=tuple(gt_ids),
        )

    n_prediction = len(ordered_predictions)
    n_gt = len(ordered_gt)
    size = n_prediction + n_gt
    # Dummy rows/columns have zero profit.  Invalid real-real pairs are worse
    # than a dummy match.  The cardinality bonus dominates any possible sum of
    # IoUs, so the assignment first maximizes match count and only then overlap.
    profit = np.zeros((size, size), dtype=np.float64)
    profit[:n_prediction, :n_gt] = -1.0e6
    valid = np.zeros((n_prediction, n_gt), dtype=bool)
    ious = np.zeros((n_prediction, n_gt), dtype=np.float64)
    cardinality_bonus = float(max(n_prediction, n_gt) + 1)
    for prediction_index, prediction in enumerate(ordered_predictions):
        for gt_index, gt_object in enumerate(ordered_gt):
            if prediction.class_name != gt_object.class_name:
                continue
            iou = _intersection_over_union(masks[prediction_index], gt_masks[gt_index])
            ious[prediction_index, gt_index] = iou
            if iou > threshold:
                valid[prediction_index, gt_index] = True
                # Do not perturb IoU: an epsilon can change a non-tied optimum.
                # The solver receives a deterministic content-ordered matrix.
                profit[prediction_index, gt_index] = cardinality_bonus + iou

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("one-to-one matching requires scipy") from exc
    row_indices, column_indices = linear_sum_assignment(-profit)
    matches = [
        IoUMatch(
            prediction_id=ordered_predictions[row].prediction_id,
            gt_id=ordered_gt[column].gt_id,
            iou=float(ious[row, column]),
        )
        for row, column in zip(row_indices.tolist(), column_indices.tolist())
        if row < n_prediction and column < n_gt and valid[row, column]
    ]
    matches.sort(
        key=lambda row: (_stable_key(row.prediction_id), _stable_key(row.gt_id))
    )
    matched_predictions = {row.prediction_id for row in matches}
    matched_gt = {row.gt_id for row in matches}
    return OneToOneMatchResult(
        matches=tuple(matches),
        unmatched_prediction_ids=tuple(
            value for value in prediction_ids if value not in matched_predictions
        ),
        unmatched_gt_ids=tuple(value for value in gt_ids if value not in matched_gt),
    )


def _has_match(
    prediction: BranchPrediction,
    gt_objects: Sequence[GroundTruthObject],
    threshold: float,
) -> bool:
    prediction_mask = _validated_mask(
        prediction.mask, owner=f"prediction {prediction.prediction_id}"
    )
    return any(
        prediction.class_name == gt_object.class_name
        and _intersection_over_union(
            prediction_mask,
            _validated_mask(gt_object.mask, owner=f"GT {gt_object.gt_id}"),
        )
        > threshold
        for gt_object in gt_objects
    )


def _metric_values(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    denominator = 2 * tp + fp + fn
    f1 = float(2 * tp / denominator) if denominator else None
    f05_denominator = 1.25 * tp + fp + 0.25 * fn
    f05 = float(1.25 * tp / f05_denominator) if f05_denominator else None
    return {"precision": precision, "recall": recall, "f1": f1, "f0_5": f05}


def _evaluate_stratum(
    *,
    ground_truth: Sequence[GroundTruthObject],
    b0_hit_ids: set[Hashable],
    predictions: Sequence[BranchPrediction],
    threshold: float,
    include_gt: Any,
    include_prediction: Any,
) -> dict[str, Any]:
    target_gt = [
        row for row in ground_truth if row.gt_id not in b0_hit_ids and include_gt(row)
    ]
    included_predictions = [row for row in predictions if include_prediction(row)]
    excluded_gt = [row for row in ground_truth if not include_gt(row)]
    b0_hit_target = [
        row for row in ground_truth if row.gt_id in b0_hit_ids and include_gt(row)
    ]
    result = match_one_to_one(included_predictions, target_gt, threshold)
    by_id = {row.prediction_id: row for row in included_predictions}
    unmatched_predictions = [by_id[value] for value in result.unmatched_prediction_ids]
    # In a size-restricted readout, a correct prediction of one excluded
    # medium/large object is ignored exactly once.  A second duplicate remains
    # a false positive; a mere "has any overlap" test would incorrectly ignore
    # every duplicate.
    outside_match = match_one_to_one(unmatched_predictions, excluded_gt, threshold)
    ignored_outside_ids = {row.prediction_id for row in outside_match.matches}
    duplicate_b0 = 0
    duplicate_branch = 0
    ignored_outside_class = len(predictions) - len(included_predictions)
    ignored_correct_outside_stratum = len(ignored_outside_ids)
    hallucination = 0
    other_fp = 0
    rescued_gt = [
        row
        for row in target_gt
        if row.gt_id in {match.gt_id for match in result.matches}
    ]
    for prediction_id in result.unmatched_prediction_ids:
        if prediction_id in ignored_outside_ids:
            continue
        prediction = by_id[prediction_id]
        if _has_match(prediction, b0_hit_target, threshold):
            duplicate_b0 += 1
        elif _has_match(prediction, rescued_gt, threshold):
            duplicate_branch += 1
        elif not _has_match(prediction, ground_truth, threshold):
            hallucination += 1
        else:
            other_fp += 1
    tp = len(result.matches)
    fp = duplicate_b0 + duplicate_branch + hallucination + other_fp
    fn = len(result.unmatched_gt_ids)
    ignored = ignored_outside_class + ignored_correct_outside_stratum
    metrics: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored": ignored,
        "duplicate_b0": duplicate_b0,
        "duplicate_missed": duplicate_branch,
        "duplicate_branch": duplicate_branch,
        "hallucination": hallucination,
        "other_fp": other_fp,
        "b0_hit_gt": len(b0_hit_target),
        "missed_gt": len(target_gt),
        "eligible_predictions": len(included_predictions),
        "target_missed_gt_count": len(target_gt),
        "traceable_prediction_count": len(predictions),
        "ignored_outside_class_count": ignored_outside_class,
        "ignored_correct_outside_stratum_count": ignored_correct_outside_stratum,
        "duplicate_b0_count": duplicate_b0,
        "duplicate_branch_count": duplicate_branch,
        "hallucination_count": hallucination,
        "other_fp_count": other_fp,
        **_metric_values(tp, fp, fn),
        "matches": [asdict(row) for row in result.matches],
    }
    return metrics


def evaluate_rescue_scene(
    *,
    scene_id: str,
    b0_predictions: Sequence[BranchPrediction],
    branch_predictions: Sequence[BranchPrediction],
    ground_truth: Sequence[GroundTruthObject],
    strata: EvaluationStrata,
    iou_threshold: float,
) -> dict[str, Any]:
    """Evaluate one scene and one IoU threshold under four frozen strata."""

    if any(
        not row.source_candidate_ids and row.source_candidate_id is None
        for row in branch_predictions
    ):
        raise ValueError("every branch prediction must have a source candidate id")

    b0_match = match_one_to_one(b0_predictions, ground_truth, iou_threshold)
    b0_hit_ids = {row.gt_id for row in b0_match.matches}
    is_small = lambda row: strata.is_small(row.bbox_diagonal_m)
    is_tail = lambda row: strata.is_tail(row.class_name)
    predicates = {
        "overall": (lambda row: True, lambda row: True),
        "small": (is_small, lambda row: True),
        "tail": (is_tail, lambda row: is_tail(row)),
        "small_tail": (
            lambda row: is_small(row) and is_tail(row),
            lambda row: is_tail(row),
        ),
    }
    return {
        "scene_id": str(scene_id),
        "iou_threshold": float(iou_threshold),
        "b0_matches": [asdict(row) for row in b0_match.matches],
        "b0_detected_gt_ids": sorted(b0_hit_ids, key=_stable_key),
        "missed_gt_ids": sorted(
            (row.gt_id for row in ground_truth if row.gt_id not in b0_hit_ids),
            key=_stable_key,
        ),
        "strata": {
            name: _evaluate_stratum(
                ground_truth=ground_truth,
                b0_hit_ids=b0_hit_ids,
                predictions=branch_predictions,
                threshold=float(iou_threshold),
                include_gt=gt_predicate,
                include_prediction=prediction_predicate,
            )
            for name, (gt_predicate, prediction_predicate) in predicates.items()
        },
    }


def aggregate_rescue_scenes(
    scene_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool counts and also average scene metrics with equal scene weight."""

    if not scene_results:
        raise ValueError("at least one scene result is required")
    thresholds = {float(row["iou_threshold"]) for row in scene_results}
    if len(thresholds) != 1:
        raise ValueError("scene results must use one IoU threshold")
    output: dict[str, Any] = {
        "iou_threshold": next(iter(thresholds)),
        "scene_count": len(scene_results),
        "strata": {},
    }
    count_names = (
        "tp",
        "fp",
        "fn",
        "ignored",
        "duplicate_b0",
        "duplicate_missed",
        "duplicate_branch",
        "hallucination",
        "other_fp",
        "b0_hit_gt",
        "missed_gt",
        "eligible_predictions",
    )
    for name in ("overall", "small", "tail", "small_tail"):
        rows = [row["strata"][name] for row in scene_results]
        pooled = {key: int(sum(int(row[key]) for row in rows)) for key in count_names}
        pooled_metrics = _metric_values(pooled["tp"], pooled["fp"], pooled["fn"])
        scene_equal = {}
        defined_scene_count = {}
        for key in ("precision", "recall", "f1", "f0_5"):
            values = [float(row[key]) for row in rows if row[key] is not None]
            scene_equal[key] = float(np.mean(values)) if values else None
            defined_scene_count[key] = len(values)
        output["strata"][name] = {
            "pooled_counts": pooled,
            "pooled_metrics": pooled_metrics,
            "scene_equal_mean": scene_equal,
            "defined_scene_count": defined_scene_count,
        }
    return output


def ground_truth_objects(
    scene: GroundTruthScene,
    xyz: np.ndarray,
    class_names: Sequence[str],
    min_region_size: int = 100,
) -> tuple[GroundTruthObject, ...]:
    coords = np.asarray(xyz, dtype=np.float64)
    if coords.shape != (len(scene.semantic), 3):
        raise ValueError(f"{scene.scene_id}: GT XYZ and labels do not align")
    result: list[GroundTruthObject] = []
    for class_id, class_name in enumerate(class_names):
        class_mask = scene.semantic == class_id
        for instance_id in sorted(
            int(value) for value in np.unique(scene.instance[class_mask])
        ):
            if instance_id < 0:
                continue
            mask = class_mask & (scene.instance == instance_id)
            if int(np.count_nonzero(mask)) < int(min_region_size):
                continue
            points = coords[mask]
            diagonal = float(np.linalg.norm(pca_sorted_extents_m(points, 1.0)))
            result.append(
                GroundTruthObject(
                    gt_id=f"{class_id}:{instance_id}",
                    class_name=str(class_name),
                    mask=mask,
                    bbox_diagonal_m=diagonal,
                )
            )
    return tuple(result)


def evaluate_scene_reconciliation(
    *,
    scene_id: str,
    b0_predictions: Sequence[BranchPrediction],
    final_predictions: Sequence[BranchPrediction],
    ground_truth: Sequence[GroundTruthObject],
    strata: EvaluationStrata,
    iou_threshold: float,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Account for every final prediction independently of candidate lineage.

    This is an object readout, not a replacement for the ScanNet AP evaluator.
    Small exported fragments remain present and are counted separately. Final
    ownership is matched to all GT once, then sliced into frozen strata; a
    correct replacement of a B0 hit is retained, never a duplicate merely
    because its candidate source is new.
    """
    b0_match = match_one_to_one(b0_predictions, ground_truth, iou_threshold)
    final_match = match_one_to_one(final_predictions, ground_truth, iou_threshold)
    b0_hit = {row.gt_id for row in b0_match.matches}
    final_hit = {row.gt_id for row in final_match.matches}
    b0_hit_prediction = {row.prediction_id for row in b0_match.matches}
    final_by_id = {row.prediction_id: row for row in final_predictions}
    gt_by_id = {row.gt_id: row for row in ground_truth}

    # Match exact unchanged geometry one-to-one, not by export ID or GT-domain
    # projection. One unchanged mask cannot exempt several duplicate outputs.
    available_b0: dict[tuple[str, str], list[BranchPrediction]] = {}
    for row in sorted(b0_predictions, key=lambda row: _stable_key(row.prediction_id)):
        available_b0.setdefault(_member_key(row), []).append(row)
    unchanged: dict[Hashable, Hashable] = {}
    for row in sorted(final_predictions, key=lambda row: (_content_key(row), _stable_key(row.prediction_id))):
        group = available_b0.get(_member_key(row), [])
        if group:
            unchanged[row.prediction_id] = group.pop(0).prediction_id

    unmatched = set(final_match.unmatched_prediction_ids)
    inherited = {
        value for value in unmatched
        if value in unchanged and unchanged[value] not in b0_hit_prediction
    }
    displaced_unchanged = {
        value for value in unmatched
        if value in unchanged and unchanged[value] in b0_hit_prediction
    }
    new_fp = unmatched - set(unchanged)
    hit_gt = [gt_by_id[value] for value in final_hit]
    duplicates = {
        value for value in unmatched
        if _has_match(final_by_id[value], hit_gt, iou_threshold)
    }
    rescued = final_hit - b0_hit
    retained = final_hit & b0_hit
    lost = b0_hit - final_hit
    missed = set(gt_by_id) - b0_hit

    gt_predicates = {
        "overall": lambda row: True,
        "small": lambda row: strata.is_small(row.bbox_diagonal_m),
        "tail": lambda row: strata.is_tail(row.class_name),
        "small_tail": lambda row: strata.is_small(row.bbox_diagonal_m) and strata.is_tail(row.class_name),
    }
    readouts: dict[str, Any] = {}
    for name, include_gt in gt_predicates.items():
        target = {row.gt_id for row in ground_truth if include_gt(row)}
        # Size is a GT property. Never exclude an FP by its predicted geometry.
        # Tail remains class-restricted; all-FP counts are reported alongside.
        include_prediction = (
            (lambda row: strata.is_tail(row.class_name))
            if name in {"tail", "small_tail"} else (lambda row: True)
        )
        eligible = {row.prediction_id for row in final_predictions if include_prediction(row)}
        count_rescue = len(rescued & target)
        count_fp = len(new_fp & eligible)
        count_fn = len((missed - rescued) & target)
        readouts[name] = {
            "rescued": count_rescue,
            "retained": len(retained & target),
            "lost": len(lost & target),
            "b0_hit_gt": len(b0_hit & target),
            "b0_missed_gt": len(missed & target),
            "final_hit_gt": len(final_hit & target),
            "gt_count": len(target),
            "new_false_positives": count_fp,
            "all_new_false_positives": len(new_fp),
            "inherited_false_positives": len(inherited & eligible),
            "unchanged_b0_displaced_predictions": len(displaced_unchanged & eligible),
            "duplicate_predictions": len(duplicates & eligible),
            "new_duplicate_predictions": len(duplicates & new_fp & eligible),
            "final_false_positives": len(unmatched & eligible),
            "all_final_false_positives": len(unmatched),
            "final_prediction_count": len(eligible),
            "all_final_prediction_count": len(final_predictions),
            "below_ap_min_region_predictions": sum(
                int(np.count_nonzero(row.mask)) < int(min_region_size)
                for row in final_predictions if include_prediction(row)
            ),
            "tp": count_rescue,
            "fp": count_fp,
            "fn": count_fn,
            **_metric_values(count_rescue, count_fp, count_fn),
        }
    return {
        "scene_id": str(scene_id),
        "iou_threshold": float(iou_threshold),
        "prediction_coverage": "all_final_exports",
        "unchanged_identity_domain": (
            "gaussian" if all(row.member_sha256 is not None for row in (*b0_predictions, *final_predictions))
            else "provided_mask_domain"
        ),
        "matching": "same_class_max_cardinality_then_total_iou_strict_threshold_content_ties",
        "min_region_size_for_ap_only": int(min_region_size),
        "b0_matches": [asdict(row) for row in b0_match.matches],
        "final_matches": [asdict(row) for row in final_match.matches],
        "rescued_gt_ids": sorted(rescued, key=_stable_key),
        "retained_gt_ids": sorted(retained, key=_stable_key),
        "lost_gt_ids": sorted(lost, key=_stable_key),
        "new_false_positive_ids": sorted(new_fp, key=_stable_key),
        "inherited_false_positive_ids": sorted(inherited, key=_stable_key),
        "unchanged_b0_displaced_prediction_ids": sorted(displaced_unchanged, key=_stable_key),
        "duplicate_prediction_ids": sorted(duplicates, key=_stable_key),
        "unchanged_prediction_pairs": [
            {"final_export_id": value, "b0_export_id": unchanged[value]}
            for value in sorted(unchanged, key=_stable_key)
        ],
        "strata": readouts,
    }


def aggregate_reconciliation_scenes(
    scene_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not scene_results:
        raise ValueError("at least one scene result is required")
    thresholds = {float(row["iou_threshold"]) for row in scene_results}
    scene_ids = [str(row["scene_id"]) for row in scene_results]
    if len(thresholds) != 1 or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("reconciliation requires unique scenes at one IoU threshold")
    result: dict[str, Any] = {
        "iou_threshold": next(iter(thresholds)), "scene_count": len(scene_results), "strata": {},
    }
    metric_names = ("precision", "recall", "f1", "f0_5")
    for name in ("overall", "small", "tail", "small_tail"):
        rows = [scene["strata"][name] for scene in scene_results]
        counts = {
            key: sum(int(row[key]) for row in rows)
            for key in rows[0] if key not in metric_names
        }
        values = {key: [float(row[key]) for row in rows if row[key] is not None] for key in metric_names}
        result["strata"][name] = {
            "pooled_counts": counts,
            "pooled_metrics": _metric_values(counts["tp"], counts["fp"], counts["fn"]),
            "scene_equal_mean": {key: float(np.mean(items)) if items else None for key, items in values.items()},
            "defined_scene_count": {key: len(items) for key, items in values.items()},
        }
    return result


def _branch_rows(
    predictions: Sequence[PredictedInstance],
    candidate_export_ids: Mapping[str, Any],
    class_names: Sequence[str],
    *,
    candidate_export_lineage: Mapping[str, Any] | None = None,
) -> tuple[BranchPrediction, ...]:
    """Build legacy rescue rows from complete export-indexed lineage only."""
    if candidate_export_lineage is None:
        raise ValueError("incomplete lineage: legacy parent-to-export map cannot prove split coverage")
    export_to_parents: dict[int, tuple[int, ...]] = {}
    for raw_export, raw_parents in candidate_export_lineage.items():
        export = _nonnegative_id(raw_export)
        if export < 0 or export in export_to_parents:
            raise ValueError("lineage export IDs must be unique nonnegative integers")
        if not isinstance(raw_parents, (list, tuple)) or not raw_parents:
            raise ValueError(f"export {export}: lineage requires a nonempty parent list")
        parents = tuple(sorted({_nonnegative_id(value) for value in raw_parents}))
        if min(parents) < 0:
            raise ValueError("candidate IDs must be nonnegative")
        export_to_parents[export] = parents
    by_export = {int(row.instance_id): row for row in predictions}
    if len(by_export) != len(predictions):
        raise ValueError("prediction export IDs must be unique")
    missing = sorted(set(export_to_parents) - set(by_export))
    if missing:
        raise ValueError(
            f"candidate source chain references missing exports: {missing}"
        )
    return tuple(
        BranchPrediction(
            prediction_id=int(export_id),
            class_name=str(class_names[row.class_id]),
            score=float(row.score),
            mask=np.asarray(row.mask, dtype=bool),
            source_candidate_id=export_to_parents[export_id][0],
            source_candidate_ids=export_to_parents[export_id],
        )
        for export_id, row in sorted(by_export.items())
        if export_id in export_to_parents
    )


def _prediction_rows(
    predictions: Sequence[PredictedInstance], class_names: Sequence[str],
    *, gaussian_labels: np.ndarray | None = None,
) -> tuple[BranchPrediction, ...]:
    return tuple(
        BranchPrediction(
            prediction_id=int(row.instance_id),
            class_name=str(class_names[row.class_id]),
            score=float(row.score),
            mask=np.asarray(row.mask, dtype=bool),
            member_sha256=(
                mask_content_sha256(np.asarray(gaussian_labels) == int(row.instance_id))
                if gaussian_labels is not None else None
            ),
        )
        for row in predictions
    )


def audit_export_lineage(
    predictions: Sequence[PredictedInstance],
    output_payload: Mapping[str, Any],
    class_names: Sequence[str],
) -> tuple[tuple[BranchPrediction, ...], dict[str, Any]]:
    """Recover historical full lineage; suppress rescue if provenance is partial.

    Historical refinement already saved lossless ``candidate_export_lineage``.
    Its lossy canonical-parent map is checked as a subset only. New v2 outputs
    additionally prove explicit export inventory and complete inverse coverage.
    """
    lineage = output_payload.get("candidate_export_lineage")
    inverse = output_payload.get("candidate_export_ids", {})
    schema = output_payload.get("candidate_export_contract_schema")
    declared = output_payload.get("refined_export_ids")
    audit: dict[str, Any] = {
        "status": "incomplete", "complete": False,
        "source": "full_export_lineage" if lineage is not None else "legacy_single_value_map",
        "explicit_refined_inventory": declared is not None,
        "final_prediction_count": len(predictions),
        "traceable_prediction_count": None,
        "reason": None,
    }
    try:
        if schema not in (None, "saga-candidate-export-lineage-v2"):
            raise ValueError(f"unsupported candidate lineage schema {schema!r}")
        if lineage is not None and not isinstance(lineage, Mapping):
            raise ValueError("export lineage must be a mapping")
        if not isinstance(inverse, Mapping):
            raise ValueError("candidate export inverse must be a mapping")
        rows = _branch_rows(predictions, inverse, class_names, candidate_export_lineage=lineage)
        parents_by_export = {int(row.prediction_id): set(row.source_candidate_ids) for row in rows}
        if schema is not None and declared is None:
            raise ValueError("v2 lineage requires refined_export_ids")
        if declared is not None:
            if not isinstance(declared, (list, tuple)):
                raise ValueError("refined_export_ids must be a list")
            inventory = [_nonnegative_id(value) for value in declared]
            if len(set(inventory)) != len(inventory) or set(inventory) != set(parents_by_export):
                raise ValueError("refined export inventory differs from full lineage")
        inverse_pairs = set()
        for parent, exports in inverse.items():
            if schema is not None and not isinstance(exports, (list, tuple)):
                raise ValueError("v2 candidate inverse values must be export lists")
            exports = exports if isinstance(exports, (list, tuple)) else [exports]
            for export in exports:
                pair = _nonnegative_id(parent), _nonnegative_id(export)
                if pair[0] not in parents_by_export.get(pair[1], set()):
                    raise ValueError("candidate inverse contradicts full export lineage")
                inverse_pairs.add(pair)
        full_pairs = {(parent, export) for export, parents in parents_by_export.items() for parent in parents}
        if schema is not None and inverse_pairs != full_pairs:
            raise ValueError("v2 inverse does not cover all parent/export relationships")
    except (ValueError, TypeError, OverflowError) as exc:
        audit["reason"] = str(exc)
        return (), audit
    audit.update({
        "status": "complete", "complete": True,
        "traceable_prediction_count": len(rows),
        "refined_export_ids": sorted(parents_by_export),
        "export_sources": {
            str(export): sorted(parents_by_export[export]) for export in sorted(parents_by_export)
        },
        "coverage_basis": "explicit_v2_inventory" if schema is not None else "historical_lossless_export_field",
    })
    return rows, audit


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def evaluate_recheck_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path,
    taxonomy: Taxonomy,
    strata: EvaluationStrata,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate B0-missed rescue and auxiliary official AP for a manifest."""

    manifest_source = Path(manifest_path)
    manifest = load_json(manifest_source)
    if manifest.get("schema") not in (
        "saga-instance-recheck-evaluation-manifest-v1",
        "saga-instance-recheck-evaluation-manifest-v2",
    ):
        raise ValueError("unsupported recheck evaluation manifest schema")
    base = manifest_source.parent
    condition_names = tuple(str(value) for value in manifest["conditions"])
    if not condition_names:
        raise ValueError("manifest must declare at least one condition")
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("condition names must be unique")
    scene_ids = [str(item["scene_id"]) for item in manifest["scenes"]]
    if not scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("manifest must contain unique scenes")
    source_files = {manifest_source.resolve()}
    for item in manifest["scenes"]:
        source_files.update(_resolve(base, item[key]).resolve() for key in ("gt_npz", "gaussian_ply", "b0_output_json"))
        for spec in item["condition_outputs"].values():
            source_files.add(_resolve(base, spec["output_json"] if isinstance(spec, dict) else spec).resolve())
    if Path(output_path).resolve() in source_files:
        raise ValueError("evaluation output must not overwrite any input artifact")
    input_hashes = {str(path): sha256_file(path) for path in sorted(source_files)}
    per_condition_scene: dict[str, list[dict[str, Any]]] = {
        name: [] for name in condition_names
    }
    per_condition_reconciliation: dict[str, list[dict[str, Any]]] = {
        name: [] for name in condition_names
    }
    incomplete_scenes: dict[str, list[str]] = {name: [] for name in condition_names}
    gt_scenes: list[GroundTruthScene] = []
    predictions_by_condition: dict[str, list[PredictedInstance]] = {
        name: [] for name in condition_names
    }
    all_b0_predictions: list[PredictedInstance] = []
    diagnostics: dict[str, Any] = {}

    for scene_item in manifest["scenes"]:
        scene_id = str(scene_item["scene_id"])
        gt_xyz, gt_scene = load_ground_truth_npz(
            _resolve(base, scene_item["gt_npz"]), scene_id
        )
        gt_scenes.append(gt_scene)
        objects = ground_truth_objects(
            gt_scene, gt_xyz, taxonomy.canonical_classes, min_region_size
        )
        transform = scene_item["gaussian_to_gt_transform"]
        gaussian_ply = _resolve(base, scene_item["gaussian_ply"])
        b0_predictions, b0_diagnostics = saga_scene_predictions(
            scene_id=scene_id,
            gt_coords=gt_xyz,
            output_json=_resolve(base, scene_item["b0_output_json"]),
            gaussian_ply=gaussian_ply,
            taxonomy=taxonomy,
            metadata_json=None,
            transform=transform,
            radius_m=radius_m,
            require_scores=True,
        )
        if float(b0_diagnostics["mapped_fraction"]) < float(
            manifest.get("minimum_mapped_fraction", 0.90)
        ):
            raise ValueError(f"{scene_id}: coordinate alignment gate failed")
        all_b0_predictions.extend(b0_predictions)
        b0_payload = load_json(_resolve(base, scene_item["b0_output_json"]))
        b0_declared = {_nonnegative_id(value) for value in b0_payload["instances"]}
        if b0_declared != {row.instance_id for row in b0_predictions}:
            raise ValueError(f"{scene_id}: B0 has declared exports outside the evaluation taxonomy")
        b0_labels = np.asarray(b0_payload["point_labels"], dtype=np.int64)
        b0_rows = _prediction_rows(
            b0_predictions, taxonomy.canonical_classes,
            gaussian_labels=b0_labels,
        )
        diagnostics[scene_id] = {"b0": b0_diagnostics, "conditions": {}}
        condition_specs = scene_item["condition_outputs"]
        for condition in condition_names:
            spec = condition_specs[condition]
            output_value = spec["output_json"] if isinstance(spec, dict) else spec
            output_path_value = _resolve(base, output_value)
            predictions, condition_diagnostics = saga_scene_predictions(
                scene_id=scene_id,
                gt_coords=gt_xyz,
                output_json=output_path_value,
                gaussian_ply=gaussian_ply,
                taxonomy=taxonomy,
                metadata_json=None,
                transform=transform,
                radius_m=radius_m,
                require_scores=True,
            )
            output_payload = load_json(output_path_value)
            declared = {_nonnegative_id(value) for value in output_payload["instances"]}
            if declared != {row.instance_id for row in predictions}:
                raise ValueError(f"{scene_id}/{condition}: declared exports were omitted by the AP input adapter")
            final_labels = np.asarray(output_payload["point_labels"], dtype=np.int64)
            if final_labels.shape != b0_labels.shape:
                raise ValueError(f"{scene_id}/{condition}: B0 and final Gaussian domains differ")
            final_rows = _prediction_rows(
                predictions, taxonomy.canonical_classes,
                gaussian_labels=final_labels,
            )
            branch, lineage_audit = audit_export_lineage(
                predictions, output_payload, taxonomy.canonical_classes,
            )
            if not lineage_audit["complete"]:
                incomplete_scenes[condition].append(scene_id)
            predictions_by_condition[condition].extend(predictions)
            diagnostics[scene_id]["conditions"][condition] = {
                **condition_diagnostics,
                "traceable_branch_predictions": lineage_audit["traceable_prediction_count"],
                "lineage": lineage_audit,
                "below_ap_min_region_predictions": sum(
                    int(np.count_nonzero(row.mask)) < int(min_region_size) for row in predictions
                ),
                "mapping_direction": "GT_points_to_nearest_Gaussian_within_radius",
            }
            for threshold in (0.25, 0.50):
                if lineage_audit["complete"]:
                    per_condition_scene[condition].append(evaluate_rescue_scene(
                        scene_id=scene_id,
                        b0_predictions=b0_rows,
                        branch_predictions=branch,
                        ground_truth=objects,
                        strata=strata,
                        iou_threshold=threshold,
                    ))
                per_condition_reconciliation[condition].append(evaluate_scene_reconciliation(
                    scene_id=scene_id, b0_predictions=b0_rows,
                    final_predictions=final_rows, ground_truth=objects, strata=strata,
                    iou_threshold=threshold, min_region_size=min_region_size,
                ))

    conditions: dict[str, Any] = {}
    for condition in condition_names:
        rows = per_condition_scene[condition]
        rescue = None if incomplete_scenes[condition] else {
            f"iou_{threshold:.2f}": {
                "aggregate": aggregate_rescue_scenes(
                    [
                        row
                        for row in rows
                        if math.isclose(row["iou_threshold"], threshold)
                    ]
                ),
                "per_scene": [
                    row for row in rows if math.isclose(row["iou_threshold"], threshold)
                ],
            }
            for threshold in (0.25, 0.50)
        }
        official = evaluate_instances(
            gt_scenes,
            predictions_by_condition[condition],
            taxonomy.canonical_classes,
            overlaps=SCANNET_OFFICIAL_OVERLAPS,
            min_region_size=min_region_size,
        )
        historical = evaluate_instances(
            gt_scenes,
            predictions_by_condition[condition],
            taxonomy.canonical_classes,
            overlaps=HISTORICAL_10_OVERLAPS,
            min_region_size=min_region_size,
        )
        conditions[condition] = {
            "rescue": rescue,
            "rescue_status": "incomplete" if incomplete_scenes[condition] else "complete",
            "incomplete_lineage_scene_ids": incomplete_scenes[condition],
            "reconciliation": {
                f"iou_{threshold:.2f}": {
                    "aggregate": aggregate_reconciliation_scenes([
                        row for row in per_condition_reconciliation[condition]
                        if math.isclose(row["iou_threshold"], threshold)
                    ]),
                    "per_scene": [
                        row for row in per_condition_reconciliation[condition]
                        if math.isclose(row["iou_threshold"], threshold)
                    ],
                } for threshold in (0.25, 0.50)
            },
            "official_9": official,
            "historical_10": historical,
        }

    b0_official = evaluate_instances(
        gt_scenes,
        all_b0_predictions,
        taxonomy.canonical_classes,
        overlaps=SCANNET_OFFICIAL_OVERLAPS,
        min_region_size=min_region_size,
    )
    b0_historical = evaluate_instances(
        gt_scenes,
        all_b0_predictions,
        taxonomy.canonical_classes,
        overlaps=HISTORICAL_10_OVERLAPS,
        min_region_size=min_region_size,
    )
    result: dict[str, Any] = {
        "schema": "saga-instance-recheck-evaluation-v2",
        "b0": {
            "official_9": b0_official,
            "historical_10": b0_historical,
        },
        "conditions": conditions,
        "diagnostics": diagnostics,
        "strata": {
            "small_diagonal_threshold_m": strata.small_diagonal_threshold_m,
            "tail_classes": list(strata.tail_classes),
        },
        "provenance": {
            "manifest_sha256": sha256_file(manifest_source),
            "taxonomy_sha256": taxonomy.content_hash,
            "radius_m": float(radius_m),
            "min_region_size": int(min_region_size),
            "input_sha256": input_hashes,
        },
    }
    if any(sha256_file(path) != digest for path, digest in input_hashes.items()):
        raise ValueError("evaluation input changed while reading; no result was written")
    result["content_sha256"] = hash_json(result)
    write_json(output_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate B0-missed object rescue for the 3D-to-2D recheck experiment"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--taxonomy")
    parser.add_argument("--strata")
    parser.add_argument("--radius-m", type=float, default=0.05)
    parser.add_argument("--min-region-size", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    taxonomy = load_taxonomy(args.taxonomy)
    strata = (
        load_evaluation_strata(args.strata) if args.strata else load_evaluation_strata()
    )
    result = evaluate_recheck_manifest(
        args.manifest,
        output_path=args.output,
        taxonomy=taxonomy,
        strata=strata,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "conditions": list(result["conditions"]),
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )


__all__ = [
    "BranchPrediction",
    "GroundTruthObject",
    "IoUMatch",
    "OneToOneMatchResult",
    "aggregate_rescue_scenes",
    "aggregate_reconciliation_scenes",
    "audit_export_lineage",
    "evaluate_recheck_manifest",
    "evaluate_rescue_scene",
    "evaluate_scene_reconciliation",
    "ground_truth_objects",
    "match_one_to_one",
    "mask_content_sha256",
]


if __name__ == "__main__":
    main()
