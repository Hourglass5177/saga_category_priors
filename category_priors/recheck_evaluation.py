from __future__ import annotations

"""Evaluate whether the 2D recheck branch rescues objects missed by B0.

This module is deliberately evaluation-only.  It consumes finalized predictions
and ground truth masks on the ScanNet point domain; it is not imported by the
candidate builder, 2D reviewer, or replay runtime.
"""

import argparse
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
    evaluator.  Stable identifiers make exact ties deterministic.
    """

    threshold = float(iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU threshold must be in [0, 1]")
    ordered_predictions = sorted(
        predictions, key=lambda row: _stable_key(row.prediction_id)
    )
    ordered_gt = sorted(ground_truth, key=lambda row: _stable_key(row.gt_id))
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
                # The perturbation is far below any reported precision and is
                # used solely to resolve mathematically exact assignment ties.
                pair_rank = prediction_index * n_gt + gt_index
                tie = 1.0e-12 / float(pair_rank + 1)
                profit[prediction_index, gt_index] = cardinality_bonus + iou + tie

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
    return {"precision": precision, "recall": recall, "f1": f1}


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

    source_ids = [row.source_candidate_id for row in branch_predictions]
    if any(value is None for value in source_ids):
        raise ValueError("every branch prediction must have a source candidate id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("branch source candidate ids must be unique")

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
        for key in ("precision", "recall", "f1"):
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


def _branch_rows(
    predictions: Sequence[PredictedInstance],
    candidate_export_ids: Mapping[str, Any],
    class_names: Sequence[str],
) -> tuple[BranchPrediction, ...]:
    export_to_candidate: dict[int, int] = {}
    for candidate_id, export_id in candidate_export_ids.items():
        export = int(export_id)
        if export in export_to_candidate:
            raise ValueError("multiple candidate ids point to one exported prediction")
        export_to_candidate[export] = int(candidate_id)
    by_export = {int(row.instance_id): row for row in predictions}
    missing = sorted(set(export_to_candidate) - set(by_export))
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
            source_candidate_id=int(export_to_candidate[export_id]),
        )
        for export_id, row in sorted(by_export.items())
        if export_id in export_to_candidate
    )


def _prediction_rows(
    predictions: Sequence[PredictedInstance], class_names: Sequence[str]
) -> tuple[BranchPrediction, ...]:
    return tuple(
        BranchPrediction(
            prediction_id=int(row.instance_id),
            class_name=str(class_names[row.class_id]),
            score=float(row.score),
            mask=np.asarray(row.mask, dtype=bool),
        )
        for row in predictions
    )


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
    if manifest.get("schema") != "saga-instance-recheck-evaluation-manifest-v1":
        raise ValueError("unsupported recheck evaluation manifest schema")
    base = manifest_source.parent
    condition_names = tuple(str(value) for value in manifest["conditions"])
    if not condition_names:
        raise ValueError("manifest must declare at least one condition")
    per_condition_scene: dict[str, list[dict[str, Any]]] = {
        name: [] for name in condition_names
    }
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
        b0_rows = _prediction_rows(b0_predictions, taxonomy.canonical_classes)
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
            branch = _branch_rows(
                predictions,
                output_payload.get("candidate_export_ids", {}),
                taxonomy.canonical_classes,
            )
            predictions_by_condition[condition].extend(predictions)
            diagnostics[scene_id]["conditions"][condition] = {
                **condition_diagnostics,
                "traceable_branch_predictions": len(branch),
            }
            for threshold in (0.25, 0.50):
                per_condition_scene[condition].append(
                    evaluate_rescue_scene(
                        scene_id=scene_id,
                        b0_predictions=b0_rows,
                        branch_predictions=branch,
                        ground_truth=objects,
                        strata=strata,
                        iou_threshold=threshold,
                    )
                )

    conditions: dict[str, Any] = {}
    for condition in condition_names:
        rows = per_condition_scene[condition]
        rescue = {
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
        "schema": "saga-instance-recheck-evaluation-v1",
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
        },
    }
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
    "evaluate_recheck_manifest",
    "evaluate_rescue_scene",
    "ground_truth_objects",
    "match_one_to_one",
]
