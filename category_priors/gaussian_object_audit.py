from __future__ import annotations

"""Precision-first 3D audit of predicted Gaussian instances against ScanNet GT.

The official evaluator projects predictions onto GT vertices (GT -> Gaussian).
This module deliberately measures the reverse direction as a diagnostic: every
predicted Gaussian must find valid, same-instance GT support or it counts against
point precision.  The diagnostic never changes the official AP protocol.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    GroundTruthScene,
    apply_transform,
    evaluate_instances,
    load_ground_truth_npz,
    load_ply_xyz,
    saga_scene_predictions,
)
from .io import load_json, write_json, write_rows
from .scene_manifest import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .viewer_io import condition_slug, write_colored_ply


CORRECT_COLOR = np.asarray((52, 199, 89), dtype=np.uint8)
SAME_CLASS_WRONG_INSTANCE_COLOR = np.asarray((255, 203, 64), dtype=np.uint8)
WRONG_CLASS_COLOR = np.asarray((230, 68, 68), dtype=np.uint8)
UNSUPPORTED_COLOR = np.asarray((112, 112, 112), dtype=np.uint8)
GT_COLOR = np.asarray((56, 132, 255), dtype=np.uint8)
VIEWER_SMALL_CATEGORY_EXAMPLES = (
    "cup",
    "switch",
    "book",
    "phone",
    "speaker",
    "lamp",
    "trash can",
)


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    if scene.get("gaussian_ply"):
        path = Path(str(scene["gaussian_ply"]))
        return path if path.is_absolute() else Path(str(scene["base_path"])) / path
    root = Path(str(scene["base_path"])) / "output_models/point_cloud/iteration_30000"
    primary = root / "point_cloud.ply"
    return primary if primary.is_file() else root / "scene_point_cloud.ply"


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


def _nearest_indices(
    query: np.ndarray, reference: np.ndarray, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Gaussian object audit requires scipy") from exc
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    indices = np.full(len(query), -1, dtype=np.int64)
    distances = np.full(len(query), math.inf, dtype=np.float64)
    if not len(query) or not len(reference):
        return indices, distances
    raw_distances, raw_indices = cKDTree(reference).query(
        query, k=1, distance_upper_bound=radius_m, workers=-1
    )
    valid = np.isfinite(raw_distances) & (raw_indices < len(reference))
    indices[valid] = raw_indices[valid]
    distances[valid] = raw_distances[valid]
    return indices, distances


def _class_id(
    payload: Mapping[str, Any], canonical_classes: Sequence[str]
) -> tuple[int, str]:
    class_name = str(payload.get("class", "")).strip().lower()
    try:
        return canonical_classes.index(class_name), class_name
    except ValueError:
        return -1, class_name


def _duplicate_flags(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["dominant_gt_instance"] is not None:
            groups[(int(row["class_id"]), int(row["dominant_gt_instance"]))].append(row)
    for group in groups.values():
        ranked = sorted(
            group,
            key=lambda row: (
                float(row["official_iou"]),
                float(row["point_precision"]),
                int(row["correct_gaussian_count"]),
                -int(row["instance_id"]),
            ),
            reverse=True,
        )
        for index, row in enumerate(ranked):
            row["duplicate_prediction"] = index > 0


def evaluate_gaussian_object_precision(
    gaussian_xyz: np.ndarray,
    point_labels: np.ndarray,
    instances_metadata: Mapping[str | int, Mapping[str, Any]],
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    radius_m: float = 0.05,
    *,
    canonical_classes: Sequence[str],
) -> dict[str, Any]:
    """Measure Gaussian -> GT point precision and auxiliary GT -> Gaussian recall.

    Unsupported and void-mapped predicted Gaussians stay in the precision
    denominator.  This is intentionally stricter than the official projection
    used to build ScanNet prediction masks.
    """
    gaussian_xyz = np.asarray(gaussian_xyz, dtype=np.float64)
    point_labels = np.asarray(point_labels, dtype=np.int64)
    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    gt_semantic = np.asarray(gt_semantic, dtype=np.int64)
    gt_instance = np.asarray(gt_instance, dtype=np.int64)
    if gaussian_xyz.ndim != 2 or gaussian_xyz.shape[1] != 3:
        raise ValueError("gaussian_xyz must have shape [N, 3]")
    if gt_xyz.ndim != 2 or gt_xyz.shape[1] != 3:
        raise ValueError("gt_xyz must have shape [M, 3]")
    if point_labels.shape != (len(gaussian_xyz),):
        raise ValueError("point_labels and gaussian_xyz differ in length")
    if gt_semantic.shape != (len(gt_xyz),) or gt_instance.shape != (len(gt_xyz),):
        raise ValueError("GT coordinates, semantic and instance arrays differ in length")

    gaussian_to_gt, gaussian_distances = _nearest_indices(
        gaussian_xyz, gt_xyz, radius_m
    )
    gt_to_gaussian, gt_distances = _nearest_indices(gt_xyz, gaussian_xyz, radius_m)
    mapped_gt_labels = np.full(len(gt_xyz), -1, dtype=np.int64)
    valid_gt_to_gaussian = gt_to_gaussian >= 0
    mapped_gt_labels[valid_gt_to_gaussian] = point_labels[
        gt_to_gaussian[valid_gt_to_gaussian]
    ]

    rows: list[dict[str, Any]] = []
    point_categories: dict[int, np.ndarray] = {}
    for raw_instance_id, payload in instances_metadata.items():
        instance_id = int(raw_instance_id)
        if instance_id < 0:
            continue
        class_id, class_name = _class_id(payload, canonical_classes)
        if class_id < 0:
            continue
        predicted_indices = np.flatnonzero(point_labels == instance_id)
        if not len(predicted_indices):
            continue
        nearest_gt = gaussian_to_gt[predicted_indices]
        valid_support = nearest_gt >= 0
        supported_gt = nearest_gt[valid_support]
        supported_semantic = gt_semantic[supported_gt]
        supported_instance = gt_instance[supported_gt]
        evaluable_support = (
            valid_support.copy()
        )
        if np.any(valid_support):
            evaluable_support[valid_support] = (
                (supported_semantic >= 0) & (supported_instance >= 0)
            )
        same_class = np.zeros(len(predicted_indices), dtype=bool)
        same_class[evaluable_support] = (
            gt_semantic[nearest_gt[evaluable_support]] == class_id
        )
        dominant_gt_instance: int | None = None
        if np.any(same_class):
            ids, counts = np.unique(
                gt_instance[nearest_gt[same_class]], return_counts=True
            )
            dominant_gt_instance = int(ids[np.argmax(counts)])

        correct = np.zeros(len(predicted_indices), dtype=bool)
        if dominant_gt_instance is not None:
            correct[evaluable_support] = (
                (gt_semantic[nearest_gt[evaluable_support]] == class_id)
                & (gt_instance[nearest_gt[evaluable_support]] == dominant_gt_instance)
            )
        same_class_wrong = same_class & ~correct
        wrong_class = evaluable_support & ~same_class
        unsupported = ~evaluable_support
        categories = np.full(len(predicted_indices), 3, dtype=np.int8)
        categories[wrong_class] = 2
        categories[same_class_wrong] = 1
        categories[correct] = 0
        point_categories[instance_id] = categories

        total = len(predicted_indices)
        correct_count = int(correct.sum())
        same_class_count = int(same_class.sum())
        gt_mask = np.zeros(len(gt_xyz), dtype=bool)
        if dominant_gt_instance is not None:
            gt_mask = (
                (gt_semantic == class_id)
                & (gt_instance == dominant_gt_instance)
            )
        predicted_on_gt = mapped_gt_labels == instance_id
        intersection = int(np.count_nonzero(predicted_on_gt & gt_mask))
        union = int(np.count_nonzero(predicted_on_gt | gt_mask))
        official_iou = intersection / union if union else 0.0
        recall = intersection / int(gt_mask.sum()) if np.any(gt_mask) else 0.0
        touched_instances = set(
            int(value)
            for value in gt_instance[nearest_gt[same_class]]
            if int(value) >= 0
        )
        rows.append(
            {
                "instance_id": instance_id,
                "class_id": class_id,
                "class_name": class_name,
                "predicted_gaussian_count": total,
                "correct_gaussian_count": correct_count,
                "same_class_wrong_instance_count": int(same_class_wrong.sum()),
                "wrong_class_count": int(wrong_class.sum()),
                "unsupported_count": int(unsupported.sum()),
                "point_precision": correct_count / total,
                "semantic_precision": same_class_count / total,
                "unsupported_fraction": float(unsupported.mean()),
                "dominant_gt_instance": dominant_gt_instance,
                "dominant_gt_point_count": int(gt_mask.sum()),
                "gt_to_gaussian_recall": recall,
                "official_iou": official_iou,
                "official_match_025": bool(official_iou > 0.25),
                "official_match_050": bool(official_iou > 0.50),
                "same_class_gt_instances_touched": len(touched_instances),
                "merge_candidate": len(touched_instances) > 1,
                "duplicate_prediction": False,
                "mean_supported_distance_m": float(
                    np.mean(gaussian_distances[predicted_indices][valid_support])
                )
                if np.any(valid_support)
                else None,
            }
        )

    _duplicate_flags(rows)
    total_predicted = sum(int(row["predicted_gaussian_count"]) for row in rows)
    total_correct = sum(int(row["correct_gaussian_count"]) for row in rows)
    precisions = [float(row["point_precision"]) for row in rows]
    recalls = [
        float(row["gt_to_gaussian_recall"])
        for row in rows
        if row["dominant_gt_instance"] is not None
    ]
    valid_gaussian_mapping = gaussian_to_gt >= 0
    valid_gt_mapping = gt_to_gaussian >= 0
    aggregate = {
        "predicted_instance_count": len(rows),
        "predicted_gaussian_count": total_predicted,
        "correct_gaussian_count": total_correct,
        "micro_point_precision": total_correct / total_predicted
        if total_predicted
        else 0.0,
        "mean_instance_point_precision": float(np.mean(precisions))
        if precisions
        else 0.0,
        "median_instance_point_precision": float(np.median(precisions))
        if precisions
        else 0.0,
        "mean_matched_gt_recall": float(np.mean(recalls)) if recalls else 0.0,
        "median_matched_gt_recall": float(np.median(recalls)) if recalls else 0.0,
        "duplicate_prediction_count": sum(
            int(bool(row["duplicate_prediction"])) for row in rows
        ),
        "merge_candidate_count": sum(
            int(bool(row["merge_candidate"])) for row in rows
        ),
        "unsupported_prediction_count": sum(
            int(row["dominant_gt_instance"] is None) for row in rows
        ),
        "gaussian_to_gt_mapped_fraction": float(valid_gaussian_mapping.mean())
        if len(valid_gaussian_mapping)
        else 0.0,
        "gt_to_gaussian_mapped_fraction": float(valid_gt_mapping.mean())
        if len(valid_gt_mapping)
        else 0.0,
        "gaussian_to_gt_median_distance_m": float(
            np.median(gaussian_distances[valid_gaussian_mapping])
        )
        if np.any(valid_gaussian_mapping)
        else None,
        "gt_to_gaussian_median_distance_m": float(
            np.median(gt_distances[valid_gt_mapping])
        )
        if np.any(valid_gt_mapping)
        else None,
    }
    return {
        "kind": "gaussian_object_precision",
        "schema_version": "1.0",
        "radius_m": float(radius_m),
        "aggregate": aggregate,
        "instances": rows,
        "point_categories": point_categories,
        "gaussian_to_gt_indices": gaussian_to_gt,
    }


def _prediction_path(
    runs_root: Path, condition: str, scene_id: str, seed: int
) -> tuple[Path, Path]:
    run_dir = runs_root / condition / scene_id / f"seed-{seed}"
    return run_dir / "output.json", run_dir / "output.json.metadata.json"


def _aggregate_condition_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(row["predicted_gaussian_count"]) for row in rows)
    correct = sum(int(row["correct_gaussian_count"]) for row in rows)
    values = [float(row["point_precision"]) for row in rows]
    recalls = [
        float(row["gt_to_gaussian_recall"])
        for row in rows
        if row["dominant_gt_instance"] is not None
    ]
    return {
        "predicted_instance_count": len(rows),
        "predicted_gaussian_count": total,
        "micro_point_precision": correct / total if total else 0.0,
        "mean_instance_point_precision": float(np.mean(values)) if values else 0.0,
        "median_instance_point_precision": float(np.median(values)) if values else 0.0,
        "mean_matched_gt_recall": float(np.mean(recalls)) if recalls else 0.0,
        "duplicate_prediction_count": sum(
            int(bool(row["duplicate_prediction"])) for row in rows
        ),
        "merge_candidate_count": sum(
            int(bool(row["merge_candidate"])) for row in rows
        ),
        "unsupported_prediction_count": sum(
            int(row["dominant_gt_instance"] is None) for row in rows
        ),
    }


def _select_viewer_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str, int]] = set()

    def add(role: str, candidates: Sequence[dict[str, Any]], key=None, reverse=False) -> None:
        ordered = list(candidates)
        if key is not None:
            ordered.sort(key=key, reverse=reverse)
        for row in ordered:
            identity = (str(row["scene_id"]), str(row["condition"]), int(row["instance_id"]))
            if identity not in used:
                used.add(identity)
                selected.append({"role": role, **row})
                return

    matched = [row for row in rows if row["dominant_gt_instance"] is not None]
    add("highest_precision", matched, key=lambda row: float(row["point_precision"]), reverse=True)
    if matched:
        median = float(np.median([float(row["point_precision"]) for row in matched]))
        add("median_precision", matched, key=lambda row: abs(float(row["point_precision"]) - median))
    add("lowest_precision", matched, key=lambda row: float(row["point_precision"]))
    add(
        "unsupported_false_positive",
        rows,
        key=lambda row: (float(row["unsupported_fraction"]), int(row["predicted_gaussian_count"])),
        reverse=True,
    )
    add(
        "tiny_small_success",
        [
            row
            for row in rows
            if row["class_name"] in VIEWER_SMALL_CATEGORY_EXAMPLES
        ],
        key=lambda row: (float(row["point_precision"]), float(row["gt_to_gaussian_recall"])),
        reverse=True,
    )
    add(
        "tiny_small_failure",
        [
            row
            for row in rows
            if row["class_name"] in VIEWER_SMALL_CATEGORY_EXAMPLES
        ],
        key=lambda row: (float(row["point_precision"]), float(row["gt_to_gaussian_recall"])),
    )
    add(
        "merge_case",
        [row for row in rows if row["merge_candidate"]],
        key=lambda row: int(row["same_class_gt_instances_touched"]),
        reverse=True,
    )
    add(
        "split_or_duplicate_case",
        [row for row in rows if row["duplicate_prediction"]],
        key=lambda row: (float(row["point_precision"]), int(row["predicted_gaussian_count"])),
        reverse=True,
    )
    add(
        "changed_vs_reference",
        [row for row in rows if float(row.get("changed_fraction_vs_reference", 0.0)) > 0],
        key=lambda row: float(row.get("changed_fraction_vs_reference", 0.0)),
        reverse=True,
    )
    for index, row in enumerate(rows):
        if len(selected) >= 6:
            break
        add(f"fallback_{index + 1}", [row])
    return selected


def _export_viewer_case(
    case: Mapping[str, Any], audit: Mapping[str, Any], gt_xyz: np.ndarray,
    gt_semantic: np.ndarray, gt_instance: np.ndarray, gaussian_xyz: np.ndarray,
    point_labels: np.ndarray, output_dir: Path,
) -> dict[str, Any]:
    instance_id = int(case["instance_id"])
    predicted_indices = np.flatnonzero(point_labels == instance_id)
    categories = np.asarray(audit["point_categories"][instance_id], dtype=np.int8)
    colors = np.empty((len(categories), 3), dtype=np.uint8)
    colors[categories == 0] = CORRECT_COLOR
    colors[categories == 1] = SAME_CLASS_WRONG_INSTANCE_COLOR
    colors[categories == 2] = WRONG_CLASS_COLOR
    colors[categories == 3] = UNSUPPORTED_COLOR
    gt_mask = np.zeros(len(gt_xyz), dtype=bool)
    if case["dominant_gt_instance"] is not None:
        gt_mask = (
            (gt_semantic == int(case["class_id"]))
            & (gt_instance == int(case["dominant_gt_instance"]))
        )
    gt_points = gt_xyz[gt_mask]
    case_dir = (
        output_dir / str(case["scene_id"]) / condition_slug(str(case["condition"]))
        / f"{case['role']}-instance-{instance_id}"
    )
    write_colored_ply(
        case_dir / "predicted_gaussians.ply",
        gaussian_xyz[predicted_indices].astype(np.float32),
        colors,
    )
    write_colored_ply(
        case_dir / "matched_gt_points.ply",
        gt_points.astype(np.float32),
        np.tile(GT_COLOR, (len(gt_points), 1)),
    )
    overlay_xyz = np.concatenate((gaussian_xyz[predicted_indices], gt_points), axis=0)
    overlay_rgb = np.concatenate((colors, np.tile(GT_COLOR, (len(gt_points), 1))), axis=0)
    write_colored_ply(
        case_dir / "overlay.ply", overlay_xyz.astype(np.float32), overlay_rgb
    )
    metrics = {
        key: value
        for key, value in case.items()
        if key not in {"point_categories", "gaussian_to_gt_indices"}
    }
    write_json(case_dir / "metrics.json", metrics)
    return {
        "role": str(case["role"]),
        "scene_id": str(case["scene_id"]),
        "condition": str(case["condition"]),
        "instance_id": instance_id,
        "directory": str(case_dir),
    }


def audit_gaussian_object_runs(
    *,
    scene_manifest: str | Path,
    gt_dir: str | Path,
    runs_root: str | Path,
    taxonomy: Taxonomy,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    seed: int,
    table_output: str | Path,
    audit_output: str | Path,
    comparison_output: str | Path,
    viewer_output: str | Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    condition_names = tuple(map(str, conditions))
    if not condition_names:
        raise ValueError("at least one condition is required")
    reference_condition = condition_names[0]
    runtime = load_scene_runtime_manifest(scene_manifest)
    gt_root = Path(gt_dir)
    runs = Path(runs_root)
    all_rows: list[dict[str, Any]] = []
    audits: dict[tuple[str, str], dict[str, Any]] = {}
    scene_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    official_scenes: dict[str, list[GroundTruthScene]] = defaultdict(list)
    official_predictions: dict[str, list[Any]] = defaultdict(list)

    for scene_id in map(str, scene_ids):
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_root / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        scene_arrays[scene_id] = (gt_xyz, gt.semantic, gt.instance, gaussian_xyz)
        reference_classes: np.ndarray | None = None
        for condition in condition_names:
            output_path, metadata_path = _prediction_path(runs, condition, scene_id, seed)
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"{scene_id}/{condition}: official AP metadata is missing: {metadata_path}"
                )
            output = load_json(output_path)
            point_labels = np.asarray(output["point_labels"], dtype=np.int64)
            audit = evaluate_gaussian_object_precision(
                gaussian_xyz,
                point_labels,
                output.get("instances", {}),
                gt_xyz,
                gt.semantic,
                gt.instance,
                radius_m,
                canonical_classes=taxonomy.canonical_classes,
            )
            audits[(scene_id, condition)] = {**audit, "point_labels": point_labels}
            class_by_point = np.full(len(point_labels), -1, dtype=np.int64)
            for raw_id, payload in output.get("instances", {}).items():
                instance_id = int(raw_id)
                if instance_id < 0:
                    continue
                class_id, _ = _class_id(payload, taxonomy.canonical_classes)
                if class_id >= 0:
                    class_by_point[point_labels == instance_id] = class_id
            if condition == reference_condition:
                reference_classes = class_by_point
            for row in audit["instances"]:
                enriched = {
                    "scene_id": scene_id,
                    "condition": condition,
                    "seed": int(seed),
                    **row,
                }
                if reference_classes is not None and condition != reference_condition:
                    mask = point_labels == int(row["instance_id"])
                    enriched["changed_fraction_vs_reference"] = float(
                        np.mean(reference_classes[mask] != class_by_point[mask])
                    ) if np.any(mask) else 0.0
                else:
                    enriched["changed_fraction_vs_reference"] = 0.0
                all_rows.append(enriched)
            predictions, _ = saga_scene_predictions(
                scene_id,
                gt_xyz,
                output_path,
                _gaussian_ply(scene),
                taxonomy,
                metadata_path,
                _transform(scene),
                radius_m,
                require_scores=True,
            )
            official_scenes[condition].append(gt)
            official_predictions[condition].extend(predictions)

    condition_summaries: dict[str, Any] = {}
    for condition in condition_names:
        rows = [row for row in all_rows if row["condition"] == condition]
        official = evaluate_instances(
            official_scenes[condition],
            official_predictions[condition],
            taxonomy.canonical_classes,
            min_region_size=min_region_size,
        )
        condition_summaries[condition] = {
            **_aggregate_condition_rows(rows),
            "official": official["aggregate"],
        }

    comparison: dict[str, Any] = {
        "kind": "gaussian_precision_comparison",
        "scene_ids": list(map(str, scene_ids)),
        "seed": int(seed),
        "radius_m": float(radius_m),
        "reference_condition": reference_condition,
        "conditions": condition_summaries,
    }
    left = condition_summaries[reference_condition]
    comparison["differences_vs_reference"] = {}
    for condition in condition_names[1:]:
        right = condition_summaries[condition]
        comparison["differences_vs_reference"][condition] = {
            "micro_point_precision": right["micro_point_precision"] - left["micro_point_precision"],
            "mean_matched_gt_recall": right["mean_matched_gt_recall"] - left["mean_matched_gt_recall"],
            "map_50_90": right["official"]["map_50_90"] - left["official"]["map_50_90"],
            "map_0.50": right["official"]["map_0.50"] - left["official"]["map_0.50"],
            "map_0.25": right["official"]["map_0.25"] - left["official"]["map_0.25"],
        }

    selected = _select_viewer_cases(all_rows)
    viewer_cases: list[dict[str, Any]] = []
    output_root = Path(viewer_output)
    for case in selected:
        scene_id = str(case["scene_id"])
        condition = str(case["condition"])
        gt_xyz, gt_semantic, gt_instance, gaussian_xyz = scene_arrays[scene_id]
        current = audits[(scene_id, condition)]
        viewer_cases.append(
            _export_viewer_case(
                case,
                current,
                gt_xyz,
                gt_semantic,
                gt_instance,
                gaussian_xyz,
                np.asarray(current["point_labels"], dtype=np.int64),
                output_root,
            )
        )
    viewer_manifest = {
        "kind": "gaussian_object_precision_viewer",
        "selection": "precision-first deterministic diagnostic cases",
        "qualitative_only": True,
        "not_for_parameter_selection": True,
        "colors": {
            "green": "same class and same GT instance",
            "yellow": "same class, different GT instance",
            "red": "wrong class",
            "gray": "no valid GT support within radius",
            "blue": "matched GT instance points",
        },
        "cases": viewer_cases,
    }
    write_json(output_root / "viewer_case_selection.json", viewer_manifest)
    write_rows(table_output, all_rows)
    audit_payload = {
        "kind": "gaussian_object_audit",
        "schema_version": "1.0",
        "scene_ids": list(map(str, scene_ids)),
        "conditions": list(condition_names),
        "reference_condition": reference_condition,
        "seed": int(seed),
        "radius_m": float(radius_m),
        "min_region_size": int(min_region_size),
        "condition_summaries": condition_summaries,
        "viewer": viewer_manifest,
        "official_ap_unchanged": True,
        "two_dimensional_metrics": False,
    }
    write_json(audit_output, audit_payload)
    write_json(comparison_output, comparison)
    return audit_payload
