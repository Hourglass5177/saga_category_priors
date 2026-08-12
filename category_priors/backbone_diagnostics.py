from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .class_first_evaluation import _scene_gaussian_path
from .evaluator import (
    apply_transform,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


def diagnose_mapped_instances(
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    mapped_labels: np.ndarray,
    instance_classes: Mapping[int, int],
    instance_scores: Mapping[int, float] | None = None,
    min_region_size: int = 100,
) -> dict[str, Any]:
    valid_gt = gt_instance >= 0
    assigned = mapped_labels >= 0
    predicted_semantic = np.full(len(mapped_labels), -1, dtype=np.int64)
    for instance_id, class_id in instance_classes.items():
        predicted_semantic[mapped_labels == int(instance_id)] = int(class_id)
    semantic_correct = assigned & valid_gt & (predicted_semantic == gt_semantic)
    gt_items: list[tuple[int, int, np.ndarray]] = []
    for class_id in np.unique(gt_semantic):
        class_mask = gt_semantic == class_id
        for instance_id in np.unique(gt_instance[class_mask]):
            mask = class_mask & (gt_instance == instance_id)
            if instance_id >= 0 and int(mask.sum()) >= min_region_size:
                gt_items.append((int(class_id), int(instance_id), mask))
    best_ious: list[float] = []
    split_counts: list[int] = []
    prediction_best_iou: dict[int, float] = {int(key): 0.0 for key in instance_classes}
    prediction_gt_hits: dict[int, int] = {int(key): 0 for key in instance_classes}
    for class_id, _, gt_mask in gt_items:
        overlaps: list[float] = []
        for prediction_id, prediction_class in instance_classes.items():
            if prediction_class != class_id:
                continue
            prediction_mask = mapped_labels == prediction_id
            intersection = int(np.count_nonzero(gt_mask & prediction_mask))
            if not intersection:
                continue
            union = int(np.count_nonzero(gt_mask | prediction_mask))
            iou = intersection / max(union, 1)
            overlaps.append(iou)
            prediction_best_iou[prediction_id] = max(
                prediction_best_iou[prediction_id], iou
            )
            if iou > 0.10:
                prediction_gt_hits[prediction_id] += 1
        best_ious.append(max(overlaps, default=0.0))
        split_counts.append(sum(value > 0.10 for value in overlaps))
    scores = instance_scores or {}
    paired = [
        (float(scores[key]), value)
        for key, value in prediction_best_iou.items()
        if key in scores
    ]
    score_iou_correlation = None
    if len(paired) >= 2:
        score_array = np.asarray([item[0] for item in paired])
        iou_array = np.asarray([item[1] for item in paired])
        if np.std(score_array) > 0 and np.std(iou_array) > 0:
            score_iou_correlation = float(np.corrcoef(score_array, iou_array)[0, 1])
    return {
        "gt_points": int(np.count_nonzero(valid_gt)),
        "assigned_gt_points": int(np.count_nonzero(assigned & valid_gt)),
        "correct_semantic_assigned_gt_points": int(np.count_nonzero(semantic_correct)),
        "assigned_gt_fraction": float(np.mean(assigned[valid_gt])) if np.any(valid_gt) else 0.0,
        "semantic_accuracy_on_assigned_gt": float(
            np.mean(predicted_semantic[assigned & valid_gt] == gt_semantic[assigned & valid_gt])
        ) if np.any(assigned & valid_gt) else 0.0,
        "correct_semantic_gt_fraction": float(np.mean(semantic_correct[valid_gt]))
        if np.any(valid_gt) else 0.0,
        "gt_instances": len(gt_items),
        "pred_instances": len(instance_classes),
        "prediction_gt_ratio": len(instance_classes) / max(len(gt_items), 1),
        "proposal_recall_025": float(np.mean(np.asarray(best_ious) >= 0.25)) if best_ious else 0.0,
        "proposal_recall_050": float(np.mean(np.asarray(best_ious) >= 0.50)) if best_ious else 0.0,
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "mean_split_count_at_010": float(np.mean(split_counts)) if split_counts else 0.0,
        "merge_predictions_at_010": int(sum(value > 1 for value in prediction_gt_hits.values())),
        "score_iou_correlation": score_iou_correlation,
    }


def diagnose_backbone_runs(
    scene_manifest: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    taxonomy: Taxonomy,
    output_json: str | Path,
    output_table: str | Path,
    conditions: Sequence[str],
    seeds: Sequence[int],
    scene_ids: Sequence[str],
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            for scene_id in scene_ids:
                scene = scenes[scene_id]
                run_dir = Path(output_root) / condition / scene_id / f"seed-{seed}"
                output = load_json(run_dir / "output.json")
                metadata_path = run_dir / "diagnostics.json"
                metadata = load_json(metadata_path) if metadata_path.is_file() else {"instances": {}}
                gt_coords, gt = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", scene_id)
                transform = scene.get("gaussian_to_gt_transform", np.eye(4).tolist())
                gaussian_coords = apply_transform(
                    load_ply_xyz(_scene_gaussian_path(scene)), transform
                )
                mapped, alignment = map_gaussians_to_gt(
                    gt_coords, gaussian_coords,
                    np.asarray(output["point_labels"], dtype=np.int64), radius_m,
                )
                instances = {
                    int(key): class_to_id[str(value.get("class", "")).strip().lower()]
                    for key, value in output.get("instances", {}).items()
                    if str(value.get("class", "")).strip().lower() in class_to_id
                }
                metadata_instances = metadata.get("instances", {})
                scores = {
                    int(key): float(value.get("score", 1.0))
                    for key, value in metadata_instances.items()
                    if isinstance(value, Mapping)
                }
                metrics = diagnose_mapped_instances(
                    gt.semantic, gt.instance, mapped, instances, scores, min_region_size
                )
                rows.append({
                    "condition": condition, "seed": int(seed), "scene_id": scene_id,
                    **alignment, **metrics,
                })
    summary: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            key: float(np.mean([row[key] for row in selected if row[key] is not None]))
            for key in selected[0]
            if key not in {"condition", "seed", "scene_id"}
        } if selected else {}
    payload = {
        "schema_version": "1.0", "kind": "backbone_audit",
        "conditions": list(conditions), "seeds": [int(value) for value in seeds],
        "scenes": list(scene_ids), "summary": summary,
    }
    write_rows(output_table, rows)
    write_json(output_json, payload)
    return payload
