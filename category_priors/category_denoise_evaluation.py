from __future__ import annotations

"""Evaluation for the frozen all-category denoising bank and its replays."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    apply_transform,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v9_metrics import (
    _bbox_diagonal,
    _gaussian_ply,
    _size_bin,
    _transform,
    evaluate_v9_predictions,
)


def _bank_labels(bank_root: Path, scene_id: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    direct = bank_root / scene_id
    scene_root = direct if direct.is_dir() else bank_root / "bank" / scene_id
    payload = load_json(scene_root / "candidates.json")
    rows = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise TypeError(f"{scene_id}: invalid candidates.json")
    with np.load(scene_root / "bank_labels.npz", allow_pickle=False) as arrays:
        for key in ("branch_full", "branch_full_id", "branch_full_labels"):
            if key in arrays.files:
                labels = np.asarray(arrays[key], dtype=np.int64)
                break
        else:
            raise ValueError(f"{scene_id}: bank is missing branch-full labels")
    return labels, [dict(row) for row in rows]


def evaluate_candidate_bank(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Measure frozen candidate quality; GT is used only in this evaluator."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins is not None else None
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    per_scene: list[dict[str, Any]] = []
    all_candidate_025 = 0
    all_candidate_050 = 0
    all_candidates = 0
    all_tiny = 0
    all_tiny_025 = 0
    all_tiny_050 = 0
    classes_050: set[str] = set()
    scenes_050: set[str] = set()
    for scene_id in map(str, scene_ids):
        scene = scenes[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        branch_labels, candidates = _bank_labels(bank_root, scene_id)
        if len(branch_labels) != len(gaussian_xyz):
            raise ValueError(f"{scene_id}: bank/Gaussian point count mismatch")
        mapped, _ = map_gaussians_to_gt(gt_xyz, gaussian_xyz, branch_labels, radius_m)

        gt_rows: list[dict[str, Any]] = []
        valid = (gt.semantic >= 0) & (gt.instance >= 0)
        for class_id, instance_id in sorted(
            set(zip(gt.semantic[valid].tolist(), gt.instance[valid].tolist()))
        ):
            mask = valid & (gt.semantic == class_id) & (gt.instance == instance_id)
            if int(mask.sum()) < int(min_region_size):
                continue
            name = taxonomy.canonical_classes[int(class_id)]
            gt_rows.append(
                {
                    "class": name,
                    "mask": mask,
                    "tiny_small": _size_bin(_bbox_diagonal(gt_xyz[mask]), size_spec)
                    in {"tiny", "small"},
                }
            )

        scene_025 = 0
        scene_050 = 0
        best_by_gt = np.zeros(len(gt_rows), dtype=np.float64)
        candidate_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = int(candidate["candidate_id"])
            class_name = str(candidate.get("branch_class", candidate.get("class_name", "")))
            if class_name not in class_to_id:
                continue
            prediction = mapped == candidate_id
            best_iou = 0.0
            best_gt = None
            for gt_index, gt_row in enumerate(gt_rows):
                if gt_row["class"] != class_name:
                    continue
                gt_mask = gt_row["mask"]
                intersection = int(np.count_nonzero(prediction & gt_mask))
                union = int(np.count_nonzero(prediction | gt_mask))
                iou = intersection / union if union else 0.0
                if iou > best_iou:
                    best_iou = float(iou)
                    best_gt = gt_index
                best_by_gt[gt_index] = max(best_by_gt[gt_index], float(iou))
            scene_025 += int(best_iou >= 0.25)
            scene_050 += int(best_iou >= 0.50)
            if best_iou >= 0.50:
                classes_050.add(class_name)
                scenes_050.add(scene_id)
            candidate_rows.append(
                {
                    "candidate_id": candidate_id,
                    "class": class_name,
                    "best_same_class_iou": best_iou,
                    "best_gt_index": best_gt,
                }
            )
        tiny_indices = [index for index, row in enumerate(gt_rows) if row["tiny_small"]]
        tiny_025 = sum(best_by_gt[index] >= 0.25 for index in tiny_indices)
        tiny_050 = sum(best_by_gt[index] >= 0.50 for index in tiny_indices)
        per_scene.append(
            {
                "scene_id": scene_id,
                "candidate_count": len(candidate_rows),
                "same_class_iou_025_count": scene_025,
                "same_class_iou_050_count": scene_050,
                "candidate_precision_025": scene_025 / len(candidate_rows)
                if candidate_rows
                else 0.0,
                "tiny_small_gt_count": len(tiny_indices),
                "tiny_small_recall_025": tiny_025 / len(tiny_indices)
                if tiny_indices
                else 0.0,
                "tiny_small_recall_050": tiny_050 / len(tiny_indices)
                if tiny_indices
                else 0.0,
                "candidates": candidate_rows,
            }
        )
        all_candidates += len(candidate_rows)
        all_candidate_025 += scene_025
        all_candidate_050 += scene_050
        all_tiny += len(tiny_indices)
        all_tiny_025 += int(tiny_025)
        all_tiny_050 += int(tiny_050)
    return {
        "candidate_count": all_candidates,
        "same_class_iou_025_count": all_candidate_025,
        "same_class_iou_050_count": all_candidate_050,
        "same_class_iou_050_scene_count": len(scenes_050),
        "same_class_iou_050_classes": sorted(classes_050),
        "candidate_precision_025": all_candidate_025 / all_candidates
        if all_candidates
        else 0.0,
        "tiny_small_gt_count": all_tiny,
        "tiny_small_recall_025": all_tiny_025 / all_tiny if all_tiny else 0.0,
        "tiny_small_recall_050": all_tiny_050 / all_tiny if all_tiny else 0.0,
        "per_scene": per_scene,
    }


def evaluate_category_denoise(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    prediction_root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    size_bins: Path | None = None,
    viewer_output: Path | None = None,
) -> dict[str, Any]:
    analysis = evaluate_v9_predictions(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        prediction_root=prediction_root,
        scene_ids=scene_ids,
        conditions=conditions,
        taxonomy=taxonomy,
        metrics_output=metrics_output,
        analysis_output=analysis_output,
        radius_m=radius_m,
        min_region_size=min_region_size,
        size_bins=size_bins,
        viewer_output=viewer_output,
    )
    # V9's shared evaluator already provides official AP, Gaussian precision,
    # GT recall, FP/TP and the strict output-contract audit.  The denoising
    # gate additionally preregisters prediction coverage, so add it here from
    # the same exported point labels rather than inventing another evaluator.
    metric_rows: list[dict[str, Any]] = []
    for condition in map(str, conditions):
        condition_result = analysis["conditions"][condition]
        point_total = 0
        assigned_total = 0
        by_scene = {
            str(row["scene_id"]): row for row in condition_result["per_scene"]
        }
        for scene_id in map(str, scene_ids):
            output = load_json(prediction_root / condition / scene_id / "output.json")
            labels = np.asarray(output["point_labels"], dtype=np.int64)
            assigned = int(np.count_nonzero(labels >= 0))
            total = len(labels)
            by_scene[scene_id]["prediction_coverage"] = (
                assigned / total if total else 0.0
            )
            by_scene[scene_id]["assigned_gaussian_count"] = assigned
            by_scene[scene_id]["gaussian_count"] = total
            point_total += total
            assigned_total += assigned
        condition_result["metrics"]["prediction_coverage"] = (
            assigned_total / point_total if point_total else 0.0
        )
        condition_result["metrics"]["assigned_gaussian_count"] = assigned_total
        condition_result["metrics"]["gaussian_count"] = point_total
        metric_rows.append(dict(condition_result["metrics"]))
    write_rows(metrics_output, metric_rows)
    analysis["schema"] = "saga-category-denoise-analysis-v1"
    analysis["candidate_bank"] = evaluate_candidate_bank(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        bank_root=bank_root,
        scene_ids=scene_ids,
        taxonomy=taxonomy,
        size_bins=size_bins,
        radius_m=radius_m,
        min_region_size=min_region_size,
    )
    write_json(analysis_output, analysis)
    return analysis
