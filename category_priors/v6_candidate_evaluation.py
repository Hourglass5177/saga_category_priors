from __future__ import annotations

"""GT-only evaluation of V6 input funnel and immutable affinity proposal banks."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v3_shadow_evaluation import _gaussian_ply, _gt_instances, _transform
from .v5_candidate import SAGA20
from .v6_candidate_runner import v6_candidate_run_paths


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(labels, dtype=bool)
    if not len(values) or target.all() or (~target).all():
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    unique, starts, counts = np.unique(values[order], return_index=True, return_counts=True)
    del unique
    for start, count in zip(starts, counts):
        if count > 1:
            ranks[order[start:start + count]] = (2 * start + count + 1) / 2.0
    positive = int(target.sum())
    negative = int((~target).sum())
    return float((ranks[target].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def _mapped_gaussian_indices(gt_xyz: np.ndarray, gaussian_xyz: np.ndarray, radius_m: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(gaussian_xyz).query(gt_xyz, k=1, distance_upper_bound=radius_m, workers=-1)
    result = np.full(len(gt_xyz), -1, dtype=np.int64)
    valid = np.isfinite(distances) & (indices < len(gaussian_xyz))
    result[valid] = indices[valid]
    return result


def evaluate_v6_candidate_banks(
    *, scene_manifest: str | Path, gt_dir: str | Path, output_root: str | Path,
    taxonomy: Taxonomy, size_bins: str | Path, scene_ids: Sequence[str], seed: int,
    table_output: str | Path, analysis_output: str | Path, radius_m: float = 0.05,
) -> dict[str, Any]:
    runtime = load_scene_runtime_manifest(scene_manifest)
    size_spec = load_json(size_bins)
    rows: list[dict[str, Any]] = []
    commit: str | None = None
    candidate_total = candidate_positive_025 = candidate_positive_050 = 0
    candidate_positive_scenes: set[str] = set()
    tiny_small_gt: list[dict[str, Any]] = []
    tiny_small_recall_025: list[bool] = []
    tiny_small_recall_050: list[bool] = []
    sam_coverages: list[float] = []
    edge_aucs: list[float] = []
    for scene_id in map(str, scene_ids):
        paths = v6_candidate_run_paths(output_root, scene_id, seed)
        bank = load_json(paths["proposals"])
        if bank.get("kind") != "v6_affinity_proposal_bank":
            raise ValueError(f"{paths['proposals']}: not a V6 affinity proposal bank")
        current_commit = str(bank["git_commit"])
        if commit is None:
            commit = current_commit
        elif commit != current_commit:
            raise ValueError("V6 candidate captures use different commits")
        with np.load(paths["proposal_labels"], allow_pickle=False) as arrays:
            full_labels = np.asarray(arrays["full_labels"], dtype=np.int64)
            codebook_winner = np.asarray(arrays["codebook_winner"], dtype=np.int64)
            codebook_score = np.asarray(arrays["codebook_score"], dtype=np.float64)
            point_views = np.asarray(arrays["point_view_count"], dtype=np.int64)
            point_vote_winner = np.asarray(arrays["point_vote_winner"], dtype=np.int64)
            point_vote_ratio = np.asarray(arrays["point_vote_ratio"], dtype=np.float64)
            edge_left = np.asarray(arrays["edge_left"], dtype=np.int64)
            edge_right = np.asarray(arrays["edge_right"], dtype=np.int64)
            edge_affinity = np.asarray(arrays["edge_affinity"], dtype=np.float64)
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", scene_id)
        gt_instances = _gt_instances(gt_xyz, gt.semantic, gt.instance, taxonomy, size_spec, scene_id)
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        mapped_full, mapping = map_gaussians_to_gt(gt_xyz, gaussian_xyz, full_labels, radius_m)
        mapped_indices = _mapped_gaussian_indices(gt_xyz, gaussian_xyz, radius_m)
        class_to_index = {name: index for index, name in enumerate(bank["class_names"])}
        candidates = {int(row["candidate_id"]): dict(row) for row in bank.get("candidates", [])}
        for instance in gt_instances:
            canonical = str(instance["canonical_class"])
            mask = np.asarray(instance["mask"], dtype=bool)
            indices = mapped_indices[mask]
            valid = indices >= 0
            mapped = indices[valid]
            class_index = class_to_index.get(canonical, -1)
            sam = float(np.mean(point_views[mapped] > 0)) if len(mapped) else 0.0
            correct_mask = float(np.mean(point_vote_winner[mapped] == class_index)) if len(mapped) and class_index >= 0 else 0.0
            semantic = float(np.mean(codebook_winner[mapped] == class_index)) if len(mapped) and class_index >= 0 else 0.0
            semantic_thresholded = float(np.mean((codebook_winner[mapped] == class_index) & (codebook_score[mapped] >= 0.70))) if len(mapped) and class_index >= 0 else 0.0
            matching = [candidate_id for candidate_id, row in candidates.items() if str(row.get("branch_class")) == canonical]
            best = max((_iou(mapped_full == candidate_id, mask) for candidate_id in matching), default=0.0)
            row = {
                "row_type": "gt_instance", "scene_id": scene_id, "seed": int(seed), "git_commit": current_commit,
                "canonical_class": canonical, "gt_instance_id": int(instance["gt_instance_id"]),
                "point_count": int(instance["point_count"]), "bbox_diag_m": float(instance["bbox_diag_m"]),
                "physical_size_bin": str(instance["physical_size_bin"]),
                "below_official_min_region_size": bool(instance["below_official_min_region_size"]),
                "sam_coverage": sam, "correct_class_mask_coverage": correct_mask,
                "codebook_top1_recall": semantic, "codebook_thresholded_recall": semantic_thresholded,
                "same_class_best_iou": float(best), "mapped_fraction": float(mapping["mapped_fraction"]),
            }
            rows.append(row)
            if canonical in SAGA20 and row["physical_size_bin"] in {"tiny", "small"}:
                tiny_small_gt.append(row)
                sam_coverages.append(sam)
                tiny_small_recall_025.append(best >= 0.25)
                tiny_small_recall_050.append(best >= 0.50)
        for candidate_id, candidate in candidates.items():
            mask = mapped_full == candidate_id
            same = [instance for instance in gt_instances if str(instance["canonical_class"]) == str(candidate["branch_class"])]
            best = max((_iou(mask, np.asarray(instance["mask"], dtype=bool)) for instance in same), default=0.0)
            candidate_total += 1
            candidate_positive_025 += int(best >= 0.25)
            candidate_positive_050 += int(best >= 0.50)
            if best >= 0.50:
                candidate_positive_scenes.add(scene_id)
            rows.append({
                "row_type": "candidate", "scene_id": scene_id, "seed": int(seed), "git_commit": current_commit,
                "candidate_id": candidate_id, "canonical_class": str(candidate["branch_class"]),
                "candidate_points": int(mask.sum()), "core_point_count": int(candidate["core_point_count"]),
                "same_class_best_iou": float(best), "effective_view_count": int(candidate["effective_view_count"]),
                "vote_ratio": float(candidate["vote"]["winner_ratio"]),
                "internal_affinity_mean": float(candidate["internal_affinity_mean"]),
            })
        # Evaluate only local edges whose endpoints both have an aligned GT label.
        gt_instance_by_gaussian: dict[int, tuple[int, int]] = {}
        for point_index, gaussian_index in enumerate(mapped_indices):
            if gaussian_index < 0 or gt.instance[point_index] < 0 or gt.semantic[point_index] < 0:
                continue
            gt_instance_by_gaussian[int(gaussian_index)] = (int(gt.semantic[point_index]), int(gt.instance[point_index]))
        known = [(left, right) for left, right in zip(edge_left, edge_right) if int(left) in gt_instance_by_gaussian and int(right) in gt_instance_by_gaussian]
        if known:
            selector = np.asarray([index for index, (left, right) in enumerate(zip(edge_left, edge_right)) if int(left) in gt_instance_by_gaussian and int(right) in gt_instance_by_gaussian], dtype=np.int64)
            labels = np.asarray([gt_instance_by_gaussian[int(edge_left[index])] == gt_instance_by_gaussian[int(edge_right[index])] for index in selector], dtype=bool)
            value = _auc(edge_affinity[selector], labels)
            if value is not None:
                edge_aucs.append(value)
                rows.append({"row_type": "edge_auroc", "scene_id": scene_id, "seed": int(seed), "git_commit": current_commit, "edge_auroc": value, "edge_count": int(len(selector))})
    write_rows(table_output, rows)
    official_tiny_small = [row for row in tiny_small_gt if not row["below_official_min_region_size"]]
    official_025 = [row["same_class_best_iou"] >= 0.25 for row in official_tiny_small]
    official_050 = [row["same_class_best_iou"] >= 0.50 for row in official_tiny_small]
    median_sam = float(np.median(sam_coverages)) if sam_coverages else 0.0
    mean_auc = float(np.mean(edge_aucs)) if edge_aucs else None
    candidate_precision = candidate_positive_025 / candidate_total if candidate_total else 0.0
    stage0 = {
        "sam_coverage_insufficient": bool(median_sam < 0.35),
        "affinity_edge_auc": mean_auc,
        "affinity_insufficient": bool(mean_auc is not None and mean_auc < 0.60),
        "recommended_next_stage": "sam_reextract" if median_sam < 0.35 else ("feature_10k_control" if mean_auc is not None and mean_auc < 0.60 else "affinity_candidate_gate"),
    }
    candidate_gate = {
        "same_class_positive_050": candidate_positive_050,
        "positive_scene_count_050": len(candidate_positive_scenes),
        "candidate_precision_025": candidate_precision,
        "tiny_small_recall_025": float(np.mean(official_025)) if official_025 else 0.0,
        "tiny_small_recall_050": float(np.mean(official_050)) if official_050 else 0.0,
        "absolute_candidate_gate_passed": bool(candidate_positive_050 >= 12 and len(candidate_positive_scenes) >= 4 and candidate_precision >= 0.05),
    }
    analysis = {
        "kind": "v6_input_and_candidate_analysis", "schema_version": "1.0", "git_commit": commit,
        "scene_count": len(scene_ids), "seed": int(seed), "stage0": stage0,
        "candidate_gate": candidate_gate, "tiny_small_gt_count": len(official_tiny_small),
        "median_tiny_small_sam_coverage": median_sam,
    }
    write_json(analysis_output, analysis)
    return analysis
