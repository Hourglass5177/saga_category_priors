from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v3_shadow import load_shadow_arrays
from .v3_shadow_evaluation import (
    _affinity_instance_metrics, _feature_ply, _gaussian_ply, _gt_instances, _iou,
    _load_affinity_features, _map_gt_to_gaussian_indices, _transform,
)
from .v4_candidate_runner import v4_candidate_run_paths
from .v4_feature_control import v4_feature_control_paths


def evaluate_v4_feature_control(
    *, scene_manifest_path: str | Path, gt_dir: str | Path, taxonomy: Taxonomy,
    size_bins_path: str | Path, scene_ids: Sequence[str], seed: int,
    candidate_root_2k: str | Path, candidate_root_10k: str | Path,
    feature_control_root: str | Path, output: str | Path, radius_m: float = 0.05,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    size_spec = load_json(size_bins_path)
    instances: list[dict[str, Any]] = []
    candidate_totals = {"2k": 0, "10k": 0}
    candidate_positive = {"2k": 0, "10k": 0}
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        gt_coords, gt_scene = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", scene_id)
        gt_instances = _gt_instances(
            gt_coords, gt_scene.semantic, gt_scene.instance, taxonomy, size_spec, scene_id
        )
        gaussian_coords = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        mapped_feature_indices = _map_gt_to_gaussian_indices(gt_coords, gaussian_coords, radius_m)
        for variant, candidate_root in (
            ("2k", candidate_root_2k), ("10k", candidate_root_10k)
        ):
            paths = v4_candidate_run_paths(candidate_root, "uniform", scene_id, seed)
            capture = load_json(paths["candidate_json"])
            arrays = load_shadow_arrays(paths["candidate_labels"])
            branch, _ = map_gaussians_to_gt(
                gt_coords, gaussian_coords, arrays["branch_labels"], radius_m
            )
            semantic_codebook, _ = map_gaussians_to_gt(
                gt_coords, gaussian_coords, arrays["semantic_top1"], radius_m
            )
            sam_raw, _ = map_gaussians_to_gt(
                gt_coords, gaussian_coords, arrays["sam_covered"].astype(np.int64), radius_m
            )
            semantic = np.full_like(semantic_codebook, -1)
            for codebook_id, name in enumerate(capture["class_names"]):
                if name in taxonomy.canonical_classes:
                    semantic[semantic_codebook == codebook_id] = taxonomy.canonical_classes.index(name)
            feature_path = (
                _feature_ply(scene) if variant == "2k"
                else v4_feature_control_paths(feature_control_root, scene_id)["feature_ply"]
            )
            affinity = _load_affinity_features(feature_path, capture["affinity_gate"])
            affinity_metrics = _affinity_instance_metrics(
                gt_instances, mapped_feature_indices, affinity
            )
            candidates = {
                int(row["candidate_id"]): str(row["branch_class"])
                for row in capture["candidates"]
            }
            candidate_totals[variant] += len(candidates)
            for candidate_id, name in candidates.items():
                mask = branch == candidate_id
                best = max(
                    (
                        _iou(mask, np.asarray(gt["mask"], dtype=bool))
                        for gt in gt_instances if str(gt["canonical_class"]) == name
                    ), default=0.0,
                )
                candidate_positive[variant] += int(best >= 0.25)
            for gt in gt_instances:
                key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
                mask = np.asarray(gt["mask"], dtype=bool)
                best = max(
                    (
                        _iou(branch == candidate_id, mask)
                        for candidate_id, name in candidates.items()
                        if name == str(gt["canonical_class"])
                    ), default=0.0,
                )
                instances.append({
                    "variant": variant, "scene_id": scene_id,
                    "canonical_class": gt["canonical_class"],
                    "gt_instance_id": int(gt["gt_instance_id"]),
                    "physical_size_bin": gt["physical_size_bin"],
                    "sam_coverage": float(np.mean(sam_raw[mask] == 1)),
                    "semantic_top1_recall": float(np.mean(semantic[mask] == int(gt["class_id"]))),
                    "affinity_same_class_margin": affinity_metrics[key]["same_class_margin"],
                    "candidate_best_iou": float(best),
                })
    paired = {}
    for row in instances:
        key = (row["scene_id"], row["canonical_class"], row["gt_instance_id"])
        paired.setdefault(key, {})[row["variant"]] = row
    complete = [value for value in paired.values() if set(value) == {"2k", "10k"}]
    small = [
        pair for pair in complete
        if pair["2k"]["physical_size_bin"] in {"tiny", "small"}
    ]
    semantic_delta = float(np.mean([
        pair["10k"]["semantic_top1_recall"] - pair["2k"]["semantic_top1_recall"]
        for pair in complete
    ])) if complete else 0.0
    recall_delta = float(np.mean([
        float(pair["10k"]["candidate_best_iou"] >= 0.25)
        - float(pair["2k"]["candidate_best_iou"] >= 0.25)
        for pair in small
    ])) if small else 0.0
    added = [
        key for key, pair in paired.items()
        if pair["10k"]["candidate_best_iou"] >= 0.25
        and pair["2k"]["candidate_best_iou"] < 0.25
    ]
    losses_by_scene = {
        scene_id: sum(
            pair["2k"]["candidate_best_iou"] >= 0.25
            and pair["10k"]["candidate_best_iou"] < 0.25
            for pair in complete if pair["2k"]["scene_id"] == scene_id
        )
        for scene_id in scene_ids
    }
    improved = (
        (semantic_delta >= 0.05 or recall_delta >= 0.10)
        and len(added) >= 2
        and max(losses_by_scene.values(), default=0) <= 1
        and candidate_totals["10k"] <= 1.5 * max(candidate_totals["2k"], 1)
    )
    payload = {
        "kind": "v4_feature_10k_control", "schema_version": "1.0",
        "scene_ids": list(scene_ids), "seed": int(seed),
        "semantic_recall_mean_delta": semantic_delta,
        "tiny_small_recall_025_delta": recall_delta,
        "new_matches_025": len(added), "losses_by_scene": losses_by_scene,
        "candidate_counts": candidate_totals,
        "candidate_precision_025": {
            key: candidate_positive[key] / candidate_totals[key]
            if candidate_totals[key] else 0.0 for key in candidate_totals
        },
        "marked_improvement": bool(improved), "instances": instances,
        "interpretation": (
            "2k feature is a material limiting factor" if improved
            else "10k control did not meet the registered material-improvement gate"
        ),
    }
    write_json(output, payload)
    return payload
