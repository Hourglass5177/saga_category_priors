from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v3_shadow import load_shadow_arrays
from .v3_shadow_evaluation import (
    _final_predictions,
    _gaussian_ply,
    _gt_instances,
    _iou,
    _transform,
)
from .v4_candidate import MODES
from .v4_candidate_runner import v4_candidate_run_paths


def evaluate_v4_candidate_runs(
    *, scene_manifest_path: str | Path, gt_dir: str | Path,
    output_root: str | Path, taxonomy: Taxonomy, size_bins_path: str | Path,
    scene_ids: Sequence[str], seed: int, table_output: str | Path,
    analysis_output: str | Path, radius_m: float = 0.05,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    size_spec = load_json(size_bins_path)
    rows: list[dict[str, Any]] = []
    commit: str | None = None
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        gt_coords, gt_scene = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", scene_id)
        gt_instances = _gt_instances(
            gt_coords, gt_scene.semantic, gt_scene.instance, taxonomy, size_spec, scene_id
        )
        gaussian_coords = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        for mode in MODES:
            paths = v4_candidate_run_paths(output_root, mode, scene_id, seed)
            capture = load_json(paths["candidate_json"])
            current_commit = str(capture["git_commit"])
            if commit is None:
                commit = current_commit
            elif commit != current_commit:
                raise ValueError("V4 captures use different Git commits")
            output = load_json(paths["output"])
            final_labels, _ = map_gaussians_to_gt(
                gt_coords, gaussian_coords,
                np.asarray(output["point_labels"], dtype=np.int64), radius_m,
            )
            final_predictions = _final_predictions(final_labels, output, taxonomy)
            arrays = load_shadow_arrays(paths["candidate_labels"])
            mapped_branch, mapping = map_gaussians_to_gt(
                gt_coords, gaussian_coords, arrays["branch_labels"], radius_m
            )
            candidates = {
                int(row["candidate_id"]): dict(row) for row in capture["candidates"]
            }
            candidate_best: dict[tuple[int, int], float] = defaultdict(float)
            global_best: dict[tuple[int, int], float] = defaultdict(float)
            for gt in gt_instances:
                key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
                gt_mask = np.asarray(gt["mask"], dtype=bool)
                global_best[key] = max(
                    (
                        _iou(np.asarray(pred["mask"], dtype=bool), gt_mask)
                        for pred in final_predictions
                        if int(pred["class_id"]) == int(gt["class_id"])
                    ), default=0.0,
                )
                candidate_best[key] = max(
                    (
                        _iou(mapped_branch == candidate_id, gt_mask)
                        for candidate_id, candidate in candidates.items()
                        if str(candidate["branch_class"]) == str(gt["canonical_class"])
                    ), default=0.0,
                )
                rows.append({
                    "scene_id": scene_id, "seed": int(seed), "mode": mode,
                    "git_commit": current_commit,
                    "row_type": "gt_instance",
                    "canonical_class": gt["canonical_class"],
                    "gt_instance_id": int(gt["gt_instance_id"]),
                    "point_count": int(gt["point_count"]),
                    "bbox_diag_m": float(gt["bbox_diag_m"]),
                    "physical_size_bin": gt["physical_size_bin"],
                    "below_official_min_region_size": bool(gt["below_official_min_region_size"]),
                    "candidate_best_iou": float(candidate_best[key]),
                    "global_best_iou": float(global_best[key]),
                    "mapped_fraction": float(mapping["mapped_fraction"]),
                })
            for candidate_id, candidate in candidates.items():
                mask = mapped_branch == candidate_id
                same = [
                    gt for gt in gt_instances
                    if str(gt["canonical_class"]) == str(candidate["branch_class"])
                ]
                best_iou = max(
                    (_iou(mask, np.asarray(gt["mask"], dtype=bool)) for gt in same),
                    default=0.0,
                )
                rows.append({
                    "scene_id": scene_id, "seed": int(seed), "mode": mode,
                    "git_commit": current_commit, "row_type": "candidate",
                    "candidate_id": int(candidate_id),
                    "canonical_class": str(candidate["branch_class"]),
                    "candidate_points": int(mask.sum()),
                    "candidate_best_iou": float(best_iou),
                    "hdbscan_persistence": candidate.get("hdbscan_persistence"),
                    "vote_branch_class_ratio": candidate.get("vote", {}).get("branch_class_ratio"),
                    "vote_winner_matches_branch": candidate.get("vote", {}).get("winner_matches_branch"),
                })

    write_rows(table_output, rows)
    gt_rows = [row for row in rows if row["row_type"] == "gt_instance"]
    candidate_rows = [row for row in rows if row["row_type"] == "candidate"]
    summaries: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        small = [
            row for row in gt_rows
            if row["mode"] == mode
            and row["physical_size_bin"] in {"tiny", "small"}
            and not row["below_official_min_region_size"]
        ]
        mode_candidates = [row for row in candidate_rows if row["mode"] == mode]
        matched = [row for row in small if float(row["candidate_best_iou"]) >= 0.25]
        positive_candidates = [
            row for row in mode_candidates if float(row["candidate_best_iou"]) >= 0.25
        ]
        summaries[mode] = {
            "official_tiny_small_count": len(small),
            "tiny_small_recall_025": float(np.mean([
                float(row["candidate_best_iou"]) >= 0.25 for row in small
            ])) if small else 0.0,
            "tiny_small_recall_050": float(np.mean([
                float(row["candidate_best_iou"]) >= 0.50 for row in small
            ])) if small else 0.0,
            "candidate_count": len(mode_candidates),
            "candidate_precision_025": len(positive_candidates) / len(mode_candidates)
            if mode_candidates else 0.0,
            "matched_scene_count_025": len({row["scene_id"] for row in matched}),
            "matched_class_count_025": len({row["canonical_class"] for row in matched}),
        }
    uniform = summaries["uniform"]
    decisions = {}
    for mode in MODES[1:]:
        summary = summaries[mode]
        uniform_matches = {
            (row["scene_id"], row["canonical_class"], row["gt_instance_id"])
            for row in gt_rows
            if row["mode"] == "uniform"
            and row["physical_size_bin"] in {"tiny", "small"}
            and not row["below_official_min_region_size"]
            and float(row["candidate_best_iou"]) >= 0.25
        }
        mode_matches = {
            (row["scene_id"], row["canonical_class"], row["gt_instance_id"])
            for row in gt_rows
            if row["mode"] == mode
            and row["physical_size_bin"] in {"tiny", "small"}
            and not row["below_official_min_region_size"]
            and float(row["candidate_best_iou"]) >= 0.25
        }
        added = mode_matches - uniform_matches
        per_scene_delta = defaultdict(float)
        for row in gt_rows:
            if row["mode"] in {"uniform", mode} and row["physical_size_bin"] in {"tiny", "small"}:
                sign = 1.0 if row["mode"] == mode else -1.0
                per_scene_delta[row["scene_id"]] += sign * float(row["candidate_best_iou"] >= 0.25)
        positive_scenes = sum(value > 0 for value in per_scene_delta.values())
        negative_scenes = sum(value < 0 for value in per_scene_delta.values())
        passed = (
            summary["tiny_small_recall_025"] - uniform["tiny_small_recall_025"] >= 0.02
            and len(added) >= 5
            and len({key[1] for key in added}) >= 2
            and len({key[0] for key in added}) >= 4
            and summary["candidate_precision_025"] >= 0.8 * uniform["candidate_precision_025"]
            and summary["candidate_count"] <= 1.5 * max(uniform["candidate_count"], 1)
            and positive_scenes > negative_scenes
        )
        decisions[mode] = {
            "passed": bool(passed), "new_matches_025": len(added),
            "new_match_scene_count": len({key[0] for key in added}),
            "new_match_class_count": len({key[1] for key in added}),
            "positive_scenes": positive_scenes, "negative_scenes": negative_scenes,
        }
    passing = [mode for mode, decision in decisions.items() if decision["passed"]]
    passing.sort(key=lambda mode: (
        summaries[mode]["tiny_small_recall_025"],
        summaries[mode]["tiny_small_recall_050"],
        summaries[mode]["candidate_precision_025"],
        -MODES.index(mode),
    ), reverse=True)
    analysis = {
        "kind": "v4_candidate_analysis", "schema_version": "1.0",
        "git_commit": commit, "scene_count": len(scene_ids), "seed": int(seed),
        "summaries": summaries, "decisions": decisions,
        "best_candidate": passing[0] if passing else None,
        "stage_b_passed": bool(passing),
    }
    write_json(analysis_output, analysis)
    return analysis
