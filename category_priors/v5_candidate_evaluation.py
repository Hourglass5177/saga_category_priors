from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v3_shadow_evaluation import _gaussian_ply, _gt_instances, _iou, _transform
from .v5_candidate import SOURCES, score_candidate
from .v5_candidate_runner import v5_candidate_run_paths


def _load_labels(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        return np.asarray(arrays["branch_labels"], dtype=np.int64)


def evaluate_v5_candidates(
    *, scene_manifest: str | Path, gt_dir: str | Path, output_root: str | Path,
    category_priors: str | Path, taxonomy: Taxonomy, size_bins: str | Path,
    scene_ids: Sequence[str], seed: int, table_output: str | Path,
    analysis_output: str | Path, radius_m: float = 0.05,
) -> dict[str, Any]:
    runtime = load_scene_runtime_manifest(scene_manifest)
    priors = load_json(category_priors)
    size_spec = load_json(size_bins)
    rows: list[dict[str, Any]] = []
    source_summary: dict[str, dict[str, Any]] = {}
    source_outputs: dict[tuple[str, str], np.ndarray] = {}
    commit: str | None = None
    for scene_id in scene_ids:
        scene = runtime[str(scene_id)]
        gt_xyz, gt = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", str(scene_id))
        gt_instances = _gt_instances(gt_xyz, gt.semantic, gt.instance, taxonomy, size_spec, str(scene_id))
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        for source in SOURCES:
            paths = v5_candidate_run_paths(output_root, source, str(scene_id), seed)
            bank = load_json(paths["proposals"])
            if bank.get("kind") != "v5_proposal_bank":
                raise ValueError(f"{paths['proposals']}: not a V5 proposal bank")
            current_commit = str(bank["git_commit"])
            if commit is None:
                commit = current_commit
            elif commit != current_commit:
                raise ValueError("V5 candidate captures use different commits")
            output = load_json(paths["output"])
            source_outputs[(source, str(scene_id))] = np.asarray(output["point_labels"], dtype=np.int64)
            branch = _load_labels(paths["proposal_labels"])
            mapped_branch, mapping = map_gaussians_to_gt(gt_xyz, gaussian_xyz, branch, radius_m)
            candidates = {int(item["candidate_id"]): dict(item) for item in bank.get("candidates", [])}
            for gt_instance in gt_instances:
                canonical = str(gt_instance["canonical_class"])
                mask = np.asarray(gt_instance["mask"], dtype=bool)
                matching = [
                    (candidate_id, candidate) for candidate_id, candidate in candidates.items()
                    if str(candidate["branch_class"]) == canonical
                ]
                best = max((_iou(mapped_branch == candidate_id, mask) for candidate_id, _ in matching), default=0.0)
                rows.append({
                    "row_type": "gt_instance", "source": source, "scene_id": str(scene_id),
                    "seed": int(seed), "git_commit": current_commit,
                    "canonical_class": canonical, "gt_instance_id": int(gt_instance["gt_instance_id"]),
                    "point_count": int(gt_instance["point_count"]),
                    "bbox_diag_m": float(gt_instance["bbox_diag_m"]),
                    "physical_size_bin": str(gt_instance["physical_size_bin"]),
                    "below_official_min_region_size": bool(gt_instance["below_official_min_region_size"]),
                    "same_class_best_iou": float(best), "mapped_fraction": float(mapping["mapped_fraction"]),
                })
            for candidate_id, candidate in candidates.items():
                mask = mapped_branch == candidate_id
                same = [
                    instance for instance in gt_instances
                    if str(instance["canonical_class"]) == str(candidate["branch_class"])
                ]
                best = max((_iou(mask, np.asarray(instance["mask"], dtype=bool)) for instance in same), default=0.0)
                uniform = score_candidate(candidate, priors, "U00-uniform")
                class_score = score_candidate(candidate, priors, "D11-combined")
                rows.append({
                    "row_type": "candidate", "source": source, "scene_id": str(scene_id),
                    "seed": int(seed), "git_commit": current_commit,
                    "candidate_id": candidate_id, "canonical_class": str(candidate["branch_class"]),
                    "candidate_points": int(mask.sum()), "same_class_best_iou": float(best),
                    "E": float(uniform["E"]), "uniform_E": float(uniform["E"]),
                    "uniform_G": float(uniform["G"]), "uniform_C": float(uniform["C"]),
                    "class_E": float(class_score["E"]), "class_G": float(class_score["G"]),
                    "class_C": float(class_score["C"]),
                    "hdbscan_persistence": candidate.get("hdbscan_persistence"),
                    "hdbscan_membership_mean": candidate.get("hdbscan_membership_mean"),
                    "assignment_confidence_mean": candidate.get("assignment_confidence_mean"),
                    "vote_branch_class_ratio": candidate.get("vote", {}).get("branch_class_ratio"),
                })

    write_rows(table_output, rows)
    gt_rows = [row for row in rows if row["row_type"] == "gt_instance"]
    candidate_rows = [row for row in rows if row["row_type"] == "candidate"]
    matches_050: dict[str, set[tuple[str, str, int]]] = {}
    for source in SOURCES:
        all_gt = [row for row in gt_rows if row["source"] == source]
        tiny_small = [
            row for row in all_gt
            if row["physical_size_bin"] in {"tiny", "small"}
            and not row["below_official_min_region_size"]
        ]
        candidates = [row for row in candidate_rows if row["source"] == source]
        positives_025 = [row for row in candidates if float(row["same_class_best_iou"]) >= 0.25]
        positives_050 = [row for row in candidates if float(row["same_class_best_iou"]) >= 0.50]
        matched = {
            (str(row["scene_id"]), str(row["canonical_class"]), int(row["gt_instance_id"]))
            for row in all_gt if float(row["same_class_best_iou"]) >= 0.50
        }
        matches_050[source] = matched
        source_summary[source] = {
            "candidate_count": len(candidates),
            "candidate_precision_025": len(positives_025) / len(candidates) if candidates else 0.0,
            "candidate_precision_050": len(positives_050) / len(candidates) if candidates else 0.0,
            "same_class_positive_050": len(positives_050),
            "positive_scene_count_050": len({row["scene_id"] for row in positives_050}),
            "official_tiny_small_count": len(tiny_small),
            "tiny_small_recall_025": float(np.mean([float(row["same_class_best_iou"]) >= 0.25 for row in tiny_small])) if tiny_small else 0.0,
            "tiny_small_recall_050": float(np.mean([float(row["same_class_best_iou"]) >= 0.50 for row in tiny_small])) if tiny_small else 0.0,
            "all_recall_050": float(np.mean([float(row["same_class_best_iou"]) >= 0.50 for row in all_gt])) if all_gt else 0.0,
        }
    decisions: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        other = next(value for value in SOURCES if value != source)
        added = matches_050[source] - matches_050[other]
        summary, comparator = source_summary[source], source_summary[other]
        relative = (
            summary["tiny_small_recall_050"] - comparator["tiny_small_recall_050"] >= 0.02
            or (len(added) >= 5 and len({item[0] for item in added}) >= 4)
        )
        source_gate = (
            relative
            and summary["candidate_precision_025"] >= 0.80 * comparator["candidate_precision_025"]
            and summary["candidate_count"] <= 1.5 * max(1, comparator["candidate_count"])
        )
        absolute = (
            summary["same_class_positive_050"] >= 12
            and summary["positive_scene_count_050"] >= 4
            and summary["candidate_precision_025"] >= 0.05
        )
        decisions[source] = {
            "source_selection_passed": bool(source_gate), "absolute_candidate_gate_passed": bool(absolute),
            "new_same_class_matches_050": len(added), "new_match_scene_count_050": len({item[0] for item in added}),
            "comparator": other,
        }
    selected = [source for source in SOURCES if decisions[source]["source_selection_passed"] and decisions[source]["absolute_candidate_gate_passed"]]
    selected.sort(key=lambda source: (
        source_summary[source]["tiny_small_recall_050"],
        decisions[source]["new_same_class_matches_050"],
        source_summary[source]["candidate_precision_025"],
        -source_summary[source]["candidate_count"],
    ), reverse=True)
    b1_unchanged = {
        scene_id: bool(np.array_equal(source_outputs[(SOURCES[0], scene_id)], source_outputs[(SOURCES[1], scene_id)]))
        for scene_id in map(str, scene_ids)
    }
    analysis = {
        "kind": "v5_candidate_source_analysis", "schema_version": "1.0", "git_commit": commit,
        "scene_count": len(scene_ids), "seed": int(seed), "sources": source_summary,
        "decisions": decisions, "selected_source": selected[0] if selected else None,
        "stage_b_passed": bool(selected), "b1_output_identical_between_sources": b1_unchanged,
    }
    write_json(analysis_output, analysis)
    return analysis
