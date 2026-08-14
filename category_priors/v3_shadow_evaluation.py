from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v3_shadow import load_shadow_arrays
from .v3_shadow_runner import v3_shadow_run_paths
from .v3_stage0 import _oriented_bbox_diagonal, classify_physical_size


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    if scene.get("gaussian_ply"):
        path = Path(str(scene["gaussian_ply"]))
        if not path.is_absolute():
            path = Path(str(scene["base_path"])) / path
        return path
    base = Path(str(scene["base_path"]))
    directory = base / "output_models" / "point_cloud" / "iteration_30000"
    primary = directory / "point_cloud.ply"
    fallback = directory / "scene_point_cloud.ply"
    return primary if primary.is_file() else fallback


def _feature_ply(scene: Mapping[str, Any]) -> Path:
    return Path(str(scene["base_path"])) / "saga" / "contrastive_feature_point_cloud.ply"


def _load_affinity_features(path: str | Path, gate: Sequence[float]) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Affinity ceiling requires plyfile") from exc
    vertex = PlyData.read(str(path))["vertex"]
    names = sorted(
        (prop.name for prop in vertex.properties if prop.name.startswith("f_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    features = np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in names])
    gate_values = np.asarray(gate, dtype=np.float64)
    if features.shape[1] != gate_values.size:
        raise ValueError("Affinity gate dimension does not match feature PLY")
    features = features * gate_values[None, :]
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def _map_gt_to_gaussian_indices(
    gt_coords: np.ndarray, gaussian_coords: np.ndarray, radius_m: float
) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, indices = cKDTree(gaussian_coords).query(
        gt_coords, k=1, distance_upper_bound=radius_m, workers=-1
    )
    mapped = np.full(len(gt_coords), -1, dtype=np.int64)
    valid = np.isfinite(distances) & (indices < len(gaussian_coords))
    mapped[valid] = indices[valid]
    return mapped


def _affinity_instance_metrics(
    gt_instances: Sequence[Mapping[str, Any]],
    mapped_indices: np.ndarray,
    features: np.ndarray,
) -> dict[tuple[int, int], dict[str, float | None]]:
    centroids: dict[tuple[int, int], np.ndarray] = {}
    instance_features: dict[tuple[int, int], np.ndarray] = {}
    for gt in gt_instances:
        key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
        indices = mapped_indices[np.asarray(gt["mask"], dtype=bool)]
        indices = indices[indices >= 0]
        values = features[indices]
        instance_features[key] = values
        if len(values):
            centroid = values.mean(axis=0)
            centroids[key] = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    result: dict[tuple[int, int], dict[str, float | None]] = {}
    for key, values in instance_features.items():
        if not len(values) or key not in centroids:
            result[key] = {"intra_cosine": None, "nearest_same_class_cosine": None, "same_class_margin": None}
            continue
        intra = float(np.mean(values @ centroids[key]))
        others = [
            float(centroids[key] @ centroid)
            for other_key, centroid in centroids.items()
            if other_key != key and other_key[0] == key[0]
        ]
        nearest = max(others) if others else None
        result[key] = {
            "intra_cosine": intra,
            "nearest_same_class_cosine": nearest,
            "same_class_margin": intra - nearest if nearest is not None else None,
        }
    return result


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.sum(left & right))
    union = int(np.sum(left | right))
    return intersection / union if union else 0.0


def _gt_instances(
    coords: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    taxonomy: Taxonomy,
    size_spec: Mapping[str, Any],
    scene_id: str,
) -> list[dict[str, Any]]:
    result = []
    valid = (semantic >= 0) & (semantic < len(taxonomy.canonical_classes)) & (instance >= 0)
    for class_id, instance_id in sorted(set(zip(semantic[valid].tolist(), instance[valid].tolist()))):
        mask = valid & (semantic == class_id) & (instance == instance_id)
        diagonal = _oriented_bbox_diagonal(coords[mask])
        result.append({
            "scene_id": scene_id,
            "class_id": int(class_id),
            "canonical_class": taxonomy.canonical_classes[int(class_id)],
            "gt_instance_id": int(instance_id),
            "mask": mask,
            "point_count": int(mask.sum()),
            "bbox_diag_m": diagonal,
            "physical_size_bin": classify_physical_size(diagonal, size_spec),
            "below_official_min_region_size": int(mask.sum()) < 100,
        })
    return result


def _final_predictions(
    mapped_labels: np.ndarray,
    output: Mapping[str, Any],
    taxonomy: Taxonomy,
) -> list[dict[str, Any]]:
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    result = []
    for raw_id, values in output.get("instances", {}).items():
        class_name = str(values.get("class", "")).strip().lower()
        if class_name not in class_to_id:
            continue
        instance_id = int(raw_id)
        result.append({
            "instance_id": instance_id,
            "class_id": class_to_id[class_name],
            "mask": mapped_labels == instance_id,
        })
    return result


def evaluate_shadow_scene_arrays(
    *,
    scene_id: str,
    mode: str,
    gt_instances: Sequence[Mapping[str, Any]],
    mapped_branch_labels: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    final_predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], float]]:
    global_best: dict[tuple[int, int], float] = {}
    for gt in gt_instances:
        key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
        global_best[key] = max(
            (
                _iou(np.asarray(pred["mask"], dtype=bool), np.asarray(gt["mask"], dtype=bool))
                for pred in final_predictions
                if int(pred["class_id"]) == int(gt["class_id"])
            ),
            default=0.0,
        )

    rows = []
    for raw in candidates:
        candidate = dict(raw)
        candidate_id = int(candidate["candidate_id"])
        branch_class = str(candidate["branch_class"])
        mask = mapped_branch_labels == candidate_id
        same = [
            gt for gt in gt_instances
            if str(gt["canonical_class"]) == branch_class
        ]
        same_scores = [_iou(mask, np.asarray(gt["mask"], dtype=bool)) for gt in same]
        any_scores = [_iou(mask, np.asarray(gt["mask"], dtype=bool)) for gt in gt_instances]
        same_index = int(np.argmax(same_scores)) if same_scores else None
        any_index = int(np.argmax(any_scores)) if any_scores else None
        best_same = same[same_index] if same_index is not None else None
        same_iou = float(same_scores[same_index]) if same_index is not None else 0.0
        any_iou = float(any_scores[any_index]) if any_index is not None else 0.0
        global_iou = (
            global_best[(int(best_same["class_id"]), int(best_same["gt_instance_id"]))]
            if best_same is not None else 0.0
        )
        vote = dict(candidate.pop("vote", {}))
        pre = dict(candidate.pop("global_pre_overlap", {}))
        final = dict(candidate.pop("global_final_overlap", {}))
        if int(candidate.get("active_branch_points", 0)) == 0:
            death = "overwritten_before_active_branch"
        elif int(candidate.get("after_knn_points", 0)) == 0:
            death = "global_knn"
        elif int(candidate.get("after_filter_points", 0)) == 0:
            death = "filter_num"
        elif not bool(vote.get("winner_matches_branch", False)):
            death = "2d_vote_or_background"
        else:
            death = "survives_registered_funnel"
        row = {
            "scene_id": scene_id,
            "mode": mode,
            **candidate,
            "mapped_candidate_points": int(mask.sum()),
            "same_class_best_iou": same_iou,
            "any_class_best_iou": any_iou,
            "matched_gt_class": best_same["canonical_class"] if best_same else None,
            "matched_gt_instance_id": int(best_same["gt_instance_id"]) if best_same else None,
            "matched_gt_point_count": int(best_same["point_count"]) if best_same else None,
            "matched_gt_bbox_diag_m": float(best_same["bbox_diag_m"]) if best_same else None,
            "matched_gt_size_bin": best_same["physical_size_bin"] if best_same else None,
            "matched_gt_below_100": bool(best_same["below_official_min_region_size"]) if best_same else None,
            "global_same_gt_best_iou": global_iou,
            "new_oracle_match_025": same_iou >= 0.25 and global_iou < 0.25,
            "new_oracle_match_050": same_iou >= 0.50 and global_iou < 0.50,
            "death_stage": death,
            "vote_branch_class_ratio": vote.get("branch_class_ratio"),
            "vote_winner": vote.get("winner"),
            "vote_winner_ratio": vote.get("winner_ratio"),
            "vote_background_ratio": vote.get("background_ratio"),
            "vote_winner_matches_branch": vote.get("winner_matches_branch"),
            "global_pre_overlap_fraction": pre.get("fraction"),
            "global_final_overlap_fraction": final.get("fraction"),
        }
        rows.append(row)
    return rows, global_best


def evaluate_v3_shadow_runs(
    *,
    scene_manifest_path: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    taxonomy: Taxonomy,
    size_bins_path: str | Path,
    scene_ids: Sequence[str],
    seed: int,
    funnel_output: str | Path,
    input_ceiling_output: str | Path,
    analysis_output: str | Path,
    radius_m: float = 0.05,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    size_spec = load_json(size_bins_path)
    class_to_id = {
        name: index for index, name in enumerate(taxonomy.canonical_classes)
    }
    all_rows: list[dict[str, Any]] = []
    ceiling_instances: list[dict[str, Any]] = []
    mapping_diagnostics = []
    evaluation_commit: str | None = None
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        gt_coords, gt_scene = load_ground_truth_npz(Path(gt_dir) / f"{scene_id}.npz", scene_id)
        gt_instances = _gt_instances(
            gt_coords, gt_scene.semantic, gt_scene.instance, taxonomy, size_spec, scene_id
        )
        gaussian_coords = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        paths = v3_shadow_run_paths(output_root, scene_id, seed)
        output = load_json(paths["output"])
        mapped_final, final_diag = map_gaussians_to_gt(
            gt_coords, gaussian_coords, np.asarray(output["point_labels"], dtype=np.int64), radius_m
        )
        final_predictions = _final_predictions(mapped_final, output, taxonomy)
        per_mode_best: dict[str, dict[tuple[int, int], float]] = {}
        semantic_mapped = None
        sam_mapped = None
        for mode in ("exact", "exclusive"):
            capture = load_json(paths[f"{mode}_json"])
            capture_commit = str(capture["git_commit"])
            if evaluation_commit is None:
                evaluation_commit = capture_commit
            elif capture_commit != evaluation_commit:
                raise ValueError("shadow captures use different Git commits")
            if int(capture["seed"]) != int(seed) or capture["scene_id"] != scene_id:
                raise ValueError("shadow capture identity does not match evaluation request")
            arrays = load_shadow_arrays(paths[f"{mode}_labels"])
            mapped_branch, branch_diag = map_gaussians_to_gt(
                gt_coords, gaussian_coords, arrays["branch_labels"], radius_m
            )
            if semantic_mapped is None:
                semantic_codebook, _ = map_gaussians_to_gt(
                    gt_coords, gaussian_coords, arrays["semantic_top1"], radius_m
                )
                semantic_mapped = np.full_like(semantic_codebook, -1)
                for codebook_id, class_name in enumerate(capture["class_names"]):
                    if class_name in class_to_id:
                        semantic_mapped[semantic_codebook == codebook_id] = class_to_id[class_name]
                sam_mapped_raw, _ = map_gaussians_to_gt(
                    gt_coords, gaussian_coords, arrays["sam_covered"].astype(np.int64), radius_m
                )
                sam_mapped = sam_mapped_raw == 1
            rows, global_best = evaluate_shadow_scene_arrays(
                scene_id=scene_id,
                mode=mode,
                gt_instances=gt_instances,
                mapped_branch_labels=mapped_branch,
                candidates=capture["candidates"],
                final_predictions=final_predictions,
            )
            for row in rows:
                row["seed"] = int(seed)
                row["git_commit"] = capture_commit
            all_rows.extend(rows)
            candidate_classes = {
                int(row["candidate_id"]): str(row["branch_class"])
                for row in capture["candidates"]
            }
            candidate_best = defaultdict(float)
            for gt in gt_instances:
                key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
                gt_mask = np.asarray(gt["mask"], dtype=bool)
                candidate_best[key] = max(
                    (
                        _iou(mapped_branch == candidate_id, gt_mask)
                        for candidate_id, class_name in candidate_classes.items()
                        if class_name == str(gt["canonical_class"])
                    ),
                    default=0.0,
                )
            per_mode_best[mode] = dict(candidate_best)
            mapping_diagnostics.append({"scene_id": scene_id, "mode": mode, **branch_diag})
        exact_capture = load_json(paths["exact_json"])
        affinity_features = _load_affinity_features(
            _feature_ply(scene), exact_capture["affinity_gate"]
        )
        if len(affinity_features) != len(gaussian_coords):
            raise ValueError(
                f"{scene_id}: affinity feature count {len(affinity_features)} does not "
                f"match Gaussian count {len(gaussian_coords)}"
            )
        mapped_feature_indices = _map_gt_to_gaussian_indices(
            gt_coords, gaussian_coords, radius_m
        )
        affinity_metrics = _affinity_instance_metrics(
            gt_instances, mapped_feature_indices, affinity_features
        )
        for gt in gt_instances:
            key = (int(gt["class_id"]), int(gt["gt_instance_id"]))
            mask = np.asarray(gt["mask"], dtype=bool)
            affinity = affinity_metrics[key]
            ceiling_instances.append({
                "scene_id": scene_id,
                "seed": int(seed),
                "git_commit": evaluation_commit,
                "canonical_class": gt["canonical_class"],
                "gt_instance_id": gt["gt_instance_id"],
                "point_count": gt["point_count"],
                "bbox_diag_m": gt["bbox_diag_m"],
                "physical_size_bin": gt["physical_size_bin"],
                "below_official_min_region_size": gt["below_official_min_region_size"],
                "sam_mask_coverage": float(np.mean(sam_mapped[mask])),
                "semantic_top1_recall": float(np.mean(semantic_mapped[mask] == int(gt["class_id"]))),
                "affinity_intra_cosine": affinity["intra_cosine"],
                "affinity_nearest_same_class_cosine": affinity["nearest_same_class_cosine"],
                "affinity_same_class_margin": affinity["same_class_margin"],
                "global_best_iou": max(
                    (_iou(np.asarray(pred["mask"], dtype=bool), mask) for pred in final_predictions if int(pred["class_id"]) == int(gt["class_id"])),
                    default=0.0,
                ),
                "exact_candidate_best_iou": per_mode_best["exact"].get(key, 0.0),
                "exclusive_candidate_best_iou": per_mode_best["exclusive"].get(key, 0.0),
            })

    write_rows(funnel_output, all_rows)
    small = [row for row in ceiling_instances if row["physical_size_bin"] in {"tiny", "small"}]
    mode_summary = {}
    for mode in ("exact", "exclusive"):
        field = f"{mode}_candidate_best_iou"
        new = [
            row for row in small
            if float(row[field]) >= 0.25 and float(row["global_best_iou"]) < 0.25
        ]
        candidate_recall = float(np.mean([float(row[field]) >= 0.25 for row in small])) if small else 0.0
        global_recall = float(np.mean([float(row["global_best_iou"]) >= 0.25 for row in small])) if small else 0.0
        mode_summary[mode] = {
            "small_instance_count": len(small),
            "small_oracle_recall_025": candidate_recall,
            "global_small_recall_025": global_recall,
            "recall_difference": candidate_recall - global_recall,
            "new_small_matches_025": len(new),
            "new_match_scene_count": len({row["scene_id"] for row in new}),
            "new_match_class_count": len({row["canonical_class"] for row in new}),
        }
    exclusive = mode_summary["exclusive"]
    gate_passed = (
        exclusive["recall_difference"] >= 0.02
        or (
            exclusive["new_small_matches_025"] >= 5
            and exclusive["new_match_class_count"] >= 2
            and exclusive["new_match_scene_count"] >= 4
        )
    )
    input_payload = {
        "kind": "v3_input_ceiling",
        "schema_version": "1.0",
        "git_commit": evaluation_commit,
        "scene_count": len(scene_ids),
        "seed": int(seed),
        "instances": ceiling_instances,
        "mapping_diagnostics": mapping_diagnostics,
    }
    analysis = {
        "kind": "v3_proposal_oracle_analysis",
        "schema_version": "1.0",
        "git_commit": evaluation_commit,
        "scene_count": len(scene_ids),
        "seed": int(seed),
        "candidate_count": len(all_rows),
        "mode_summary": mode_summary,
        "death_stage_counts": {
            mode: dict(sorted(Counter(row["death_stage"] for row in all_rows if row["mode"] == mode).items()))
            for mode in ("exact", "exclusive")
        },
        "stage1_minimum_information_gate": {
            "passed": gate_passed,
            "rule": "exclusive small recall delta >=0.02 OR >=5 new IoU25 matches across >=2 classes and >=4 scenes",
            "next_action": "choose_one_protection_from_death_type" if gate_passed else "extend_shadow_to_remaining16",
        },
    }
    write_json(input_ceiling_output, input_payload)
    write_json(analysis_output, analysis)
    return analysis
