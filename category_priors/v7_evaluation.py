from __future__ import annotations

"""Official and precision-first evaluation for immutable V7 object banks."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .evaluator import (
    GroundTruthScene,
    PredictedInstance,
    apply_transform,
    evaluate_instances,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from .gaussian_object_audit import (
    _export_viewer_case,
    _select_viewer_cases,
    evaluate_gaussian_object_precision,
)
from .io import load_json, write_json, write_rows
from .taxonomy import Taxonomy, load_taxonomy


def _runtime_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("scenes", payload)
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    return {str(row["scene_id"]): dict(row) for row in rows}


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    explicit = scene.get("gaussian_ply")
    if explicit:
        result = Path(str(explicit))
        return result if result.is_absolute() else Path(str(scene["base_path"])) / result
    root = Path(str(scene["base_path"])) / "output_models/point_cloud/iteration_30000"
    # Registered automatic-evaluation assets use scene_point_cloud.ply; its
    # row order is the one checked against the SAGA feature PLY by V8 worker.
    # Falling back to point_cloud.ply is only for older assets that do not
    # contain the registered scene file.
    primary = root / "scene_point_cloud.ply"
    return primary if primary.is_file() else root / "point_cloud.ply"


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    )


def _gaussian_opacity(path: Path) -> np.ndarray:
    """Load physical opacity weights; fall back to unit weights for old PLYs."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    if "opacity" not in (vertex.dtype.names or ()):
        return np.ones(len(vertex), dtype=np.float64)
    logits = np.asarray(vertex["opacity"], dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _opacity_precision(
    audit: Mapping[str, Any], point_labels: np.ndarray, opacity: np.ndarray
) -> tuple[float, float, float]:
    predicted_weight = 0.0
    correct_weight = 0.0
    instance_values: list[float] = []
    for row in audit["instances"]:
        instance_id = int(row["instance_id"])
        selected = np.flatnonzero(point_labels == instance_id)
        categories = np.asarray(audit["point_categories"][instance_id], dtype=np.int8)
        weights = opacity[selected]
        total = float(weights.sum())
        correct = float(weights[categories == 0].sum())
        predicted_weight += total
        correct_weight += correct
        instance_values.append(correct / total if total else 0.0)
    return (
        correct_weight,
        predicted_weight,
        float(np.mean(instance_values)) if instance_values else 0.0,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def _bbox_diagonal(coords: np.ndarray) -> float:
    points = np.asarray(coords, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)))


def _size_bin(diagonal: float, spec: Mapping[str, Any] | None) -> str | None:
    if spec is None:
        return None
    limits = spec["boundaries_m"]
    if diagonal <= float(limits["tiny_max_m"]):
        return "tiny"
    if diagonal <= float(limits["small_max_m"]):
        return "small"
    if diagonal <= float(limits["medium_max_m"]):
        return "medium"
    return "large"


def evaluate_v7_bank(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    rows_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
) -> dict[str, Any]:
    scenes = _runtime_rows(runtime_manifest)
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    size_spec = load_json(size_bins) if size_bins else None
    rows: list[dict[str, Any]] = []
    gt_support: dict[tuple[str, int, int], dict[str, Any]] = {}
    oracle_single_best: dict[tuple[str, int, int], float] = {}
    oracle_association_best: dict[tuple[str, int, int], float] = {}
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        gt_coords, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_coords = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        bank = load_json(bank_root / scene_id / "object_bank.json")
        with np.load(bank_root / scene_id / "object_bank.npz", allow_pickle=False) as arrays:
            gaussian_candidate_labels = np.asarray(arrays["candidate_labels"], dtype=np.int64)
            fragment_indptr = np.asarray(arrays["fragment_full_indptr"], dtype=np.int64)
            fragment_ids = np.asarray(arrays["fragment_full_ids"], dtype=np.int64)
            fragment_classes = np.asarray(arrays["fragment_class"], dtype=np.int64)
        mapped_candidates, mapping = map_gaussians_to_gt(
            gt_coords, gaussian_coords, gaussian_candidate_labels, radius_m
        )
        gt_distances, gt_gaussian_indices = cKDTree(gaussian_coords).query(
            gt_coords, k=1, distance_upper_bound=radius_m, workers=-1
        )
        gt_gaussian_indices = np.asarray(gt_gaussian_indices, dtype=np.int64)
        valid_gt_gaussian = np.isfinite(gt_distances) & (gt_gaussian_indices < len(gaussian_coords))
        valid_gt = (gt.semantic >= 0) & (gt.instance >= 0)
        for class_id, instance_id in sorted(set(zip(
            gt.semantic[valid_gt].tolist(), gt.instance[valid_gt].tolist()
        ))):
            mask = valid_gt & (gt.semantic == class_id) & (gt.instance == instance_id)
            diagonal = _bbox_diagonal(gt_coords[mask])
            gt_support[(scene_id, int(class_id), int(instance_id))] = {
                "mask": mask,
                "point_count": int(mask.sum()),
                "size_bin": _size_bin(diagonal, size_spec),
            }
        fragment_mapped_masks: list[np.ndarray] = []
        fragment_best_gt: list[tuple[str, int, int] | None] = []
        for fragment_index, class_code in enumerate(fragment_classes):
            selected = fragment_ids[fragment_indptr[fragment_index]:fragment_indptr[fragment_index + 1]]
            selected_mask = np.zeros(len(gaussian_coords), dtype=bool)
            selected_mask[selected] = True
            fragment_mask = np.zeros(len(gt_coords), dtype=bool)
            fragment_mask[valid_gt_gaussian] = selected_mask[
                gt_gaussian_indices[valid_gt_gaussian]
            ]
            class_name = bank["classes"][int(class_code)] if 0 <= class_code < len(bank["classes"]) else ""
            class_id = class_to_id.get(class_name, -1)
            matches = [
                (_iou(fragment_mask, support["mask"]), key)
                for key, support in gt_support.items()
                if key[0] == scene_id and key[1] == class_id
            ]
            matches.sort(key=lambda item: (-item[0], item[1][2]))
            best_key = matches[0][1] if matches and matches[0][0] > 0 else None
            fragment_mapped_masks.append(fragment_mask)
            fragment_best_gt.append(best_key)
            if best_key is not None:
                oracle_single_best[best_key] = max(
                    oracle_single_best.get(best_key, 0.0), float(matches[0][0])
                )
        grouped: dict[tuple[str, int, int], np.ndarray] = {}
        for fragment_mask, key in zip(fragment_mapped_masks, fragment_best_gt):
            if key is not None:
                grouped[key] = grouped.get(key, np.zeros(len(gt_coords), dtype=bool)) | fragment_mask
        for key, mask in grouped.items():
            oracle_association_best[key] = max(
                oracle_association_best.get(key, 0.0), _iou(mask, gt_support[key]["mask"])
            )
        for candidate in bank.get("candidates", []):
            candidate_id = int(candidate["candidate_id"])
            class_name = str(candidate["branch_class"])
            class_id = class_to_id.get(class_name, -1)
            candidate_mask = mapped_candidates == candidate_id
            matches = []
            if class_id >= 0:
                for key, support in gt_support.items():
                    if key[0] == scene_id and key[1] == class_id:
                        matches.append((
                            _iou(candidate_mask, support["mask"]), key[2], support
                        ))
            matches.sort(key=lambda item: (-item[0], item[1]))
            best_iou, best_instance, support = matches[0] if matches else (0.0, None, None)
            rows.append({
                "scene_id": scene_id,
                "candidate_id": candidate_id,
                "class": class_name,
                "base_score": float(candidate["base_score"]),
                "same_class_best_iou": float(best_iou),
                "matched_gt_instance": best_instance,
                "matched_gt_point_count": support["point_count"] if support else None,
                "matched_gt_size_bin": support["size_bin"] if support else None,
                "match_025": bool(best_iou >= 0.25),
                "match_050": bool(best_iou >= 0.50),
                "mapped_candidate_points": int(np.count_nonzero(candidate_mask)),
                "mapping_fraction": float(mapping["mapped_fraction"]),
                "core_point_count": int(candidate["core_point_count"]),
                "full_point_count": int(candidate["full_point_count"]),
                "effective_view_count": int(candidate["effective_view_count"]),
            })

    best_by_gt: dict[tuple[str, int, int], float] = {
        key: 0.0 for key, support in gt_support.items() if support["point_count"] >= 100
    }
    for row in rows:
        if row["matched_gt_instance"] is None:
            continue
        class_id = class_to_id.get(str(row["class"]), -1)
        key = (str(row["scene_id"]), class_id, int(row["matched_gt_instance"]))
        if key in best_by_gt:
            best_by_gt[key] = max(best_by_gt[key], float(row["same_class_best_iou"]))
    tiny_small = [
        key for key in best_by_gt
        if gt_support[key]["size_bin"] in {"tiny", "small"}
    ]
    valid_gt_keys = list(best_by_gt)
    oracle_tiny_small = [
        key for key in valid_gt_keys
        if gt_support[key]["size_bin"] in {"tiny", "small"}
    ]
    scores = np.asarray([row["base_score"] for row in rows], dtype=np.float64)
    ious = np.asarray([row["same_class_best_iou"] for row in rows], dtype=np.float64)
    correlation = float(spearmanr(scores, ious).statistic) if len(rows) >= 2 else 0.0
    if not np.isfinite(correlation):
        correlation = 0.0
    analysis = {
        "schema": "saga-v7-bank-analysis-v1",
        "scene_count": len(scene_ids),
        "candidate_count": len(rows),
        "match_025_count": sum(int(row["match_025"]) for row in rows),
        "match_050_count": sum(int(row["match_050"]) for row in rows),
        "candidate_precision_025": (
            sum(int(row["match_025"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "official_valid_gt_count": len(best_by_gt),
        "recall_025": (
            sum(value >= 0.25 for value in best_by_gt.values()) / len(best_by_gt)
            if best_by_gt else 0.0
        ),
        "recall_050": (
            sum(value >= 0.50 for value in best_by_gt.values()) / len(best_by_gt)
            if best_by_gt else 0.0
        ),
        "tiny_small_official_gt_count": len(tiny_small),
        "tiny_small_recall_025": (
            sum(best_by_gt[key] >= 0.25 for key in tiny_small) / len(tiny_small)
            if tiny_small else 0.0
        ),
        "tiny_small_recall_050": (
            sum(best_by_gt[key] >= 0.50 for key in tiny_small) / len(tiny_small)
            if tiny_small else 0.0
        ),
        "score_iou_spearman": correlation,
        "positive_050_scene_count": len({
            row["scene_id"] for row in rows if row["match_050"]
        }),
        "oracles": {
            "single_mask_match_050_count": sum(
                oracle_single_best.get(key, 0.0) >= 0.50 for key in valid_gt_keys
            ),
            "association_match_050_count": sum(
                oracle_association_best.get(key, 0.0) >= 0.50 for key in valid_gt_keys
            ),
            "association_tiny_small_recall_025": (
                sum(oracle_association_best.get(key, 0.0) >= 0.25 for key in oracle_tiny_small)
                / len(oracle_tiny_small) if oracle_tiny_small else 0.0
            ),
        },
    }
    write_rows(rows_output, rows)
    write_json(analysis_output, analysis)
    return analysis


def evaluate_v7_replays(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    replay_root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    viewer_output: Path | None = None,
    size_bins: Path | None = None,
) -> dict[str, Any]:
    scenes = _runtime_rows(runtime_manifest)
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    size_spec = load_json(size_bins) if size_bins else None
    metrics_rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {"schema": "saga-v7-analysis-v1", "conditions": {}}
    viewer_rows: list[dict[str, Any]] = []
    viewer_audits: dict[tuple[str, str], dict[str, Any]] = {}
    viewer_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for condition in conditions:
        all_gt: list[GroundTruthScene] = []
        all_predictions: list[PredictedInstance] = []
        precision_aggregates: list[dict[str, Any]] = []
        radius_aggregates: dict[str, list[dict[str, Any]]] = {"0.02": [], "0.05": [], "0.10": []}
        opacity_correct = 0.0
        opacity_total = 0.0
        opacity_instance_macro: list[float] = []
        tiny_small_total = 0
        tiny_small_hit_025 = 0
        tiny_small_hit_050 = 0
        official_gt_total = 0
        official_gt_hit_025 = 0
        per_scene: list[dict[str, Any]] = []
        for scene_id in scene_ids:
            scene = scenes[scene_id]
            gt_coords, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
            output = load_json(replay_root / condition / scene_id / "output.json")
            diagnostics = load_json(replay_root / condition / scene_id / "diagnostics.json")
            gaussian_labels = np.asarray(output["point_labels"], dtype=np.int64)
            gaussian_coords = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
            gaussian_opacity = _gaussian_opacity(_gaussian_ply(scene))
            mapped_labels, _ = map_gaussians_to_gt(
                gt_coords, gaussian_coords, gaussian_labels, radius_m
            )
            predictions: list[PredictedInstance] = []
            metadata = diagnostics.get("instances", output.get("instances", {}))
            for raw_id, values in output.get("instances", {}).items():
                class_name = str(values.get("class", ""))
                if class_name not in class_to_id:
                    continue
                instance_id = int(raw_id)
                meta = metadata.get(str(instance_id), {})
                predictions.append(PredictedInstance(
                    scene_id=scene_id,
                    instance_id=instance_id,
                    class_id=class_to_id[class_name],
                    score=float(meta.get("score", 1.0)),
                    mask=mapped_labels == instance_id,
                ))
            scene_result = evaluate_instances(
                [gt], predictions, taxonomy.canonical_classes,
                min_region_size=min_region_size,
            )
            scene_tiny_total = 0
            scene_tiny_hit_025 = 0
            scene_tiny_hit_050 = 0
            scene_official_total = 0
            scene_official_hit_025 = 0
            valid_gt = (gt.semantic >= 0) & (gt.instance >= 0)
            for class_id, instance_id in sorted(set(zip(
                gt.semantic[valid_gt].tolist(), gt.instance[valid_gt].tolist()
            ))):
                gt_mask = valid_gt & (gt.semantic == class_id) & (gt.instance == instance_id)
                if int(gt_mask.sum()) < min_region_size:
                    continue
                same_class = [
                    item.mask for item in predictions
                    if item.class_id == int(class_id)
                ]
                best_iou = max(
                    (_iou(gt_mask, mask) for mask in same_class), default=0.0
                )
                scene_official_total += 1
                scene_official_hit_025 += int(best_iou >= 0.25)
                if _size_bin(_bbox_diagonal(gt_coords[gt_mask]), size_spec) not in {"tiny", "small"}:
                    continue
                scene_tiny_total += 1
                scene_tiny_hit_025 += int(best_iou >= 0.25)
                scene_tiny_hit_050 += int(best_iou >= 0.50)
            tiny_small_total += scene_tiny_total
            tiny_small_hit_025 += scene_tiny_hit_025
            tiny_small_hit_050 += scene_tiny_hit_050
            official_gt_total += scene_official_total
            official_gt_hit_025 += scene_official_hit_025
            per_scene.append({
                "scene_id": scene_id, **scene_result["aggregate"],
                "tiny_small_official_gt_count": scene_tiny_total,
                "tiny_small_recall_025": scene_tiny_hit_025 / scene_tiny_total if scene_tiny_total else 0.0,
                "tiny_small_recall_050": scene_tiny_hit_050 / scene_tiny_total if scene_tiny_total else 0.0,
                "official_gt_count": scene_official_total,
                "official_gt_recall_025": (
                    scene_official_hit_025 / scene_official_total
                    if scene_official_total else 0.0
                ),
            })
            audit = evaluate_gaussian_object_precision(
                gaussian_coords, gaussian_labels, output.get("instances", {}),
                gt_coords, gt.semantic, gt.instance, radius_m,
                canonical_classes=taxonomy.canonical_classes,
            )
            precision_aggregates.append(audit["aggregate"])
            for diagnostic_radius in (0.02, 0.05, 0.10):
                radius_audit = audit if diagnostic_radius == radius_m else evaluate_gaussian_object_precision(
                    gaussian_coords, gaussian_labels, output.get("instances", {}),
                    gt_coords, gt.semantic, gt.instance, diagnostic_radius,
                    canonical_classes=taxonomy.canonical_classes,
                )
                radius_aggregates[f"{diagnostic_radius:.2f}"].append(radius_audit["aggregate"])
            weighted_correct, weighted_total, weighted_macro = _opacity_precision(
                audit, gaussian_labels, gaussian_opacity
            )
            opacity_correct += weighted_correct
            opacity_total += weighted_total
            opacity_instance_macro.append(weighted_macro)
            if viewer_output is not None:
                viewer_audits[(scene_id, condition)] = {**audit, "point_labels": gaussian_labels}
                viewer_arrays[scene_id] = (gt_coords, gt.semantic, gt.instance, gaussian_coords)
                viewer_rows.extend({"scene_id": scene_id, "condition": condition, **item} for item in audit["instances"])
            all_gt.append(gt)
            all_predictions.extend(predictions)
        official = evaluate_instances(
            all_gt, all_predictions, taxonomy.canonical_classes,
            min_region_size=min_region_size,
        )
        aggregate = official["aggregate"]
        total_gaussians = sum(row["predicted_gaussian_count"] for row in precision_aggregates)
        total_correct = sum(row["correct_gaussian_count"] for row in precision_aggregates)
        total_unsupported_instances = sum(row["unsupported_prediction_count"] for row in precision_aggregates)
        total_instances = sum(row["predicted_instance_count"] for row in precision_aggregates)
        row = {
            "condition": condition,
            "scene_count": len(scene_ids),
            **aggregate,
            "predicted_instance_count": sum(row["predicted_instance_count"] for row in precision_aggregates),
            "gaussian_micro_precision": total_correct / total_gaussians if total_gaussians else 0.0,
            "opacity_weighted_micro_precision": opacity_correct / opacity_total if opacity_total else 0.0,
            "opacity_weighted_instance_macro_precision": float(np.mean(opacity_instance_macro)) if opacity_instance_macro else 0.0,
            "unsupported_instance_fraction": (
                total_unsupported_instances / total_instances if total_instances else 0.0
            ),
            "mean_matched_gt_recall": float(np.mean([
                item["mean_matched_gt_recall"] for item in precision_aggregates
            ])) if precision_aggregates else 0.0,
            "official_gt_recall_025": (
                official_gt_hit_025 / official_gt_total
                if official_gt_total else 0.0
            ),
            "mean_scene_map_50_95": float(np.mean([item["map_50_95"] for item in per_scene])),
            "tiny_small_official_gt_count": tiny_small_total,
            "tiny_small_recall_025": tiny_small_hit_025 / tiny_small_total if tiny_small_total else 0.0,
            "tiny_small_recall_050": tiny_small_hit_050 / tiny_small_total if tiny_small_total else 0.0,
            "positive_scene_count": None,
        }
        metrics_rows.append(row)
        analysis["conditions"][condition] = {
            "official": official,
            "precision": row,
            "per_scene": per_scene,
            "radius_sensitivity": {
                key: {
                    "gaussian_micro_precision": (
                        sum(item["correct_gaussian_count"] for item in values)
                        / max(sum(item["predicted_gaussian_count"] for item in values), 1)
                    ),
                    "unsupported_instance_fraction": (
                        sum(item["unsupported_prediction_count"] for item in values)
                        / max(sum(item["predicted_instance_count"] for item in values), 1)
                    ),
                }
                for key, values in radius_aggregates.items()
            },
        }
    if viewer_output is not None:
        selected = _select_viewer_cases(viewer_rows)
        cases = []
        for case in selected:
            scene_id = str(case["scene_id"])
            condition = str(case["condition"])
            gt_xyz, gt_semantic, gt_instance, gaussian_xyz = viewer_arrays[scene_id]
            current = viewer_audits[(scene_id, condition)]
            cases.append(_export_viewer_case(
                case, current, gt_xyz, gt_semantic, gt_instance, gaussian_xyz,
                np.asarray(current["point_labels"], dtype=np.int64), viewer_output,
            ))
        analysis["viewer"] = {"directory": str(viewer_output), "cases": cases}
    write_rows(metrics_output, metrics_rows)
    write_json(analysis_output, analysis)
    return analysis


def main_bank(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate V7 candidate banks")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes", required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--size-bins", type=Path)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path)
    args = parser.parse_args(argv)
    evaluate_v7_bank(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        bank_root=args.bank_root, scene_ids=args.scenes,
        taxonomy=load_taxonomy(args.taxonomy), rows_output=args.rows_output,
        analysis_output=args.analysis_output, size_bins=args.size_bins,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate V7 replay outputs")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes", required=True)
    parser.add_argument("--condition", action="append", dest="conditions", required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args(argv)
    evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=args.replay_root, scene_ids=args.scenes,
        conditions=args.conditions, taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=args.metrics_output, analysis_output=args.analysis_output,
        size_bins=args.size_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
