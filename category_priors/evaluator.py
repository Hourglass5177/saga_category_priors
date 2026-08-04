from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import hash_json, load_json, sha256_file, write_json
from .taxonomy import Taxonomy

OVERLAPS = tuple(np.arange(0.50, 0.96, 0.05).round(2).tolist())


@dataclass(frozen=True)
class GroundTruthScene:
    scene_id: str
    semantic: np.ndarray
    instance: np.ndarray


@dataclass(frozen=True)
class PredictedInstance:
    scene_id: str
    instance_id: int
    class_id: int
    score: float
    mask: np.ndarray


def load_ground_truth_npz(
    path: str | Path, scene_id: str | None = None
) -> tuple[np.ndarray, GroundTruthScene]:
    payload = np.load(path)
    coords = np.asarray(payload["coords"], dtype=np.float64)
    semantic = np.asarray(payload["semantic"], dtype=np.int64)
    instance = np.asarray(payload["instance"], dtype=np.int64)
    if (
        coords.ndim != 2
        or coords.shape[1] != 3
        or semantic.shape != (len(coords),)
        or instance.shape != (len(coords),)
    ):
        raise ValueError(f"{path}: invalid GT array shapes")
    return coords, GroundTruthScene(scene_id or Path(path).stem, semantic, instance)


def load_ply_xyz(path: str | Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PLY evaluation requires plyfile") from exc
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)


def apply_transform(
    points: np.ndarray, matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("scene transform must be 4x4")
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    result = homogeneous @ transform.T
    return result[:, :3] / result[:, 3:4]


def map_gaussians_to_gt(
    gt_coords: np.ndarray,
    gaussian_coords: np.ndarray,
    gaussian_labels: np.ndarray,
    radius_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Nearest-neighbor evaluation requires scipy") from exc
    if len(gaussian_coords) != len(gaussian_labels):
        raise ValueError("Gaussian coordinate/label count mismatch")
    tree = cKDTree(gaussian_coords)
    distances, indices = tree.query(
        gt_coords, k=1, distance_upper_bound=radius_m, workers=-1
    )
    mapped = np.full(len(gt_coords), -1, dtype=np.int64)
    valid = np.isfinite(distances) & (indices < len(gaussian_labels))
    mapped[valid] = gaussian_labels[indices[valid]]
    diagnostics = {
        "mapped_fraction": float(valid.mean()) if len(valid) else 0.0,
        "median_nn_distance_m": float(np.median(distances[valid]))
        if np.any(valid)
        else math.inf,
        "p95_nn_distance_m": float(np.quantile(distances[valid], 0.95))
        if np.any(valid)
        else math.inf,
    }
    return mapped, diagnostics


def saga_scene_predictions(
    scene_id: str,
    gt_coords: np.ndarray,
    output_json: str | Path,
    gaussian_ply: str | Path,
    taxonomy: Taxonomy,
    metadata_json: str | Path | None,
    transform: Sequence[Sequence[float]],
    radius_m: float = 0.05,
    require_scores: bool = True,
) -> tuple[list[PredictedInstance], dict[str, float]]:
    output = load_json(output_json)
    gaussian_labels = np.asarray(output["point_labels"], dtype=np.int64)
    gaussian_coords = apply_transform(load_ply_xyz(gaussian_ply), transform)
    mapped_labels, diagnostics = map_gaussians_to_gt(
        gt_coords, gaussian_coords, gaussian_labels, radius_m
    )
    metadata = load_json(metadata_json) if metadata_json else {"instances": {}}
    metadata_instances = metadata.get("instances", {})
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    predictions: list[PredictedInstance] = []
    for raw_instance_id, properties in output.get("instances", {}).items():
        instance_id = int(raw_instance_id)
        class_name = str(properties.get("class", "")).strip().lower()
        if class_name not in class_to_id:
            continue
        meta = metadata_instances.get(
            str(instance_id), metadata_instances.get(instance_id, {})
        )
        if "score" not in meta and require_scores:
            raise ValueError(
                f"{scene_id}: instance {instance_id} is missing an AP score"
            )
        metadata_class = str(meta.get("class", class_name)).strip().lower()
        if metadata_class != class_name:
            raise ValueError(
                f"{scene_id}: instance {instance_id} class mismatch between output and metadata"
            )
        score = float(meta.get("score", 1.0))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{scene_id}: invalid instance score {score}")
        predictions.append(
            PredictedInstance(
                scene_id=scene_id,
                instance_id=instance_id,
                class_id=class_to_id[class_name],
                score=score,
                mask=mapped_labels == instance_id,
            )
        )
    return predictions, diagnostics


def _ground_truth_instances(
    scene: GroundTruthScene, class_id: int, min_region_size: int
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    class_mask = scene.semantic == class_id
    for instance_id in np.unique(scene.instance[class_mask]):
        if instance_id < 0:
            continue
        mask = class_mask & (scene.instance == instance_id)
        if int(mask.sum()) >= min_region_size:
            result.append(mask)
    return result


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    recall = np.concatenate(([0.0], recalls, [1.0]))
    precision = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.flatnonzero(recall[1:] != recall[:-1])
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def evaluate_instances(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    overlaps: Sequence[float] = OVERLAPS,
    min_region_size: int = 100,
) -> dict[str, Any]:
    gt_by_scene = {scene.scene_id: scene for scene in ground_truth}
    pred_by_class: dict[int, list[PredictedInstance]] = {
        index: [] for index in range(len(class_names))
    }
    for prediction in predictions:
        if prediction.scene_id not in gt_by_scene:
            raise ValueError(
                f"Prediction references unknown scene: {prediction.scene_id}"
            )
        pred_by_class.setdefault(prediction.class_id, []).append(prediction)

    thresholds = sorted({float(value) for value in overlaps} | {0.25})
    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(class_names):
        gt_masks = {
            scene.scene_id: _ground_truth_instances(scene, class_id, min_region_size)
            for scene in ground_truth
        }
        total_gt = sum(len(masks) for masks in gt_masks.values())
        class_result: dict[str, float | None] = {}
        for threshold in thresholds:
            if total_gt == 0:
                class_result[f"ap_{threshold:.2f}"] = None
                continue
            matched = {
                scene_id: np.zeros(len(masks), dtype=bool)
                for scene_id, masks in gt_masks.items()
            }
            true_positive: list[float] = []
            false_positive: list[float] = []
            sorted_predictions = sorted(
                pred_by_class.get(class_id, []),
                key=lambda item: (-item.score, item.scene_id, item.instance_id),
            )
            for prediction in sorted_predictions:
                scene = gt_by_scene[prediction.scene_id]
                pred_count = int(prediction.mask.sum())
                if pred_count < min_region_size:
                    continue
                valid_prediction = prediction.mask & (scene.semantic >= 0)
                void_fraction = float(
                    (prediction.mask & (scene.semantic < 0)).sum() / max(pred_count, 1)
                )
                best_iou = -1.0
                best_index = -1
                for index, gt_mask in enumerate(gt_masks[prediction.scene_id]):
                    if matched[prediction.scene_id][index]:
                        continue
                    intersection = int(np.count_nonzero(valid_prediction & gt_mask))
                    union = (
                        int(valid_prediction.sum()) + int(gt_mask.sum()) - intersection
                    )
                    iou = intersection / union if union else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_index >= 0 and best_iou >= threshold:
                    matched[prediction.scene_id][best_index] = True
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                elif void_fraction > threshold:
                    continue
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
            if true_positive:
                tp = np.cumsum(np.asarray(true_positive, dtype=np.float64))
                fp = np.cumsum(np.asarray(false_positive, dtype=np.float64))
                recall = tp / total_gt
                precision = tp / np.maximum(tp + fp, 1e-12)
                ap = average_precision(recall, precision)
            else:
                ap = 0.0
            class_result[f"ap_{threshold:.2f}"] = ap
        valid_main = [class_result[f"ap_{threshold:.2f}"] for threshold in overlaps]
        class_result["ap_50_95"] = (
            float(np.mean([value for value in valid_main if value is not None]))
            if any(value is not None for value in valid_main)
            else None
        )
        class_result["gt_instances"] = total_gt
        class_result["pred_instances"] = len(pred_by_class.get(class_id, []))
        per_class[class_name] = class_result

    aggregate: dict[str, float | None] = {}
    for threshold in thresholds:
        values = [value[f"ap_{threshold:.2f}"] for value in per_class.values()]
        finite = [float(value) for value in values if value is not None]
        aggregate[f"map_{threshold:.2f}"] = float(np.mean(finite)) if finite else None
    main_values = [aggregate[f"map_{threshold:.2f}"] for threshold in overlaps]
    aggregate["map_50_95"] = (
        float(np.mean([value for value in main_values if value is not None]))
        if any(value is not None for value in main_values)
        else None
    )
    return {
        "schema_version": "1.0",
        "protocol": "ScanNet200-SAGA20",
        "overlaps": list(overlaps),
        "min_region_size": min_region_size,
        "aggregate": aggregate,
        "per_class": per_class,
    }


def evaluate_manifest(
    manifest_path: str | Path,
    taxonomy: Taxonomy,
    output_path: str | Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    minimum_mapped_fraction = float(manifest.get("minimum_mapped_fraction", 0.90))
    if not 0.0 < minimum_mapped_fraction <= 1.0:
        raise ValueError("minimum_mapped_fraction must be in (0, 1]")
    ground_truth: list[GroundTruthScene] = []
    predictions: list[PredictedInstance] = []
    diagnostics: dict[str, Any] = {}
    base = Path(manifest_path).parent
    for item in manifest["scenes"]:
        scene_id = str(item["scene_id"])
        gt_path = base / item["gt_npz"]
        coords, gt_scene = load_ground_truth_npz(gt_path, scene_id)
        scene_predictions, scene_diagnostics = saga_scene_predictions(
            scene_id=scene_id,
            gt_coords=coords,
            output_json=base / item["output_json"],
            gaussian_ply=base / item["gaussian_ply"],
            taxonomy=taxonomy,
            metadata_json=base / item["metadata_json"],
            transform=item["gaussian_to_gt_transform"],
            radius_m=radius_m,
            require_scores=True,
        )
        if (
            scene_diagnostics["median_nn_distance_m"] > radius_m
            or scene_diagnostics["mapped_fraction"] < minimum_mapped_fraction
        ):
            raise ValueError(f"{scene_id}: coordinate alignment gate failed")
        ground_truth.append(gt_scene)
        predictions.extend(scene_predictions)
        diagnostics[scene_id] = scene_diagnostics
    result = evaluate_instances(
        ground_truth,
        predictions,
        taxonomy.canonical_classes,
        min_region_size=min_region_size,
    )
    result["diagnostics"] = diagnostics
    result["provenance"] = {
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "taxonomy_sha256": taxonomy.content_hash,
    }
    result["content_sha256"] = hash_json(result)
    write_json(output_path, result)
    return result
