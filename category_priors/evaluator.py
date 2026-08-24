from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import hash_json, load_json, sha256_file, write_json
from .instance_projection import project_declared_instances
from .taxonomy import Taxonomy

OVERLAPS = tuple(np.arange(0.50, 0.96, 0.05).round(2).tolist())
PROTOCOL_VERSION = "scannet-official-instance-v1"


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
    output_instances = output.get("instances", {})
    projection = project_declared_instances(
        output["point_labels"], output_instances
    )
    gaussian_labels = projection.point_labels
    gaussian_coords = apply_transform(load_ply_xyz(gaussian_ply), transform)
    mapped_labels, diagnostics = map_gaussians_to_gt(
        gt_coords, gaussian_coords, gaussian_labels, radius_m
    )
    diagnostics.update(projection.numeric_stats())
    diagnostics["gt_nearest_declared_fraction"] = (
        float(np.mean(mapped_labels >= 0)) if len(mapped_labels) else 0.0
    )
    metadata = load_json(metadata_json) if metadata_json else {"instances": {}}
    metadata_instances = metadata.get("instances", {})
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    predictions: list[PredictedInstance] = []
    for raw_instance_id, properties in output_instances.items():
        instance_id = int(raw_instance_id)
        if instance_id < 0:
            continue
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
    """Integrate an already constructed precision/recall curve.

    This helper is retained for callers that construct a conventional precision
    envelope.  ScanNet instance evaluation uses its own score-threshold curve and
    integration rule, implemented by ``_scannet_average_precision`` below.
    """
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


def _scannet_average_precision(
    y_true: np.ndarray, y_score: np.ndarray, hard_false_negatives: int
) -> float:
    """Port ScanNet's official instance AP integration without interpolation."""
    if len(y_score) == 0:
        return 0.0

    score_order = np.argsort(y_score)
    score_sorted = y_score[score_order]
    true_sorted = y_true[score_order]
    true_cumsum = np.cumsum(true_sorted)
    _, unique_indices = np.unique(score_sorted, return_index=True)

    precision = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    recall = np.zeros(len(unique_indices) + 1, dtype=np.float64)
    num_examples = len(score_sorted)
    num_true_examples = float(true_cumsum[-1])
    # The appended zero is used by the official code for the idx == 0 case.
    true_cumsum = np.append(true_cumsum, 0.0)
    for result_index, score_index in enumerate(unique_indices):
        cumsum = float(true_cumsum[score_index - 1])
        true_positives = num_true_examples - cumsum
        false_positives = num_examples - int(score_index) - true_positives
        false_negatives = cumsum + hard_false_negatives
        precision[result_index] = true_positives / (
            true_positives + false_positives
        )
        recall[result_index] = true_positives / (
            true_positives + false_negatives
        )

    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_convolution = np.append(recall[0], recall)
    recall_for_convolution = np.append(recall_for_convolution, 0.0)
    step_widths = np.convolve(
        recall_for_convolution, np.asarray([-0.5, 0.0, 0.5]), "valid"
    )
    return float(np.dot(precision, step_widths))


def _all_ground_truth_instances(
    scene: GroundTruthScene, class_id: int
) -> list[np.ndarray]:
    """Return every non-empty GT instance, including regions later ignored as small."""
    result: list[np.ndarray] = []
    class_mask = scene.semantic == class_id
    for instance_id in np.unique(scene.instance[class_mask]):
        if instance_id < 0:
            continue
        mask = class_mask & (scene.instance == instance_id)
        if np.any(mask):
            result.append(mask)
    return result


def _prepare_class_matches(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_scene: dict[str, list[tuple[int, PredictedInstance]]],
    class_id: int,
    min_region_size: int,
    valid_semantic_ids: np.ndarray,
) -> tuple[
    dict[
        str,
        tuple[
            list[np.ndarray],
            list[int],
            list[tuple[int, PredictedInstance, int, int, list[tuple[int, int]]]],
        ],
    ],
    int,
    bool,
]:
    """Associate predictions with GT once, as ScanNet does before AP thresholds."""
    prepared: dict[
        str,
        tuple[
            list[np.ndarray],
            list[int],
            list[tuple[int, PredictedInstance, int, int, list[tuple[int, int]]]],
        ],
    ] = {}
    total_gt = 0
    has_prediction = False

    for scene in ground_truth:
        all_gt = _all_ground_truth_instances(scene, class_id)
        gt_counts = [int(mask.sum()) for mask in all_gt]
        total_gt += sum(count >= min_region_size for count in gt_counts)
        void_mask = ~np.isin(scene.semantic, valid_semantic_ids)

        scene_predictions: list[
            tuple[int, PredictedInstance, int, int, list[tuple[int, int]]]
        ] = []
        for prediction_index, prediction in predictions_by_scene.get(
            scene.scene_id, []
        ):
            pred_count = int(np.count_nonzero(prediction.mask))
            # ScanNet discards predicted regions smaller than min_region_size
            # before deciding whether the class has predictions.
            if pred_count < min_region_size:
                continue
            has_prediction = True
            intersections = [
                (gt_index, int(np.count_nonzero(prediction.mask & gt_mask)))
                for gt_index, gt_mask in enumerate(all_gt)
            ]
            intersections = [item for item in intersections if item[1] > 0]
            scene_predictions.append(
                (
                    prediction_index,
                    prediction,
                    pred_count,
                    int(np.count_nonzero(prediction.mask & void_mask)),
                    intersections,
                )
            )
        prepared[scene.scene_id] = (all_gt, gt_counts, scene_predictions)

    return prepared, total_gt, has_prediction


def _evaluate_class_at_overlap(
    ground_truth: Sequence[GroundTruthScene],
    prepared: dict[
        str,
        tuple[
            list[np.ndarray],
            list[int],
            list[tuple[int, PredictedInstance, int, int, list[tuple[int, int]]]],
        ],
    ],
    total_gt: int,
    has_prediction: bool,
    overlap_threshold: float,
    min_region_size: int,
) -> float | None:
    """Evaluate one class/overlap using ScanNet's official matching semantics."""

    if total_gt == 0:
        return None
    if not has_prediction:
        return 0.0

    prediction_visited = {
        prediction_index: False
        for _, _, scene_predictions in prepared.values()
        for prediction_index, _, _, _, _ in scene_predictions
    }
    y_true: list[float] = []
    y_score: list[float] = []
    hard_false_negatives = 0

    for scene in ground_truth:
        _, gt_counts, scene_predictions = prepared[scene.scene_id]
        valid_gt_indices = [
            index
            for index, count in enumerate(gt_counts)
            if count >= min_region_size
        ]
        current_true = [1.0] * len(valid_gt_indices)
        current_score = [-math.inf] * len(valid_gt_indices)
        current_match = [False] * len(valid_gt_indices)

        for local_gt_index, gt_index in enumerate(valid_gt_indices):
            found_match = False
            gt_count = gt_counts[gt_index]
            for (
                prediction_index,
                prediction,
                pred_count,
                _,
                intersections,
            ) in scene_predictions:
                if prediction_visited[prediction_index]:
                    continue
                intersection = next(
                    (
                        value
                        for matched_gt_index, value in intersections
                        if matched_gt_index == gt_index
                    ),
                    0,
                )
                if intersection == 0:
                    continue
                overlap = intersection / (gt_count + pred_count - intersection)
                # The official ScanNet evaluator uses a strict comparison.
                if overlap > overlap_threshold:
                    confidence = float(prediction.score)
                    if current_match[local_gt_index]:
                        maximum = max(current_score[local_gt_index], confidence)
                        minimum = min(current_score[local_gt_index], confidence)
                        current_score[local_gt_index] = maximum
                        current_true.append(0.0)
                        current_score.append(minimum)
                        current_match.append(True)
                    else:
                        found_match = True
                        current_match[local_gt_index] = True
                        current_score[local_gt_index] = confidence
                        prediction_visited[prediction_index] = True
            if not found_match:
                hard_false_negatives += 1

        y_true.extend(
            value for value, matched in zip(current_true, current_match) if matched
        )
        y_score.extend(
            value for value, matched in zip(current_score, current_match) if matched
        )

        # Unmatched predictions count as false positives unless enough of their
        # support is void or belongs to a same-class GT region excluded for size.
        for (
            _,
            prediction,
            pred_count,
            void_intersection,
            intersections,
        ) in scene_predictions:
            found_gt = False
            for gt_index, intersection in intersections:
                overlap = intersection / (
                    gt_counts[gt_index] + pred_count - intersection
                )
                if overlap > overlap_threshold:
                    found_gt = True
                    break
            if found_gt:
                continue
            ignored = void_intersection + sum(
                intersection
                for gt_index, intersection in intersections
                if gt_counts[gt_index] < min_region_size
            )
            ignored_fraction = ignored / pred_count
            if ignored_fraction <= overlap_threshold:
                y_true.append(0.0)
                y_score.append(float(prediction.score))

    return _scannet_average_precision(
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_score, dtype=np.float64),
        hard_false_negatives,
    )


def evaluate_instances(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    overlaps: Sequence[float] = OVERLAPS,
    min_region_size: int = 100,
) -> dict[str, Any]:
    gt_by_scene = {scene.scene_id: scene for scene in ground_truth}
    if len(gt_by_scene) != len(ground_truth):
        raise ValueError("Ground-truth scene ids must be unique")
    for scene in ground_truth:
        if scene.semantic.shape != scene.instance.shape or scene.semantic.ndim != 1:
            raise ValueError(f"{scene.scene_id}: invalid GT array shapes")

    pred_by_class_and_scene: dict[
        int, dict[str, list[tuple[int, PredictedInstance]]]
    ] = {
        index: {} for index in range(len(class_names))
    }
    pred_counts = {index: 0 for index in range(len(class_names))}
    for prediction_index, prediction in enumerate(predictions):
        if prediction.scene_id not in gt_by_scene:
            raise ValueError(
                f"Prediction references unknown scene: {prediction.scene_id}"
            )
        scene = gt_by_scene[prediction.scene_id]
        if prediction.mask.shape != scene.semantic.shape:
            raise ValueError(
                f"{prediction.scene_id}: prediction mask shape does not match GT"
            )
        if 0 <= prediction.class_id < len(class_names):
            pred_by_class_and_scene[prediction.class_id].setdefault(
                prediction.scene_id, []
            ).append((prediction_index, prediction))
            pred_counts[prediction.class_id] += 1

    thresholds = sorted({float(value) for value in overlaps} | {0.25})
    valid_semantic_ids = np.arange(len(class_names), dtype=np.int64)
    per_class: dict[str, Any] = {}
    for class_id, class_name in enumerate(class_names):
        class_result: dict[str, float | None] = {}
        prepared, total_gt, has_prediction = _prepare_class_matches(
            ground_truth,
            pred_by_class_and_scene[class_id],
            class_id,
            min_region_size,
            valid_semantic_ids,
        )
        for threshold in thresholds:
            ap = _evaluate_class_at_overlap(
                ground_truth,
                prepared,
                total_gt,
                has_prediction,
                threshold,
                min_region_size,
            )
            class_result[f"ap_{threshold:.2f}"] = ap
        valid_main = [class_result[f"ap_{threshold:.2f}"] for threshold in overlaps]
        class_result["ap_50_95"] = (
            float(np.mean([value for value in valid_main if value is not None]))
            if any(value is not None for value in valid_main)
            else None
        )
        class_result["gt_instances"] = total_gt
        class_result["pred_instances"] = pred_counts[class_id]
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
        "protocol_version": PROTOCOL_VERSION,
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
