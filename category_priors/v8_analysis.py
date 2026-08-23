from __future__ import annotations

"""Offline V8 lifting and object-bank analysis.

Ground truth is deliberately confined to this module.  Runtime workers write
immutable Gaussian-index banks; this module maps them into GT-point space,
computes diagnostic oracles, and applies the preregistered stage gates.
"""

import json
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import GroundTruthScene, OVERLAPS, PredictedInstance
from .v8_evaluation import evaluate_fragment_oracles


V8_LIFTING_ARMS = ("G-M1", "G-AM", "S-M1", "S-AM")
_GATE_EPSILON = 1e-12


@dataclass(frozen=True)
class ScanNetSceneAPEvent:
    """One scene/class/IoU contribution to the official ScanNet AP curve."""

    gt_count: int
    y_true: np.ndarray
    y_score: np.ndarray
    hard_false_negatives: int


@dataclass(frozen=True)
class ScanNetSceneAPEvents:
    """Precomputed, scene-separable ScanNet AP events.

    ``events[scene][class][overlap]`` can be replicated by a scene's
    multinomial bootstrap weight without re-running matching.  Matching is
    scene-local in the official protocol, whereas the AP curve is pooled.
    """

    scene_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    overlaps: tuple[float, ...]
    min_region_size: int
    events: tuple[tuple[tuple[ScanNetSceneAPEvent, ...], ...], ...]


def _all_class_gt_masks(scene: GroundTruthScene, class_id: int) -> list[np.ndarray]:
    class_mask = scene.semantic == int(class_id)
    return [
        class_mask & (scene.instance == instance_id)
        for instance_id in np.unique(scene.instance[class_mask])
        if int(instance_id) >= 0
    ]


def _scene_class_ap_event(
    scene: GroundTruthScene,
    predictions: Sequence[tuple[int, PredictedInstance]],
    class_id: int,
    overlap: float,
    min_region_size: int,
    valid_semantic_ids: np.ndarray,
) -> ScanNetSceneAPEvent:
    """Port the official matching loop while retaining scene-local events."""
    all_gt = _all_class_gt_masks(scene, class_id)
    gt_counts = [int(mask.sum()) for mask in all_gt]
    valid_gt_indices = [
        index for index, count in enumerate(gt_counts)
        if count >= int(min_region_size)
    ]
    void_mask = ~np.isin(scene.semantic, valid_semantic_ids)
    prepared: list[
        tuple[int, PredictedInstance, int, int, list[tuple[int, int]]]
    ] = []
    for prediction_index, prediction in predictions:
        pred_count = int(np.count_nonzero(prediction.mask))
        if pred_count < int(min_region_size):
            continue
        intersections = [
            (gt_index, int(np.count_nonzero(prediction.mask & gt_mask)))
            for gt_index, gt_mask in enumerate(all_gt)
        ]
        prepared.append(
            (
                int(prediction_index),
                prediction,
                pred_count,
                int(np.count_nonzero(prediction.mask & void_mask)),
                [item for item in intersections if item[1] > 0],
            )
        )

    prediction_visited = {row[0]: False for row in prepared}
    current_true = [1.0] * len(valid_gt_indices)
    current_score = [-np.inf] * len(valid_gt_indices)
    current_match = [False] * len(valid_gt_indices)
    hard_false_negatives = 0
    threshold = float(overlap)
    for local_gt_index, gt_index in enumerate(valid_gt_indices):
        found_match = False
        gt_count = gt_counts[gt_index]
        for prediction_index, prediction, pred_count, _, intersections in prepared:
            if prediction_visited[prediction_index]:
                continue
            intersection = next(
                (
                    value for matched_gt_index, value in intersections
                    if matched_gt_index == gt_index
                ),
                0,
            )
            if not intersection:
                continue
            iou = intersection / (gt_count + pred_count - intersection)
            if iou > threshold:
                confidence = float(prediction.score)
                if current_match[local_gt_index]:
                    previous_score = current_score[local_gt_index]
                    current_score[local_gt_index] = max(previous_score, confidence)
                    current_true.append(0.0)
                    current_score.append(min(previous_score, confidence))
                    current_match.append(True)
                else:
                    found_match = True
                    current_match[local_gt_index] = True
                    current_score[local_gt_index] = confidence
                    prediction_visited[prediction_index] = True
        if not found_match:
            hard_false_negatives += 1

    y_true = [
        value for value, matched in zip(current_true, current_match) if matched
    ]
    y_score = [
        value for value, matched in zip(current_score, current_match) if matched
    ]
    for _, prediction, pred_count, void_intersection, intersections in prepared:
        found_gt = False
        for gt_index, intersection in intersections:
            iou = intersection / (gt_counts[gt_index] + pred_count - intersection)
            if iou > threshold:
                found_gt = True
                break
        if found_gt:
            continue
        ignored = void_intersection + sum(
            intersection
            for gt_index, intersection in intersections
            if gt_counts[gt_index] < int(min_region_size)
        )
        if ignored / pred_count <= threshold:
            y_true.append(0.0)
            y_score.append(float(prediction.score))

    return ScanNetSceneAPEvent(
        gt_count=len(valid_gt_indices),
        y_true=np.asarray(y_true, dtype=np.float64),
        y_score=np.asarray(y_score, dtype=np.float64),
        hard_false_negatives=int(hard_false_negatives),
    )


def precompute_scannet_scene_ap_events(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    *,
    overlaps: Sequence[float] = OVERLAPS,
    min_region_size: int = 100,
) -> ScanNetSceneAPEvents:
    """Precompute exact official AP events independently for every scene."""
    scenes = tuple(ground_truth)
    scene_by_id = {str(scene.scene_id): scene for scene in scenes}
    if len(scene_by_id) != len(scenes):
        raise ValueError("ground-truth scene IDs must be unique")
    normalized_overlaps = tuple(float(value) for value in overlaps)
    normalized_classes = tuple(map(str, class_names))
    pred_by_class_scene: dict[int, dict[str, list[tuple[int, PredictedInstance]]]] = {
        index: {} for index in range(len(normalized_classes))
    }
    for prediction_index, prediction in enumerate(predictions):
        scene_id = str(prediction.scene_id)
        if scene_id not in scene_by_id:
            raise ValueError(f"prediction references unknown scene: {scene_id}")
        if prediction.mask.shape != scene_by_id[scene_id].semantic.shape:
            raise ValueError(f"{scene_id}: prediction mask shape does not match GT")
        if 0 <= int(prediction.class_id) < len(normalized_classes):
            pred_by_class_scene[int(prediction.class_id)].setdefault(
                scene_id, []
            ).append((prediction_index, prediction))
    valid_semantic_ids = np.arange(len(normalized_classes), dtype=np.int64)
    table: list[tuple[tuple[ScanNetSceneAPEvent, ...], ...]] = []
    for scene in scenes:
        class_rows: list[tuple[ScanNetSceneAPEvent, ...]] = []
        for class_id in range(len(normalized_classes)):
            class_rows.append(
                tuple(
                    _scene_class_ap_event(
                        scene,
                        pred_by_class_scene[class_id].get(scene.scene_id, ()),
                        class_id,
                        overlap,
                        min_region_size,
                        valid_semantic_ids,
                    )
                    for overlap in normalized_overlaps
                )
            )
        table.append(tuple(class_rows))
    return ScanNetSceneAPEvents(
        scene_ids=tuple(str(scene.scene_id) for scene in scenes),
        class_names=normalized_classes,
        overlaps=normalized_overlaps,
        min_region_size=int(min_region_size),
        events=tuple(table),
    )


def _weighted_ap_samples(
    scene_events: Sequence[ScanNetSceneAPEvent],
    scene_weights: np.ndarray,
) -> np.ndarray:
    """Vectorized exact ScanNet AP for many scene-multiplicity rows."""
    weights = np.asarray(scene_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(scene_events):
        raise ValueError("scene_weights must have one column per scene")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("scene weights must be finite and non-negative")
    gt_counts = np.asarray([event.gt_count for event in scene_events], dtype=np.float64)
    hard_fns = np.asarray(
        [event.hard_false_negatives for event in scene_events], dtype=np.float64
    )
    weighted_gt = weights @ gt_counts
    result = np.full(len(weights), np.nan, dtype=np.float64)
    active = weighted_gt > 0
    if not np.any(active):
        return result

    nonempty_scores = [event.y_score for event in scene_events if len(event.y_score)]
    if not nonempty_scores:
        result[active] = 0.0
        return result
    unique_scores = np.unique(np.concatenate(nonempty_scores))
    positives = np.zeros((len(scene_events), len(unique_scores)), dtype=np.float64)
    totals = np.zeros_like(positives)
    for scene_index, event in enumerate(scene_events):
        if event.y_true.shape != event.y_score.shape:
            raise ValueError("AP event truth and score arrays must align")
        if not len(event.y_score):
            continue
        score_indices = np.searchsorted(unique_scores, event.y_score)
        np.add.at(totals[scene_index], score_indices, 1.0)
        np.add.at(positives[scene_index], score_indices, event.y_true)

    weighted_positive = weights @ positives
    weighted_total = weights @ totals
    hard_false_negatives = weights @ hard_fns
    positive_total = weighted_positive.sum(axis=1, keepdims=True)
    positive_before = np.concatenate(
        (
            np.zeros((len(weights), 1), dtype=np.float64),
            np.cumsum(weighted_positive[:, :-1], axis=1),
        ),
        axis=1,
    )
    true_positive = positive_total - positive_before
    examples_at_or_above = np.cumsum(weighted_total[:, ::-1], axis=1)[:, ::-1]
    false_positive = examples_at_or_above - true_positive
    false_negative = positive_before + hard_false_negatives[:, None]
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_positive) > 0,
    )
    recall = np.divide(
        true_positive,
        true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=(true_positive + false_negative) > 0,
    )
    precision_curve = np.concatenate(
        (precision, np.ones((len(weights), 1), dtype=np.float64)), axis=1
    )
    recall_curve = np.concatenate(
        (recall, np.zeros((len(weights), 1), dtype=np.float64)), axis=1
    )
    padded_recall = np.concatenate(
        (recall_curve[:, :1], recall_curve, np.zeros((len(weights), 1))), axis=1
    )
    step_widths = 0.5 * (padded_recall[:, :-2] - padded_recall[:, 2:])
    ap = np.sum(precision_curve * step_widths, axis=1)
    result[active] = ap[active]
    return result


def scannet_map_samples_from_scene_weights(
    events: ScanNetSceneAPEvents,
    scene_weights: np.ndarray,
) -> np.ndarray:
    """Recompute pooled official mAP for every scene-multiplicity sample."""
    weights = np.asarray(scene_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != len(events.scene_ids):
        raise ValueError("scene_weights must have one column per event scene")
    overlap_maps: list[np.ndarray] = []
    for overlap_index in range(len(events.overlaps)):
        class_aps = np.column_stack(
            [
                _weighted_ap_samples(
                    [
                        events.events[scene_index][class_index][overlap_index]
                        for scene_index in range(len(events.scene_ids))
                    ],
                    weights,
                )
                for class_index in range(len(events.class_names))
            ]
        )
        counts = np.count_nonzero(np.isfinite(class_aps), axis=1)
        overlap_map = np.divide(
            np.nansum(class_aps, axis=1),
            counts,
            out=np.full(len(weights), np.nan, dtype=np.float64),
            where=counts > 0,
        )
        overlap_maps.append(overlap_map)
    values = np.column_stack(overlap_maps)
    counts = np.count_nonzero(np.isfinite(values), axis=1)
    return np.divide(
        np.nansum(values, axis=1),
        counts,
        out=np.full(len(weights), np.nan, dtype=np.float64),
        where=counts > 0,
    )


def pooled_scannet_metrics_from_scene_weights(
    events: ScanNetSceneAPEvents,
    scene_weights: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return official pooled per-class/per-IoU AP under scene weights."""
    weights = np.ones(len(events.scene_ids), dtype=np.float64) if scene_weights is None else np.asarray(scene_weights, dtype=np.float64)
    if weights.shape != (len(events.scene_ids),):
        raise ValueError("scene_weights must have one value per event scene")
    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(events.class_names):
        row: dict[str, float | None] = {}
        for overlap_index, overlap in enumerate(events.overlaps):
            value = _weighted_ap_samples(
                [
                    events.events[scene_index][class_index][overlap_index]
                    for scene_index in range(len(events.scene_ids))
                ],
                weights[None, :],
            )[0]
            row[f"ap_{overlap:.2f}"] = float(value) if np.isfinite(value) else None
        valid = [value for value in row.values() if value is not None]
        row["ap_50_95"] = float(np.mean(valid)) if valid else None
        per_class[class_name] = row
    aggregate: dict[str, float | None] = {}
    for overlap in events.overlaps:
        values = [row[f"ap_{overlap:.2f}"] for row in per_class.values()]
        valid = [float(value) for value in values if value is not None]
        aggregate[f"map_{overlap:.2f}"] = float(np.mean(valid)) if valid else None
    valid_main = [value for value in aggregate.values() if value is not None]
    aggregate["map_50_95"] = float(np.mean(valid_main)) if valid_main else None
    return {"aggregate": aggregate, "per_class": per_class}


def paired_scannet_scene_bootstrap(
    reference: ScanNetSceneAPEvents,
    treatment: ScanNetSceneAPEvents,
    *,
    samples: int = 10_000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """Paired physical-scene bootstrap of the pooled official ScanNet mAP."""
    identity = (
        reference.scene_ids,
        reference.class_names,
        reference.overlaps,
        reference.min_region_size,
    )
    if identity != (
        treatment.scene_ids,
        treatment.class_names,
        treatment.overlaps,
        treatment.min_region_size,
    ):
        raise ValueError("paired AP event tables must describe the same protocol/scenes")
    if samples <= 0 or not reference.scene_ids:
        raise ValueError("bootstrap requires positive samples and at least one scene")
    scene_count = len(reference.scene_ids)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        scene_count,
        np.full(scene_count, 1.0 / scene_count),
        size=int(samples),
    )
    reference_samples = scannet_map_samples_from_scene_weights(reference, weights)
    treatment_samples = scannet_map_samples_from_scene_weights(treatment, weights)
    deltas = treatment_samples - reference_samples
    finite = deltas[np.isfinite(deltas)]
    if not len(finite):
        raise ValueError("no bootstrap replicate has an evaluable ScanNet mAP")
    reference_point = pooled_scannet_metrics_from_scene_weights(reference)
    treatment_point = pooled_scannet_metrics_from_scene_weights(treatment)
    delta_point = float(
        treatment_point["aggregate"]["map_50_95"]
        - reference_point["aggregate"]["map_50_95"]
    )
    return {
        "schema": "saga-v8-paired-scannet-bootstrap-v1",
        "scene_count": scene_count,
        "samples": int(samples),
        "seed": int(seed),
        "statistic": "pooled_official_scannet_map_50_95",
        "reference_map_50_95": reference_point["aggregate"]["map_50_95"],
        "treatment_map_50_95": treatment_point["aggregate"]["map_50_95"],
        "delta_map_50_95": delta_point,
        "paired_bootstrap_ci95": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "finite_sample_count": int(len(finite)),
    }


def _runtime_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenes", payload)
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    return {str(row["scene_id"]): dict(row) for row in rows}


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    explicit = scene.get("gaussian_ply")
    if explicit:
        path = Path(str(explicit))
        return path if path.is_absolute() else Path(str(scene["base_path"])) / path
    root = Path(str(scene["base_path"])) / "output_models/point_cloud/iteration_30000"
    primary = root / "scene_point_cloud.ply"
    return primary if primary.is_file() else root / "point_cloud.ply"


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def paired_scannet_scene_bootstrap_from_replays(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    replay_root: Path,
    scene_ids: Sequence[str],
    reference_condition: str,
    treatment_condition: str,
    class_names: Sequence[str],
    samples: int = 10_000,
    seed: int = 20260804,
    radius_m: float = 0.05,
    overlaps: Sequence[float] = OVERLAPS,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Build paired AP events from frozen V8 replays and bootstrap them.

    Runtime construction remains GT-free.  This path-level helper is strictly
    evaluator-side and uses one shared set of physical-scene multinomial draws
    for the two frozen conditions.
    """
    from .evaluator import (
        apply_transform,
        load_ground_truth_npz,
        load_ply_xyz,
        map_gaussians_to_gt,
    )

    runtime = _runtime_rows(runtime_manifest)
    normalized_classes = tuple(str(name).strip().lower() for name in class_names)
    class_to_id = {name: index for index, name in enumerate(normalized_classes)}
    conditions = (str(reference_condition), str(treatment_condition))
    ground_truth: list[GroundTruthScene] = []
    predictions: dict[str, list[PredictedInstance]] = {
        condition: [] for condition in conditions
    }
    for scene_id in map(str, scene_ids):
        if scene_id not in runtime:
            raise KeyError(f"scene missing from runtime manifest: {scene_id}")
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        ground_truth.append(gt)
        gaussian_xyz = apply_transform(
            load_ply_xyz(_gaussian_ply(scene)), _transform(scene)
        )
        for condition in conditions:
            run_dir = replay_root / condition / scene_id
            output = json.loads((run_dir / "output.json").read_text(encoding="utf-8"))
            diagnostics = json.loads(
                (run_dir / "diagnostics.json").read_text(encoding="utf-8")
            )
            gaussian_labels = np.asarray(output["point_labels"], dtype=np.int64)
            mapped_labels, _ = map_gaussians_to_gt(
                gt_xyz, gaussian_xyz, gaussian_labels, radius_m
            )
            instance_metadata = diagnostics.get("instances", {})
            for raw_instance_id, properties in output.get("instances", {}).items():
                instance_id = int(raw_instance_id)
                class_name = str(properties.get("class", "")).strip().lower()
                if class_name not in class_to_id:
                    continue
                metadata = instance_metadata.get(
                    str(instance_id), instance_metadata.get(instance_id, {})
                )
                score = float(metadata.get("score", properties.get("score", 1.0)))
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(
                        f"{condition}/{scene_id}/{instance_id}: invalid score {score}"
                    )
                predictions[condition].append(
                    PredictedInstance(
                        scene_id=scene_id,
                        instance_id=instance_id,
                        class_id=class_to_id[class_name],
                        score=score,
                        mask=mapped_labels == instance_id,
                    )
                )
    reference_events = precompute_scannet_scene_ap_events(
        ground_truth,
        predictions[conditions[0]],
        normalized_classes,
        overlaps=overlaps,
        min_region_size=min_region_size,
    )
    treatment_events = precompute_scannet_scene_ap_events(
        ground_truth,
        predictions[conditions[1]],
        normalized_classes,
        overlaps=overlaps,
        min_region_size=min_region_size,
    )
    result = paired_scannet_scene_bootstrap(
        reference_events, treatment_events, samples=samples, seed=seed
    )
    return {
        **result,
        "reference_condition": conditions[0],
        "treatment_condition": conditions[1],
    }


def load_lifting_bank(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load the frozen V8 lifting-bank schema while tolerating extra fields."""
    metadata_path = directory / "lifting_bank.json"
    arrays_path = directory / "lifting_bank.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "saga-v8-lifting-bank-v1":
        raise ValueError(f"{metadata_path}: unexpected lifting-bank schema")
    required = (
        "xyz_m",
        "fragment_full_indptr",
        "fragment_full_ids",
        "fragment_core_indptr",
        "fragment_core_ids",
        "fragment_frame",
        "fragment_mask_index",
        "fragment_source_class",
        "frame_visible_indptr",
        "frame_visible_ids",
    )
    with np.load(arrays_path, allow_pickle=False) as payload:
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"{arrays_path}: missing arrays {missing}")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    point_count = int(metadata["point_count"])
    fragment_count = int(metadata["fragment_count"])
    if arrays["xyz_m"].shape != (point_count, 3):
        raise ValueError(f"{arrays_path}: xyz_m does not match point_count")
    if arrays["fragment_full_indptr"].shape != (fragment_count + 1,):
        raise ValueError(f"{arrays_path}: invalid fragment_full_indptr")
    if arrays["fragment_source_class"].shape != (fragment_count,):
        raise ValueError(f"{arrays_path}: invalid fragment_source_class")
    return metadata, arrays


def load_object_bank(directory: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Thin loader for a V8 object bank; analysis uses only named arrays."""
    metadata_path = directory / "object_bank.json"
    arrays_path = directory / "object_bank.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if metadata.get("schema") != "saga-v8-object-bank-v1":
        raise ValueError(f"{metadata_path}: unexpected object-bank schema")
    point_count = int(metadata["point_count"])
    for key, classifier in (("mv", "mv-label"), ("codebook", "codebook")):
        candidate_count = int(metadata["classifiers"][classifier]["candidate_count"])
        required = (
            f"core_candidate_id_{key}",
            f"full_candidate_indptr_{key}",
            f"full_candidate_ids_{key}",
            f"core_candidate_indptr_{key}",
            f"core_candidate_ids_{key}",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"{arrays_path}: missing arrays {missing}")
        if arrays[f"core_candidate_id_{key}"].shape != (point_count,):
            raise ValueError(f"{arrays_path}: invalid {key} core candidate labels")
        if arrays[f"full_candidate_indptr_{key}"].shape != (candidate_count + 1,):
            raise ValueError(f"{arrays_path}: invalid {key} candidate indptr")
    return metadata, arrays


def unpack_ragged(indptr: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    pointers = np.asarray(indptr, dtype=np.int64)
    flat = np.asarray(values, dtype=np.int64)
    if pointers.ndim != 1 or not len(pointers) or pointers[0] != 0:
        raise ValueError("ragged indptr must be one-dimensional and begin at zero")
    if pointers[-1] != len(flat) or np.any(np.diff(pointers) < 0):
        raise ValueError("ragged indptr is inconsistent with values")
    return [flat[pointers[index]:pointers[index + 1]] for index in range(len(pointers) - 1)]


def _weighted_sparse_iou(
    left_ids: np.ndarray,
    left_mass: np.ndarray,
    right_ids: np.ndarray,
    right_mass: np.ndarray,
) -> float:
    union_ids = np.union1d(left_ids, right_ids)
    if not len(union_ids):
        return 0.0
    left = np.zeros(len(union_ids), dtype=np.float64)
    right = np.zeros(len(union_ids), dtype=np.float64)
    left[np.searchsorted(union_ids, left_ids)] = np.asarray(left_mass, dtype=np.float64)
    right[np.searchsorted(union_ids, right_ids)] = np.asarray(right_mass, dtype=np.float64)
    denominator = float(np.maximum(left, right).sum())
    return float(np.minimum(left, right).sum() / denominator) if denominator else 0.0


def _late_fragment_class_ids(
    arrays: Mapping[str, np.ndarray],
    *,
    min_weighted_iou: float = 0.25,
) -> list[int]:
    """Give class-agnostic fragments the same frozen Grounded-mask evidence
    used by the registered MV-label classifier.

    This is evaluator-side semantic attribution only.  It does not change a
    fragment membership or enter the geometric oracle/track construction.
    """
    fragment_full = unpack_ragged(
        arrays["fragment_full_indptr"], arrays["fragment_full_ids"]
    )
    fragment_mass = unpack_ragged(
        arrays["fragment_full_indptr"], arrays["fragment_full_mass"]
    )
    semantic_full = unpack_ragged(
        arrays["semantic_fragment_full_indptr"],
        arrays["semantic_fragment_full_ids"],
    )
    semantic_mass = unpack_ragged(
        arrays["semantic_fragment_full_indptr"],
        arrays["semantic_fragment_full_mass"],
    )
    semantic_by_frame: dict[int, list[int]] = {}
    for index, frame_id in enumerate(np.asarray(arrays["semantic_fragment_frame"])):
        semantic_by_frame.setdefault(int(frame_id), []).append(index)
    labels: list[int] = []
    for index, frame_id in enumerate(np.asarray(arrays["fragment_frame"])):
        best_score = float(min_weighted_iou)
        best_class = -1
        for semantic_index in semantic_by_frame.get(int(frame_id), ()):
            class_id = int(arrays["semantic_fragment_class"][semantic_index])
            if class_id < 0:
                continue
            score = _weighted_sparse_iou(
                fragment_full[index], fragment_mass[index],
                semantic_full[semantic_index], semantic_mass[semantic_index],
            )
            if (
                score >= float(min_weighted_iou)
                and (
                    best_class < 0
                    or score > best_score + 1e-12
                    or (abs(score - best_score) <= 1e-12 and class_id < best_class)
                )
            ):
                best_score = score
                best_class = class_id
        labels.append(best_class)
    return labels


def map_gt_points_to_nearest_gaussian(
    gt_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    radius_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map each GT point to one nearest Gaussian, or ``-1`` when unsupported."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("V8 nearest-Gaussian mapping requires scipy") from exc
    gt = np.asarray(gt_xyz, dtype=np.float64)
    gaussians = np.asarray(gaussian_xyz, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1:] != (3,):
        raise ValueError("gt_xyz must be Nx3")
    if gaussians.ndim != 2 or gaussians.shape[1:] != (3,):
        raise ValueError("gaussian_xyz must be Mx3")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    distances, indices = cKDTree(gaussians).query(
        gt, k=1, distance_upper_bound=float(radius_m), workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(gaussians))
    mapped = np.full(len(gt), -1, dtype=np.int64)
    mapped[valid] = np.asarray(indices[valid], dtype=np.int64)
    finite = np.asarray(distances[valid], dtype=np.float64)
    return mapped, {
        "mapped_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "median_nn_distance_m": float(np.median(finite)) if len(finite) else float("inf"),
        "p95_nn_distance_m": (
            float(np.quantile(finite, 0.95)) if len(finite) else float("inf")
        ),
    }


def gaussian_sets_to_gt_point_ids(
    gaussian_sets: Sequence[np.ndarray | Sequence[int]],
    gt_nearest_gaussian: np.ndarray,
    gaussian_count: int,
) -> list[np.ndarray]:
    """Convert sparse Gaussian memberships to sparse shared GT-point IDs."""
    mapping = np.asarray(gt_nearest_gaussian, dtype=np.int64)
    if mapping.ndim != 1 or gaussian_count < 0:
        raise ValueError("invalid GT-to-Gaussian mapping")
    valid_gt = np.flatnonzero((mapping >= 0) & (mapping < gaussian_count))
    mapped_gaussians = mapping[valid_gt]
    order = np.argsort(mapped_gaussians, kind="stable")
    sorted_gt = valid_gt[order]
    counts = np.bincount(mapped_gaussians, minlength=gaussian_count)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    memberships: list[np.ndarray] = []
    for set_index, raw_ids in enumerate(gaussian_sets):
        ids = np.unique(np.asarray(raw_ids, dtype=np.int64))
        if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= gaussian_count):
            raise ValueError(f"gaussian_sets[{set_index}] contains an invalid ID")
        pieces = [sorted_gt[indptr[index]:indptr[index + 1]] for index in ids]
        memberships.append(
            np.sort(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
        )
    return memberships


def _bbox_diagonal(coords: np.ndarray) -> float:
    points = np.asarray(coords, dtype=np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)))


def _is_tiny_small(diagonal: float, size_spec: Mapping[str, Any] | None) -> bool:
    if size_spec is None:
        return False
    boundaries = size_spec.get("boundaries_m", size_spec)
    return diagonal <= float(boundaries["small_max_m"])


def ground_truth_memberships(
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    *,
    min_region_size: int = 100,
    size_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one sparse row per labeled GT instance."""
    xyz = np.asarray(gt_xyz, dtype=np.float64)
    semantic = np.asarray(gt_semantic, dtype=np.int64)
    instance = np.asarray(gt_instance, dtype=np.int64)
    if xyz.shape != (len(semantic), 3) or instance.shape != semantic.shape:
        raise ValueError("GT coordinates, semantic, and instance arrays do not align")
    valid_points = (semantic >= 0) & (instance >= 0)
    keys = sorted(set(zip(semantic[valid_points].tolist(), instance[valid_points].tolist())))
    point_ids: list[np.ndarray] = []
    class_ids: list[int] = []
    instance_ids: list[int] = []
    official_valid: list[bool] = []
    tiny_small: list[bool] = []
    for class_id, instance_id in keys:
        ids = np.flatnonzero(
            valid_points & (semantic == class_id) & (instance == instance_id)
        )
        point_ids.append(ids)
        class_ids.append(int(class_id))
        instance_ids.append(int(instance_id))
        official_valid.append(len(ids) >= int(min_region_size))
        tiny_small.append(_is_tiny_small(_bbox_diagonal(xyz[ids]), size_spec))
    return {
        "point_ids": point_ids,
        "class_ids": class_ids,
        "instance_ids": instance_ids,
        "official_valid": official_valid,
        "tiny_small": tiny_small,
    }


def _sparse_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return float(intersection / union) if union else 0.0


def _fragment_precision(
    fragments: Sequence[np.ndarray],
    gt_instances: Sequence[np.ndarray],
    gt_valid: Sequence[bool],
    threshold: float = 0.25,
) -> tuple[int, float]:
    official = [gt for gt, keep in zip(gt_instances, gt_valid) if keep]
    matched = sum(
        max((_sparse_iou(fragment, gt) for gt in official), default=0.0) >= threshold
        for fragment in fragments
    )
    return int(matched), matched / len(fragments) if fragments else 0.0


def evaluate_fragment_scene_arrays(
    *,
    scene_id: str,
    combination: str,
    fragment_gaussian_ids: Sequence[np.ndarray | Sequence[int]],
    fragment_class_ids: Sequence[Hashable],
    gt_nearest_gaussian: np.ndarray,
    gaussian_count: int,
    gt_memberships: Mapping[str, Any],
    mask_count: int = 0,
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one immutable lifting arm after Gaussian-to-GT mapping."""
    fragments = gaussian_sets_to_gt_point_ids(
        fragment_gaussian_ids, gt_nearest_gaussian, gaussian_count
    )
    oracle = evaluate_fragment_oracles(
        fragments,
        fragment_class_ids,
        gt_memberships["point_ids"],
        gt_memberships["class_ids"],
        gt_instance_ids=gt_memberships["instance_ids"],
        gt_valid=gt_memberships["official_valid"],
        gt_is_tiny_small=gt_memberships["tiny_small"],
    )
    official = oracle["aggregate"]["official_valid"]
    tiny_small = oracle["aggregate"]["tiny_small_official_valid"]
    fragment_match_count, fragment_precision = _fragment_precision(
        fragments,
        gt_memberships["point_ids"],
        gt_memberships["official_valid"],
    )
    return {
        "scene_id": str(scene_id),
        "combination": str(combination),
        "fragment_count": len(fragments),
        "fragment_match_025_count": fragment_match_count,
        "fragment_precision_025": fragment_precision,
        "mask_count": int(mask_count),
        "runtime_seconds": float(runtime_seconds),
        "official_gt_count": int(official["gt_count"]),
        "geometric_greedy_match_050_count": int(
            official["geometric_greedy_upper_bound"]["match_050_count"]
        ),
        "semantic_greedy_match_050_count": int(
            official["semantic_greedy_upper_bound"]["match_050_count"]
        ),
        "tiny_small_gt_count": int(tiny_small["gt_count"]),
        "tiny_small_geometric_match_025_count": int(
            tiny_small["geometric_greedy_upper_bound"]["match_025_count"]
        ),
        "tiny_small_semantic_match_025_count": int(
            tiny_small["semantic_greedy_upper_bound"]["match_025_count"]
        ),
        "oracle": oracle,
    }


def aggregate_lifting_factorial_rows(
    scene_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate Stage-1 rows by arm using count-weighted denominators."""
    result: list[dict[str, Any]] = []
    for combination in V8_LIFTING_ARMS:
        selected = [row for row in scene_rows if row["combination"] == combination]
        official_gt = sum(int(row["official_gt_count"]) for row in selected)
        match_050 = sum(
            int(row["geometric_greedy_match_050_count"]) for row in selected
        )
        semantic_match_050 = sum(
            int(row["semantic_greedy_match_050_count"]) for row in selected
        )
        tiny_gt = sum(int(row["tiny_small_gt_count"]) for row in selected)
        tiny_match = sum(
            int(row["tiny_small_geometric_match_025_count"]) for row in selected
        )
        tiny_semantic_match = sum(
            int(row["tiny_small_semantic_match_025_count"]) for row in selected
        )
        fragments = sum(int(row["fragment_count"]) for row in selected)
        fragment_matches = sum(int(row["fragment_match_025_count"]) for row in selected)
        result.append(
            {
                "combination": combination,
                "scene_count": len(selected),
                "official_gt_count": official_gt,
                "geometric_greedy_match_050_count": match_050,
                "geometric_greedy_recall_050": (
                    match_050 / official_gt if official_gt else 0.0
                ),
                "semantic_greedy_match_050_count": semantic_match_050,
                "semantic_greedy_recall_050": (
                    semantic_match_050 / official_gt if official_gt else 0.0
                ),
                "tiny_small_gt_count": tiny_gt,
                "tiny_small_recall_025": tiny_match / tiny_gt if tiny_gt else 0.0,
                "tiny_small_semantic_recall_025": (
                    tiny_semantic_match / tiny_gt if tiny_gt else 0.0
                ),
                "fragment_count": fragments,
                "fragment_match_025_count": fragment_matches,
                "fragment_precision_025": (
                    fragment_matches / fragments if fragments else 0.0
                ),
                "mask_count": sum(int(row.get("mask_count", 0)) for row in selected),
                "runtime_seconds": sum(
                    float(row.get("runtime_seconds", 0.0)) for row in selected
                ),
            }
        )
    return result


def stage1_combination_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "geometric_match_050_at_least_6": (
            int(row["geometric_greedy_match_050_count"]) >= 6
        ),
        "tiny_small_recall_025_at_least_020": (
            float(row["tiny_small_recall_025"]) >= 0.20 - _GATE_EPSILON
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _factor_effect(
    treatment: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    match_delta = (
        int(treatment["geometric_greedy_match_050_count"])
        - int(control["geometric_greedy_match_050_count"])
    )
    recall_delta = (
        float(treatment["tiny_small_recall_025"])
        - float(control["tiny_small_recall_025"])
    )
    return {
        "match_050_delta": match_delta,
        "tiny_small_recall_025_delta": recall_delta,
        "substantive": (
            match_delta >= 2 or recall_delta >= 0.05 - _GATE_EPSILON
        ),
    }


def select_stage1_combination(
    arm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply exact V8 Stage-1 gates and the preregistered tie breaks."""
    by_name = {str(row["combination"]): dict(row) for row in arm_rows}
    missing = [name for name in V8_LIFTING_ARMS if name not in by_name]
    if missing:
        raise ValueError(f"missing V8 lifting arms: {missing}")
    gates = {name: stage1_combination_gate(by_name[name]) for name in V8_LIFTING_ARMS}
    qualified = [name for name in V8_LIFTING_ARMS if gates[name]["passed"]]

    def ranking(name: str) -> tuple[Any, ...]:
        row = by_name[name]
        mask, lifting = name.split("-")
        return (
            -int(row["geometric_greedy_match_050_count"]),
            -float(row["tiny_small_recall_025"]),
            -float(row["fragment_precision_025"]),
            int(row.get("mask_count", 0)),
            float(row.get("runtime_seconds", 0.0)),
            0 if lifting == "M1" else 1,
            0 if mask == "G" else 1,
        )

    selected = min(qualified, key=ranking) if qualified else None
    return {
        "passed": selected is not None,
        "selected_combination": selected,
        "qualified_combinations": qualified,
        "gates": gates,
        "factor_effects": {
            "mask_at_M1": _factor_effect(by_name["S-M1"], by_name["G-M1"]),
            "mask_at_AM": _factor_effect(by_name["S-AM"], by_name["G-AM"]),
            "lifting_at_G": _factor_effect(by_name["G-AM"], by_name["G-M1"]),
            "lifting_at_S": _factor_effect(by_name["S-AM"], by_name["S-M1"]),
        },
    }


def select_late_classifier(
    candidate_gt_point_ids: Sequence[np.ndarray | Sequence[int]],
    mv_class_ids: Sequence[Hashable],
    codebook_class_ids: Sequence[Hashable],
    gt_point_ids: Sequence[np.ndarray | Sequence[int]],
    gt_class_ids: Sequence[Hashable],
    *,
    gt_valid: Sequence[bool] | None = None,
    min_geometric_iou: float = 0.25,
    mv_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Choose MV-label within two percentage points of codebook accuracy."""
    count = len(candidate_gt_point_ids)
    if len(mv_class_ids) != count or len(codebook_class_ids) != count:
        raise ValueError("classifier outputs must have one entry per candidate")
    valid = (
        np.ones(len(gt_point_ids), dtype=bool)
        if gt_valid is None
        else np.asarray(gt_valid, dtype=bool)
    )
    if valid.shape != (len(gt_point_ids),) or len(gt_class_ids) != len(gt_point_ids):
        raise ValueError("GT metadata does not align")
    candidates = [np.unique(np.asarray(ids, dtype=np.int64)) for ids in candidate_gt_point_ids]
    ground_truth = [np.unique(np.asarray(ids, dtype=np.int64)) for ids in gt_point_ids]
    eligible: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        matches = [
            (_sparse_iou(candidate, gt), gt_index)
            for gt_index, gt in enumerate(ground_truth)
            if valid[gt_index]
        ]
        matches.sort(key=lambda item: (-item[0], item[1]))
        if not matches or matches[0][0] < float(min_geometric_iou):
            continue
        iou, gt_index = matches[0]
        gt_class = gt_class_ids[gt_index]
        eligible.append(
            {
                "candidate_index": candidate_index,
                "gt_index": gt_index,
                "geometric_iou": float(iou),
                "mv_correct": bool(mv_class_ids[candidate_index] == gt_class),
                "codebook_correct": bool(
                    codebook_class_ids[candidate_index] == gt_class
                ),
            }
        )
    denominator = len(eligible)
    mv_accuracy = (
        sum(int(row["mv_correct"]) for row in eligible) / denominator
        if denominator else 0.0
    )
    codebook_accuracy = (
        sum(int(row["codebook_correct"]) for row in eligible) / denominator
        if denominator else 0.0
    )
    selected = (
        "MV-label"
        if mv_accuracy >= codebook_accuracy - mv_tolerance - _GATE_EPSILON
        else "codebook"
    )
    return {
        "eligible_candidate_count": denominator,
        "mv_correct_count": sum(int(row["mv_correct"]) for row in eligible),
        "codebook_correct_count": sum(
            int(row["codebook_correct"]) for row in eligible
        ),
        "mv_label_accuracy": mv_accuracy,
        "codebook_accuracy": codebook_accuracy,
        "selected_classifier": selected,
        "mv_tolerance": float(mv_tolerance),
        "candidates": eligible,
    }


def aggregate_late_classifier_results(
    scene_results: Sequence[Mapping[str, Any]],
    *,
    mv_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select one classifier from candidate-weighted multi-scene counts."""
    denominator = sum(int(row["eligible_candidate_count"]) for row in scene_results)
    mv_correct = sum(int(row["mv_correct_count"]) for row in scene_results)
    codebook_correct = sum(int(row["codebook_correct_count"]) for row in scene_results)
    mv_accuracy = mv_correct / denominator if denominator else 0.0
    codebook_accuracy = codebook_correct / denominator if denominator else 0.0
    selected = (
        "MV-label"
        if mv_accuracy >= codebook_accuracy - mv_tolerance - _GATE_EPSILON
        else "codebook"
    )
    return {
        "scene_count": len(scene_results),
        "eligible_candidate_count": denominator,
        "mv_correct_count": mv_correct,
        "codebook_correct_count": codebook_correct,
        "mv_label_accuracy": mv_accuracy,
        "codebook_accuracy": codebook_accuracy,
        "selected_classifier": selected,
        "mv_tolerance": float(mv_tolerance),
    }


def evaluate_object_bank_arrays(
    *,
    candidate_labels: np.ndarray,
    gt_nearest_gaussian: np.ndarray,
    gaussian_count: int,
    mv_class_ids: Sequence[Hashable],
    codebook_class_ids: Sequence[Hashable],
    gt_memberships: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate late classifiers on one frozen object bank."""
    labels = np.asarray(candidate_labels, dtype=np.int64)
    if labels.shape != (gaussian_count,):
        raise ValueError("candidate_labels must have one value per Gaussian")
    candidate_ids = [int(value) for value in np.unique(labels) if value >= 0]
    gaussian_sets = [np.flatnonzero(labels == value) for value in candidate_ids]
    memberships = gaussian_sets_to_gt_point_ids(
        gaussian_sets, gt_nearest_gaussian, gaussian_count
    )
    if len(mv_class_ids) != len(candidate_ids) or len(codebook_class_ids) != len(candidate_ids):
        raise ValueError("candidate class arrays do not align with candidate IDs")
    selection = select_late_classifier(
        memberships,
        mv_class_ids,
        codebook_class_ids,
        gt_memberships["point_ids"],
        gt_memberships["class_ids"],
        gt_valid=gt_memberships["official_valid"],
    )
    return {
        "candidate_ids": candidate_ids,
        "candidate_gt_point_ids": memberships,
        "classifier_selection": selection,
    }


def evaluate_loaded_object_bank_classifiers(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    gt_nearest_gaussian: np.ndarray,
    gt_memberships: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare MV/codebook labels on the same frozen geometric tracks."""
    point_count = int(metadata["point_count"])
    valid_track_ids = np.asarray(arrays["valid_track_ids"], dtype=np.int64)
    track_full_ids = unpack_ragged(
        arrays["track_full_indptr"], arrays["track_full_ids"]
    )
    if valid_track_ids.shape != (len(track_full_ids),):
        raise ValueError("valid track IDs and geometric track masks do not align")
    if len(np.unique(valid_track_ids)) != len(valid_track_ids):
        raise ValueError("valid geometric track IDs must be unique")
    geometry_by_track = {
        int(track_id): ids
        for track_id, ids in zip(valid_track_ids.tolist(), track_full_ids)
    }

    by_classifier: dict[str, dict[int, Hashable]] = {}
    for key, classifier in (("mv", "mv-label"), ("codebook", "codebook")):
        candidates = list(metadata["classifiers"][classifier]["candidates"])
        full_ids = unpack_ragged(
            arrays[f"full_candidate_indptr_{key}"],
            arrays[f"full_candidate_ids_{key}"],
        )
        if len(candidates) != len(full_ids):
            raise ValueError(f"{classifier} candidate metadata does not align")
        tracks: dict[int, Hashable] = {}
        for index, (candidate, ids) in enumerate(zip(candidates, full_ids)):
            candidate_id = int(candidate.get("candidate_id", index))
            if candidate_id != index:
                raise ValueError(f"{classifier} candidate IDs must be contiguous")
            track_id = int(candidate["track_id"])
            if track_id not in geometry_by_track:
                raise ValueError(
                    f"{classifier} candidate references invalid track {track_id}"
                )
            if not np.array_equal(
                np.sort(ids), np.sort(geometry_by_track[track_id])
            ):
                raise ValueError(f"track {track_id} geometry differs by classifier")
            if track_id in tracks:
                raise ValueError(f"{classifier} materialized track {track_id} twice")
            tracks[track_id] = candidate["class_id"]
        by_classifier[classifier] = tracks

    # The registered accuracy denominator is every valid geometric track with
    # IoU >= .25.  A track on which both classifiers abstain must therefore
    # remain present and count as an error for both; restricting this list to
    # materialized candidates would inflate both accuracies and can change the
    # two-percentage-point MV tie break.
    track_ids = sorted(geometry_by_track)
    gaussian_sets: list[np.ndarray] = []
    mv_classes: list[Hashable] = []
    codebook_classes: list[Hashable] = []
    for track_id in track_ids:
        gaussian_sets.append(geometry_by_track[track_id])
        mv_classes.append(by_classifier["mv-label"].get(track_id, -1))
        codebook_classes.append(by_classifier["codebook"].get(track_id, -1))
    memberships = gaussian_sets_to_gt_point_ids(
        gaussian_sets, gt_nearest_gaussian, point_count
    )
    selection = select_late_classifier(
        memberships,
        mv_classes,
        codebook_classes,
        gt_memberships["point_ids"],
        gt_memberships["class_ids"],
        gt_valid=gt_memberships["official_valid"],
    )
    return {
        "track_ids": track_ids,
        "candidate_gt_point_ids": memberships,
        "classifier_selection": selection,
    }


def stage2_bank_health_gate(
    bank: Mapping[str, Any],
    b1_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every registered V8 Stage-2 uniform-bank health threshold."""
    precision_gain = (
        float(bank["gaussian_micro_precision"])
        - float(b1_fixed["gaussian_micro_precision"])
    )
    unsupported_reduction = (
        float(b1_fixed["unsupported_instance_fraction"])
        - float(bank["unsupported_instance_fraction"])
    )
    baseline_instances = int(b1_fixed["predicted_instance_count"])
    instance_limit = 1.25 * baseline_instances
    checks = {
        "geometric_match_050_at_least_16": int(bank["geometric_match_050_count"]) >= 16,
        "geometric_match_050_scenes_at_least_4": (
            int(bank["geometric_match_050_scene_count"]) >= 4
        ),
        "same_class_match_050_at_least_12": (
            int(bank["same_class_match_050_count"]) >= 12
        ),
        "same_class_match_050_scenes_at_least_4": (
            int(bank["same_class_match_050_scene_count"]) >= 4
        ),
        "same_class_precision_025_at_least_010": (
            float(bank["same_class_candidate_precision_025"])
            >= 0.10 - _GATE_EPSILON
        ),
        "tiny_small_recall_025_at_least_020": (
            float(bank["tiny_small_recall_025"]) >= 0.20 - _GATE_EPSILON
        ),
        "precision_or_unsupported_improved": (
            precision_gain >= 0.05 - _GATE_EPSILON
            or unsupported_reduction >= 0.10 - _GATE_EPSILON
        ),
        "gt_recall_drop_at_most_005": (
            float(bank["gt_recall"])
            >= float(b1_fixed["gt_recall"]) - 0.05 - _GATE_EPSILON
        ),
        "u00_map_drop_at_most_0001": (
            float(bank["map_50_95"])
            >= float(b1_fixed["map_50_95"]) - 0.001 - _GATE_EPSILON
        ),
        "u00_ap50_drop_at_most_0002": (
            float(bank["ap50"])
            >= float(b1_fixed["ap50"]) - 0.002 - _GATE_EPSILON
        ),
        "instance_count_at_most_1_25x": (
            int(bank["predicted_instance_count"]) <= instance_limit
        ),
        "score_iou_spearman_at_least_020": (
            float(bank["score_iou_spearman"]) >= 0.20 - _GATE_EPSILON
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "precision_gain": precision_gain,
        "unsupported_reduction": unsupported_reduction,
        "instance_limit": instance_limit,
    }


def evaluate_v8_lifting_factorial(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    arm_roots: Mapping[str, Path],
    scene_ids: Sequence[str],
    canonical_classes: Sequence[str],
    size_spec: Mapping[str, Any] | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Path-level offline integration for the two-scene V8 2x2 audit."""
    from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz

    missing_arms = [name for name in V8_LIFTING_ARMS if name not in arm_roots]
    if missing_arms:
        raise ValueError(f"missing arm roots: {missing_arms}")
    runtime = _runtime_rows(runtime_manifest)
    class_to_id = {str(name): index for index, name in enumerate(canonical_classes)}
    scene_rows: list[dict[str, Any]] = []
    mapping_diagnostics: dict[str, Any] = {}
    for scene_id in scene_ids:
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(_gaussian_ply(scene)), _transform(scene)
        )
        nearest, diagnostics = map_gt_points_to_nearest_gaussian(
            gt_xyz, gaussian_xyz, radius_m
        )
        mapping_diagnostics[scene_id] = diagnostics
        gt_rows = ground_truth_memberships(
            gt_xyz,
            gt.semantic,
            gt.instance,
            min_region_size=min_region_size,
            size_spec=size_spec,
        )
        for arm in V8_LIFTING_ARMS:
            metadata, arrays = load_lifting_bank(Path(arm_roots[arm]) / scene_id)
            expected_mask, expected_lifting = arm.split("-")
            if str(metadata.get("scene_id")) != scene_id:
                raise ValueError(f"{scene_id}/{arm}: bank scene identity differs")
            if str(metadata.get("mask_source")) != expected_mask:
                raise ValueError(f"{scene_id}/{arm}: bank mask source differs")
            if str(metadata.get("lifting_source")) != expected_lifting:
                raise ValueError(f"{scene_id}/{arm}: bank lifting source differs")
            if int(metadata["point_count"]) != len(gaussian_xyz):
                raise ValueError(f"{scene_id}/{arm}: bank and Gaussian counts differ")
            fragments = unpack_ragged(
                arrays["fragment_full_indptr"], arrays["fragment_full_ids"]
            )
            raw_classes = np.asarray(arrays["fragment_source_class"], dtype=np.int64)
            names = list(metadata.get("classes", []))
            source_class_ids = (
                _late_fragment_class_ids(arrays)
                if expected_mask == "S"
                else raw_classes.tolist()
            )
            fragment_classes = [
                class_to_id.get(names[value], -1)
                if 0 <= value < len(names)
                else -1
                for value in source_class_ids
            ]
            frame_mask_pairs = np.column_stack(
                (arrays["fragment_frame"], arrays["fragment_mask_index"])
            )
            retained_mask_count = (
                len(np.unique(frame_mask_pairs, axis=0)) if len(frame_mask_pairs) else 0
            )
            scene_rows.append(
                evaluate_fragment_scene_arrays(
                    scene_id=scene_id,
                    combination=arm,
                    fragment_gaussian_ids=fragments,
                    fragment_class_ids=fragment_classes,
                    gt_nearest_gaussian=nearest,
                    gaussian_count=len(gaussian_xyz),
                    gt_memberships=gt_rows,
                    mask_count=int(metadata.get("mask_count", retained_mask_count)),
                    runtime_seconds=float(metadata.get("runtime_seconds", 0.0)),
                )
            )
    arm_rows = aggregate_lifting_factorial_rows(scene_rows)
    return {
        "schema": "saga-v8-lifting-factorial-analysis-v1",
        "scene_rows": scene_rows,
        "arm_rows": arm_rows,
        "selection": select_stage1_combination(arm_rows),
        "mapping": mapping_diagnostics,
    }


def _best_iou_and_class(
    candidate: np.ndarray,
    gt_sets: Sequence[np.ndarray],
    gt_classes: Sequence[int],
    gt_valid: Sequence[bool],
    *,
    candidate_class: int | None,
) -> float:
    return max(
        (
            _sparse_iou(candidate, gt)
            for gt, gt_class, valid in zip(gt_sets, gt_classes, gt_valid)
            if valid and (candidate_class is None or int(gt_class) == candidate_class)
        ),
        default=0.0,
    )


def _rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return 0.0
    from scipy.stats import spearmanr

    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def evaluate_v8_object_banks(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    canonical_classes: Sequence[str],
    size_spec: Mapping[str, Any] | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate frozen geometry and choose the registered late classifier.

    This function never mutates a bank and never exposes GT to tracking.  It
    first selects MV-label/codebook from same geometric tracks, then reports
    the exact candidate-health quantities needed by the Stage-2 gate.
    """
    from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz

    runtime = _runtime_rows(runtime_manifest)
    class_to_id = {str(name): index for index, name in enumerate(canonical_classes)}
    prepared: list[dict[str, Any]] = []
    classifier_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        nearest, mapping = map_gt_points_to_nearest_gaussian(
            gt_xyz, gaussian_xyz, radius_m
        )
        gt_rows = ground_truth_memberships(
            gt_xyz,
            gt.semantic,
            gt.instance,
            min_region_size=min_region_size,
            size_spec=size_spec,
        )
        metadata, arrays = load_object_bank(bank_root / scene_id)
        if int(metadata["point_count"]) != len(gaussian_xyz):
            raise ValueError(f"{scene_id}: object bank and Gaussian counts differ")
        classifier_result = evaluate_loaded_object_bank_classifiers(
            metadata=metadata,
            arrays=arrays,
            gt_nearest_gaussian=nearest,
            gt_memberships=gt_rows,
        )
        classifier_rows.append(classifier_result["classifier_selection"])
        track_sets = unpack_ragged(
            arrays["track_full_indptr"], arrays["track_full_ids"]
        )
        track_gt_sets = gaussian_sets_to_gt_point_ids(
            track_sets, nearest, len(gaussian_xyz)
        )
        prepared.append(
            {
                "scene_id": scene_id,
                "metadata": metadata,
                "arrays": arrays,
                "mapping": mapping,
                "nearest": nearest,
                "gaussian_count": len(gaussian_xyz),
                "gt": gt_rows,
                "track_gt_sets": track_gt_sets,
            }
        )
    classifier_selection = aggregate_late_classifier_results(classifier_rows)
    classifier = (
        "mv-label"
        if classifier_selection["selected_classifier"] == "MV-label"
        else "codebook"
    )
    key = "mv" if classifier == "mv-label" else "codebook"
    per_scene: list[dict[str, Any]] = []
    all_scores: list[float] = []
    all_ious: list[float] = []
    for row in prepared:
        metadata = row["metadata"]
        arrays = row["arrays"]
        gt_rows = row["gt"]
        geometric_ious = [
            _best_iou_and_class(
                candidate,
                gt_rows["point_ids"],
                gt_rows["class_ids"],
                gt_rows["official_valid"],
                candidate_class=None,
            )
            for candidate in row["track_gt_sets"]
        ]
        candidates = list(metadata["classifiers"][classifier]["candidates"])
        candidate_sets = unpack_ragged(
            arrays[f"full_candidate_indptr_{key}"],
            arrays[f"full_candidate_ids_{key}"],
        )
        # Reuse the already materialized point mapping.  Store it explicitly
        # here rather than letting GT enter any worker/bank code path.
        candidate_gt_sets = gaussian_sets_to_gt_point_ids(
            candidate_sets, row["nearest"], int(row["gaussian_count"])
        )
        candidate_ious: list[float] = []
        for candidate, points in zip(candidates, candidate_gt_sets):
            class_id = class_to_id.get(str(candidate["branch_class"]), -1)
            iou = _best_iou_and_class(
                points,
                gt_rows["point_ids"],
                gt_rows["class_ids"],
                gt_rows["official_valid"],
                candidate_class=class_id,
            )
            candidate_ious.append(iou)
            all_scores.append(float(candidate["base_score"]))
            all_ious.append(iou)
        tiny_indices = [
            index
            for index, (valid, tiny) in enumerate(
                zip(gt_rows["official_valid"], gt_rows["tiny_small"])
            )
            if valid and tiny
        ]
        tiny_matches = 0
        for gt_index in tiny_indices:
            class_id = int(gt_rows["class_ids"][gt_index])
            best = max(
                (
                    _sparse_iou(points, gt_rows["point_ids"][gt_index])
                    for candidate, points in zip(candidates, candidate_gt_sets)
                    if class_to_id.get(str(candidate["branch_class"]), -1) == class_id
                ),
                default=0.0,
            )
            tiny_matches += int(best >= 0.25)
        per_scene.append(
            {
                "scene_id": row["scene_id"],
                "geometric_match_050_count": int(
                    np.count_nonzero(np.asarray(geometric_ious) >= 0.50)
                ),
                "same_class_match_050_count": int(
                    np.count_nonzero(np.asarray(candidate_ious) >= 0.50)
                ),
                "same_class_match_025_count": int(
                    np.count_nonzero(np.asarray(candidate_ious) >= 0.25)
                ),
                "candidate_count": len(candidates),
                "candidate_iou_by_id": {
                    str(int(candidate.get("candidate_id", index))): float(iou)
                    for index, (candidate, iou) in enumerate(
                        zip(candidates, candidate_ious)
                    )
                },
                "tiny_small_gt_count": len(tiny_indices),
                "tiny_small_match_025_count": tiny_matches,
            }
        )
    geometric_match = sum(row["geometric_match_050_count"] for row in per_scene)
    same_match = sum(row["same_class_match_050_count"] for row in per_scene)
    match_025 = sum(row["same_class_match_025_count"] for row in per_scene)
    candidate_count = sum(row["candidate_count"] for row in per_scene)
    tiny_gt = sum(row["tiny_small_gt_count"] for row in per_scene)
    tiny_match = sum(row["tiny_small_match_025_count"] for row in per_scene)
    return {
        "schema": "saga-v8-bank-analysis-v1",
        "classifier_selection": classifier_selection,
        "selected_classifier": classifier,
        "per_scene": per_scene,
        "geometric_match_050_count": geometric_match,
        "geometric_match_050_scene_count": sum(
            row["geometric_match_050_count"] > 0 for row in per_scene
        ),
        "same_class_match_050_count": same_match,
        "same_class_match_050_scene_count": sum(
            row["same_class_match_050_count"] > 0 for row in per_scene
        ),
        "same_class_candidate_precision_025": (
            match_025 / candidate_count if candidate_count else 0.0
        ),
        "tiny_small_recall_025": tiny_match / tiny_gt if tiny_gt else 0.0,
        "score_iou_spearman": _rank_correlation(all_scores, all_ious),
    }
