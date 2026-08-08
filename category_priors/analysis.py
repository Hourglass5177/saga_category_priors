from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    OVERLAPS,
    GroundTruthScene,
    PredictedInstance,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .io import hash_json, load_json, sha256_file, write_json
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class _CurveEvents:
    """Score-ordered sufficient statistics for one class/IoU curve."""

    scores: np.ndarray
    true: np.ndarray
    scene_indices: np.ndarray
    unique_score_indices: np.ndarray
    hard_false_negatives: np.ndarray
    ground_truth_instances: np.ndarray


@dataclass(frozen=True)
class CompiledEvaluation:
    """Mask-free sufficient statistics for repeated confirmatory analyses."""

    scene_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    overlaps: tuple[float, ...]
    evaluation_overlaps: tuple[float, ...]
    min_region_size: int
    prediction_counts: tuple[int, ...]
    curves: tuple[tuple[_CurveEvents, ...], ...]


@dataclass(frozen=True)
class _PreparedSwapCurve:
    """Group-level count contrasts for exact pooled-mAP condition swaps."""

    reference_true_positive: np.ndarray
    reference_false_positive: np.ndarray
    treatment_true_positive: np.ndarray
    treatment_false_positive: np.ndarray
    delta_true_positive: np.ndarray
    delta_false_positive: np.ndarray
    ground_truth_instances: float


PredictionReplicates = Mapping[
    str,
    Sequence[PredictedInstance]
    | CompiledEvaluation
    | Mapping[
        str | int,
        Sequence[PredictedInstance] | CompiledEvaluation,
    ],
]


def _normalise_prediction_replicates(
    predictions_by_condition: PredictionReplicates,
) -> dict[
    str,
    dict[str, tuple[PredictedInstance, ...] | CompiledEvaluation],
]:
    """Make technical replicates explicit and reject unbalanced seed sets."""
    normalised: dict[
        str,
        dict[str, tuple[PredictedInstance, ...] | CompiledEvaluation],
    ] = {}
    for condition, value in predictions_by_condition.items():
        if isinstance(value, CompiledEvaluation):
            replicates = {"default": value}
        elif isinstance(value, Mapping):
            replicates: dict[
                str,
                tuple[PredictedInstance, ...] | CompiledEvaluation,
            ] = {}
            for raw_seed, predictions in value.items():
                seed = str(raw_seed)
                if seed in replicates:
                    raise ValueError(
                        f"{condition}: duplicate technical replicate {seed!r}"
                    )
                replicates[seed] = (
                    predictions
                    if isinstance(predictions, CompiledEvaluation)
                    else tuple(predictions)
                )
            if not replicates:
                raise ValueError(f"{condition}: no technical replicates were supplied")
        else:
            replicates = {"default": tuple(value)}
        normalised[str(condition)] = replicates

    if not normalised:
        raise ValueError("No prediction conditions were supplied")
    seed_sets = {
        condition: set(replicates)
        for condition, replicates in normalised.items()
    }
    expected = next(iter(seed_sets.values()))
    mismatched = {
        condition: sorted(seeds)
        for condition, seeds in seed_sets.items()
        if seeds != expected
    }
    if mismatched:
        raise ValueError(
            "All conditions must contain the same technical replicates; "
            f"expected={sorted(expected)}, mismatched={mismatched}"
        )
    return normalised


def _ground_truth_masks(
    scene: GroundTruthScene,
    class_id: int,
) -> list[np.ndarray]:
    class_mask = scene.semantic == class_id
    return [
        class_mask & (scene.instance == instance_id)
        for instance_id in np.unique(scene.instance[class_mask])
        if instance_id >= 0
    ]


def _scene_curve_events(
    scene: GroundTruthScene,
    predictions: Sequence[PredictedInstance],
    class_id: int,
    overlap: float,
    class_count: int,
    min_region_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Port ScanNet's GT-first matching into weightable per-scene events."""
    all_gt_masks = _ground_truth_masks(scene, class_id)
    valid_gt_masks = [
        mask for mask in all_gt_masks if int(mask.sum()) >= min_region_size
    ]
    small_gt_masks = [
        mask for mask in all_gt_masks if int(mask.sum()) < min_region_size
    ]
    class_predictions: list[PredictedInstance] = []
    for prediction in predictions:
        if prediction.scene_id != scene.scene_id or prediction.class_id != class_id:
            continue
        if prediction.mask.shape != scene.semantic.shape:
            raise ValueError(
                f"{scene.scene_id}: prediction {prediction.instance_id} "
                "mask shape mismatch"
            )
        if int(prediction.mask.sum()) >= min_region_size:
            class_predictions.append(prediction)

    visited = np.zeros(len(class_predictions), dtype=bool)
    true: list[float] = []
    scores: list[float] = []
    hard_false_negatives = 0
    for gt_mask in valid_gt_masks:
        found_match = False
        matched = False
        matched_score = -np.inf
        gt_count = int(gt_mask.sum())
        for pred_index, prediction in enumerate(class_predictions):
            if visited[pred_index]:
                continue
            intersection = int(np.count_nonzero(gt_mask & prediction.mask))
            if not intersection:
                continue
            union = gt_count + int(prediction.mask.sum()) - intersection
            iou = intersection / union if union else 0.0
            if iou <= overlap:  # ScanNet uses a strict `>` comparison.
                continue
            if matched:
                scores.append(min(matched_score, float(prediction.score)))
                true.append(0.0)
                matched_score = max(matched_score, float(prediction.score))
            else:
                found_match = True
                matched = True
                matched_score = float(prediction.score)
                visited[pred_index] = True
        if found_match:
            scores.append(matched_score)
            true.append(1.0)
        else:
            hard_false_negatives += 1

    valid_semantic = (scene.semantic >= 0) & (scene.semantic < class_count)
    ignore_mask = ~valid_semantic
    for mask in small_gt_masks:
        ignore_mask |= mask
    for prediction in class_predictions:
        pred_count = int(prediction.mask.sum())
        matched_gt = False
        for gt_mask in all_gt_masks:
            intersection = int(np.count_nonzero(gt_mask & prediction.mask))
            union = int(gt_mask.sum()) + pred_count - intersection
            if union and intersection / union > overlap:
                matched_gt = True
                break
        if matched_gt:
            continue
        ignore_fraction = float(np.count_nonzero(prediction.mask & ignore_mask)) / max(
            pred_count, 1
        )
        if ignore_fraction <= overlap:
            scores.append(float(prediction.score))
            true.append(0.0)

    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(true, dtype=np.float64),
        hard_false_negatives,
        len(valid_gt_masks),
    )


def compile_predictions(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    min_region_size: int = 100,
) -> CompiledEvaluation:
    """Compile dense prediction masks into compact, reusable curve events."""
    scene_ids = tuple(scene.scene_id for scene in ground_truth)
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Ground-truth scene IDs must be unique")
    scene_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    unknown = sorted({item.scene_id for item in predictions} - set(scene_ids))
    if unknown:
        raise ValueError(f"Predictions reference unknown scenes: {unknown}")
    predictions_by_scene: dict[str, list[PredictedInstance]] = {
        scene_id: [] for scene_id in scene_ids
    }
    prediction_counts = np.zeros(len(class_names), dtype=np.int64)
    for prediction in predictions:
        scene = ground_truth[scene_index[prediction.scene_id]]
        if prediction.mask.shape != scene.semantic.shape:
            raise ValueError(
                f"{prediction.scene_id}: prediction mask shape does not match GT"
            )
        predictions_by_scene[prediction.scene_id].append(prediction)
        if 0 <= prediction.class_id < len(class_names):
            prediction_counts[prediction.class_id] += 1

    overlaps = tuple(float(value) for value in OVERLAPS)
    evaluation_overlaps = tuple(sorted(set(overlaps) | {0.25}))
    curves: list[tuple[_CurveEvents, ...]] = []
    for class_id in range(len(class_names)):
        class_curves: list[_CurveEvents] = []
        for overlap in evaluation_overlaps:
            score_parts: list[np.ndarray] = []
            true_parts: list[np.ndarray] = []
            index_parts: list[np.ndarray] = []
            hard_fn = np.zeros(len(scene_ids), dtype=np.float64)
            gt_count = np.zeros(len(scene_ids), dtype=np.float64)
            for scene in ground_truth:
                index = scene_index[scene.scene_id]
                scores, true, misses, instances = _scene_curve_events(
                    scene,
                    predictions_by_scene[scene.scene_id],
                    class_id,
                    overlap,
                    len(class_names),
                    min_region_size,
                )
                if len(scores):
                    score_parts.append(scores)
                    true_parts.append(true)
                    index_parts.append(np.full(len(scores), index, dtype=np.int64))
                hard_fn[index] = misses
                gt_count[index] = instances
            scores = (
                np.concatenate(score_parts)
                if score_parts
                else np.empty(0, dtype=np.float64)
            )
            true = (
                np.concatenate(true_parts)
                if true_parts
                else np.empty(0, dtype=np.float64)
            )
            indices = (
                np.concatenate(index_parts)
                if index_parts
                else np.empty(0, dtype=np.int64)
            )
            order = np.argsort(scores, kind="stable")
            sorted_scores = scores[order]
            class_curves.append(
                _CurveEvents(
                    scores=sorted_scores,
                    true=true[order],
                    scene_indices=indices[order],
                    unique_score_indices=np.unique(
                        sorted_scores, return_index=True
                    )[1],
                    hard_false_negatives=hard_fn,
                    ground_truth_instances=gt_count,
                )
            )
        curves.append(tuple(class_curves))
    return CompiledEvaluation(
        scene_ids=scene_ids,
        class_names=tuple(class_names),
        overlaps=overlaps,
        evaluation_overlaps=evaluation_overlaps,
        min_region_size=min_region_size,
        prediction_counts=tuple(int(value) for value in prediction_counts),
        curves=tuple(curves),
    )


def merge_compiled_evaluations(
    parts: Sequence[CompiledEvaluation],
) -> CompiledEvaluation:
    """Merge disjoint scene compilations without reconstructing dense masks."""
    if not parts:
        raise ValueError("At least one compiled evaluation is required")
    first = parts[0]
    protocol = (
        first.class_names,
        first.overlaps,
        first.evaluation_overlaps,
        first.min_region_size,
    )
    scene_ids: list[str] = []
    prediction_counts = np.zeros(len(first.class_names), dtype=np.int64)
    for part in parts:
        if (
            part.class_names,
            part.overlaps,
            part.evaluation_overlaps,
            part.min_region_size,
        ) != protocol:
            raise ValueError("Compiled evaluations use incompatible protocols")
        duplicate = sorted(set(scene_ids) & set(part.scene_ids))
        if duplicate:
            raise ValueError(f"Compiled evaluations repeat scenes: {duplicate}")
        scene_ids.extend(part.scene_ids)
        prediction_counts += np.asarray(part.prediction_counts, dtype=np.int64)

    curves: list[tuple[_CurveEvents, ...]] = []
    for class_index in range(len(first.class_names)):
        class_curves: list[_CurveEvents] = []
        for overlap_index in range(len(first.evaluation_overlaps)):
            scores: list[np.ndarray] = []
            true: list[np.ndarray] = []
            scene_indices: list[np.ndarray] = []
            hard_false_negatives: list[np.ndarray] = []
            ground_truth_instances: list[np.ndarray] = []
            scene_offset = 0
            for part in parts:
                curve = part.curves[class_index][overlap_index]
                scores.append(curve.scores)
                true.append(curve.true)
                scene_indices.append(curve.scene_indices + scene_offset)
                hard_false_negatives.append(curve.hard_false_negatives)
                ground_truth_instances.append(curve.ground_truth_instances)
                scene_offset += len(part.scene_ids)
            merged_scores = np.concatenate(scores)
            merged_true = np.concatenate(true)
            merged_scene_indices = np.concatenate(scene_indices)
            order = np.argsort(merged_scores, kind="stable")
            merged_scores = merged_scores[order]
            class_curves.append(
                _CurveEvents(
                    scores=merged_scores,
                    true=merged_true[order],
                    scene_indices=merged_scene_indices[order],
                    unique_score_indices=np.unique(
                        merged_scores, return_index=True
                    )[1],
                    hard_false_negatives=np.concatenate(hard_false_negatives),
                    ground_truth_instances=np.concatenate(ground_truth_instances),
                )
            )
        curves.append(tuple(class_curves))
    return CompiledEvaluation(
        scene_ids=tuple(scene_ids),
        class_names=first.class_names,
        overlaps=first.overlaps,
        evaluation_overlaps=first.evaluation_overlaps,
        min_region_size=first.min_region_size,
        prediction_counts=tuple(int(value) for value in prediction_counts),
        curves=tuple(curves),
    )


def _weighted_curve_ap(curve: _CurveEvents, weights: np.ndarray) -> float | None:
    weighted_gt = float(np.dot(weights, curve.ground_truth_instances))
    if weighted_gt <= 0.0:
        return None
    if not len(curve.scores):
        return 0.0
    event_weights = weights[curve.scene_indices]
    active = event_weights > 0.0
    if not np.any(active):
        return 0.0
    if np.all(active):
        true = curve.true
        unique_score_indices = curve.unique_score_indices
    else:
        scores = curve.scores[active]
        event_weights = event_weights[active]
        true = curve.true[active]
        unique_score_indices = np.unique(scores, return_index=True)[1]
    weighted_true = event_weights * true
    true_cumsum = np.cumsum(weighted_true)
    weight_cumsum = np.cumsum(event_weights)
    total_true = float(true_cumsum[-1])
    total_weight = float(weight_cumsum[-1])
    hard_fn = float(np.dot(weights, curve.hard_false_negatives))
    true_cumsum = np.append(true_cumsum, 0.0)
    weight_cumsum = np.append(weight_cumsum, 0.0)
    precision = np.zeros(len(unique_score_indices) + 1, dtype=np.float64)
    recall = np.zeros_like(precision)
    for result_index, score_index in enumerate(unique_score_indices):
        true_before = float(true_cumsum[score_index - 1])
        weight_before = float(weight_cumsum[score_index - 1])
        true_positive = total_true - true_before
        false_positive = (total_weight - weight_before) - true_positive
        false_negative = true_before + hard_fn
        precision[result_index] = true_positive / max(
            true_positive + false_positive, np.finfo(np.float64).tiny
        )
        recall[result_index] = true_positive / max(
            true_positive + false_negative, np.finfo(np.float64).tiny
        )
    precision[-1] = 1.0
    recall[-1] = 0.0
    recall_for_convolution = np.concatenate(([recall[0]], recall, [0.0]))
    step_widths = np.convolve(
        recall_for_convolution, [-0.5, 0.0, 0.5], mode="valid"
    )
    return float(np.dot(precision, step_widths))


def _weighted_compiled_metric(
    compiled: CompiledEvaluation,
    weights: np.ndarray,
    *,
    require_all_classes: bool,
) -> float:
    if weights.shape != (len(compiled.scene_ids),):
        raise ValueError("Scene weights do not match the compiled scenes")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError(
            "Scene weights must be non-negative with positive total weight"
        )
    class_values: list[float] = []
    missing_classes: list[str] = []
    for class_name, curves in zip(compiled.class_names, compiled.curves, strict=True):
        overlap_values = [
            value
            for overlap, curve in zip(
                compiled.evaluation_overlaps, curves, strict=True
            )
            if overlap in compiled.overlaps
            if (value := _weighted_curve_ap(curve, weights)) is not None
        ]
        if not overlap_values:
            missing_classes.append(class_name)
        else:
            class_values.append(float(np.mean(overlap_values)))
    if require_all_classes and missing_classes:
        raise ValueError(
            "Positive-weight bootstrap requires GT support for every registered class; "
            f"missing={missing_classes}"
        )
    if not class_values:
        raise ValueError("mAP is undefined because the selected scenes contain no GT")
    return float(np.mean(class_values))


def weighted_scene_metric(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance] | CompiledEvaluation,
    class_names: Sequence[str],
    scene_weights: Mapping[str, float],
    min_region_size: int = 100,
    *,
    require_all_classes: bool = True,
) -> float:
    """Evaluate ScanNet mAP with one common weight for a scene's GT, TP and FP."""
    compiled = (
        predictions
        if isinstance(predictions, CompiledEvaluation)
        else compile_predictions(
            ground_truth, predictions, class_names, min_region_size
        )
    )
    if (
        compiled.scene_ids != tuple(scene.scene_id for scene in ground_truth)
        or compiled.class_names != tuple(class_names)
        or compiled.min_region_size != min_region_size
    ):
        raise ValueError(
            "Compiled predictions do not match the requested scenes, classes, "
            "or min_region_size"
        )
    missing = sorted(set(compiled.scene_ids) - set(scene_weights))
    if missing:
        raise ValueError(f"Missing scene weights: {missing}")
    weights = np.asarray(
        [float(scene_weights[scene_id]) for scene_id in compiled.scene_ids],
        dtype=np.float64,
    )
    return _weighted_compiled_metric(
        compiled, weights, require_all_classes=require_all_classes
    )


def evaluate_compiled(compiled: CompiledEvaluation) -> dict[str, Any]:
    """Return official aggregate/per-class AP without revisiting dense masks."""
    weights = np.ones(len(compiled.scene_ids), dtype=np.float64)
    per_class: dict[str, Any] = {}
    for class_index, (class_name, curves) in enumerate(
        zip(compiled.class_names, compiled.curves, strict=True)
    ):
        class_result: dict[str, float | int | None] = {}
        for overlap, curve in zip(
            compiled.evaluation_overlaps, curves, strict=True
        ):
            class_result[f"ap_{overlap:.2f}"] = _weighted_curve_ap(curve, weights)
        main_values = [
            class_result[f"ap_{overlap:.2f}"] for overlap in compiled.overlaps
        ]
        class_result["ap_50_95"] = (
            float(np.mean([value for value in main_values if value is not None]))
            if any(value is not None for value in main_values)
            else None
        )
        class_result["gt_instances"] = int(
            curves[0].ground_truth_instances.sum()
        )
        class_result["pred_instances"] = compiled.prediction_counts[class_index]
        per_class[class_name] = class_result

    aggregate: dict[str, float | None] = {}
    for overlap in compiled.evaluation_overlaps:
        values = [item[f"ap_{overlap:.2f}"] for item in per_class.values()]
        finite = [float(value) for value in values if value is not None]
        aggregate[f"map_{overlap:.2f}"] = (
            float(np.mean(finite)) if finite else None
        )
    main_values = [
        aggregate[f"map_{overlap:.2f}"] for overlap in compiled.overlaps
    ]
    aggregate["map_50_95"] = (
        float(np.mean([value for value in main_values if value is not None]))
        if any(value is not None for value in main_values)
        else None
    )
    return {
        "schema_version": "1.0",
        "protocol": "ScanNet200-SAGA20",
        "overlaps": list(compiled.overlaps),
        "min_region_size": compiled.min_region_size,
        "aggregate": aggregate,
        "per_class": per_class,
    }


def _validate_physical_groups(
    ground_truth: Sequence[GroundTruthScene],
    physical_group_by_scene: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    scene_ids = tuple(scene.scene_id for scene in ground_truth)
    missing = sorted(set(scene_ids) - set(physical_group_by_scene))
    if missing:
        raise ValueError(f"Missing physical-scene IDs for: {missing}")
    groups = tuple(sorted({str(physical_group_by_scene[scene]) for scene in scene_ids}))
    if not groups:
        raise ValueError("No physical scene groups were supplied")
    indices = {
        group: np.asarray(
            [
                index
                for index, scene_id in enumerate(scene_ids)
                if str(physical_group_by_scene[scene_id]) == group
            ],
            dtype=np.int64,
        )
        for group in groups
    }
    return groups, indices


def _compiled_replicates(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: PredictionReplicates,
    class_names: Sequence[str],
    min_region_size: int,
) -> tuple[dict[str, dict[str, CompiledEvaluation]], tuple[str, ...]]:
    normalised = _normalise_prediction_replicates(predictions_by_condition)
    expected_scenes = tuple(scene.scene_id for scene in ground_truth)
    expected_classes = tuple(class_names)
    result: dict[str, dict[str, CompiledEvaluation]] = {}
    for condition, replicates in normalised.items():
        result[condition] = {}
        for seed, predictions in replicates.items():
            compiled = (
                predictions
                if isinstance(predictions, CompiledEvaluation)
                else compile_predictions(
                    ground_truth, predictions, class_names, min_region_size
                )
            )
            if (
                compiled.scene_ids != expected_scenes
                or compiled.class_names != expected_classes
                or compiled.overlaps != tuple(float(value) for value in OVERLAPS)
                or compiled.min_region_size != min_region_size
            ):
                raise ValueError(
                    f"{condition}/{seed}: compiled predictions do not match "
                    "the requested scenes, classes, or min_region_size"
                )
            result[condition][seed] = compiled
    seeds = tuple(sorted(next(iter(normalised.values()))))
    return result, seeds


def _condition_metric(
    replicates: Mapping[str, CompiledEvaluation],
    weights: np.ndarray,
    *,
    require_all_classes: bool,
) -> float:
    # Seeds are technical replicates: average them inside every resample.
    return float(
        np.mean(
            [
                _weighted_compiled_metric(
                    compiled, weights, require_all_classes=require_all_classes
                )
                for compiled in replicates.values()
            ]
        )
    )


def _positive_group_weights(
    rng: np.random.Generator,
    groups: Sequence[str],
    group_indices: Mapping[str, np.ndarray],
    scene_count: int,
) -> np.ndarray:
    draws = np.maximum(
        rng.exponential(scale=1.0, size=len(groups)),
        np.finfo(np.float64).tiny,
    )
    weights = np.empty(scene_count, dtype=np.float64)
    for group, draw in zip(groups, draws, strict=True):
        weights[group_indices[group]] = draw
    return weights


def _curve_counts_by_group(
    curve: _CurveEvents,
    thresholds: np.ndarray,
    scene_group_indices: np.ndarray,
    group_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    true_positive = np.zeros((group_count, len(thresholds)), dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    event_groups = scene_group_indices[curve.scene_indices]
    for group_index in range(group_count):
        selected = event_groups == group_index
        scores = curve.scores[selected]
        true = curve.true[selected]
        if not len(scores):
            continue
        first_active = np.searchsorted(scores, thresholds, side="left")
        true_cumsum = np.concatenate(([0.0], np.cumsum(true)))
        group_true_positive = true_cumsum[-1] - true_cumsum[first_active]
        active_predictions = len(scores) - first_active
        true_positive[group_index] = group_true_positive
        false_positive[group_index] = active_predictions - group_true_positive
    return true_positive, false_positive


def _prepare_swap_curves(
    reference: CompiledEvaluation,
    treatment: CompiledEvaluation,
    group_indices: Mapping[str, np.ndarray],
    groups: Sequence[str],
) -> tuple[_PreparedSwapCurve, ...]:
    if (
        reference.scene_ids != treatment.scene_ids
        or reference.class_names != treatment.class_names
        or reference.overlaps != treatment.overlaps
        or reference.evaluation_overlaps != treatment.evaluation_overlaps
        or reference.min_region_size != treatment.min_region_size
    ):
        raise ValueError("Paired compiled predictions use incompatible protocols")
    scene_group_indices = np.empty(len(reference.scene_ids), dtype=np.int64)
    for group_index, group in enumerate(groups):
        scene_group_indices[group_indices[group]] = group_index

    prepared: list[_PreparedSwapCurve] = []
    for reference_class, treatment_class in zip(
        reference.curves, treatment.curves, strict=True
    ):
        for overlap, reference_curve, treatment_curve in zip(
            reference.evaluation_overlaps,
            reference_class,
            treatment_class,
            strict=True,
        ):
            if overlap not in reference.overlaps:
                continue
            thresholds = np.unique(
                np.concatenate((reference_curve.scores, treatment_curve.scores))
            )
            reference_tp, reference_fp = _curve_counts_by_group(
                reference_curve,
                thresholds,
                scene_group_indices,
                len(groups),
            )
            treatment_tp, treatment_fp = _curve_counts_by_group(
                treatment_curve,
                thresholds,
                scene_group_indices,
                len(groups),
            )
            reference_gt = float(reference_curve.ground_truth_instances.sum())
            treatment_gt = float(treatment_curve.ground_truth_instances.sum())
            if reference_gt != treatment_gt or reference_gt <= 0.0:
                raise ValueError(
                    "Paired pooled permutation requires fixed GT support for "
                    "every registered class"
                )
            prepared.append(
                _PreparedSwapCurve(
                    reference_true_positive=reference_tp.sum(axis=0),
                    reference_false_positive=reference_fp.sum(axis=0),
                    treatment_true_positive=treatment_tp.sum(axis=0),
                    treatment_false_positive=treatment_fp.sum(axis=0),
                    delta_true_positive=treatment_tp - reference_tp,
                    delta_false_positive=treatment_fp - reference_fp,
                    ground_truth_instances=reference_gt,
                )
            )
    return tuple(prepared)


def _batch_scannet_ap(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    ground_truth_instances: float,
) -> np.ndarray:
    if true_positive.shape != false_positive.shape:
        raise ValueError("TP and FP arrays must have the same shape")
    if true_positive.shape[1] == 0:
        return np.zeros(true_positive.shape[0], dtype=np.float64)
    true_positive = np.clip(true_positive, 0.0, ground_truth_instances)
    false_positive = np.maximum(false_positive, 0.0)
    denominator = true_positive + false_positive
    precision = np.divide(
        true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0.0,
    )
    recall = true_positive / ground_truth_instances
    active_count = true_positive + false_positive
    exact_count = active_count - np.concatenate(
        (
            active_count[:, 1:],
            np.zeros((len(active_count), 1), dtype=np.float64),
        ),
        axis=1,
    )
    active = exact_count > 0.5

    threshold_count = precision.shape[1]
    indices = np.broadcast_to(
        np.arange(threshold_count, dtype=np.int64), active.shape
    )
    active_indices = np.where(active, indices, threshold_count)
    nearest_active = np.minimum.accumulate(active_indices[:, ::-1], axis=1)[:, ::-1]
    next_active = np.concatenate(
        (
            nearest_active[:, 1:],
            np.full((len(active), 1), threshold_count, dtype=np.int64),
        ),
        axis=1,
    )
    padded_precision = np.concatenate(
        (precision, np.ones((len(precision), 1), dtype=np.float64)), axis=1
    )
    padded_recall = np.concatenate(
        (recall, np.zeros((len(recall), 1), dtype=np.float64)), axis=1
    )
    next_precision = np.take_along_axis(
        padded_precision, next_active, axis=1
    )
    next_recall = np.take_along_axis(padded_recall, next_active, axis=1)
    contributions = 0.5 * (precision + next_precision) * (recall - next_recall)
    return np.sum(np.where(active, contributions, 0.0), axis=1)


def _evaluate_swap_batch(
    swaps: np.ndarray,
    prepared_by_seed: Sequence[Sequence[_PreparedSwapCurve]],
) -> tuple[np.ndarray, np.ndarray]:
    reference_metric = np.zeros(len(swaps), dtype=np.float64)
    treatment_metric = np.zeros(len(swaps), dtype=np.float64)
    curve_count = 0
    for prepared_curves in prepared_by_seed:
        curve_count += len(prepared_curves)
        for curve in prepared_curves:
            reference_tp_change = swaps @ curve.delta_true_positive
            reference_fp_change = swaps @ curve.delta_false_positive
            reference_metric += _batch_scannet_ap(
                curve.reference_true_positive[None, :] + reference_tp_change,
                curve.reference_false_positive[None, :] + reference_fp_change,
                curve.ground_truth_instances,
            )
            treatment_metric += _batch_scannet_ap(
                curve.treatment_true_positive[None, :] - reference_tp_change,
                curve.treatment_false_positive[None, :] - reference_fp_change,
                curve.ground_truth_instances,
            )
    if curve_count == 0:
        raise ValueError("No pooled AP curves were prepared")
    return reference_metric / curve_count, treatment_metric / curve_count


def paired_scene_bootstrap(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: PredictionReplicates,
    physical_group_by_scene: Mapping[str, str],
    class_names: Sequence[str],
    reference: str,
    treatment: str,
    samples: int = 10_000,
    seed: int = 20260804,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Positive-weight physical-scene bootstrap for a paired mAP difference."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if (
        reference not in predictions_by_condition
        or treatment not in predictions_by_condition
    ):
        raise KeyError("Both reference and treatment predictions are required")
    selected = {
        reference: predictions_by_condition[reference],
        treatment: predictions_by_condition[treatment],
    }
    compiled, technical_replicates = _compiled_replicates(
        ground_truth, selected, class_names, min_region_size
    )
    groups, group_indices = _validate_physical_groups(
        ground_truth, physical_group_by_scene
    )
    point_weights = np.ones(len(ground_truth), dtype=np.float64)
    point_reference = _condition_metric(
        compiled[reference], point_weights, require_all_classes=True
    )
    point_treatment = _condition_metric(
        compiled[treatment], point_weights, require_all_classes=True
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        weights = _positive_group_weights(
            rng, groups, group_indices, len(ground_truth)
        )
        differences[index] = _condition_metric(
            compiled[treatment], weights, require_all_classes=True
        ) - _condition_metric(compiled[reference], weights, require_all_classes=True)
    low, high = np.quantile(differences, (0.025, 0.975))
    return {
        "reference": reference,
        "treatment": treatment,
        "reference_map_50_95": point_reference,
        "treatment_map_50_95": point_treatment,
        "difference": point_treatment - point_reference,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": samples,
        "bootstrap_method": "physical_scene_exp1_positive_weights",
        "technical_replicates": list(technical_replicates),
        "technical_replicate_aggregation": "mean_within_resample",
        "seed": seed,
    }


def paired_scene_permutation_test(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: PredictionReplicates,
    physical_group_by_scene: Mapping[str, str],
    class_names: Sequence[str],
    reference: str,
    treatment: str,
    samples: int = 50_000,
    seed: int = 20260804,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Swap whole physical scenes and recompute the fixed-class pooled mAP."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if (
        reference not in predictions_by_condition
        or treatment not in predictions_by_condition
    ):
        raise KeyError("Both reference and treatment predictions are required")
    selected = {
        reference: predictions_by_condition[reference],
        treatment: predictions_by_condition[treatment],
    }
    compiled, technical_replicates = _compiled_replicates(
        ground_truth, selected, class_names, min_region_size
    )
    groups, group_indices = _validate_physical_groups(
        ground_truth, physical_group_by_scene
    )
    point_weights = np.ones(len(ground_truth), dtype=np.float64)
    reference_point = _condition_metric(
        compiled[reference], point_weights, require_all_classes=True
    )
    treatment_point = _condition_metric(
        compiled[treatment], point_weights, require_all_classes=True
    )
    observed = treatment_point - reference_point
    prepared_by_seed = [
        _prepare_swap_curves(
            compiled[reference][technical_seed],
            compiled[treatment][technical_seed],
            group_indices,
            groups,
        )
        for technical_seed in technical_replicates
    ]
    no_swap_reference, no_swap_treatment = _evaluate_swap_batch(
        np.zeros((1, len(groups)), dtype=np.float64), prepared_by_seed
    )
    reconstructed = float(no_swap_treatment[0] - no_swap_reference[0])
    if not np.isclose(reconstructed, observed, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Compiled scene-swap statistic does not reproduce the pooled point estimate"
        )

    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = samples
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 8.0
    while remaining:
        batch = min(remaining, 4096)
        swaps = rng.integers(
            0, 2, size=(batch, len(groups)), dtype=np.int8
        ).astype(np.float64)
        permuted_reference, permuted_treatment = _evaluate_swap_batch(
            swaps, prepared_by_seed
        )
        differences = permuted_treatment - permuted_reference
        extreme += int(
            np.count_nonzero(np.abs(differences) >= abs(observed) - tolerance)
        )
        remaining -= batch
    return {
        "reference": reference,
        "treatment": treatment,
        "reference_map_50_95": reference_point,
        "treatment_map_50_95": treatment_point,
        "statistic": "fixed_class_pooled_map_50_95_difference",
        "observed": float(observed),
        "p_two_sided": float((extreme + 1) / (samples + 1)),
        "extreme_permutations": extreme,
        "permutation_samples": samples,
        "permutation_method": "paired_physical_scene_whole_prediction_swap",
        "technical_replicates": list(technical_replicates),
        "technical_replicate_aggregation": "mean_pooled_map_within_permutation",
        "seed": seed,
    }


def factorial_bootstrap(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: PredictionReplicates,
    condition_bits: Mapping[str, Sequence[int]],
    physical_group_by_scene: Mapping[str, str],
    class_names: Sequence[str],
    samples: int = 10_000,
    seed: int = 20260804,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Estimate the seven orthogonal 2^3 effects with scene-level uncertainty."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if set(condition_bits) != set(predictions_by_condition):
        raise ValueError("condition_bits must cover exactly the factorial conditions")
    if any(
        len(bits) != 3 or any(bit not in (0, 1) for bit in bits)
        for bits in condition_bits.values()
    ):
        raise ValueError("Each factorial condition must contain three binary factors")
    expected_combinations = {
        (size, smooth, small)
        for size in (0, 1)
        for smooth in (0, 1)
        for small in (0, 1)
    }
    observed_combinations = {tuple(bits) for bits in condition_bits.values()}
    if observed_combinations != expected_combinations:
        raise ValueError(
            "The factorial analysis requires each of the eight 2^3 "
            "combinations exactly once"
        )
    compiled, technical_replicates = _compiled_replicates(
        ground_truth, predictions_by_condition, class_names, min_region_size
    )
    groups, group_indices = _validate_physical_groups(
        ground_truth, physical_group_by_scene
    )
    rng = np.random.default_rng(seed)
    term_masks = {
        "size": 0b100,
        "smooth": 0b010,
        "small": 0b001,
        "size:smooth": 0b110,
        "size:small": 0b101,
        "smooth:small": 0b011,
        "size:smooth:small": 0b111,
    }
    def contrasts(metrics: Mapping[str, float]) -> np.ndarray:
        values = np.empty(len(term_masks), dtype=np.float64)
        for term_index, term_mask in enumerate(term_masks.values()):
            positive: list[float] = []
            negative: list[float] = []
            for name, bits in condition_bits.items():
                coded = tuple(1 if bit else -1 for bit in bits)
                signs = (
                    coded[0] if term_mask & 0b100 else 1,
                    coded[1] if term_mask & 0b010 else 1,
                    coded[2] if term_mask & 0b001 else 1,
                )
                contrast = signs[0] * signs[1] * signs[2]
                (positive if contrast > 0 else negative).append(metrics[name])
            values[term_index] = float(np.mean(positive) - np.mean(negative))
        return values

    point_weights = np.ones(len(ground_truth), dtype=np.float64)
    point_metrics = {
        condition: _condition_metric(
            replicates, point_weights, require_all_classes=True
        )
        for condition, replicates in compiled.items()
    }
    point_effects = contrasts(point_metrics)
    effects = np.empty((samples, len(term_masks)), dtype=np.float64)
    for sample_index in range(samples):
        weights = _positive_group_weights(
            rng, groups, group_indices, len(ground_truth)
        )
        metrics = {
            condition: _condition_metric(
                replicates, weights, require_all_classes=True
            )
            for condition, replicates in compiled.items()
        }
        effects[sample_index] = contrasts(metrics)

    names = tuple(term_masks)
    raw_p = []
    for index in range(len(names)):
        lower = (int(np.count_nonzero(effects[:, index] <= 0.0)) + 1) / (samples + 1)
        upper = (int(np.count_nonzero(effects[:, index] >= 0.0)) + 1) / (samples + 1)
        raw_p.append(min(1.0, 2.0 * min(lower, upper)))
    adjusted = holm_adjust(raw_p)
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        low, high = np.quantile(effects[:, index], (0.025, 0.975))
        result[name] = {
            "effect": float(point_effects[index]),
            "bootstrap_mean_effect": float(np.mean(effects[:, index])),
            "ci95": [float(low), float(high)],
            "p_two_sided": raw_p[index],
            "p_holm": adjusted[index],
        }
    return {
        "effects": result,
        "holm_family": list(names),
        "bootstrap_samples": samples,
        "bootstrap_method": "physical_scene_exp1_positive_weights",
        "technical_replicates": list(technical_replicates),
        "technical_replicate_aggregation": "mean_within_resample",
        "seed": seed,
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def analyze_manifest(
    manifest_path: str | Path,
    taxonomy: Taxonomy,
    output_path: str | Path,
    samples: int = 10_000,
    seed: int = 20260804,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    permutation_samples: int = 50_000,
) -> dict[str, Any]:
    """Run the registered paired and 2^3 analyses from one locked manifest."""
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "analysis_manifest":
        raise ValueError("Expected an analysis_manifest")
    base = Path(manifest_path).parent
    minimum_mapped_fraction = float(manifest.get("minimum_mapped_fraction", 0.90))
    if not 0.0 < minimum_mapped_fraction <= 1.0:
        raise ValueError("minimum_mapped_fraction must be in (0, 1]")
    ground_truth: list[GroundTruthScene] = []
    coordinates: dict[str, np.ndarray] = {}
    physical_groups: dict[str, str] = {}
    for item in manifest["scenes"]:
        scene_id = str(item["scene_id"])
        coords, scene = load_ground_truth_npz(base / item["gt_npz"], scene_id)
        coordinates[scene_id] = coords
        ground_truth.append(scene)
        physical_groups[scene_id] = str(item["physical_scene_id"])

    required_scenes = set(coordinates)
    predictions_by_condition: dict[
        str,
        Sequence[PredictedInstance]
        | Mapping[str | int, Sequence[PredictedInstance]],
    ] = {}
    alignment: dict[str, dict[str, Any]] = {}
    for condition, items in manifest["conditions"].items():
        items_by_seed: dict[str, dict[str, Mapping[str, Any]]] = {}
        for item in items:
            technical_seed = str(item.get("seed", "default"))
            scene_id = str(item["scene_id"])
            by_scene = items_by_seed.setdefault(technical_seed, {})
            if scene_id in by_scene:
                raise ValueError(
                    f"{condition}/{technical_seed}: duplicate scene {scene_id}"
                )
            by_scene[scene_id] = item
        condition_predictions: dict[str, list[PredictedInstance]] = {}
        alignment[condition] = {}
        for technical_seed, items_by_scene in sorted(items_by_seed.items()):
            if set(items_by_scene) != required_scenes:
                missing = sorted(required_scenes - set(items_by_scene))
                extra = sorted(set(items_by_scene) - required_scenes)
                raise ValueError(
                    f"{condition}/{technical_seed}: scene mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            replicate_predictions: list[PredictedInstance] = []
            alignment[condition][technical_seed] = {}
            for scene_id in sorted(required_scenes):
                item = items_by_scene[scene_id]
                predictions, diagnostics = saga_scene_predictions(
                    scene_id,
                    coordinates[scene_id],
                    base / item["output_json"],
                    base / item["gaussian_ply"],
                    taxonomy,
                    base / item["metadata_json"],
                    item["gaussian_to_gt_transform"],
                    radius_m,
                    require_scores=True,
                )
                if (
                    diagnostics["median_nn_distance_m"] > radius_m
                    or diagnostics["mapped_fraction"] < minimum_mapped_fraction
                ):
                    raise ValueError(
                        f"{condition}/{technical_seed}/{scene_id}: "
                        "coordinate alignment gate failed"
                    )
                replicate_predictions.extend(predictions)
                alignment[condition][technical_seed][scene_id] = diagnostics
            condition_predictions[technical_seed] = replicate_predictions
        predictions_by_condition[str(condition)] = condition_predictions

    paired_results = []
    for comparison in manifest.get("paired_comparisons", []):
        reference = str(comparison["reference"])
        treatment = str(comparison["treatment"])
        result = paired_scene_bootstrap(
            ground_truth,
            predictions_by_condition,
            physical_groups,
            taxonomy.canonical_classes,
            reference,
            treatment,
            samples,
            seed,
            min_region_size,
        )
        result["permutation_test"] = paired_scene_permutation_test(
            ground_truth,
            predictions_by_condition,
            physical_groups,
            taxonomy.canonical_classes,
            reference,
            treatment,
            permutation_samples,
            seed,
            min_region_size,
        )
        paired_results.append(result)

    factorial_bits = {
        str(name): tuple(int(value) for value in bits)
        for name, bits in manifest.get("factorial_bits", {}).items()
    }
    factorial_result = None
    if factorial_bits:
        factorial_predictions = {
            name: predictions_by_condition[name] for name in factorial_bits
        }
        factorial_result = factorial_bootstrap(
            ground_truth,
            factorial_predictions,
            factorial_bits,
            physical_groups,
            taxonomy.canonical_classes,
            samples,
            seed,
            min_region_size,
        )

    payload = {
        "schema_version": "1.0",
        "kind": "confirmatory_analysis",
        "manifest_sha256": sha256_file(manifest_path),
        "bootstrap_unit": "physical_scene",
        "bootstrap_samples": samples,
        "permutation_samples": permutation_samples,
        "seed": seed,
        "radius_m": radius_m,
        "min_region_size": min_region_size,
        "paired": paired_results,
        "factorial": factorial_result,
        "alignment": alignment,
    }
    payload["content_sha256"] = hash_json(payload)
    write_json(output_path, payload)
    return payload
