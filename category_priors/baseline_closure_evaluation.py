from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .evaluator import GroundTruthScene, PredictedInstance, evaluate_instances

# ScanNet's reference evaluator averages its main AP over .50, .55, ..., .90.
# The project historically added .95 while retaining an "official" label.  Keep
# both definitions explicit so old numbers can be reproduced without conflating
# the two protocols.
SCANNET_OFFICIAL_OVERLAPS = tuple(np.arange(0.50, 0.95, 0.05).round(2).tolist())
HISTORICAL_OVERLAPS = tuple(np.arange(0.50, 0.96, 0.05).round(2).tolist())

ScoreMode = Literal["unit", "final_vote", "gt_oracle"]
ScoreKey = tuple[str, int]


@dataclass(frozen=True)
class AdaptedPredictions:
    predictions: tuple[PredictedInstance, ...]
    mode: ScoreMode
    diagnostic_only: bool


def _score_key(prediction: PredictedInstance) -> ScoreKey:
    return prediction.scene_id, int(prediction.instance_id)


def _validate_score(value: float, key: ScoreKey) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{key}: invalid instance score {value}")
    return score


def _oracle_iou_scores(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    min_region_size: int,
) -> dict[ScoreKey, float]:
    gt_by_scene = {scene.scene_id: scene for scene in ground_truth}
    if len(gt_by_scene) != len(ground_truth):
        raise ValueError("Ground-truth scene ids must be unique")

    masks: dict[tuple[str, int], list[np.ndarray]] = {}
    for scene in ground_truth:
        if scene.semantic.shape != scene.instance.shape or scene.semantic.ndim != 1:
            raise ValueError(f"{scene.scene_id}: invalid GT array shapes")
        for class_id in np.unique(scene.semantic):
            class_id = int(class_id)
            if class_id < 0:
                continue
            class_mask = scene.semantic == class_id
            class_instances: list[np.ndarray] = []
            for instance_id in np.unique(scene.instance[class_mask]):
                instance_id = int(instance_id)
                if instance_id < 0:
                    continue
                mask = class_mask & (scene.instance == instance_id)
                if int(mask.sum()) >= min_region_size:
                    class_instances.append(mask)
            masks[(scene.scene_id, class_id)] = class_instances

    scores: dict[ScoreKey, float] = {}
    for prediction in predictions:
        scene = gt_by_scene.get(prediction.scene_id)
        if scene is None:
            raise ValueError(
                f"Prediction references unknown scene: {prediction.scene_id}"
            )
        if prediction.mask.shape != scene.semantic.shape:
            raise ValueError(
                f"{prediction.scene_id}: prediction mask shape does not match GT"
            )
        key = _score_key(prediction)
        if key in scores:
            raise ValueError(f"Duplicate prediction identity: {key}")
        prediction_mask = np.asarray(prediction.mask, dtype=bool)
        prediction_count = int(prediction_mask.sum())
        best_iou = 0.0
        if prediction_count:
            for gt_mask in masks.get(
                (prediction.scene_id, int(prediction.class_id)), []
            ):
                intersection = int(np.count_nonzero(prediction_mask & gt_mask))
                if not intersection:
                    continue
                union = prediction_count + int(gt_mask.sum()) - intersection
                best_iou = max(best_iou, intersection / union)
        scores[key] = best_iou
    return scores


def adapt_prediction_scores(
    predictions: Sequence[PredictedInstance],
    mode: ScoreMode,
    *,
    final_vote_scores: Mapping[ScoreKey, float] | None = None,
    ground_truth: Sequence[GroundTruthScene] | None = None,
    min_region_size: int = 100,
) -> AdaptedPredictions:
    """Return score-adapted copies without mutating baseline predictions.

    ``unit`` removes confidence calibration from the comparison. ``final_vote``
    consumes an explicitly supplied final semantic-vote confidence for every
    instance. ``gt_oracle`` ranks by best same-class GT IoU and is marked
    diagnostic-only because it reads evaluation ground truth.
    """

    if min_region_size <= 0:
        raise ValueError("min_region_size must be positive")
    if mode not in {"unit", "final_vote", "gt_oracle"}:
        raise ValueError(f"Unknown score mode: {mode}")

    oracle_scores: Mapping[ScoreKey, float] | None = None
    if mode == "final_vote":
        if final_vote_scores is None:
            raise ValueError("final_vote mode requires final_vote_scores")
    elif mode == "gt_oracle":
        if ground_truth is None:
            raise ValueError("gt_oracle mode requires ground_truth")
        oracle_scores = _oracle_iou_scores(ground_truth, predictions, min_region_size)

    adapted: list[PredictedInstance] = []
    seen: set[ScoreKey] = set()
    for prediction in predictions:
        key = _score_key(prediction)
        if key in seen:
            raise ValueError(f"Duplicate prediction identity: {key}")
        seen.add(key)
        if mode == "unit":
            score = 1.0
        elif mode == "final_vote":
            assert final_vote_scores is not None
            if key not in final_vote_scores:
                raise ValueError(f"Missing final-vote score for {key}")
            score = _validate_score(final_vote_scores[key], key)
        else:
            assert oracle_scores is not None
            score = _validate_score(oracle_scores[key], key)
        adapted.append(
            PredictedInstance(
                scene_id=prediction.scene_id,
                instance_id=int(prediction.instance_id),
                class_id=int(prediction.class_id),
                score=score,
                mask=np.asarray(prediction.mask, dtype=bool).copy(),
            )
        )

    return AdaptedPredictions(
        predictions=tuple(adapted),
        mode=mode,
        diagnostic_only=mode == "gt_oracle",
    )


def _restrict_to_classes(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_ids: set[int],
) -> tuple[tuple[GroundTruthScene, ...], tuple[PredictedInstance, ...]]:
    restricted_gt: list[GroundTruthScene] = []
    for scene in ground_truth:
        keep = np.isin(scene.semantic, np.asarray(sorted(class_ids), dtype=np.int64))
        restricted_gt.append(
            GroundTruthScene(
                scene_id=scene.scene_id,
                semantic=np.where(keep, scene.semantic, -1).astype(np.int64),
                instance=np.where(keep, scene.instance, -1).astype(np.int64),
            )
        )
    restricted_predictions = tuple(
        prediction
        for prediction in predictions
        if int(prediction.class_id) in class_ids
    )
    return tuple(restricted_gt), restricted_predictions


def _protocol_result(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    overlaps: Sequence[float],
    min_region_size: int,
    *,
    protocol_name: str,
    protocol_version: str,
    primary_metric: str,
) -> dict[str, object]:
    result = evaluate_instances(
        ground_truth,
        predictions,
        class_names,
        overlaps=overlaps,
        min_region_size=min_region_size,
    )
    # evaluate_instances historically names every main average map_50_95.
    # Rename that field for the official nine-threshold view so downstream code
    # cannot accidentally report .50-.90 as .50-.95.
    raw_primary = result["aggregate"].pop("map_50_95")
    result["aggregate"][primary_metric] = raw_primary
    for class_result in result["per_class"].values():
        raw_class_primary = class_result.pop("ap_50_95")
        class_result[primary_metric.replace("map_", "ap_")] = raw_class_primary
    result["protocol"] = protocol_name
    result["protocol_version"] = protocol_version
    result["primary_metric"] = primary_metric
    return result


def evaluate_baseline_closure(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    *,
    predictable_classes: Sequence[str],
    score_mode: ScoreMode = "unit",
    final_vote_scores: Mapping[ScoreKey, float] | None = None,
    min_region_size: int = 100,
) -> dict[str, object]:
    """Evaluate one frozen baseline under both protocol and class views."""

    normalized_classes = tuple(str(name).strip().lower() for name in class_names)
    if not normalized_classes or len(set(normalized_classes)) != len(
        normalized_classes
    ):
        raise ValueError("class_names must be non-empty and unique")
    class_to_id = {name: index for index, name in enumerate(normalized_classes)}
    normalized_predictable = tuple(
        dict.fromkeys(str(name).strip().lower() for name in predictable_classes)
    )
    unknown = sorted(set(normalized_predictable) - set(class_to_id))
    if unknown:
        raise ValueError(f"Unknown predictable classes: {unknown}")
    if not normalized_predictable:
        raise ValueError("predictable_classes must be non-empty")

    adapted = adapt_prediction_scores(
        predictions,
        score_mode,
        final_vote_scores=final_vote_scores,
        ground_truth=ground_truth,
        min_region_size=min_region_size,
    )
    predictable_ids = {class_to_id[name] for name in normalized_predictable}
    restricted_gt, restricted_predictions = _restrict_to_classes(
        ground_truth, adapted.predictions, predictable_ids
    )

    protocol_specs = (
        (
            "scannet_official_9",
            SCANNET_OFFICIAL_OVERLAPS,
            "ScanNet-official-instance-9-threshold",
            "scannet-official-instance-9-v1",
            "map_50_90",
        ),
        (
            "historical_10",
            HISTORICAL_OVERLAPS,
            "ScanNet200-SAGA20-historical-10-threshold",
            "saga20-historical-instance-10-v1",
            "map_50_95",
        ),
    )
    protocols: dict[str, object] = {}
    for (
        key,
        overlaps,
        protocol_name,
        protocol_version,
        primary_metric,
    ) in protocol_specs:
        protocols[key] = {
            "overlaps": list(overlaps),
            "primary_metric": primary_metric,
            "full_saga20": _protocol_result(
                ground_truth,
                adapted.predictions,
                normalized_classes,
                overlaps,
                min_region_size,
                protocol_name=protocol_name,
                protocol_version=protocol_version,
                primary_metric=primary_metric,
            ),
            "predictable_intersection": _protocol_result(
                restricted_gt,
                restricted_predictions,
                normalized_classes,
                overlaps,
                min_region_size,
                protocol_name=protocol_name,
                protocol_version=protocol_version,
                primary_metric=primary_metric,
            ),
        }

    return {
        "schema_version": "1.0",
        "kind": "baseline_closure_evaluation",
        "score_adapter": {
            "mode": adapted.mode,
            "diagnostic_only": adapted.diagnostic_only,
        },
        "class_views": {
            "full_saga20": list(normalized_classes),
            "predictable_intersection": list(normalized_predictable),
        },
        "min_region_size": min_region_size,
        "protocols": protocols,
    }
