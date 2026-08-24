from __future__ import annotations

"""Neutral official/precision metrics used by the V9 continuation stages.

This module intentionally does not import an earlier experimental pipeline.
It is the single evaluator for legacy references and frozen ObjectBank replay
predictions in stages 3--6.  Ground truth is read only here, never by feature,
lifting, bank, or replay workers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .evaluator import (
    OVERLAPS,
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
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v9_evaluation import CandidateSupport, GroundTruthSupport, evaluate_object_candidates
from .v9_replay import validate_prediction_contract
from .v9_runner import load_v9_candidate_bank


@dataclass(frozen=True)
class ScanNetSceneAPEvent:
    """One scene/class/IoU contribution to an official ScanNet AP curve."""

    gt_count: int
    y_true: np.ndarray
    y_score: np.ndarray
    hard_false_negatives: int


@dataclass(frozen=True)
class ScanNetSceneAPEvents:
    """Scene-separable events from which pooled official AP can be recomputed."""

    scene_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    overlaps: tuple[float, ...]
    min_region_size: int
    events: tuple[tuple[tuple[ScanNetSceneAPEvent, ...], ...], ...]


# Frozen before Stage 2 is interpreted.  Association truth is intentionally
# limited to fragments with non-trivial, predominantly single-object support;
# one accidentally mapped point is not evidence that two fragments share an
# object identity.
ASSOCIATION_TRUTH_MIN_INTERSECTION = 3
ASSOCIATION_TRUTH_MIN_IOU = 0.05
ASSOCIATION_TRUTH_MIN_PURITY = 0.50


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    explicit = scene.get("gaussian_ply")
    if explicit:
        value = Path(str(explicit))
        return value if value.is_absolute() else Path(str(scene["base_path"])) / value
    root = Path(str(scene["base_path"])) / "output_models/point_cloud/iteration_30000"
    registered = root / "scene_point_cloud.ply"
    return registered if registered.is_file() else root / "point_cloud.ply"


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


def _output_dir(root: Path, condition: str, scene_id: str) -> Path:
    direct = root / condition / scene_id
    if (direct / "output.json").is_file():
        return direct
    seeded = direct / "seed-42"
    if (seeded / "output.json").is_file():
        return seeded
    raise FileNotFoundError(direct / "output.json")


def _size_bin(diagonal_m: float, spec: Mapping[str, Any] | None) -> str | None:
    if spec is None:
        return None
    limits = spec["boundaries_m"]
    if diagonal_m <= float(limits["tiny_max_m"]):
        return "tiny"
    if diagonal_m <= float(limits["small_max_m"]):
        return "small"
    if diagonal_m <= float(limits["medium_max_m"]):
        return "medium"
    return "large"


def _bbox_diagonal(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if not len(xyz):
        return 0.0
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    if len(xyz) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(centered.max(axis=0) - centered.min(axis=0)))


def _contract_audit(labels: np.ndarray, instances: Mapping[str, Any]) -> dict[str, int]:
    declared: set[int] = set()
    negative = 0
    for raw_id in instances:
        instance_id = int(raw_id)
        if instance_id < 0:
            negative += 1
        else:
            declared.add(instance_id)
    assigned = set(map(int, np.unique(labels[labels >= 0])))
    return {
        "orphan_gaussian_count": int(
            np.count_nonzero(np.isin(labels, sorted(assigned - declared)))
        ),
        "negative_metadata_count": negative,
    }


def _greedy_tp_fp(
    predictions: Sequence[PredictedInstance],
    gt: GroundTruthScene,
    *,
    threshold: float = 0.25,
    min_region_size: int = 100,
) -> tuple[int, int]:
    valid = (gt.semantic >= 0) & (gt.instance >= 0)
    gt_masks: dict[tuple[int, int], np.ndarray] = {}
    for class_id, instance_id in sorted(
        set(zip(gt.semantic[valid].tolist(), gt.instance[valid].tolist()))
    ):
        mask = valid & (gt.semantic == class_id) & (gt.instance == instance_id)
        if int(mask.sum()) >= min_region_size:
            gt_masks[(int(class_id), int(instance_id))] = mask
    used: set[tuple[int, int]] = set()
    true_positive = 0
    false_positive = 0
    for prediction in sorted(
        predictions, key=lambda row: (-float(row.score), int(row.instance_id))
    ):
        if int(np.count_nonzero(prediction.mask)) < min_region_size:
            continue
        best: tuple[float, tuple[int, int] | None] = (0.0, None)
        for key, gt_mask in gt_masks.items():
            if key in used or key[0] != int(prediction.class_id):
                continue
            intersection = int(np.count_nonzero(prediction.mask & gt_mask))
            union = int(np.count_nonzero(prediction.mask | gt_mask))
            iou = intersection / union if union else 0.0
            if iou > best[0]:
                best = (iou, key)
        if best[1] is not None and best[0] > threshold:
            used.add(best[1])
            true_positive += 1
        else:
            false_positive += 1
    return true_positive, false_positive


def _unique_official_gt_coverage(
    gt: GroundTruthScene,
    predictions: Sequence[PredictedInstance],
    *,
    min_region_size: int,
) -> dict[str, float | int]:
    """Measure GT coverage once per official-valid instance.

    A missed GT contributes zero.  Multiple predictions that target the same
    GT cannot increase its weight: only the largest same-class intersection is
    retained.  The macro result therefore answers the registered instance
    recall question, while the micro result exposes point-mass effects.
    """

    valid = (gt.semantic >= 0) & (gt.instance >= 0)
    recall_sum = 0.0
    gt_instance_count = 0
    gt_point_count = 0
    covered_point_count = 0
    for class_id, instance_id in sorted(
        set(zip(gt.semantic[valid].tolist(), gt.instance[valid].tolist()))
    ):
        gt_mask = valid & (gt.semantic == class_id) & (gt.instance == instance_id)
        point_count = int(np.count_nonzero(gt_mask))
        if point_count < int(min_region_size):
            continue
        best_intersection = max(
            (
                int(np.count_nonzero(gt_mask & prediction.mask))
                for prediction in predictions
                if int(prediction.class_id) == int(class_id)
            ),
            default=0,
        )
        gt_instance_count += 1
        gt_point_count += point_count
        covered_point_count += best_intersection
        recall_sum += best_intersection / point_count
    return {
        "official_gt_instance_count": gt_instance_count,
        "official_gt_point_count": gt_point_count,
        "best_covered_gt_point_count": covered_point_count,
        "gt_instance_recall_sum": recall_sum,
        "gt_instance_macro_recall": (
            recall_sum / gt_instance_count if gt_instance_count else 0.0
        ),
        "gt_point_micro_recall": (
            covered_point_count / gt_point_count if gt_point_count else 0.0
        ),
    }


def _all_class_gt_masks(
    scene: GroundTruthScene, class_id: int
) -> list[np.ndarray]:
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
    """Run the official scene-local matching loop and retain its AP events."""

    all_gt = _all_class_gt_masks(scene, class_id)
    gt_counts = [int(mask.sum()) for mask in all_gt]
    valid_gt_indices = [
        index
        for index, count in enumerate(gt_counts)
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
        for (
            prediction_index,
            prediction,
            pred_count,
            _,
            intersections,
        ) in prepared:
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
            if not intersection:
                continue
            iou = intersection / (gt_count + pred_count - intersection)
            if iou > threshold:
                confidence = float(prediction.score)
                if current_match[local_gt_index]:
                    previous_score = current_score[local_gt_index]
                    current_score[local_gt_index] = max(
                        previous_score, confidence
                    )
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
    """Precompute the exact official matching events independently per scan."""

    scenes = tuple(ground_truth)
    scene_by_id = {str(scene.scene_id): scene for scene in scenes}
    if len(scene_by_id) != len(scenes):
        raise ValueError("ground-truth scene IDs must be unique")
    normalized_overlaps = tuple(float(value) for value in overlaps)
    normalized_classes = tuple(map(str, class_names))
    pred_by_class_scene: dict[
        int, dict[str, list[tuple[int, PredictedInstance]]]
    ] = {index: {} for index in range(len(normalized_classes))}
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
    scene_events: Sequence[ScanNetSceneAPEvent], scene_weights: np.ndarray
) -> np.ndarray:
    """Vectorized exact ScanNet AP for several scan-multiplicity rows."""

    weights = np.asarray(scene_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(scene_events):
        raise ValueError("scene_weights must have one column per scan")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("scene weights must be finite and non-negative")
    gt_counts = np.asarray(
        [event.gt_count for event in scene_events], dtype=np.float64
    )
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
        (
            recall_curve[:, :1],
            recall_curve,
            np.zeros((len(weights), 1), dtype=np.float64),
        ),
        axis=1,
    )
    step_widths = 0.5 * (padded_recall[:, :-2] - padded_recall[:, 2:])
    ap = np.sum(precision_curve * step_widths, axis=1)
    result[active] = ap[active]
    return result


def scannet_map_samples_from_scene_weights(
    events: ScanNetSceneAPEvents, scene_weights: np.ndarray
) -> np.ndarray:
    """Recompute pooled official mAP for every scan-multiplicity sample."""

    weights = np.asarray(scene_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = weights[None, :]
    if weights.ndim != 2 or weights.shape[1] != len(events.scene_ids):
        raise ValueError("scene_weights must have one column per event scan")
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
    """Return official pooled per-class/per-IoU AP under scan weights."""

    weights = (
        np.ones(len(events.scene_ids), dtype=np.float64)
        if scene_weights is None
        else np.asarray(scene_weights, dtype=np.float64)
    )
    if weights.shape != (len(events.scene_ids),):
        raise ValueError("scene_weights must have one value per event scan")
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
            row[f"ap_{overlap:.2f}"] = (
                float(value) if np.isfinite(value) else None
            )
        valid = [value for value in row.values() if value is not None]
        row["ap_50_95"] = float(np.mean(valid)) if valid else None
        per_class[class_name] = row
    aggregate: dict[str, float | None] = {}
    for overlap in events.overlaps:
        values = [row[f"ap_{overlap:.2f}"] for row in per_class.values()]
        valid = [float(value) for value in values if value is not None]
        aggregate[f"map_{overlap:.2f}"] = (
            float(np.mean(valid)) if valid else None
        )
    valid_main = [value for value in aggregate.values() if value is not None]
    aggregate["map_50_95"] = (
        float(np.mean(valid_main)) if valid_main else None
    )
    return {"aggregate": aggregate, "per_class": per_class}


def paired_scannet_scene_bootstrap(
    reference: ScanNetSceneAPEvents,
    treatment: ScanNetSceneAPEvents,
    *,
    physical_scene_ids: Sequence[str] | None = None,
    samples: int = 10_000,
    seed: int = 20_260_804,
    batch_size: int = 512,
) -> dict[str, Any]:
    """Paired physical-scene bootstrap of pooled official ScanNet mAP."""

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
        raise ValueError("paired AP event tables must describe one protocol")
    if samples <= 0 or batch_size <= 0 or not reference.scene_ids:
        raise ValueError("bootstrap requires positive sizes and at least one scan")
    groups = tuple(
        physical_scene_ids
        if physical_scene_ids is not None
        else (scene_id.rsplit("_", 1)[0] for scene_id in reference.scene_ids)
    )
    if len(groups) != len(reference.scene_ids):
        raise ValueError("physical_scene_ids must have one entry per scan")
    unique_groups = tuple(dict.fromkeys(map(str, groups)))
    group_index = {name: index for index, name in enumerate(unique_groups)}
    scan_group_indices = np.asarray(
        [group_index[str(name)] for name in groups], dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    deltas: list[np.ndarray] = []
    remaining = int(samples)
    while remaining:
        count = min(int(batch_size), remaining)
        group_weights = rng.multinomial(
            len(unique_groups),
            np.full(len(unique_groups), 1.0 / len(unique_groups)),
            size=count,
        )
        scan_weights = group_weights[:, scan_group_indices]
        reference_samples = scannet_map_samples_from_scene_weights(
            reference, scan_weights
        )
        treatment_samples = scannet_map_samples_from_scene_weights(
            treatment, scan_weights
        )
        deltas.append(treatment_samples - reference_samples)
        remaining -= count
    delta_samples = np.concatenate(deltas)
    finite = delta_samples[np.isfinite(delta_samples)]
    if not len(finite):
        raise ValueError("no bootstrap replicate has an evaluable ScanNet mAP")
    reference_point = pooled_scannet_metrics_from_scene_weights(reference)
    treatment_point = pooled_scannet_metrics_from_scene_weights(treatment)
    delta_point = float(
        treatment_point["aggregate"]["map_50_95"]
        - reference_point["aggregate"]["map_50_95"]
    )
    return {
        "schema": "saga-v9-paired-scannet-bootstrap-v1",
        "statistic": "pooled_official_scannet_map_50_95",
        "scan_count": len(reference.scene_ids),
        "physical_scene_count": len(unique_groups),
        "samples": int(samples),
        "seed": int(seed),
        "reference_map_50_95": reference_point["aggregate"]["map_50_95"],
        "treatment_map_50_95": treatment_point["aggregate"]["map_50_95"],
        "delta_map_50_95": delta_point,
        "paired_bootstrap_ci95": [
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
        ],
        "finite_sample_count": int(len(finite)),
    }


def evaluate_v9_predictions(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prediction_root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    viewer_output: Path | None = None,
) -> dict[str, Any]:
    """Evaluate strict V9 predictions once under all registered diagnostics."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    missing = sorted(set(map(str, scene_ids)).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    size_spec = load_json(size_bins) if size_bins else None
    rows: list[dict[str, Any]] = []
    analysis: dict[str, Any] = {
        "schema": "saga-v9-object-system-analysis-v1",
        "conditions": {},
    }
    viewer_rows: list[dict[str, Any]] = []
    viewer_audits: dict[tuple[str, str], dict[str, Any]] = {}
    viewer_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for condition in map(str, conditions):
        all_gt: list[GroundTruthScene] = []
        all_predictions: list[PredictedInstance] = []
        per_scene: list[dict[str, Any]] = []
        audit_aggregates: list[dict[str, Any]] = []
        total_tiny = 0
        matched_tiny_050 = 0
        orphan = 0
        negative = 0
        total_tp = 0
        total_fp = 0
        gt_instance_count = 0
        gt_instance_recall_sum = 0.0
        gt_point_count = 0
        covered_gt_point_count = 0
        matched_prediction_recalls: list[float] = []
        for scene_id in map(str, scene_ids):
            scene = scenes[scene_id]
            gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
            output = load_json(_output_dir(prediction_root, condition, scene_id) / "output.json")
            labels = np.asarray(output.get("point_labels"), dtype=np.int64)
            instances = output.get("instances")
            if labels.ndim != 1 or not isinstance(instances, Mapping):
                raise ValueError(f"{condition}/{scene_id}: invalid prediction payload")
            validate_prediction_contract(labels, instances)
            contract = _contract_audit(labels, instances)
            orphan += contract["orphan_gaussian_count"]
            negative += contract["negative_metadata_count"]
            gaussian_xyz = apply_transform(
                load_ply_xyz(_gaussian_ply(scene)), _transform(scene)
            )
            if len(labels) != len(gaussian_xyz):
                raise ValueError(f"{condition}/{scene_id}: point count mismatch")
            mapped_labels, _ = map_gaussians_to_gt(
                gt_xyz, gaussian_xyz, labels, radius_m
            )
            predictions: list[PredictedInstance] = []
            for raw_id, metadata in instances.items():
                instance_id = int(raw_id)
                class_name = str(metadata.get("class", ""))
                if instance_id < 0 or class_name not in class_to_id:
                    continue
                predictions.append(
                    PredictedInstance(
                        scene_id=scene_id,
                        instance_id=instance_id,
                        class_id=class_to_id[class_name],
                        score=float(metadata.get("score", 1.0)),
                        mask=mapped_labels == instance_id,
                    )
                )
            scene_official = evaluate_instances(
                [gt], predictions, taxonomy.canonical_classes,
                min_region_size=min_region_size,
            )
            gt_coverage = _unique_official_gt_coverage(
                gt, predictions, min_region_size=min_region_size
            )
            gt_instance_count += int(gt_coverage["official_gt_instance_count"])
            gt_instance_recall_sum += float(gt_coverage["gt_instance_recall_sum"])
            gt_point_count += int(gt_coverage["official_gt_point_count"])
            covered_gt_point_count += int(
                gt_coverage["best_covered_gt_point_count"]
            )
            valid_gt = (gt.semantic >= 0) & (gt.instance >= 0)
            scene_tiny = 0
            scene_tiny_hit = 0
            for class_id, instance_id in sorted(
                set(zip(gt.semantic[valid_gt].tolist(), gt.instance[valid_gt].tolist()))
            ):
                mask = valid_gt & (gt.semantic == class_id) & (gt.instance == instance_id)
                if int(mask.sum()) < min_region_size:
                    continue
                if _size_bin(_bbox_diagonal(gt_xyz[mask]), size_spec) not in {"tiny", "small"}:
                    continue
                same_class = [row.mask for row in predictions if row.class_id == int(class_id)]
                best = 0.0
                for pred_mask in same_class:
                    intersection = int(np.count_nonzero(mask & pred_mask))
                    union = int(np.count_nonzero(mask | pred_mask))
                    best = max(best, intersection / union if union else 0.0)
                scene_tiny += 1
                scene_tiny_hit += int(best > 0.50)
            audit = evaluate_gaussian_object_precision(
                gaussian_xyz,
                labels,
                instances,
                gt_xyz,
                gt.semantic,
                gt.instance,
                radius_m,
                canonical_classes=taxonomy.canonical_classes,
            )
            if viewer_output is not None:
                viewer_audits[(scene_id, condition)] = {
                    **audit,
                    "point_labels": labels,
                }
                viewer_arrays[scene_id] = (
                    gt_xyz,
                    gt.semantic,
                    gt.instance,
                    gaussian_xyz,
                )
                viewer_rows.extend(
                    {
                        "scene_id": scene_id,
                        "condition": condition,
                        **dict(instance_row),
                    }
                    for instance_row in audit["instances"]
                )
            audit_aggregates.append(audit["aggregate"])
            matched_prediction_recalls.extend(
                float(instance_row["gt_to_gaussian_recall"])
                for instance_row in audit["instances"]
                if instance_row["dominant_gt_instance"] is not None
            )
            tp, fp = _greedy_tp_fp(
                predictions, gt, min_region_size=min_region_size
            )
            total_tp += tp
            total_fp += fp
            total_tiny += scene_tiny
            matched_tiny_050 += scene_tiny_hit
            per_scene.append(
                {
                    "scene_id": scene_id,
                    **scene_official["aggregate"],
                    "predicted_instance_count": int(audit["aggregate"]["predicted_instance_count"]),
                    "gaussian_micro_precision": float(audit["aggregate"]["micro_point_precision"]),
                    "unsupported_instance_fraction": (
                        int(audit["aggregate"]["unsupported_prediction_count"])
                        / max(int(audit["aggregate"]["predicted_instance_count"]), 1)
                    ),
                    # Compatibility alias consumed by the frozen Stage-3 gate.
                    # Its semantics are now unique official-GT macro coverage.
                    "gt_recall": float(gt_coverage["gt_instance_macro_recall"]),
                    "gt_instance_macro_recall": float(
                        gt_coverage["gt_instance_macro_recall"]
                    ),
                    "gt_point_micro_recall": float(
                        gt_coverage["gt_point_micro_recall"]
                    ),
                    "official_gt_instance_count": int(
                        gt_coverage["official_gt_instance_count"]
                    ),
                    "official_gt_point_count": int(
                        gt_coverage["official_gt_point_count"]
                    ),
                    "predicted_instance_mean_matched_gt_recall": float(
                        audit["aggregate"]["mean_matched_gt_recall"]
                    ),
                    "tiny_small_gt_count": scene_tiny,
                    "tiny_small_match_050_count": scene_tiny_hit,
                    "tiny_small_recall_050": scene_tiny_hit / scene_tiny if scene_tiny else 0.0,
                    "true_positive_count": tp,
                    "false_positive_count": fp,
                    **contract,
                }
            )
            all_gt.append(gt)
            all_predictions.extend(predictions)
        official = evaluate_instances(
            all_gt, all_predictions, taxonomy.canonical_classes,
            min_region_size=min_region_size,
        )
        total_gaussians = sum(int(row["predicted_gaussian_count"]) for row in audit_aggregates)
        total_correct = sum(int(row["correct_gaussian_count"]) for row in audit_aggregates)
        total_instances = sum(int(row["predicted_instance_count"]) for row in audit_aggregates)
        total_unsupported = sum(int(row["unsupported_prediction_count"]) for row in audit_aggregates)
        row = {
            "condition": condition,
            "scene_count": len(per_scene),
            **official["aggregate"],
            "ap50": official["aggregate"].get("map_0.50"),
            "predicted_instance_count": total_instances,
            "gaussian_micro_precision": total_correct / total_gaussians if total_gaussians else 0.0,
            "unsupported_instance_fraction": total_unsupported / total_instances if total_instances else 0.0,
            # Compatibility alias; unlike the old prediction-weighted value,
            # every official-valid GT contributes exactly once and misses are 0.
            "gt_recall": (
                gt_instance_recall_sum / gt_instance_count
                if gt_instance_count
                else 0.0
            ),
            "gt_instance_macro_recall": (
                gt_instance_recall_sum / gt_instance_count
                if gt_instance_count
                else 0.0
            ),
            "gt_point_micro_recall": (
                covered_gt_point_count / gt_point_count if gt_point_count else 0.0
            ),
            "official_gt_instance_count": gt_instance_count,
            "official_gt_point_count": gt_point_count,
            "predicted_instance_mean_matched_gt_recall": (
                float(np.mean(matched_prediction_recalls))
                if matched_prediction_recalls
                else 0.0
            ),
            "tiny_small_gt_count": total_tiny,
            "tiny_small_match_050_count": matched_tiny_050,
            "tiny_small_recall_050": matched_tiny_050 / total_tiny if total_tiny else 0.0,
            "true_positive_count": total_tp,
            "false_positive_count": total_fp,
            "orphan_gaussian_count": orphan,
            "negative_metadata_count": negative,
        }
        rows.append(row)
        analysis["conditions"][condition] = {
            "official": official,
            "metrics": row,
            "per_scene": per_scene,
        }
    if viewer_output is not None:
        cases: list[dict[str, Any]] = []
        for case in _select_viewer_cases(viewer_rows):
            scene_id = str(case["scene_id"])
            condition = str(case["condition"])
            gt_xyz, gt_semantic, gt_instance, gaussian_xyz = viewer_arrays[scene_id]
            audit = viewer_audits[(scene_id, condition)]
            cases.append(
                _export_viewer_case(
                    case,
                    audit,
                    gt_xyz,
                    gt_semantic,
                    gt_instance,
                    gaussian_xyz,
                    np.asarray(audit["point_labels"], dtype=np.int64),
                    viewer_output,
                )
            )
        analysis["viewer"] = {"directory": str(viewer_output), "cases": cases}
    write_rows(metrics_output, rows)
    write_json(analysis_output, analysis)
    return analysis


def _nearest_gaussians(
    gt_xyz: np.ndarray, gaussian_xyz: np.ndarray, radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = cKDTree(gaussian_xyz).query(
        gt_xyz, k=1, distance_upper_bound=float(radius_m), workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(gaussian_xyz))
    return np.asarray(indices, dtype=np.int64), valid


def _mapped_support(
    gaussian_ids: np.ndarray,
    nearest: np.ndarray,
    valid: np.ndarray,
    *,
    sentinel: int,
) -> np.ndarray:
    selected = valid & np.isin(nearest, np.asarray(gaussian_ids, dtype=np.int64))
    support = np.flatnonzero(selected).astype(np.int64)
    return support if len(support) else np.asarray([sentinel], dtype=np.int64)


def _gt_supports(
    scene_id: str,
    gt_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_names: Sequence[str],
    size_spec: Mapping[str, Any] | None,
    min_region_size: int,
) -> list[GroundTruthSupport]:
    result: list[GroundTruthSupport] = []
    valid = (semantic >= 0) & (instance >= 0)
    for class_id, instance_id in sorted(
        set(zip(semantic[valid].tolist(), instance[valid].tolist()))
    ):
        mask = valid & (semantic == class_id) & (instance == instance_id)
        ids = np.flatnonzero(mask).astype(np.int64)
        if not len(ids) or not 0 <= int(class_id) < len(class_names):
            continue
        result.append(
            GroundTruthSupport(
                scene_id=scene_id,
                instance_id=int(instance_id),
                support_ids=ids,
                class_name=str(class_names[int(class_id)]),
                size_bin=_size_bin(_bbox_diagonal(gt_xyz[ids]), size_spec),
                support_count=len(ids),
                official_valid=len(ids) >= int(min_region_size),
            )
        )
    return result


def _qualifying_association_gt(
    support: np.ndarray,
    gt_supports: Sequence[GroundTruthSupport],
) -> tuple[str, int] | None:
    """Return a reliable dominant GT identity for one fragment, if any."""

    fragment = np.asarray(support, dtype=np.int64)
    if fragment.ndim != 1:
        raise ValueError("fragment support must be one-dimensional")
    scores: list[tuple[float, float, int, str, int]] = []
    for gt in gt_supports:
        intersection = int(
            len(np.intersect1d(fragment, gt.support_ids, assume_unique=True))
        )
        union = len(fragment) + len(gt.support_ids) - intersection
        iou = intersection / union if union else 0.0
        purity = intersection / len(fragment) if len(fragment) else 0.0
        scores.append(
            (iou, purity, intersection, str(gt.class_name), int(gt.instance_id))
        )
    if not scores:
        return None
    best = min(
        scores,
        key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]),
    )
    iou, purity, intersection, class_name, instance_id = best
    if (
        intersection < ASSOCIATION_TRUTH_MIN_INTERSECTION
        or iou < ASSOCIATION_TRUTH_MIN_IOU
        or purity < ASSOCIATION_TRUTH_MIN_PURITY
    ):
        return None
    return class_name, instance_id


def _association_diagnostics(
    bank_metadata: Mapping[str, Any],
    gt_supports: Sequence[GroundTruthSupport],
    nearest: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    source = Path(str(bank_metadata["source_lifting_bank"]))
    with np.load(source / "lifting_bank.npz", allow_pickle=False) as arrays:
        fragment_ids = np.asarray(arrays["fragment_id"], dtype=np.int64)
        frames = np.asarray(arrays["fragment_frame"], dtype=np.int64)
        indptr = np.asarray(arrays["fragment_full_indptr"], dtype=np.int64)
        values = np.asarray(arrays["fragment_full_ids"], dtype=np.int64)
    best_gt: dict[int, tuple[str, int] | None] = {}
    for index, fragment_id in enumerate(fragment_ids):
        support = _mapped_support(
            values[indptr[index] : indptr[index + 1]],
            nearest,
            valid,
            sentinel=len(nearest) + 1_000_000 + index,
        )
        best_gt[int(fragment_id)] = _qualifying_association_gt(
            support, gt_supports
        )
    frame_by_fragment = {
        int(fragment_id): int(frame)
        for fragment_id, frame in zip(fragment_ids, frames)
    }
    positive: set[tuple[int, int]] = set()
    ordered = sorted(map(int, fragment_ids))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if frame_by_fragment[left] != frame_by_fragment[right] and best_gt[left] is not None and best_gt[left] == best_gt[right]:
                positive.add((left, right))
    predicted = {
        tuple(sorted((int(row["left_fragment_id"]), int(row["right_fragment_id"]))))
        for row in bank_metadata.get("accepted_edges", ())
    }
    true_positive = len(predicted & positive)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(positive) if positive else 0.0
    return {
        "predicted_pair_count": len(predicted),
        "oracle_positive_pair_count": len(positive),
        "true_positive_pair_count": true_positive,
        "association_pair_precision": precision,
        "association_pair_recall": recall,
        "association_pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "association_truth_eligible_fragment_count": sum(
            value is not None for value in best_gt.values()
        ),
        "association_truth_thresholds": {
            "min_intersection": ASSOCIATION_TRUTH_MIN_INTERSECTION,
            "min_iou": ASSOCIATION_TRUTH_MIN_IOU,
            "min_purity": ASSOCIATION_TRUTH_MIN_PURITY,
        },
    }


def evaluate_v9_candidate_banks(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    association_mode: str,
    classifier: str,
    taxonomy: Taxonomy,
    rows_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Offline candidate/association evaluation without a V3--V8 helper."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins else None
    candidates: list[CandidateSupport] = []
    ground_truth: list[GroundTruthSupport] = []
    association_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        scene = scenes[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(load_ply_xyz(_gaussian_ply(scene)), _transform(scene))
        nearest, valid = _nearest_gaussians(gt_xyz, gaussian_xyz, radius_m)
        metadata, bank = load_v9_candidate_bank(
            bank_root / association_mode / scene_id, classifier
        )
        gt_rows = _gt_supports(
            scene_id,
            gt_xyz,
            gt.semantic,
            gt.instance,
            taxonomy.canonical_classes,
            size_spec,
            min_region_size,
        )
        ground_truth.extend(gt_rows)
        for candidate_id, (row, full_ids) in enumerate(zip(bank.candidates, bank.full_ids)):
            candidates.append(
                CandidateSupport(
                    scene_id=scene_id,
                    candidate_id=candidate_id,
                    support_ids=_mapped_support(
                        full_ids,
                        nearest,
                        valid,
                        sentinel=len(gt_xyz) + 1_000_000 + candidate_id,
                    ),
                    class_name=str(row["branch_class"]),
                    score=float(row["base_score"]),
                )
            )
        association_rows.append(
            {
                "scene_id": scene_id,
                **_association_diagnostics(metadata, gt_rows, nearest, valid),
            }
        )
    result = evaluate_object_candidates(candidates, ground_truth)
    predicted = sum(int(row["predicted_pair_count"]) for row in association_rows)
    positives = sum(int(row["oracle_positive_pair_count"]) for row in association_rows)
    true_positive = sum(int(row["true_positive_pair_count"]) for row in association_rows)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / positives if positives else 0.0
    result.update(
        {
            "association_mode": association_mode,
            "classifier": classifier,
            "association_pair_precision": precision,
            "association_pair_recall": recall,
            "association_pair_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "association_truth_thresholds": {
                "min_intersection": ASSOCIATION_TRUTH_MIN_INTERSECTION,
                "min_iou": ASSOCIATION_TRUTH_MIN_IOU,
                "min_purity": ASSOCIATION_TRUTH_MIN_PURITY,
            },
            "association_per_scene": association_rows,
        }
    )
    write_rows(rows_output, result["per_candidate"])
    write_json(analysis_output, result)
    return result


def metrics_by_condition(analysis: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(condition): dict(payload["metrics"])
        for condition, payload in analysis["conditions"].items()
    }


def scene_metrics(
    analysis: Mapping[str, Any], condition: str
) -> list[dict[str, Any]]:
    return [dict(row) for row in analysis["conditions"][condition]["per_scene"]]


def physical_scene_macro_delta(
    analysis: Mapping[str, Any], *, reference: str, treatment: str
) -> dict[str, Any]:
    ref = {str(row["scene_id"]): row for row in scene_metrics(analysis, reference)}
    trt = {str(row["scene_id"]): row for row in scene_metrics(analysis, treatment)}
    if not ref or set(ref) != set(trt):
        raise ValueError("paired conditions must contain the same non-empty scene set")
    grouped: dict[str, list[float]] = {}
    for scene_id in sorted(ref):
        physical = scene_id.rsplit("_", 1)[0]
        grouped.setdefault(physical, []).append(
            float(trt[scene_id]["map_50_95"]) - float(ref[scene_id]["map_50_95"])
        )
    means = {key: float(np.mean(values)) for key, values in sorted(grouped.items())}
    return {
        "physical_scene_count": len(means),
        "physical_scene_deltas": means,
        "macro_delta_map_50_95": float(np.mean(list(means.values()))),
    }


def paired_physical_scene_bootstrap(
    analysis: Mapping[str, Any],
    *,
    reference: str,
    treatment: str,
    samples: int = 10_000,
    seed: int = 20_260_804,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    macro = physical_scene_macro_delta(
        analysis, reference=reference, treatment=treatment
    )
    deltas = np.asarray(
        list(macro["physical_scene_deltas"].values()), dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    draws = deltas[indices].mean(axis=1)
    return {
        "schema": "saga-v9-paired-physical-scene-bootstrap-v1",
        "samples": samples,
        "seed": seed,
        "physical_scene_count": len(deltas),
        "delta_map_50_95": float(np.mean(deltas)),
        "paired_bootstrap_ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def paired_scannet_bootstrap_from_predictions(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prediction_root: Path,
    scene_ids: Sequence[str],
    reference_condition: str,
    treatment_condition: str,
    taxonomy: Taxonomy,
    samples: int = 10_000,
    seed: int = 20_260_804,
    radius_m: float = 0.05,
    overlaps: Sequence[float] = OVERLAPS,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Load frozen strict replays and bootstrap pooled official ScanNet AP."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    normalized_scenes = tuple(map(str, scene_ids))
    if len(set(normalized_scenes)) != len(normalized_scenes):
        raise ValueError("paired bootstrap scene_ids must be unique scans")
    missing = sorted(set(normalized_scenes).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    class_names = tuple(map(str, taxonomy.canonical_classes))
    class_to_id = {name: index for index, name in enumerate(class_names)}
    conditions = (str(reference_condition), str(treatment_condition))
    ground_truth: list[GroundTruthScene] = []
    predictions: dict[str, list[PredictedInstance]] = {
        condition: [] for condition in conditions
    }
    for scene_id in normalized_scenes:
        scene = scenes[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        ground_truth.append(gt)
        gaussian_xyz = apply_transform(
            load_ply_xyz(_gaussian_ply(scene)), _transform(scene)
        )
        for condition in conditions:
            output = load_json(
                _output_dir(prediction_root, condition, scene_id) / "output.json"
            )
            labels = np.asarray(output.get("point_labels"), dtype=np.int64)
            instances = output.get("instances")
            if labels.ndim != 1 or not isinstance(instances, Mapping):
                raise ValueError(f"{condition}/{scene_id}: invalid prediction payload")
            validate_prediction_contract(labels, instances)
            if len(labels) != len(gaussian_xyz):
                raise ValueError(f"{condition}/{scene_id}: point count mismatch")
            mapped_labels, _ = map_gaussians_to_gt(
                gt_xyz, gaussian_xyz, labels, radius_m
            )
            for raw_id, metadata in instances.items():
                instance_id = int(raw_id)
                class_name = str(metadata.get("class", ""))
                if instance_id < 0 or class_name not in class_to_id:
                    continue
                score = float(metadata.get("score", np.nan))
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
        class_names,
        overlaps=overlaps,
        min_region_size=min_region_size,
    )
    treatment_events = precompute_scannet_scene_ap_events(
        ground_truth,
        predictions[conditions[1]],
        class_names,
        overlaps=overlaps,
        min_region_size=min_region_size,
    )
    result = paired_scannet_scene_bootstrap(
        reference_events,
        treatment_events,
        physical_scene_ids=tuple(
            scene_id.rsplit("_", 1)[0] for scene_id in normalized_scenes
        ),
        samples=samples,
        seed=seed,
    )
    return {
        **result,
        "reference_condition": conditions[0],
        "treatment_condition": conditions[1],
    }
