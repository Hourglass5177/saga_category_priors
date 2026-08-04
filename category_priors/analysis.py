from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .io import hash_json, load_json, sha256_file, write_json
from .taxonomy import Taxonomy


def _metric(
    ground_truth: Sequence[GroundTruthScene],
    predictions: Sequence[PredictedInstance],
    class_names: Sequence[str],
    min_region_size: int = 100,
) -> float:
    result = evaluate_instances(
        ground_truth,
        predictions,
        class_names,
        min_region_size=min_region_size,
    )
    value = result["aggregate"]["map_50_95"]
    if value is None:
        raise ValueError("mAP is undefined for this bootstrap sample")
    return float(value)


def _resample_groups(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: Mapping[str, Sequence[PredictedInstance]],
    physical_group_by_scene: Mapping[str, str],
    sampled_groups: Sequence[str],
) -> tuple[list[GroundTruthScene], dict[str, list[PredictedInstance]]]:
    gt_by_scene = {scene.scene_id: scene for scene in ground_truth}
    scenes_by_group: dict[str, list[str]] = {}
    for scene_id in gt_by_scene:
        scenes_by_group.setdefault(physical_group_by_scene[scene_id], []).append(
            scene_id
        )
    pred_lookup: dict[str, dict[str, list[PredictedInstance]]] = {}
    for condition, predictions in predictions_by_condition.items():
        by_scene: dict[str, list[PredictedInstance]] = {}
        for prediction in predictions:
            by_scene.setdefault(prediction.scene_id, []).append(prediction)
        pred_lookup[condition] = by_scene

    sampled_gt: list[GroundTruthScene] = []
    sampled_predictions = {condition: [] for condition in predictions_by_condition}
    for occurrence, group in enumerate(sampled_groups):
        for scene_id in sorted(scenes_by_group[group]):
            clone_id = f"{scene_id}#bootstrap-{occurrence}"
            sampled_gt.append(replace(gt_by_scene[scene_id], scene_id=clone_id))
            for condition in predictions_by_condition:
                sampled_predictions[condition].extend(
                    replace(prediction, scene_id=clone_id)
                    for prediction in pred_lookup[condition].get(scene_id, [])
                )
    return sampled_gt, sampled_predictions


def paired_scene_bootstrap(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: Mapping[str, Sequence[PredictedInstance]],
    physical_group_by_scene: Mapping[str, str],
    class_names: Sequence[str],
    reference: str,
    treatment: str,
    samples: int = 10_000,
    seed: int = 20260804,
    min_region_size: int = 100,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if (
        reference not in predictions_by_condition
        or treatment not in predictions_by_condition
    ):
        raise KeyError("Both reference and treatment predictions are required")
    groups = sorted(set(physical_group_by_scene.values()))
    if not groups:
        raise ValueError("No physical scene groups were supplied")
    point_reference = _metric(
        ground_truth, predictions_by_condition[reference], class_names, min_region_size
    )
    point_treatment = _metric(
        ground_truth, predictions_by_condition[treatment], class_names, min_region_size
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True).tolist()
        sampled_gt, sampled_pred = _resample_groups(
            ground_truth,
            predictions_by_condition,
            physical_group_by_scene,
            sampled_groups,
        )
        differences[index] = _metric(
            sampled_gt, sampled_pred[treatment], class_names, min_region_size
        ) - _metric(sampled_gt, sampled_pred[reference], class_names, min_region_size)
    low, high = np.quantile(differences, (0.025, 0.975))
    return {
        "reference": reference,
        "treatment": treatment,
        "reference_map_50_95": point_reference,
        "treatment_map_50_95": point_treatment,
        "difference": point_treatment - point_reference,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": samples,
        "seed": seed,
    }


def factorial_bootstrap(
    ground_truth: Sequence[GroundTruthScene],
    predictions_by_condition: Mapping[str, Sequence[PredictedInstance]],
    condition_bits: Mapping[str, Sequence[int]],
    physical_group_by_scene: Mapping[str, str],
    class_names: Sequence[str],
    samples: int = 10_000,
    seed: int = 20260804,
    min_region_size: int = 100,
) -> dict[str, Any]:
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
            "The factorial analysis requires each of the eight 2^3 combinations exactly once"
        )
    groups = sorted(set(physical_group_by_scene.values()))
    if not groups:
        raise ValueError("No physical scene groups were supplied")
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
    effects = np.empty((samples, len(term_masks)), dtype=np.float64)
    for sample_index in range(samples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True).tolist()
        sampled_gt, sampled_pred = _resample_groups(
            ground_truth,
            predictions_by_condition,
            physical_group_by_scene,
            sampled_groups,
        )
        metrics = {
            condition: _metric(sampled_gt, predictions, class_names, min_region_size)
            for condition, predictions in sampled_pred.items()
        }
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
            effects[sample_index, term_index] = float(
                np.mean(positive) - np.mean(negative)
            )

    names = tuple(term_masks)
    raw_p = [
        min(
            1.0,
            2.0
            * min(
                float(np.mean(effects[:, index] <= 0)),
                float(np.mean(effects[:, index] >= 0)),
            ),
        )
        for index in range(len(names))
    ]
    adjusted = holm_adjust(raw_p)
    result: dict[str, Any] = {}
    for index, name in enumerate(names):
        low, high = np.quantile(effects[:, index], (0.025, 0.975))
        result[name] = {
            "bootstrap_mean_effect": float(np.mean(effects[:, index])),
            "ci95": [float(low), float(high)],
            "p_two_sided": raw_p[index],
            "p_holm": adjusted[index],
        }
    return {
        "effects": result,
        "holm_family": list(names),
        "bootstrap_samples": samples,
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
    predictions_by_condition: dict[str, list[PredictedInstance]] = {}
    alignment: dict[str, dict[str, Any]] = {}
    for condition, items in manifest["conditions"].items():
        items_by_scene = {str(item["scene_id"]): item for item in items}
        if set(items_by_scene) != required_scenes:
            missing = sorted(required_scenes - set(items_by_scene))
            extra = sorted(set(items_by_scene) - required_scenes)
            raise ValueError(
                f"{condition}: scene mismatch; missing={missing}, extra={extra}"
            )
        condition_predictions: list[PredictedInstance] = []
        alignment[condition] = {}
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
                    f"{condition}/{scene_id}: coordinate alignment gate failed"
                )
            condition_predictions.extend(predictions)
            alignment[condition][scene_id] = diagnostics
        predictions_by_condition[str(condition)] = condition_predictions

    paired_results = []
    for comparison in manifest.get("paired_comparisons", []):
        paired_results.append(
            paired_scene_bootstrap(
                ground_truth,
                predictions_by_condition,
                physical_groups,
                taxonomy.canonical_classes,
                str(comparison["reference"]),
                str(comparison["treatment"]),
                samples,
                seed,
                min_region_size,
            )
        )

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
