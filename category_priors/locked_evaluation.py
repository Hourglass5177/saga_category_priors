from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    CompiledEvaluation,
    compile_predictions,
    evaluate_compiled,
    factorial_bootstrap,
    paired_scene_bootstrap,
    paired_scene_permutation_test,
    merge_compiled_evaluations,
)
from .evaluator import (
    PROTOCOL_VERSION,
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .io import load_json, write_json, write_rows
from .locked import SMALL_CATEGORIES, expand_locked_runs, validate_locked_plan
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


def _scene_gaussian_path(scene: Mapping[str, Any]) -> Path:
    base_path = Path(scene["base_path"])
    raw = scene.get("gaussian_ply") or scene.get("point_cloud_path")
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else (base_path / path).resolve()
    preferred = (
        base_path
        / "output_models"
        / "point_cloud"
        / "iteration_30000"
        / "point_cloud.ply"
    )
    fallback = preferred.with_name("scene_point_cloud.ply")
    return fallback if not preferred.is_file() and fallback.is_file() else preferred


def _locked_run_paths(
    output_root: Path, condition: str, scene_id: str, run_seed: int
) -> tuple[Path, Path, Path]:
    run_dir = output_root / condition / scene_id / f"seed-{run_seed}"
    return (
        run_dir / "output.json",
        run_dir / "output.json.metadata.json",
        run_dir / "run.json",
    )


def _load_ground_truth(
    scene_ids: list[str], gt_dir: str | Path
) -> tuple[list[GroundTruthScene], dict[str, np.ndarray]]:
    base = Path(gt_dir).resolve()
    ground_truth: list[GroundTruthScene] = []
    coordinates: dict[str, np.ndarray] = {}
    for scene_id in scene_ids:
        coords, scene = load_ground_truth_npz(base / f"{scene_id}.npz", scene_id)
        coordinates[scene_id] = coords
        ground_truth.append(scene)
    return ground_truth, coordinates


def _compile_locked_outputs(
    scene_ids: list[str],
    conditions: list[str],
    seeds: list[int],
    ground_truth: list[GroundTruthScene],
    coordinates: Mapping[str, np.ndarray],
    runtime_scenes: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    taxonomy: Taxonomy,
    radius_m: float,
    minimum_mapped_fraction: float,
    min_region_size: int,
) -> tuple[
    dict[str, dict[str, CompiledEvaluation]],
    dict[str, dict[str, dict[str, dict[str, float]]]],
    dict[tuple[str, str, int], float],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, Any]],
]:
    root = Path(output_root).resolve()
    compiled: dict[str, dict[str, CompiledEvaluation]] = {
        condition: {} for condition in conditions
    }
    diagnostics: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        condition: {} for condition in conditions
    }
    runtimes: dict[tuple[str, str, int], float] = {}
    attempts: dict[tuple[str, str, int], dict[str, Any]] = {}
    per_scene_metrics: dict[str, dict[str, dict[str, float]]] = {
        "P000-B2": {},
        "P111-combined": {},
    }
    evaluated: dict[str, dict[str, Any]] = {condition: {} for condition in conditions}
    ground_truth_by_scene = {scene.scene_id: scene for scene in ground_truth}
    for condition in conditions:
        for run_seed in seeds:
            seed_key = str(run_seed)
            compiled_scene_parts: list[CompiledEvaluation] = []
            diagnostics[condition][seed_key] = {}
            if condition in per_scene_metrics:
                per_scene_metrics[condition][seed_key] = {}
            for scene_id in scene_ids:
                if scene_id not in runtime_scenes:
                    raise ValueError(f"Missing locked runtime scene: {scene_id}")
                output_json, metadata_json, run_json = _locked_run_paths(
                    root, condition, scene_id, run_seed
                )
                record = load_json(run_json)
                expected = {
                    "scene_id": scene_id,
                    "condition": condition,
                    "run_seed": run_seed,
                }
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Locked run identity mismatch: {run_json}")
                if record.get("status") != "complete":
                    raise ValueError(f"Locked run is incomplete: {run_json}")
                metadata = load_json(metadata_json)
                metadata_run = metadata.get("run", {})
                if (
                    metadata.get("kind") != "saga_instance_metadata"
                    or metadata_run.get("scene_id") != scene_id
                    or metadata_run.get("condition") != condition
                    or int(metadata_run.get("seed", -1)) != run_seed
                ):
                    raise ValueError(f"Locked metadata identity mismatch: {metadata_json}")
                scene_runtime = runtime_scenes[scene_id]
                transform = scene_runtime.get(
                    "gaussian_to_gt_transform", np.eye(4).tolist()
                )
                scene_predictions, scene_diagnostics = saga_scene_predictions(
                    scene_id,
                    coordinates[scene_id],
                    output_json,
                    _scene_gaussian_path(scene_runtime),
                    taxonomy,
                    metadata_json,
                    transform,
                    radius_m,
                    require_scores=True,
                )
                if (
                    scene_diagnostics["median_nn_distance_m"] > radius_m
                    or scene_diagnostics["mapped_fraction"] < minimum_mapped_fraction
                ):
                    raise ValueError(f"{condition}/{scene_id}: alignment failed")
                diagnostics[condition][seed_key][scene_id] = scene_diagnostics
                runtimes[(condition, scene_id, run_seed)] = float(
                    record["runtime_seconds"]
                )
                attempts[(condition, scene_id, run_seed)] = {
                    "attempts": int(record.get("attempts", 1)),
                    "first_attempt_failed": bool(
                        record.get("first_attempt_failed", False)
                    ),
                    "recovered": bool(record.get("recovered", False)),
                }
                scene_compiled = compile_predictions(
                    [ground_truth_by_scene[scene_id]],
                    scene_predictions,
                    taxonomy.canonical_classes,
                    min_region_size,
                )
                if condition in per_scene_metrics:
                    scene_result = evaluate_compiled(scene_compiled)
                    per_scene_metrics[condition][seed_key][scene_id] = float(
                        scene_result["aggregate"]["map_50_95"]
                    )
                compiled_scene_parts.append(scene_compiled)
                del scene_predictions
            compiled_seed = merge_compiled_evaluations(compiled_scene_parts)
            compiled[condition][seed_key] = compiled_seed
            evaluated[condition][seed_key] = evaluate_compiled(compiled_seed)
            # This is the critical 90GiB boundary: only one scene's dense masks
            # exist at once; compact curve events survive across scenes.
            del compiled_scene_parts
    return compiled, diagnostics, runtimes, attempts, per_scene_metrics, evaluated


def _condition_metric_rows(
    scene_count: int,
    evaluated: Mapping[str, Mapping[str, Mapping[str, Any]]],
    runtimes: Mapping[tuple[str, str, int], float],
    attempts: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    small_set = set(SMALL_CATEGORIES)
    for condition, replicates in evaluated.items():
        for raw_seed, result in replicates.items():
            run_seed = int(raw_seed)
            small_values = [
                float(item["ap_50_95"])
                for name, item in result["per_class"].items()
                if name in small_set and item["ap_50_95"] is not None
            ]
            runtime_values = [
                value
                for (item_condition, _, item_seed), value in runtimes.items()
                if item_condition == condition and item_seed == run_seed
            ]
            attempt_values = [
                value
                for (item_condition, _, item_seed), value in attempts.items()
                if item_condition == condition and item_seed == run_seed
            ]
            aggregate = result["aggregate"]
            rows.append(
                {
                    "split": "val-locked",
                    "protocol_version": PROTOCOL_VERSION,
                    "condition": condition,
                    "run_seed": run_seed,
                    "scene_count": scene_count,
                    "map_50_95": aggregate["map_50_95"],
                    "map_0.50": aggregate["map_0.50"],
                    "map_0.25": aggregate["map_0.25"],
                    "small_category_map_50_95": float(np.mean(small_values))
                    if small_values
                    else None,
                    "runtime_seconds": float(np.mean(runtime_values)),
                    "first_attempt_failure_rate": float(
                        np.mean(
                            [item["first_attempt_failed"] for item in attempt_values]
                        )
                    ),
                    "recovered_run_count": int(
                        sum(item["recovered"] for item in attempt_values)
                    ),
                    "final_failure_rate": 0.0,
                }
            )
    return rows


def _class_support(
    ground_truth: list[GroundTruthScene], taxonomy: Taxonomy, min_region_size: int
) -> dict[str, dict[str, int]]:
    support: dict[str, dict[str, int]] = {}
    for class_id, name in enumerate(taxonomy.canonical_classes):
        instances = 0
        scenes = 0
        for scene in ground_truth:
            scene_instances = 0
            class_mask = scene.semantic == class_id
            for instance_id in np.unique(scene.instance[class_mask]):
                if instance_id < 0:
                    continue
                if int(np.count_nonzero(class_mask & (scene.instance == instance_id))) >= min_region_size:
                    scene_instances += 1
            instances += scene_instances
            scenes += scene_instances > 0
        support[name] = {
            "gt_instances": instances,
            "physical_scene_support": scenes,
        }
    return support


def _qualitative_ranking(
    scene_ids: list[str],
    per_scene_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[dict[str, Any]]:
    reference = per_scene_metrics["P000-B2"]
    treatment = per_scene_metrics["P111-combined"]
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        seed_deltas = []
        for seed in sorted(reference):
            seed_deltas.append(
                float(treatment[seed][scene_id]) - float(reference[seed][scene_id])
            )
        rows.append({"scene_id": scene_id, "delta_map_50_95": float(np.mean(seed_deltas))})
    rows.sort(key=lambda item: (item["delta_map_50_95"], item["scene_id"]))
    return rows


def evaluate_locked_plan(
    plan_path: str | Path,
    scene_manifest_path: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    taxonomy: Taxonomy,
    metrics_path: str | Path,
    analysis_path: str | Path,
) -> dict[str, Any]:
    plan = load_json(plan_path)
    validate_locked_plan(plan)
    expected_taxonomy = plan["taxonomy"]
    if (
        taxonomy.benchmark_name != expected_taxonomy["benchmark_name"]
        or list(taxonomy.canonical_classes)
        != list(expected_taxonomy["canonical_classes"])
    ):
        raise ValueError("Evaluation taxonomy differs from locked_plan.json")
    scene_ids = [str(item["scene_id"]) for item in plan["scenes"]]
    physical_groups = {
        str(item["scene_id"]): str(item["physical_scene_id"])
        for item in plan["scenes"]
    }
    ground_truth, coordinates = _load_ground_truth(scene_ids, gt_dir)
    runtime_scenes = load_scene_runtime_manifest(scene_manifest_path)
    config = plan["analysis"]
    (
        compiled,
        diagnostics,
        runtimes,
        attempts,
        per_scene_metrics,
        condition_results,
    ) = _compile_locked_outputs(
        scene_ids,
        [str(value) for value in plan["conditions"]],
        [int(value) for value in plan["seeds"]],
        ground_truth,
        coordinates,
        runtime_scenes,
        output_root,
        taxonomy,
        float(config["radius_m"]),
        float(config["minimum_mapped_fraction"]),
        int(config["min_region_size"]),
    )
    metric_rows = _condition_metric_rows(
        len(ground_truth),
        condition_results,
        runtimes,
        attempts,
    )
    write_rows(metrics_path, metric_rows)

    comparison = config["primary_comparison"]
    paired = paired_scene_bootstrap(
        ground_truth,
        compiled,
        physical_groups,
        taxonomy.canonical_classes,
        str(comparison["reference"]),
        str(comparison["treatment"]),
        samples=int(config["bootstrap"]["samples"]),
        seed=int(plan["randomization_seed"]),
        min_region_size=int(config["min_region_size"]),
    )
    permutation = paired_scene_permutation_test(
        ground_truth,
        compiled,
        physical_groups,
        taxonomy.canonical_classes,
        str(comparison["reference"]),
        str(comparison["treatment"]),
        samples=int(config["permutation"]["samples"]),
        seed=int(plan["randomization_seed"]),
        min_region_size=int(config["min_region_size"]),
    )
    factorial_predictions = {
        name: compiled[name] for name in config["factorial_bits"]
    }
    factorial = factorial_bootstrap(
        ground_truth,
        factorial_predictions,
        config["factorial_bits"],
        physical_groups,
        taxonomy.canonical_classes,
        samples=int(config["bootstrap"]["samples"]),
        seed=int(plan["randomization_seed"]),
        min_region_size=int(config["min_region_size"]),
    )
    secondary = []
    for reference in ("B0-legacy", "B1-other-classes"):
        secondary.append(
            paired_scene_bootstrap(
                ground_truth,
                compiled,
                physical_groups,
                taxonomy.canonical_classes,
                reference,
                "P111-combined",
                samples=int(config["bootstrap"]["samples"]),
                seed=int(plan["randomization_seed"]),
                min_region_size=int(config["min_region_size"]),
            )
        )
    for treatment in ("P111-no-gate", "P111-no-shrink"):
        secondary.append(
            paired_scene_bootstrap(
                ground_truth,
                compiled,
                physical_groups,
                taxonomy.canonical_classes,
                "P111-combined",
                treatment,
                samples=int(config["bootstrap"]["samples"]),
                seed=int(plan["randomization_seed"]),
                min_region_size=int(config["min_region_size"]),
            )
        )

    alpha = float(config["alpha"])
    ci = paired["ci95"]
    success = (
        float(paired["difference"]) > 0
        and float(permutation["p_two_sided"]) < alpha
        and float(ci[0]) > 0
    )
    ranking = _qualitative_ranking(
        scene_ids,
        per_scene_metrics,
    )
    middle = len(ranking) // 2
    payload = {
        "schema_version": "1.0",
        "kind": "confirmatory_analysis",
        "split": "val-locked",
        "experimental_unit": "physical_scene",
        "scene_count": len(scene_ids),
        "physical_scene_count": len(set(physical_groups.values())),
        "technical_replicates": [int(value) for value in plan["seeds"]],
        "primary": {
            "reference": comparison["reference"],
            "treatment": comparison["treatment"],
            "metric": config["primary_metric"],
            "bootstrap": paired,
            "permutation": permutation,
            "success": success,
            "decision": "stable improvement supported"
            if success
            else "stable improvement not supported",
        },
        "secondary": secondary,
        "factorial": factorial,
        "small_categories": list(SMALL_CATEGORIES),
        "class_support": _class_support(
            ground_truth, taxonomy, int(config["min_region_size"])
        ),
        "per_condition_seed_metrics": condition_results,
        "execution": {
            "run_count": len(attempts),
            "first_attempt_failure_count": int(
                sum(item["first_attempt_failed"] for item in attempts.values())
            ),
            "recovered_run_count": int(
                sum(item["recovered"] for item in attempts.values())
            ),
            "final_failure_count": 0,
        },
        "alignment": diagnostics,
        "qualitative_cases": {
            "worst": ranking[0],
            "median": ranking[middle],
            "best": ranking[-1],
            "ranking": ranking,
        },
    }
    write_json(analysis_path, payload)
    return payload


def evaluate_tune_seed_execution(
    schedule_path: str | Path,
    execution_path: str | Path,
    scene_manifest_path: str | Path,
    gt_dir: str | Path,
    taxonomy: Taxonomy,
    metrics_path: str | Path,
    *,
    config_id: str | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> list[dict[str, Any]]:
    """Evaluate the two-condition tune seed audit without extra manifests/hashes."""
    schedule = load_json(schedule_path)
    if schedule.get("kind") != "run_schedule" or schedule.get("split") != "val-tune":
        raise ValueError("Expected a val-tune run_schedule")
    execution = load_json(execution_path)
    if execution.get("kind") != "run_execution":
        raise ValueError("Expected a run_execution")
    execution_by_sequence = {
        int(item["sequence"]): item for item in execution.get("runs", [])
    }
    runtime_scenes = load_scene_runtime_manifest(scene_manifest_path)
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for run in schedule["runs"]:
        if config_id is not None and str(run.get("config_id")) != config_id:
            continue
        grouped[(str(run["condition"]), int(run["run_seed"]))].append(run)
    if not grouped:
        raise ValueError(f"No tune runs matched config_id={config_id!r}")
    rows: list[dict[str, Any]] = []
    for (condition, run_seed), runs in sorted(grouped.items()):
        scene_ids = sorted(str(run["scene_id"]) for run in runs)
        ground_truth, coordinates = _load_ground_truth(scene_ids, gt_dir)
        predictions: list[PredictedInstance] = []
        runtimes = []
        for run in runs:
            record = execution_by_sequence.get(int(run["sequence"]))
            if not record or record.get("status") not in {"complete", "skipped_complete"}:
                raise ValueError(f"Incomplete tune seed run: {run['sequence']}")
            scene_id = str(run["scene_id"])
            scene = runtime_scenes[scene_id]
            scene_predictions, diagnostics = saga_scene_predictions(
                scene_id,
                coordinates[scene_id],
                record["output_json"],
                _scene_gaussian_path(scene),
                taxonomy,
                record["metadata_json"],
                scene.get("gaussian_to_gt_transform", np.eye(4).tolist()),
                radius_m,
                require_scores=True,
            )
            if diagnostics["mapped_fraction"] < float(
                scene.get("minimum_mapped_fraction", 0.90)
            ):
                raise ValueError(f"{condition}/{scene_id}: alignment failed")
            predictions.extend(scene_predictions)
            # A resumed run may be recorded as ``skipped_complete`` without a
            # wall-clock duration in the new execution file.  Its predictions
            # are still valid; runtime is a secondary diagnostic, so summarize
            # the durations that were actually observed instead of rejecting
            # the entire seed audit.
            if record.get("runtime_seconds") is not None:
                runtimes.append(float(record["runtime_seconds"]))
        evaluated = evaluate_instances(
            ground_truth,
            predictions,
            taxonomy.canonical_classes,
            min_region_size=min_region_size,
        )
        rows.append(
            {
                "split": "val-tune",
                "protocol_version": PROTOCOL_VERSION,
                "condition": condition,
                "run_seed": run_seed,
                "scene_count": len(scene_ids),
                "map_50_95": evaluated["aggregate"]["map_50_95"],
                "map_0.50": evaluated["aggregate"]["map_0.50"],
                "map_0.25": evaluated["aggregate"]["map_0.25"],
                "runtime_seconds": float(np.mean(runtimes)) if runtimes else None,
                "runtime_observed_runs": len(runtimes),
                "failure_rate": 0.0,
            }
        )
    write_rows(metrics_path, rows)
    return rows
