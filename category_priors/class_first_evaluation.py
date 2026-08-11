from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    CompiledEvaluation,
    compile_predictions,
    evaluate_compiled,
    merge_compiled_evaluations,
    paired_scene_bootstrap,
)
from .class_first_runner import CLASS_FIRST_CONDITIONS
from .evaluator import (
    PROTOCOL_VERSION,
    GroundTruthScene,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .io import load_json, write_json, write_rows
from .locked import SMALL_CATEGORIES
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import Taxonomy


CLASS_FIRST_DIAGNOSTIC_FIELDS = (
    "candidate_points",
    "sampled_points",
    "hdbscan_noise_points",
    "sor_removed_points",
    "rescued_points",
    "final_instances",
    "assigned_points",
    "coverage",
)


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


def _run_paths(
    output_root: str | Path, condition: str, scene_id: str, seed: int
) -> tuple[Path, Path]:
    run_dir = Path(output_root).resolve() / condition / scene_id / f"seed-{seed}"
    return run_dir / "output.json", run_dir / "diagnostics.json"


def _selection_items(
    payload: Mapping[str, Any], selection_split: str
) -> list[Mapping[str, Any] | str]:
    if isinstance(payload.get("scenes"), list):
        return list(payload["scenes"])
    selection = payload.get("selection")
    if isinstance(selection, Mapping) and isinstance(selection.get(selection_split), list):
        return list(selection[selection_split])
    raise ValueError("Selection JSON does not contain a usable scene list")


def resolve_evaluation_scenes(
    runtime_scenes: Mapping[str, Mapping[str, Any]],
    *,
    scene_ids: Sequence[str] | None = None,
    scene_list_path: str | Path | None = None,
    selection_path: str | Path | None = None,
    selection_split: str = "locked",
) -> tuple[list[str], dict[str, str]]:
    supplied = sum(
        value is not None for value in (scene_ids, scene_list_path, selection_path)
    )
    if supplied > 1:
        raise ValueError("Use only one of scene_ids, scene_list_path, or selection_path")

    selected: list[str] = []
    physical: dict[str, str] = {}
    if selection_path is not None:
        payload = load_json(selection_path)
        if not isinstance(payload, Mapping):
            raise ValueError("Selection JSON must be an object")
        for item in _selection_items(payload, selection_split):
            if isinstance(item, Mapping):
                scene_id = str(item["scene_id"])
                if item.get("physical_scene_id") is not None:
                    physical[scene_id] = str(item["physical_scene_id"])
            else:
                scene_id = str(item)
            selected.append(scene_id)
    elif scene_list_path is not None:
        selected = [
            line.strip()
            for line in Path(scene_list_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif scene_ids is not None:
        selected = [str(value) for value in scene_ids]
    else:
        selected = list(runtime_scenes)

    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Evaluation scenes must be nonempty and unique")
    missing = sorted(set(selected) - set(runtime_scenes))
    if missing:
        raise ValueError(f"Scene runtime manifest is missing scenes: {missing}")
    for scene_id in selected:
        runtime = runtime_scenes[scene_id]
        physical.setdefault(
            scene_id,
            str(runtime.get("physical_scene_id", physical_scene_id(scene_id))),
        )
    return selected, physical


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _add_numeric(target: dict[str, list[float]], name: str, value: Any) -> None:
    number = _numeric(value)
    if number is not None:
        target.setdefault(name, []).append(number)


def _flatten_numeric(
    value: Any,
    target: dict[str, list[float]],
    prefix: str = "",
    depth: int = 0,
) -> None:
    if depth > 4:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in {"command", "instances", "point_labels"}:
                continue
            name = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(item, target, name, depth + 1)
    elif prefix:
        _add_numeric(target, prefix, value)


def _new_diagnostic_accumulator() -> dict[str, Any]:
    return {
        "alignment": {},
        "runner": {},
        "class_first_totals": {},
        "class_first_classes": {},
        "numeric_fallback": {},
    }


def _accumulate_diagnostics(
    accumulator: dict[str, Any],
    diagnostics: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> None:
    for name, value in alignment.items():
        _add_numeric(accumulator["alignment"], str(name), value)
    runner = diagnostics.get("runner")
    if isinstance(runner, Mapping):
        for name in ("runtime_seconds", "point_count", "instance_count"):
            _add_numeric(accumulator["runner"], name, runner.get(name))

    class_first = diagnostics.get("class_first")
    if not isinstance(class_first, Mapping):
        _flatten_numeric(diagnostics, accumulator["numeric_fallback"])
        return
    totals = class_first.get("totals")
    total_values = totals if isinstance(totals, Mapping) else class_first
    for name in CLASS_FIRST_DIAGNOSTIC_FIELDS:
        _add_numeric(
            accumulator["class_first_totals"], name, total_values.get(name)
        )
    classes = class_first.get("classes")
    if isinstance(classes, Mapping):
        for class_name, values in classes.items():
            if not isinstance(values, Mapping):
                continue
            class_accumulator = accumulator["class_first_classes"].setdefault(
                str(class_name), {}
            )
            for name in CLASS_FIRST_DIAGNOSTIC_FIELDS:
                _add_numeric(class_accumulator, name, values.get(name))


def _summarize_values(values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw in sorted(values.items()):
        array = np.asarray(raw, dtype=np.float64)
        if not len(array):
            continue
        result[name] = {
            "count": int(len(array)),
            "sum": float(array.sum()),
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }
    return result


def _summarize_diagnostics(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "alignment": _summarize_values(accumulator["alignment"]),
        "runner": _summarize_values(accumulator["runner"]),
        "class_first_totals": _summarize_values(
            accumulator["class_first_totals"]
        ),
        "class_first_classes": {
            class_name: _summarize_values(values)
            for class_name, values in sorted(
                accumulator["class_first_classes"].items()
            )
        },
        "numeric_fallback": _summarize_values(accumulator["numeric_fallback"]),
    }


def _validate_run_diagnostics(
    diagnostics: Mapping[str, Any], condition: str, scene_id: str, seed: int
) -> None:
    run = diagnostics.get("run")
    identity = run if isinstance(run, Mapping) else diagnostics
    if (
        diagnostics.get("status") != "complete"
        or identity.get("condition") != condition
        or identity.get("scene_id") != scene_id
        or int(identity.get("seed", -1)) != seed
    ):
        raise ValueError(
            f"Class-first run identity/status mismatch: {condition}/{scene_id}/seed-{seed}"
        )


def _class_support_for_scene(
    scene: GroundTruthScene, class_count: int, min_region_size: int
) -> tuple[np.ndarray, np.ndarray]:
    instances = np.zeros(class_count, dtype=np.int64)
    scenes = np.zeros(class_count, dtype=np.int64)
    for class_id in range(class_count):
        class_mask = scene.semantic == class_id
        count = sum(
            int(np.count_nonzero(class_mask & (scene.instance == instance_id)))
            >= min_region_size
            for instance_id in np.unique(scene.instance[class_mask])
            if instance_id >= 0
        )
        instances[class_id] = count
        scenes[class_id] = count > 0
    return instances, scenes


def _small_category_mean(result: Mapping[str, Any]) -> float | None:
    values = [
        float(result["per_class"][name]["ap_50_95"])
        for name in SMALL_CATEGORIES
        if name in result["per_class"]
        and result["per_class"][name]["ap_50_95"] is not None
    ]
    return float(np.mean(values)) if values else None


def _qualitative_ranking(
    scene_ids: Sequence[str],
    per_scene_metrics: Mapping[str, Mapping[str, Mapping[str, float | None]]],
    reference: str,
    treatment: str,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        reference_values = [
            per_scene_metrics[reference][str(seed)][scene_id] for seed in seeds
        ]
        treatment_values = [
            per_scene_metrics[treatment][str(seed)][scene_id] for seed in seeds
        ]
        paired = [
            (float(left), float(right))
            for left, right in zip(reference_values, treatment_values, strict=True)
            if left is not None and right is not None
        ]
        if not paired:
            continue
        reference_mean = float(np.mean([item[0] for item in paired]))
        treatment_mean = float(np.mean([item[1] for item in paired]))
        rows.append(
            {
                "scene_id": scene_id,
                "reference_map_50_95": reference_mean,
                "treatment_map_50_95": treatment_mean,
                "delta_map_50_95": treatment_mean - reference_mean,
            }
        )
    rows.sort(key=lambda item: (item["delta_map_50_95"], item["scene_id"]))
    return rows


def evaluate_class_first_runs(
    scene_manifest_path: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    taxonomy: Taxonomy,
    *,
    metrics_path: str | Path,
    analysis_path: str | Path,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] = (42, 3407, 20260804),
    scene_ids: Sequence[str] | None = None,
    scene_list_path: str | Path | None = None,
    selection_path: str | Path | None = None,
    selection_split: str = "locked",
    reference: str | None = None,
    treatment: str | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260804,
    radius_m: float = 0.05,
    minimum_mapped_fraction: float = 0.90,
    min_region_size: int = 100,
    split: str = "class-first",
) -> dict[str, Any]:
    selected_conditions = list(conditions or CLASS_FIRST_CONDITIONS)
    unknown = sorted(set(selected_conditions) - set(CLASS_FIRST_CONDITIONS))
    if not selected_conditions or unknown:
        raise ValueError(f"Invalid class-first conditions: {unknown}")
    selected_seeds = [int(value) for value in seeds]
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    if (reference is None) != (treatment is None):
        raise ValueError("reference and treatment must be supplied together")
    if reference is not None and (
        reference not in selected_conditions or treatment not in selected_conditions
    ):
        raise ValueError("reference and treatment must be selected conditions")

    runtime_scenes = load_scene_runtime_manifest(scene_manifest_path)
    selected_scene_ids, physical_groups = resolve_evaluation_scenes(
        runtime_scenes,
        scene_ids=scene_ids,
        scene_list_path=scene_list_path,
        selection_path=selection_path,
        selection_split=selection_split,
    )
    class_names = tuple(taxonomy.canonical_classes)
    compiled_parts: dict[str, dict[str, list[CompiledEvaluation]]] = {
        condition: {str(seed): [] for seed in selected_seeds}
        for condition in selected_conditions
    }
    per_scene_metrics: dict[str, dict[str, dict[str, float | None]]] = {
        condition: {str(seed): {} for seed in selected_seeds}
        for condition in selected_conditions
    }
    diagnostic_accumulators = {
        condition: {
            str(seed): _new_diagnostic_accumulator() for seed in selected_seeds
        }
        for condition in selected_conditions
    }
    support_instances = np.zeros(len(class_names), dtype=np.int64)
    support_physical_groups = [set() for _ in class_names]
    ground_truth_stubs: list[GroundTruthScene] = []
    gt_root = Path(gt_dir).resolve()

    for scene_id in selected_scene_ids:
        coordinates, ground_truth = load_ground_truth_npz(
            gt_root / f"{scene_id}.npz", scene_id
        )
        instance_counts, scene_counts = _class_support_for_scene(
            ground_truth, len(class_names), min_region_size
        )
        support_instances += instance_counts
        for class_index in np.flatnonzero(scene_counts):
            support_physical_groups[int(class_index)].add(physical_groups[scene_id])
        scene_runtime = runtime_scenes[scene_id]
        transform = scene_runtime.get("gaussian_to_gt_transform", np.eye(4).tolist())
        for condition in selected_conditions:
            for seed in selected_seeds:
                output_json, diagnostics_json = _run_paths(
                    output_root, condition, scene_id, seed
                )
                diagnostics = load_json(diagnostics_json)
                if not isinstance(diagnostics, Mapping):
                    raise ValueError(f"Expected diagnostics object: {diagnostics_json}")
                _validate_run_diagnostics(diagnostics, condition, scene_id, seed)
                predictions, alignment = saga_scene_predictions(
                    scene_id,
                    coordinates,
                    output_json,
                    _scene_gaussian_path(scene_runtime),
                    taxonomy,
                    diagnostics_json,
                    transform,
                    radius_m,
                    require_scores=True,
                )
                if (
                    alignment["median_nn_distance_m"] > radius_m
                    or alignment["mapped_fraction"] < minimum_mapped_fraction
                ):
                    raise ValueError(f"{condition}/{scene_id}: alignment failed")
                scene_compiled = compile_predictions(
                    [ground_truth], predictions, class_names, min_region_size
                )
                compiled_parts[condition][str(seed)].append(scene_compiled)
                scene_result = evaluate_compiled(scene_compiled)
                per_scene_metrics[condition][str(seed)][scene_id] = scene_result[
                    "aggregate"
                ]["map_50_95"]
                _accumulate_diagnostics(
                    diagnostic_accumulators[condition][str(seed)],
                    diagnostics,
                    alignment,
                )
                del predictions, scene_compiled, scene_result
        ground_truth_stubs.append(
            GroundTruthScene(
                scene_id,
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )
        )
        del coordinates, ground_truth

    compiled: dict[str, dict[str, CompiledEvaluation]] = {}
    evaluated: dict[str, dict[str, dict[str, Any]]] = {}
    diagnostic_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    metric_rows: list[dict[str, Any]] = []
    for condition in selected_conditions:
        compiled[condition] = {}
        evaluated[condition] = {}
        diagnostic_summaries[condition] = {}
        for seed in selected_seeds:
            seed_key = str(seed)
            merged = merge_compiled_evaluations(compiled_parts[condition][seed_key])
            compiled[condition][seed_key] = merged
            result = evaluate_compiled(merged)
            evaluated[condition][seed_key] = result
            diagnostic_summary = _summarize_diagnostics(
                diagnostic_accumulators[condition][seed_key]
            )
            diagnostic_summaries[condition][seed_key] = diagnostic_summary
            runner_summary = diagnostic_summary["runner"]
            class_first_summary = diagnostic_summary["class_first_totals"]
            metric_rows.append(
                {
                    "split": split,
                    "protocol_version": PROTOCOL_VERSION,
                    "condition": condition,
                    "run_seed": seed,
                    "scene_count": len(selected_scene_ids),
                    "map_50_95": result["aggregate"]["map_50_95"],
                    "map_0.50": result["aggregate"]["map_0.50"],
                    "map_0.25": result["aggregate"]["map_0.25"],
                    "small_category_map_50_95": _small_category_mean(result),
                    "runtime_seconds_mean": runner_summary.get(
                        "runtime_seconds", {}
                    ).get("mean"),
                    "point_count_mean": runner_summary.get("point_count", {}).get(
                        "mean"
                    ),
                    "instance_count_mean": runner_summary.get(
                        "instance_count", {}
                    ).get("mean"),
                    "coverage_mean": class_first_summary.get("coverage", {}).get(
                        "mean"
                    ),
                }
            )
    write_rows(metrics_path, metric_rows)

    comparison: dict[str, Any] | None = None
    qualitative_cases: dict[str, Any] | None = None
    if reference is not None and treatment is not None:
        bootstrap = paired_scene_bootstrap(
            ground_truth_stubs,
            {reference: compiled[reference], treatment: compiled[treatment]},
            physical_groups,
            class_names,
            reference,
            treatment,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            min_region_size=min_region_size,
        )
        ranking = _qualitative_ranking(
            selected_scene_ids,
            per_scene_metrics,
            reference,
            treatment,
            selected_seeds,
        )
        comparison = {
            "reference": reference,
            "treatment": treatment,
            "metric": "map_50_95",
            "bootstrap": bootstrap,
        }
        if ranking:
            qualitative_cases = {
                "worst": ranking[0],
                "median": ranking[len(ranking) // 2],
                "best": ranking[-1],
                "ranking": ranking,
            }

    class_support = {
        name: {
            "gt_instances": int(support_instances[index]),
            "physical_scene_support": len(support_physical_groups[index]),
            "globally_evaluable": bool(support_instances[index] > 0),
        }
        for index, name in enumerate(class_names)
    }
    payload = {
        "schema_version": "1.0",
        "kind": "class_first_analysis",
        "split": split,
        "protocol_version": PROTOCOL_VERSION,
        "scene_count": len(selected_scene_ids),
        "physical_scene_count": len(set(physical_groups.values())),
        "scenes": [
            {
                "scene_id": scene_id,
                "physical_scene_id": physical_groups[scene_id],
            }
            for scene_id in selected_scene_ids
        ],
        "conditions": selected_conditions,
        "technical_replicates": selected_seeds,
        "small_categories": list(SMALL_CATEGORIES),
        "class_support": class_support,
        "per_condition_seed_metrics": evaluated,
        "diagnostics": diagnostic_summaries,
        "comparison": comparison,
        "qualitative_cases": qualitative_cases,
    }
    write_json(analysis_path, payload)
    return payload
