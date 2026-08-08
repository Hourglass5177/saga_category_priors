from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import evaluate_manifest
from .io import (
    hash_json,
    load_json,
    sha256_file,
    write_json,
    write_rows,
)
from .mapping import (
    DEFAULT_MAPPING_CONFIG,
    build_mapping_config,
    validate_mapping_config,
)
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import Taxonomy


def _validate_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not expected or hash_json(unsigned) != expected:
        raise ValueError(f"{label} content hash mismatch")


def _search_kind(design: Mapping[str, Any]) -> str:
    kind = str(design.get("kind", ""))
    if kind == "global_search_design":
        return "global"
    if kind == "prior_search_design":
        return "prior"
    raise ValueError(f"Unsupported search design kind: {kind}")


def _resolve_manifest_path(manifest_path: str | Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    return path.resolve()


def materialize_search_mappings(
    design_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    priors_path: str | Path,
    taxonomy_path: str | Path,
    scene_selection_path: str | Path,
    base_mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    design = load_json(design_path)
    _validate_content_hash(design, "Search design")
    kind = _search_kind(design)
    if kind == "prior" and base_mapping_path is None:
        raise ValueError("Prior search requires the selected global base mapping")
    if kind == "global" and base_mapping_path is not None:
        raise ValueError("Global search must start from the registered defaults")

    base_mapping: dict[str, Any] | None = None
    if base_mapping_path is not None:
        base_mapping = load_json(base_mapping_path)
        validate_mapping_config(base_mapping)
        if (
            base_mapping["provenance"]["category_priors_sha256"]
            != sha256_file(priors_path)
        ):
            raise ValueError("Base mapping does not match category priors")

    output_dir = Path(output_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in design["configurations"]:
        config_id = str(item["config_id"])
        parameters = dict(item["parameters"])
        if kind == "global":
            global_parameters = parameters
            prior_parameters = DEFAULT_MAPPING_CONFIG["coefficients"]
        else:
            assert base_mapping is not None
            global_parameters = base_mapping["baseline"]
            prior_parameters = parameters
        mapping = build_mapping_config(
            global_parameters,
            prior_parameters,
            priors_path,
            taxonomy_path,
            scene_selection_path,
        )
        mapping["provenance"].update(
            {
                "search_kind": kind,
                "search_config_id": config_id,
                "search_design_sha256": sha256_file(design_path),
            }
        )
        if base_mapping_path is not None:
            mapping["provenance"]["base_mapping_sha256"] = sha256_file(
                base_mapping_path
            )
        mapping.pop("content_sha256", None)
        mapping["content_sha256"] = hash_json(mapping)
        target = output_dir / f"{config_id}.json"
        write_json(target, mapping)
        records.append(
            {
                "config_id": config_id,
                "parameters": parameters,
                "path": os.path.relpath(target, manifest_path.parent).replace(
                    "\\", "/"
                ),
                "sha256": sha256_file(target),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "search_mapping_manifest",
        "search_kind": kind,
        "split": "val-tune",
        "search_design_sha256": sha256_file(design_path),
        "category_priors_sha256": sha256_file(priors_path),
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "scene_selection_sha256": sha256_file(scene_selection_path),
        "base_mapping_sha256": sha256_file(base_mapping_path)
        if base_mapping_path
        else None,
        "configurations": records,
    }
    manifest["content_sha256"] = hash_json(manifest)
    write_json(manifest_path, manifest)
    return manifest


def build_search_schedule(
    scene_selection_path: str | Path,
    mapping_manifest_path: str | Path,
    run_seeds: Sequence[int] = (42,),
    randomization_seed: int = 20260804,
) -> dict[str, Any]:
    mapping_manifest = load_json(mapping_manifest_path)
    if mapping_manifest.get("kind") != "search_mapping_manifest":
        raise ValueError("Expected a search_mapping_manifest")
    _validate_content_hash(mapping_manifest, "Search mapping manifest")
    if mapping_manifest.get("split") != "val-tune":
        raise ValueError("Search schedules are val-tune only")
    kind = str(mapping_manifest["search_kind"])
    condition = "P000-B2" if kind == "global" else "P111-combined"
    configurations = list(mapping_manifest["configurations"])
    if not configurations:
        raise ValueError("Search mapping manifest has no configurations")
    if not run_seeds or len(run_seeds) != len({int(seed) for seed in run_seeds}):
        raise ValueError("run_seeds must be nonempty and unique")

    selection = load_json(scene_selection_path)
    scenes = sorted(str(scene) for scene in selection["selection"]["tune"])
    mapping_base = Path(mapping_manifest_path).resolve().parent
    resolved_configs = []
    for item in configurations:
        path = Path(item["path"])
        if not path.is_absolute():
            path = mapping_base / path
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Mapping hash mismatch: {item['config_id']}")
        resolved_configs.append({**item, "path": str(path.resolve())})

    rng = np.random.default_rng(randomization_seed)
    runs: list[dict[str, Any]] = []
    sequence = 0
    for scene_id in scenes:
        for run_seed in sorted(int(seed) for seed in run_seeds):
            order = rng.permutation(len(resolved_configs)).tolist()
            for within_block_order, config_index in enumerate(order):
                config = resolved_configs[config_index]
                runs.append(
                    {
                        "sequence": sequence,
                        "block": f"{scene_id}/seed-{run_seed}",
                        "scene_id": scene_id,
                        "run_seed": run_seed,
                        "within_block_order": within_block_order,
                        "condition": condition,
                        "config_id": config["config_id"],
                        "mapping_path": config["path"],
                        "mapping_sha256": config["sha256"],
                    }
                )
                sequence += 1
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "run_schedule",
        "split": "val-tune",
        "search_kind": kind,
        "condition": condition,
        "randomization": "configuration order shuffled within scene/seed blocks",
        "randomization_seed": randomization_seed,
        "scene_selection_sha256": sha256_file(scene_selection_path),
        "mapping_manifest_sha256": sha256_file(mapping_manifest_path),
        "configurations": [item["config_id"] for item in resolved_configs],
        "seeds": sorted(int(seed) for seed in run_seeds),
        "runs": runs,
    }
    payload["content_sha256"] = hash_json(payload)
    return payload


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


def evaluate_search_execution(
    schedule_path: str | Path,
    execution_path: str | Path,
    scene_manifest_path: str | Path,
    gt_manifest_path: str | Path,
    taxonomy: Taxonomy,
    output_dir: str | Path,
    metrics_path: str | Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> list[dict[str, Any]]:
    schedule = load_json(schedule_path)
    if schedule.get("kind") != "run_schedule" or not schedule.get("search_kind"):
        raise ValueError("Expected a search run_schedule")
    _validate_content_hash(schedule, "Search schedule")
    if schedule.get("split") != "val-tune":
        raise ValueError("Search evaluation is val-tune only")
    execution = load_json(execution_path)
    if execution.get("kind") != "run_execution":
        raise ValueError("Expected a run_execution manifest")
    if execution.get("schedule_sha256") != sha256_file(schedule_path):
        raise ValueError("Execution does not match search schedule")

    scenes = load_scene_runtime_manifest(scene_manifest_path)
    gt_manifest = load_json(gt_manifest_path)
    if gt_manifest.get("kind") != "canonical_ground_truth":
        raise ValueError("Expected a canonical_ground_truth manifest")
    gt_base = Path(gt_manifest_path).resolve().parent
    gt_by_scene = {
        str(item["scene_id"]): (gt_base / item["path"]).resolve()
        for item in gt_manifest["scenes"]
    }
    execution_by_sequence = {
        int(item["sequence"]): item for item in execution.get("runs", [])
    }
    scheduled_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for run in schedule["runs"]:
        scheduled_groups[(str(run["config_id"]), int(run["run_seed"]))].append(run)

    output_dir = Path(output_dir).resolve()
    manifest_dir = output_dir / "manifests"
    result_dir = output_dir / "results"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for (config_id, run_seed), group in sorted(scheduled_groups.items()):
        manifest_scenes: list[dict[str, Any]] = []
        runtimes: list[float] = []
        for run in sorted(group, key=lambda item: str(item["scene_id"])):
            sequence = int(run["sequence"])
            if sequence not in execution_by_sequence:
                raise ValueError(f"Missing execution record for sequence {sequence}")
            record = execution_by_sequence[sequence]
            if record.get("status") not in {"complete", "skipped_complete"}:
                raise ValueError(
                    f"Incomplete search run {sequence}: {record.get('status')}"
                )
            if "runtime_seconds" not in record:
                raise ValueError(f"Search run {sequence} is missing runtime_seconds")
            runtimes.append(float(record["runtime_seconds"]))
            scene_id = str(run["scene_id"])
            if scene_id not in scenes or scene_id not in gt_by_scene:
                raise ValueError(f"Missing runtime or GT scene: {scene_id}")
            scene = scenes[scene_id]
            gaussian_path = _scene_gaussian_path(scene)
            transform = scene.get("gaussian_to_gt_transform", np.eye(4).tolist())
            manifest_scenes.append(
                {
                    "scene_id": scene_id,
                    "gt_npz": str(gt_by_scene[scene_id]),
                    "output_json": str(Path(record["output_json"]).resolve()),
                    "metadata_json": str(Path(record["metadata_json"]).resolve()),
                    "gaussian_ply": str(gaussian_path),
                    "gaussian_to_gt_transform": transform,
                }
            )
        evaluation_manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "kind": "evaluation_manifest",
            "split": "val-tune",
            "search_kind": schedule["search_kind"],
            "condition": schedule["condition"],
            "config_id": config_id,
            "run_seed": run_seed,
            "minimum_mapped_fraction": min(
                float(scene.get("minimum_mapped_fraction", 0.90))
                for scene in (scenes[str(run["scene_id"])] for run in group)
            ),
            "scenes": manifest_scenes,
        }
        evaluation_manifest["content_sha256"] = hash_json(evaluation_manifest)
        stem = f"{config_id}-seed-{run_seed}"
        evaluation_manifest_path = manifest_dir / f"{stem}.json"
        evaluation_result_path = result_dir / f"{stem}.json"
        write_json(evaluation_manifest_path, evaluation_manifest)
        evaluated = evaluate_manifest(
            evaluation_manifest_path,
            taxonomy,
            evaluation_result_path,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        diagnostics = list(evaluated["diagnostics"].values())
        aggregate = evaluated["aggregate"]
        metric_rows.append(
            {
                "config_id": config_id,
                "split": "val-tune",
                "protocol_version": evaluated["protocol_version"],
                "run_seed": run_seed,
                "scene_count": len(group),
                "map_50_95": aggregate["map_50_95"],
                "map_0.50": aggregate["map_0.50"],
                "map_0.25": aggregate["map_0.25"],
                "runtime_seconds": float(np.mean(runtimes)),
                "failure_rate": 0.0,
                "minimum_mapped_fraction": min(
                    float(item["mapped_fraction"]) for item in diagnostics
                ),
                "median_nn_distance_m": float(
                    np.median(
                        [float(item["median_nn_distance_m"]) for item in diagnostics]
                    )
                ),
                "evaluation_result": str(evaluation_result_path),
                "evaluation_result_sha256": sha256_file(evaluation_result_path),
            }
        )
    if not metric_rows:
        raise ValueError("No search evaluations were produced")
    if any(
        row["map_50_95"] is None or not math.isfinite(float(row["map_50_95"]))
        for row in metric_rows
    ):
        raise ValueError("Search evaluation produced a non-finite primary metric")
    write_rows(metrics_path, metric_rows)
    return metric_rows
