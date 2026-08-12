from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest


LEGACY_PRIOR_CONDITIONS = {
    "L1-uniform": "uniform",
    "D-size": "size",
    "D-smooth": "smooth",
    "D-small": "small",
    "D-combined": "combined",
}


def legacy_prior_run_paths(
    output_root: str | Path, condition: str, scene_id: str, seed: int
) -> dict[str, Path]:
    directory = Path(output_root).resolve() / condition / scene_id / f"seed-{seed}"
    return {
        "run_dir": directory,
        "output": directory / "output.json",
        "pending_output": directory / "output.pending.json",
        "diagnostics": directory / "diagnostics.json",
        "pending_diagnostics": directory / "diagnostics.pending.json",
        "progress": directory / "progress.txt",
        "log": directory / "postprocess.log",
    }


def build_legacy_prior_command(
    pipeline: str | Path,
    scene: Mapping[str, Any],
    output_root: str | Path,
    condition: str,
    scene_id: str,
    seed: int,
    category_priors: str | Path,
    config: str | Path,
    score: str = "unit",
    semantic_source: str = "gaussian",
) -> tuple[list[str], dict[str, Path]]:
    if condition not in LEGACY_PRIOR_CONDITIONS:
        raise ValueError(f"unknown legacy-prior condition: {condition}")
    paths = legacy_prior_run_paths(output_root, condition, scene_id, seed)
    command = [
        "bash", str(Path(pipeline).resolve()),
        "--stage", "postprocess",
        "--base-path", str(scene["base_path"]),
        "--python", str(scene["python_bin"]),
        "--json-path", str(paths["pending_output"]),
        "--prior-metadata-path", str(paths["pending_diagnostics"]),
        "--progress-path", str(paths["progress"]),
        "--scene-scale-m-per-unit", str(float(scene["scene_scale_m_per_unit"])),
        "--seed", str(int(seed)),
        "--clustering-mode", "legacy-prior",
        "--legacy-prior-mode", LEGACY_PRIOR_CONDITIONS[condition],
        "--legacy-prior-score", score,
        "--legacy-prior-semantic-source", semantic_source,
        "--category-priors", str(Path(category_priors).resolve()),
        "--legacy-prior-config", str(Path(config).resolve()),
        "--minimal-metadata",
    ]
    return command, paths


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("point_labels"), list) and isinstance(
        payload.get("instances"), Mapping
    )


def _complete(paths: Mapping[str, Path], condition: str, scene: str, seed: int) -> bool:
    if not _valid_output(paths["output"]):
        return False
    try:
        payload = load_json(paths["diagnostics"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    run = payload.get("run", {})
    return (
        payload.get("status") == "complete"
        and run.get("scene_id") == scene
        and run.get("condition") == condition
        and run.get("seed") == seed
    )


def execute_legacy_prior_runs(
    scene_manifest: str | Path,
    output_root: str | Path,
    pipeline: str | Path,
    category_priors: str | Path,
    config: str | Path,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    scene_ids: Sequence[str] | None = None,
    score: str = "unit",
    semantic_source: str = "gaussian",
    resume: bool = True,
    continue_on_error: bool = False,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    selected_conditions = list(conditions or LEGACY_PRIOR_CONDITIONS)
    selected_seeds = [int(value) for value in (seeds or (42,))]
    selected_scenes = list(scene_ids or scenes)
    unknown = sorted(set(selected_conditions) - set(LEGACY_PRIOR_CONDITIONS))
    missing = sorted(set(selected_scenes) - set(scenes))
    if unknown or missing:
        raise ValueError(f"unknown conditions={unknown}, missing scenes={missing}")
    runs = [
        (scene_id, seed, condition)
        for scene_id in selected_scenes
        for seed in selected_seeds
        for condition in selected_conditions
    ]
    if max_runs is not None:
        runs = runs[:max_runs]
    records: list[dict[str, Any]] = []
    for scene_id, seed, condition in runs:
        command, paths = build_legacy_prior_command(
            pipeline, scenes[scene_id], output_root, condition, scene_id, seed,
            category_priors, config, score, semantic_source,
        )
        record = {"scene_id": scene_id, "condition": condition, "seed": seed}
        if resume and _complete(paths, condition, scene_id, seed):
            record["status"] = "skipped_complete"
            records.append(record)
            continue
        if dry_run:
            record.update({"status": "planned", "command": command})
            records.append(record)
            continue
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        paths["pending_output"].unlink(missing_ok=True)
        paths["pending_diagnostics"].unlink(missing_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        runtime = time.perf_counter() - started
        status = "complete" if result.returncode == 0 and _valid_output(paths["pending_output"]) else "failed"
        diagnostics: dict[str, Any] = {}
        if paths["pending_diagnostics"].is_file():
            candidate = load_json(paths["pending_diagnostics"])
            if isinstance(candidate, Mapping):
                diagnostics = dict(candidate)
        diagnostics.pop("content_sha256", None)
        diagnostics.update({
            "schema_version": "1.0", "kind": "legacy_prior_diagnostics",
            "status": status,
            "run": {
                "scene_id": scene_id,
                "physical_scene_id": str(scenes[scene_id].get("physical_scene_id", scene_id)),
                "condition": condition,
                "legacy_prior_mode": LEGACY_PRIOR_CONDITIONS[condition],
                "seed": seed,
            },
            "runner": {
                "command": command, "runtime_seconds": runtime,
                "return_code": result.returncode, "log": str(paths["log"]),
            },
        })
        if status == "complete":
            output = load_json(paths["pending_output"])
            diagnostics["runner"].update({
                "point_count": len(output["point_labels"]),
                "instance_count": len(output["instances"]),
            })
            os.replace(paths["pending_output"], paths["output"])
        write_json(paths["diagnostics"], diagnostics)
        paths["pending_diagnostics"].unlink(missing_ok=True)
        record["status"] = status
        records.append(record)
        if status == "failed" and not continue_on_error:
            break
    return {
        "kind": "legacy_prior_execution", "planned": len(runs),
        "complete": sum(r["status"] in {"complete", "skipped_complete"} for r in records),
        "failed": sum(r["status"] == "failed" for r in records),
        "records": records,
    }
