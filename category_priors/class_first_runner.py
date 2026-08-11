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


CLASS_FIRST_CONDITIONS: dict[str, str] = {
    "U0-uniform": "uniform",
    "D-size": "size",
    "D-smooth": "smooth",
    "D-small": "small",
    "D-combined": "combined",
}


def _run_paths(
    output_root: str | Path,
    condition: str,
    scene_id: str,
    seed: int,
) -> dict[str, Path]:
    run_dir = Path(output_root).resolve() / condition / scene_id / f"seed-{seed}"
    return {
        "run_dir": run_dir,
        "output_json": run_dir / "output.json",
        "pending_output_json": run_dir / "output.pending.json",
        "diagnostics_json": run_dir / "diagnostics.json",
        "pending_diagnostics_json": run_dir / "diagnostics.pending.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
    }


def build_class_first_command(
    pipeline_path: str | Path,
    scene: Mapping[str, Any],
    output_root: str | Path,
    condition: str,
    scene_id: str,
    seed: int,
    category_priors_path: str | Path,
    class_first_config_path: str | Path,
) -> tuple[list[str], dict[str, Path]]:
    if condition not in CLASS_FIRST_CONDITIONS:
        raise ValueError(f"Unknown class-first condition: {condition}")
    if not scene.get("python_bin"):
        raise ValueError(f"{scene_id}: scene runtime manifest must define python_bin")

    paths = _run_paths(output_root, condition, scene_id, seed)
    command = [
        "bash",
        str(Path(pipeline_path).resolve()),
        "--stage",
        "postprocess",
        "--base-path",
        str(scene["base_path"]),
        "--python",
        str(scene["python_bin"]),
        "--json-path",
        str(paths["pending_output_json"]),
        "--prior-metadata-path",
        str(paths["pending_diagnostics_json"]),
        "--progress-path",
        str(paths["progress"]),
        "--scene-scale-m-per-unit",
        str(float(scene["scene_scale_m_per_unit"])),
        "--seed",
        str(int(seed)),
        "--clustering-mode",
        "class-first",
        "--class-prior-mode",
        CLASS_FIRST_CONDITIONS[condition],
        "--category-priors",
        str(Path(category_priors_path).resolve()),
        "--class-first-config",
        str(Path(class_first_config_path).resolve()),
        "--minimal-metadata",
    ]
    return command, paths


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and isinstance(payload.get("point_labels"), list)
        and isinstance(payload.get("instances"), Mapping)
    )


def _diagnostic_identity(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    run = payload.get("run")
    return run if isinstance(run, Mapping) else payload


def _completed_run(
    paths: Mapping[str, Path], condition: str, scene_id: str, seed: int
) -> bool:
    if not _valid_output(paths["output_json"]):
        return False
    try:
        diagnostics = load_json(paths["diagnostics_json"])
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(diagnostics, Mapping) or diagnostics.get("status") != "complete":
        return False
    identity = _diagnostic_identity(diagnostics)
    return (
        identity.get("scene_id") == scene_id
        and identity.get("condition") == condition
        and identity.get("seed") == seed
    )


def _read_optional_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _valid_mapping(path: Path) -> bool:
    try:
        return isinstance(load_json(path), Mapping)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_diagnostics(
    paths: Mapping[str, Path],
    scene: Mapping[str, Any],
    condition: str,
    scene_id: str,
    seed: int,
    command: Sequence[str],
    status: str,
    runtime_seconds: float | None = None,
    return_code: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    diagnostics = _read_optional_mapping(paths["pending_diagnostics_json"])
    diagnostics.pop("content_sha256", None)
    diagnostics.setdefault("schema_version", "1.0")
    diagnostics.setdefault("kind", "class_first_diagnostics")
    run = diagnostics.get("run")
    run = dict(run) if isinstance(run, Mapping) else {}
    run.update(
        {
            "scene_id": scene_id,
            "physical_scene_id": str(scene.get("physical_scene_id", scene_id)),
            "condition": condition,
            "class_prior_mode": CLASS_FIRST_CONDITIONS[condition],
            "seed": int(seed),
            "output_json": str(paths["output_json"]),
        }
    )
    diagnostics["run"] = run
    diagnostics["status"] = status
    diagnostics["runner"] = {
        "command": [str(value) for value in command],
        "log": str(paths["log"]),
        "runtime_seconds": runtime_seconds,
        "return_code": return_code,
    }
    if error:
        diagnostics["runner"]["error"] = error
    if status == "complete":
        output_path = (
            paths["pending_output_json"]
            if paths["pending_output_json"].is_file()
            else paths["output_json"]
        )
        output = load_json(output_path)
        diagnostics["runner"]["point_count"] = len(output["point_labels"])
        diagnostics["runner"]["instance_count"] = len(output["instances"])
    write_json(paths["diagnostics_json"], diagnostics)
    return diagnostics


def _selected_values(
    values: Sequence[Any] | None, defaults: Sequence[Any], label: str
) -> list[Any]:
    selected = list(defaults if values is None else values)
    if not selected:
        raise ValueError(f"At least one {label} is required")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate {label} values are not allowed")
    return selected


def execute_class_first_runs(
    scene_manifest_path: str | Path,
    output_root: str | Path,
    pipeline_path: str | Path,
    category_priors_path: str | Path,
    class_first_config_path: str | Path,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    scene_ids: Sequence[str] | None = None,
    resume: bool = True,
    continue_on_error: bool = False,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    selected_conditions = _selected_values(
        conditions, tuple(CLASS_FIRST_CONDITIONS), "condition"
    )
    unknown_conditions = sorted(set(selected_conditions) - set(CLASS_FIRST_CONDITIONS))
    if unknown_conditions:
        raise ValueError(f"Unknown class-first conditions: {unknown_conditions}")
    selected_seeds = [int(value) for value in _selected_values(seeds, (42,), "seed")]
    selected_scene_ids = [
        str(value)
        for value in _selected_values(scene_ids, tuple(scenes), "scene")
    ]
    missing_scenes = sorted(set(selected_scene_ids) - set(scenes))
    if missing_scenes:
        raise ValueError(f"Scene runtime manifest is missing scenes: {missing_scenes}")
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive")

    runs = [
        (scene_id, seed, condition)
        for scene_id in selected_scene_ids
        for seed in selected_seeds
        for condition in selected_conditions
    ]
    if max_runs is not None:
        runs = runs[:max_runs]

    records: list[dict[str, Any]] = []
    for scene_id, seed, condition in runs:
        scene = scenes[scene_id]
        command, paths = build_class_first_command(
            pipeline_path,
            scene,
            output_root,
            condition,
            scene_id,
            seed,
            category_priors_path,
            class_first_config_path,
        )
        record: dict[str, Any] = {
            "scene_id": scene_id,
            "condition": condition,
            "class_prior_mode": CLASS_FIRST_CONDITIONS[condition],
            "seed": seed,
            "run_dir": str(paths["run_dir"]),
        }
        if resume and _completed_run(paths, condition, scene_id, seed):
            record["status"] = "skipped_complete"
            records.append(record)
            continue
        if dry_run:
            record["status"] = "planned"
            record["command"] = command
            records.append(record)
            continue

        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        paths["pending_output_json"].unlink(missing_ok=True)
        paths["pending_diagnostics_json"].unlink(missing_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=Path(pipeline_path).resolve().parent,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        runtime_seconds = time.perf_counter() - started
        record.update(
            {
                "runtime_seconds": runtime_seconds,
                "return_code": completed.returncode,
                "log": str(paths["log"]),
            }
        )
        error: str | None = None
        if completed.returncode != 0:
            error = f"postprocess exited with return code {completed.returncode}"
        elif not _valid_output(paths["pending_output_json"]):
            error = "postprocess did not produce a parseable output.json"
        elif not _valid_mapping(paths["pending_diagnostics_json"]):
            error = "postprocess did not produce parseable diagnostics.json"

        if error is None:
            try:
                os.replace(paths["pending_output_json"], paths["output_json"])
                _write_diagnostics(
                    paths,
                    scene,
                    condition,
                    scene_id,
                    seed,
                    command,
                    "complete",
                    runtime_seconds,
                    completed.returncode,
                )
                paths["pending_diagnostics_json"].unlink(missing_ok=True)
                record["status"] = "complete"
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                error = f"{type(exc).__name__}: {exc}"

        if error is not None:
            _write_diagnostics(
                paths,
                scene,
                condition,
                scene_id,
                seed,
                command,
                "failed",
                runtime_seconds,
                completed.returncode,
                error,
            )
            record["status"] = "failed"
            record["error"] = error
        records.append(record)
        if record["status"] == "failed" and not continue_on_error:
            raise RuntimeError(f"Class-first run failed; inspect {paths['log']}: {error}")

    return {
        "planned": len(runs),
        "complete": sum(item["status"] == "complete" for item in records),
        "skipped": sum(item["status"] == "skipped_complete" for item in records),
        "failed": sum(item["status"] == "failed" for item in records),
        "runs": records,
    }
