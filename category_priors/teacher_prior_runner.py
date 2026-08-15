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


TEACHER_PRIOR_CONDITIONS: dict[str, str] = {
    "off": "off",
    "original": "original",
    "U0-all-uniform": "all-uniform",
    "D-size": "size",
    "D-smooth": "smooth",
    "D-small": "small",
    "D-combined": "combined",
    "U0-multi-anchor": "all-uniform",
    "D-combined-multi-anchor": "combined",
    # Offline replay condition: it reuses the original B1 output and therefore
    # is never dispatched through this runner, but the shared evaluator must
    # recognize its condition identity.
    "B1-proposal-protected": "original",
}
TEACHER_PRIOR_EXPERIMENT_CONDITIONS = (
    "U0-all-uniform",
    "D-size",
    "D-smooth",
    "D-small",
    "D-combined",
)

_PARAMETERIZED_MODES = frozenset(
    {"all-uniform", "size", "smooth", "small", "combined"}
)

TEACHER_PRIOR_PROTECTION: dict[str, str] = {
    condition: (
        "multi-anchor" if condition.endswith("-multi-anchor") else "off"
    )
    for condition in TEACHER_PRIOR_CONDITIONS
}


def teacher_prior_run_paths(
    output_root: str | Path,
    condition: str,
    scene_id: str,
    seed: int,
) -> dict[str, Path]:
    run_dir = Path(output_root).resolve() / condition / scene_id / f"seed-{seed}"
    return {
        "run_dir": run_dir,
        "output": run_dir / "output.json",
        "pending_output": run_dir / "output.pending.json",
        "diagnostics": run_dir / "diagnostics.json",
        "pending_diagnostics": run_dir / "diagnostics.pending.json",
        "progress": run_dir / "progress.pending.txt",
        "log": run_dir / "postprocess.log",
    }


def build_teacher_prior_command(
    pipeline: str | Path,
    scene: Mapping[str, Any],
    output_root: str | Path,
    condition: str,
    scene_id: str,
    seed: int,
    category_params: str | Path | None = None,
) -> tuple[list[str], dict[str, Path]]:
    if condition not in TEACHER_PRIOR_CONDITIONS:
        raise ValueError(f"unknown teacher-prior condition: {condition}")
    if not scene.get("python_bin"):
        raise ValueError(f"{scene_id}: scene runtime manifest must define python_bin")
    mode = TEACHER_PRIOR_CONDITIONS[condition]
    if mode in _PARAMETERIZED_MODES and category_params is None:
        raise ValueError(f"{condition} requires teacher category parameters")

    paths = teacher_prior_run_paths(output_root, condition, scene_id, seed)
    command = [
        "bash",
        str(Path(pipeline).resolve()),
        "--stage",
        "postprocess",
        "--base-path",
        str(scene["base_path"]),
        "--python",
        str(scene["python_bin"]),
        "--json-path",
        str(paths["pending_output"]),
        "--prior-metadata-path",
        str(paths["pending_diagnostics"]),
        "--progress-path",
        str(paths["progress"]),
        "--scene-scale-m-per-unit",
        str(float(scene["scene_scale_m_per_unit"])),
        "--seed",
        str(int(seed)),
        "--teacher-prior-mode",
        mode,
        "--minimal-metadata",
    ]
    if mode in _PARAMETERIZED_MODES:
        command.extend(
            ["--teacher-category-params", str(Path(category_params).resolve())]
        )
    protection = TEACHER_PRIOR_PROTECTION[condition]
    if protection != "off":
        command.extend(["--teacher-evidence-protection", protection])
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


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _complete(paths: Mapping[str, Path]) -> bool:
    return _valid_output(paths["output"])


def _write_diagnostics(
    paths: Mapping[str, Path],
    scene: Mapping[str, Any],
    condition: str,
    scene_id: str,
    seed: int,
    command: Sequence[str],
    status: str,
    runtime_seconds: float,
    return_code: int,
    error: str | None = None,
) -> None:
    diagnostics = _load_mapping(paths["pending_diagnostics"])
    diagnostics.pop("content_sha256", None)
    diagnostics["schema_version"] = "1.0"
    diagnostics["kind"] = "teacher_prior_diagnostics"
    diagnostics["status"] = status
    diagnostics["run"] = {
        "scene_id": scene_id,
        "physical_scene_id": str(scene.get("physical_scene_id", scene_id)),
        "condition": condition,
        "teacher_prior_mode": TEACHER_PRIOR_CONDITIONS[condition],
        "teacher_evidence_protection": TEACHER_PRIOR_PROTECTION[condition],
        "seed": int(seed),
        "output_json": str(paths["output"]),
    }
    diagnostics["runner"] = {
        "command": [str(value) for value in command],
        "runtime_seconds": float(runtime_seconds),
        "return_code": int(return_code),
        "log": str(paths["log"]),
    }
    output_path = (
        paths["pending_output"]
        if paths["pending_output"].is_file()
        else paths["output"]
    )
    if status == "complete" and _valid_output(output_path):
        output = load_json(output_path)
        diagnostics["runner"]["point_count"] = len(output["point_labels"])
        diagnostics["runner"]["instance_count"] = len(output["instances"])
    if error:
        diagnostics["runner"]["error"] = error
    write_json(paths["diagnostics"], diagnostics)


def _selected(
    values: Sequence[Any] | None, defaults: Sequence[Any], label: str
) -> list[Any]:
    result = list(defaults if values is None else values)
    if not result:
        raise ValueError(f"at least one {label} is required")
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate {label} values are not allowed")
    return result


def execute_teacher_prior_runs(
    scene_manifest: str | Path,
    output_root: str | Path,
    pipeline: str | Path,
    category_params: str | Path | None = None,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    scene_ids: Sequence[str] | None = None,
    resume: bool = True,
    continue_on_error: bool = False,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    selected_conditions = [
        str(value)
        for value in _selected(
            conditions,
            TEACHER_PRIOR_EXPERIMENT_CONDITIONS,
            "condition",
        )
    ]
    unknown = sorted(set(selected_conditions) - set(TEACHER_PRIOR_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown teacher-prior conditions: {unknown}")
    if category_params is None and any(
        TEACHER_PRIOR_CONDITIONS[condition] in _PARAMETERIZED_MODES
        for condition in selected_conditions
    ):
        raise ValueError("parameterized teacher-prior conditions require --teacher-category-params")

    selected_seeds = [
        int(value) for value in _selected(seeds, (42,), "seed")
    ]
    selected_scene_ids = [
        str(value) for value in _selected(scene_ids, tuple(scenes), "scene")
    ]
    missing = sorted(set(selected_scene_ids) - set(scenes))
    if missing:
        raise ValueError(f"scene runtime manifest is missing scenes: {missing}")
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
        command, paths = build_teacher_prior_command(
            pipeline,
            scenes[scene_id],
            output_root,
            condition,
            scene_id,
            seed,
            category_params,
        )
        record: dict[str, Any] = {
            "scene_id": scene_id,
            "condition": condition,
            "teacher_prior_mode": TEACHER_PRIOR_CONDITIONS[condition],
            "teacher_evidence_protection": TEACHER_PRIOR_PROTECTION[condition],
            "seed": seed,
            "run_dir": str(paths["run_dir"]),
        }
        if resume and _complete(paths):
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
        paths["progress"].unlink(missing_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                cwd=Path(pipeline).resolve().parent,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        runtime_seconds = time.perf_counter() - started
        error: str | None = None
        if result.returncode != 0:
            error = f"postprocess exited with return code {result.returncode}"
        elif not _valid_output(paths["pending_output"]):
            error = "postprocess did not produce a parseable output.json"
        elif not _load_mapping(paths["pending_diagnostics"]):
            error = "postprocess did not produce parseable diagnostics.json"

        status = "complete" if error is None else "failed"
        if status == "complete":
            try:
                os.replace(paths["pending_output"], paths["output"])
            except OSError as exc:
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
        _write_diagnostics(
            paths,
            scenes[scene_id],
            condition,
            scene_id,
            seed,
            command,
            status,
            runtime_seconds,
            result.returncode,
            error,
        )
        paths["pending_diagnostics"].unlink(missing_ok=True)
        paths["progress"].unlink(missing_ok=True)
        record.update(
            {
                "status": status,
                "runtime_seconds": runtime_seconds,
                "return_code": result.returncode,
                "log": str(paths["log"]),
            }
        )
        if error:
            record["error"] = error
        records.append(record)
        if status == "failed" and not continue_on_error:
            raise RuntimeError(f"teacher-prior run failed; inspect {paths['log']}: {error}")

    return {
        "kind": "teacher_prior_execution",
        "planned": len(runs),
        "complete": sum(item["status"] == "complete" for item in records),
        "skipped": sum(item["status"] == "skipped_complete" for item in records),
        "failed": sum(item["status"] == "failed" for item in records),
        "runs": records,
    }
