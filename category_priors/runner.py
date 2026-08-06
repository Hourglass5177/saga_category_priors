from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import hash_json, load_json, sha256_file, write_json

CONDITION_OPTIONS: dict[str, tuple[str, ...]] = {
    "B0-legacy": ("--prior-mode", "off", "--disable-other-classes"),
    "B1-other-classes": ("--prior-mode", "off"),
    "P000-B2": ("--prior-mode", "global"),
    "P001-small": ("--prior-mode", "small"),
    "P010-smooth": ("--prior-mode", "smooth"),
    "P011-smooth-small": ("--prior-mode", "smooth-small"),
    "P100-size": ("--prior-mode", "size"),
    "P101-size-small": ("--prior-mode", "size-small"),
    "P110-size-smooth": ("--prior-mode", "size-smooth"),
    "P111-combined": ("--prior-mode", "combined"),
    "P111-no-gate": ("--prior-mode", "combined", "--prior-gate", "off"),
    "P111-no-shrink": ("--prior-mode", "combined", "--prior-shrink", "off"),
}


def _validate_content_hash(payload: Mapping[str, Any], label: str) -> None:
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not expected or hash_json(unsigned) != expected:
        raise ValueError(f"{label} content hash mismatch")


def load_scene_runtime_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if payload.get("kind") != "scene_runtime_manifest":
        raise ValueError("Expected a scene_runtime_manifest")
    scenes: dict[str, dict[str, Any]] = {}
    base = Path(path).parent
    for item in payload["scenes"]:
        scene_id = str(item["scene_id"])
        if scene_id in scenes:
            raise ValueError(f"Duplicate runtime scene: {scene_id}")
        scale = float(item["scene_scale_m_per_unit"])
        if scale <= 0:
            raise ValueError(f"{scene_id}: scene_scale_m_per_unit must be positive")
        base_path = Path(item["base_path"])
        if not base_path.is_absolute():
            base_path = (base / base_path).resolve()
        scenes[scene_id] = {
            **item,
            "base_path": str(base_path),
            "scene_scale_m_per_unit": scale,
        }
        if item.get("python_bin"):
            python_bin = Path(str(item["python_bin"]))
            if not python_bin.is_absolute():
                python_bin = (base / python_bin).resolve()
            scenes[scene_id]["python_bin"] = str(python_bin)
    return scenes


def build_postprocess_command(
    pipeline_path: str | Path,
    run: Mapping[str, Any],
    scene: Mapping[str, Any],
    output_root: str | Path,
    priors_path: str | Path | None,
    mapping_path: str | Path | None,
) -> tuple[list[str], dict[str, Path]]:
    condition = str(run["condition"])
    if condition not in CONDITION_OPTIONS:
        raise ValueError(f"Unregistered condition: {condition}")
    run_dir = Path(output_root) / condition
    if run.get("config_id"):
        run_dir = run_dir / str(run["config_id"])
    run_dir = run_dir / str(run["scene_id"]) / f"seed-{int(run['run_seed'])}"
    paths = {
        "run_dir": run_dir,
        "output_json": run_dir / "output.json",
        "metadata_json": run_dir / "output.json.metadata.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
    }
    command = [
        "bash",
        str(Path(pipeline_path).resolve()),
        "--stage",
        "postprocess",
        "--base-path",
        str(scene["base_path"]),
        "--json-path",
        str(paths["output_json"]),
        "--prior-metadata-path",
        str(paths["metadata_json"]),
        "--progress-path",
        str(paths["progress"]),
        "--max-contributor-cache-path",
        str(output_root / ".cache" / "max_contributors" / str(run["scene_id"])),
        "--scene-scale-m-per-unit",
        str(float(scene["scene_scale_m_per_unit"])),
        "--seed",
        str(int(run["run_seed"])),
        *CONDITION_OPTIONS[condition],
    ]
    if scene.get("python_bin"):
        command[2:2] = ["--python", str(scene["python_bin"])]
    if condition.startswith("P"):
        if priors_path is None or mapping_path is None:
            raise ValueError(f"{condition} requires priors and mapping paths")
        command.extend(
            [
                "--prior-config",
                str(Path(priors_path).resolve()),
                "--prior-mapping-config",
                str(Path(mapping_path).resolve()),
            ]
        )
    return command, paths


def execute_schedule(
    schedule_path: str | Path,
    scene_manifest_path: str | Path,
    output_root: str | Path,
    result_path: str | Path,
    pipeline_path: str | Path,
    priors_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    continue_on_error: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    schedule = load_json(schedule_path)
    if schedule.get("kind") != "run_schedule":
        raise ValueError("Expected a run_schedule")
    _validate_content_hash(schedule, "Run schedule")
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    selected_runs = list(schedule["runs"])
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive")
        selected_runs = selected_runs[:max_runs]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    previous_by_sequence: dict[int, Mapping[str, Any]] = {}
    result_path = Path(result_path).resolve()
    if resume and result_path.is_file():
        previous = load_json(result_path)
        if previous.get("schedule_sha256") != sha256_file(schedule_path):
            raise ValueError("Existing execution manifest belongs to another schedule")
        previous_by_sequence = {
            int(item["sequence"]): item for item in previous.get("runs", [])
        }

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "run_execution",
        "schedule_sha256": sha256_file(schedule_path),
        "scene_manifest_sha256": sha256_file(scene_manifest_path),
        "pipeline_sha256": sha256_file(pipeline_path),
        "category_priors_sha256": sha256_file(priors_path) if priors_path else None,
        "prior_mapping_sha256": sha256_file(mapping_path) if mapping_path else None,
        "dry_run": dry_run,
        "resume": resume,
        "runs": [],
    }
    for run in selected_runs:
        scene_id = str(run["scene_id"])
        if scene_id not in scenes:
            raise ValueError(
                f"Schedule scene is missing from runtime manifest: {scene_id}"
            )
        run_mapping_path = run.get("mapping_path", mapping_path)
        if run.get("mapping_sha256"):
            if run_mapping_path is None or sha256_file(run_mapping_path) != run.get(
                "mapping_sha256"
            ):
                raise ValueError(
                    f"{run.get('config_id', scene_id)}: mapping hash mismatch"
                )
        command, paths = build_postprocess_command(
            pipeline_path,
            run,
            scenes[scene_id],
            output_root,
            priors_path,
            run_mapping_path,
        )
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "sequence": int(run["sequence"]),
            "scene_id": scene_id,
            "condition": str(run["condition"]),
            "run_seed": int(run["run_seed"]),
            "command": command,
            "output_json": str(paths["output_json"]),
            "metadata_json": str(paths["metadata_json"]),
            "log": str(paths["log"]),
        }
        if run.get("config_id"):
            record["config_id"] = str(run["config_id"])
            record["mapping_path"] = str(Path(run_mapping_path).resolve())
            record["mapping_sha256"] = sha256_file(run_mapping_path)
        if (
            resume
            and paths["output_json"].is_file()
            and paths["metadata_json"].is_file()
        ):
            record["status"] = "skipped_complete"
            previous_record = previous_by_sequence.get(int(run["sequence"]), {})
            if "runtime_seconds" in previous_record:
                record["runtime_seconds"] = float(previous_record["runtime_seconds"])
        elif dry_run:
            record["status"] = "planned"
        else:
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
            record["return_code"] = completed.returncode
            record["runtime_seconds"] = time.perf_counter() - started
            record["status"] = "complete" if completed.returncode == 0 else "failed"
            if completed.returncode == 0 and not (
                paths["output_json"].is_file() and paths["metadata_json"].is_file()
            ):
                record["status"] = "failed_missing_outputs"
        result["runs"].append(record)
        unsigned = dict(result)
        unsigned.pop("content_sha256", None)
        result["content_sha256"] = hash_json(unsigned)
        write_json(result_path, result)
        if str(record["status"]).startswith("failed") and not continue_on_error:
            raise RuntimeError(f"Run failed; inspect {paths['log']}")
    return result
