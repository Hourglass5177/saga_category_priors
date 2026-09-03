from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import hash_json, load_json, sha256_file, write_json
from .locked import (
    expand_locked_runs,
    summarize_mapping,
    summarize_priors,
    validate_locked_plan,
)

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
    # Runtime manifests predate the embedded hash contract and this loader is
    # shared by those historical audit paths.  Requiring a hash here would
    # break valid read-only reproductions; when a manifest does declare one,
    # however, silently ignoring a mismatch would defeat the lock entirely.
    if "content_sha256" in payload:
        _validate_content_hash(payload, "Scene runtime manifest")
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
    staging: bool = False,
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
        "run_record": run_dir / "run.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
    }
    command_output = (
        run_dir / "output.pending.json" if staging else paths["output_json"]
    )
    command_metadata = (
        run_dir / "output.pending.metadata.json" if staging else paths["metadata_json"]
    )
    paths["staged_output_json"] = command_output
    paths["staged_metadata_json"] = command_metadata
    command = [
        "bash",
        str(Path(pipeline_path).resolve()),
        "--stage",
        "postprocess",
        "--base-path",
        str(scene["base_path"]),
        "--json-path",
        str(command_output),
        "--prior-metadata-path",
        str(command_metadata),
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
    if staging:
        command.append("--minimal-metadata")
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


def _completed_locked_run(paths: Mapping[str, Path], run: Mapping[str, Any]) -> bool:
    try:
        output = load_json(paths["output_json"])
        metadata = load_json(paths["metadata_json"])
        record = load_json(paths["run_record"])
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return False
    if not isinstance(output.get("point_labels"), list) or not isinstance(
        output.get("instances"), dict
    ):
        return False
    if metadata.get("kind") != "saga_instance_metadata":
        return False
    expected = {
        "run_id": str(run["run_id"]),
        "scene_id": str(run["scene_id"]),
        "condition": str(run["condition"]),
        "run_seed": int(run["run_seed"]),
    }
    metadata_run = metadata.get("run", {})
    metadata_expected = {
        "scene_id": expected["scene_id"],
        "condition": expected["condition"],
        "seed": expected["run_seed"],
    }
    return (
        record.get("status") == "complete"
        and all(record.get(key) == value for key, value in expected.items())
        and all(
            metadata_run.get(key) == value
            for key, value in metadata_expected.items()
        )
    )


def _finalize_locked_run(
    paths: Mapping[str, Path],
    run: Mapping[str, Any],
    runtime_seconds: float,
    attempt_number: int,
    first_attempt_failed: bool,
) -> None:
    output = load_json(paths["staged_output_json"])
    metadata = load_json(paths["staged_metadata_json"])
    if not isinstance(output.get("point_labels"), list) or not isinstance(
        output.get("instances"), dict
    ):
        raise ValueError(f"{run['run_id']}: invalid postprocess output")
    if metadata.get("kind") != "saga_instance_metadata":
        raise ValueError(f"{run['run_id']}: invalid postprocess metadata")
    metadata.pop("content_sha256", None)
    metadata_run = dict(metadata.get("run", {}))
    for key in (
        "output_json_sha256",
        "category_priors_sha256",
        "prior_mapping_config_sha256",
    ):
        metadata_run.pop(key, None)
    metadata_run.update(
        {
            "scene_id": str(run["scene_id"]),
            "physical_scene_id": str(run["physical_scene_id"]),
            "condition": str(run["condition"]),
            "seed": int(run["run_seed"]),
            "output_json": str(paths["output_json"]),
        }
    )
    metadata["run"] = metadata_run
    os.replace(paths["staged_output_json"], paths["output_json"])
    write_json(paths["metadata_json"], metadata)
    paths["staged_metadata_json"].unlink(missing_ok=True)
    write_json(
        paths["run_record"],
        {
            "schema_version": "1.0",
            "kind": "locked_run",
            "run_id": str(run["run_id"]),
            "scene_id": str(run["scene_id"]),
            "physical_scene_id": str(run["physical_scene_id"]),
            "condition": str(run["condition"]),
            "run_seed": int(run["run_seed"]),
            "runtime_seconds": float(runtime_seconds),
            "attempts": int(attempt_number),
            "first_attempt_failed": bool(first_attempt_failed),
            "recovered": bool(first_attempt_failed),
            "status": "complete",
        },
    )


def _evict_scene_cache(output_root: Path, scene_id: str) -> bool:
    cache_root = (output_root / ".cache" / "max_contributors").resolve()
    target = (cache_root / scene_id).resolve()
    if target.parent != cache_root:
        raise ValueError(f"Unsafe scene cache path: {target}")
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _deployed_code_commit(pipeline_repo: Path) -> str:
    """Prefer a real checkout identity; use a marker only for exported code."""
    completed = subprocess.run(
        ["git", "-C", str(pipeline_repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    commit_marker = pipeline_repo / ".category_priors_commit"
    return (
        commit_marker.read_text(encoding="utf-8").strip()
        if commit_marker.is_file()
        else ""
    )


def execute_locked_plan(
    plan_path: str | Path,
    scene_manifest_path: str | Path,
    output_root: str | Path,
    progress_path: str | Path,
    pipeline_path: str | Path,
    priors_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    """Run a frozen locked plan with field-based resume and one failed-run retry."""
    plan = load_json(plan_path)
    validate_locked_plan(plan)
    all_runs = expand_locked_runs(plan)
    runs = list(all_runs)
    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive")
        runs = runs[:max_runs]
    scenes = load_scene_runtime_manifest(scene_manifest_path)
    planned_scene_ids = {str(item["scene_id"]) for item in plan["scenes"]}
    if not planned_scene_ids.issubset(scenes):
        missing = sorted(planned_scene_ids - set(scenes))
        raise ValueError(f"Locked runtime manifest is missing scenes: {missing}")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = Path(progress_path).resolve()
    plan_base = Path(plan_path).resolve().parent

    def plan_input(name: str) -> Path:
        path = Path(str(plan["inputs"][name]))
        return path.resolve() if path.is_absolute() else (plan_base / path).resolve()

    planned_priors = plan_input("category_priors")
    planned_mapping = plan_input("prior_mapping")
    if priors_path is not None and Path(priors_path).resolve() != planned_priors:
        raise ValueError("Locked priors path must match locked_plan.json")
    if mapping_path is not None and Path(mapping_path).resolve() != planned_mapping:
        raise ValueError("Locked mapping path must match locked_plan.json")
    priors_path = planned_priors
    mapping_path = planned_mapping
    if summarize_priors(load_json(priors_path)) != plan["priors"]:
        raise ValueError("Locked category priors differ from the frozen plan")
    if summarize_mapping(load_json(mapping_path)) != plan["parameters"]:
        raise ValueError("Locked prior mapping differs from the frozen plan")
    if not dry_run:
        pipeline_repo = Path(pipeline_path).resolve().parent
        deployed_commit = _deployed_code_commit(pipeline_repo)
        if deployed_commit != str(plan["code_commit"]):
            raise ValueError("Deployed pipeline commit differs from the frozen plan")
    selected_run_ids = {str(run["run_id"]) for run in runs}
    records: dict[str, dict[str, Any]] = {}
    if progress_path.is_file():
        previous = load_json(progress_path)
        records = {
            str(item["run_id"]): dict(item)
            for item in previous.get("runs", [])
            if str(item.get("run_id", "")) in selected_run_ids
        }

    expected_by_scene: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        expected_by_scene.setdefault(str(run["scene_id"]), []).append(run)

    expected_ids_by_scene = {
        scene_id: {str(run["run_id"]) for run in scene_runs}
        for scene_id, scene_runs in expected_by_scene.items()
    }
    complete_run_ids: set[str] = set()

    pending = list(runs)
    for attempt in range(2):
        failed: list[dict[str, Any]] = []
        for run in pending:
            scene_id = str(run["scene_id"])
            command, paths = build_postprocess_command(
                pipeline_path,
                run,
                scenes[scene_id],
                output_root,
                priors_path,
                mapping_path,
                staging=True,
            )
            paths["run_dir"].mkdir(parents=True, exist_ok=True)
            if _completed_locked_run(paths, run):
                record = dict(load_json(paths["run_record"]))
                record["status"] = "skipped_complete"
                previous_record = records.get(str(run["run_id"]), {})
                for key in ("attempts", "first_attempt_failed", "recovered"):
                    if key in previous_record:
                        record[key] = previous_record[key]
            elif dry_run:
                record = {
                    "run_id": run["run_id"],
                    "scene_id": scene_id,
                    "condition": run["condition"],
                    "run_seed": run["run_seed"],
                    "status": "planned",
                    "command": command,
                }
            else:
                # run.json is the commit marker.  Invalidate an old marker
                # before replacing output/metadata so an interrupted rerun
                # cannot be mistaken for a complete transaction.
                paths["run_record"].unlink(missing_ok=True)
                paths["staged_output_json"].unlink(missing_ok=True)
                paths["staged_metadata_json"].unlink(missing_ok=True)
                started = time.perf_counter()
                attempt_log = paths["run_dir"] / f"postprocess-attempt-{attempt + 1}.log"
                with attempt_log.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        cwd=Path(pipeline_path).resolve().parent,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                runtime_seconds = time.perf_counter() - started
                record = {
                    "run_id": run["run_id"],
                    "scene_id": scene_id,
                    "physical_scene_id": run["physical_scene_id"],
                    "condition": run["condition"],
                    "run_seed": run["run_seed"],
                    "runtime_seconds": runtime_seconds,
                    "return_code": completed.returncode,
                    "log": str(attempt_log),
                    "attempts": attempt + 1,
                    "first_attempt_failed": bool(
                        records.get(str(run["run_id"]), {}).get(
                            "first_attempt_failed", False
                        )
                    ),
                    "status": "failed",
                }
                if completed.returncode == 0:
                    try:
                        first_attempt_failed = bool(
                            attempt > 0
                            or records.get(str(run["run_id"]), {}).get(
                                "first_attempt_failed", False
                            )
                        )
                        _finalize_locked_run(
                            paths,
                            run,
                            runtime_seconds,
                            attempt + 1,
                            first_attempt_failed,
                        )
                        record["status"] = "complete"
                        record["first_attempt_failed"] = first_attempt_failed
                    except (OSError, ValueError, TypeError) as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                if record["status"] != "complete":
                    if attempt == 0:
                        record["first_attempt_failed"] = True
                    failed.append(dict(run))
                elif record["first_attempt_failed"]:
                    record["recovered"] = True
            records[str(run["run_id"])] = record
            if record.get("status") in {"complete", "skipped_complete"}:
                complete_run_ids.add(str(run["run_id"]))
            progress = {
                "schema_version": "1.0",
                "kind": "locked_progress",
                "split": "val-locked",
                "planned": len(runs),
                "complete": sum(
                    item.get("status") in {"complete", "skipped_complete"}
                    for item in records.values()
                ),
                "failed": sum(item.get("status") == "failed" for item in records.values()),
                "attempt": attempt + 1,
                "runs": [records[key] for key in sorted(records)],
            }
            write_json(progress_path, progress)
            if (
                not dry_run
                and expected_ids_by_scene[scene_id].issubset(complete_run_ids)
            ):
                _evict_scene_cache(output_root, scene_id)
        if dry_run or not failed:
            break
        pending = failed

    result = load_json(progress_path)
    if not dry_run and result["complete"] != len(runs):
        failures = [
            item["run_id"] for item in result["runs"] if item.get("status") == "failed"
        ]
        raise RuntimeError(f"Locked execution incomplete; failures={failures}")
    return result


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
