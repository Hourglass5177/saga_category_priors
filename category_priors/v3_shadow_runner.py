from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest


def v3_shadow_run_paths(
    output_root: str | Path, scene_id: str, seed: int
) -> dict[str, Path]:
    run_dir = Path(output_root).resolve() / scene_id / f"seed-{int(seed)}"
    return {
        "run_dir": run_dir,
        "output": run_dir / "output.json",
        "pending_output": run_dir / "output.pending.json",
        "metadata": run_dir / "output.metadata.json",
        "pending_metadata": run_dir / "output.pending.metadata.json",
        "runner": run_dir / "runner.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
        "exact_json": run_dir / "shadow-exact.json",
        "exact_labels": run_dir / "branch-labels-exact.npz",
        "exclusive_json": run_dir / "shadow-exclusive.json",
        "exclusive_labels": run_dir / "branch-labels-exclusive.npz",
    }


def build_v3_shadow_command(
    pipeline: str | Path,
    scene: Mapping[str, Any],
    output_root: str | Path,
    scene_id: str,
    seed: int,
    git_commit: str,
) -> tuple[list[str], dict[str, Path]]:
    paths = v3_shadow_run_paths(output_root, scene_id, seed)
    if not scene.get("python_bin"):
        raise ValueError(f"{scene_id}: runtime manifest must define python_bin")
    command = [
        "bash", str(Path(pipeline).resolve()),
        "--stage", "postprocess",
        "--base-path", str(scene["base_path"]),
        "--python", str(scene["python_bin"]),
        "--json-path", str(paths["pending_output"]),
        "--prior-metadata-path", str(paths["pending_metadata"]),
        "--progress-path", str(paths["progress"]),
        "--scene-scale-m-per-unit", str(float(scene["scene_scale_m_per_unit"])),
        "--seed", str(int(seed)),
        "--teacher-prior-mode", "original",
        "--minimal-metadata",
        "--v3-shadow-mode", "both",
        "--v3-shadow-output", str(paths["run_dir"] / "shadow-{mode}.json"),
        "--v3-branch-labels-output", str(paths["run_dir"] / "branch-labels-{mode}.npz"),
        "--v3-shadow-git-commit", str(git_commit),
        "--v3-shadow-scene-id", scene_id,
    ]
    return command, paths


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("point_labels"), list) and isinstance(
        payload.get("instances"), Mapping
    )


def _complete(paths: Mapping[str, Path]) -> bool:
    if not _valid_output(paths["output"]):
        return False
    for mode in ("exact", "exclusive"):
        try:
            payload = load_json(paths[f"{mode}_json"])
        except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if payload.get("kind") != "v3_shadow_capture" or payload.get("mode") != mode:
            return False
        if not paths[f"{mode}_labels"].is_file():
            return False
    return True


def execute_v3_shadow_runs(
    *,
    scene_manifest: str | Path,
    output_root: str | Path,
    pipeline: str | Path,
    git_commit: str,
    scene_ids: Sequence[str],
    seeds: Sequence[int] = (42,),
    resume: bool = True,
    continue_on_error: bool = False,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    selected_scenes = [str(value) for value in scene_ids]
    if not selected_scenes or len(selected_scenes) != len(set(selected_scenes)):
        raise ValueError("scene_ids must be nonempty and unique")
    missing = sorted(set(selected_scenes) - set(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    selected_seeds = [int(value) for value in seeds]
    if not selected_seeds or len(selected_seeds) != len(set(selected_seeds)):
        raise ValueError("seeds must be nonempty and unique")
    runs = [(scene_id, seed) for scene_id in selected_scenes for seed in selected_seeds]
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        runs = runs[:max_runs]
    records: list[dict[str, Any]] = []
    for scene_id, seed in runs:
        command, paths = build_v3_shadow_command(
            pipeline, scenes[scene_id], output_root, scene_id, seed, git_commit
        )
        record = {"scene_id": scene_id, "seed": seed, "run_dir": str(paths["run_dir"])}
        if resume and _complete(paths):
            record["status"] = "skipped_complete"
            records.append(record)
            continue
        if dry_run:
            record.update({"status": "planned", "command": command})
            records.append(record)
            continue
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        runtime = time.perf_counter() - started
        if result.returncode == 0 and _valid_output(paths["pending_output"]):
            paths["pending_output"].replace(paths["output"])
            if paths["pending_metadata"].is_file():
                paths["pending_metadata"].replace(paths["metadata"])
        status = "complete" if result.returncode == 0 and _complete(paths) else "failed"
        runner_payload = {
            "kind": "v3_shadow_run",
            "schema_version": "1.0",
            "git_commit": git_commit,
            "scene_id": scene_id,
            "seed": seed,
            "status": status,
            "runtime_seconds": runtime,
            "return_code": result.returncode,
            "command": command,
        }
        write_json(paths["runner"], runner_payload)
        record.update({"status": status, "runtime_seconds": runtime})
        records.append(record)
        if status != "complete" and not continue_on_error:
            raise RuntimeError(f"V3 shadow failed for {scene_id} seed {seed}; see {paths['log']}")
    return {
        "kind": "v3_shadow_execution",
        "git_commit": git_commit,
        "total": len(records),
        "complete": sum(row["status"] in {"complete", "skipped_complete"} for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "runs": records,
    }
