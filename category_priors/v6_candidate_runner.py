from __future__ import annotations

"""Small single-process runner for immutable V6 affinity proposal banks."""

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest


def v6_candidate_run_paths(
    output_root: str | Path, scene_id: str, seed: int,
) -> dict[str, Path]:
    run_dir = Path(output_root).resolve() / str(scene_id) / f"seed-{int(seed)}"
    return {
        "run_dir": run_dir,
        "output": run_dir / "output.json",
        "pending_output": run_dir / "output.pending.json",
        "diagnostics": run_dir / "diagnostics.json",
        "pending_diagnostics": run_dir / "diagnostics.pending.json",
        "proposals": run_dir / "v6_proposals.json",
        "proposal_labels": run_dir / "v6_proposal_labels.npz",
        "runner": run_dir / "runner.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
    }


def build_v6_candidate_command(
    *, pipeline: str | Path, scene: Mapping[str, Any], output_root: str | Path,
    scene_id: str, seed: int, git_commit: str,
) -> tuple[list[str], dict[str, Path]]:
    paths = v6_candidate_run_paths(output_root, scene_id, seed)
    command = [
        "bash", str(Path(pipeline).resolve()), "--stage", "postprocess",
        "--base-path", str(scene["base_path"]), "--python", str(scene["python_bin"]),
        "--json-path", str(paths["pending_output"]),
        "--prior-metadata-path", str(paths["pending_diagnostics"]),
        "--progress-path", str(paths["progress"]),
        "--scene-scale-m-per-unit", str(float(scene["scene_scale_m_per_unit"])),
        "--seed", str(int(seed)), "--teacher-prior-mode", "original",
        "--minimal-metadata", "--v6-candidate-mode", "affinity-first",
        "--v6-candidate-output", str(paths["proposals"]),
        "--v6-candidate-labels-output", str(paths["proposal_labels"]),
        "--v6-git-commit", str(git_commit), "--v6-scene-id", str(scene_id),
    ]
    return command, paths


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("point_labels"), list) and isinstance(payload.get("instances"), Mapping)


def _complete(paths: Mapping[str, Path], command: Sequence[str] | None = None) -> bool:
    if not _valid_output(paths["output"]) or not paths["diagnostics"].is_file() or not paths["proposal_labels"].is_file():
        return False
    try:
        proposal = load_json(paths["proposals"])
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if proposal.get("kind") != "v6_affinity_proposal_bank":
        return False
    if command is not None:
        try:
            runner = load_json(paths["runner"])
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if runner.get("kind") != "v6_candidate_run" or runner.get("command") != list(command):
            return False
    return True


def execute_v6_candidate_runs(
    *, scene_manifest: str | Path, output_root: str | Path, pipeline: str | Path,
    git_commit: str, scene_ids: Sequence[str], seeds: Sequence[int] = (42,),
    resume: bool = True, continue_on_error: bool = False, dry_run: bool = False,
    max_runs: int | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    runs = [(str(scene_id), int(seed)) for scene_id in scene_ids for seed in seeds]
    missing = sorted({scene_id for scene_id, _ in runs} - set(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    if max_runs is not None:
        runs = runs[:int(max_runs)]
    records: list[dict[str, Any]] = []
    for scene_id, seed in runs:
        command, paths = build_v6_candidate_command(
            pipeline=pipeline, scene=scenes[scene_id], output_root=output_root,
            scene_id=scene_id, seed=seed, git_commit=git_commit,
        )
        record = {"scene_id": scene_id, "seed": seed, "run_dir": str(paths["run_dir"])}
        if resume and _complete(paths, command):
            records.append({**record, "status": "skipped_complete"})
            continue
        if dry_run:
            records.append({**record, "status": "planned", "command": command})
            continue
        paths["run_dir"].mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        runtime = time.perf_counter() - started
        if completed.returncode == 0 and _valid_output(paths["pending_output"]):
            paths["pending_output"].replace(paths["output"])
            if paths["pending_diagnostics"].is_file():
                paths["pending_diagnostics"].replace(paths["diagnostics"])
        status = "complete" if completed.returncode == 0 and _complete(paths) else "failed"
        write_json(paths["runner"], {
            "kind": "v6_candidate_run", "git_commit": str(git_commit),
            "scene_id": scene_id, "seed": seed, "status": status,
            "runtime_seconds": runtime, "return_code": completed.returncode, "command": command,
        })
        if status == "complete" and not _complete(paths, command):
            status = "failed"
            payload = load_json(paths["runner"])
            payload["status"] = status
            write_json(paths["runner"], payload)
        records.append({**record, "status": status, "runtime_seconds": runtime})
        if status == "failed" and not continue_on_error:
            raise RuntimeError(f"V6 candidate run failed: {scene_id}/seed-{seed}")
    return {
        "kind": "v6_candidate_execution", "git_commit": str(git_commit), "total": len(records),
        "complete": sum(row["status"] in {"complete", "skipped_complete"} for row in records),
        "failed": sum(row["status"] == "failed" for row in records), "runs": records,
    }
