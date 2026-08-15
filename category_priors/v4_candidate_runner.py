from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .v4_candidate import MODES


def v4_candidate_run_paths(
    output_root: str | Path, mode: str, scene_id: str, seed: int
) -> dict[str, Path]:
    if mode not in MODES:
        raise ValueError(f"unsupported V4 candidate mode: {mode}")
    run_dir = Path(output_root).resolve() / mode / scene_id / f"seed-{int(seed)}"
    return {
        "run_dir": run_dir,
        "output": run_dir / "output.json",
        "pending_output": run_dir / "output.pending.json",
        "metadata": run_dir / "output.metadata.json",
        "pending_metadata": run_dir / "output.pending.metadata.json",
        "candidate_json": run_dir / "v4-candidates.json",
        "candidate_labels": run_dir / "v4-candidate-labels.npz",
        "runner": run_dir / "runner.json",
        "progress": run_dir / "progress.txt",
        "log": run_dir / "postprocess.log",
    }


def build_v4_candidate_command(
    *,
    pipeline: str | Path,
    scene: Mapping[str, Any],
    output_root: str | Path,
    mode: str,
    scene_id: str,
    seed: int,
    git_commit: str,
    category_priors: str | Path,
    feature_ply: str | Path | None = None,
    scale_gate: str | Path | None = None,
) -> tuple[list[str], dict[str, Path]]:
    paths = v4_candidate_run_paths(output_root, mode, scene_id, seed)
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
        "--v4-candidate-mode", mode,
        "--category-priors", str(Path(category_priors).resolve()),
        "--v4-candidate-output", str(paths["candidate_json"]),
        "--v4-candidate-labels-output", str(paths["candidate_labels"]),
        "--v4-git-commit", str(git_commit),
        "--v4-scene-id", scene_id,
    ]
    if feature_ply is not None:
        command += ["--contrastive-feature-point-cloud-path", str(Path(feature_ply).resolve())]
    if scale_gate is not None:
        command += ["--scale-gate-path", str(Path(scale_gate).resolve())]
    return command, paths


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload.get("point_labels"), list) and isinstance(payload.get("instances"), Mapping)


def _complete(paths: Mapping[str, Path], mode: str) -> bool:
    if not _valid_output(paths["output"]) or not paths["candidate_labels"].is_file():
        return False
    try:
        payload = load_json(paths["candidate_json"])
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return payload.get("kind") == "v4_candidate_capture" and payload.get("mode") == mode


def execute_v4_candidate_runs(
    *,
    scene_manifest: str | Path,
    output_root: str | Path,
    pipeline: str | Path,
    git_commit: str,
    category_priors: str | Path,
    scene_ids: Sequence[str],
    modes: Sequence[str] = MODES,
    seeds: Sequence[int] = (42,),
    resume: bool = True,
    continue_on_error: bool = False,
    dry_run: bool = False,
    max_runs: int | None = None,
    feature_control_root: str | Path | None = None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    requested_modes = [str(value) for value in modes]
    if not requested_modes or any(mode not in MODES for mode in requested_modes):
        raise ValueError(f"modes must be selected from {MODES}")
    if feature_control_root is not None and requested_modes != ["uniform"]:
        raise ValueError("10k feature-control candidates are restricted to uniform mode")
    runs = [
        (mode, str(scene_id), int(seed))
        for scene_id in scene_ids for mode in requested_modes for seed in seeds
    ]
    missing = sorted({scene_id for _, scene_id, _ in runs} - set(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    if max_runs is not None:
        runs = runs[: int(max_runs)]
    records = []
    for mode, scene_id, seed in runs:
        feature_ply = None
        scale_gate = None
        if feature_control_root is not None:
            from .v4_feature_control import CONTROL_SCENES, v4_feature_control_paths
            if scene_id not in CONTROL_SCENES:
                raise ValueError(f"10k feature control is restricted to {CONTROL_SCENES}")
            assets = v4_feature_control_paths(feature_control_root, scene_id)
            feature_ply = assets["feature_ply"]
            scale_gate = assets["scale_gate"]
        command, paths = build_v4_candidate_command(
            pipeline=pipeline, scene=scenes[scene_id], output_root=output_root,
            mode=mode, scene_id=scene_id, seed=seed, git_commit=git_commit,
            category_priors=category_priors,
            feature_ply=feature_ply, scale_gate=scale_gate,
        )
        record = {"mode": mode, "scene_id": scene_id, "seed": seed, "run_dir": str(paths["run_dir"])}
        if resume and _complete(paths, mode):
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
        status = "complete" if result.returncode == 0 and _complete(paths, mode) else "failed"
        write_json(paths["runner"], {
            "kind": "v4_candidate_run", "git_commit": git_commit,
            "mode": mode, "scene_id": scene_id, "seed": seed,
            "status": status, "runtime_seconds": runtime,
            "return_code": result.returncode, "command": command,
        })
        record.update({"status": status, "runtime_seconds": runtime})
        records.append(record)
        if status == "failed" and not continue_on_error:
            raise RuntimeError(f"V4 candidate run failed: {mode}/{scene_id}/seed-{seed}")
    return {
        "kind": "v4_candidate_execution", "git_commit": git_commit,
        "total": len(records),
        "complete": sum(row["status"] in {"complete", "skipped_complete"} for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "runs": records,
    }
