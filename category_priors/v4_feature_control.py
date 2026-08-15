from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import write_json
from .runner import load_scene_runtime_manifest


CONTROL_SCENES = ("scene0011_00", "scene0608_00")


def v4_feature_control_paths(output_root: str | Path, scene_id: str) -> dict[str, Path]:
    root = Path(output_root).resolve() / scene_id
    return {
        "root": root,
        "feature_ply": root / "contrastive_feature_point_cloud_10k.ply",
        "scale_gate": root / "scale_gate_10k.pt",
        "progress": root / "train_progress.txt",
        "log": root / "train_10k.log",
        "record": root / "train_10k.json",
    }


def build_v4_feature_control_command(
    pipeline: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    output_root: str | Path,
) -> tuple[list[str], dict[str, Path]]:
    paths = v4_feature_control_paths(output_root, scene_id)
    return [
        "bash", str(Path(pipeline).resolve()),
        "--stage", "train",
        "--base-path", str(scene["base_path"]),
        "--python", str(scene["python_bin"]),
        "--feature-iterations", "10000",
        "--contrastive-feature-point-cloud-path", str(paths["feature_ply"]),
        "--scale-gate-path", str(paths["scale_gate"]),
        "--progress-path", str(paths["progress"]),
    ], paths


def execute_v4_feature_controls(
    *, scene_manifest: str | Path, output_root: str | Path, pipeline: str | Path,
    git_commit: str, scene_ids: Sequence[str] | None = CONTROL_SCENES,
    resume: bool = True, dry_run: bool = False,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    selected = [str(value) for value in (scene_ids or CONTROL_SCENES)]
    if set(selected) - set(CONTROL_SCENES):
        raise ValueError(f"V4 10k control is restricted to {CONTROL_SCENES}")
    records = []
    for scene_id in selected:
        command, paths = build_v4_feature_control_command(
            pipeline, scenes[scene_id], scene_id, output_root
        )
        if resume and paths["feature_ply"].is_file() and paths["scale_gate"].is_file():
            records.append({"scene_id": scene_id, "status": "skipped_complete"})
            continue
        if dry_run:
            records.append({"scene_id": scene_id, "status": "planned", "command": command})
            continue
        paths["root"].mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with paths["log"].open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        runtime = time.perf_counter() - started
        status = "complete" if (
            result.returncode == 0 and paths["feature_ply"].is_file() and paths["scale_gate"].is_file()
        ) else "failed"
        payload = {
            "kind": "v4_feature_10k_control_run", "git_commit": git_commit,
            "scene_id": scene_id, "iterations": 10000, "status": status,
            "runtime_seconds": runtime, "return_code": result.returncode,
            "feature_ply": str(paths["feature_ply"]), "scale_gate": str(paths["scale_gate"]),
            "command": command,
        }
        write_json(paths["record"], payload)
        records.append(payload)
        if status == "failed":
            raise RuntimeError(f"V4 10k control failed for {scene_id}; see {paths['log']}")
    return {"kind": "v4_feature_10k_control_execution", "git_commit": git_commit, "runs": records}
