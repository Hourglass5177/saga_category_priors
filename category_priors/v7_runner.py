from __future__ import annotations

"""Minimal sequential runner for V7 object banks."""

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class RuntimeScene:
    scene_id: str
    base_path: Path
    python_bin: Path
    scene_scale_m_per_unit: float


def bank_is_complete(output_dir: Path) -> bool:
    json_path = output_dir / "object_bank.json"
    npz_path = output_dir / "object_bank.npz"
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        bank = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as arrays:
            count = int(bank["point_count"])
            return all(
                arrays[name].shape == (count,)
                for name in ("core_track_id", "final_track_id", "candidate_labels")
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def load_runtime_scenes(path: Path) -> dict[str, RuntimeScene]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenes", payload)
    if isinstance(rows, dict):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    result: dict[str, RuntimeScene] = {}
    for row in rows:
        scene = RuntimeScene(
            scene_id=str(row["scene_id"]),
            base_path=Path(row["base_path"]),
            python_bin=Path(row["python_bin"]),
            scene_scale_m_per_unit=float(row.get("scene_scale_m_per_unit", 1.0)),
        )
        result[scene.scene_id] = scene
    return result


def run_v7_banks(
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    output_root: Path,
    repo_root: Path,
    *,
    halo: bool,
) -> dict[str, Any]:
    scenes = load_runtime_scenes(runtime_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    started = time.monotonic()
    for scene_id in scene_ids:
        if scene_id not in scenes:
            raise KeyError(f"scene missing from runtime manifest: {scene_id}")
        scene = scenes[scene_id]
        output_dir = output_root / scene_id
        if bank_is_complete(output_dir):
            summary.append({"scene_id": scene_id, "status": "reused"})
            print(f"[{scene_id}] reused", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(scene.python_bin), "-m", "category_priors.v7_worker",
            "--scene-id", scene_id,
            "--base-path", str(scene.base_path),
            "--output-dir", str(output_dir),
            "--scene-scale-m-per-unit", str(scene.scene_scale_m_per_unit),
            "--halo", "on" if halo else "off",
        ]
        print(f"[{scene_id}] starting", flush=True)
        scene_started = time.monotonic()
        with (output_dir / "worker.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode != 0 or not bank_is_complete(output_dir):
            raise RuntimeError(
                f"{scene_id} V7 worker failed with exit {completed.returncode}; "
                f"see {output_dir / 'worker.log'}"
            )
        elapsed = time.monotonic() - scene_started
        summary.append({"scene_id": scene_id, "status": "completed", "seconds": elapsed})
        print(f"[{scene_id}] completed in {elapsed:.1f}s", flush=True)
    result = {
        "schema": "saga-v7-run-summary-v1",
        "halo_enabled": bool(halo),
        "runtime_seconds": float(time.monotonic() - started),
        "scenes": summary,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--halo", choices=("off", "on"), default="off")
    args = parser.parse_args(argv)
    run_v7_banks(
        args.runtime_manifest, args.scenes, args.output_root, args.repo_root,
        halo=args.halo == "on",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
