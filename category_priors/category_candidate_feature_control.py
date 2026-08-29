from __future__ import annotations

"""Same-source two-scene 10k feature control for candidate formation.

This is intentionally a narrow wrapper around the already validated PMR-3
training primitive.  It preserves every scene input used by the active
candidate bank (30k Gaussian order, cameras, masks, labels, mask scales and
seed) and changes only the feature-training budget to 10,000 iterations.
"""

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_json, write_json
from .pmr3_scale_capacity import (
    FEATURE_SEED,
    FINAL_ITERATION,
    MIN_DISK_FREE_GIB,
    build_training_command,
    materialize_native_snapshot,
    validate_training_trajectory,
)
from .runner import load_scene_runtime_manifest


CONTROL_SCHEMA = "saga-category-candidate-same-source-10k-features-v1"
CONTROL_BANK_IDENTITY_SCHEMA = "saga-category-candidate-10k-bank-input-v1"


def _scene_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str
) -> Path:
    base = Path(str(scene["base_path"])).resolve()
    for key in keys:
        value = scene.get(key)
        if value:
            path = Path(str(value)).expanduser()
            return path.resolve() if path.is_absolute() else (base / path).resolve()
    return (base / default).resolve()


def same_source_training_assets(scene: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve precisely the paths forwarded by the active bank runner."""

    base = Path(str(scene["base_path"])).resolve()
    if scene.get("point_cloud_path") or scene.get("gaussian_ply"):
        point_cloud = _scene_path(
            scene, ("point_cloud_path", "gaussian_ply"), ""
        )
    else:
        point_cloud = (
            base / "output_models/point_cloud/iteration_30000/point_cloud.ply"
        ).resolve()
        fallback = (
            base
            / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
        ).resolve()
        if not point_cloud.is_file() and fallback.is_file():
            point_cloud = fallback
    assets = {
        "images": _scene_path(
            scene, ("images_path",), "fastRecon/dense/sparse/0/images"
        ),
        "sparse": _scene_path(
            scene, ("sparse_path",), "fastRecon/dense/sparse/0"
        ),
        "point_cloud": point_cloud,
        "masks": _scene_path(scene, ("masks_path",), "saga/masks"),
        "labels": _scene_path(
            scene, ("grounded_labels_path", "labels_path"), "saga/labels"
        ),
        "label_features": _scene_path(
            scene, ("label_features_path",), "saga/labels/label_features.pt"
        ),
        "mask_scales": _scene_path(
            scene, ("mask_scales_path",), "saga/mask_scales"
        ),
    }
    for name in ("images", "sparse", "masks", "labels", "mask_scales"):
        if not assets[name].is_dir():
            raise FileNotFoundError(
                f"same-source 10k training {name} not found: {assets[name]}"
            )
    for name in ("point_cloud", "label_features"):
        if not assets[name].is_file():
            raise FileNotFoundError(
                f"same-source 10k training {name} not found: {assets[name]}"
            )
    return assets


def _path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def run_same_source_feature_10k(
    *,
    workspace: Path,
    python_bin: Path,
    runtime_manifest: Path,
    training_root: Path,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    """Train or strictly reuse the registered two-scene 10k trajectories."""

    workspace = Path(workspace).resolve()
    python_bin = Path(python_bin).resolve()
    runtime_manifest = Path(runtime_manifest).resolve()
    training_root = Path(training_root).resolve()
    if not (workspace / "run_pipeline.sh").is_file():
        raise FileNotFoundError(workspace / "run_pipeline.sh")
    if not python_bin.is_file():
        raise FileNotFoundError(python_bin)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    selected = tuple(map(str, scene_ids))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("10k feature control requires unique scene IDs")
    missing = sorted(set(selected).difference(scenes))
    if missing:
        raise ValueError(f"10k runtime manifest is missing scenes: {missing}")

    training_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for scene_id in selected:
        scene = scenes[scene_id]
        assets = same_source_training_assets(scene)
        trajectory = training_root / scene_id
        manifest_path = trajectory / "training_manifest.json"
        reused = False
        validation: Mapping[str, Any] | None = None
        if manifest_path.is_file():
            try:
                manifest = load_json(manifest_path)
                if manifest.get("status") == "training_complete":
                    materialize_native_snapshot(trajectory)
                validation = validate_training_trajectory(
                    trajectory_root=trajectory,
                    training_assets=assets,
                )
                reused = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                validation = None
        if validation is None:
            free_gib = shutil.disk_usage(training_root.parent).free / 1024**3
            if free_gib < MIN_DISK_FREE_GIB:
                raise RuntimeError(
                    f"disk safety floor reached: {free_gib:.2f} GiB "
                    f"< {MIN_DISK_FREE_GIB:.0f} GiB"
                )
            trajectory.mkdir(parents=True, exist_ok=True)
            command = build_training_command(
                workspace=workspace,
                python_bin=python_bin,
                scene_base=Path(str(scene["base_path"])),
                trajectory_root=trajectory,
                scene_assets=assets,
            )
            environment = os.environ.copy()
            environment.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
            environment.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
            environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
            with (trajectory / "training.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=workspace,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{scene_id}: same-source 10k training exited "
                    f"{completed.returncode}; inspect {trajectory / 'training.log'}"
                )
            materialize_native_snapshot(trajectory)
            validation = validate_training_trajectory(
                trajectory_root=trajectory,
                training_assets=assets,
            )
        results.append(
            {
                "scene_id": scene_id,
                "reused": reused,
                "input_identity": {
                    name: _path_identity(path) for name, path in assets.items()
                },
                **dict(validation),
            }
        )

    payload = {
        "schema": CONTROL_SCHEMA,
        "status": "complete",
        "scene_ids": list(selected),
        "feature_seed": int(FEATURE_SEED),
        "feature_iterations": int(FINAL_ITERATION),
        "only_training_budget_changed": True,
        "runtime_manifest": str(runtime_manifest),
        "scenes": results,
    }
    write_json(training_root / "same_source_10k_manifest.json", payload)
    return payload


def materialize_feature_runtime_manifest(
    *,
    source_manifest: Path,
    training_payload: Mapping[str, Any],
    output: Path,
) -> Path:
    """Bind the candidate worker to the validated 10k feature/gate pair."""

    if (
        training_payload.get("schema") != CONTROL_SCHEMA
        or training_payload.get("status") != "complete"
        or int(training_payload.get("feature_iterations", -1)) != FINAL_ITERATION
    ):
        raise ValueError("invalid same-source 10k training payload")
    by_scene = {
        str(row["scene_id"]): row for row in training_payload.get("scenes", ())
    }
    payload = load_json(source_manifest)
    if payload.get("kind") != "scene_runtime_manifest":
        raise ValueError("source runtime manifest has the wrong kind")
    rows = payload.get("scenes")
    if not isinstance(rows, list):
        raise TypeError("source runtime manifest scenes must be a list")
    rewritten = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        scene_id = str(row["scene_id"])
        if scene_id in by_scene:
            checkpoints = by_scene[scene_id].get("checkpoints", {})
            checkpoint = checkpoints.get("10k") if isinstance(checkpoints, Mapping) else None
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"{scene_id}: validated 10k checkpoint is missing")
            feature = Path(str(checkpoint["feature_ply"])).resolve()
            gate = Path(str(checkpoint["scale_gate"])).resolve()
            if not feature.is_file() or not gate.is_file():
                raise FileNotFoundError(f"{scene_id}: validated 10k assets disappeared")
            row["contrastive_feature_point_cloud_path"] = str(feature)
            row["scale_gate_path"] = str(gate)
            row["feature_training_iterations"] = FINAL_ITERATION
            row["feature_training_seed"] = FEATURE_SEED
            seen.add(scene_id)
        rewritten.append(row)
    missing = sorted(set(by_scene).difference(seen))
    if missing:
        raise ValueError(f"source runtime manifest omitted 10k scenes: {missing}")
    derived = {**dict(payload), "scenes": rewritten}
    output = Path(output).resolve()
    write_json(output, derived)
    # Parse it through the production loader before any candidate process sees it.
    load_scene_runtime_manifest(output)
    return output


def bind_control_candidate_root(
    *,
    runtime_manifest: Path,
    control_root: Path,
    scene_ids: Sequence[str],
    sample_cap: int,
    seed: int,
) -> Path:
    """Prevent a completed bank from being reused with different 10k assets."""

    runtime_manifest = Path(runtime_manifest).resolve()
    control_root = Path(control_root).resolve()
    scenes = load_scene_runtime_manifest(runtime_manifest)
    rows = []
    for scene_id in map(str, scene_ids):
        scene = scenes[scene_id]
        feature = _scene_path(
            scene,
            (
                "contrastive_feature_point_cloud_path",
                "feature_point_cloud_path",
                "feature_ply_path",
            ),
            "saga/contrastive_feature_point_cloud.ply",
        )
        gate = _scene_path(scene, ("scale_gate_path",), "saga/scale_gate.pt")
        if not feature.is_file() or not gate.is_file():
            raise FileNotFoundError(f"{scene_id}: bound 10k feature pair is missing")
        rows.append(
            {
                "scene_id": scene_id,
                "feature": _path_identity(feature),
                "scale_gate": _path_identity(gate),
            }
        )
    expected = {
        "schema": CONTROL_BANK_IDENTITY_SCHEMA,
        "runtime_manifest": str(runtime_manifest),
        "scene_ids": list(map(str, scene_ids)),
        "sample_cap": int(sample_cap),
        "seed": int(seed),
        "feature_assets": rows,
    }
    path = control_root / "feature_input_identity.json"
    if path.is_file():
        if load_json(path) != expected:
            raise ValueError(
                "10k candidate bank root is bound to different feature inputs"
            )
        return path
    if (control_root / "bank").is_dir() and any(
        (control_root / "bank").rglob("bank_labels.npz")
    ):
        raise ValueError("unbound 10k candidate banks already exist")
    control_root.mkdir(parents=True, exist_ok=True)
    write_json(path, expected)
    return path


__all__ = [
    "CONTROL_SCHEMA",
    "CONTROL_BANK_IDENTITY_SCHEMA",
    "bind_control_candidate_root",
    "materialize_feature_runtime_manifest",
    "run_same_source_feature_10k",
    "same_source_training_assets",
]
