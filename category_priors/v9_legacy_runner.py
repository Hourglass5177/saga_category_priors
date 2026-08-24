from __future__ import annotations

"""Corrected legacy B0/B1 runs over frozen V9 10k feature assets.

This adapter exists solely for the registered ``F10k-B0/F10k-B1`` input
comparison.  It invokes the current legacy postprocessor directly, in a fixed
order, and never trains features, downloads assets, reads ground truth, or
shares outputs with an historical run.
"""

import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .prediction_contract import normalize_prediction
from .runner import load_scene_runtime_manifest
from .v9_feature_training import (
    V9_FEATURE_ITERATIONS,
    V9_FEATURE_SCHEMA,
    V9_FEATURE_SEED,
    V9FeaturePaths,
    v9_feature_training_paths,
)


V9_LEGACY_SCHEMA = "saga-v9-f10k-legacy-v1"
V9_LEGACY_CONDITIONS = ("F10k-B0", "F10k-B1")
EXPECTED_CGROUP_MAX_BYTES = 90 * 1024**3

CLASSES_32 = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)
SELECTED_CLASSES_28 = tuple(
    name for name in CLASSES_32 if name not in {"wall", "floor", "ceiling", "person"}
)
OTHER_CLASSES_8 = (
    "switch", "socket", "book", "remote", "key", "cup", "vase", "phone",
)

TRACE_ARRAYS = (
    "global_sample_core",
    "global_full_assignment",
    "other_class_candidates",
    "branch_class_before_merge",
    "merged_partition",
    "post_global_knn",
    "post_filter",
    "post_attach",
    "final_internal_labels",
    "exported_prediction",
)


@dataclass(frozen=True)
class V9LegacyPaths:
    root: Path
    output: Path
    diagnostics: Path
    stage_trace: Path
    stage_trace_metadata: Path
    progress: Path
    log: Path
    record: Path


@dataclass(frozen=True)
class V9LegacyInvocation:
    scene_id: str
    condition: str
    command: tuple[str, ...]
    cwd: Path
    paths: V9LegacyPaths
    identity: Mapping[str, Any]


Executor = Callable[[V9LegacyInvocation], int]


def v9_legacy_paths(
    output_root: str | Path, condition: str, scene_id: str
) -> V9LegacyPaths:
    if condition not in V9_LEGACY_CONDITIONS:
        raise ValueError(f"unknown V9 legacy condition: {condition}")
    if not scene_id or Path(scene_id).name != scene_id:
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    root = Path(output_root).resolve() / condition / scene_id / f"seed-{V9_FEATURE_SEED}"
    trace = root / "stage_trace.npz"
    return V9LegacyPaths(
        root=root,
        output=root / "output.json",
        diagnostics=root / "diagnostics.json",
        stage_trace=trace,
        stage_trace_metadata=trace.with_suffix(".json"),
        progress=root / "progress.txt",
        log=root / "postprocess.log",
        record=root / "run.json",
    )


def _resolve_scene_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str
) -> Path:
    value: Any = None
    for key in keys:
        if scene.get(key) not in {None, ""}:
            value = scene[key]
            break
    path = Path(str(value if value is not None else default))
    if not path.is_absolute():
        path = Path(str(scene["base_path"])) / path
    return path.resolve()


def _default_point_cloud(base: Path) -> Path:
    standard = base / "output_models/point_cloud/iteration_30000/point_cloud.ply"
    alternate = base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
    return alternate if not standard.is_file() and alternate.is_file() else standard


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_complete_feature_record(
    feature_paths: V9FeaturePaths,
    scene_id: str,
    expected_git_commit: str | None,
) -> dict[str, Any]:
    try:
        record = load_json(feature_paths.record)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{scene_id}: V9 10k feature record is missing or corrupt") from exc
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{scene_id}: V9 10k feature identity is missing")
    if (
        record.get("kind") != "v9_feature_training_run"
        or record.get("status") != "complete"
        or identity.get("schema") != V9_FEATURE_SCHEMA
        or str(identity.get("scene_id")) != scene_id
        or int(identity.get("iterations", -1)) != V9_FEATURE_ITERATIONS
        or int(identity.get("seed", -1)) != V9_FEATURE_SEED
    ):
        raise ValueError(f"{scene_id}: V9 10k feature record is not complete/registered")
    if expected_git_commit is not None and record.get("git_commit") != expected_git_commit:
        raise ValueError(
            f"{scene_id}: feature commit {record.get('git_commit')!r} != "
            f"{expected_git_commit!r}"
        )
    outputs = identity.get("outputs")
    if not isinstance(outputs, Mapping) or (
        Path(str(outputs.get("feature_ply", ""))).resolve()
        != feature_paths.feature_ply.resolve()
        or Path(str(outputs.get("scale_gate", ""))).resolve()
        != feature_paths.scale_gate.resolve()
    ):
        raise ValueError(f"{scene_id}: V9 feature record points at different outputs")
    if not feature_paths.feature_ply.is_file() or feature_paths.feature_ply.stat().st_size < 16:
        raise FileNotFoundError(feature_paths.feature_ply)
    with feature_paths.feature_ply.open("rb") as handle:
        if handle.read(3) != b"ply":
            raise ValueError(f"{scene_id}: invalid V9 feature PLY")
    if not feature_paths.scale_gate.is_file() or feature_paths.scale_gate.stat().st_size == 0:
        raise FileNotFoundError(feature_paths.scale_gate)
    try:
        if int(feature_paths.progress.read_text(encoding="utf-8").strip()) != 100:
            raise ValueError
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{scene_id}: V9 feature progress is incomplete") from exc
    return dict(record)


def _validate_scene_assets(
    scene: Mapping[str, Any], workspace: Path
) -> dict[str, Path]:
    base = Path(str(scene["base_path"])).resolve()
    python_value = scene.get("python_bin")
    if not python_value:
        raise ValueError("scene runtime entry is missing python_bin")
    python_bin = Path(str(python_value))
    python_bin = (base / python_bin).resolve() if not python_bin.is_absolute() else python_bin.resolve()
    point_cloud = (
        _resolve_scene_path(scene, ("point_cloud_path",), "")
        if scene.get("point_cloud_path")
        else _default_point_cloud(base).resolve()
    )
    labels = _resolve_scene_path(
        scene, ("grounded_labels_path", "labels_path"), "saga/labels"
    )
    assets = {
        "python": python_bin,
        "postprocess": workspace / "postprocess.py",
        "images": _resolve_scene_path(
            scene, ("images_path",), "fastRecon/dense/sparse/0/images"
        ),
        "sparse": _resolve_scene_path(
            scene, ("sparse_path",), "fastRecon/dense/sparse/0"
        ),
        "point_cloud": point_cloud,
        "masks": _resolve_scene_path(
            scene, ("grounded_masks_path", "masks_path"), "saga/masks"
        ),
        "labels": labels,
        "label_features": _resolve_scene_path(
            scene,
            ("grounded_label_features_path", "label_features_path"),
            str(labels / "label_features.pt"),
        ),
        "mask_scales": _resolve_scene_path(
            scene,
            ("grounded_mask_scales_path", "mask_scales_path"),
            "saga/mask_scales",
        ),
    }
    for name in ("images", "sparse", "masks", "labels", "mask_scales"):
        if not assets[name].is_dir():
            raise FileNotFoundError(f"{name} directory not found: {assets[name]}")
    for name in ("python", "postprocess", "point_cloud", "label_features"):
        if not assets[name].is_file():
            raise FileNotFoundError(f"{name} file not found: {assets[name]}")
    return assets


def build_v9_legacy_invocation(
    *,
    workspace: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    feature_root: str | Path,
    output_root: str | Path,
    condition: str,
    git_commit: str,
    feature_git_commit: str | None = None,
) -> V9LegacyInvocation:
    if condition not in V9_LEGACY_CONDITIONS:
        raise ValueError(f"unknown V9 legacy condition: {condition}")
    if float(scene.get("scene_scale_m_per_unit", 0.0)) <= 0:
        raise ValueError(f"{scene_id}: scene_scale_m_per_unit must be positive")
    workspace_path = Path(workspace).resolve()
    assets = _validate_scene_assets(scene, workspace_path)
    feature_paths = v9_feature_training_paths(feature_root, scene_id)
    feature_record = _load_complete_feature_record(
        feature_paths, scene_id, feature_git_commit
    )
    paths = v9_legacy_paths(output_root, condition, scene_id)
    command = [
        str(assets["python"]),
        str(assets["postprocess"]),
        "--progress_path", str(paths.progress),
        "--stage_trace_path", str(paths.stage_trace),
        "--sh_degree", "0",
        "--feature_dim", "32",
        "--semantic_feature_dim", "32",
        "--images_path", str(assets["images"]),
        "--sparse_path", str(assets["sparse"]),
        "--masks_path", str(assets["masks"]),
        "--labels_path", str(assets["labels"]),
        "--label_features_path", str(assets["label_features"]),
        "--mask_scales_path", str(assets["mask_scales"]),
        "--point_cloud_path", str(assets["point_cloud"]),
        "--contrastive_feature_point_cloud_path", str(feature_paths.feature_ply),
        "--scale_gate_path", str(feature_paths.scale_gate),
        "--json_path", str(paths.output),
        "--prior_metadata_path", str(paths.diagnostics),
        "--classes", *CLASSES_32,
        "--selected_classes", *SELECTED_CLASSES_28,
        "--other_classes", *OTHER_CLASSES_8,
        "--prior_mode", "off",
        "--clustering-mode", "legacy",
        "--teacher-prior-mode", "original",
        "--teacher-evidence-protection", "off",
        "--v7-causal-ablation", "L0",
        "--scene_scale_m_per_unit", str(float(scene["scene_scale_m_per_unit"])),
        "--seed", str(V9_FEATURE_SEED),
        "--minimal_metadata",
    ]
    if condition == "F10k-B0":
        command.append("--disable_other_classes")
    identity = {
        "schema": V9_LEGACY_SCHEMA,
        "git_commit": str(git_commit),
        "scene_id": scene_id,
        "condition": condition,
        "seed": V9_FEATURE_SEED,
        "workspace": str(workspace_path),
        "postprocess": _file_identity(assets["postprocess"]),
        "feature_record": {
            "path": str(feature_paths.record.resolve()),
            "git_commit": feature_record.get("git_commit"),
            "identity": feature_record.get("identity"),
        },
        "feature_ply": _file_identity(feature_paths.feature_ply),
        "scale_gate": _file_identity(feature_paths.scale_gate),
        "label_features": _file_identity(assets["label_features"]),
        "command": command,
    }
    return V9LegacyInvocation(
        scene_id=scene_id,
        condition=condition,
        command=tuple(command),
        cwd=workspace_path,
        paths=paths,
        identity=identity,
    )


def _prediction_contract_is_complete(path: Path) -> tuple[bool, int]:
    try:
        payload = load_json(path)
        raw_labels = payload.get("point_labels")
        instances = payload.get("instances")
        if not isinstance(raw_labels, list) or not raw_labels:
            return False, 0
        labels = np.asarray(raw_labels)
        integer_labels = labels.astype(np.int64)
        if labels.ndim != 1 or not np.array_equal(labels, integer_labels):
            return False, 0
        if not isinstance(instances, Mapping):
            return False, 0
        declared: set[int] = set()
        for raw_id, metadata in instances.items():
            instance_id = int(raw_id)
            if str(instance_id) != str(raw_id) or instance_id < 0:
                return False, 0
            if not isinstance(metadata, Mapping):
                return False, 0
            score = metadata.get("score")
            if (
                not isinstance(metadata.get("class"), str)
                or not metadata["class"]
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not bool(np.any(integer_labels == instance_id))
            ):
                return False, 0
            declared.add(instance_id)
        assigned = set(map(int, np.unique(integer_labels[integer_labels >= 0])))
        expected = set(range(len(declared)))
        audit = payload.get("prediction_contract")
        if (
            assigned != declared
            or declared != expected
            or not isinstance(audit, Mapping)
            or audit.get("schema") != "saga-strict-prediction-contract-v1"
            or int(audit.get("point_count", -1)) != len(integer_labels)
        ):
            return False, 0
        return True, int(len(integer_labels))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False, 0


def _stage_trace_is_complete(paths: V9LegacyPaths, point_count: int) -> bool:
    if not paths.stage_trace.is_file() or not paths.stage_trace_metadata.is_file():
        return False
    try:
        metadata = load_json(paths.stage_trace_metadata)
        if (
            metadata.get("schema") != "saga-v9-legacy-stage-trace-v1"
            or int(metadata.get("point_count", -1)) != point_count
            or metadata.get("level") != "L0"
        ):
            return False
        raw_instances = metadata.get("raw_instances")
        if not isinstance(raw_instances, Mapping):
            return False
        output = load_json(paths.output)
        output_labels = np.asarray(output.get("point_labels"), dtype=np.int64)
        output_instances = output.get("instances")
        if output_labels.shape != (point_count,) or not isinstance(output_instances, Mapping):
            return False
        with np.load(paths.stage_trace, allow_pickle=False) as arrays:
            if not set(TRACE_ARRAYS).issubset(arrays.files) or not all(
                np.asarray(arrays[name]).shape == (point_count,)
                for name in TRACE_ARRAYS
            ):
                return False
            internal = np.asarray(arrays["final_internal_labels"])
            exported = np.asarray(arrays["exported_prediction"])
            if (
                not np.issubdtype(internal.dtype, np.integer)
                or not np.issubdtype(exported.dtype, np.integer)
                or not np.array_equal(exported, output_labels)
            ):
                return False
            projected = normalize_prediction(internal, raw_instances)
            return bool(
                np.array_equal(projected.point_labels, exported)
                and projected.instances == dict(output_instances)
            )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def v9_legacy_run_complete(
    paths: V9LegacyPaths, identity: Mapping[str, Any]
) -> bool:
    try:
        record = load_json(paths.record)
        diagnostics = load_json(paths.diagnostics)
        progress = float(paths.progress.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    valid_output, point_count = _prediction_contract_is_complete(paths.output)
    return (
        record.get("kind") == "v9_f10k_legacy_run"
        and record.get("status") == "complete"
        and record.get("identity") == dict(identity)
        and isinstance(diagnostics, Mapping)
        and progress >= 1.0
        and valid_output
        and _stage_trace_is_complete(paths, point_count)
    )


def read_v9_legacy_resources(
    output_root: str | Path,
    *,
    cgroup_root: str | Path | None = "/sys/fs/cgroup",
    disk_floor_gib: float = 80.0,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    free_gib = float(shutil.disk_usage(root).free / 1024**3)
    if free_gib < disk_floor_gib:
        raise RuntimeError(
            f"V9 legacy requires at least {disk_floor_gib:.1f} GiB free; "
            f"found {free_gib:.1f} GiB"
        )
    cgroup: dict[str, Any] | None = None
    if cgroup_root is not None:
        cgroup_path = Path(cgroup_root)
        maximum_text = (cgroup_path / "memory.max").read_text(encoding="utf-8").strip()
        maximum = None if maximum_text == "max" else int(maximum_text)
        current = int((cgroup_path / "memory.current").read_text(encoding="utf-8").strip())
        events = {
            key: int(value)
            for key, value in (
                line.split()
                for line in (cgroup_path / "memory.events").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        }
        if maximum != EXPECTED_CGROUP_MAX_BYTES:
            raise RuntimeError(f"expected 90 GiB cgroup, found memory.max={maximum}")
        if current >= maximum:
            raise RuntimeError("cgroup memory.current has reached memory.max")
        cgroup = {"current": current, "max": maximum, "events": events}
    return {"disk_free_gib": free_gib, "cgroup": cgroup}


def _default_executor(invocation: V9LegacyInvocation) -> int:
    environment = os.environ.copy()
    module_root = invocation.cwd / "submodules/diff-gaussian-rasterization-max-contributor"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(module_root), str(invocation.cwd)]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    environment["PYTHONHASHSEED"] = str(V9_FEATURE_SEED)
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    with invocation.paths.log.open("a", encoding="utf-8", newline="\n") as log:
        result = subprocess.run(
            invocation.command,
            cwd=invocation.cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def execute_v9_legacy_runs(
    *,
    scene_manifest: str | Path,
    feature_root: str | Path,
    output_root: str | Path,
    workspace: str | Path,
    git_commit: str,
    scene_ids: Sequence[str],
    feature_git_commit: str | None = None,
    conditions: Sequence[str] = V9_LEGACY_CONDITIONS,
    resume: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = False,
    cgroup_root: str | Path | None = "/sys/fs/cgroup",
    disk_floor_gib: float = 80.0,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Run scenes sequentially, with B0 before B1 for every scene."""

    selected_scenes = tuple(map(str, scene_ids))
    selected_conditions = tuple(map(str, conditions))
    if len(selected_scenes) != len(set(selected_scenes)):
        raise ValueError("scene_ids contains duplicates")
    if not selected_scenes:
        raise ValueError("at least one scene is required")
    if len(selected_conditions) != len(set(selected_conditions)):
        raise ValueError("conditions contains duplicates")
    unknown = set(selected_conditions).difference(V9_LEGACY_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown V9 legacy conditions: {sorted(unknown)}")
    ordered_conditions = tuple(
        condition for condition in V9_LEGACY_CONDITIONS if condition in selected_conditions
    )
    scenes = load_scene_runtime_manifest(scene_manifest)
    missing = sorted(set(selected_scenes).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")

    root = Path(output_root).resolve()
    resource_start = read_v9_legacy_resources(
        root, cgroup_root=cgroup_root, disk_floor_gib=disk_floor_gib
    )
    run_executor = executor or _default_executor
    rows: list[dict[str, Any]] = []
    summary_path = root / "execution_summary.json"
    for scene_id in selected_scenes:
        for condition in ordered_conditions:
            invocation = build_v9_legacy_invocation(
                workspace=workspace,
                scene=scenes[scene_id],
                scene_id=scene_id,
                feature_root=feature_root,
                output_root=root,
                condition=condition,
                git_commit=git_commit,
                feature_git_commit=feature_git_commit,
            )
            if resume and v9_legacy_run_complete(invocation.paths, invocation.identity):
                row = {
                    "scene_id": scene_id,
                    "condition": condition,
                    "status": "skipped_complete",
                    "root": str(invocation.paths.root),
                }
                rows.append(row)
                continue
            if dry_run:
                rows.append({
                    "scene_id": scene_id,
                    "condition": condition,
                    "status": "planned",
                    "root": str(invocation.paths.root),
                    "command": list(invocation.command),
                })
                continue

            resources_before = read_v9_legacy_resources(
                root, cgroup_root=cgroup_root, disk_floor_gib=disk_floor_gib
            )
            invocation.paths.root.mkdir(parents=True, exist_ok=True)
            running = {
                "kind": "v9_f10k_legacy_run",
                "schema": V9_LEGACY_SCHEMA,
                "status": "running",
                "identity": dict(invocation.identity),
                "resources_before": resources_before,
            }
            write_json(invocation.paths.record, running)
            started = time.monotonic()
            return_code = run_executor(invocation)
            runtime_seconds = float(time.monotonic() - started)
            # Validate data artifacts before committing the final run record;
            # the normal completeness check intentionally rejects "running".
            valid_output, point_count = _prediction_contract_is_complete(
                invocation.paths.output
            )
            artifacts_complete = (
                return_code == 0
                and valid_output
                and _stage_trace_is_complete(invocation.paths, point_count)
                and invocation.paths.diagnostics.is_file()
            )
            resources_after = read_v9_legacy_resources(
                root, cgroup_root=cgroup_root, disk_floor_gib=disk_floor_gib
            )
            status = "complete" if artifacts_complete else "failed"
            final_record = {
                **running,
                "status": status,
                "return_code": int(return_code),
                "runtime_seconds": runtime_seconds,
                "resources_after": resources_after,
            }
            write_json(invocation.paths.record, final_record)
            if status == "complete" and not v9_legacy_run_complete(
                invocation.paths, invocation.identity
            ):
                status = "failed"
                final_record["status"] = status
                final_record["failure"] = "final completeness validation failed"
                write_json(invocation.paths.record, final_record)
            row = {
                "scene_id": scene_id,
                "condition": condition,
                "status": status,
                "root": str(invocation.paths.root),
                "runtime_seconds": runtime_seconds,
            }
            rows.append(row)
            write_json(summary_path, {
                "kind": "v9_f10k_legacy_execution",
                "schema": V9_LEGACY_SCHEMA,
                "git_commit": str(git_commit),
                "seed": V9_FEATURE_SEED,
                "runs": rows,
            })
            if status == "failed" and not continue_on_error:
                raise RuntimeError(
                    f"V9 legacy {condition}/{scene_id} failed; see "
                    f"{invocation.paths.log}"
                )

    summary = {
        "kind": "v9_f10k_legacy_execution",
        "schema": V9_LEGACY_SCHEMA,
        "git_commit": str(git_commit),
        "seed": V9_FEATURE_SEED,
        "resource_start": resource_start,
        "resource_end": read_v9_legacy_resources(
            root, cgroup_root=cgroup_root, disk_floor_gib=disk_floor_gib
        ),
        "total": len(rows),
        "complete": sum(
            row["status"] in {"complete", "skipped_complete"} for row in rows
        ),
        "failed": sum(row["status"] == "failed" for row in rows),
        "runs": rows,
    }
    write_json(summary_path, summary)
    return summary
