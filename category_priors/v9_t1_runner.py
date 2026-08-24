from __future__ import annotations

"""Corrected teacher-legacy B0/B1 runs over the frozen existing features.

``T1-B0`` and ``T1-B1`` are the registered V9 teacher-structure controls.  They
use the scene's already trained affinity/semantic feature PLY and scale gate;
this module never trains, downloads, or substitutes the V9 10k features.  The
only live postprocessor is the corrected current implementation: alpha*T_prev
contributor lifting, the teacher's original ``other_classes`` branch, the
strict prediction contract, and a complete L0 stage trace.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .v9_feature_training import V9_FEATURE_SEED
from .v9_legacy_runner import (
    CLASSES_32,
    OTHER_CLASSES_8,
    SELECTED_CLASSES_28,
    _default_executor,
    _file_identity,
    _prediction_contract_is_complete,
    _resolve_scene_path,
    _stage_trace_is_complete,
    _validate_scene_assets,
    read_v9_legacy_resources,
)


V9_T1_SCHEMA = "saga-v9-t1-legacy-v1"
V9_T1_CONDITIONS = ("T1-B0", "T1-B1")
V9_T1_DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)


@dataclass(frozen=True)
class V9T1Paths:
    root: Path
    output: Path
    diagnostics: Path
    stage_trace: Path
    stage_trace_metadata: Path
    progress: Path
    log: Path
    record: Path


@dataclass(frozen=True)
class V9T1Invocation:
    scene_id: str
    condition: str
    command: tuple[str, ...]
    cwd: Path
    paths: V9T1Paths
    identity: Mapping[str, Any]


Executor = Callable[[V9T1Invocation], int]


def v9_t1_paths(output_root: str | Path, condition: str, scene_id: str) -> V9T1Paths:
    if condition not in V9_T1_CONDITIONS:
        raise ValueError(f"unknown V9 T1 condition: {condition}")
    if not scene_id or Path(scene_id).name != scene_id:
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    root = Path(output_root).resolve() / condition / scene_id / f"seed-{V9_FEATURE_SEED}"
    trace = root / "stage_trace.npz"
    return V9T1Paths(
        root=root,
        output=root / "output.json",
        diagnostics=root / "diagnostics.json",
        stage_trace=trace,
        stage_trace_metadata=trace.with_suffix(".json"),
        progress=root / "progress.txt",
        log=root / "postprocess.log",
        record=root / "run.json",
    )


def _resolve_existing_features(scene: Mapping[str, Any]) -> tuple[Path, Path]:
    """Resolve, but never materialize, the immutable per-scene feature assets."""

    feature = _resolve_scene_path(
        scene,
        (
            "contrastive_feature_point_cloud_path",
            "feature_point_cloud_path",
            "feature_ply_path",
        ),
        "saga/contrastive_feature_point_cloud.ply",
    )
    scale_gate = _resolve_scene_path(
        scene,
        ("scale_gate_path",),
        "saga/scale_gate.pt",
    )
    if not feature.is_file():
        raise FileNotFoundError(f"existing feature PLY not found: {feature}")
    try:
        with feature.open("rb") as handle:
            if handle.read(3) != b"ply":
                raise ValueError(f"invalid existing feature PLY: {feature}")
    except OSError as exc:
        raise ValueError(f"cannot read existing feature PLY: {feature}") from exc
    if not scale_gate.is_file() or scale_gate.stat().st_size == 0:
        raise FileNotFoundError(f"existing scale gate not found: {scale_gate}")
    return feature.resolve(), scale_gate.resolve()


def build_v9_t1_invocation(
    *,
    workspace: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    output_root: str | Path,
    condition: str,
    git_commit: str,
) -> V9T1Invocation:
    if condition not in V9_T1_CONDITIONS:
        raise ValueError(f"unknown V9 T1 condition: {condition}")
    if float(scene.get("scene_scale_m_per_unit", 0.0)) <= 0:
        raise ValueError(f"{scene_id}: scene_scale_m_per_unit must be positive")
    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("git_commit must be non-empty")

    workspace_path = Path(workspace).resolve()
    assets = _validate_scene_assets(scene, workspace_path)
    feature_ply, scale_gate = _resolve_existing_features(scene)
    paths = v9_t1_paths(output_root, condition, scene_id)
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
        "--contrastive_feature_point_cloud_path", str(feature_ply),
        "--scale_gate_path", str(scale_gate),
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
    if condition == "T1-B0":
        command.append("--disable_other_classes")

    identity = {
        "schema": V9_T1_SCHEMA,
        "git_commit": commit,
        "scene_id": scene_id,
        "condition": condition,
        "seed": V9_FEATURE_SEED,
        "workspace": str(workspace_path),
        "input_budget": "existing-scene-feature-2k",
        "contributor_weight": "alpha_times_t_prev",
        "teacher_prior_mode": "original",
        "causal_level": "L0",
        "postprocess": _file_identity(assets["postprocess"]),
        "feature_ply": _file_identity(feature_ply),
        "scale_gate": _file_identity(scale_gate),
        "label_features": _file_identity(assets["label_features"]),
        "command": command,
    }
    return V9T1Invocation(
        scene_id=scene_id,
        condition=condition,
        command=tuple(command),
        cwd=workspace_path,
        paths=paths,
        identity=identity,
    )


def v9_t1_run_complete(paths: V9T1Paths, identity: Mapping[str, Any]) -> bool:
    try:
        record = load_json(paths.record)
        diagnostics = load_json(paths.diagnostics)
        progress = float(paths.progress.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    valid_output, point_count = _prediction_contract_is_complete(paths.output)
    return (
        record.get("kind") == "v9_t1_legacy_run"
        and record.get("status") == "complete"
        and record.get("identity") == dict(identity)
        and isinstance(diagnostics, Mapping)
        and progress >= 1.0
        and valid_output
        and _stage_trace_is_complete(paths, point_count)
    )


def execute_v9_t1_runs(
    *,
    scene_manifest: str | Path,
    output_root: str | Path,
    workspace: str | Path,
    git_commit: str,
    scene_ids: Sequence[str] = V9_T1_DEV8,
    conditions: Sequence[str] = V9_T1_CONDITIONS,
    resume: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = False,
    cgroup_root: str | Path | None = "/sys/fs/cgroup",
    disk_floor_gib: float = 80.0,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Run corrected T1-B0 then T1-B1, scene by scene, with exact resume."""

    selected_scenes = tuple(map(str, scene_ids))
    selected_conditions = tuple(map(str, conditions))
    if not selected_scenes or len(selected_scenes) != len(set(selected_scenes)):
        raise ValueError("scene_ids must be non-empty and unique")
    if len(selected_conditions) != len(set(selected_conditions)):
        raise ValueError("conditions contains duplicates")
    unknown = set(selected_conditions).difference(V9_T1_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown V9 T1 conditions: {sorted(unknown)}")
    ordered_conditions = tuple(
        condition for condition in V9_T1_CONDITIONS if condition in selected_conditions
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
            invocation = build_v9_t1_invocation(
                workspace=workspace,
                scene=scenes[scene_id],
                scene_id=scene_id,
                output_root=root,
                condition=condition,
                git_commit=git_commit,
            )
            if resume and v9_t1_run_complete(invocation.paths, invocation.identity):
                rows.append({
                    "scene_id": scene_id,
                    "condition": condition,
                    "status": "skipped_complete",
                    "root": str(invocation.paths.root),
                })
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
                "kind": "v9_t1_legacy_run",
                "schema": V9_T1_SCHEMA,
                "status": "running",
                "identity": dict(invocation.identity),
                "resources_before": resources_before,
            }
            write_json(invocation.paths.record, running)
            started = time.monotonic()
            return_code = int(run_executor(invocation))
            runtime_seconds = float(time.monotonic() - started)
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
                "return_code": return_code,
                "runtime_seconds": runtime_seconds,
                "resources_after": resources_after,
            }
            write_json(invocation.paths.record, final_record)
            if status == "complete" and not v9_t1_run_complete(
                invocation.paths, invocation.identity
            ):
                status = "failed"
                final_record["status"] = status
                final_record["failure"] = "final completeness validation failed"
                write_json(invocation.paths.record, final_record)
            rows.append({
                "scene_id": scene_id,
                "condition": condition,
                "status": status,
                "root": str(invocation.paths.root),
                "runtime_seconds": runtime_seconds,
            })
            write_json(summary_path, {
                "kind": "v9_t1_legacy_execution",
                "schema": V9_T1_SCHEMA,
                "git_commit": str(git_commit),
                "seed": V9_FEATURE_SEED,
                "runs": rows,
            })
            if status == "failed" and not continue_on_error:
                raise RuntimeError(
                    f"V9 T1 {condition}/{scene_id} failed; see {invocation.paths.log}"
                )

    summary = {
        "kind": "v9_t1_legacy_execution",
        "schema": V9_T1_SCHEMA,
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
