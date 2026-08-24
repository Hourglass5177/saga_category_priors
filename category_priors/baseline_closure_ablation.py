from __future__ import annotations

"""Contributor repair and structural ablations for the teacher closeout.

This stage deliberately consumes the already trained ``full950/adaptive``
feature assets.  It never reads GT and never changes those assets.  The exact
patched-950 implementation is run first; the current L0 harness is allowed to
drive L1--L3 only after their instance partitions are mechanically equivalent.
"""

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .baseline_closure import (
    CLOSURE_SCENES,
    TAXONOMIES,
    RuntimeScene,
    asset_paths,
    feature_is_complete,
    feature_paths,
    load_runtime_scenes,
    output_is_complete,
    output_paths,
    validate_scene_inputs,
)
from .baseline_closure_runner import (
    EXPECTED_CGROUP_MAX_BYTES,
    StageInvocation,
    _ensure_stage_parents,
    execute_stage,
    read_cgroup_snapshot,
)

STRUCTURAL_LEVELS = ("L0", "L1", "L2", "L3")
FIXED_VARIANT = "full950-contributor-fixed"
HARNESS_VARIANT = "current-causal-harness"
DISABLED_OTHER_CLASS = "__disabled__"

Executor = Callable[[StageInvocation], int]


def _default_executor(invocation: StageInvocation) -> int:
    invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    local_module_roots = (
        invocation.cwd / "submodules/diff-gaussian-rasterization-max-contributor",
        invocation.cwd,
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in local_module_roots]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    with invocation.log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            list(invocation.command),
            cwd=invocation.cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return int(result.returncode)


def _common_args(
    scene: RuntimeScene, closure_root: Path, *, feature_budget: str
) -> list[str]:
    assets = asset_paths(closure_root, scene.scene_id, "bfc18")
    feature_variant = (
        "full950-iterations-cli"
        if feature_budget in {"adaptive-iterations-cli", "10000"}
        else "full950"
    )
    source_budget = (
        "adaptive" if feature_budget == "adaptive-iterations-cli" else feature_budget
    )
    features = feature_paths(
        closure_root, scene.scene_id, feature_variant, source_budget, "bfc18"
    )
    return [
        "--sh_degree",
        "0",
        "--feature_dim",
        "32",
        "--images_path",
        str(scene.images_path),
        "--sparse_path",
        str(scene.sparse_path),
        "--masks_path",
        str(assets.masks),
        "--labels_path",
        str(assets.labels),
        "--label_features_path",
        str(assets.label_features),
        "--mask_scales_path",
        str(assets.mask_scales),
        "--point_cloud_path",
        str(scene.point_cloud_path),
        "--contrastive_feature_point_cloud_path",
        str(features.point_cloud),
        "--scale_gate_path",
        str(features.scale_gate),
    ]


def build_fixed_invocation(
    scene: RuntimeScene,
    closure_root: Path,
    fixed_workspace: Path,
    *,
    condition: str,
    feature_budget: str = "adaptive",
) -> StageInvocation:
    taxonomy = TAXONOMIES["bfc18"]
    target = output_paths(
        closure_root,
        scene.scene_id,
        FIXED_VARIANT,
        feature_budget,
        condition,
        "bfc18",
    )
    if condition == "B0-global":
        other = (DISABLED_OTHER_CLASS,)
    elif condition == "B1-original":
        other = taxonomy.other_classes
    else:
        raise ValueError(f"unknown fixed condition: {condition}")
    command = (
        str(scene.python_bin),
        str(fixed_workspace / "postprocess.py"),
        "--progress_path",
        str(target.progress),
        *_common_args(scene, closure_root, feature_budget=feature_budget),
        "--json_path",
        str(target.output_json),
        "--classes",
        *taxonomy.classes,
        "--selected_classes",
        *taxonomy.selected_classes,
        "--other_classes",
        *other,
    )
    return StageInvocation(
        stage="contributor-fixed",
        scene_id=scene.scene_id,
        variant_id=FIXED_VARIANT,
        budget=feature_budget,
        condition=condition,
        command=command,
        cwd=fixed_workspace,
        log_path=target.root / "postprocess.log",
    )


def build_harness_invocation(
    scene: RuntimeScene,
    closure_root: Path,
    current_workspace: Path,
    *,
    level: str,
    condition: str,
    scene_scale_m_per_unit: float,
) -> StageInvocation:
    if level not in STRUCTURAL_LEVELS:
        raise ValueError(f"unknown structural level: {level}")
    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    taxonomy = TAXONOMIES["bfc18"]
    output_condition = f"{level}-{condition}"
    target = output_paths(
        closure_root,
        scene.scene_id,
        HARNESS_VARIANT,
        "adaptive",
        output_condition,
        "bfc18",
    )
    command = [
        str(scene.python_bin),
        str(current_workspace / "postprocess.py"),
        "--progress_path",
        str(target.progress),
        *_common_args(scene, closure_root, feature_budget="adaptive"),
        "--json_path",
        str(target.output_json),
        "--classes",
        *taxonomy.classes,
        "--selected_classes",
        *taxonomy.selected_classes,
        "--other_classes",
        *taxonomy.other_classes,
        "--teacher-prior-mode",
        "original",
        "--v7-causal-ablation",
        level,
        "--scene_scale_m_per_unit",
        str(float(scene_scale_m_per_unit)),
        "--seed",
        "42",
    ]
    if condition == "B0-global":
        command.append("--disable_other_classes")
    elif condition != "B1-original":
        raise ValueError(f"unknown harness condition: {condition}")
    return StageInvocation(
        stage="causal-ablation",
        scene_id=scene.scene_id,
        variant_id=HARNESS_VARIANT,
        budget="adaptive",
        condition=output_condition,
        command=tuple(command),
        cwd=current_workspace,
        log_path=target.root / "postprocess.log",
    )


def _load_output(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"prediction must be an object: {path}")
    return payload


def canonical_point_partition(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, int] | None, ...]:
    """Canonicalize instance IDs while retaining each instance class."""

    raw_labels = payload.get("point_labels")
    raw_instances = payload.get("instances")
    if not isinstance(raw_labels, list) or not isinstance(raw_instances, Mapping):
        raise TypeError("prediction is missing point_labels/instances")
    members: dict[int, list[int]] = {}
    for index, raw in enumerate(raw_labels):
        value = int(raw)
        if value >= 0:
            members.setdefault(value, []).append(index)
    records: list[tuple[str, tuple[int, ...], int]] = []
    for instance_id, indices in members.items():
        metadata = raw_instances.get(str(instance_id), raw_instances.get(instance_id))
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("class"), str
        ):
            raise TypeError(f"instance {instance_id} has no class metadata")
        records.append((str(metadata["class"]), tuple(indices), instance_id))
    records.sort(key=lambda item: (item[0], item[1][0], len(item[1]), item[1]))
    canonical: list[tuple[str, int] | None] = [None] * len(raw_labels)
    for rank, (class_name, indices, _raw_id) in enumerate(records):
        for index in indices:
            canonical[index] = (class_name, rank)
    return tuple(canonical)


def compare_partitions(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = canonical_point_partition(_load_output(left_path))
    right = canonical_point_partition(_load_output(right_path))
    if len(left) != len(right):
        raise ValueError("prediction point counts differ")
    changed = sum(a != b for a, b in zip(left, right))
    return {
        "left": str(left_path),
        "right": str(right_path),
        "point_count": len(left),
        "changed_points": changed,
        "changed_fraction": changed / max(len(left), 1),
        "equivalent": changed == 0,
    }


def _scene_scales(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: Any = (
        payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    )
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    if not isinstance(rows, list):
        raise TypeError("runtime manifest must contain scenes")
    return {
        str(row["scene_id"]): float(row.get("scene_scale_m_per_unit", 1.0))
        for row in rows
    }


def _resource_snapshot(closure_root: Path, cgroup_root: Path) -> dict[str, Any]:
    free_gib = shutil.disk_usage(closure_root).free / 1024**3
    if free_gib < 80.0:
        raise RuntimeError(f"less than 80 GiB free: {free_gib:.1f}")
    cgroup = read_cgroup_snapshot(cgroup_root)
    if cgroup["max"] != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(f"expected 90 GiB cgroup; found {cgroup['max']}")
    if cgroup["current"] >= cgroup["max"]:
        raise RuntimeError("memory.current reached memory.max")
    return {"disk_free_gib": free_gib, "cgroup": cgroup}


def run_ablation_closeout(
    *,
    runtime_manifest: Path,
    closure_root: Path,
    fixed_workspace: Path,
    current_workspace: Path,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    executor: Executor = _default_executor,
) -> dict[str, Any]:
    scenes = load_runtime_scenes(runtime_manifest)
    scales = _scene_scales(runtime_manifest)
    taxonomy = TAXONOMIES["bfc18"]
    for required in CLOSURE_SCENES:
        if required not in scenes or required not in scales:
            raise KeyError(f"runtime manifest is missing {required}")
    if not (fixed_workspace / "postprocess.py").is_file():
        raise FileNotFoundError(fixed_workspace / "postprocess.py")
    if not (current_workspace / "postprocess.py").is_file():
        raise FileNotFoundError(current_workspace / "postprocess.py")
    closure_root.mkdir(parents=True, exist_ok=True)
    initial = _resource_snapshot(closure_root, cgroup_root)
    records: list[dict[str, Any]] = []

    for scene_id in CLOSURE_SCENES:
        scene = scenes[scene_id]
        validate_scene_inputs(scene)
        features = feature_paths(closure_root, scene_id, "full950", "adaptive", "bfc18")
        if not feature_is_complete(scene, features):
            raise RuntimeError(f"full950/adaptive feature is incomplete: {scene_id}")
        fixed_budgets = (
            ("adaptive", "adaptive-iterations-cli", "10000")
            if scene_id == "scene0064_01"
            else ("adaptive",)
        )
        for fixed_budget in fixed_budgets:
            feature_variant = (
                "full950-iterations-cli"
                if fixed_budget in {"adaptive-iterations-cli", "10000"}
                else "full950"
            )
            source_budget = (
                "adaptive"
                if fixed_budget == "adaptive-iterations-cli"
                else fixed_budget
            )
            fixed_features = feature_paths(
                closure_root,
                scene_id,
                feature_variant,
                source_budget,
                "bfc18",
            )
            if not feature_is_complete(scene, fixed_features):
                raise RuntimeError(
                    f"{feature_variant}/{source_budget} feature is incomplete: {scene_id}"
                )
            for condition in ("B0-global", "B1-original"):
                invocation = build_fixed_invocation(
                    scene,
                    closure_root,
                    fixed_workspace,
                    condition=condition,
                    feature_budget=fixed_budget,
                )
                _ensure_stage_parents(invocation)
                target = output_paths(
                    closure_root,
                    scene_id,
                    FIXED_VARIANT,
                    fixed_budget,
                    condition,
                    "bfc18",
                )
                result = execute_stage(
                    invocation,
                    is_complete=lambda s=scene, p=target: output_is_complete(
                        s, p, taxonomy
                    ),
                    executor=executor,
                )
                records.append(
                    {
                        **result,
                        "scene_id": scene_id,
                        "budget": fixed_budget,
                        "condition": condition,
                    }
                )

        for condition in ("B0-global", "B1-original"):
            invocation = build_harness_invocation(
                scene,
                closure_root,
                current_workspace,
                level="L0",
                condition=condition,
                scene_scale_m_per_unit=scales[scene_id],
            )
            _ensure_stage_parents(invocation)
            target = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"L0-{condition}",
                "bfc18",
            )
            result = execute_stage(
                invocation,
                is_complete=lambda s=scene, p=target: output_is_complete(
                    s, p, taxonomy
                ),
                executor=executor,
            )
            records.append(
                {**result, "scene_id": scene_id, "condition": f"L0-{condition}"}
            )

    parity: list[dict[str, Any]] = []
    for scene_id in CLOSURE_SCENES:
        for condition in ("B0-global", "B1-original"):
            fixed = output_paths(
                closure_root, scene_id, FIXED_VARIANT, "adaptive", condition, "bfc18"
            )
            harness = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"L0-{condition}",
                "bfc18",
            )
            comparison = compare_partitions(fixed.output_json, harness.output_json)
            parity.append({**comparison, "scene_id": scene_id, "condition": condition})
    if not all(row["equivalent"] for row in parity):
        summary = {
            "schema": "saga-teacher-baseline-structural-v1",
            "status": "stopped-current-harness-parity-failed",
            "initial_resources": initial,
            "final_resources": _resource_snapshot(closure_root, cgroup_root),
            "parity": parity,
            "records": records,
        }
        (closure_root / "structural_run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    for scene_id in CLOSURE_SCENES:
        scene = scenes[scene_id]
        for level in ("L1", "L2", "L3"):
            invocation = build_harness_invocation(
                scene,
                closure_root,
                current_workspace,
                level=level,
                condition="B1-original",
                scene_scale_m_per_unit=scales[scene_id],
            )
            _ensure_stage_parents(invocation)
            target = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"{level}-B1-original",
                "bfc18",
            )
            result = execute_stage(
                invocation,
                is_complete=lambda s=scene, p=target: output_is_complete(
                    s, p, taxonomy
                ),
                executor=executor,
            )
            records.append({**result, "scene_id": scene_id, "condition": level})

    summary = {
        "schema": "saga-teacher-baseline-structural-v1",
        "status": "completed",
        "initial_resources": initial,
        "final_resources": _resource_snapshot(closure_root, cgroup_root),
        "parity": parity,
        "records": records,
    }
    (closure_root / "structural_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--fixed-workspace", type=Path, required=True)
    parser.add_argument("--current-workspace", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    result = run_ablation_closeout(
        runtime_manifest=args.runtime_manifest,
        closure_root=args.closure_root,
        fixed_workspace=args.fixed_workspace,
        current_workspace=args.current_workspace,
        cgroup_root=args.cgroup_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
