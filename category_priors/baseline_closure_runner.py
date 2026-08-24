from __future__ import annotations

"""Sequential and resumable runner for the frozen teacher-baseline closeout.

The runner is intentionally mechanical: it prepares no source variant, reads no
ground truth, performs no evaluation, and never downloads or deletes anything.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .baseline_closure import (
    CLOSURE_SCENES,
    REGISTERED_RUNS,
    SOURCE_VARIANTS,
    TAXONOMIES,
    AssetPaths,
    FeaturePaths,
    OutputPaths,
    RunSpec,
    RuntimeScene,
    SourceWorkspace,
    TaxonomySpec,
    assert_isolated_output,
    asset_paths,
    feature_is_complete,
    feature_paths,
    load_runtime_scenes,
    load_source_workspaces,
    masks_are_complete,
    output_is_complete,
    output_paths,
    record_masks_completion,
    scales_are_complete,
    validate_scene_inputs,
    validate_source_workspace,
)

DISABLED_OTHER_CLASS = "__disabled__"
EXPECTED_CGROUP_MAX_BYTES = 90 * 1024**3


@dataclass(frozen=True)
class StageInvocation:
    stage: str
    scene_id: str
    variant_id: str
    budget: str | None
    condition: str | None
    command: tuple[str, ...]
    cwd: Path
    log_path: Path


Executor = Callable[[StageInvocation], int]


def _common_model_args(scene: RuntimeScene, assets: AssetPaths) -> list[str]:
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
    ]


def build_masks_invocation(
    scene: RuntimeScene,
    assets: AssetPaths,
    workspace: SourceWorkspace,
    taxonomy: TaxonomySpec,
    *,
    sam_checkpoint: Path,
    groundingdino_checkpoint: Path,
    groundingdino_config: Path,
) -> StageInvocation:
    command = (
        str(scene.python_bin),
        str(workspace.root / "grounded_SAM_masks.py"),
        "--progress_path",
        str(assets.masks_progress),
        "--images_path",
        str(scene.images_path),
        "--masks_path",
        str(assets.masks),
        "--labels_path",
        str(assets.labels),
        "--label_features_path",
        str(assets.label_features),
        "--sam_checkpoint_path",
        str(sam_checkpoint),
        "--groundingdino_checkpoint_path",
        str(groundingdino_checkpoint),
        "--groundingdino_config_path",
        str(groundingdino_config),
        "--downsample",
        "1",
        "--classes",
        *taxonomy.classes,
    )
    return StageInvocation(
        stage="masks",
        scene_id=scene.scene_id,
        variant_id="full950",
        budget=None,
        condition=None,
        command=command,
        cwd=workspace.root,
        log_path=assets.root / "masks.log",
    )


def build_scale_invocation(
    scene: RuntimeScene,
    assets: AssetPaths,
    workspace: SourceWorkspace,
) -> StageInvocation:
    command = (
        str(scene.python_bin),
        str(workspace.root / "get_scale.py"),
        "--progress_path",
        str(assets.scale_progress),
        "--sh_degree",
        "0",
        "--masks_path",
        str(assets.masks),
        "--point_cloud_path",
        str(scene.point_cloud_path),
        "--sparse_path",
        str(scene.sparse_path),
        "--images_path",
        str(scene.images_path),
        "--mask_scales_path",
        str(assets.mask_scales),
    )
    return StageInvocation(
        stage="scale",
        scene_id=scene.scene_id,
        variant_id="full950",
        budget=None,
        condition=None,
        command=command,
        cwd=workspace.root,
        log_path=assets.root / "scale.log",
    )


def build_train_invocation(
    scene: RuntimeScene,
    assets: AssetPaths,
    features: FeaturePaths,
    workspace: SourceWorkspace,
    *,
    budget: str,
) -> StageInvocation:
    command = [
        str(scene.python_bin),
        str(workspace.root / "train_contrastive_feature.py"),
        "--progress_path",
        str(features.progress),
        *_common_model_args(scene, assets),
        "--contrastive_feature_point_cloud_path",
        str(features.point_cloud),
        "--scale_gate_path",
        str(features.scale_gate),
        "--num_sampled_rays",
        "1000",
    ]
    if budget == "10000":
        if workspace.variant_id != "full950-iterations-cli":
            raise ValueError(
                "the 10k control requires the registered integer iterations CLI variant"
            )
        command.extend(("--iterations", "10000"))
    elif budget != "adaptive":
        raise ValueError(f"unknown feature budget: {budget}")
    return StageInvocation(
        stage="train",
        scene_id=scene.scene_id,
        variant_id=workspace.variant_id,
        budget=budget,
        condition=None,
        command=tuple(command),
        cwd=workspace.root,
        log_path=features.root / "train.log",
    )


def build_postprocess_invocation(
    scene: RuntimeScene,
    assets: AssetPaths,
    features: FeaturePaths,
    output: OutputPaths,
    workspace: SourceWorkspace,
    taxonomy: TaxonomySpec,
    *,
    budget: str,
    condition: str,
) -> StageInvocation:
    if condition == "B0-global":
        other_classes = (DISABLED_OTHER_CLASS,)
    elif condition == "B1-original":
        other_classes = taxonomy.other_classes
    else:
        raise ValueError(f"unknown baseline condition: {condition}")
    command = (
        str(scene.python_bin),
        str(workspace.root / "postprocess.py"),
        "--progress_path",
        str(output.progress),
        *_common_model_args(scene, assets),
        "--contrastive_feature_point_cloud_path",
        str(features.point_cloud),
        "--scale_gate_path",
        str(features.scale_gate),
        "--json_path",
        str(output.output_json),
        "--classes",
        *taxonomy.classes,
        "--selected_classes",
        *taxonomy.selected_classes,
        "--other_classes",
        *other_classes,
    )
    return StageInvocation(
        stage="postprocess",
        scene_id=scene.scene_id,
        variant_id=workspace.variant_id,
        budget=budget,
        condition=condition,
        command=command,
        cwd=workspace.root,
        log_path=output.root / "postprocess.log",
    )


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
        completed = subprocess.run(
            list(invocation.command),
            cwd=invocation.cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return int(completed.returncode)


def execute_stage(
    invocation: StageInvocation,
    *,
    is_complete: Callable[[], bool],
    executor: Executor = _default_executor,
    on_success: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if is_complete():
        return {
            "stage": invocation.stage,
            "status": "reused",
            "command": list(invocation.command),
        }
    invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    returncode = int(executor(invocation))
    seconds = float(time.monotonic() - started)
    if returncode != 0:
        raise RuntimeError(
            f"{invocation.stage} failed with exit {returncode}; see {invocation.log_path}"
        )
    if on_success is not None:
        on_success()
    if not is_complete():
        raise RuntimeError(
            f"{invocation.stage} returned success but its artifact is incomplete; "
            f"see {invocation.log_path}"
        )
    return {
        "stage": invocation.stage,
        "status": "completed",
        "seconds": seconds,
        "command": list(invocation.command),
    }


def read_cgroup_snapshot(root: Path) -> dict[str, Any]:
    maximum_text = (root / "memory.max").read_text(encoding="utf-8").strip()
    current = int((root / "memory.current").read_text(encoding="utf-8").strip())
    maximum = None if maximum_text == "max" else int(maximum_text)
    events: dict[str, int] = {}
    for line in (root / "memory.events").read_text(encoding="utf-8").splitlines():
        key, value = line.split()
        events[key] = int(value)
    return {"current": current, "max": maximum, "events": events}


def validate_cgroup(root: Path | None) -> dict[str, Any] | None:
    if root is None:
        return None
    snapshot = read_cgroup_snapshot(root)
    if snapshot["max"] != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(
            f"expected a 90 GiB cgroup, found memory.max={snapshot['max']}"
        )
    if snapshot["current"] >= snapshot["max"]:
        raise RuntimeError("cgroup memory.current has reached memory.max")
    return snapshot


def _assert_disk_floor(output_root: Path, floor_gib: float) -> float:
    free_gib = shutil.disk_usage(output_root).free / 1024**3
    if free_gib < floor_gib:
        raise RuntimeError(
            f"baseline closeout requires at least {floor_gib:.1f} GiB free; "
            f"found {free_gib:.1f} GiB"
        )
    return float(free_gib)


def _ensure_stage_parents(invocation: StageInvocation) -> None:
    invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
    for option in (
        "--progress_path",
        "--masks_path",
        "--labels_path",
        "--mask_scales_path",
        "--contrastive_feature_point_cloud_path",
        "--scale_gate_path",
        "--json_path",
    ):
        if option not in invocation.command:
            continue
        index = invocation.command.index(option)
        path = Path(invocation.command[index + 1])
        if option in {"--masks_path", "--labels_path", "--mask_scales_path"}:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)


def _run_registered_spec(
    spec: RunSpec,
    *,
    scenes: Mapping[str, RuntimeScene],
    workspaces: Mapping[str, SourceWorkspace],
    output_root: Path,
    taxonomy: TaxonomySpec,
    executor: Executor,
    disk_floor_gib: float,
    cgroup_root: Path | None,
) -> list[dict[str, Any]]:
    workspace = workspaces[spec.variant_id]
    records: list[dict[str, Any]] = []
    for scene_id in spec.scene_ids:
        scene = scenes[scene_id]
        assets = asset_paths(output_root, scene_id, taxonomy.taxonomy_id)
        features = feature_paths(
            output_root, scene_id, spec.variant_id, spec.budget, taxonomy.taxonomy_id
        )
        invocation = build_train_invocation(
            scene,
            assets,
            features,
            workspace,
            budget=spec.budget,
        )
        _ensure_stage_parents(invocation)
        _assert_disk_floor(output_root, disk_floor_gib)
        validate_cgroup(cgroup_root)
        result = execute_stage(
            invocation,
            is_complete=lambda current_scene=scene, current_paths=features: (
                feature_is_complete(current_scene, current_paths)
            ),
            executor=executor,
        )
        records.append(
            {
                **result,
                "scene_id": scene_id,
                "variant_id": spec.variant_id,
                "budget": spec.budget,
            }
        )
        for condition in spec.conditions:
            outputs = output_paths(
                output_root,
                scene_id,
                spec.variant_id,
                spec.budget,
                condition,
                taxonomy.taxonomy_id,
            )
            postprocess = build_postprocess_invocation(
                scene,
                assets,
                features,
                outputs,
                workspace,
                taxonomy,
                budget=spec.budget,
                condition=condition,
            )
            _ensure_stage_parents(postprocess)
            _assert_disk_floor(output_root, disk_floor_gib)
            validate_cgroup(cgroup_root)
            result = execute_stage(
                postprocess,
                is_complete=lambda current_scene=scene, current_paths=outputs: (
                    output_is_complete(current_scene, current_paths, taxonomy)
                ),
                executor=executor,
            )
            records.append(
                {
                    **result,
                    "scene_id": scene_id,
                    "variant_id": spec.variant_id,
                    "budget": spec.budget,
                    "condition": condition,
                }
            )
    return records


def run_baseline_closure(
    *,
    runtime_manifest: Path,
    workspace_manifest: Path,
    output_root: Path,
    sam_checkpoint: Path,
    groundingdino_checkpoint: Path,
    groundingdino_config: Path,
    taxonomy_id: str = "bfc18",
    executor: Executor = _default_executor,
    disk_floor_gib: float = 80.0,
    cgroup_root: Path | None = Path("/sys/fs/cgroup"),
) -> dict[str, Any]:
    if taxonomy_id != "bfc18":
        raise ValueError(
            "the registered closeout runs only bfc18; tip28 is provenance-only"
        )
    taxonomy = TAXONOMIES[taxonomy_id]
    scenes = load_runtime_scenes(runtime_manifest)
    workspaces = load_source_workspaces(workspace_manifest)
    required_scenes = set(CLOSURE_SCENES)
    required_variants = {spec.variant_id for spec in REGISTERED_RUNS}
    if missing := required_scenes - scenes.keys():
        raise KeyError(
            f"runtime manifest is missing registered scenes: {sorted(missing)}"
        )
    if missing := required_variants - workspaces.keys():
        raise KeyError(
            f"workspace manifest is missing registered variants: {sorted(missing)}"
        )
    selected_scenes = [scenes[key] for key in CLOSURE_SCENES]
    selected_workspaces = [workspaces[key] for key in sorted(required_variants)]
    assert_isolated_output(output_root, selected_scenes, selected_workspaces)
    for scene in selected_scenes:
        validate_scene_inputs(scene)
    for workspace in selected_workspaces:
        validate_source_workspace(workspace)
    for checkpoint in (sam_checkpoint, groundingdino_checkpoint, groundingdino_config):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"required existing checkpoint/config is missing: {checkpoint}"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    initial_cgroup = validate_cgroup(cgroup_root)
    initial_free = _assert_disk_floor(output_root, disk_floor_gib)
    records: list[dict[str, Any]] = []
    asset_workspace = workspaces["full950"]
    for scene in selected_scenes:
        assets = asset_paths(output_root, scene.scene_id, taxonomy.taxonomy_id)
        masks = build_masks_invocation(
            scene,
            assets,
            asset_workspace,
            taxonomy,
            sam_checkpoint=sam_checkpoint,
            groundingdino_checkpoint=groundingdino_checkpoint,
            groundingdino_config=groundingdino_config,
        )
        _ensure_stage_parents(masks)
        _assert_disk_floor(output_root, disk_floor_gib)
        validate_cgroup(cgroup_root)
        result = execute_stage(
            masks,
            is_complete=lambda current_scene=scene, current_paths=assets: (
                masks_are_complete(current_scene, current_paths, taxonomy)
            ),
            executor=executor,
            on_success=lambda current_scene=scene, current_paths=assets: (
                record_masks_completion(current_scene, current_paths, taxonomy)
            ),
        )
        records.append({**result, "scene_id": scene.scene_id})
        scale = build_scale_invocation(scene, assets, asset_workspace)
        _ensure_stage_parents(scale)
        _assert_disk_floor(output_root, disk_floor_gib)
        validate_cgroup(cgroup_root)
        result = execute_stage(
            scale,
            is_complete=lambda current_paths=assets: scales_are_complete(current_paths),
            executor=executor,
        )
        records.append({**result, "scene_id": scene.scene_id})
    for spec in REGISTERED_RUNS:
        records.extend(
            _run_registered_spec(
                spec,
                scenes=scenes,
                workspaces=workspaces,
                output_root=output_root,
                taxonomy=taxonomy,
                executor=executor,
                disk_floor_gib=disk_floor_gib,
                cgroup_root=cgroup_root,
            )
        )
    summary = {
        "schema": "saga-teacher-baseline-closure-run-v1",
        "baseline": "teacher-handoff-bfc2192",
        "primary_commit": SOURCE_VARIANTS["literal-bfc"].exact_commit,
        "full950_commit": SOURCE_VARIANTS["full950"].exact_commit,
        "source_variants": {
            key: {
                "base_commit": SOURCE_VARIANTS[key].base_commit,
                "exact_commit": SOURCE_VARIANTS[key].exact_commit,
                "patch_set": SOURCE_VARIANTS[key].patch_set,
            }
            for key in sorted({spec.variant_id for spec in REGISTERED_RUNS})
        },
        "taxonomy_id": taxonomy.taxonomy_id,
        "scenes": list(CLOSURE_SCENES),
        "initial_disk_free_gib": initial_free,
        "initial_cgroup": initial_cgroup,
        "final_disk_free_gib": _assert_disk_floor(output_root, disk_floor_gib),
        "final_cgroup": validate_cgroup(cgroup_root),
        "records": records,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--groundingdino-checkpoint", type=Path, required=True)
    parser.add_argument("--groundingdino-config", type=Path, required=True)
    parser.add_argument("--disk-floor-gib", type=float, default=80.0)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    result = run_baseline_closure(
        runtime_manifest=args.runtime_manifest,
        workspace_manifest=args.workspace_manifest,
        output_root=args.output_root,
        sam_checkpoint=args.sam_checkpoint,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        groundingdino_config=args.groundingdino_config,
        disk_floor_gib=args.disk_floor_gib,
        cgroup_root=args.cgroup_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
