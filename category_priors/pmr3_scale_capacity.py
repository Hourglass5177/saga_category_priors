from __future__ import annotations

"""Paired native-budget/10k scale-gate capacity control (PMR-3).

Training and prompt segmentation never read ground truth.  Ground truth enters
only ``evaluate_checkpoint`` after masks for U and the frozen nine-point grid
have been materialized.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, read_rows, write_json, write_rows
from .prompt_prior_diagnostics import GRID9, choose_grid_oracle, scale_key
from .prompt_prior_experiment import (
    SIMILARITY_THRESHOLD,
    _camera_list,
    _feature_model,
    _gt_path,
    _nearest,
    _pipeline,
    _scene_assets,
    _size_bin,
    _transform,
    evaluate_prompt_pair_arrays,
)
from .runner import load_scene_runtime_manifest


SCENES = ("scene0591_02", "scene0645_00")
FINAL_ITERATION = 10_000
FEATURE_SEED = 0
EXPECTED_OBJECT_COUNTS = {"scene0591_02": 15, "scene0645_00": 19}
MIN_DISK_FREE_GIB = 80.0


def _producer_commit() -> str | None:
    value = os.environ.get("SAGA_EXPERIMENT_COMMIT")
    return value if value else None


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _file_identity_matches(value: Any, path: Path) -> bool:
    try:
        return dict(value) == _file_identity(path)
    except (OSError, TypeError, ValueError):
        return False


def _runtime_prompts(prompts_root: Path, scene_id: str) -> dict[str, dict[str, Any]]:
    payload = load_json(
        prompts_root / "prompts" / scene_id / "runtime_prompts.json"
    )
    if payload.get("gt_fields_present") is not False:
        raise ValueError(f"{scene_id}: runtime prompts are not GT-isolated")
    rows = {
        str(row["prompt_id"]): dict(row) for row in payload.get("prompts", [])
    }
    expected = EXPECTED_OBJECT_COUNTS.get(scene_id)
    if expected is not None and len(rows) != expected:
        raise ValueError(
            f"{scene_id}: registered prompt count changed: {len(rows)} != {expected}"
        )
    return rows


def _evaluation_prompts(
    prompts_root: Path, scene_id: str
) -> dict[str, dict[str, Any]]:
    payload = load_json(
        prompts_root / "prompts" / scene_id / "evaluation_prompts.json"
    )
    rows = {
        str(row["prompt_id"]): dict(row) for row in payload.get("prompts", [])
    }
    expected = EXPECTED_OBJECT_COUNTS.get(scene_id)
    if expected is not None and len(rows) != expected:
        raise ValueError(
            f"{scene_id}: registered evaluation count changed: {len(rows)} != {expected}"
        )
    return rows


def build_training_command(
    *,
    workspace: Path,
    python_bin: Path,
    scene_base: Path,
    trajectory_root: Path,
    scene_assets: Mapping[str, Path] | None = None,
) -> list[str]:
    final_root = trajectory_root / f"iteration_{FINAL_ITERATION}"
    assets = dict(scene_assets or {})
    command = [
        "bash",
        str(workspace / "run_pipeline.sh"),
        "--python",
        str(python_bin),
        "--stage",
        "train",
        "--base-path",
        str(scene_base),
        "--feature-iterations",
        str(FINAL_ITERATION),
        "--feature-seed",
        str(FEATURE_SEED),
        "--num-sampled-rays",
        "1000",
        "--feature-snapshot-root",
        str(trajectory_root),
        "--contrastive-feature-point-cloud-path",
        str(final_root / "contrastive_feature_point_cloud.ply"),
        "--scale-gate-path",
        str(final_root / "scale_gate.pt"),
        "--progress-path",
        str(trajectory_root / "progress"),
        "--sh-degree",
        "0",
        "--feature-dim",
        "32",
        "--downsample",
        "1",
    ]
    option_names = {
        "images": "--images-path",
        "sparse": "--sparse-path",
        "point_cloud": "--point-cloud-path",
        "masks": "--masks-path",
        "labels": "--labels-path",
        "label_features": "--label-features-path",
        "mask_scales": "--mask-scales-path",
    }
    for name, option in option_names.items():
        if name in assets:
            command.extend((option, str(assets[name])))
    return command


def _training_assets(
    scene: Mapping[str, Any], prompt_assets: Mapping[str, Path]
) -> dict[str, Path]:
    base = Path(str(scene["base_path"])).resolve()
    assets = {
        "images": prompt_assets["images"].resolve(),
        "sparse": prompt_assets["sparse"].resolve(),
        "point_cloud": prompt_assets["point_cloud"].resolve(),
        "masks": (base / "saga/masks").resolve(),
        "labels": (base / "saga/labels").resolve(),
        "label_features": (base / "saga/labels/label_features.pt").resolve(),
        "mask_scales": prompt_assets["mask_scales"].resolve(),
    }
    for name in ("images", "sparse", "masks", "labels", "mask_scales"):
        if not assets[name].is_dir():
            raise FileNotFoundError(f"PMR-3 training {name} not found: {assets[name]}")
    for name in ("point_cloud", "label_features"):
        if not assets[name].is_file():
            raise FileNotFoundError(f"PMR-3 training {name} not found: {assets[name]}")
    return assets


def _snapshot_paths(
    trajectory_root: Path, checkpoint: str
) -> tuple[Path, Path, int]:
    manifest = load_json(trajectory_root / "training_manifest.json")
    if (
        manifest.get("kind") != "pmr3_scale_training_trajectory"
        or manifest.get("status") != "complete"
        or int(manifest.get("seed", -1)) != FEATURE_SEED
        or int(manifest.get("final_iteration", -1)) != FINAL_ITERATION
        or int(manifest.get("num_sampled_rays", -1)) != 1000
        or int(manifest.get("native_iteration", -1))
        != min(10 * int(manifest.get("train_camera_count", -1)), FINAL_ITERATION)
        or (
            _producer_commit() is not None
            and manifest.get("producer_commit") != _producer_commit()
        )
    ):
        raise ValueError(f"invalid PMR-3 training manifest: {trajectory_root}")
    if checkpoint == "native":
        directory = Path(str(manifest["native_snapshot"]))
        iteration = int(manifest["native_iteration"])
    elif checkpoint == "10k":
        directory = Path(str(manifest["final_snapshot"]))
        iteration = int(manifest["final_iteration"])
    else:
        raise ValueError(f"unknown checkpoint: {checkpoint}")
    metadata = load_json(directory / "snapshot.json")
    feature = directory / "contrastive_feature_point_cloud.ply"
    gate = directory / "scale_gate.pt"
    if (
        metadata.get("kind") != "pmr3_scale_training_snapshot"
        or metadata.get("status") != "complete"
        or int(metadata.get("iteration", -1)) != iteration
        or int(metadata.get("seed", -1)) != FEATURE_SEED
        or int(metadata.get("smooth_k", -1)) != 16
        or Path(str(metadata.get("feature_ply", ""))).resolve()
        != feature.resolve()
        or Path(str(metadata.get("scale_gate", ""))).resolve() != gate.resolve()
        or not feature.is_file()
        or feature.stat().st_size <= 0
        or not gate.is_file()
        or gate.stat().st_size <= 0
    ):
        raise ValueError(f"incomplete PMR-3 snapshot: {directory}")
    return feature.resolve(), gate.resolve(), iteration


def materialize_native_snapshot(trajectory_root: Path) -> dict[str, Any]:
    """KNN-smooth the raw native snapshot after its optimizer has gone away."""

    import torch

    manifest_path = trajectory_root / "training_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") == "complete":
        feature, gate, iteration = _snapshot_paths(trajectory_root, "native")
        return {
            "status": "reused",
            "iteration": iteration,
            "feature_ply": str(feature),
            "scale_gate": str(gate),
        }
    if (
        manifest.get("kind") != "pmr3_scale_training_trajectory"
        or manifest.get("status") != "training_complete"
        or int(manifest.get("seed", -1)) != FEATURE_SEED
        or int(manifest.get("final_iteration", -1)) != FINAL_ITERATION
    ):
        raise ValueError(f"native snapshot cannot be materialized: {trajectory_root}")
    directory = Path(str(manifest["native_snapshot"]))
    metadata_path = directory / "snapshot.json"
    metadata = load_json(metadata_path)
    raw_feature = Path(str(metadata.get("raw_feature_ply", "")))
    gate = directory / "scale_gate.pt"
    feature = directory / "contrastive_feature_point_cloud.ply"
    if (
        metadata.get("kind") == "pmr3_scale_training_snapshot"
        and metadata.get("status") == "complete"
        and feature.is_file()
        and gate.is_file()
    ):
        manifest["status"] = "complete"
        manifest["native_materialization"] = (
            "post-training-traditional-knn" + str(int(metadata["smooth_k"]))
        )
        write_json(manifest_path, manifest)
        return {
            "status": "recovered-after-materialization",
            "iteration": int(metadata["iteration"]),
            "feature_ply": str(feature),
            "scale_gate": str(gate),
        }
    if (
        metadata.get("kind") != "pmr3_scale_training_snapshot"
        or metadata.get("status") != "raw_complete"
        or not raw_feature.is_file()
        or not gate.is_file()
    ):
        raise ValueError(f"raw native snapshot is incomplete: {directory}")
    feature_part = directory / "contrastive_feature_point_cloud.part.ply"
    model = _feature_model(raw_feature)
    model.save_ply(
        str(feature_part),
        smooth_weights=None,
        smooth_type="traditional",
        smooth_K=int(metadata["smooth_k"]),
    )
    feature_part.replace(feature)
    del model
    torch.cuda.empty_cache()
    metadata.update(
        {
            "status": "complete",
            "feature_ply": str(feature.resolve()),
            "scale_gate": str(gate.resolve()),
            "materialization": (
                "post-training-traditional-knn" + str(int(metadata["smooth_k"]))
            ),
        }
    )
    write_json(metadata_path, metadata)
    manifest["status"] = "complete"
    manifest["native_materialization"] = metadata["materialization"]
    write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "iteration": int(metadata["iteration"]),
        "feature_ply": str(feature),
        "scale_gate": str(gate),
    }


def _validate_feature_and_gate(
    feature: Path, gate: Path, *, expected_points: int
) -> None:
    import torch
    from plyfile import PlyData

    ply = PlyData.read(str(feature))
    vertices = ply.elements[0]
    if int(vertices.count) != int(expected_points):
        raise ValueError(f"{feature}: unexpected point count")
    property_names = {item.name for item in vertices.properties}
    expected_affinity = {f"f_{index}" for index in range(32)}
    expected_semantic = {f"sf_{index}" for index in range(32)}
    if not expected_affinity <= property_names or not expected_semantic <= property_names:
        raise ValueError(f"{feature}: missing affinity or semantic feature fields")
    for name in sorted(expected_affinity | expected_semantic):
        if not np.isfinite(np.asarray(vertices[name])).all():
            raise ValueError(f"{feature}: non-finite feature values in {name}")
    state = torch.load(gate, map_location="cpu")
    if (
        not isinstance(state, Mapping)
        or tuple(state.get("0.weight", torch.empty(0)).shape) != (32, 1)
        or tuple(state.get("0.bias", torch.empty(0)).shape) != (32,)
        or not all(torch.isfinite(value).all().item() for value in state.values())
    ):
        raise ValueError(f"{gate}: invalid 1-to-32 scale gate state")


def validate_training_trajectory(
    *, trajectory_root: Path, training_assets: Mapping[str, Path]
) -> dict[str, Any]:
    manifest = load_json(trajectory_root / "training_manifest.json")
    manifest_fields = {
        "images": "images_path",
        "sparse": "sparse_path",
        "point_cloud": "point_cloud_path",
        "masks": "masks_path",
        "labels": "labels_path",
        "label_features": "label_features_path",
        "mask_scales": "mask_scales_path",
    }
    for name, field in manifest_fields.items():
        if Path(str(manifest.get(field, ""))).resolve() != training_assets[name].resolve():
            raise ValueError(
                f"{trajectory_root}: training input changed for {name}"
            )
    if int(manifest.get("smooth_k", -1)) != 16:
        raise ValueError(f"{trajectory_root}: smooth_k is not the frozen value 16")
    pairs = {}
    reference_xyz = load_ply_xyz(training_assets["point_cloud"])
    for checkpoint in ("native", "10k"):
        feature, gate, iteration = _snapshot_paths(trajectory_root, checkpoint)
        feature_xyz = load_ply_xyz(feature)
        if feature_xyz.shape != reference_xyz.shape or not np.allclose(
            feature_xyz, reference_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                f"{trajectory_root}/{checkpoint}: RGB and feature XYZ/order differ"
            )
        _validate_feature_and_gate(
            feature, gate, expected_points=int(len(reference_xyz))
        )
        pairs[checkpoint] = {
            "iteration": iteration,
            "feature_ply": str(feature),
            "scale_gate": str(gate),
            "point_count": int(len(feature_xyz)),
        }
    return {
        "kind": "pmr3_validated_training_trajectory",
        "status": "complete",
        "seed": FEATURE_SEED,
        "checkpoints": pairs,
    }


def run_training_trajectories(
    *,
    workspace: Path,
    python_bin: Path,
    runtime_manifest: Path,
    training_root: Path,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    results = []
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        training_assets = _training_assets(scene, assets)
        trajectory = training_root / scene_id
        reused = False
        needs_training = True
        manifest_path = trajectory / "training_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = load_json(manifest_path)
                if manifest.get("status") == "training_complete":
                    materialize_native_snapshot(trajectory)
                validation = validate_training_trajectory(
                    trajectory_root=trajectory,
                    training_assets=training_assets,
                )
                reused = True
                needs_training = False
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                needs_training = True
        if needs_training:
            free_gib = shutil.disk_usage(training_root.parent).free / 1024**3
            if free_gib < MIN_DISK_FREE_GIB:
                raise RuntimeError(
                    f"disk safety floor reached: {free_gib:.2f} GiB < {MIN_DISK_FREE_GIB}"
                )
            trajectory.mkdir(parents=True, exist_ok=True)
            command = build_training_command(
                workspace=workspace,
                python_bin=python_bin,
                scene_base=Path(str(scene["base_path"])),
                trajectory_root=trajectory,
                scene_assets=training_assets,
            )
            environment = os.environ.copy()
            environment["CUDA_HOME"] = "/usr/local/cuda-12.8"
            environment["TORCH_CUDA_ARCH_LIST"] = "12.0"
            with (trajectory / "training.log").open(
                "a", encoding="utf-8"
            ) as log:
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
                    f"{scene_id}: 10k training exited {completed.returncode}"
                )
            materialize_native_snapshot(trajectory)
            validation = validate_training_trajectory(
                trajectory_root=trajectory,
                training_assets=training_assets,
            )
        results.append({"scene_id": scene_id, "reused": reused, **validation})
    payload = {
        "kind": "pmr3_training_manifest",
        "status": "complete",
        "scene_ids": list(scene_ids),
        "seed": FEATURE_SEED,
        "final_iteration": FINAL_ITERATION,
        "producer_commit": _producer_commit(),
        "scenes": results,
    }
    write_json(training_root / "pmr3_training_manifest.json", payload)
    return payload


def _checkpoint_assets(
    scene: Mapping[str, Any], trajectory_root: Path, checkpoint: str
) -> tuple[dict[str, Path], int]:
    assets = _scene_assets(scene)
    feature, gate, iteration = _snapshot_paths(trajectory_root, checkpoint)
    return {**assets, "feature_ply": feature, "scale_gate": gate}, iteration


def _result_complete(
    path: Path,
    *,
    prompt: Mapping[str, Any],
    point_count: int,
    feature: Path,
    gate: Path,
    checkpoint: str,
    checkpoint_iteration: int,
    uniform_scale_input: float | None = None,
) -> bool:
    try:
        metadata = load_json(path.with_suffix(".json"))
        with np.load(path) as payload:
            expected = {"U_global", *(scale_key(value) for value in GRID9)}
            valid = set(payload.files) == expected and all(
                np.asarray(payload[key]).shape == (point_count,)
                and set(np.unique(np.asarray(payload[key])).tolist()) <= {0, 1}
                for key in expected
            )
        return bool(
            valid
            and metadata.get("kind") == "pmr3_scale_capacity_masks"
            and metadata.get("status") == "complete"
            and metadata.get("checkpoint") == checkpoint
            and int(metadata.get("checkpoint_iteration", -1))
            == int(checkpoint_iteration)
            and metadata.get("prompt") == dict(prompt)
            and Path(str(metadata.get("feature_ply", ""))).resolve()
            == feature.resolve()
            and Path(str(metadata.get("scale_gate", ""))).resolve()
            == gate.resolve()
            and _file_identity_matches(metadata.get("feature_identity"), feature)
            and _file_identity_matches(metadata.get("gate_identity"), gate)
            and (
                uniform_scale_input is None
                or math.isclose(
                    float(metadata.get("uniform_scale_input", float("nan"))),
                    float(uniform_scale_input),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            and math.isclose(
                float(metadata.get("similarity_threshold", float("nan"))),
                SIMILARITY_THRESHOLD,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def segment_checkpoint(
    *,
    runtime_manifest: Path,
    prompts_root: Path,
    parameters_path: Path,
    training_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    checkpoint: str,
) -> dict[str, Any]:
    """Render U plus the fixed grid.  This runtime path intentionally has no GT."""

    import torch
    import torch.nn.functional as functional
    from gaussian_renderer import render_contrastive_feature

    scenes = load_scene_runtime_manifest(runtime_manifest)
    parameters = load_json(parameters_path)
    completed = 0
    reused = 0
    for scene_id in scene_ids:
        prompts = _runtime_prompts(prompts_root, scene_id)
        assets, iteration = _checkpoint_assets(
            scenes[scene_id], training_root / scene_id, checkpoint
        )
        feature_xyz = load_ply_xyz(assets["feature_ply"])
        scene_xyz = load_ply_xyz(assets["point_cloud"])
        if feature_xyz.shape != scene_xyz.shape or not np.allclose(
            feature_xyz, scene_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{scene_id}: RGB and feature XYZ/order differ")
        pending = []
        uniform_scale_input = float(
            parameters["scenes"][scene_id]["global_scale_input"]
        )
        scene_output = output_root / checkpoint / scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        for prompt_id, prompt in sorted(prompts.items()):
            target = scene_output / f"{prompt_id}.npz"
            if _result_complete(
                target,
                prompt=prompt,
                point_count=len(feature_xyz),
                feature=assets["feature_ply"],
                gate=assets["scale_gate"],
                checkpoint=checkpoint,
                checkpoint_iteration=iteration,
                uniform_scale_input=uniform_scale_input,
            ):
                reused += 1
            else:
                pending.append(prompt)
        if not pending:
            continue
        model = _feature_model(assets["feature_ply"])
        gate_model = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias=True), torch.nn.Sigmoid()
        ).cuda()
        gate_model.load_state_dict(
            torch.load(assets["scale_gate"], map_location="cuda")
        )
        gate_model.eval()
        cameras = {camera.image_name: camera for camera in _camera_list(assets)}
        point_features = model.get_point_features.detach()
        background = torch.zeros(32, dtype=torch.float32, device="cuda")
        pipe = _pipeline()
        query_raw: dict[str, Any] = {}
        by_camera: dict[str, list[Mapping[str, Any]]] = {}
        for prompt in pending:
            by_camera.setdefault(str(prompt["image_name"]), []).append(prompt)
        for image_name, rows in sorted(by_camera.items()):
            camera = deepcopy(cameras[image_name])
            camera.feature_height = camera.image_height
            camera.feature_width = camera.image_width
            with torch.no_grad():
                rendered = render_contrastive_feature(
                    camera, model, pipe, background, norm_point_features=True
                )["render"].detach()
            for prompt in rows:
                x, y = int(prompt["x"]), int(prompt["y"])
                query_raw[str(prompt["prompt_id"])] = rendered[:, y, x].clone()
            del rendered
        scales_by_prompt = {
            str(prompt["prompt_id"]): uniform_scale_input
            for prompt in pending
        }
        masks = {str(prompt["prompt_id"]): {} for prompt in pending}
        gate_rows: dict[str, list[float]] = {}
        scale_items = [("U_global", None)] + [
            (scale_key(value), float(value)) for value in GRID9
        ]
        for key, fixed_value in scale_items:
            value = (
                next(iter(scales_by_prompt.values()))
                if fixed_value is None
                else fixed_value
            )
            with torch.no_grad():
                gate_value = gate_model(
                    torch.tensor([value], dtype=torch.float32, device="cuda")
                ).squeeze(0)
                points = functional.normalize(
                    point_features * gate_value.unsqueeze(0), dim=1, p=2
                )
            for prompt in pending:
                prompt_id = str(prompt["prompt_id"])
                with torch.no_grad():
                    query = functional.normalize(
                        query_raw[prompt_id] * gate_value, dim=0, p=2
                    )
                    mask = (points @ query) > SIMILARITY_THRESHOLD
                masks[prompt_id][key] = mask.detach().cpu().numpy().astype(np.uint8)
            gate_rows[key] = gate_value.detach().cpu().numpy().astype(float).tolist()
            del points
        for prompt in pending:
            prompt_id = str(prompt["prompt_id"])
            target = scene_output / f"{prompt_id}.npz"
            temporary = target.with_suffix(".part.npz")
            np.savez_compressed(temporary, **masks[prompt_id])
            temporary.replace(target)
            write_json(
                target.with_suffix(".json"),
                {
                    "kind": "pmr3_scale_capacity_masks",
                    "status": "complete",
                    "scene_id": scene_id,
                    "checkpoint": checkpoint,
                    "checkpoint_iteration": iteration,
                    "prompt": dict(prompt),
                    "feature_ply": str(assets["feature_ply"]),
                    "scale_gate": str(assets["scale_gate"]),
                    "feature_identity": _file_identity(assets["feature_ply"]),
                    "gate_identity": _file_identity(assets["scale_gate"]),
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "uniform_scale_input": scales_by_prompt[prompt_id],
                    "grid_scales": {scale_key(value): value for value in GRID9},
                    "grid_gates": gate_rows,
                    "gt_accessed": False,
                },
            )
            completed += 1
        del model, gate_model
        torch.cuda.empty_cache()
    payload = {
        "kind": "pmr3_scale_capacity_segmentation",
        "status": "complete",
        "checkpoint": checkpoint,
        "scene_ids": list(scene_ids),
        "completed_prompts": completed,
        "reused_prompts": reused,
        "gt_accessed": False,
        "producer_commit": _producer_commit(),
    }
    write_json(output_root / checkpoint / "segmentation.json", payload)
    return payload


def evaluate_checkpoint(
    *,
    runtime_manifest: Path,
    prompts_root: Path,
    gt_dir: Path,
    training_root: Path,
    masks_root: Path,
    scene_ids: Sequence[str],
    checkpoint: str,
    table_output: Path,
    analysis_output: Path,
    size_bins: Path | None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins else None
    object_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    checkpoint_iterations: dict[str, int] = {}
    for scene_id in scene_ids:
        runtime = _runtime_prompts(prompts_root, scene_id)
        evaluation = _evaluation_prompts(prompts_root, scene_id)
        if set(runtime) != set(evaluation):
            raise ValueError(f"{scene_id}: runtime/evaluation prompt IDs differ")
        assets, iteration = _checkpoint_assets(
            scenes[scene_id], training_root / scene_id, checkpoint
        )
        checkpoint_iterations[scene_id] = iteration
        gt_xyz, gt = load_ground_truth_npz(
            _gt_path(scenes[scene_id], gt_dir, scene_id), scene_id
        )
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["feature_ply"]), _transform(scenes[scene_id])
        )
        gt_to_gaussian, gt_distance = _nearest(gt_xyz, gaussian_xyz)
        gaussian_to_gt, gaussian_distance = _nearest(gaussian_xyz, gt_xyz)
        for prompt_id in sorted(runtime):
            prompt = runtime[prompt_id]
            target = evaluation[prompt_id]
            path = masks_root / checkpoint / scene_id / f"{prompt_id}.npz"
            if not _result_complete(
                path,
                prompt=prompt,
                point_count=len(gaussian_xyz),
                feature=assets["feature_ply"],
                gate=assets["scale_gate"],
                checkpoint=checkpoint,
                checkpoint_iteration=iteration,
            ):
                raise ValueError(f"incomplete PMR-3 mask result: {path}")
            with np.load(path) as payload:
                arrays = {key: np.asarray(payload[key]).astype(bool) for key in payload.files}
            common = {
                "target_class_id": int(target["class_id"]),
                "target_instance_id": int(target["gt_instance_id"]),
                "gt_semantic": gt.semantic,
                "gt_instance": gt.instance,
                "gt_to_gaussian_index": gt_to_gaussian,
                "gt_to_gaussian_distance_m": gt_distance,
                "gaussian_to_gt_index": gaussian_to_gt,
                "gaussian_to_gt_distance_m": gaussian_distance,
            }
            uniform = evaluate_prompt_pair_arrays(mask=arrays["U_global"], **common)
            base = {
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "checkpoint": checkpoint,
                "checkpoint_iteration": iteration,
                "class_name": str(target["class_name"]),
                "size_bin": _size_bin(float(target["bbox_diagonal_m"]), size_spec),
            }
            table_rows.append({**base, "condition": "U_global", **uniform})
            candidates = [
                {"condition": "U_global", "scale_input": float("nan"), **uniform}
            ]
            for value in GRID9:
                key = scale_key(value)
                metrics = evaluate_prompt_pair_arrays(mask=arrays[key], **common)
                table_rows.append(
                    {**base, "condition": key, "scale_input": value, **metrics}
                )
                candidates.append(
                    {"condition": key, "scale_input": value, **metrics}
                )
            uniform_scale = float(
                load_json(path.with_suffix(".json"))["uniform_scale_input"]
            )
            candidates[0]["scale_input"] = uniform_scale
            best = choose_grid_oracle(candidates, uniform_scale=uniform_scale)
            table_rows.append({**base, "condition": "GridOracle", **best})
            object_rows.append(
                {
                    **base,
                    "iou_u": float(uniform["iou"]),
                    "precision_u": float(uniform["gaussian_precision"]),
                    "recall_u": float(uniform["gt_recall"]),
                    "grid_best_condition": str(best["condition"]),
                    "grid_best_scale": float(best["scale_input"]),
                    "grid_best_iou": float(best["iou"]),
                    "grid_best_precision": float(best["gaussian_precision"]),
                    "grid_best_recall": float(best["gt_recall"]),
                    "grid_delta_iou": float(best["iou"] - uniform["iou"]),
                    "grid_delta_precision": float(
                        best["gaussian_precision"] - uniform["gaussian_precision"]
                    ),
                    "grid_delta_recall": float(
                        best["gt_recall"] - uniform["gt_recall"]
                    ),
                }
            )
    write_rows(table_output, table_rows)
    scene_results = []
    for scene_id in scene_ids:
        rows = [row for row in object_rows if row["scene_id"] == scene_id]
        scene_results.append(
            {
                "scene_id": scene_id,
                "object_count": len(rows),
                "grid_mean_delta_iou": float(np.mean([r["grid_delta_iou"] for r in rows])),
                "grid_mean_delta_precision": float(
                    np.mean([r["grid_delta_precision"] for r in rows])
                ),
                "grid_mean_delta_recall": float(
                    np.mean([r["grid_delta_recall"] for r in rows])
                ),
                "object_fraction_at_least_0p02": float(
                    np.mean([r["grid_delta_iou"] >= 0.02 for r in rows])
                ),
            }
        )
    size_results: dict[str, dict[str, Any]] = {}
    for size_name in sorted(
        {str(row["size_bin"]) for row in object_rows if row.get("size_bin") is not None}
    ):
        per_scene = []
        for scene_id in scene_ids:
            rows = [
                row
                for row in object_rows
                if row["scene_id"] == scene_id and row.get("size_bin") == size_name
            ]
            if rows:
                per_scene.append(
                    {
                        "scene_id": scene_id,
                        "object_count": len(rows),
                        "grid_mean_delta_iou": float(
                            np.mean([row["grid_delta_iou"] for row in rows])
                        ),
                        "grid_mean_delta_precision": float(
                            np.mean([row["grid_delta_precision"] for row in rows])
                        ),
                        "grid_mean_delta_recall": float(
                            np.mean([row["grid_delta_recall"] for row in rows])
                        ),
                    }
                )
        size_results[size_name] = {
            "object_count": int(sum(row["object_count"] for row in per_scene)),
            "scene_count": len(per_scene),
            "scene_equal_grid_delta_iou": float(
                np.mean([row["grid_mean_delta_iou"] for row in per_scene])
            ),
            "scene_equal_grid_delta_precision": float(
                np.mean([row["grid_mean_delta_precision"] for row in per_scene])
            ),
            "scene_equal_grid_delta_recall": float(
                np.mean([row["grid_mean_delta_recall"] for row in per_scene])
            ),
            "scene_results": per_scene,
        }
    analysis = {
        "kind": "pmr3_scale_capacity_checkpoint_analysis",
        "status": "complete",
        "checkpoint": checkpoint,
        "checkpoint_iterations": checkpoint_iterations,
        "scene_ids": list(scene_ids),
        "object_count": len(object_rows),
        "scene_equal_grid_delta_iou": float(
            np.mean([row["grid_mean_delta_iou"] for row in scene_results])
        ),
        "scene_equal_grid_delta_precision": float(
            np.mean([row["grid_mean_delta_precision"] for row in scene_results])
        ),
        "object_fraction_at_least_0p02": float(
            np.mean([row["grid_delta_iou"] >= 0.02 for row in object_rows])
        ),
        "scene_results": scene_results,
        "size_results": size_results,
        "object_results": object_rows,
        "gt_used_only_for_evaluation": True,
        "producer_commit": _producer_commit(),
    }
    write_json(analysis_output, analysis)
    return analysis


def analyze_pair(
    *,
    native_analysis: Path,
    tenk_analysis: Path,
    output: Path,
    native_table: Path | None = None,
    tenk_table: Path | None = None,
    combined_table: Path | None = None,
    historical_analysis: Path | None = None,
) -> dict[str, Any]:
    native = load_json(native_analysis)
    tenk = load_json(tenk_analysis)
    native_keys = {
        (row["scene_id"], row["prompt_id"]) for row in native["object_results"]
    }
    tenk_keys = {
        (row["scene_id"], row["prompt_id"]) for row in tenk["object_results"]
    }
    if native_keys != tenk_keys or len(tenk_keys) != sum(EXPECTED_OBJECT_COUNTS.values()):
        raise ValueError("native/10k registered objects are not exactly paired")
    tenk_mean = float(tenk["scene_equal_grid_delta_iou"])
    native_mean = float(native["scene_equal_grid_delta_iou"])
    scene_values = {
        row["scene_id"]: float(row["grid_mean_delta_iou"])
        for row in tenk["scene_results"]
    }
    checks = {
        "tenk_scene_equal_capacity_at_least_0p02": tenk_mean >= 0.02,
        "both_scenes_capacity_at_least_0p01": all(
            scene_values.get(scene_id, float("-inf")) >= 0.01
            for scene_id in SCENES
        ),
        "object_fraction_at_least_25pct": float(
            tenk["object_fraction_at_least_0p02"]
        )
        >= 0.25,
        "tenk_minus_native_capacity_at_least_0p01": tenk_mean - native_mean >= 0.01,
        "tenk_precision_loss_no_more_than_1pp": float(
            tenk["scene_equal_grid_delta_precision"]
        )
        >= -0.01,
    }
    passed = all(checks.values())
    payload = {
        "kind": "pmr3_scale_capacity_analysis",
        "status": "complete",
        "scene_ids": list(SCENES),
        "object_count": len(tenk_keys),
        "native_scene_equal_capacity": native_mean,
        "tenk_scene_equal_capacity": tenk_mean,
        "tenk_minus_native_capacity": tenk_mean - native_mean,
        "tenk_scene_equal_delta_precision": float(
            tenk["scene_equal_grid_delta_precision"]
        ),
        "tenk_scene_results": tenk["scene_results"],
        "tenk_object_fraction_at_least_0p02": float(
            tenk["object_fraction_at_least_0p02"]
        ),
        "checks": checks,
        "passed": passed,
        "decision": (
            "proceed-to-train-only-native-class-scale-holdout"
            if passed
            else "stop-native-training-extension-does-not-restore-scale-capacity"
        ),
        "conclusion_boundary": (
            "training-budget and scale-gate capacity only; category priors are not tested"
        ),
        "producer_commit": _producer_commit(),
    }
    if historical_analysis is not None:
        historical = load_json(historical_analysis)
        historical_rows = [
            row
            for row in historical.get("object_results", [])
            if row.get("scene_id") in SCENES
        ]
        historical_scene_means = []
        for scene_id in SCENES:
            rows = [row for row in historical_rows if row.get("scene_id") == scene_id]
            if rows:
                historical_scene_means.append(
                    float(np.mean([float(row["grid_delta_iou"]) for row in rows]))
                )
        historical_capacity = (
            float(np.mean(historical_scene_means))
            if len(historical_scene_means) == len(SCENES)
            else None
        )
        payload["historical_native_context"] = {
            "path": str(historical_analysis.resolve()),
            "scene_equal_capacity": historical_capacity,
            "native_absolute_difference": (
                abs(native_mean - historical_capacity)
                if historical_capacity is not None
                else None
            ),
            "warning_difference_exceeds_0p02": bool(
                historical_capacity is not None
                and abs(native_mean - historical_capacity) > 0.02
            ),
            "changes_preregistered_decision": False,
        }
    supplied_tables = (native_table, tenk_table, combined_table)
    if any(value is not None for value in supplied_tables):
        if not all(value is not None for value in supplied_tables):
            raise ValueError(
                "native-table, tenk-table and combined-table must be supplied together"
            )
        combined_rows = read_rows(native_table) + read_rows(tenk_table)
        write_rows(combined_table, combined_rows)
        payload["combined_metrics"] = str(combined_table.resolve())
    write_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--workspace", required=True, type=Path)
    train.add_argument("--python", required=True, type=Path)
    train.add_argument("--runtime-manifest", required=True, type=Path)
    train.add_argument("--training-root", required=True, type=Path)
    train.add_argument("--scene", action="append", required=True)

    segment = commands.add_parser("segment")
    segment.add_argument("--runtime-manifest", required=True, type=Path)
    segment.add_argument("--prompts-root", required=True, type=Path)
    segment.add_argument("--parameters", required=True, type=Path)
    segment.add_argument("--training-root", required=True, type=Path)
    segment.add_argument("--output-root", required=True, type=Path)
    segment.add_argument("--scene", action="append", required=True)
    segment.add_argument("--checkpoint", choices=("native", "10k"), required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--runtime-manifest", required=True, type=Path)
    evaluate.add_argument("--prompts-root", required=True, type=Path)
    evaluate.add_argument("--gt-dir", required=True, type=Path)
    evaluate.add_argument("--training-root", required=True, type=Path)
    evaluate.add_argument("--masks-root", required=True, type=Path)
    evaluate.add_argument("--scene", action="append", required=True)
    evaluate.add_argument("--checkpoint", choices=("native", "10k"), required=True)
    evaluate.add_argument("--table-output", required=True, type=Path)
    evaluate.add_argument("--analysis-output", required=True, type=Path)
    evaluate.add_argument("--size-bins", type=Path)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--native-analysis", required=True, type=Path)
    analyze.add_argument("--tenk-analysis", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--native-table", type=Path)
    analyze.add_argument("--tenk-table", type=Path)
    analyze.add_argument("--combined-table", type=Path)
    analyze.add_argument("--historical-analysis", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train":
        result = run_training_trajectories(
            workspace=args.workspace,
            python_bin=args.python,
            runtime_manifest=args.runtime_manifest,
            training_root=args.training_root,
            scene_ids=args.scene,
        )
    elif args.command == "segment":
        result = segment_checkpoint(
            runtime_manifest=args.runtime_manifest,
            prompts_root=args.prompts_root,
            parameters_path=args.parameters,
            training_root=args.training_root,
            output_root=args.output_root,
            scene_ids=args.scene,
            checkpoint=args.checkpoint,
        )
    elif args.command == "evaluate":
        result = evaluate_checkpoint(
            runtime_manifest=args.runtime_manifest,
            prompts_root=args.prompts_root,
            gt_dir=args.gt_dir,
            training_root=args.training_root,
            masks_root=args.masks_root,
            scene_ids=args.scene,
            checkpoint=args.checkpoint,
            table_output=args.table_output,
            analysis_output=args.analysis_output,
            size_bins=args.size_bins,
        )
    else:
        result = analyze_pair(
            native_analysis=args.native_analysis,
            tenk_analysis=args.tenk_analysis,
            output=args.output,
            native_table=args.native_table,
            tenk_table=args.tenk_table,
            combined_table=args.combined_table,
            historical_analysis=args.historical_analysis,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
