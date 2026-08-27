from __future__ import annotations

"""Causal diagnostics for the native SAGA prompt scale gate.

This module deliberately does not construct objects, clusters, or automatic
proposals.  Ground truth is permitted only in ``audit-directions``,
``prepare-capacity`` and ``evaluate-capacity``.  ``segment-capacity`` consumes
an explicitly GT-derived oracle plan, but has no GT path and only changes the
single normalized scalar passed to the already-trained scale gate.
"""

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, write_json, write_rows
from .prompt_prior import training_quantile_uniform
from .prompt_prior_experiment import (
    DEV8,
    PROMPT_RADIUS_M,
    SIMILARITY_THRESHOLD,
    _camera_list,
    _feature_model,
    _gt_path,
    _mask_scale_values,
    _nearest,
    _pipeline,
    _scene_assets,
    _scene_model,
    _transform,
    evaluate_prompt_pair_arrays,
)
from .runner import load_scene_runtime_manifest

GRID5 = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_SUPPLEMENT = (0.125, 0.375, 0.625, 0.875)
GRID9 = tuple(sorted(GRID5 + GRID_SUPPLEMENT))
RADIUS_SENSITIVITY_M = (0.02, 0.05, 0.10)
FORMULA_VERSION = "native-visible-mask-historical-get-scale-v1"


def _producer_commit() -> str | None:
    value = os.environ.get("SAGA_EXPERIMENT_COMMIT")
    return value if value else None


def _file_identity(path: Path) -> dict[str, Any]:
    """Return a lightweight resume identity without inventing artifact hashes."""

    resolved = path.resolve()
    if resolved.is_dir():
        files = []
        for child in sorted(value for value in resolved.rglob("*") if value.is_file()):
            stat = child.stat()
            files.append(
                {
                    "relative_path": child.relative_to(resolved).as_posix(),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        return {
            "path": str(resolved),
            "kind": "directory",
            "file_count": len(files),
            "files": files,
        }
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


def scale_key(value: float) -> str:
    scaled = round(float(value) * 1000.0)
    if not math.isclose(float(value), scaled / 1000.0, abs_tol=1e-12):
        raise ValueError(f"scale must have at most three decimals: {value}")
    return f"s_{scaled:04d}"


def classify_direction_change(
    uniform: np.ndarray,
    data: np.ndarray,
    *,
    gaussian_to_gt_index: np.ndarray,
    gaussian_to_gt_distance_m: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    target_class_id: int,
    target_instance_id: int,
    radius_m: float,
) -> dict[str, Any]:
    """Classify every U/D XOR Gaussian without dropping unsupported points."""

    u = np.asarray(uniform, dtype=bool)
    d = np.asarray(data, dtype=bool)
    index = np.asarray(gaussian_to_gt_index, dtype=np.int64)
    distance = np.asarray(gaussian_to_gt_distance_m, dtype=np.float64)
    if u.shape != d.shape or u.shape != index.shape or u.shape != distance.shape:
        raise ValueError("direction arrays must have identical one-dimensional shape")
    supported = distance <= float(radius_m)
    semantic = np.asarray(gt_semantic)
    instance = np.asarray(gt_instance)
    mapped_semantic = np.full(len(u), -1, dtype=np.int64)
    mapped_instance = np.full(len(u), -1, dtype=np.int64)
    mapped_semantic[supported] = semantic[index[supported]]
    mapped_instance[supported] = instance[index[supported]]
    target = (
        supported
        & (mapped_semantic == int(target_class_id))
        & (mapped_instance == int(target_instance_id))
    )
    same_class_other = (
        supported
        & (mapped_semantic == int(target_class_id))
        & (mapped_instance != int(target_instance_id))
    )
    wrong_class = supported & (mapped_semantic != int(target_class_id))
    unsupported = ~supported
    added = d & ~u
    removed = u & ~d

    counts = {
        "added_target": int(np.count_nonzero(added & target)),
        "added_same_class_other": int(np.count_nonzero(added & same_class_other)),
        "added_wrong_class": int(np.count_nonzero(added & wrong_class)),
        "added_unsupported": int(np.count_nonzero(added & unsupported)),
        "removed_target": int(np.count_nonzero(removed & target)),
        "removed_same_class_other": int(np.count_nonzero(removed & same_class_other)),
        "removed_wrong_class": int(np.count_nonzero(removed & wrong_class)),
        "removed_unsupported": int(np.count_nonzero(removed & unsupported)),
    }
    helpful = counts["added_target"] + sum(
        counts[name]
        for name in (
            "removed_same_class_other",
            "removed_wrong_class",
            "removed_unsupported",
        )
    )
    harmful = counts["removed_target"] + sum(
        counts[name]
        for name in (
            "added_same_class_other",
            "added_wrong_class",
            "added_unsupported",
        )
    )
    changed = helpful + harmful
    return {
        **counts,
        "helpful_count": int(helpful),
        "harmful_count": int(harmful),
        "changed_count": int(changed),
        "help_ratio": float(helpful / changed) if changed else None,
        "direction": float((helpful - harmful) / changed) if changed else 0.0,
    }


def native_visible_mask_scale(
    depth: np.ndarray,
    footprint: np.ndarray,
    *,
    fov_x: float,
    fov_y: float,
    historical_axis_order: bool = True,
    require_valid_depth: bool = False,
) -> dict[str, Any]:
    """Reproduce the historical ``get_scale.py`` visible-mask scale.

    ``historical_axis_order=True`` intentionally preserves the public code's
    row/column swap because the frozen gate was trained in that scale domain.
    The corrected variant is emitted only as a static diagnostic.
    """

    # Historical PyTorch execution used float32 tensors.  Keep the same dtype
    # so an oracle scale cannot drift merely because NumPy defaulted to float64.
    depth_array = np.asarray(depth, dtype=np.float32).squeeze()
    mask = np.asarray(footprint, dtype=bool)
    if depth_array.ndim != 2 or mask.shape != depth_array.shape:
        raise ValueError("depth and footprint must be matching two-dimensional arrays")
    padded = np.pad(mask.astype(np.int16), 1, mode="constant")
    neighborhood = np.zeros_like(mask, dtype=np.int16)
    for dy in range(3):
        for dx in range(3):
            neighborhood += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    selected = neighborhood >= 5
    if require_valid_depth:
        selected &= np.isfinite(depth_array) & (depth_array > 0.0)
    y, x = np.indices(depth_array.shape, dtype=np.float32)
    cx = np.float32(depth_array.shape[1] / 2.0)
    cy = np.float32(depth_array.shape[0] / 2.0)
    fx = np.float32(cx / np.tan(float(fov_x) / 2.0))
    fy = np.float32(cy / np.tan(float(fov_y) / 2.0))
    if historical_axis_order:
        x3 = (y - cx) * depth_array / fx
        y3 = (x - cy) * depth_array / fy
    else:
        x3 = (x - cx) * depth_array / fx
        y3 = (y - cy) * depth_array / fy
    points = np.stack((x3[selected], y3[selected], depth_array[selected]), axis=1)
    eligible = len(points) >= 2 and bool(np.all(np.isfinite(points)))
    value = (
        float(
            np.linalg.norm(
                np.float32(2.0) * np.std(points, axis=0, ddof=1, dtype=np.float32)
            )
        )
        if eligible
        else None
    )
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        eligible = False
        value = None
    return {
        "eligible": bool(eligible),
        "selected_pixel_count": len(points),
        "raw_scale_scene_units": value,
    }


def choose_grid_oracle(
    rows: Sequence[Mapping[str, Any]], *, uniform_scale: float
) -> Mapping[str, Any]:
    """Choose highest IoU, then closest to U, then smaller scale."""

    if not rows:
        raise ValueError("grid oracle requires at least one candidate")
    return min(
        rows,
        key=lambda row: (
            -float(row["iou"]),
            abs(float(row["scale_input"]) - float(uniform_scale)),
            float(row["scale_input"]),
        ),
    )


def _prompt_maps(root: Path, scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_payload = load_json(root / "prompts" / scene_id / "runtime_prompts.json")
    evaluation_payload = load_json(
        root / "prompts" / scene_id / "evaluation_prompts.json"
    )
    runtime = {str(row["prompt_id"]): row for row in runtime_payload["prompts"]}
    evaluation = {str(row["prompt_id"]): row for row in evaluation_payload["prompts"]}
    if set(runtime) != set(evaluation):
        raise ValueError(f"{scene_id}: runtime/evaluation prompt IDs differ")
    return runtime, evaluation


def _load_binary_pair(
    path: Path, expected_points: int
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as payload:
        uniform = np.asarray(payload["U_global"])
        data = np.asarray(payload["D_class"])
    if uniform.shape != (expected_points,) or data.shape != (expected_points,):
        raise ValueError(f"invalid mask shape: {path}")
    for name, value in (("U_global", uniform), ("D_class", data)):
        unique = set(np.unique(value).tolist())
        if not unique <= {0, 1}:
            raise ValueError(f"{path}: {name} is not binary: {unique}")
    return uniform.astype(bool), data.astype(bool)


@lru_cache(maxsize=256)
def _gate_vector_cached(scale_gate: str, scale: float) -> tuple[float, ...]:
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Sigmoid())
    model.load_state_dict(torch.load(scale_gate, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        value = model(torch.tensor([float(scale)], dtype=torch.float32)).squeeze(0)
    return tuple(value.numpy().astype(np.float64).tolist())


def _gate_vector(scale_gate: Path, scale: float) -> np.ndarray:
    return np.asarray(
        _gate_vector_cached(str(scale_gate.resolve()), float(scale)), dtype=np.float64
    )


def _validate_old_result(
    *,
    path: Path,
    expected_points: int,
    prompt: Mapping[str, Any],
    feature_ply: Path,
    scale_gate: Path,
    scale_u: float,
    scale_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    uniform, data = _load_binary_pair(path, expected_points)
    metadata = load_json(path.with_suffix(".json"))
    if (
        metadata.get("kind") != "prompt_prior_mask_pair"
        or metadata.get("status") != "complete"
        or metadata.get("prompt") != dict(prompt)
    ):
        raise ValueError(f"old result identity mismatch: {path}")
    if not math.isclose(
        float(metadata.get("similarity_threshold", float("nan"))),
        SIMILARITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"old similarity threshold mismatch: {path}")
    if Path(str(metadata.get("feature_ply", ""))).resolve() != feature_ply.resolve():
        raise ValueError(f"old feature path mismatch: {path}")
    if Path(str(metadata.get("scale_gate", ""))).resolve() != scale_gate.resolve():
        raise ValueError(f"old scale-gate path mismatch: {path}")
    for condition, expected in (("U-global", scale_u), ("D-class", scale_d)):
        row = metadata["conditions"][condition]
        if not math.isclose(float(row["scale_input"]), expected, abs_tol=1e-12):
            raise ValueError(f"old scale input mismatch: {path}/{condition}")
        recomputed = _gate_vector(scale_gate, expected)
        if not np.allclose(recomputed, np.asarray(row["gate"]), atol=1e-6, rtol=0.0):
            raise ValueError(f"old gate vector mismatch: {path}/{condition}")
    return uniform, data


def audit_directions(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prompts_root: Path,
    masks_root: Path,
    parameters_path: Path,
    scene_ids: Sequence[str],
    table_output: Path,
    analysis_output: Path,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    parameters = load_json(parameters_path)
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        runtime, evaluation = _prompt_maps(prompts_root, scene_id)
        gt_xyz, gt = load_ground_truth_npz(_gt_path(scene, gt_dir, scene_id), scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["feature_ply"]), _transform(scene)
        )
        gaussian_to_gt, gaussian_distance = _nearest(gaussian_xyz, gt_xyz)
        scene_parameters = parameters["scenes"][scene_id]
        for prompt_id in sorted(runtime):
            prompt = runtime[prompt_id]
            target = evaluation[prompt_id]
            scale_u = float(scene_parameters["global_scale_input"])
            scale_d = float(
                scene_parameters["class_scale_inputs"].get(
                    str(prompt["class_name"]), scale_u
                )
            )
            uniform, data = _validate_old_result(
                path=masks_root / scene_id / f"{prompt_id}.npz",
                expected_points=len(gaussian_xyz),
                prompt=prompt,
                feature_ply=assets["feature_ply"],
                scale_gate=assets["scale_gate"],
                scale_u=scale_u,
                scale_d=scale_d,
            )
            base: dict[str, Any] = {
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "class_name": str(target["class_name"]),
                "class_id": int(target["class_id"]),
                "gt_instance_id": int(target["gt_instance_id"]),
                "scale_input_u": scale_u,
                "scale_input_d": scale_d,
            }
            for radius in RADIUS_SENSITIVITY_M:
                result = classify_direction_change(
                    uniform,
                    data,
                    gaussian_to_gt_index=gaussian_to_gt,
                    gaussian_to_gt_distance_m=gaussian_distance,
                    gt_semantic=gt.semantic,
                    gt_instance=gt.instance,
                    target_class_id=int(target["class_id"]),
                    target_instance_id=int(target["gt_instance_id"]),
                    radius_m=radius,
                )
                suffix = f"{round(radius * 100):02d}cm"
                base.update({f"{key}_{suffix}": value for key, value in result.items()})
            rows.append(base)
    write_rows(table_output, rows)
    if tuple(scene_ids) == DEV8 and len(rows) != 124:
        raise ValueError(
            f"registered DEV8 direction audit expected 124 objects, got {len(rows)}"
        )
    scene_rows = []
    for scene_id in scene_ids:
        selected = [row for row in rows if row["scene_id"] == scene_id]
        scene_row: dict[str, Any] = {
            "scene_id": scene_id,
            "object_count": len(selected),
        }
        for radius in RADIUS_SENSITIVITY_M:
            suffix = f"{round(radius * 100):02d}cm"
            ratios = [
                float(row[f"help_ratio_{suffix}"])
                for row in selected
                if row[f"help_ratio_{suffix}"] is not None
            ]
            scene_row.update(
                {
                    f"changed_object_count_{suffix}": len(ratios),
                    f"mean_object_help_ratio_{suffix}": float(np.mean(ratios))
                    if ratios
                    else None,
                    f"mean_object_direction_{suffix}": float(
                        np.mean([float(row[f"direction_{suffix}"]) for row in selected])
                    )
                    if selected
                    else 0.0,
                }
            )
        scene_rows.append(scene_row)
    available_scene_ratios = [
        float(row["mean_object_help_ratio_05cm"])
        for row in scene_rows
        if row["mean_object_help_ratio_05cm"] is not None
    ]
    scene_equal_ratio = (
        float(np.mean(available_scene_ratios)) if available_scene_ratios else None
    )
    positive = sum(float(row["mean_object_direction_05cm"]) > 0.0 for row in scene_rows)
    negative = sum(float(row["mean_object_direction_05cm"]) < 0.0 for row in scene_rows)
    if scene_equal_ratio is not None and scene_equal_ratio >= 0.60 and positive >= 5:
        conclusion = "directionally-beneficial"
    elif scene_equal_ratio is not None and scene_equal_ratio <= 0.40 and negative >= 5:
        conclusion = "directionally-harmful"
    else:
        conclusion = "mixed-or-boundary-perturbation"
    sensitivity = {}
    for radius in RADIUS_SENSITIVITY_M:
        suffix = f"{round(radius * 100):02d}cm"
        ratios = [
            float(row[f"mean_object_help_ratio_{suffix}"])
            for row in scene_rows
            if row[f"mean_object_help_ratio_{suffix}"] is not None
        ]
        sensitivity[suffix] = {
            "scene_equal_help_ratio": float(np.mean(ratios)) if ratios else None,
            "positive_scene_count": sum(
                float(row[f"mean_object_direction_{suffix}"]) > 0.0
                for row in scene_rows
            ),
            "negative_scene_count": sum(
                float(row[f"mean_object_direction_{suffix}"]) < 0.0
                for row in scene_rows
            ),
        }
    analysis = {
        "kind": "prompt_prior_direction_audit",
        "status": "complete",
        "main_radius_m": 0.05,
        "sensitivity_radii_m": list(RADIUS_SENSITIVITY_M),
        "scene_ids": list(scene_ids),
        "object_count": len(rows),
        "scene_results": scene_rows,
        "radius_sensitivity": sensitivity,
        "scene_equal_help_ratio_05cm": scene_equal_ratio,
        "positive_scene_count": positive,
        "negative_scene_count": negative,
        "direction_conclusion": conclusion,
        "does_not_gate_capacity_control": True,
        "producer_commit": _producer_commit(),
    }
    write_json(analysis_output, analysis)
    return analysis


def prepare_capacity(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prompts_root: Path,
    parameters_path: Path,
    scene_ids: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    """Materialize GT-derived O-instance scales; never writes prediction masks."""

    import torch

    from gaussian_renderer import render_with_depth, render_with_max_contributor

    scenes = load_scene_runtime_manifest(runtime_manifest)
    parameters = load_json(parameters_path)
    scene_rows: dict[str, Any] = {}
    run_identity = {
        "producer_commit": _producer_commit(),
        "runtime_manifest": _file_identity(runtime_manifest),
        "parameters": _file_identity(parameters_path),
        "prompts_root": str(prompts_root.resolve()),
        "gt_dir": str(gt_dir.resolve()),
    }
    if output.is_file():
        try:
            existing = load_json(output)
            if (
                existing.get("kind") == "prompt_prior_scale_capacity_plan"
                and existing.get("formula_version") == FORMULA_VERSION
                and existing.get("scene_ids") == list(scene_ids)
                and existing.get("run_identity") == run_identity
            ):
                scene_rows.update(existing.get("scenes", {}))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        runtime, evaluation = _prompt_maps(prompts_root, scene_id)
        gt_path = _gt_path(scene, gt_dir, scene_id)
        existing_scene = scene_rows.get(scene_id)
        if existing_scene is not None:
            existing_prompts = {
                str(row.get("prompt_id")): {
                    "prompt": row.get("prompt"),
                    "target": row.get("target"),
                }
                for row in existing_scene.get("prompts", [])
            }
            if (
                existing_prompts
                == {
                    prompt_id: {
                        "prompt": dict(prompt),
                        "target": dict(evaluation[prompt_id]),
                    }
                    for prompt_id, prompt in runtime.items()
                }
                and int(existing_scene.get("point_count", -1))
                == len(load_ply_xyz(assets["point_cloud"]))
                and _file_identity_matches(
                    existing_scene.get("point_cloud"), assets["point_cloud"]
                )
                and _file_identity_matches(
                    existing_scene.get("feature_ply"), assets["feature_ply"]
                )
                and _file_identity_matches(
                    existing_scene.get("scale_gate"), assets["scale_gate"]
                )
                and _file_identity_matches(
                    existing_scene.get("mask_scales"), assets["mask_scales"]
                )
                and _file_identity_matches(existing_scene.get("gt"), gt_path)
            ):
                continue
        gt_xyz, gt = load_ground_truth_npz(gt_path, scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["point_cloud"]), _transform(scene)
        )
        gaussian_to_gt, gaussian_distance = _nearest(gaussian_xyz, gt_xyz)
        model = _scene_model(assets["point_cloud"])
        # Historical readColmapSceneInfo sorts camera infos by image name
        # before get_scale.py reuses cameras[0]'s FoV for every view.
        camera_list = sorted(_camera_list(assets), key=lambda camera: camera.image_name)
        cameras = {camera.image_name: camera for camera in camera_list}
        first_camera = camera_list[0]
        pipe = _pipeline()
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        per_camera: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        prompt_rows: list[dict[str, Any]] = []
        raw_scales = _mask_scale_values(assets["mask_scales"])
        metres_per_unit = float(scene.get("scene_scale_m_per_unit", 1.0))
        physical_scales = raw_scales * metres_per_unit
        for prompt_id in sorted(runtime):
            prompt = runtime[prompt_id]
            target = evaluation[prompt_id]
            image_name = str(prompt["image_name"])
            camera = cameras[image_name]
            if image_name not in per_camera:
                with torch.no_grad():
                    contributor_result = render_with_max_contributor(
                        camera, model, pipe, background
                    )
                    depth_result = render_with_depth(camera, model, pipe, background)
                contributor = (
                    contributor_result["max_contributor"]
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                weight = (
                    contributor_result["max_contribute"]
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                depth = depth_result["depth"].detach().cpu().numpy().squeeze()
                per_camera[image_name] = (contributor, weight, depth)
            contributor, weight, depth = per_camera[image_name]
            target_gaussian = np.zeros(len(gaussian_xyz), dtype=bool)
            supported = gaussian_distance <= PROMPT_RADIUS_M
            nearest = gaussian_to_gt[supported]
            target_gaussian[supported] = (
                gt.semantic[nearest] == int(target["class_id"])
            ) & (gt.instance[nearest] == int(target["gt_instance_id"]))
            valid_pixel = (
                (contributor >= 0)
                & (contributor < len(target_gaussian))
                & (weight > 0.0)
            )
            footprint = np.zeros(contributor.shape, dtype=bool)
            footprint[valid_pixel] = target_gaussian[contributor[valid_pixel]]
            historical = native_visible_mask_scale(
                depth,
                footprint,
                fov_x=float(first_camera.FoVx),
                fov_y=float(first_camera.FoVy),
                historical_axis_order=True,
                require_valid_depth=False,
            )
            corrected = native_visible_mask_scale(
                depth,
                footprint,
                fov_x=float(camera.FoVx),
                fov_y=float(camera.FoVy),
                historical_axis_order=False,
                require_valid_depth=True,
            )
            native_m = (
                float(historical["raw_scale_scene_units"]) * metres_per_unit
                if historical["eligible"]
                else None
            )
            scale_input = (
                training_quantile_uniform(physical_scales, native_m)
                if native_m is not None
                else None
            )
            prompt_rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": dict(prompt),
                    "target": dict(target),
                    "gt_derived_oracle": True,
                    "formula_version": FORMULA_VERSION,
                    "historical_native_scale": historical,
                    "historical_native_scale_m": native_m,
                    "o_instance_scale_input": scale_input,
                    "corrected_geometry_diagnostic": corrected,
                }
            )
        scene_rows[scene_id] = {
            "point_count": len(gaussian_xyz),
            "point_cloud": _file_identity(assets["point_cloud"]),
            "feature_ply": _file_identity(assets["feature_ply"]),
            "scale_gate": _file_identity(assets["scale_gate"]),
            "mask_scales": _file_identity(assets["mask_scales"]),
            "gt": _file_identity(gt_path),
            "scene_scale_m_per_unit": metres_per_unit,
            "historical_first_camera": {
                "image_name": str(first_camera.image_name),
                "fov_x": float(first_camera.FoVx),
                "fov_y": float(first_camera.FoVy),
            },
            "global_scale_input": float(
                parameters["scenes"][scene_id]["global_scale_input"]
            ),
            "class_scale_inputs": dict(
                parameters["scenes"][scene_id]["class_scale_inputs"]
            ),
            "prompts": prompt_rows,
        }
        write_json(
            output,
            {
                "kind": "prompt_prior_scale_capacity_plan",
                "status": "running",
                "gt_derived_oracle": True,
                "formula_version": FORMULA_VERSION,
                "scene_ids": list(scene_ids),
                "initial_grid": list(GRID5),
                "registered_supplemental_grid": list(GRID_SUPPLEMENT),
                "run_identity": run_identity,
                "scenes": scene_rows,
            },
        )
        del model
        torch.cuda.empty_cache()
    payload = {
        "kind": "prompt_prior_scale_capacity_plan",
        "status": "complete",
        "gt_derived_oracle": True,
        "formula_version": FORMULA_VERSION,
        "scene_ids": list(scene_ids),
        "initial_grid": list(GRID5),
        "registered_supplemental_grid": list(GRID_SUPPLEMENT),
        "run_identity": run_identity,
        "scenes": scene_rows,
    }
    write_json(output, payload)
    return payload


def _capacity_complete(
    path: Path,
    *,
    expected_points: int,
    expected_prompt: Mapping[str, Any],
    expected_scales: Sequence[float],
    expect_o_instance: bool,
    feature_ply: Path,
    scale_gate: Path,
    o_instance_scale_input: float | None,
) -> bool:
    try:
        _load_capacity_result(
            path,
            expected_points=expected_points,
            expected_prompt=expected_prompt,
            expected_scales=expected_scales,
            expect_o_instance=expect_o_instance,
            feature_ply=feature_ply,
            scale_gate=scale_gate,
            o_instance_scale_input=o_instance_scale_input,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
        return False
    return True


def _load_capacity_result(
    path: Path,
    *,
    expected_points: int,
    expected_prompt: Mapping[str, Any],
    expected_scales: Sequence[float],
    expect_o_instance: bool,
    feature_ply: Path,
    scale_gate: Path,
    o_instance_scale_input: float | None,
) -> dict[str, np.ndarray]:
    """Load one capacity result only after validating its full experiment identity."""

    metadata = load_json(path.with_suffix(".json"))
    expected = {scale_key(value) for value in expected_scales}
    if expect_o_instance:
        expected.add("O_instance")
    if (
        metadata.get("kind") != "prompt_prior_scale_capacity_masks"
        or metadata.get("status") != "complete"
        or metadata.get("prompt") != dict(expected_prompt)
        or metadata.get("formula_version") != FORMULA_VERSION
        or set(metadata.get("completed_keys", [])) < expected
    ):
        raise ValueError(f"capacity result identity mismatch: {path}")
    if not math.isclose(
        float(metadata.get("similarity_threshold", float("nan"))),
        SIMILARITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"capacity similarity threshold mismatch: {path}")
    if Path(str(metadata.get("feature_ply", ""))).resolve() != feature_ply.resolve():
        raise ValueError(f"capacity feature path mismatch: {path}")
    if Path(str(metadata.get("scale_gate", ""))).resolve() != scale_gate.resolve():
        raise ValueError(f"capacity scale-gate path mismatch: {path}")
    producer = _producer_commit()
    if producer is not None and metadata.get("producer_commit") != producer:
        raise ValueError(f"capacity producer commit mismatch: {path}")
    scale_map = metadata.get("grid_scales", {})
    gates = metadata.get("gates", {})
    for value in expected_scales:
        key = scale_key(value)
        if not math.isclose(
            float(scale_map.get(key, float("nan"))), value, abs_tol=1e-12
        ):
            raise ValueError(f"capacity scale mapping mismatch: {path}/{key}")
        if not np.allclose(
            _gate_vector(scale_gate, value),
            np.asarray(gates.get(key)),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"capacity gate vector mismatch: {path}/{key}")
    if expect_o_instance:
        if o_instance_scale_input is None or not math.isclose(
            float(metadata.get("o_instance_scale_input", float("nan"))),
            float(o_instance_scale_input),
            abs_tol=1e-12,
        ):
            raise ValueError(f"capacity O-instance scale mismatch: {path}")
        if not np.allclose(
            _gate_vector(scale_gate, float(o_instance_scale_input)),
            np.asarray(gates.get("O_instance")),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"capacity O-instance gate mismatch: {path}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path) as payload:
        if not expected <= set(payload.files):
            raise ValueError(f"capacity result is incomplete: {path}")
        for key in expected:
            value = np.asarray(payload[key])
            if value.shape != (expected_points,) or not set(
                np.unique(value).tolist()
            ) <= {0, 1}:
                raise ValueError(f"capacity mask is invalid: {path}/{key}")
            arrays[key] = value.astype(bool)
    return arrays


def segment_capacity(
    *,
    runtime_manifest: Path,
    plan_path: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    grid_values: Sequence[float],
) -> dict[str, Any]:
    """Run fixed gate inputs.  Intentionally accepts no GT path."""

    import torch
    from torch.nn import functional

    from gaussian_renderer import render_contrastive_feature

    scenes = load_scene_runtime_manifest(runtime_manifest)
    plan = load_json(plan_path)
    plan_run_identity = plan.get("run_identity", {})
    if (
        plan.get("kind") != "prompt_prior_scale_capacity_plan"
        or plan.get("status") != "complete"
        or plan.get("gt_derived_oracle") is not True
        or plan.get("formula_version") != FORMULA_VERSION
        or plan.get("scene_ids") != list(scene_ids)
        or not _file_identity_matches(
            plan_run_identity.get("runtime_manifest"), runtime_manifest
        )
        or (
            _producer_commit() is not None
            and plan.get("run_identity", {}).get("producer_commit")
            != _producer_commit()
        )
    ):
        raise ValueError("capacity plan must be explicitly marked GT-derived oracle")
    completed = 0
    reused = 0
    requested_scales = tuple(sorted({float(value) for value in grid_values}))
    for value in requested_scales:
        if value not in GRID9:
            raise ValueError(f"unregistered capacity grid value: {value}")
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        scene_plan = plan["scenes"][scene_id]
        for name in ("point_cloud", "feature_ply", "scale_gate", "mask_scales"):
            if not _file_identity_matches(scene_plan.get(name), assets[name]):
                raise ValueError(f"{scene_id}: capacity plan {name} identity changed")
        prompt_rows = {str(row["prompt_id"]): row for row in scene_plan["prompts"]}
        feature_xyz = load_ply_xyz(assets["feature_ply"])
        scene_xyz = load_ply_xyz(assets["point_cloud"])
        if feature_xyz.shape != scene_xyz.shape or not np.allclose(
            feature_xyz, scene_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{scene_id}: RGB and feature Gaussian XYZ/order differ")
        pending = []
        scene_output = output_root / scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        for prompt_id, row in sorted(prompt_rows.items()):
            target = scene_output / f"{prompt_id}.npz"
            eligible = row.get("o_instance_scale_input") is not None
            if _capacity_complete(
                target,
                expected_points=len(feature_xyz),
                expected_prompt=row["prompt"],
                expected_scales=requested_scales,
                expect_o_instance=eligible,
                feature_ply=assets["feature_ply"],
                scale_gate=assets["scale_gate"],
                o_instance_scale_input=row.get("o_instance_scale_input"),
            ):
                reused += 1
            else:
                pending.append(row)
        if not pending:
            continue
        model = _feature_model(assets["feature_ply"])
        gate_model = torch.nn.Sequential(
            torch.nn.Linear(1, 32, bias=True), torch.nn.Sigmoid()
        ).cuda()
        gate_model.load_state_dict(torch.load(assets["scale_gate"]))
        gate_model.eval()
        cameras = {camera.image_name: camera for camera in _camera_list(assets)}
        point_features = model.get_point_features.detach()
        background = torch.zeros(32, dtype=torch.float32, device="cuda")
        pipe = _pipeline()
        query_raw: dict[str, Any] = {}
        by_camera: dict[str, list[Mapping[str, Any]]] = {}
        for row in pending:
            by_camera.setdefault(str(row["prompt"]["image_name"]), []).append(row)
        for image_name, camera_rows in sorted(by_camera.items()):
            camera = deepcopy(cameras[image_name])
            camera.feature_height = camera.image_height
            camera.feature_width = camera.image_width
            with torch.no_grad():
                rendered = render_contrastive_feature(
                    camera, model, pipe, background, norm_point_features=True
                )["render"].detach()
            for row in camera_rows:
                prompt = row["prompt"]
                x, y = int(prompt["x"]), int(prompt["y"])
                query_raw[str(row["prompt_id"])] = rendered[:, y, x].clone()
            del rendered
        masks_by_prompt: dict[str, dict[str, np.ndarray]] = {
            str(row["prompt_id"]): {} for row in pending
        }
        gate_by_key: dict[str, list[float]] = {}
        for value in requested_scales:
            key = scale_key(value)
            with torch.no_grad():
                gate = gate_model(
                    torch.tensor([value], dtype=torch.float32, device="cuda")
                ).squeeze(0)
                points = functional.normalize(
                    point_features * gate.unsqueeze(0), dim=1, p=2
                )
                for row in pending:
                    prompt_id = str(row["prompt_id"])
                    query = functional.normalize(
                        query_raw[prompt_id] * gate, dim=0, p=2
                    )
                    mask = (points @ query) > SIMILARITY_THRESHOLD
                    masks_by_prompt[prompt_id][key] = (
                        mask.detach().cpu().numpy().astype(np.uint8)
                    )
            gate_by_key[key] = gate.detach().cpu().numpy().astype(float).tolist()
            del points
        for row in pending:
            prompt_id = str(row["prompt_id"])
            value = row.get("o_instance_scale_input")
            if value is None:
                continue
            with torch.no_grad():
                gate = gate_model(
                    torch.tensor([float(value)], dtype=torch.float32, device="cuda")
                ).squeeze(0)
                points = functional.normalize(
                    point_features * gate.unsqueeze(0), dim=1, p=2
                )
                query = functional.normalize(query_raw[prompt_id] * gate, dim=0, p=2)
                mask = (points @ query) > SIMILARITY_THRESHOLD
            masks_by_prompt[prompt_id]["O_instance"] = (
                mask.detach().cpu().numpy().astype(np.uint8)
            )
            gate_by_key[f"O_instance:{prompt_id}"] = (
                gate.detach().cpu().numpy().astype(float).tolist()
            )
            del points
        for row in pending:
            prompt_id = str(row["prompt_id"])
            target = scene_output / f"{prompt_id}.npz"
            existing: dict[str, np.ndarray] = {}
            existing_metadata: dict[str, Any] = {}
            if target.is_file():
                try:
                    with np.load(target) as payload:
                        existing = {
                            key: np.asarray(payload[key]) for key in payload.files
                        }
                    existing_metadata = load_json(target.with_suffix(".json"))
                    base_identity_matches = (
                        existing_metadata.get("kind")
                        == "prompt_prior_scale_capacity_masks"
                        and existing_metadata.get("prompt") == row["prompt"]
                        and existing_metadata.get("formula_version") == FORMULA_VERSION
                        and math.isclose(
                            float(
                                existing_metadata.get(
                                    "similarity_threshold", float("nan")
                                )
                            ),
                            SIMILARITY_THRESHOLD,
                            abs_tol=0.0,
                        )
                        and Path(
                            str(existing_metadata.get("feature_ply", ""))
                        ).resolve()
                        == assets["feature_ply"].resolve()
                        and Path(str(existing_metadata.get("scale_gate", ""))).resolve()
                        == assets["scale_gate"].resolve()
                        and (
                            _producer_commit() is None
                            or existing_metadata.get("producer_commit")
                            == _producer_commit()
                        )
                    )
                    if not base_identity_matches:
                        existing = {}
                        existing_metadata = {}
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    existing = {}
                    existing_metadata = {}
            existing.update(masks_by_prompt[prompt_id])
            temporary = target.with_suffix(".part.npz")
            np.savez_compressed(temporary, **existing)
            temporary.replace(target)
            completed_keys = sorted(existing)
            all_gates = dict(existing_metadata.get("gates", {}))
            for key in completed_keys:
                if key in gate_by_key:
                    all_gates[key] = gate_by_key[key]
            oracle_gate_key = f"O_instance:{prompt_id}"
            if "O_instance" in completed_keys and oracle_gate_key in gate_by_key:
                all_gates["O_instance"] = gate_by_key[oracle_gate_key]
            write_json(
                target.with_suffix(".json"),
                {
                    "kind": "prompt_prior_scale_capacity_masks",
                    "status": "complete",
                    "scene_id": scene_id,
                    "prompt": row["prompt"],
                    "feature_ply": str(assets["feature_ply"]),
                    "scale_gate": str(assets["scale_gate"]),
                    "similarity_threshold": SIMILARITY_THRESHOLD,
                    "formula_version": FORMULA_VERSION,
                    "producer_commit": _producer_commit(),
                    "gt_derived_o_instance": row.get("o_instance_scale_input")
                    is not None,
                    "o_instance_scale_input": row.get("o_instance_scale_input"),
                    "grid_scales": {
                        scale_key(value): value
                        for value in GRID9
                        if scale_key(value) in existing
                    },
                    "gates": all_gates,
                    "completed_keys": completed_keys,
                },
            )
            completed += 1
        del model, gate_model
        torch.cuda.empty_cache()
    result = {
        "kind": "prompt_prior_scale_capacity_segmentation",
        "status": "complete",
        "scene_ids": list(scene_ids),
        "grid_values": list(requested_scales),
        "completed_prompts": completed,
        "reused_prompts": reused,
        "producer_commit": _producer_commit(),
    }
    write_json(output_root / "segmentation.json", result)
    return result


def _size_bin(diagonal_m: float, spec: Mapping[str, Any] | None) -> str | None:
    if spec is None:
        return None
    boundaries = spec["boundaries_m"]
    if diagonal_m <= float(boundaries["tiny_max_m"]):
        return "tiny"
    if diagonal_m <= float(boundaries["small_max_m"]):
        return "small"
    if diagonal_m <= float(boundaries["medium_max_m"]):
        return "medium"
    return "large"


def evaluate_capacity(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prompts_root: Path,
    old_masks_root: Path,
    capacity_masks_root: Path,
    plan_path: Path,
    scene_ids: Sequence[str],
    grid_values: Sequence[float],
    table_output: Path,
    analysis_output: Path,
    size_bins: Path | None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    plan = load_json(plan_path)
    plan_run_identity = plan.get("run_identity", {})
    if (
        plan.get("kind") != "prompt_prior_scale_capacity_plan"
        or plan.get("status") != "complete"
        or plan.get("formula_version") != FORMULA_VERSION
        or plan.get("scene_ids") != list(scene_ids)
        or not _file_identity_matches(
            plan_run_identity.get("runtime_manifest"), runtime_manifest
        )
        or plan_run_identity.get("prompts_root") != str(prompts_root.resolve())
        or plan_run_identity.get("gt_dir") != str(gt_dir.resolve())
        or (
            _producer_commit() is not None
            and plan.get("run_identity", {}).get("producer_commit")
            != _producer_commit()
        )
    ):
        raise ValueError("capacity plan identity mismatch")
    size_spec = load_json(size_bins) if size_bins else None
    grid = tuple(sorted({float(value) for value in grid_values}))
    rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        scene_plan = plan["scenes"][scene_id]
        for name in ("point_cloud", "feature_ply", "scale_gate", "mask_scales"):
            if not _file_identity_matches(scene_plan.get(name), assets[name]):
                raise ValueError(f"{scene_id}: capacity plan {name} identity changed")
        if not _file_identity_matches(
            scene_plan.get("gt"), _gt_path(scene, gt_dir, scene_id)
        ):
            raise ValueError(f"{scene_id}: capacity plan GT identity changed")
        runtime, evaluation = _prompt_maps(prompts_root, scene_id)
        gt_xyz, gt = load_ground_truth_npz(_gt_path(scene, gt_dir, scene_id), scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["feature_ply"]), _transform(scene)
        )
        gt_to_gaussian, gt_distance = _nearest(gt_xyz, gaussian_xyz)
        gaussian_to_gt, gaussian_distance = _nearest(gaussian_xyz, gt_xyz)
        plan_rows = {
            str(row["prompt_id"]): row for row in plan["scenes"][scene_id]["prompts"]
        }
        for prompt_id in sorted(runtime):
            prompt = runtime[prompt_id]
            target = evaluation[prompt_id]
            plan_row = plan_rows[prompt_id]
            if plan_row.get("prompt") != dict(prompt) or plan_row.get("target") != dict(
                target
            ):
                raise ValueError(
                    f"{scene_id}/{prompt_id}: frozen capacity prompt/target changed"
                )
            scale_u = float(plan["scenes"][scene_id]["global_scale_input"])
            scale_d = float(
                plan["scenes"][scene_id]["class_scale_inputs"].get(
                    str(prompt["class_name"]), scale_u
                )
            )
            uniform, _ = _validate_old_result(
                path=old_masks_root / scene_id / f"{prompt_id}.npz",
                expected_points=len(gaussian_xyz),
                prompt=prompt,
                feature_ply=assets["feature_ply"],
                scale_gate=assets["scale_gate"],
                scale_u=scale_u,
                scale_d=scale_d,
            )
            o_scale = plan_row.get("o_instance_scale_input")
            capacity = _load_capacity_result(
                capacity_masks_root / scene_id / f"{prompt_id}.npz",
                expected_points=len(gaussian_xyz),
                expected_prompt=prompt,
                expected_scales=grid,
                expect_o_instance=o_scale is not None,
                feature_ply=assets["feature_ply"],
                scale_gate=assets["scale_gate"],
                o_instance_scale_input=o_scale,
            )
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
            uniform_metrics = evaluate_prompt_pair_arrays(mask=uniform, **common)
            base_row = {
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "class_name": str(target["class_name"]),
                "class_id": int(target["class_id"]),
                "gt_instance_id": int(target["gt_instance_id"]),
                "size_bin": _size_bin(float(target["bbox_diagonal_m"]), size_spec),
            }
            rows.append(
                {
                    **base_row,
                    "condition": "U-global",
                    "scale_input": scale_u,
                    **uniform_metrics,
                }
            )
            grid_candidates: list[dict[str, Any]] = [
                {
                    "condition": "U-global",
                    "scale_input": scale_u,
                    **uniform_metrics,
                }
            ]
            for value in grid:
                key = scale_key(value)
                if key not in capacity:
                    raise ValueError(
                        f"missing capacity mask {scene_id}/{prompt_id}/{key}"
                    )
                metrics = evaluate_prompt_pair_arrays(mask=capacity[key], **common)
                row = {
                    **base_row,
                    "condition": key,
                    "scale_input": float(value),
                    **metrics,
                }
                rows.append(row)
                grid_candidates.append(row)
            best = choose_grid_oracle(
                grid_candidates,
                uniform_scale=scale_u,
            )
            rows.append(
                {
                    **base_row,
                    "condition": "GridOracle",
                    "selected_condition": str(best["condition"]),
                    "scale_input": float(best["scale_input"]),
                    **{
                        key: value
                        for key, value in best.items()
                        if key not in {"condition", "scale_input"}
                    },
                }
            )
            oracle_metrics = None
            if o_scale is not None:
                if "O_instance" not in capacity:
                    raise ValueError(f"missing O-instance mask {scene_id}/{prompt_id}")
                oracle_metrics = evaluate_prompt_pair_arrays(
                    mask=capacity["O_instance"], **common
                )
                rows.append(
                    {
                        **base_row,
                        "condition": "O-instance",
                        "scale_input": float(o_scale),
                        **oracle_metrics,
                    }
                )
            object_rows.append(
                {
                    "scene_id": scene_id,
                    "prompt_id": prompt_id,
                    "class_name": str(target["class_name"]),
                    "size_bin": _size_bin(float(target["bbox_diagonal_m"]), size_spec),
                    "iou_u": float(uniform_metrics["iou"]),
                    "precision_u": float(uniform_metrics["gaussian_precision"]),
                    "recall_u": float(uniform_metrics["gt_recall"]),
                    "grid_best_condition": str(best["condition"]),
                    "grid_best_scale": float(best["scale_input"]),
                    "grid_best_iou": float(best["iou"]),
                    "grid_delta_iou": float(best["iou"] - uniform_metrics["iou"]),
                    "o_instance_eligible": oracle_metrics is not None,
                    "o_instance_iou": float(oracle_metrics["iou"])
                    if oracle_metrics
                    else None,
                    "o_instance_precision": float(oracle_metrics["gaussian_precision"])
                    if oracle_metrics
                    else None,
                    "o_instance_recall": float(oracle_metrics["gt_recall"])
                    if oracle_metrics
                    else None,
                    "o_instance_delta_iou": float(
                        oracle_metrics["iou"] - uniform_metrics["iou"]
                    )
                    if oracle_metrics
                    else None,
                    "o_instance_delta_precision": float(
                        oracle_metrics["gaussian_precision"]
                        - uniform_metrics["gaussian_precision"]
                    )
                    if oracle_metrics
                    else None,
                    "o_instance_delta_recall": float(
                        oracle_metrics["gt_recall"] - uniform_metrics["gt_recall"]
                    )
                    if oracle_metrics
                    else None,
                }
            )
    write_rows(table_output, rows)

    scene_results = []
    for scene_id in scene_ids:
        selected = [row for row in object_rows if row["scene_id"] == scene_id]
        eligible = [row for row in selected if row["o_instance_eligible"]]
        tiny_eligible = [
            row for row in eligible if row["size_bin"] in {"tiny", "small"}
        ]
        scene_results.append(
            {
                "scene_id": scene_id,
                "object_count": len(selected),
                "grid_mean_delta_iou": float(
                    np.mean([row["grid_delta_iou"] for row in selected])
                ),
                "o_instance_eligible_count": len(eligible),
                "o_instance_eligible_fraction": float(len(eligible) / len(selected))
                if selected
                else 0.0,
                "o_instance_mean_delta_iou": float(
                    np.mean([row["o_instance_delta_iou"] for row in eligible])
                )
                if eligible
                else None,
                "o_instance_mean_delta_precision": float(
                    np.mean([row["o_instance_delta_precision"] for row in eligible])
                )
                if eligible
                else None,
                "o_instance_mean_delta_recall": float(
                    np.mean([row["o_instance_delta_recall"] for row in eligible])
                )
                if eligible
                else None,
                "o_instance_tiny_small_count": len(tiny_eligible),
                "o_instance_tiny_small_mean_delta_iou": float(
                    np.mean([row["o_instance_delta_iou"] for row in tiny_eligible])
                )
                if tiny_eligible
                else None,
                "o_instance_tiny_small_mean_delta_recall": float(
                    np.mean([row["o_instance_delta_recall"] for row in tiny_eligible])
                )
                if tiny_eligible
                else None,
            }
        )
    grid_mean = float(np.mean([row["grid_mean_delta_iou"] for row in scene_results]))
    grid_positive_scenes = sum(
        row["grid_mean_delta_iou"] >= 0.01 for row in scene_results
    )
    grid_object_fraction = float(
        np.mean([row["grid_delta_iou"] >= 0.02 for row in object_rows])
    )
    grid_passed = (
        grid_mean >= 0.02 and grid_positive_scenes >= 5 and grid_object_fraction >= 0.25
    )
    eligible_scenes = [
        row for row in scene_results if row["o_instance_mean_delta_iou"] is not None
    ]
    o_mean = (
        float(np.mean([row["o_instance_mean_delta_iou"] for row in eligible_scenes]))
        if eligible_scenes
        else None
    )
    o_precision = (
        float(
            np.mean([row["o_instance_mean_delta_precision"] for row in eligible_scenes])
        )
        if eligible_scenes
        else None
    )
    o_positive = sum(row["o_instance_mean_delta_iou"] > 0 for row in eligible_scenes)
    tiny_scene_results = [
        {
            "scene_id": row["scene_id"],
            "object_count": row["o_instance_tiny_small_count"],
            "mean_delta_iou": row["o_instance_tiny_small_mean_delta_iou"],
            "mean_delta_recall": row["o_instance_tiny_small_mean_delta_recall"],
        }
        for row in scene_results
        if row["o_instance_tiny_small_count"] > 0
    ]
    tiny_iou = (
        float(np.mean([row["mean_delta_iou"] for row in tiny_scene_results]))
        if tiny_scene_results
        else None
    )
    tiny_recall = (
        float(np.mean([row["mean_delta_recall"] for row in tiny_scene_results]))
        if tiny_scene_results
        else None
    )
    tiny_signal = (
        tiny_iou is not None
        and tiny_recall is not None
        and (
            (tiny_iou > 0 and tiny_recall >= -0.02)
            or (tiny_recall > 0 and tiny_iou >= -0.02)
        )
    )
    overall_eligible_fraction = float(
        sum(row["o_instance_eligible_count"] for row in scene_results)
        / sum(row["object_count"] for row in scene_results)
    )
    o_coverage_passed = overall_eligible_fraction >= 0.80 and all(
        row["o_instance_eligible_fraction"] >= 0.50 for row in scene_results
    )
    o_passed = (
        o_coverage_passed
        and o_mean is not None
        and o_mean >= 0.02
        and o_positive >= 5
        and o_precision is not None
        and o_precision >= -0.01
        and tiny_signal
    )
    complete_grid = set(grid) == set(GRID9)
    if not grid_passed and not complete_grid:
        decision = "run-registered-nine-point-supplement"
    elif not grid_passed:
        decision = "stop-current-2k-scale-gate-route-no-broad-capacity"
    elif not o_coverage_passed:
        decision = "stop-o-instance-insufficient-coverage"
    elif not o_passed:
        decision = "stop-physical-scale-mapping-not-predictive"
    else:
        decision = "proceed-to-train-only-native-class-scale-holdout"
    analysis = {
        "kind": "prompt_prior_scale_capacity_analysis",
        "status": "complete",
        "gt_derived_oracles": True,
        "scene_ids": list(scene_ids),
        "object_count": len(object_rows),
        "grid_values": list(grid),
        "grid_scene_equal_mean_delta_iou": grid_mean,
        "grid_scenes_at_least_0p01": grid_positive_scenes,
        "grid_object_fraction_at_least_0p02": grid_object_fraction,
        "grid_broad_capacity_passed": grid_passed,
        "o_instance_scene_equal_mean_delta_iou": o_mean,
        "o_instance_positive_scenes": o_positive,
        "o_instance_scene_equal_delta_precision": o_precision,
        "o_instance_overall_eligible_fraction": overall_eligible_fraction,
        "o_instance_coverage_passed": o_coverage_passed,
        "o_instance_tiny_small_delta_iou": tiny_iou,
        "o_instance_tiny_small_delta_recall": tiny_recall,
        "o_instance_tiny_small_scene_results": tiny_scene_results,
        "o_instance_passed": o_passed,
        "scene_results": scene_results,
        "object_results": object_rows,
        "decision": decision,
        "producer_commit": _producer_commit(),
        "size_bins_identity": _file_identity(size_bins) if size_bins else None,
        "conclusion_boundary": "gate capacity and physical-scale mapping diagnostics only; not a deployable category-prior method",
    }
    write_json(analysis_output, analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-directions")
    audit.add_argument("--runtime-manifest", required=True, type=Path)
    audit.add_argument("--gt-dir", required=True, type=Path)
    audit.add_argument("--prompts-root", required=True, type=Path)
    audit.add_argument("--masks-root", required=True, type=Path)
    audit.add_argument("--parameters", required=True, type=Path)
    audit.add_argument("--scene", action="append", required=True)
    audit.add_argument("--table-output", required=True, type=Path)
    audit.add_argument("--analysis-output", required=True, type=Path)

    prepare = commands.add_parser("prepare-capacity")
    prepare.add_argument("--runtime-manifest", required=True, type=Path)
    prepare.add_argument("--gt-dir", required=True, type=Path)
    prepare.add_argument("--prompts-root", required=True, type=Path)
    prepare.add_argument("--parameters", required=True, type=Path)
    prepare.add_argument("--scene", action="append", required=True)
    prepare.add_argument("--output", required=True, type=Path)

    segment = commands.add_parser("segment-capacity")
    segment.add_argument("--runtime-manifest", required=True, type=Path)
    segment.add_argument("--plan", required=True, type=Path)
    segment.add_argument("--output-root", required=True, type=Path)
    segment.add_argument("--scene", action="append", required=True)
    segment.add_argument("--grid", action="append", required=True, type=float)

    evaluate = commands.add_parser("evaluate-capacity")
    evaluate.add_argument("--runtime-manifest", required=True, type=Path)
    evaluate.add_argument("--gt-dir", required=True, type=Path)
    evaluate.add_argument("--prompts-root", required=True, type=Path)
    evaluate.add_argument("--old-masks-root", required=True, type=Path)
    evaluate.add_argument("--capacity-masks-root", required=True, type=Path)
    evaluate.add_argument("--plan", required=True, type=Path)
    evaluate.add_argument("--scene", action="append", required=True)
    evaluate.add_argument("--grid", action="append", required=True, type=float)
    evaluate.add_argument("--table-output", required=True, type=Path)
    evaluate.add_argument("--analysis-output", required=True, type=Path)
    evaluate.add_argument("--size-bins", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit-directions":
        result = audit_directions(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            prompts_root=args.prompts_root,
            masks_root=args.masks_root,
            parameters_path=args.parameters,
            scene_ids=args.scene,
            table_output=args.table_output,
            analysis_output=args.analysis_output,
        )
    elif args.command == "prepare-capacity":
        result = prepare_capacity(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            prompts_root=args.prompts_root,
            parameters_path=args.parameters,
            scene_ids=args.scene,
            output=args.output,
        )
    elif args.command == "segment-capacity":
        result = segment_capacity(
            runtime_manifest=args.runtime_manifest,
            plan_path=args.plan,
            output_root=args.output_root,
            scene_ids=args.scene,
            grid_values=args.grid,
        )
    else:
        result = evaluate_capacity(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            prompts_root=args.prompts_root,
            old_masks_root=args.old_masks_root,
            capacity_masks_root=args.capacity_masks_root,
            plan_path=args.plan,
            scene_ids=args.scene,
            grid_values=args.grid,
            table_output=args.table_output,
            analysis_output=args.analysis_output,
            size_bins=args.size_bins,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
