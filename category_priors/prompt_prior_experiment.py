from __future__ import annotations

"""Minimal prompt-conditioned category-size experiment.

This module deliberately keeps the public SAGA prompt path intact.  Ground
truth is allowed only in ``prepare`` (to freeze one oracle click per object)
and ``evaluate``.  The ``segment`` command has no GT argument and changes only
the scalar input of the trained scale gate between U-global and D-class.
"""

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, write_json, write_rows
from .prompt_prior import (
    empirical_cdf,
    materialize_prompt_prior,
    training_quantile_uniform,
)
from .runner import load_scene_runtime_manifest
from .taxonomy import default_taxonomy_path, load_taxonomy
from .v9_t1_runner import _resolve_existing_features
from .v9_legacy_runner import _default_point_cloud, _resolve_scene_path


DEV2 = ("scene0645_00", "scene0025_01")
DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
CONDITIONS = ("U-global", "D-class")
SIMILARITY_THRESHOLD = 0.75
PROMPT_RADIUS_M = 0.05


def _scene_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str
) -> Path:
    return _resolve_scene_path(scene, tuple(keys), default).resolve()


def _scene_assets(scene: Mapping[str, Any]) -> dict[str, Path]:
    base = Path(str(scene["base_path"])).resolve()
    feature, gate = _resolve_existing_features(scene)
    point_cloud = (
        _scene_path(scene, ("point_cloud_path",), "")
        if scene.get("point_cloud_path")
        else _default_point_cloud(base).resolve()
    )
    assets = {
        "images": _scene_path(
            scene, ("images_path",), "fastRecon/dense/sparse/0/images"
        ),
        "sparse": _scene_path(scene, ("sparse_path",), "fastRecon/dense/sparse/0"),
        "point_cloud": point_cloud,
        "feature_ply": feature,
        "scale_gate": gate,
        "mask_scales": _scene_path(
            scene, ("grounded_mask_scales_path", "mask_scales_path"),
            "saga/mask_scales",
        ),
    }
    for name in ("images", "sparse", "mask_scales"):
        if not assets[name].is_dir():
            raise FileNotFoundError(f"{name} directory not found: {assets[name]}")
    for name in ("point_cloud", "feature_ply", "scale_gate"):
        if not assets[name].is_file():
            raise FileNotFoundError(f"{name} file not found: {assets[name]}")
    # PMR is registered against the native 2k handoff assets.  Refuse a
    # manifest that silently mixes a later 10k feature/gate with the original
    # Grounded-SAM mask-scale distribution.
    expected = {
        "feature_ply": (base / "saga/contrastive_feature_point_cloud.ply").resolve(),
        "scale_gate": (base / "saga/scale_gate.pt").resolve(),
        "mask_scales": (base / "saga/mask_scales").resolve(),
    }
    for name, expected_path in expected.items():
        if assets[name] != expected_path:
            raise ValueError(
                f"prompt experiment requires native 2k {name}: "
                f"{expected_path}; got {assets[name]}"
            )
    return assets


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _gt_path(scene: Mapping[str, Any], gt_dir: str | Path, scene_id: str) -> Path:
    if scene.get("gt_npz"):
        path = Path(str(scene["gt_npz"]))
        return path if path.is_absolute() else Path(str(scene["base_path"])) / path
    return Path(gt_dir).resolve() / f"{scene_id}.npz"


def _bbox_diagonal(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if not len(xyz):
        return 0.0
    return float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))


def choose_interior_prompt(footprint: np.ndarray) -> tuple[int, int]:
    """Return deterministic ``(x, y)`` furthest from a footprint boundary."""

    mask = np.asarray(footprint, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("footprint must be a non-empty 2-D mask")
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover - cloud dependency
        raise RuntimeError("prompt preparation requires scipy") from exc
    distance = distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(x), int(y)


def _camera_list(assets: Mapping[str, Path]) -> list[Any]:
    from scene.dataset_readers import (
        readColmapCameras,
        read_extrinsics_binary,
        read_extrinsics_text,
        read_intrinsics_binary,
        read_intrinsics_text,
    )
    from utils.camera_utils import cameraList_from_camInfos

    sparse = assets["sparse"]
    try:
        cameras = readColmapCameras(
            read_extrinsics_binary(str(sparse / "images.bin")),
            read_intrinsics_binary(str(sparse / "cameras.bin")),
            str(assets["images"]),
        )
    except (FileNotFoundError, OSError, ValueError):
        cameras = readColmapCameras(
            read_extrinsics_text(str(sparse / "images.txt")),
            read_intrinsics_text(str(sparse / "cameras.txt")),
            str(assets["images"]),
        )
    args = SimpleNamespace(resolution=1, data_device="cuda")
    return cameraList_from_camInfos(cameras, 1, args)


def _scene_model(path: Path) -> Any:
    from scene import GaussianModel

    model = GaussianModel(0)
    model.load_ply(str(path))
    return model


def _feature_model(path: Path) -> Any:
    from scene import FeatureGaussianModel

    model = FeatureGaussianModel(32, 32)
    model.load_ply(str(path))
    return model


def _pipeline() -> SimpleNamespace:
    return SimpleNamespace(
        compute_cov3D_python=False,
        convert_SHs_python=False,
        debug=False,
    )


def _nearest(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - cloud dependency
        raise RuntimeError("prompt experiment requires scipy") from exc
    distance, index = cKDTree(np.asarray(reference, dtype=np.float64)).query(
        np.asarray(query, dtype=np.float64), k=1, workers=-1
    )
    return np.asarray(index, dtype=np.int64), np.asarray(distance, dtype=np.float64)


def _object_rows(
    gt_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    canonical_classes: Sequence[str],
    selected_classes: set[str],
    min_region_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(canonical_classes):
        if class_name not in selected_classes:
            continue
        ids = np.unique(instance[(semantic == class_id) & (instance >= 0)])
        for instance_id in ids:
            mask = (semantic == class_id) & (instance == int(instance_id))
            count = int(np.count_nonzero(mask))
            if count < int(min_region_size):
                continue
            rows.append(
                {
                    "class_id": int(class_id),
                    "class_name": str(class_name),
                    "gt_instance_id": int(instance_id),
                    "gt_point_count": count,
                    "bbox_diagonal_m": _bbox_diagonal(gt_xyz[mask]),
                }
            )
    return sorted(
        rows, key=lambda row: (row["class_name"], row["gt_instance_id"])
    )


def _mark_mechanical(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["bbox_diagonal_m"]),
            rows[index]["class_name"],
            rows[index]["gt_instance_id"],
        ),
    )
    selected = set(ranked[:2] + ranked[-2:])
    for index, row in enumerate(rows):
        row["mechanical_selected"] = index in selected


def _prepare_scene_prompts(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    gt_path: Path,
    output_root: Path,
    canonical_classes: Sequence[str],
    selected_classes: set[str],
    min_region_size: int,
) -> dict[str, Any]:
    """Build frozen prompt/evaluation files; this is the only GT-aware worker."""

    import torch
    from gaussian_renderer import render_with_max_contributor

    scene_root = output_root / "prompts" / scene_id
    runtime_path = scene_root / "runtime_prompts.json"
    evaluation_path = scene_root / "evaluation_prompts.json"
    if runtime_path.is_file() and evaluation_path.is_file():
        runtime = load_json(runtime_path)
        evaluation = load_json(evaluation_path)
        if runtime.get("status") == evaluation.get("status") == "complete":
            return {"runtime": str(runtime_path), "evaluation": str(evaluation_path)}

    assets = _scene_assets(scene)
    gt_xyz, gt = load_ground_truth_npz(gt_path, scene_id)
    scene_xyz = apply_transform(load_ply_xyz(assets["point_cloud"]), _transform(scene))
    feature_xyz = apply_transform(load_ply_xyz(assets["feature_ply"]), _transform(scene))
    if scene_xyz.shape != feature_xyz.shape or not np.allclose(
        scene_xyz, feature_xyz, atol=1e-6, rtol=0.0
    ):
        raise ValueError(f"{scene_id}: RGB and feature Gaussian XYZ/order differ")

    objects = _object_rows(
        gt_xyz,
        gt.semantic,
        gt.instance,
        canonical_classes,
        selected_classes,
        min_region_size,
    )
    target_by_identity = {
        (row["class_id"], row["gt_instance_id"]): index
        for index, row in enumerate(objects)
    }
    nearest_gt, distance = _nearest(scene_xyz, gt_xyz)
    gaussian_target = np.full(len(scene_xyz), -1, dtype=np.int32)
    valid = distance <= PROMPT_RADIUS_M
    for gaussian_id in np.flatnonzero(valid):
        gt_id = int(nearest_gt[gaussian_id])
        key = (int(gt.semantic[gt_id]), int(gt.instance[gt_id]))
        target = target_by_identity.get(key)
        if target is not None:
            gaussian_target[gaussian_id] = int(target)

    model = _scene_model(assets["point_cloud"])
    cameras = _camera_list(assets)
    pipe = _pipeline()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    best: list[dict[str, Any] | None] = [None] * len(objects)
    with torch.no_grad():
        for camera in cameras:
            rendered = render_with_max_contributor(camera, model, pipe, background)
            contributor = rendered["max_contributor"].detach().cpu().numpy().squeeze()
            weight = rendered["max_contribute"].detach().cpu().numpy().squeeze()
            valid_pixel = (
                (contributor >= 0)
                & (contributor < len(gaussian_target))
                & (weight > 0)
            )
            target_image = np.full(contributor.shape, -1, dtype=np.int32)
            target_image[valid_pixel] = gaussian_target[contributor[valid_pixel]]
            visible = target_image[target_image >= 0]
            if not len(visible):
                continue
            counts = np.bincount(visible, minlength=len(objects))
            for target in np.flatnonzero(counts):
                if best[target] is not None and int(counts[target]) <= int(
                    best[target]["visible_pixel_count"]
                ):
                    continue
                x, y = choose_interior_prompt(target_image == int(target))
                best[target] = {
                    "image_name": str(camera.image_name),
                    "x": x,
                    "y": y,
                    "visible_pixel_count": int(counts[target]),
                }

    available: list[dict[str, Any]] = []
    for index, (row, prompt) in enumerate(zip(objects, best)):
        if prompt is None:
            continue
        available.append(
            {
                **row,
                **prompt,
                "prompt_id": f"p{index:04d}",
            }
        )
    _mark_mechanical(available)
    runtime_prompts = [
        {
            "prompt_id": row["prompt_id"],
            "scene_id": scene_id,
            "image_name": row["image_name"],
            "x": row["x"],
            "y": row["y"],
            "class_name": row["class_name"],
            "mechanical_selected": row["mechanical_selected"],
        }
        for row in available
    ]
    evaluation_prompts = [
        {
            "prompt_id": row["prompt_id"],
            "scene_id": scene_id,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "gt_instance_id": row["gt_instance_id"],
            "gt_point_count": row["gt_point_count"],
            "bbox_diagonal_m": row["bbox_diagonal_m"],
            "mechanical_selected": row["mechanical_selected"],
        }
        for row in available
    ]
    write_json(
        runtime_path,
        {
            "kind": "prompt_prior_runtime_prompts",
            "status": "complete",
            "scene_id": scene_id,
            "coordinate_order": "json_xy_tensor_yx",
            "gt_fields_present": False,
            "prompts": runtime_prompts,
        },
    )
    write_json(
        evaluation_path,
        {
            "kind": "prompt_prior_evaluation_prompts",
            "status": "complete",
            "scene_id": scene_id,
            "prompts": evaluation_prompts,
        },
    )
    del model
    torch.cuda.empty_cache()
    return {"runtime": str(runtime_path), "evaluation": str(evaluation_path)}


def _mask_scale_values(path: Path) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    for item in sorted(path.glob("*.pt")):
        tensor = torch.load(item, map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy().reshape(-1)
            array = array[np.isfinite(array) & (array >= 0)]
            if len(array):
                values.append(array.astype(np.float64, copy=False))
    if not values:
        raise ValueError(f"no finite mask scales found in {path}")
    return np.concatenate(values)


def _materialize_parameters(
    *,
    priors_path: Path,
    scenes: Mapping[str, Mapping[str, Any]],
    scene_ids: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    table = materialize_prompt_prior(load_json(priors_path))
    scene_rows: dict[str, Any] = {}
    for scene_id in scene_ids:
        assets = _scene_assets(scenes[scene_id])
        raw_scales = _mask_scale_values(
            assets["mask_scales"]
        )
        metres_per_unit = float(
            scenes[scene_id].get("scene_scale_m_per_unit", 1.0)
        )
        if not math.isfinite(metres_per_unit) or metres_per_unit <= 0.0:
            raise ValueError(f"{scene_id}: invalid scene_scale_m_per_unit")
        scales = raw_scales * metres_per_unit
        class_inputs = {
            class_name: training_quantile_uniform(scales, float(typical_diag_m))
            for class_name, typical_diag_m in table[
                "class_typical_diag_m"
            ].items()
        }
        scene_rows[scene_id] = {
            "mask_scale_count": int(len(scales)),
            "scene_scale_m_per_unit": metres_per_unit,
            "mask_scale_min_m": float(np.min(scales)),
            "mask_scale_max_m": float(np.max(scales)),
            "normalization": "training_quantile_transformer_uniform",
            "feature_ply": {
                "path": str(assets["feature_ply"]),
                "size_bytes": int(assets["feature_ply"].stat().st_size),
                "mtime_ns": int(assets["feature_ply"].stat().st_mtime_ns),
            },
            "scale_gate": {
                "path": str(assets["scale_gate"]),
                "size_bytes": int(assets["scale_gate"].stat().st_size),
                "mtime_ns": int(assets["scale_gate"].stat().st_mtime_ns),
            },
            "mask_scales_path": str(assets["mask_scales"]),
            "global_scale_input": training_quantile_uniform(
                scales, float(table["global_typical_diag_m"])
            ),
            "global_scale_input_ecdf_diagnostic": empirical_cdf(
                scales, float(table["global_typical_diag_m"])
            ),
            "class_scale_inputs": class_inputs,
        }
    payload = {
        "kind": "prompt_prior_params",
        "status": "complete",
        "prior_source": str(priors_path.resolve()),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "table": table,
        "scenes": scene_rows,
    }
    write_json(output, payload)
    return payload


def prepare_prompts(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    category_priors: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    taxonomy_path: Path,
    min_region_size: int = 100,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    taxonomy = load_taxonomy(taxonomy_path)
    selected = {
        "chair", "table", "plant", "tv", "painting", "sofa", "cabinet",
        "bed", "socket", "book", "switch", "door", "window", "lamp",
        "speaker", "fan", "refrigerator", "cup", "phone", "trash can",
    }
    results = []
    for scene_id in scene_ids:
        if scene_id not in scenes:
            raise KeyError(f"scene missing from runtime manifest: {scene_id}")
        results.append(
            _prepare_scene_prompts(
                scene_id=scene_id,
                scene=scenes[scene_id],
                gt_path=_gt_path(scenes[scene_id], gt_dir, scene_id),
                output_root=output_root,
                canonical_classes=taxonomy.canonical_classes,
                selected_classes=selected,
                min_region_size=min_region_size,
            )
        )
    params = _materialize_parameters(
        priors_path=category_priors,
        scenes=scenes,
        scene_ids=scene_ids,
        output=output_root / "prompt_prior_params.json",
    )
    summary = {
        "kind": "prompt_prior_preparation",
        "status": "complete",
        "scene_ids": list(scene_ids),
        "prompt_files": results,
        "parameters": str(output_root / "prompt_prior_params.json"),
        "parameter_scene_count": len(params["scenes"]),
    }
    write_json(output_root / "preparation.json", summary)
    return summary


def _condition_scale(
    parameters: Mapping[str, Any], scene_id: str, class_name: str, condition: str
) -> float:
    scene = parameters["scenes"][scene_id]
    if condition == "U-global":
        return float(scene["global_scale_input"])
    if condition != "D-class":
        raise ValueError(f"unknown prompt-prior condition: {condition}")
    return float(
        scene["class_scale_inputs"].get(
            class_name, scene["global_scale_input"]
        )
    )


def _mask_change_count(uniform: np.ndarray, data: np.ndarray) -> int:
    """Count exact pointwise treatment changes, independent of scene size."""
    uniform_array = np.asarray(uniform, dtype=bool)
    data_array = np.asarray(data, dtype=bool)
    if uniform_array.shape != data_array.shape:
        raise ValueError("uniform and data masks must have the same shape")
    return int(np.count_nonzero(uniform_array != data_array))


def _complete_prompt_result(
    path: Path,
    expected_points: int,
    *,
    expected_prompt: Mapping[str, Any],
    feature_ply: Path,
    scale_gate: Path,
    expected_scales: Mapping[str, float],
) -> bool:
    metadata = path.with_suffix(".json")
    try:
        with np.load(path) as payload:
            valid = all(
                key in payload
                and np.asarray(payload[key]).shape == (expected_points,)
                for key in ("U_global", "D_class")
            )
        row = load_json(metadata)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    if not valid or row.get("status") != "complete":
        return False
    if row.get("prompt") != dict(expected_prompt):
        return False
    if not math.isclose(
        float(row.get("similarity_threshold", float("nan"))),
        SIMILARITY_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        return False
    if Path(str(row.get("feature_ply", ""))).resolve() != feature_ply.resolve():
        return False
    if Path(str(row.get("scale_gate", ""))).resolve() != scale_gate.resolve():
        return False
    try:
        return all(
            math.isclose(
                float(row["conditions"][condition]["scale_input"]),
                float(expected_scales[condition]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for condition in CONDITIONS
        )
    except (KeyError, TypeError, ValueError):
        return False


def segment_prompts(
    *,
    runtime_manifest: Path,
    prompts_root: Path,
    parameters_path: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    mechanical_only: bool,
) -> dict[str, Any]:
    """Run SAGA masks.  This function intentionally has no GT input."""

    import torch
    import torch.nn.functional as functional
    from gaussian_renderer import render_contrastive_feature

    scenes = load_scene_runtime_manifest(runtime_manifest)
    parameters = load_json(parameters_path)
    completed = 0
    skipped = 0
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        runtime = load_json(
            prompts_root / "prompts" / scene_id / "runtime_prompts.json"
        )
        if runtime.get("gt_fields_present") is not False:
            raise ValueError(f"{scene_id}: runtime prompt file is not GT-isolated")
        prompts = [
            row for row in runtime["prompts"]
            if not mechanical_only or bool(row.get("mechanical_selected"))
        ]
        feature_xyz = load_ply_xyz(assets["feature_ply"])
        scene_xyz = load_ply_xyz(assets["point_cloud"])
        if feature_xyz.shape != scene_xyz.shape or not np.allclose(
            feature_xyz, scene_xyz, atol=1e-6, rtol=0.0
        ):
            raise ValueError(f"{scene_id}: RGB and feature Gaussian XYZ/order differ")
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
        scene_output = output_root / scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        pending_by_camera: dict[str, list[Mapping[str, Any]]] = {}
        for prompt in prompts:
            prompt_id = str(prompt["prompt_id"])
            target = scene_output / f"{prompt_id}.npz"
            expected_scales = {
                condition: _condition_scale(
                    parameters,
                    scene_id,
                    str(prompt["class_name"]),
                    condition,
                )
                for condition in CONDITIONS
            }
            if _complete_prompt_result(
                target,
                len(point_features),
                expected_prompt=prompt,
                feature_ply=assets["feature_ply"],
                scale_gate=assets["scale_gate"],
                expected_scales=expected_scales,
            ):
                skipped += 1
                continue
            image_name = str(prompt["image_name"])
            if image_name not in cameras:
                raise KeyError(f"{scene_id}: prompt camera not found: {image_name}")
            pending_by_camera.setdefault(image_name, []).append(prompt)

        notebook_reference_path = scene_output / "notebook_scale_0p5_reference.json"
        notebook_reference_done = notebook_reference_path.is_file()
        for image_name in sorted(pending_by_camera):
            camera = deepcopy(cameras[image_name])
            camera.feature_height = camera.image_height
            camera.feature_width = camera.image_width
            with torch.no_grad():
                rendered = render_contrastive_feature(
                    camera,
                    model,
                    pipe,
                    background,
                    norm_point_features=True,
                )["render"].detach()
            for prompt in pending_by_camera[image_name]:
                prompt_id = str(prompt["prompt_id"])
                target = scene_output / f"{prompt_id}.npz"
                x, y = int(prompt["x"]), int(prompt["y"])
                if not (0 <= y < rendered.shape[1] and 0 <= x < rendered.shape[2]):
                    raise ValueError(
                        f"{scene_id}/{prompt_id}: prompt pixel out of range"
                    )
                masks: dict[str, np.ndarray] = {}
                diagnostics: dict[str, Any] = {}
                if mechanical_only and not notebook_reference_done:
                    with torch.no_grad():
                        half_gate = gate_model(
                            torch.tensor(
                                [0.5], dtype=torch.float32, device="cuda"
                            )
                        ).squeeze(0)
                        notebook_map = functional.normalize(
                            rendered.permute(1, 2, 0) * half_gate,
                            dim=-1,
                            p=2,
                        )
                        notebook_query = notebook_map[y, x]
                        notebook_points = functional.normalize(
                            point_features * half_gate.unsqueeze(0),
                            dim=1,
                            p=2,
                        )
                        notebook_similarity = torch.einsum(
                            "c,nc->n", notebook_query, notebook_points
                        )
                        worker_query = functional.normalize(
                            rendered[:, y, x] * half_gate, dim=0, p=2
                        )
                        worker_similarity = notebook_points @ worker_query
                    reference_equal = bool(
                        torch.equal(
                            notebook_similarity > SIMILARITY_THRESHOLD,
                            worker_similarity > SIMILARITY_THRESHOLD,
                        )
                    )
                    maximum_error = float(
                        torch.max(
                            torch.abs(notebook_similarity - worker_similarity)
                        ).item()
                    )
                    write_json(
                        notebook_reference_path,
                        {
                            "kind": "prompt_prior_notebook_reference",
                            "status": "complete",
                            "scene_id": scene_id,
                            "prompt_id": prompt_id,
                            "scale_input": 0.5,
                            "mask_exact": reference_equal,
                            "maximum_similarity_error": maximum_error,
                            "passed": reference_equal and maximum_error <= 1e-6,
                        },
                    )
                    notebook_reference_done = True
                for condition in CONDITIONS:
                    scale = _condition_scale(
                        parameters, scene_id, str(prompt["class_name"]), condition
                    )
                    with torch.no_grad():
                        gate = gate_model(
                            torch.tensor(
                                [scale], dtype=torch.float32, device="cuda"
                            )
                        ).squeeze(0)
                        query = functional.normalize(
                            rendered[:, y, x] * gate, dim=0, p=2
                        )
                        points = functional.normalize(
                            point_features * gate.unsqueeze(0), dim=1, p=2
                        )
                        similarity = points @ query
                        mask = similarity > SIMILARITY_THRESHOLD
                        repeat_similarity = points @ query
                        repeat_mask = repeat_similarity > SIMILARITY_THRESHOLD
                    key = "U_global" if condition == "U-global" else "D_class"
                    masks[key] = mask.detach().cpu().numpy().astype(np.uint8)
                    diagnostics[condition] = {
                        "scale_input": scale,
                        "gate": gate.detach().cpu().numpy().astype(float).tolist(),
                        "predicted_gaussian_count": int(mask.sum().item()),
                        "similarity_min": float(similarity.min().item()),
                        "similarity_max": float(similarity.max().item()),
                        "similarity_mean": float(similarity.mean().item()),
                        "query_self_similarity": float((query @ query).item()),
                        "repeat_mask_exact": bool(torch.equal(mask, repeat_mask)),
                    }
                temporary = target.with_suffix(".part.npz")
                np.savez_compressed(
                    temporary,
                    U_global=masks["U_global"],
                    D_class=masks["D_class"],
                )
                temporary.replace(target)
                change_count = _mask_change_count(
                    masks["U_global"], masks["D_class"]
                )
                change = float(change_count / len(masks["U_global"]))
                write_json(
                    target.with_suffix(".json"),
                    {
                        "kind": "prompt_prior_mask_pair",
                        "status": "complete",
                        "scene_id": scene_id,
                        "prompt": dict(prompt),
                        "feature_ply": str(assets["feature_ply"]),
                        "scale_gate": str(assets["scale_gate"]),
                        "similarity_threshold": SIMILARITY_THRESHOLD,
                        "conditions": diagnostics,
                        "mask_change_count": change_count,
                        "mask_change_fraction": change,
                    },
                )
                completed += 1
            del rendered
            torch.cuda.empty_cache()
        del model, gate_model
        torch.cuda.empty_cache()
        if mechanical_only:
            reference = load_json(notebook_reference_path)
            if reference.get("passed") is not True:
                raise ValueError(f"{scene_id}: Notebook scale=0.5 parity failed")
    result = {
        "kind": "prompt_prior_segmentation",
        "status": "complete",
        "scene_ids": list(scene_ids),
        "mechanical_only": bool(mechanical_only),
        "completed_prompts": completed,
        "reused_prompts": skipped,
    }
    write_json(output_root / "segmentation.json", result)
    return result


def evaluate_prompt_pair_arrays(
    *,
    mask: np.ndarray,
    target_class_id: int,
    target_instance_id: int,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gt_to_gaussian_index: np.ndarray,
    gt_to_gaussian_distance_m: np.ndarray,
    gaussian_to_gt_index: np.ndarray,
    gaussian_to_gt_distance_m: np.ndarray,
    radius_m: float = PROMPT_RADIUS_M,
) -> dict[str, Any]:
    prediction = np.asarray(mask, dtype=bool)
    target = (
        (np.asarray(gt_semantic) == int(target_class_id))
        & (np.asarray(gt_instance) == int(target_instance_id))
    )
    valid_gt = np.asarray(gt_to_gaussian_distance_m) <= float(radius_m)
    projected = np.zeros(len(target), dtype=bool)
    projected[valid_gt] = prediction[
        np.asarray(gt_to_gaussian_index, dtype=np.int64)[valid_gt]
    ]
    intersection = int(np.count_nonzero(projected & target))
    union = int(np.count_nonzero(projected | target))
    predicted_ids = np.flatnonzero(prediction)
    supported = (
        np.asarray(gaussian_to_gt_distance_m)[predicted_ids] <= float(radius_m)
    )
    nearest_gt = np.asarray(gaussian_to_gt_index, dtype=np.int64)[predicted_ids]
    correct = np.zeros(len(predicted_ids), dtype=bool)
    correct[supported] = (
        (np.asarray(gt_semantic)[nearest_gt[supported]] == int(target_class_id))
        & (np.asarray(gt_instance)[nearest_gt[supported]] == int(target_instance_id))
    )
    total = int(len(predicted_ids))
    return {
        "iou": intersection / union if union else 0.0,
        "gaussian_precision": int(correct.sum()) / total if total else 0.0,
        "gt_recall": intersection / int(target.sum()) if np.any(target) else 0.0,
        "predicted_gaussian_count": total,
        "correct_gaussian_count": int(correct.sum()),
        "unsupported_gaussian_count": int(np.count_nonzero(~supported)),
        "projected_gt_point_count": int(np.count_nonzero(projected)),
    }


def _size_bin(diagonal_m: float, size_spec: Mapping[str, Any] | None) -> str | None:
    if size_spec is None:
        return None
    limits = size_spec["boundaries_m"]
    if diagonal_m <= float(limits["tiny_max_m"]):
        return "tiny"
    if diagonal_m <= float(limits["small_max_m"]):
        return "small"
    if diagonal_m <= float(limits["medium_max_m"]):
        return "medium"
    return "large"


def _export_prompt_viewers(
    *,
    rows: Sequence[Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
    gt_dir: Path,
    masks_root: Path,
    viewer_root: Path,
    scope: str,
) -> list[dict[str, Any]]:
    """Export low/median/high paired masks as real 3-D point clouds."""

    from .viewer_materials import _write_colored_ply

    if not rows:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["delta_iou"]),
            str(row["scene_id"]),
            str(row["prompt_id"]),
        ),
    )
    positions = (0, len(ranked) // 2, len(ranked) - 1)
    labels = ("lowest", "median", "highest")
    exported: list[dict[str, Any]] = []
    for label, position in zip(labels, positions):
        row = ranked[position]
        scene_id = str(row["scene_id"])
        prompt_id = str(row["prompt_id"])
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["feature_ply"]), _transform(scene)
        )
        gt_xyz, gt = load_ground_truth_npz(
            _gt_path(scene, gt_dir, scene_id), scene_id
        )
        target = (
            (gt.semantic == int(row["class_id"]))
            & (gt.instance == int(row["gt_instance_id"]))
        )
        with np.load(masks_root / scene_id / f"{prompt_id}.npz") as payload:
            uniform = np.asarray(payload["U_global"], dtype=bool)
            data = np.asarray(payload["D_class"], dtype=bool)
        common = uniform & data
        uniform_only = uniform & ~data
        data_only = data & ~uniform
        destination = viewer_root / scope / f"{label}-{scene_id}-{prompt_id}"
        _write_colored_ply(
            destination / "U_global_gaussians.ply",
            gaussian_xyz[uniform],
            np.tile(np.asarray([[255, 180, 0]], dtype=np.uint8), (uniform.sum(), 1)),
        )
        _write_colored_ply(
            destination / "D_class_gaussians.ply",
            gaussian_xyz[data],
            np.tile(np.asarray([[0, 220, 255]], dtype=np.uint8), (data.sum(), 1)),
        )
        _write_colored_ply(
            destination / "target_gt_points.ply",
            gt_xyz[target],
            np.tile(np.asarray([[40, 100, 255]], dtype=np.uint8), (target.sum(), 1)),
        )
        overlay_xyz = np.concatenate(
            (
                gt_xyz[target],
                gaussian_xyz[common],
                gaussian_xyz[uniform_only],
                gaussian_xyz[data_only],
            ),
            axis=0,
        )
        overlay_rgb = np.concatenate(
            (
                np.tile(
                    np.asarray([[40, 100, 255]], dtype=np.uint8),
                    (target.sum(), 1),
                ),
                np.tile(
                    np.asarray([[0, 220, 80]], dtype=np.uint8),
                    (common.sum(), 1),
                ),
                np.tile(
                    np.asarray([[255, 180, 0]], dtype=np.uint8),
                    (uniform_only.sum(), 1),
                ),
                np.tile(
                    np.asarray([[255, 0, 220]], dtype=np.uint8),
                    (data_only.sum(), 1),
                ),
            ),
            axis=0,
        )
        _write_colored_ply(destination / "overlay.ply", overlay_xyz, overlay_rgb)
        metrics = {
            "kind": "prompt_prior_viewer_case",
            "scope": scope,
            "rank": label,
            "colors": {
                "target_gt": "blue",
                "shared_prediction": "green",
                "U_global_only": "orange",
                "D_class_only": "magenta",
            },
            **dict(row),
        }
        write_json(destination / "metrics.json", metrics)
        exported.append(
            {
                "rank": label,
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "path": str(destination),
            }
        )
    return exported


def evaluate_prompts(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prompts_root: Path,
    masks_root: Path,
    scene_ids: Sequence[str],
    table_output: Path,
    analysis_output: Path,
    size_bins: Path | None,
    mechanical_only: bool,
    viewer_root: Path | None,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins else None
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        assets = _scene_assets(scene)
        gt_xyz, gt = load_ground_truth_npz(
            _gt_path(scene, gt_dir, scene_id), scene_id
        )
        gaussian_xyz = apply_transform(
            load_ply_xyz(assets["feature_ply"]), _transform(scene)
        )
        gt_to_gaussian, gt_distance = _nearest(gt_xyz, gaussian_xyz)
        gaussian_to_gt, gaussian_distance = _nearest(gaussian_xyz, gt_xyz)
        evaluation = load_json(
            prompts_root / "prompts" / scene_id / "evaluation_prompts.json"
        )
        selected = [
            prompt for prompt in evaluation["prompts"]
            if not mechanical_only or bool(prompt.get("mechanical_selected"))
        ]
        for prompt in selected:
            prompt_id = str(prompt["prompt_id"])
            mask_path = masks_root / scene_id / f"{prompt_id}.npz"
            metadata = load_json(mask_path.with_suffix(".json"))
            with np.load(mask_path) as masks:
                mask_change_count = _mask_change_count(
                    masks["U_global"], masks["D_class"]
                )
                mask_union_count = int(
                    np.count_nonzero(masks["U_global"] | masks["D_class"])
                )
                condition_results = {
                    "U-global": evaluate_prompt_pair_arrays(
                        mask=masks["U_global"],
                        target_class_id=int(prompt["class_id"]),
                        target_instance_id=int(prompt["gt_instance_id"]),
                        gt_semantic=gt.semantic,
                        gt_instance=gt.instance,
                        gt_to_gaussian_index=gt_to_gaussian,
                        gt_to_gaussian_distance_m=gt_distance,
                        gaussian_to_gt_index=gaussian_to_gt,
                        gaussian_to_gt_distance_m=gaussian_distance,
                    ),
                    "D-class": evaluate_prompt_pair_arrays(
                        mask=masks["D_class"],
                        target_class_id=int(prompt["class_id"]),
                        target_instance_id=int(prompt["gt_instance_id"]),
                        gt_semantic=gt.semantic,
                        gt_instance=gt.instance,
                        gt_to_gaussian_index=gt_to_gaussian,
                        gt_to_gaussian_distance_m=gt_distance,
                        gaussian_to_gt_index=gaussian_to_gt,
                        gaussian_to_gt_distance_m=gaussian_distance,
                    ),
                }
            row: dict[str, Any] = {
                "scene_id": scene_id,
                "prompt_id": prompt_id,
                "class_name": str(prompt["class_name"]),
                "class_id": int(prompt["class_id"]),
                "gt_instance_id": int(prompt["gt_instance_id"]),
                "gt_point_count": int(prompt["gt_point_count"]),
                "bbox_diagonal_m": float(prompt["bbox_diagonal_m"]),
                "size_bin": _size_bin(float(prompt["bbox_diagonal_m"]), size_spec),
                "mechanical_selected": bool(prompt.get("mechanical_selected")),
                "mask_change_count": mask_change_count,
                "mask_change_fraction": float(
                    mask_change_count / len(gaussian_xyz)
                ),
                "mask_change_fraction_of_union": float(
                    mask_change_count / mask_union_count
                )
                if mask_union_count
                else 0.0,
                "gate_delta_linf": float(
                    np.max(
                        np.abs(
                            np.asarray(
                                metadata["conditions"]["D-class"]["gate"],
                                dtype=np.float64,
                            )
                            - np.asarray(
                                metadata["conditions"]["U-global"]["gate"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "scale_input_u": float(
                    metadata["conditions"]["U-global"]["scale_input"]
                ),
                "scale_input_d": float(
                    metadata["conditions"]["D-class"]["scale_input"]
                ),
            }
            for condition, values in condition_results.items():
                suffix = "u" if condition == "U-global" else "d"
                row.update({f"{key}_{suffix}": value for key, value in values.items()})
            row.update(
                {
                    "delta_iou": row["iou_d"] - row["iou_u"],
                    "delta_gaussian_precision": (
                        row["gaussian_precision_d"] - row["gaussian_precision_u"]
                    ),
                    "delta_gt_recall": row["gt_recall_d"] - row["gt_recall_u"],
                }
            )
            rows.append(row)
    write_rows(table_output, rows)
    scene_rows = []
    for scene_id in scene_ids:
        selected = [row for row in rows if row["scene_id"] == scene_id]
        if not selected:
            continue
        scene_rows.append(
            {
                "scene_id": scene_id,
                "object_count": len(selected),
                "mean_delta_iou": float(np.mean([row["delta_iou"] for row in selected])),
                "mean_delta_gaussian_precision": float(
                    np.mean([row["delta_gaussian_precision"] for row in selected])
                ),
                "mean_delta_gt_recall": float(
                    np.mean([row["delta_gt_recall"] for row in selected])
                ),
            }
        )
    scale_changed = sum(
        abs(float(row["scale_input_d"]) - float(row["scale_input_u"])) >= 0.05
        for row in rows
    )
    gate_changed = sum(float(row["gate_delta_linf"]) > 1e-6 for row in rows)
    # This is an implementation-activation check, not an effect-size screen.
    # A small prompted object may contain far fewer than 1% of all scene
    # Gaussians, so a full-scene 1% denominator would make it impossible for
    # precisely the small objects under study to pass.  Effect magnitude is
    # assessed later with object IoU at the physical-scene level.
    mask_changed = sum(int(row["mask_change_count"]) > 0 for row in rows)
    changed_scene_ids = {
        str(row["scene_id"])
        for row in rows
        if int(row["mask_change_count"]) > 0
    }
    large_full_scene_changes = sum(
        float(row["mask_change_fraction"]) >= 0.01 for row in rows
    )
    classes = sorted({str(row["class_name"]) for row in rows})
    mean_delta = float(np.mean([row["mean_delta_iou"] for row in scene_rows])) if scene_rows else 0.0
    precision_delta = (
        float(np.mean([row["mean_delta_gaussian_precision"] for row in scene_rows]))
        if scene_rows else 0.0
    )
    tiny_small_scene_rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        selected = [
            row
            for row in rows
            if row["scene_id"] == scene_id
            and row["size_bin"] in {"tiny", "small"}
        ]
        if selected:
            tiny_small_scene_rows.append(
                {
                    "scene_id": scene_id,
                    "object_count": len(selected),
                    "mean_delta_iou": float(
                        np.mean([row["delta_iou"] for row in selected])
                    ),
                    "mean_delta_gt_recall": float(
                        np.mean([row["delta_gt_recall"] for row in selected])
                    ),
                }
            )
    tiny_iou_delta = (
        float(
            np.mean(
                [row["mean_delta_iou"] for row in tiny_small_scene_rows]
            )
        )
        if tiny_small_scene_rows
        else None
    )
    tiny_recall_delta = (
        float(
            np.mean(
                [
                    row["mean_delta_gt_recall"]
                    for row in tiny_small_scene_rows
                ]
            )
        )
        if tiny_small_scene_rows
        else None
    )
    if scene_rows:
        scene_delta = np.asarray(
            [row["mean_delta_iou"] for row in scene_rows], dtype=np.float64
        )
        generator = np.random.default_rng(42)
        bootstrap = scene_delta[
            generator.integers(0, len(scene_delta), size=(10_000, len(scene_delta)))
        ].mean(axis=1)
        delta_iou_interval = [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ]
    else:
        delta_iou_interval = [0.0, 0.0]
    scene_object_counts = {
        row["scene_id"]: int(row["object_count"]) for row in scene_rows
    }
    nonempty_outputs = all(
        int(row["predicted_gaussian_count_u"]) > 0
        and int(row["predicted_gaussian_count_d"]) > 0
        for row in rows
    )
    query_self_valid = all(
        abs(
            float(
                load_json(
                    masks_root / row["scene_id"] / f"{row['prompt_id']}.json"
                )["conditions"][condition]["query_self_similarity"]
            )
            - 1.0
        )
        <= 1e-5
        for row in rows
        for condition in CONDITIONS
    )
    repeat_masks_valid = all(
        bool(
            load_json(
                masks_root / row["scene_id"] / f"{row['prompt_id']}.json"
            )["conditions"][condition]["repeat_mask_exact"]
        )
        for row in rows
        for condition in CONDITIONS
    )
    notebook_reference_valid = (
        all(
            bool(
                load_json(
                    masks_root
                    / scene_id
                    / "notebook_scale_0p5_reference.json"
                ).get("passed")
            )
            for scene_id in scene_ids
        )
        if mechanical_only
        else True
    )
    mechanical_gate = {
        "object_count_at_least_8": len(rows) >= 8,
        "each_registered_scene_has_at_least_4_objects": all(
            scene_object_counts.get(scene_id, 0) >= 4 for scene_id in scene_ids
        ),
        "scale_changed_objects_at_least_4": scale_changed >= 4,
        "gate_changed_objects_at_least_4": gate_changed >= 4,
        "mask_changed_objects_at_least_2": mask_changed >= 2,
        "each_registered_scene_has_changed_object": all(
            scene_id in changed_scene_ids for scene_id in scene_ids
        ),
        "all_outputs_nonempty": nonempty_outputs,
        "all_query_self_similarities_are_one": query_self_valid,
        "all_repeat_masks_are_pointwise_identical": repeat_masks_valid,
        "notebook_scale_0p5_parity_passed": notebook_reference_valid,
    }
    mechanism_gate = {
        "all_eight_registered_scenes_present": len(scene_rows) == 8,
        "mean_delta_iou_at_least_0p02": mean_delta >= 0.02,
        "positive_scenes_at_least_5": sum(
            row["mean_delta_iou"] > 0 for row in scene_rows
        ) >= 5,
        "precision_drop_at_most_0p01": precision_delta >= -0.01,
        "tiny_small_signal": (
            tiny_iou_delta is not None
            and tiny_recall_delta is not None
            and (
                (tiny_iou_delta > 0 and tiny_recall_delta >= -0.02)
                or (tiny_recall_delta > 0 and tiny_iou_delta >= -0.02)
            )
        ),
    }
    scope = "mechanical2" if mechanical_only else "paired8"
    viewer_cases = (
        _export_prompt_viewers(
            rows=rows,
            scenes=scenes,
            gt_dir=gt_dir,
            masks_root=masks_root,
            viewer_root=viewer_root,
            scope=scope,
        )
        if viewer_root is not None
        else []
    )
    analysis = {
        "kind": "prompt_prior_analysis",
        "status": "complete",
        "scope": scope,
        "scene_ids": list(scene_ids),
        "object_count": len(rows),
        "class_names": classes,
        "scene_results": scene_rows,
        "mean_scene_delta_iou": mean_delta,
        "scene_paired_bootstrap95_delta_iou": delta_iou_interval,
        "mean_scene_delta_gaussian_precision": precision_delta,
        "tiny_small_delta_iou": tiny_iou_delta,
        "tiny_small_delta_gt_recall": tiny_recall_delta,
        "tiny_small_scene_results": tiny_small_scene_rows,
        "mechanical_intervention_audit": {
            "definition": "at_least_one_pointwise_mask_difference",
            "changed_object_count": mask_changed,
            "changed_scene_ids": sorted(changed_scene_ids),
            "gate_changed_object_count": gate_changed,
            "deprecated_full_scene_1pct_changed_object_count": (
                large_full_scene_changes
            ),
            "amendment_reason": (
                "the former full-scene 1% denominator was invalid for small "
                "prompted objects; this gate only verifies that the treatment "
                "entered the output, while IoU measures effect magnitude"
            ),
        },
        "component_count_diagnostic": {
            "status": "not_applicable",
            "reason": (
                "the native SAGA prompt output is an unordered Boolean "
                "Gaussian mask and defines no adjacency graph; inventing one "
                "would add a second post-processing method"
            ),
        },
        "mechanical_gate": mechanical_gate,
        "mechanical_passed": all(mechanical_gate.values()),
        "mechanism_gate": mechanism_gate,
        "mechanism_passed": all(mechanism_gate.values()) if not mechanical_only else None,
        "viewer_cases": viewer_cases,
        "conclusion_boundary": (
            "known-object known-class native SAGA size-gate mechanism only; "
            "not automatic instance detection"
        ),
    }
    write_json(analysis_output, analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--runtime-manifest", required=True, type=Path)
    prepare.add_argument("--gt-dir", required=True, type=Path)
    prepare.add_argument("--category-priors", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--scene", action="append", required=True)
    prepare.add_argument(
        "--taxonomy", type=Path, default=default_taxonomy_path()
    )
    prepare.add_argument("--min-region-size", type=int, default=100)

    segment = commands.add_parser("segment")
    segment.add_argument("--runtime-manifest", required=True, type=Path)
    segment.add_argument("--prompts-root", required=True, type=Path)
    segment.add_argument("--parameters", required=True, type=Path)
    segment.add_argument("--output-root", required=True, type=Path)
    segment.add_argument("--scene", action="append", required=True)
    segment.add_argument("--mechanical-only", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--runtime-manifest", required=True, type=Path)
    evaluate.add_argument("--gt-dir", required=True, type=Path)
    evaluate.add_argument("--prompts-root", required=True, type=Path)
    evaluate.add_argument("--masks-root", required=True, type=Path)
    evaluate.add_argument("--scene", action="append", required=True)
    evaluate.add_argument("--table-output", required=True, type=Path)
    evaluate.add_argument("--analysis-output", required=True, type=Path)
    evaluate.add_argument("--size-bins", type=Path)
    evaluate.add_argument("--viewer-root", type=Path)
    evaluate.add_argument("--mechanical-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_prompts(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            category_priors=args.category_priors,
            output_root=args.output_root,
            scene_ids=args.scene,
            taxonomy_path=args.taxonomy,
            min_region_size=args.min_region_size,
        )
    elif args.command == "segment":
        result = segment_prompts(
            runtime_manifest=args.runtime_manifest,
            prompts_root=args.prompts_root,
            parameters_path=args.parameters,
            output_root=args.output_root,
            scene_ids=args.scene,
            mechanical_only=args.mechanical_only,
        )
    else:
        result = evaluate_prompts(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            prompts_root=args.prompts_root,
            masks_root=args.masks_root,
            scene_ids=args.scene,
            table_output=args.table_output,
            analysis_output=args.analysis_output,
            size_bins=args.size_bins,
            mechanical_only=args.mechanical_only,
            viewer_root=args.viewer_root,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
