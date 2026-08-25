from __future__ import annotations

"""V10 lifting-only worker over already trained scene assets.

This module deliberately has no training or download entry point.  It reuses
the native V9 M1-core / alpha-mass-full lifting primitives, but identifies the
actual runtime-manifest inputs directly instead of requiring a V9 10k feature
training record.
"""

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .v9_feature_training import validate_v8_sam_everything_source
from .v9_lifting import (
    AttributionMass,
    DEFAULT_CLASSES,
    FragmentConfig,
    V9_LIFTING_SCHEMA,
    fragments_from_mass,
    hybrid_fragments,
    lifting_bank_is_complete as v9_lifting_bank_is_complete,
    mass_from_max_contributor,
)


V10_LIFTING_SCHEMA = "saga-v10-lifting-bank-v1"
V10_LIFTING_IDENTITY_SCHEMA = "saga-v10-lifting-identity-v1"
COMPATIBLE_LIFTING_SCHEMAS = (V9_LIFTING_SCHEMA, V10_LIFTING_SCHEMA)


@dataclass(frozen=True)
class V10LiftingInputs:
    base_path: Path
    rgb_ply: Path
    feature_ply: Path
    sparse: Path
    images: Path
    grounded_masks: Path
    grounded_labels: Path
    label_features: Path
    segment_everything_root: Path


def _resolve_scene_path(
    scene: Mapping[str, Any], keys: Sequence[str], default: str | Path
) -> Path:
    base = Path(str(scene["base_path"])).resolve()
    value: Any = default
    for key in keys:
        if scene.get(key) not in (None, ""):
            value = scene[key]
            break
    target = Path(str(value))
    return (base / target).resolve() if not target.is_absolute() else target.resolve()


def _default_rgb_ply(base: Path) -> Path:
    candidates = (
        base / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_v10_lifting_inputs(
    scene: Mapping[str, Any],
    *,
    segment_everything_root: Path | None = None,
    feature_ply: Path | None = None,
    label_features: Path | None = None,
) -> V10LiftingInputs:
    """Resolve immutable lifting inputs without consulting a training record."""

    base = Path(str(scene["base_path"])).resolve()
    labels = _resolve_scene_path(
        scene, ("grounded_labels_path", "labels_path"), "saga/labels"
    )
    selected_feature = (
        Path(feature_ply).resolve()
        if feature_ply is not None
        else _resolve_scene_path(
            scene,
            (
                "feature_ply_path",
                "contrastive_feature_point_cloud_path",
                "feature_ply",
            ),
            "saga/contrastive_feature_point_cloud.ply",
        )
    )
    selected_labels = (
        Path(label_features).resolve()
        if label_features is not None
        else _resolve_scene_path(
            scene,
            ("grounded_label_features_path", "label_features_path"),
            labels / "label_features.pt",
        )
    )
    if segment_everything_root is None:
        packed_value = next(
            (
                scene[key]
                for key in (
                    "segment_everything_root",
                    "sam_everything_packed_path",
                    "sam_everything_root",
                )
                if scene.get(key) not in (None, "")
            ),
            None,
        )
        if packed_value is None:
            raise ValueError("runtime scene lacks a packed SAM-everything root")
        packed = Path(str(packed_value))
        segment_everything_root = (
            (base / packed).resolve() if not packed.is_absolute() else packed.resolve()
        )
    else:
        segment_everything_root = Path(segment_everything_root).resolve()
    rgb = (
        _resolve_scene_path(scene, ("point_cloud_path",), "")
        if scene.get("point_cloud_path")
        else _default_rgb_ply(base).resolve()
    )
    result = V10LiftingInputs(
        base_path=base,
        rgb_ply=rgb,
        feature_ply=selected_feature,
        sparse=_resolve_scene_path(scene, ("sparse_path",), "fastRecon/dense/sparse/0"),
        images=_resolve_scene_path(
            scene, ("images_path",), "fastRecon/dense/sparse/0/images"
        ),
        grounded_masks=_resolve_scene_path(
            scene, ("grounded_masks_path", "masks_path"), "saga/masks"
        ),
        grounded_labels=labels,
        label_features=selected_labels,
        segment_everything_root=segment_everything_root,
    )
    missing = [
        str(path)
        for path in (
            result.rgb_ply,
            result.feature_ply,
            result.sparse,
            result.images,
            result.grounded_masks,
            result.grounded_labels,
            result.label_features,
            result.segment_everything_root,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing V10 lifting inputs: {missing}")
    return result


def _file_identity(path: Path) -> dict[str, Any]:
    target = Path(path).resolve()
    stat = target.stat()
    return {
        "path": str(target),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _directory_identity(path: Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    return {
        "path": str(root),
        "files": [
            {
                "relative_path": item.relative_to(root).as_posix(),
                "size_bytes": int(item.stat().st_size),
                "mtime_ns": int(item.stat().st_mtime_ns),
            }
            for item in files
        ],
    }


def _sparse_identity(path: Path) -> dict[str, Any]:
    root = Path(path).resolve()
    names = (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
    )
    files = [_file_identity(root / name) for name in names if (root / name).is_file()]
    if not files:
        raise FileNotFoundError(f"COLMAP sparse model is empty: {root}")
    return {"path": str(root), "files": files}


def build_v10_lifting_identity(
    *,
    scene_id: str,
    git_commit: str,
    inputs: V10LiftingInputs,
    classes: Sequence[str] = DEFAULT_CLASSES,
    config: FragmentConfig = FragmentConfig(),
) -> dict[str, Any]:
    """Return a no-hash identity for every source consumed by lifting."""

    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("V10 lifting git_commit must be non-empty")
    sam = validate_v8_sam_everything_source(inputs.segment_everything_root)
    return {
        "schema": V10_LIFTING_IDENTITY_SCHEMA,
        "scene_id": str(scene_id),
        "git_commit": commit,
        "rgb_ply": _file_identity(inputs.rgb_ply),
        "feature_ply": _file_identity(inputs.feature_ply),
        "label_features": _file_identity(inputs.label_features),
        "sparse": _sparse_identity(inputs.sparse),
        "images": _directory_identity(inputs.images),
        "grounded_masks": _directory_identity(inputs.grounded_masks),
        "grounded_labels": _directory_identity(inputs.grounded_labels),
        "segment_everything": sam,
        "classes": list(map(str, classes)),
        "fragment_config": asdict(config),
    }


def _valid_ragged(indptr: np.ndarray, values: np.ndarray, rows: int, points: int) -> bool:
    return bool(
        np.issubdtype(indptr.dtype, np.integer)
        and np.issubdtype(values.dtype, np.integer)
        and indptr.shape == (rows + 1,)
        and int(indptr[0]) == 0
        and np.all(np.diff(indptr) >= 0)
        and int(indptr[-1]) == len(values)
        and np.all(values >= 0)
        and np.all(values < points)
    )


def _v10_arrays_are_valid(metadata: Mapping[str, Any], path: Path) -> bool:
    n = int(metadata["point_count"])
    fragments = int(metadata["fragment_count"])
    frames = int(metadata["frame_count"])
    semantic_fragments = int(metadata["semantic_fragment_count"])
    classes = metadata["classes"]
    if (
        n <= 0
        or fragments < 0
        or frames <= 0
        or semantic_fragments < 0
        or not isinstance(classes, list)
        or not classes
        or len(set(classes)) != len(classes)
    ):
        return False
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "xyz_m", "affinity", "semantic", "label_features",
            "fragment_full_indptr", "fragment_full_ids", "fragment_full_mass",
            "fragment_core_indptr", "fragment_core_ids", "fragment_core_mass",
            "fragment_id", "fragment_frame", "fragment_mask_index",
            "fragment_conflict_ratio", "frame_visible_indptr", "frame_visible_ids",
            "frame_visible_mass", "frame_geometry_abstained", "frame_grounded_missing",
            "semantic_fragment_full_indptr", "semantic_fragment_full_ids",
            "semantic_fragment_full_mass", "semantic_fragment_frame",
            "semantic_fragment_class",
        }
        if not required.issubset(arrays.files):
            return False
        xyz = np.asarray(arrays["xyz_m"])
        affinity = np.asarray(arrays["affinity"])
        semantic = np.asarray(arrays["semantic"])
        codebook = np.asarray(arrays["label_features"])
        if (
            xyz.shape != (n, 3)
            or affinity.ndim != 2
            or affinity.shape[0] != n
            or affinity.shape[1] <= 0
            or semantic.ndim != 2
            or semantic.shape[0] != n
            or semantic.shape[1] <= 0
            or codebook.shape != (len(classes), semantic.shape[1])
            or any(np.any(~np.isfinite(value)) for value in (xyz, affinity, semantic, codebook))
            or np.any(np.linalg.norm(codebook, axis=1) <= 0)
            or not np.allclose(np.linalg.norm(codebook, axis=1), 1.0, atol=1e-4, rtol=1e-4)
        ):
            return False
        if (
            arrays["fragment_id"].shape != (fragments,)
            or not np.issubdtype(arrays["fragment_id"].dtype, np.integer)
            or len(np.unique(arrays["fragment_id"])) != fragments
            or arrays["fragment_frame"].shape != (fragments,)
            or np.any(arrays["fragment_frame"] < 0)
            or np.any(arrays["fragment_frame"] >= frames)
            or arrays["fragment_mask_index"].shape != (fragments,)
            or np.any(arrays["fragment_mask_index"] < 0)
            or arrays["fragment_conflict_ratio"].shape != (fragments,)
            or np.any(~np.isfinite(arrays["fragment_conflict_ratio"]))
            or np.any(arrays["fragment_conflict_ratio"] < 0)
            or np.any(arrays["fragment_conflict_ratio"] > 1)
            or arrays["frame_geometry_abstained"].shape != (frames,)
            or arrays["frame_geometry_abstained"].dtype != np.bool_
            or arrays["frame_grounded_missing"].shape != (frames,)
            or arrays["frame_grounded_missing"].dtype != np.bool_
        ):
            return False
        full_i = np.asarray(arrays["fragment_full_indptr"])
        full_v = np.asarray(arrays["fragment_full_ids"])
        core_i = np.asarray(arrays["fragment_core_indptr"])
        core_v = np.asarray(arrays["fragment_core_ids"])
        if not _valid_ragged(full_i, full_v, fragments, n) or not _valid_ragged(
            core_i, core_v, fragments, n
        ):
            return False
        full_mass = np.asarray(arrays["fragment_full_mass"])
        core_mass = np.asarray(arrays["fragment_core_mass"])
        if (
            full_mass.shape != full_v.shape
            or core_mass.shape != core_v.shape
            or np.any(~np.isfinite(full_mass))
            or np.any(~np.isfinite(core_mass))
            or np.any(full_mass < 0)
            or np.any(core_mass < 0)
        ):
            return False
        for index in range(fragments):
            full = full_v[int(full_i[index]) : int(full_i[index + 1])]
            core = core_v[int(core_i[index]) : int(core_i[index + 1])]
            if (
                (len(full) and np.any(np.diff(full) <= 0))
                or (len(core) and np.any(np.diff(core) <= 0))
                or not np.all(np.isin(core, full))
            ):
                return False
        visible_i = np.asarray(arrays["frame_visible_indptr"])
        visible_v = np.asarray(arrays["frame_visible_ids"])
        visible_mass = np.asarray(arrays["frame_visible_mass"])
        if (
            not _valid_ragged(visible_i, visible_v, frames, n)
            or visible_mass.shape != visible_v.shape
            or np.any(~np.isfinite(visible_mass))
            or np.any(visible_mass < 0)
        ):
            return False
        semantic_frame = np.asarray(arrays["semantic_fragment_frame"])
        semantic_class = np.asarray(arrays["semantic_fragment_class"])
        semantic_i = np.asarray(arrays["semantic_fragment_full_indptr"])
        semantic_v = np.asarray(arrays["semantic_fragment_full_ids"])
        semantic_mass = np.asarray(arrays["semantic_fragment_full_mass"])
        return bool(
            semantic_frame.shape == (semantic_fragments,)
            and semantic_class.shape == (semantic_fragments,)
            and _valid_ragged(semantic_i, semantic_v, semantic_fragments, n)
            and semantic_mass.shape == semantic_v.shape
            and np.all(np.isfinite(semantic_mass))
            and np.all(semantic_mass >= 0)
            and np.all(semantic_frame >= 0)
            and np.all(semantic_frame < frames)
            and np.all(semantic_class >= 0)
            and np.all(semantic_class < len(classes))
        )


def v10_lifting_bank_is_complete(
    directory: Path,
    *,
    expected_scene_id: str | None = None,
    expected_git_commit: str | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> bool:
    try:
        directory = Path(directory)
        metadata = load_json(directory / "lifting_bank.json")
        identity = metadata.get("identity")
        if (
            metadata.get("schema") != V10_LIFTING_SCHEMA
            or metadata.get("lifting_source") != "M1-core+AM-full"
            or metadata.get("mask_source") != "SAM-everything"
            or metadata.get("feature_source") != "runtime-manifest-trained-feature"
            or metadata.get("config") != asdict(FragmentConfig())
            or not isinstance(identity, Mapping)
            or identity.get("schema") != V10_LIFTING_IDENTITY_SCHEMA
            or identity.get("scene_id") != metadata.get("scene_id")
            or identity.get("git_commit") != metadata.get("git_commit")
        ):
            return False
        if expected_scene_id is not None and metadata.get("scene_id") != expected_scene_id:
            return False
        if expected_git_commit is not None and metadata.get("git_commit") != expected_git_commit:
            return False
        if expected_identity is not None and dict(identity) != dict(expected_identity):
            return False
        return _v10_arrays_are_valid(metadata, directory / "lifting_bank.npz")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def compatible_lifting_bank_is_complete(
    directory: Path,
    *,
    expected_scene_id: str | None = None,
) -> bool:
    """Accept exactly the immutable V9 native and V10 lifting schemas."""

    try:
        schema = load_json(Path(directory) / "lifting_bank.json").get("schema")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if schema == V9_LIFTING_SCHEMA:
        return v9_lifting_bank_is_complete(
            Path(directory), expected_scene_id=expected_scene_id
        )
    if schema == V10_LIFTING_SCHEMA:
        return v10_lifting_bank_is_complete(
            Path(directory), expected_scene_id=expected_scene_id
        )
    return False


def load_compatible_lifting_bank(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not compatible_lifting_bank_is_complete(Path(directory)):
        raise ValueError(f"invalid V9/V10 lifting bank: {directory}")
    metadata = load_json(Path(directory) / "lifting_bank.json")
    with np.load(Path(directory) / "lifting_bank.npz", allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    return metadata, arrays


def _camera_paths(inputs: V10LiftingInputs) -> dict[str, Path]:
    return {
        "rgb": inputs.rgb_ply,
        "feature": inputs.feature_ply,
        "sparse": inputs.sparse,
        "images": inputs.images,
        "masks": inputs.grounded_masks,
        "labels": inputs.grounded_labels,
    }


def run_v10_lifting_bank(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    output_dir: Path,
    git_commit: str,
    segment_everything_root: Path | None = None,
    feature_ply: Path | None = None,
    label_features: Path | None = None,
    classes: Sequence[str] = DEFAULT_CLASSES,
) -> dict[str, Any]:
    """Generate one isolated lifting bank; never train or mutate source assets."""

    scale = float(scene.get("scene_scale_m_per_unit", 0.0))
    if scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    inputs = resolve_v10_lifting_inputs(
        scene,
        segment_everything_root=segment_everything_root,
        feature_ply=feature_ply,
        label_features=label_features,
    )
    identity = build_v10_lifting_identity(
        scene_id=scene_id, git_commit=git_commit, inputs=inputs, classes=classes
    )
    output_dir = Path(output_dir).resolve()
    try:
        existing = load_json(output_dir / "lifting_bank.json")
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, Mapping) and existing.get("schema") == V9_LIFTING_SCHEMA:
        if v9_lifting_bank_is_complete(output_dir, expected_scene_id=scene_id):
            return dict(existing)
    if isinstance(existing, Mapping) and existing.get("schema") == V10_LIFTING_SCHEMA:
        if v10_lifting_bank_is_complete(
            output_dir,
            expected_scene_id=scene_id,
            expected_git_commit=git_commit,
            expected_identity=identity,
        ):
            return dict(existing)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    complete_parts = sorted(
        path
        for path in output_dir.parent.glob(f".{output_dir.name}.part-*")
        if path.is_dir()
        and v10_lifting_bank_is_complete(
            path,
            expected_scene_id=scene_id,
            expected_git_commit=git_commit,
            expected_identity=identity,
        )
    )
    if complete_parts:
        if output_dir.exists():
            if output_dir.is_dir() and not any(output_dir.iterdir()):
                output_dir.rmdir()
            else:
                raise RuntimeError(
                    f"refusing to replace occupied lifting artifact: {output_dir}"
                )
        staging = complete_parts[0]
        os.replace(staging, output_dir)
        return dict(load_json(output_dir / "lifting_bank.json"))
    if output_dir.exists():
        if output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
        else:
            occupied = sorted(str(path) for path in output_dir.iterdir())
            raise RuntimeError(
                f"refusing to overwrite incomplete lifting artifact: {occupied}"
            )

    import torch
    from gaussian_renderer import render_with_max_contributor
    from scene import FeatureGaussianModel, GaussianModel
    from .v9_lifting_worker import (
        _load_cameras,
        _record,
        _serialize,
        grounded_payload,
        render_alpha_mass,
        segment_everything_payload,
    )

    started = time.monotonic()
    rgb = GaussianModel(0)
    rgb.load_ply(str(inputs.rgb_ply))
    feature = FeatureGaussianModel(32, 32)
    feature.load_ply(str(inputs.feature_ply))
    rgb_xyz = rgb.get_xyz.detach().cpu().numpy()
    feature_xyz = feature.get_xyz.detach().cpu().numpy()
    if rgb_xyz.shape != feature_xyz.shape or not np.allclose(
        rgb_xyz, feature_xyz, atol=1e-6, rtol=0
    ):
        raise ValueError("RGB and feature Gaussian order differs")
    point_count = len(rgb_xyz)
    affinity = feature.get_point_features.detach().cpu().numpy()
    semantic = feature.get_point_semantic_features.detach().cpu().numpy()
    affinity /= np.maximum(np.linalg.norm(affinity, axis=1, keepdims=True), 1e-12)
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-12)
    loaded_labels = torch.load(inputs.label_features, map_location="cpu")
    if not isinstance(loaded_labels, torch.Tensor) or loaded_labels.ndim != 2:
        raise ValueError("label feature codebook must be a matrix")
    label_matrix = loaded_labels.detach().cpu().numpy().astype(np.float32)
    if label_matrix.shape != (len(classes), semantic.shape[1]):
        raise ValueError("label codebook does not match semantic feature")
    label_matrix /= np.maximum(np.linalg.norm(label_matrix, axis=1, keepdims=True), 1e-12)

    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    geometry_records: list[Any] = []
    semantic_records: list[Any] = []
    geometry_offset = 0
    semantic_offset = 0
    cameras = sorted(_load_cameras(_camera_paths(inputs)), key=lambda item: item.image_name)
    for frame_id, camera in enumerate(cameras):
        geometry_payload = segment_everything_payload(
            inputs.segment_everything_root, camera
        )
        semantic_payload = grounded_payload(camera)
        geometry_masks = geometry_payload.masks
        assert geometry_masks is not None
        semantic_masks = (
            np.zeros((0, camera.image_height, camera.image_width), dtype=bool)
            if semantic_payload.masks is None
            else semantic_payload.masks
        )
        combined_masks = np.concatenate((geometry_masks, semantic_masks), axis=0)
        alpha_all = render_alpha_mass(
            camera, rgb, pipeline, background, combined_masks, point_count
        )
        boundary = len(geometry_masks)
        alpha_geometry = AttributionMass(
            "AM",
            alpha_all.inside_mass[:boundary],
            alpha_all.visible_mass,
            alpha_all.valid_pixel_count,
            False,
        )
        alpha_semantic = AttributionMass(
            "AM",
            alpha_all.inside_mass[boundary:],
            alpha_all.visible_mass,
            alpha_all.valid_pixel_count,
            semantic_payload.abstained,
        )
        with torch.no_grad():
            rendered = render_with_max_contributor(camera, rgb, pipeline, background)
        maximum = mass_from_max_contributor(
            rendered["max_contributor"].detach().cpu().numpy(),
            rendered["max_contribute"].detach().cpu().numpy(),
            geometry_masks,
            point_count,
        )
        geometry_fragments = hybrid_fragments(
            maximum, alpha_geometry, frame_id, geometry_offset
        )
        semantic_fragments = fragments_from_mass(
            alpha_semantic,
            frame_id,
            semantic_offset,
            config=FragmentConfig(fragment_min_core=0, fragment_min_full=1),
        )
        geometry_records.append(
            _record(
                frame_id,
                str(camera.image_name),
                alpha_geometry,
                geometry_fragments,
                geometry_payload.labels,
            )
        )
        semantic_records.append(
            _record(
                frame_id,
                str(camera.image_name),
                alpha_semantic,
                semantic_fragments,
                semantic_payload.labels,
            )
        )
        geometry_offset += geometry_payload.count
        semantic_offset += semantic_payload.count

    metadata = {
        "schema": V10_LIFTING_SCHEMA,
        "scene_id": str(scene_id),
        "git_commit": str(git_commit),
        "identity": identity,
        "lifting_source": "M1-core+AM-full",
        "mask_source": "SAM-everything",
        "feature_source": "runtime-manifest-trained-feature",
        "point_count": int(point_count),
        "frame_count": len(geometry_records),
        "fragment_count": sum(len(row.fragments) for row in geometry_records),
        "semantic_fragment_count": sum(len(row.fragments) for row in semantic_records),
        "mask_count": sum(row.mask_count for row in geometry_records),
        "grounded_abstention_frame_count": sum(row.abstained for row in semantic_records),
        "classes": list(classes),
        "config": asdict(FragmentConfig()),
        "arrays_npz": "lifting_bank.npz",
        "runtime_seconds": float(time.monotonic() - started),
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.part-", dir=output_dir.parent)
    )
    _serialize(
        staging,
        xyz_m=feature_xyz * scale,
        affinity=affinity,
        semantic=semantic,
        label_features=label_matrix,
        geometry=geometry_records,
        semantics=semantic_records,
    )
    write_json(staging / "lifting_bank.json", metadata)
    if not v10_lifting_bank_is_complete(
        staging,
        expected_scene_id=scene_id,
        expected_git_commit=git_commit,
        expected_identity=identity,
    ):
        raise RuntimeError("serialized V10 lifting bank is invalid")
    if output_dir.exists():
        raise RuntimeError(f"refusing to replace occupied lifting artifact: {output_dir}")
    os.replace(staging, output_dir)
    return metadata


LiftingWorker = Callable[..., Mapping[str, Any]]


def ensure_v10_lifting_banks(
    *,
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    output_root: Path,
    git_commit: str,
    segment_everything_root: Path | None = None,
    segment_everything_by_scene: Mapping[str, Path] | None = None,
    feature_ply_by_scene: Mapping[str, Path] | None = None,
    label_features: Path | None = None,
    worker: LiftingWorker | None = None,
) -> dict[str, Any]:
    """Generate only missing scenes, sequentially, from registered runtime assets."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    active_worker = run_v10_lifting_bank if worker is None else worker
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for scene_id in map(str, scene_ids):
        if scene_id not in scenes:
            raise KeyError(f"runtime manifest lacks {scene_id}")
        target = output_root / scene_id
        try:
            existing_metadata = load_json(target / "lifting_bank.json")
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_metadata = None
        if (
            isinstance(existing_metadata, Mapping)
            and existing_metadata.get("schema") == V9_LIFTING_SCHEMA
            and v9_lifting_bank_is_complete(target, expected_scene_id=scene_id)
        ):
            records.append(
                {
                    "scene_id": scene_id,
                    "status": "reused",
                    "schema": existing_metadata["schema"],
                    "path": str(target),
                }
            )
            continue
        scene = scenes[scene_id]
        sam_scene = (
            Path(segment_everything_by_scene[scene_id])
            if segment_everything_by_scene is not None
            and scene_id in segment_everything_by_scene
            else (
                Path(segment_everything_root) / scene_id
                if segment_everything_root is not None
                else None
            )
        )
        selected_feature = (
            Path(feature_ply_by_scene[scene_id])
            if feature_ply_by_scene is not None and scene_id in feature_ply_by_scene
            else None
        )
        # Resolve before invoking the GPU worker.  Missing inputs therefore
        # fail without launching any process, downloader, or trainer.
        inputs = resolve_v10_lifting_inputs(
            scene,
            segment_everything_root=sam_scene,
            feature_ply=selected_feature,
            label_features=label_features,
        )
        if (
            isinstance(existing_metadata, Mapping)
            and existing_metadata.get("schema") == V10_LIFTING_SCHEMA
        ):
            existing_identity = existing_metadata.get("identity")
            producer_commit = (
                str(existing_identity.get("git_commit", "")).strip()
                if isinstance(existing_identity, Mapping)
                else ""
            )
            expected_identity = (
                build_v10_lifting_identity(
                    scene_id=scene_id,
                    git_commit=producer_commit,
                    inputs=inputs,
                )
                if producer_commit
                else None
            )
            if expected_identity is not None and v10_lifting_bank_is_complete(
                target,
                expected_scene_id=scene_id,
                expected_git_commit=producer_commit,
                expected_identity=expected_identity,
            ):
                records.append(
                    {
                        "scene_id": scene_id,
                        "status": "reused",
                        "schema": V10_LIFTING_SCHEMA,
                        "path": str(target),
                    }
                )
                continue
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"refusing to overwrite incomplete lifting directory: {target}")
        scene_started = time.monotonic()
        metadata = active_worker(
            scene_id=scene_id,
            scene=scene,
            output_dir=target,
            git_commit=git_commit,
            segment_everything_root=sam_scene,
            feature_ply=selected_feature,
            label_features=label_features,
        )
        if not compatible_lifting_bank_is_complete(target, expected_scene_id=scene_id):
            raise RuntimeError(f"V10 worker left an incomplete lifting bank: {target}")
        records.append(
            {
                "scene_id": scene_id,
                "status": "completed",
                "schema": str(metadata["schema"]),
                "path": str(target),
                "seconds": float(time.monotonic() - scene_started),
            }
        )
    summary = {
        "schema": "saga-v10-lifting-run-summary-v1",
        "scene_count": len(records),
        "runs": records,
        "runtime_seconds": float(time.monotonic() - started),
    }
    write_json(output_root / "v10_lifting_summary.json", summary)
    return summary


run_v10_lifting_banks = ensure_v10_lifting_banks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--segment-everything-root", type=Path)
    parser.add_argument("--label-features", type=Path)
    args = parser.parse_args(argv)
    result = ensure_v10_lifting_banks(
        runtime_manifest=args.runtime_manifest,
        scene_ids=args.scenes,
        output_root=args.output_root,
        git_commit=args.git_commit,
        segment_everything_root=args.segment_everything_root,
        label_features=args.label_features,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
