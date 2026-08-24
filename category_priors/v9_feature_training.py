from __future__ import annotations

"""Isolated, dual-source 10k feature training for the V9 ObjectBank.

The affinity head is supervised by class-agnostic SAM-everything masks while
the semantic head is supervised by Grounded-SAM masks and labels.  This module
only prepares and executes that one registered training condition; it does not
change the historical scene assets or the legacy pipeline entrypoint.
"""

import json
import os
import subprocess
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest


V9_FEATURE_ITERATIONS = 10_000
V9_FEATURE_SEED = 42
V9_FEATURE_SCHEMA = "saga-v9-dual-source-feature-v1"


@dataclass(frozen=True)
class V9FeatureInputs:
    python_bin: Path
    images: Path
    sparse: Path
    point_cloud: Path
    affinity_masks: Path
    affinity_mask_scales: Path
    semantic_masks: Path
    semantic_labels: Path
    semantic_mask_scales: Path
    semantic_label_features: Path


@dataclass(frozen=True)
class V9FeaturePaths:
    root: Path
    model: Path
    feature_ply: Path
    scale_gate: Path
    progress: Path
    log: Path
    record: Path


@dataclass(frozen=True)
class V9AffinityInputPaths:
    root: Path
    masks: Path
    mask_scales: Path
    scale_model: Path
    scale_progress: Path
    scale_log: Path
    materialization_record: Path
    scale_record: Path


def v9_feature_training_paths(
    output_root: str | Path, scene_id: str
) -> V9FeaturePaths:
    scene_name = str(scene_id)
    if not scene_name or Path(scene_name).name != scene_name:
        raise ValueError(f"invalid scene ID: {scene_id!r}")
    root = Path(output_root).resolve() / scene_name
    return V9FeaturePaths(
        root=root,
        model=root / "model",
        feature_ply=root / "contrastive_feature_point_cloud_10k.ply",
        scale_gate=root / "scale_gate_10k.pt",
        progress=root / "train_progress.txt",
        log=root / "train_10k.log",
        record=root / "train_10k.json",
    )


def v9_affinity_input_paths(
    output_root: str | Path, scene_id: str
) -> V9AffinityInputPaths:
    feature_paths = v9_feature_training_paths(output_root, scene_id)
    root = feature_paths.root / "affinity-inputs"
    return V9AffinityInputPaths(
        root=root,
        masks=root / "sam_everything_masks",
        mask_scales=root / "sam_everything_mask_scales",
        scale_model=root / "scale-model",
        scale_progress=root / "scale_progress.txt",
        scale_log=root / "get_scale.log",
        materialization_record=root / "sam_everything_materialization.json",
        scale_record=root / "get_scale.json",
    )


def _scene_path(
    scene: Mapping[str, Any], key: str, default: str | None = None
) -> Path:
    value = scene.get(key, default)
    if value in {None, ""}:
        raise ValueError(f"scene runtime entry is missing {key!r}")
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(str(scene["base_path"])) / path
    return path.resolve()


def _default_point_cloud(base_path: Path) -> Path:
    standard = base_path / "output_models/point_cloud/iteration_30000/point_cloud.ply"
    alternate = base_path / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
    return alternate if not standard.is_file() and alternate.is_file() else standard


def resolve_v9_feature_inputs(scene: Mapping[str, Any]) -> V9FeatureInputs:
    """Resolve one scene while keeping both supervision sources explicit.

    ``sam_everything_*`` is intentionally required.  Falling back to the
    Grounded-SAM directory would silently turn the registered dual-source arm
    into the historical single-source training condition.
    """

    base = Path(str(scene["base_path"])).resolve()
    semantic_labels = _scene_path(
        scene, "grounded_labels_path", "saga/labels"
    )
    point_cloud = (
        _scene_path(scene, "point_cloud_path")
        if scene.get("point_cloud_path")
        else _default_point_cloud(base).resolve()
    )
    python_value = scene.get("python_bin")
    if not python_value:
        raise ValueError("scene runtime entry is missing 'python_bin'")
    python_bin = Path(str(python_value))
    if not python_bin.is_absolute():
        python_bin = (base / python_bin).resolve()
    else:
        python_bin = python_bin.resolve()
    return V9FeatureInputs(
        python_bin=python_bin,
        images=_scene_path(
            scene, "images_path", "fastRecon/dense/sparse/0/images"
        ),
        sparse=_scene_path(
            scene, "sparse_path", "fastRecon/dense/sparse/0"
        ),
        point_cloud=point_cloud,
        affinity_masks=_scene_path(scene, "sam_everything_masks_path"),
        affinity_mask_scales=_scene_path(
            scene, "sam_everything_mask_scales_path"
        ),
        semantic_masks=_scene_path(
            scene, "grounded_masks_path", "saga/masks"
        ),
        semantic_labels=semantic_labels,
        semantic_mask_scales=_scene_path(
            scene, "grounded_mask_scales_path", "saga/mask_scales"
        ),
        semantic_label_features=_scene_path(
            scene,
            "grounded_label_features_path",
            str(semantic_labels / "label_features.pt"),
        ),
    )


def _pt_files(
    directory: Path,
    *,
    exclude: Path | None = None,
    allow_empty: bool = False,
) -> dict[str, Path]:
    files = {
        path.stem: path.resolve()
        for path in directory.glob("*.pt")
        if path.is_file() and (exclude is None or path.resolve() != exclude.resolve())
    }
    if not files and not allow_empty:
        raise ValueError(f"no trainer-ready .pt frames found under {directory}")
    return files


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _ply_header(path: Path) -> dict[str, Any]:
    """Parse enough of a PLY header to prove a feature artifact is complete."""

    if not path.is_file():
        raise FileNotFoundError(path)
    lines: list[str] = []
    header_bytes = 0
    with path.open("rb") as handle:
        while header_bytes <= 1_048_576:
            raw_line = handle.readline()
            if not raw_line:
                break
            header_bytes += len(raw_line)
            try:
                line = raw_line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"PLY header is not ASCII: {path}") from error
            lines.append(line)
            if line == "end_header":
                break
    if not lines or lines[0] != "ply" or lines[-1] != "end_header":
        raise ValueError(f"incomplete PLY header: {path}")
    format_rows = [line.split() for line in lines if line.startswith("format ")]
    if len(format_rows) != 1 or format_rows[0][1] not in {
        "ascii",
        "binary_little_endian",
        "binary_big_endian",
    }:
        raise ValueError(f"unsupported PLY format: {path}")
    vertex_count: int | None = None
    properties: list[str] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and len(parts) == 3 and parts[0] == "property":
            properties.append(parts[2])
        elif in_vertex and parts[:2] == ["property", "list"]:
            raise ValueError(f"list-valued vertex property is unsupported: {path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"PLY has no non-empty vertex element: {path}")
    return {
        "format": format_rows[0][1],
        "vertex_count": vertex_count,
        "properties": properties,
        "header_bytes": header_bytes,
        "file_size": path.stat().st_size,
    }


def _directory_identity(directory: Path, files: Mapping[str, Path]) -> dict[str, Any]:
    stats = [path.stat() for path in files.values()]
    return {
        "path": str(directory.resolve()),
        "file_count": len(files),
        "total_bytes": int(sum(item.st_size for item in stats)),
        "latest_mtime_ns": (
            int(max(item.st_mtime_ns for item in stats)) if stats else None
        ),
    }


def validate_v9_feature_inputs(inputs: V9FeatureInputs) -> dict[str, Any]:
    for label, directory in (
        ("images", inputs.images),
        ("sparse", inputs.sparse),
        ("affinity masks", inputs.affinity_masks),
        ("affinity mask scales", inputs.affinity_mask_scales),
        ("semantic masks", inputs.semantic_masks),
        ("semantic labels", inputs.semantic_labels),
        ("semantic mask scales", inputs.semantic_mask_scales),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")
    for label, path in (
        ("Python interpreter", inputs.python_bin),
        ("30k point cloud", inputs.point_cloud),
        ("semantic label features", inputs.semantic_label_features),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    if inputs.affinity_masks == inputs.semantic_masks:
        raise ValueError(
            "affinity and semantic masks resolve to the same directory; "
            "V9 requires SAM-everything affinity masks and Grounded-SAM semantic masks"
        )

    affinity = _pt_files(inputs.affinity_masks)
    affinity_scales = _pt_files(inputs.affinity_mask_scales)
    semantic = _pt_files(inputs.semantic_masks, allow_empty=True)
    semantic_labels = _pt_files(
        inputs.semantic_labels,
        exclude=inputs.semantic_label_features,
        allow_empty=True,
    )
    semantic_scales = _pt_files(inputs.semantic_mask_scales, allow_empty=True)
    expected = set(affinity)
    if set(affinity_scales) != expected:
        missing = sorted(expected - set(affinity_scales))[:5]
        extra = sorted(set(affinity_scales) - expected)[:5]
        raise ValueError(
            "SAM-everything mask scales frame identity differs from masks; "
            f"missing={missing}, extra={extra}"
        )

    # Grounded-SAM is allowed to abstain on a frame.  The three semantic
    # artifacts must agree with each other and may form only a subset of the
    # class-agnostic affinity frames; the trainer disables semantic loss on
    # the missing frames instead of fabricating background supervision.
    semantic_expected = set(semantic)
    for label, frames in (
        ("Grounded-SAM labels", semantic_labels),
        ("Grounded-SAM mask scales", semantic_scales),
    ):
        if set(frames) != semantic_expected:
            missing = sorted(semantic_expected - set(frames))[:5]
            extra = sorted(set(frames) - semantic_expected)[:5]
            raise ValueError(
                f"{label} frame identity differs from Grounded-SAM masks; "
                f"missing={missing}, extra={extra}"
            )
    if not semantic_expected.issubset(expected):
        raise ValueError(
            "Grounded-SAM supervision references frames absent from SAM-everything: "
            f"{sorted(semantic_expected - expected)[:5]}"
        )

    image_stems = {
        path.stem
        for path in inputs.images.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if not expected.issubset(image_stems):
        raise ValueError(
            "mask supervision references frames absent from the image directory: "
            f"{sorted(expected - image_stems)[:5]}"
        )

    frame_stems = sorted(expected)
    semantic_frame_stems = sorted(semantic_expected)
    return {
        "schema": V9_FEATURE_SCHEMA,
        "frame_count": len(frame_stems),
        "frame_stems": frame_stems,
        "semantic_frame_count": len(semantic_frame_stems),
        "semantic_frame_stems": semantic_frame_stems,
        "semantic_abstention_frame_count": len(expected - semantic_expected),
        "python_bin": _file_identity(inputs.python_bin),
        "point_cloud": {
            **_file_identity(inputs.point_cloud),
            "ply": _ply_header(inputs.point_cloud),
        },
        "semantic_label_features": _file_identity(inputs.semantic_label_features),
        "images": str(inputs.images.resolve()),
        "sparse": str(inputs.sparse.resolve()),
        "affinity_masks": _directory_identity(inputs.affinity_masks, affinity),
        "affinity_mask_scales": _directory_identity(
            inputs.affinity_mask_scales, affinity_scales
        ),
        "semantic_masks": _directory_identity(inputs.semantic_masks, semantic),
        "semantic_labels": _directory_identity(
            inputs.semantic_labels, semantic_labels
        ),
        "semantic_mask_scales": _directory_identity(
            inputs.semantic_mask_scales, semantic_scales
        ),
    }


def _torch(torch_module: Any | None = None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        import torch
    except ImportError as error:  # pragma: no cover - cloud runtime has torch
        raise RuntimeError(
            "materializing trainer masks and validating mask scales requires torch"
        ) from error
    return torch


def _read_v8_packed_mask(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            packed = np.asarray(payload["packed"], dtype=np.uint8)
            count = int(np.asarray(payload["count"]).item())
            height = int(np.asarray(payload["height"]).item())
            width = int(np.asarray(payload["width"]).item())
    except (OSError, ValueError, KeyError, EOFError) as error:
        raise ValueError(f"invalid V8 SAM-everything frame: {path}") from error
    if count < 0 or height <= 0 or width <= 0:
        raise ValueError(f"invalid V8 SAM-everything dimensions: {path}")
    expected_shape = (count, (height * width + 7) // 8)
    if packed.shape != expected_shape:
        raise ValueError(
            f"packed SAM mask shape {packed.shape} != {expected_shape}: {path}"
        )
    masks = np.unpackbits(
        packed, axis=1, count=height * width
    ).reshape(count, height, width).astype(np.bool_, copy=False)
    # Repacking is an inexpensive exactness check and catches malformed tail
    # bits or accidental shape reinterpretation before writing trainer inputs.
    if not np.array_equal(
        np.packbits(masks.reshape(count, height * width), axis=1), packed
    ):
        raise ValueError(f"SAM mask unpack/repack identity failed: {path}")
    return masks, {"count": count, "height": height, "width": width}


def validate_v8_sam_everything_source(
    source_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    summary_path = source / "summary.json"
    if not source.is_dir() or not summary_path.is_file():
        raise FileNotFoundError(f"V8 SAM-everything summary not found: {summary_path}")
    payload = load_json(summary_path)
    if payload.get("schema") not in {
        "saga-v8-segment-everything-v1",  # read-only registered dev8 source
        "saga-v9-segment-everything-v1",
    }:
        raise ValueError(f"unexpected SAM-everything schema: {summary_path}")
    rows = payload.get("images")
    if not isinstance(rows, list) or int(payload.get("image_count", -1)) != len(rows):
        raise ValueError(f"SAM-everything summary image count is invalid: {summary_path}")
    expected_stems: set[str] = set()
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"SAM-everything summary row is invalid: {summary_path}")
        stem = Path(str(row["image"])).stem
        if not stem or stem in expected_stems:
            raise ValueError(f"duplicate/invalid SAM-everything frame stem: {stem!r}")
        expected_stems.add(stem)
        path = source / f"{stem}.npz"
        _, metadata = _read_v8_packed_mask(path)
        registered = (
            int(row["mask_count"]), int(row["height"]), int(row["width"])
        )
        observed = (
            metadata["count"], metadata["height"], metadata["width"]
        )
        if observed != registered:
            raise ValueError(
                f"SAM-everything summary/frame mismatch for {stem}: "
                f"{registered} != {observed}"
            )
        stat = path.stat()
        records.append({
            "stem": stem,
            **metadata,
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        })
    actual_stems = {path.stem for path in source.glob("*.npz") if path.is_file()}
    if actual_stems != expected_stems:
        raise ValueError(
            "packed SAM-everything stems differ from summary; "
            f"missing={sorted(expected_stems - actual_stems)[:5]}, "
            f"extra={sorted(actual_stems - expected_stems)[:5]}"
        )
    if not records or not any(row["count"] >= 2 for row in records):
        raise ValueError(
            "SAM-everything source has no frame with the two masks required by training"
        )
    observed_mask_count = int(sum(row["count"] for row in records))
    if int(payload.get("mask_count", -1)) != observed_mask_count:
        raise ValueError(
            "SAM-everything summary total mask count does not match frame payloads"
        )
    return {
        "schema": "saga-v9-sam-everything-source-v1",
        "source_root": str(source),
        "summary": _file_identity(summary_path),
        "frame_count": len(records),
        "mask_count": observed_mask_count,
        "frames": records,
    }


def _materialized_mask_matches(
    path: Path, expected: np.ndarray, torch_module: Any
) -> bool:
    if not path.is_file():
        return False
    try:
        tensor = torch_module.load(path, map_location="cpu")
        expected_dtype = getattr(torch_module, "bool", None)
        if expected_dtype is not None and getattr(tensor, "dtype", None) != expected_dtype:
            return False
        observed = np.asarray(tensor.cpu().numpy(), dtype=np.bool_)
    except (OSError, ValueError, TypeError, EOFError, RuntimeError, AttributeError):
        return False
    return observed.shape == expected.shape and np.array_equal(observed, expected)


def materialize_v9_sam_everything_masks(
    source_root: str | Path,
    target_root: str | Path,
    *,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Unpack V8 NPZ masks losslessly into isolated trainer-ready bool tensors."""

    source_identity = validate_v8_sam_everything_source(source_root)
    source = Path(source_root).resolve()
    target = Path(target_root).resolve()
    target.mkdir(parents=True, exist_ok=True)
    expected_stems = {str(row["stem"]) for row in source_identity["frames"]}
    extras = {
        path.stem for path in target.glob("*.pt") if path.is_file()
    } - expected_stems
    if extras:
        raise RuntimeError(
            "isolated SAM materialization contains foreign frames: "
            f"{sorted(extras)[:5]}"
        )
    torch_api = _torch(torch_module)
    reused = 0
    written = 0
    for row in source_identity["frames"]:
        stem = str(row["stem"])
        masks, _ = _read_v8_packed_mask(source / f"{stem}.npz")
        destination = target / f"{stem}.pt"
        if _materialized_mask_matches(destination, masks, torch_api):
            reused += 1
            continue
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            torch_api.save(torch_api.from_numpy(masks.copy()).bool(), temporary)
            if not _materialized_mask_matches(temporary, masks, torch_api):
                raise RuntimeError(
                    f"materialized SAM mask failed exact verification: {destination}"
                )
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        written += 1
    return {
        "kind": "v9_sam_everything_materialization",
        "schema": V9_FEATURE_SCHEMA,
        "source": source_identity,
        "target_root": str(target),
        "frame_count": len(expected_stems),
        "mask_count": source_identity["mask_count"],
        "written": written,
        "reused": reused,
    }


def build_v9_affinity_scale_command(
    *,
    workspace: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    output_root: str | Path,
) -> tuple[list[str], V9AffinityInputPaths]:
    workspace_path = Path(workspace).resolve()
    scale_script = workspace_path / "get_scale.py"
    if not scale_script.is_file():
        raise FileNotFoundError(f"scale script not found: {scale_script}")
    paths = v9_affinity_input_paths(output_root, scene_id)
    base = Path(str(scene["base_path"])).resolve()
    python_value = scene.get("python_bin")
    if not python_value:
        raise ValueError("scene runtime entry is missing 'python_bin'")
    python_bin = Path(str(python_value))
    python_bin = (
        (base / python_bin).resolve() if not python_bin.is_absolute()
        else python_bin.resolve()
    )
    if not python_bin.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python_bin}")
    images = _scene_path(
        scene, "images_path", "fastRecon/dense/sparse/0/images"
    )
    sparse = _scene_path(
        scene, "sparse_path", "fastRecon/dense/sparse/0"
    )
    point_cloud = (
        _scene_path(scene, "point_cloud_path")
        if scene.get("point_cloud_path")
        else _default_point_cloud(base).resolve()
    )
    for label, directory in (("images", images), ("sparse", sparse)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")
    if not point_cloud.is_file():
        raise FileNotFoundError(f"30k point cloud not found: {point_cloud}")
    command = [
        str(python_bin),
        str(scale_script),
        "--model_path", str(paths.scale_model),
        "--progress_path", str(paths.scale_progress),
        "--sh_degree", "0",
        "--images_path", str(images),
        "--sparse_path", str(sparse),
        "--point_cloud_path", str(point_cloud),
        "--masks_path", str(paths.masks),
        "--mask_scales_path", str(paths.mask_scales),
    ]
    return command, paths


def _scale_outputs_complete(
    paths: V9AffinityInputPaths,
    source_identity: Mapping[str, Any],
    torch_module: Any,
) -> bool:
    expected = {str(row["stem"]): int(row["count"]) for row in source_identity["frames"]}
    actual = {
        path.stem: path for path in paths.mask_scales.glob("*.pt") if path.is_file()
    }
    if set(actual) != set(expected):
        return False
    for stem, count in expected.items():
        try:
            tensor = torch_module.load(actual[stem], map_location="cpu")
            shape = tuple(tensor.shape)
        except (OSError, ValueError, TypeError, EOFError, RuntimeError, AttributeError):
            return False
        if shape != (count,):
            return False
    return _progress_complete(paths.scale_progress)


def prepare_v9_affinity_inputs(
    *,
    workspace: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    packed_masks_root: str | Path,
    output_root: str | Path,
    git_commit: str,
    resume: bool = True,
    dry_run: bool = False,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Materialize exact masks and generate physical mask scales, resumably."""

    source_identity = validate_v8_sam_everything_source(packed_masks_root)
    command, paths = build_v9_affinity_scale_command(
        workspace=workspace,
        scene=scene,
        scene_id=scene_id,
        output_root=output_root,
    )
    identity = {
        "schema": V9_FEATURE_SCHEMA,
        "scene_id": str(scene_id),
        "git_commit": str(git_commit),
        "source": source_identity,
        "masks": str(paths.masks),
        "mask_scales": str(paths.mask_scales),
        "command": command,
    }
    if dry_run:
        return {
            "kind": "v9_affinity_input_preparation",
            "status": "planned",
            "identity": identity,
            "scene_overrides": {
                "sam_everything_masks_path": str(paths.masks),
                "sam_everything_mask_scales_path": str(paths.mask_scales),
            },
        }

    torch_api = _torch(torch_module)
    materialization = materialize_v9_sam_everything_masks(
        packed_masks_root, paths.masks, torch_module=torch_api
    )
    write_json(paths.materialization_record, materialization)
    existing_scale_files = {
        path.stem for path in paths.mask_scales.glob("*.pt") if path.is_file()
    } if paths.mask_scales.is_dir() else set()
    try:
        record = load_json(paths.scale_record)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        record = None
    if record is None and existing_scale_files:
        raise RuntimeError(
            f"{scene_id}: isolated scale outputs exist without a run record"
        )
    if record is not None and record.get("identity") != identity:
        raise RuntimeError(
            f"{scene_id}: existing scale run belongs to a different input identity"
        )
    if resume and record is not None and record.get("status") == "complete" and (
        _scale_outputs_complete(paths, source_identity, torch_api)
    ):
        status = "skipped_complete"
        runtime = float(record.get("runtime_seconds", 0.0))
    else:
        paths.mask_scales.mkdir(parents=True, exist_ok=True)
        paths.scale_model.mkdir(parents=True, exist_ok=True)
        running = {
            "kind": "v9_affinity_scale_run",
            "status": "running",
            "identity": identity,
        }
        write_json(paths.scale_record, running)
        started = time.perf_counter()
        with paths.scale_log.open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(
                command,
                cwd=Path(workspace).resolve(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        runtime = float(time.perf_counter() - started)
        status = "complete" if (
            result.returncode == 0
            and _scale_outputs_complete(paths, source_identity, torch_api)
        ) else "failed"
        write_json(paths.scale_record, {
            **running,
            "status": status,
            "return_code": int(result.returncode),
            "runtime_seconds": runtime,
        })
        if status == "failed":
            raise RuntimeError(
                f"V9 affinity scale generation failed for {scene_id}; "
                f"see {paths.scale_log}"
            )
    return {
        "kind": "v9_affinity_input_preparation",
        "status": status,
        "scene_id": str(scene_id),
        "runtime_seconds": runtime,
        "materialization": materialization,
        "scene_overrides": {
            "sam_everything_masks_path": str(paths.masks),
            "sam_everything_mask_scales_path": str(paths.mask_scales),
        },
    }


def _ensure_isolated_outputs(
    scene: Mapping[str, Any], paths: V9FeaturePaths
) -> None:
    base = Path(str(scene["base_path"])).resolve()
    historical = {
        (base / "saga/contrastive_feature_point_cloud.ply").resolve(),
        (base / "saga/scale_gate.pt").resolve(),
    }
    if paths.feature_ply.resolve() in historical or paths.scale_gate.resolve() in historical:
        raise ValueError("V9 feature outputs collide with historical scene assets")


def build_v9_feature_training_command(
    *,
    workspace: str | Path,
    scene: Mapping[str, Any],
    scene_id: str,
    output_root: str | Path,
) -> tuple[list[str], V9FeaturePaths, dict[str, Any]]:
    workspace_path = Path(workspace).resolve()
    trainer = workspace_path / "train_contrastive_feature.py"
    if not trainer.is_file():
        raise FileNotFoundError(f"feature trainer not found: {trainer}")
    inputs = resolve_v9_feature_inputs(scene)
    input_identity = validate_v9_feature_inputs(inputs)
    paths = v9_feature_training_paths(output_root, scene_id)
    _ensure_isolated_outputs(scene, paths)

    command = [
        str(inputs.python_bin),
        str(trainer),
        "--model_path", str(paths.model),
        "--progress_path", str(paths.progress),
        "--sh_degree", "0",
        "--feature_dim", "32",
        "--semantic_feature_dim", "32",
        "--images_path", str(inputs.images),
        "--sparse_path", str(inputs.sparse),
        "--point_cloud_path", str(inputs.point_cloud),
        "--masks_path", str(inputs.affinity_masks),
        "--mask_scales_path", str(inputs.affinity_mask_scales),
        "--semantic_masks_path", str(inputs.semantic_masks),
        "--semantic_labels_path", str(inputs.semantic_labels),
        "--semantic_mask_scales_path", str(inputs.semantic_mask_scales),
        "--semantic_label_features_path", str(inputs.semantic_label_features),
        "--contrastive_feature_point_cloud_path", str(paths.feature_ply),
        "--scale_gate_path", str(paths.scale_gate),
        "--num_sampled_rays", "1000",
        "--iterations", str(V9_FEATURE_ITERATIONS),
        "--seed", str(V9_FEATURE_SEED),
    ]
    identity = {
        "schema": V9_FEATURE_SCHEMA,
        "scene_id": str(scene_id),
        "iterations": V9_FEATURE_ITERATIONS,
        "seed": V9_FEATURE_SEED,
        "workspace": str(workspace_path),
        "trainer": _file_identity(trainer),
        "inputs": input_identity,
        "outputs": {
            "feature_ply": str(paths.feature_ply),
            "scale_gate": str(paths.scale_gate),
        },
        "command": command,
    }
    return command, paths, identity


def _valid_feature_ply(path: Path, *, expected_vertex_count: int) -> bool:
    if not path.is_file():
        return False
    try:
        header = _ply_header(path)
        properties = set(header["properties"])
        required = {
            "x", "y", "z", "opacity",
            *(f"f_{index}" for index in range(32)),
            *(f"sf_{index}" for index in range(32)),
        }
        if int(header["vertex_count"]) != int(expected_vertex_count):
            return False
        if not required.issubset(properties):
            return False
        # A header-only/truncated file cannot be accepted even if its declared
        # vertex count and property names look plausible.
        return int(header["file_size"]) > int(header["header_bytes"])
    except (OSError, ValueError, TypeError):
        return False


def _valid_scale_gate(path: Path) -> bool:
    """Check the integrity of the modern torch.save ZIP container."""

    if not path.is_file() or path.stat().st_size < 64 or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
            names = archive.namelist()
            data_entries = [
                name
                for name in names
                if "/data/" in name or name.startswith("data/")
            ]
            return (
                any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
                and len(data_entries) >= 2
            )
    except (OSError, zipfile.BadZipFile):
        return False


def _expected_vertex_count(identity: Mapping[str, Any]) -> int:
    try:
        return int(identity["inputs"]["point_cloud"]["ply"]["vertex_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("V9 feature identity lacks source PLY vertex count") from error


def _progress_complete(path: Path) -> bool:
    try:
        return int(path.read_text(encoding="utf-8").strip()) == 100
    except (FileNotFoundError, OSError, ValueError):
        return False


def v9_feature_training_complete(
    paths: V9FeaturePaths, identity: Mapping[str, Any], git_commit: str
) -> bool:
    try:
        record = load_json(paths.record)
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return (
        record.get("kind") == "v9_feature_training_run"
        and record.get("status") == "complete"
        and record.get("git_commit") == str(git_commit)
        and record.get("identity") == dict(identity)
        and _valid_feature_ply(
            paths.feature_ply,
            expected_vertex_count=_expected_vertex_count(identity),
        )
        and _valid_scale_gate(paths.scale_gate)
        and _progress_complete(paths.progress)
    )


def _identity_conflict(
    paths: V9FeaturePaths, identity: Mapping[str, Any], git_commit: str
) -> str | None:
    data_outputs_exist = paths.feature_ply.exists() or paths.scale_gate.exists()
    if not paths.record.is_file():
        return "isolated feature outputs exist without a run record" if data_outputs_exist else None
    try:
        record = load_json(paths.record)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return "run record is not parseable" if data_outputs_exist else None
    if record.get("identity") != dict(identity) or record.get("git_commit") != str(git_commit):
        return "existing isolated run belongs to a different input or code identity"
    return None


def execute_v9_feature_training(
    *,
    scene_manifest: str | Path,
    output_root: str | Path,
    workspace: str | Path,
    git_commit: str,
    scene_ids: Sequence[str],
    resume: bool = True,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    scenes = load_scene_runtime_manifest(scene_manifest)
    selected = [str(scene_id) for scene_id in scene_ids]
    missing = sorted(set(selected) - set(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    if len(selected) != len(set(selected)):
        raise ValueError("scene_ids contains duplicates")

    records: list[dict[str, Any]] = []
    workspace_path = Path(workspace).resolve()
    for scene_id in selected:
        command, paths, identity = build_v9_feature_training_command(
            workspace=workspace_path,
            scene=scenes[scene_id],
            scene_id=scene_id,
            output_root=output_root,
        )
        conflict = _identity_conflict(paths, identity, git_commit)
        if conflict:
            raise RuntimeError(f"{scene_id}: {conflict}; refusing to overwrite it")
        if resume and v9_feature_training_complete(paths, identity, git_commit):
            records.append({
                "scene_id": scene_id,
                "status": "skipped_complete",
                "root": str(paths.root),
            })
            continue
        if dry_run:
            records.append({
                "scene_id": scene_id,
                "status": "planned",
                "root": str(paths.root),
                "command": command,
            })
            continue

        paths.root.mkdir(parents=True, exist_ok=True)
        paths.model.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        running_record = {
            "kind": "v9_feature_training_run",
            "git_commit": str(git_commit),
            "scene_id": scene_id,
            "status": "running",
            "identity": identity,
        }
        write_json(paths.record, running_record)
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = str(V9_FEATURE_SEED)
        with paths.log.open("w", encoding="utf-8", newline="\n") as log:
            result = subprocess.run(
                command,
                cwd=workspace_path,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        runtime = float(time.perf_counter() - started)
        status = "complete" if (
            result.returncode == 0
            and _valid_feature_ply(
                paths.feature_ply,
                expected_vertex_count=_expected_vertex_count(identity),
            )
            and _valid_scale_gate(paths.scale_gate)
            and _progress_complete(paths.progress)
        ) else "failed"
        final_record = {
            **running_record,
            "status": status,
            "runtime_seconds": runtime,
            "return_code": int(result.returncode),
            "feature_ply": str(paths.feature_ply),
            "scale_gate": str(paths.scale_gate),
        }
        write_json(paths.record, final_record)
        records.append({
            "scene_id": scene_id,
            "status": status,
            "root": str(paths.root),
            "runtime_seconds": runtime,
        })
        if status == "failed" and not continue_on_error:
            raise RuntimeError(
                f"V9 10k feature training failed for {scene_id}; see {paths.log}"
            )

    return {
        "kind": "v9_feature_training_execution",
        "schema": V9_FEATURE_SCHEMA,
        "git_commit": str(git_commit),
        "iterations": V9_FEATURE_ITERATIONS,
        "seed": V9_FEATURE_SEED,
        "total": len(records),
        "complete": sum(
            row["status"] in {"complete", "skipped_complete"} for row in records
        ),
        "failed": sum(row["status"] == "failed" for row in records),
        "runs": records,
    }
