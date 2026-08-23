from __future__ import annotations

"""Sequential, resumable runners for V8 lifting banks.

The runner deliberately delegates GPU work to each scene's configured Python
environment.  It neither reads ground truth nor decides experimental gates.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .v7_runner import RuntimeScene, load_runtime_scenes
from .v8_masks import SAM_EVERYTHING_CONFIG, _mask_file_is_complete


MASK_SOURCES = ("G", "S")
LIFTING_SOURCES = ("M1", "AM")
LIFTING_ARMS = tuple(
    f"{mask}-{lifting}"
    for mask in MASK_SOURCES
    for lifting in LIFTING_SOURCES
)


def _valid_ragged(
    indptr: np.ndarray,
    ids: np.ndarray,
    row_count: int,
    point_count: int,
) -> bool:
    pointers = np.asarray(indptr, dtype=np.int64)
    values = np.asarray(ids, dtype=np.int64)
    return bool(
        pointers.shape == (int(row_count) + 1,)
        and int(pointers[0]) == 0
        and np.all(np.diff(pointers) >= 0)
        and int(pointers[-1]) == len(values)
        and np.all(values >= 0)
        and np.all(values < int(point_count))
    )


def lifting_bank_is_complete(
    output_dir: Path,
    *,
    expected_scene_id: str | None = None,
    expected_mask_source: str | None = None,
    expected_lifting_source: str | None = None,
    expected_contributor_audit: bool | None = None,
) -> bool:
    metadata_path = output_dir / "lifting_bank.json"
    arrays_path = output_dir / "lifting_bank.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != "saga-v8-lifting-bank-v1":
            return False
        if expected_scene_id is not None and metadata.get("scene_id") != expected_scene_id:
            return False
        if expected_mask_source is not None and metadata.get("mask_source") != expected_mask_source:
            return False
        if expected_lifting_source is not None and metadata.get("lifting_source") != expected_lifting_source:
            return False
        if (
            expected_contributor_audit is True
            and metadata.get("contributor_audit_requested") is not True
        ):
            return False
        point_count = int(metadata["point_count"])
        fragment_count = int(metadata["fragment_count"])
        frame_count = int(metadata["frame_count"])
        with np.load(arrays_path, allow_pickle=False) as arrays:
            required = {
                "xyz_m",
                "opacity",
                "affinity",
                "semantic",
                "label_features",
                "fragment_full_indptr",
                "fragment_full_ids",
                "fragment_full_mass",
                "fragment_core_indptr",
                "fragment_core_ids",
                "fragment_core_mass",
                "fragment_core_ratio",
                "fragment_id",
                "fragment_frame",
                "fragment_mask_index",
                "fragment_source_class",
                "frame_visible_indptr",
                "frame_visible_ids",
                "frame_visible_mass",
                "frame_geometry_abstained",
                "frame_grounded_missing",
                "semantic_fragment_full_indptr",
                "semantic_fragment_full_ids",
                "semantic_fragment_full_mass",
                "semantic_fragment_id",
                "semantic_fragment_frame",
                "semantic_fragment_mask_index",
                "semantic_fragment_class",
            }
            if not required.issubset(arrays.files):
                return False
            shape_ok = (
                arrays["xyz_m"].shape == (point_count, 3)
                and arrays["opacity"].shape == (point_count,)
                and arrays["affinity"].shape[0] == point_count
                and arrays["semantic"].shape[0] == point_count
                and arrays["label_features"].ndim == 2
                and arrays["fragment_full_indptr"].shape == (fragment_count + 1,)
                and arrays["fragment_core_indptr"].shape == (fragment_count + 1,)
                and arrays["fragment_full_mass"].shape
                == arrays["fragment_full_ids"].shape
                and arrays["fragment_core_mass"].shape
                == arrays["fragment_core_ids"].shape
                and arrays["fragment_core_ratio"].shape
                == arrays["fragment_core_ids"].shape
                and arrays["fragment_id"].shape == (fragment_count,)
                and arrays["fragment_frame"].shape == (fragment_count,)
                and arrays["fragment_mask_index"].shape == (fragment_count,)
                and arrays["fragment_source_class"].shape == (fragment_count,)
                and arrays["frame_visible_indptr"].shape == (frame_count + 1,)
                and arrays["frame_visible_mass"].shape
                == arrays["frame_visible_ids"].shape
                and arrays["frame_geometry_abstained"].shape == (frame_count,)
                and arrays["frame_grounded_missing"].shape == (frame_count,)
            )
            if not shape_ok:
                return False
            semantic_count = len(arrays["semantic_fragment_id"])
            return (
                _valid_ragged(
                    arrays["fragment_full_indptr"], arrays["fragment_full_ids"],
                    fragment_count, point_count,
                )
                and _valid_ragged(
                    arrays["fragment_core_indptr"], arrays["fragment_core_ids"],
                    fragment_count, point_count,
                )
                and arrays["semantic_fragment_full_mass"].shape
                == arrays["semantic_fragment_full_ids"].shape
                and _valid_ragged(
                    arrays["frame_visible_indptr"], arrays["frame_visible_ids"],
                    frame_count, point_count,
                )
                and _valid_ragged(
                    arrays["semantic_fragment_full_indptr"],
                    arrays["semantic_fragment_full_ids"],
                    semantic_count, point_count,
                )
                and arrays["semantic_fragment_frame"].shape == (semantic_count,)
                and arrays["semantic_fragment_mask_index"].shape == (semantic_count,)
                and arrays["semantic_fragment_class"].shape == (semantic_count,)
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _scene_image_root(scene: RuntimeScene) -> Path:
    return scene.base_path / "fastRecon/dense/sparse/0/images"


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log_path}"
        )


def ensure_segment_everything(
    scene: RuntimeScene,
    *,
    repo_root: Path,
    output_root: Path,
    sam_checkpoint: Path,
) -> Path:
    target = output_root / scene.scene_id
    summary_path = target / "summary.json"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = payload.get("images", ())
            if (
                payload.get("schema") == "saga-v8-segment-everything-v1"
                and payload.get("config") == SAM_EVERYTHING_CONFIG
                and int(payload.get("image_count", -1)) > 0
                and len(rows) == int(payload["image_count"])
                and {
                    str(row["image"]) for row in rows
                } == {
                    path.name
                    for path in _scene_image_root(scene).iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                }
                and all(
                    _mask_file_is_complete(
                        target / f"{Path(str(row['image'])).stem}.npz",
                        int(row["height"]),
                        int(row["width"]),
                    )
                    for row in rows
                )
            ):
                return target
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    target.mkdir(parents=True, exist_ok=True)
    _run_logged(
        (
            str(scene.python_bin),
            "-m",
            "category_priors.v8_masks",
            "--image-root",
            str(_scene_image_root(scene)),
            "--output-root",
            str(target),
            "--sam-checkpoint",
            str(sam_checkpoint),
        ),
        cwd=repo_root,
        log_path=target / "sam_everything.log",
    )
    if not summary_path.is_file():
        raise RuntimeError(f"SAM-everything summary missing: {summary_path}")
    return target


def run_v8_lifting_banks(
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    output_root: Path,
    repo_root: Path,
    *,
    mask_source: str,
    lifting_source: str,
    sam_masks_root: Path | None = None,
    sam_checkpoint: Path | None = None,
    label_features: Path | None = None,
    feature_ply_by_scene: Mapping[str, Path] | None = None,
    contributor_audit: bool = False,
) -> dict[str, Any]:
    mask_source = str(mask_source).upper()
    lifting_source = str(lifting_source).upper()
    if mask_source not in MASK_SOURCES:
        raise ValueError(f"unknown mask source: {mask_source}")
    if lifting_source not in LIFTING_SOURCES:
        raise ValueError(f"unknown lifting source: {lifting_source}")
    scenes = load_runtime_scenes(runtime_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for scene_id in map(str, scene_ids):
        free_gib = shutil.disk_usage(output_root).free / 1024**3
        if free_gib < 82.0:
            raise RuntimeError(
                "V8 requires an 80 GiB floor plus a 2 GiB per-scene write "
                f"reserve; found {free_gib:.1f} GiB"
            )
        scene = scenes.get(scene_id)
        if scene is None:
            raise KeyError(f"scene missing from runtime manifest: {scene_id}")
        sam_scene_root: Path | None = None
        if mask_source == "S":
            if sam_masks_root is None or sam_checkpoint is None:
                raise ValueError("S mask source requires sam_masks_root and sam_checkpoint")
            sam_scene_root = ensure_segment_everything(
                scene,
                repo_root=repo_root,
                output_root=sam_masks_root,
                sam_checkpoint=sam_checkpoint,
            )
        target = output_root / scene_id
        if lifting_bank_is_complete(
            target,
            expected_scene_id=scene_id,
            expected_mask_source=mask_source,
            expected_lifting_source=lifting_source,
            expected_contributor_audit=True if contributor_audit else None,
        ):
            records.append({"scene_id": scene_id, "status": "reused"})
            print(f"[{mask_source}-{lifting_source}/{scene_id}] reused", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [
            str(scene.python_bin),
            "-m",
            "category_priors.v8_worker",
            "--scene-id",
            scene_id,
            "--base-path",
            str(scene.base_path),
            "--output-dir",
            str(target),
            "--scene-scale-m-per-unit",
            str(scene.scene_scale_m_per_unit),
            "--mask-source",
            mask_source,
            "--lifting-source",
            lifting_source,
        ]
        if sam_scene_root is not None:
            command.extend(("--sam-mask-root", str(sam_scene_root)))
        if label_features is not None:
            command.extend(("--label-features", str(label_features)))
        if feature_ply_by_scene is not None and scene_id in feature_ply_by_scene:
            command.extend(("--feature-ply", str(feature_ply_by_scene[scene_id])))
        if contributor_audit:
            command.append("--contributor-audit")
        print(f"[{mask_source}-{lifting_source}/{scene_id}] starting", flush=True)
        scene_started = time.monotonic()
        _run_logged(command, cwd=repo_root, log_path=target / "worker.log")
        if not lifting_bank_is_complete(
            target,
            expected_scene_id=scene_id,
            expected_mask_source=mask_source,
            expected_lifting_source=lifting_source,
            expected_contributor_audit=True if contributor_audit else None,
        ):
            raise RuntimeError(f"incomplete V8 lifting bank: {target}")
        free_after_gib = shutil.disk_usage(output_root).free / 1024**3
        if free_after_gib < 80.0:
            raise RuntimeError(
                f"V8 disk floor was crossed after {scene_id}: {free_after_gib:.1f} GiB"
            )
        seconds = float(time.monotonic() - scene_started)
        records.append(
            {"scene_id": scene_id, "status": "completed", "seconds": seconds}
        )
        print(
            f"[{mask_source}-{lifting_source}/{scene_id}] completed in {seconds:.1f}s",
            flush=True,
        )
    result = {
        "schema": "saga-v8-lifting-run-summary-v1",
        "mask_source": mask_source,
        "lifting_source": lifting_source,
        "runtime_seconds": float(time.monotonic() - started),
        "scenes": records,
    }
    (output_root / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def run_v8_lifting_factorial(
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    output_root: Path,
    repo_root: Path,
    *,
    sam_masks_root: Path,
    sam_checkpoint: Path,
    label_features: Path | None = None,
    contributor_audit: bool = False,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mask_source in MASK_SOURCES:
        for lifting_source in LIFTING_SOURCES:
            arm = f"{mask_source}-{lifting_source}"
            results[arm] = run_v8_lifting_banks(
                runtime_manifest,
                scene_ids,
                output_root / arm,
                repo_root,
                mask_source=mask_source,
                lifting_source=lifting_source,
                sam_masks_root=sam_masks_root,
                sam_checkpoint=sam_checkpoint,
                label_features=label_features,
                contributor_audit=(
                    contributor_audit
                    and mask_source == "G"
                    and lifting_source == "M1"
                ),
            )
    return {"schema": "saga-v8-lifting-factorial-run-v1", "arms": results}
