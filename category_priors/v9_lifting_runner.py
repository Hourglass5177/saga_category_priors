from __future__ import annotations

"""Sequential native V9 lifting runner; no V3--V8 runtime dependency."""

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .v9_lifting import (
    DEFAULT_CLASSES,
    FragmentConfig,
    build_lifting_identity,
    lifting_bank_is_complete,
)
from .v9_masks import sam_directory_is_complete
from .v9_feature_training import validate_v8_sam_everything_source


def _registered_sam_directory_is_complete(
    directory: Path, image_root: Path
) -> bool:
    """Require both the registered summary and every packed frame payload."""

    try:
        validate_v8_sam_everything_source(directory)
        summary = load_json(Path(directory) / "summary.json")
        if Path(str(summary.get("image_root", ""))).resolve() != Path(
            image_root
        ).resolve():
            return False
        return sam_directory_is_complete(directory, image_root)
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
        return False


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command), cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        )
    if completed.returncode:
        raise RuntimeError(
            f"native V9 subprocess exited {completed.returncode}; see {log_path}"
        )


def ensure_v9_segment_everything(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    repo_root: Path,
    output_root: Path,
    sam_checkpoint: Path,
    reusable_root: Path | None = None,
) -> Path:
    """Return a complete packed-mask directory without mutating old assets."""

    image_root = Path(str(scene["base_path"])) / "fastRecon/dense/sparse/0/images"
    if reusable_root is not None:
        reusable = Path(reusable_root) / str(scene_id)
        if _registered_sam_directory_is_complete(reusable, image_root):
            return reusable
    target = Path(output_root) / str(scene_id)
    if _registered_sam_directory_is_complete(target, image_root):
        return target
    python_bin = Path(str(scene.get("python_bin", "")))
    if not python_bin.is_file():
        raise FileNotFoundError(f"scene Python does not exist: {python_bin}")
    _run_logged(
        (
            str(python_bin),
            "-m",
            "category_priors.v9_masks",
            "--image-root",
            str(image_root),
            "--output-root",
            str(target),
            "--sam-checkpoint",
            str(sam_checkpoint),
        ),
        cwd=repo_root,
        log_path=target / "sam_everything.log",
    )
    if not _registered_sam_directory_is_complete(target, image_root):
        raise RuntimeError(f"incomplete native V9 SAM masks for {scene_id}")
    return target


def run_v9_lifting_banks(
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    output_root: Path,
    repo_root: Path,
    *,
    sam_masks_root: Path,
    sam_checkpoint: Path | None = None,
    label_features: Path,
    feature_ply_by_scene: Mapping[str, Path],
    git_commit: str,
    sam_scene_roots: Mapping[str, Path] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Freeze deterministic hybrid lifting banks from isolated 10k features."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for scene_id in map(str, scene_ids):
        if shutil.disk_usage(output_root).free / 1024**3 < 82.0:
            raise RuntimeError("V9 needs the 80 GiB floor plus 2 GiB write reserve")
        if scene_id not in scenes:
            raise KeyError(f"runtime manifest lacks {scene_id}")
        if scene_id not in feature_ply_by_scene:
            raise ValueError(f"10k feature PLY missing for {scene_id}")
        scene = scenes[scene_id]
        target = output_root / scene_id
        python_bin = Path(str(scene.get("python_bin", "")))
        if not python_bin.is_file():
            raise FileNotFoundError(f"scene Python does not exist: {python_bin}")
        if sam_scene_roots is not None and scene_id in sam_scene_roots:
            sam_scene = Path(sam_scene_roots[scene_id])
            image_root = Path(str(scene["base_path"])) / "fastRecon/dense/sparse/0/images"
            if not _registered_sam_directory_is_complete(sam_scene, image_root):
                raise ValueError(f"declared SAM root is incomplete for {scene_id}")
        else:
            if sam_checkpoint is None:
                raise ValueError("sam_checkpoint is required when no scene root is supplied")
            sam_scene = ensure_v9_segment_everything(
                scene_id=scene_id,
                scene=scene,
                repo_root=repo_root,
                output_root=output_root.parent / "sam-everything",
                sam_checkpoint=sam_checkpoint,
                reusable_root=sam_masks_root,
            )
        feature_ply = Path(feature_ply_by_scene[scene_id]).resolve()
        feature_record = feature_ply.parent / "train_10k.json"
        expected_identity = build_lifting_identity(
            scene_id=scene_id,
            git_commit=git_commit,
            feature_ply=feature_ply,
            feature_record=feature_record,
            label_features=Path(label_features),
            segment_everything_root=sam_scene,
            classes=DEFAULT_CLASSES,
            config=FragmentConfig(),
        )
        if lifting_bank_is_complete(
            target,
            expected_scene_id=scene_id,
            expected_git_commit=git_commit,
            expected_identity=expected_identity,
        ):
            records.append({"scene_id": scene_id, "status": "reused"})
            continue
        command = (
            str(python_bin),
            "-m",
            "category_priors.v9_lifting_worker",
            "--scene-id",
            scene_id,
            "--base-path",
            str(scene["base_path"]),
            "--output-dir",
            str(target),
            "--scene-scale-m-per-unit",
            str(float(scene["scene_scale_m_per_unit"])),
            "--segment-everything-root",
            str(sam_scene),
            "--feature-ply",
            str(feature_ply),
            "--feature-record",
            str(feature_record),
            "--label-features",
            str(label_features),
            "--git-commit",
            str(git_commit),
        )
        target.mkdir(parents=True, exist_ok=True)
        scene_started = time.monotonic()
        _run_logged(command, cwd=repo_root, log_path=target / "worker.log")
        if not lifting_bank_is_complete(
            target,
            expected_scene_id=scene_id,
            expected_git_commit=git_commit,
            expected_identity=expected_identity,
        ):
            raise RuntimeError(f"native V9 worker left an incomplete bank: {target}")
        records.append(
            {
                "scene_id": scene_id,
                "status": "completed",
                "seconds": float(time.monotonic() - scene_started),
            }
        )
    summary = {
        "schema": "saga-v9-native-lifting-run-summary-v1",
        "scene_count": len(records),
        "runs": records,
        "runtime_seconds": float(time.monotonic() - started),
    }
    write_json(output_root / "run_summary.json", summary)
    return summary
