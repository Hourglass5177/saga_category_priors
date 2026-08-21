from __future__ import annotations

"""GPU worker for deterministic V7 cross-view object banks.

Ground truth is intentionally absent from this module.  It renders corrected
maximum-contributor IDs and weights, lifts existing 2D masks, and serializes a
compact object bank for offline evaluation and prior replay.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch

from gaussian_renderer import render_with_max_contributor
from scene import FeatureGaussianModel, GaussianModel
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import cameraList_from_camInfos

from .v7_objects import (
    V7Config,
    associate_fragments,
    attach_unique_halo,
    build_consensus_core,
    lift_frame,
    materialize_instances,
)


DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


def _git_commit(repo: Path) -> str:
    marker = repo / "GIT_COMMIT"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted-preflight"


def _scene_paths(base_path: Path) -> dict[str, Path]:
    rgb_candidates = (
        base_path / "output_models/point_cloud/iteration_30000/scene_point_cloud.ply",
        base_path / "output_models/point_cloud/iteration_30000/point_cloud.ply",
    )
    rgb = next((path for path in rgb_candidates if path.is_file()), rgb_candidates[0])
    return {
        "rgb_ply": rgb,
        "feature_ply": base_path / "saga/contrastive_feature_point_cloud.ply",
        "sparse": base_path / "fastRecon/dense/sparse/0",
        "images": base_path / "fastRecon/dense/sparse/0/images",
        "masks": base_path / "saga/masks",
        "labels": base_path / "saga/labels",
    }


def _load_cameras(paths: dict[str, Path]) -> list[Any]:
    sparse = paths["sparse"]
    try:
        extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
        intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
    except (FileNotFoundError, OSError):
        extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
        intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
    infos = readColmapCameras(
        extrinsics,
        intrinsics,
        str(paths["images"]),
        masks_folder=str(paths["masks"]),
        labels_folder=str(paths["labels"]),
    )
    camera_args = SimpleNamespace(resolution=1, data_device="cuda")
    return cameraList_from_camInfos(infos, 1, camera_args)


def _resize_masks(masks: torch.Tensor, height: int, width: int) -> torch.Tensor:
    masks = masks.detach().cpu()
    if masks.ndim != 3:
        raise ValueError(f"mask tensor must be MxHxW, got {tuple(masks.shape)}")
    if masks.shape[-2:] != (height, width):
        masks = torch.nn.functional.interpolate(
            masks.float().unsqueeze(1), size=(height, width), mode="nearest"
        ).squeeze(1)
    return masks.bool()


def bank_is_complete(output_dir: Path) -> bool:
    json_path = output_dir / "object_bank.json"
    npz_path = output_dir / "object_bank.npz"
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        bank = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as arrays:
            count = int(bank["point_count"])
            required = ("core_track_id", "final_track_id", "candidate_labels")
            return all(arrays[name].shape == (count,) for name in required)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _ragged(rows: Sequence[np.ndarray], dtype: np.dtype[Any]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=dtype) for row in rows])
        if int(indptr[-1]) else np.empty(0, dtype=dtype)
    )
    return indptr, values


def run_v7_bank(
    scene_id: str,
    base_path: Path,
    output_dir: Path,
    scene_scale_m_per_unit: float,
    *,
    halo: bool,
    classes: Sequence[str] = DEFAULT_CLASSES,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    paths = _scene_paths(base_path)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing scene assets: {missing}")

    rgb = GaussianModel(0)
    rgb.load_ply(str(paths["rgb_ply"]))
    feature = FeatureGaussianModel(32, 32)
    feature.load_ply(str(paths["feature_ply"]))
    rgb_xyz = rgb.get_xyz.detach().cpu().numpy()
    feature_xyz = feature.get_xyz.detach().cpu().numpy()
    if rgb_xyz.shape != feature_xyz.shape or not np.allclose(
        rgb_xyz, feature_xyz, rtol=0.0, atol=1e-6
    ):
        raise ValueError("RGB and feature Gaussian XYZ/order do not match")
    point_count = len(rgb_xyz)
    xyz_m = feature_xyz.astype(np.float64) * float(scene_scale_m_per_unit)
    affinity = feature.get_point_features.detach().cpu().numpy()
    affinity /= np.maximum(np.linalg.norm(affinity, axis=1, keepdims=True), 1e-12)

    cameras = _load_cameras(paths)
    pipeline = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    frames = []
    next_fragment_id = 0
    empty_frames = 0
    invalid_pixels = 0
    for frame_id, camera in enumerate(sorted(cameras, key=lambda item: item.image_name)):
        masks = camera.original_masks
        labels = camera.labels
        if masks is None and labels is None:
            masks = torch.zeros(
                (0, camera.image_height, camera.image_width), dtype=torch.bool
            )
            labels = torch.zeros(0, dtype=torch.long)
            empty_frames += 1
        elif masks is None or labels is None:
            raise ValueError(
                f"{scene_id}/{camera.image_name}: mask and label must both exist or both be absent"
            )
        masks = _resize_masks(masks, camera.image_height, camera.image_width)
        labels = torch.as_tensor(labels).detach().cpu().reshape(-1).long()
        if len(masks) != len(labels):
            raise ValueError(
                f"{scene_id}/{camera.image_name}: {len(masks)} masks != {len(labels)} labels"
            )
        rendered = render_with_max_contributor(camera, rgb, pipeline, background)
        ids = rendered["max_contributor"].detach().cpu().numpy().astype(np.int64)
        weights = rendered["max_contribute"].detach().cpu().numpy().astype(np.float64)
        if ids.shape != (camera.image_height, camera.image_width):
            raise ValueError(f"unexpected contributor shape {ids.shape}")
        invalid_pixels += int(np.count_nonzero((ids < 0) | (weights <= 0)))
        evidence = lift_frame(
            ids,
            weights,
            masks.numpy(),
            labels.numpy(),
            frame_id,
            point_count,
            fragment_id_start=next_fragment_id,
        )
        frames.append(evidence)
        next_fragment_id += len(evidence.fragments)

    fragments = [fragment for frame in frames for fragment in frame.fragments]
    tracks = associate_fragments(fragments)
    core = build_consensus_core(tracks, fragments, frames, point_count)
    final_track_id = (
        attach_unique_halo(xyz_m, affinity, tracks, core)
        if halo else core.core_track_id.copy()
    )
    candidate_labels, candidates = materialize_instances(
        xyz_m, tracks, fragments, core, final_track_id, classes
    )

    fragment_full_indptr, fragment_full_ids = _ragged(
        [fragment.full_ids for fragment in fragments], np.int32
    )
    fragment_core_indptr, fragment_core_ids = _ragged(
        [fragment.core_ids for fragment in fragments], np.int32
    )
    track_indptr, track_fragment_ids = _ragged(
        [np.asarray(track.fragment_ids) for track in tracks], np.int32
    )
    np.savez_compressed(
        output_dir / "object_bank.npz",
        xyz_m=xyz_m.astype(np.float32),
        fragment_full_indptr=fragment_full_indptr,
        fragment_full_ids=fragment_full_ids,
        fragment_core_indptr=fragment_core_indptr,
        fragment_core_ids=fragment_core_ids,
        fragment_frame=np.asarray([item.frame_id for item in fragments], dtype=np.int32),
        fragment_class=np.asarray([item.class_id for item in fragments], dtype=np.int16),
        track_fragment_indptr=track_indptr,
        track_fragment_ids=track_fragment_ids,
        core_track_id=core.core_track_id,
        final_track_id=final_track_id,
        candidate_labels=candidate_labels,
        positive_views=core.positive_views,
        visible_views=core.visible_views,
        background_views=core.background_views,
        conflict_views=core.conflict_views,
        assignment_margin=core.assignment_margin,
    )
    repo = repo_root or Path(__file__).resolve().parents[1]
    bank = {
        "schema": "saga-v7-object-bank-v1",
        "scene_id": scene_id,
        "git_commit": _git_commit(repo),
        "point_count": point_count,
        "frame_count": len(frames),
        "empty_background_frame_count": empty_frames,
        "invalid_or_zero_contribution_pixel_count": invalid_pixels,
        "fragment_count": len(fragments),
        "track_count": len(tracks),
        "valid_track_count": len(core.valid_track_ids),
        "candidate_count": len(candidates),
        "halo_enabled": bool(halo),
        "classes": list(classes),
        "config": V7Config().as_json(),
        "candidates": candidates,
        "arrays_npz": "object_bank.npz",
        "runtime_seconds": float(time.monotonic() - started),
    }
    (output_dir / "object_bank.json").write_text(
        json.dumps(bank, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not bank_is_complete(output_dir):
        raise RuntimeError("serialized V7 object bank failed validation")
    return bank


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-scale-m-per-unit", type=float, default=1.0)
    parser.add_argument("--halo", choices=("off", "on"), default="off")
    args = parser.parse_args(argv)
    if bank_is_complete(args.output_dir):
        print(f"complete bank exists, skipping: {args.output_dir}")
        return 0
    bank = run_v7_bank(
        args.scene_id,
        args.base_path,
        args.output_dir,
        args.scene_scale_m_per_unit,
        halo=args.halo == "on",
    )
    print(json.dumps({key: bank[key] for key in (
        "scene_id", "fragment_count", "track_count", "candidate_count", "runtime_seconds"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
