from __future__ import annotations

"""Offline geometric lifting oracle for native V9 artifacts (GT-only side)."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .v9_lifting import load_lifting_bank


def runtime_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text("utf-8"))
    rows = payload.get("scenes", payload)
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    return {str(row["scene_id"]): dict(row) for row in rows}


def gaussian_ply(scene: Mapping[str, Any]) -> Path:
    if scene.get("gaussian_ply"):
        path = Path(str(scene["gaussian_ply"]))
        return path if path.is_absolute() else Path(str(scene["base_path"])) / path
    root = Path(str(scene["base_path"])) / "output_models/point_cloud/iteration_30000"
    primary = root / "scene_point_cloud.ply"
    return primary if primary.is_file() else root / "point_cloud.ply"


def transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )


def unpack_ragged(indptr: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(indptr[index]):int(indptr[index + 1])], dtype=np.int64)
        for index in range(len(indptr) - 1)
    ]


def map_gt_to_gaussian(
    gt_xyz: np.ndarray, gaussian_xyz: np.ndarray, radius_m: float
) -> tuple[np.ndarray, dict[str, float]]:
    distances, indices = cKDTree(gaussian_xyz).query(
        gt_xyz, k=1, distance_upper_bound=float(radius_m), workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(gaussian_xyz))
    mapped = np.full(len(gt_xyz), -1, dtype=np.int64)
    mapped[valid] = indices[valid]
    return mapped, {
        "mapped_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "median_nn_distance_m": float(np.median(distances[valid])) if np.any(valid) else float("inf"),
    }


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = len(np.intersect1d(left, right, assume_unique=True))
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _greedy_iou(fragments: Sequence[np.ndarray], gt: np.ndarray) -> float:
    if not fragments:
        return 0.0
    best = max(fragments, key=lambda fragment: _iou(fragment, gt))
    support = np.asarray(best, dtype=np.int64)
    score = _iou(support, gt)
    remaining = [fragment for fragment in fragments if fragment is not best]
    while remaining:
        choices = [(_iou(np.union1d(support, fragment), gt), index) for index, fragment in enumerate(remaining)]
        next_score, index = max(choices, key=lambda row: (row[0], -row[1]))
        if next_score <= score + 1e-12:
            break
        support = np.union1d(support, remaining.pop(index))
        score = next_score
    return float(score)


def evaluate_fragment_oracle(
    *,
    scene_id: str,
    fragment_gaussian_ids: Sequence[np.ndarray],
    gt_nearest_gaussian: np.ndarray,
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    size_spec: Mapping[str, Any],
    min_region_size: int,
) -> dict[str, Any]:
    valid_map = gt_nearest_gaussian >= 0
    fragments = [
        np.flatnonzero(valid_map & np.isin(gt_nearest_gaussian, ids)).astype(np.int64)
        for ids in fragment_gaussian_ids
    ]
    valid_gt = (gt_semantic >= 0) & (gt_instance >= 0)
    official = 0
    matches_050 = 0
    tiny_count = 0
    tiny_matches_025 = 0
    per_gt: list[dict[str, Any]] = []
    small_max = float(size_spec.get("boundaries_m", size_spec)["small_max_m"])
    for class_id, instance_id in sorted(set(zip(gt_semantic[valid_gt], gt_instance[valid_gt]))):
        ids = np.flatnonzero(valid_gt & (gt_semantic == class_id) & (gt_instance == instance_id))
        if len(ids) < int(min_region_size):
            continue
        official += 1
        score = _greedy_iou(fragments, ids)
        matches_050 += int(score >= 0.50)
        tiny = float(np.linalg.norm(np.ptp(gt_xyz[ids], axis=0))) <= small_max
        tiny_count += int(tiny)
        tiny_matches_025 += int(tiny and score >= 0.25)
        per_gt.append({
            "class_id": int(class_id), "instance_id": int(instance_id),
            "point_count": len(ids), "tiny_small": tiny, "greedy_iou": score,
        })
    matched_fragments = sum(
        max((_iou(fragment, np.flatnonzero(valid_gt & (gt_instance == instance_id)))
             for instance_id in np.unique(gt_instance[valid_gt])), default=0.0) >= 0.25
        for fragment in fragments
    )
    return {
        "scene_id": scene_id,
        "fragment_count": len(fragments),
        "fragment_precision_025": matched_fragments / len(fragments) if fragments else 0.0,
        "official_gt_count": official,
        "geometric_greedy_match_050_count": matches_050,
        "tiny_small_gt_count": tiny_count,
        "tiny_small_geometric_match_025_count": tiny_matches_025,
        "per_gt": per_gt,
    }


__all__ = [
    "evaluate_fragment_oracle", "gaussian_ply", "load_lifting_bank",
    "map_gt_to_gaussian", "runtime_rows", "transform", "unpack_ragged",
]
