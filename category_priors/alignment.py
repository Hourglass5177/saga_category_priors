from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import hash_json, load_json, sha256_file, write_json


def _validate_manifest_hash(payload: dict[str, Any]) -> None:
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not expected or hash_json(unsigned) != expected:
        raise ValueError("Scene preparation manifest content hash mismatch")


def _qvec_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid COLMAP quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1 - 2 * y * y - 2 * z * z,
                2 * x * y - 2 * w * z,
                2 * x * z + 2 * w * y,
            ],
            [
                2 * x * y + 2 * w * z,
                1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * w * x,
            ],
            [
                2 * x * z - 2 * w * y,
                2 * y * z + 2 * w * x,
                1 - 2 * x * x - 2 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def _read_camera_centers(images_txt: Path) -> np.ndarray:
    centers: list[np.ndarray] = []
    with images_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 10:
                continue
            try:
                int(fields[0])
                int(fields[8])
                qvec = [float(value) for value in fields[1:5]]
                tvec = np.asarray([float(value) for value in fields[5:8]])
            except ValueError:
                continue
            rotation = _qvec_to_rotation(qvec)
            centers.append(-(rotation.T @ tvec))
    if len(centers) < 2:
        raise ValueError(f"Expected at least two registered cameras: {images_txt}")
    result = np.asarray(centers, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("COLMAP camera centers contain non-finite values")
    return result


def _nearest_diagnostics(
    reference: np.ndarray,
    query: np.ndarray,
    radius_m: float,
) -> dict[str, float | None]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Alignment audit requires scipy") from exc
    tree = cKDTree(reference)
    distances, indices = tree.query(
        query, k=1, distance_upper_bound=radius_m, workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(reference))
    return {
        "mapped_fraction": float(valid.mean()) if len(valid) else 0.0,
        "median_nn_distance_m": (
            float(np.median(distances[valid])) if np.any(valid) else None
        ),
        "p95_nn_distance_m": (
            float(np.quantile(distances[valid], 0.95)) if np.any(valid) else None
        ),
    }


def audit_saga_alignment(
    preparation_manifest_path: str | Path,
    gt_npz_path: str | Path,
    output_path: str | Path,
    gaussian_ply_path: str | Path | None = None,
    radius_m: float = 0.05,
    minimum_mapped_fraction: float = 0.90,
    camera_padding_m: float = 2.0,
) -> dict[str, Any]:
    if radius_m <= 0 or camera_padding_m < 0:
        raise ValueError("Alignment distances must be non-negative and radius positive")
    if not 0 < minimum_mapped_fraction <= 1:
        raise ValueError("minimum_mapped_fraction must be in (0, 1]")

    manifest_path = Path(preparation_manifest_path).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("kind") != "scannet_saga_scene":
        raise ValueError("Expected a scannet_saga_scene preparation manifest")
    _validate_manifest_hash(manifest)
    base_path = Path(manifest["base_path"])
    if not base_path.is_absolute():
        base_path = (manifest_path.parent / base_path).resolve()
    initial_path = base_path / manifest["initial_point_cloud"]["path"]
    cloud_path = (
        Path(gaussian_ply_path).resolve()
        if gaussian_ply_path is not None
        else initial_path.resolve()
    )
    gt_path = Path(gt_npz_path).resolve()
    gt_coords, _ = load_ground_truth_npz(gt_path, str(manifest["scene_id"]))
    transform = manifest["gaussian_to_gt_transform"]
    cloud_coords = apply_transform(load_ply_xyz(cloud_path), transform)
    if len(gt_coords) == 0 or len(cloud_coords) == 0:
        raise ValueError("Alignment audit requires nonempty GT and point clouds")
    if not np.all(np.isfinite(gt_coords)) or not np.all(np.isfinite(cloud_coords)):
        raise ValueError("Alignment point clouds contain non-finite coordinates")

    gt_to_cloud = _nearest_diagnostics(cloud_coords, gt_coords, radius_m)
    cloud_sample = cloud_coords
    if len(cloud_sample) > 200_000:
        sample_indices = np.linspace(
            0, len(cloud_sample) - 1, 200_000, dtype=np.int64
        )
        cloud_sample = cloud_sample[sample_indices]
    cloud_to_gt = _nearest_diagnostics(gt_coords, cloud_sample, radius_m)

    gt_min, gt_max = gt_coords.min(axis=0), gt_coords.max(axis=0)
    cloud_min, cloud_max = cloud_coords.min(axis=0), cloud_coords.max(axis=0)
    gt_diagonal = float(np.linalg.norm(gt_max - gt_min))
    cloud_diagonal = float(np.linalg.norm(cloud_max - cloud_min))
    extent_ratio = cloud_diagonal / gt_diagonal if gt_diagonal > 0 else None

    images_txt = (
        base_path / "fastRecon" / "dense" / "sparse" / "0" / "images.txt"
    )
    camera_centers = _read_camera_centers(images_txt)
    lower = gt_min - camera_padding_m
    upper = gt_max + camera_padding_m
    inside = np.all((camera_centers >= lower) & (camera_centers <= upper), axis=1)
    camera_inside_fraction = float(inside.mean())
    outside_components = np.maximum(
        np.maximum(lower - camera_centers, camera_centers - upper), 0.0
    )
    maximum_camera_outside_distance = float(
        np.linalg.norm(outside_components, axis=1).max()
    )

    failures: list[str] = []
    if float(manifest["scene_scale_m_per_unit"]) != 1.0:
        failures.append("scene_scale_m_per_unit_is_not_one")
    if not np.allclose(transform, np.eye(4), atol=1e-8):
        failures.append("gaussian_to_gt_transform_is_not_identity")
    if gt_to_cloud["mapped_fraction"] < minimum_mapped_fraction:
        failures.append("gt_to_cloud_coverage_below_threshold")
    if camera_inside_fraction < minimum_mapped_fraction:
        failures.append("camera_trajectory_outside_padded_gt_bounds")

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "saga_alignment_audit",
        "scene_id": str(manifest["scene_id"]),
        "passed": not failures,
        "failures": failures,
        "thresholds": {
            "radius_m": radius_m,
            "minimum_mapped_fraction": minimum_mapped_fraction,
            "camera_padding_m": camera_padding_m,
        },
        "point_cloud_role": "trained_gaussians"
        if gaussian_ply_path is not None
        else "prepared_initial_points",
        "gt_to_cloud": gt_to_cloud,
        "cloud_to_gt_sample": cloud_to_gt,
        "geometry": {
            "gt_vertices": len(gt_coords),
            "cloud_vertices": len(cloud_coords),
            "gt_bounds_m": [gt_min.tolist(), gt_max.tolist()],
            "cloud_bounds_m": [cloud_min.tolist(), cloud_max.tolist()],
            "gt_diagonal_m": gt_diagonal,
            "cloud_diagonal_m": cloud_diagonal,
            "cloud_to_gt_extent_ratio": extent_ratio,
        },
        "cameras": {
            "count": len(camera_centers),
            "inside_padded_gt_fraction": camera_inside_fraction,
            "maximum_outside_distance_m": maximum_camera_outside_distance,
            "bounds_m": [
                camera_centers.min(axis=0).tolist(),
                camera_centers.max(axis=0).tolist(),
            ],
        },
        "provenance": {
            "preparation_manifest": str(manifest_path),
            "preparation_manifest_sha256": sha256_file(manifest_path),
            "gt_npz": str(gt_path),
            "gt_npz_sha256": sha256_file(gt_path),
            "point_cloud": str(cloud_path),
            "point_cloud_sha256": sha256_file(cloud_path),
        },
    }
    payload["content_sha256"] = hash_json(payload)
    write_json(output_path, payload)
    if failures:
        raise RuntimeError("SAGA alignment audit failed: " + ", ".join(failures))
    return payload
