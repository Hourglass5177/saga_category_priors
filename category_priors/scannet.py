from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file
from .taxonomy import Taxonomy

FIXED_RADII_M = (0.02, 0.05, 0.10, 0.20)
RELATIVE_RADII = (0.02, 0.05, 0.10)


@dataclass(frozen=True)
class ScanNetSceneFiles:
    scene_id: str
    scene_dir: Path
    mesh: Path
    aggregation: Path
    segments: Path
    metadata: Path


def physical_scene_id(scene_id: str) -> str:
    return scene_id.rsplit("_", 1)[0] if "_" in scene_id else scene_id


def discover_scene_files(dataset_root: str | Path, scene_id: str) -> ScanNetSceneFiles:
    scene_dir = Path(dataset_root) / scene_id
    result = ScanNetSceneFiles(
        scene_id=scene_id,
        scene_dir=scene_dir,
        mesh=scene_dir / f"{scene_id}_vh_clean_2.ply",
        aggregation=scene_dir / f"{scene_id}.aggregation.json",
        segments=scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        metadata=scene_dir / f"{scene_id}.txt",
    )
    missing = [
        str(path)
        for path in (result.mesh, result.aggregation, result.segments, result.metadata)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{scene_id}: missing required files: {missing}")
    return result


def read_axis_alignment(metadata_path: str | Path) -> np.ndarray:
    for line in (
        Path(metadata_path).read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        if line.strip().startswith("axisAlignment"):
            values = [float(item) for item in line.split("=", 1)[1].strip().split()]
            if len(values) != 16:
                raise ValueError(
                    f"{metadata_path}: axisAlignment must contain 16 values"
                )
            matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{metadata_path}: non-finite axisAlignment")
            return matrix
    raise ValueError(f"{metadata_path}: axisAlignment not found")


def load_mesh(mesh_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ScanNet extraction requires plyfile") from exc
    ply = PlyData.read(str(mesh_path))
    vertex = ply["vertex"]
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64
    )
    face = ply["face"]
    faces = np.asarray(
        [np.asarray(item, dtype=np.int64) for item in face["vertex_indices"]],
        dtype=object,
    )
    if any(len(item) != 3 for item in faces):
        raise ValueError(f"{mesh_path}: expected a triangular mesh")
    return vertices, np.stack(faces).astype(np.int64)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
    )
    transformed = homogeneous @ transform.T
    w = transformed[:, 3:4]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("Invalid homogeneous transform")
    return transformed[:, :3] / w


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def sample_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    areas: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    valid = areas > 0
    faces = faces[valid]
    areas = areas[valid]
    if len(faces) == 0 or count <= 0:
        return np.empty((0, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(
        len(faces), size=count, replace=True, p=areas / areas.sum()
    )
    triangles = vertices[faces[face_indices]]
    u = rng.random(count)
    v = rng.random(count)
    sqrt_u = np.sqrt(u)
    barycentric = np.column_stack((1.0 - sqrt_u, sqrt_u * (1.0 - v), sqrt_u * v))
    return np.einsum("ni,nij->nj", barycentric, triangles)


def voxelize(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    keys = np.floor(points / voxel_size_m + 0.5).astype(np.int64)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(unique_indices)]


def pca_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 3:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        extents = maximum - minimum
        return extents, (minimum + maximum) / 2.0, np.eye(3)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    local = (points - center) @ axes
    minimum = local.min(axis=0)
    maximum = local.max(axis=0)
    extents = maximum - minimum
    local_center = (minimum + maximum) / 2.0
    world_center = center + local_center @ axes.T
    return extents, world_center, axes


def voxel_connected_components(points: np.ndarray, voxel_size_m: float) -> int:
    if len(points) == 0:
        return 0
    occupied = {
        tuple(item) for item in np.floor(points / voxel_size_m + 0.5).astype(np.int64)
    }
    components = 0
    offsets = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while occupied:
        components += 1
        stack = [occupied.pop()]
        while stack:
            current = stack.pop()
            for offset in offsets:
                neighbor = (
                    current[0] + offset[0],
                    current[1] + offset[1],
                    current[2] + offset[2],
                )
                if neighbor in occupied:
                    occupied.remove(neighbor)
                    stack.append(neighbor)
    return components


def _rates_for_radius(
    tree: Any,
    points: np.ndarray,
    labels: np.ndarray,
    radius: float | np.ndarray,
    query_global_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if query_global_indices is None:
        if len(points) != len(labels):
            raise ValueError("Subset radius queries must provide query_global_indices")
        query_global_indices = np.arange(len(points), dtype=np.int64)
    else:
        query_global_indices = np.asarray(query_global_indices, dtype=np.int64)
        if len(query_global_indices) != len(points):
            raise ValueError("query_global_indices must match the query point count")
    neighborhoods = tree.query_ball_point(points, radius, workers=-1)
    same_rates = np.ones(len(points), dtype=np.float64)
    boundary = np.zeros(len(points), dtype=np.float64)
    for index, neighbors in enumerate(neighborhoods):
        neighbor_indices = np.asarray(neighbors, dtype=np.int64)
        neighbor_indices = neighbor_indices[
            neighbor_indices != query_global_indices[index]
        ]
        if len(neighbor_indices) == 0:
            continue
        same = labels[neighbor_indices] == labels[query_global_indices[index]]
        same_rates[index] = float(same.mean())
        boundary[index] = float(np.any(~same))
    return same_rates, boundary


def _instance_assignment(
    segment_indices: np.ndarray,
    groups: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, str]]:
    vertex_instance = np.full(len(segment_indices), -1, dtype=np.int64)
    raw_labels: dict[int, str] = {}
    segment_to_vertices: dict[int, np.ndarray] = {
        int(segment): np.flatnonzero(segment_indices == segment)
        for segment in np.unique(segment_indices)
    }
    for fallback_id, group in enumerate(groups):
        instance_id = int(group.get("id", fallback_id))
        raw_labels[instance_id] = str(group.get("label", "")).strip().lower()
        for segment in group.get("segments", []):
            indices = segment_to_vertices.get(int(segment))
            if indices is not None:
                vertex_instance[indices] = instance_id
    return vertex_instance, raw_labels


def extract_scene_stats(
    files: ScanNetSceneFiles,
    taxonomy: Taxonomy,
    split: str,
    dataset: str = "scannet200",
    voxel_size_m: float = 0.02,
    oversample_factor: float = 4.0,
    max_samples_per_instance: int = 500_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ScanNet extraction requires scipy") from exc

    vertices, faces = load_mesh(files.mesh)
    vertices = apply_transform(vertices, read_axis_alignment(files.metadata))
    segments_payload = json.loads(files.segments.read_text(encoding="utf-8"))
    aggregation_payload = json.loads(files.aggregation.read_text(encoding="utf-8"))
    segment_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    if len(segment_indices) != len(vertices):
        raise ValueError(f"{files.scene_id}: segment/vertex count mismatch")
    groups = aggregation_payload.get("segGroups", [])
    vertex_instance, raw_labels = _instance_assignment(segment_indices, groups)

    face_instances = vertex_instance[faces]
    valid_face_mask = np.all(face_instances == face_instances[:, :1], axis=1) & (
        face_instances[:, 0] >= 0
    )
    areas = triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    valid_area = float(areas[valid_face_mask].sum())

    instance_points: dict[int, np.ndarray] = {}
    instance_face_area: dict[int, float] = {}
    for instance_id in sorted(set(face_instances[valid_face_mask, 0].tolist())):
        instance_mask = valid_face_mask & (face_instances[:, 0] == instance_id)
        selected_faces = faces[instance_mask]
        selected_areas = areas[instance_mask]
        surface_area = float(selected_areas.sum())
        sample_count = max(
            30, math.ceil(oversample_factor * surface_area / (voxel_size_m**2))
        )
        sample_count = min(sample_count, max_samples_per_instance)
        sampled = sample_faces(
            vertices,
            selected_faces,
            selected_areas,
            sample_count,
            _stable_seed(files.scene_id, instance_id),
        )
        instance_points[int(instance_id)] = voxelize(sampled, voxel_size_m)
        instance_face_area[int(instance_id)] = surface_area

    nonempty = [
        (instance_id, points)
        for instance_id, points in instance_points.items()
        if len(points)
    ]
    if nonempty:
        all_points = np.concatenate([item[1] for item in nonempty], axis=0)
        all_labels = np.concatenate(
            [np.full(len(item[1]), item[0], dtype=np.int64) for item in nonempty],
            axis=0,
        )
        tree = cKDTree(all_points)
        fixed_rates: dict[float, np.ndarray] = {}
        fixed_boundary: dict[float, np.ndarray] = {}
        for radius in FIXED_RADII_M:
            fixed_rates[radius], fixed_boundary[radius] = _rates_for_radius(
                tree, all_points, all_labels, radius
            )
    else:
        all_points = np.empty((0, 3), dtype=np.float64)
        all_labels = np.empty((0,), dtype=np.int64)
        tree = None
        fixed_rates = {}
        fixed_boundary = {}

    source_hashes = {
        path.name: sha256_file(path)
        for path in (files.mesh, files.aggregation, files.segments, files.metadata)
    }
    rows: list[dict[str, Any]] = []
    for instance_id, points in nonempty:
        raw_label = raw_labels.get(instance_id, "")
        canonical = taxonomy.map_label(dataset, raw_label)
        if canonical is None:
            continue
        extents, center, axes = pca_obb(points)
        sorted_extents = np.sort(np.maximum(extents, 0.0))
        diagonal = float(np.linalg.norm(extents))
        point_mask = all_labels == instance_id
        same_fixed = {
            f"{radius:.2f}": float(np.median(fixed_rates[radius][point_mask]))
            for radius in FIXED_RADII_M
        }
        boundary_fixed = {
            f"{radius:.2f}": float(np.mean(fixed_boundary[radius][point_mask]))
            for radius in FIXED_RADII_M
        }
        same_relative: dict[str, float] = {}
        if tree is not None:
            instance_indices = np.flatnonzero(point_mask)
            for fraction in RELATIVE_RADII:
                rate, _ = _rates_for_radius(
                    tree,
                    all_points[instance_indices],
                    all_labels,
                    max(fraction * diagonal, voxel_size_m),
                    instance_indices,
                )
                same_relative[f"{fraction:.2f}"] = float(np.median(rate))
        surface_area = instance_face_area[instance_id]
        quality_valid = bool(
            len(points) >= 30
            and surface_area > 0
            and diagonal > 0
            and np.all(np.isfinite(points))
        )
        rows.append(
            {
                "schema_version": "1.0",
                "dataset": dataset,
                "split": split,
                "scene_id": files.scene_id,
                "physical_scene_id": physical_scene_id(files.scene_id),
                "instance_id": int(instance_id),
                "raw_label": raw_label,
                "canonical_class": canonical,
                "parent_class": taxonomy.parent_for(canonical),
                "units": "meters",
                "metric_scale_valid": True,
                "voxel_size_m": float(voxel_size_m),
                "voxel_count": len(points),
                "surface_area_m2": surface_area,
                "obb_extent_0_m": float(extents[0]),
                "obb_extent_1_m": float(extents[1]),
                "obb_extent_2_m": float(extents[2]),
                "extent_short_m": float(sorted_extents[0]),
                "extent_mid_m": float(sorted_extents[1]),
                "extent_long_m": float(sorted_extents[2]),
                "bbox_diag_m": diagonal,
                "bbox_volume_m3": float(np.prod(extents)),
                "centroid_x_m": float(center[0]),
                "centroid_y_m": float(center[1]),
                "centroid_z_m": float(center[2]),
                "obb_axes_json": json.dumps(axes.tolist(), separators=(",", ":")),
                "same_instance_fixed_json": json.dumps(
                    same_fixed, sort_keys=True, separators=(",", ":")
                ),
                "same_instance_relative_json": json.dumps(
                    same_relative, sort_keys=True, separators=(",", ":")
                ),
                "boundary_fixed_json": json.dumps(
                    boundary_fixed, sort_keys=True, separators=(",", ":")
                ),
                "connected_components": int(
                    voxel_connected_components(points, voxel_size_m)
                ),
                "quality_valid": quality_valid,
                "source_hashes_json": json.dumps(
                    source_hashes, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    audit = {
        "schema_version": "1.0",
        "dataset": dataset,
        "split": split,
        "scene_id": files.scene_id,
        "physical_scene_id": physical_scene_id(files.scene_id),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "aggregation_instance_count": len(groups),
        "mapped_instance_count": len(rows),
        "valid_face_area_fraction": valid_area / total_area if total_area > 0 else 0.0,
        "source_hashes": source_hashes,
        "status": "ok",
    }
    return rows, audit


def prepare_scene_ground_truth(
    files: ScanNetSceneFiles,
    taxonomy: Taxonomy,
    dataset: str = "scannet200",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned mesh vertices with canonical semantic and instance ids.

    Semantic id ``-1`` marks ScanNet200 categories outside the registered SAGA20
    protocol. Instance ids are namespaced only within the scene, as required by the
    evaluator.
    """
    vertices, _ = load_mesh(files.mesh)
    vertices = apply_transform(vertices, read_axis_alignment(files.metadata))
    segments_payload = json.loads(files.segments.read_text(encoding="utf-8"))
    aggregation_payload = json.loads(files.aggregation.read_text(encoding="utf-8"))
    segment_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    if len(segment_indices) != len(vertices):
        raise ValueError(f"{files.scene_id}: segment/vertex count mismatch")
    vertex_instance, raw_labels = _instance_assignment(
        segment_indices, aggregation_payload.get("segGroups", [])
    )
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    semantic = np.full(len(vertices), -1, dtype=np.int64)
    instance = np.full(len(vertices), -1, dtype=np.int64)
    for raw_instance_id, raw_label in raw_labels.items():
        canonical = taxonomy.map_label(dataset, raw_label)
        if canonical is None:
            continue
        mask = vertex_instance == raw_instance_id
        semantic[mask] = class_to_id[canonical]
        instance[mask] = int(raw_instance_id)
    return vertices, semantic, instance


def read_scene_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_scene_ids(scene_ids: Iterable[str]) -> list[str]:
    normalized = [
        str(scene_id).strip() for scene_id in scene_ids if str(scene_id).strip()
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Scene list contains duplicates")
    return normalized
