from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any

import numpy as np

from .io import hash_json, sha256_file, write_json
from .scannet import apply_transform, discover_scene_files, read_axis_alignment


COLOR_COMPRESSION = {0: "raw", 1: "png", 2: "jpeg"}
DEPTH_COMPRESSION = {0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


@dataclass(frozen=True)
class SensHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression: str
    depth_compression: str
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    num_frames: int


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    payload = handle.read(size)
    if len(payload) != size:
        raise EOFError(f"Unexpected end of .sens stream: wanted {size}, got {len(payload)}")
    return payload


def _unpack(handle: BinaryIO, fmt: str) -> tuple[Any, ...]:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(handle, size))


def read_sens_header(handle: BinaryIO) -> SensHeader:
    version = int(_unpack(handle, "I")[0])
    if version != 4:
        raise ValueError(f"Unsupported .sens version: {version}")
    sensor_name_size = int(_unpack(handle, "Q")[0])
    sensor_name = _read_exact(handle, sensor_name_size).decode(
        "utf-8", errors="replace"
    )
    matrices = [
        np.asarray(_unpack(handle, "16f"), dtype=np.float64).reshape(4, 4)
        for _ in range(4)
    ]
    color_code = int(_unpack(handle, "i")[0])
    depth_code = int(_unpack(handle, "i")[0])
    if color_code not in COLOR_COMPRESSION:
        raise ValueError(f"Unknown .sens color compression code: {color_code}")
    if depth_code not in DEPTH_COMPRESSION:
        raise ValueError(f"Unknown .sens depth compression code: {depth_code}")
    color_width, color_height, depth_width, depth_height = (
        int(value) for value in _unpack(handle, "4I")
    )
    depth_shift = float(_unpack(handle, "f")[0])
    num_frames = int(_unpack(handle, "Q")[0])
    return SensHeader(
        version=version,
        sensor_name=sensor_name,
        intrinsic_color=matrices[0],
        extrinsic_color=matrices[1],
        intrinsic_depth=matrices[2],
        extrinsic_depth=matrices[3],
        color_compression=COLOR_COMPRESSION[color_code],
        depth_compression=DEPTH_COMPRESSION[depth_code],
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        num_frames=num_frames,
    )


def _selected_frame_indices(
    num_frames: int, frame_stride: int, max_frames: int | None
) -> set[int]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    candidates = np.arange(0, num_frames, frame_stride, dtype=np.int64)
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if len(candidates) > max_frames:
            positions = np.rint(np.linspace(0, len(candidates) - 1, max_frames)).astype(
                np.int64
            )
            candidates = candidates[np.unique(positions)]
    return {int(value) for value in candidates}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, payload: str) -> None:
    _atomic_write_bytes(path, payload.encode("utf-8"))


def _rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def _valid_pose(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return False
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        return False
    rotation = pose[:3, :3]
    return bool(
        np.isclose(np.linalg.det(rotation), 1.0, atol=2e-2)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-2)
    )


def _extract_frames_and_colmap_images(
    sens_path: Path,
    image_dir: Path,
    axis_alignment: np.ndarray,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[SensHeader, list[dict[str, Any]], list[str]]:
    with sens_path.open("rb") as handle:
        header = read_sens_header(handle)
        if header.color_compression not in {"jpeg", "png"}:
            raise ValueError(
                "SAGA scene preparation supports JPEG or PNG .sens color frames"
            )
        selected = _selected_frame_indices(
            header.num_frames, frame_stride=frame_stride, max_frames=max_frames
        )
        image_records: list[dict[str, Any]] = []
        image_lines = [
            "# Image list with two lines of data per image:",
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
            "# POINTS2D[] as (X, Y, POINT3D_ID)",
        ]
        extension = ".jpg" if header.color_compression == "jpeg" else ".png"
        for frame_index in range(header.num_frames):
            camera_to_world = np.asarray(
                _unpack(handle, "16f"), dtype=np.float64
            ).reshape(4, 4)
            timestamp_color, timestamp_depth, color_size, depth_size = _unpack(
                handle, "4Q"
            )
            if frame_index not in selected:
                handle.seek(int(color_size) + int(depth_size), os.SEEK_CUR)
                continue
            color_payload = _read_exact(handle, int(color_size))
            handle.seek(int(depth_size), os.SEEK_CUR)
            if not _valid_pose(camera_to_world):
                continue
            aligned_camera_to_world = axis_alignment @ camera_to_world
            world_to_camera = np.linalg.inv(aligned_camera_to_world)
            qvec = _rotation_matrix_to_qvec(world_to_camera[:3, :3])
            tvec = world_to_camera[:3, 3]
            filename = f"frame-{frame_index:06d}{extension}"
            _atomic_write_bytes(image_dir / filename, color_payload)
            image_id = len(image_records) + 1
            values = [*qvec.tolist(), *tvec.tolist()]
            image_lines.append(
                f"{image_id} "
                + " ".join(f"{value:.17g}" for value in values)
                + f" 1 {filename}"
            )
            image_lines.append("")
            image_records.append(
                {
                    "image_id": image_id,
                    "frame_index": frame_index,
                    "filename": filename,
                    "timestamp_color": int(timestamp_color),
                    "timestamp_depth": int(timestamp_depth),
                }
            )
    if len(image_records) < 2:
        raise ValueError(
            f"Only {len(image_records)} valid selected frames; at least two are required"
        )
    return header, image_records, image_lines


def _load_aligned_initial_points(
    mesh_path: Path,
    axis_alignment: np.ndarray,
    max_initial_points: int,
    seed_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if max_initial_points <= 0:
        raise ValueError("max_initial_points must be positive")
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ScanNet SAGA preparation requires plyfile") from exc
    vertex = PlyData.read(str(mesh_path))["vertex"]
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64
    )
    points = apply_transform(points, axis_alignment)
    names = set(vertex.data.dtype.names or ())
    if {"red", "green", "blue"} <= names:
        colors = np.column_stack(
            (vertex["red"], vertex["green"], vertex["blue"])
        ).astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 127, dtype=np.uint8)
    if len(points) > max_initial_points:
        seed = int.from_bytes(
            hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(len(points), size=max_initial_points, replace=False)
        )
        points = points[indices]
        colors = colors[indices]
    return points, colors


def _write_initial_point_cloud(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("ScanNet SAGA preparation requires plyfile") from exc
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    vertices = np.empty(len(points), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(
        str(temporary)
    )
    os.replace(temporary, path)


def prepare_saga_scene(
    dataset_root: str | Path,
    scene_id: str,
    sens_path: str | Path,
    output_root: str | Path,
    frame_stride: int = 20,
    max_frames: int | None = 200,
    max_initial_points: int = 200_000,
) -> dict[str, Any]:
    """Prepare metric, axis-aligned ScanNet RGB/poses for SAGA/3DGS.

    The .sens binary layout follows ScanNet's MIT-licensed SensReader. Color
    frames remain compressed; depth payloads are intentionally skipped because
    the registered postprocess-only experiment needs RGB, poses, and intrinsics.
    """
    files = discover_scene_files(dataset_root, scene_id)
    sens_path = Path(sens_path).resolve()
    if not sens_path.is_file() or sens_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing nonempty .sens file: {sens_path}")
    base_path = Path(output_root).resolve() / scene_id
    sparse_path = base_path / "fastRecon" / "dense" / "sparse" / "0"
    image_dir = sparse_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    axis_alignment = read_axis_alignment(files.metadata)
    header, image_records, image_lines = _extract_frames_and_colmap_images(
        sens_path,
        image_dir,
        axis_alignment,
        frame_stride=frame_stride,
        max_frames=max_frames,
    )
    intrinsic = header.intrinsic_color
    fx, fy, cx, cy = (
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
    )
    if min(fx, fy) <= 0 or not np.all(np.isfinite([fx, fy, cx, cy])):
        raise ValueError("Invalid ScanNet color intrinsics")
    cameras_text = (
        "# Camera list with one line of data per camera:\n"
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        f"1 PINHOLE {header.color_width} {header.color_height} "
        f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n"
    )
    _atomic_write_text(sparse_path / "cameras.txt", cameras_text)
    _atomic_write_text(sparse_path / "images.txt", "\n".join(image_lines) + "\n")

    points, colors = _load_aligned_initial_points(
        files.mesh,
        axis_alignment,
        max_initial_points=max_initial_points,
        seed_key=scene_id,
    )
    point_cloud_path = sparse_path / "points3D.ply"
    _write_initial_point_cloud(point_cloud_path, points, colors)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "scannet_saga_scene",
        "scene_id": scene_id,
        "base_path": str(base_path).replace("\\", "/"),
        "scene_scale_m_per_unit": 1.0,
        "gaussian_to_gt_transform": np.eye(4).tolist(),
        "coordinate_system": "ScanNet axisAlignment applied to cameras and mesh",
        "source": {
            "sens_path": str(sens_path).replace("\\", "/"),
            "sens_sha256": sha256_file(sens_path),
            "mesh_path": str(files.mesh.resolve()).replace("\\", "/"),
            "mesh_sha256": sha256_file(files.mesh),
            "metadata_path": str(files.metadata.resolve()).replace("\\", "/"),
            "metadata_sha256": sha256_file(files.metadata),
        },
        "axis_alignment": axis_alignment.tolist(),
        "frame_selection": {
            "source_frames": header.num_frames,
            "frame_stride": frame_stride,
            "max_frames": max_frames,
            "selected_valid_frames": len(image_records),
            "frames": image_records,
        },
        "camera": {
            "model": "PINHOLE",
            "width": header.color_width,
            "height": header.color_height,
            "parameters": [fx, fy, cx, cy],
            "sensor_name": header.sensor_name,
        },
        "initial_point_cloud": {
            "path": str(point_cloud_path.relative_to(base_path)).replace("\\", "/"),
            "vertices": len(points),
            "sha256": sha256_file(point_cloud_path),
        },
    }
    manifest["content_sha256"] = hash_json(manifest)
    write_json(base_path / "scene_preparation_manifest.json", manifest)
    return manifest
