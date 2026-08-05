from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from category_priors.alignment import audit_saga_alignment
from category_priors.io import hash_json
from category_priors.scannet_saga import prepare_saga_scene, read_sens_header


def _write_sens(path: Path, poses: list[np.ndarray]) -> None:
    identity = np.eye(4, dtype=np.float32)
    sensor_name = b"fake-scannet-sensor"
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", 4))
        handle.write(struct.pack("<Q", len(sensor_name)))
        handle.write(sensor_name)
        for matrix in (identity, identity, identity, identity):
            handle.write(struct.pack("<16f", *matrix.ravel()))
        handle.write(struct.pack("<ii", 2, 1))
        handle.write(struct.pack("<4I", 4, 3, 2, 2))
        handle.write(struct.pack("<f", 1000.0))
        handle.write(struct.pack("<Q", len(poses)))
        for index, pose in enumerate(poses):
            color = b"\xff\xd8" + bytes([index]) + b"\xff\xd9"
            depth = b"depth"
            handle.write(struct.pack("<16f", *pose.astype(np.float32).ravel()))
            handle.write(struct.pack("<4Q", index, index, len(color), len(depth)))
            handle.write(color)
            handle.write(depth)


def _write_scene_files(root: Path, scene_id: str) -> None:
    scene = root / scene_id
    scene.mkdir(parents=True)
    vertices = np.array(
        [
            (0.0, 0.0, 0.0, 255, 0, 0),
            (1.0, 0.0, 0.0, 0, 255, 0),
            (0.0, 1.0, 0.0, 0, 0, 255),
            (0.0, 0.0, 1.0, 255, 255, 255),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    PlyData([PlyElement.describe(vertices, "vertex")]).write(
        str(scene / f"{scene_id}_vh_clean_2.ply")
    )
    (scene / f"{scene_id}.aggregation.json").write_text("{}", encoding="utf-8")
    (scene / f"{scene_id}_vh_clean_2.0.010000.segs.json").write_text(
        "{}", encoding="utf-8"
    )
    alignment = np.eye(4)
    alignment[0, 3] = 2.0
    (scene / f"{scene_id}.txt").write_text(
        "axisAlignment = " + " ".join(str(value) for value in alignment.ravel()),
        encoding="utf-8",
    )


def test_read_sens_header_and_prepare_axis_aligned_scene(tmp_path: Path) -> None:
    scene_id = "scene0000_00"
    dataset_root = tmp_path / "scans"
    _write_scene_files(dataset_root, scene_id)
    poses = [np.eye(4) for _ in range(5)]
    poses[2] = np.full((4, 4), np.nan)
    sens_path = dataset_root / scene_id / f"{scene_id}.sens"
    _write_sens(sens_path, poses)

    with sens_path.open("rb") as handle:
        header = read_sens_header(handle)
    assert header.num_frames == 5
    assert header.color_compression == "jpeg"
    assert header.color_width == 4

    manifest = prepare_saga_scene(
        dataset_root,
        scene_id,
        sens_path,
        tmp_path / "prepared",
        frame_stride=2,
        max_frames=3,
        max_initial_points=3,
    )

    base = tmp_path / "prepared" / scene_id
    sparse = base / "fastRecon" / "dense" / "sparse" / "0"
    assert manifest["scene_scale_m_per_unit"] == 1.0
    assert manifest["gaussian_to_gt_transform"] == np.eye(4).tolist()
    assert manifest["frame_selection"]["selected_valid_frames"] == 2
    assert sorted(path.name for path in (sparse / "images").glob("*.jpg")) == [
        "frame-000000.jpg",
        "frame-000004.jpg",
    ]
    assert "PINHOLE 4 3 1 1 0 0" in (sparse / "cameras.txt").read_text()
    assert len(
        [
            line
            for line in (sparse / "images.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ]
    ) == 2

    point_vertex = PlyData.read(str(sparse / "points3D.ply"))["vertex"]
    assert len(point_vertex) == 3
    assert np.min(point_vertex["x"]) >= 2.0
    unsigned = dict(manifest)
    assert unsigned.pop("content_sha256") == hash_json(unsigned)


def test_alignment_audit_passes_and_writes_failure_diagnostics(tmp_path: Path) -> None:
    scene_id = "scene0000_00"
    dataset_root = tmp_path / "scans"
    _write_scene_files(dataset_root, scene_id)
    sens_path = dataset_root / scene_id / f"{scene_id}.sens"
    _write_sens(sens_path, [np.eye(4) for _ in range(4)])
    manifest = prepare_saga_scene(
        dataset_root,
        scene_id,
        sens_path,
        tmp_path / "prepared",
        frame_stride=1,
        max_frames=4,
        max_initial_points=4,
    )
    base = Path(manifest["base_path"])
    initial_path = base / manifest["initial_point_cloud"]["path"]
    vertices = PlyData.read(str(initial_path))["vertex"]
    coords = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    gt_path = tmp_path / "gt.npz"
    np.savez_compressed(
        gt_path,
        coords=coords,
        semantic=np.zeros(len(coords), dtype=np.int64),
        instance=np.zeros(len(coords), dtype=np.int64),
    )

    audit_path = tmp_path / "alignment.json"
    audit = audit_saga_alignment(
        base / "scene_preparation_manifest.json", gt_path, audit_path
    )
    assert audit["passed"] is True
    assert audit["gt_to_cloud"]["mapped_fraction"] == 1.0
    assert audit["cameras"]["inside_padded_gt_fraction"] == 1.0

    bad_vertices = np.array(vertices.data, copy=True)
    bad_vertices["x"] += 100.0
    bad_path = tmp_path / "bad.ply"
    PlyData([PlyElement.describe(bad_vertices, "vertex")]).write(str(bad_path))
    failed_path = tmp_path / "failed-alignment.json"
    with pytest.raises(RuntimeError, match="coverage_below_threshold"):
        audit_saga_alignment(
            base / "scene_preparation_manifest.json",
            gt_path,
            failed_path,
            gaussian_ply_path=bad_path,
        )
    assert failed_path.is_file()
