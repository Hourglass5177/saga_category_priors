from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image


def _stub_module(monkeypatch: pytest.MonkeyPatch, name: str, **members: object) -> None:
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)


def _load_dataset_readers(monkeypatch: pytest.MonkeyPatch):
    _stub_module(monkeypatch, "torch", tensor=object, load=lambda _path: None)
    _stub_module(monkeypatch, "scene", __path__=[])
    _stub_module(
        monkeypatch,
        "scene.colmap_loader",
        read_extrinsics_text=lambda *_: None,
        read_intrinsics_text=lambda *_: None,
        qvec2rotmat=lambda _qvec: np.eye(3),
        read_extrinsics_binary=lambda *_: None,
        read_intrinsics_binary=lambda *_: None,
        read_points3D_binary=lambda *_: None,
        read_points3D_text=lambda *_: None,
    )
    _stub_module(monkeypatch, "utils", __path__=[])
    _stub_module(
        monkeypatch,
        "utils.graphics_utils",
        getWorld2View2=lambda *_: np.eye(4),
        focal2fov=lambda focal, pixels: float(focal) / float(pixels),
        fov2focal=lambda fov, pixels: float(fov) * float(pixels),
    )
    _stub_module(monkeypatch, "utils.sh_utils", SH2RGB=lambda value: value)
    _stub_module(monkeypatch, "scene.gaussian_model", BasicPointCloud=tuple)
    _stub_module(monkeypatch, "plyfile", PlyData=object, PlyElement=object)

    path = Path(__file__).parents[1] / "scene" / "dataset_readers.py"
    spec = importlib.util.spec_from_file_location("dataset_readers_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_image(path: Path) -> None:
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)


def test_image_resolution_priority_and_unique_fuzzy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers = _load_dataset_readers(monkeypatch)
    exact = tmp_path / "frame.jpg"
    exact_stem = tmp_path / "other.png"
    fuzzy = tmp_path / "rgb_000123_color.jpeg"
    _touch_image(exact)
    _touch_image(exact_stem)
    _touch_image(fuzzy)

    assert readers._resolve_colmap_image_path(tmp_path, "frame.jpg") == str(exact)
    assert readers._resolve_colmap_image_path(tmp_path, "other.bmp") == str(
        exact_stem
    )
    assert readers._resolve_colmap_image_path(tmp_path, "000123.jpg") == str(fuzzy)


@pytest.mark.parametrize(
    ("colmap_name", "filenames", "message"),
    [
        ("frame.bmp", ("frame.jpg", "frame.png"), "Ambiguous image stem"),
        (
            "000123.jpg",
            ("rgb_000123.jpg", "depth_000123.png"),
            "Ambiguous fuzzy image match",
        ),
    ],
)
def test_image_resolution_rejects_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    colmap_name: str,
    filenames: tuple[str, ...],
    message: str,
) -> None:
    readers = _load_dataset_readers(monkeypatch)
    for filename in filenames:
        _touch_image(tmp_path / filename)

    with pytest.raises(ValueError, match=message):
        readers._resolve_colmap_image_path(tmp_path, colmap_name)


def test_read_colmap_cameras_records_the_resolved_image_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readers = _load_dataset_readers(monkeypatch)
    actual = tmp_path / "frame.png"
    _touch_image(actual)

    class Extrinsic:
        name = "frame.jpg"
        camera_id = 1
        qvec = np.asarray([1.0, 0.0, 0.0, 0.0])
        tvec = np.zeros(3)

    class Intrinsic:
        id = 1
        height = 2
        width = 2
        model = "PINHOLE"
        params = np.asarray([1.0, 1.0, 1.0, 1.0])

    cameras = readers.readColmapCameras(
        {0: Extrinsic()}, {1: Intrinsic()}, str(tmp_path)
    )

    assert len(cameras) == 1
    assert cameras[0].image_path == str(actual)
    assert cameras[0].image.size == (2, 2)
