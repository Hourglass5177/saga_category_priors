from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from category_priors.object_verification.artifacts import file_digest
from category_priors.object_verification.frozen_scene import infer_sh_degree, load_text_cameras

def test_native_sh_degree_requires_exact_registered_coefficients():
    dc = [f"f_dc_{i}" for i in range(3)]
    assert infer_sh_degree(dc) == 0
    assert infer_sh_degree(dc + [f"f_rest_{i}" for i in range(45)]) == 3
    with pytest.raises(ValueError):
        infer_sh_degree(dc + ["f_rest_1"])
    with pytest.raises(ValueError):
        infer_sh_degree(dc + [f"f_rest_{i}" for i in range(6)])

def test_text_camera_image_name_sort_and_explicit_rgb_identity(tmp_path):
    sparse, images = tmp_path / "sparse", tmp_path / "images"
    sparse.mkdir(); images.mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 12 8 10 10 6.2 4.2\n")
    (sparse / "images.txt").write_text("2 1 0 0 0 0 0 0 1 frame-2.jpg\n\n1 1 0 0 0 0 0 0 1 frame-1.jpg\n\n")
    for name in ("frame-1.png", "frame-2.png"):
        Image.fromarray(np.zeros((8, 12, 3), np.uint8)).save(images / name)
    files = {str(p): file_digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    cameras = load_text_cameras(sparse, images, scene_id="s", scale_m_per_unit=1, registered_files=files)
    assert [(c.old_index, c.uid) for c in cameras] == [(0, "s:frame-1"), (1, "s:frame-2")]
    (images / "frame-1.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_text_cameras(sparse, images, scene_id="s", scale_m_per_unit=1, registered_files=files)
