from __future__ import annotations

"""Small, experiment-neutral helpers for 3D audit visualizations."""

from pathlib import Path

import numpy as np


def condition_slug(condition: str) -> str:
    """Return a stable, filesystem-friendly condition name."""

    return "".join(
        character.lower() if character.isalnum() else "_"
        for character in condition
    ).strip("_")


def write_colored_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write XYZ points with uint8 RGB colors to a binary PLY file."""

    from plyfile import PlyData, PlyElement

    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3)")
    if rgb.shape != xyz.shape:
        raise ValueError("rgb must have the same (N, 3) shape as xyz")

    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
