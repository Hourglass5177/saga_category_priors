from __future__ import annotations

"""Small geometry helpers shared by category-prior experiments."""

from typing import Any

import numpy as np

from .scannet import pca_obb


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def pca_sorted_extents_m(
    points_scene: Any, scene_scale_m_per_unit: float
) -> np.ndarray:
    """Return train-prior-compatible PCA box extents in ascending metric order."""

    points = _as_numpy(points_scene, np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_scene must contain at least one 3D point")
    scale = float(scene_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    extents, _, _ = pca_obb(points * scale)
    return np.sort(np.maximum(np.asarray(extents, dtype=np.float64), 0.0))


def unit_cube_coordinates(points: Any) -> np.ndarray:
    """Min-max normalise 3D points without dividing by a zero-length axis.

    The teacher baseline normalises each scene axis independently before its
    spatial distance is mixed with the affinity distance.  A planar scene (or
    a small synthetic regression fixture) can have a constant axis; that axis
    contributes zero distance instead of producing NaNs.
    """

    array = _as_numpy(points)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("points must contain at least one 3D point")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("points must be finite numeric values")
    minimum = array.min(axis=0)
    span = array.max(axis=0) - minimum
    denominator = np.where(span > 0, span, np.ones_like(span))
    result = (array - minimum) / denominator
    return np.asarray(result, dtype=array.dtype)


def nonnegative_cluster_ids(labels: Any) -> tuple[int, ...]:
    """Return sorted cluster IDs without assuming that noise ``-1`` exists."""

    array = _as_numpy(labels)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("cluster labels must be a one-dimensional integer array")
    return tuple(int(value) for value in np.unique(array) if int(value) >= 0)


__all__ = [
    "nonnegative_cluster_ids",
    "pca_sorted_extents_m",
    "unit_cube_coordinates",
]
