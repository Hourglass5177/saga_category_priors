from __future__ import annotations

"""Exact legacy KNN/filter replay shared by candidate experiments.

The implementation intentionally preserves SciPy ``KDTree`` neighbour order,
includes the query point itself, and resolves majority ties by the nearest
occurring label.  Those details are required for pointwise baseline parity.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

GLOBAL_KNN_K = 256
GLOBAL_MIN_COUNT = 10


@dataclass(frozen=True)
class LegacyKNNFilterResult:
    after_knn: np.ndarray
    after_filter: np.ndarray
    k_effective: int
    min_count: int
    instance_count_before_filter: int
    instance_count_after_filter: int
    removed_instance_ids: tuple[int, ...]


def legacy_filter_small_clusters(
    source_labels: Any, min_count: int = GLOBAL_MIN_COUNT
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply the teacher baseline's final point-count filter."""

    labels = _as_numpy(source_labels, np.int64)
    if labels.ndim != 1 or np.any(labels < -1):
        raise ValueError("source_labels must be one-dimensional and use only -1 as negative")
    threshold = int(min_count)
    if threshold < 1:
        raise ValueError("min_count must be positive")
    output = labels.copy()
    values, counts = np.unique(labels[labels >= 0], return_counts=True)
    removed = tuple(
        int(value) for value, count in zip(values, counts) if int(count) < threshold
    )
    for value in removed:
        output[labels == value] = -1
    return _readonly(output), removed


def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _readonly(value: Any, dtype: Any = np.int64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _majority_vote_nearest_tie(
    neighbor_labels: np.ndarray, label_values: np.ndarray
) -> np.ndarray:
    rows, width = neighbor_labels.shape
    encoded = np.searchsorted(label_values, neighbor_labels)
    row_ids = np.repeat(np.arange(rows, dtype=np.int64), width)
    keys = row_ids * len(label_values) + encoded.reshape(-1)
    unique_keys, first, counts = np.unique(
        keys, return_index=True, return_counts=True
    )
    pair_rows = unique_keys // len(label_values)
    pair_labels = unique_keys % len(label_values)
    maximum = np.zeros(rows, dtype=np.int64)
    np.maximum.at(maximum, pair_rows, counts)
    tied = counts == maximum[pair_rows]
    best_first = np.full(rows, keys.size, dtype=np.int64)
    np.minimum.at(best_first, pair_rows[tied], first[tied])
    chosen = tied & (first == best_first[pair_rows])
    output = np.empty(rows, dtype=label_values.dtype)
    output[pair_rows[chosen]] = label_values[pair_labels[chosen]]
    return output


def legacy_knn_filter(
    xyz_scene: Any,
    source_labels: Any,
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    *,
    chunk_size: int = 8_192,
) -> LegacyKNNFilterResult:
    """Replay historical ``filter3d`` followed by ``filter_num`` exactly."""

    from scipy.spatial import KDTree

    xyz = _as_numpy(xyz_scene, np.float64)
    labels = _as_numpy(source_labels, np.int64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or labels.shape != (len(xyz),):
        raise ValueError("xyz_scene and source_labels must describe the same 3D points")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite")
    if np.any(labels < -1):
        raise ValueError("source_labels may only use -1 as its negative label")

    point_count = len(xyz)
    k_effective = min(max(int(k), 1), point_count) if point_count else 0
    voted = np.empty(point_count, dtype=np.int64)
    if point_count:
        tree = KDTree(xyz)
        label_values = np.unique(labels)
        chunk = max(int(chunk_size), 1)
        for start in range(0, point_count, chunk):
            stop = min(start + chunk, point_count)
            _, neighbor_indices = tree.query(xyz[start:stop], k=k_effective)
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64).reshape(
                stop - start, k_effective
            )
            voted[start:stop] = _majority_vote_nearest_tie(
                labels[neighbor_indices], label_values
            )

    values = np.unique(voted[voted >= 0])
    filtered, removed = legacy_filter_small_clusters(voted, min_count)
    return LegacyKNNFilterResult(
        after_knn=_readonly(voted),
        after_filter=_readonly(filtered),
        k_effective=int(k_effective),
        min_count=int(min_count),
        instance_count_before_filter=int(len(values)),
        instance_count_after_filter=int(len(np.unique(filtered[filtered >= 0]))),
        removed_instance_ids=removed,
    )


__all__ = [
    "GLOBAL_KNN_K",
    "GLOBAL_MIN_COUNT",
    "LegacyKNNFilterResult",
    "legacy_filter_small_clusters",
    "legacy_knn_filter",
]
