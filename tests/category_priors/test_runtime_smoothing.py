from __future__ import annotations

import numpy as np

from category_priors.runtime import _majority_neighbor_labels


def reference_majority(
    labels: np.ndarray,
    current: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    ks: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    result = current.copy()
    for row in range(len(current)):
        k = int(ks[row])
        valid = (
            np.isfinite(distances[row, :k])
            & (indices[row, :k] < len(labels))
            & (distances[row, :k] <= radii[row])
        )
        neighbors = labels[indices[row, :k][valid]]
        if len(neighbors):
            values, counts = np.unique(neighbors, return_counts=True)
            result[row] = values[np.argmax(counts)]
    return result


def test_vectorized_majority_matches_legacy_randomized() -> None:
    rng = np.random.default_rng(20260806)
    for _ in range(50):
        labels = rng.integers(-1, 12, size=200, dtype=np.int64)
        rows, max_k = 37, 16
        indices = rng.integers(0, len(labels) + 3, size=(rows, max_k))
        distances = rng.random((rows, max_k))
        distances[indices >= len(labels)] = np.inf
        ks = rng.integers(1, max_k + 1, size=rows)
        radii = rng.choice([0.15, 0.4, 1.0, np.inf], size=rows)
        current = labels[:rows]
        expected = reference_majority(labels, current, distances, indices, ks, radii)
        actual = _majority_neighbor_labels(
            labels, current, distances, indices, ks, radii
        )
        np.testing.assert_array_equal(actual, expected)


def test_vectorized_majority_preserves_smallest_label_tie_break() -> None:
    labels = np.array([4, 2, 4, 2, 9], dtype=np.int64)
    current = np.array([9], dtype=np.int64)
    indices = np.array([[0, 1, 2, 3]], dtype=np.int64)
    distances = np.zeros_like(indices, dtype=np.float64)
    actual = _majority_neighbor_labels(
        labels,
        current,
        distances,
        indices,
        np.array([4]),
        np.array([np.inf]),
    )
    np.testing.assert_array_equal(actual, np.array([2]))
