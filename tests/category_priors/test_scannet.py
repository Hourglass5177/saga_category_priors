from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from category_priors.scannet import _rates_for_radius


def test_subset_neighborhood_removes_true_self_index() -> None:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    labels = np.asarray([1, 2, 2])

    rates, boundary = _rates_for_radius(
        cKDTree(points),
        points[[1]],
        labels,
        0.02,
        np.asarray([1]),
    )

    assert rates.tolist() == [0.0]
    assert boundary.tolist() == [1.0]
