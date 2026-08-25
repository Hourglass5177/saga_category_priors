from __future__ import annotations

import numpy as np

from category_priors.v10_evaluation import _size_bin_mapping


def test_size_bin_mapping_uses_metric_gt_extent_and_only_marks_tiny_small() -> None:
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    semantic = np.asarray([0, 0, 1, 1])
    instance = np.asarray([4, 4, 5, 5])
    spec = {
        "boundaries_m": {"tiny_max_m": 0.2, "small_max_m": 1.0}
    }
    result = _size_bin_mapping(xyz, semantic, instance, ("chair", "table"), spec)
    assert result == {("chair", 4): "tiny"}
