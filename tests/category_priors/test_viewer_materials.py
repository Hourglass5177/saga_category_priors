from __future__ import annotations

import numpy as np

from category_priors.viewer_materials import (
    _instance_colors,
    _prediction_labels,
    _sample_indices,
    _semantic_colors,
)


def test_sample_indices_are_deterministic_and_bounded() -> None:
    assert _sample_indices(3, 5).tolist() == [0, 1, 2]
    sampled = _sample_indices(10, 4)
    assert sampled.tolist() == [0, 3, 6, 9]
    assert len(np.unique(sampled)) == 4


def test_prediction_semantics_follow_instance_metadata() -> None:
    instances, semantics = _prediction_labels(
        {
            "point_labels": [0, 0, 3, -1],
            "instances": {"0": {"class": "chair"}, "3": {"class": "cup"}},
        },
        ["chair", "table", "cup"],
    )
    assert instances.tolist() == [0, 0, 3, -1]
    assert semantics.tolist() == [0, 0, 2, -1]


def test_colors_are_stable_and_unknown_is_gray() -> None:
    labels = np.asarray([-1, 0, 1, 0])
    instance_colors = _instance_colors(labels)
    semantic_colors = _semantic_colors(labels)
    assert instance_colors[0].tolist() == [96, 96, 96]
    assert semantic_colors[0].tolist() == [96, 96, 96]
    assert np.array_equal(instance_colors[1], instance_colors[3])
    assert np.array_equal(semantic_colors[1], semantic_colors[3])
    assert not np.array_equal(instance_colors[1], instance_colors[2])
