from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from category_priors.scannet import _rates_for_radius, physical_scene_id
from category_priors.selection import select_scenes
from category_priors.taxonomy import load_taxonomy


def test_subset_neighborhood_removes_true_self_index() -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]])
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


def test_24_48_selection_has_no_physical_scene_leakage() -> None:
    rows = []
    for index in range(90):
        scene_id = f"scene{index:04d}_00"
        rows.append(
            {
                "split": "val",
                "scene_id": scene_id,
                "physical_scene_id": physical_scene_id(scene_id),
                "canonical_class": "chair" if index % 2 else "cup",
            }
        )
    result = select_scenes(rows, load_taxonomy())
    tune = result["selection"]["tune"]
    locked = result["selection"]["locked"]
    assert len(tune) == 24
    assert len(locked) == 48
    assert {physical_scene_id(scene) for scene in tune}.isdisjoint(
        {physical_scene_id(scene) for scene in locked}
    )
