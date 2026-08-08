from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from category_priors.scannet import _rates_for_radius, physical_scene_id
from category_priors.selection import (
    select_locked_evaluation_scenes,
    select_scenes,
)
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


def test_locked_evaluation_selection_uses_independent_candidate_scenes() -> None:
    taxonomy = load_taxonomy()
    rows = []
    locked = []
    locked_replacements = []
    for index in range(60):
        category = taxonomy.canonical_classes[index % len(taxonomy.canonical_classes)]
        for scan_index in range(2):
            scene_id = f"scene{index:04d}_{scan_index:02d}"
            (locked if index < 24 else locked_replacements).append(scene_id)
            rows.extend(
                {
                    "split": "val",
                    "scene_id": scene_id,
                    "canonical_class": category,
                }
                for _ in range(scan_index + 1)
            )
    rows.append(
        {
            "split": "val",
            "scene_id": "scene9999_00",
            "canonical_class": "chair",
        }
    )
    previous_selection = {
        "selection": {
            "tune": ["scene1000_00"],
            "tune_replacements": ["scene1001_00"],
            "locked": locked,
            "locked_replacements": locked_replacements,
        }
    }

    first = select_locked_evaluation_scenes(rows, taxonomy, previous_selection)
    second = select_locked_evaluation_scenes(rows, taxonomy, previous_selection)
    selected = first["scenes"]
    candidate_set = set(locked + locked_replacements)

    assert first == second
    assert first["kind"] == "locked_evaluation_scenes"
    assert first["split"] == "val-locked"
    assert len(selected) == 48
    assert len({item["physical_scene_id"] for item in selected}) == 48
    assert {item["scene_id"] for item in selected} <= candidate_set
    assert set(first["coverage"]) == set(taxonomy.canonical_classes)
    assert all(
        first["coverage"][category] > 0
        for category in taxonomy.canonical_classes
    )
    assert first["candidate_scan_count"] == 120
    assert first["candidate_physical_scene_count"] == 60
    assert "sha256" not in json.dumps(first).lower()


def test_locked_evaluation_selection_breaks_ties_by_scene_id_and_removes_group() -> None:
    rows = [
        {"split": "val", "scene_id": scene, "canonical_class": "chair"}
        for scene in ("scene0100_00", "scene0100_01", "scene0101_00")
    ]
    previous_selection = {
        "selection": {
            "tune": [],
            "tune_replacements": [],
            "locked": ["scene0100_01", "scene0101_00"],
            "locked_replacements": ["scene0100_00"],
        }
    }

    result = select_locked_evaluation_scenes(
        rows,
        load_taxonomy(),
        previous_selection,
        locked_budget=2,
    )

    assert [item["scene_id"] for item in result["scenes"]] == [
        "scene0100_00",
        "scene0101_00",
    ]


def test_locked_evaluation_selection_rejects_tune_physical_scene_overlap() -> None:
    rows = [
        {"split": "val", "scene_id": "scene0100_00", "canonical_class": "chair"}
    ]
    previous_selection = {
        "selection": {
            "tune": [],
            "tune_replacements": ["scene0100_01"],
            "locked": ["scene0100_00"],
            "locked_replacements": [],
        }
    }

    with pytest.raises(ValueError, match="Physical scene leakage"):
        select_locked_evaluation_scenes(
            rows,
            load_taxonomy(),
            previous_selection,
            locked_budget=1,
        )
