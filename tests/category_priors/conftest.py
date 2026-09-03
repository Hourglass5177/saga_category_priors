from __future__ import annotations

import json
from pathlib import Path

import pytest

from category_priors.priors import fit_priors
from category_priors.taxonomy import load_taxonomy


def make_stats_rows(scene_count: int = 6) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene_index in range(scene_count):
        for offset, category in enumerate(("chair", "cup")):
            diagonal = 1.0 if category == "chair" else 0.12
            area = 2.0 if category == "chair" else 0.03
            consistency = 0.90 if category == "chair" else 0.65
            rows.append(
                {
                    "dataset": "scannet200",
                    "split": "train",
                    "scene_id": f"scene{scene_index:04d}_00",
                    "physical_scene_id": f"scene{scene_index:04d}",
                    "instance_id": offset,
                    "canonical_class": category,
                    "units": "meters",
                    "quality_valid": True,
                    "bbox_diag_m": diagonal * (1.0 + 0.01 * scene_index),
                    "surface_area_m2": area * (1.0 + 0.01 * scene_index),
                    "bbox_volume_m3": diagonal**3,
                    "extent_short_m": diagonal * 0.5,
                    "extent_mid_m": diagonal * 0.7,
                    "extent_long_m": diagonal,
                    "voxel_count": 1000 if category == "chair" else 80,
                    "same_instance_fixed_json": json.dumps(
                        {
                            "0.02": consistency,
                            "0.05": consistency,
                            "0.10": consistency,
                            "0.20": consistency,
                        }
                    ),
                    "same_instance_relative_json": json.dumps(
                        {"0.02": consistency, "0.05": consistency, "0.10": consistency}
                    ),
                    "boundary_fixed_json": json.dumps(
                        {"0.02": 0.1, "0.05": 0.1, "0.10": 0.1, "0.20": 0.1}
                    ),
                }
            )
    return rows


@pytest.fixture()
def fitted_priors(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "stats.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    return fit_priors(
        make_stats_rows(),
        load_taxonomy(),
        source,
        bootstrap_samples=40,
        min_physical_scenes=3,
        shrink_tau=3.0,
    )


@pytest.fixture()
def stats_rows() -> list[dict[str, object]]:
    return make_stats_rows()
