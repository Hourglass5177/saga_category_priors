from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from category_priors.io import hash_json, sha256_file, write_json
from category_priors.mapping import latin_hypercube_design
from category_priors.search import (
    build_search_schedule,
    evaluate_search_execution,
    materialize_search_mappings,
)
from category_priors.taxonomy import load_taxonomy


def test_materialized_global_search_is_block_complete(tmp_path: Path) -> None:
    design_path = tmp_path / "global_design.json"
    write_json(design_path, latin_hypercube_design("global", samples=3, seed=7))
    priors = tmp_path / "priors.json"
    taxonomy = tmp_path / "taxonomy.json"
    selection = tmp_path / "selection.json"
    write_json(priors, {"source": "train"})
    write_json(taxonomy, {"classes": ["chair"]})
    write_json(
        selection,
        {"selection": {"tune": ["scene0001_00", "scene0002_00"], "locked": []}},
    )
    manifest_path = tmp_path / "mapping_manifest.json"
    manifest = materialize_search_mappings(
        design_path,
        tmp_path / "mappings",
        manifest_path,
        priors,
        taxonomy,
        selection,
    )
    assert len(manifest["configurations"]) == 3
    assert all(
        (_resolve := manifest_path.parent / item["path"]).is_file()
        and sha256_file(_resolve) == item["sha256"]
        for item in manifest["configurations"]
    )

    schedule = build_search_schedule(selection, manifest_path, (42,), 11)
    assert schedule["split"] == "val-tune"
    assert schedule["condition"] == "P000-B2"
    assert len(schedule["runs"]) == 6
    for block in {run["block"] for run in schedule["runs"]}:
        assert {
            run["config_id"] for run in schedule["runs"] if run["block"] == block
        } == {"global-000", "global-001", "global-002"}


def _write_xyz_ply(path: Path, coords: np.ndarray) -> None:
    vertices = np.empty(len(coords), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertices["x"], vertices["y"], vertices["z"] = coords.T
    PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path))


def test_search_evaluation_writes_tune_metric_row(tmp_path: Path) -> None:
    scene_id = "scene0000_00"
    coords = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    np.savez_compressed(
        gt_dir / f"{scene_id}.npz",
        coords=coords,
        semantic=np.zeros(3, dtype=np.int64),
        instance=np.ones(3, dtype=np.int64),
    )
    gt_manifest = gt_dir / "manifest.json"
    write_json(
        gt_manifest,
        {
            "kind": "canonical_ground_truth",
            "scenes": [{"scene_id": scene_id, "path": f"{scene_id}.npz"}],
        },
    )

    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    gaussian_ply = scene_dir / "point_cloud.ply"
    _write_xyz_ply(gaussian_ply, coords)
    scene_manifest = tmp_path / "scenes.json"
    write_json(
        scene_manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "base_path": str(scene_dir),
                    "scene_scale_m_per_unit": 1.0,
                    "gaussian_ply": str(gaussian_ply),
                    "gaussian_to_gt_transform": np.eye(4).tolist(),
                }
            ],
        },
    )
    output_json = tmp_path / "output.json"
    metadata_json = tmp_path / "output.json.metadata.json"
    write_json(
        output_json,
        {
            "point_labels": [10, 10, 10],
            "instances": {"10": {"class": "chair"}},
        },
    )
    write_json(
        metadata_json,
        {"instances": {"10": {"class": "chair", "score": 0.9}}},
    )
    schedule_path = tmp_path / "schedule.json"
    schedule = {
        "kind": "run_schedule",
        "split": "val-tune",
        "search_kind": "global",
        "condition": "P000-B2",
        "runs": [
            {
                "sequence": 0,
                "scene_id": scene_id,
                "run_seed": 42,
                "config_id": "global-000",
            }
        ],
    }
    schedule["content_sha256"] = hash_json(schedule)
    write_json(schedule_path, schedule)
    execution_path = tmp_path / "execution.json"
    write_json(
        execution_path,
        {
            "kind": "run_execution",
            "schedule_sha256": sha256_file(schedule_path),
            "runs": [
                {
                    "sequence": 0,
                    "status": "complete",
                    "runtime_seconds": 2.5,
                    "output_json": str(output_json),
                    "metadata_json": str(metadata_json),
                }
            ],
        },
    )
    rows = evaluate_search_execution(
        schedule_path,
        execution_path,
        scene_manifest,
        gt_manifest,
        load_taxonomy(),
        tmp_path / "evaluation",
        tmp_path / "metrics.parquet",
        min_region_size=1,
    )
    assert len(rows) == 1
    assert rows[0]["split"] == "val-tune"
    assert rows[0]["scene_count"] == 1
    assert rows[0]["map_50_95"] == 1.0
    assert rows[0]["runtime_seconds"] == 2.5
