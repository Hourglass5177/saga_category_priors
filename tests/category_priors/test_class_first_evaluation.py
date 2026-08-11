from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import category_priors.class_first_evaluation as evaluation_module
from category_priors.class_first_evaluation import (
    evaluate_class_first_runs,
    resolve_evaluation_scenes,
)
from category_priors.evaluator import PredictedInstance
from category_priors.io import load_json, read_rows, write_json
from category_priors.taxonomy import load_taxonomy


SCENES = ("scene0000_00", "scene0001_00", "scene0002_00")
SEEDS = (42, 3407, 20260804)


def _make_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    runtime_path = tmp_path / "runtime.json"
    write_json(
        runtime_path,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "physical_scene_id": scene_id.rsplit("_", 1)[0],
                    "base_path": str(tmp_path / "assets" / scene_id),
                    "gaussian_ply": "gaussians.ply",
                    "scene_scale_m_per_unit": 1.0,
                }
                for scene_id in SCENES
            ],
        },
    )
    selection_path = tmp_path / "selection.json"
    write_json(
        selection_path,
        {
            "kind": "locked_evaluation_scenes",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "physical_scene_id": scene_id.rsplit("_", 1)[0],
                }
                for scene_id in SCENES
            ],
        },
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for scene_id in SCENES:
        coords = np.column_stack(
            (
                np.linspace(0.0, 1.0, 120),
                np.zeros(120),
                np.zeros(120),
            )
        )
        np.savez_compressed(
            gt_dir / f"{scene_id}.npz",
            coords=coords,
            semantic=np.full(120, 9, dtype=np.int64),
            instance=np.zeros(120, dtype=np.int64),
        )
    output_root = tmp_path / "runs"
    for condition in ("U0-uniform", "D-combined"):
        for scene_id in SCENES:
            for seed in SEEDS:
                run_dir = output_root / condition / scene_id / f"seed-{seed}"
                run_dir.mkdir(parents=True)
                write_json(run_dir / "output.json", {"point_labels": [], "instances": {}})
                write_json(
                    run_dir / "diagnostics.json",
                    {
                        "kind": "class_first_scores",
                        "status": "complete",
                        "run": {
                            "scene_id": scene_id,
                            "condition": condition,
                            "seed": seed,
                        },
                        "runner": {
                            "runtime_seconds": 2.0,
                            "point_count": 120,
                            "instance_count": int(
                                condition == "D-combined" and scene_id == SCENES[0]
                            ),
                        },
                        "class_first": {
                            "totals": {
                                "candidate_points": 120,
                                "sampled_points": 20,
                                "hdbscan_noise_points": 2,
                                "sor_removed_points": 1,
                                "rescued_points": 3,
                                "final_instances": int(
                                    condition == "D-combined"
                                    and scene_id == SCENES[0]
                                ),
                                "assigned_points": 120,
                                "coverage": 1.0,
                            },
                            "classes": {
                                "book": {
                                    "candidate_points": 120,
                                    "sampled_points": 20,
                                    "final_instances": 1,
                                    "coverage": 1.0,
                                }
                            },
                        },
                        "instances": {},
                    },
                )
    return runtime_path, selection_path, gt_dir, output_root


def test_selection_json_preserves_physical_scene_ids(tmp_path) -> None:
    runtime_path, selection_path, _, _ = _make_inputs(tmp_path)
    runtime = evaluation_module.load_scene_runtime_manifest(runtime_path)

    scenes, physical = resolve_evaluation_scenes(
        runtime, selection_path=selection_path
    )

    assert scenes == list(SCENES)
    assert physical[SCENES[0]] == "scene0000"


def test_evaluate_class_first_runs_streams_scenes_and_bootstraps(
    tmp_path, monkeypatch
) -> None:
    runtime_path, selection_path, gt_dir, output_root = _make_inputs(tmp_path)
    book_class_id = 9
    seen: list[tuple[str, str]] = []

    def fake_scene_predictions(
        scene_id,
        gt_coords,
        output_json,
        gaussian_ply,
        taxonomy,
        metadata_json,
        transform,
        radius_m,
        require_scores,
    ):
        condition = Path(output_json).parents[2].name
        seen.append((condition, scene_id))
        predictions = []
        if condition == "D-combined" and scene_id == SCENES[0]:
            predictions.append(
                PredictedInstance(
                    scene_id,
                    0,
                    book_class_id,
                    0.9,
                    np.ones(len(gt_coords), dtype=bool),
                )
            )
        return predictions, {
            "mapped_fraction": 1.0,
            "median_nn_distance_m": 0.0,
            "p95_nn_distance_m": 0.0,
        }

    monkeypatch.setattr(
        evaluation_module, "saga_scene_predictions", fake_scene_predictions
    )
    metrics_path = tmp_path / "class_first_metrics.parquet"
    analysis_path = tmp_path / "class_first_analysis.json"

    payload = evaluate_class_first_runs(
        runtime_path,
        gt_dir,
        output_root,
        load_taxonomy(),
        metrics_path=metrics_path,
        analysis_path=analysis_path,
        conditions=["U0-uniform", "D-combined"],
        seeds=SEEDS,
        selection_path=selection_path,
        reference="U0-uniform",
        treatment="D-combined",
        bootstrap_samples=100,
    )

    rows = read_rows(metrics_path)
    assert len(rows) == 6
    treatment_rows = [row for row in rows if row["condition"] == "D-combined"]
    assert all(row["map_50_95"] > 0 for row in treatment_rows)
    assert all(
        row["small_category_map_50_95"] == pytest.approx(row["map_50_95"])
        for row in treatment_rows
    )
    assert len(seen) == len(SCENES) * len(SEEDS) * 2
    assert payload["comparison"]["bootstrap"]["difference"] > 0
    assert set(payload["comparison"]["bootstrap"]["technical_replicates"]) == {
        str(seed) for seed in SEEDS
    }
    assert payload["qualitative_cases"]["best"]["scene_id"] == SCENES[0]
    summary = payload["diagnostics"]["D-combined"]["42"]
    assert summary["class_first_totals"]["candidate_points"]["sum"] == 360
    assert summary["alignment"]["mapped_fraction"]["mean"] == 1.0
    assert load_json(analysis_path)["kind"] == "class_first_analysis"
