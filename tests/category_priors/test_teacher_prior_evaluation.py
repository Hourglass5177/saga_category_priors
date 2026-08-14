from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import category_priors.class_first_evaluation as base_evaluation
from category_priors.evaluator import PredictedInstance
from category_priors.io import load_json, read_rows, write_json
from category_priors.taxonomy import load_taxonomy
from category_priors.teacher_prior_evaluation import (
    evaluate_teacher_prior_runs,
    partition_change_fraction,
)


SCENES = ("scene0000_00", "scene0001_00", "scene0002_00")
SEEDS = (42, 3407)


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
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(SCENES) + "\n", encoding="utf-8")
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for scene_id in SCENES:
        coords = np.column_stack(
            (np.linspace(0.0, 1.0, 120), np.zeros(120), np.zeros(120))
        )
        np.savez_compressed(
            gt_dir / f"{scene_id}.npz",
            coords=coords,
            semantic=np.full(120, 9, dtype=np.int64),
            instance=np.zeros(120, dtype=np.int64),
        )
    output_root = tmp_path / "runs"
    for condition in ("U0-all-uniform", "D-small"):
        for scene_id in SCENES:
            for seed in SEEDS:
                run_dir = output_root / condition / scene_id / f"seed-{seed}"
                run_dir.mkdir(parents=True)
                changed = int(condition == "D-small" and scene_id == SCENES[0])
                point_labels = np.zeros(120, dtype=np.int64)
                if changed:
                    point_labels[:12] = 1
                write_json(
                    run_dir / "output.json",
                    {
                        "point_labels": point_labels.tolist(),
                        "instances": {
                            "0": {"class": "chair"},
                            **({"1": {"class": "chair"}} if changed else {}),
                        },
                    },
                )
                write_json(
                    run_dir / "diagnostics.json",
                    {
                        "kind": "teacher_prior_diagnostics",
                        "status": "complete",
                        "run": {
                            "scene_id": scene_id,
                            "condition": condition,
                            "seed": seed,
                        },
                        "runner": {
                            "runtime_seconds": 2.0,
                            "point_count": 120,
                            "instance_count": changed,
                        },
                        "teacher_prior": {
                            "totals": {
                                "changed_points": 12 * changed,
                                "candidate_points": 120,
                                "sampled_points": 20,
                                "noise_points": 2,
                                "rescued_points": 3 * changed,
                                "accepted_instances": changed,
                                "coverage": float(changed),
                            }
                        },
                    },
                )
    return runtime_path, scene_list, gt_dir, output_root


def test_teacher_evaluation_filters_and_averages_technical_seeds(
    tmp_path, monkeypatch
) -> None:
    runtime_path, scene_list, gt_dir, output_root = _make_inputs(tmp_path)
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
        if condition == "D-small" and scene_id == SCENES[0]:
            predictions.append(
                PredictedInstance(
                    scene_id,
                    0,
                    9,
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
        base_evaluation, "saga_scene_predictions", fake_scene_predictions
    )
    metrics_path = tmp_path / "teacher_prior_metrics.parquet"
    analysis_path = tmp_path / "teacher_prior_analysis.json"

    payload = evaluate_teacher_prior_runs(
        runtime_path,
        gt_dir,
        output_root,
        load_taxonomy(),
        metrics_path=metrics_path,
        analysis_path=analysis_path,
        conditions=["U0-all-uniform", "D-small"],
        seeds=SEEDS,
        scene_list_path=scene_list,
        reference="U0-all-uniform",
        treatment="D-small",
        bootstrap_samples=100,
    )

    rows = read_rows(metrics_path)
    assert len(rows) == 4
    treatment_rows = [row for row in rows if row["condition"] == "D-small"]
    assert all(row["map_50_95"] > 0 for row in treatment_rows)
    assert all(
        row["small_category_map_50_95"] == pytest.approx(row["map_50_95"])
        for row in treatment_rows
    )
    assert len(seen) == len(SCENES) * len(SEEDS) * 2
    bootstrap = payload["comparison"]["bootstrap"]
    assert bootstrap["difference"] > 0
    assert set(bootstrap["technical_replicates"]) == {"42", "3407"}
    assert payload["best_median_worst"]["best"]["scene_id"] == SCENES[0]
    summary = payload["structural_diagnostics"]["D-small"]["42"]
    assert summary["teacher_prior"]["totals.changed_points"]["sum"] == 12
    assert summary["alignment"]["mapped_fraction"]["mean"] == 1.0
    assert payload["condition_modes"]["U0-all-uniform"] == "all-uniform"
    intervention = payload["intervention_diagnostics"]["D-small"]
    assert intervention["max_partition_change_fraction"] == pytest.approx(0.1)
    assert intervention["mean_partition_change_fraction"] == pytest.approx(1 / 30)
    assert intervention["mean_coverage_delta"] == pytest.approx(0.0)
    assert intervention["mean_instance_ratio"] == pytest.approx(4 / 3)
    saved = load_json(analysis_path)
    assert saved["kind"] == "teacher_prior_analysis"
    assert "book" in saved["per_condition_seed_metrics"]["D-small"]["42"][
        "per_class"
    ]


def test_partition_change_fraction_ignores_instance_id_renaming() -> None:
    reference = [0, 0, 1, 1, -1, -1]
    renamed = [7, 7, 3, 3, -1, -1]
    split = [7, 8, 3, 3, -1, -1]

    assert partition_change_fraction(reference, renamed) == 0.0
    assert partition_change_fraction(reference, split) == pytest.approx(1 / 6)
