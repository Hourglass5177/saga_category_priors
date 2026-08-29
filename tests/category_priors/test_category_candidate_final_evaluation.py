from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import category_priors.category_candidate_final_evaluation as final_eval


def _scene_row(
    scene_id: str,
    *,
    map_value: float,
    ap50: float,
    instances: int,
    precision: float,
    coverage: float,
    tiny_count: int = 10,
    tiny_hits: int = 2,
    tp: int = 5,
    fp: int = 2,
) -> dict:
    return {
        "scene_id": scene_id,
        "map_50_95": map_value,
        "map_0.50": ap50,
        "predicted_instance_count": instances,
        "gaussian_micro_precision": precision,
        "prediction_coverage": coverage,
        "gt_recall": 0.5,
        "tiny_small_gt_count": tiny_count,
        "tiny_small_match_050_count": tiny_hits,
        "tiny_small_recall_050": tiny_hits / tiny_count if tiny_count else 0.0,
        "true_positive_count": tp,
        "false_positive_count": fp,
    }


def _analysis(rows_by_condition: dict[str, list[dict]]) -> dict:
    return {
        "conditions": {
            condition: {
                "metrics": {
                    key: sum(float(row[key]) for row in rows) / len(rows)
                    for key in (
                        "map_50_95",
                        "map_0.50",
                        "predicted_instance_count",
                        "gaussian_micro_precision",
                        "prediction_coverage",
                        "gt_recall",
                    )
                },
                "per_scene": rows,
            }
            for condition, rows in rows_by_condition.items()
        }
    }


def test_physical_scene_equal_metrics_average_scans_before_scenes() -> None:
    analysis = _analysis(
        {
            "U": [
                _scene_row(
                    "scene0001_00",
                    map_value=0.1,
                    ap50=0.2,
                    instances=10,
                    precision=0.7,
                    coverage=0.8,
                    tiny_count=2,
                    tiny_hits=2,
                ),
                _scene_row(
                    "scene0001_01",
                    map_value=0.3,
                    ap50=0.4,
                    instances=14,
                    precision=0.9,
                    coverage=1.0,
                    tiny_count=8,
                    tiny_hits=0,
                ),
                _scene_row(
                    "scene0002_00",
                    map_value=0.6,
                    ap50=0.7,
                    instances=20,
                    precision=0.5,
                    coverage=0.6,
                    tiny_count=5,
                    tiny_hits=5,
                ),
            ]
        }
    )

    result = final_eval.physical_scene_equal_condition_metrics(
        analysis, condition="U"
    )

    assert result["physical_scene_count"] == 2
    by_id = {
        row["physical_scene_id"]: row for row in result["per_physical_scene"]
    }
    assert by_id["scene0001"]["map_50_95"] == pytest.approx(0.2)
    # Repeated scans are averaged before physical scenes: (1.0 + 0.0) / 2.
    assert by_id["scene0001"]["tiny_small_recall_050"] == pytest.approx(0.5)
    assert result["macro"]["map_50_95"] == pytest.approx(0.4)
    assert result["macro"]["tiny_small_recall_050"] == pytest.approx(0.75)


def test_candidate_survival_reports_a_difference_erased_by_filter() -> None:
    rows = {
        "U": [
            {
                "scene_id": "scene0001_00",
                "candidate_id": 2,
                "accepted": False,
                "pre_knn_owned_count": 0,
                "post_knn_total_count": 0,
                "post_filter_total_count": 0,
                "_final_point_indices": np.asarray([], dtype=np.int64),
            }
        ],
        "D": [
            {
                "scene_id": "scene0001_00",
                "candidate_id": 2,
                "accepted": True,
                "pre_knn_owned_count": 4,
                "post_knn_total_count": 1,
                "post_filter_total_count": 0,
                "_final_point_indices": np.asarray([], dtype=np.int64),
            }
        ],
    }

    result = final_eval.candidate_survival_intervention(
        rows, uniform_condition="U", data_condition="D"
    )

    assert result["pre_knn_difference_count"] == 1
    assert result["post_knn_difference_count"] == 1
    assert result["post_filter_difference_count"] == 0
    assert result["difference_erased_by_legacy_knn_filter"] is True
    assert result["no_mechanical_intervention"] is False


def test_equal_post_filter_counts_with_different_members_are_not_erased() -> None:
    rows = {
        "U": [
            {
                "scene_id": "scene0001_00",
                "candidate_id": 2,
                "accepted": True,
                "pre_knn_owned_count": 2,
                "post_knn_total_count": 2,
                "post_filter_total_count": 2,
                "_final_point_indices": np.asarray([0, 1], dtype=np.int64),
            }
        ],
        "D": [
            {
                "scene_id": "scene0001_00",
                "candidate_id": 2,
                "accepted": True,
                "pre_knn_owned_count": 3,
                "post_knn_total_count": 2,
                "post_filter_total_count": 2,
                "_final_point_indices": np.asarray([2, 3], dtype=np.int64),
            }
        ],
    }

    result = final_eval.candidate_survival_intervention(
        rows, uniform_condition="U", data_condition="D"
    )

    assert result["pre_knn_difference_count"] == 1
    assert result["post_filter_count_difference_count"] == 0
    assert result["post_filter_difference_count"] == 1
    assert result["post_filter_difference_basis"] == (
        "exact_exported_point_membership"
    )
    assert result["difference_erased_by_legacy_knn_filter"] is False
    assert result["per_candidate"][0]["post_filter_count_changed"] is False
    assert result["per_candidate"][0]["post_filter_changed"] is True


def test_survival_loader_reads_the_runtime_metadata_contract(tmp_path: Path) -> None:
    scene = "scene0001_00"
    for condition in ("uniform", "class"):
        root = tmp_path / condition / scene
        root.mkdir(parents=True)
        (root / "output.json").write_text(
            '{"point_labels": [-1, 0, 0], "instances": '
            '{"0": {"class": "chair", "score": 0.5}}}',
            encoding="utf-8",
        )
        (root / "diagnostics.json").write_text(
            '{"category_denoise": {"candidate_survival": '
            '[{"candidate_id": 3, "accepted": true, '
            '"post_filter_total_count": 2, "final_instance_id": 0}]}}',
            encoding="utf-8",
        )

    rows = final_eval._load_survival_rows(
        tmp_path, ("uniform", "class"), (scene,)
    )
    assert len(rows["uniform"]) == 1
    row = rows["uniform"][0]
    assert {
        key: value for key, value in row.items() if key != "_final_point_indices"
    } == {
        "scene_id": scene,
        "candidate_id": 3,
        "accepted": True,
        "post_filter_total_count": 2,
        "final_instance_id": 0,
    }
    assert np.array_equal(row["_final_point_indices"], np.asarray([1, 2]))


def _passing_dev8_analysis() -> dict:
    scene_ids = [f"scene{index:04d}_00" for index in range(8)]
    return _analysis(
        {
            "B0": [
                _scene_row(
                    scene,
                    map_value=0.1000,
                    ap50=0.2000,
                    instances=10,
                    precision=0.80,
                    coverage=0.80,
                )
                for scene in scene_ids
            ],
            "U": [
                _scene_row(
                    scene,
                    map_value=0.0995,
                    ap50=0.1985,
                    instances=12,
                    precision=0.79,
                    coverage=0.795,
                )
                for scene in scene_ids
            ],
            "D": [
                _scene_row(
                    scene,
                    map_value=0.1020 if index < 5 else 0.0990,
                    ap50=0.1990,
                    instances=12,
                    precision=0.785,
                    coverage=0.795,
                    tiny_hits=3,
                )
                for index, scene in enumerate(scene_ids)
            ],
        }
    )


def test_dev8_gate_combines_uniform_health_and_registered_data_gain() -> None:
    result = final_eval.evaluate_registered_gate(
        _passing_dev8_analysis(),
        stage="dev8",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
    )

    assert result["uniform_health"]["passed"]
    assert result["uniform_health"]["predicted_instance_ratio"] == pytest.approx(1.2)
    data = result["data_minus_uniform"]
    assert data["positive_physical_scene_count"] == 5
    assert data["delta_tiny_small_recall_050"] == pytest.approx(0.1)
    assert data["checks"]["registered_primary_gain"]
    assert data["passed"]
    assert result["passed"]


def test_uniform_health_failure_blocks_an_otherwise_positive_data_result() -> None:
    analysis = _passing_dev8_analysis()
    for row in analysis["conditions"]["U"]["per_scene"]:
        row["predicted_instance_count"] = 13

    result = final_eval.evaluate_registered_gate(
        analysis,
        stage="dev8",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
    )

    assert not result["uniform_health"]["checks"]["instance_count_at_most_1.25x"]
    assert not result["uniform_health"]["passed"]
    assert not result["passed"]


def test_holdout_tune_and_final_have_distinct_frozen_gates() -> None:
    scene_ids = [f"scene{index:04d}_00" for index in range(5)]
    base = [
        _scene_row(
            scene,
            map_value=0.1,
            ap50=0.2,
            instances=10,
            precision=0.8,
            coverage=0.8,
        )
        for scene in scene_ids
    ]
    data = [
        _scene_row(
            scene,
            map_value=0.11 if index < 3 else 0.095,
            ap50=0.2,
            instances=10,
            precision=0.8,
            coverage=0.8,
            tiny_hits=3,
        )
        for index, scene in enumerate(scene_ids)
    ]
    analysis = _analysis({"B0": deepcopy(base), "U": deepcopy(base), "D": data})

    holdout = final_eval.evaluate_registered_gate(
        analysis,
        stage="holdout",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
    )
    tune = final_eval.evaluate_registered_gate(
        analysis,
        stage="tune",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
    )
    final = final_eval.evaluate_registered_gate(
        analysis,
        stage="final",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
        final_bootstrap={
            "delta_map_50_95": 0.004,
            "paired_bootstrap_ci95": [0.001, 0.008],
            "samples": 10_000,
        },
    )

    assert holdout["data_minus_uniform"]["passed"]
    assert tune["data_minus_uniform"]["passed"]
    assert final["data_minus_uniform"]["passed"]

    failed_final = final_eval.evaluate_registered_gate(
        analysis,
        stage="final",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
        final_bootstrap={
            "delta_map_50_95": 0.004,
            "paired_bootstrap_ci95": [-0.001, 0.008],
            "samples": 10_000,
        },
    )
    assert not failed_final["data_minus_uniform"]["passed"]


def test_final_bootstrap_wrapper_forwards_registered_10000_sample_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_bootstrap(**kwargs):
        captured.update(kwargs)
        return {"samples": kwargs["samples"], "delta_map_50_95": 0.003}

    monkeypatch.setattr(
        final_eval, "paired_scannet_bootstrap_from_predictions", fake_bootstrap
    )
    taxonomy = object()

    result = final_eval.final_paired_scannet_bootstrap(
        runtime_manifest=Path("runtime.json"),
        gt_dir=Path("gt"),
        replay_root=Path("replays"),
        scene_ids=("scene0001_00", "scene0002_00"),
        uniform_condition="U",
        data_condition="D",
        taxonomy=taxonomy,  # type: ignore[arg-type]
    )

    assert result["samples"] == 10_000
    assert captured["prediction_root"] == Path("replays")
    assert captured["reference_condition"] == "U"
    assert captured["treatment_condition"] == "D"
    assert captured["scene_ids"] == ("scene0001_00", "scene0002_00")
    assert captured["taxonomy"] is taxonomy


def test_file_entrypoint_accepts_a_direct_b0_condition_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = "scene0001_00"
    b0_condition_root = tmp_path / "historical" / "B0"
    replay_root = tmp_path / "replay"
    (b0_condition_root / scene).mkdir(parents=True)
    (b0_condition_root / scene / "output.json").write_text("{}", encoding="utf-8")
    for condition in ("U", "D"):
        (replay_root / condition / scene).mkdir(parents=True)
        (replay_root / condition / scene / "output.json").write_text(
            "{}", encoding="utf-8"
        )
    base_row = _scene_row(
        scene,
        map_value=0.1,
        ap50=0.2,
        instances=10,
        precision=0.8,
        coverage=0.8,
        tiny_hits=2,
    )
    calls = []

    def fake_evaluator(**kwargs):
        calls.append(kwargs)
        condition_rows = {}
        for condition in kwargs["conditions"]:
            row = deepcopy(base_row)
            if condition == "D":
                row["map_50_95"] += 0.003
                row["tiny_small_match_050_count"] = 3
                row["tiny_small_recall_050"] = 0.3
            condition_rows[condition] = {
                "metrics": dict(row),
                "per_scene": [row],
            }
        return {"conditions": condition_rows}

    monkeypatch.setattr(final_eval, "evaluate_v9_predictions", fake_evaluator)
    monkeypatch.setattr(final_eval, "_add_prediction_coverage", lambda *a, **k: None)
    monkeypatch.setattr(final_eval, "_load_survival_rows", lambda *a, **k: {})
    monkeypatch.setattr(final_eval, "write_rows", lambda *a, **k: None)
    monkeypatch.setattr(final_eval, "write_json", lambda *a, **k: None)

    result = final_eval.evaluate_candidate_final_stage(
        runtime_manifest=tmp_path / "runtime.json",
        gt_dir=tmp_path / "gt",
        b0_root=b0_condition_root,
        replay_root=replay_root,
        scene_ids=(scene,),
        taxonomy=object(),  # type: ignore[arg-type]
        stage="dev8",
        output_dir=tmp_path / "out",
        b0_condition="B0",
        uniform_condition="U",
        data_condition="D",
    )

    assert calls[0]["prediction_root"] == b0_condition_root.parent
    assert calls[0]["conditions"] == ("B0",)
    assert result["schema"] == final_eval.SCHEMA
