from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import category_priors.baseline_closure_analysis as analysis_module
from category_priors.baseline_closure_analysis import (
    _b1_minus_b0,
    _output_runs,
    bfc18_saga20_intersection,
    build_parser,
)
from category_priors.evaluator import GroundTruthScene
from category_priors.teacher_prior import SAGA20_CLASSES


def test_bfc18_saga20_common10_is_registered_and_ordered() -> None:
    assert bfc18_saga20_intersection(SAGA20_CLASSES) == (
        "chair",
        "table",
        "plant",
        "tv",
        "painting",
        "sofa",
        "cabinet",
        "bed",
        "socket",
        "book",
    )


def test_b1_minus_b0_pairs_same_variant_budget_and_view() -> None:
    common = {
        "record_type": "condition",
        "variant_id": "full950",
        "budget": "adaptive",
        "score_mode": "unit",
        "protocol_key": "scannet_official_9",
        "class_view": "full_saga20",
        "scene_count": 3,
        "scene_ids": "a|b|c",
        "map_0.25": 0.2,
        "primary_score": 0.1,
    }
    rows = [
        {**common, "condition": "B0-global"},
        {**common, "condition": "B1-original", "map_0.25": 0.5, "primary_score": 0.3},
        {
            **common,
            "condition": "B1-original",
            "budget": "10000",
            "map_0.25": 0.9,
            "primary_score": 0.8,
        },
    ]
    deltas = _b1_minus_b0(rows)
    assert len(deltas) == 1
    assert deltas[0]["condition"] == "B1-original_minus_B0-global"
    assert deltas[0]["map_0.25"] == 0.3
    assert deltas[0]["primary_score"] == 0.19999999999999998


def test_cli_parser_accepts_minimal_evaluation_paths() -> None:
    args = build_parser().parse_args(
        [
            "--closure-root",
            "closure",
            "--gt-dir",
            "gt",
            "--runtime-manifest",
            "runtime.json",
            "--output-dir",
            "artifacts",
        ]
    )
    assert str(args.closure_root) == "closure"
    assert args.min_region_size == 100
    assert args.radius_m == 0.05


def test_structural_outputs_are_registered_without_conversion(tmp_path) -> None:
    target = (
        tmp_path
        / "outputs"
        / "bfc18"
        / "current-causal-harness"
        / "adaptive"
        / "L2-B1-original"
        / "scene0064_01"
        / "output.json"
    )
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    rows = list(_output_runs(tmp_path))
    assert (
        "current-L2",
        "adaptive",
        "B1-original",
        "scene0064_01",
        target,
    ) in rows


def test_analysis_artifact_exposes_full_orphan_projection_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_json = tmp_path / "output.json"
    output_json.write_text(
        json.dumps(
            {
                "point_labels": [4, 9],
                "instances": {"4": {"class": "chair"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        analysis_module,
        "_output_runs",
        lambda _root: iter(
            [("full950", "adaptive", "B1-original", "scene", output_json)]
        ),
    )
    monkeypatch.setattr(
        analysis_module, "_runtime_rows", lambda _path: {"scene": {}}
    )
    monkeypatch.setattr(
        analysis_module, "_gaussian_ply", lambda _row: tmp_path / "unused.ply"
    )
    monkeypatch.setattr(analysis_module, "_transform", lambda _row: np.eye(4))
    monkeypatch.setattr(analysis_module, "CLOSURE_SCENES", ())
    monkeypatch.setattr(
        analysis_module,
        "load_ground_truth_npz",
        lambda _path, scene_id: (
            np.zeros((2, 3)),
            GroundTruthScene(
                scene_id,
                semantic=np.asarray([0, 0]),
                instance=np.asarray([1, 1]),
            ),
        ),
    )
    monkeypatch.setattr(
        analysis_module,
        "saga_scene_predictions",
        lambda **_kwargs: ([], {}),
    )
    monkeypatch.setattr(
        analysis_module,
        "evaluate_baseline_closure",
        lambda *_args, **_kwargs: {"aggregate": {}},
    )
    monkeypatch.setattr(analysis_module, "_metric_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(analysis_module, "write_rows", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(analysis_module, "write_json", lambda *_args, **_kwargs: None)

    result = analysis_module.evaluate_teacher_handoff(
        closure_root=tmp_path / "closure",
        gt_dir=tmp_path / "gt",
        runtime_manifest=tmp_path / "runtime.json",
        output_dir=tmp_path / "artifacts",
    )

    projection = result["declared_instance_projection"]["runs"][0]
    assert projection["orphan_instance_ids"] == [9]
    assert projection["orphan_counts"] == {"9": 1}
    assert projection["declared_gaussian_count"] == 1
    assert projection["orphan_gaussian_fraction"] == 0.5
