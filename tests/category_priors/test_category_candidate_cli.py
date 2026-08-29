from __future__ import annotations

import pytest

from category_priors.category_candidate_evaluation import evaluate_category_candidates
from category_priors.cli import build_parser
from category_priors.taxonomy import load_taxonomy


def test_candidate_formation_commands_have_strict_gt_boundary() -> None:
    parser = build_parser()
    repair = parser.parse_args(
        [
            "repair-category-candidates",
            "--runtime-manifest",
            "runtime.json",
            "--output-root",
            "runs",
            "--category-priors",
            "priors.json",
            "--scene",
            "scene0645_00",
            "--sample-cap",
            "10000",
        ]
    )
    diagnose = parser.parse_args(
        [
            "diagnose-category-candidates",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--trace-output",
            "trace.parquet",
            "--analysis-output",
            "root.json",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate-category-candidates",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--metrics-output",
            "metrics.parquet",
            "--analysis-output",
            "analysis.json",
            "--phase",
            "dev2",
        ]
    )

    assert repair.command == "repair-category-candidates"
    assert repair.sample_cap == 10000
    assert not hasattr(repair, "gt_dir")
    assert diagnose.command == "diagnose-category-candidates"
    assert evaluate.phase == "dev2"
    assert evaluate.selected_condition is None

    dev8 = parser.parse_args(
        [
            "evaluate-category-candidates",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--metrics-output",
            "metrics.parquet",
            "--analysis-output",
            "analysis.json",
            "--phase",
            "dev8",
            "--selected-condition",
            "C2-raw-anchored-envelope",
        ]
    )
    assert dev8.selected_condition == "C2-raw-anchored-envelope"

    representation = parser.parse_args(
        [
            "diagnose-candidate-representation",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--metrics-output",
            "representation.parquet",
            "--analysis-output",
            "representation.json",
        ]
    )
    assert representation.command == "diagnose-candidate-representation"

    replay = parser.parse_args(
        [
            "replay-category-candidates",
            "--runtime-manifest",
            "runtime.json",
            "--bank-root",
            "bank",
            "--output-root",
            "runs",
            "--category-priors",
            "priors.json",
            "--scene",
            "scene0645_00",
            "--mode",
            "uniform",
            "--score-threshold",
            "0.15",
        ]
    )
    assert replay.command == "replay-category-candidates"
    assert replay.score_threshold == 0.15


def test_candidate_evaluation_phase_is_registered() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "evaluate-category-candidates",
                "--runtime-manifest",
                "runtime.json",
                "--gt-dir",
                "gt",
                "--run-root",
                "runs",
                "--scene",
                "scene0645_00",
                "--metrics-output",
                "metrics.parquet",
                "--analysis-output",
                "analysis.json",
                "--phase",
                "final",
            ]
        )


def test_dev8_requires_the_dev2_frozen_repair_condition(tmp_path) -> None:
    common = {
        "runtime_manifest": tmp_path / "runtime.json",
        "gt_dir": tmp_path / "gt",
        "run_root": tmp_path / "runs",
        "scene_ids": ("scene0645_00",),
        "taxonomy": load_taxonomy(),
        "metrics_output": tmp_path / "metrics.parquet",
        "analysis_output": tmp_path / "analysis.json",
    }
    with pytest.raises(ValueError, match="dev8 requires"):
        evaluate_category_candidates(**common, phase="dev8")
    with pytest.raises(ValueError, match="must be omitted"):
        evaluate_category_candidates(
            **common,
            phase="dev2",
            selected_condition="C1-consistent-envelope",
        )
