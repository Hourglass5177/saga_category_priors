from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from category_priors.backbone_diagnostics import diagnose_mapped_instances
from category_priors.legacy_prior import (
    LegacyPriorConfig,
    empirical_scale_quantile,
    radius_vote_labels,
    rescue_halo,
    resolve_class_parameters,
)
from category_priors.legacy_prior_runner import build_legacy_prior_command


def _summary(value: float) -> dict[str, float]:
    return {"q05": value, "q25": value, "q50": value, "q75": value, "q95": value}


def _priors() -> dict:
    node = {
        "shrunk": {
            "geometry": {
                "log_bbox_diag_m": _summary(float(np.log(0.5))),
                "log_surface_area_m2": _summary(float(np.log(0.04))),
            },
            "neighborhood": {
                "boundary_fixed:0.02": _summary(0.01),
                "boundary_fixed:0.05": _summary(0.04),
                "boundary_fixed:0.10": _summary(0.08),
                "boundary_fixed:0.20": _summary(0.30),
            },
        }
    }
    return {"categories": {"cup": node}}


def test_prior_parameters_use_metric_density_and_independent_min_samples() -> None:
    config = LegacyPriorConfig(alpha=0.05, min_samples=3, boundary_beta=0.10)
    params = resolve_class_parameters(
        _priors(), config, "cup", "combined", 1000, 10000.0, [0.1, 0.5, 1.0]
    )
    assert params["min_samples"] == 3
    assert 3 <= params["min_cluster_size"] <= 20
    assert params["smoothing_radius_m"] == 0.10
    assert params["scale_gate_input"] == 2 / 3
    assert params["rescue_enabled"] is True


def test_unknown_class_is_strict_uniform() -> None:
    params = resolve_class_parameters(
        _priors(), LegacyPriorConfig(), "unknown", "combined", 100, 1000.0, []
    )
    assert params["supported"] is False
    assert params["min_cluster_size"] == 5
    assert params["spatial_scale_m"] is None
    assert params["smoothing_radius_m"] is None
    assert params["rescue_enabled"] is False


def test_radius_vote_ignores_noise_and_halo_requires_consensus() -> None:
    xyz = np.asarray([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [1, 0, 0]], dtype=float)
    labels = np.asarray([2, 2, 2, -1, -1])
    smoothed = radius_vote_labels(labels, xyz, 0.04, 8)
    assert smoothed[3] == 2
    recovered, count = rescue_halo(labels, xyz, 0.04, 4, 3)
    assert recovered[3] == 2 and recovered[4] == -1 and count == 1


def test_empirical_scale_quantile() -> None:
    assert empirical_scale_quantile([0.1, 0.2, 0.4, 1.0], 0.3) == 0.5


def test_runner_builds_isolated_legacy_prior_command(tmp_path: Path) -> None:
    pipeline = tmp_path / "run_pipeline.sh"
    pipeline.write_text("#!/bin/sh\n", encoding="utf-8")
    scene = {
        "base_path": str(tmp_path / "scene"),
        "python_bin": "/python",
        "scene_scale_m_per_unit": 1.0,
    }
    command, paths = build_legacy_prior_command(
        pipeline, scene, tmp_path / "runs", "D-small", "scene0000_00", 3407,
        tmp_path / "priors.json", tmp_path / "config.json",
    )
    assert "legacy-prior" in command and "small" in command
    assert paths["run_dir"].as_posix().endswith("D-small/scene0000_00/seed-3407")
    assert "max-contributor-cache" not in " ".join(command)


def test_backbone_diagnostic_detects_fragmentation_and_semantic_error() -> None:
    gt_semantic = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    gt_instance = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    mapped = np.asarray([0, 0, 1, 1, 2, 2, 2, -1])
    result = diagnose_mapped_instances(
        gt_semantic, gt_instance, mapped, {0: 0, 1: 0, 2: 0}, min_region_size=1
    )
    assert result["gt_instances"] == 2
    assert result["pred_instances"] == 3
    assert result["mean_split_count_at_010"] == 1.0
    assert result["semantic_accuracy_on_assigned_gt"] < 1.0
