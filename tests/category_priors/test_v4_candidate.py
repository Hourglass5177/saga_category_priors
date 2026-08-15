from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from category_priors.v4_candidate import (
    V4CandidateConfig,
    nested_permutation,
    resolve_v4_candidate_parameters,
)
from category_priors.v4_candidate_runner import build_v4_candidate_command
from category_priors.v4_candidate_runner import _complete, v4_candidate_run_paths
from category_priors.v4_feature_control import (
    CONTROL_SCENES,
    build_v4_feature_control_command,
)
from category_priors import cli


def _priors() -> dict:
    node = {
        "shrunk": {
            "geometry": {
                "log_bbox_diag_m": {"q50": float(np.log(0.5))},
                "log_surface_area_m2": {"q50": float(np.log(0.2))},
            }
        }
    }
    return {"categories": {"cup": node}}


def test_v4_factors_are_separate_and_min_samples_is_fixed() -> None:
    kwargs = dict(
        priors=_priors(), class_name="cup", candidate_count=1000,
        surface_density=1000.0, mask_scales=[0.1, 0.3, 0.7],
        config=V4CandidateConfig(),
    )
    uniform = resolve_v4_candidate_parameters(mode="uniform", **kwargs)
    scale = resolve_v4_candidate_parameters(mode="class-scale", **kwargs)
    core = resolve_v4_candidate_parameters(mode="class-core", **kwargs)
    combined = resolve_v4_candidate_parameters(mode="combined", **kwargs)
    assert uniform["scale_gate_input"] == core["scale_gate_input"] == 1.0
    assert scale["scale_gate_input"] == combined["scale_gate_input"] == 2 / 3
    assert uniform["min_cluster_size"] == scale["min_cluster_size"] == 5
    assert core["min_cluster_size"] == combined["min_cluster_size"]
    assert core["min_cluster_size"] != 5
    assert {row["min_samples"] for row in (uniform, scale, core, combined)} == {3}
    assert "spatial_scale_m" not in combined


def test_v4_unknown_class_falls_back_to_uniform() -> None:
    params = resolve_v4_candidate_parameters(
        _priors(), "combined", "unknown", 500, 1000.0, [0.1, 0.2]
    )
    assert params["supported"] is False
    assert params["scale_gate_input"] == 1.0
    assert params["min_cluster_size"] == 5


def test_v4_sampling_is_nested_and_repeatable() -> None:
    first = nested_permutation(100, 42, "cup")
    second = nested_permutation(100, 42, "cup")
    assert np.array_equal(first, second)
    assert np.array_equal(first[:20], first[:40][:20])
    assert not np.array_equal(first, nested_permutation(100, 42, "chair"))


def test_v4_candidate_command_keeps_b1_and_shadow_outputs_isolated(tmp_path: Path) -> None:
    scene = {
        "base_path": "/data/scene",
        "python_bin": "/env/bin/python",
        "scene_scale_m_per_unit": 1.0,
    }
    command, paths = build_v4_candidate_command(
        pipeline=tmp_path / "run_pipeline.sh", scene=scene,
        output_root=tmp_path / "runs", mode="combined",
        scene_id="scene0011_00", seed=42, git_commit="abc",
        category_priors=tmp_path / "priors.json",
    )
    assert command[command.index("--teacher-prior-mode") + 1] == "original"
    assert "--v4-candidate-mode" in command
    assert str(paths["candidate_json"]) in command
    assert paths["candidate_json"] != paths["output"]


def test_v4_10k_control_paths_cannot_overwrite_scene_assets(tmp_path: Path) -> None:
    scene = {"base_path": "/data/scene", "python_bin": "/env/bin/python"}
    command, paths = build_v4_feature_control_command(
        tmp_path / "run_pipeline.sh", scene, CONTROL_SCENES[0], tmp_path / "control"
    )
    assert command[command.index("--feature-iterations") + 1] == "10000"
    assert str(paths["feature_ply"]) in command
    assert str(paths["scale_gate"]) in command
    assert "/data/scene/saga/contrastive_feature_point_cloud.ply" not in command


def test_v4_cli_forwards_feature_control_root(monkeypatch, capsys) -> None:
    captured = {}
    monkeypatch.setattr(
        cli, "execute_v4_candidate_runs",
        lambda **kwargs: captured.update(kwargs) or {"kind": "ok"},
    )
    cli.command_run_v4_candidates(Namespace(
        scene_manifest="manifest.json", output_root="runs", pipeline="pipeline.sh",
        git_commit="abc", category_priors="priors.json", scene=["scene0011_00"],
        mode=["uniform"], seed=[42], no_resume=False, continue_on_error=False,
        dry_run=True, max_runs=None, feature_control_root="control-assets",
    ))
    assert captured["feature_control_root"] == "control-assets"
    assert captured["modes"] == ["uniform"]


def test_v4_resume_requires_the_same_asset_command(tmp_path: Path) -> None:
    paths = v4_candidate_run_paths(tmp_path, "uniform", "scene0011_00", 42)
    paths["run_dir"].mkdir(parents=True)
    paths["output"].write_text('{"point_labels": [], "instances": {}}', encoding="utf-8")
    paths["candidate_json"].write_text(
        '{"kind": "v4_candidate_capture", "mode": "uniform"}', encoding="utf-8"
    )
    np.savez_compressed(paths["candidate_labels"], point_count=np.asarray([0]))
    paths["runner"].write_text(
        json.dumps({"kind": "v4_candidate_run", "command": ["2k-feature"]}),
        encoding="utf-8",
    )
    assert _complete(paths, "uniform", ["2k-feature"])
    assert not _complete(paths, "uniform", ["10k-feature"])
