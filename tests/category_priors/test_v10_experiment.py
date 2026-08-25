from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from category_priors.io import write_json
from category_priors.v10_experiment import (
    _analysis_payload,
    _registered_final_scenes,
    _run_lifting_subprocess,
    build_parser,
    run_v10_experiment,
)


def test_final_scene_registration_requires_the_locked_replacement(tmp_path: Path) -> None:
    scenes = tuple(
        "scene0019_01" if index == 19 else f"scene{index:04d}_00"
        for index in range(48)
    )
    runtime = tmp_path / "locked-runtime.json"
    selection = tmp_path / "locked-scenes.json"
    write_json(
        runtime,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "base_path": str(tmp_path / scene_id),
                    "scene_scale_m_per_unit": 1.0,
                }
                for scene_id in scenes
            ],
        },
    )
    write_json(
        selection,
        {"kind": "locked_evaluation_scenes", "scenes": list(scenes)},
    )
    assert _registered_final_scenes(runtime, selection) == scenes


def test_v10_experiment_parser_exposes_all_production_roots() -> None:
    args = build_parser().parse_args(
        [
            "--runtime-manifest", "tune.json",
            "--locked-runtime-manifest", "locked.json",
            "--locked-evaluation-scenes", "locked-scenes.json",
            "--workspace", "workspace",
            "--runs-root", "runs",
            "--artifacts-root", "artifacts",
            "--v9-artifacts-root", "v9-artifacts",
            "--v9-lifting-root", "v9-lifting",
            "--gt-dir", "gt",
            "--locked-gt-dir", "locked-gt",
            "--sam-checkpoint", "sam.pth",
            "--label-features", "labels.pt",
            "--size-bins", "sizes.json",
            "--category-priors", "priors.json",
            "--b1-fixed-prediction-root", "b1",
            "--git-commit", "consumer",
        ]
    )
    assert args.b1_fixed_condition == "T1-B1"
    assert args.git_commit == "consumer"


def test_analysis_payload_is_self_contained_and_includes_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_experiment as module

    artifacts = tmp_path / "artifacts"
    viewer = artifacts / "viewer"
    viewer.mkdir(parents=True)
    write_json(artifacts / "v10_v9_closeout.json", {"closed": True})
    write_json(
        artifacts / "v10_tune24_metrics.json",
        {"conditions": {"U000": {"metrics": {"map_50_95": 0.1}}}},
    )
    monkeypatch.setattr(module, "_resource_snapshot", lambda _path: {"ok": True})

    payload = _analysis_payload(
        result={"state": "stopped", "checkpoint": "stage1"},
        runs_root=tmp_path / "runs",
        artifacts_root=artifacts,
    )

    assert payload["state"] == "stopped"
    assert payload["resource_snapshot"] == {"ok": True}
    assert payload["artifacts"]["viewer"] == str(viewer.resolve())
    assert payload["tune24_metrics_summary"]["U000"]["map_50_95"] == 0.1


def test_lifting_subprocess_uses_scene_python_and_keeps_log_outside_target(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_experiment as module

    python_bin = tmp_path / "scene-env" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        module, "compatible_lifting_bank_is_complete", lambda *_args, **_kwargs: True
    )
    output_root = tmp_path / "lifting"
    _run_lifting_subprocess(
        scene_id="scene0000_00",
        scene={"python_bin": str(python_bin)},
        runtime_manifest=tmp_path / "runtime.json",
        output_root=output_root,
        sam_scene=tmp_path / "sam" / "scene0000_00",
        label_features=tmp_path / "labels.pt",
        workspace=tmp_path,
        git_commit="consumer",
    )

    assert calls[0][0][0] == str(python_bin.resolve())
    assert calls[0][1]["cwd"] == tmp_path
    assert (output_root / "_logs" / "scene0000_00.log").is_file()
    assert not (output_root / "scene0000_00" / "lifting_worker.log").exists()


def test_experiment_boundary_always_writes_analysis_on_early_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_experiment as module

    monkeypatch.setattr(
        module,
        "_run_v10_experiment_impl",
        lambda _args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(module, "_resource_snapshot", lambda _path: {"ok": True})
    args = argparse.Namespace(
        artifacts_root=tmp_path / "artifacts",
        runs_root=tmp_path / "runs",
        git_commit="consumer",
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_v10_experiment(args)

    analysis = (tmp_path / "artifacts" / "v10_analysis.json").read_text("utf-8")
    assert '"checkpoint": "experiment-boundary-exception"' in analysis
    assert '"error": "boom"' in analysis
