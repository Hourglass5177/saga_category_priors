from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import category_priors.class_first_runner as runner_module
from category_priors.class_first_runner import (
    CLASS_FIRST_CONDITIONS,
    build_class_first_command,
    execute_class_first_runs,
)
from category_priors.io import load_json, write_json


def _scene_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "scene_runtime.json"
    write_json(
        path,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0000_00",
                    "physical_scene_id": "scene0000",
                    "base_path": "assets/scene0000_00",
                    "scene_scale_m_per_unit": 1.25,
                    "python_bin": "env/bin/python",
                }
            ],
        },
    )
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    pipeline = tmp_path / "run_pipeline.sh"
    priors = tmp_path / "category_priors.json"
    config = tmp_path / "class_first_params.json"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    write_json(priors, {"kind": "category_priors"})
    write_json(config, {"kind": "class_first_config"})
    return pipeline, priors, config


def _successful_subprocess(calls: list[list[str]]):
    def fake_run(command, **kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--json-path") + 1])
        diagnostics = Path(command[command.index("--prior-metadata-path") + 1])
        write_json(output, {"point_labels": [0, 0], "instances": {"0": {}}})
        write_json(
            diagnostics,
            {"kind": "class_first_scores", "scores": {"accepted": 1}},
        )
        kwargs["stdout"].write("postprocess complete\n")
        return SimpleNamespace(returncode=0)

    return fake_run


def test_command_maps_conditions_and_uses_simple_run_paths(tmp_path) -> None:
    command, paths = build_class_first_command(
        tmp_path / "run_pipeline.sh",
        {
            "base_path": str(tmp_path / "scene"),
            "scene_scale_m_per_unit": 1.0,
            "python_bin": str(tmp_path / "python"),
        },
        tmp_path / "runs",
        "D-small",
        "scene0000_00",
        3407,
        tmp_path / "priors.json",
        tmp_path / "config.json",
    )

    assert paths["run_dir"] == (
        tmp_path / "runs" / "D-small" / "scene0000_00" / "seed-3407"
    )
    assert command[command.index("--clustering-mode") + 1] == "class-first"
    assert command[command.index("--class-prior-mode") + 1] == "small"
    assert command[command.index("--json-path") + 1].endswith(
        "output.pending.json"
    )
    assert command[command.index("--prior-metadata-path") + 1].endswith(
        "diagnostics.pending.json"
    )
    assert "--minimal-metadata" in command
    assert "--max-contributor-cache-path" not in command
    assert "--repo-path" not in command


def test_execute_preserves_scores_and_resumes_by_run_identity(
    tmp_path, monkeypatch
) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, priors, config = _inputs(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", _successful_subprocess(calls)
    )

    first = execute_class_first_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        priors,
        config,
        conditions=["U0-uniform"],
        seeds=[42],
    )
    second = execute_class_first_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        priors,
        config,
        conditions=["U0-uniform"],
        seeds=[42],
    )

    run_dir = tmp_path / "runs" / "U0-uniform" / "scene0000_00" / "seed-42"
    diagnostics = load_json(run_dir / "diagnostics.json")
    assert first["complete"] == 1
    assert second["skipped"] == 1
    assert len(calls) == 1
    assert diagnostics["scores"] == {"accepted": 1}
    assert diagnostics["status"] == "complete"
    assert diagnostics["run"]["scene_id"] == "scene0000_00"
    assert diagnostics["run"]["condition"] == "U0-uniform"
    assert diagnostics["run"]["seed"] == 42
    assert (run_dir / "output.json").is_file()
    assert (run_dir / "postprocess.log").read_text(encoding="utf-8") == (
        "postprocess complete\n"
    )


def test_corrupt_output_is_rerun_even_with_matching_diagnostics(
    tmp_path, monkeypatch
) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, priors, config = _inputs(tmp_path)
    run_dir = tmp_path / "runs" / "D-size" / "scene0000_00" / "seed-42"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "output.json", {})
    write_json(
        run_dir / "diagnostics.json",
        {
            "status": "complete",
            "run": {
                "scene_id": "scene0000_00",
                "condition": "D-size",
                "seed": 42,
            },
        },
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", _successful_subprocess(calls)
    )

    result = execute_class_first_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        priors,
        config,
        conditions=["D-size"],
        seeds=[42],
    )

    assert result["complete"] == 1
    assert len(calls) == 1
    assert load_json(run_dir / "output.json")["point_labels"] == [0, 0]


def test_dry_run_expands_manifest_without_schedule_or_files(tmp_path) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, priors, config = _inputs(tmp_path)
    output_root = tmp_path / "runs"

    result = execute_class_first_runs(
        manifest,
        output_root,
        pipeline,
        priors,
        config,
        seeds=[42, 3407],
        dry_run=True,
    )

    assert result["planned"] == 2 * len(CLASS_FIRST_CONDITIONS)
    assert all(item["status"] == "planned" for item in result["runs"])
    assert not output_root.exists()
    run_dirs = {item["run_dir"] for item in result["runs"]}
    assert len(run_dirs) == result["planned"]
