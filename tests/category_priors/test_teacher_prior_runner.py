from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import category_priors.teacher_prior_runner as runner_module
from category_priors.cli import build_parser
from category_priors.io import load_json, write_json
from category_priors.teacher_prior_runner import (
    TEACHER_PRIOR_EXPERIMENT_CONDITIONS,
    build_teacher_prior_command,
    execute_teacher_prior_runs,
)


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


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    pipeline = tmp_path / "run_pipeline.sh"
    params = tmp_path / "teacher_category_params.json"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    write_json(params, {"kind": "category_priors"})
    return pipeline, params


def _successful_subprocess(calls: list[list[str]]):
    def fake_run(command, **kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--json-path") + 1])
        diagnostics = Path(command[command.index("--prior-metadata-path") + 1])
        write_json(output, {"point_labels": [0, 0], "instances": {"0": {}}})
        write_json(
            diagnostics,
            {
                "teacher_prior": {
                    "totals": {"changed_points": 2, "coverage": 1.0}
                }
            },
        )
        kwargs["stdout"].write("postprocess complete\n")
        return SimpleNamespace(returncode=0)

    return fake_run


def test_command_uses_shared_params_and_exact_mode(tmp_path) -> None:
    params = tmp_path / "teacher_category_params.json"
    scene = {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(tmp_path / "python"),
    }

    command, paths = build_teacher_prior_command(
        tmp_path / "run_pipeline.sh",
        scene,
        tmp_path / "runs",
        "U0-all-uniform",
        "scene0000_00",
        3407,
        params,
    )

    assert paths["run_dir"] == (
        tmp_path / "runs" / "U0-all-uniform" / "scene0000_00" / "seed-3407"
    )
    assert command[command.index("--teacher-prior-mode") + 1] == "all-uniform"
    assert command[command.index("--teacher-category-params") + 1] == str(
        params.resolve()
    )
    assert command[command.index("--json-path") + 1].endswith(
        "output.pending.json"
    )
    assert "--teacher-branch-preservation" not in command
    assert not any("schedule" in value or "cache" in value for value in command)


def test_multi_anchor_conditions_select_same_modes_with_one_structure_flag(tmp_path) -> None:
    scene = {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(tmp_path / "python"),
    }
    expected = {
        "U0-multi-anchor": "all-uniform",
        "D-combined-multi-anchor": "combined",
    }
    for condition, mode in expected.items():
        command, paths = build_teacher_prior_command(
            tmp_path / "run_pipeline.sh", scene, tmp_path / "runs",
            condition, "scene0000_00", 42,
            tmp_path / "teacher_category_params.json",
        )
        assert command[command.index("--teacher-prior-mode") + 1] == mode
        assert command[command.index("--teacher-evidence-protection") + 1] == "multi-anchor"
        assert paths["run_dir"].parts[-3:] == (condition, "scene0000_00", "seed-42")


def test_off_and_original_never_receive_category_params(tmp_path) -> None:
    scene = {
        "base_path": str(tmp_path / "scene"),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(tmp_path / "python"),
    }

    for condition in ("off", "original"):
        command, _ = build_teacher_prior_command(
            tmp_path / "run_pipeline.sh",
            scene,
            tmp_path / "runs",
            condition,
            "scene0000_00",
            42,
            tmp_path / "unused-params.json",
        )
        assert command[command.index("--teacher-prior-mode") + 1] == condition
        assert "--teacher-category-params" not in command


def test_execute_resumes_complete_run_and_reruns_corrupt_output(
    tmp_path, monkeypatch
) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, params = _inputs(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", _successful_subprocess(calls)
    )

    first = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        params,
        conditions=["D-small"],
        seeds=[42],
    )
    second = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        params,
        conditions=["D-small"],
        seeds=[42],
    )

    run_dir = tmp_path / "runs" / "D-small" / "scene0000_00" / "seed-42"
    diagnostics = load_json(run_dir / "diagnostics.json")
    assert first["complete"] == 1
    assert second["skipped"] == 1
    assert len(calls) == 1
    assert diagnostics["teacher_prior"]["totals"]["changed_points"] == 2
    assert diagnostics["run"]["teacher_prior_mode"] == "small"
    assert diagnostics["runner"]["point_count"] == 2
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "diagnostics.json",
        "output.json",
        "postprocess.log",
    ]

    (run_dir / "diagnostics.json").unlink()
    output_only_resume = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        params,
        conditions=["D-small"],
        seeds=[42],
    )
    assert output_only_resume["skipped"] == 1
    assert len(calls) == 1

    write_json(run_dir / "output.json", {})
    rerun = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        params,
        conditions=["D-small"],
        seeds=[42],
    )
    assert rerun["complete"] == 1
    assert len(calls) == 2
    assert load_json(run_dir / "output.json")["point_labels"] == [0, 0]


def test_default_dry_run_is_five_experimental_conditions(tmp_path) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, params = _inputs(tmp_path)

    result = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        params,
        seeds=[42, 3407],
        dry_run=True,
    )

    assert result["planned"] == 2 * len(TEACHER_PRIOR_EXPERIMENT_CONDITIONS)
    assert {item["condition"] for item in result["runs"]} == set(
        TEACHER_PRIOR_EXPERIMENT_CONDITIONS
    )
    assert not (tmp_path / "runs").exists()


def test_cli_exposes_runner_and_evaluator_without_branch_flag() -> None:
    parser = build_parser()
    build = parser.parse_args(
        [
            "build-teacher-category-params",
            "--category-priors",
            "priors.json",
            "--output",
            "params.json",
            "--branch-preservation",
            "--restore-after-global-filter",
        ]
    )
    assert build.branch_preservation is True
    assert build.restore_after_global_filter is True
    args = parser.parse_args(
        [
            "run-teacher-prior",
            "--scene-manifest",
            "runtime.json",
            "--output-root",
            "runs",
            "--teacher-category-params",
            "params.json",
            "--condition",
            "U0-all-uniform",
        ]
    )

    assert args.condition == ["U0-all-uniform"]
    assert not hasattr(args, "teacher_branch_preservation")
    evaluation = parser.parse_args(
        [
            "evaluate-teacher-prior",
            "--scene-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--output-root",
            "runs",
            "--scene",
            "scene0000_00",
            "--reference",
            "U0-all-uniform",
            "--treatment",
            "D-small",
        ]
    )
    assert evaluation.selection_split == "tune"
