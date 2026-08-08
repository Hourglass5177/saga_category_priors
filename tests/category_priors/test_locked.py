from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import category_priors.runner as runner_module
from category_priors.io import hash_json, write_json
from category_priors.evaluator import PROTOCOL_VERSION
from category_priors.locked import (
    assess_seed_sensitivity,
    build_locked_plan,
    expand_locked_runs,
)
from category_priors.locked_evaluation import evaluate_locked_plan
from category_priors.mapping import DEFAULT_MAPPING_CONFIG, REGISTERED_CONDITIONS
from category_priors.runner import execute_locked_plan
from category_priors.taxonomy import load_taxonomy


def seed_rows(values: dict[tuple[str, int], float]) -> list[dict[str, object]]:
    return [
        {
            "split": "val-tune",
            "protocol_version": PROTOCOL_VERSION,
            "condition": condition,
            "run_seed": seed,
            "map_50_95": value,
        }
        for (condition, seed), value in values.items()
    ]


def test_seed_sensitivity_selects_one_seed_only_when_both_rules_pass() -> None:
    stable = seed_rows(
        {
            ("P000-B2", 42): 0.0400,
            ("P000-B2", 3407): 0.0405,
            ("P000-B2", 20260804): 0.0410,
            ("P111-combined", 42): 0.0440,
            ("P111-combined", 3407): 0.0445,
            ("P111-combined", 20260804): 0.0450,
        }
    )
    assert assess_seed_sensitivity(stable)["selected_locked_seeds"] == [42]

    reversed_once = [dict(row) for row in stable]
    for row in reversed_once:
        if row["condition"] == "P111-combined" and row["run_seed"] == 3407:
            row["map_50_95"] = 0.039
    assert assess_seed_sensitivity(reversed_once)["selected_locked_seeds"] == [
        42,
        3407,
        20260804,
    ]


def make_plan(tmp_path, seeds=(42,)) -> tuple[dict[str, object], object]:
    locked = tmp_path / "locked.json"
    write_json(
        locked,
        {
            "kind": "locked_evaluation_scenes",
            "split": "val-locked",
            "scenes": [
                {
                    "scene_id": f"scene{index:04d}_00",
                    "physical_scene_id": f"scene{index:04d}",
                }
                for index in range(48)
            ],
        },
    )
    mapping = {
        **DEFAULT_MAPPING_CONFIG,
        "provenance": {"tuning_split": "val-tune"},
    }
    mapping["content_sha256"] = hash_json(mapping)
    mapping_path = tmp_path / "mapping.json"
    write_json(mapping_path, mapping)
    priors = tmp_path / "priors.json"
    prior_payload = {
        "schema_version": "1.0",
        "kind": "category_priors",
        "provenance": {
            "datasets": ["scannet200"],
            "splits": ["train"],
            "row_count": 20,
        },
        "normalization": {"units": "meters"},
        "fit_config": {"min_physical_scenes": 5},
        "categories": {f"class-{index}": {} for index in range(20)},
        "fallback": {"unknown": "legacy_global"},
    }
    prior_payload["content_sha256"] = hash_json(prior_payload)
    write_json(priors, prior_payload)
    plan = build_locked_plan(locked, priors, mapping_path, None, "abc123", seeds)
    return plan, priors


def make_runtime(tmp_path: Path) -> tuple[Path, Path]:
    scene_manifest = tmp_path / "runtime.json"
    write_json(
        scene_manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": f"scene{index:04d}_00",
                    "base_path": str(tmp_path / f"scene{index:04d}_00"),
                    "scene_scale_m_per_unit": 1.0,
                }
                for index in range(48)
            ],
        },
    )
    pipeline = tmp_path / "run_pipeline.sh"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return scene_manifest, pipeline


def successful_fake_runner(calls: list[list[str]]):
    def run(command, **_kwargs):
        command = [str(value) for value in command]
        if command and command[0] == "git":
            return SimpleNamespace(returncode=1, stdout="")
        calls.append(command)
        output = Path(command[command.index("--json-path") + 1])
        metadata = Path(command[command.index("--prior-metadata-path") + 1])
        cache = Path(command[command.index("--max-contributor-cache-path") + 1])
        write_json(output, {"point_labels": [0], "instances": {}})
        write_json(metadata, {"kind": "saga_instance_metadata", "instances": {}})
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "reproducible.bin").write_bytes(b"cache")
        return SimpleNamespace(returncode=0)

    return run


def test_real_git_commit_takes_precedence_over_stale_export_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".category_priors_commit").write_text(
        "stale-marker\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="real-git-head\n"
        ),
    )
    assert runner_module._deployed_code_commit(tmp_path) == "real-git-head"


def test_locked_plan_expands_a_seeded_complete_block_design(tmp_path) -> None:
    plan, _ = make_plan(tmp_path)
    assert "content_sha256" not in plan
    assert Path(plan["inputs"]["category_priors"]).is_absolute()
    assert isinstance(plan["priors"]["categories"], dict)
    assert plan["conditions"] == list(REGISTERED_CONDITIONS)
    first = expand_locked_runs(plan)
    second = expand_locked_runs(plan)
    assert first == second
    assert len(first) == 48 * 12
    assert len({row["run_id"] for row in first}) == len(first)
    for scene_id in {row["scene_id"] for row in first}:
        assert {
            row["condition"] for row in first if row["scene_id"] == scene_id
        } == set(REGISTERED_CONDITIONS)


def test_run_locked_dry_run_uses_lightweight_progress(tmp_path) -> None:
    plan, priors = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    scene_manifest, pipeline = make_runtime(tmp_path)
    progress = tmp_path / "progress.json"
    result = execute_locked_plan(
        plan_path,
        scene_manifest,
        tmp_path / "runs",
        progress,
        pipeline,
        priors,
        tmp_path / "mapping.json",
        dry_run=True,
        max_runs=2,
    )
    assert result["planned"] == 2
    assert len(result["runs"]) == 2
    assert "content_sha256" not in result
    assert all("--minimal-metadata" in row["command"] for row in result["runs"])
    json.loads(progress.read_text(encoding="utf-8"))


def test_run_locked_resume_corrupt_output_and_progress_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    scene_manifest, pipeline = make_runtime(tmp_path)
    (tmp_path / ".category_priors_commit").write_text("abc123\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", successful_fake_runner(calls)
    )
    output_root = tmp_path / "runs"
    progress = tmp_path / "progress.json"

    execute_locked_plan(
        plan_path, scene_manifest, output_root, progress, pipeline, max_runs=2
    )
    assert len(calls) == 2
    execute_locked_plan(
        plan_path, scene_manifest, output_root, progress, pipeline, max_runs=2
    )
    assert len(calls) == 2

    first_run = expand_locked_runs(plan)[0]
    first_output = (
        output_root
        / str(first_run["condition"])
        / str(first_run["scene_id"])
        / f"seed-{first_run['run_seed']}"
        / "output.json"
    )
    first_output.write_text("not json", encoding="utf-8")
    result = execute_locked_plan(
        plan_path, scene_manifest, output_root, progress, pipeline, max_runs=1
    )
    assert len(calls) == 3
    assert result["planned"] == 1
    assert len(result["runs"]) == 1


def test_run_locked_retries_once_and_retains_attempt_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    scene_manifest, pipeline = make_runtime(tmp_path)
    (tmp_path / ".category_priors_commit").write_text("abc123\n", encoding="utf-8")
    calls: list[list[str]] = []
    success = successful_fake_runner(calls)

    def fail_then_succeed(command, **kwargs):
        if str(command[0]) == "git":
            return SimpleNamespace(returncode=1, stdout="")
        if not calls:
            calls.append([str(value) for value in command])
            return SimpleNamespace(returncode=1)
        return success(command, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "run", fail_then_succeed)
    output_root = tmp_path / "runs"
    result = execute_locked_plan(
        plan_path,
        scene_manifest,
        output_root,
        tmp_path / "progress.json",
        pipeline,
        max_runs=1,
    )
    record = result["runs"][0]
    assert len(calls) == 2
    assert record["status"] == "complete"
    assert record["first_attempt_failed"] is True
    assert record["recovered"] is True
    run_dir = Path(record["log"]).parent
    assert (run_dir / "postprocess-attempt-1.log").is_file()
    assert (run_dir / "postprocess-attempt-2.log").is_file()


def test_run_locked_keeps_seeds_separate_and_evicts_complete_scene_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = make_plan(tmp_path, seeds=(42, 3407, 20260804))
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    scene_manifest, pipeline = make_runtime(tmp_path)
    (tmp_path / ".category_priors_commit").write_text("abc123\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", successful_fake_runner(calls)
    )
    output_root = tmp_path / "runs"
    execute_locked_plan(
        plan_path,
        scene_manifest,
        output_root,
        tmp_path / "progress.json",
        pipeline,
        max_runs=13,
    )
    assert list(output_root.glob("*/scene0000_00/seed-42/output.json"))
    assert list(output_root.glob("*/scene0000_00/seed-3407/output.json"))
    assert not list(output_root.glob("*/scene0000_00/seed-20260804/output.json"))

    # Finish the full three-seed block for the first scene. The reusable cache
    # is then released immediately even though the remaining 47 scenes are not run.
    execute_locked_plan(
        plan_path,
        scene_manifest,
        output_root,
        tmp_path / "progress.json",
        pipeline,
        max_runs=36,
    )
    assert not (output_root / ".cache" / "max_contributors" / "scene0000_00").exists()


def test_seed_sensitivity_rejects_non_tune_rows() -> None:
    with pytest.raises(ValueError, match="val-tune"):
        assess_seed_sensitivity(
            [
                {
                    "split": "val-locked",
                    "protocol_version": PROTOCOL_VERSION,
                    "condition": "P000-B2",
                    "run_seed": 42,
                    "map_50_95": 0.1,
                }
            ]
        )


def test_locked_evaluation_rejects_taxonomy_outside_plan(tmp_path: Path) -> None:
    plan, _ = make_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)
    wrong_taxonomy = replace(load_taxonomy(), benchmark_name="different-benchmark")
    with pytest.raises(ValueError, match="taxonomy differs"):
        evaluate_locked_plan(
            plan_path,
            tmp_path / "runtime.json",
            tmp_path / "gt",
            tmp_path / "runs",
            wrong_taxonomy,
            tmp_path / "metrics.parquet",
            tmp_path / "analysis.json",
        )
