from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from category_priors import category_fragment_merge_experiment as experiment
from category_priors.category_fragment_merge_experiment import (
    DEV2,
    DEV8,
    EXPECTED_CGROUP_MAX_BYTES,
    FragmentMergeExperimentConfig,
    FragmentMergeExperimentHooks,
    check_fragment_merge_resources,
    run_category_fragment_merge_experiment,
)
from category_priors.io import load_json


def _config(
    tmp_path: Path, *, runtime_name: str = "runtime.json"
) -> FragmentMergeExperimentConfig:
    return FragmentMergeExperimentConfig(
        runtime_manifest=tmp_path / runtime_name,
        gt_dir=tmp_path / "gt",
        category_priors=tmp_path / "priors.json",
        output_root=tmp_path / "output",
        size_bins=tmp_path / "size-bins.json",
        taxonomy=tmp_path / "taxonomy.json",
    )


def _hooks(
    calls: list[tuple],
    *,
    dev2_passed: bool = True,
    dev2_graph_passed: bool = True,
    dev2_raw_identity_passed: bool | None = None,
    dev8_passed: bool = True,
    fail_once_at: str | None = None,
) -> FragmentMergeExperimentHooks:
    failed: set[str] = set()

    def resources(root: Path):
        calls.append(("resources", Path(root)))
        return {
            "disk_available_gib": 100.0,
            "memory_current_bytes": 1,
            "memory_max_bytes": EXPECTED_CGROUP_MAX_BYTES,
            "memory_events": "oom 0",
            "host_free_used": False,
        }

    def maybe_fail(name: str) -> None:
        if fail_once_at == name and name not in failed:
            failed.add(name)
            raise RuntimeError(f"simulated {name}")

    def build(**kwargs):
        phase = str(kwargs["phase"])
        maybe_fail(f"{phase}_build")
        scenes = tuple(kwargs["scene_ids"])
        calls.append((f"{phase}_build", scenes))
        return {
            "status": "complete",
            "stage": "build-fragment-graph",
            "scene_ids": list(scenes),
        }

    def merge(**kwargs):
        phase = str(kwargs["phase"])
        maybe_fail(f"{phase}_merge")
        scenes = tuple(kwargs["scene_ids"])
        calls.append((f"{phase}_merge", scenes))
        return {
            "status": "complete",
            "stage": "merge-fragment-graph",
            "scene_ids": list(scenes),
        }

    def evaluate(**kwargs):
        phase = str(kwargs["phase"])
        maybe_fail(f"{phase}_evaluate")
        scenes = tuple(kwargs["scene_ids"])
        calls.append((f"{phase}_evaluate", scenes))
        passed = dev2_passed if phase == "dev2" else dev8_passed
        result = {
            "schema": "saga-category-fragment-merge-evaluation-v1",
            "phase": phase,
            "scene_ids": list(scenes),
            "passed": passed,
            "conclusion": f"{phase}-{'passed' if passed else 'failed'}",
        }
        if phase == "dev2":
            result["graph_passed"] = dev2_graph_passed
            if dev2_raw_identity_passed is not None:
                result["raw_fragment_identity"] = {
                    "passed": dev2_raw_identity_passed
                }
        return result

    return FragmentMergeExperimentHooks(
        check_resources=resources,
        build_graphs=build,
        merge_graphs=merge,
        evaluate=evaluate,
    )


def _functional(calls: list[tuple]) -> list[tuple]:
    return [row for row in calls if row[0] != "resources"]


def test_dev2_pass_automatically_runs_fixed_dev8_and_completes(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_fragment_merge_experiment(config, _hooks(calls))

    assert result["status"] == "complete"
    assert result["checkpoint"] == "dev8_evaluated"
    assert result["dev2_passed"] is True
    assert result["dev8_passed"] is True
    assert result["category_prior_tested"] is True
    assert result["category_prior_replayed"] is True
    assert result["category_prior_evaluable"] is True
    assert _functional(calls) == [
        ("dev2_build", DEV2),
        ("dev2_merge", DEV2),
        ("dev2_evaluate", DEV2),
        ("dev8_build", DEV8),
        ("dev8_merge", DEV8),
        ("dev8_evaluate", DEV8),
    ]
    assert len([row for row in calls if row[0] == "resources"]) == 6
    assert load_json(config.state_path) == result


def test_dev2_gate_failure_stops_without_any_dev8_work(tmp_path: Path) -> None:
    calls: list[tuple] = []

    result = run_category_fragment_merge_experiment(
        _config(tmp_path), _hooks(calls, dev2_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "dev2_gate_failed"
    assert result["dev2_passed"] is False
    assert result["dev8_passed"] is None
    assert result["stop_reason"] == "dev2-failed"
    assert not any(row[0].startswith("dev8") for row in calls)


def test_dev2_graph_failure_records_replay_but_not_prior_test(tmp_path: Path) -> None:
    calls: list[tuple] = []

    result = run_category_fragment_merge_experiment(
        _config(tmp_path),
        _hooks(calls, dev2_passed=False, dev2_graph_passed=False),
    )

    assert result["status"] == "stopped"
    assert result["category_prior_replayed"] is True
    assert result["category_prior_evaluable"] is False
    assert result["category_prior_tested"] is False
    assert not any(row[0].startswith("dev8") for row in calls)


def test_dev2_raw_identity_failure_is_not_marked_prior_evaluable(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []

    result = run_category_fragment_merge_experiment(
        _config(tmp_path),
        _hooks(
            calls,
            dev2_passed=False,
            dev2_graph_passed=True,
            dev2_raw_identity_passed=False,
        ),
    )

    assert result["status"] == "stopped"
    assert result["category_prior_replayed"] is True
    assert result["category_prior_evaluable"] is False
    assert result["category_prior_tested"] is False
    assert not any(row[0].startswith("dev8") for row in calls)


def test_dev8_failure_is_a_completed_negative_result_not_an_expansion(tmp_path: Path) -> None:
    calls: list[tuple] = []

    result = run_category_fragment_merge_experiment(
        _config(tmp_path), _hooks(calls, dev8_passed=False)
    )

    assert result["status"] == "complete"
    assert result["dev8_passed"] is False
    assert result["conclusion"] == "dev8-failed"
    assert result["identity"]["holdout_authorized"] is False
    assert result["identity"]["final_authorized"] is False


def test_interrupted_stage_records_error_and_resumes_at_that_stage(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls, fail_once_at="dev2_merge")

    with pytest.raises(RuntimeError, match="simulated dev2_merge"):
        run_category_fragment_merge_experiment(config, hooks)

    interrupted = load_json(config.state_path)
    assert interrupted["status"] == "error"
    assert interrupted["next_stage"] == "dev2_merge"
    assert interrupted["last_error"]["stage"] == "dev2_merge"
    assert "RuntimeError" in interrupted["last_error"]["traceback"]

    result = run_category_fragment_merge_experiment(config, hooks)

    assert result["status"] == "complete"
    functional = _functional(calls)
    assert sum(row[0] == "dev2_build" for row in functional) == 1
    assert sum(row[0] == "dev2_merge" for row in functional) == 1
    assert sum(row[0] == "dev2_evaluate" for row in functional) == 1


def test_existing_state_rejects_changed_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_category_fragment_merge_experiment(config, _hooks([]))

    with pytest.raises(ValueError, match="identity differs"):
        run_category_fragment_merge_experiment(
            _config(tmp_path, runtime_name="different.json"), _hooks([])
        )


def test_resource_guard_uses_only_df_and_cgroup_v2(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(
        f"{EXPECTED_CGROUP_MAX_BYTES}\n", encoding="utf-8"
    )
    (cgroup / "memory.events").write_text(
        "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n",
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(
            stdout=(
                "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                "/dev/test 500000000 1 200000000 1% /workspace\n"
            )
        )

    result = check_fragment_merge_resources(
        tmp_path / "output", cgroup_root=cgroup, run=fake_run
    )

    assert commands == [["df", "-Pk", str(tmp_path / "output")]]
    assert result["disk_available_gib"] > 80.0
    assert result["memory_max_bytes"] == EXPECTED_CGROUP_MAX_BYTES
    assert result["host_free_used"] is False


@pytest.mark.parametrize(
    ("available_kib", "maximum", "message"),
    [
        (70 * 1024**2, EXPECTED_CGROUP_MAX_BYTES, "at least 80 GiB"),
        (100 * 1024**2, 89 * 1024**3, "memory.max=90 GiB"),
    ],
)
def test_resource_guard_enforces_disk_and_exact_memory_limit(
    tmp_path: Path, available_kib: int, maximum: int, message: str
) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(f"{maximum}\n", encoding="utf-8")
    (cgroup / "memory.events").write_text("oom 0\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        return SimpleNamespace(
            stdout=(
                "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                f"/dev/test 500000000 1 {available_kib} 1% /workspace\n"
            )
        )

    with pytest.raises(RuntimeError, match=message):
        check_fragment_merge_resources(
            tmp_path / "output", cgroup_root=cgroup, run=fake_run
        )


def test_config_freezes_seed_and_parser_exposes_only_dev2_dev8_inputs(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="seed 42"):
        FragmentMergeExperimentConfig(
            runtime_manifest=tmp_path / "runtime.json",
            gt_dir=tmp_path / "gt",
            category_priors=tmp_path / "priors.json",
            output_root=tmp_path / "output",
            seed=7,
        )

    args = experiment.build_parser().parse_args(
        [
            "--runtime-manifest",
            str(tmp_path / "runtime.json"),
            "--gt-dir",
            str(tmp_path / "gt"),
            "--category-priors",
            str(tmp_path / "priors.json"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )
    assert args.seed == 42
    assert not hasattr(args, "holdout")
    assert not hasattr(args, "final")


def test_module_main_builds_config_and_prints_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[FragmentMergeExperimentConfig] = []

    def fake_run(config: FragmentMergeExperimentConfig):
        observed.append(config)
        return {"schema": "test", "status": "complete"}

    monkeypatch.setattr(experiment, "run_category_fragment_merge_experiment", fake_run)
    result = experiment.main(
        [
            "--runtime-manifest",
            str(tmp_path / "runtime.json"),
            "--gt-dir",
            str(tmp_path / "gt"),
            "--category-priors",
            str(tmp_path / "priors.json"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    assert observed[0].seed == 42
    assert '"status": "complete"' in capsys.readouterr().out
