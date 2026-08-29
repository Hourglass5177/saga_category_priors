from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import category_priors.category_cluster_experiment as experiment
from category_priors.category_cluster_bank import (
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
from category_priors.category_cluster_experiment import (
    COMPLETE_STATUS,
    DEV2,
    DEV8,
    EXPECTED_CGROUP_MAX_BYTES,
    FROZEN_SELECTION_SCHEMA,
    G1_CONDITIONS,
    HOLDOUT5,
    PRIMARY_CONDITIONS,
    ClusterExperimentConfig,
    ClusterExperimentHooks,
    check_cluster_experiment_resources,
    run_category_cluster_experiment,
)
from category_priors.io import load_json, write_json, write_rows


def _config(tmp_path: Path, *, runtime_name: str = "runtime.json") -> ClusterExperimentConfig:
    return ClusterExperimentConfig(
        runtime_manifest=tmp_path / runtime_name,
        gt_dir=tmp_path / "gt",
        locked_runtime_manifest=tmp_path / "locked-runtime.json",
        locked_gt_dir=tmp_path / "locked-gt",
        locked_evaluation_scenes=tmp_path / "locked-scenes.json",
        repo_root=tmp_path / "repo",
        category_priors=tmp_path / "priors.json",
        prior_oracle_root=tmp_path / "prior-oracle",
        reference_bank_root=tmp_path / "reference-bank",
        reference_trace_root=tmp_path / "reference-trace",
        output_root=tmp_path / "experiment",
        size_bins=tmp_path / "size-bins.json",
    )


def _tune24() -> tuple[str, ...]:
    extras = tuple(
        f"{scene.rsplit('_', 1)[0]}_99" for scene in (DEV8 + HOLDOUT5)[:11]
    )
    return DEV8 + HOLDOUT5 + extras


def _final48() -> tuple[str, ...]:
    return ("scene0019_01",) + tuple(
        f"scene8{index:03d}_00" for index in range(47)
    )


def _gate(passed: bool) -> dict:
    return {"passed": passed, "checks": {"registered": passed}}


def _analysis(
    *,
    phase: str,
    scenes: tuple[str, ...],
    selected: str | None,
    primary_pass: bool,
    include_g1: bool = False,
    g1_pass: bool = False,
    dev8_pass: bool = False,
) -> dict:
    if phase == "dev8":
        assert selected is not None
        gates = {selected: _gate(dev8_pass)}
        conditions = {R0_LEGACY: {}, selected: {}}
        selected_gate = gates[selected]
        tier = "frozen_dev2_selection"
    else:
        gates = {
            R1_METRIC_HDBSCAN: _gate(primary_pass and selected == R1_METRIC_HDBSCAN),
            R2_ANCHORED_HDBSCAN: _gate(primary_pass and selected == R2_ANCHORED_HDBSCAN),
        }
        conditions = {
            R0_LEGACY: {},
            R1_METRIC_HDBSCAN: {},
            R2_ANCHORED_HDBSCAN: {},
        }
        if include_g1:
            gates[G1_MUTUAL_LOCAL_GRAPH] = _gate(g1_pass)
            conditions[G1_MUTUAL_LOCAL_GRAPH] = {}
        selected_gate = gates.get(selected) if selected is not None else None
        tier = (
            "registered_graph_fallback"
            if selected == G1_MUTUAL_LOCAL_GRAPH
            else "primary_hdbscan_repair" if selected is not None else None
        )
    return {
        "schema": "saga-category-cluster-evaluation-v1",
        "phase": phase,
        "scene_ids": list(scenes),
        "conditions": conditions,
        "gates": gates,
        "selected_condition": selected,
        "selected_gate": selected_gate,
        "selection_tier": tier,
        "category_prior_tested": False,
    }


def _audit() -> dict:
    return {
        "schema": "saga-category-cluster-distance-audit-v1",
        "r0_identity_passed": True,
        "corrected_distance_contract_measured": True,
        "corrected_distance_contract_passed": True,
        "determinism_passed": True,
        "scenes": [
            {
                "scene_id": scene,
                "r0_raw_identity_checks": {
                    "sample_rank_exact": True,
                    "hdbscan_labels_exact": True,
                    "hdbscan_membership_atol_1e-6": True,
                },
                "r0_determinism": {
                    "condition": R0_LEGACY,
                    "measured_this_scene": True,
                    "violation_count": 0,
                },
                "corrected_conditions": [
                    {
                        "condition": condition,
                        "global_typical_diag_m": 1.25,
                        "distance_matrix_count": 1,
                        "corrected_distance_contract_measured": True,
                        "corrected_distance_contract_passed": True,
                        "determinism_measured_this_scene": True,
                        "determinism_violation_count": 0,
                    }
                    for condition in (R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN)
                ],
            }
            for scene in DEV2
        ],
    }


def _hooks(
    calls: list[tuple],
    *,
    primary_selected: str | None = R1_METRIC_HDBSCAN,
    g1_passed: bool = False,
    dev8_passed: bool = True,
    oracle_passed: bool = True,
    candidate_prior_passed: bool = True,
    replay_fail_stage: str | None = None,
    fail_once_at: str | None = None,
    build_roots: list[tuple[str, Path]] | None = None,
    audit_roots: list[Path] | None = None,
    evaluation_roots: list[tuple[str, Path]] | None = None,
) -> ClusterExperimentHooks:
    failed = False

    def maybe_fail(name: str) -> None:
        nonlocal failed
        if name == fail_once_at and not failed:
            failed = True
            raise RuntimeError(f"simulated {name} interruption")

    def resources(root: Path) -> dict:
        calls.append(("resources", Path(root)))
        return {
            "disk_available_gib": 100.0,
            "memory_max_bytes": EXPECTED_CGROUP_MAX_BYTES,
            "host_free_used": False,
        }

    def validate_inputs(**kwargs):
        calls.append(("validate_inputs",))
        return {
            "scene_ids": list(DEV8),
            "tune24_scene_ids": list(_tune24()),
            "final48_scene_ids": list(_final48()),
            "tune_physical_scene_count": 13,
            "final_physical_scene_count": 48,
            "gt_boundary": "offline_evaluation_only",
        }

    def build_banks(**kwargs):
        scenes = tuple(kwargs["scene_ids"])
        conditions = tuple(kwargs["conditions"])
        if scenes == DEV2 and conditions == PRIMARY_CONDITIONS:
            name = "build_dev2_primary"
        elif scenes == DEV2 and conditions == G1_CONDITIONS:
            name = "build_dev2_g1"
        elif scenes == DEV8:
            name = "build_dev8"
        elif scenes == HOLDOUT5:
            name = "build_holdout5"
        elif scenes == _tune24():
            name = "build_tune24"
        elif scenes == _final48():
            name = "build_final48"
        else:
            raise AssertionError(f"unregistered test build scenes: {scenes}")
        if build_roots is not None:
            build_roots.append((name, Path(kwargs["run_root"])))
        calls.append((name, scenes, conditions))
        maybe_fail(name)
        return {
            "total": len(scenes),
            "complete": len(scenes),
            "conditions": list(conditions),
            "reference_identity_required": bool(
                kwargs.get("require_reference_identity", True)
            ),
            "determinism_mode": (
                "measured_this_scene"
                if bool(kwargs.get("verify_determinism"))
                else "algorithm_contract_reference"
            ),
            "determinism_reference": (
                None
                if bool(kwargs.get("verify_determinism"))
                else {"schema": "test-dev2-determinism-reference"}
            ),
            "runs": [
                {"scene_id": scene, "status": "complete"} for scene in scenes
            ],
        }

    def audit_distance(**kwargs):
        if audit_roots is not None:
            audit_roots.append(Path(kwargs["run_root"]))
        calls.append(("audit_distance",))
        payload = _audit()
        write_json(kwargs["output_path"], payload)
        return payload

    def evaluate_banks(**kwargs):
        phase = str(kwargs["phase"])
        if evaluation_roots is not None:
            evaluation_roots.append((phase, Path(kwargs["run_root"])))
        primary_analysis = kwargs["primary_analysis"]
        if phase == "dev8":
            name = "evaluate_dev8"
            selected = str(kwargs["selected_condition"])
            result = _analysis(
                phase="dev8",
                scenes=DEV8,
                selected=selected,
                primary_pass=False,
                dev8_pass=dev8_passed,
            )
        elif primary_analysis is None:
            name = "evaluate_dev2_primary"
            result = _analysis(
                phase="dev2",
                scenes=DEV2,
                selected=primary_selected,
                primary_pass=primary_selected in {
                    R1_METRIC_HDBSCAN,
                    R2_ANCHORED_HDBSCAN,
                },
            )
        else:
            name = "evaluate_dev2_g1"
            assert Path(primary_analysis).is_file()
            result = _analysis(
                phase="dev2",
                scenes=DEV2,
                selected=G1_MUTUAL_LOCAL_GRAPH if g1_passed else None,
                primary_pass=False,
                include_g1=True,
                g1_pass=g1_passed,
            )
        calls.append((name, tuple(kwargs["scene_ids"]), kwargs["selected_condition"]))
        write_rows(kwargs["metrics_output"], [{"phase": phase, "test": True}])
        write_json(kwargs["analysis_output"], result)
        return result

    def evaluate_prior_oracle(**kwargs):
        maybe_fail("prior_oracle_v2")
        calls.append(("prior_oracle_v2",))
        result = {"passed": oracle_passed, "checks": {"registered": oracle_passed}}
        write_json(kwargs["output"], result)
        return result

    def evaluate_candidate_prior(**kwargs):
        maybe_fail("candidate_prior_dev8")
        calls.append(("candidate_prior_dev8", kwargs["selected_condition"]))
        result = {
            "passed": candidate_prior_passed,
            "acceptance_threshold": None,
            "gates": {"registered": candidate_prior_passed},
        }
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(kwargs["metrics_output"], [{"candidate_id": 0, "test": True}])
        write_json(output_dir / "dev8_analysis.json", result)
        return result

    def select_candidate_threshold(**kwargs):
        maybe_fail("select_threshold_dev2")
        calls.append(("select_threshold_dev2", kwargs["selected_condition"]))
        result = {
            "selected_threshold": 0.15,
            "scene_ids": list(DEV2),
            "score_source": "uniform",
            "tie_rule": "exact_tie_choose_higher_threshold",
            "grid_rows": [],
        }
        write_json(kwargs["output"], result)
        return result

    def replay_final_stage(**kwargs):
        stage = str(kwargs["stage"])
        maybe_fail(f"replay_{stage}")
        scenes = tuple(kwargs["scene_ids"])
        calls.append((f"replay_{stage}", scenes, kwargs["selected_condition"]))
        output_dir = Path(kwargs["output_dir"]) / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(
            output_dir / f"{stage}_condition_metrics.parquet",
            [{"stage": stage, "test": True}],
        )
        result = {
            "passed": stage != replay_fail_stage,
            "uniform_health": {},
            "data_minus_uniform": {},
        }
        write_json(output_dir / f"{stage}_analysis.json", result)
        return result

    return ClusterExperimentHooks(
        check_resources=resources,
        validate_inputs=validate_inputs,
        build_banks=build_banks,
        audit_distance=audit_distance,
        evaluate_banks=evaluate_banks,
        evaluate_prior_oracle=evaluate_prior_oracle,
        evaluate_candidate_prior=evaluate_candidate_prior,
        select_candidate_threshold=select_candidate_threshold,
        replay_final_stage=replay_final_stage,
    )


def _functional_calls(calls: list[tuple]) -> list[tuple]:
    return [row for row in calls if row[0] != "resources"]


def test_primary_dev2_winner_runs_frozen_prior_pipeline_to_final48(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(config, _hooks(calls))

    assert result["status"] == COMPLETE_STATUS
    assert result["checkpoint"] == "final48_passed"
    assert result["selected_condition"] == R1_METRIC_HDBSCAN
    assert result["category_prior_tested"] is True
    assert result["g1_tested"] is False
    assert _functional_calls(calls)[:6] == [
        ("validate_inputs",),
        ("build_dev2_primary", DEV2, PRIMARY_CONDITIONS),
        ("audit_distance",),
        ("evaluate_dev2_primary", DEV2, None),
        ("build_dev8", DEV8, (R0_LEGACY, R1_METRIC_HDBSCAN)),
        ("evaluate_dev8", DEV8, R1_METRIC_HDBSCAN),
    ]
    assert ("prior_oracle_v2",) in calls
    assert ("candidate_prior_dev8", R1_METRIC_HDBSCAN) in calls
    assert ("build_holdout5", HOLDOUT5, (R0_LEGACY, R1_METRIC_HDBSCAN)) in calls
    assert ("build_tune24", _tune24(), (R0_LEGACY, R1_METRIC_HDBSCAN)) in calls
    assert ("build_final48", _final48(), (R0_LEGACY, R1_METRIC_HDBSCAN)) in calls
    frozen = load_json(config.frozen_selection_path)
    assert frozen["schema"] == FROZEN_SELECTION_SCHEMA
    assert frozen["selected_condition"] == R1_METRIC_HDBSCAN
    assert frozen["scene_ids"] == list(DEV2)
    assert frozen["selected_gate"]["passed"] is True
    assert config.dev2_metrics_path.is_file()
    assert config.primary_dev2_metrics_path.is_file()
    assert config.dev8_analysis_path.is_file()
    assert load_json(config.state_path) == result


def test_dev2_direct_evidence_root_is_never_reused_by_dev8(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    build_roots: list[tuple[str, Path]] = []
    audit_roots: list[Path] = []
    evaluation_roots: list[tuple[str, Path]] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(
        config,
        _hooks(
            calls,
            build_roots=build_roots,
            audit_roots=audit_roots,
            evaluation_roots=evaluation_roots,
        ),
    )

    assert result["status"] == COMPLETE_STATUS
    assert config.dev2_run_root.resolve() != config.run_root.resolve()
    assert config.run_root.resolve() != config.final_run_root.resolve()
    assert ("build_dev2_primary", config.dev2_run_root) in build_roots
    assert ("build_dev8", config.run_root) in build_roots
    assert ("build_holdout5", config.run_root) in build_roots
    assert ("build_tune24", config.run_root) in build_roots
    assert ("build_final48", config.final_run_root) in build_roots
    assert audit_roots == [config.dev2_run_root]
    assert ("dev2", config.dev2_run_root) in evaluation_roots
    assert ("dev8", config.run_root) in evaluation_roots
    stored_identity = load_json(config.state_path)["identity"]
    assert Path(stored_identity["dev2_run_root"]) == config.dev2_run_root.resolve()
    assert Path(stored_identity["run_root"]) == config.run_root.resolve()
    assert not any(
        name != "build_dev2_primary" and root == config.dev2_run_root
        for name, root in build_roots
    )


def test_terminal_recovery_regenerates_distance_audit_from_direct_dev2_root(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    build_roots: list[tuple[str, Path]] = []
    audit_roots: list[Path] = []
    config = _config(tmp_path)
    hooks = _hooks(
        calls,
        build_roots=build_roots,
        audit_roots=audit_roots,
    )
    assert run_category_cluster_experiment(config, hooks)["status"] == COMPLETE_STATUS
    direct_build_count = sum(
        name == "build_dev2_primary" for name, _ in build_roots
    )
    config.distance_audit_path.write_text("{broken", encoding="utf-8")

    recovered = run_category_cluster_experiment(config, hooks)

    assert recovered["status"] == COMPLETE_STATUS
    assert load_json(config.distance_audit_path)["determinism_passed"] is True
    assert audit_roots == [config.dev2_run_root, config.dev2_run_root]
    assert sum(name == "build_dev2_primary" for name, _ in build_roots) == (
        direct_build_count
    )


def test_terminal_recovery_regenerates_dev2_analysis_without_rebuilding_direct_bank(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    build_roots: list[tuple[str, Path]] = []
    evaluation_roots: list[tuple[str, Path]] = []
    config = _config(tmp_path)
    hooks = _hooks(
        calls,
        build_roots=build_roots,
        evaluation_roots=evaluation_roots,
    )
    assert run_category_cluster_experiment(config, hooks)["status"] == COMPLETE_STATUS
    config.dev2_analysis_path.write_text("{broken", encoding="utf-8")

    recovered = run_category_cluster_experiment(config, hooks)

    assert recovered["status"] == COMPLETE_STATUS
    assert load_json(config.dev2_analysis_path)["phase"] == "dev2"
    assert sum(row[0] == "evaluate_dev2_primary" for row in calls) == 2
    assert sum(name == "build_dev2_primary" for name, _ in build_roots) == 1
    assert [root for phase, root in evaluation_roots if phase == "dev2"] == [
        config.dev2_run_root,
        config.dev2_run_root,
    ]


def test_g1_runs_only_after_both_primary_arms_fail(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(
        config,
        _hooks(calls, primary_selected=None, g1_passed=True),
    )

    assert result["status"] == COMPLETE_STATUS
    assert result["selected_condition"] == G1_MUTUAL_LOCAL_GRAPH
    assert result["g1_authorized"] is True
    assert result["g1_tested"] is True
    functional = _functional_calls(calls)
    assert functional.index(("evaluate_dev2_primary", DEV2, None)) < functional.index(
        ("build_dev2_g1", DEV2, G1_CONDITIONS)
    )
    assert functional.index(("build_dev2_g1", DEV2, G1_CONDITIONS)) < functional.index(
        ("evaluate_dev2_g1", DEV2, None)
    )
    assert (
        "build_dev8",
        DEV8,
        (R0_LEGACY, G1_MUTUAL_LOCAL_GRAPH),
    ) in functional


def test_g1_direct_build_uses_the_same_isolated_dev2_evidence_root(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    build_roots: list[tuple[str, Path]] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(
        config,
        _hooks(
            calls,
            primary_selected=None,
            g1_passed=True,
            build_roots=build_roots,
        ),
    )

    assert result["status"] == COMPLETE_STATUS
    assert ("build_dev2_primary", config.dev2_run_root) in build_roots
    assert ("build_dev2_g1", config.dev2_run_root) in build_roots
    assert ("build_dev8", config.run_root) in build_roots


def test_all_dev2_repairs_fail_stops_without_prior_or_dev8(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(
        config,
        _hooks(calls, primary_selected=None, g1_passed=False),
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "dev2_all_registered_repairs_failed"
    assert result["category_prior_tested"] is False
    assert not any(row[0] == "build_dev8" for row in calls)
    assert not config.dev8_metrics_path.exists()


def test_failed_dev8_health_stops_at_candidate_space(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_cluster_experiment(
        config,
        _hooks(calls, dev8_passed=False),
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "dev8_candidate_health_failed"
    assert "category prior was not tested" in result["stop_reason"]
    assert result["category_prior_tested"] is False


def test_prior_oracle_failure_stops_before_same_bank_prior(tmp_path: Path) -> None:
    calls: list[tuple] = []
    result = run_category_cluster_experiment(
        _config(tmp_path), _hooks(calls, oracle_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "prior_oracle_v2_gate_failed"
    assert result["prior_capacity_tested"] is True
    assert result["category_prior_tested"] is False
    assert not any(row[0] == "candidate_prior_dev8" for row in calls)


def test_threshold_free_prior_failure_blocks_threshold_and_replay(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    result = run_category_cluster_experiment(
        _config(tmp_path), _hooks(calls, candidate_prior_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "candidate_prior_dev8_gate_failed"
    assert result["category_prior_tested"] is True
    assert not any(row[0] == "select_threshold_dev2" for row in calls)
    assert not any(row[0].startswith("replay_") for row in calls)


@pytest.mark.parametrize(
    ("failed_stage", "checkpoint", "forbidden_build"),
    [
        ("dev8", "legacy_replay_dev8_gate_failed", "build_holdout5"),
        ("holdout", "legacy_replay_holdout5_gate_failed", "build_tune24"),
        ("tune", "legacy_replay_tune24_gate_failed", "build_final48"),
        ("final", "legacy_replay_final48_gate_failed", None),
    ],
)
def test_replay_gate_stops_at_registered_boundary(
    tmp_path: Path,
    failed_stage: str,
    checkpoint: str,
    forbidden_build: str | None,
) -> None:
    calls: list[tuple] = []
    result = run_category_cluster_experiment(
        _config(tmp_path), _hooks(calls, replay_fail_stage=failed_stage)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == checkpoint
    if forbidden_build is not None:
        assert not any(row[0] == forbidden_build for row in calls)
    for row in calls:
        if row[0] in {"build_holdout5", "build_tune24", "build_final48"}:
            assert row[2] == (R0_LEGACY, result["selected_condition"])


def test_downstream_interruption_resumes_without_repeating_cluster_selection(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls, fail_once_at="replay_holdout")

    with pytest.raises(RuntimeError, match="simulated replay_holdout"):
        run_category_cluster_experiment(config, hooks)
    assert load_json(config.state_path)["next_stage"] == "replay_holdout5"

    result = run_category_cluster_experiment(config, hooks)
    assert result["status"] == COMPLETE_STATUS
    assert sum(row[0] == "evaluate_dev2_primary" for row in calls) == 1
    assert sum(row[0] == "candidate_prior_dev8" for row in calls) == 1


def test_interrupted_dev8_build_resumes_without_repeating_completed_stages(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls, fail_once_at="build_dev8")

    with pytest.raises(RuntimeError, match="simulated build_dev8"):
        run_category_cluster_experiment(config, hooks)

    interrupted = load_json(config.state_path)
    assert interrupted["status"] == "error"
    assert interrupted["next_stage"] == "build_dev8"
    result = run_category_cluster_experiment(config, hooks)

    assert result["status"] == COMPLETE_STATUS
    assert sum(row[0] == "evaluate_dev2_primary" for row in calls) == 1
    assert sum(row[0] == "build_dev8" for row in calls) == 2


def test_tampered_frozen_selection_blocks_resume(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls, fail_once_at="build_dev8")
    with pytest.raises(RuntimeError):
        run_category_cluster_experiment(config, hooks)
    frozen = load_json(config.frozen_selection_path)
    frozen["selected_condition"] = R2_ANCHORED_HDBSCAN
    write_json(config.frozen_selection_path, frozen)

    with pytest.raises(ValueError, match="state and DEV2 frozen selection disagree"):
        run_category_cluster_experiment(config, hooks)
    assert sum(row[0] == "build_dev8" for row in calls) == 1


def test_existing_state_rejects_changed_experiment_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_category_cluster_experiment(config, _hooks([]))

    with pytest.raises(ValueError, match="identity differs"):
        run_category_cluster_experiment(
            _config(tmp_path, runtime_name="different-runtime.json"), _hooks([])
        )


def test_terminal_state_recovers_missing_final_metrics_from_last_stage(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls)
    assert run_category_cluster_experiment(config, hooks)["status"] == COMPLETE_STATUS
    config.replay_metrics_path("final").unlink()

    recovered = run_category_cluster_experiment(config, hooks)

    assert recovered["status"] == COMPLETE_STATUS
    assert config.replay_metrics_path("final").is_file()
    assert sum(row[0] == "replay_final" for row in calls) == 2
    assert sum(row[0] == "candidate_prior_dev8" for row in calls) == 1


def test_terminal_state_rebuilds_tampered_frozen_selection(tmp_path: Path) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    hooks = _hooks(calls)
    assert run_category_cluster_experiment(config, hooks)["status"] == COMPLETE_STATUS
    frozen = load_json(config.frozen_selection_path)
    frozen["selected_condition"] = R2_ANCHORED_HDBSCAN
    write_json(config.frozen_selection_path, frozen)

    recovered = run_category_cluster_experiment(config, hooks)

    assert recovered["status"] == COMPLETE_STATUS
    assert load_json(config.frozen_selection_path)["selected_condition"] == (
        R1_METRIC_HDBSCAN
    )
    assert sum(row[0] == "evaluate_dev2_primary" for row in calls) == 2


def test_resource_guard_uses_df_and_cgroup_only(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(
        f"{EXPECTED_CGROUP_MAX_BYTES}\n", encoding="utf-8"
    )
    (cgroup / "memory.events").write_text("oom 0\noom_kill 0\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return SimpleNamespace(
            stdout=(
                "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                "/dev/test 500000000 1 200000000 1% /workspace\n"
            )
        )

    result = check_cluster_experiment_resources(
        tmp_path / "output", cgroup_root=cgroup, run=fake_run
    )

    assert commands == [["df", "-Pk", str(tmp_path / "output")]]
    assert result["disk_available_gib"] > 80
    assert result["memory_max_bytes"] == EXPECTED_CGROUP_MAX_BYTES
    assert result["host_free_used"] is False


def test_audit_contract_failure_is_recoverable_error_not_method_stop(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    base = _hooks(calls)

    def bad_audit(**kwargs):
        payload = _audit()
        payload["scenes"][0]["corrected_conditions"][0][
            "corrected_distance_contract_passed"
        ] = False
        write_json(kwargs["output_path"], payload)
        return payload

    hooks = ClusterExperimentHooks(
        check_resources=base.check_resources,
        validate_inputs=base.validate_inputs,
        build_banks=base.build_banks,
        audit_distance=bad_audit,
        evaluate_banks=base.evaluate_banks,
        evaluate_prior_oracle=base.evaluate_prior_oracle,
        evaluate_candidate_prior=base.evaluate_candidate_prior,
        select_candidate_threshold=base.select_candidate_threshold,
        replay_final_stage=base.replay_final_stage,
    )
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="evaluation is blocked"):
        run_category_cluster_experiment(config, hooks)

    state = load_json(config.state_path)
    assert state["status"] == "error"
    assert state["next_stage"] == "audit_distance"
    assert state["category_prior_tested"] is False


def test_cli_parser_preserves_registered_paths_and_seed(tmp_path: Path) -> None:
    args = experiment.build_parser().parse_args(
        [
            "--runtime-manifest",
            str(tmp_path / "runtime.json"),
            "--gt-dir",
            str(tmp_path / "gt"),
            "--locked-runtime-manifest",
            str(tmp_path / "locked-runtime.json"),
            "--locked-gt-dir",
            str(tmp_path / "locked-gt"),
            "--locked-evaluation-scenes",
            str(tmp_path / "locked-scenes.json"),
            "--repo-root",
            str(tmp_path / "repo"),
            "--category-priors",
            str(tmp_path / "priors.json"),
            "--prior-oracle-root",
            str(tmp_path / "prior-oracle"),
            "--reference-bank-root",
            str(tmp_path / "reference-bank"),
            "--reference-trace-root",
            str(tmp_path / "reference-trace"),
            "--output-root",
            str(tmp_path / "output"),
            "--size-bins",
            str(tmp_path / "size-bins.json"),
            "--taxonomy",
            str(tmp_path / "taxonomy.json"),
            "--python-bin",
            str(tmp_path / "python"),
        ]
    )

    assert args.seed == 42
    assert Path(args.runtime_manifest) == tmp_path / "runtime.json"
    assert Path(args.locked_runtime_manifest) == tmp_path / "locked-runtime.json"
    assert Path(args.locked_gt_dir) == tmp_path / "locked-gt"
    assert Path(args.locked_evaluation_scenes) == tmp_path / "locked-scenes.json"
    assert Path(args.prior_oracle_root) == tmp_path / "prior-oracle"
    assert Path(args.reference_bank_root) == tmp_path / "reference-bank"
    assert Path(args.reference_trace_root) == tmp_path / "reference-trace"


def test_module_main_builds_config_and_prints_terminal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: list[ClusterExperimentConfig] = []

    def fake_run(config: ClusterExperimentConfig) -> dict:
        observed.append(config)
        return {
            "schema": "test",
            "status": COMPLETE_STATUS,
            "category_prior_tested": True,
        }

    monkeypatch.setattr(experiment, "run_category_cluster_experiment", fake_run)
    argv = [
        "--runtime-manifest",
        str(tmp_path / "runtime.json"),
        "--gt-dir",
        str(tmp_path / "gt"),
        "--locked-runtime-manifest",
        str(tmp_path / "locked-runtime.json"),
        "--locked-gt-dir",
        str(tmp_path / "locked-gt"),
        "--locked-evaluation-scenes",
        str(tmp_path / "locked-scenes.json"),
        "--repo-root",
        str(tmp_path / "repo"),
        "--category-priors",
        str(tmp_path / "priors.json"),
        "--prior-oracle-root",
        str(tmp_path / "prior-oracle"),
        "--reference-bank-root",
        str(tmp_path / "reference-bank"),
        "--reference-trace-root",
        str(tmp_path / "reference-trace"),
        "--output-root",
        str(tmp_path / "output"),
        "--size-bins",
        str(tmp_path / "size-bins.json"),
    ]

    assert experiment.main(argv) == 0
    assert len(observed) == 1
    assert observed[0].seed == 42
    assert observed[0].output_root == tmp_path / "output"
    printed = capsys.readouterr().out
    assert '"status": "complete"' in printed
    assert '"category_prior_tested": true' in printed
