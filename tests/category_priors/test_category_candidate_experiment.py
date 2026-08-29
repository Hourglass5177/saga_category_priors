from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import category_priors.category_candidate_experiment as experiment
from category_priors.category_candidate_experiment import (
    ACCEPTANCE_ARTIFACT_FILENAMES,
    CandidateExperimentConfig,
    CandidateExperimentHooks,
    DEV2,
    DEV8,
    EXPECTED_CGROUP_MAX_BYTES,
    HOLDOUT5,
    REGISTERED_10K_SCHEMA,
    ROOT_ACTION_NESTED,
    ROOT_ACTION_REPAIR,
    ROOT_ACTION_REPRESENTATION,
    REPRESENTATION_ACTION_10K,
    check_experiment_resources,
    nested_sampling_gate,
    run_category_candidate_experiment,
    same_source_feature_10k_gate,
)
from category_priors.io import load_json, write_json, write_rows


def _tune24() -> tuple[str, ...]:
    base = DEV8 + HOLDOUT5
    repeats = tuple(f"{scene.rsplit('_', 1)[0]}_99" for scene in base[:11])
    return base + repeats


def _final48() -> tuple[str, ...]:
    return ("scene0019_01",) + tuple(
        f"scene{index:04d}_00" for index in range(1000, 1047)
    )


def _config(tmp_path: Path) -> CandidateExperimentConfig:
    return CandidateExperimentConfig(
        runtime_manifest=tmp_path / "runtime.json",
        gt_dir=tmp_path / "gt",
        locked_runtime_manifest=tmp_path / "locked-runtime.json",
        locked_gt_dir=tmp_path / "locked-gt",
        locked_evaluation_scenes=tmp_path / "locked-scenes.json",
        repo_root=tmp_path / "repo",
        category_priors=tmp_path / "priors.json",
        prior_oracle_root=tmp_path / "oracle",
        reference_bank_root=tmp_path / "reference",
        output_root=tmp_path / "experiment",
        size_bins=tmp_path / "size-bins.json",
    )


def _hooks(
    calls: list[tuple],
    *,
    diagnoses: list[dict] | None = None,
    nested_passed: bool = True,
    parity_passed: bool = True,
    representation_action: str = ROOT_ACTION_REPAIR,
    feature10k_passed: bool = True,
    dev2_passed: bool = True,
    dev8_passed: bool = True,
    oracle_passed: bool = True,
    prior_passed: bool = True,
    replay_dev8_passed: bool = True,
    replay_holdout_passed: bool = True,
    replay_tune_passed: bool = True,
    replay_final_passed: bool = True,
    fail_once_at: str | None = None,
) -> CandidateExperimentHooks:
    diagnosis_queue = list(
        diagnoses
        or [{"diagnosable_object_count": 8, "next_action": ROOT_ACTION_REPAIR}]
    )
    failed = False

    def maybe_fail(name: str) -> None:
        nonlocal failed
        if name == fail_once_at and not failed:
            failed = True
            raise RuntimeError(f"simulated {name} interruption")

    def resources(root: Path) -> dict:
        calls.append(("resources", Path(root)))
        maybe_fail("resources")
        return {
            "disk_available_gib": 100.0,
            "memory_max_bytes": EXPECTED_CGROUP_MAX_BYTES,
            "host_free_used": False,
        }

    def validate_inputs(**kwargs):
        calls.append(("validate_inputs",))
        maybe_fail("validate_inputs")
        return {
            "tune24_scene_ids": list(_tune24()),
            "final48_scene_ids": list(_final48()),
            "tune_physical_scene_count": 13,
            "final_physical_scene_count": 48,
        }

    def repair(**kwargs):
        calls.append(
            (
                "repair",
                tuple(kwargs["scene_ids"]),
                int(kwargs["sample_cap"]),
                bool(kwargs["require_reference_identity"]),
            )
        )
        maybe_fail("repair")
        return {
            "complete": len(kwargs["scene_ids"]),
            "total": len(kwargs["scene_ids"]),
            "sample_cap": kwargs["sample_cap"],
            "reference_identity_required": kwargs["require_reference_identity"],
        }

    def diagnose(**kwargs):
        calls.append(("diagnose", tuple(kwargs["scene_ids"])))
        maybe_fail("diagnose")
        if not diagnosis_queue:
            raise AssertionError("unexpected extra diagnosis")
        result = diagnosis_queue.pop(0)
        write_rows(kwargs["trace_output"], [])
        write_json(kwargs["analysis_output"], result)
        return result

    def parity(**kwargs):
        calls.append(("parity", tuple(kwargs["scene_ids"])))
        maybe_fail("parity")
        return {"passed": parity_passed, "scenes": []}

    def nested(**kwargs):
        calls.append(("nested", tuple(kwargs["scene_ids"])))
        maybe_fail("nested")
        return {
            "passed": nested_passed,
            "checks": {"registered": nested_passed},
            "nested_trace_identity": True,
        }

    def representation(**kwargs):
        calls.append(("representation", tuple(kwargs["scene_ids"])))
        maybe_fail("representation")
        return {
            "next_action": representation_action,
            "representation_bottleneck_triggered": (
                representation_action == REPRESENTATION_ACTION_10K
            ),
            "mean_local_affinity_edge_auroc": 0.5,
            "oracle_seed_recall_025": 0.1,
        }

    def feature_10k(**kwargs):
        calls.append(("feature_10k", tuple(kwargs["scene_ids"])))
        maybe_fail("feature_10k")
        return {
            "schema": REGISTERED_10K_SCHEMA,
            "passed": feature10k_passed,
            "checks": {"registered": feature10k_passed},
            "mean_affinity_auroc_delta": 0.06 if feature10k_passed else 0.01,
            "same_class_iou025_candidate_delta": 2 if feature10k_passed else 0,
            "feature_iterations": 10_000,
            "scene_ids": list(DEV2),
            "control_run_root": str(kwargs["control_run_root"]),
            "control_runtime_manifest": str(kwargs["control_runtime_manifest"]),
        }

    def evaluate_repair(**kwargs):
        phase = str(kwargs["phase"])
        calls.append(
            (
                "evaluate_repair",
                phase,
                tuple(kwargs["scene_ids"]),
                kwargs["selected_condition"],
            )
        )
        maybe_fail(f"evaluate_repair_{phase}")
        if phase == "dev2":
            assert kwargs.get("frozen_repair_artifact") is None
            selected = "C1-consistent-envelope" if dev2_passed else None
            result = {
                "schema": "saga-category-candidate-repair-analysis-v1",
                "phase": "dev2",
                "scene_ids": list(DEV2),
                "passed": dev2_passed,
                "selected_condition": selected,
                "dev2_arm_gates": {
                    "C1-consistent-envelope": {"passed": dev2_passed}
                },
            }
            write_rows(kwargs["metrics_output"], [])
            write_json(kwargs["analysis_output"], result)
            return result
        artifact = kwargs.get("frozen_repair_artifact")
        assert isinstance(artifact, Path) and artifact.is_file()
        result = {
            "passed": dev8_passed,
            "selected_condition": kwargs["selected_condition"],
            "dev8_health_gate": {"passed": dev8_passed},
        }
        write_rows(kwargs["metrics_output"], [])
        write_json(kwargs["analysis_output"], result)
        return result

    def oracle(**kwargs):
        calls.append(("oracle",))
        maybe_fail("oracle")
        result = {
            "passed": oracle_passed,
            "checks": {"registered": oracle_passed},
        }
        write_json(kwargs["output"], result)
        return result

    def prior(**kwargs):
        calls.append(("prior", kwargs["selected_condition"]))
        maybe_fail("prior")
        result = {
            "passed": prior_passed,
            "acceptance_threshold": None,
            "gates": {"registered": prior_passed},
        }
        write_rows(kwargs["metrics_output"], [])
        write_json(Path(kwargs["output_dir"]) / "dev8_analysis.json", result)
        return result

    def threshold(**kwargs):
        calls.append(("threshold", kwargs["selected_condition"]))
        maybe_fail("threshold")
        return {
            "selected_threshold": 0.15,
            "scene_ids": list(DEV2),
            "score_source": "uniform",
            "tie_rule": "exact_tie_choose_higher_threshold",
        }

    def replay(**kwargs):
        stage = str(kwargs["stage"])
        calls.append(
            (
                "replay",
                stage,
                tuple(kwargs["scene_ids"]),
                kwargs["selected_condition"],
                kwargs["threshold"],
            )
        )
        maybe_fail(f"replay_{stage}")
        passed = {
            "dev8": replay_dev8_passed,
            "holdout": replay_holdout_passed,
            "tune": replay_tune_passed,
            "final": replay_final_passed,
        }[stage]
        result = {
            "passed": passed,
            "uniform_health": {"passed": passed},
            "data_minus_uniform": {"passed": passed},
        }
        write_rows(
            Path(kwargs["output_dir"])
            / "evaluation"
            / f"{stage}_condition_metrics.parquet",
            [],
        )
        return result

    return CandidateExperimentHooks(
        check_resources=resources,
        validate_inputs=validate_inputs,
        repair=repair,
        check_b0_parity=parity,
        diagnose=diagnose,
        nested_sampling_control=nested,
        representation_diagnostic=representation,
        feature_10k_control=feature_10k,
        evaluate_repair=evaluate_repair,
        evaluate_prior_oracle=oracle,
        evaluate_candidate_prior=prior,
        select_candidate_threshold=threshold,
        replay_final_stage=replay,
    )


def test_fixed_scene_contract_is_exact_and_disjoint() -> None:
    assert DEV2 == ("scene0645_00", "scene0025_01")
    assert DEV8[:2] == DEV2
    assert len(DEV8) == 8
    assert HOLDOUT5 == (
        "scene0231_00",
        "scene0608_00",
        "scene0356_00",
        "scene0011_00",
        "scene0593_00",
    )
    assert not set(DEV8).intersection(HOLDOUT5)


def test_happy_path_freezes_dev2_choices_and_stops_after_holdout(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)

    result = run_category_candidate_experiment(config, _hooks(calls))

    assert result["status"] == "complete"
    assert result["checkpoint"] == "final48_passed"
    assert result["frozen_repair_condition"] == "C1-consistent-envelope"
    assert result["frozen_threshold"] == 0.15
    functional = [row for row in calls if row[0] != "resources"]
    assert functional == [
        ("validate_inputs",),
        ("repair", DEV2, 5_000, True),
        ("parity", DEV2),
        ("diagnose", DEV2),
        ("evaluate_repair", "dev2", DEV2, None),
        ("repair", DEV8, 5_000, True),
        (
            "evaluate_repair",
            "dev8",
            DEV8,
            "C1-consistent-envelope",
        ),
        ("oracle",),
        ("prior", "C1-consistent-envelope"),
        ("threshold", "C1-consistent-envelope"),
        ("replay", "dev8", DEV8, "C1-consistent-envelope", 0.15),
        ("repair", HOLDOUT5, 5_000, False),
        ("replay", "holdout", HOLDOUT5, "C1-consistent-envelope", 0.15),
        ("repair", _tune24(), 5_000, False),
        ("replay", "tune", _tune24(), "C1-consistent-envelope", 0.15),
        ("repair", _final48(), 5_000, False),
        ("replay", "final", _final48(), "C1-consistent-envelope", 0.15),
    ]
    persisted = load_json(config.state_path)
    assert persisted == result
    assert not list(config.output_root.glob(".experiment_state.json.*.tmp"))
    expected = set(ACCEPTANCE_ARTIFACT_FILENAMES) | {"viewer/"}
    assert set(result["acceptance_artifacts"]) == expected
    assert all(
        Path(entry["path"]).exists()
        for entry in result["acceptance_artifacts"].values()
    )
    public_analysis = load_json(
        config.acceptance_artifact("category_denoise_v2_analysis.json")
    )
    assert public_analysis["status"] == "complete"
    assert public_analysis["checkpoint"] == "final48_passed"
    assert set(public_analysis["acceptance_artifacts"]) == expected


def test_less_than_eight_objects_extends_trace_before_repair_evaluation(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    hooks = _hooks(
        calls,
        diagnoses=[
            {"diagnosable_object_count": 5, "next_action": "extend-trace-only-to-dev8"},
            {"diagnosable_object_count": 12, "next_action": ROOT_ACTION_REPAIR},
        ],
    )

    result = run_category_candidate_experiment(_config(tmp_path), hooks)

    assert result["status"] == "complete"
    functional = [row for row in calls if row[0] != "resources"]
    assert functional[:7] == [
        ("validate_inputs",),
        ("repair", DEV2, 5_000, True),
        ("parity", DEV2),
        ("diagnose", DEV2),
        ("repair", DEV8, 5_000, True),
        ("diagnose", DEV8),
        ("evaluate_repair", "dev2", DEV2, None),
    ]


def test_dev8_with_seven_objects_routes_raw_majority_to_representation(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    hooks = _hooks(
        calls,
        diagnoses=[
            {
                "diagnosable_object_count": 1,
                "next_action": "extend-trace-only-to-dev8",
            },
            {
                "diagnosable_object_count": 7,
                "next_action": "extend-trace-only-to-dev8",
                "sample_starved_is_majority_of_failures": False,
                "raw_clustering_is_majority_of_sufficiently_sampled_failures": True,
            },
        ],
        representation_action=ROOT_ACTION_REPAIR,
        dev2_passed=False,
    )

    result = run_category_candidate_experiment(_config(tmp_path), hooks)

    functional = [row for row in calls if row[0] != "resources"]
    assert ("representation", DEV8) in functional
    assert result["checkpoint"] == "repair_dev2_gate_failed"


def test_migrates_erroneous_dev8_insufficient_stop_without_retracing(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    state = experiment._initial_state(config)
    state.update(
        {
            "status": "stopped",
            "checkpoint": "root_diagnosis_insufficient_on_dev8",
            "next_stage": None,
            "stop_reason": "legacy controller bug",
            "root_scene_ids": list(DEV8),
            "tune24_scene_ids": list(_tune24()),
            "final48_scene_ids": list(_final48()),
        }
    )
    write_json(
        config.acceptance_artifact("candidate_formation_root_cause.json"),
        {
            "diagnosable_object_count": 7,
            "next_action": "extend-trace-only-to-dev8",
            "sample_starved_is_majority_of_failures": False,
            "raw_clustering_is_majority_of_sufficiently_sampled_failures": True,
        },
    )
    experiment._write_state(config, state)
    hooks = _hooks(
        calls,
        representation_action=ROOT_ACTION_REPAIR,
        dev2_passed=False,
    )

    result = run_category_candidate_experiment(config, hooks)

    functional = [row for row in calls if row[0] != "resources"]
    assert functional[0] == ("representation", DEV8)
    assert not any(row[0] in {"repair", "diagnose"} for row in functional)
    assert result["checkpoint"] == "repair_dev2_gate_failed"


def test_nested_sampling_branch_freezes_10k_only_after_gate_passes(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    result = run_category_candidate_experiment(
        _config(tmp_path),
        _hooks(
            calls,
            diagnoses=[
                {"diagnosable_object_count": 9, "next_action": ROOT_ACTION_NESTED}
            ],
        ),
    )

    assert result["status"] == "complete"
    assert result["active_sample_cap"] == 10_000
    assert Path(result["active_run_root"]).name == "formation_10k"
    assert ("nested", DEV2) in calls
    assert ("repair", DEV8, 10_000, False) in calls


def test_representation_branch_runs_same_source_10k_and_stops_for_approval(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple] = []
    stopped = run_category_candidate_experiment(
        config,
        _hooks(
            calls,
            diagnoses=[
                {
                    "diagnosable_object_count": 9,
                    "next_action": ROOT_ACTION_REPRESENTATION,
                }
            ],
            representation_action=REPRESENTATION_ACTION_10K,
        ),
    )

    assert stopped["status"] == "stopped"
    assert stopped["checkpoint"] == "feature_10k_control_passed_requires_expansion"
    assert stopped["feature_10k_control_tested"] is True
    assert stopped["feature_10k_control_passed"] is True
    assert Path(stopped["active_run_root"]).name == "formation_feature_10k"
    assert Path(stopped["active_runtime_manifest"]).name == (
        "feature_10k_runtime_manifest.json"
    )
    assert ("feature_10k", DEV2) in calls
    assert not any(row[0] == "evaluate_repair" for row in calls)


def test_representation_10k_failure_is_terminal(tmp_path: Path) -> None:
    calls: list[tuple] = []
    stopped = run_category_candidate_experiment(
        _config(tmp_path),
        _hooks(
            calls,
            diagnoses=[
                {
                    "diagnosable_object_count": 9,
                    "next_action": ROOT_ACTION_REPRESENTATION,
                }
            ],
            representation_action=REPRESENTATION_ACTION_10K,
            feature10k_passed=False,
        ),
    )
    assert stopped["status"] == "stopped"
    assert stopped["checkpoint"] == "feature_10k_control_failed"
    assert stopped["feature_10k_control_passed"] is False
    assert ("feature_10k", DEV2) in calls


def test_same_source_feature_10k_gate_requires_both_registered_gains() -> None:
    def representation(auc: float) -> dict:
        return {"mean_local_affinity_edge_auroc": auc}

    def candidates(count: int) -> dict:
        return {
            "conditions": {
                "C0-legacy": {"same_class_iou_025_count": count}
            }
        }

    passed = same_source_feature_10k_gate(
        source_representation=representation(0.50),
        control_representation=representation(0.56),
        source_candidates=candidates(4),
        control_candidates=candidates(6),
    )
    assert passed["passed"] is True
    auc_only = same_source_feature_10k_gate(
        source_representation=representation(0.50),
        control_representation=representation(0.56),
        source_candidates=candidates(4),
        control_candidates=candidates(5),
    )
    assert auc_only["passed"] is False


def test_gate_failure_is_terminal_and_restart_invokes_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple] = []
    stopped = run_category_candidate_experiment(
        config, _hooks(calls, dev2_passed=False)
    )
    assert stopped["status"] == "stopped"
    assert stopped["checkpoint"] == "repair_dev2_gate_failed"
    assert not any(row[0] in {"oracle", "prior", "replay"} for row in calls)

    restarted_calls: list[tuple] = []
    same = run_category_candidate_experiment(config, _hooks(restarted_calls))
    assert same == stopped
    assert restarted_calls == []


def test_off_vs_b0_parity_failure_precedes_all_gt_diagnosis(tmp_path: Path) -> None:
    calls: list[tuple] = []
    result = run_category_candidate_experiment(
        _config(tmp_path), _hooks(calls, parity_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "b0_parity_dev2_failed"
    assert ("parity", DEV2) in calls
    assert not any(row[0] == "diagnose" for row in calls)
    assert not any(row[0] == "evaluate_repair" for row in calls)


def test_holdout_pass_continues_to_tune_and_tune_failure_blocks_final48(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    result = run_category_candidate_experiment(
        _config(tmp_path), _hooks(calls, replay_tune_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "legacy_replay_tune24_gate_failed"
    assert ("replay", "holdout", HOLDOUT5, "C1-consistent-envelope", 0.15) in calls
    assert ("replay", "tune", _tune24(), "C1-consistent-envelope", 0.15) in calls
    assert not any(row[0] == "replay" and row[1] == "final" for row in calls)


def test_failed_threshold_free_prior_gate_never_selects_dev2_threshold(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    result = run_category_candidate_experiment(
        _config(tmp_path), _hooks(calls, prior_passed=False)
    )

    assert result["status"] == "stopped"
    assert result["checkpoint"] == "candidate_prior_dev8_gate_failed"
    assert ("prior", "C1-consistent-envelope") in calls
    assert not any(row[0] == "threshold" for row in calls)
    assert not any(row[0] == "replay" for row in calls)


def test_interrupted_stage_is_atomic_and_retried_without_repeating_history(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="simulated evaluate_repair_dev2"):
        run_category_candidate_experiment(
            config,
            _hooks([], fail_once_at="evaluate_repair_dev2"),
        )
    failed = load_json(config.state_path)
    assert failed["status"] == "error"
    assert failed["next_stage"] == "evaluate_repair_dev2"
    assert failed["last_error"]["type"] == "RuntimeError"

    calls: list[tuple] = []
    completed = run_category_candidate_experiment(config, _hooks(calls))
    assert completed["status"] == "complete"
    assert calls[0] == ("evaluate_repair", "dev2", DEV2, None)
    history_stages = [row["stage"] for row in completed["history"]]
    assert history_stages.count("repair_dev2_5k") == 1
    assert history_stages.count("b0_parity_dev2") == 1
    assert history_stages.count("diagnose_root_dev2") == 1


def test_resource_guard_uses_df_and_exact_cgroup_not_host_free(
    tmp_path: Path,
) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024", encoding="utf-8")
    (cgroup / "memory.max").write_text(
        str(EXPECTED_CGROUP_MAX_BYTES), encoding="utf-8"
    )
    (cgroup / "memory.events").write_text("oom 0\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        available_kib = 100 * 1024**2
        return SimpleNamespace(
            stdout=(
                "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
                f"mock 200000000 1 {available_kib} 1% /workspace\n"
            )
        )

    result = check_experiment_resources(
        tmp_path / "output", cgroup_root=cgroup, run=fake_run
    )
    assert commands == [["df", "-Pk", str(tmp_path / "output")]]
    assert result["disk_available_gib"] == 100.0
    assert result["memory_max_bytes"] == EXPECTED_CGROUP_MAX_BYTES
    assert result["host_free_used"] is False

    (cgroup / "memory.max").write_text("max", encoding="utf-8")
    with pytest.raises(RuntimeError, match="90 GiB"):
        check_experiment_resources(tmp_path / "output", cgroup_root=cgroup, run=fake_run)


def test_nested_sampling_gate_requires_gain_count_bound_and_identity() -> None:
    source = [
        {
            "scene_id": "a",
            "gt_class": "chair",
            "gt_instance_id": index,
            "best_raw_cluster_id": index,
            "best_raw_iou": 0.3,
        }
        for index in range(2)
    ] + [
        {
            "scene_id": "a",
            "gt_class": "chair",
            "gt_instance_id": index + 2,
            "best_raw_cluster_id": index + 2,
            "best_raw_iou": 0.0,
        }
        for index in range(8)
    ]
    nested = [dict(row) for row in source]
    nested[2].update({"best_raw_cluster_id": 10, "best_raw_iou": 0.4})
    nested[3].update({"best_raw_cluster_id": 11, "best_raw_iou": 0.4})
    passed = nested_sampling_gate(
        source,
        nested,
        source_candidate_count=100,
        nested_candidate_count=140,
        nested_trace_identity=True,
    )
    assert passed["passed"]
    assert passed["new_raw_iou025_cluster_count"] == 2

    failed = nested_sampling_gate(
        source,
        nested,
        source_candidate_count=100,
        nested_candidate_count=151,
        nested_trace_identity=False,
    )
    assert not failed["passed"]
    assert not failed["checks"]["nested_trace_identity"]
    assert not failed["checks"]["candidate_count_at_most_1.5x"]


def test_replay_identity_binds_bank_prior_seed_knn_filter_and_q_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    for path, text in (
        (config.runtime_manifest, "runtime"),
        (config.category_priors, "priors"),
        (config.size_bins, "size-bins"),
    ):
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        experiment,
        "_selected_bank_replay_identity",
        lambda **kwargs: {
            "scenes": [{"scene_id": DEV2[0], "bank_sha256": "bank"}],
            "bank_signature": "bank-signature",
            "candidate_id_q_signature": "q-id-signature",
        },
    )

    identity = experiment._expected_replay_identity(
        config=config,
        runtime_manifest=config.runtime_manifest,
        run_root=tmp_path / "formation",
        scene_ids=DEV2,
        selected_condition="C1-consistent-envelope",
        threshold=0.15,
        stage="dev8",
    )

    assert identity["seed"] == 42
    assert identity["knn_k"] == 256
    assert identity["min_count"] == 10
    assert identity["category_priors_sha256"]
    assert identity["bank_signature"] == "bank-signature"
    assert identity["candidate_id_q_signature"] == "q-id-signature"
    path = experiment._bind_replay_identity(tmp_path / "replay", identity)
    assert load_json(path) == identity
    with pytest.raises(ValueError, match="recovery identity"):
        experiment._bind_replay_identity(
            tmp_path / "replay", {**identity, "threshold": 0.20}
        )

    orphan = tmp_path / "orphan"
    (orphan / "runs" / "replay" / "uniform" / DEV2[0]).mkdir(parents=True)
    (orphan / "runs" / "replay" / "uniform" / DEV2[0] / "output.json").write_text(
        "{}", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unbound replay outputs"):
        experiment._bind_replay_identity(orphan, identity)


def test_replay_output_validation_rejects_q_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import category_priors.category_denoise as denoise

    bank = SimpleNamespace(
        candidates=(
            {"candidate_id": 0, "base_score": 0.4},
            {"candidate_id": 2, "base_score": 0.7},
        ),
        point_count=300,
    )
    monkeypatch.setattr(denoise, "load_candidate_bank", lambda path: bank)
    replay_root = tmp_path / "replay"
    for mode in ("uniform", "class"):
        write_json(
            replay_root / "replay" / mode / DEV2[0] / "diagnostics.json",
            {
                "category_denoise": {
                    "action": "candidate-replay",
                    "mode": mode,
                    "scene_id": DEV2[0],
                    "score_threshold": 0.15,
                    "knn_k_effective": 256,
                    "filter_min_count": 10,
                    "protected_or_reinserted_point_count": 0,
                    "secondary_class_vote_applied": False,
                    "decisions": [
                        {"candidate_id": 0, "Q": 0.4},
                        {"candidate_id": 2, "Q": 0.7},
                    ],
                    "candidate_survival": [
                        {"candidate_id": 0},
                        {"candidate_id": 2},
                    ],
                }
            },
        )
    experiment._validate_replay_outputs(
        replay_run_root=replay_root,
        source_run_root=tmp_path / "source",
        scene_ids=(DEV2[0],),
        selected_condition="C1-consistent-envelope",
        threshold=0.15,
    )

    payload = load_json(
        replay_root / "replay" / "class" / DEV2[0] / "diagnostics.json"
    )
    payload["category_denoise"]["decisions"][1]["Q"] = 0.6
    write_json(
        replay_root / "replay" / "class" / DEV2[0] / "diagnostics.json",
        payload,
    )
    with pytest.raises(ValueError, match="candidate ID/Q identity changed"):
        experiment._validate_replay_outputs(
            replay_run_root=replay_root,
            source_run_root=tmp_path / "source",
            scene_ids=(DEV2[0],),
            selected_condition="C1-consistent-envelope",
            threshold=0.15,
        )


def test_real_off_b0_parity_compares_point_labels_and_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import category_priors.category_denoise_runner as runner

    config = _config(tmp_path)
    run_root = tmp_path / "runs"

    def prediction(path: Path, labels: list[int]) -> None:
        write_json(
            path,
            {
                "point_labels": labels,
                "instances": {"0": {"class": "chair", "score": 0.5}},
            },
        )

    prediction(run_root / "b0-off" / DEV2[0] / "output.json", [-1, 0, 0])
    prediction(run_root / "b0" / DEV2[0] / "output.json", [-1, 0, 0])
    monkeypatch.setattr(runner, "run_category_denoise_b0_control", lambda *a, **k: {})
    passed = experiment._default_check_b0_parity(
        config=config, run_root=run_root, scene_ids=(DEV2[0],)
    )
    assert passed["passed"]

    prediction(run_root / "b0" / DEV2[0] / "output.json", [-1, -1, 0])
    failed = experiment._default_check_b0_parity(
        config=config, run_root=run_root, scene_ids=(DEV2[0],)
    )
    assert not failed["passed"]
    assert failed["scenes"][0]["changed_point_count"] == 1
