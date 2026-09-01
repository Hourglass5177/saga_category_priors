from __future__ import annotations

import json
from pathlib import Path

import pytest

from category_priors.clean_baseline import two_step_experiment as experiment


COMMIT = "1" * 40
DEV2 = experiment.REGISTERED_DEV2_SCENE_IDS
DEV8 = experiment.REGISTERED_DEV8_SCENE_IDS


@pytest.fixture(autouse=True)
def _registered_runtime_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Orchestration tests use synthetic assets, not a committed cloud checkout."""

    monkeypatch.setattr(experiment, "_verify_registered_checkout", lambda _: None)
    monkeypatch.setattr(experiment, "_preflight_two_step_roots", lambda **_: None)


def _manifest(path: Path) -> Path:
    payload = {
        "schema": "saga-clean-mask-contract-manifest-v1",
        "dev2_scene_ids": list(DEV2),
        "dev8_scene_ids": list(DEV8),
        "scenes": [
            {
                "scene_id": scene_id,
                "source_evidence_request": {"scene": {"scene_id": scene_id}},
            }
            for scene_id in DEV8
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_two_step_stops_only_at_failed_technical_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    prepared = False

    def audit(*args, **kwargs):
        del args, kwargs
        return {"technical_gates": {"passed": False, "parity": False}}

    def prepare(**kwargs):
        nonlocal prepared
        del kwargs
        prepared = True
        raise AssertionError("step two must not start")

    monkeypatch.setattr(experiment, "audit_clean_baseline_manifest", audit)
    monkeypatch.setattr(experiment, "_prepare_flat_mask_control", prepare)
    with pytest.raises(RuntimeError, match="technical integrity"):
        experiment.run_clean_baseline_two_step(
            manifest_path=manifest,
            output_root=tmp_path / "artifacts",
            run_root=tmp_path / "runs",
            producer_commit=COMMIT,
        )
    assert not prepared
    status = json.loads(
        (tmp_path / "artifacts" / "clean_two_step_status.json").read_text()
    )
    assert status["status"] == "stopped"
    assert status["stage"] == "stopped-step1-technical-integrity"


def test_scientific_failure_completes_without_midrun_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")

    monkeypatch.setattr(
        experiment,
        "audit_clean_baseline_manifest",
        lambda *args, **kwargs: {"technical_gates": {"passed": True}},
    )
    preparation = {
        "scene_ids": list(DEV2),
        "storage_root": str(tmp_path / "runs" / "mask-control-inputs"),
        "scenes": {
            scene_id: {
                "scene_id": scene_id,
                "frame_count": 1,
                "generated_frame_count": 1,
                "historical_hierarchy_exact_frame_count": 0,
                "hierarchy_mask_count": 2,
                "flat_mask_count": 2,
                "union_changed_pixel_count": 0,
                "flat_overlap_pixel_count": 0,
                "mechanical_contract_pass": True,
                "input_binding_pass": True,
                "flat_repeat_identity_pass": True,
                "repeat_generated_frame_count": 0,
                "repeat_input_manifest_before": [],
                "repeat_input_manifest_after": [],
                "stranded_part_files": [],
                "hierarchy_evidence_request": str(tmp_path / f"{scene_id}-h.json"),
                "flat_evidence_request": str(tmp_path / f"{scene_id}-p.json"),
            }
            for scene_id in DEV2
        },
    }
    monkeypatch.setattr(
        experiment, "_prepare_flat_mask_control", lambda **kwargs: preparation
    )
    paired = {
        scene_id: {
            arm: {
                "bank_dir": str(tmp_path / "bank" / arm / scene_id),
                "output": str(tmp_path / "out" / arm / scene_id / "output.json"),
                "diagnostics": str(
                    tmp_path / "out" / arm / scene_id / "diagnostics.json"
                ),
            }
            for arm in experiment.REGISTERED_ARMS
        }
        for scene_id in DEV2
    }
    monkeypatch.setattr(experiment, "_run_paired_conditions", lambda **kwargs: paired)
    repeat_audits = {}
    for scene_id in DEV2:
        path = tmp_path / f"{scene_id}-repeat.json"
        path.write_text("{}", encoding="utf-8")
        repeat_audits[scene_id] = {"audit_path": str(path)}
    monkeypatch.setattr(
        experiment, "_run_flat_full_repeat", lambda **kwargs: repeat_audits
    )

    # The runtime import is satisfied by replacing the real evaluator entry.
    from category_priors.clean_baseline import mask_ablation

    monkeypatch.setattr(
        mask_ablation,
        "evaluate_mask_contract_ablation_manifest",
        lambda *args, **kwargs: {
            "analysis": {
                "mechanical_gate": {"passed": True},
                "scientific_gate": {"passed": False},
                "conclusion": "flat-mask-control-failed",
            },
        },
    )
    result = experiment.run_clean_baseline_two_step(
        manifest_path=manifest,
        output_root=tmp_path / "artifacts",
        run_root=tmp_path / "runs",
        producer_commit=COMMIT,
    )
    assert result["status"] == "complete"
    assert result["decision"] == "flat-mask-control-failed"
    analysis = json.loads(
        (tmp_path / "artifacts" / "clean_two_step_analysis.json").read_text()
    )
    assert analysis["category_prior_tested"] is False
    assert analysis["affinity_feature_used_for_geometric_association"] is False
    assert analysis["geometric_identity_unit"] == "complete-frame-mask-observation"
    assert analysis["semantic_category_role"] == "late-object-classification-only"
    assert analysis["step2_scientific_gates"]["passed"] is False


def test_manifest_requires_exact_two_dev_scenes(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    payload = json.loads(manifest.read_text())
    payload["dev2_scene_ids"] = [DEV2[0]]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _, loaded = experiment._load_manifest(manifest)
    with pytest.raises(ValueError, match="exact frozen DEV2"):
        experiment._dev2_scene_ids(loaded)


def test_input_repeat_gate_stops_before_any_gpu_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    monkeypatch.setattr(
        experiment,
        "audit_clean_baseline_manifest",
        lambda *args, **kwargs: {"technical_gates": {"passed": True}},
    )
    preparation = {
        "scene_ids": list(DEV2),
        "storage_root": str(tmp_path / "inputs"),
        "scenes": {
            scene_id: {
                "frame_count": 1,
                "generated_frame_count": 0,
                "historical_hierarchy_exact_frame_count": 0,
                "hierarchy_mask_count": 1,
                "flat_mask_count": 1,
                "union_changed_pixel_count": 0,
                "flat_overlap_pixel_count": 0,
                "mechanical_contract_pass": True,
                "input_binding_pass": True,
                "flat_repeat_identity_pass": scene_id != DEV2[0],
                "repeat_generated_frame_count": 0,
                "repeat_input_manifest_before": [],
                "repeat_input_manifest_after": [],
                "stranded_part_files": [],
            }
            for scene_id in DEV2
        },
    }
    monkeypatch.setattr(
        experiment, "_prepare_flat_mask_control", lambda **kwargs: preparation
    )
    gpu_started = False

    def run_paired(**kwargs):
        nonlocal gpu_started
        del kwargs
        gpu_started = True
        raise AssertionError("lifting must not start")

    monkeypatch.setattr(experiment, "_run_paired_conditions", run_paired)
    with pytest.raises(RuntimeError, match="input contract"):
        experiment.run_clean_baseline_two_step(
            manifest_path=manifest,
            output_root=tmp_path / "artifacts",
            run_root=tmp_path / "runs",
            producer_commit=COMMIT,
        )
    assert gpu_started is False


def test_root_preflight_fails_before_status_or_directory_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")

    def reject(**kwargs):
        del kwargs
        raise ValueError("source asset overlap")

    monkeypatch.setattr(experiment, "_preflight_two_step_roots", reject)
    with pytest.raises(ValueError, match="source asset overlap"):
        experiment.run_clean_baseline_two_step(
            manifest_path=manifest,
            output_root=tmp_path / "artifacts",
            run_root=tmp_path / "runs",
            producer_commit=COMMIT,
        )
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "runs").exists()


def test_complete_status_is_revalidated_instead_of_blindly_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    artifacts = tmp_path / "artifacts"
    runs = tmp_path / "runs"
    artifacts.mkdir()
    identity = experiment._run_identity(
        manifest_file=manifest.resolve(),
        producer_commit=COMMIT,
        artifacts=artifacts.resolve(),
        runs=runs.resolve(),
    )
    (artifacts / "clean_two_step_status.json").write_text(
        json.dumps(
            {
                "schema": experiment.STATUS_SCHEMA,
                "status": "complete",
                "run_identity": identity,
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def audit(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return {"technical_gates": {"passed": False}}

    monkeypatch.setattr(experiment, "audit_clean_baseline_manifest", audit)
    with pytest.raises(RuntimeError, match="technical integrity"):
        experiment.run_clean_baseline_two_step(
            manifest_path=manifest,
            output_root=artifacts,
            run_root=runs,
            producer_commit=COMMIT,
        )
    assert calls == 1


def test_standalone_audit_checks_source_asset_isolation_before_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    evaluated = False

    def reject(**kwargs):
        del kwargs
        raise ValueError("source asset overlap")

    def audit(*args, **kwargs):
        nonlocal evaluated
        del args, kwargs
        evaluated = True
        raise AssertionError("evaluation must not start")

    monkeypatch.setattr(experiment, "_preflight_source_asset_roots", reject)
    monkeypatch.setattr(
        experiment, "preflight_audit_output_directory", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(experiment, "audit_clean_baseline_manifest", audit)
    with pytest.raises(ValueError, match="source asset overlap"):
        experiment.audit_clean_baseline(
            manifest_path=manifest,
            output_root=tmp_path / "artifacts",
        )
    assert evaluated is False


def test_prepare_reenumerates_and_rejects_new_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path / "manifest.json")
    _, payload = experiment._load_manifest(manifest)
    calls: dict[str, int] = {}

    def prepare(*, scene_id, source_request, output_root, producer_commit):
        del source_request, producer_commit
        calls[scene_id] = calls.get(scene_id, 0) + 1
        metadata = output_root / "sam-metadata" / scene_id
        metadata.mkdir(parents=True, exist_ok=True)
        (metadata / "frame.npz").write_bytes(b"stable")
        if calls[scene_id] == 2:
            (metadata / "late-write.part").write_bytes(b"partial")
        return {
            "scene_id": scene_id,
            "frame_count": 1,
            "generated_frame_count": int(calls[scene_id] == 1),
            "mechanical_contract_pass": True,
            "input_binding_pass": True,
        }

    monkeypatch.setattr(experiment, "prepare_flat_mask_control_scene", prepare)
    summary = experiment._prepare_flat_mask_control(
        payload=payload,
        storage_root=tmp_path / "inputs",
        producer_commit=COMMIT,
    )

    for scene_id in DEV2:
        row = summary["scenes"][scene_id]
        assert row["flat_repeat_identity_pass"] is False
        assert row["stranded_part_files"] == [
            f"sam-metadata/{scene_id}/late-write.part"
        ]
        assert row["repeat_input_manifest_before"] != row["repeat_input_manifest_after"]
    assert experiment._input_preflight_pass(summary) is False
