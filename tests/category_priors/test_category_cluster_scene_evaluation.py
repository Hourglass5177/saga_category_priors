from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors import category_cluster_runner
from category_priors import category_cluster_scene_evaluation as scene_evaluation
from category_priors import cli
from category_priors.category_candidate_clustering import (
    G1_MUTUAL_LOCAL_GRAPH,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
from category_priors.category_cluster_evaluation import R0_LEGACY
from category_priors.category_cluster_scene_evaluation import (
    DEV2_SCENE_IDS,
    DEV8_SCENE_IDS,
    _evaluation_scene,
    evaluate_category_cluster_run,
)
from category_priors.taxonomy import load_taxonomy


def test_evaluation_scene_propagates_real_gt_instance_ids(monkeypatch) -> None:
    objects = (
        SimpleNamespace(class_id=0, instance_id=17, size_bin="tiny"),
        SimpleNamespace(class_id=1, instance_id=42, size_bin="large"),
    )
    context = {
        "objects": objects,
        "mapping": SimpleNamespace(
            gt_to_gaussian=SimpleNamespace(
                indices=np.asarray([0, 1, 2, 3], dtype=np.int64)
            ),
            gaussian_to_gt=SimpleNamespace(
                indices=np.asarray([0, 1, 2, 3, -1], dtype=np.int64)
            ),
        ),
        "object_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
    }
    monkeypatch.setattr(scene_evaluation, "_scene_context", lambda **_: context)

    observed = _evaluation_scene(
        scene_id="scene0645_00",
        scene={},
        gt_dir=Path("gt"),
        taxonomy=load_taxonomy(),
        size_spec=None,
        radius_m=0.05,
        min_region_size=100,
    )

    assert observed.gt_object_instance_ids.tolist() == [17, 42]
    assert observed.gt_object_size_bins == ("tiny", "large")


def test_dev2_requires_the_exact_registered_physical_scene_set(tmp_path) -> None:
    common = {
        "runtime_manifest": tmp_path / "runtime.json",
        "gt_dir": tmp_path / "gt",
        "run_root": tmp_path / "runs",
        "taxonomy": load_taxonomy(),
        "phase": "dev2",
        "metrics_output": tmp_path / "metrics.parquet",
        "analysis_output": tmp_path / "analysis.json",
    }

    with pytest.raises(ValueError, match="exact registered scene set"):
        evaluate_category_cluster_run(
            **common,
            scene_ids=(DEV2_SCENE_IDS[0],),
        )


def _touch_registered_banks(root: Path, scene_ids: tuple[str, ...]) -> None:
    for scene_id in scene_ids:
        for condition in (
            R0_LEGACY,
            R1_METRIC_HDBSCAN,
            R2_ANCHORED_HDBSCAN,
            G1_MUTUAL_LOCAL_GRAPH,
        ):
            path = root / "bank" / scene_id / condition / "bank_labels.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


def test_g1_cannot_leak_in_when_a_primary_dev2_arm_passed(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    _touch_registered_banks(run_root, DEV2_SCENE_IDS)
    monkeypatch.setattr(
        scene_evaluation,
        "load_scene_runtime_manifest",
        lambda _: {scene_id: {} for scene_id in DEV2_SCENE_IDS},
    )
    monkeypatch.setattr(scene_evaluation, "_evaluation_scene", lambda **_: object())
    monkeypatch.setattr(scene_evaluation, "load_candidate_bank", lambda _: object())
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps(
            {
                "phase": "dev2",
                "selected_condition": R1_METRIC_HDBSCAN,
                "gates": {
                    R1_METRIC_HDBSCAN: {"passed": True},
                    R2_ANCHORED_HDBSCAN: {"passed": False},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only after both primary DEV2 arms fail"):
        evaluate_category_cluster_run(
            runtime_manifest=tmp_path / "runtime.json",
            gt_dir=tmp_path / "gt",
            run_root=run_root,
            scene_ids=DEV2_SCENE_IDS,
            taxonomy=load_taxonomy(),
            phase="dev2",
            metrics_output=tmp_path / "metrics.parquet",
            analysis_output=tmp_path / "analysis.json",
            primary_analysis=primary,
        )


def test_primary_dev2_recovery_ignores_a_stale_g1_bank(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    _touch_registered_banks(run_root, DEV2_SCENE_IDS)
    monkeypatch.setattr(
        scene_evaluation,
        "load_scene_runtime_manifest",
        lambda _: {scene_id: {} for scene_id in DEV2_SCENE_IDS},
    )
    monkeypatch.setattr(scene_evaluation, "_evaluation_scene", lambda **_: object())
    loaded_conditions: list[str] = []

    def fake_load(path):
        loaded_conditions.append(Path(path).name)
        return object()

    monkeypatch.setattr(scene_evaluation, "load_candidate_bank", fake_load)
    observed: dict[str, object] = {}

    def fake_evaluate(_scenes, banks, **_kwargs):
        observed["conditions"] = tuple(banks)
        return {
            "phase": "dev2",
            "conditions": {
                condition: {"per_scene": []}
                for condition in (
                    R0_LEGACY,
                    R1_METRIC_HDBSCAN,
                    R2_ANCHORED_HDBSCAN,
                )
            },
        }

    monkeypatch.setattr(
        scene_evaluation, "evaluate_cluster_candidate_banks", fake_evaluate
    )
    monkeypatch.setattr(scene_evaluation, "write_rows", lambda *_: None)
    monkeypatch.setattr(scene_evaluation, "write_json", lambda *_: None)

    evaluate_category_cluster_run(
        runtime_manifest=tmp_path / "runtime.json",
        gt_dir=tmp_path / "gt",
        run_root=run_root,
        scene_ids=DEV2_SCENE_IDS,
        taxonomy=load_taxonomy(),
        phase="dev2",
        metrics_output=tmp_path / "metrics.parquet",
        analysis_output=tmp_path / "analysis.json",
        primary_analysis=None,
    )

    assert G1_MUTUAL_LOCAL_GRAPH not in loaded_conditions
    assert observed["conditions"] == (
        R0_LEGACY,
        R1_METRIC_HDBSCAN,
        R2_ANCHORED_HDBSCAN,
    )


def test_dev8_requires_a_matching_passed_frozen_dev2_selection(tmp_path) -> None:
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "phase": "dev2",
                "selected_condition": R2_ANCHORED_HDBSCAN,
                "selected_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not authorize DEV8"):
        evaluate_category_cluster_run(
            runtime_manifest=tmp_path / "runtime.json",
            gt_dir=tmp_path / "gt",
            run_root=tmp_path / "runs",
            scene_ids=DEV8_SCENE_IDS,
            taxonomy=load_taxonomy(),
            phase="dev8",
            selected_condition=R1_METRIC_HDBSCAN,
            frozen_selection_artifact=frozen,
            metrics_output=tmp_path / "metrics.parquet",
            analysis_output=tmp_path / "analysis.json",
        )


def test_dev2_writes_one_scalar_parquet_row_per_condition_and_scene(
    tmp_path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    for scene_id in DEV2_SCENE_IDS:
        for condition in (R0_LEGACY, R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN):
            path = run_root / "bank" / scene_id / condition / "bank_labels.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(
        scene_evaluation,
        "load_scene_runtime_manifest",
        lambda _: {scene_id: {} for scene_id in DEV2_SCENE_IDS},
    )
    monkeypatch.setattr(scene_evaluation, "_evaluation_scene", lambda **_: object())
    monkeypatch.setattr(scene_evaluation, "load_candidate_bank", lambda _: object())
    result = {
        "phase": "dev2",
        "conditions": {
            condition: {
                "per_scene": [
                    {
                        "scene_id": scene_id,
                        "candidate_count": 3,
                        "candidate_precision_025": 0.25,
                        "candidate_rows": [{"candidate_id": 0}],
                    }
                    for scene_id in DEV2_SCENE_IDS
                ]
            }
            for condition in (R0_LEGACY, R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN)
        },
    }
    monkeypatch.setattr(
        scene_evaluation,
        "evaluate_cluster_candidate_banks",
        lambda *_, **__: result,
    )
    written: dict[str, object] = {}
    monkeypatch.setattr(
        scene_evaluation,
        "write_rows",
        lambda path, rows: written.update(path=path, rows=list(rows)),
    )
    monkeypatch.setattr(
        scene_evaluation,
        "write_json",
        lambda path, value: written.update(analysis_path=path, analysis=value),
    )

    observed = evaluate_category_cluster_run(
        runtime_manifest=tmp_path / "runtime.json",
        gt_dir=tmp_path / "gt",
        run_root=run_root,
        scene_ids=tuple(reversed(DEV2_SCENE_IDS)),
        taxonomy=load_taxonomy(),
        phase="dev2",
        metrics_output=tmp_path / "metrics.parquet",
        analysis_output=tmp_path / "analysis.json",
    )

    rows = written["rows"]
    assert observed is result
    assert len(rows) == 3 * len(DEV2_SCENE_IDS)
    assert {
        (row["condition"], row["scene_id"])
        for row in rows
    } == {
        (condition, scene_id)
        for condition in (R0_LEGACY, R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN)
        for scene_id in DEV2_SCENE_IDS
    }
    assert all(row["phase"] == "dev2" for row in rows)
    assert all("candidate_rows" not in row for row in rows)


def test_cluster_cli_parsers_keep_gt_out_of_bank_and_audit_commands() -> None:
    parser = cli.build_parser()
    bank = parser.parse_args(
        [
            "run-category-cluster-bank",
            "--runtime-manifest",
            "runtime.json",
            "--output-root",
            "runs",
            "--category-priors",
            "priors.json",
            "--scene",
            "scene0645_00",
            "--condition",
            R1_METRIC_HDBSCAN,
        ]
    )
    audit = parser.parse_args(
        [
            "audit-category-cluster-distance",
            "--run-root",
            "runs",
            "--reference-bank-root",
            "reference-bank",
            "--reference-trace-root",
            "reference-trace",
            "--scene",
            "scene0645_00",
            "--output",
            "audit.json",
        ]
    )
    evaluate = parser.parse_args(
        [
            "evaluate-category-cluster-bank",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--phase",
            "dev2",
            "--metrics-output",
            "metrics.parquet",
            "--analysis-output",
            "analysis.json",
        ]
    )

    assert bank.condition == [R1_METRIC_HDBSCAN]
    assert not hasattr(bank, "gt_dir")
    assert not hasattr(audit, "gt_dir")
    assert evaluate.gt_dir == "gt"
    assert evaluate.phase == "dev2"


def test_three_cluster_cli_dispatchers_forward_registered_arguments(
    monkeypatch, capsys
) -> None:
    calls: dict[str, dict] = {}

    def capture(name):
        def inner(**kwargs):
            calls[name] = kwargs
            return {"action": name}

        return inner

    monkeypatch.setattr(
        category_cluster_runner,
        "run_category_cluster_bank",
        capture("run"),
    )
    monkeypatch.setattr(
        category_cluster_runner,
        "audit_category_cluster_distance",
        capture("audit"),
    )
    monkeypatch.setattr(
        scene_evaluation,
        "evaluate_category_cluster_run",
        capture("evaluate"),
    )
    monkeypatch.setattr(cli, "load_taxonomy", lambda _: "taxonomy")
    parser = cli.build_parser()

    run_args = parser.parse_args(
        [
            "run-category-cluster-bank",
            "--runtime-manifest",
            "runtime.json",
            "--output-root",
            "runs",
            "--category-priors",
            "priors.json",
            "--reference-bank-root",
            "reference",
            "--verify-determinism",
            "--python-bin",
            "python",
            "--scene",
            "scene0645_00",
            "--condition",
            R2_ANCHORED_HDBSCAN,
        ]
    )
    run_args.func(run_args)
    audit_args = parser.parse_args(
        [
            "audit-category-cluster-distance",
            "--run-root",
            "runs",
            "--reference-bank-root",
            "reference-bank",
            "--reference-trace-root",
            "reference-trace",
            "--scene",
            "scene0645_00",
            "--output",
            "audit.json",
        ]
    )
    audit_args.func(audit_args)
    evaluate_args = parser.parse_args(
        [
            "evaluate-category-cluster-bank",
            "--runtime-manifest",
            "runtime.json",
            "--gt-dir",
            "gt",
            "--run-root",
            "runs",
            "--scene",
            "scene0645_00",
            "--phase",
            "dev8",
            "--selected-condition",
            R2_ANCHORED_HDBSCAN,
            "--frozen-selection-artifact",
            "frozen.json",
            "--metrics-output",
            "metrics.parquet",
            "--analysis-output",
            "analysis.json",
        ]
    )
    evaluate_args.func(evaluate_args)

    assert calls["run"]["conditions"] == [R2_ANCHORED_HDBSCAN]
    assert calls["run"]["reference_bank_root"] == Path("reference")
    assert calls["run"]["verify_determinism"] is True
    assert calls["run"]["determinism_reference"] is None
    assert calls["audit"]["reference_trace_root"] == Path("reference-trace")
    assert calls["evaluate"]["taxonomy"] == "taxonomy"
    assert calls["evaluate"]["selected_condition"] == R2_ANCHORED_HDBSCAN
    assert calls["evaluate"]["frozen_selection_artifact"] == Path("frozen.json")
    assert capsys.readouterr().out.count('"action"') == 3


def test_distance_audit_forwards_measured_corrected_contract(
    tmp_path, monkeypatch
) -> None:
    sample_rank = np.asarray([0, 1, -1], dtype=np.int64)
    labels = np.asarray([0, 0, -1], dtype=np.int64)
    membership = np.asarray([0.9, 0.8, 0.0], dtype=np.float64)

    def fake_bank(path):
        condition = Path(path).name
        diagnostics = {
            "determinism_measured_this_scene": True,
            "determinism_contract_verified": True,
            "determinism_violation_count": 0,
        }
        if condition in {R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN}:
            diagnostics.update({
                "global_typical_diag_m": 1.25,
                "raw_member_count": 2,
                "raw_member_retained_count": 2,
                "core_outside_full_count": 0,
                "distance_matrix_count": 1,
                "distance_all_finite": True,
                "distance_symmetry_max_abs": 0.0,
                "distance_diagonal_max_abs": 0.0,
                "distance_min": 0.0,
                "distance_max": 0.75,
                "corrected_distance_contract_measured": True,
                "corrected_distance_contract_passed": True,
            })
        return SimpleNamespace(candidates=({},), diagnostics=diagnostics)

    monkeypatch.setattr(category_cluster_runner, "load_candidate_bank", fake_bank)
    monkeypatch.setattr(
        category_cluster_runner,
        "compare_candidate_bank_identity",
        lambda *_: SimpleNamespace(
            matches=True, mismatches=(), max_abs_differences={}
        ),
    )
    monkeypatch.setattr(
        category_cluster_runner,
        "load_candidate_formation_trace",
        lambda *_: SimpleNamespace(
            sample_rank=sample_rank,
            hdbscan_labels=labels,
            hdbscan_membership=membership,
        ),
    )
    monkeypatch.setattr(
        category_cluster_runner,
        "load_cluster_raw_audit",
        lambda *_: {
            "sample_rank": sample_rank,
            "hdbscan_labels": labels,
            "hdbscan_membership": membership,
        },
    )

    observed = category_cluster_runner.audit_category_cluster_distance(
        run_root=tmp_path / "run",
        scene_ids=("scene0645_00",),
        reference_bank_root=tmp_path / "reference-bank",
        reference_trace_root=tmp_path / "reference-trace",
        output_path=tmp_path / "audit.json",
    )

    assert observed["r0_identity_passed"] is True
    assert observed["corrected_distance_contract_passed"] is True
    assert observed["corrected_distance_contract_measured"] is True
    assert observed["determinism_passed"] is True
    for row in observed["scenes"][0]["corrected_conditions"]:
        assert row["distance_matrix_count"] == 1
        assert row["distance_all_finite"] is True
        assert row["distance_symmetry_max_abs"] == 0.0
        assert row["distance_diagonal_max_abs"] == 0.0
        assert row["corrected_distance_contract_passed"] is True
        assert row["corrected_distance_contract_measured"] is True
        assert row["determinism_measured_this_scene"] is True


def test_distance_audit_rejects_unmeasured_empty_corrected_matrices(
    tmp_path, monkeypatch
) -> None:
    sample_rank = np.asarray([0, 1, -1], dtype=np.int64)
    labels = np.asarray([0, 0, -1], dtype=np.int64)
    membership = np.asarray([0.9, 0.8, 0.0], dtype=np.float64)

    def fake_bank(path):
        condition = Path(path).name
        diagnostics = {
            "determinism_measured_this_scene": True,
            "determinism_contract_verified": True,
            "determinism_violation_count": 0,
        }
        if condition in {R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN}:
            diagnostics.update(
                {
                    "global_typical_diag_m": 1.25,
                    "distance_matrix_count": 0,
                    "corrected_distance_contract_measured": False,
                    # A stale truthy stamp must not make an empty measurement pass.
                    "corrected_distance_contract_passed": True,
                }
            )
        return SimpleNamespace(candidates=(), diagnostics=diagnostics)

    monkeypatch.setattr(category_cluster_runner, "load_candidate_bank", fake_bank)
    monkeypatch.setattr(
        category_cluster_runner,
        "compare_candidate_bank_identity",
        lambda *_: SimpleNamespace(matches=True, mismatches=(), max_abs_differences={}),
    )
    monkeypatch.setattr(
        category_cluster_runner,
        "load_candidate_formation_trace",
        lambda *_: SimpleNamespace(
            sample_rank=sample_rank,
            hdbscan_labels=labels,
            hdbscan_membership=membership,
        ),
    )
    monkeypatch.setattr(
        category_cluster_runner,
        "load_cluster_raw_audit",
        lambda *_: {
            "sample_rank": sample_rank,
            "hdbscan_labels": labels,
            "hdbscan_membership": membership,
        },
    )

    with pytest.raises(ValueError, match="corrected distance contract failed"):
        category_cluster_runner.audit_category_cluster_distance(
            run_root=tmp_path / "run",
            scene_ids=("scene0645_00",),
            reference_bank_root=tmp_path / "reference-bank",
            reference_trace_root=tmp_path / "reference-trace",
            output_path=tmp_path / "audit.json",
        )
