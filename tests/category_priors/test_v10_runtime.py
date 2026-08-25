from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from category_priors.io import write_json
from category_priors.v10_pipeline import DEV2, DEV8
from category_priors.v10_runtime import FilesystemV10Config, FilesystemV10Hooks


def _config(tmp_path: Path, *, ensure_lifting=None) -> FilesystemV10Config:
    closeout = tmp_path / "v10_v9_closeout.json"
    write_json(closeout, {"passed": True})
    return FilesystemV10Config(
        runtime_manifest=tmp_path / "runtime.json",
        gt_dir=tmp_path / "gt",
        lifting_root=tmp_path / "lifting",
        bank_root=tmp_path / "banks",
        replay_root=tmp_path / "replay",
        artifacts_root=tmp_path / "artifacts",
        category_priors=tmp_path / "priors.json",
        size_bins=tmp_path / "sizes.json",
        b1_fixed_prediction_root=tmp_path / "b1",
        b1_fixed_condition="B1-fixed",
        v9_closeout=closeout,
        git_commit="commit-v10",
        locked_runtime_manifest=tmp_path / "locked-runtime.json",
        locked_gt_dir=tmp_path / "locked-gt",
        ensure_lifting=ensure_lifting,
    )


def _official_analysis(scene_ids, conditions) -> dict[str, Any]:
    blocks = {}
    for condition in conditions:
        delta = 0.003 if condition != "U000" else 0.0
        rows = [
            {
                "scene_id": scene,
                "map_50_95": 0.10 + delta,
                "tiny_small_recall_050": 0.20 + (0.02 if delta else 0.0),
                "false_positive_count": 11 if delta else 10,
                "true_positive_count": 10,
            }
            for scene in scene_ids
        ]
        blocks[condition] = {
            "metrics": {
                "map_50_95": 0.10 + delta,
                "ap50": 0.20 + delta,
                "predicted_instance_count": 20,
                "gaussian_micro_precision": 0.40,
                "unsupported_instance_fraction": 0.20,
                "gt_recall": 0.60,
                "orphan_gaussian_count": 0,
                "negative_metadata_count": 0,
                "tiny_small_recall_050": 0.20 + (0.02 if delta else 0.0),
            },
            "per_scene": rows,
        }
    return {"schema": "saga-v10-object-system-analysis-v1", "conditions": blocks}


def test_missing_lifting_never_trains_or_downloads_implicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_runtime as module

    monkeypatch.setattr(
        module, "compatible_lifting_bank_is_complete", lambda *args, **kwargs: False
    )
    hooks = FilesystemV10Hooks(_config(tmp_path))
    with pytest.raises(RuntimeError, match="DEV2 must reuse"):
        hooks.ensure_banks(scene_ids=(DEV2[0],), structure_conditions=("VC1",))
    with pytest.raises(RuntimeError, match="No training or download was started"):
        hooks.ensure_banks(scene_ids=("scene9999_00",), structure_conditions=("VC1",))


def test_nondev_missing_lifting_uses_explicit_callback_then_resumable_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_runtime as module

    complete: set[str] = set()
    calls: list[Any] = []

    def callback(scene_id: str) -> None:
        calls.append(("lifting", scene_id))
        complete.add(scene_id)

    def fake_complete(path: Path, *, expected_scene_id=None, **kwargs) -> bool:
        return str(expected_scene_id) in complete

    def fake_run(**kwargs):
        calls.append(("banks", kwargs))
        return {"schema": "fake", "runs": []}

    monkeypatch.setattr(module, "compatible_lifting_bank_is_complete", fake_complete)
    monkeypatch.setattr(module, "run_v10_banks", fake_run)
    hooks = FilesystemV10Hooks(_config(tmp_path, ensure_lifting=callback))
    result = hooks.ensure_banks(
        scene_ids=("scene9999_00",), structure_conditions=("VC1",)
    )
    assert result["schema"] == "fake"
    assert calls[0] == ("lifting", "scene9999_00")
    assert calls[1][0] == "banks"
    assert calls[1][1]["conditions"] == ("VC1",)


def test_dev2_audit_adapts_gate_metrics_and_reuses_named_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_runtime as module

    calls: list[dict[str, Any]] = []

    def fake_audit(**kwargs):
        calls.append(kwargs)
        return {
            "schema": "saga-v10-association-audit-v1",
            "scene_ids": list(DEV2),
            "conditions": {
                f"{condition}/mv-label": {
                    "gate_metrics": {
                        "candidate_count": 10,
                        "geometric_match_050_count": 6,
                        "geometric_candidate_precision_025": 0.2,
                        "geometric_tiny_small_recall_025": 0.3,
                        "identifiable_association_precision": 0.6,
                    }
                }
                for condition in ("P0R0", "P1R0", "P0R1", "P1R1", "VC1")
            },
        }

    monkeypatch.setattr(module, "audit_v10_associations", fake_audit)
    hooks = FilesystemV10Hooks(_config(tmp_path))
    monkeypatch.setattr(
        hooks, "_audit_source_identity", lambda **_kwargs: {"test_fixture": True}
    )
    conditions = ("P0R0", "P1R0", "P0R1", "P1R1", "VC1")
    result = hooks.audit_dev2_structures(
        scene_ids=DEV2, structure_conditions=conditions
    )
    assert set(result) == set(conditions)
    assert result["VC1"]["identifiable_association_precision"] == 0.6
    assert calls[0]["rows_output"].name == "v10_association_funnel2.parquet"
    assert calls[0]["classifiers"] == ("mv-label",)


def test_health_and_prior_factorial_are_shaped_for_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    hooks = FilesystemV10Hooks(_config(tmp_path))
    official = _official_analysis(DEV8, ("U000", "D100"))
    b1 = _official_analysis(DEV8, ("B1-fixed",))
    b1["conditions"]["B1-fixed"]["metrics"].update(
        {
            "gaussian_micro_precision": 0.30,
            "unsupported_instance_fraction": 0.35,
            "gt_recall": 0.62,
        }
    )
    final = {
        "candidate_match_050_count": 18,
        "candidate_match_050_scene_count": 5,
        "same_class_candidate_match_050_count": 14,
        "same_class_candidate_match_050_scene_count": 5,
        "same_class_candidate_precision_025": 0.2,
        "same_class_tiny_small_recall_025": 0.25,
    }
    monkeypatch.setattr(hooks, "_evaluate_v10", lambda **kwargs: official)
    monkeypatch.setattr(hooks, "_evaluate_b1", lambda *args, **kwargs: b1)
    monkeypatch.setattr(
        hooks,
        "_dev8_audit",
        lambda: (
            {
                "conditions": {
                    "VC1/mv-label": {"stages": {"final_candidate": final}}
                }
            },
            tmp_path / "rows.parquet",
        ),
    )
    monkeypatch.setattr(hooks, "_score_iou", lambda **kwargs: 0.3)
    health = hooks.uniform_health_inputs_dev8(
        scene_ids=DEV8,
        structure_condition="VC1",
        classifier="mv-label",
        acceptance_threshold=0.2,
    )
    assert health["bank"]["same_class_match_050_count"] == 14
    assert health["bank"]["score_iou_spearman"] == 0.3
    assert health["b1_fixed"]["gaussian_micro_precision"] == 0.30

    def fake_diagnostics(*, scene_id, condition, **kwargs):
        accepted = [] if condition == "U000" else [0]
        labels = [-1, -1] if condition == "U000" else [0, 0]
        instances = {} if condition == "U000" else {"0": {"candidate_id": 0}}
        score = 0.5 if condition == "U000" else 0.6
        return (
            {
                "candidate_scores": [{"candidate_id": 0, "score": score}],
                "accepted_candidate_ids": accepted,
                "instances": instances,
            },
            {"point_labels": labels},
        )

    monkeypatch.setattr(hooks, "_replay_diagnostics", fake_diagnostics)
    factorial = hooks.prior_factorial_dev8(
        scene_ids=DEV8,
        structure_condition="VC1",
        classifier="mv-label",
        acceptance_threshold=0.2,
        prior_conditions=("U000", "D100"),
    )
    assert factorial["D100"]["candidate_score_deltas"] == pytest.approx([0.1] * 8)
    assert factorial["D100"]["accepted_or_ownership_changed"]
    assert len(factorial["D100"]["rows"]) == 8


def test_final48_uses_locked_protocol_and_official_paired_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from category_priors import v10_runtime as module

    hooks = FilesystemV10Hooks(_config(tmp_path))
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        hooks,
        "ensure_banks",
        lambda **kwargs: calls.append(("banks", kwargs)) or {},
    )
    monkeypatch.setattr(
        hooks,
        "_evaluate_v10",
        lambda **kwargs: calls.append(("evaluate", kwargs)) or {},
    )
    monkeypatch.setattr(
        hooks, "_replay_source_identity", lambda **_kwargs: {"test_fixture": True}
    )

    def fake_bootstrap(**kwargs):
        calls.append(("bootstrap", kwargs))
        return {
            "delta_map_50_95": 0.003,
            "paired_bootstrap_ci95": [0.001, 0.005],
        }

    monkeypatch.setattr(module, "paired_scannet_bootstrap_from_predictions", fake_bootstrap)
    scenes = tuple(f"locked{index:04d}_00" for index in range(48))
    result = hooks.final48_bootstrap(
        scene_ids=scenes,
        structure_condition="VC1",
        classifier="mv-label",
        acceptance_threshold=0.2,
        data_condition="D100",
    )
    assert result["delta_map_50_95"] == 0.003
    evaluate_call = next(payload for name, payload in calls if name == "evaluate")
    assert evaluate_call["runtime_manifest"] == hooks.config.locked_runtime_manifest
    bootstrap_call = next(payload for name, payload in calls if name == "bootstrap")
    assert bootstrap_call["samples"] == 10_000
    assert bootstrap_call["prediction_root"].parts[-2:] == ("VC1", "mv-label")
