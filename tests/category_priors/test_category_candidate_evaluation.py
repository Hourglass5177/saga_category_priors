from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from category_priors import category_candidate_evaluation as candidate_eval
from category_priors.category_denoise import CandidateBank
from category_priors.io import sha256_file, write_json


def _write_frozen_arm(
    tmp_path: Path,
    run_root: Path,
    condition: str = "C1-consistent-envelope",
) -> Path:
    selection = tmp_path / "repair_dev2.analysis.json"
    write_json(
        selection,
        {
            "schema": candidate_eval.REPAIR_ANALYSIS_SCHEMA,
            "phase": "dev2",
            "scene_ids": list(candidate_eval.DEV2_SCENE_IDS),
            "selected_condition": condition,
            "dev2_arm_gates": {condition: {"passed": True}},
        },
    )
    artifact = tmp_path / "frozen_repair_arm.json"
    write_json(
        artifact,
        {
            "schema": candidate_eval.FROZEN_REPAIR_ARM_SCHEMA,
            "condition": condition,
            "selected_on_scene_ids": list(candidate_eval.DEV2_SCENE_IDS),
            "sample_cap": 5000,
            "run_root": str(run_root.resolve()),
            "selection_analysis": str(selection.resolve()),
            "selection_analysis_sha256": sha256_file(selection),
        },
    )
    return artifact


def _scene_metrics(scene_id: str, condition: str, phase: str) -> dict[str, Any]:
    if phase == "dev8":
        repaired = condition != "C0-legacy"
        candidate_count = 2
        count_025 = 2 if repaired else 0
        count_050 = 2 if repaired else 0
        unsupported = 0.02 if repaired else 0.20
        tiny_recall = 1.0 if repaired else 0.0
        best_iou = [0.8 if repaired else 0.0]
    elif condition == "C0-legacy":
        candidate_count = 4
        count_025 = 2
        count_050 = 1
        unsupported = 0.20
        tiny_recall = 0.5
        best_iou = [0.7, 0.2]
    elif condition == "C1-consistent-envelope":
        candidate_count = 3
        count_025 = 2
        count_050 = 1
        unsupported = 0.20
        tiny_recall = 0.5
        # This deliberately violates the old, unregistered per-GT drop gate.
        best_iou = [0.5, 0.8]
    else:
        candidate_count = 6
        count_025 = 2
        count_050 = 1
        unsupported = 0.20
        tiny_recall = 0.5
        best_iou = [0.7, 0.2]
    return {
        "scene_id": scene_id,
        "candidate_count": candidate_count,
        "same_class_iou_025_count": count_025,
        "same_class_iou_050_count": count_050,
        "candidate_precision_025": count_025 / candidate_count,
        "candidate_precision_050": count_050 / candidate_count,
        "unsupported_fraction": unsupported,
        "tiny_small_gt_count": 2,
        "tiny_small_recall_025": tiny_recall,
        "tiny_small_recall_050": tiny_recall,
        "core_subset_full_violation_count": 0,
        "best_iou_by_gt": best_iou,
        "candidate_rows": [
            {
                "full_point_count": 1,
                "unsupported_fraction": unsupported,
            }
            for _ in range(candidate_count)
        ],
    }


def _patch_evaluation_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase: str,
) -> tuple[list[tuple[str, tuple[str, ...]]], list[list[dict[str, Any]]]]:
    loaded: list[tuple[str, tuple[str, ...]]] = []
    written_rows: list[list[dict[str, Any]]] = []

    monkeypatch.setattr(
        candidate_eval,
        "load_scene_runtime_manifest",
        lambda _path: {
            scene_id: {} for scene_id in candidate_eval.DEV8_SCENE_IDS
        },
    )
    monkeypatch.setattr(
        candidate_eval,
        "_scene_context",
        lambda **_kwargs: {},
    )

    def load_banks(
        _root: Path, scene_id: str, conditions: tuple[str, ...]
    ) -> dict[str, str]:
        normalized = tuple(conditions)
        loaded.append((scene_id, normalized))
        return {condition: condition for condition in normalized}

    monkeypatch.setattr(candidate_eval, "_load_scene_banks", load_banks)
    monkeypatch.setattr(
        candidate_eval,
        "_candidate_scene_metrics",
        lambda scene_id, bank, _context, _taxonomy: _scene_metrics(
            scene_id, str(bank), phase
        ),
    )
    monkeypatch.setattr(
        candidate_eval,
        "write_rows",
        lambda _path, rows: written_rows.append([dict(row) for row in rows]),
    )
    monkeypatch.setattr(candidate_eval, "write_json", lambda _path, _value: None)
    return loaded, written_rows


def _common(tmp_path: Path, run_root: Path) -> dict[str, Any]:
    return {
        "runtime_manifest": tmp_path / "runtime.json",
        "gt_dir": tmp_path / "gt",
        "run_root": run_root,
        "taxonomy": object(),
        "metrics_output": tmp_path / "metrics.jsonl",
        "analysis_output": tmp_path / "analysis.json",
    }


def test_dev8_loads_and_writes_only_c0_and_the_frozen_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    artifact = _write_frozen_arm(tmp_path, run_root)
    loaded, written_rows = _patch_evaluation_io(monkeypatch, phase="dev8")

    analysis = candidate_eval.evaluate_category_candidates(
        **_common(tmp_path, run_root),
        scene_ids=tuple(reversed(candidate_eval.DEV8_SCENE_IDS)),
        phase="dev8",
        frozen_repair_artifact=artifact,
    )

    expected = ("C0-legacy", "C1-consistent-envelope")
    assert loaded == [(scene_id, expected) for scene_id in candidate_eval.DEV8_SCENE_IDS]
    assert set(analysis["conditions"]) == set(expected)
    assert "C2-raw-anchored-envelope" not in analysis["conditions"]
    assert analysis["scene_ids"] == list(candidate_eval.DEV8_SCENE_IDS)
    assert analysis["selected_condition"] == "C1-consistent-envelope"
    assert analysis["dev2_arm_gates"] == {}
    assert analysis["dev8_health_gate"]["passed"] is True
    assert [row["condition"] for row in written_rows[0]] == list(expected)


def test_frozen_artifact_is_authoritative_and_bound_to_the_run_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    artifact = _write_frozen_arm(tmp_path, run_root)
    common = _common(tmp_path, run_root)

    with pytest.raises(ValueError, match="disagrees"):
        candidate_eval.evaluate_category_candidates(
            **common,
            scene_ids=candidate_eval.DEV8_SCENE_IDS,
            phase="dev8",
            selected_condition="C2-raw-anchored-envelope",
            frozen_repair_artifact=artifact,
        )
    with pytest.raises(ValueError, match="different run_root"):
        candidate_eval.evaluate_category_candidates(
            **{**common, "run_root": tmp_path / "other-run"},
            scene_ids=candidate_eval.DEV8_SCENE_IDS,
            phase="dev8",
            frozen_repair_artifact=artifact,
        )
    write_json(tmp_path / "repair_dev2.analysis.json", {"tampered": True})
    with pytest.raises(ValueError, match="changed after freezing"):
        candidate_eval.evaluate_category_candidates(
            **common,
            scene_ids=candidate_eval.DEV8_SCENE_IDS,
            phase="dev8",
            frozen_repair_artifact=artifact,
        )


def test_registered_scene_sets_reject_duplicates_and_subsets(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    common = _common(tmp_path, run_root)
    with pytest.raises(ValueError, match="duplicate"):
        candidate_eval.evaluate_category_candidates(
            **common,
            scene_ids=("scene0645_00", "scene0645_00"),
            phase="dev2",
        )
    with pytest.raises(ValueError, match="exactly the registered"):
        candidate_eval.evaluate_category_candidates(
            **common,
            scene_ids=("scene0645_00",),
            phase="dev2",
        )
    with pytest.raises(ValueError, match="DEV2 or DEV8"):
        candidate_eval._registered_scene_ids(("scene0645_00",))
    artifact = _write_frozen_arm(tmp_path, run_root)
    with pytest.raises(ValueError, match="exactly the registered"):
        candidate_eval.evaluate_category_candidates(
            **common,
            scene_ids=candidate_eval.DEV2_SCENE_IDS,
            phase="dev8",
            frozen_repair_artifact=artifact,
        )


def test_dev2_scene_improvement_and_gt_drop_are_registered_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    _patch_evaluation_io(monkeypatch, phase="dev2")

    analysis = candidate_eval.evaluate_category_candidates(
        **_common(tmp_path, run_root),
        scene_ids=candidate_eval.DEV2_SCENE_IDS,
        phase="dev2",
    )

    gate = analysis["dev2_arm_gates"]["C1-consistent-envelope"]
    assert gate["checks"]["at_least_one_scene_improved"] is True
    assert gate["checks"]["per_gt_drop_at_most_0.05"] is False
    assert gate["passed"] is False
    assert analysis["selected_condition"] is None


def test_postprocess_survival_is_loaded_and_classified(
    tmp_path: Path,
) -> None:
    root = tmp_path / "uniform"
    scene_id = candidate_eval.DEV2_SCENE_IDS[0]
    write_json(
        root / scene_id / "diagnostics.json",
        {
            "category_denoise": {
                "candidate_survival": [
                    {
                        "candidate_id": 4,
                        "accepted": True,
                        "survived_post_knn": True,
                        "survived_post_filter": False,
                    },
                    {
                        "candidate_id": 5,
                        "accepted": True,
                        "survived_post_knn": False,
                        "survived_post_filter": False,
                    },
                ]
            }
        },
    )

    rows = candidate_eval._load_scene_postprocess_survival(root, scene_id)
    assert candidate_eval._postprocess_loss_stage(rows[4]) == "post_filter"
    assert candidate_eval._postprocess_loss_stage(rows[5]) == "post_knn"
    assert candidate_eval._candidate_failure_status(
        sampled_count=10,
        best_raw_f1=0.8,
        retained_candidate_id=4,
        best_raw_iou=0.6,
        full_iou=0.6,
        best_raw_precision=0.8,
        full_precision=0.8,
        postprocess_loss_stage="post_filter",
    ) == "postprocess_loss"
    # Earlier formation failures retain causal precedence over postprocess loss.
    assert candidate_eval._candidate_failure_status(
        sampled_count=10,
        best_raw_f1=0.8,
        retained_candidate_id=4,
        best_raw_iou=0.8,
        full_iou=0.6,
        best_raw_precision=0.8,
        full_precision=0.8,
        postprocess_loss_stage="post_filter",
    ) == "full_assignment_loss"


def test_c0_survival_replay_projects_historical_core_to_exported_full() -> None:
    bank = CandidateBank(
        class_names=("chair",),
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.asarray([-1, -1, -1, -1], dtype=np.int64),
        semantic_top1=np.zeros(4, dtype=np.int64),
        semantic_top1_score=np.ones(4, dtype=np.float64),
        branch_full_labels=np.asarray([0, 0, 0, -1], dtype=np.int64),
        # Point 3 is a historical raw core member that legacy full assignment
        # moved outside candidate 0.  Diagnosis must preserve and report this
        # anomaly without making the common replay reject C0 itself.
        branch_core_labels=np.asarray([0, 0, -1, 0], dtype=np.int64),
        assignment_confidence=np.asarray([0.9, 0.8, 0.7, 0.6]),
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "base_score": 0.8,
            },
        ),
        diagnostics={},
    )

    survival = candidate_eval._all_c0_postprocess_survival(
        bank, np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )

    assert survival[0]["accepted"] is True
