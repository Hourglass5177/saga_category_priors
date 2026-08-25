from __future__ import annotations

from pathlib import Path
from typing import Any

from category_priors.v10_orchestrator import (
    STRUCTURE_CONDITIONS,
    run_v10_orchestrator,
)
from category_priors.v10_pipeline import DEV8, HOLDOUT5
from category_priors.v10_replay import V10_PRIOR_CONDITIONS


def _structure_row(*, candidate_count: int = 10, precision: float = 0.20) -> dict[str, Any]:
    return {
        "candidate_count": candidate_count,
        "geometric_match_050_count": 7,
        "geometric_candidate_precision_025": precision,
        "geometric_tiny_small_recall_025": 0.30,
        "identifiable_association_precision": 0.60,
    }


def _uniform_health(*, orphan_count: int = 0) -> dict[str, dict[str, Any]]:
    return {
        "b1_fixed": {
            "gaussian_micro_precision": 0.30,
            "unsupported_instance_fraction": 0.40,
            "gt_recall": 0.60,
            "map_50_95": 0.050,
            "ap50": 0.100,
            "predicted_instance_count": 20,
        },
        "bank": {
            "geometric_match_050_count": 18,
            "geometric_match_050_scene_count": 5,
            "same_class_match_050_count": 14,
            "same_class_match_050_scene_count": 5,
            "same_class_candidate_precision_025": 0.20,
            "tiny_small_recall_025": 0.25,
            "gaussian_micro_precision": 0.36,
            "unsupported_instance_fraction": 0.35,
            "gt_recall": 0.56,
            "map_50_95": 0.050,
            "ap50": 0.100,
            "predicted_instance_count": 22,
            "score_iou_spearman": 0.30,
            "orphan_gaussian_count": orphan_count,
            "negative_metadata_count": 0,
        },
    }


def _comparison_rows(scene_ids: tuple[str, ...], delta: float = 0.003) -> dict[str, Any]:
    uniform = [
        {
            "scene_id": scene,
            "physical_scene_id": scene.split("_")[0],
            "map_50_95": 0.10,
            "tiny_small_recall_050": 0.20,
            "false_positive_count": 10,
            "true_positive_count": 10,
        }
        for scene in scene_ids
    ]
    data = [
        {
            **row,
            "map_50_95": row["map_50_95"] + delta,
            "tiny_small_recall_050": row["tiny_small_recall_050"] + 0.02,
            "false_positive_count": 11,
        }
        for row in uniform
    ]
    return {"uniform_rows": uniform, "data_rows": data}


class FakeHooks:
    def __init__(
        self,
        *,
        fail_stage1: bool = False,
        fail_stage2: bool = False,
        fail_prior: bool = False,
    ) -> None:
        self.fail_stage1 = fail_stage1
        self.fail_stage2 = fail_stage2
        self.fail_prior = fail_prior
        self.calls: list[tuple[str, Any]] = []

    def closeout_v9(self) -> dict[str, Any]:
        self.calls.append(("closeout", None))
        return {"passed": True, "v9_identity_conclusion_withdrawn": True}

    def ensure_banks(self, *, scene_ids, structure_conditions):
        self.calls.append(("banks", (tuple(scene_ids), tuple(structure_conditions))))
        return {"complete": True, "count": len(scene_ids) * len(structure_conditions)}

    def audit_dev2_structures(self, *, scene_ids, structure_conditions):
        self.calls.append(("audit_dev2", (tuple(scene_ids), tuple(structure_conditions))))
        rows = {condition: _structure_row() for condition in structure_conditions}
        rows["P0R0"] = _structure_row(candidate_count=10)
        rows["VC1"] = _structure_row(
            candidate_count=12,
            precision=0.09 if self.fail_stage1 else 0.20,
        )
        return rows

    def classifier_metrics_dev8(self, *, scene_ids, structure_condition):
        self.calls.append(("classifiers", (tuple(scene_ids), structure_condition)))
        return {
            "mv-label": {
                "geometric_candidate_match_025_count": 100,
                "late_classifier_correct_025_count": 60,
            },
            "codebook": {
                "geometric_candidate_match_025_count": 100,
                "late_classifier_correct_025_count": 61,
            },
        }

    def threshold_sweep_dev2(
        self, *, scene_ids, structure_condition, classifier, thresholds
    ):
        self.calls.append(("thresholds", (tuple(scene_ids), classifier)))
        return [
            {
                "acceptance_threshold": value,
                "map_50_95": 0.20 if value in {0.15, 0.20} else 0.10,
                "structure_passed": True,
            }
            for value in thresholds
        ]

    def uniform_health_inputs_dev8(
        self, *, scene_ids, structure_condition, classifier, acceptance_threshold
    ):
        self.calls.append(
            ("uniform_health", (tuple(scene_ids), classifier, acceptance_threshold))
        )
        return _uniform_health(orphan_count=1 if self.fail_stage2 else 0)

    def prior_factorial_dev8(
        self,
        *,
        scene_ids,
        structure_condition,
        classifier,
        acceptance_threshold,
        prior_conditions,
    ):
        self.calls.append(("prior", (tuple(scene_ids), tuple(prior_conditions))))
        comparison = _comparison_rows(tuple(scene_ids))
        uniform = comparison["uniform_rows"]
        result: dict[str, Any] = {}
        for condition in prior_conditions:
            if condition == "U000":
                rows = uniform
                deltas: list[float] = []
                changed = False
                mean_map = 0.10
            elif self.fail_prior:
                rows = uniform
                deltas = []
                changed = False
                mean_map = 0.10
            else:
                rows = comparison["data_rows"]
                deltas = [0.02] + [0.0] * 9
                changed = True
                mean_map = 0.104 if condition == "D100" else 0.103
            result[condition] = {
                "rows": rows,
                "aggregate": {
                    "map_50_95": mean_map,
                    "tiny_small_recall_050": 0.22,
                    "ap50": 0.30,
                },
                "candidate_score_deltas": deltas,
                "accepted_or_ownership_changed": changed,
            }
        return result

    def holdout5_comparison(
        self,
        *,
        scene_ids,
        structure_condition,
        classifier,
        acceptance_threshold,
        data_condition,
    ):
        self.calls.append(("holdout", (tuple(scene_ids), data_condition)))
        return _comparison_rows(tuple(scene_ids))

    def tune24_comparison(
        self,
        *,
        scene_ids,
        structure_condition,
        classifier,
        acceptance_threshold,
        data_condition,
    ):
        self.calls.append(("tune24", (tuple(scene_ids), data_condition)))
        return _comparison_rows(tuple(scene_ids))

    def final48_bootstrap(
        self,
        *,
        scene_ids,
        structure_condition,
        classifier,
        acceptance_threshold,
        data_condition,
    ):
        self.calls.append(("final48", (tuple(scene_ids), data_condition)))
        return {
            "delta_map_50_95": 0.003,
            "paired_bootstrap_ci95": [0.001, 0.005],
        }


def _run(tmp_path: Path, hooks: FakeHooks) -> dict[str, Any]:
    tune = tuple(f"scene{index:04d}_00" for index in range(24))
    final = tuple(f"locked{index:04d}_00" for index in range(48))
    return run_v10_orchestrator(
        hooks=hooks,
        artifacts_root=tmp_path,
        git_commit="commit-v10",
        tune24_scene_ids=tune,
        final48_scene_ids=final,
    )


def test_stage1_failure_writes_v10b_and_never_tests_priors(tmp_path: Path) -> None:
    hooks = FakeHooks(fail_stage1=True)
    result = _run(tmp_path, hooks)
    assert result["state"] == "stopped"
    assert result["approval_required"]
    assert not result["category_prior_tested"]
    assert Path(result["v10b_proposal"]).exists()
    assert "请批准或拒绝" in Path(result["v10b_proposal"]).read_text("utf-8")
    assert "prior" not in [name for name, _ in hooks.calls]


def test_stage2_failure_writes_v10b_after_dev8_and_stops(tmp_path: Path) -> None:
    hooks = FakeHooks(fail_stage2=True)
    result = _run(tmp_path, hooks)
    assert result["state"] == "stopped"
    assert result["checkpoint"].startswith("stage2-dev8-uniform-health")
    assert result["approval_required"]
    assert "prior" not in [name for name, _ in hooks.calls]


def test_prior_failure_is_a_prior_result_not_an_identity_approval(tmp_path: Path) -> None:
    hooks = FakeHooks(fail_prior=True)
    result = _run(tmp_path, hooks)
    assert result["state"] == "stopped"
    assert result["checkpoint"] == "stage3-prior-no-dev8-benefit"
    assert result["category_prior_tested"]
    assert not result["approval_required"]
    assert not (tmp_path / "V10B_IDENTITY_TRAINING_PROPOSAL.md").exists()


def test_complete_path_runs_registered_sequence_and_final_gate(tmp_path: Path) -> None:
    hooks = FakeHooks()
    result = _run(tmp_path, hooks)
    assert result["state"] == "complete"
    assert result["category_prior_supported"]
    assert result["selected_classifier"] == "mv-label"
    assert result["acceptance_threshold"] == 0.20
    assert result["selected_data_condition"] == "D100"
    assert result["final48_gate"]["passed"]

    bank_calls = [payload for name, payload in hooks.calls if name == "banks"]
    assert bank_calls[0] == (("scene0645_00", "scene0025_01"), STRUCTURE_CONDITIONS)
    assert bank_calls[1] == (DEV8, ("VC1",))
    prior_call = next(payload for name, payload in hooks.calls if name == "prior")
    assert prior_call == (DEV8, V10_PRIOR_CONDITIONS)
    holdout_call = next(payload for name, payload in hooks.calls if name == "holdout")
    assert holdout_call[0] == HOLDOUT5
    assert (tmp_path / "v10_orchestrator_status.json").exists()
