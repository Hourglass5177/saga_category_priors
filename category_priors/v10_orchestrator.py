from __future__ import annotations

"""Recoverable, preregistered stage controller for the SAGA V10 experiment.

The controller deliberately owns decisions, not GPU/runtime details.  Concrete
deployments inject idempotent hooks that build banks and evaluate their outputs.
This keeps ground truth out of the ObjectBank worker while making it impossible
for a shell script to continue past a failed registered gate.
"""

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .io import write_json
from .v10_pipeline import (
    DEV2,
    DEV8,
    HOLDOUT5,
    PAIR_RECONSTRUCTION_ARMS,
    PRIOR_THRESHOLDS,
    VIEW_CONSENSUS_ARM,
    final48_gate,
    holdout5_gate,
    physical_scene_macro_gate,
    select_best_prior_condition,
    select_late_classifier,
    select_pair_reconstruction_arm,
    select_uniform_threshold,
    stage1_structure_gate,
    stage2_uniform_health_gate,
    stage3_prior_gate,
    write_v10b_identity_training_proposal,
)
from .v10_replay import V10_PRIOR_CONDITIONS


STRUCTURE_CONDITIONS = (*PAIR_RECONSTRUCTION_ARMS, VIEW_CONSENSUS_ARM)


class V10OrchestratorHooks(Protocol):
    """Idempotent runtime/evaluation operations required by the controller.

    Hook implementations may reuse complete artifacts, but must raise on a
    damaged output instead of silently returning partial metrics.
    """

    def closeout_v9(self) -> Mapping[str, Any]: ...

    def ensure_banks(
        self,
        *,
        scene_ids: Sequence[str],
        structure_conditions: Sequence[str],
    ) -> Mapping[str, Any]: ...

    def audit_dev2_structures(
        self,
        *,
        scene_ids: Sequence[str],
        structure_conditions: Sequence[str],
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def classifier_metrics_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def threshold_sweep_dev2(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        thresholds: Sequence[float],
    ) -> Sequence[Mapping[str, Any]]: ...

    def uniform_health_inputs_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def prior_factorial_dev8(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        prior_conditions: Sequence[str],
    ) -> Mapping[str, Mapping[str, Any]]: ...

    def holdout5_comparison(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...

    def tune24_comparison(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]: ...

    def final48_bootstrap(
        self,
        *,
        scene_ids: Sequence[str],
        structure_condition: str,
        classifier: str,
        acceptance_threshold: float,
        data_condition: str,
    ) -> Mapping[str, Any]: ...


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value


def _status(
    path: Path,
    *,
    state: str,
    checkpoint: str,
    git_commit: str,
    history: Sequence[Mapping[str, Any]],
    **payload: Any,
) -> dict[str, Any]:
    result = {
        "schema": "saga-v10-orchestrator-status-v1",
        "state": str(state),
        "checkpoint": str(checkpoint),
        "git_commit": str(git_commit),
        "updated_at_unix": time.time(),
        "history": _jsonable(history),
        **_jsonable(payload),
    }
    write_json(path, result)
    return result


def _append_history(
    history: list[dict[str, Any]], checkpoint: str, **payload: Any
) -> None:
    history.append(
        {
            "checkpoint": str(checkpoint),
            "completed_at_unix": time.time(),
            **_jsonable(payload),
        }
    )


def _require_keys(
    values: Mapping[str, Any], expected: Sequence[str], *, label: str
) -> None:
    actual = set(map(str, values))
    wanted = set(map(str, expected))
    if actual != wanted:
        raise ValueError(
            f"{label} must contain exactly {sorted(wanted)}; got {sorted(actual)}"
        )


def _scene_ids(
    values: Sequence[str], *, label: str, expected_count: int
) -> tuple[str, ...]:
    result = tuple(map(str, values))
    if len(result) != expected_count or len(set(result)) != expected_count:
        raise ValueError(f"{label} must contain {expected_count} unique scene IDs")
    return result


def _stop_for_v10b(
    *,
    status_path: Path,
    proposal_path: Path,
    git_commit: str,
    history: list[dict[str, Any]],
    failed_stage: str,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    proposal = write_v10b_identity_training_proposal(
        proposal_path,
        failed_stage=failed_stage,
        diagnosis={
            "gate_schema": gate.get("schema", "unknown"),
            **dict(gate.get("checks", {})),
        },
    )
    return _status(
        status_path,
        state="stopped",
        checkpoint=f"{failed_stage}-v10b-approval-required",
        git_commit=git_commit,
        history=history,
        stop_reason="V10 ObjectBank structure did not pass its preregistered gate",
        category_prior_tested=False,
        approval_required=True,
        required_user_action="approve-or-reject-v10b-identity-head-control",
        v10b_proposal=proposal,
        failed_gate=gate,
    )


def run_v10_orchestrator(
    *,
    hooks: V10OrchestratorHooks,
    artifacts_root: Path,
    git_commit: str,
    tune24_scene_ids: Sequence[str] = (),
    final48_scene_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Run V10 from V9 closeout through the final48 registered decision.

    Complete artifacts are reused by the injected hooks.  Structural failures
    at Stage 1 or Stage 2 always produce the V10B proposal and stop before any
    category-prior replay.  Later failures report a prior result and never
    mutate the frozen bank or retry with different thresholds.
    """

    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("git_commit must be non-empty")
    root = Path(artifacts_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "v10_orchestrator_status.json"
    proposal_path = root / "V10B_IDENTITY_TRAINING_PROPOSAL.md"
    history: list[dict[str, Any]] = []

    try:
        _status(
            status_path,
            state="running",
            checkpoint="stage0-v9-closeout",
            git_commit=commit,
            history=history,
            category_prior_tested=False,
        )
        closeout = dict(hooks.closeout_v9())
        if closeout.get("passed") is False or closeout.get("state") == "failed":
            _append_history(history, "stage0-v9-closeout", result=closeout)
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage0-evaluation-closeout-failed",
                git_commit=commit,
                history=history,
                stop_reason="V9 evaluation correction did not close reproducibly",
                category_prior_tested=False,
                approval_required=False,
                closeout=closeout,
            )
        _append_history(history, "stage0-v9-closeout", result=closeout)

        _status(
            status_path,
            state="running",
            checkpoint="stage1-dev2-five-structure-banks",
            git_commit=commit,
            history=history,
            category_prior_tested=False,
        )
        bank_summary = hooks.ensure_banks(
            scene_ids=DEV2,
            structure_conditions=STRUCTURE_CONDITIONS,
        )
        structure_metrics = hooks.audit_dev2_structures(
            scene_ids=DEV2,
            structure_conditions=STRUCTURE_CONDITIONS,
        )
        _require_keys(
            structure_metrics,
            STRUCTURE_CONDITIONS,
            label="DEV2 structure audit",
        )
        causal_selection = select_pair_reconstruction_arm(
            [
                {"condition": arm, **dict(structure_metrics[arm])}
                for arm in PAIR_RECONSTRUCTION_ARMS
            ]
        )
        stage1_gate = stage1_structure_gate(
            structure_metrics[VIEW_CONSENSUS_ARM],
            p0r0_candidate_count=int(
                structure_metrics["P0R0"]["candidate_count"]
            ),
        )
        _append_history(
            history,
            "stage1-dev2-structure",
            bank_summary=bank_summary,
            causal_selection=causal_selection,
            gate=stage1_gate,
        )
        if not stage1_gate["passed"]:
            return _stop_for_v10b(
                status_path=status_path,
                proposal_path=proposal_path,
                git_commit=commit,
                history=history,
                failed_stage="stage1-dev2-structure",
                gate=stage1_gate,
            )

        _status(
            status_path,
            state="running",
            checkpoint="stage2-dev8-uniform-health",
            git_commit=commit,
            history=history,
            category_prior_tested=False,
        )
        dev8_bank_summary = hooks.ensure_banks(
            scene_ids=DEV8,
            structure_conditions=(VIEW_CONSENSUS_ARM,),
        )
        classifier_metrics = hooks.classifier_metrics_dev8(
            scene_ids=DEV8,
            structure_condition=VIEW_CONSENSUS_ARM,
        )
        classifier_selection = select_late_classifier(classifier_metrics)
        classifier = str(classifier_selection["selected"])
        threshold_rows = hooks.threshold_sweep_dev2(
            scene_ids=DEV2,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            thresholds=PRIOR_THRESHOLDS,
        )
        threshold_selection = select_uniform_threshold(threshold_rows)
        if not threshold_selection["passed"]:
            threshold_gate = {
                "schema": "saga-v10-stage2-threshold-gate-v1",
                "passed": False,
                "checks": {"registered_uniform_threshold_exists": False},
            }
            _append_history(
                history,
                "stage2-dev8-uniform-threshold",
                classifier_selection=classifier_selection,
                threshold_selection=threshold_selection,
            )
            return _stop_for_v10b(
                status_path=status_path,
                proposal_path=proposal_path,
                git_commit=commit,
                history=history,
                failed_stage="stage2-dev8-uniform-threshold",
                gate=threshold_gate,
            )
        threshold = float(threshold_selection["selected_threshold"])
        health_inputs = hooks.uniform_health_inputs_dev8(
            scene_ids=DEV8,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            acceptance_threshold=threshold,
        )
        _require_keys(
            health_inputs,
            ("bank", "b1_fixed"),
            label="DEV8 uniform health inputs",
        )
        health_gate = stage2_uniform_health_gate(
            health_inputs["bank"],
            b1_fixed=health_inputs["b1_fixed"],
        )
        _append_history(
            history,
            "stage2-dev8-uniform-health",
            bank_summary=dev8_bank_summary,
            classifier_selection=classifier_selection,
            threshold_selection=threshold_selection,
            gate=health_gate,
        )
        if not health_gate["passed"]:
            return _stop_for_v10b(
                status_path=status_path,
                proposal_path=proposal_path,
                git_commit=commit,
                history=history,
                failed_stage="stage2-dev8-uniform-health",
                gate=health_gate,
            )

        _status(
            status_path,
            state="running",
            checkpoint="stage3-dev8-prior-factorial",
            git_commit=commit,
            history=history,
            selected_classifier=classifier,
            acceptance_threshold=threshold,
            category_prior_tested=False,
        )
        factorial = hooks.prior_factorial_dev8(
            scene_ids=DEV8,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            acceptance_threshold=threshold,
            prior_conditions=V10_PRIOR_CONDITIONS,
        )
        _require_keys(
            factorial,
            V10_PRIOR_CONDITIONS,
            label="DEV8 prior factorial",
        )
        uniform_rows = factorial["U000"]["rows"]
        prior_gates: dict[str, dict[str, Any]] = {}
        aggregate_metrics: dict[str, Mapping[str, Any]] = {
            condition: factorial[condition]["aggregate"]
            for condition in V10_PRIOR_CONDITIONS
        }
        for condition in V10_PRIOR_CONDITIONS:
            if condition == "U000":
                continue
            evidence = factorial[condition]
            prior_gates[condition] = stage3_prior_gate(
                uniform_rows,
                evidence["rows"],
                candidate_score_deltas=evidence.get("candidate_score_deltas", ()),
                accepted_or_ownership_changed=bool(
                    evidence.get("accepted_or_ownership_changed", False)
                ),
            )
        prior_selection = select_best_prior_condition(
            prior_gates, aggregate_metrics
        )
        _append_history(
            history,
            "stage3-dev8-prior-factorial",
            gates=prior_gates,
            selection=prior_selection,
        )
        if not prior_selection["passed"]:
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage3-prior-no-dev8-benefit",
                git_commit=commit,
                history=history,
                stop_reason=(
                    "No data-driven prior condition passed the preregistered "
                    "DEV8 benefit and stability gate"
                ),
                category_prior_tested=True,
                approval_required=False,
                selected_classifier=classifier,
                acceptance_threshold=threshold,
                prior_gates=prior_gates,
            )
        best_data = str(prior_selection["selected"])

        _status(
            status_path,
            state="running",
            checkpoint="stage4-holdout5",
            git_commit=commit,
            history=history,
            category_prior_tested=True,
            selected_classifier=classifier,
            acceptance_threshold=threshold,
            selected_data_condition=best_data,
        )
        holdout = hooks.holdout5_comparison(
            scene_ids=HOLDOUT5,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            acceptance_threshold=threshold,
            data_condition=best_data,
        )
        _require_keys(
            holdout,
            ("uniform_rows", "data_rows"),
            label="holdout5 comparison",
        )
        holdout_gate = holdout5_gate(
            holdout["uniform_rows"], holdout["data_rows"]
        )
        _append_history(history, "stage4-holdout5", gate=holdout_gate)
        if not holdout_gate["passed"]:
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage4-holdout5-prior-not-stable",
                git_commit=commit,
                history=history,
                stop_reason="Best DEV8 prior did not replicate on canonical holdout5",
                category_prior_tested=True,
                approval_required=False,
                selected_data_condition=best_data,
                holdout_gate=holdout_gate,
            )

        tune_ids = _scene_ids(
            tune24_scene_ids, label="tune24_scene_ids", expected_count=24
        )
        tune = hooks.tune24_comparison(
            scene_ids=tune_ids,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            acceptance_threshold=threshold,
            data_condition=best_data,
        )
        _require_keys(
            tune,
            ("uniform_rows", "data_rows"),
            label="tune24 comparison",
        )
        tune_gate = physical_scene_macro_gate(
            tune["uniform_rows"], tune["data_rows"]
        )
        _append_history(history, "stage4-tune24-physical-macro", gate=tune_gate)
        if not tune_gate["passed"]:
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage4-tune24-prior-not-stable",
                git_commit=commit,
                history=history,
                stop_reason=(
                    "Best prior did not reach +0.002 macro mAP across physical scenes"
                ),
                category_prior_tested=True,
                approval_required=False,
                selected_data_condition=best_data,
                tune24_gate=tune_gate,
            )

        final_ids = _scene_ids(
            final48_scene_ids, label="final48_scene_ids", expected_count=48
        )
        bootstrap = hooks.final48_bootstrap(
            scene_ids=final_ids,
            structure_condition=VIEW_CONSENSUS_ARM,
            classifier=classifier,
            acceptance_threshold=threshold,
            data_condition=best_data,
        )
        final_gate = final48_gate(bootstrap)
        _append_history(history, "stage4-final48", gate=final_gate)
        if not final_gate["passed"]:
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage4-final48-no-stable-prior-improvement",
                git_commit=commit,
                history=history,
                stop_reason=(
                    "V10 proposal-level category prior did not pass the final48 "
                    "effect-size and paired-bootstrap gate"
                ),
                category_prior_tested=True,
                approval_required=False,
                selected_data_condition=best_data,
                final48_gate=final_gate,
            )
        return _status(
            status_path,
            state="complete",
            checkpoint="stage4-final48-category-prior-supported",
            git_commit=commit,
            history=history,
            category_prior_tested=True,
            category_prior_supported=True,
            selected_classifier=classifier,
            acceptance_threshold=threshold,
            selected_data_condition=best_data,
            final48_gate=final_gate,
        )
    except BaseException as error:
        _status(
            status_path,
            state="failed",
            checkpoint="orchestrator-exception",
            git_commit=commit,
            history=history,
            category_prior_tested=False,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
