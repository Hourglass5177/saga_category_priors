from __future__ import annotations

"""Recoverable controller for the section-30 candidate-prior experiment.

The controller is deliberately separate from the command registry and the
postprocess worker.  It owns sequencing, immutable experimental identity,
resource guards, atomic state, and registered stop rules; all candidate and
evaluation algorithms remain in their dedicated modules.  Ground truth is
only passed to diagnostic/evaluation hooks and never to repair or replay.
"""

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import hash_json, load_json, read_rows, sha256_file, write_json, write_rows


DEV2 = ("scene0645_00", "scene0025_01")
DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)

INITIAL_SAMPLE_CAP = 5_000
NESTED_SAMPLE_CAP = 10_000
MIN_DISK_FREE_GIB = 80.0
EXPECTED_CGROUP_MAX_BYTES = 90 * 1024**3
STATE_SCHEMA = "saga-category-candidate-experiment-state-v1"
REPLAY_IDENTITY_SCHEMA = "saga-category-candidate-replay-identity-v1"
FROZEN_ARM_SCHEMA = "saga-category-candidate-frozen-repair-arm-v1"
REGISTERED_10K_SCHEMA = "saga-category-candidate-same-source-10k-control-v2"
REPAIR_CONDITIONS = (
    "C1-consistent-envelope",
    "C2-raw-anchored-envelope",
)
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)

# Public acceptance artifacts from the section-30 preregistration.  Internal
# checkpoints may keep more descriptive names, but these names are the stable
# hand-off contract consumed by status monitors and the final report.
ACCEPTANCE_ARTIFACT_FILENAMES = (
    "candidate_formation_trace_dev2.parquet",
    "candidate_formation_root_cause.json",
    "candidate_repair_dev2.parquet",
    "candidate_repair_dev8.parquet",
    "prior_oracle_v2.json",
    "candidate_prior_dev8.parquet",
    "category_denoise_v2_dev8.parquet",
    "category_denoise_v2_holdout5.parquet",
    "category_denoise_v2_tune24.parquet",
    "category_denoise_v2_final48.parquet",
    "category_denoise_v2_analysis.json",
)

ROOT_ACTION_EXTEND = "extend-trace-only-to-dev8"
ROOT_ACTION_NESTED = "compare-nested-sample-cap-5000-10000"
ROOT_ACTION_REPRESENTATION = "evaluate-affinity-auroc-and-oracle-seed"
ROOT_ACTION_REPAIR = "evaluate-C1-C2-full-assignment-repair"
REPRESENTATION_ACTION_10K = "run-two-scene-10k-feature-control"

TERMINAL_STATUSES = frozenset({"complete", "stopped"})
RESOURCE_GUARDED_STAGES = frozenset(
    {
        "repair_dev2_5k",
        "b0_parity_dev2",
        "repair_dev8_trace",
        "nested_10k_sampling_control",
        "feature_10k_control",
        "repair_dev8_frozen",
        "legacy_replay_dev8",
        "repair_holdout5",
        "legacy_replay_holdout5",
        "repair_tune24",
        "legacy_replay_tune24",
        "repair_final48",
        "legacy_replay_final48",
    }
)


@dataclass(frozen=True)
class CandidateExperimentConfig:
    runtime_manifest: Path
    gt_dir: Path
    locked_runtime_manifest: Path
    locked_gt_dir: Path
    locked_evaluation_scenes: Path
    repo_root: Path
    category_priors: Path
    prior_oracle_root: Path
    reference_bank_root: Path
    output_root: Path
    size_bins: Path
    python_bin: Path | None = None
    registered_10k_control: Path | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.size_bins is None:
            raise ValueError("section 30 requires the frozen size-bin artifact")
        for name in (
            "runtime_manifest",
            "gt_dir",
            "locked_runtime_manifest",
            "locked_gt_dir",
            "locked_evaluation_scenes",
            "repo_root",
            "category_priors",
            "prior_oracle_root",
            "reference_bank_root",
            "output_root",
            "size_bins",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("python_bin", "registered_10k_control"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if isinstance(self.seed, bool) or int(self.seed) != 42:
            raise ValueError("section 30 freezes the experiment seed at 42")

    @property
    def state_path(self) -> Path:
        return self.output_root / "experiment_state.json"

    @property
    def artifacts_root(self) -> Path:
        return self.output_root / "artifacts"

    @property
    def effective_registered_10k_control(self) -> Path:
        return self.registered_10k_control or (
            self.artifacts_root / "registered_10k_feature_control.json"
        )

    @property
    def viewer_root(self) -> Path:
        return self.artifacts_root / "viewer"

    def acceptance_artifact(self, filename: str) -> Path:
        if filename not in ACCEPTANCE_ARTIFACT_FILENAMES:
            raise KeyError(f"unregistered acceptance artifact: {filename}")
        return self.artifacts_root / filename


Hook = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class CandidateExperimentHooks:
    """Injectable execution boundary; defaults call the real project modules."""

    check_resources: Callable[[Path], Mapping[str, Any]]
    validate_inputs: Hook
    repair: Hook
    check_b0_parity: Hook
    diagnose: Hook
    nested_sampling_control: Hook
    representation_diagnostic: Hook
    feature_10k_control: Hook
    evaluate_repair: Hook
    evaluate_prior_oracle: Hook
    evaluate_candidate_prior: Hook
    select_candidate_threshold: Hook
    replay_final_stage: Hook


def check_experiment_resources(
    root: Path,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Enforce the cloud contract using ``df`` and cgroup v2 only.

    Host-level ``free`` is intentionally neither invoked nor interpreted.
    ``df -Pk`` reports available KiB, which is converted to GiB without a
    decimal-unit ambiguity.
    """

    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    completed = run(
        ["df", "-Pk", str(destination)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in str(completed.stdout).splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError("df did not return a filesystem data row")
    fields = lines[-1].split()
    if len(fields) < 6:
        raise RuntimeError("df output does not contain an available-KiB field")
    try:
        available_kib = int(fields[3])
    except ValueError as exc:
        raise RuntimeError("df available-KiB field is not an integer") from exc
    available_gib = available_kib / 1024**2
    if available_gib < MIN_DISK_FREE_GIB:
        raise RuntimeError(
            f"candidate experiment requires at least {MIN_DISK_FREE_GIB:.0f} GiB "
            f"available according to df; found {available_gib:.1f} GiB"
        )

    current_path = Path(cgroup_root) / "memory.current"
    maximum_path = Path(cgroup_root) / "memory.max"
    events_path = Path(cgroup_root) / "memory.events"
    for path in (current_path, maximum_path, events_path):
        if not path.is_file():
            raise RuntimeError(f"required cgroup v2 resource file is missing: {path}")
    maximum_text = maximum_path.read_text(encoding="utf-8").strip()
    if maximum_text == "max":
        raise RuntimeError("expected cgroup memory.max=90 GiB; found max")
    try:
        maximum = int(maximum_text)
        current = int(current_path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise RuntimeError("cgroup memory values must be integers") from exc
    if maximum != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(
            "expected cgroup memory.max=90 GiB; "
            f"found {maximum_text} bytes"
        )
    if current >= maximum:
        raise RuntimeError("cgroup memory.current has reached memory.max")
    return {
        "disk_source": "df-Pk",
        "disk_available_kib": available_kib,
        "disk_available_gib": available_gib,
        "memory_current_bytes": current,
        "memory_max_bytes": maximum,
        "memory_events": events_path.read_text(encoding="utf-8").strip(),
        "host_free_used": False,
    }


def _resolved(path: Path | None) -> str | None:
    return str(path.resolve()) if path is not None else None


def _optional_sha256(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def _experiment_identity(config: CandidateExperimentConfig) -> dict[str, Any]:
    return {
        "runtime_manifest": _resolved(config.runtime_manifest),
        "runtime_manifest_sha256": _optional_sha256(config.runtime_manifest),
        "gt_dir": _resolved(config.gt_dir),
        "locked_runtime_manifest": _resolved(config.locked_runtime_manifest),
        "locked_runtime_manifest_sha256": _optional_sha256(
            config.locked_runtime_manifest
        ),
        "locked_gt_dir": _resolved(config.locked_gt_dir),
        "locked_evaluation_scenes": _resolved(config.locked_evaluation_scenes),
        "locked_evaluation_scenes_sha256": _optional_sha256(
            config.locked_evaluation_scenes
        ),
        "repo_root": _resolved(config.repo_root),
        "category_priors": _resolved(config.category_priors),
        "category_priors_sha256": _optional_sha256(config.category_priors),
        "prior_oracle_root": _resolved(config.prior_oracle_root),
        "reference_bank_root": _resolved(config.reference_bank_root),
        "output_root": _resolved(config.output_root),
        "size_bins": _resolved(config.size_bins),
        "size_bins_sha256": _optional_sha256(config.size_bins),
        "python_bin": _resolved(config.python_bin),
        "registered_10k_control": _resolved(
            config.effective_registered_10k_control
        ),
        "seed": int(config.seed),
        "legacy_knn_k": 256,
        "legacy_min_count": 10,
        "initial_sample_cap": INITIAL_SAMPLE_CAP,
        "nested_sample_cap": NESTED_SAMPLE_CAP,
        "dev2": list(DEV2),
        "dev8": list(DEV8),
        "holdout5": list(HOLDOUT5),
        "smoothness_enabled": False,
        "protected_or_reinserted_points_allowed": False,
    }


def _initial_state(config: CandidateExperimentConfig) -> dict[str, Any]:
    formation = (config.output_root / "formation_5k").resolve()
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "checkpoint": "initialized",
        "next_stage": "validate_inputs",
        "current_stage": None,
        "identity": _experiment_identity(config),
        "history": [],
        "active_sample_cap": INITIAL_SAMPLE_CAP,
        "active_run_root": str(formation),
        "active_runtime_manifest": str(config.runtime_manifest.resolve()),
        "feature_10k_control_tested": False,
        "feature_10k_control_passed": None,
        "root_scene_ids": list(DEV2),
        "root_trace_rows": None,
        "tune24_scene_ids": None,
        "final48_scene_ids": None,
        "frozen_repair_condition": None,
        "frozen_repair_arm_artifact": None,
        "frozen_threshold": None,
        "prior_capacity_tested": False,
        "candidate_prior_tested": False,
        "legacy_replay_tested": False,
    }


def _acceptance_artifact_status(
    config: CandidateExperimentConfig,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename in ACCEPTANCE_ARTIFACT_FILENAMES:
        path = config.acceptance_artifact(filename)
        result[filename] = {
            "path": str(path.resolve()),
            "available": path.is_file(),
        }
    result["viewer/"] = {
        "path": str(config.viewer_root.resolve()),
        "available": config.viewer_root.is_dir(),
    }
    return result


def _write_acceptance_analysis(
    config: CandidateExperimentConfig, state: Mapping[str, Any]
) -> None:
    """Keep the public analysis/status artifact current at every checkpoint."""

    analysis_path = config.acceptance_artifact(
        "category_denoise_v2_analysis.json"
    )
    statuses = _acceptance_artifact_status(config)
    # This file is being materialized by this call, so report it as available
    # even on the first checkpoint write.
    statuses[analysis_path.name]["available"] = True
    payload = {
        "schema": "saga-category-denoise-v2-analysis-v1",
        "status": state.get("status"),
        "checkpoint": state.get("checkpoint"),
        "current_stage": state.get("current_stage"),
        "next_stage": state.get("next_stage"),
        "stop_reason": state.get("stop_reason"),
        "last_error": _json_safe(state.get("last_error")),
        "frozen_repair_condition": state.get("frozen_repair_condition"),
        "frozen_threshold": state.get("frozen_threshold"),
        "active_run_root": state.get("active_run_root"),
        "active_runtime_manifest": state.get("active_runtime_manifest"),
        "feature_10k_control_tested": bool(
            state.get("feature_10k_control_tested", False)
        ),
        "feature_10k_control_passed": state.get("feature_10k_control_passed"),
        "candidate_prior_tested": bool(state.get("candidate_prior_tested", False)),
        "legacy_replay_tested": bool(state.get("legacy_replay_tested", False)),
        "history": _json_safe(state.get("history", [])),
        "acceptance_artifacts": statuses,
    }
    write_json(analysis_path, payload)


def _write_state(config: CandidateExperimentConfig, state: Mapping[str, Any]) -> None:
    # write_json uses a same-directory temporary file and os.replace.
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    config.viewer_root.mkdir(parents=True, exist_ok=True)
    mutable = dict(state)
    _write_acceptance_analysis(config, mutable)
    mutable["acceptance_artifacts"] = _acceptance_artifact_status(config)
    if isinstance(state, dict):
        state["acceptance_artifacts"] = mutable["acceptance_artifacts"]
    write_json(config.state_path, mutable)


def _copy_acceptance_artifact(source: Path, destination: Path) -> bool:
    """Publish a completed internal checkpoint under its stable public name."""

    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file():
        return False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return True


def _load_state(config: CandidateExperimentConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        state = _initial_state(config)
        _write_state(config, state)
        return state
    value = load_json(config.state_path)
    if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
        raise ValueError(f"invalid candidate experiment state: {config.state_path}")
    state = dict(value)
    state.setdefault("active_runtime_manifest", str(config.runtime_manifest.resolve()))
    state.setdefault("feature_10k_control_tested", False)
    state.setdefault("feature_10k_control_passed", None)
    expected = _experiment_identity(config)
    if state.get("identity") != expected:
        raise ValueError(
            "experiment identity differs from the existing recoverable state"
        )
    if not isinstance(state.get("history"), list):
        raise TypeError("experiment state history must be a list")
    return state


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _summary(result: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {
        key: _json_safe(result[key])
        for key in keys
        if key in result
    }


def _record(
    state: dict[str, Any], stage: str, result: Mapping[str, Any], keys: Sequence[str]
) -> None:
    state["history"].append({"stage": stage, **_summary(result, keys)})
    state["checkpoint"] = stage
    state["current_stage"] = None
    state.pop("last_error", None)


def _stop(
    config: CandidateExperimentConfig,
    state: dict[str, Any],
    *,
    checkpoint: str,
    reason: str,
) -> dict[str, Any]:
    state.update(
        {
            "status": "stopped",
            "checkpoint": checkpoint,
            "current_stage": None,
            "next_stage": None,
            "stop_reason": reason,
        }
    )
    _write_state(config, state)
    return state


def _freeze_repair_arm(
    config: CandidateExperimentConfig,
    state: dict[str, Any],
    *,
    condition: str,
) -> str:
    if condition not in REPAIR_CONDITIONS:
        raise ValueError("only a registered C1/C2 arm may be frozen")
    analysis_path = config.artifacts_root / "repair_dev2.analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError(
            "cannot freeze the DEV2 repair arm without its selection analysis: "
            f"{analysis_path}"
        )
    payload = {
        "schema": FROZEN_ARM_SCHEMA,
        "condition": condition,
        "selected_on_scene_ids": list(DEV2),
        "sample_cap": int(state["active_sample_cap"]),
        "run_root": str(Path(state["active_run_root"]).resolve()),
        "selection_analysis": str(analysis_path.resolve()),
        "selection_analysis_sha256": sha256_file(analysis_path),
        "tie_rule": "iou050_iou025_precision_unsupported_count_then_C1",
    }
    path = config.artifacts_root / "frozen_repair_arm.json"
    if path.is_file():
        existing = load_json(path)
        if existing != payload:
            raise ValueError("existing DEV2 frozen-arm artifact has changed")
    else:
        write_json(path, payload)
    state["frozen_repair_condition"] = condition
    state["frozen_repair_arm_artifact"] = str(path.resolve())
    return condition


def _frozen_repair_arm(
    config: CandidateExperimentConfig, state: Mapping[str, Any]
) -> str:
    raw_path = state.get("frozen_repair_arm_artifact")
    if raw_path is None:
        raise ValueError("DEV2 frozen-arm artifact is missing from state")
    path = Path(str(raw_path))
    expected_path = (config.artifacts_root / "frozen_repair_arm.json").resolve()
    if path.resolve() != expected_path or not path.is_file():
        raise ValueError("DEV2 frozen-arm artifact path is missing or changed")
    payload = load_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != FROZEN_ARM_SCHEMA:
        raise ValueError("DEV2 frozen-arm artifact has the wrong schema")
    if tuple(map(str, payload.get("selected_on_scene_ids", ()))) != DEV2:
        raise ValueError("repair arm was not frozen exclusively on DEV2")
    condition = str(payload.get("condition", ""))
    if condition not in REPAIR_CONDITIONS:
        raise ValueError("frozen repair arm is not registered")
    if condition != state.get("frozen_repair_condition"):
        raise ValueError("state and frozen repair-arm artifact disagree")
    if int(payload.get("sample_cap", -1)) != int(state["active_sample_cap"]):
        raise ValueError("frozen repair arm sample cap changed")
    if Path(str(payload.get("run_root", ""))).resolve() != Path(
        str(state["active_run_root"])
    ).resolve():
        raise ValueError("frozen repair arm run root changed")
    analysis_sha = payload.get("selection_analysis_sha256")
    analysis_path = Path(str(payload.get("selection_analysis", "")))
    if analysis_sha is not None and (
        not analysis_path.is_file() or sha256_file(analysis_path) != analysis_sha
    ):
        raise ValueError("DEV2 repair-selection analysis changed after freezing")
    return condition


def _route_root_action(
    config: CandidateExperimentConfig,
    state: dict[str, Any],
    result: Mapping[str, Any],
    *,
    expanded_to_dev8: bool,
) -> dict[str, Any] | None:
    action = str(result.get("next_action", ""))
    if action == ROOT_ACTION_EXTEND:
        if expanded_to_dev8:
            # The preregistration uses eight objects only to decide whether the
            # two-scene trace must be widened to DEV8.  DEV8 is the terminal
            # trace scope; if it still contains fewer than eight diagnosable
            # objects, route the available evidence through the registered
            # sampling/raw-clustering/assignment decision tree instead of
            # inventing an unregistered sample-size stop.
            if bool(result.get("sample_starved_is_majority_of_failures", False)):
                state["next_stage"] = "nested_10k_sampling_control"
            elif bool(
                result.get(
                    "raw_clustering_is_majority_of_sufficiently_sampled_failures",
                    False,
                )
            ):
                state["next_stage"] = "representation_diagnostic"
            else:
                state["next_stage"] = "evaluate_repair_dev2"
        else:
            state["next_stage"] = "repair_dev8_trace"
    elif action == ROOT_ACTION_NESTED:
        state["next_stage"] = "nested_10k_sampling_control"
    elif action == ROOT_ACTION_REPRESENTATION:
        state["next_stage"] = "representation_diagnostic"
    elif action == ROOT_ACTION_REPAIR:
        state["next_stage"] = "evaluate_repair_dev2"
    else:
        raise ValueError(f"unregistered root-cause next_action: {action!r}")
    return None


def _validated_scene_sets(
    result: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    tune = tuple(map(str, result.get("tune24_scene_ids", ())))
    final = tuple(map(str, result.get("final48_scene_ids", ())))
    if len(tune) != 24 or len(set(tune)) != 24:
        raise ValueError("tune runtime must contain exactly 24 unique scans")
    if not set(DEV8).union(HOLDOUT5).issubset(tune):
        raise ValueError("tune24 does not contain the frozen DEV8 and HOLDOUT5")
    if len({scene.rsplit("_", 1)[0] for scene in tune}) != 13:
        raise ValueError("tune24 must represent exactly 13 physical scenes")
    if len(final) != 48 or len(set(final)) != 48:
        raise ValueError("locked final runtime must contain exactly 48 unique scans")
    if len({scene.rsplit("_", 1)[0] for scene in final}) != 48:
        raise ValueError("final48 must contain 48 distinct physical scenes")
    if "scene0019_01" not in final or "scene0019_00" in final:
        raise ValueError("final48 must use the registered scene0019_01 scan")
    return tune, final


def run_category_candidate_experiment(
    config: CandidateExperimentConfig,
    hooks: CandidateExperimentHooks | None = None,
) -> dict[str, Any]:
    """Run or resume the fixed DEV2 -> DEV8 -> HOLDOUT5 state machine."""

    active_hooks = hooks or default_candidate_experiment_hooks()
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    state = _load_state(config)
    if (
        state.get("status") == "stopped"
        and state.get("checkpoint") == "root_diagnosis_insufficient_on_dev8"
    ):
        # Resume states written by the short-lived controller bug fixed above.
        # The completed DEV8 trace and analysis remain authoritative; only the
        # next registered branch is restored.
        analysis_path = config.acceptance_artifact(
            "candidate_formation_root_cause.json"
        )
        result = load_json(analysis_path)
        if not isinstance(result, Mapping):
            raise TypeError("DEV8 root-cause analysis must be a mapping")
        state.update(
            {
                "status": "running",
                "current_stage": None,
                "next_stage": None,
                "stop_reason": None,
                "last_error": None,
            }
        )
        _route_root_action(config, state, result, expanded_to_dev8=True)
        _write_state(config, state)
    if state.get("status") in TERMINAL_STATUSES:
        _write_state(config, state)
        return state
    if state.get("status") == "awaiting_registered_10k_control":
        # Migrate the short-lived broken checkpoint which waited for an
        # external JSON.  The controller now owns the registered same-source
        # training, candidate regeneration and comparison end to end.
        state.update(
            {
                "status": "running",
                "current_stage": None,
                "next_stage": "feature_10k_control",
                "awaiting_reason": None,
            }
        )
        _write_state(config, state)

    while state.get("status") not in TERMINAL_STATUSES:
        stage = str(state.get("next_stage") or "")
        if not stage:
            raise RuntimeError("running experiment state has no next_stage")
        state["status"] = "running"
        state["current_stage"] = stage
        _write_state(config, state)

        try:
            if stage in RESOURCE_GUARDED_STAGES:
                resources = active_hooks.check_resources(config.output_root)
                state["last_resources"] = _json_safe(resources)
                _write_state(config, state)

            if stage == "validate_inputs":
                result = active_hooks.validate_inputs(config=config)
                tune, final = _validated_scene_sets(result)
                _record(
                    state,
                    stage,
                    result,
                    (
                        "tune24_scene_ids",
                        "final48_scene_ids",
                        "tune_physical_scene_count",
                        "final_physical_scene_count",
                    ),
                )
                state["tune24_scene_ids"] = list(tune)
                state["final48_scene_ids"] = list(final)
                state["next_stage"] = "repair_dev2_5k"

            elif stage == "repair_dev2_5k":
                run_root = Path(state["active_run_root"])
                result = active_hooks.repair(
                    config=config,
                    run_root=run_root,
                    scene_ids=DEV2,
                    sample_cap=INITIAL_SAMPLE_CAP,
                    require_reference_identity=True,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["next_stage"] = "b0_parity_dev2"

            elif stage == "b0_parity_dev2":
                result = active_hooks.check_b0_parity(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=DEV2,
                )
                _record(state, stage, result, ("passed", "scenes"))
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="b0_parity_dev2_failed",
                        reason="category-repair B0 is not pointwise identical to true off",
                    )
                state["next_stage"] = "diagnose_root_dev2"

            elif stage in {"diagnose_root_dev2", "diagnose_root_dev8"}:
                expanded = stage.endswith("dev8")
                scenes = DEV8 if expanded else DEV2
                if expanded:
                    trace_path = config.artifacts_root / (
                        f"root_dev8_{state['active_sample_cap']}.parquet"
                    )
                    # DEV8 is only reached when DEV2 has too few diagnosable
                    # objects.  Its result therefore supersedes the preliminary
                    # DEV2 root-cause summary under the same public contract.
                    analysis_path = config.acceptance_artifact(
                        "candidate_formation_root_cause.json"
                    )
                else:
                    trace_path = config.acceptance_artifact(
                        "candidate_formation_trace_dev2.parquet"
                    )
                    analysis_path = config.acceptance_artifact(
                        "candidate_formation_root_cause.json"
                    )
                result = active_hooks.diagnose(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=scenes,
                    trace_output=trace_path,
                    analysis_output=analysis_path,
                )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "diagnosable_object_count",
                        "failure_status_counts",
                        "next_action",
                    ),
                )
                state["root_scene_ids"] = list(scenes)
                state["root_trace_rows"] = str(trace_path.resolve())
                stopped = _route_root_action(
                    config, state, result, expanded_to_dev8=expanded
                )
                if stopped is not None:
                    return stopped

            elif stage == "repair_dev8_trace":
                result = active_hooks.repair(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=DEV8,
                    sample_cap=int(state["active_sample_cap"]),
                    require_reference_identity=True,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["next_stage"] = "diagnose_root_dev8"

            elif stage == "nested_10k_sampling_control":
                nested_root = (config.output_root / "formation_10k").resolve()
                result = active_hooks.nested_sampling_control(
                    config=config,
                    source_run_root=Path(state["active_run_root"]),
                    nested_run_root=nested_root,
                    scene_ids=tuple(map(str, state["root_scene_ids"])),
                    source_trace_rows=Path(state["root_trace_rows"]),
                )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "passed",
                        "checks",
                        "new_raw_iou025_cluster_count",
                        "raw_recall025_delta",
                        "candidate_count_ratio",
                        "nested_trace_identity",
                    ),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="nested_10k_sampling_control_failed",
                        reason="the registered nested 5k/10k sampling gate failed",
                    )
                state["active_sample_cap"] = NESTED_SAMPLE_CAP
                state["active_run_root"] = str(nested_root)
                state["next_stage"] = "evaluate_repair_dev2"

            elif stage == "representation_diagnostic":
                result = active_hooks.representation_diagnostic(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=tuple(map(str, state["root_scene_ids"])),
                    metrics_output=config.artifacts_root / "representation.parquet",
                    analysis_output=config.artifacts_root / "representation.analysis.json",
                )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "mean_local_affinity_edge_auroc",
                        "oracle_seed_recall_025",
                        "representation_bottleneck_triggered",
                        "next_action",
                    ),
                )
                action = str(result.get("next_action", ""))
                if action == ROOT_ACTION_REPAIR:
                    state["next_stage"] = "evaluate_repair_dev2"
                elif action == REPRESENTATION_ACTION_10K:
                    state["next_stage"] = "feature_10k_control"
                else:
                    raise ValueError(
                        f"unregistered representation next_action: {action!r}"
                    )

            elif stage == "feature_10k_control":
                result = active_hooks.feature_10k_control(
                    config=config,
                    source_run_root=Path(state["active_run_root"]),
                    source_runtime_manifest=Path(
                        str(state["active_runtime_manifest"])
                    ),
                    scene_ids=DEV2,
                    sample_cap=int(state["active_sample_cap"]),
                    training_root=config.output_root / "feature_10k_control",
                    control_run_root=config.output_root / "formation_feature_10k",
                    control_runtime_manifest=(
                        config.artifacts_root / "feature_10k_runtime_manifest.json"
                    ),
                    output=config.effective_registered_10k_control,
                )
                passed = bool(result.get("passed", False))
                state["feature_10k_control_tested"] = True
                state["feature_10k_control_passed"] = passed
                _record(
                    state,
                    stage,
                    result,
                    (
                        "passed",
                        "checks",
                        "mean_affinity_auroc_delta",
                        "same_class_iou025_candidate_delta",
                        "feature_iterations",
                        "scene_ids",
                        "control_run_root",
                        "control_runtime_manifest",
                    ),
                )
                if not passed:
                    return _stop(
                        config,
                        state,
                        checkpoint="feature_10k_control_failed",
                        reason=(
                            "the registered same-source two-scene 10k feature "
                            "control did not improve mean affinity AUROC by 0.05 "
                            "and add at least two same-class IoU>=0.25 candidates"
                        ),
                    )
                state["active_run_root"] = str(
                    Path(str(result["control_run_root"])).resolve()
                )
                state["active_runtime_manifest"] = str(
                    Path(str(result["control_runtime_manifest"])).resolve()
                )
                return _stop(
                    config,
                    state,
                    checkpoint="feature_10k_control_passed_requires_expansion",
                    reason=(
                        "the same-source two-scene 10k control passed, confirming "
                        "that the 2k representation is a bottleneck; the registered "
                        "scope authorizes only this positive control, so DEV8 10k "
                        "training requires separate approval"
                    ),
                )

            elif stage == "evaluate_repair_dev2":
                result = active_hooks.evaluate_repair(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=DEV2,
                    phase="dev2",
                    selected_condition=None,
                    frozen_repair_artifact=None,
                    metrics_output=config.acceptance_artifact(
                        "candidate_repair_dev2.parquet"
                    ),
                    analysis_output=config.artifacts_root / "repair_dev2.analysis.json",
                )
                selected = result.get("selected_condition")
                gate = (
                    result.get("dev2_arm_gates", {}).get(selected, {})
                    if isinstance(result.get("dev2_arm_gates"), Mapping)
                    else {}
                )
                passed = bool(result.get("passed", gate.get("passed", False)))
                _record(
                    state,
                    stage,
                    {**dict(result), "passed": passed},
                    ("passed", "selected_condition", "dev2_arm_gates"),
                )
                if selected not in REPAIR_CONDITIONS or not passed:
                    return _stop(
                        config,
                        state,
                        checkpoint="repair_dev2_gate_failed",
                        reason="no registered C1/C2 repair arm passed DEV2",
                    )
                _freeze_repair_arm(
                    config, state, condition=str(selected)
                )
                state["next_stage"] = "repair_dev8_frozen"

            elif stage == "repair_dev8_frozen":
                _frozen_repair_arm(config, state)
                cap = int(state["active_sample_cap"])
                result = active_hooks.repair(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=DEV8,
                    sample_cap=cap,
                    require_reference_identity=cap == INITIAL_SAMPLE_CAP,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["next_stage"] = "evaluate_repair_dev8"

            elif stage == "evaluate_repair_dev8":
                frozen_condition = _frozen_repair_arm(config, state)
                result = active_hooks.evaluate_repair(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=DEV8,
                    phase="dev8",
                    selected_condition=frozen_condition,
                    frozen_repair_artifact=Path(
                        str(state["frozen_repair_arm_artifact"])
                    ),
                    metrics_output=config.acceptance_artifact(
                        "candidate_repair_dev8.parquet"
                    ),
                    analysis_output=config.artifacts_root / "repair_dev8.analysis.json",
                )
                gate = result.get("dev8_health_gate", {})
                passed = bool(result.get("passed", gate.get("passed", False)))
                _record(
                    state,
                    stage,
                    {**dict(result), "passed": passed},
                    ("passed", "selected_condition", "dev8_health_gate"),
                )
                if str(result.get("selected_condition")) != frozen_condition:
                    raise ValueError("DEV8 changed the repair arm frozen on DEV2")
                if not passed:
                    return _stop(
                        config,
                        state,
                        checkpoint="repair_dev8_health_gate_failed",
                        reason="the frozen repair arm did not make the DEV8 candidate space healthy",
                    )
                state["next_stage"] = "prior_oracle_v2"

            elif stage == "prior_oracle_v2":
                result = active_hooks.evaluate_prior_oracle(
                    config=config,
                    output=config.acceptance_artifact("prior_oracle_v2.json"),
                )
                state["prior_capacity_tested"] = True
                _record(state, stage, result, ("passed", "checks", "failed_checks"))
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="prior_oracle_v2_gate_failed",
                        reason="the registered complete/fragment/merge prior capacity gate failed",
                    )
                state["next_stage"] = "candidate_prior_dev8"

            elif stage == "candidate_prior_dev8":
                frozen_condition = _frozen_repair_arm(config, state)
                result = active_hooks.evaluate_candidate_prior(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    selected_condition=frozen_condition,
                    repair_analysis=config.artifacts_root / "repair_dev8.analysis.json",
                    output_dir=config.artifacts_root / "candidate_prior",
                    metrics_output=config.acceptance_artifact(
                        "candidate_prior_dev8.parquet"
                    ),
                )
                state["candidate_prior_tested"] = True
                _record(
                    state,
                    stage,
                    result,
                    (
                        "passed",
                        "acceptance_threshold",
                        "gates",
                        "mechanical_effect",
                        "candidate_ap",
                    ),
                )
                if result.get("acceptance_threshold", None) is not None:
                    raise ValueError(
                        "DEV8 candidate-prior gate must be threshold-free"
                    )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="candidate_prior_dev8_gate_failed",
                        reason="the same-bank threshold-free candidate prior gate failed",
                    )
                state["next_stage"] = "select_threshold_dev2"

            elif stage == "select_threshold_dev2":
                frozen_condition = _frozen_repair_arm(config, state)
                result = active_hooks.select_candidate_threshold(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    selected_condition=frozen_condition,
                    repair_analysis=config.artifacts_root / "repair_dev8.analysis.json",
                    output=config.artifacts_root
                    / "candidate_prior"
                    / "dev2_threshold.json",
                )
                threshold = float(result.get("selected_threshold", -1.0))
                if threshold not in THRESHOLD_GRID:
                    raise ValueError(
                        "candidate prior returned a threshold outside the frozen grid"
                    )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "selected_threshold",
                        "scene_ids",
                        "score_source",
                        "tie_rule",
                        "grid_rows",
                    ),
                )
                state["frozen_threshold"] = threshold
                state["next_stage"] = "legacy_replay_dev8"

            elif stage == "legacy_replay_dev8":
                frozen_condition = _frozen_repair_arm(config, state)
                result = active_hooks.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=Path(state["active_run_root"]),
                    selected_condition=frozen_condition,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=DEV8,
                    stage="dev8",
                    output_dir=config.output_root / "legacy_replay_dev8",
                )
                _copy_acceptance_artifact(
                    config.output_root
                    / "legacy_replay_dev8"
                    / "evaluation"
                    / "dev8_condition_metrics.parquet",
                    config.acceptance_artifact(
                        "category_denoise_v2_dev8.parquet"
                    ),
                )
                state["legacy_replay_tested"] = True
                _record(
                    state,
                    stage,
                    result,
                    ("passed", "uniform_health", "data_minus_uniform", "candidate_survival_intervention"),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_dev8_gate_failed",
                        reason="U/D did not pass the registered shared-legacy DEV8 output gate",
                    )
                state["next_stage"] = "repair_holdout5"

            elif stage == "repair_holdout5":
                _frozen_repair_arm(config, state)
                result = active_hooks.repair(
                    config=config,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=HOLDOUT5,
                    sample_cap=int(state["active_sample_cap"]),
                    require_reference_identity=False,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["next_stage"] = "legacy_replay_holdout5"

            elif stage == "legacy_replay_holdout5":
                frozen_condition = _frozen_repair_arm(config, state)
                result = active_hooks.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=Path(state["active_run_root"]),
                    selected_condition=frozen_condition,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=HOLDOUT5,
                    stage="holdout",
                    output_dir=config.output_root / "legacy_replay_holdout5",
                )
                _copy_acceptance_artifact(
                    config.output_root
                    / "legacy_replay_holdout5"
                    / "evaluation"
                    / "holdout_condition_metrics.parquet",
                    config.acceptance_artifact(
                        "category_denoise_v2_holdout5.parquet"
                    ),
                )
                _record(
                    state,
                    stage,
                    result,
                    ("passed", "uniform_health", "data_minus_uniform", "candidate_survival_intervention"),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_holdout5_gate_failed",
                        reason="U/D did not pass the five-scene canonical holdout gate",
                    )
                state["next_stage"] = "repair_tune24"

            elif stage == "repair_tune24":
                _frozen_repair_arm(config, state)
                tune_scenes = tuple(map(str, state["tune24_scene_ids"]))
                result = active_hooks.repair(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=Path(state["active_run_root"]),
                    scene_ids=tune_scenes,
                    sample_cap=int(state["active_sample_cap"]),
                    require_reference_identity=False,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["next_stage"] = "legacy_replay_tune24"

            elif stage == "legacy_replay_tune24":
                frozen_condition = _frozen_repair_arm(config, state)
                tune_scenes = tuple(map(str, state["tune24_scene_ids"]))
                result = active_hooks.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=Path(state["active_run_root"]),
                    selected_condition=frozen_condition,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=tune_scenes,
                    stage="tune",
                    output_dir=config.output_root / "legacy_replay_tune24",
                )
                _copy_acceptance_artifact(
                    config.output_root
                    / "legacy_replay_tune24"
                    / "evaluation"
                    / "tune_condition_metrics.parquet",
                    config.acceptance_artifact(
                        "category_denoise_v2_tune24.parquet"
                    ),
                )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "passed",
                        "uniform_health",
                        "data_minus_uniform",
                        "candidate_survival_intervention",
                    ),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_tune24_gate_failed",
                        reason="U/D did not pass the 13-physical-scene tune24 gate",
                    )
                state["next_stage"] = "repair_final48"

            elif stage == "repair_final48":
                _frozen_repair_arm(config, state)
                final_scenes = tuple(map(str, state["final48_scene_ids"]))
                final_root = (config.output_root / "formation_final48").resolve()
                result = active_hooks.repair(
                    config=config,
                    runtime_manifest=config.locked_runtime_manifest,
                    run_root=final_root,
                    scene_ids=final_scenes,
                    sample_cap=int(state["active_sample_cap"]),
                    require_reference_identity=False,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("complete", "total", "sample_cap", "reference_identity_required"),
                )
                state["final_run_root"] = str(final_root)
                state["next_stage"] = "legacy_replay_final48"

            elif stage == "legacy_replay_final48":
                frozen_condition = _frozen_repair_arm(config, state)
                final_scenes = tuple(map(str, state["final48_scene_ids"]))
                result = active_hooks.replay_final_stage(
                    config=config,
                    runtime_manifest=config.locked_runtime_manifest,
                    gt_dir=config.locked_gt_dir,
                    run_root=Path(state["final_run_root"]),
                    selected_condition=frozen_condition,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=final_scenes,
                    stage="final",
                    output_dir=config.output_root / "legacy_replay_final48",
                )
                _copy_acceptance_artifact(
                    config.output_root
                    / "legacy_replay_final48"
                    / "evaluation"
                    / "final_condition_metrics.parquet",
                    config.acceptance_artifact(
                        "category_denoise_v2_final48.parquet"
                    ),
                )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "passed",
                        "uniform_health",
                        "data_minus_uniform",
                        "candidate_survival_intervention",
                    ),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_final48_gate_failed",
                        reason="U/D did not pass the locked final48 bootstrap gate",
                    )
                state.update(
                    {
                        "status": "complete",
                        "checkpoint": "final48_passed",
                        "current_stage": None,
                        "next_stage": None,
                    }
                )
                _write_state(config, state)
                return state

            else:
                raise ValueError(f"unknown candidate experiment stage: {stage}")

            _write_state(config, state)
        except BaseException as exc:
            state.update(
                {
                    "status": "error",
                    "checkpoint": f"{stage}_error",
                    "current_stage": None,
                    "next_stage": stage,
                    "last_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            _write_state(config, state)
            raise
    return state


def _default_validate_inputs(**kwargs: Any) -> Mapping[str, Any]:
    from .runner import load_scene_runtime_manifest

    config: CandidateExperimentConfig = kwargs["config"]
    required_files = (
        config.runtime_manifest,
        config.locked_runtime_manifest,
        config.locked_evaluation_scenes,
        config.category_priors,
        config.size_bins,
    )
    required_dirs = (
        config.gt_dir,
        config.locked_gt_dir,
        config.repo_root,
        config.prior_oracle_root,
        config.reference_bank_root,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(path)
    if config.python_bin is not None and not config.python_bin.is_file():
        raise FileNotFoundError(config.python_bin)

    tune = tuple(load_scene_runtime_manifest(config.runtime_manifest))
    final = tuple(load_scene_runtime_manifest(config.locked_runtime_manifest))
    locked_payload = load_json(config.locked_evaluation_scenes)
    locked_rows = (
        locked_payload.get("scenes", locked_payload)
        if isinstance(locked_payload, Mapping)
        else locked_payload
    )
    if not isinstance(locked_rows, Sequence) or isinstance(
        locked_rows, (str, bytes)
    ):
        raise TypeError("locked evaluation scenes must be a sequence")
    locked = tuple(
        str(row["scene_id"] if isinstance(row, Mapping) else row)
        for row in locked_rows
    )
    if len(locked) != len(set(locked)) or set(locked) != set(final):
        raise ValueError(
            "locked runtime and locked evaluation scene artifact disagree"
        )
    return {
        "tune24_scene_ids": list(tune),
        "final48_scene_ids": list(final),
        "tune_physical_scene_count": len(
            {scene.rsplit("_", 1)[0] for scene in tune}
        ),
        "final_physical_scene_count": len(
            {scene.rsplit("_", 1)[0] for scene in final}
        ),
        "size_bins_sha256": sha256_file(config.size_bins),
        "category_priors_sha256": sha256_file(config.category_priors),
    }


def _default_repair(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_runner import repair_category_candidates

    config: CandidateExperimentConfig = kwargs["config"]
    reference = (
        config.reference_bank_root
        if bool(kwargs["require_reference_identity"])
        else None
    )
    return repair_category_candidates(
        kwargs.get("runtime_manifest", config.runtime_manifest),
        kwargs["run_root"],
        config.repo_root,
        config.category_priors,
        kwargs["scene_ids"],
        reference_bank_root=reference,
        seed=config.seed,
        sample_cap=int(kwargs["sample_cap"]),
        python_bin=config.python_bin,
    )


def _default_check_b0_parity(**kwargs: Any) -> Mapping[str, Any]:
    import numpy as np

    from .category_denoise_runner import run_category_denoise_b0_control
    from .prediction_contract import validate_prediction_contract

    config: CandidateExperimentConfig = kwargs["config"]
    run_root = Path(kwargs["run_root"])
    scenes = tuple(map(str, kwargs["scene_ids"]))
    run_category_denoise_b0_control(
        config.runtime_manifest,
        run_root,
        config.repo_root,
        config.category_priors,
        scenes,
        seed=config.seed,
        python_bin=config.python_bin,
    )
    rows: list[dict[str, Any]] = []
    for scene_id in scenes:
        disabled = load_json(run_root / "b0-off" / scene_id / "output.json")
        observed = load_json(run_root / "b0" / scene_id / "output.json")
        left = np.asarray(disabled.get("point_labels"), dtype=np.int64)
        right = np.asarray(observed.get("point_labels"), dtype=np.int64)
        left_instances = disabled.get("instances")
        right_instances = observed.get("instances")
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError(f"{scene_id}: B0 predictions must contain label vectors")
        if not isinstance(left_instances, Mapping) or not isinstance(
            right_instances, Mapping
        ):
            raise TypeError(f"{scene_id}: B0 predictions require instance tables")
        validate_prediction_contract(left, left_instances)
        validate_prediction_contract(right, right_instances)
        same_shape = left.shape == right.shape
        changed = (
            int(np.count_nonzero(left != right)) if same_shape else max(len(left), len(right))
        )
        instances_exact = left_instances == right_instances
        rows.append(
            {
                "scene_id": scene_id,
                "point_count_off": len(left),
                "point_count_b0": len(right),
                "changed_point_count": changed,
                "point_labels_exact": same_shape and changed == 0,
                "instances_exact": bool(instances_exact),
                "passed": same_shape and changed == 0 and bool(instances_exact),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "scenes": rows}


def _default_diagnose(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_evaluation import diagnose_category_candidates
    from .taxonomy import load_taxonomy

    config: CandidateExperimentConfig = kwargs["config"]
    return diagnose_category_candidates(
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
        run_root=kwargs["run_root"],
        scene_ids=kwargs["scene_ids"],
        taxonomy=load_taxonomy(),
        trace_output=kwargs["trace_output"],
        analysis_output=kwargs["analysis_output"],
        size_bins=config.size_bins,
    )


def _raw_cluster_keys(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, int]]:
    result: set[tuple[str, str, int]] = set()
    for row in rows:
        cluster = row.get("best_raw_cluster_id")
        if cluster is not None and float(row.get("best_raw_iou", 0.0)) >= 0.25:
            result.add(
                (
                    str(row["scene_id"]),
                    str(row.get("gt_class", "")),
                    int(cluster),
                )
            )
    return result


def nested_sampling_gate(
    source_rows: Sequence[Mapping[str, Any]],
    nested_rows: Sequence[Mapping[str, Any]],
    *,
    source_candidate_count: int,
    nested_candidate_count: int,
    nested_trace_identity: bool,
) -> dict[str, Any]:
    """Apply the preregistered nested 5k/10k sampling gate."""

    source_keys = _raw_cluster_keys(source_rows)
    nested_keys = _raw_cluster_keys(nested_rows)
    source_objects = sorted(
        (
            str(row.get("scene_id", "")),
            str(row.get("gt_class", "")),
            str(row.get("gt_instance_id", "")),
        )
        for row in source_rows
    )
    nested_objects = sorted(
        (
            str(row.get("scene_id", "")),
            str(row.get("gt_class", "")),
            str(row.get("gt_instance_id", "")),
        )
        for row in nested_rows
    )
    # HDBSCAN cluster integers are not identities across the 5k and 10k fits.
    # The registered intervention is therefore a count gain, not a set
    # difference of unstable numeric labels.
    new_count = max(0, len(nested_keys) - len(source_keys))
    source_recall = (
        sum(float(row.get("best_raw_iou", 0.0)) >= 0.25 for row in source_rows)
        / len(source_rows)
        if source_rows
        else 0.0
    )
    nested_recall = (
        sum(float(row.get("best_raw_iou", 0.0)) >= 0.25 for row in nested_rows)
        / len(nested_rows)
        if nested_rows
        else 0.0
    )
    ratio = nested_candidate_count / max(source_candidate_count, 1)
    checks = {
        "nested_trace_identity": bool(nested_trace_identity),
        "diagnosable_object_universe_unchanged": source_objects == nested_objects,
        "raw_formation_improved": new_count >= 2
        or nested_recall - source_recall >= 0.10,
        "candidate_count_at_most_1.5x": nested_candidate_count
        <= 1.5 * max(source_candidate_count, 1),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "new_raw_iou025_cluster_count": new_count,
        "source_raw_recall025": source_recall,
        "nested_raw_recall025": nested_recall,
        "raw_recall025_delta": nested_recall - source_recall,
        "source_candidate_count": int(source_candidate_count),
        "nested_candidate_count": int(nested_candidate_count),
        "candidate_count_ratio": ratio,
        "nested_trace_identity": bool(nested_trace_identity),
    }


def _assert_nested_trace_identity(
    source_root: Path, nested_root: Path, scene_ids: Sequence[str]
) -> bool:
    import numpy as np

    from .category_candidate_trace import load_candidate_formation_trace

    for scene_id in map(str, scene_ids):
        source = load_candidate_formation_trace(
            source_root / "candidate_trace" / scene_id
        )
        nested = load_candidate_formation_trace(
            nested_root / "candidate_trace" / scene_id
        )
        left_class = np.asarray(source.semantic_selected_class_index)
        right_class = np.asarray(nested.semantic_selected_class_index)
        if not np.array_equal(left_class, right_class):
            raise ValueError(f"{scene_id}: 5k/10k semantic selection differs")
        left_rank = np.asarray(source.sample_rank, dtype=np.int64)
        right_rank = np.asarray(nested.sample_rank, dtype=np.int64)
        if left_rank.shape != right_rank.shape:
            raise ValueError(f"{scene_id}: 5k/10k sample-rank axes differ")
        selected = left_rank >= 0
        if not np.array_equal(left_rank[selected], right_rank[selected]):
            raise ValueError(f"{scene_id}: 10k is not a nested extension of 5k")
    return True


def _candidate_count(root: Path, scene_ids: Sequence[str]) -> int:
    from .category_denoise import load_candidate_bank

    return sum(
        len(
            load_candidate_bank(
                root / "bank" / str(scene_id) / "C0-legacy"
            ).candidates
        )
        for scene_id in scene_ids
    )


def _default_nested_sampling_control(**kwargs: Any) -> Mapping[str, Any]:
    config: CandidateExperimentConfig = kwargs["config"]
    nested_root = Path(kwargs["nested_run_root"])
    scenes = tuple(map(str, kwargs["scene_ids"]))
    _default_repair(
        config=config,
        run_root=nested_root,
        scene_ids=scenes,
        sample_cap=NESTED_SAMPLE_CAP,
        require_reference_identity=False,
    )
    nested_rows_path = config.artifacts_root / "root_nested_10k.parquet"
    _default_diagnose(
        config=config,
        run_root=nested_root,
        scene_ids=scenes,
        trace_output=nested_rows_path,
        analysis_output=config.artifacts_root / "root_nested_10k.analysis.json",
    )
    identity = _assert_nested_trace_identity(
        Path(kwargs["source_run_root"]), nested_root, scenes
    )
    return nested_sampling_gate(
        read_rows(kwargs["source_trace_rows"]),
        read_rows(nested_rows_path),
        source_candidate_count=_candidate_count(
            Path(kwargs["source_run_root"]), scenes
        ),
        nested_candidate_count=_candidate_count(nested_root, scenes),
        nested_trace_identity=identity,
    )


def _default_representation_diagnostic(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_representation import (
        evaluate_candidate_representation,
    )
    from .taxonomy import load_taxonomy

    config: CandidateExperimentConfig = kwargs["config"]
    return evaluate_candidate_representation(
        runtime_manifest=Path(
            kwargs.get("runtime_manifest", config.runtime_manifest)
        ),
        gt_dir=config.gt_dir,
        run_root=kwargs["run_root"],
        scene_ids=kwargs["scene_ids"],
        taxonomy=load_taxonomy(),
        metrics_output=kwargs["metrics_output"],
        analysis_output=kwargs["analysis_output"],
        size_bins=config.size_bins,
    )


def _c0_iou025_candidate_count(result: Mapping[str, Any]) -> int:
    conditions = result.get("conditions")
    if not isinstance(conditions, Mapping):
        raise TypeError("candidate evaluation lacks its conditions table")
    c0 = conditions.get("C0-legacy")
    if not isinstance(c0, Mapping):
        raise TypeError("candidate evaluation lacks C0-legacy")
    return int(c0.get("same_class_iou_025_count", -1))


def same_source_feature_10k_gate(
    *,
    source_representation: Mapping[str, Any],
    control_representation: Mapping[str, Any],
    source_candidates: Mapping[str, Any],
    control_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered two-scene feature-capacity positive-control gate."""

    source_auc = float(source_representation["mean_local_affinity_edge_auroc"])
    control_auc = float(control_representation["mean_local_affinity_edge_auroc"])
    source_iou025 = _c0_iou025_candidate_count(source_candidates)
    control_iou025 = _c0_iou025_candidate_count(control_candidates)
    auc_delta = control_auc - source_auc
    iou025_delta = control_iou025 - source_iou025
    checks = {
        "mean_affinity_auroc_improved_at_least_0.05": auc_delta >= 0.05,
        "added_at_least_two_same_class_iou025_candidates": iou025_delta >= 2,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "source_mean_affinity_auroc": source_auc,
        "control_mean_affinity_auroc": control_auc,
        "mean_affinity_auroc_delta": auc_delta,
        "source_same_class_iou025_candidate_count": source_iou025,
        "control_same_class_iou025_candidate_count": control_iou025,
        "same_class_iou025_candidate_delta": iou025_delta,
    }


def _default_feature_10k_control(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_feature_control import (
        bind_control_candidate_root,
        materialize_feature_runtime_manifest,
        run_same_source_feature_10k,
    )

    config: CandidateExperimentConfig = kwargs["config"]
    scenes = tuple(map(str, kwargs["scene_ids"]))
    if scenes != DEV2:
        raise ValueError("the same-source 10k control is frozen to DEV2")
    source_root = Path(kwargs["source_run_root"]).resolve()
    source_manifest = Path(kwargs["source_runtime_manifest"]).resolve()
    control_root = Path(kwargs["control_run_root"]).resolve()
    control_manifest = Path(kwargs["control_runtime_manifest"]).resolve()
    sample_cap = int(kwargs["sample_cap"])
    training = run_same_source_feature_10k(
        workspace=config.repo_root,
        python_bin=(config.python_bin or Path(sys.executable)),
        runtime_manifest=source_manifest,
        training_root=Path(kwargs["training_root"]),
        scene_ids=DEV2,
    )
    materialize_feature_runtime_manifest(
        source_manifest=source_manifest,
        training_payload=training,
        output=control_manifest,
    )
    bind_control_candidate_root(
        runtime_manifest=control_manifest,
        control_root=control_root,
        scene_ids=DEV2,
        sample_cap=sample_cap,
        seed=config.seed,
    )
    _default_repair(
        config=config,
        runtime_manifest=control_manifest,
        run_root=control_root,
        scene_ids=DEV2,
        sample_cap=sample_cap,
        require_reference_identity=False,
    )

    source_representation = _default_representation_diagnostic(
        config=config,
        runtime_manifest=source_manifest,
        run_root=source_root,
        scene_ids=DEV2,
        metrics_output=config.artifacts_root / "representation_2k_control.parquet",
        analysis_output=config.artifacts_root / "representation_2k_control.json",
    )
    control_representation = _default_representation_diagnostic(
        config=config,
        runtime_manifest=control_manifest,
        run_root=control_root,
        scene_ids=DEV2,
        metrics_output=config.artifacts_root / "representation_10k_control.parquet",
        analysis_output=config.artifacts_root / "representation_10k_control.json",
    )
    source_candidates = _default_evaluate_repair(
        config=config,
        runtime_manifest=source_manifest,
        run_root=source_root,
        scene_ids=DEV2,
        phase="dev2",
        selected_condition=None,
        frozen_repair_artifact=None,
        metrics_output=config.artifacts_root / "candidate_2k_control.parquet",
        analysis_output=config.artifacts_root / "candidate_2k_control.json",
    )
    control_candidates = _default_evaluate_repair(
        config=config,
        runtime_manifest=control_manifest,
        run_root=control_root,
        scene_ids=DEV2,
        phase="dev2",
        selected_condition=None,
        frozen_repair_artifact=None,
        metrics_output=config.artifacts_root / "candidate_10k_control.parquet",
        analysis_output=config.artifacts_root / "candidate_10k_control.json",
    )
    gate = same_source_feature_10k_gate(
        source_representation=source_representation,
        control_representation=control_representation,
        source_candidates=source_candidates,
        control_candidates=control_candidates,
    )
    payload = {
        "schema": REGISTERED_10K_SCHEMA,
        "scene_ids": list(DEV2),
        "feature_iterations": 10_000,
        "feature_seed": int(training["feature_seed"]),
        "sample_cap": sample_cap,
        "source_run_root": str(source_root),
        "source_runtime_manifest": str(source_manifest),
        "control_run_root": str(control_root),
        "control_runtime_manifest": str(control_manifest),
        "same_source_inputs": True,
        "only_training_budget_changed": True,
        "training": dict(training),
        **gate,
    }
    write_json(kwargs["output"], payload)
    return payload


def _default_evaluate_repair(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_evaluation import evaluate_category_candidates
    from .taxonomy import load_taxonomy

    config: CandidateExperimentConfig = kwargs["config"]
    return evaluate_category_candidates(
        runtime_manifest=Path(
            kwargs.get("runtime_manifest", config.runtime_manifest)
        ),
        gt_dir=config.gt_dir,
        run_root=kwargs["run_root"],
        scene_ids=kwargs["scene_ids"],
        taxonomy=load_taxonomy(),
        metrics_output=kwargs["metrics_output"],
        analysis_output=kwargs["analysis_output"],
        phase=kwargs["phase"],
        selected_condition=kwargs["selected_condition"],
        frozen_repair_artifact=kwargs.get("frozen_repair_artifact"),
        size_bins=config.size_bins,
    )


def _default_evaluate_prior_oracle(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_prior_oracle_v2 import evaluate_prior_oracle_v2

    config: CandidateExperimentConfig = kwargs["config"]
    return evaluate_prior_oracle_v2(
        prior_oracle_root=config.prior_oracle_root,
        category_priors=config.category_priors,
        output=kwargs["output"],
    )


def _prepare_candidate_prior_inputs(
    *,
    config: CandidateExperimentConfig,
    run_root: Path,
    selected: str,
    repair_analysis: Path,
    scene_ids: Sequence[str],
    include_official_gt: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    from .category_candidate_evaluation import _scene_context
    from .category_candidate_prior_evaluation import (
        join_candidate_prior_rows,
        normalize_official_gt_rows,
    )
    from .category_candidate_prior_v2 import score_same_bank_candidate_priors
    from .category_denoise import load_candidate_bank
    from .runner import load_scene_runtime_manifest
    from .taxonomy import load_taxonomy

    priors = load_json(config.category_priors)
    repair = load_json(repair_analysis)
    condition = repair["conditions"][selected]
    labels_by_scene = {
        str(row["scene_id"]): row["candidate_rows"]
        for row in condition["per_scene"]
    }
    candidate_rows: list[dict[str, Any]] = []
    uniform_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        bank = load_candidate_bank(
            run_root / "bank" / scene_id / selected
        )
        candidates: list[dict[str, Any]] = []
        for raw in bank.candidates:
            row = {"scene_id": scene_id, **dict(raw)}
            row.setdefault(
                "core_point_count", int(row["trusted_core_point_count"])
            )
            candidates.append(row)
        scores = score_same_bank_candidate_priors(candidates, priors)
        candidate_rows.extend(candidates)
        uniform_rows.extend({**dict(row), "score": float(row["S"])} for row in scores.uniform)
        class_rows.extend({**dict(row), "score": float(row["S"])} for row in scores.class_shrunk)
        for raw in labels_by_scene[scene_id]:
            target = raw.get("best_same_class_instance_id")
            label_rows.append(
                {
                    "scene_id": scene_id,
                    "candidate_id": int(raw["candidate_id"]),
                    "same_class_iou": float(raw["best_same_class_iou"]),
                    "matched_gt_class": str(raw["branch_class"])
                    if target is not None
                    else None,
                    "matched_gt_instance_id": int(target)
                    if target is not None
                    else None,
                    "matched_gt_size_bin": raw.get("best_same_class_size_bin"),
                }
            )

    examples = join_candidate_prior_rows(
        candidate_rows=candidate_rows,
        uniform_score_rows=uniform_rows,
        class_score_rows=class_rows,
        label_rows=label_rows,
    )
    if not include_official_gt:
        return examples, ()

    taxonomy = load_taxonomy()
    scenes = load_scene_runtime_manifest(config.runtime_manifest)
    size_spec = load_json(config.size_bins) if config.size_bins is not None else None
    official_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        context = _scene_context(
            scene_id=scene_id,
            scene=scenes[scene_id],
            gt_dir=config.gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=0.05,
            min_region_size=100,
        )
        official_rows.extend(
            {
                "scene_id": scene_id,
                "class_name": item.class_name,
                "instance_id": item.instance_id,
                "size_bin": item.size_bin,
            }
            for item in context["objects"]
        )
    return examples, normalize_official_gt_rows(official_rows)


def _default_evaluate_candidate_prior(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_prior_evaluation import evaluate_candidate_prior_dev8

    config: CandidateExperimentConfig = kwargs["config"]
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    examples, official_gt = _prepare_candidate_prior_inputs(
        config=config,
        run_root=Path(kwargs["run_root"]),
        selected=str(kwargs["selected_condition"]),
        repair_analysis=Path(kwargs["repair_analysis"]),
        scene_ids=DEV8,
        include_official_gt=True,
    )
    evaluation = evaluate_candidate_prior_dev8(
        examples=examples,
        official_gt=official_gt,
    )
    evaluation_payload = evaluation.to_dict()
    write_rows(
        kwargs["metrics_output"],
        [
            {
                "scene_id": row.scene_id,
                "candidate_id": row.candidate_id,
                "branch_class": row.branch_class,
                "trusted_core_point_count": row.core_point_count,
                "Q": row.q_value,
                "S_uniform": row.uniform_score,
                "S_class": row.class_score,
                "uniform_support_pass": row.uniform_support_pass,
                "class_support_pass": row.class_support_pass,
                "same_class_iou": row.same_class_iou,
                "matched_gt_class": row.matched_gt_class,
                "matched_gt_instance_id": row.matched_gt_instance_id,
                "matched_gt_size_bin": row.matched_gt_size_bin,
            }
            for row in examples
        ],
    )
    write_json(output_dir / "dev8_analysis.json", evaluation_payload)
    return evaluation_payload


def _default_select_candidate_threshold(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_prior_evaluation import select_uniform_threshold_dev2

    config: CandidateExperimentConfig = kwargs["config"]
    examples, _ = _prepare_candidate_prior_inputs(
        config=config,
        run_root=Path(kwargs["run_root"]),
        selected=str(kwargs["selected_condition"]),
        repair_analysis=Path(kwargs["repair_analysis"]),
        scene_ids=DEV2,
        include_official_gt=False,
    )
    threshold = select_uniform_threshold_dev2(examples)
    payload = threshold.to_dict()
    write_json(kwargs["output"], payload)
    return payload


def _materialize_selected_banks(
    source_root: Path,
    destination_root: Path,
    scene_ids: Sequence[str],
    selected_condition: str,
) -> None:
    from .category_candidate_trace import assert_candidate_bank_identity
    from .category_denoise import load_candidate_bank, save_candidate_bank

    for scene_id in map(str, scene_ids):
        source = load_candidate_bank(
            source_root / "bank" / scene_id / selected_condition
        )
        destination = destination_root / scene_id
        try:
            existing = load_candidate_bank(destination)
            assert_candidate_bank_identity(source, existing)
            continue
        except (FileNotFoundError, OSError, EOFError, KeyError, TypeError, ValueError):
            pass
        save_candidate_bank(source, destination)
        assert_candidate_bank_identity(source, load_candidate_bank(destination))


def _selected_bank_replay_identity(
    *,
    run_root: Path,
    scene_ids: Sequence[str],
    selected_condition: str,
) -> dict[str, Any]:
    from .category_denoise import load_candidate_bank

    scenes: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        bank_root = run_root / "bank" / scene_id / selected_condition
        bank_path = bank_root / "bank_labels.npz"
        bank = load_candidate_bank(bank_root)
        candidates = []
        for raw in sorted(bank.candidates, key=lambda row: int(row["candidate_id"])):
            candidate_id = int(raw["candidate_id"])
            q_value = float(raw.get("Q", raw.get("base_score")))
            candidates.append({"candidate_id": candidate_id, "Q": q_value})
            q_rows.append(
                {"scene_id": scene_id, "candidate_id": candidate_id, "Q": q_value}
            )
        scenes.append(
            {
                "scene_id": scene_id,
                "bank_sha256": sha256_file(bank_path),
                "point_count": bank.point_count,
                "bank_seed": bank.seed,
                "candidate_count": len(candidates),
                "candidate_id_q_signature": hash_json(candidates),
            }
        )
    return {
        "scenes": scenes,
        "bank_signature": hash_json(scenes),
        "candidate_id_q_signature": hash_json(q_rows),
    }


def _expected_replay_identity(
    *,
    config: CandidateExperimentConfig,
    runtime_manifest: Path,
    run_root: Path,
    scene_ids: Sequence[str],
    selected_condition: str,
    threshold: float,
    stage: str,
) -> dict[str, Any]:
    bank = _selected_bank_replay_identity(
        run_root=run_root,
        scene_ids=scene_ids,
        selected_condition=selected_condition,
    )
    return {
        "schema": REPLAY_IDENTITY_SCHEMA,
        "stage": str(stage),
        "runtime_manifest": str(Path(runtime_manifest).resolve()),
        "runtime_manifest_sha256": sha256_file(runtime_manifest),
        "category_priors": str(config.category_priors.resolve()),
        "category_priors_sha256": sha256_file(config.category_priors),
        "size_bins_sha256": sha256_file(config.size_bins),
        "seed": int(config.seed),
        "knn_k": 256,
        "min_count": 10,
        "threshold": float(threshold),
        "modes": ["uniform", "class"],
        "selected_condition": selected_condition,
        "scene_ids": list(map(str, scene_ids)),
        **bank,
    }


def _bind_replay_identity(output_dir: Path, expected: Mapping[str, Any]) -> Path:
    path = output_dir / "replay_identity.json"
    if path.is_file():
        if load_json(path) != expected:
            raise ValueError("replay recovery identity differs from frozen inputs")
        return path
    replay_root = output_dir / "runs" / "replay"
    if replay_root.is_dir() and any(replay_root.rglob("output.json")):
        raise ValueError("unbound replay outputs exist without a recovery identity")
    write_json(path, dict(expected))
    return path


def _validate_replay_outputs(
    *,
    replay_run_root: Path,
    source_run_root: Path,
    scene_ids: Sequence[str],
    selected_condition: str,
    threshold: float,
) -> None:
    from .category_denoise import load_candidate_bank

    for scene_id in map(str, scene_ids):
        bank = load_candidate_bank(
            source_run_root / "bank" / scene_id / selected_condition
        )
        expected_q = {
            int(row["candidate_id"]): float(row.get("Q", row.get("base_score")))
            for row in bank.candidates
        }
        expected_k = min(256, bank.point_count) if bank.point_count else 0
        for mode in ("uniform", "class"):
            path = replay_run_root / "replay" / mode / scene_id / "diagnostics.json"
            diagnostics = load_json(path)
            payload = (
                diagnostics.get("category_denoise")
                if isinstance(diagnostics, Mapping)
                else None
            )
            if not isinstance(payload, Mapping):
                raise TypeError(f"invalid replay diagnostics: {mode}/{scene_id}")
            checks = {
                "action": payload.get("action") == "candidate-replay",
                "mode": payload.get("mode") == mode,
                "scene": payload.get("scene_id") == scene_id,
                "threshold": abs(float(payload.get("score_threshold", -1.0)) - threshold)
                <= 1e-12,
                "knn": int(payload.get("knn_k_effective", -1)) == expected_k,
                "filter": int(payload.get("filter_min_count", -1)) == 10,
                "unprotected": int(
                    payload.get("protected_or_reinserted_point_count", -1)
                )
                == 0,
                "no_second_vote": payload.get("secondary_class_vote_applied") is False,
            }
            if not all(checks.values()):
                failed = sorted(key for key, value in checks.items() if not value)
                raise ValueError(
                    f"{mode}/{scene_id}: replay identity checks failed: {failed}"
                )
            decisions = payload.get("decisions")
            survival = payload.get("candidate_survival")
            if not isinstance(decisions, list) or not isinstance(survival, list):
                raise TypeError(f"{mode}/{scene_id}: replay rows are missing")
            observed_q = {
                int(row["candidate_id"]): float(row.get("Q", row.get("base_score")))
                for row in decisions
            }
            survival_ids = {int(row["candidate_id"]) for row in survival}
            if observed_q != expected_q or survival_ids != set(expected_q):
                raise ValueError(f"{mode}/{scene_id}: candidate ID/Q identity changed")


def _normalize_replay_for_evaluation(
    replay_run_root: Path,
    evaluation_root: Path,
    scene_ids: Sequence[str],
) -> None:
    for mode in ("uniform", "class"):
        for scene_id in map(str, scene_ids):
            source = replay_run_root / "replay" / mode / scene_id
            destination = evaluation_root / mode / scene_id
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "output.json", destination / "output.json")
            diagnostics = load_json(source / "diagnostics.json")
            payload = diagnostics.get("category_denoise")
            if not isinstance(payload, Mapping):
                raise TypeError(f"invalid replay diagnostics: {mode}/{scene_id}")
            survival = payload.get("candidate_survival")
            if not isinstance(survival, list):
                raise TypeError(f"replay diagnostics lack survival rows: {mode}/{scene_id}")
            # Preserve the canonical replay diagnostics contract.  The final
            # evaluator consumes category_denoise.candidate_survival and also
            # needs the replay identity fields for safe resume validation.
            write_json(destination / "diagnostics.json", diagnostics)


def _default_replay_final_stage(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_final_evaluation import evaluate_candidate_final_stage
    from .category_candidate_runner import replay_repaired_category_candidates
    from .taxonomy import load_taxonomy

    config: CandidateExperimentConfig = kwargs["config"]
    output_dir = Path(kwargs["output_dir"])
    runtime_manifest = Path(kwargs.get("runtime_manifest", config.runtime_manifest))
    gt_dir = Path(kwargs.get("gt_dir", config.gt_dir))
    run_root = Path(kwargs["run_root"])
    selected_condition = str(kwargs["selected_condition"])
    threshold = float(kwargs["threshold"])
    expected_identity = _expected_replay_identity(
        config=config,
        runtime_manifest=runtime_manifest,
        run_root=run_root,
        scene_ids=kwargs["scene_ids"],
        selected_condition=selected_condition,
        threshold=threshold,
        stage=str(kwargs["stage"]),
    )
    identity_path = _bind_replay_identity(output_dir, expected_identity)
    bank_root = output_dir / "selected_bank"
    _materialize_selected_banks(
        run_root,
        bank_root,
        kwargs["scene_ids"],
        selected_condition,
    )
    replay_repaired_category_candidates(
        runtime_manifest,
        bank_root,
        output_dir / "runs",
        config.repo_root,
        config.category_priors,
        kwargs["scene_ids"],
        score_threshold=threshold,
        seed=config.seed,
        python_bin=config.python_bin,
    )
    if load_json(identity_path) != expected_identity:
        raise ValueError("replay identity artifact changed during execution")
    if _expected_replay_identity(
        config=config,
        runtime_manifest=runtime_manifest,
        run_root=run_root,
        scene_ids=kwargs["scene_ids"],
        selected_condition=selected_condition,
        threshold=threshold,
        stage=str(kwargs["stage"]),
    ) != expected_identity:
        raise ValueError("replay inputs changed during execution")
    _validate_replay_outputs(
        replay_run_root=output_dir / "runs",
        source_run_root=run_root,
        scene_ids=kwargs["scene_ids"],
        selected_condition=selected_condition,
        threshold=threshold,
    )
    evaluation_root = output_dir / "evaluation_predictions"
    _normalize_replay_for_evaluation(
        output_dir / "runs", evaluation_root, kwargs["scene_ids"]
    )
    return evaluate_candidate_final_stage(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        b0_root=run_root / "b0",
        replay_root=evaluation_root,
        scene_ids=kwargs["scene_ids"],
        taxonomy=load_taxonomy(),
        stage=kwargs["stage"],
        output_dir=output_dir / "evaluation",
        b0_condition="B0-global",
        uniform_condition="uniform",
        data_condition="class",
        size_bins=config.size_bins,
    )


def default_candidate_experiment_hooks() -> CandidateExperimentHooks:
    return CandidateExperimentHooks(
        check_resources=check_experiment_resources,
        validate_inputs=_default_validate_inputs,
        repair=_default_repair,
        check_b0_parity=_default_check_b0_parity,
        diagnose=_default_diagnose,
        nested_sampling_control=_default_nested_sampling_control,
        representation_diagnostic=_default_representation_diagnostic,
        feature_10k_control=_default_feature_10k_control,
        evaluate_repair=_default_evaluate_repair,
        evaluate_prior_oracle=_default_evaluate_prior_oracle,
        evaluate_candidate_prior=_default_evaluate_candidate_prior,
        select_candidate_threshold=_default_select_candidate_threshold,
        replay_final_stage=_default_replay_final_stage,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume the preregistered section-30 candidate experiment"
    )
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-gt-dir", type=Path, required=True)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--prior-oracle-root", type=Path, required=True)
    parser.add_argument("--reference-bank-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument(
        "--registered-10k-control",
        type=Path,
        help=(
            "Optional JSON output path for the controller-owned, same-source "
            "two-scene 10k feature control"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_category_candidate_experiment(
        CandidateExperimentConfig(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            locked_runtime_manifest=args.locked_runtime_manifest,
            locked_gt_dir=args.locked_gt_dir,
            locked_evaluation_scenes=args.locked_evaluation_scenes,
            repo_root=args.repo_root,
            category_priors=args.category_priors,
            prior_oracle_root=args.prior_oracle_root,
            reference_bank_root=args.reference_bank_root,
            output_root=args.output_root,
            size_bins=args.size_bins,
            python_bin=args.python_bin,
            registered_10k_control=args.registered_10k_control,
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["status"] in {"complete", "awaiting_registered_10k_control"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CandidateExperimentConfig",
    "CandidateExperimentHooks",
    "DEV2",
    "DEV8",
    "HOLDOUT5",
    "REGISTERED_10K_SCHEMA",
    "check_experiment_resources",
    "default_candidate_experiment_hooks",
    "nested_sampling_gate",
    "run_category_candidate_experiment",
]
