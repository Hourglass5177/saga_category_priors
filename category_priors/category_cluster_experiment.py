from __future__ import annotations

"""Recoverable controller for the section-31 clustering and prior experiment.

The controller owns only the preregistered execution order and artifact gates:

``DEV2 R0/R1/R2 -> conditional DEV2 G1 -> frozen selection -> DEV8``.

Candidate construction remains GT-free in :mod:`category_cluster_runner`.
Ground truth is only passed to offline evaluation.  A healthy DEV8 cluster
bank continues through the frozen section-30.4/30.5 same-bank prior, shared
legacy replay, holdout, tune24 and locked final48 gates.
"""

import argparse
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .category_cluster_bank import (
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
from .category_cluster_scene_evaluation import DEV2_SCENE_IDS, DEV8_SCENE_IDS
from .io import load_json, read_rows, write_json


DEV2 = DEV2_SCENE_IDS
DEV8 = DEV8_SCENE_IDS
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)
PRIMARY_CONDITIONS = (
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
G1_CONDITIONS = (R0_LEGACY, G1_MUTUAL_LOCAL_GRAPH)
PRIMARY_REPAIRS = (R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN)

STATE_SCHEMA = "saga-category-cluster-experiment-state-v1"
FROZEN_SELECTION_SCHEMA = "saga-category-cluster-frozen-selection-v1"
READY_STATUS = "ready_for_prior"
COMPLETE_STATUS = "complete"
TERMINAL_STATUSES = frozenset({"stopped", COMPLETE_STATUS})
EXPECTED_CGROUP_MAX_BYTES = 90 * 1024**3
MIN_DISK_FREE_GIB = 80.0
THRESHOLD_GRID = (0.05, 0.10, 0.15, 0.20, 0.25)


@dataclass(frozen=True)
class ClusterExperimentConfig:
    runtime_manifest: Path
    gt_dir: Path
    locked_runtime_manifest: Path
    locked_gt_dir: Path
    locked_evaluation_scenes: Path
    repo_root: Path
    category_priors: Path
    prior_oracle_root: Path
    reference_bank_root: Path
    reference_trace_root: Path
    output_root: Path
    size_bins: Path
    taxonomy: Path | None = None
    python_bin: Path | None = None
    seed: int = 42

    def __post_init__(self) -> None:
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
            "reference_trace_root",
            "output_root",
            "size_bins",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("taxonomy", "python_bin"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if isinstance(self.seed, bool) or int(self.seed) != 42:
            raise ValueError("section 31 freezes the experiment seed at 42")

    @property
    def state_path(self) -> Path:
        return self.output_root / "category_cluster_experiment_state.json"

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs"

    @property
    def dev2_run_root(self) -> Path:
        """Immutable direct-repeatability banks used by DEV2 recovery.

        DEV2 is a subset of DEV8.  Keeping its directly rebuilt banks under
        the ordinary run root allowed the later DEV8 reference-mode build to
        replace the measured evidence for those two scenes.  The controller
        deliberately keeps the two evidence classes in separate roots.
        """

        return self.output_root / "dev2_measured_runs"

    @property
    def artifacts_root(self) -> Path:
        return self.output_root / "artifacts"

    @property
    def distance_audit_path(self) -> Path:
        return self.artifacts_root / "cluster_distance_audit.json"

    @property
    def dev2_metrics_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev2.parquet"

    @property
    def dev2_analysis_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev2_analysis.json"

    @property
    def primary_dev2_metrics_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev2_primary.parquet"

    @property
    def primary_dev2_analysis_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev2_primary_analysis.json"

    @property
    def frozen_selection_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev2_selection.json"

    @property
    def dev8_metrics_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev8.parquet"

    @property
    def dev8_analysis_path(self) -> Path:
        return self.artifacts_root / "cluster_repair_dev8_analysis.json"

    @property
    def prior_oracle_path(self) -> Path:
        return self.artifacts_root / "prior_oracle_v2.json"

    @property
    def candidate_prior_metrics_path(self) -> Path:
        return self.artifacts_root / "candidate_prior_dev8.parquet"

    @property
    def candidate_prior_root(self) -> Path:
        return self.artifacts_root / "candidate_prior"

    @property
    def threshold_path(self) -> Path:
        return self.candidate_prior_root / "dev2_threshold.json"

    @property
    def final_run_root(self) -> Path:
        return self.output_root / "final48_cluster"

    @property
    def final_analysis_path(self) -> Path:
        return self.artifacts_root / "category_denoise_v3_analysis.json"

    def replay_root(self, stage: str) -> Path:
        if stage not in {"dev8", "holdout", "tune", "final"}:
            raise ValueError(f"unregistered replay stage: {stage}")
        return self.output_root / f"legacy_replay_{stage}"

    def replay_metrics_path(self, stage: str) -> Path:
        suffix = {
            "dev8": "dev8",
            "holdout": "holdout5",
            "tune": "tune24",
            "final": "final48",
        }.get(stage)
        if suffix is None:
            raise ValueError(f"unregistered replay stage: {stage}")
        return self.artifacts_root / f"category_denoise_v3_{suffix}.parquet"


Hook = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class ClusterExperimentHooks:
    """Injectable boundaries used by focused state-machine tests."""

    check_resources: Callable[[Path], Mapping[str, Any]]
    validate_inputs: Hook
    build_banks: Hook
    audit_distance: Hook
    evaluate_banks: Hook
    evaluate_prior_oracle: Hook
    evaluate_candidate_prior: Hook
    select_candidate_threshold: Hook
    replay_final_stage: Hook


def check_cluster_experiment_resources(
    root: Path,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Enforce the cloud disk/cgroup contract without reading host ``free``."""

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
        raise RuntimeError("df output lacks the available-KiB field")
    try:
        available_kib = int(fields[3])
    except ValueError as exc:
        raise RuntimeError("df available-KiB field is not an integer") from exc
    available_gib = available_kib / 1024**2
    if available_gib < MIN_DISK_FREE_GIB:
        raise RuntimeError(
            f"cluster experiment requires at least {MIN_DISK_FREE_GIB:.0f} GiB "
            f"available; found {available_gib:.1f} GiB"
        )

    current_path = Path(cgroup_root) / "memory.current"
    maximum_path = Path(cgroup_root) / "memory.max"
    events_path = Path(cgroup_root) / "memory.events"
    for path in (current_path, maximum_path, events_path):
        if not path.is_file():
            raise RuntimeError(f"required cgroup v2 resource file is missing: {path}")
    maximum_text = maximum_path.read_text(encoding="utf-8").strip()
    if maximum_text == "max":
        raise RuntimeError("expected memory.max=90 GiB; found max")
    try:
        maximum = int(maximum_text)
        current = int(current_path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise RuntimeError("cgroup memory values must be integers") from exc
    if maximum != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(
            "expected cgroup memory.max=90 GiB; " f"found {maximum_text} bytes"
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


def _identity(config: ClusterExperimentConfig) -> dict[str, Any]:
    return {
        "runtime_manifest": _resolved(config.runtime_manifest),
        "gt_dir": _resolved(config.gt_dir),
        "locked_runtime_manifest": _resolved(config.locked_runtime_manifest),
        "locked_gt_dir": _resolved(config.locked_gt_dir),
        "locked_evaluation_scenes": _resolved(
            config.locked_evaluation_scenes
        ),
        "repo_root": _resolved(config.repo_root),
        "category_priors": _resolved(config.category_priors),
        "prior_oracle_root": _resolved(config.prior_oracle_root),
        "reference_bank_root": _resolved(config.reference_bank_root),
        "reference_trace_root": _resolved(config.reference_trace_root),
        "output_root": _resolved(config.output_root),
        "dev2_run_root": _resolved(config.dev2_run_root),
        "run_root": _resolved(config.run_root),
        "size_bins": _resolved(config.size_bins),
        "taxonomy": _resolved(config.taxonomy),
        "python_bin": _resolved(config.python_bin),
        "seed": int(config.seed),
        "dev2": list(DEV2),
        "dev8": list(DEV8),
        "primary_conditions": list(PRIMARY_CONDITIONS),
        "conditional_graph_fallback": G1_MUTUAL_LOCAL_GRAPH,
        "category_prior_tested": False,
        "legacy_replay_tested": False,
    }


def _initial_state(config: ClusterExperimentConfig) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "checkpoint": "initialized",
        "current_stage": None,
        "next_stage": "validate_inputs",
        "identity": _identity(config),
        "history": [],
        "selected_condition": None,
        "frozen_selection_artifact": None,
        "g1_authorized": False,
        "g1_tested": False,
        "tune24_scene_ids": None,
        "final48_scene_ids": None,
        "frozen_threshold": None,
        "prior_capacity_tested": False,
        "category_prior_tested": False,
        "legacy_replay_tested": False,
    }


def _write_state(config: ClusterExperimentConfig, state: Mapping[str, Any]) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    config.dev2_run_root.mkdir(parents=True, exist_ok=True)
    config.run_root.mkdir(parents=True, exist_ok=True)
    write_json(config.state_path, _json_safe(state))
    write_json(
        config.final_analysis_path,
        {
            "schema": "saga-category-denoise-v3-analysis-v1",
            "status": state.get("status"),
            "checkpoint": state.get("checkpoint"),
            "current_stage": state.get("current_stage"),
            "next_stage": state.get("next_stage"),
            "stop_reason": state.get("stop_reason"),
            "last_error": _json_safe(state.get("last_error")),
            "selected_condition": state.get("selected_condition"),
            "frozen_threshold": state.get("frozen_threshold"),
            "g1_tested": bool(state.get("g1_tested", False)),
            "prior_capacity_tested": bool(
                state.get("prior_capacity_tested", False)
            ),
            "category_prior_tested": bool(
                state.get("category_prior_tested", False)
            ),
            "legacy_replay_tested": bool(
                state.get("legacy_replay_tested", False)
            ),
            "history": _json_safe(state.get("history", [])),
        },
    )


def _load_state(config: ClusterExperimentConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        state = _initial_state(config)
        _write_state(config, state)
        return state
    value = load_json(config.state_path)
    if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
        raise ValueError(f"invalid cluster experiment state: {config.state_path}")
    state = dict(value)
    if state.get("identity") != _identity(config):
        raise ValueError("cluster experiment identity differs from recoverable state")
    if not isinstance(state.get("history"), list):
        raise TypeError("cluster experiment history must be a list")
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


def _record(
    state: dict[str, Any],
    stage: str,
    result: Mapping[str, Any],
    keys: Sequence[str],
) -> None:
    summary = {
        key: _json_safe(result[key])
        for key in keys
        if key in result
    }
    state["history"].append({"stage": stage, **summary})
    state["checkpoint"] = stage
    state["current_stage"] = None
    state.pop("last_error", None)


def _stop(
    config: ClusterExperimentConfig,
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


def _continue_to_prior(
    config: ClusterExperimentConfig, state: dict[str, Any]
) -> dict[str, Any]:
    state.update(
        {
            "status": "running",
            "checkpoint": "dev8_health_passed",
            "current_stage": None,
            "next_stage": "prior_oracle_v2",
            "stop_reason": None,
            "prior_ready": True,
            "category_prior_tested": False,
        }
    )
    _write_state(config, state)
    return state


def _complete(
    config: ClusterExperimentConfig, state: dict[str, Any]
) -> dict[str, Any]:
    state.update(
        {
            "status": COMPLETE_STATUS,
            "checkpoint": "final48_passed",
            "current_stage": None,
            "next_stage": None,
            "stop_reason": None,
            "category_prior_tested": True,
            "legacy_replay_tested": True,
        }
    )
    _write_state(config, state)
    return state


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
        raise ValueError("locked final runtime must contain 48 unique scans")
    if len({scene.rsplit("_", 1)[0] for scene in final}) != 48:
        raise ValueError("final48 must contain 48 distinct physical scenes")
    if "scene0019_01" not in final or "scene0019_00" in final:
        raise ValueError("final48 must use the registered scene0019_01 scan")
    return tune, final


def _validate_build_result(
    result: Mapping[str, Any],
    *,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    reference_identity_required: bool,
    verify_determinism: bool,
) -> None:
    expected_scenes = tuple(map(str, scene_ids))
    if int(result.get("total", -1)) != len(expected_scenes) or int(
        result.get("complete", -1)
    ) != len(expected_scenes):
        raise ValueError("cluster bank stage is not complete for every registered scene")
    observed_conditions = tuple(map(str, result.get("conditions", ())))
    if observed_conditions != tuple(map(str, conditions)):
        raise ValueError("cluster bank stage used a different condition set")
    if bool(result.get("reference_identity_required")) != bool(
        reference_identity_required
    ):
        raise ValueError("cluster bank stage used the wrong reference-identity boundary")
    expected_determinism_mode = (
        "measured_this_scene"
        if verify_determinism
        else "algorithm_contract_reference"
    )
    if result.get("determinism_mode") != expected_determinism_mode:
        raise ValueError("cluster bank stage used the wrong determinism mode")
    if verify_determinism:
        if result.get("determinism_reference") is not None:
            raise ValueError("directly measured banks must not cite an algorithm reference")
    elif not isinstance(result.get("determinism_reference"), Mapping):
        raise ValueError("later-stage banks require a verified DEV2 determinism reference")
    runs = result.get("runs")
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise TypeError("cluster bank result must include per-scene runs")
    observed_scenes = tuple(str(row.get("scene_id")) for row in runs)
    if len(observed_scenes) != len(set(observed_scenes)) or set(
        observed_scenes
    ) != set(expected_scenes):
        raise ValueError("cluster bank result does not cover the registered scene set")
    if any(
        str(row.get("status")) not in {"complete", "skipped_complete"}
        for row in runs
    ):
        raise ValueError("cluster bank result contains an incomplete scene")


def _audit_contract_passed(result: Mapping[str, Any]) -> bool:
    if not bool(result.get("r0_identity_passed")):
        return False
    if not bool(result.get("corrected_distance_contract_measured")):
        return False
    if not bool(result.get("determinism_passed")):
        return False
    # The integration audit may expose one of these equivalent aggregate keys.
    # If present, it is a hard gate; older audited payloads remain valid only
    # when every per-scene corrected arm reports the required metric scale.
    for key in (
        "corrected_distance_contract_passed",
        "distance_contract_passed",
    ):
        if key in result and not bool(result[key]):
            return False
    scenes = result.get("scenes")
    if not isinstance(scenes, Sequence) or isinstance(scenes, (str, bytes)):
        return False
    if {str(row.get("scene_id")) for row in scenes} != set(DEV2):
        return False
    for row in scenes:
        checks = row.get("r0_raw_identity_checks")
        if not isinstance(checks, Mapping) or not checks or not all(
            bool(value) for value in checks.values()
        ):
            return False
        corrected = row.get("corrected_conditions")
        if not isinstance(corrected, Sequence) or isinstance(
            corrected, (str, bytes)
        ):
            return False
        by_condition = {str(item.get("condition")): item for item in corrected}
        if set(PRIMARY_REPAIRS).difference(by_condition):
            return False
        if any(
            float(by_condition[condition].get("global_typical_diag_m", 0.0)) <= 0
            for condition in PRIMARY_REPAIRS
        ):
            return False
        if any(
            not bool(
                by_condition[condition].get(
                    "corrected_distance_contract_measured"
                )
            )
            or int(by_condition[condition].get("distance_matrix_count", 0)) <= 0
            for condition in PRIMARY_REPAIRS
        ):
            return False
        if any(
            not bool(
                by_condition[condition].get("corrected_distance_contract_passed")
            )
            for condition in PRIMARY_REPAIRS
        ):
            return False
        r0_determinism = row.get("r0_determinism")
        if not isinstance(r0_determinism, Mapping):
            return False
        if (
            not bool(r0_determinism.get("measured_this_scene"))
            or int(r0_determinism.get("violation_count", -1)) != 0
        ):
            return False
        if any(
            not bool(
                by_condition[condition].get("determinism_measured_this_scene")
            )
            or int(
                by_condition[condition].get(
                    "determinism_violation_count", -1
                )
            )
            != 0
            for condition in PRIMARY_REPAIRS
        ):
            return False
    return True


def _validate_analysis(
    result: Mapping[str, Any],
    *,
    phase: str,
    expected_scenes: Sequence[str],
) -> None:
    if result.get("phase") != phase:
        raise ValueError(f"cluster analysis phase must be {phase}")
    scenes = tuple(map(str, result.get("scene_ids", ())))
    if len(scenes) != len(set(scenes)) or set(scenes) != set(expected_scenes):
        raise ValueError(f"{phase} analysis does not use the registered scenes")
    if bool(result.get("category_prior_tested", True)):
        raise ValueError("cluster repair analysis must not test category priors")
    if R0_LEGACY not in result.get("conditions", {}):
        raise ValueError("cluster analysis is missing R0-legacy")
    gates = result.get("gates")
    if not isinstance(gates, Mapping):
        raise TypeError("cluster analysis gates must be a mapping")


def _require_evaluation_artifacts(
    *,
    metrics_path: Path,
    analysis_path: Path,
    stage: str,
    expected_analysis: Mapping[str, Any],
) -> None:
    missing = [
        str(path)
        for path in (metrics_path, analysis_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"{stage} evaluation artifacts are incomplete: {missing}"
        )
    stored = load_json(analysis_path)
    if stored != _json_safe(expected_analysis):
        raise RuntimeError(
            f"{stage} analysis artifact differs from the evaluated result"
        )


def _primary_failed(result: Mapping[str, Any]) -> bool:
    gates = result.get("gates", {})
    if not isinstance(gates, Mapping):
        raise TypeError("primary DEV2 gates must be a mapping")
    missing = [condition for condition in PRIMARY_REPAIRS if condition not in gates]
    if missing:
        raise ValueError(f"primary DEV2 analysis lacks gates for {missing}")
    return all(not bool((gates[condition] or {}).get("passed")) for condition in PRIMARY_REPAIRS)


def _freeze_selection(
    config: ClusterExperimentConfig,
    state: dict[str, Any],
    analysis: Mapping[str, Any],
) -> str:
    condition = str(analysis.get("selected_condition") or "")
    if condition not in (*PRIMARY_REPAIRS, G1_MUTUAL_LOCAL_GRAPH):
        raise ValueError("DEV2 selected an unregistered cluster repair")
    gate = analysis.get("selected_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed")):
        raise ValueError("DEV2 selection does not have a passing registered gate")
    payload = {
        "schema": FROZEN_SELECTION_SCHEMA,
        "phase": "dev2",
        "scene_ids": list(DEV2),
        "selected_condition": condition,
        "selected_gate": _json_safe(gate),
        "selection_tier": analysis.get("selection_tier"),
        "tie_rule": (
            "iou050_iou025_precision_unsupported_candidate_count_"
            "then_structural_simplicity_R1"
        ),
        "category_prior_tested": False,
    }
    path = config.frozen_selection_path
    if path.is_file() and bool(state.pop("replace_invalid_frozen_selection", False)):
        write_json(path, payload)
    elif path.is_file():
        if load_json(path) != payload:
            raise ValueError("existing DEV2 frozen selection has changed")
    else:
        write_json(path, payload)
    state["selected_condition"] = condition
    state["frozen_selection_artifact"] = str(path.resolve())
    return condition


def _load_frozen_selection(
    config: ClusterExperimentConfig, state: Mapping[str, Any]
) -> str:
    raw = state.get("frozen_selection_artifact")
    path = Path(str(raw)) if raw else config.frozen_selection_path
    if path.resolve() != config.frozen_selection_path.resolve() or not path.is_file():
        raise ValueError("DEV2 frozen selection artifact is missing or moved")
    payload = load_json(path)
    if not isinstance(payload, Mapping) or payload.get("schema") != FROZEN_SELECTION_SCHEMA:
        raise ValueError("DEV2 frozen selection artifact has the wrong schema")
    if payload.get("phase") != "dev2" or tuple(
        map(str, payload.get("scene_ids", ()))
    ) != DEV2:
        raise ValueError("DEV2 frozen selection artifact uses the wrong stage")
    condition = str(payload.get("selected_condition") or "")
    if condition not in (*PRIMARY_REPAIRS, G1_MUTUAL_LOCAL_GRAPH):
        raise ValueError("DEV2 frozen selection contains an unregistered condition")
    if condition != state.get("selected_condition"):
        raise ValueError("state and DEV2 frozen selection disagree")
    if not bool((payload.get("selected_gate") or {}).get("passed")):
        raise ValueError("DEV2 frozen selection does not contain a passing gate")
    return condition


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _publish_replay_metrics(
    config: ClusterExperimentConfig, *, stage: str
) -> None:
    source_stem = {
        "dev8": "dev8",
        "holdout": "holdout",
        "tune": "tune",
        "final": "final",
    }.get(stage)
    if source_stem is None:
        raise ValueError(f"unregistered replay stage: {stage}")
    _copy_file(
        config.replay_root(stage)
        / "evaluation"
        / f"{source_stem}_condition_metrics.parquet",
        config.replay_metrics_path(stage),
    )


def _validate_dev8_prior_handoff(
    config: ClusterExperimentConfig, state: Mapping[str, Any]
) -> str:
    selected = _load_frozen_selection(config, state)
    analysis = load_json(config.dev8_analysis_path)
    _validate_analysis(analysis, phase="dev8", expected_scenes=DEV8)
    if analysis.get("selected_condition") != selected:
        raise ValueError("DEV8 analysis and frozen cluster selection disagree")
    if not bool((analysis.get("selected_gate") or {}).get("passed")):
        raise ValueError("category prior requires a passing DEV8 candidate gate")
    if bool(analysis.get("category_prior_tested", True)):
        raise ValueError("DEV8 candidate-health analysis crossed the GT/prior boundary")
    return selected


def _json_mapping_valid(path: Path) -> bool:
    try:
        return path.is_file() and isinstance(load_json(path), Mapping)
    except (OSError, EOFError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _rows_artifact_valid(path: Path) -> bool:
    try:
        return path.is_file() and bool(read_rows(path))
    except (OSError, EOFError, TypeError, ValueError, ImportError):
        return False


def _terminal_recovery_stage(
    config: ClusterExperimentConfig, state: Mapping[str, Any]
) -> tuple[str, str, bool] | None:
    """Return the earliest invalid completed stage and whether selection is bad."""

    checkpoint = str(state.get("checkpoint") or "")
    completed_rank = {
        "dev2_all_registered_repairs_failed": 1,
        "dev8_candidate_health_failed": 2,
        "prior_oracle_v2_gate_failed": 3,
        "candidate_prior_dev8_gate_failed": 4,
        "legacy_replay_dev8_gate_failed": 5,
        "legacy_replay_holdout5_gate_failed": 6,
        "legacy_replay_tune24_gate_failed": 7,
        "legacy_replay_final48_gate_failed": 8,
        "final48_passed": 8,
    }.get(checkpoint)
    if completed_rank is None:
        raise ValueError(f"unregistered terminal checkpoint: {checkpoint}")

    selected = str(state.get("selected_condition") or "")
    try:
        distance_audit_valid = _audit_contract_passed(
            load_json(config.distance_audit_path)
        )
    except (OSError, EOFError, TypeError, ValueError, json.JSONDecodeError):
        distance_audit_valid = False
    if not distance_audit_valid:
        return "audit_distance", "cluster distance audit is invalid", False

    primary_valid = _rows_artifact_valid(config.primary_dev2_metrics_path) and (
        _json_mapping_valid(config.primary_dev2_analysis_path)
    )
    if not primary_valid:
        return "evaluate_dev2_primary", "DEV2 primary artifacts are invalid", False

    dev2_valid = _rows_artifact_valid(config.dev2_metrics_path) and (
        _json_mapping_valid(config.dev2_analysis_path)
    )
    if not dev2_valid:
        if bool(state.get("g1_tested")):
            return "evaluate_dev2_g1", "DEV2 G1 artifacts are invalid", False
        return "evaluate_dev2_primary", "DEV2 primary artifacts are invalid", False

    if selected:
        try:
            _load_frozen_selection(config, state)
        except (OSError, EOFError, TypeError, ValueError, json.JSONDecodeError):
            stage = (
                "evaluate_dev2_g1"
                if selected == G1_MUTUAL_LOCAL_GRAPH
                else "evaluate_dev2_primary"
            )
            return stage, "DEV2 frozen selection is invalid", True

    if completed_rank >= 2:
        dev8_valid = _rows_artifact_valid(config.dev8_metrics_path) and (
            _json_mapping_valid(config.dev8_analysis_path)
        )
        if not dev8_valid:
            return "evaluate_dev8", "DEV8 evaluation artifacts are invalid", False
    if completed_rank >= 3 and not _json_mapping_valid(config.prior_oracle_path):
        return "prior_oracle_v2", "prior oracle artifact is invalid", False
    if completed_rank >= 4:
        candidate_analysis = config.candidate_prior_root / "dev8_analysis.json"
        if not _rows_artifact_valid(config.candidate_prior_metrics_path) or not (
            _json_mapping_valid(candidate_analysis)
        ):
            return (
                "candidate_prior_dev8",
                "candidate-prior artifacts are invalid",
                False,
            )
    if completed_rank >= 5:
        if not _json_mapping_valid(config.threshold_path):
            return "select_threshold_dev2", "DEV2 threshold artifact is invalid", False
        threshold = load_json(config.threshold_path)
        if float(threshold.get("selected_threshold", -1.0)) not in THRESHOLD_GRID:
            return "select_threshold_dev2", "DEV2 threshold is outside the grid", False

    replay_stages = (
        (5, "dev8", "replay_dev8"),
        (6, "holdout", "replay_holdout5"),
        (7, "tune", "replay_tune24"),
        (8, "final", "replay_final48"),
    )
    for rank, public_stage, controller_stage in replay_stages:
        if completed_rank < rank:
            continue
        analysis = (
            config.replay_root(public_stage)
            / "evaluation"
            / f"{public_stage}_analysis.json"
        )
        if not _rows_artifact_valid(
            config.replay_metrics_path(public_stage)
        ) or not _json_mapping_valid(analysis):
            return (
                controller_stage,
                f"{public_stage} replay artifacts are invalid",
                False,
            )
    return None


def _recover_terminal_state(
    config: ClusterExperimentConfig, state: dict[str, Any]
) -> dict[str, Any]:
    recovery = _terminal_recovery_stage(config, state)
    if recovery is None:
        return state
    stage, reason, replace_selection = recovery
    state.update(
        {
            "status": "running",
            "checkpoint": f"{state.get('checkpoint')}_artifact_recovery",
            "current_stage": None,
            "next_stage": stage,
            "stop_reason": None,
            "last_error": {
                "type": "ArtifactRecovery",
                "message": reason,
            },
        }
    )
    if replace_selection:
        state["replace_invalid_frozen_selection"] = True
    _write_state(config, state)
    return state


def run_category_cluster_experiment(
    config: ClusterExperimentConfig,
    hooks: ClusterExperimentHooks | None = None,
) -> dict[str, Any]:
    """Run or resume the frozen Stage-0/DEV2/DEV8 state machine."""

    active = hooks or default_cluster_experiment_hooks()
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    config.dev2_run_root.mkdir(parents=True, exist_ok=True)
    config.run_root.mkdir(parents=True, exist_ok=True)
    state = _load_state(config)
    if state.get("status") == READY_STATUS:
        if state.get("checkpoint") != "dev8_health_passed":
            raise ValueError("ready_for_prior state lacks the DEV8 health checkpoint")
        state.update(
            {
                "status": "running",
                "current_stage": None,
                "next_stage": "prior_oracle_v2",
                "prior_ready": True,
            }
        )
        _write_state(config, state)
    if state.get("status") in TERMINAL_STATUSES:
        state = _recover_terminal_state(config, state)
    if state.get("status") in TERMINAL_STATUSES:
        _write_state(config, state)
        return state

    resource_stages = {
        "build_dev2_primary",
        "build_dev2_g1",
        "build_dev8",
        "replay_dev8",
        "build_holdout5",
        "replay_holdout5",
        "build_tune24",
        "replay_tune24",
        "build_final48",
        "replay_final48",
    }
    while state.get("status") not in TERMINAL_STATUSES:
        stage = str(state.get("next_stage") or "")
        if not stage:
            raise RuntimeError("running cluster experiment has no next_stage")
        state.update({"status": "running", "current_stage": stage})
        _write_state(config, state)
        try:
            if stage in resource_stages:
                resources = active.check_resources(config.output_root)
                state["last_resources"] = _json_safe(resources)
                _write_state(config, state)

            if stage == "validate_inputs":
                result = active.validate_inputs(config=config)
                scenes = tuple(map(str, result.get("scene_ids", ())))
                if len(scenes) != len(set(scenes)) or set(scenes) != set(DEV8):
                    raise ValueError("validated runtime does not contain exact DEV8")
                tune, final = _validated_scene_sets(result)
                _record(
                    state,
                    stage,
                    result,
                    (
                        "scene_ids",
                        "gt_boundary",
                        "tune24_scene_ids",
                        "final48_scene_ids",
                        "tune_physical_scene_count",
                        "final_physical_scene_count",
                    ),
                )
                state["tune24_scene_ids"] = list(tune)
                state["final48_scene_ids"] = list(final)
                state["next_stage"] = "build_dev2_primary"

            elif stage == "build_dev2_primary":
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=config.dev2_run_root,
                    scene_ids=DEV2,
                    conditions=PRIMARY_CONDITIONS,
                    require_reference_identity=True,
                    verify_determinism=True,
                    determinism_reference=None,
                )
                _validate_build_result(
                    result,
                    scene_ids=DEV2,
                    conditions=PRIMARY_CONDITIONS,
                    reference_identity_required=True,
                    verify_determinism=True,
                )
                _record(state, stage, result, ("total", "complete", "conditions"))
                state["next_stage"] = "audit_distance"

            elif stage == "audit_distance":
                result = active.audit_distance(
                    config=config,
                    run_root=config.dev2_run_root,
                    output_path=config.distance_audit_path,
                )
                if not config.distance_audit_path.is_file():
                    raise FileNotFoundError(config.distance_audit_path)
                if not _audit_contract_passed(result):
                    raise RuntimeError(
                        "R0 identity or corrected-distance contract failed; "
                        "DEV2 evaluation is blocked"
                    )
                _record(
                    state,
                    stage,
                    result,
                    (
                        "r0_identity_passed",
                        "corrected_distance_contract_passed",
                        "distance_contract_passed",
                    ),
                )
                state["next_stage"] = "evaluate_dev2_primary"

            elif stage == "evaluate_dev2_primary":
                result = active.evaluate_banks(
                    config=config,
                    run_root=config.dev2_run_root,
                    phase="dev2",
                    scene_ids=DEV2,
                    selected_condition=None,
                    primary_analysis=None,
                    frozen_selection_artifact=None,
                    metrics_output=config.dev2_metrics_path,
                    analysis_output=config.dev2_analysis_path,
                )
                _validate_analysis(result, phase="dev2", expected_scenes=DEV2)
                conditions = result.get("conditions", {})
                gates = result.get("gates", {})
                if G1_MUTUAL_LOCAL_GRAPH in conditions or G1_MUTUAL_LOCAL_GRAPH in gates:
                    raise ValueError(
                        "primary DEV2 evaluation must not contain the conditional G1 arm"
                    )
                _require_evaluation_artifacts(
                    metrics_path=config.dev2_metrics_path,
                    analysis_path=config.dev2_analysis_path,
                    stage="DEV2 primary",
                    expected_analysis=result,
                )
                _copy_file(config.dev2_metrics_path, config.primary_dev2_metrics_path)
                _copy_file(config.dev2_analysis_path, config.primary_dev2_analysis_path)
                _record(
                    state,
                    stage,
                    result,
                    ("selected_condition", "selection_tier", "gates"),
                )
                selected = result.get("selected_condition")
                if selected in PRIMARY_REPAIRS:
                    _freeze_selection(config, state, result)
                    state["next_stage"] = "build_dev8"
                elif selected is not None:
                    raise ValueError("primary DEV2 selected a non-primary condition")
                elif _primary_failed(result):
                    state["g1_authorized"] = True
                    state["next_stage"] = "build_dev2_g1"
                else:
                    raise ValueError(
                        "primary DEV2 produced no selection without both gates failing"
                    )

            elif stage == "build_dev2_g1":
                if not bool(state.get("g1_authorized")):
                    raise ValueError("G1 was not authorized by failed primary DEV2 gates")
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=config.dev2_run_root,
                    scene_ids=DEV2,
                    conditions=G1_CONDITIONS,
                    require_reference_identity=True,
                    verify_determinism=True,
                    determinism_reference=None,
                )
                _validate_build_result(
                    result,
                    scene_ids=DEV2,
                    conditions=G1_CONDITIONS,
                    reference_identity_required=True,
                    verify_determinism=True,
                )
                state["g1_tested"] = True
                _record(state, stage, result, ("total", "complete", "conditions"))
                state["next_stage"] = "evaluate_dev2_g1"

            elif stage == "evaluate_dev2_g1":
                if not bool(state.get("g1_authorized")):
                    raise ValueError("G1 evaluation lacks primary-failure authorization")
                if not config.primary_dev2_analysis_path.is_file():
                    raise FileNotFoundError(config.primary_dev2_analysis_path)
                result = active.evaluate_banks(
                    config=config,
                    run_root=config.dev2_run_root,
                    phase="dev2",
                    scene_ids=DEV2,
                    selected_condition=None,
                    primary_analysis=config.primary_dev2_analysis_path,
                    frozen_selection_artifact=None,
                    metrics_output=config.dev2_metrics_path,
                    analysis_output=config.dev2_analysis_path,
                )
                _validate_analysis(result, phase="dev2", expected_scenes=DEV2)
                conditions = result.get("conditions", {})
                gates = result.get("gates", {})
                if (
                    G1_MUTUAL_LOCAL_GRAPH not in conditions
                    or G1_MUTUAL_LOCAL_GRAPH not in gates
                ):
                    raise ValueError(
                        "conditional DEV2 evaluation is missing the registered G1 arm"
                    )
                _require_evaluation_artifacts(
                    metrics_path=config.dev2_metrics_path,
                    analysis_path=config.dev2_analysis_path,
                    stage="DEV2 G1",
                    expected_analysis=result,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("selected_condition", "selection_tier", "gates"),
                )
                if result.get("selected_condition") == G1_MUTUAL_LOCAL_GRAPH:
                    _freeze_selection(config, state, result)
                    state["next_stage"] = "build_dev8"
                elif result.get("selected_condition") is None:
                    return _stop(
                        config,
                        state,
                        checkpoint="dev2_all_registered_repairs_failed",
                        reason=(
                            "R1, R2 and the preregistered G1 fallback all failed "
                            "DEV2; category prior was not tested"
                        ),
                    )
                else:
                    raise ValueError("conditional G1 evaluation selected an invalid arm")

            elif stage == "build_dev8":
                selected = _load_frozen_selection(config, state)
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=config.run_root,
                    scene_ids=DEV8,
                    conditions=(R0_LEGACY, selected),
                    require_reference_identity=True,
                    verify_determinism=False,
                    determinism_reference=config.dev2_analysis_path,
                )
                _validate_build_result(
                    result,
                    scene_ids=DEV8,
                    conditions=(R0_LEGACY, selected),
                    reference_identity_required=True,
                    verify_determinism=False,
                )
                _record(
                    state,
                    stage,
                    result,
                    ("total", "complete", "conditions"),
                )
                state["next_stage"] = "evaluate_dev8"

            elif stage == "evaluate_dev8":
                selected = _load_frozen_selection(config, state)
                result = active.evaluate_banks(
                    config=config,
                    run_root=config.run_root,
                    phase="dev8",
                    scene_ids=DEV8,
                    selected_condition=selected,
                    primary_analysis=None,
                    frozen_selection_artifact=config.frozen_selection_path,
                    metrics_output=config.dev8_metrics_path,
                    analysis_output=config.dev8_analysis_path,
                )
                _validate_analysis(result, phase="dev8", expected_scenes=DEV8)
                _require_evaluation_artifacts(
                    metrics_path=config.dev8_metrics_path,
                    analysis_path=config.dev8_analysis_path,
                    stage="DEV8",
                    expected_analysis=result,
                )
                if result.get("selected_condition") != selected:
                    raise ValueError("DEV8 did not evaluate the frozen DEV2 selection")
                _record(
                    state,
                    stage,
                    result,
                    ("selected_condition", "selected_gate", "gates"),
                )
                if bool((result.get("selected_gate") or {}).get("passed")):
                    _continue_to_prior(config, state)
                else:
                    return _stop(
                        config,
                        state,
                        checkpoint="dev8_candidate_health_failed",
                        reason=(
                            "the frozen DEV2 repair failed the DEV8 candidate-health "
                            "gate; category prior was not tested"
                        ),
                    )

            elif stage == "prior_oracle_v2":
                _validate_dev8_prior_handoff(config, state)
                result = active.evaluate_prior_oracle(
                    config=config,
                    output=config.prior_oracle_path,
                )
                if not config.prior_oracle_path.is_file():
                    raise FileNotFoundError(config.prior_oracle_path)
                if load_json(config.prior_oracle_path) != _json_safe(result):
                    raise RuntimeError("prior oracle artifact differs from its result")
                state["prior_capacity_tested"] = True
                _record(state, stage, result, ("passed", "checks", "failed_checks"))
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="prior_oracle_v2_gate_failed",
                        reason=(
                            "the frozen complete/fragment/merge prior-capacity "
                            "gate failed"
                        ),
                    )
                state["next_stage"] = "candidate_prior_dev8"

            elif stage == "candidate_prior_dev8":
                selected = _validate_dev8_prior_handoff(config, state)
                analysis_path = config.candidate_prior_root / "dev8_analysis.json"
                result = active.evaluate_candidate_prior(
                    config=config,
                    run_root=config.run_root,
                    selected_condition=selected,
                    repair_analysis=config.dev8_analysis_path,
                    output_dir=config.candidate_prior_root,
                    metrics_output=config.candidate_prior_metrics_path,
                )
                state["category_prior_tested"] = True
                _require_evaluation_artifacts(
                    metrics_path=config.candidate_prior_metrics_path,
                    analysis_path=analysis_path,
                    stage="DEV8 candidate prior",
                    expected_analysis=result,
                )
                if result.get("acceptance_threshold") is not None:
                    raise ValueError("DEV8 candidate-prior gate must be threshold-free")
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
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="candidate_prior_dev8_gate_failed",
                        reason="the same-bank threshold-free candidate-prior gate failed",
                    )
                state["next_stage"] = "select_threshold_dev2"

            elif stage == "select_threshold_dev2":
                selected = _validate_dev8_prior_handoff(config, state)
                result = active.select_candidate_threshold(
                    config=config,
                    run_root=config.run_root,
                    selected_condition=selected,
                    repair_analysis=config.dev8_analysis_path,
                    output=config.threshold_path,
                )
                if not config.threshold_path.is_file():
                    raise FileNotFoundError(config.threshold_path)
                if load_json(config.threshold_path) != _json_safe(result):
                    raise RuntimeError("DEV2 threshold artifact differs from its result")
                threshold = float(result.get("selected_threshold", -1.0))
                if threshold not in THRESHOLD_GRID:
                    raise ValueError("selected threshold is outside the frozen DEV2 grid")
                state["frozen_threshold"] = threshold
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
                state["next_stage"] = "replay_dev8"

            elif stage == "replay_dev8":
                selected = _load_frozen_selection(config, state)
                result = active.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=config.run_root,
                    selected_condition=selected,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=DEV8,
                    stage="dev8",
                    output_dir=config.replay_root("dev8"),
                )
                _publish_replay_metrics(config, stage="dev8")
                state["legacy_replay_tested"] = True
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
                        checkpoint="legacy_replay_dev8_gate_failed",
                        reason="U/D failed the frozen shared-legacy DEV8 output gate",
                    )
                state["next_stage"] = "build_holdout5"

            elif stage == "build_holdout5":
                selected = _load_frozen_selection(config, state)
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=config.run_root,
                    scene_ids=HOLDOUT5,
                    conditions=(R0_LEGACY, selected),
                    require_reference_identity=False,
                    verify_determinism=False,
                    determinism_reference=config.dev2_analysis_path,
                )
                _validate_build_result(
                    result,
                    scene_ids=HOLDOUT5,
                    conditions=(R0_LEGACY, selected),
                    reference_identity_required=False,
                    verify_determinism=False,
                )
                _record(state, stage, result, ("total", "complete", "conditions"))
                state["next_stage"] = "replay_holdout5"

            elif stage == "replay_holdout5":
                selected = _load_frozen_selection(config, state)
                result = active.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=config.run_root,
                    selected_condition=selected,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=HOLDOUT5,
                    stage="holdout",
                    output_dir=config.replay_root("holdout"),
                )
                _publish_replay_metrics(config, stage="holdout")
                _record(
                    state,
                    stage,
                    result,
                    ("passed", "uniform_health", "data_minus_uniform"),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_holdout5_gate_failed",
                        reason="U/D failed the five-scene canonical holdout gate",
                    )
                state["next_stage"] = "build_tune24"

            elif stage == "build_tune24":
                selected = _load_frozen_selection(config, state)
                tune = tuple(map(str, state["tune24_scene_ids"]))
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    run_root=config.run_root,
                    scene_ids=tune,
                    conditions=(R0_LEGACY, selected),
                    require_reference_identity=False,
                    verify_determinism=False,
                    determinism_reference=config.dev2_analysis_path,
                )
                _validate_build_result(
                    result,
                    scene_ids=tune,
                    conditions=(R0_LEGACY, selected),
                    reference_identity_required=False,
                    verify_determinism=False,
                )
                _record(state, stage, result, ("total", "complete", "conditions"))
                state["next_stage"] = "replay_tune24"

            elif stage == "replay_tune24":
                selected = _load_frozen_selection(config, state)
                tune = tuple(map(str, state["tune24_scene_ids"]))
                result = active.replay_final_stage(
                    config=config,
                    runtime_manifest=config.runtime_manifest,
                    gt_dir=config.gt_dir,
                    run_root=config.run_root,
                    selected_condition=selected,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=tune,
                    stage="tune",
                    output_dir=config.replay_root("tune"),
                )
                _publish_replay_metrics(config, stage="tune")
                _record(
                    state,
                    stage,
                    result,
                    ("passed", "uniform_health", "data_minus_uniform"),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_tune24_gate_failed",
                        reason="U/D failed the 13-physical-scene tune24 gate",
                    )
                state["next_stage"] = "build_final48"

            elif stage == "build_final48":
                selected = _load_frozen_selection(config, state)
                final = tuple(map(str, state["final48_scene_ids"]))
                result = active.build_banks(
                    config=config,
                    runtime_manifest=config.locked_runtime_manifest,
                    run_root=config.final_run_root,
                    scene_ids=final,
                    conditions=(R0_LEGACY, selected),
                    require_reference_identity=False,
                    verify_determinism=False,
                    determinism_reference=config.dev2_analysis_path,
                )
                _validate_build_result(
                    result,
                    scene_ids=final,
                    conditions=(R0_LEGACY, selected),
                    reference_identity_required=False,
                    verify_determinism=False,
                )
                _record(state, stage, result, ("total", "complete", "conditions"))
                state["next_stage"] = "replay_final48"

            elif stage == "replay_final48":
                selected = _load_frozen_selection(config, state)
                final = tuple(map(str, state["final48_scene_ids"]))
                result = active.replay_final_stage(
                    config=config,
                    runtime_manifest=config.locked_runtime_manifest,
                    gt_dir=config.locked_gt_dir,
                    run_root=config.final_run_root,
                    selected_condition=selected,
                    threshold=float(state["frozen_threshold"]),
                    scene_ids=final,
                    stage="final",
                    output_dir=config.replay_root("final"),
                )
                _publish_replay_metrics(config, stage="final")
                _record(
                    state,
                    stage,
                    result,
                    ("passed", "uniform_health", "data_minus_uniform"),
                )
                if not bool(result.get("passed", False)):
                    return _stop(
                        config,
                        state,
                        checkpoint="legacy_replay_final48_gate_failed",
                        reason="U/D failed the locked final48 bootstrap gate",
                    )
                return _complete(config, state)

            else:
                raise ValueError(f"unknown cluster experiment stage: {stage}")

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
    from .taxonomy import load_taxonomy

    config: ClusterExperimentConfig = kwargs["config"]
    required_files = {
        "runtime manifest": config.runtime_manifest,
        "locked runtime manifest": config.locked_runtime_manifest,
        "locked evaluation scenes": config.locked_evaluation_scenes,
        "category priors": config.category_priors,
        "size bins": config.size_bins,
        "pipeline": config.repo_root / "run_pipeline.sh",
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    for label, path in {
        "GT directory": config.gt_dir,
        "locked GT directory": config.locked_gt_dir,
        "prior oracle root": config.prior_oracle_root,
        "reference bank root": config.reference_bank_root,
        "reference trace root": config.reference_trace_root,
    }.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if config.taxonomy is not None and not config.taxonomy.is_file():
        raise FileNotFoundError(config.taxonomy)
    if config.python_bin is not None and not config.python_bin.is_file():
        raise FileNotFoundError(config.python_bin)
    if (
        config.taxonomy is not None
        and load_taxonomy(config.taxonomy) != load_taxonomy()
    ):
        raise ValueError(
            "downstream frozen prior/replay evaluators require the canonical taxonomy"
        )
    scenes = load_scene_runtime_manifest(config.runtime_manifest)
    final_scenes = load_scene_runtime_manifest(config.locked_runtime_manifest)
    missing = sorted(set(DEV8).union(HOLDOUT5).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest lacks registered tune scenes: {missing}")
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
    final = tuple(final_scenes)
    if len(locked) != len(set(locked)) or set(locked) != set(final):
        raise ValueError("locked runtime and locked scene artifact disagree")
    priors = load_json(config.category_priors)
    if not isinstance(priors, Mapping) or "global" not in priors:
        raise ValueError("category priors must contain the frozen global statistics")
    size_bins = load_json(config.size_bins)
    if not isinstance(size_bins, Mapping):
        raise TypeError("size-bin artifact must be a JSON object")
    return {
        "scene_ids": list(DEV8),
        "tune24_scene_ids": list(scenes),
        "final48_scene_ids": list(final),
        "tune_physical_scene_count": len(
            {scene.rsplit("_", 1)[0] for scene in scenes}
        ),
        "final_physical_scene_count": len(
            {scene.rsplit("_", 1)[0] for scene in final}
        ),
        "gt_boundary": "offline_evaluation_only",
        "seed": int(config.seed),
    }


def _default_build_banks(**kwargs: Any) -> Mapping[str, Any]:
    from .category_cluster_runner import run_category_cluster_bank

    config: ClusterExperimentConfig = kwargs["config"]
    return run_category_cluster_bank(
        runtime_manifest=kwargs.get("runtime_manifest", config.runtime_manifest),
        output_root=kwargs.get("run_root", config.run_root),
        repo_root=config.repo_root,
        category_priors=config.category_priors,
        scene_ids=kwargs["scene_ids"],
        conditions=kwargs["conditions"],
        reference_bank_root=(
            config.reference_bank_root
            if bool(kwargs.get("require_reference_identity", True))
            else None
        ),
        verify_determinism=bool(kwargs.get("verify_determinism", False)),
        determinism_reference=kwargs.get("determinism_reference"),
        seed=config.seed,
        python_bin=config.python_bin,
    )


def _default_audit_distance(**kwargs: Any) -> Mapping[str, Any]:
    from .category_cluster_runner import audit_category_cluster_distance

    config: ClusterExperimentConfig = kwargs["config"]
    return audit_category_cluster_distance(
        run_root=kwargs.get("run_root", config.dev2_run_root),
        scene_ids=DEV2,
        reference_bank_root=config.reference_bank_root,
        reference_trace_root=config.reference_trace_root,
        output_path=kwargs["output_path"],
    )


def _default_evaluate_banks(**kwargs: Any) -> Mapping[str, Any]:
    from .category_cluster_scene_evaluation import evaluate_category_cluster_run
    from .taxonomy import load_taxonomy

    config: ClusterExperimentConfig = kwargs["config"]
    phase = str(kwargs["phase"])
    default_run_root = config.dev2_run_root if phase == "dev2" else config.run_root
    return evaluate_category_cluster_run(
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
        run_root=kwargs.get("run_root", default_run_root),
        scene_ids=kwargs["scene_ids"],
        taxonomy=load_taxonomy(config.taxonomy),
        phase=phase,
        selected_condition=kwargs["selected_condition"],
        primary_analysis=kwargs["primary_analysis"],
        frozen_selection_artifact=kwargs["frozen_selection_artifact"],
        metrics_output=kwargs["metrics_output"],
        analysis_output=kwargs["analysis_output"],
        size_bins=config.size_bins,
    )


def _default_evaluate_prior_oracle(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_experiment import _default_evaluate_prior_oracle as run

    return run(**kwargs)


def _default_evaluate_candidate_prior(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_experiment import _default_evaluate_candidate_prior as run

    return run(**kwargs)


def _default_select_candidate_threshold(**kwargs: Any) -> Mapping[str, Any]:
    from .category_candidate_experiment import (
        _default_select_candidate_threshold as run,
    )

    return run(**kwargs)


def _default_replay_final_stage(**kwargs: Any) -> Mapping[str, Any]:
    # The reused helper stores content hashes only inside replay_identity.json
    # for recovery.  It does not create a standalone SHA/hash artifact.
    from .category_candidate_experiment import _default_replay_final_stage as run

    return run(**kwargs)


def default_cluster_experiment_hooks() -> ClusterExperimentHooks:
    return ClusterExperimentHooks(
        check_resources=check_cluster_experiment_resources,
        validate_inputs=_default_validate_inputs,
        build_banks=_default_build_banks,
        audit_distance=_default_audit_distance,
        evaluate_banks=_default_evaluate_banks,
        evaluate_prior_oracle=_default_evaluate_prior_oracle,
        evaluate_candidate_prior=_default_evaluate_candidate_prior,
        select_candidate_threshold=_default_select_candidate_threshold,
        replay_final_stage=_default_replay_final_stage,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the preregistered section-31 clustering repair and frozen "
            "section-30 prior-validation stages"
        )
    )
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--locked-runtime-manifest", required=True)
    parser.add_argument("--locked-gt-dir", required=True)
    parser.add_argument("--locked-evaluation-scenes", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--category-priors", required=True)
    parser.add_argument("--prior-oracle-root", required=True)
    parser.add_argument("--reference-bank-root", required=True)
    parser.add_argument("--reference-trace-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--size-bins", required=True)
    parser.add_argument("--taxonomy")
    parser.add_argument("--python-bin")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_category_cluster_experiment(
        ClusterExperimentConfig(
            runtime_manifest=Path(args.runtime_manifest),
            gt_dir=Path(args.gt_dir),
            locked_runtime_manifest=Path(args.locked_runtime_manifest),
            locked_gt_dir=Path(args.locked_gt_dir),
            locked_evaluation_scenes=Path(args.locked_evaluation_scenes),
            repo_root=Path(args.repo_root),
            category_priors=Path(args.category_priors),
            prior_oracle_root=Path(args.prior_oracle_root),
            reference_bank_root=Path(args.reference_bank_root),
            reference_trace_root=Path(args.reference_trace_root),
            output_root=Path(args.output_root),
            size_bins=Path(args.size_bins),
            taxonomy=Path(args.taxonomy) if args.taxonomy else None,
            python_bin=Path(args.python_bin) if args.python_bin else None,
            seed=args.seed,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPLETE_STATUS",
    "DEV2",
    "DEV8",
    "EXPECTED_CGROUP_MAX_BYTES",
    "G1_MUTUAL_LOCAL_GRAPH",
    "HOLDOUT5",
    "PRIMARY_CONDITIONS",
    "READY_STATUS",
    "ClusterExperimentConfig",
    "ClusterExperimentHooks",
    "check_cluster_experiment_resources",
    "default_cluster_experiment_hooks",
    "run_category_cluster_experiment",
]
