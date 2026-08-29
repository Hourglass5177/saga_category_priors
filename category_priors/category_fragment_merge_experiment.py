from __future__ import annotations

"""Recoverable DEV2 -> conditional DEV8 controller for section 33.

The controller owns only stage order, resource guards and persisted state.
Ground-truth-free graph/merge work remains in
``category_fragment_merge_runner``; GT is opened only by the offline scene
evaluation adapter.  No holdout, tune24 or final48 stage is authorized here.
"""

import argparse
import json
import subprocess
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .category_fragment_merge_evaluation import DEV2_SCENE_IDS, DEV8_SCENE_IDS
from .category_fragment_merge_runner import (
    build_category_fragment_graphs,
    merge_category_fragment_graphs,
)
from .category_fragment_merge_scene_evaluation import (
    evaluate_category_fragment_merge_run,
)
from .io import load_json, write_json
from .taxonomy import load_taxonomy

STATE_SCHEMA = "saga-category-fragment-merge-experiment-state-v1"
DEV2 = DEV2_SCENE_IDS
DEV8 = DEV8_SCENE_IDS
SEED = 42
EXPECTED_CGROUP_MAX_BYTES = 90 * 1024**3
MIN_DISK_FREE_GIB = 80.0
TERMINAL_STATUSES = frozenset({"stopped", "complete"})
STAGES = (
    "dev2_build",
    "dev2_merge",
    "dev2_evaluate",
    "dev8_build",
    "dev8_merge",
    "dev8_evaluate",
)


@dataclass(frozen=True)
class FragmentMergeExperimentConfig:
    runtime_manifest: Path
    gt_dir: Path
    category_priors: Path
    output_root: Path
    size_bins: Path | None = None
    taxonomy: Path | None = None
    seed: int = SEED
    radius_m: float = 0.05
    min_region_size: int = 100

    def __post_init__(self) -> None:
        for name in (
            "runtime_manifest",
            "gt_dir",
            "category_priors",
            "output_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in ("size_bins", "taxonomy"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if isinstance(self.seed, bool) or int(self.seed) != SEED:
            raise ValueError(f"fragment merge experiment requires seed {SEED}")
        radius = float(self.radius_m)
        if not 0.0 < radius < float("inf"):
            raise ValueError("radius_m must be finite and positive")
        if (
            isinstance(self.min_region_size, bool)
            or int(self.min_region_size) != self.min_region_size
            or int(self.min_region_size) <= 0
        ):
            raise ValueError("min_region_size must be a positive integer")

    @property
    def state_path(self) -> Path:
        return self.output_root / "category_fragment_merge_experiment_state.json"

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs"

    @property
    def artifacts_root(self) -> Path:
        return self.output_root / "artifacts"

    def metrics_path(self, phase: str) -> Path:
        if phase not in {"dev2", "dev8"}:
            raise ValueError("phase must be dev2 or dev8")
        return self.artifacts_root / f"category_fragment_merge_{phase}.parquet"

    def analysis_path(self, phase: str) -> Path:
        if phase not in {"dev2", "dev8"}:
            raise ValueError("phase must be dev2 or dev8")
        # The scene evaluator explicitly uses this DEV2 filename as the DEV8
        # authorization artifact, so keep both phases beside each other.
        return self.artifacts_root / f"category_fragment_merge_{phase}_analysis.json"


Hook = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class FragmentMergeExperimentHooks:
    check_resources: Callable[[Path], Mapping[str, Any]]
    build_graphs: Hook
    merge_graphs: Hook
    evaluate: Hook


def check_fragment_merge_resources(
    root: Path,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Enforce only the registered ``df`` and cgroup-v2 resource contract."""

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
            f"fragment merge requires at least {MIN_DISK_FREE_GIB:.0f} GiB "
            f"available; found {available_gib:.1f} GiB"
        )

    cgroup = Path(cgroup_root)
    current_path = cgroup / "memory.current"
    maximum_path = cgroup / "memory.max"
    events_path = cgroup / "memory.events"
    for path in (current_path, maximum_path, events_path):
        if not path.is_file():
            raise RuntimeError(f"required cgroup v2 resource file is missing: {path}")
    maximum_text = maximum_path.read_text(encoding="utf-8").strip()
    if maximum_text == "max":
        raise RuntimeError("expected memory.max=90 GiB; found max")
    try:
        current = int(current_path.read_text(encoding="utf-8").strip())
        maximum = int(maximum_text)
    except ValueError as exc:
        raise RuntimeError("cgroup memory values must be integers") from exc
    if maximum != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(
            f"expected cgroup memory.max=90 GiB; found {maximum_text} bytes"
        )
    if current >= maximum:
        raise RuntimeError("cgroup memory.current has reached memory.max")
    events = events_path.read_text(encoding="utf-8").strip()
    return {
        "disk_source": "df-Pk",
        "disk_available_kib": available_kib,
        "disk_available_gib": available_gib,
        "memory_current_bytes": current,
        "memory_max_bytes": maximum,
        "memory_events": events,
        "host_free_used": False,
    }


def _resolved(path: Path | None) -> str | None:
    return None if path is None else str(path.resolve())


def _identity(config: FragmentMergeExperimentConfig) -> dict[str, Any]:
    return {
        "runtime_manifest": _resolved(config.runtime_manifest),
        "gt_dir": _resolved(config.gt_dir),
        "category_priors": _resolved(config.category_priors),
        "output_root": _resolved(config.output_root),
        "run_root": _resolved(config.run_root),
        "size_bins": _resolved(config.size_bins),
        "taxonomy": _resolved(config.taxonomy),
        "seed": int(config.seed),
        "radius_m": float(config.radius_m),
        "min_region_size": int(config.min_region_size),
        "dev2_scene_ids": list(DEV2),
        "dev8_scene_ids": list(DEV8),
        "authorized_stages": list(STAGES),
        "holdout_authorized": False,
        "final_authorized": False,
    }


def _initial_state(config: FragmentMergeExperimentConfig) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": "running",
        "checkpoint": "initialized",
        "current_stage": None,
        "next_stage": "dev2_build",
        "identity": _identity(config),
        "history": [],
        "dev2_passed": None,
        "dev8_passed": None,
        "category_prior_replayed": False,
        "category_prior_evaluable": False,
        "category_prior_tested": False,
    }


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


def _write_state(
    config: FragmentMergeExperimentConfig, state: Mapping[str, Any]
) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.run_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    # ``write_json`` writes a sibling temporary file and replaces atomically.
    write_json(config.state_path, _json_safe(state))


def _load_state(config: FragmentMergeExperimentConfig) -> dict[str, Any]:
    if not config.state_path.is_file():
        state = _initial_state(config)
        _write_state(config, state)
        return state
    payload = load_json(config.state_path)
    if not isinstance(payload, Mapping) or payload.get("schema") != STATE_SCHEMA:
        raise ValueError(f"invalid fragment merge state: {config.state_path}")
    state = dict(payload)
    if state.get("identity") != _identity(config):
        raise ValueError("fragment merge experiment identity differs from state")
    if not isinstance(state.get("history"), list):
        raise TypeError("fragment merge experiment history must be a list")
    if state.get("next_stage") not in {*STAGES, None}:
        raise ValueError("fragment merge state contains an unknown next stage")
    return state


def _validate_runner_result(
    result: Mapping[str, Any], *, stage: str, scene_ids: Sequence[str]
) -> None:
    if result.get("status") != "complete":
        raise ValueError(f"{stage} runner did not complete")
    observed = tuple(map(str, result.get("scene_ids", ())))
    if observed != tuple(scene_ids):
        raise ValueError(f"{stage} runner used a different scene set or order")
    expected_stage = {
        "dev2_build": "build-fragment-graph",
        "dev8_build": "build-fragment-graph",
        "dev2_merge": "merge-fragment-graph",
        "dev8_merge": "merge-fragment-graph",
    }[stage]
    if result.get("stage") != expected_stage:
        raise ValueError(f"{stage} runner returned the wrong stage")


def _validate_evaluation_result(
    result: Mapping[str, Any], *, phase: str, scene_ids: Sequence[str]
) -> None:
    if result.get("phase") != phase:
        raise ValueError(f"{phase} evaluator returned a different phase")
    if tuple(map(str, result.get("scene_ids", ()))) != tuple(scene_ids):
        raise ValueError(f"{phase} evaluator used a different scene set or order")
    if not isinstance(result.get("passed"), bool):
        raise TypeError(f"{phase} evaluator must report boolean passed")


def _default_hooks(config: FragmentMergeExperimentConfig) -> FragmentMergeExperimentHooks:
    taxonomy = load_taxonomy(config.taxonomy)

    def resources(root: Path) -> Mapping[str, Any]:
        return check_fragment_merge_resources(root)

    def build(**kwargs: Any) -> Mapping[str, Any]:
        return build_category_fragment_graphs(
            runtime_manifest=config.runtime_manifest,
            category_priors=config.category_priors,
            output_root=config.run_root,
            scene_ids=kwargs["scene_ids"],
            seed=config.seed,
        )

    def merge(**kwargs: Any) -> Mapping[str, Any]:
        return merge_category_fragment_graphs(
            runtime_manifest=config.runtime_manifest,
            category_priors=config.category_priors,
            output_root=config.run_root,
            scene_ids=kwargs["scene_ids"],
            seed=config.seed,
        )

    def evaluate(**kwargs: Any) -> Mapping[str, Any]:
        phase = str(kwargs["phase"])
        return evaluate_category_fragment_merge_run(
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            run_root=config.run_root,
            scene_ids=kwargs["scene_ids"],
            taxonomy=taxonomy,
            phase=phase,
            metrics_output=config.metrics_path(phase),
            analysis_output=config.analysis_path(phase),
            size_bins=config.size_bins,
            radius_m=config.radius_m,
            min_region_size=config.min_region_size,
        )

    return FragmentMergeExperimentHooks(
        check_resources=resources,
        build_graphs=build,
        merge_graphs=merge,
        evaluate=evaluate,
    )


def _stage_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "schema",
        "status",
        "stage",
        "phase",
        "scene_ids",
        "passed",
        "conclusion",
    )
    return {key: _json_safe(result[key]) for key in keep if key in result}


def _mark_error(
    config: FragmentMergeExperimentConfig,
    state: dict[str, Any],
    stage: str,
    exc: BaseException,
) -> None:
    state.update(
        {
            "status": "error",
            "checkpoint": state.get("checkpoint", "initialized"),
            "current_stage": None,
            "next_stage": stage,
            "last_error": {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        }
    )
    _write_state(config, state)


def run_category_fragment_merge_experiment(
    config: FragmentMergeExperimentConfig,
    hooks: FragmentMergeExperimentHooks | None = None,
) -> dict[str, Any]:
    """Run or resume the preregistered DEV2 and conditional DEV8 stages."""

    active_hooks = hooks or _default_hooks(config)
    state = _load_state(config)
    if state.get("status") in TERMINAL_STATUSES:
        return state

    while state.get("next_stage") is not None:
        stage = str(state["next_stage"])
        if stage not in STAGES:
            raise ValueError(f"unregistered fragment merge stage: {stage}")
        phase = "dev2" if stage.startswith("dev2") else "dev8"
        scene_ids = DEV2 if phase == "dev2" else DEV8
        state.update(
            {
                "status": "running",
                "current_stage": stage,
                "next_stage": stage,
            }
        )
        state.pop("last_error", None)
        _write_state(config, state)
        try:
            resources = active_hooks.check_resources(config.output_root)
            if stage.endswith("_build"):
                result = active_hooks.build_graphs(
                    phase=phase,
                    scene_ids=scene_ids,
                )
                _validate_runner_result(result, stage=stage, scene_ids=scene_ids)
                next_stage = f"{phase}_merge"
            elif stage.endswith("_merge"):
                result = active_hooks.merge_graphs(
                    phase=phase,
                    scene_ids=scene_ids,
                )
                _validate_runner_result(result, stage=stage, scene_ids=scene_ids)
                next_stage = f"{phase}_evaluate"
            else:
                result = active_hooks.evaluate(
                    phase=phase,
                    scene_ids=scene_ids,
                    metrics_output=config.metrics_path(phase),
                    analysis_output=config.analysis_path(phase),
                )
                _validate_evaluation_result(
                    result,
                    phase=phase,
                    scene_ids=scene_ids,
                )
                state[f"{phase}_passed"] = bool(result["passed"])
                prior_evaluable = bool(
                    result.get("graph_passed", phase == "dev8")
                )
                state["category_prior_replayed"] = True
                state["category_prior_evaluable"] = bool(
                    state.get("category_prior_evaluable", False)
                    or prior_evaluable
                )
                state["category_prior_tested"] = state[
                    "category_prior_evaluable"
                ]
                if phase == "dev2" and not result["passed"]:
                    state.update(
                        {
                            "status": "stopped",
                            "checkpoint": "dev2_gate_failed",
                            "current_stage": None,
                            "next_stage": None,
                            "stop_reason": str(
                                result.get(
                                    "conclusion",
                                    "DEV2 category fragment merge gate failed",
                                )
                            ),
                        }
                    )
                    state["history"].append(
                        {
                            "stage": stage,
                            "resources": _json_safe(resources),
                            **_stage_summary(result),
                        }
                    )
                    _write_state(config, state)
                    return state
                if phase == "dev2":
                    next_stage = "dev8_build"
                else:
                    state.update(
                        {
                            "status": "complete",
                            "checkpoint": "dev8_evaluated",
                            "current_stage": None,
                            "next_stage": None,
                            "conclusion": result.get("conclusion"),
                        }
                    )
                    state["history"].append(
                        {
                            "stage": stage,
                            "resources": _json_safe(resources),
                            **_stage_summary(result),
                        }
                    )
                    _write_state(config, state)
                    return state
        except BaseException as exc:
            _mark_error(config, state, stage, exc)
            raise

        state["history"].append(
            {
                "stage": stage,
                "resources": _json_safe(resources),
                **_stage_summary(result),
            }
        )
        state.update(
            {
                "status": "running",
                "checkpoint": stage,
                "current_stage": None,
                "next_stage": next_stage,
            }
        )
        _write_state(config, state)
    return state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the recoverable SAGA category fragment merge experiment"
    )
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--category-priors", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--size-bins")
    parser.add_argument("--taxonomy")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--radius-m", type=float, default=0.05)
    parser.add_argument("--min-region-size", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FragmentMergeExperimentConfig(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        category_priors=Path(args.category_priors),
        output_root=Path(args.output_root),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        taxonomy=Path(args.taxonomy) if args.taxonomy else None,
        seed=args.seed,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    result = run_category_fragment_merge_experiment(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEV2",
    "DEV8",
    "EXPECTED_CGROUP_MAX_BYTES",
    "MIN_DISK_FREE_GIB",
    "SEED",
    "STAGES",
    "STATE_SCHEMA",
    "FragmentMergeExperimentConfig",
    "FragmentMergeExperimentHooks",
    "build_parser",
    "check_fragment_merge_resources",
    "main",
    "run_category_fragment_merge_experiment",
]
