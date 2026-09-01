from __future__ import annotations

"""Narrow command-line boundary for the clean alpha-mask baseline.

GPU/rendering orchestration is injected as a worker callable.  Keeping that
dependency behind this boundary lets the evidence, consensus, and evaluation
algorithms remain importable in CPU-only test environments.
"""

import argparse
import importlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..io import load_json
from ..taxonomy import load_taxonomy
from .evaluation import (
    FORMAL_CONDITIONS,
    evaluate_clean_baseline_manifest,
    evaluation_is_complete,
    prediction_is_complete,
)

DEFAULT_EVIDENCE_WORKER = (
    "category_priors.clean_baseline.evidence:build_alpha_mask_evidence"
)
DEFAULT_CONSENSUS_RUNNER = (
    "category_priors.clean_baseline.pipeline:run_consensus_condition"
)
DEFAULT_PRIOR_RUNNER = (
    "category_priors.clean_baseline.pipeline:replay_size_prior_condition"
)


def _resolve_callable(spec: str) -> Callable[..., Any]:
    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("worker/runner must use the form 'module:function'")
    try:
        module = importlib.import_module(module_name)
        value = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"clean-baseline worker is unavailable: {spec}; "
            "deploy the scene worker before running this command"
        ) from exc
    if not callable(value):
        raise TypeError(f"worker/runner is not callable: {spec}")
    return value


def _call_with_registered_kwargs(
    function: Callable[..., Any], kwargs: Mapping[str, Any]
) -> Any:
    signature = inspect.signature(function)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**dict(kwargs))
    unknown = sorted(set(kwargs).difference(signature.parameters))
    if unknown:
        raise TypeError(
            f"{function.__module__}.{function.__name__} does not accept {unknown}"
        )
    return function(**dict(kwargs))


def _evidence_complete(
    directory: str | Path,
    scene_id: str,
    *,
    expected_source: Mapping[str, Any],
) -> bool:
    try:
        from .evidence import evidence_bank_is_complete

        return bool(
            evidence_bank_is_complete(
                directory,
                expected_scene_id=scene_id,
                expected_source=expected_source,
            )
        )
    except (ImportError, FileNotFoundError, OSError, TypeError, ValueError):
        return False


def _build_alpha_mask_evidence(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    request = load_json(args.request_json)
    if not isinstance(request, Mapping):
        raise TypeError("evidence request must be a JSON object")
    from .evidence import evidence_request_source

    expected_source = evidence_request_source(
        scene_id=args.scene_id,
        request=request,
    )
    if _evidence_complete(
        output_dir,
        args.scene_id,
        expected_source=expected_source,
    ):
        return {
            "command": "build-alpha-mask-evidence",
            "scene_id": args.scene_id,
            "status": "skipped-complete",
            "output_dir": str(output_dir),
        }
    worker = _resolve_callable(args.worker)
    _call_with_registered_kwargs(
        worker,
        {
            "scene_id": args.scene_id,
            "request": dict(request),
            "output_dir": output_dir,
        },
    )
    if not _evidence_complete(
        output_dir,
        args.scene_id,
        expected_source=expected_source,
    ):
        raise RuntimeError("evidence worker returned without a complete evidence bank")
    return {
        "command": "build-alpha-mask-evidence",
        "scene_id": args.scene_id,
        "status": "complete",
        "output_dir": str(output_dir),
    }


def _run_condition(
    args: argparse.Namespace,
    *,
    command: str,
    registered_conditions: frozenset[str],
) -> dict[str, Any]:
    if args.condition not in registered_conditions:
        raise ValueError(f"{command}: unregistered condition {args.condition}")
    if args.condition == "D-oracle-class":
        raise ValueError("D-oracle-class is evaluation-only")
    output_dir = Path(args.output_dir).resolve()
    output_json = output_dir / "output.json"
    runner = _resolve_callable(args.runner)
    runner_result = _call_with_registered_kwargs(
        runner,
        {
            "scene_id": args.scene_id,
            "bank_dir": Path(args.bank_dir).resolve(),
            "condition": args.condition,
            "output_dir": output_dir,
            "priors_path": (
                None if args.priors is None else Path(args.priors).resolve()
            ),
        },
    )
    if not prediction_is_complete(
        output_json,
        expected_scene_id=args.scene_id,
        expected_condition=args.condition,
        expected_gaussian_count=args.gaussian_count,
    ):
        raise RuntimeError("condition runner returned without a complete prediction")
    return {
        "command": command,
        "scene_id": args.scene_id,
        "condition": args.condition,
        "status": (
            str(runner_result.get("runner_status", "complete"))
            if isinstance(runner_result, Mapping)
            else "complete"
        ),
        "output_json": str(output_json),
    }


def _run_mask_consensus(args: argparse.Namespace) -> dict[str, Any]:
    return _run_condition(
        args,
        command="run-mask-consensus",
        registered_conditions=frozenset({"C0-no-prior", "U-global"}),
    )


def _replay_size_prior(args: argparse.Namespace) -> dict[str, Any]:
    if args.priors is None:
        raise ValueError("replay-size-prior requires --priors")
    return _run_condition(
        args,
        command="replay-size-prior",
        registered_conditions=frozenset({"U-global", "D-predicted"}),
    )


def _evaluate_clean_baseline(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    if manifest.get("kind") != "clean_baseline_evaluation_manifest":
        raise ValueError("expected a clean_baseline_evaluation_manifest")
    scene_ids = [str(item["scene_id"]) for item in manifest.get("scenes", [])]
    conditions = [str(value) for value in manifest.get("conditions", [])]
    taxonomy = load_taxonomy(args.taxonomy)
    result = evaluate_clean_baseline_manifest(
        args.manifest,
        class_names=taxonomy.canonical_classes,
        output_path=args.output,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    return {
        "command": "evaluate-clean-baseline",
        "status": str(result.get("runner_status", "complete")),
        "output": str(Path(args.output).resolve()),
        "scene_count": len(result["scene_ids"]),
        "conditions": result["conditions"],
    }


def _audit_clean_baseline(args: argparse.Namespace) -> dict[str, Any]:
    from .two_step_experiment import audit_clean_baseline

    return audit_clean_baseline(
        manifest_path=Path(args.manifest).resolve(),
        output_root=Path(args.output_root).resolve(),
    )


def _prepare_flat_mask_control(args: argparse.Namespace) -> dict[str, Any]:
    from .two_step_experiment import prepare_flat_mask_control

    return prepare_flat_mask_control(
        manifest_path=Path(args.manifest).resolve(),
        output_root=Path(args.output_root).resolve(),
        producer_commit=str(args.producer_commit),
    )


def _run_clean_baseline_two_step(args: argparse.Namespace) -> dict[str, Any]:
    from .two_step_experiment import run_clean_baseline_two_step

    return run_clean_baseline_two_step(
        manifest_path=Path(args.manifest).resolve(),
        output_root=Path(args.output_root).resolve(),
        run_root=Path(args.run_root).resolve(),
        producer_commit=str(args.producer_commit),
    )


def _add_condition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--gaussian-count", required=True, type=int)
    parser.add_argument("--priors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAGA clean alpha-mask consensus baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evidence = subparsers.add_parser(
        "build-alpha-mask-evidence",
        help="build or validate a sparse alpha-mask evidence bank",
    )
    evidence.add_argument("--scene-id", required=True)
    evidence.add_argument("--request-json", required=True)
    evidence.add_argument("--output-dir", required=True)
    evidence.add_argument("--worker", default=DEFAULT_EVIDENCE_WORKER)
    evidence.set_defaults(handler=_build_alpha_mask_evidence)

    consensus = subparsers.add_parser(
        "run-mask-consensus",
        help="form class-agnostic objects from one evidence bank",
    )
    _add_condition_arguments(consensus)
    consensus.add_argument("--runner", default=DEFAULT_CONSENSUS_RUNNER)
    consensus.set_defaults(handler=_run_mask_consensus)

    replay = subparsers.add_parser(
        "replay-size-prior",
        help="replay global or predicted-class size constraints",
    )
    _add_condition_arguments(replay)
    replay.add_argument("--runner", default=DEFAULT_PRIOR_RUNNER)
    replay.set_defaults(handler=_replay_size_prior)

    evaluate = subparsers.add_parser(
        "evaluate-clean-baseline",
        help="run official AP for paired formal C0/U/D predictions",
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--taxonomy")
    evaluate.add_argument("--radius-m", type=float, default=0.05)
    evaluate.add_argument("--min-region-size", type=int, default=100)
    evaluate.set_defaults(handler=_evaluate_clean_baseline)

    audit = subparsers.add_parser(
        "audit-clean-baseline",
        help="read-only corrected DEV8 metrics and production-stage funnel",
    )
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output-root", required=True)
    audit.set_defaults(handler=_audit_clean_baseline)

    prepare_flat = subparsers.add_parser(
        "prepare-flat-mask-control",
        help="generate one SAM stack and derive paired hierarchy/flat masks",
    )
    prepare_flat.add_argument("--manifest", required=True)
    prepare_flat.add_argument("--output-root", required=True)
    prepare_flat.add_argument("--producer-commit", required=True)
    prepare_flat.set_defaults(handler=_prepare_flat_mask_control)

    two_step = subparsers.add_parser(
        "run-clean-baseline-two-step",
        help="run the metric correction and paired H'/P DEV2 control",
    )
    two_step.add_argument("--manifest", required=True)
    two_step.add_argument("--output-root", required=True)
    two_step.add_argument("--run-root", required=True)
    two_step.add_argument("--producer-commit", required=True)
    two_step.set_defaults(handler=_run_clean_baseline_two_step)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
