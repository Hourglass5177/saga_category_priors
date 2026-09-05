from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from arguments import ModelParams, PipelineParams

from .pipeline import run_scene
from .runtime_io import json_atomic
from .seeds import prepare_reservoir


def _commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def _prepare(args: argparse.Namespace) -> None:
    seeds = prepare_reservoir(
        candidate_bank_path=args.candidate_bank,
        stage_trace_path=args.stage_trace,
        b0_output_path=args.b0_output,
        output_dir=args.output_dir,
        provenance={"commit": _commit(), "scene_id": args.scene_id},
    )
    print(json.dumps({"candidate_count": len(seeds), "output_dir": args.output_dir}, indent=2))


def _refine(args: argparse.Namespace) -> None:
    result = run_scene(args)
    print(json.dumps(result["outputs"], ensure_ascii=False, indent=2))


def _replay(args: argparse.Namespace) -> None:
    # Replay is deliberately a validation/index operation for now: profile
    # outputs are already produced from one shared 2D evidence cache by refine.
    root = Path(args.output_dir)
    payload = json.loads((root / "iterative_refinement.json").read_text(encoding="utf-8"))
    required = {"stable", "balanced", "coverage"}
    if set(payload.get("outputs", {})) != required:
        raise ValueError("refine output does not contain all registered profiles")
    for profile in required:
        if not (root / profile / "output.json").is_file():
            raise FileNotFoundError(root / profile / "output.json")
    print(json.dumps(payload["outputs"], ensure_ascii=False, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    # Keep GT in a separate process/module.  This wrapper invokes the existing
    # authoritative evaluator and never imports GT into the runtime process.
    from ..evaluation_strata import load_evaluation_strata
    from ..recheck_evaluation import evaluate_recheck_manifest
    from ..taxonomy import load_taxonomy

    evaluate_recheck_manifest(
        args.manifest, output_path=args.output,
        taxonomy=load_taxonomy(args.taxonomy),
        strata=load_evaluation_strata(args.strata) if args.strata else load_evaluation_strata(),
        radius_m=args.radius_m, min_region_size=args.min_region_size,
    )


def _validate_alpha(args: argparse.Namespace) -> None:
    from .alpha_validation import validate_alpha_backend
    result = validate_alpha_backend(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-iterative-refinement")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="build the branch-only candidate reservoir")
    prepare.add_argument("--candidate-bank", required=True)
    prepare.add_argument("--stage-trace", required=True)
    prepare.add_argument("--b0-output", required=True)
    prepare.add_argument("--scene-id", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.set_defaults(func=_prepare)

    refine = commands.add_parser("refine", help="run two-round 2D--3D refinement")
    ModelParams(refine)
    PipelineParams(refine)
    refine.add_argument("--candidate-bank", required=True)
    refine.add_argument("--reservoir", required=True)
    refine.add_argument("--priors", required=True)
    refine.add_argument("--output-dir", required=True)
    refine.add_argument("--condition", choices=("global", "class"), required=True)
    refine.add_argument("--sam-checkpoint-path", required=True)
    refine.add_argument("--groundingdino-checkpoint-path", required=True)
    refine.add_argument("--groundingdino-config-path", required=True)
    refine.add_argument("--classes", nargs="+")
    refine.add_argument("--label-threshold", type=float, default=.3)
    refine.add_argument("--scale-threshold", type=float, default=.8)
    refine.add_argument("--opcity-threshold", type=float, default=.005)
    refine.add_argument("--alpha-backend", choices=("fused", "gradient-reference"), default="fused")
    refine.add_argument("--alpha-cache-dir")
    refine.add_argument("--alpha-cache-mode", choices=("readwrite", "readonly", "off"), default="readwrite")
    refine.add_argument("--review-cache-source", help="read-only prior DINO/SAM cache root")
    refine.set_defaults(func=_refine)

    replay = commands.add_parser("replay", help="validate/reuse cached profile outputs")
    replay.add_argument("--output-dir", required=True)
    replay.set_defaults(func=_replay)

    evaluate = commands.add_parser("evaluate", help="evaluate outputs in a separate GT process")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--taxonomy")
    evaluate.add_argument("--strata")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--radius-m", type=float, default=.05)
    evaluate.add_argument("--min-region-size", type=int, default=100)
    evaluate.set_defaults(func=_evaluate)

    validate = commands.add_parser("validate-alpha", help="validate fused alpha mass against the frozen gradient reference")
    ModelParams(validate)
    PipelineParams(validate)
    validate.add_argument("--review-cache-source", required=True)
    validate.add_argument("--alpha-cache-dir", required=True)
    validate.add_argument("--output-dir", required=True)
    validate.set_defaults(func=_validate_alpha)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
