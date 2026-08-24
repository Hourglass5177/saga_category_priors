from __future__ import annotations

"""Small public CLI for the active SAGA baseline and V9 experiment.

Retired B2/class-first/prior-v2/V3-V6 experiment entry points intentionally do
not live here; their exact implementations remain available through Git.
"""

import argparse
import json
from pathlib import Path

from .alignment import audit_saga_alignment
from .baseline_closure_analysis import evaluate_teacher_handoff
from .evaluator import evaluate_manifest
from .gaussian_object_audit import audit_gaussian_object_runs
from .io import read_rows
from .priors import fit_priors, write_priors
from .taxonomy import load_taxonomy
from .v8_bank import (
    CLASSIFIERS as V8_CLASSIFIERS,
    build_v8_object_bank,
    replay_v8_priors,
)
from .v8_evaluation import evaluate_v8_replays
from .v8_replay import CONDITIONS as V8_CONDITIONS
from .v8_runner import run_v8_lifting_banks, run_v8_lifting_factorial
from .v9_feature_training import execute_v9_feature_training
from .v9_replay import CONDITION_FACTORS as V9_CONDITIONS
from .v9_runner import (
    ASSOCIATION_MODES as V9_ASSOCIATION_MODES,
    CLASSIFIERS as V9_CLASSIFIERS,
    replay_v9_priors,
    run_v9_banks,
)
from .v9_metrics import evaluate_v9_candidate_banks, evaluate_v9_predictions


def _fit(args: argparse.Namespace) -> None:
    payload = fit_priors(
        read_rows(args.stats), load_taxonomy(args.taxonomy), args.stats,
        seed=args.seed, bootstrap_samples=args.bootstrap_samples,
        min_physical_scenes=args.min_physical_scenes, shrink_tau=args.shrink_tau,
    )
    write_priors(args.output, payload)


def _evaluate(args: argparse.Namespace) -> None:
    evaluate_manifest(
        args.manifest, load_taxonomy(args.taxonomy), args.output,
        args.radius_m, args.min_region_size,
    )


def _audit_alignment(args: argparse.Namespace) -> None:
    audit_saga_alignment(
        args.preparation_manifest, args.gt_npz, args.output,
        gaussian_ply_path=args.gaussian_ply, radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        camera_padding_m=args.camera_padding_m, minimal=args.minimal,
    )


def _audit(args: argparse.Namespace) -> None:
    payload = audit_gaussian_object_runs(
        scene_manifest=args.scene_manifest, gt_dir=args.gt_dir,
        runs_root=args.runs_root, taxonomy=load_taxonomy(args.taxonomy),
        scene_ids=args.scene, conditions=args.condition, seed=args.seed,
        table_output=args.table_output, audit_output=args.audit_output,
        comparison_output=args.comparison_output, viewer_output=args.viewer_output,
        radius_m=args.radius_m, min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_v8_lifting_audit(args: argparse.Namespace) -> None:
    payload = run_v8_lifting_factorial(
        Path(args.runtime_manifest),
        args.scene,
        Path(args.output_root),
        Path(args.repo_root),
        sam_masks_root=Path(args.sam_masks_root),
        sam_checkpoint=Path(args.sam_checkpoint),
        label_features=Path(args.label_features) if args.label_features else None,
        contributor_audit=args.contributor_audit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_v8_bank(args: argparse.Namespace) -> None:
    lifting_root = Path(args.output_root) / "lifting"
    run_v8_lifting_banks(
        Path(args.runtime_manifest),
        args.scene,
        lifting_root,
        Path(args.repo_root),
        mask_source=args.mask_source,
        lifting_source=args.lifting_source,
        sam_masks_root=Path(args.sam_masks_root) if args.sam_masks_root else None,
        sam_checkpoint=Path(args.sam_checkpoint) if args.sam_checkpoint else None,
        label_features=Path(args.label_features) if args.label_features else None,
    )
    records = [
        build_v8_object_bank(
            lifting_root / scene_id,
            Path(args.output_root) / scene_id,
        )
        for scene_id in args.scene
    ]
    print(json.dumps({"schema": "saga-v8-bank-run-v1", "banks": records}, ensure_ascii=False, indent=2))


def _replay_v8(args: argparse.Namespace) -> None:
    payload = replay_v8_priors(
        bank_root=args.bank_root,
        output_root=args.output_root,
        scene_ids=args.scene,
        classifier=args.classifier,
        conditions=args.condition or V8_CONDITIONS,
        category_priors=args.category_priors,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_v8(args: argparse.Namespace) -> None:
    payload = evaluate_v8_replays(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        replay_root=Path(args.replay_root),
        scene_ids=args.scene,
        conditions=args.condition or V8_CONDITIONS,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
        size_bins=Path(args.size_bins) if args.size_bins else None,
        viewer_output=Path(args.viewer_output) if args.viewer_output else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _audit_teacher_baseline(args: argparse.Namespace) -> None:
    payload = evaluate_teacher_handoff(
        closure_root=Path(args.closure_root),
        gt_dir=Path(args.gt_dir),
        runtime_manifest=Path(args.runtime_manifest),
        output_dir=Path(args.output_dir),
        taxonomy=load_taxonomy(args.taxonomy),
        min_region_size=args.min_region_size,
        radius_m=args.radius_m,
        final_vote_scores_path=(
            Path(args.final_vote_scores) if args.final_vote_scores else None
        ),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _train_object_features_10k(args: argparse.Namespace) -> None:
    payload = execute_v9_feature_training(
        scene_manifest=Path(args.scene_manifest),
        output_root=Path(args.output_root),
        workspace=Path(args.workspace),
        git_commit=args.git_commit,
        scene_ids=args.scene,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_object_bank(args: argparse.Namespace) -> None:
    payload = run_v9_banks(
        lifting_root=args.lifting_root,
        output_root=args.output_root,
        scene_ids=args.scene,
        association_modes=args.association_mode,
        git_commit=args.git_commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_object_priors(args: argparse.Namespace) -> None:
    payload = replay_v9_priors(
        bank_root=Path(args.bank_root) / args.association_mode,
        output_root=args.output_root,
        scene_ids=args.scene,
        classifier=args.classifier,
        conditions=args.condition or tuple(V9_CONDITIONS),
        category_priors=args.category_priors,
        acceptance_threshold=args.acceptance_threshold,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_object_system(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    if args.evaluation_target == "bank":
        payload = evaluate_v9_candidate_banks(
            runtime_manifest=Path(args.runtime_manifest),
            gt_dir=Path(args.gt_dir),
            bank_root=Path(args.input_root),
            scene_ids=args.scene,
            association_mode=args.association_mode,
            classifier=args.classifier,
            taxonomy=taxonomy,
            rows_output=Path(args.metrics_output),
            analysis_output=Path(args.analysis_output),
            size_bins=Path(args.size_bins) if args.size_bins else None,
            radius_m=args.radius_m,
            min_region_size=args.min_region_size,
        )
    else:
        payload = evaluate_v9_predictions(
            runtime_manifest=Path(args.runtime_manifest),
            gt_dir=Path(args.gt_dir),
            prediction_root=Path(args.input_root),
            scene_ids=args.scene,
            conditions=args.condition or tuple(V9_CONDITIONS),
            taxonomy=taxonomy,
            metrics_output=Path(args.metrics_output),
            analysis_output=Path(args.analysis_output),
            radius_m=args.radius_m,
            min_region_size=args.min_region_size,
            size_bins=Path(args.size_bins) if args.size_bins else None,
            viewer_output=Path(args.viewer_output) if args.viewer_output else None,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAGA category-prior utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit frozen train-only category statistics")
    fit.add_argument("--stats", required=True)
    fit.add_argument("--taxonomy")
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=20260804)
    fit.add_argument("--bootstrap-samples", type=int, default=2000)
    fit.add_argument("--min-physical-scenes", type=int, default=5)
    fit.add_argument("--shrink-tau", type=float, default=20.0)
    fit.set_defaults(func=_fit)

    evaluate = commands.add_parser("evaluate", help="run the official ScanNet evaluator")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--taxonomy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--radius-m", type=float, default=0.05)
    evaluate.add_argument("--min-region-size", type=int, default=100)
    evaluate.set_defaults(func=_evaluate)

    alignment = commands.add_parser("audit-saga-alignment", help="audit metric Gaussian/GT alignment")
    alignment.add_argument("--preparation-manifest", required=True)
    alignment.add_argument("--gt-npz", required=True)
    alignment.add_argument("--gaussian-ply")
    alignment.add_argument("--output", required=True)
    alignment.add_argument("--radius-m", type=float, default=0.05)
    alignment.add_argument("--minimum-mapped-fraction", type=float, default=0.90)
    alignment.add_argument("--camera-padding-m", type=float, default=2.0)
    alignment.add_argument("--minimal", action="store_true")
    alignment.set_defaults(func=_audit_alignment)

    audit = commands.add_parser("audit-gaussian-objects", help="precision-first B0/B1 audit")
    audit.add_argument("--scene-manifest", required=True)
    audit.add_argument("--gt-dir", required=True)
    audit.add_argument("--runs-root", required=True)
    audit.add_argument("--taxonomy")
    audit.add_argument("--scene", action="append", required=True)
    audit.add_argument("--condition", action="append", required=True)
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--table-output", required=True)
    audit.add_argument("--audit-output", required=True)
    audit.add_argument("--comparison-output", required=True)
    audit.add_argument("--viewer-output", required=True)
    audit.add_argument("--radius-m", type=float, default=0.05)
    audit.add_argument("--min-region-size", type=int, default=100)
    audit.set_defaults(func=_audit)

    lifting_v8 = commands.add_parser(
        "run-v8-lifting-audit", help="run the frozen V8 mask-by-lifting factorial"
    )
    lifting_v8.add_argument("--runtime-manifest", required=True)
    lifting_v8.add_argument("--output-root", required=True)
    lifting_v8.add_argument("--repo-root", default=".")
    lifting_v8.add_argument("--sam-masks-root", required=True)
    lifting_v8.add_argument("--sam-checkpoint", required=True)
    lifting_v8.add_argument("--label-features")
    lifting_v8.add_argument("--scene", action="append", required=True)
    lifting_v8.add_argument("--contributor-audit", action="store_true")
    lifting_v8.set_defaults(func=_run_v8_lifting_audit)

    bank_v8 = commands.add_parser(
        "run-v8-bank", help="build deterministic V8 tracks from a selected lifting arm"
    )
    bank_v8.add_argument("--runtime-manifest", required=True)
    bank_v8.add_argument("--output-root", required=True)
    bank_v8.add_argument("--repo-root", default=".")
    bank_v8.add_argument("--mask-source", choices=("G", "S"), required=True)
    bank_v8.add_argument("--lifting-source", choices=("M1", "AM"), required=True)
    bank_v8.add_argument("--sam-masks-root")
    bank_v8.add_argument("--sam-checkpoint")
    bank_v8.add_argument("--label-features", required=True)
    bank_v8.add_argument("--scene", action="append", required=True)
    bank_v8.set_defaults(func=_run_v8_bank)

    replay_v8 = commands.add_parser(
        "replay-v8-priors", help="replay frozen U/D scores over immutable V8 banks"
    )
    replay_v8.add_argument("--bank-root", required=True)
    replay_v8.add_argument("--output-root", required=True)
    replay_v8.add_argument("--category-priors", required=True)
    replay_v8.add_argument("--classifier", choices=V8_CLASSIFIERS, required=True)
    replay_v8.add_argument("--scene", action="append", required=True)
    replay_v8.add_argument("--condition", action="append", choices=V8_CONDITIONS)
    replay_v8.set_defaults(func=_replay_v8)

    evaluate_v8 = commands.add_parser(
        "evaluate-v8", help="official AP and Gaussian diagnostics for V8 replay outputs"
    )
    evaluate_v8.add_argument("--runtime-manifest", required=True)
    evaluate_v8.add_argument("--gt-dir", required=True)
    evaluate_v8.add_argument("--replay-root", required=True)
    evaluate_v8.add_argument("--scene", action="append", required=True)
    evaluate_v8.add_argument("--condition", action="append", choices=V8_CONDITIONS)
    evaluate_v8.add_argument("--taxonomy")
    evaluate_v8.add_argument("--metrics-output", required=True)
    evaluate_v8.add_argument("--analysis-output", required=True)
    evaluate_v8.add_argument("--size-bins")
    evaluate_v8.add_argument("--viewer-output")
    evaluate_v8.add_argument("--radius-m", type=float, default=0.05)
    evaluate_v8.add_argument("--min-region-size", type=int, default=100)
    evaluate_v8.set_defaults(func=_evaluate_v8)

    teacher = commands.add_parser(
        "audit-teacher-baseline",
        help="forensic evaluation of the reconstructed teacher handoff",
    )
    teacher.add_argument("--closure-root", required=True)
    teacher.add_argument("--gt-dir", required=True)
    teacher.add_argument("--runtime-manifest", required=True)
    teacher.add_argument("--output-dir", required=True)
    teacher.add_argument("--taxonomy")
    teacher.add_argument("--final-vote-scores")
    teacher.add_argument("--radius-m", type=float, default=0.05)
    teacher.add_argument("--min-region-size", type=int, default=100)
    teacher.set_defaults(func=_audit_teacher_baseline)

    train_v9 = commands.add_parser(
        "train-object-features-10k",
        help="train isolated 10k affinity/semantic features from separate sources",
    )
    train_v9.add_argument("--scene-manifest", required=True)
    train_v9.add_argument("--output-root", required=True)
    train_v9.add_argument("--workspace", default=".")
    train_v9.add_argument("--git-commit", required=True)
    train_v9.add_argument("--scene", action="append", required=True)
    train_v9.add_argument("--dry-run", action="store_true")
    train_v9.add_argument("--no-resume", action="store_true")
    train_v9.add_argument("--continue-on-error", action="store_true")
    train_v9.set_defaults(func=_train_object_features_10k)

    bank_v9 = commands.add_parser(
        "run-object-bank", help="build deterministic A0-A3 V9 object banks"
    )
    bank_v9.add_argument("--lifting-root", required=True)
    bank_v9.add_argument("--output-root", required=True)
    bank_v9.add_argument("--scene", action="append", required=True)
    bank_v9.add_argument(
        "--association-mode",
        action="append",
        choices=V9_ASSOCIATION_MODES,
        required=True,
    )
    bank_v9.add_argument("--git-commit", required=True)
    bank_v9.set_defaults(func=_run_object_bank)

    replay_v9 = commands.add_parser(
        "replay-object-priors", help="run frozen 2^3 category-prior replay"
    )
    replay_v9.add_argument("--bank-root", required=True)
    replay_v9.add_argument("--output-root", required=True)
    replay_v9.add_argument("--category-priors", required=True)
    replay_v9.add_argument("--association-mode", choices=V9_ASSOCIATION_MODES, required=True)
    replay_v9.add_argument("--classifier", choices=V9_CLASSIFIERS, required=True)
    replay_v9.add_argument("--scene", action="append", required=True)
    replay_v9.add_argument("--condition", action="append", choices=tuple(V9_CONDITIONS))
    replay_v9.add_argument("--acceptance-threshold", type=float, required=True)
    replay_v9.set_defaults(func=_replay_object_priors)

    evaluate_v9 = commands.add_parser(
        "evaluate-object-system",
        help="evaluate a frozen V9 bank or strict replay outputs",
    )
    evaluate_v9.add_argument("--evaluation-target", choices=("bank", "replay"), required=True)
    evaluate_v9.add_argument("--runtime-manifest", required=True)
    evaluate_v9.add_argument("--gt-dir", required=True)
    evaluate_v9.add_argument("--input-root", required=True)
    evaluate_v9.add_argument("--scene", action="append", required=True)
    evaluate_v9.add_argument("--association-mode", choices=V9_ASSOCIATION_MODES, default="A1")
    evaluate_v9.add_argument("--classifier", choices=V9_CLASSIFIERS, default="mv-label")
    evaluate_v9.add_argument("--condition", action="append", choices=tuple(V9_CONDITIONS))
    evaluate_v9.add_argument("--taxonomy")
    evaluate_v9.add_argument("--metrics-output", required=True)
    evaluate_v9.add_argument("--analysis-output", required=True)
    evaluate_v9.add_argument("--size-bins")
    evaluate_v9.add_argument("--viewer-output")
    evaluate_v9.add_argument("--radius-m", type=float, default=0.05)
    evaluate_v9.add_argument("--min-region-size", type=int, default=100)
    evaluate_v9.set_defaults(func=_evaluate_object_system)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
