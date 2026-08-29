from __future__ import annotations

"""Public CLI for the active SAGA candidate-formation experiment.

Candidate commands are dependency-isolated from historical experiment
handlers.  The latter remain callable for forensic reproducibility but are
loaded lazily and are not part of the active runner.
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

# Parser choices are deliberately values, not imports from the retired V8--V10
# implementations.  Candidate-formation commands must remain usable on a
# clean machine even when an old experiment's optional CUDA/Python dependency
# is unavailable.  The corresponding implementation is imported only if that
# legacy command is actually invoked.
V8_CLASSIFIERS = ("mv-label", "codebook")
V8_CONDITIONS = ("U00", "D10", "D01", "D11")
V9_ASSOCIATION_MODES = ("A0", "A1", "A2", "A3")
V9_CLASSIFIERS = ("mv-label", "codebook")
V9_CONDITIONS = (
    "U000",
    "D100",
    "D010",
    "D001",
    "D110",
    "D101",
    "D011",
    "D111",
)
V10_STRUCTURE_CONDITIONS = ("P0R0", "P1R0", "P0R1", "P1R1", "VC1")
V10_CLASSIFIERS = V9_CLASSIFIERS
V10_PRIOR_CONDITIONS = V9_CONDITIONS


def _build_category_fragment_graph(args: argparse.Namespace) -> None:
    from .category_fragment_merge_runner import build_category_fragment_graphs

    payload = build_category_fragment_graphs(
        runtime_manifest=Path(args.runtime_manifest),
        category_priors=Path(args.category_priors),
        output_root=Path(args.output_root),
        scene_ids=args.scene,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _merge_category_fragments(args: argparse.Namespace) -> None:
    from .category_fragment_merge_runner import merge_category_fragment_graphs

    payload = merge_category_fragment_graphs(
        runtime_manifest=Path(args.runtime_manifest),
        category_priors=Path(args.category_priors),
        output_root=Path(args.output_root),
        scene_ids=args.scene,
        modes=args.mode,
        seed=args.seed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_fragment_merge(args: argparse.Namespace) -> None:
    from .category_fragment_merge_scene_evaluation import (
        evaluate_category_fragment_merge_run,
    )

    payload = evaluate_category_fragment_merge_run(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        phase=args.phase,
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_category_denoise_bank(args: argparse.Namespace) -> None:
    from .category_denoise_runner import run_category_denoise_bank

    payload = run_category_denoise_bank(
        runtime_manifest=Path(args.runtime_manifest),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        category_priors=Path(args.category_priors),
        scene_ids=args.scene,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _repair_category_candidates(args: argparse.Namespace) -> None:
    from .category_candidate_runner import repair_category_candidates

    payload = repair_category_candidates(
        runtime_manifest=Path(args.runtime_manifest),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        category_priors=Path(args.category_priors),
        scene_ids=args.scene,
        reference_bank_root=(
            Path(args.reference_bank_root) if args.reference_bank_root else None
        ),
        seed=args.seed,
        sample_cap=args.sample_cap,
        python_bin=Path(args.python_bin) if args.python_bin else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_category_cluster_bank(args: argparse.Namespace) -> None:
    from .category_cluster_runner import run_category_cluster_bank

    payload = run_category_cluster_bank(
        runtime_manifest=Path(args.runtime_manifest),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        category_priors=Path(args.category_priors),
        scene_ids=args.scene,
        conditions=args.condition,
        reference_bank_root=(
            Path(args.reference_bank_root) if args.reference_bank_root else None
        ),
        verify_determinism=bool(args.verify_determinism),
        determinism_reference=(
            Path(args.determinism_reference)
            if args.determinism_reference
            else None
        ),
        seed=args.seed,
        python_bin=Path(args.python_bin) if args.python_bin else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _audit_category_cluster_distance(args: argparse.Namespace) -> None:
    from .category_cluster_runner import audit_category_cluster_distance

    payload = audit_category_cluster_distance(
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        reference_bank_root=Path(args.reference_bank_root),
        reference_trace_root=Path(args.reference_trace_root),
        output_path=Path(args.output),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_feature_routing_factorial(args: argparse.Namespace) -> None:
    from .category_feature_routing_factorial import run_feature_routing_factorial

    payload = run_feature_routing_factorial(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        category_priors=Path(args.category_priors),
        feature_10k_root=Path(args.feature_10k_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        output_dir=Path(args.output_dir),
        size_bins=Path(args.size_bins) if args.size_bins else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_cluster_bank(args: argparse.Namespace) -> None:
    from .category_cluster_scene_evaluation import evaluate_category_cluster_run

    payload = evaluate_category_cluster_run(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        phase=args.phase,
        selected_condition=args.selected_condition,
        primary_analysis=(
            Path(args.primary_analysis) if args.primary_analysis else None
        ),
        frozen_selection_artifact=(
            Path(args.frozen_selection_artifact)
            if args.frozen_selection_artifact
            else None
        ),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _diagnose_category_candidates(args: argparse.Namespace) -> None:
    from .category_candidate_evaluation import diagnose_category_candidates

    payload = diagnose_category_candidates(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        trace_output=Path(args.trace_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_candidates(args: argparse.Namespace) -> None:
    from .category_candidate_evaluation import evaluate_category_candidates

    payload = evaluate_category_candidates(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        phase=args.phase,
        selected_condition=args.selected_condition,
        frozen_repair_artifact=(
            Path(args.frozen_repair_artifact)
            if args.frozen_repair_artifact
            else None
        ),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _diagnose_candidate_representation(args: argparse.Namespace) -> None:
    from .category_candidate_representation import evaluate_candidate_representation

    payload = evaluate_candidate_representation(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        run_root=Path(args.run_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_category_candidates(args: argparse.Namespace) -> None:
    from .category_candidate_runner import replay_repaired_category_candidates

    payload = replay_repaired_category_candidates(
        runtime_manifest=Path(args.runtime_manifest),
        bank_root=Path(args.bank_root),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        category_priors=Path(args.category_priors),
        scene_ids=args.scene,
        modes=args.mode,
        score_threshold=args.score_threshold,
        selected_condition=args.selected_condition,
        seed=args.seed,
        python_bin=Path(args.python_bin) if args.python_bin else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_candidate_final(args: argparse.Namespace) -> None:
    from .category_candidate_final_evaluation import evaluate_candidate_final_stage

    physical_map = None
    if args.physical_scene_map:
        raw = json.loads(Path(args.physical_scene_map).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("physical scene map must be a JSON object")
        physical_map = {str(key): str(value) for key, value in raw.items()}
    payload = evaluate_candidate_final_stage(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        b0_root=Path(args.b0_root),
        replay_root=Path(args.replay_root),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        stage=args.phase,
        output_dir=Path(args.output_dir),
        physical_scene_by_scan=physical_map,
        size_bins=Path(args.size_bins),
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
        write_viewer=not args.no_viewer,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_category_denoise(args: argparse.Namespace) -> None:
    from .category_denoise_runner import replay_category_denoise

    payload = replay_category_denoise(
        runtime_manifest=Path(args.runtime_manifest),
        bank_root=Path(args.bank_root),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
        category_priors=Path(args.category_priors),
        scene_ids=args.scene,
        mode=args.mode,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_denoise(args: argparse.Namespace) -> None:
    from .category_denoise_evaluation import evaluate_category_denoise

    payload = evaluate_category_denoise(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        bank_root=Path(args.bank_root),
        prediction_root=Path(args.prediction_root),
        scene_ids=args.scene,
        conditions=args.condition,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
        size_bins=Path(args.size_bins) if args.size_bins else None,
        viewer_output=Path(args.viewer_output) if args.viewer_output else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _diagnose_category_denoise_funnel(args: argparse.Namespace) -> None:
    from .category_denoise_diagnostic_runner import (
        diagnose_category_denoise_funnel,
    )

    payload = diagnose_category_denoise_funnel(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        bank_root=Path(args.bank_root),
        category_priors=Path(args.category_priors),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        output_dir=Path(args.output_dir),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _prepare_category_denoise_knn_oracle(args: argparse.Namespace) -> None:
    from .category_denoise_diagnostic_runner import (
        prepare_category_denoise_knn_oracle,
    )

    payload = prepare_category_denoise_knn_oracle(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        bank_root=Path(args.bank_root),
        output=Path(args.output),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        iou_threshold=args.iou_threshold,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_category_denoise_knn_oracle(args: argparse.Namespace) -> None:
    from .category_denoise_diagnostic_runner import (
        replay_category_denoise_knn_oracle,
    )

    payload = replay_category_denoise_knn_oracle(
        runtime_manifest=Path(args.runtime_manifest),
        bank_root=Path(args.bank_root),
        b0_root=Path(args.b0_root),
        oracle_plan=Path(args.oracle_plan),
        output_root=Path(args.output_root),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_category_denoise_knn_oracle(args: argparse.Namespace) -> None:
    from .category_denoise_diagnostic_evaluation import (
        evaluate_category_denoise_knn_oracle,
    )

    payload = evaluate_category_denoise_knn_oracle(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        prediction_root=Path(args.prediction_root),
        oracle_plan=Path(args.oracle_plan),
        output_dir=Path(args.output_dir),
        taxonomy=load_taxonomy(args.taxonomy),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _diagnose_category_prior_oracle(args: argparse.Namespace) -> None:
    from .category_denoise_diagnostic_runner import (
        diagnose_category_prior_oracle,
    )

    payload = diagnose_category_prior_oracle(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        category_priors=Path(args.category_priors),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        output_dir=Path(args.output_dir),
        scene_ids=args.scene,
        taxonomy=load_taxonomy(args.taxonomy),
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
    from .v8_runner import run_v8_lifting_factorial

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
    from .v8_bank import build_v8_object_bank
    from .v8_runner import run_v8_lifting_banks

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
    from .v8_bank import replay_v8_priors

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
    from .v8_evaluation import evaluate_v8_replays

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
    from .v9_feature_training import execute_v9_feature_training

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
    from .v9_runner import run_v9_banks

    payload = run_v9_banks(
        lifting_root=args.lifting_root,
        output_root=args.output_root,
        scene_ids=args.scene,
        association_modes=args.association_mode,
        git_commit=args.git_commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_object_priors(args: argparse.Namespace) -> None:
    from .v9_runner import replay_v9_priors

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
    from .v9_metrics import evaluate_v9_candidate_banks, evaluate_v9_predictions

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


def _audit_v10_association(args: argparse.Namespace) -> None:
    from .v10_evaluation import audit_v10_associations

    payload = audit_v10_associations(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        bank_root=Path(args.bank_root),
        scene_ids=args.scene,
        conditions=args.condition,
        classifiers=args.classifier or V10_CLASSIFIERS,
        taxonomy=load_taxonomy(args.taxonomy),
        rows_output=Path(args.rows_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _run_v10_view_consensus(args: argparse.Namespace) -> None:
    from .v10_runner import run_v10_banks

    payload = run_v10_banks(
        lifting_root=args.lifting_root,
        output_root=args.output_root,
        scene_ids=args.scene,
        conditions=args.condition or V10_STRUCTURE_CONDITIONS,
        git_commit=args.git_commit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _replay_v10(args: argparse.Namespace) -> None:
    from .v10_replay import replay_v10_priors

    payload = replay_v10_priors(
        bank_root=args.bank_root,
        output_root=args.output_root,
        scene_ids=args.scene,
        structure_conditions=args.structure_condition,
        prior_conditions=args.condition or V10_PRIOR_CONDITIONS,
        classifier=args.classifier,
        category_priors=args.category_priors,
        acceptance_threshold=args.acceptance_threshold,
        git_commit=args.git_commit,
        nms_core_iou=args.nms_core_iou,
        min_points=args.min_points,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _evaluate_v10(args: argparse.Namespace) -> None:
    from .v10_evaluation import evaluate_v10_replays

    payload = evaluate_v10_replays(
        runtime_manifest=Path(args.runtime_manifest),
        gt_dir=Path(args.gt_dir),
        replay_root=Path(args.replay_root),
        structure_condition=args.structure_condition,
        classifier=args.classifier,
        scene_ids=args.scene,
        conditions=args.condition or V10_PRIOR_CONDITIONS,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_output=Path(args.metrics_output),
        analysis_output=Path(args.analysis_output),
        size_bins=Path(args.size_bins) if args.size_bins else None,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
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

    audit_v10 = commands.add_parser(
        "audit-v10-association",
        help="audit real accepted V10 pairs and the persisted eight-stage funnel",
    )
    audit_v10.add_argument("--runtime-manifest", required=True)
    audit_v10.add_argument("--gt-dir", required=True)
    audit_v10.add_argument("--bank-root", required=True)
    audit_v10.add_argument("--scene", action="append", required=True)
    audit_v10.add_argument(
        "--condition",
        action="append",
        choices=V10_STRUCTURE_CONDITIONS,
        required=True,
    )
    audit_v10.add_argument("--taxonomy")
    audit_v10.add_argument(
        "--classifier", action="append", choices=V10_CLASSIFIERS
    )
    audit_v10.add_argument("--size-bins")
    audit_v10.add_argument("--rows-output", required=True)
    audit_v10.add_argument("--analysis-output", required=True)
    audit_v10.add_argument("--radius-m", type=float, default=0.05)
    audit_v10.add_argument("--min-region-size", type=int, default=100)
    audit_v10.set_defaults(func=_audit_v10_association)

    bank_v10 = commands.add_parser(
        "run-v10-view-consensus",
        help="build the frozen V10 pair/reconstruction arms and VC1 bank",
    )
    bank_v10.add_argument("--lifting-root", required=True)
    bank_v10.add_argument("--output-root", required=True)
    bank_v10.add_argument("--scene", action="append", required=True)
    bank_v10.add_argument(
        "--condition", action="append", choices=V10_STRUCTURE_CONDITIONS
    )
    bank_v10.add_argument("--git-commit", required=True)
    bank_v10.set_defaults(func=_run_v10_view_consensus)

    replay_v10 = commands.add_parser(
        "replay-v10-priors",
        help="replay the frozen V10 2^3 category-prior factorial",
    )
    replay_v10.add_argument("--bank-root", required=True)
    replay_v10.add_argument("--output-root", required=True)
    replay_v10.add_argument("--category-priors", required=True)
    replay_v10.add_argument("--scene", action="append", required=True)
    replay_v10.add_argument(
        "--structure-condition",
        action="append",
        choices=V10_STRUCTURE_CONDITIONS,
        required=True,
    )
    replay_v10.add_argument(
        "--condition", action="append", choices=V10_PRIOR_CONDITIONS
    )
    replay_v10.add_argument("--acceptance-threshold", type=float, required=True)
    replay_v10.add_argument("--classifier", choices=V10_CLASSIFIERS, required=True)
    replay_v10.add_argument("--nms-core-iou", type=float, default=0.50)
    replay_v10.add_argument("--min-points", type=int, default=10)
    replay_v10.add_argument("--git-commit", required=True)
    replay_v10.set_defaults(func=_replay_v10)

    evaluate_v10 = commands.add_parser(
        "evaluate-v10",
        help="run official AP and Gaussian diagnostics on strict V10 replays",
    )
    evaluate_v10.add_argument("--runtime-manifest", required=True)
    evaluate_v10.add_argument("--gt-dir", required=True)
    evaluate_v10.add_argument("--replay-root", required=True)
    evaluate_v10.add_argument(
        "--structure-condition", choices=V10_STRUCTURE_CONDITIONS, required=True
    )
    evaluate_v10.add_argument("--classifier", choices=V10_CLASSIFIERS, required=True)
    evaluate_v10.add_argument("--scene", action="append", required=True)
    evaluate_v10.add_argument(
        "--condition", action="append", choices=V10_PRIOR_CONDITIONS
    )
    evaluate_v10.add_argument("--taxonomy")
    evaluate_v10.add_argument("--metrics-output", required=True)
    evaluate_v10.add_argument("--analysis-output", required=True)
    evaluate_v10.add_argument("--size-bins")
    evaluate_v10.add_argument("--viewer-output")
    evaluate_v10.add_argument("--radius-m", type=float, default=0.05)
    evaluate_v10.add_argument("--min-region-size", type=int, default=100)
    evaluate_v10.set_defaults(func=_evaluate_v10)

    fragment_graph = commands.add_parser(
        "build-category-fragment-graph",
        help="build the frozen raw-fragment graph without GT or category priors",
    )
    fragment_graph.add_argument("--runtime-manifest", required=True)
    fragment_graph.add_argument("--category-priors", required=True)
    fragment_graph.add_argument("--output-root", required=True)
    fragment_graph.add_argument("--scene", action="append", required=True)
    fragment_graph.add_argument("--seed", type=int, default=42)
    fragment_graph.set_defaults(func=_build_category_fragment_graph)

    fragment_merge = commands.add_parser(
        "merge-category-fragments",
        help="replay global or class statistics over one frozen fragment graph",
    )
    fragment_merge.add_argument("--runtime-manifest", required=True)
    fragment_merge.add_argument("--category-priors", required=True)
    fragment_merge.add_argument("--output-root", required=True)
    fragment_merge.add_argument("--scene", action="append", required=True)
    fragment_merge.add_argument(
        "--mode",
        action="append",
        choices=("global", "class"),
        required=True,
    )
    fragment_merge.add_argument("--seed", type=int, default=42)
    fragment_merge.set_defaults(func=_merge_category_fragments)

    fragment_evaluate = commands.add_parser(
        "evaluate-category-fragment-merge",
        help="evaluate the frozen graph oracle and paired U/D fragment assembly",
    )
    fragment_evaluate.add_argument("--runtime-manifest", required=True)
    fragment_evaluate.add_argument("--gt-dir", required=True)
    fragment_evaluate.add_argument("--run-root", required=True)
    fragment_evaluate.add_argument("--scene", action="append", required=True)
    fragment_evaluate.add_argument(
        "--phase", choices=("dev2", "dev8"), required=True
    )
    fragment_evaluate.add_argument("--taxonomy")
    fragment_evaluate.add_argument("--size-bins")
    fragment_evaluate.add_argument("--metrics-output", required=True)
    fragment_evaluate.add_argument("--analysis-output", required=True)
    fragment_evaluate.add_argument("--radius-m", type=float, default=0.05)
    fragment_evaluate.add_argument("--min-region-size", type=int, default=100)
    fragment_evaluate.set_defaults(func=_evaluate_category_fragment_merge)

    cluster_bank = commands.add_parser(
        "run-category-cluster-bank",
        help="build exact R0 and registered repaired instance-candidate banks",
    )
    cluster_bank.add_argument("--runtime-manifest", required=True)
    cluster_bank.add_argument("--output-root", required=True)
    cluster_bank.add_argument("--repo-root", default=".")
    cluster_bank.add_argument("--category-priors", required=True)
    cluster_bank.add_argument("--reference-bank-root")
    cluster_bank.add_argument(
        "--verify-determinism",
        action="store_true",
        help="independently rebuild every requested DEV2 bank and compare pointwise",
    )
    cluster_bank.add_argument(
        "--determinism-reference",
        help="measured DEV2 cluster-analysis JSON used by later stages",
    )
    cluster_bank.add_argument("--python-bin")
    cluster_bank.add_argument("--seed", type=int, default=42)
    cluster_bank.add_argument("--scene", action="append", required=True)
    cluster_bank.add_argument(
        "--condition",
        action="append",
        choices=(
            "R0-legacy",
            "R1-corrected-distance-legacy-expand",
            "R2-corrected-distance-anchored-expand",
            "G1-mutual-local-graph",
        ),
    )
    cluster_bank.set_defaults(func=_run_category_cluster_bank)

    cluster_audit = commands.add_parser(
        "audit-category-cluster-distance",
        help="verify exact R0 identity and corrected metric registration",
    )
    cluster_audit.add_argument("--run-root", required=True)
    cluster_audit.add_argument("--reference-bank-root", required=True)
    cluster_audit.add_argument("--reference-trace-root", required=True)
    cluster_audit.add_argument("--scene", action="append", required=True)
    cluster_audit.add_argument("--output", required=True)
    cluster_audit.set_defaults(func=_audit_category_cluster_distance)

    cluster_evaluate = commands.add_parser(
        "evaluate-category-cluster-bank",
        help="evaluate DEV2/DEV8 repaired candidate banks with frozen gates",
    )
    cluster_evaluate.add_argument("--runtime-manifest", required=True)
    cluster_evaluate.add_argument("--gt-dir", required=True)
    cluster_evaluate.add_argument("--run-root", required=True)
    cluster_evaluate.add_argument("--scene", action="append", required=True)
    cluster_evaluate.add_argument("--taxonomy")
    cluster_evaluate.add_argument("--size-bins")
    cluster_evaluate.add_argument("--phase", choices=("dev2", "dev8"), required=True)
    cluster_evaluate.add_argument(
        "--selected-condition",
        choices=(
            "R1-corrected-distance-legacy-expand",
            "R2-corrected-distance-anchored-expand",
            "G1-mutual-local-graph",
        ),
    )
    cluster_evaluate.add_argument(
        "--primary-analysis",
        help="required only for conditional G1 DEV2 evaluation",
    )
    cluster_evaluate.add_argument(
        "--frozen-selection-artifact",
        help="required for DEV8 and must authorize --selected-condition",
    )
    cluster_evaluate.add_argument("--metrics-output", required=True)
    cluster_evaluate.add_argument("--analysis-output", required=True)
    cluster_evaluate.add_argument("--radius-m", type=float, default=0.05)
    cluster_evaluate.add_argument("--min-region-size", type=int, default=100)
    cluster_evaluate.set_defaults(func=_evaluate_category_cluster_bank)

    feature_route = commands.add_parser(
        "run-feature-routing-factorial",
        help="isolate feature representation and semantic routing at raw HDBSCAN",
    )
    feature_route.add_argument("--runtime-manifest", required=True)
    feature_route.add_argument("--gt-dir", required=True)
    feature_route.add_argument("--category-priors", required=True)
    feature_route.add_argument("--feature-10k-root", required=True)
    feature_route.add_argument("--scene", action="append", required=True)
    feature_route.add_argument("--taxonomy")
    feature_route.add_argument("--size-bins")
    feature_route.add_argument("--output-dir", required=True)
    feature_route.set_defaults(func=_run_feature_routing_factorial)

    candidate_repair = commands.add_parser(
        "repair-category-candidates",
        help="trace one HDBSCAN run and build C0/C1/C2 candidate banks",
    )
    candidate_repair.add_argument("--runtime-manifest", required=True)
    candidate_repair.add_argument("--output-root", required=True)
    candidate_repair.add_argument("--repo-root", default=".")
    candidate_repair.add_argument("--category-priors", required=True)
    candidate_repair.add_argument("--reference-bank-root")
    candidate_repair.add_argument("--python-bin")
    candidate_repair.add_argument("--seed", type=int, default=42)
    candidate_repair.add_argument("--sample-cap", type=int, default=5000)
    candidate_repair.add_argument("--scene", action="append", required=True)
    candidate_repair.set_defaults(func=_repair_category_candidates)

    candidate_diagnose = commands.add_parser(
        "diagnose-category-candidates",
        help="classify sampling, raw clustering, and full assignment failures",
    )
    candidate_diagnose.add_argument("--runtime-manifest", required=True)
    candidate_diagnose.add_argument("--gt-dir", required=True)
    candidate_diagnose.add_argument("--run-root", required=True)
    candidate_diagnose.add_argument("--scene", action="append", required=True)
    candidate_diagnose.add_argument("--taxonomy")
    candidate_diagnose.add_argument("--size-bins")
    candidate_diagnose.add_argument("--trace-output", required=True)
    candidate_diagnose.add_argument("--analysis-output", required=True)
    candidate_diagnose.add_argument("--radius-m", type=float, default=0.05)
    candidate_diagnose.add_argument("--min-region-size", type=int, default=100)
    candidate_diagnose.set_defaults(func=_diagnose_category_candidates)

    candidate_evaluate = commands.add_parser(
        "evaluate-category-candidates",
        help="evaluate and gate C0/C1/C2 candidate banks",
    )
    candidate_evaluate.add_argument("--runtime-manifest", required=True)
    candidate_evaluate.add_argument("--gt-dir", required=True)
    candidate_evaluate.add_argument("--run-root", required=True)
    candidate_evaluate.add_argument("--scene", action="append", required=True)
    candidate_evaluate.add_argument("--taxonomy")
    candidate_evaluate.add_argument("--size-bins")
    candidate_evaluate.add_argument("--metrics-output", required=True)
    candidate_evaluate.add_argument("--analysis-output", required=True)
    candidate_evaluate.add_argument("--phase", choices=("dev2", "dev8"), required=True)
    candidate_evaluate.add_argument(
        "--selected-condition",
        choices=("C1-consistent-envelope", "C2-raw-anchored-envelope"),
        help="DEV2-frozen repair arm; required for DEV8 and forbidden for DEV2",
    )
    candidate_evaluate.add_argument(
        "--frozen-repair-artifact",
        help="required for DEV8; binds evaluation to the DEV2-selected arm",
    )
    candidate_evaluate.add_argument("--radius-m", type=float, default=0.05)
    candidate_evaluate.add_argument("--min-region-size", type=int, default=100)
    candidate_evaluate.set_defaults(func=_evaluate_category_candidates)

    representation = commands.add_parser(
        "diagnose-candidate-representation",
        help="measure local affinity AUROC and a GT-seed offline upper control",
    )
    representation.add_argument("--runtime-manifest", required=True)
    representation.add_argument("--gt-dir", required=True)
    representation.add_argument("--run-root", required=True)
    representation.add_argument("--scene", action="append", required=True)
    representation.add_argument("--taxonomy")
    representation.add_argument("--size-bins")
    representation.add_argument("--metrics-output", required=True)
    representation.add_argument("--analysis-output", required=True)
    representation.add_argument("--radius-m", type=float, default=0.05)
    representation.add_argument("--min-region-size", type=int, default=100)
    representation.set_defaults(func=_diagnose_candidate_representation)

    candidate_replay = commands.add_parser(
        "replay-category-candidates",
        help="replay accepted repaired candidates through legacy KNN/filter",
    )
    candidate_replay.add_argument("--runtime-manifest", required=True)
    candidate_replay.add_argument("--bank-root", required=True)
    candidate_replay.add_argument("--output-root", required=True)
    candidate_replay.add_argument("--repo-root", default=".")
    candidate_replay.add_argument("--category-priors", required=True)
    candidate_replay.add_argument("--scene", action="append", required=True)
    candidate_replay.add_argument(
        "--mode", action="append", choices=("uniform", "class"), required=True
    )
    candidate_replay.add_argument("--score-threshold", type=float, required=True)
    candidate_replay.add_argument(
        "--selected-condition",
        choices=("C1-consistent-envelope", "C2-raw-anchored-envelope"),
        help=(
            "read repair-layout banks from bank/<scene>/<condition>; omit only "
            "when --bank-root is already a flat selected-bank root"
        ),
    )
    candidate_replay.add_argument("--seed", type=int, default=42)
    candidate_replay.add_argument("--python-bin")
    candidate_replay.set_defaults(func=_replay_category_candidates)

    candidate_final = commands.add_parser(
        "evaluate-category-candidate-final",
        help="evaluate frozen B0/U/D candidate replays at a registered stage",
    )
    candidate_final.add_argument("--runtime-manifest", required=True)
    candidate_final.add_argument("--gt-dir", required=True)
    candidate_final.add_argument("--b0-root", required=True)
    candidate_final.add_argument("--replay-root", required=True)
    candidate_final.add_argument("--scene", action="append", required=True)
    candidate_final.add_argument(
        "--phase", choices=("dev8", "holdout", "tune", "final"), required=True
    )
    candidate_final.add_argument("--output-dir", required=True)
    candidate_final.add_argument("--taxonomy")
    candidate_final.add_argument("--size-bins", required=True)
    candidate_final.add_argument("--physical-scene-map")
    candidate_final.add_argument("--radius-m", type=float, default=0.05)
    candidate_final.add_argument("--min-region-size", type=int, default=100)
    candidate_final.add_argument("--no-viewer", action="store_true")
    candidate_final.set_defaults(func=_evaluate_category_candidate_final)

    denoise_bank = commands.add_parser(
        "run-category-denoise-bank",
        help="build one immutable all-category candidate bank per scene",
    )
    denoise_bank.add_argument("--runtime-manifest", required=True)
    denoise_bank.add_argument("--output-root", required=True)
    denoise_bank.add_argument("--repo-root", default=".")
    denoise_bank.add_argument("--category-priors", required=True)
    denoise_bank.add_argument("--scene", action="append", required=True)
    denoise_bank.set_defaults(func=_run_category_denoise_bank)

    denoise_replay = commands.add_parser(
        "replay-category-denoise",
        help="replay uniform or class statistics over an immutable denoising bank",
    )
    denoise_replay.add_argument("--runtime-manifest", required=True)
    denoise_replay.add_argument("--bank-root", required=True)
    denoise_replay.add_argument("--output-root", required=True)
    denoise_replay.add_argument("--repo-root", default=".")
    denoise_replay.add_argument("--category-priors", required=True)
    denoise_replay.add_argument("--mode", choices=("uniform", "class"), required=True)
    denoise_replay.add_argument("--scene", action="append", required=True)
    denoise_replay.set_defaults(func=_replay_category_denoise)

    denoise_evaluate = commands.add_parser(
        "evaluate-category-denoise",
        help="official AP and candidate diagnostics for all-category denoising",
    )
    denoise_evaluate.add_argument("--runtime-manifest", required=True)
    denoise_evaluate.add_argument("--gt-dir", required=True)
    denoise_evaluate.add_argument("--bank-root", required=True)
    denoise_evaluate.add_argument("--prediction-root", required=True)
    denoise_evaluate.add_argument("--scene", action="append", required=True)
    denoise_evaluate.add_argument("--condition", action="append", required=True)
    denoise_evaluate.add_argument("--taxonomy")
    denoise_evaluate.add_argument("--metrics-output", required=True)
    denoise_evaluate.add_argument("--analysis-output", required=True)
    denoise_evaluate.add_argument("--viewer-output")
    denoise_evaluate.add_argument("--size-bins")
    denoise_evaluate.add_argument("--radius-m", type=float, default=0.05)
    denoise_evaluate.add_argument("--min-region-size", type=int, default=100)
    denoise_evaluate.set_defaults(func=_evaluate_category_denoise)

    funnel = commands.add_parser(
        "diagnose-category-denoise-funnel",
        help="diagnose semantic, core, full-candidate, and score bottlenecks",
    )
    funnel.add_argument("--runtime-manifest", required=True)
    funnel.add_argument("--gt-dir", required=True)
    funnel.add_argument("--bank-root", required=True)
    funnel.add_argument("--category-priors", required=True)
    funnel.add_argument("--size-bins")
    funnel.add_argument("--output-dir", required=True)
    funnel.add_argument("--scene", action="append", required=True)
    funnel.add_argument("--taxonomy")
    funnel.add_argument("--radius-m", type=float, default=0.05)
    funnel.add_argument("--min-region-size", type=int, default=100)
    funnel.set_defaults(func=_diagnose_category_denoise_funnel)

    prepare_knn = commands.add_parser(
        "prepare-category-denoise-knn-oracle",
        help="select evaluation-only same-class IoU>=0.50 candidates",
    )
    prepare_knn.add_argument("--runtime-manifest", required=True)
    prepare_knn.add_argument("--gt-dir", required=True)
    prepare_knn.add_argument("--bank-root", required=True)
    prepare_knn.add_argument("--output", required=True)
    prepare_knn.add_argument("--scene", action="append", required=True)
    prepare_knn.add_argument("--taxonomy")
    prepare_knn.add_argument("--iou-threshold", type=float, default=0.50)
    prepare_knn.add_argument("--radius-m", type=float, default=0.05)
    prepare_knn.add_argument("--min-region-size", type=int, default=100)
    prepare_knn.set_defaults(func=_prepare_category_denoise_knn_oracle)

    replay_knn = commands.add_parser(
        "replay-category-denoise-knn-oracle",
        help="GT-free O1/O2 replay of a frozen oracle candidate plan",
    )
    replay_knn.add_argument("--runtime-manifest", required=True)
    replay_knn.add_argument("--bank-root", required=True)
    replay_knn.add_argument("--b0-root", required=True)
    replay_knn.add_argument("--oracle-plan", required=True)
    replay_knn.add_argument("--output-root", required=True)
    replay_knn.set_defaults(func=_replay_category_denoise_knn_oracle)

    evaluate_knn = commands.add_parser(
        "evaluate-category-denoise-knn-oracle",
        help="evaluate fixed B0 metadata and O1/O2 KNN counterfactuals",
    )
    evaluate_knn.add_argument("--runtime-manifest", required=True)
    evaluate_knn.add_argument("--gt-dir", required=True)
    evaluate_knn.add_argument("--prediction-root", required=True)
    evaluate_knn.add_argument("--oracle-plan", required=True)
    evaluate_knn.add_argument("--output-dir", required=True)
    evaluate_knn.add_argument("--size-bins")
    evaluate_knn.add_argument("--taxonomy")
    evaluate_knn.add_argument("--radius-m", type=float, default=0.05)
    evaluate_knn.add_argument("--min-region-size", type=int, default=100)
    evaluate_knn.set_defaults(func=_evaluate_category_denoise_knn_oracle)

    prior_oracle = commands.add_parser(
        "diagnose-category-prior-oracle",
        help="test prior calibration on complete GT-derived Gaussian objects",
    )
    prior_oracle.add_argument("--runtime-manifest", required=True)
    prior_oracle.add_argument("--gt-dir", required=True)
    prior_oracle.add_argument("--category-priors", required=True)
    prior_oracle.add_argument("--size-bins")
    prior_oracle.add_argument("--output-dir", required=True)
    prior_oracle.add_argument("--scene", action="append", required=True)
    prior_oracle.add_argument("--taxonomy")
    prior_oracle.add_argument("--radius-m", type=float, default=0.05)
    prior_oracle.add_argument("--min-region-size", type=int, default=100)
    prior_oracle.set_defaults(func=_diagnose_category_prior_oracle)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
