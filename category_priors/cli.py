from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import audit_saga_alignment
from .backbone_diagnostics import diagnose_backbone_runs
from .analysis import analyze_manifest
from .class_first import write_class_first_params
from .class_first_runner import CLASS_FIRST_CONDITIONS, execute_class_first_runs
from .class_first_evaluation import evaluate_class_first_runs
from .legacy_prior import load_legacy_prior_config, write_legacy_prior_params
from .legacy_prior_runner import (
    LEGACY_PRIOR_CONDITIONS,
    execute_legacy_prior_runs,
)
from .teacher_prior_evaluation import evaluate_teacher_prior_runs
from .teacher_prior import materialize_teacher_prior
from .teacher_prior_runner import (
    TEACHER_PRIOR_CONDITIONS,
    execute_teacher_prior_runs,
)
from .download import (
    MINIMAL_FILE_TYPES,
    download_scannet_saga_scenes,
    download_scannet_subset,
)
from .evaluator import evaluate_manifest
from .io import hash_json, load_json, read_rows, sha256_file, write_json, write_rows
from .locked import assess_seed_sensitivity, build_locked_plan
from .locked_evaluation import evaluate_locked_plan, evaluate_tune_seed_execution
from .mapping import (
    DEFAULT_MAPPING_CONFIG,
    build_lock_manifest,
    build_mapping_config,
    build_run_schedule,
    choose_best_config,
    latin_hypercube_design,
    validate_mapping_config,
)
from .priors import fit_priors, validate_priors, write_priors
from .runner import execute_locked_plan, execute_schedule
from .scannet import (
    discover_scene_files,
    extract_scene_stats,
    prepare_scene_ground_truth,
    read_scene_ids,
    validate_scene_ids,
)
from .scannet_saga import prepare_saga_scene
from .search import (
    build_search_schedule,
    evaluate_search_execution,
    materialize_search_mappings,
)
from .selection import select_locked_evaluation_scenes, select_scenes
from .taxonomy import load_taxonomy


def command_download_scannet(args: argparse.Namespace) -> None:
    download_scannet_subset(
        official_downloader=args.official_downloader,
        scene_list=args.scene_list,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        accept_tos=args.accept_tos,
        file_types=tuple(args.file_types or MINIMAL_FILE_TYPES),
        include_label_map=not args.no_label_map,
        workers=args.workers,
        retries=args.retries,
        timeout_s=args.timeout_s,
        min_free_gb=args.min_free_gb,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def command_download_scannet_saga(args: argparse.Namespace) -> None:
    download_scannet_saga_scenes(
        official_downloader=args.official_downloader,
        scene_list=args.scene_list,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        accept_tos=args.accept_tos,
        workers=args.workers,
        retries=args.retries,
        timeout_s=args.timeout_s,
        min_free_gb=args.min_free_gb,
        limit=args.limit,
        dry_run=args.dry_run,
    )


def _extract_one(
    dataset_root: str,
    scene_id: str,
    taxonomy_path: str | None,
    split: str,
    dataset: str,
    voxel_size_m: float,
    oversample_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    taxonomy = load_taxonomy(taxonomy_path)
    files = discover_scene_files(dataset_root, scene_id)
    return extract_scene_stats(
        files,
        taxonomy,
        split,
        dataset,
        voxel_size_m,
        oversample_factor,
    )


def command_extract(args: argparse.Namespace) -> None:
    scene_ids = validate_scene_ids(read_scene_ids(args.scene_list))
    taxonomy_path = str(Path(args.taxonomy).resolve()) if args.taxonomy else None
    label_map_hash = sha256_file(args.label_map) if args.label_map else None
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if args.workers == 1:
        for scene_id in scene_ids:
            try:
                scene_rows, audit = _extract_one(
                    args.dataset_root,
                    scene_id,
                    taxonomy_path,
                    args.split,
                    args.dataset,
                    args.voxel_size_m,
                    args.oversample_factor,
                )
                rows.extend(scene_rows)
                audits.append(audit)
            except Exception as exc:  # noqa: BLE001 - preserve audit evidence before failing
                errors.append(
                    {"scene_id": scene_id, "error": f"{type(exc).__name__}: {exc}"}
                )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _extract_one,
                    args.dataset_root,
                    scene_id,
                    taxonomy_path,
                    args.split,
                    args.dataset,
                    args.voxel_size_m,
                    args.oversample_factor,
                ): scene_id
                for scene_id in scene_ids
            }
            for future in as_completed(futures):
                scene_id = futures[future]
                try:
                    scene_rows, audit = future.result()
                    rows.extend(scene_rows)
                    audits.append(audit)
                except Exception as exc:  # noqa: BLE001 - worker errors belong in the audit
                    errors.append(
                        {"scene_id": scene_id, "error": f"{type(exc).__name__}: {exc}"}
                    )
    rows.sort(key=lambda row: (row["scene_id"], row["instance_id"]))
    audits.sort(key=lambda row: row["scene_id"])
    if label_map_hash:
        for row in rows:
            row["label_map_sha256"] = label_map_hash
    write_rows(args.output, rows)
    audit_payload = {
        "schema_version": "1.0",
        "kind": "scannet_statistics_audit",
        "dataset": args.dataset,
        "split": args.split,
        "scene_count_requested": len(scene_ids),
        "scene_count_succeeded": len(audits),
        "row_count": len(rows),
        "taxonomy_sha256": load_taxonomy(taxonomy_path).content_hash,
        "label_map_sha256": label_map_hash,
        "statistics_table": Path(args.output).name,
        "statistics_table_sha256": sha256_file(args.output),
        "scenes": audits,
        "errors": sorted(errors, key=lambda item: item["scene_id"]),
    }
    audit_payload["content_sha256"] = hash_json(audit_payload)
    write_json(args.audit_output, audit_payload)
    if errors and not args.allow_scene_errors:
        raise RuntimeError(
            f"Statistics extraction failed for {len(errors)} scenes; see {args.audit_output}"
        )


def command_prepare_gt(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    scene_ids = validate_scene_ids(read_scene_ids(args.scene_list))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "kind": "canonical_ground_truth",
        "dataset": args.dataset,
        "taxonomy_sha256": taxonomy.content_hash,
        "scenes": [],
    }
    for scene_id in scene_ids:
        files = discover_scene_files(args.dataset_root, scene_id)
        coords, semantic, instance = prepare_scene_ground_truth(
            files, taxonomy, args.dataset
        )
        target = output_dir / f"{scene_id}.npz"
        np.savez_compressed(target, coords=coords, semantic=semantic, instance=instance)
        manifest["scenes"].append(
            {
                "scene_id": scene_id,
                "path": target.name,
                "sha256": sha256_file(target),
                "vertices": len(coords),
                "mapped_vertices": int((semantic >= 0).sum()),
            }
        )
    manifest["content_sha256"] = hash_json(manifest)
    write_json(output_dir / "manifest.json", manifest)


def command_prepare_saga_scene(args: argparse.Namespace) -> None:
    payload = prepare_saga_scene(
        dataset_root=args.dataset_root,
        scene_id=args.scene_id,
        sens_path=args.sens,
        output_root=args.output_root,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        max_initial_points=args.max_initial_points,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "scene_id": payload["scene_id"],
                "base_path": payload["base_path"],
                "selected_valid_frames": payload["frame_selection"][
                    "selected_valid_frames"
                ],
                "initial_points": payload["initial_point_cloud"]["vertices"],
            },
            ensure_ascii=False,
        )
    )


def command_audit_saga_alignment(args: argparse.Namespace) -> None:
    payload = audit_saga_alignment(
        preparation_manifest_path=args.preparation_manifest,
        gt_npz_path=args.gt_npz,
        output_path=args.output,
        gaussian_ply_path=args.gaussian_ply,
        radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        camera_padding_m=args.camera_padding_m,
        minimal=args.minimal,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "scene_id": payload["scene_id"],
                "point_cloud_role": payload["point_cloud_role"],
                "mapped_fraction": payload["gt_to_cloud"]["mapped_fraction"],
                "camera_inside_fraction": payload["cameras"][
                    "inside_padded_gt_fraction"
                ],
            },
            ensure_ascii=False,
        )
    )


def command_fit(args: argparse.Namespace) -> None:
    rows = read_rows(args.stats)
    taxonomy = load_taxonomy(args.taxonomy)
    payload = fit_priors(
        rows,
        taxonomy,
        args.stats,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        min_physical_scenes=args.min_physical_scenes,
        shrink_tau=args.shrink_tau,
    )
    write_priors(args.output, payload)


def command_select_scenes(args: argparse.Namespace) -> None:
    rows = read_rows(args.stats)
    taxonomy = load_taxonomy(args.taxonomy)
    payload = select_scenes(
        rows,
        taxonomy,
        args.tune_budget,
        args.locked_budget,
        args.tune_target,
        args.locked_target,
        args.seed,
    )
    write_json(args.output, payload)


def command_select_locked_scenes(args: argparse.Namespace) -> None:
    rows = read_rows(args.stats)
    taxonomy = load_taxonomy(args.taxonomy)
    previous_selection = load_json(args.scene_selection)
    payload = select_locked_evaluation_scenes(
        rows,
        taxonomy,
        previous_selection,
        locked_budget=args.locked_budget,
        target_per_class=args.locked_target,
    )
    write_json(args.output, payload)


def command_assess_seeds(args: argparse.Namespace) -> None:
    rows = []
    for path in args.metrics:
        rows.extend(read_rows(path))
    decision = assess_seed_sensitivity(
        rows, maximum_range=args.maximum_range
    )
    write_json(args.output, decision)


def command_build_locked_plan(args: argparse.Namespace) -> None:
    decision = load_json(args.seed_decision)
    if decision.get("kind") != "seed_sensitivity_decision":
        raise ValueError("Expected a seed_sensitivity_decision")
    commit = args.code_commit
    if not commit:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    payload = build_locked_plan(
        args.locked_scenes,
        args.priors,
        args.mapping,
        args.taxonomy,
        commit,
        decision["selected_locked_seeds"],
        randomization_seed=args.seed,
    )
    payload["seed_sensitivity"] = {
        "decision": decision["decision"],
        "ranges": decision["ranges"],
        "treatment_minus_reference": decision["treatment_minus_reference"],
    }
    write_json(args.output, payload)


def command_search_design(args: argparse.Namespace) -> None:
    write_json(args.output, latin_hypercube_design(args.kind, args.samples, args.seed))


def command_materialize_search(args: argparse.Namespace) -> None:
    materialize_search_mappings(
        design_path=args.design,
        output_dir=args.output_dir,
        manifest_path=args.output,
        priors_path=args.priors,
        taxonomy_path=args.taxonomy,
        scene_selection_path=args.scene_selection,
        base_mapping_path=args.base_mapping,
    )


def command_search_schedule(args: argparse.Namespace) -> None:
    write_json(
        args.output,
        build_search_schedule(
            args.scene_selection,
            args.mapping_manifest,
            args.run_seed or (42,),
            args.seed,
        ),
    )


def command_select_config(args: argparse.Namespace) -> None:
    rows = read_rows(args.metrics)
    design = load_json(args.design)
    selected = choose_best_config(rows, design, args.tie_ap)
    selected["provenance"] = {
        "split": "val-tune",
        "metrics_sha256": sha256_file(args.metrics),
        "search_design_sha256": sha256_file(args.design),
    }
    selected["content_sha256"] = hash_json(selected)
    write_json(args.output, selected)


def command_build_mapping(args: argparse.Namespace) -> None:
    global_best = load_json(args.global_best)
    prior_best = load_json(args.prior_best) if args.prior_best else None
    payload = build_mapping_config(
        global_best["parameters"],
        prior_best["parameters"]
        if prior_best
        else DEFAULT_MAPPING_CONFIG["coefficients"],
        args.priors,
        args.taxonomy,
        args.scene_selection,
    )
    payload["provenance"]["mapping_stage"] = (
        "global+prior" if prior_best else "global-only"
    )
    payload.pop("content_sha256", None)
    payload["content_sha256"] = hash_json(payload)
    write_json(args.output, payload)


def command_lock(args: argparse.Namespace) -> None:
    paths = {}
    for item in args.artifact:
        if "=" not in item:
            raise ValueError("--artifact values must use NAME=PATH")
        name, path = item.split("=", 1)
        paths[name] = path
    commit = args.code_commit
    if not commit:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    write_json(args.output, build_lock_manifest(commit, paths))


def command_schedule(args: argparse.Namespace) -> None:
    write_json(
        args.output,
        build_run_schedule(
            args.scene_selection,
            args.split,
            args.condition or None,
            args.run_seed or (42, 3407, 20260804),
            args.seed,
        ),
    )


def command_run_experiment(args: argparse.Namespace) -> None:
    execute_schedule(
        args.schedule,
        args.scene_manifest,
        args.output_root,
        args.output,
        args.pipeline,
        args.priors,
        args.mapping,
        args.dry_run,
        not args.no_resume,
        args.continue_on_error,
        args.max_runs,
    )


def command_run_locked(args: argparse.Namespace) -> None:
    execute_locked_plan(
        args.plan,
        args.scene_manifest,
        args.output_root,
        args.progress,
        args.pipeline,
        None,
        None,
        args.dry_run,
        args.max_runs,
    )


def command_run_class_first(args: argparse.Namespace) -> None:
    result = execute_class_first_runs(
        scene_manifest_path=args.scene_manifest,
        output_root=args.output_root,
        pipeline_path=args.pipeline,
        category_priors_path=args.category_priors,
        class_first_config_path=args.class_first_config,
        conditions=args.condition,
        seeds=args.seed,
        scene_ids=args.scene,
        resume=not args.no_resume,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_build_class_first_params(args: argparse.Namespace) -> None:
    payload = write_class_first_params(
        args.category_priors,
        args.class_first_config,
        args.output,
        mode=args.mode,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_evaluate_class_first(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    payload = evaluate_class_first_runs(
        scene_manifest_path=args.scene_manifest,
        gt_dir=args.gt_dir,
        output_root=output_root,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_path=args.metrics_output
        or output_root / "class_first_metrics.parquet",
        analysis_path=args.analysis_output
        or output_root / "class_first_analysis.json",
        conditions=args.condition,
        seeds=args.seed or (42, 3407, 20260804),
        scene_ids=args.scene,
        scene_list_path=args.scene_list,
        selection_path=args.selection,
        selection_split=args.selection_split,
        reference=args.reference,
        treatment=args.treatment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        min_region_size=args.min_region_size,
        split=args.split,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_run_teacher_prior(args: argparse.Namespace) -> None:
    result = execute_teacher_prior_runs(
        scene_manifest=args.scene_manifest,
        output_root=args.output_root,
        pipeline=args.pipeline,
        category_params=args.teacher_category_params,
        conditions=args.condition,
        seeds=args.seed,
        scene_ids=args.scene,
        resume=not args.no_resume,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_build_teacher_category_params(args: argparse.Namespace) -> None:
    payload = materialize_teacher_prior(
        load_json(args.category_priors),
        branch_preservation=args.branch_preservation,
    )
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_evaluate_teacher_prior(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    payload = evaluate_teacher_prior_runs(
        scene_manifest_path=args.scene_manifest,
        gt_dir=args.gt_dir,
        output_root=output_root,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_path=args.metrics_output
        or output_root / "teacher_prior_metrics.parquet",
        analysis_path=args.analysis_output
        or output_root / "teacher_prior_analysis.json",
        conditions=args.condition,
        seeds=args.seed or (42,),
        scene_ids=args.scene,
        scene_list_path=args.scene_list,
        selection_path=args.selection,
        selection_split=args.selection_split,
        reference=args.reference,
        treatment=args.treatment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        min_region_size=args.min_region_size,
        split=args.split,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_run_prior_v2(args: argparse.Namespace) -> None:
    result = execute_legacy_prior_runs(
        scene_manifest=args.scene_manifest,
        output_root=args.output_root,
        pipeline=args.pipeline,
        category_priors=args.category_priors,
        config=args.legacy_prior_config,
        conditions=args.condition,
        seeds=args.seed,
        scene_ids=args.scene,
        score=args.score,
        semantic_source=args.semantic_source,
        resume=not args.no_resume,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
        max_runs=args.max_runs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_build_prior_v2_params(args: argparse.Namespace) -> None:
    payload = write_legacy_prior_params(
        load_json(args.category_priors),
        load_legacy_prior_config(args.legacy_prior_config),
        args.output,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_evaluate_prior_v2(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    payload = evaluate_class_first_runs(
        scene_manifest_path=args.scene_manifest,
        gt_dir=args.gt_dir,
        output_root=output_root,
        taxonomy=load_taxonomy(args.taxonomy),
        metrics_path=args.metrics_output or output_root / "prior_v2_metrics.parquet",
        analysis_path=args.analysis_output or output_root / "prior_v2_analysis.json",
        conditions=args.condition,
        seeds=args.seed or (42,),
        scene_ids=args.scene,
        scene_list_path=args.scene_list,
        selection_path=args.selection,
        selection_split=args.selection_split,
        reference=args.reference,
        treatment=args.treatment,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        min_region_size=args.min_region_size,
        split=args.split,
        supported_conditions=tuple(LEGACY_PRIOR_CONDITIONS),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_diagnose_backbone(args: argparse.Namespace) -> None:
    payload = diagnose_backbone_runs(
        scene_manifest=args.scene_manifest,
        gt_dir=args.gt_dir,
        output_root=args.output_root,
        taxonomy=load_taxonomy(args.taxonomy),
        output_json=args.output,
        output_table=args.table,
        conditions=args.condition,
        seeds=args.seed,
        scene_ids=args.scene,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_evaluate_seed_audit(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    evaluate_tune_seed_execution(
        args.schedule,
        args.execution,
        args.scene_manifest,
        args.gt_dir,
        taxonomy,
        args.output,
        config_id=args.config_id,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )


def command_evaluate_locked(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    evaluate_locked_plan(
        args.plan,
        args.scene_manifest,
        args.gt_dir,
        args.output_root,
        taxonomy,
        args.metrics_output,
        args.analysis_output,
    )


def command_evaluate(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    evaluate_manifest(
        args.manifest, taxonomy, args.output, args.radius_m, args.min_region_size
    )


def command_evaluate_search(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    evaluate_search_execution(
        schedule_path=args.schedule,
        execution_path=args.execution,
        scene_manifest_path=args.scene_manifest,
        gt_manifest_path=args.gt_manifest,
        taxonomy=taxonomy,
        output_dir=args.output_dir,
        metrics_path=args.output,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )


def command_analyze(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    analyze_manifest(
        args.manifest,
        taxonomy,
        args.output,
        args.bootstrap_samples,
        args.seed,
        args.radius_m,
        args.min_region_size,
    )


def command_validate(args: argparse.Namespace) -> None:
    priors = load_json(args.priors)
    validate_priors(priors)
    mapping = load_json(args.mapping)
    validate_mapping_config(mapping)
    if mapping["provenance"]["category_priors_sha256"] != sha256_file(args.priors):
        raise ValueError("Mapping config does not match the supplied prior file")
    print(
        json.dumps(
            {"status": "ok", "priors": args.priors, "mapping": args.mapping},
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Category-prior research utilities for SAGA"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser(
        "download-scannet",
        help="Download a licensed ScanNet split's registered minimal statistics files",
    )
    download.add_argument("--official-downloader", required=True)
    download.add_argument("--scene-list", required=True)
    download.add_argument("--out-dir", required=True)
    download.add_argument("--manifest", required=True)
    download.add_argument(
        "--type",
        dest="file_types",
        action="append",
        choices=MINIMAL_FILE_TYPES,
        help="repeat to override the registered four-file minimal set",
    )
    download.add_argument("--no-label-map", action="store_true")
    download.add_argument("--workers", type=int, default=2)
    download.add_argument("--retries", type=int, default=3)
    download.add_argument("--timeout-s", type=float, default=120.0)
    download.add_argument("--min-free-gb", type=float, default=80.0)
    download.add_argument("--limit", type=int)
    download.add_argument("--dry-run", action="store_true")
    download.add_argument("--accept-tos", action="store_true")
    download.set_defaults(func=command_download_scannet)

    download_saga = subparsers.add_parser(
        "download-scannet-saga",
        help="Download resumable .sens streams for already selected SAGA scenes",
    )
    download_saga.add_argument("--official-downloader", required=True)
    download_saga.add_argument("--scene-list", required=True)
    download_saga.add_argument("--out-dir", required=True)
    download_saga.add_argument("--manifest", required=True)
    download_saga.add_argument("--workers", type=int, default=1)
    download_saga.add_argument("--retries", type=int, default=8)
    download_saga.add_argument("--timeout-s", type=float, default=300.0)
    download_saga.add_argument("--min-free-gb", type=float, default=80.0)
    download_saga.add_argument("--limit", type=int)
    download_saga.add_argument("--dry-run", action="store_true")
    download_saga.add_argument("--accept-tos", action="store_true")
    download_saga.set_defaults(func=command_download_scannet_saga)

    extract = subparsers.add_parser(
        "extract", help="Extract ScanNet per-instance statistics"
    )
    extract.add_argument("--dataset-root", required=True)
    extract.add_argument("--scene-list", required=True)
    extract.add_argument("--split", choices=("train", "val"), required=True)
    extract.add_argument("--dataset", default="scannet200")
    extract.add_argument("--taxonomy")
    extract.add_argument("--label-map")
    extract.add_argument("--output", required=True)
    extract.add_argument("--audit-output", required=True)
    extract.add_argument("--voxel-size-m", type=float, default=0.02)
    extract.add_argument("--oversample-factor", type=float, default=4.0)
    extract.add_argument("--workers", type=int, default=1)
    extract.add_argument("--allow-scene-errors", action="store_true")
    extract.set_defaults(func=command_extract)

    prepare_gt = subparsers.add_parser(
        "prepare-gt", help="Build canonical SAGA20 vertex GT"
    )
    prepare_gt.add_argument("--dataset-root", required=True)
    prepare_gt.add_argument("--scene-list", required=True)
    prepare_gt.add_argument("--dataset", default="scannet200")
    prepare_gt.add_argument("--taxonomy")
    prepare_gt.add_argument("--output-dir", required=True)
    prepare_gt.set_defaults(func=command_prepare_gt)

    prepare_saga = subparsers.add_parser(
        "prepare-saga-scene",
        help="Export a metric, axis-aligned ScanNet .sens scene for SAGA/3DGS",
    )
    prepare_saga.add_argument("--dataset-root", required=True)
    prepare_saga.add_argument("--scene-id", required=True)
    prepare_saga.add_argument("--sens", required=True)
    prepare_saga.add_argument("--output-root", required=True)
    prepare_saga.add_argument("--frame-stride", type=int, default=20)
    prepare_saga.add_argument("--max-frames", type=int, default=200)
    prepare_saga.add_argument("--max-initial-points", type=int, default=200_000)
    prepare_saga.set_defaults(func=command_prepare_saga_scene)

    audit_alignment = subparsers.add_parser(
        "audit-saga-alignment",
        help="Gate a prepared or trained SAGA point cloud against canonical GT",
    )
    audit_alignment.add_argument("--preparation-manifest", required=True)
    audit_alignment.add_argument("--gt-npz", required=True)
    audit_alignment.add_argument("--gaussian-ply")
    audit_alignment.add_argument("--output", required=True)
    audit_alignment.add_argument("--radius-m", type=float, default=0.05)
    audit_alignment.add_argument(
        "--minimum-mapped-fraction", type=float, default=0.90
    )
    audit_alignment.add_argument("--camera-padding-m", type=float, default=2.0)
    audit_alignment.add_argument(
        "--minimal",
        action="store_true",
        help="Run all alignment checks without validating or writing file hashes",
    )
    audit_alignment.set_defaults(func=command_audit_saga_alignment)

    fit = subparsers.add_parser("fit", help="Fit train-only hierarchical priors")
    fit.add_argument("--stats", required=True)
    fit.add_argument("--taxonomy")
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=20260804)
    fit.add_argument("--bootstrap-samples", type=int, default=2000)
    fit.add_argument("--min-physical-scenes", type=int, default=5)
    fit.add_argument("--shrink-tau", type=float, default=20.0)
    fit.set_defaults(func=command_fit)

    select = subparsers.add_parser(
        "select-scenes", help="Create grouped 24/48 validation selection"
    )
    select.add_argument("--stats", required=True)
    select.add_argument("--taxonomy")
    select.add_argument("--output", required=True)
    select.add_argument("--tune-budget", type=int, default=24)
    select.add_argument("--locked-budget", type=int, default=48)
    select.add_argument("--tune-target", type=int, default=10)
    select.add_argument("--locked-target", type=int, default=20)
    select.add_argument("--seed", type=int, default=20260804)
    select.set_defaults(func=command_select_scenes)

    select_locked = subparsers.add_parser(
        "select-locked-scenes",
        help="Choose 48 physically independent scans from the frozen locked pool",
    )
    select_locked.add_argument("--stats", required=True)
    select_locked.add_argument("--scene-selection", required=True)
    select_locked.add_argument("--taxonomy")
    select_locked.add_argument("--output", required=True)
    select_locked.add_argument("--locked-budget", type=int, default=48)
    select_locked.add_argument("--locked-target", type=int, default=20)
    select_locked.set_defaults(func=command_select_locked_scenes)

    assess_seeds = subparsers.add_parser(
        "assess-seeds", help="Apply the preregistered tune seed-sensitivity rule"
    )
    assess_seeds.add_argument("--metrics", action="append", required=True)
    assess_seeds.add_argument("--maximum-range", type=float, default=0.002)
    assess_seeds.add_argument("--output", required=True)
    assess_seeds.set_defaults(func=command_assess_seeds)

    locked_plan = subparsers.add_parser(
        "build-locked-plan", help="Freeze the lightweight confirmatory protocol"
    )
    locked_plan.add_argument("--locked-scenes", required=True)
    locked_plan.add_argument("--seed-decision", required=True)
    locked_plan.add_argument("--priors", required=True)
    locked_plan.add_argument("--mapping", required=True)
    locked_plan.add_argument("--taxonomy")
    locked_plan.add_argument("--code-commit")
    locked_plan.add_argument("--seed", type=int, default=20260804)
    locked_plan.add_argument("--output", required=True)
    locked_plan.set_defaults(func=command_build_locked_plan)

    search = subparsers.add_parser(
        "search-design", help="Generate a registered LHS search design"
    )
    search.add_argument("--kind", choices=("global", "prior"), required=True)
    search.add_argument("--samples", type=int, default=32)
    search.add_argument("--seed", type=int, default=20260804)
    search.add_argument("--output", required=True)
    search.set_defaults(func=command_search_design)

    materialize = subparsers.add_parser(
        "materialize-search",
        help="Materialize every LHS row as a hashed executable mapping",
    )
    materialize.add_argument("--design", required=True)
    materialize.add_argument("--priors", required=True)
    materialize.add_argument("--taxonomy", required=True)
    materialize.add_argument("--scene-selection", required=True)
    materialize.add_argument("--base-mapping")
    materialize.add_argument("--output-dir", required=True)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(func=command_materialize_search)

    search_schedule = subparsers.add_parser(
        "search-schedule",
        help="Randomize executable search configurations within scene/seed blocks",
    )
    search_schedule.add_argument("--scene-selection", required=True)
    search_schedule.add_argument("--mapping-manifest", required=True)
    search_schedule.add_argument("--run-seed", action="append", type=int)
    search_schedule.add_argument("--seed", type=int, default=20260804)
    search_schedule.add_argument("--output", required=True)
    search_schedule.set_defaults(func=command_search_schedule)

    choose = subparsers.add_parser(
        "select-config", help="Select the registered best tuning config"
    )
    choose.add_argument("--metrics", required=True)
    choose.add_argument("--design", required=True)
    choose.add_argument(
        "--tie-ap",
        type=float,
        default=0.002,
        help="AP fraction used for the runtime tie-break (0.002 = 0.2 AP points)",
    )
    choose.add_argument("--output", required=True)
    choose.set_defaults(func=command_select_config)

    mapping = subparsers.add_parser(
        "build-mapping", help="Build validation-derived mapping config"
    )
    mapping.add_argument("--global-best", required=True)
    mapping.add_argument(
        "--prior-best",
        help="omit after global search; defaults are retained for the prior search base",
    )
    mapping.add_argument("--priors", required=True)
    mapping.add_argument("--taxonomy", required=True)
    mapping.add_argument("--scene-selection", required=True)
    mapping.add_argument("--output", required=True)
    mapping.set_defaults(func=command_build_mapping)

    lock = subparsers.add_parser(
        "lock", help="Freeze the confirmatory experiment manifest"
    )
    lock.add_argument("--artifact", action="append", required=True)
    lock.add_argument("--code-commit")
    lock.add_argument("--output", required=True)
    lock.set_defaults(func=command_lock)

    schedule = subparsers.add_parser(
        "schedule", help="Randomize condition order within scene/seed blocks"
    )
    schedule.add_argument("--scene-selection", required=True)
    schedule.add_argument("--split", choices=("tune", "locked"), required=True)
    schedule.add_argument("--condition", action="append")
    schedule.add_argument("--run-seed", action="append", type=int)
    schedule.add_argument("--seed", type=int, default=20260804)
    schedule.add_argument("--output", required=True)
    schedule.set_defaults(func=command_schedule)

    run_experiment = subparsers.add_parser(
        "run-experiment", help="Execute or dry-run a registered postprocess schedule"
    )
    run_experiment.add_argument("--schedule", required=True)
    run_experiment.add_argument("--scene-manifest", required=True)
    run_experiment.add_argument("--output-root", required=True)
    run_experiment.add_argument(
        "--output", required=True, help="Execution status manifest"
    )
    run_experiment.add_argument("--pipeline", default="run_pipeline.sh")
    run_experiment.add_argument("--priors")
    run_experiment.add_argument("--mapping")
    run_experiment.add_argument("--dry-run", action="store_true")
    run_experiment.add_argument("--no-resume", action="store_true")
    run_experiment.add_argument("--continue-on-error", action="store_true")
    run_experiment.add_argument("--max-runs", type=int)
    run_experiment.set_defaults(func=command_run_experiment)

    run_locked = subparsers.add_parser(
        "run-locked", help="Execute a frozen locked plan with lightweight resume"
    )
    run_locked.add_argument("--plan", required=True)
    run_locked.add_argument("--scene-manifest", required=True)
    run_locked.add_argument("--output-root", required=True)
    run_locked.add_argument("--progress", required=True)
    run_locked.add_argument("--pipeline", default="run_pipeline.sh")
    run_locked.add_argument("--dry-run", action="store_true")
    run_locked.add_argument("--max-runs", type=int)
    run_locked.set_defaults(func=command_run_locked)

    run_class_first = subparsers.add_parser(
        "run-class-first",
        help="Run lightweight class-first postprocess experiments without a schedule",
    )
    run_class_first.add_argument("--scene-manifest", required=True)
    run_class_first.add_argument("--output-root", required=True)
    run_class_first.add_argument("--category-priors", required=True)
    run_class_first.add_argument("--class-first-config", required=True)
    run_class_first.add_argument("--pipeline", default="run_pipeline.sh")
    run_class_first.add_argument(
        "--condition", action="append", choices=tuple(CLASS_FIRST_CONDITIONS)
    )
    run_class_first.add_argument("--seed", action="append", type=int)
    run_class_first.add_argument("--scene", action="append")
    run_class_first.add_argument("--no-resume", action="store_true")
    run_class_first.add_argument("--continue-on-error", action="store_true")
    run_class_first.add_argument("--dry-run", action="store_true")
    run_class_first.add_argument("--max-runs", type=int)
    run_class_first.set_defaults(func=command_run_class_first)

    class_first_params = subparsers.add_parser(
        "build-class-first-params",
        help="Write the explicit no-hash d/A/b/m/K/rescue table",
    )
    class_first_params.add_argument("--category-priors", required=True)
    class_first_params.add_argument("--class-first-config", required=True)
    class_first_params.add_argument("--output", required=True)
    class_first_params.add_argument(
        "--mode",
        choices=("uniform", "size", "smooth", "small", "combined"),
        default="combined",
    )
    class_first_params.set_defaults(func=command_build_class_first_params)

    evaluate_class_first = subparsers.add_parser(
        "evaluate-class-first",
        help="Evaluate and compare class-first runs without schedules or locks",
    )
    evaluate_class_first.add_argument("--scene-manifest", required=True)
    evaluate_class_first.add_argument("--gt-dir", required=True)
    evaluate_class_first.add_argument("--output-root", required=True)
    evaluate_class_first.add_argument("--taxonomy")
    evaluate_class_first.add_argument("--metrics-output")
    evaluate_class_first.add_argument("--analysis-output")
    evaluate_class_first.add_argument(
        "--condition", action="append", choices=tuple(CLASS_FIRST_CONDITIONS)
    )
    evaluate_class_first.add_argument("--seed", action="append", type=int)
    scene_source = evaluate_class_first.add_mutually_exclusive_group()
    scene_source.add_argument("--scene", action="append")
    scene_source.add_argument("--scene-list")
    scene_source.add_argument("--selection")
    evaluate_class_first.add_argument(
        "--selection-split", choices=("tune", "locked"), default="locked"
    )
    evaluate_class_first.add_argument(
        "--reference", choices=tuple(CLASS_FIRST_CONDITIONS)
    )
    evaluate_class_first.add_argument(
        "--treatment", choices=tuple(CLASS_FIRST_CONDITIONS)
    )
    evaluate_class_first.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate_class_first.add_argument("--bootstrap-seed", type=int, default=20260804)
    evaluate_class_first.add_argument("--radius-m", type=float, default=0.05)
    evaluate_class_first.add_argument(
        "--minimum-mapped-fraction", type=float, default=0.90
    )
    evaluate_class_first.add_argument("--min-region-size", type=int, default=100)
    evaluate_class_first.add_argument("--split", default="class-first")
    evaluate_class_first.set_defaults(func=command_evaluate_class_first)

    run_teacher_prior = subparsers.add_parser(
        "run-teacher-prior",
        help="Run lightweight teacher-style category-prior postprocess experiments",
    )
    run_teacher_prior.add_argument("--scene-manifest", required=True)
    run_teacher_prior.add_argument("--output-root", required=True)
    run_teacher_prior.add_argument("--teacher-category-params")
    run_teacher_prior.add_argument("--pipeline", default="run_pipeline.sh")
    run_teacher_prior.add_argument(
        "--condition", action="append", choices=tuple(TEACHER_PRIOR_CONDITIONS)
    )
    run_teacher_prior.add_argument("--seed", action="append", type=int)
    run_teacher_prior.add_argument("--scene", action="append")
    run_teacher_prior.add_argument("--no-resume", action="store_true")
    run_teacher_prior.add_argument("--continue-on-error", action="store_true")
    run_teacher_prior.add_argument("--dry-run", action="store_true")
    run_teacher_prior.add_argument("--max-runs", type=int)
    run_teacher_prior.set_defaults(func=command_run_teacher_prior)

    build_teacher_params = subparsers.add_parser(
        "build-teacher-category-params",
        help="Materialize one readable train-only parameter table for teacher-prior runs",
    )
    build_teacher_params.add_argument("--category-priors", required=True)
    build_teacher_params.add_argument("--output", required=True)
    build_teacher_params.add_argument("--branch-preservation", action="store_true")
    build_teacher_params.set_defaults(func=command_build_teacher_category_params)

    evaluate_teacher_prior = subparsers.add_parser(
        "evaluate-teacher-prior",
        help="Evaluate teacher-style category-prior runs with the official protocol",
    )
    evaluate_teacher_prior.add_argument("--scene-manifest", required=True)
    evaluate_teacher_prior.add_argument("--gt-dir", required=True)
    evaluate_teacher_prior.add_argument("--output-root", required=True)
    evaluate_teacher_prior.add_argument("--taxonomy")
    evaluate_teacher_prior.add_argument("--metrics-output")
    evaluate_teacher_prior.add_argument("--analysis-output")
    evaluate_teacher_prior.add_argument(
        "--condition", action="append", choices=tuple(TEACHER_PRIOR_CONDITIONS)
    )
    evaluate_teacher_prior.add_argument("--seed", action="append", type=int)
    teacher_scene_source = evaluate_teacher_prior.add_mutually_exclusive_group()
    teacher_scene_source.add_argument("--scene", action="append")
    teacher_scene_source.add_argument("--scene-list")
    teacher_scene_source.add_argument("--selection")
    evaluate_teacher_prior.add_argument(
        "--selection-split", choices=("tune", "locked"), default="tune"
    )
    evaluate_teacher_prior.add_argument(
        "--reference", choices=tuple(TEACHER_PRIOR_CONDITIONS)
    )
    evaluate_teacher_prior.add_argument(
        "--treatment", choices=tuple(TEACHER_PRIOR_CONDITIONS)
    )
    evaluate_teacher_prior.add_argument(
        "--bootstrap-samples", type=int, default=10000
    )
    evaluate_teacher_prior.add_argument(
        "--bootstrap-seed", type=int, default=20260804
    )
    evaluate_teacher_prior.add_argument("--radius-m", type=float, default=0.05)
    evaluate_teacher_prior.add_argument(
        "--minimum-mapped-fraction", type=float, default=0.90
    )
    evaluate_teacher_prior.add_argument("--min-region-size", type=int, default=100)
    evaluate_teacher_prior.add_argument("--split", default="teacher-prior")
    evaluate_teacher_prior.set_defaults(func=command_evaluate_teacher_prior)

    run_prior_v2 = subparsers.add_parser(
        "run-prior-v2",
        help="Run proposal-first legacy category-prior postprocess experiments",
    )
    run_prior_v2.add_argument("--scene-manifest", required=True)
    run_prior_v2.add_argument("--output-root", required=True)
    run_prior_v2.add_argument("--category-priors", required=True)
    run_prior_v2.add_argument("--legacy-prior-config", required=True)
    run_prior_v2.add_argument("--pipeline", default="run_pipeline.sh")
    run_prior_v2.add_argument(
        "--condition", action="append", choices=tuple(LEGACY_PRIOR_CONDITIONS)
    )
    run_prior_v2.add_argument("--seed", action="append", type=int)
    run_prior_v2.add_argument("--scene", action="append")
    run_prior_v2.add_argument(
        "--score", choices=("unit", "vote", "assignment"), default="unit"
    )
    run_prior_v2.add_argument(
        "--semantic-source", choices=("gaussian", "vote"), default="gaussian"
    )
    run_prior_v2.add_argument("--no-resume", action="store_true")
    run_prior_v2.add_argument("--continue-on-error", action="store_true")
    run_prior_v2.add_argument("--dry-run", action="store_true")
    run_prior_v2.add_argument("--max-runs", type=int)
    run_prior_v2.set_defaults(func=command_run_prior_v2)

    prior_v2_params = subparsers.add_parser(
        "build-prior-v2-params", help="Write the no-hash proposal-first prior table"
    )
    prior_v2_params.add_argument("--category-priors", required=True)
    prior_v2_params.add_argument("--legacy-prior-config", required=True)
    prior_v2_params.add_argument("--output", required=True)
    prior_v2_params.set_defaults(func=command_build_prior_v2_params)

    evaluate_prior_v2 = subparsers.add_parser(
        "evaluate-prior-v2", help="Evaluate proposal-first prior-v2 runs"
    )
    evaluate_prior_v2.add_argument("--scene-manifest", required=True)
    evaluate_prior_v2.add_argument("--gt-dir", required=True)
    evaluate_prior_v2.add_argument("--output-root", required=True)
    evaluate_prior_v2.add_argument("--taxonomy")
    evaluate_prior_v2.add_argument("--metrics-output")
    evaluate_prior_v2.add_argument("--analysis-output")
    evaluate_prior_v2.add_argument(
        "--condition", action="append", choices=tuple(LEGACY_PRIOR_CONDITIONS)
    )
    evaluate_prior_v2.add_argument("--seed", action="append", type=int)
    prior_v2_scenes = evaluate_prior_v2.add_mutually_exclusive_group()
    prior_v2_scenes.add_argument("--scene", action="append")
    prior_v2_scenes.add_argument("--scene-list")
    prior_v2_scenes.add_argument("--selection")
    evaluate_prior_v2.add_argument(
        "--selection-split", choices=("tune", "locked"), default="tune"
    )
    evaluate_prior_v2.add_argument(
        "--reference", choices=tuple(LEGACY_PRIOR_CONDITIONS), default="L1-uniform"
    )
    evaluate_prior_v2.add_argument(
        "--treatment", choices=tuple(LEGACY_PRIOR_CONDITIONS)
    )
    evaluate_prior_v2.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate_prior_v2.add_argument("--bootstrap-seed", type=int, default=20260804)
    evaluate_prior_v2.add_argument("--radius-m", type=float, default=0.05)
    evaluate_prior_v2.add_argument(
        "--minimum-mapped-fraction", type=float, default=0.90
    )
    evaluate_prior_v2.add_argument("--min-region-size", type=int, default=100)
    evaluate_prior_v2.add_argument("--split", default="prior-v2")
    evaluate_prior_v2.set_defaults(func=command_evaluate_prior_v2)

    diagnose_backbone = subparsers.add_parser(
        "diagnose-backbone", help="Measure coverage, semantic errors and proposal fragmentation"
    )
    diagnose_backbone.add_argument("--scene-manifest", required=True)
    diagnose_backbone.add_argument("--gt-dir", required=True)
    diagnose_backbone.add_argument("--output-root", required=True)
    diagnose_backbone.add_argument("--taxonomy")
    diagnose_backbone.add_argument("--condition", action="append", required=True)
    diagnose_backbone.add_argument("--seed", action="append", type=int, required=True)
    diagnose_backbone.add_argument("--scene", action="append", required=True)
    diagnose_backbone.add_argument("--output", required=True)
    diagnose_backbone.add_argument("--table", required=True)
    diagnose_backbone.add_argument("--radius-m", type=float, default=0.05)
    diagnose_backbone.add_argument("--min-region-size", type=int, default=100)
    diagnose_backbone.set_defaults(func=command_diagnose_backbone)

    evaluate_seed_audit = subparsers.add_parser(
        "evaluate-seed-audit",
        help="Evaluate the two-condition val-tune seed sensitivity run",
    )
    evaluate_seed_audit.add_argument("--schedule", required=True)
    evaluate_seed_audit.add_argument("--execution", required=True)
    evaluate_seed_audit.add_argument("--scene-manifest", required=True)
    evaluate_seed_audit.add_argument("--gt-dir", required=True)
    evaluate_seed_audit.add_argument("--taxonomy")
    evaluate_seed_audit.add_argument("--config-id")
    evaluate_seed_audit.add_argument("--output", required=True)
    evaluate_seed_audit.add_argument("--radius-m", type=float, default=0.05)
    evaluate_seed_audit.add_argument("--min-region-size", type=int, default=100)
    evaluate_seed_audit.set_defaults(func=command_evaluate_seed_audit)

    evaluate_locked = subparsers.add_parser(
        "evaluate-locked",
        help="Run the frozen confirmatory metrics and statistical analysis",
    )
    evaluate_locked.add_argument("--plan", required=True)
    evaluate_locked.add_argument("--scene-manifest", required=True)
    evaluate_locked.add_argument("--gt-dir", required=True)
    evaluate_locked.add_argument("--output-root", required=True)
    evaluate_locked.add_argument("--taxonomy")
    evaluate_locked.add_argument("--metrics-output", required=True)
    evaluate_locked.add_argument("--analysis-output", required=True)
    evaluate_locked.set_defaults(func=command_evaluate_locked)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate SAGA outputs on canonical vertex GT"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--taxonomy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--radius-m", type=float, default=0.05)
    evaluate.add_argument("--min-region-size", type=int, default=100)
    evaluate.set_defaults(func=command_evaluate)

    evaluate_search = subparsers.add_parser(
        "evaluate-search",
        help="Evaluate all complete val-tune configurations and write metrics",
    )
    evaluate_search.add_argument("--schedule", required=True)
    evaluate_search.add_argument("--execution", required=True)
    evaluate_search.add_argument("--scene-manifest", required=True)
    evaluate_search.add_argument("--gt-manifest", required=True)
    evaluate_search.add_argument("--taxonomy")
    evaluate_search.add_argument("--output-dir", required=True)
    evaluate_search.add_argument("--output", required=True)
    evaluate_search.add_argument("--radius-m", type=float, default=0.05)
    evaluate_search.add_argument("--min-region-size", type=int, default=100)
    evaluate_search.set_defaults(func=command_evaluate_search)

    analyze = subparsers.add_parser(
        "analyze", help="Run locked paired and 2^3 scene-group bootstrap analyses"
    )
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--taxonomy")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260804)
    analyze.add_argument("--radius-m", type=float, default=0.05)
    analyze.add_argument("--min-region-size", type=int, default=100)
    analyze.set_defaults(func=command_analyze)

    validate = subparsers.add_parser(
        "validate", help="Validate prior/mapping provenance and hashes"
    )
    validate.add_argument("--priors", required=True)
    validate.add_argument("--mapping", required=True)
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers must be at least 1")
    if getattr(args, "bootstrap_samples", 1) < 1:
        parser.error("--bootstrap-samples must be at least 1")
    args.func(args)


if __name__ == "__main__":
    main()
