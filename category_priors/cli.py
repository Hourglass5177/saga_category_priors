from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_manifest
from .download import (
    MINIMAL_FILE_TYPES,
    download_scannet_saga_scenes,
    download_scannet_subset,
)
from .evaluator import evaluate_manifest
from .io import hash_json, load_json, read_rows, sha256_file, write_json, write_rows
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
from .runner import execute_schedule
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
from .selection import select_scenes
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
