from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import analyze_manifest
from .evaluator import evaluate_manifest
from .io import hash_json, load_json, read_rows, sha256_file, write_json, write_rows
from .mapping import (
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
from .selection import select_scenes
from .taxonomy import load_taxonomy


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
    prior_best = load_json(args.prior_best)
    payload = build_mapping_config(
        global_best["parameters"],
        prior_best["parameters"],
        args.priors,
        args.taxonomy,
        args.scene_selection,
    )
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

    choose = subparsers.add_parser(
        "select-config", help="Select the registered best tuning config"
    )
    choose.add_argument("--metrics", required=True)
    choose.add_argument("--design", required=True)
    choose.add_argument("--tie-ap", type=float, default=0.2)
    choose.add_argument("--output", required=True)
    choose.set_defaults(func=command_select_config)

    mapping = subparsers.add_parser(
        "build-mapping", help="Build validation-derived mapping config"
    )
    mapping.add_argument("--global-best", required=True)
    mapping.add_argument("--prior-best", required=True)
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
