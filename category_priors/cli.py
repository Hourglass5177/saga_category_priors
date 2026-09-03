from __future__ import annotations

"""Small public CLI for reusable category-prior utilities.

Experiment-specific runners deliberately live outside this entry point. Git
history preserves their old command lines; the active CLI exposes only data
preparation, prior fitting, evaluation, and read-only audits.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .alignment import audit_saga_alignment
from .evaluator import evaluate_manifest
from .gaussian_object_audit import audit_gaussian_object_runs
from .io import hash_json, read_rows, sha256_file, write_json
from .priors import fit_priors, write_priors
from .scannet import (
    discover_scene_files,
    prepare_scene_ground_truth,
    read_scene_ids,
    validate_scene_ids,
)
from .taxonomy import load_taxonomy


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _fit(args: argparse.Namespace) -> None:
    payload = fit_priors(
        read_rows(args.stats),
        load_taxonomy(args.taxonomy),
        args.stats,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        min_physical_scenes=args.min_physical_scenes,
        shrink_tau=args.shrink_tau,
    )
    write_priors(args.output, payload)


def _evaluate(args: argparse.Namespace) -> None:
    evaluate_manifest(
        args.manifest,
        load_taxonomy(args.taxonomy),
        args.output,
        args.radius_m,
        args.min_region_size,
    )


def _write_ground_truth_npz(
    target: Path,
    coords: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
) -> None:
    temporary = target.with_suffix(target.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            coords=np.asarray(coords),
            semantic=np.asarray(semantic),
            instance=np.asarray(instance),
        )
    temporary.replace(target)


def _prepare_gt(args: argparse.Namespace) -> None:
    taxonomy = load_taxonomy(args.taxonomy)
    scene_ids = validate_scene_ids(read_scene_ids(args.scene_list))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "kind": "canonical_ground_truth",
        "dataset": args.dataset,
        "taxonomy_sha256": taxonomy.content_hash,
        "scenes": [],
    }
    scenes = manifest["scenes"]
    assert isinstance(scenes, list)
    for scene_id in scene_ids:
        files = discover_scene_files(args.dataset_root, scene_id)
        coords, semantic, instance = prepare_scene_ground_truth(
            files, taxonomy, args.dataset
        )
        target = output_dir / f"{scene_id}.npz"
        _write_ground_truth_npz(target, coords, semantic, instance)
        scenes.append(
            {
                "scene_id": scene_id,
                "path": target.name,
                "sha256": sha256_file(target),
                "vertices": int(len(coords)),
                "mapped_vertices": int(np.count_nonzero(semantic >= 0)),
            }
        )
    manifest["content_sha256"] = hash_json(manifest)
    write_json(output_dir / "manifest.json", manifest)


def _audit_alignment(args: argparse.Namespace) -> None:
    payload = audit_saga_alignment(
        args.preparation_manifest,
        args.gt_npz,
        args.output,
        gaussian_ply_path=args.gaussian_ply,
        radius_m=args.radius_m,
        minimum_mapped_fraction=args.minimum_mapped_fraction,
        camera_padding_m=args.camera_padding_m,
        minimal=args.minimal,
    )
    if payload is not None:
        _print_json(payload)


def _audit_gaussian_objects(args: argparse.Namespace) -> None:
    payload = audit_gaussian_object_runs(
        scene_manifest=args.scene_manifest,
        gt_dir=args.gt_dir,
        runs_root=args.runs_root,
        taxonomy=load_taxonomy(args.taxonomy),
        scene_ids=args.scene,
        conditions=args.condition,
        seed=args.seed,
        table_output=args.table_output,
        audit_output=args.audit_output,
        comparison_output=args.comparison_output,
        viewer_output=args.viewer_output,
        radius_m=args.radius_m,
        min_region_size=args.min_region_size,
    )
    _print_json(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAGA reusable evaluation utilities")
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit frozen train-only category priors")
    fit.add_argument("--stats", required=True)
    fit.add_argument("--taxonomy")
    fit.add_argument("--output", required=True)
    fit.add_argument("--seed", type=int, default=20260804)
    fit.add_argument("--bootstrap-samples", type=int, default=2000)
    fit.add_argument("--min-physical-scenes", type=int, default=5)
    fit.add_argument("--shrink-tau", type=float, default=20.0)
    fit.set_defaults(func=_fit)

    evaluate = commands.add_parser(
        "evaluate", help="run the ScanNet official nine-threshold evaluator"
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--taxonomy")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--radius-m", type=float, default=0.05)
    evaluate.add_argument("--min-region-size", type=int, default=100)
    evaluate.set_defaults(func=_evaluate)

    prepare_gt = commands.add_parser(
        "prepare-gt", help="build canonical SAGA20 vertex ground truth"
    )
    prepare_gt.add_argument("--dataset-root", required=True)
    prepare_gt.add_argument("--scene-list", required=True)
    prepare_gt.add_argument("--dataset", default="scannet200")
    prepare_gt.add_argument("--taxonomy")
    prepare_gt.add_argument("--output-dir", required=True)
    prepare_gt.set_defaults(func=_prepare_gt)

    alignment = commands.add_parser(
        "audit-saga-alignment", help="audit metric Gaussian/GT alignment"
    )
    alignment.add_argument("--preparation-manifest", required=True)
    alignment.add_argument("--gt-npz", required=True)
    alignment.add_argument("--gaussian-ply")
    alignment.add_argument("--output", required=True)
    alignment.add_argument("--radius-m", type=float, default=0.05)
    alignment.add_argument("--minimum-mapped-fraction", type=float, default=0.90)
    alignment.add_argument("--camera-padding-m", type=float, default=2.0)
    alignment.add_argument("--minimal", action="store_true")
    alignment.set_defaults(func=_audit_alignment)

    audit = commands.add_parser(
        "audit-gaussian-objects", help="run the precision-first Gaussian audit"
    )
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
    audit.set_defaults(func=_audit_gaussian_objects)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
