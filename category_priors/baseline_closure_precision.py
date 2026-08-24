from __future__ import annotations

"""Precision-first Gaussian audit for all completed baseline-closeout arms."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_closure_analysis import (
    _gaussian_ply,
    _output_runs,
    _runtime_rows,
    _transform,
)
from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .gaussian_object_audit import (
    _aggregate_condition_rows,
    _export_viewer_case,
    _select_viewer_cases,
    evaluate_gaussian_object_precision,
)
from .io import load_json, write_json, write_rows
from .instance_projection import project_declared_instances
from .taxonomy import Taxonomy, load_taxonomy

AUDIT_RADII_M = (0.02, 0.05, 0.10)
VIEWER_RADIUS_M = 0.05


def evaluate_teacher_handoff_precision(
    *,
    closure_root: Path,
    gt_dir: Path,
    runtime_manifest: Path,
    output_dir: Path,
    taxonomy: Taxonomy | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy()
    output_runs = list(_output_runs(closure_root))
    if not output_runs:
        raise FileNotFoundError(f"No baseline output.json found under {closure_root}")
    runtime = _runtime_rows(runtime_manifest)

    scene_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    viewer_rows: list[dict[str, Any]] = []
    viewer_audits: dict[tuple[str, str], dict[str, Any]] = {}
    point_labels: dict[tuple[str, str], np.ndarray] = {}
    condition_rows: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    projection_runs: list[dict[str, Any]] = []

    for variant_id, budget, condition, scene_id, output_json in output_runs:
        if scene_id not in runtime:
            raise KeyError(f"runtime manifest is missing {scene_id}")
        if scene_id not in scene_arrays:
            gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
            gaussian_xyz = apply_transform(
                load_ply_xyz(_gaussian_ply(runtime[scene_id])),
                _transform(runtime[scene_id]),
            )
            scene_arrays[scene_id] = (
                gt_xyz,
                gt.semantic,
                gt.instance,
                gaussian_xyz,
            )
        gt_xyz, gt_semantic, gt_instance, gaussian_xyz = scene_arrays[scene_id]
        output = load_json(output_json)
        output_instances = output.get("instances", {})
        projection = project_declared_instances(
            output["point_labels"], output_instances
        )
        labels = projection.point_labels
        declared_ids = set(projection.declared_instance_ids)
        declared_instances = {
            raw_id: metadata
            for raw_id, metadata in output_instances.items()
            if int(raw_id) in declared_ids
        }
        if labels.shape != (len(gaussian_xyz),):
            raise ValueError(
                f"{output_json}: point_labels length differs from Gaussian PLY"
            )
        label = f"{variant_id}/{budget}/{condition}"
        projection_runs.append(
            {
                "variant_id": variant_id,
                "budget": budget,
                "condition": condition,
                "condition_label": label,
                "scene_id": scene_id,
                "output_json": str(output_json),
                **projection.stats(),
            }
        )
        for radius_m in AUDIT_RADII_M:
            audit = evaluate_gaussian_object_precision(
                gaussian_xyz,
                labels,
                declared_instances,
                gt_xyz,
                gt_semantic,
                gt_instance,
                radius_m,
                canonical_classes=taxonomy.canonical_classes,
            )
            current_rows: list[dict[str, Any]] = []
            for instance in audit["instances"]:
                row = {
                    "variant_id": variant_id,
                    "budget": budget,
                    "condition": condition,
                    "condition_label": label,
                    "scene_id": scene_id,
                    "radius_m": radius_m,
                    **instance,
                }
                rows.append(row)
                current_rows.append(row)
                if radius_m == VIEWER_RADIUS_M:
                    viewer_rows.append({**row, "condition": label})
            condition_rows.setdefault(
                (variant_id, budget, condition, radius_m), []
            ).extend(current_rows)
            if radius_m == VIEWER_RADIUS_M:
                viewer_audits[(scene_id, label)] = {
                    **audit,
                    "point_labels": labels,
                }
                point_labels[(scene_id, label)] = labels

    summaries: list[dict[str, Any]] = []
    for (variant_id, budget, condition, radius_m), current in sorted(
        condition_rows.items()
    ):
        summaries.append(
            {
                "variant_id": variant_id,
                "budget": budget,
                "condition": condition,
                "radius_m": radius_m,
                **_aggregate_condition_rows(current),
            }
        )

    selected = _select_viewer_cases(viewer_rows)
    viewer_root = output_dir / "viewer"
    viewer_cases: list[dict[str, Any]] = []
    for case in selected:
        scene_id = str(case["scene_id"])
        label = str(case["condition"])
        gt_xyz, gt_semantic, gt_instance, gaussian_xyz = scene_arrays[scene_id]
        viewer_cases.append(
            _export_viewer_case(
                case,
                viewer_audits[(scene_id, label)],
                gt_xyz,
                gt_semantic,
                gt_instance,
                gaussian_xyz,
                point_labels[(scene_id, label)],
                viewer_root,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "teacher_handoff_gaussian_precision.parquet"
    audit_path = output_dir / "teacher_handoff_gaussian_audit.json"
    write_rows(table_path, rows)
    payload = {
        "schema": "saga-teacher-handoff-gaussian-audit-v1",
        "direction": "predicted Gaussian to nearest ScanNet GT point",
        "radii_m": list(AUDIT_RADII_M),
        "unsupported_gaussians_count_as_false_positive": True,
        "orphan_gaussians_count_as_predictions": False,
        "official_ap_unchanged": True,
        "two_dimensional_metrics": False,
        "declared_instance_projection": {
            "semantics": (
                "non-negative point labels absent from instances metadata are "
                "reported and projected to background without changing output.json"
            ),
            "orphan_gaussians_count_as_predictions": False,
            "runs": projection_runs,
        },
        "summaries": summaries,
        "viewer": {
            "qualitative_only": True,
            "selection_radius_m": VIEWER_RADIUS_M,
            "cases": viewer_cases,
            "colors": {
                "green": "same class and same GT instance",
                "yellow": "same class, wrong GT instance",
                "red": "wrong class",
                "gray": "no GT support within 5 cm",
                "blue": "matched GT instance points",
            },
        },
        "table": table_path.name,
    }
    write_json(audit_path, payload)
    write_json(viewer_root / "viewer_case_selection.json", payload["viewer"])
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_teacher_handoff_precision(
        closure_root=args.closure_root,
        gt_dir=args.gt_dir,
        runtime_manifest=args.runtime_manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
