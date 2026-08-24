from __future__ import annotations

"""Evaluate the frozen teacher-handoff baseline without changing its outputs.

The controller deliberately treats the handoff's unscored ``output.json`` as a
set of fixed instance masks.  ``unit`` is therefore the primary AP adapter.
Optional final-vote ratios may be supplied as read-only sidecar data; GT oracle
ranking is always emitted only as a diagnostic sensitivity result.
"""

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_closure import (
    BFC18_CLASSES,
    CLOSURE_SCENES,
    REGISTERED_RUNS,
    asset_paths,
    feature_paths,
    load_runtime_scenes,
    output_paths,
)
from .baseline_closure_evaluation import evaluate_baseline_closure
from .evaluator import (
    GroundTruthScene,
    PredictedInstance,
    load_ground_truth_npz,
    saga_scene_predictions,
)
from .io import load_json, write_json, write_rows
from .instance_projection import project_declared_instances
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy, load_taxonomy

PRIMARY_SCORE_MODE = "unit"
FINAL_VOTE_FIELDS = ("final_vote_ratio", "vote_ratio", "semantic_confidence")


def bfc18_saga20_intersection(class_names: Sequence[str]) -> tuple[str, ...]:
    """Return the registered bfc18 ∩ SAGA20 view in canonical SAGA20 order."""

    bfc18 = {str(name).strip().lower() for name in BFC18_CLASSES}
    return tuple(
        str(name).strip().lower()
        for name in class_names
        if str(name).strip().lower() in bfc18
    )


def _runtime_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Load either the project scene-runtime format or the closeout format."""

    payload = load_json(path)
    if isinstance(payload, Mapping) and payload.get("kind") == "scene_runtime_manifest":
        return load_scene_runtime_manifest(path)
    closeout = load_runtime_scenes(path)
    return {
        scene_id: {
            "scene_id": scene_id,
            "base_path": str(scene.base_path),
            "gaussian_ply": str(scene.point_cloud_path),
            "gaussian_to_gt_transform": np.eye(4).tolist(),
        }
        for scene_id, scene in closeout.items()
    }


def _gaussian_ply(scene: Mapping[str, Any]) -> Path:
    raw = scene.get("gaussian_ply") or scene.get("point_cloud_path")
    base = Path(str(scene["base_path"]))
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else (base / path)
    root = base / "output_models" / "point_cloud" / "iteration_30000"
    teacher = root / "scene_point_cloud.ply"
    return teacher if teacher.is_file() else root / "point_cloud.ply"


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get("gaussian_to_gt_transform", np.eye(4).tolist())


def _read_final_vote_sidecar(path: Path | None) -> dict[tuple[str, int], float]:
    """Read ``{scene_id: {instance_id: score}}`` or a list of score rows."""

    if path is None:
        return {}
    payload = load_json(path)
    scores: dict[tuple[str, int], float] = {}
    if isinstance(payload, Mapping) and isinstance(payload.get("scores"), Sequence):
        payload = payload["scores"]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for row in payload:
            if not isinstance(row, Mapping):
                raise TypeError("final-vote score rows must be objects")
            instance_id = int(row["instance_id"])
            if instance_id >= 0:
                scores[(str(row["scene_id"]), instance_id)] = float(row["score"])
    elif isinstance(payload, Mapping):
        for scene_id, values in payload.items():
            if not isinstance(values, Mapping):
                raise TypeError("final-vote score mapping must be nested by scene")
            for instance_id, score in values.items():
                instance_id = int(instance_id)
                if instance_id >= 0:
                    scores[(str(scene_id), instance_id)] = float(score)
    else:
        raise TypeError("final-vote score sidecar must be a mapping or row list")
    for key, value in scores.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key}: invalid final-vote score {value}")
    return scores


def _scores_embedded_in_output(
    scene_id: str, output_json: Path
) -> dict[tuple[str, int], float]:
    """Read an explicitly persisted vote ratio, never manufacture one."""

    payload = load_json(output_json)
    instances = payload.get("instances", {})
    if not isinstance(instances, Mapping):
        raise TypeError(f"{output_json}: instances must be an object")
    result: dict[tuple[str, int], float] = {}
    for raw_id, values in instances.items():
        if not isinstance(values, Mapping):
            continue
        instance_id = int(raw_id)
        if instance_id < 0:
            continue
        for field in FINAL_VOTE_FIELDS:
            if field in values:
                result[(scene_id, instance_id)] = float(values[field])
                break
    return result


def _output_runs(closure_root: Path) -> Iterable[tuple[str, str, str, str, Path]]:
    """Yield completed historical, fixed-contributor and causal outputs."""

    for spec in REGISTERED_RUNS:
        for condition in spec.conditions:
            for scene_id in spec.scene_ids:
                path = output_paths(
                    closure_root, scene_id, spec.variant_id, spec.budget, condition
                ).output_json
                if path.is_file():
                    yield spec.variant_id, spec.budget, condition, scene_id, path
    structural = (
        (
            "full950-contributor-fixed",
            "full950-contributor-fixed",
            "B0-global",
            "B0-global",
        ),
        (
            "full950-contributor-fixed",
            "full950-contributor-fixed",
            "B1-original",
            "B1-original",
        ),
        ("current-L0", "current-causal-harness", "B0-global", "L0-B0-global"),
        ("current-L0", "current-causal-harness", "B1-original", "L0-B1-original"),
        ("current-L1", "current-causal-harness", "B1-original", "L1-B1-original"),
        ("current-L2", "current-causal-harness", "B1-original", "L2-B1-original"),
        ("current-L3", "current-causal-harness", "B1-original", "L3-B1-original"),
    )
    for reported_variant, path_variant, condition, path_condition in structural:
        for scene_id in CLOSURE_SCENES:
            path = output_paths(
                closure_root,
                scene_id,
                path_variant,
                "adaptive",
                path_condition,
            ).output_json
            if path.is_file():
                yield reported_variant, "adaptive", condition, scene_id, path
    for budget in ("adaptive-iterations-cli", "10000"):
        for condition in ("B0-global", "B1-original"):
            path = output_paths(
                closure_root,
                "scene0064_01",
                "full950-contributor-fixed",
                budget,
                condition,
            ).output_json
            if path.is_file():
                yield (
                    "full950-contributor-fixed",
                    budget,
                    condition,
                    "scene0064_01",
                    path,
                )


def _metric_rows(
    result: Mapping[str, Any],
    *,
    variant_id: str,
    budget: str,
    condition: str,
    scene_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    adapter = result["score_adapter"]
    for protocol_key, protocol in result["protocols"].items():
        for view_key in ("full_saga20", "predictable_intersection"):
            evaluation = protocol[view_key]
            aggregate = evaluation["aggregate"]
            primary_metric = str(evaluation["primary_metric"])
            rows.append(
                {
                    "record_type": "condition",
                    "variant_id": variant_id,
                    "budget": budget,
                    "condition": condition,
                    "score_mode": adapter["mode"],
                    "diagnostic_only": bool(adapter["diagnostic_only"]),
                    "protocol_key": protocol_key,
                    "protocol": evaluation["protocol"],
                    "protocol_version": evaluation["protocol_version"],
                    "class_view": (
                        "bfc18_intersect_saga20_common10"
                        if view_key == "predictable_intersection"
                        else view_key
                    ),
                    "class_count": len(result["class_views"][view_key]),
                    "scene_count": len(scene_ids),
                    "scene_ids": "|".join(scene_ids),
                    "primary_metric": primary_metric,
                    "primary_score": aggregate[primary_metric],
                    "map_0.25": aggregate["map_0.25"],
                }
            )
    return rows


def _b1_minus_b0(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pair condition rows by variant/budget/protocol/view/score adapter."""

    keyed = {
        (
            row["variant_id"],
            row["budget"],
            row["score_mode"],
            row["protocol_key"],
            row["class_view"],
            row["scene_ids"],
        ): row
        for row in rows
        if row["record_type"] == "condition" and row["condition"] == "B0-global"
    }
    deltas: list[dict[str, Any]] = []
    for row in rows:
        if row["record_type"] != "condition" or row["condition"] != "B1-original":
            continue
        key = (
            row["variant_id"],
            row["budget"],
            row["score_mode"],
            row["protocol_key"],
            row["class_view"],
            row["scene_ids"],
        )
        baseline = keyed.get(key)
        if baseline is None:
            continue
        delta = dict(row)
        delta.update(
            {
                "record_type": "b1_minus_b0",
                "condition": "B1-original_minus_B0-global",
                "reference_condition": "B0-global",
                "treatment_condition": "B1-original",
                "primary_score": _subtract(
                    row["primary_score"], baseline["primary_score"]
                ),
                "map_0.25": _subtract(row["map_0.25"], baseline["map_0.25"]),
            }
        )
        deltas.append(delta)
    return deltas


def _subtract(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _json_safe_metrics(metrics: Mapping[str, float]) -> dict[str, float | None]:
    """Keep unmapped-distance diagnostics representable in strict JSON."""

    return {
        str(key): float(value) if math.isfinite(float(value)) else None
        for key, value in metrics.items()
    }


def evaluate_teacher_handoff(
    *,
    closure_root: Path,
    gt_dir: Path,
    runtime_manifest: Path,
    output_dir: Path,
    taxonomy: Taxonomy | None = None,
    min_region_size: int = 100,
    radius_m: float = 0.05,
    final_vote_scores_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every available registered B0/B1 output and write handoff artifacts."""

    if min_region_size <= 0:
        raise ValueError("min_region_size must be positive")
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    taxonomy = taxonomy or load_taxonomy()
    class_names = taxonomy.canonical_classes
    common10 = bfc18_saga20_intersection(class_names)
    if len(common10) != 10:
        raise ValueError("registered bfc18 ∩ SAGA20 view must contain common10")
    runtime = _runtime_rows(runtime_manifest)
    sidecar_scores = _read_final_vote_sidecar(final_vote_scores_path)
    grouped: dict[tuple[str, str, str], list[tuple[str, Path]]] = {}
    for variant_id, budget, condition, scene_id, output_json in _output_runs(
        closure_root
    ):
        grouped.setdefault((variant_id, budget, condition), []).append(
            (scene_id, output_json)
        )
    if not grouped:
        raise FileNotFoundError(
            f"No registered closure output.json found under {closure_root}"
        )

    all_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    projection_runs: list[dict[str, Any]] = []
    for (variant_id, budget, condition), scene_outputs in sorted(grouped.items()):
        ground_truth: list[GroundTruthScene] = []
        predictions: list[PredictedInstance] = []
        final_vote_scores: dict[tuple[str, int], float] = {}
        alignment: dict[str, dict[str, float | None]] = {}
        for scene_id, output_json in sorted(scene_outputs):
            if scene_id not in runtime:
                raise KeyError(f"runtime manifest is missing {scene_id}")
            raw_output = load_json(output_json)
            projection = project_declared_instances(
                raw_output["point_labels"], raw_output.get("instances", {})
            )
            projection_runs.append(
                {
                    "variant_id": variant_id,
                    "budget": budget,
                    "condition": condition,
                    "scene_id": scene_id,
                    "output_json": str(output_json),
                    **projection.stats(),
                }
            )
            gt_coords, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
            scene_predictions, diagnostics = saga_scene_predictions(
                scene_id=scene_id,
                gt_coords=gt_coords,
                output_json=output_json,
                gaussian_ply=_gaussian_ply(runtime[scene_id]),
                taxonomy=taxonomy,
                metadata_json=None,
                transform=_transform(runtime[scene_id]),
                radius_m=radius_m,
                require_scores=False,
            )
            ground_truth.append(gt)
            predictions.extend(scene_predictions)
            alignment[scene_id] = _json_safe_metrics(diagnostics)
            final_vote_scores.update(_scores_embedded_in_output(scene_id, output_json))
        final_vote_scores.update(sidecar_scores)
        expected_keys = {(item.scene_id, int(item.instance_id)) for item in predictions}
        modes = [PRIMARY_SCORE_MODE, "gt_oracle"]
        if expected_keys <= final_vote_scores.keys():
            modes.insert(1, "final_vote")
        else:
            unavailable.append(
                {
                    "variant_id": variant_id,
                    "budget": budget,
                    "condition": condition,
                    "score_mode": "final_vote",
                    "reason": "missing_final_vote_ratio",
                    "missing_prediction_count": len(
                        expected_keys - final_vote_scores.keys()
                    ),
                }
            )
        for mode in modes:
            result = evaluate_baseline_closure(
                ground_truth,
                predictions,
                class_names,
                predictable_classes=common10,
                score_mode=mode,  # type: ignore[arg-type]
                final_vote_scores=final_vote_scores if mode == "final_vote" else None,
                min_region_size=min_region_size,
            )
            evaluations.append(
                {
                    "variant_id": variant_id,
                    "budget": budget,
                    "condition": condition,
                    "scene_ids": [scene_id for scene_id, _ in sorted(scene_outputs)],
                    "alignment": alignment,
                    "evaluation": result,
                }
            )
            all_rows.extend(
                _metric_rows(
                    result,
                    variant_id=variant_id,
                    budget=budget,
                    condition=condition,
                    scene_ids=[scene_id for scene_id, _ in sorted(scene_outputs)],
                )
            )
    all_rows.extend(_b1_minus_b0(all_rows))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "teacher_handoff_metrics.parquet"
    analysis_path = output_dir / "teacher_handoff_analysis.json"
    write_rows(metrics_path, all_rows)
    run_summary_path = closure_root / "run_summary.json"
    run_summary = load_json(run_summary_path) if run_summary_path.is_file() else None
    provenance = {
        "schema": "saga-teacher-handoff-provenance-v1",
        "teacher_handoff_anchor": "bfc21922384cc991a71b5e51429354b5d6b06375",
        "full950_repair_anchor": "95073c640a77984c6af24abb276147e4315abcd1",
        "public_upstream_anchor": "96e5021",
        "taxonomy": "bfc18",
        "scenes": list(CLOSURE_SCENES),
        "run_summary": run_summary,
        "no_artifact_hashes": True,
    }
    write_json(output_dir / "teacher_handoff_provenance.json", provenance)
    asset_rows: list[dict[str, Any]] = []
    for scene_id in CLOSURE_SCENES:
        assets = asset_paths(closure_root, scene_id, "bfc18")
        masks_summary = (
            load_json(assets.masks_summary) if assets.masks_summary.is_file() else None
        )
        feature_rows: list[dict[str, Any]] = []
        for spec in REGISTERED_RUNS:
            if scene_id not in spec.scene_ids:
                continue
            paths = feature_paths(
                closure_root,
                scene_id,
                spec.variant_id,
                spec.budget,
                "bfc18",
            )
            feature_rows.append(
                {
                    "variant_id": spec.variant_id,
                    "budget": spec.budget,
                    "point_cloud": str(paths.point_cloud),
                    "point_cloud_present": paths.point_cloud.is_file(),
                    "scale_gate": str(paths.scale_gate),
                    "scale_gate_present": paths.scale_gate.is_file(),
                }
            )
        asset_rows.append(
            {
                "scene_id": scene_id,
                "asset_root": str(assets.root),
                "masks_summary": masks_summary,
                "features": feature_rows,
            }
        )
    write_json(
        output_dir / "teacher_handoff_asset_audit.json",
        {
            "schema": "saga-teacher-handoff-asset-audit-v1",
            "taxonomy": "bfc18",
            "isolated_from_existing_32_class_assets": True,
            "scenes": asset_rows,
        },
    )
    analysis = {
        "schema_version": "1.0",
        "kind": "teacher_handoff_baseline_closure_analysis",
        "primary_score_adapter": {
            "mode": PRIMARY_SCORE_MODE,
            "diagnostic_only": False,
            "description": "unit score is the primary adapter because handoff output.json has no native instance confidence",
        },
        "sensitivity_score_adapters": [
            {
                "mode": "final_vote",
                "diagnostic_only": False,
                "description": "read-only final-vote-ratio adapter; not a native handoff confidence",
            },
            {
                "mode": "gt_oracle",
                "label": "rank-oracle",
                "diagnostic_only": True,
                "description": "GT oracle ranking reads evaluation ground truth and is diagnostic_only",
            },
        ],
        "class_views": {
            "full_saga20": list(class_names),
            "bfc18_intersect_saga20_common10": list(common10),
        },
        "min_region_size": min_region_size,
        "radius_m": radius_m,
        "metrics_path": metrics_path.name,
        "declared_instance_projection": {
            "semantics": (
                "non-negative point labels absent from instances metadata are "
                "reported and projected to background without changing output.json"
            ),
            "orphan_gaussians_count_as_predictions": False,
            "runs": projection_runs,
        },
        "unavailable_sensitivity": unavailable,
        "evaluations": evaluations,
    }
    write_json(analysis_path, analysis)
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-vote-scores", type=Path)
    parser.add_argument("--min-region-size", type=int, default=100)
    parser.add_argument("--radius-m", type=float, default=0.05)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = evaluate_teacher_handoff(
        closure_root=args.closure_root,
        gt_dir=args.gt_dir,
        runtime_manifest=args.runtime_manifest,
        output_dir=args.output_dir,
        final_vote_scores_path=args.final_vote_scores,
        min_region_size=args.min_region_size,
        radius_m=args.radius_m,
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
