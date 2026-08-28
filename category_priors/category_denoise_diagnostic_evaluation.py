from __future__ import annotations

"""Evaluation-only reporting for the frozen KNN oracle replay."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise import load_candidate_bank
from .category_denoise_diagnostic_runner import (
    _load_scene_arrays,
    _selection_matches,
    _validate_knn_oracle_plan,
)
from .category_denoise_diagnostics import gaussian_to_gt_mapping
from .evaluator import map_gaussians_to_gt
from .io import load_json, read_rows, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v9_metrics import evaluate_v9_predictions


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _candidate_metrics(
    *,
    candidate_mask: np.ndarray,
    raw_stage_labels: np.ndarray,
    raw_label: int,
    gaussian_xyz_metric: np.ndarray,
    gt_xyz: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    matched_class_id: int,
    matched_instance_id: int,
    radius_m: float,
    gaussian_to_gt_indices: np.ndarray,
    class_count: int,
) -> dict[str, Any]:
    observed = np.asarray(raw_stage_labels, dtype=np.int64) == int(raw_label)
    mapped, _ = map_gaussians_to_gt(
        gt_xyz,
        gaussian_xyz_metric,
        np.where(observed, 0, -1).astype(np.int64),
        radius_m,
    )
    predicted_gt = mapped == 0
    target_gt = (gt_semantic == int(matched_class_id)) & (
        gt_instance == int(matched_instance_id)
    )
    intersection = int(np.count_nonzero(predicted_gt & target_gt))
    union = int(np.count_nonzero(predicted_gt | target_gt))

    nearest = np.asarray(gaussian_to_gt_indices, dtype=np.int64)
    spatially_supported = observed & (nearest >= 0)
    supported_observed = np.zeros(len(observed), dtype=bool)
    supported_ids = np.flatnonzero(spatially_supported)
    if len(supported_ids):
        gt_ids = nearest[supported_ids]
        supported_observed[supported_ids] = (
            (gt_semantic[gt_ids] >= 0)
            & (gt_semantic[gt_ids] < int(class_count))
            & (gt_instance[gt_ids] >= 0)
        )
    target_gaussian = np.zeros(len(observed), dtype=bool)
    supported_ids = np.flatnonzero(supported_observed)
    if len(supported_ids):
        gt_ids = nearest[supported_ids]
        target_gaussian[supported_ids] = (
            (gt_semantic[gt_ids] == int(matched_class_id))
            & (gt_instance[gt_ids] == int(matched_instance_id))
        )
    point_count = int(np.count_nonzero(observed))
    supported_count = int(np.count_nonzero(supported_observed))
    target_count = int(np.count_nonzero(target_gaussian))
    original = np.asarray(candidate_mask, dtype=bool)
    retained = int(np.count_nonzero(observed & original))
    outside = int(np.count_nonzero(observed & ~original))
    return {
        "point_count": point_count,
        "retained_original_point_count": retained,
        "retained_original_fraction": _safe_divide(retained, int(original.sum())),
        "gained_outside_point_count": outside,
        "official_fixed_target_iou": _safe_divide(intersection, union),
        "official_fixed_target_coverage": _safe_divide(
            intersection, int(np.count_nonzero(target_gt))
        ),
        "gaussian_target_precision": _safe_divide(target_count, point_count),
        "gaussian_supported_purity": _safe_divide(target_count, supported_count),
        "gaussian_unsupported_fraction": _safe_divide(
            point_count - supported_count, point_count
        ),
    }


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _scene_equal_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        if row.get(key) is not None:
            by_scene.setdefault(str(row["scene_id"]), []).append(float(row[key]))
    scene_values = [float(np.mean(values)) for values in by_scene.values() if values]
    return float(np.mean(scene_values)) if scene_values else None


def _delta(value: float | None, reference: float | None) -> float | None:
    return None if value is None or reference is None else float(value - reference)


def evaluate_category_denoise_knn_oracle(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    prediction_root: Path,
    oracle_plan: Path,
    output_dir: Path,
    taxonomy: Taxonomy,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate E2 without changing its GT-free replay results."""

    plan = load_json(oracle_plan)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    selected, scene_plans = _validate_knn_oracle_plan(plan, scenes)
    if float(radius_m) != float(plan["radius_m"]):
        raise ValueError("evaluation radius_m must match the frozen oracle plan")
    if int(min_region_size) != int(plan["min_region_size"]):
        raise ValueError(
            "evaluation min_region_size must match the frozen oracle plan"
        )
    replay_summary = load_json(prediction_root / "knn_oracle_replay.json")
    if (
        not isinstance(replay_summary, Mapping)
        or replay_summary.get("schema")
        != "saga-category-denoise-knn-oracle-replay-v1"
        or replay_summary.get("status") != "complete"
        or replay_summary.get("evaluation_only_plan") is not True
        or replay_summary.get("gt_used_by_replay") is not False
        or tuple(replay_summary.get("scene_ids", ())) != selected
    ):
        raise ValueError("KNN oracle replay summary is missing or inconsistent")
    sources = {
        "runtime_manifest": str(Path(runtime_manifest).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "prediction_root": str(Path(prediction_root).resolve()),
        "oracle_plan": str(Path(oracle_plan).resolve()),
        "size_bins": str(Path(size_bins).resolve()) if size_bins is not None else None,
        "taxonomy_classes": list(taxonomy.canonical_classes),
    }
    try:
        existing = load_json(output_dir / "knn_oracle_analysis.json")
        read_rows(output_dir / "knn_oracle_candidates.parquet")
        read_rows(output_dir / "knn_oracle_metrics.parquet")
    except (OSError, ValueError, TypeError, KeyError):
        existing = None
    if isinstance(existing, Mapping):
        identity = existing.get("input_identity", {})
        if (
            existing.get("schema")
            == "saga-category-denoise-knn-oracle-analysis-v1"
            and existing.get("status") == "complete"
            and isinstance(identity, Mapping)
            and tuple(identity.get("scene_ids", ())) == selected
            and float(identity.get("radius_m", -1.0)) == float(radius_m)
            and int(identity.get("min_region_size", -1)) == int(min_region_size)
            and identity.get("sources") == sources
        ):
            result = dict(existing)
            result["status"] = "skipped_complete"
            return result

    candidate_rows: list[dict[str, Any]] = []
    replay_scene_rows: list[dict[str, Any]] = []
    full_prediction_available = True
    for scene_id in selected:
        scene_plan = scene_plans[scene_id]
        # Banks are not copied into the replay root.  Resolve them from the
        # path recorded by the replay diagnostics instead of guessing.
        scene_diag_path = prediction_root / "scenes" / scene_id / "diagnostics.json"
        scene_diag = load_json(scene_diag_path)
        expected_candidates = scene_plan.get("candidates", ())
        expected_ids = [int(row["candidate_id"]) for row in expected_candidates]
        if (
            not isinstance(scene_diag, Mapping)
            or scene_diag.get("schema")
            != "saga-category-denoise-knn-oracle-replay-scene-v1"
            or scene_diag.get("status") != "complete"
            or scene_diag.get("scene_id") != scene_id
            or int(scene_diag.get("point_count", -1))
            != int(scene_plan["point_count"])
            or scene_diag.get("bank_schema") != scene_plan.get("bank_schema")
            or int(scene_diag.get("bank_seed", -1)) != int(scene_plan["bank_seed"])
            or scene_diag.get("candidate_ids") != expected_ids
            or not _selection_matches(expected_candidates, scene_diag.get("selection"))
            or scene_diag.get("gt_used_by_replay") is not False
        ):
            raise TypeError(f"{scene_diag_path}: invalid replay diagnostics")
        # The immutable source bank is needed only for O0 masks.  The replay
        # root records no alternate bank, so the caller's oracle plan carries
        # candidate sizes while the source masks are loaded from the bank path
        # saved alongside the replay below.  Older in-progress output without
        # this field is rejected rather than silently using another bank.
        bank_path = scene_diag.get("bank_path")
        if not bank_path:
            raise ValueError(f"{scene_id}: replay diagnostics is missing bank_path")
        bank = load_candidate_bank(Path(str(bank_path)))
        if (
            bank.point_count != int(scene_plan["point_count"])
            or bank.schema != str(scene_plan["bank_schema"])
            or bank.seed != int(scene_plan["bank_seed"])
        ):
            raise ValueError(f"{scene_id}: replay bank/plan identity mismatch")
        if any(candidate_id >= len(bank.candidates) for candidate_id in expected_ids):
            raise ValueError(f"{scene_id}: plan refers to an unknown bank candidate")
        _, xyz_metric, gt_xyz, ground_truth = _load_scene_arrays(
            scene_id=scene_id, scene=scenes[scene_id], gt_dir=gt_dir
        )
        gaussian_mapping = gaussian_to_gt_mapping(xyz_metric, gt_xyz, radius_m)
        with np.load(
            prediction_root / "scenes" / scene_id / "oracle_replay_labels.npz",
            allow_pickle=False,
        ) as arrays:
            o1_after_knn = np.asarray(arrays["o1_after_knn"], dtype=np.int64)
            o1_after_filter = np.asarray(arrays["o1_after_filter"], dtype=np.int64)
            o2_after_filter = np.asarray(arrays["o2_after_filter"], dtype=np.int64)
        expected_shape = (int(scene_plan["point_count"]),)
        if any(
            value.shape != expected_shape
            for value in (o1_after_knn, o1_after_filter, o2_after_filter)
        ):
            raise ValueError(f"{scene_id}: replay label arrays have an invalid shape")

        raw_labels = scene_diag["candidate_raw_labels"]
        o1_raw = {int(key): int(value) for key, value in raw_labels["O1-unprotected"].items()}
        o2_raw = {int(key): int(value) for key, value in raw_labels["O2-protected"].items()}
        if set(o1_raw) != set(expected_ids) or set(o2_raw) != set(expected_ids):
            raise ValueError(f"{scene_id}: replay candidate raw-label mapping is incomplete")
        if (
            len(set(o1_raw.values())) != len(o1_raw)
            or len(set(o2_raw.values())) != len(o2_raw)
            or any(value < 0 for value in (*o1_raw.values(), *o2_raw.values()))
        ):
            raise ValueError(f"{scene_id}: replay candidate raw labels are invalid")
        for candidate in scene_plan.get("candidates", ()):
            candidate_id = int(candidate["candidate_id"])
            original = np.asarray(bank.branch_full_labels) == candidate_id
            # O0 is the frozen candidate before KNN.  Give it a synthetic
            # local label so the common metric function stays exact.
            o0 = _candidate_metrics(
                candidate_mask=original,
                raw_stage_labels=np.where(original, 0, -1),
                raw_label=0,
                gaussian_xyz_metric=xyz_metric,
                gt_xyz=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                matched_class_id=int(candidate["matched_gt_class_id"]),
                matched_instance_id=int(candidate["matched_gt_instance_id"]),
                radius_m=radius_m,
                gaussian_to_gt_indices=gaussian_mapping.indices,
                class_count=len(taxonomy.canonical_classes),
            )
            o1_knn = _candidate_metrics(
                candidate_mask=original,
                raw_stage_labels=o1_after_knn,
                raw_label=o1_raw[candidate_id],
                gaussian_xyz_metric=xyz_metric,
                gt_xyz=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                matched_class_id=int(candidate["matched_gt_class_id"]),
                matched_instance_id=int(candidate["matched_gt_instance_id"]),
                radius_m=radius_m,
                gaussian_to_gt_indices=gaussian_mapping.indices,
                class_count=len(taxonomy.canonical_classes),
            )
            o1_filter = _candidate_metrics(
                candidate_mask=original,
                raw_stage_labels=o1_after_filter,
                raw_label=o1_raw[candidate_id],
                gaussian_xyz_metric=xyz_metric,
                gt_xyz=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                matched_class_id=int(candidate["matched_gt_class_id"]),
                matched_instance_id=int(candidate["matched_gt_instance_id"]),
                radius_m=radius_m,
                gaussian_to_gt_indices=gaussian_mapping.indices,
                class_count=len(taxonomy.canonical_classes),
            )
            o2 = _candidate_metrics(
                candidate_mask=original,
                raw_stage_labels=o2_after_filter,
                raw_label=o2_raw[candidate_id],
                gaussian_xyz_metric=xyz_metric,
                gt_xyz=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                matched_class_id=int(candidate["matched_gt_class_id"]),
                matched_instance_id=int(candidate["matched_gt_instance_id"]),
                radius_m=radius_m,
                gaussian_to_gt_indices=gaussian_mapping.indices,
                class_count=len(taxonomy.canonical_classes),
            )
            row: dict[str, Any] = {**dict(candidate)}
            for prefix, values in (
                ("O0", o0),
                ("O1_post_knn", o1_knn),
                ("O1_post_filter", o1_filter),
                ("O2", o2),
            ):
                row.update({f"{prefix}_{key}": value for key, value in values.items()})
            row["O1_post_filter_iou_delta_from_O0"] = (
                row["O1_post_filter_official_fixed_target_iou"]
                - row["O0_official_fixed_target_iou"]
            )
            row["O2_iou_delta_from_O0"] = (
                row["O2_official_fixed_target_iou"]
                - row["O0_official_fixed_target_iou"]
            )
            row["O2_exact_full_preserved"] = bool(np.array_equal(
                o2_after_filter == o2_raw[candidate_id], original
            ))
            candidate_rows.append(row)

        full_prediction_available &= bool(scene_diag["full_prediction_available"])
        replay_scene_rows.append(
            {
                "scene_id": scene_id,
                "candidate_count": int(scene_diag["candidate_count"]),
                "full_prediction_available": bool(
                    scene_diag["full_prediction_available"]
                ),
                "mapping_error": scene_diag.get("mapping_error"),
                **(scene_diag.get("collateral") or {}),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "knn_oracle_metrics.parquet"
    full_analysis: dict[str, Any] | None = None
    conditions = ("B0", "O1-unprotected", "O2-protected")
    if full_prediction_available:
        full_analysis = evaluate_v9_predictions(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            prediction_root=prediction_root,
            scene_ids=selected,
            conditions=conditions,
            taxonomy=taxonomy,
            metrics_output=metrics_path,
            analysis_output=output_dir / "knn_oracle_analysis.json",
            size_bins=size_bins,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        metric_rows = read_rows(metrics_path)
        for row in metric_rows:
            row["metric_protocol_note"] = (
                "repository historical 0.50:0.95 ten-threshold mean; AP25 is separate"
            )
            row["full_prediction_available"] = True
    else:
        # B0 is still complete for every requested scene.  O1/O2 aggregate AP
        # is intentionally absent because a subset-only number would be biased.
        full_analysis = evaluate_v9_predictions(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            prediction_root=prediction_root,
            scene_ids=selected,
            conditions=("B0",),
            taxonomy=taxonomy,
            metrics_output=metrics_path,
            analysis_output=output_dir / "knn_oracle_analysis.json",
            size_bins=size_bins,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        metric_rows = read_rows(metrics_path)
        for condition in ("O1-unprotected", "O2-protected"):
            metric_rows.append(
                {
                    "condition": condition,
                    "scene_count": len(selected),
                    "full_prediction_available": False,
                    "map_50_95": None,
                    "map_0.50": None,
                    "ap50": None,
                    "map_0.25": None,
                    "metric_protocol_note": "aggregate AP withheld because exact B0 mapping failed in at least one requested scene",
                }
            )
        for row in metric_rows:
            if row.get("condition") == "B0":
                row["full_prediction_available"] = True

    for row in metric_rows:
        row["repository_map_50_95_10_thresholds"] = row.get("map_50_95")
        row["AP50"] = row.get("ap50", row.get("map_0.50"))
        row["AP25"] = row.get("map_0.25")
    write_rows(metrics_path, metric_rows)
    write_rows(output_dir / "knn_oracle_candidates.parquet", candidate_rows)
    o0_iou = _scene_equal_mean(candidate_rows, "O0_official_fixed_target_iou")
    o1_iou = _scene_equal_mean(
        candidate_rows, "O1_post_filter_official_fixed_target_iou"
    )
    o2_iou = _scene_equal_mean(candidate_rows, "O2_official_fixed_target_iou")
    o1_precision = _scene_equal_mean(
        candidate_rows, "O1_post_filter_gaussian_target_precision"
    )
    o2_precision = _scene_equal_mean(candidate_rows, "O2_gaussian_target_precision")
    o1_coverage = _scene_equal_mean(
        candidate_rows, "O1_post_filter_official_fixed_target_coverage"
    )
    o2_coverage = _scene_equal_mean(candidate_rows, "O2_official_fixed_target_coverage")
    analysis = {
        "schema": "saga-category-denoise-knn-oracle-analysis-v1",
        "status": "complete",
        "input_identity": {
            "scene_ids": list(selected),
            "radius_m": float(radius_m),
            "min_region_size": int(min_region_size),
            "oracle_iou_threshold": float(plan["iou_threshold"]),
            "evaluation_only": True,
            "sources": sources,
        },
        "candidate_count": len(candidate_rows),
        "scene_count": len(selected),
        "full_prediction_available": full_prediction_available,
        "candidate_mechanics": {
            "scene_equal_O0_iou": o0_iou,
            "scene_equal_O1_post_filter_iou": o1_iou,
            "scene_equal_O2_iou": o2_iou,
            "O1_iou_delta_from_O0": _delta(o1_iou, o0_iou),
            "O2_iou_delta_from_O0": _delta(o2_iou, o0_iou),
            "O2_minus_O1_precision": _delta(o2_precision, o1_precision),
            "O2_minus_O1_coverage": _delta(o2_coverage, o1_coverage),
            "O2_exact_full_preserved_count": sum(
                bool(row["O2_exact_full_preserved"]) for row in candidate_rows
            ),
            "O1_survived_after_filter_count": sum(
                int(row["O1_post_filter_point_count"] > 0) for row in candidate_rows
            ),
            "per_scene": replay_scene_rows,
        },
        "full_prediction": full_analysis,
        "metric_protocol_note": (
            "map_50_95 is this repository's historical mean over ten thresholds "
            "from 0.50 through 0.95; it is not labeled as an externally comparable "
            "ScanNet benchmark number."
        ),
        "interpretation": (
            "This is a deterministic mechanism check on GT-selected candidates. "
            "O2 is an oracle upper bound, not a deployable acceptance rule, and "
            "the small selected set is not treated as a significance sample."
        ),
    }
    write_json(output_dir / "knn_oracle_analysis.json", analysis)
    return analysis


__all__ = ["evaluate_category_denoise_knn_oracle"]
