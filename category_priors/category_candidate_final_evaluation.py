from __future__ import annotations

"""Final offline evaluation for candidate repair and legacy replay.

Ground truth enters only through this module.  Candidate construction, prior
replay, KNN and filtering remain GT-free.  The file-backed entry point accepts
one frozen B0 prediction root and one U/D replay root, reuses the repository's
official ScanNet/Gaussian evaluator, and applies the preregistered DEV8,
holdout, tune, or final gate without changing any prediction.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json, write_rows
from .taxonomy import Taxonomy
from .v9_metrics import (
    evaluate_v9_predictions,
    paired_scannet_bootstrap_from_predictions,
)


SCHEMA = "saga-category-candidate-final-evaluation-v1"
STAGES = frozenset({"dev8", "holdout", "tune", "final"})


def _finite_number(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _metric(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value is not None:
            return _finite_number(value, name)
    raise KeyError(f"metric row is missing all aliases {names}")


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    # Keep reports strict-JSON serialisable while still making a non-zero over
    # zero ratio unambiguously fail every registered upper bound.
    return 0.0 if numerator <= 0.0 else 1.0e12


def _physical_id(scene_id: str, explicit: Mapping[str, str] | None) -> str:
    if explicit is None:
        return str(scene_id).rsplit("_", 1)[0]
    if scene_id not in explicit:
        raise ValueError(f"physical_scene_by_scan is missing {scene_id}")
    value = str(explicit[scene_id])
    if not value:
        raise ValueError(f"{scene_id} has an empty physical scene ID")
    return value


def _condition_payload(
    analysis: Mapping[str, Any], condition: str
) -> Mapping[str, Any]:
    conditions = analysis.get("conditions")
    if not isinstance(conditions, Mapping) or condition not in conditions:
        raise KeyError(f"analysis is missing condition {condition}")
    payload = conditions[condition]
    if not isinstance(payload, Mapping):
        raise TypeError(f"condition {condition} payload must be a mapping")
    if not isinstance(payload.get("metrics"), Mapping) or not isinstance(
        payload.get("per_scene"), Sequence
    ):
        raise TypeError(f"condition {condition} has incomplete metrics")
    return payload


def physical_scene_equal_condition_metrics(
    analysis: Mapping[str, Any],
    *,
    condition: str,
    physical_scene_by_scan: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Average repeated scans within a physical scene, then scenes equally."""

    payload = _condition_payload(analysis, condition)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    seen: set[str] = set()
    for raw_row in payload["per_scene"]:
        if not isinstance(raw_row, Mapping):
            raise TypeError("per_scene rows must be mappings")
        scene_id = str(raw_row.get("scene_id", ""))
        if not scene_id or scene_id in seen:
            raise ValueError("per_scene rows require unique non-empty scene IDs")
        seen.add(scene_id)
        grouped.setdefault(
            _physical_id(scene_id, physical_scene_by_scan), []
        ).append(raw_row)
    if not grouped:
        raise ValueError("physical-scene aggregation requires at least one scan")

    metric_aliases = {
        "map_50_95": ("map_50_95",),
        "ap50": ("map_0.50", "ap50"),
        "predicted_instance_count": ("predicted_instance_count",),
        "gaussian_micro_precision": ("gaussian_micro_precision",),
        "prediction_coverage": ("prediction_coverage",),
        "gt_recall": ("gt_recall", "gt_instance_macro_recall"),
    }
    physical_rows: list[dict[str, Any]] = []
    for physical, rows in sorted(grouped.items()):
        result: dict[str, Any] = {
            "physical_scene_id": physical,
            "scan_ids": sorted(str(row["scene_id"]) for row in rows),
            "scan_count": len(rows),
        }
        for output_name, aliases in metric_aliases.items():
            result[output_name] = float(
                np.mean([_metric(row, *aliases) for row in rows])
            )
        tiny_count = sum(int(row.get("tiny_small_gt_count", 0)) for row in rows)
        tiny_hits = sum(
            int(row.get("tiny_small_match_050_count", 0)) for row in rows
        )
        true_positive = sum(int(row.get("true_positive_count", 0)) for row in rows)
        false_positive = sum(int(row.get("false_positive_count", 0)) for row in rows)
        # A scan is a technical/repeated observation within one physical
        # environment.  Follow the registered hierarchy: first average scan
        # metrics, then give each physical environment equal weight.
        tiny_scan_recall = float(
            np.mean([_metric(row, "tiny_small_recall_050") for row in rows])
        )
        fp_tp_scan_ratio = float(
            np.mean(
                [
                    _ratio(
                        int(row.get("false_positive_count", 0)),
                        int(row.get("true_positive_count", 0)),
                    )
                    for row in rows
                ]
            )
        )
        result.update(
            {
                "tiny_small_gt_count": tiny_count,
                "tiny_small_match_050_count": tiny_hits,
                "tiny_small_recall_050": tiny_scan_recall,
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "fp_tp_ratio": fp_tp_scan_ratio,
            }
        )
        physical_rows.append(result)

    macro_names = (
        "map_50_95",
        "ap50",
        "predicted_instance_count",
        "gaussian_micro_precision",
        "prediction_coverage",
        "gt_recall",
        "tiny_small_recall_050",
        "fp_tp_ratio",
    )
    macro = {
        name: float(np.mean([row[name] for row in physical_rows]))
        for name in macro_names
    }
    return {
        "condition": condition,
        "scan_count": len(seen),
        "physical_scene_count": len(physical_rows),
        "macro": macro,
        "per_physical_scene": physical_rows,
    }


def candidate_survival_intervention(
    rows_by_condition: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    uniform_condition: str,
    data_condition: str,
) -> dict[str, Any]:
    """Determine whether legacy KNN/filter erased a U/D candidate difference."""

    def normalize(condition: str) -> dict[tuple[str, int], Mapping[str, Any]]:
        if condition not in rows_by_condition:
            return {}
        result: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in rows_by_condition[condition]:
            scene_id = str(row.get("scene_id", ""))
            candidate_id = int(row["candidate_id"])
            key = (scene_id, candidate_id)
            if not scene_id or key in result:
                raise ValueError(
                    f"{condition} survival rows need unique scene/candidate keys"
                )
            if "_final_point_indices" not in row:
                raise ValueError(
                    f"{condition}/{scene_id}/{candidate_id}: survival diagnostics "
                    "must be joined with the final output mask"
                )
            final_indices = np.asarray(
                row["_final_point_indices"], dtype=np.int64
            )
            if final_indices.ndim != 1 or (
                len(final_indices)
                and (
                    int(final_indices.min()) < 0
                    or len(np.unique(final_indices)) != len(final_indices)
                    or not np.all(final_indices[:-1] < final_indices[1:])
                )
            ):
                raise ValueError(
                    f"{condition}/{scene_id}/{candidate_id}: final point indices "
                    "must be a sorted unique non-negative vector"
                )
            result[key] = {**dict(row), "_final_point_indices": final_indices}
        return result

    diagnostics_available = (
        uniform_condition in rows_by_condition
        and data_condition in rows_by_condition
    )
    uniform = normalize(uniform_condition)
    data = normalize(data_condition)
    if diagnostics_available and set(uniform) != set(data):
        raise ValueError("U and D survival diagnostics do not cover the same bank")
    keys = sorted(set(uniform) | set(data))
    paired_rows: list[dict[str, Any]] = []
    pre_difference_count = 0
    post_knn_difference_count = 0
    post_filter_difference_count = 0
    post_filter_count_difference_count = 0
    for scene_id, candidate_id in keys:
        left = uniform.get((scene_id, candidate_id), {})
        right = data.get((scene_id, candidate_id), {})
        left_accepted = bool(left.get("accepted", False))
        right_accepted = bool(right.get("accepted", False))
        left_pre = int(left.get("pre_knn_owned_count", 0))
        right_pre = int(right.get("pre_knn_owned_count", 0))
        left_knn = int(left.get("post_knn_total_count", 0))
        right_knn = int(right.get("post_knn_total_count", 0))
        left_filter = int(left.get("post_filter_total_count", 0))
        right_filter = int(right.get("post_filter_total_count", 0))
        left_final_indices = np.asarray(
            left.get("_final_point_indices", ()), dtype=np.int64
        )
        right_final_indices = np.asarray(
            right.get("_final_point_indices", ()), dtype=np.int64
        )
        pre_changed = (left_accepted != right_accepted) or left_pre != right_pre
        knn_changed = left_knn != right_knn
        filter_count_changed = left_filter != right_filter
        # Equal point counts do not imply equal masks.  The registered
        # "erased by legacy KNN/filter" conclusion is about the actual final
        # prediction, so compare exact exported point membership.
        filter_changed = not np.array_equal(
            left_final_indices, right_final_indices
        )
        pre_difference_count += int(pre_changed)
        post_knn_difference_count += int(knn_changed)
        post_filter_difference_count += int(filter_changed)
        post_filter_count_difference_count += int(filter_count_changed)
        paired_rows.append(
            {
                "scene_id": scene_id,
                "candidate_id": candidate_id,
                "uniform_accepted": left_accepted,
                "data_accepted": right_accepted,
                "uniform_pre_knn_count": left_pre,
                "data_pre_knn_count": right_pre,
                "uniform_post_knn_count": left_knn,
                "data_post_knn_count": right_knn,
                "uniform_post_filter_count": left_filter,
                "data_post_filter_count": right_filter,
                "pre_knn_changed": pre_changed,
                "post_knn_changed": knn_changed,
                "post_filter_count_changed": filter_count_changed,
                "uniform_post_filter_point_indices_count": len(
                    left_final_indices
                ),
                "data_post_filter_point_indices_count": len(
                    right_final_indices
                ),
                "post_filter_changed": filter_changed,
            }
        )
    erased = (
        pre_difference_count > 0 and post_filter_difference_count == 0
        if diagnostics_available
        else None
    )
    return {
        "diagnostics_available": diagnostics_available,
        "candidate_pair_count": len(keys),
        "pre_knn_difference_count": pre_difference_count,
        "post_knn_difference_count": post_knn_difference_count,
        "post_filter_difference_count": post_filter_difference_count,
        "post_filter_count_difference_count": (
            post_filter_count_difference_count
        ),
        "post_filter_difference_basis": "exact_exported_point_membership",
        "difference_erased_by_legacy_knn_filter": erased,
        "no_mechanical_intervention": (
            pre_difference_count == 0 if diagnostics_available else None
        ),
        "per_candidate": paired_rows,
    }


def _paired_physical_deltas(
    uniform: Mapping[str, Any], data: Mapping[str, Any]
) -> tuple[dict[str, dict[str, float]], int]:
    left = {
        str(row["physical_scene_id"]): row
        for row in uniform["per_physical_scene"]
    }
    right = {
        str(row["physical_scene_id"]): row for row in data["per_physical_scene"]
    }
    if set(left) != set(right):
        raise ValueError("U and D physical scene sets differ")
    fields = (
        "map_50_95",
        "ap50",
        "gaussian_micro_precision",
        "prediction_coverage",
        "tiny_small_recall_050",
        "fp_tp_ratio",
    )
    result: dict[str, dict[str, float]] = {}
    positive = 0
    for physical in sorted(left):
        result[physical] = {
            name: float(right[physical][name]) - float(left[physical][name])
            for name in fields
        }
        positive += int(result[physical]["map_50_95"] > 0.0)
    return result, positive


def evaluate_registered_gate(
    analysis: Mapping[str, Any],
    *,
    stage: str,
    b0_condition: str,
    uniform_condition: str,
    data_condition: str,
    physical_scene_by_scan: Mapping[str, str] | None = None,
    survival_rows_by_condition: Mapping[
        str, Sequence[Mapping[str, Any]]
    ]
    | None = None,
    final_bootstrap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the frozen structure and D-minus-U gate for one study stage."""

    normalized_stage = str(stage).lower()
    if normalized_stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    summaries = {
        condition: physical_scene_equal_condition_metrics(
            analysis,
            condition=condition,
            physical_scene_by_scan=physical_scene_by_scan,
        )
        for condition in (b0_condition, uniform_condition, data_condition)
    }
    b0 = summaries[b0_condition]["macro"]
    uniform = summaries[uniform_condition]["macro"]
    data = summaries[data_condition]["macro"]
    physical_deltas, positive_count = _paired_physical_deltas(
        summaries[uniform_condition], summaries[data_condition]
    )
    physical_count = len(physical_deltas)

    u_delta_map = uniform["map_50_95"] - b0["map_50_95"]
    u_delta_ap50 = uniform["ap50"] - b0["ap50"]
    u_instance_ratio = _ratio(
        uniform["predicted_instance_count"], b0["predicted_instance_count"]
    )
    u_coverage_delta = uniform["prediction_coverage"] - b0["prediction_coverage"]
    u_checks = {
        "map_drop_within_0.001": u_delta_map >= -0.001,
        "ap50_drop_within_0.002": u_delta_ap50 >= -0.002,
        "instance_count_at_most_1.25x": u_instance_ratio <= 1.25,
        "coverage_drop_within_0.01": u_coverage_delta >= -0.01,
    }

    delta_map = data["map_50_95"] - uniform["map_50_95"]
    delta_ap50 = data["ap50"] - uniform["ap50"]
    delta_tiny = (
        data["tiny_small_recall_050"] - uniform["tiny_small_recall_050"]
    )
    delta_gaussian_precision = (
        data["gaussian_micro_precision"] - uniform["gaussian_micro_precision"]
    )
    fp_tp_worsening = _ratio(data["fp_tp_ratio"], uniform["fp_tp_ratio"]) - 1.0
    if data["fp_tp_ratio"] == uniform["fp_tp_ratio"] == 0.0:
        fp_tp_worsening = 0.0
    common = {
        "delta_map_50_95": delta_map,
        "delta_ap50": delta_ap50,
        "delta_tiny_small_recall_050": delta_tiny,
        "delta_gaussian_micro_precision": delta_gaussian_precision,
        "fp_tp_relative_worsening": fp_tp_worsening,
        "positive_physical_scene_count": positive_count,
        "physical_scene_count": physical_count,
    }

    if normalized_stage == "dev8":
        minimum_positive = min(5, physical_count)
        checks = {
            "registered_primary_gain": delta_map >= 0.002
            or (delta_tiny >= 0.01 and delta_map >= -0.0005),
            "at_least_5_of_8_positive": positive_count >= minimum_positive,
            "fp_tp_worsening_at_most_20pct": fp_tp_worsening <= 0.20,
            "gaussian_precision_drop_within_0.01": delta_gaussian_precision
            >= -0.01,
        }
    elif normalized_stage == "holdout":
        checks = {
            "mean_delta_map_positive": delta_map > 0.0,
            "at_least_3_of_5_positive": positive_count >= min(3, physical_count),
            "tiny_small_recall_positive": delta_tiny > 0.0,
        }
    elif normalized_stage == "tune":
        checks = {"physical_scene_macro_delta_at_least_0.002": delta_map >= 0.002}
    else:
        if final_bootstrap is None:
            raise ValueError("final stage requires a paired bootstrap result")
        bootstrap_delta = _finite_number(
            final_bootstrap.get("delta_map_50_95"), "bootstrap delta_map_50_95"
        )
        ci = final_bootstrap.get("paired_bootstrap_ci95")
        if (
            not isinstance(ci, Sequence)
            or isinstance(ci, (str, bytes))
            or len(ci) != 2
        ):
            raise ValueError("final bootstrap is missing a two-sided CI")
        ci_low = _finite_number(ci[0], "bootstrap CI lower bound")
        ci_high = _finite_number(ci[1], "bootstrap CI upper bound")
        if ci_low > ci_high:
            raise ValueError("final bootstrap CI is reversed")
        checks = {
            "pooled_delta_map_at_least_0.002": bootstrap_delta >= 0.002,
            "paired_bootstrap_ci_lower_above_zero": ci_low > 0.0,
            "bootstrap_used_10000_samples": int(final_bootstrap.get("samples", 0))
            == 10_000,
        }
        common["official_pooled_bootstrap"] = dict(final_bootstrap)

    survival = candidate_survival_intervention(
        survival_rows_by_condition or {},
        uniform_condition=uniform_condition,
        data_condition=data_condition,
    )
    return {
        "schema": SCHEMA,
        "stage": normalized_stage,
        "conditions": summaries,
        "uniform_health": {
            "delta_map_50_95": u_delta_map,
            "delta_ap50": u_delta_ap50,
            "predicted_instance_ratio": u_instance_ratio,
            "prediction_coverage_delta": u_coverage_delta,
            "checks": u_checks,
            "passed": all(u_checks.values()),
        },
        "data_minus_uniform": {
            **common,
            "checks": checks,
            "passed": all(checks.values()),
            "per_physical_scene_delta": physical_deltas,
        },
        "candidate_survival_intervention": survival,
        # U-vs-B0 is the registered structural admission gate on DEV8.  Once
        # it has passed, holdout/tune/final test only the frozen D-vs-U claim;
        # adding a fresh U-vs-B0 stop rule there would be an unregistered gate.
        "passed": all(checks.values()) and (
            all(u_checks.values()) if normalized_stage == "dev8" else True
        ),
    }


def final_paired_scannet_bootstrap(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    replay_root: Path,
    scene_ids: Sequence[str],
    uniform_condition: str,
    data_condition: str,
    taxonomy: Taxonomy,
    samples: int = 10_000,
    seed: int = 20_260_804,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Run the registered paired official-AP bootstrap on frozen final replays."""

    return paired_scannet_bootstrap_from_predictions(
        runtime_manifest=Path(runtime_manifest),
        gt_dir=Path(gt_dir),
        prediction_root=Path(replay_root),
        scene_ids=tuple(map(str, scene_ids)),
        reference_condition=str(uniform_condition),
        treatment_condition=str(data_condition),
        taxonomy=taxonomy,
        samples=int(samples),
        seed=int(seed),
        radius_m=float(radius_m),
        min_region_size=int(min_region_size),
    )


def _condition_root(root: Path, condition: str, scene_id: str) -> tuple[Path, str]:
    direct = root / condition / scene_id
    if (direct / "output.json").is_file() or (
        direct / "seed-42" / "output.json"
    ).is_file():
        return root, condition
    direct_condition_root = root / scene_id
    if (direct_condition_root / "output.json").is_file() or (
        direct_condition_root / "seed-42" / "output.json"
    ).is_file():
        return root.parent, root.name
    return root, condition


def _prediction_dir(root: Path, condition: str, scene_id: str) -> Path:
    direct = root / condition / scene_id
    if (direct / "output.json").is_file():
        return direct
    seeded = direct / "seed-42"
    if (seeded / "output.json").is_file():
        return seeded
    raise FileNotFoundError(direct / "output.json")


def _add_prediction_coverage(
    analysis: dict[str, Any],
    *,
    prediction_root: Path,
    conditions: Sequence[str],
    scene_ids: Sequence[str],
) -> None:
    for condition in map(str, conditions):
        payload = analysis["conditions"][condition]
        per_scene = {str(row["scene_id"]): row for row in payload["per_scene"]}
        assigned_total = 0
        point_total = 0
        for scene_id in map(str, scene_ids):
            output = load_json(
                _prediction_dir(prediction_root, condition, scene_id) / "output.json"
            )
            labels = np.asarray(output.get("point_labels"), dtype=np.int64)
            if labels.ndim != 1:
                raise ValueError(f"{condition}/{scene_id}: point_labels must be a vector")
            assigned = int(np.count_nonzero(labels >= 0))
            per_scene[scene_id]["prediction_coverage"] = (
                assigned / len(labels) if len(labels) else 0.0
            )
            per_scene[scene_id]["assigned_gaussian_count"] = assigned
            per_scene[scene_id]["gaussian_count"] = len(labels)
            assigned_total += assigned
            point_total += len(labels)
        payload["metrics"]["prediction_coverage"] = (
            assigned_total / point_total if point_total else 0.0
        )
        payload["metrics"]["assigned_gaussian_count"] = assigned_total
        payload["metrics"]["gaussian_count"] = point_total


def _load_survival_rows(
    replay_root: Path, conditions: Sequence[str], scene_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {str(name): [] for name in conditions}
    for condition in map(str, conditions):
        for scene_id in map(str, scene_ids):
            scene_root = _prediction_dir(replay_root, condition, scene_id)
            output = load_json(scene_root / "output.json")
            labels = np.asarray(output.get("point_labels"), dtype=np.int64)
            if labels.ndim != 1:
                raise ValueError(
                    f"{condition}/{scene_id}: point_labels must be a vector"
                )
            path = scene_root / "diagnostics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = load_json(path)
            category = (
                payload.get("category_denoise")
                if isinstance(payload, Mapping)
                else None
            )
            rows = (
                category.get("candidate_survival")
                if isinstance(category, Mapping)
                else None
            )
            if not isinstance(rows, list):
                raise TypeError(
                    f"{condition}/{scene_id}: diagnostics lacks "
                    "category_denoise.candidate_survival"
                )
            for row in rows:
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"{condition}/{scene_id}: candidate survival rows must "
                        "be mappings"
                    )
                if "final_instance_id" not in row:
                    raise KeyError(
                        f"{condition}/{scene_id}: candidate survival row lacks "
                        "final_instance_id"
                    )
                final_id = row.get("final_instance_id")
                if final_id is None:
                    final_indices = np.empty(0, dtype=np.int64)
                else:
                    if isinstance(final_id, bool):
                        raise TypeError(
                            f"{condition}/{scene_id}: final_instance_id must be "
                            "an integer or null"
                        )
                    try:
                        final_id_int = int(final_id)
                    except (TypeError, ValueError) as exc:
                        raise TypeError(
                            f"{condition}/{scene_id}: final_instance_id must be "
                            "an integer or null"
                        ) from exc
                    if final_id_int < 0 or str(final_id_int) != str(final_id):
                        raise ValueError(
                            f"{condition}/{scene_id}: invalid final_instance_id"
                        )
                    final_indices = np.flatnonzero(labels == final_id_int).astype(
                        np.int64, copy=False
                    )
                expected_count = int(row.get("post_filter_total_count", 0))
                if len(final_indices) != expected_count:
                    raise ValueError(
                        f"{condition}/{scene_id}/{row.get('candidate_id')}: "
                        "diagnostic post-filter count does not match output mask"
                    )
                result[condition].append(
                    {
                        "scene_id": scene_id,
                        **dict(row),
                        "_final_point_indices": final_indices,
                    }
                )
    return result


def evaluate_candidate_final_stage(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    b0_root: Path,
    replay_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    stage: str,
    output_dir: Path,
    b0_condition: str = "B0-global",
    uniform_condition: str = "U-uniform",
    data_condition: str = "D-class",
    physical_scene_by_scan: Mapping[str, str] | None = None,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    final_bootstrap_samples: int = 10_000,
    final_bootstrap_seed: int = 20_260_804,
    write_viewer: bool = True,
) -> dict[str, Any]:
    """Evaluate B0/U/D roots and write the registered stage decision."""

    normalized_stage = str(stage).lower()
    if normalized_stage not in STAGES:
        raise ValueError(f"stage must be one of {sorted(STAGES)}")
    condition_names = (
        str(b0_condition),
        str(uniform_condition),
        str(data_condition),
    )
    if len(set(condition_names)) != 3 or any(not name for name in condition_names):
        raise ValueError("B0, uniform, and data conditions must be distinct and non-empty")
    normalized_scenes = tuple(map(str, scene_ids))
    if not normalized_scenes or len(set(normalized_scenes)) != len(normalized_scenes):
        raise ValueError("scene_ids must be a non-empty unique sequence")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    b0_eval_root, resolved_b0 = _condition_root(
        Path(b0_root), str(b0_condition), normalized_scenes[0]
    )
    replay_eval_root = Path(replay_root)
    b0_analysis = evaluate_v9_predictions(
        runtime_manifest=Path(runtime_manifest),
        gt_dir=Path(gt_dir),
        prediction_root=b0_eval_root,
        scene_ids=normalized_scenes,
        conditions=(resolved_b0,),
        taxonomy=taxonomy,
        metrics_output=destination / f"{normalized_stage}_b0_metrics.parquet",
        analysis_output=destination / f"{normalized_stage}_b0_analysis.json",
        size_bins=size_bins,
        radius_m=radius_m,
        min_region_size=min_region_size,
        viewer_output=(destination / "viewer" / "b0") if write_viewer else None,
    )
    replay_analysis = evaluate_v9_predictions(
        runtime_manifest=Path(runtime_manifest),
        gt_dir=Path(gt_dir),
        prediction_root=replay_eval_root,
        scene_ids=normalized_scenes,
        conditions=(str(uniform_condition), str(data_condition)),
        taxonomy=taxonomy,
        metrics_output=destination / f"{normalized_stage}_replay_metrics.parquet",
        analysis_output=destination / f"{normalized_stage}_replay_analysis.json",
        size_bins=size_bins,
        radius_m=radius_m,
        min_region_size=min_region_size,
        viewer_output=(destination / "viewer" / "replay") if write_viewer else None,
    )
    _add_prediction_coverage(
        b0_analysis,
        prediction_root=b0_eval_root,
        conditions=(resolved_b0,),
        scene_ids=normalized_scenes,
    )
    _add_prediction_coverage(
        replay_analysis,
        prediction_root=replay_eval_root,
        conditions=(str(uniform_condition), str(data_condition)),
        scene_ids=normalized_scenes,
    )
    merged = {
        "schema": SCHEMA,
        "conditions": {
            str(b0_condition): b0_analysis["conditions"][resolved_b0],
            str(uniform_condition): replay_analysis["conditions"][
                str(uniform_condition)
            ],
            str(data_condition): replay_analysis["conditions"][str(data_condition)],
        },
    }
    bootstrap = None
    if normalized_stage == "final":
        bootstrap = final_paired_scannet_bootstrap(
            runtime_manifest=Path(runtime_manifest),
            gt_dir=Path(gt_dir),
            replay_root=replay_eval_root,
            scene_ids=normalized_scenes,
            uniform_condition=str(uniform_condition),
            data_condition=str(data_condition),
            taxonomy=taxonomy,
            samples=final_bootstrap_samples,
            seed=final_bootstrap_seed,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
    survival = _load_survival_rows(
        replay_eval_root,
        (str(uniform_condition), str(data_condition)),
        normalized_scenes,
    )
    decision = evaluate_registered_gate(
        merged,
        stage=normalized_stage,
        b0_condition=str(b0_condition),
        uniform_condition=str(uniform_condition),
        data_condition=str(data_condition),
        physical_scene_by_scan=physical_scene_by_scan,
        survival_rows_by_condition=survival,
        final_bootstrap=bootstrap,
    )
    rows = [
        {
            "condition": condition,
            **dict(payload["metrics"]),
            **{
                f"physical_macro_{name}": value
                for name, value in decision["conditions"][condition]["macro"].items()
            },
        }
        for condition, payload in merged["conditions"].items()
    ]
    write_rows(destination / f"{normalized_stage}_condition_metrics.parquet", rows)
    write_json(destination / f"{normalized_stage}_analysis.json", decision)
    return decision


__all__ = [
    "SCHEMA",
    "STAGES",
    "candidate_survival_intervention",
    "evaluate_candidate_final_stage",
    "evaluate_registered_gate",
    "final_paired_scannet_bootstrap",
    "physical_scene_equal_condition_metrics",
]
