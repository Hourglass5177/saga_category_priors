from __future__ import annotations

"""Re-score frozen E3 full/fragment/merge rows with the section-30 prior."""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_prior_v2 import score_same_bank_candidate_priors
from .io import load_json, read_rows, write_json


SIZE_G_FLOOR = math.exp(-12.5)
SIZE_G_LOW_THRESHOLD = 0.001
FULL_G_LOW_FRACTION_MAX = 0.10
DISCRIMINATION_IMPROVEMENT_MIN = 0.02
DISCRIMINATION_DEGRADATION_MAX = 0.02
FULL_SUPPORT_RATE_DROP_MAX = 0.05
GATE_TOLERANCE = 1e-12


def _candidate(
    row: Mapping[str, Any], prefix: str, candidate_id: int
) -> dict[str, Any]:
    return {
        "candidate_id": int(candidate_id),
        "branch_class": str(row["class_name"]),
        "metric_extents_m": [
            float(row[f"{prefix}metric_extent_short_m"]),
            float(row[f"{prefix}metric_extent_mid_m"]),
            float(row[f"{prefix}metric_extent_long_m"]),
        ],
        "trusted_core_point_count": int(row[f"{prefix}gaussian_count"]),
        "Q": 1.0,
    }


def _scan_physical_equal(
    values: Sequence[tuple[str, str, float]],
) -> float | None:
    """Average rows within scan, then scans within each physical scene."""

    by_scan: dict[tuple[str, str], list[float]] = defaultdict(list)
    for physical_scene_id, scene_id, value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("oracle aggregate values must be finite")
        by_scan[(str(physical_scene_id), str(scene_id))].append(numeric)
    if not by_scan:
        return None
    by_physical: dict[str, list[float]] = defaultdict(list)
    for (physical_scene_id, _scene_id), rows in by_scan.items():
        by_physical[physical_scene_id].append(float(np.mean(rows)))
    return float(
        np.mean(
            [
                np.mean(scan_values)
                for _physical, scan_values in sorted(by_physical.items())
            ]
        )
    )


def prior_oracle_v2_gate_checks(
    *,
    full_g_low_fraction: float,
    fragment_d_minus_u: float | None,
    merge_d_minus_u: float | None,
    uniform_full_support_rate: float | None,
    class_full_support_rate: float | None,
    full_median_above_fragment_and_merge: bool,
) -> dict[str, bool]:
    """Evaluate the four registered section-30 capacity gates."""

    low_fraction = float(full_g_low_fraction)
    low_fraction_valid = math.isfinite(low_fraction) and (
        0.0 <= low_fraction <= FULL_G_LOW_FRACTION_MAX + GATE_TOLERANCE
    )
    discrimination_valid = False
    if fragment_d_minus_u is not None and merge_d_minus_u is not None:
        fragment = float(fragment_d_minus_u)
        merge = float(merge_d_minus_u)
        if (
            math.isfinite(fragment)
            and math.isfinite(merge)
            and -1.0 <= fragment <= 1.0
            and -1.0 <= merge <= 1.0
        ):
            discrimination_valid = (
                fragment + GATE_TOLERANCE >= DISCRIMINATION_IMPROVEMENT_MIN
                and merge + GATE_TOLERANCE
                >= -DISCRIMINATION_DEGRADATION_MAX
            ) or (
                merge + GATE_TOLERANCE >= DISCRIMINATION_IMPROVEMENT_MIN
                and fragment + GATE_TOLERANCE
                >= -DISCRIMINATION_DEGRADATION_MAX
            )
    support_valid = False
    if uniform_full_support_rate is not None and class_full_support_rate is not None:
        uniform_support = float(uniform_full_support_rate)
        class_support = float(class_full_support_rate)
        support_valid = (
            math.isfinite(uniform_support)
            and math.isfinite(class_support)
            and 0.0 <= uniform_support <= 1.0
            and 0.0 <= class_support <= 1.0
            and class_support + GATE_TOLERANCE
            >= uniform_support - FULL_SUPPORT_RATE_DROP_MAX
        )
    return {
        "full_G_le_0.001_fraction_at_most_0.10": bool(low_fraction_valid),
        "full_median_G_above_fragment_and_merge": bool(
            full_median_above_fragment_and_merge
        ),
        "class_improves_one_discrimination_without_harming_other": bool(
            discrimination_valid
        ),
        "class_full_support_rate_not_lower_by_more_than_0.05": bool(
            support_valid
        ),
    }


def evaluate_prior_oracle_v2_rows(
    *,
    object_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    priors: Mapping[str, Any],
    main_radius_m: float = 0.05,
) -> dict[str, Any]:
    main = float(main_radius_m)
    if not math.isfinite(main) or main <= 0.0:
        raise ValueError("main_radius_m must be positive and finite")
    objects = [
        row
        for row in object_rows
        if bool(row.get("eligible", False))
        and np.isclose(float(row["radius_m"]), main, rtol=0.0, atol=1e-12)
    ]
    pairs = [
        row
        for row in pair_rows
        if np.isclose(float(row["radius_m"]), main, rtol=0.0, atol=1e-12)
    ]
    object_scored: list[dict[str, Any]] = []
    for index, row in enumerate(objects):
        candidate = _candidate(row, "", index)
        scores = score_same_bank_candidate_priors((candidate,), priors)
        object_scored.append(
            {
                "scene_id": str(row["scene_id"]),
                "physical_scene_id": str(
                    row.get("physical_scene_id", row["scene_id"])
                ),
                "U_G": float(scores.uniform[0]["G"]),
                "D_G": float(scores.class_shrunk[0]["G"]),
                "U_support": bool(scores.uniform[0]["support_pass"]),
                "D_support": bool(scores.class_shrunk[0]["support_pass"]),
            }
        )
    pair_scored: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        negative_type = str(row["negative_type"])
        if negative_type not in {"fragment", "merge"}:
            raise ValueError(f"unsupported oracle negative_type: {negative_type}")
        candidates = (
            _candidate(row, "full_", 2 * index),
            _candidate(row, "negative_", 2 * index + 1),
        )
        scores = score_same_bank_candidate_priors(candidates, priors)
        scored: dict[str, Any] = {
            "scene_id": str(row["scene_id"]),
            "physical_scene_id": str(
                row.get("physical_scene_id", row["scene_id"])
            ),
            "negative_type": negative_type,
        }
        for label, score_rows in (
            ("U", scores.uniform),
            ("D", scores.class_shrunk),
        ):
            full, negative = score_rows
            full_score = float(full["S"]) if full["support_pass"] else 0.0
            negative_score = (
                float(negative["S"]) if negative["support_pass"] else 0.0
            )
            if np.isclose(full_score, negative_score, rtol=1e-9, atol=1e-12):
                accuracy = 0.5
            else:
                accuracy = float(full_score > negative_score)
            scored[f"{label}_full_G"] = float(full["G"])
            scored[f"{label}_negative_G"] = float(negative["G"])
            scored[f"{label}_accuracy"] = accuracy
        pair_scored.append(scored)

    full_d = np.asarray([row["D_G"] for row in object_scored], dtype=np.float64)
    medians: dict[str, Any] = {
        "full_D_G": float(np.median(full_d)) if len(full_d) else None
    }
    effects: dict[str, Any] = {}
    for negative_type in ("fragment", "merge"):
        selected = [
            row for row in pair_scored if row["negative_type"] == negative_type
        ]
        d_negative = [float(row["D_negative_G"]) for row in selected]
        medians[f"{negative_type}_D_G"] = (
            float(np.median(d_negative)) if d_negative else None
        )
        u_accuracy = _scan_physical_equal(
            [
                (
                    row["physical_scene_id"],
                    row["scene_id"],
                    row["U_accuracy"],
                )
                for row in selected
            ]
        )
        d_accuracy = _scan_physical_equal(
            [
                (
                    row["physical_scene_id"],
                    row["scene_id"],
                    row["D_accuracy"],
                )
                for row in selected
            ]
        )
        effects[negative_type] = {
            "pair_count": len(selected),
            "U_scene_equal_accuracy": u_accuracy,
            "D_scene_equal_accuracy": d_accuracy,
            "D_minus_U": (
                float(d_accuracy - u_accuracy)
                if d_accuracy is not None and u_accuracy is not None
                else None
            ),
        }
    full_low_fraction = (
        float(
            np.mean(
                full_d <= SIZE_G_LOW_THRESHOLD
            )
        )
        if len(full_d)
        else 1.0
    )
    u_support = _scan_physical_equal(
        [
            (
                row["physical_scene_id"],
                row["scene_id"],
                float(row["U_support"]),
            )
            for row in object_scored
        ]
    )
    d_support = _scan_physical_equal(
        [
            (
                row["physical_scene_id"],
                row["scene_id"],
                float(row["D_support"]),
            )
            for row in object_scored
        ]
    )
    fragment_delta = effects["fragment"]["D_minus_U"]
    merge_delta = effects["merge"]["D_minus_U"]
    medians_valid = (
        medians["full_D_G"] is not None
        and medians["fragment_D_G"] is not None
        and medians["merge_D_G"] is not None
        and medians["full_D_G"] > medians["fragment_D_G"]
        and medians["full_D_G"] > medians["merge_D_G"]
    )
    checks = prior_oracle_v2_gate_checks(
        full_g_low_fraction=full_low_fraction,
        fragment_d_minus_u=fragment_delta,
        merge_d_minus_u=merge_delta,
        uniform_full_support_rate=u_support,
        class_full_support_rate=d_support,
        full_median_above_fragment_and_merge=medians_valid,
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "saga-category-prior-oracle-v2",
        "main_radius_m": main,
        "object_count": len(object_scored),
        "pair_count": len(pair_scored),
        "G_formula_floor": SIZE_G_FLOOR,
        "G_low_threshold": SIZE_G_LOW_THRESHOLD,
        "full_D_G_le_0.001_fraction": full_low_fraction,
        "medians": medians,
        "diagnostics": {},
        "effects": effects,
        "U_full_support_scene_equal_rate": u_support,
        "D_full_support_scene_equal_rate": d_support,
        "aggregation": "rows-within-scan_then-scans-within-physical-scene",
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": all(checks.values()),
        "category_prior_tested_on_predictions": False,
        "gt_boundary": "complete_object_capacity_control_only",
    }


def evaluate_prior_oracle_v2(
    *,
    prior_oracle_root: Path,
    category_priors: Path,
    output: Path,
    main_radius_m: float = 0.05,
) -> dict[str, Any]:
    result = evaluate_prior_oracle_v2_rows(
        object_rows=read_rows(prior_oracle_root / "prior_oracle_objects.parquet"),
        pair_rows=read_rows(prior_oracle_root / "prior_oracle_pairs.parquet"),
        priors=load_json(category_priors),
        main_radius_m=main_radius_m,
    )
    write_json(output, result)
    return result


__all__ = [
    "DISCRIMINATION_DEGRADATION_MAX",
    "DISCRIMINATION_IMPROVEMENT_MIN",
    "FULL_G_LOW_FRACTION_MAX",
    "FULL_SUPPORT_RATE_DROP_MAX",
    "SIZE_G_LOW_THRESHOLD",
    "SIZE_G_FLOOR",
    "evaluate_prior_oracle_v2",
    "evaluate_prior_oracle_v2_rows",
    "prior_oracle_v2_gate_checks",
]
