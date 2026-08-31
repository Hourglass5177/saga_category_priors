from __future__ import annotations

"""Independent validation gates for the clean alpha-mask baseline.

This module intentionally consumes already-evaluated, per-scan metrics.  It
does not read predictions or ground truth and it cannot change the clean
baseline.  Its only job is to preserve the preregistered independent unit:
multiple scans of one physical environment are averaged before environments
are given equal weight.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..scannet import physical_scene_id


HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)

UNIFORM_CONDITION = "U-global"
DATA_CONDITION = "D-predicted"
PAIRED_CONDITIONS = (UNIFORM_CONDITION, DATA_CONDITION)

VALIDATION_OBSERVATION_SCHEMA = "saga-clean-baseline-validation-observation-v1"
PHYSICAL_PAIR_ROW_SCHEMA = "saga-clean-baseline-physical-pair-row-v1"
HOLDOUT5_RESULT_SCHEMA = "saga-clean-baseline-holdout5-result-v1"
TUNE24_RESULT_SCHEMA = "saga-clean-baseline-tune24-result-v1"
FINAL48_RESULT_SCHEMA = "saga-clean-baseline-final48-result-v1"
BOOTSTRAP_SCHEMA = "saga-clean-baseline-paired-scene-bootstrap-v1"

FINAL_BOOTSTRAP_SAMPLES = 10_000
FINAL_BOOTSTRAP_SEED = 20_260_804


def _unit_interval(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")
    return result


@dataclass(frozen=True)
class ValidationObservation:
    """One condition's evaluated metrics for one ScanNet scan."""

    scene_id: str
    physical_scene_id: str
    condition: str
    map_50_95: float
    tiny_small_recall_050: float

    def __post_init__(self) -> None:
        scene = str(self.scene_id).strip()
        condition = str(self.condition).strip()
        physical = str(self.physical_scene_id).strip()
        if not scene:
            raise ValueError("scene_id must be non-empty")
        expected_physical = physical_scene_id(scene)
        if physical != expected_physical:
            raise ValueError(
                f"{scene}: physical_scene_id must be {expected_physical!r}, "
                f"not {physical!r}"
            )
        if condition not in PAIRED_CONDITIONS:
            raise ValueError(
                f"condition must be one of {PAIRED_CONDITIONS}, not {condition!r}"
            )
        object.__setattr__(self, "scene_id", scene)
        object.__setattr__(self, "physical_scene_id", physical)
        object.__setattr__(self, "condition", condition)
        object.__setattr__(
            self,
            "map_50_95",
            _unit_interval(self.map_50_95, "map_50_95"),
        )
        object.__setattr__(
            self,
            "tiny_small_recall_050",
            _unit_interval(
                self.tiny_small_recall_050, "tiny_small_recall_050"
            ),
        )

    @classmethod
    def from_row(
        cls, row: "ValidationObservation | Mapping[str, Any]"
    ) -> "ValidationObservation":
        if isinstance(row, cls):
            return row
        if not isinstance(row, Mapping):
            raise TypeError("validation rows must be mappings or observations")
        scene_id = str(row["scene_id"])
        return cls(
            scene_id=scene_id,
            physical_scene_id=str(
                row.get("physical_scene_id", physical_scene_id(scene_id))
            ),
            condition=str(row["condition"]),
            map_50_95=row["map_50_95"],
            tiny_small_recall_050=row["tiny_small_recall_050"],
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "schema": VALIDATION_OBSERVATION_SCHEMA,
            "scene_id": self.scene_id,
            "physical_scene_id": self.physical_scene_id,
            "condition": self.condition,
            "map_50_95": self.map_50_95,
            "tiny_small_recall_050": self.tiny_small_recall_050,
        }


def _normalized_pairs(
    rows: Sequence[ValidationObservation | Mapping[str, Any]],
) -> dict[str, dict[str, ValidationObservation]]:
    if not rows:
        raise ValueError("at least one paired validation row is required")
    by_scene: dict[str, dict[str, ValidationObservation]] = defaultdict(dict)
    for raw in rows:
        row = ValidationObservation.from_row(raw)
        if row.condition in by_scene[row.scene_id]:
            raise ValueError(
                f"duplicate {row.scene_id}/{row.condition} validation row"
            )
        by_scene[row.scene_id][row.condition] = row
    for scene_id, pair in by_scene.items():
        missing = sorted(set(PAIRED_CONDITIONS).difference(pair))
        extra = sorted(set(pair).difference(PAIRED_CONDITIONS))
        if missing or extra:
            raise ValueError(
                f"{scene_id}: paired conditions differ; missing={missing}, "
                f"extra={extra}"
            )
    return dict(by_scene)


def _require_scene_set(
    pairs: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    expected_set = set(map(str, expected))
    actual_set = set(map(str, pairs))
    if len(expected_set) != len(tuple(expected)):
        raise ValueError(f"{label}: expected scene list contains duplicates")
    if actual_set != expected_set:
        raise ValueError(
            f"{label}: scene set differs; missing={sorted(expected_set-actual_set)}, "
            f"unexpected={sorted(actual_set-expected_set)}"
        )


def physical_scene_pair_rows(
    rows: Sequence[ValidationObservation | Mapping[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    """Pair U/D per scan, then average scans within each physical scene."""

    pairs = _normalized_pairs(rows)
    scans_by_physical: dict[str, list[str]] = defaultdict(list)
    for scene_id in pairs:
        scans_by_physical[physical_scene_id(scene_id)].append(scene_id)

    result: list[dict[str, Any]] = []
    for physical in sorted(scans_by_physical):
        scene_ids = sorted(scans_by_physical[physical])
        uniform = [pairs[scene][UNIFORM_CONDITION] for scene in scene_ids]
        data = [pairs[scene][DATA_CONDITION] for scene in scene_ids]
        uniform_map = float(np.mean([row.map_50_95 for row in uniform]))
        data_map = float(np.mean([row.map_50_95 for row in data]))
        uniform_tiny = float(
            np.mean([row.tiny_small_recall_050 for row in uniform])
        )
        data_tiny = float(
            np.mean([row.tiny_small_recall_050 for row in data])
        )
        result.append(
            {
                "schema": PHYSICAL_PAIR_ROW_SCHEMA,
                "stage": str(stage),
                "physical_scene_id": physical,
                # A pipe-separated stable scalar keeps this row directly
                # writable to parquet without nested-list adapter semantics.
                "scene_ids": "|".join(scene_ids),
                "scan_count": len(scene_ids),
                "uniform_map_50_95": uniform_map,
                "data_map_50_95": data_map,
                "delta_map_50_95": data_map - uniform_map,
                "uniform_tiny_small_recall_050": uniform_tiny,
                "data_tiny_small_recall_050": data_tiny,
                "delta_tiny_small_recall_050": data_tiny - uniform_tiny,
            }
        )
    return result


def _macro_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one physical-scene row is required")
    map_deltas = np.asarray(
        [float(row["delta_map_50_95"]) for row in rows], dtype=np.float64
    )
    tiny_deltas = np.asarray(
        [float(row["delta_tiny_small_recall_050"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "physical_scene_count": len(rows),
        "scan_count": int(sum(int(row["scan_count"]) for row in rows)),
        "mean_delta_map_50_95": float(map_deltas.mean()),
        "mean_delta_tiny_small_recall_050": float(tiny_deltas.mean()),
        "positive_map_scene_count": int(np.count_nonzero(map_deltas > 0.0)),
        "zero_map_scene_count": int(np.count_nonzero(map_deltas == 0.0)),
        "negative_map_scene_count": int(np.count_nonzero(map_deltas < 0.0)),
    }


def evaluate_holdout5(
    rows: Sequence[ValidationObservation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen five-canonical-scene replication gate."""

    pairs = _normalized_pairs(rows)
    _require_scene_set(pairs, HOLDOUT5, "holdout5")
    physical_rows = physical_scene_pair_rows(rows, stage="holdout5")
    if len(physical_rows) != 5 or any(row["scan_count"] != 1 for row in physical_rows):
        raise ValueError("holdout5 must contain five distinct physical scenes")
    macro = _macro_summary(physical_rows)
    checks = {
        "mean_delta_map_positive": macro["mean_delta_map_50_95"] > 0.0,
        "at_least_3_of_5_positive": macro["positive_map_scene_count"] >= 3,
        "tiny_small_delta_positive": (
            macro["mean_delta_tiny_small_recall_050"] > 0.0
        ),
    }
    return {
        "schema": HOLDOUT5_RESULT_SCHEMA,
        "stage": "holdout5",
        "expected_scene_ids": list(HOLDOUT5),
        "conditions": list(PAIRED_CONDITIONS),
        "macro": macro,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": physical_rows,
    }


def evaluate_tune24(
    rows: Sequence[ValidationObservation | Mapping[str, Any]],
) -> dict[str, Any]:
    """Group 24 scans into 13 physical scenes and apply the macro gate."""

    pairs = _normalized_pairs(rows)
    if len(pairs) != 24:
        raise ValueError(f"tune24 requires exactly 24 scans, got {len(pairs)}")
    physical_rows = physical_scene_pair_rows(rows, stage="tune24")
    if len(physical_rows) != 13:
        raise ValueError(
            "tune24 must resolve to exactly 13 physical scenes, got "
            f"{len(physical_rows)}"
        )
    macro = _macro_summary(physical_rows)
    checks = {
        "physical_scene_macro_delta_at_least_0.002": (
            macro["mean_delta_map_50_95"] >= 0.002
        )
    }
    return {
        "schema": TUNE24_RESULT_SCHEMA,
        "stage": "tune24",
        "conditions": list(PAIRED_CONDITIONS),
        "aggregation_order": "scan-within-physical-scene-then-equal-physical-scenes",
        "macro": macro,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": physical_rows,
    }


def validate_final48_scene_ids(scene_ids: Sequence[str]) -> tuple[str, ...]:
    """Require 48 unique scans representing 48 unique physical scenes."""

    normalized = tuple(str(value).strip() for value in scene_ids)
    if len(normalized) != 48 or any(not value for value in normalized):
        raise ValueError("final48 requires exactly 48 non-empty scene IDs")
    if len(set(normalized)) != 48:
        raise ValueError("final48 contains duplicate scan IDs")
    by_physical: dict[str, list[str]] = defaultdict(list)
    for scene_id in normalized:
        by_physical[physical_scene_id(scene_id)].append(scene_id)
    duplicates = {
        key: sorted(values)
        for key, values in by_physical.items()
        if len(values) != 1
    }
    if duplicates:
        raise ValueError(
            "final48 contains repeated physical scenes: "
            + "; ".join(
                f"{key}={','.join(values)}" for key, values in sorted(duplicates.items())
            )
        )
    return tuple(sorted(normalized))


def paired_scene_bootstrap(
    physical_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = FINAL_BOOTSTRAP_SAMPLES,
    seed: int = FINAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap of paired physical-scene deltas."""

    if int(samples) != FINAL_BOOTSTRAP_SAMPLES:
        raise ValueError("the registered final bootstrap requires 10,000 samples")
    if not physical_rows:
        raise ValueError("physical_rows cannot be empty")
    deltas = np.asarray(
        [float(row["delta_map_50_95"]) for row in physical_rows],
        dtype=np.float64,
    )
    if not np.isfinite(deltas).all():
        raise ValueError("physical-scene deltas must be finite")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        len(deltas),
        size=(FINAL_BOOTSTRAP_SAMPLES, len(deltas)),
        endpoint=False,
    )
    boot = deltas[indices].mean(axis=1)
    ci = np.quantile(boot, (0.025, 0.975))
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "unit": "physical_scene",
        "paired": True,
        "method": "nonparametric_percentile_bootstrap_of_scene_deltas",
        "samples": FINAL_BOOTSTRAP_SAMPLES,
        "seed": int(seed),
        "physical_scene_count": len(deltas),
        "observed_mean_delta_map_50_95": float(deltas.mean()),
        "bootstrap_mean_delta_map_50_95": float(boot.mean()),
        "ci95_lower": float(ci[0]),
        "ci95_upper": float(ci[1]),
    }


def evaluate_final48(
    rows: Sequence[ValidationObservation | Mapping[str, Any]],
    *,
    samples: int = FINAL_BOOTSTRAP_SAMPLES,
    seed: int = FINAL_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Apply the locked 48-physical-scene paired bootstrap gate."""

    pairs = _normalized_pairs(rows)
    scene_ids = validate_final48_scene_ids(tuple(pairs))
    physical_rows = physical_scene_pair_rows(rows, stage="final48")
    if len(physical_rows) != 48 or any(row["scan_count"] != 1 for row in physical_rows):
        raise AssertionError("validated final48 did not produce 48 independent rows")
    macro = _macro_summary(physical_rows)
    bootstrap = paired_scene_bootstrap(
        physical_rows,
        samples=int(samples),
        seed=int(seed),
    )
    checks = {
        "delta_map_at_least_0.002": macro["mean_delta_map_50_95"] >= 0.002,
        "paired_bootstrap_ci_lower_above_zero": bootstrap["ci95_lower"] > 0.0,
        "bootstrap_used_10000_samples": bootstrap["samples"] == 10_000,
        "all_physical_scenes_unique": True,
    }
    return {
        "schema": FINAL48_RESULT_SCHEMA,
        "stage": "final48",
        "scene_ids": list(scene_ids),
        "conditions": list(PAIRED_CONDITIONS),
        "macro": macro,
        "bootstrap": bootstrap,
        "checks": checks,
        "passed": all(checks.values()),
        "rows": physical_rows,
    }


__all__ = [
    "BOOTSTRAP_SCHEMA",
    "DATA_CONDITION",
    "FINAL48_RESULT_SCHEMA",
    "FINAL_BOOTSTRAP_SAMPLES",
    "FINAL_BOOTSTRAP_SEED",
    "HOLDOUT5",
    "HOLDOUT5_RESULT_SCHEMA",
    "PAIRED_CONDITIONS",
    "PHYSICAL_PAIR_ROW_SCHEMA",
    "TUNE24_RESULT_SCHEMA",
    "UNIFORM_CONDITION",
    "VALIDATION_OBSERVATION_SCHEMA",
    "ValidationObservation",
    "evaluate_final48",
    "evaluate_holdout5",
    "evaluate_tune24",
    "paired_scene_bootstrap",
    "physical_scene_pair_rows",
    "validate_final48_scene_ids",
]
