from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import hash_json, sha256_file, write_json
from .taxonomy import Taxonomy

LOG_GEOMETRY_FIELDS = (
    "bbox_diag_m",
    "surface_area_m2",
    "bbox_volume_m3",
    "extent_short_m",
    "extent_mid_m",
    "extent_long_m",
    "voxel_count",
)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantiles: Sequence[float]
) -> list[float]:
    if not values:
        raise ValueError("weighted_quantile requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if array.shape != weight.shape or np.any(weight < 0) or weight.sum() <= 0:
        raise ValueError("Invalid weighted quantile inputs")
    order = np.argsort(array, kind="mergesort")
    array = array[order]
    weight = weight[order]
    positions = (np.cumsum(weight) - 0.5 * weight) / weight.sum()
    return np.interp(np.asarray(quantiles, dtype=np.float64), positions, array).tolist()


def _scene_balanced(
    rows: Sequence[Mapping[str, Any]], getter: Any
) -> tuple[list[float], list[float]]:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = getter(row)
        if value is None or not math.isfinite(float(value)):
            continue
        by_scene[str(row["physical_scene_id"])].append(float(value))
    values: list[float] = []
    weights: list[float] = []
    for scene_id in sorted(by_scene):
        scene_values = by_scene[scene_id]
        per_instance_weight = 1.0 / len(scene_values)
        values.extend(scene_values)
        weights.extend([per_instance_weight] * len(scene_values))
    return values, weights


def _summary(values: Sequence[float], weights: Sequence[float]) -> dict[str, float]:
    result = weighted_quantile(values, weights, QUANTILES)
    return {
        f"q{int(quantile * 100):02d}": float(value)
        for quantile, value in zip(QUANTILES, result)
    }


def _json_metric(row: Mapping[str, Any], field: str, key: str) -> float | None:
    raw = row.get(field)
    if raw in (None, ""):
        return None
    try:
        payload = raw if isinstance(raw, dict) else json.loads(str(raw))
        value = float(payload[key])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if math.isfinite(value) else None


def _support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scene_counts: dict[str, int] = defaultdict(int)
    scans: set[str] = set()
    for row in rows:
        scene_counts[str(row["physical_scene_id"])] += 1
        scans.add(str(row["scene_id"]))
    valid_count = sum(_as_bool(row.get("quality_valid", False)) for row in rows)
    return {
        "physical_scenes": len(scene_counts),
        "scans": len(scans),
        "instances": len(rows),
        "n_eff": int(sum(min(count, 5) for count in scene_counts.values())),
        "quality_fraction": float(valid_count / len(rows)) if rows else 0.0,
    }


def _bootstrap_relative_width(
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> float:
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_positive(row.get("bbox_diag_m"))
        if value is not None:
            by_scene[str(row["physical_scene_id"])].append(math.log(value))
    scene_values = [float(np.median(by_scene[key])) for key in sorted(by_scene)]
    if len(scene_values) < 2:
        return 1.0
    array = np.asarray(scene_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        boot[index] = float(np.median(rng.choice(array, size=len(array), replace=True)))
    low, high = np.quantile(boot, (0.025, 0.975))
    center = float(np.median(array))
    # Convert the log-scale interval back to a relative multiplicative width.
    # This remains well behaved for categories whose typical diagonal is near 1 m.
    return float((math.exp(high) - math.exp(low)) / max(math.exp(center), 1e-12))


def _raw_node(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    valid_rows = [row for row in rows if _as_bool(row.get("quality_valid", False))]
    geometry: dict[str, dict[str, float]] = {}
    for field in LOG_GEOMETRY_FIELDS:
        values, weights = _scene_balanced(
            valid_rows,
            lambda row, field=field: (
                math.log(value)
                if (value := _finite_positive(row.get(field))) is not None
                else None
            ),
        )
        if values:
            geometry[f"log_{field}"] = _summary(values, weights)

    neighborhood: dict[str, dict[str, float]] = {}
    for field, keys in (
        ("same_instance_fixed_json", ("0.02", "0.05", "0.10", "0.20")),
        ("same_instance_relative_json", ("0.02", "0.05", "0.10")),
        ("boundary_fixed_json", ("0.02", "0.05", "0.10", "0.20")),
    ):
        for key in keys:
            values, weights = _scene_balanced(
                valid_rows,
                lambda row, field=field, key=key: _json_metric(row, field, key),
            )
            if values:
                neighborhood[f"{field.removesuffix('_json')}:{key}"] = _summary(
                    values, weights
                )

    support = _support(rows)
    relative_width = _bootstrap_relative_width(valid_rows, seed, bootstrap_samples)
    return {
        "name": name,
        "support": support,
        "uncertainty": {"relative_bootstrap_ci_width": relative_width},
        "raw": {"geometry": geometry, "neighborhood": neighborhood},
    }


def _interpolate_summaries(
    child: Mapping[str, dict[str, float]],
    fallback: Mapping[str, dict[str, float]],
    weight: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in sorted(set(child) | set(fallback)):
        if metric not in child:
            result[metric] = dict(fallback[metric])
            continue
        if metric not in fallback:
            result[metric] = dict(child[metric])
            continue
        result[metric] = {
            quantile: float(
                weight * child[metric][quantile]
                + (1.0 - weight) * fallback[metric][quantile]
            )
            for quantile in child[metric]
            if quantile in fallback[metric]
        }
    return result


def _activate_node(
    node: dict[str, Any],
    fallback: Mapping[str, Any],
    min_physical_scenes: int,
    shrink_tau: float,
) -> dict[str, Any]:
    support = node["support"]
    active = bool(
        support["physical_scenes"] >= min_physical_scenes and node["raw"]["geometry"]
    )
    weight = (
        float(support["n_eff"] / (support["n_eff"] + shrink_tau)) if active else 0.0
    )
    width = float(node["uncertainty"]["relative_bootstrap_ci_width"])
    reliability = float(
        np.clip(weight * support["quality_fraction"] * math.exp(-width), 0.0, 1.0)
    )
    node["active"] = active
    node["shrink_weight"] = weight
    node["reliability"] = reliability
    node["shrunk"] = {
        "geometry": _interpolate_summaries(
            node["raw"]["geometry"], fallback["shrunk"]["geometry"], weight
        ),
        "neighborhood": _interpolate_summaries(
            node["raw"]["neighborhood"], fallback["shrunk"]["neighborhood"], weight
        ),
    }
    return node


def fit_priors(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Taxonomy,
    source_table: str | Path,
    seed: int = 20260804,
    bootstrap_samples: int = 2000,
    min_physical_scenes: int = 5,
    shrink_tau: float = 20.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No statistics rows were provided")
    seen_splits = {str(row.get("split", "")).strip().lower() for row in rows}
    if seen_splits != {"train"}:
        raise ValueError(f"Priors are train-only; found splits: {sorted(seen_splits)}")
    seen_units = {str(row.get("units", "")).strip().lower() for row in rows}
    if seen_units != {"meters"}:
        raise ValueError(
            f"Priors require metric data; found units: {sorted(seen_units)}"
        )
    unknown_classes = {str(row["canonical_class"]) for row in rows} - set(
        taxonomy.canonical_classes
    )
    if unknown_classes:
        raise ValueError(f"Unknown canonical classes: {sorted(unknown_classes)}")

    global_node = _raw_node("global", rows, seed, bootstrap_samples)
    global_node["active"] = True
    global_node["shrink_weight"] = 1.0
    global_node["reliability"] = 1.0
    global_node["shrunk"] = {
        "geometry": global_node["raw"]["geometry"],
        "neighborhood": global_node["raw"]["neighborhood"],
    }

    parent_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    class_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        canonical = str(row["canonical_class"])
        class_rows[canonical].append(row)
        parent_rows[taxonomy.parent_for(canonical)].append(row)

    parents: dict[str, dict[str, Any]] = {}
    for parent in sorted(parent_rows):
        if parent == "global":
            continue
        node = _raw_node(
            parent,
            parent_rows[parent],
            seed ^ _stable_name_seed(parent),
            bootstrap_samples,
        )
        node["fallback"] = "global"
        parents[parent] = _activate_node(
            node, global_node, min_physical_scenes, shrink_tau
        )

    categories: dict[str, dict[str, Any]] = {}
    for canonical in taxonomy.canonical_classes:
        raw_rows = class_rows.get(canonical, [])
        if not raw_rows:
            categories[canonical] = {
                "name": canonical,
                "parent": taxonomy.parent_for(canonical),
                "fallback": taxonomy.parent_for(canonical),
                "active": False,
                "support": {
                    "physical_scenes": 0,
                    "scans": 0,
                    "instances": 0,
                    "n_eff": 0,
                    "quality_fraction": 0.0,
                },
                "reliability": 0.0,
                "small_score": 0.0,
            }
            continue
        parent = taxonomy.parent_for(canonical)
        fallback = parents.get(parent, global_node)
        node = _raw_node(
            canonical, raw_rows, seed ^ _stable_name_seed(canonical), bootstrap_samples
        )
        node["parent"] = parent
        node["fallback"] = parent if parent in parents else "global"
        categories[canonical] = _activate_node(
            node, fallback, min_physical_scenes, shrink_tau
        )

    active_categories = [node for node in categories.values() if node.get("active")]
    area_medians = [
        node["shrunk"]["geometry"]["log_surface_area_m2"]["q50"]
        for node in active_categories
    ]
    diag_medians = [
        node["shrunk"]["geometry"]["log_bbox_diag_m"]["q50"]
        for node in active_categories
    ]
    for node in categories.values():
        if not node.get("active"):
            node["small_score"] = 0.0
            continue
        area = node["shrunk"]["geometry"]["log_surface_area_m2"]["q50"]
        diag = node["shrunk"]["geometry"]["log_bbox_diag_m"]["q50"]
        area_rank = float(np.mean(np.asarray(area_medians) <= area))
        diag_rank = float(np.mean(np.asarray(diag_medians) <= diag))
        node["small_score"] = float(
            np.clip(1.0 - 0.5 * (area_rank + diag_rank), 0.0, 1.0)
        )

    source_path = Path(source_table)
    payload = {
        "schema_version": "1.0",
        "kind": "category_priors",
        "provenance": {
            "datasets": sorted({str(row["dataset"]) for row in rows}),
            "splits": ["train"],
            "source_table": source_path.name,
            "source_table_sha256": sha256_file(source_path),
            "taxonomy_sha256": taxonomy.content_hash,
            "row_count": len(rows),
            "seed": seed,
        },
        "normalization": {
            "units": "meters",
            "voxel_size_m": 0.02,
            "fixed_radii_m": [0.02, 0.05, 0.10, 0.20],
            "relative_radii": [0.02, 0.05, 0.10],
        },
        "fit_config": {
            "bootstrap_samples": bootstrap_samples,
            "min_physical_scenes": min_physical_scenes,
            "shrink_tau": shrink_tau,
            "scene_instance_cap": 5,
        },
        "global": global_node,
        "parents": parents,
        "categories": categories,
        "fallback": {"unknown": "legacy_global", "low_confidence": "legacy_global"},
    }
    payload["content_sha256"] = hash_json(payload)
    return payload


def _stable_name_seed(value: str) -> int:
    return int.from_bytes(value.encode("utf-8"), "little", signed=False) % (2**31 - 1)


def validate_priors(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("kind") != "category_priors"
        or payload.get("schema_version") != "1.0"
    ):
        raise ValueError("Unsupported category prior schema")
    if payload.get("provenance", {}).get("splits") != ["train"]:
        raise ValueError("Category priors must contain exactly the train split")
    expected = payload.get("content_sha256")
    if not expected:
        raise ValueError("Category priors are missing content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if hash_json(unsigned) != expected:
        raise ValueError("Category prior content hash mismatch")


def write_priors(path: str | Path, payload: Mapping[str, Any]) -> None:
    validate_priors(payload)
    write_json(path, payload)
