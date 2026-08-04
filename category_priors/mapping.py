from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import hash_json, load_json, sha256_file

GLOBAL_SEARCH_SPACE: dict[str, tuple[float, float, str]] = {
    "feature_ratio": (0.30, 0.70, "linear"),
    "instance_threshold": (0.20, 0.50, "linear"),
    "min_cluster_size": (5.0, 20.0, "integer"),
    "knn_k": (64.0, 256.0, "integer"),
    "semantic_threshold": (0.55, 0.80, "linear"),
}

PRIOR_SEARCH_SPACE: dict[str, tuple[float, float, str]] = {
    "alpha_m": (0.25, 2.00, "log"),
    "support_exponent": (0.50, 1.00, "linear"),
    "alpha_r": (0.25, 1.50, "linear"),
    "alpha_k": (0.02, 0.15, "linear"),
}

DEFAULT_MAPPING_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "kind": "prior_mapping",
    "baseline": {
        "feature_ratio": 0.5,
        "instance_threshold": 0.3,
        "min_cluster_size": 10,
        "knn_k": 256,
        "semantic_threshold": 0.7,
        "sample_num": 10000,
        "cluster_selection_epsilon": 0.01,
    },
    "coefficients": {
        "alpha_m": 1.0,
        "support_exponent": 0.75,
        "alpha_r": 1.0,
        "alpha_k": 0.05,
    },
    "fixed": {
        "density_k": 16,
        "density_sample_cap": 50000,
        "min_cluster_bounds": [3, 64],
        "knn_k_bounds": [8, 256],
        "class_sample_min": 128,
        "class_sample_max": 5000,
        "total_sample_budget": 10000,
        "small_cluster_discount": 0.5,
        "consistency_ratio_bounds": [0.5, 1.5],
        "evaluation_radius_m": 0.05,
    },
}

REGISTERED_CONDITIONS = (
    "B0-legacy",
    "B1-other-classes",
    "P000-B2",
    "P001-small",
    "P010-smooth",
    "P011-smooth-small",
    "P100-size",
    "P101-size-small",
    "P110-size-smooth",
    "P111-combined",
    "P111-no-gate",
    "P111-no-shrink",
)


def latin_hypercube_design(
    kind: str, samples: int = 32, seed: int = 20260804
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    space = (
        GLOBAL_SEARCH_SPACE
        if kind == "global"
        else PRIOR_SEARCH_SPACE
        if kind == "prior"
        else None
    )
    if space is None:
        raise ValueError("kind must be 'global' or 'prior'")
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}
    for name, (low, high, scale) in space.items():
        unit = (np.arange(samples, dtype=np.float64) + rng.random(samples)) / samples
        rng.shuffle(unit)
        if scale == "log":
            values = np.exp(math.log(low) + unit * (math.log(high) - math.log(low)))
        else:
            values = low + unit * (high - low)
        if scale == "integer":
            values = np.rint(values).astype(np.int64)
        columns[name] = values
    configurations = []
    for index in range(samples):
        params = {
            name: (
                int(values[index])
                if np.issubdtype(values.dtype, np.integer)
                else float(values[index])
            )
            for name, values in columns.items()
        }
        configurations.append(
            {"config_id": f"{kind}-{index:03d}", "parameters": params}
        )
    payload = {
        "schema_version": "1.0",
        "kind": f"{kind}_search_design",
        "seed": seed,
        "samples": samples,
        "search_space": {
            name: {"low": low, "high": high, "scale": scale}
            for name, (low, high, scale) in space.items()
        },
        "configurations": configurations,
    }
    payload["content_sha256"] = hash_json(payload)
    return payload


def choose_best_config(
    metric_rows: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    tie_ap: float = 0.2,
) -> dict[str, Any]:
    observed_splits = {str(row.get("split", "")).strip().lower() for row in metric_rows}
    if observed_splits != {"val-tune"}:
        raise ValueError(
            f"Configuration search is val-tune only; found splits: {sorted(observed_splits)}"
        )
    by_config: dict[str, list[Mapping[str, Any]]] = {}
    for row in metric_rows:
        by_config.setdefault(str(row["config_id"]), []).append(row)
    candidates: list[dict[str, Any]] = []
    params_by_id = {
        item["config_id"]: item["parameters"] for item in design["configurations"]
    }
    for config_id, rows in by_config.items():
        if config_id not in params_by_id:
            raise ValueError(f"Metric references unknown config_id: {config_id}")
        map_values = [float(row["map_50_95"]) for row in rows]
        runtimes = [float(row.get("runtime_seconds", math.inf)) for row in rows]
        candidates.append(
            {
                "config_id": config_id,
                "map_50_95": float(np.mean(map_values)),
                "runtime_seconds": float(np.mean(runtimes)),
                "parameters": params_by_id[config_id],
                "scene_count": len(rows),
            }
        )
    if not candidates:
        raise ValueError("No tuning metrics were provided")
    best_ap = max(item["map_50_95"] for item in candidates)
    finalists = [item for item in candidates if best_ap - item["map_50_95"] <= tie_ap]
    finalists.sort(
        key=lambda item: (
            item["runtime_seconds"],
            _baseline_distance(item["parameters"]),
            item["config_id"],
        )
    )
    return finalists[0]


def _baseline_distance(parameters: Mapping[str, Any]) -> float:
    baseline = {
        **DEFAULT_MAPPING_CONFIG["baseline"],
        **DEFAULT_MAPPING_CONFIG["coefficients"],
    }
    distance = 0.0
    for name, value in parameters.items():
        reference = float(baseline.get(name, value))
        distance += abs(float(value) - reference) / max(abs(reference), 1e-6)
    return distance


def build_mapping_config(
    global_parameters: Mapping[str, Any],
    prior_parameters: Mapping[str, Any],
    priors_path: str | Path,
    taxonomy_path: str | Path,
    scene_selection_path: str | Path,
) -> dict[str, Any]:
    payload = {
        **DEFAULT_MAPPING_CONFIG,
        "baseline": {**DEFAULT_MAPPING_CONFIG["baseline"], **dict(global_parameters)},
        "coefficients": {
            **DEFAULT_MAPPING_CONFIG["coefficients"],
            **dict(prior_parameters),
        },
        "provenance": {
            "category_priors_sha256": sha256_file(priors_path),
            "taxonomy_sha256": sha256_file(taxonomy_path),
            "scene_selection_sha256": sha256_file(scene_selection_path),
            "tuning_split": "val-tune",
        },
    }
    payload["content_sha256"] = hash_json(payload)
    return payload


def validate_mapping_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != "1.0" or payload.get("kind") != "prior_mapping":
        raise ValueError("Unsupported prior mapping schema")
    if payload.get("provenance", {}).get("tuning_split") != "val-tune":
        raise ValueError("Mapping config must be derived from val-tune")
    expected = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    if not expected or hash_json(unsigned) != expected:
        raise ValueError("Prior mapping content hash mismatch")


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    validate_mapping_config(payload)
    return payload


def build_lock_manifest(
    code_commit: str,
    paths: Mapping[str, str | Path],
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] = (42, 3407, 20260804),
) -> dict[str, Any]:
    if not code_commit.strip():
        raise ValueError("code_commit is required")
    manifest = {
        "schema_version": "1.0",
        "kind": "experiment_lock",
        "code_commit": code_commit,
        "artifacts": {
            name: {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
        "conditions": list(conditions or REGISTERED_CONDITIONS),
        "seeds": [int(seed) for seed in seeds],
        "status": "locked",
    }
    manifest["content_sha256"] = hash_json(manifest)
    return manifest


def build_run_schedule(
    scene_selection_path: str | Path,
    split: str,
    conditions: Sequence[str] | None = REGISTERED_CONDITIONS,
    seeds: Sequence[int] = (42, 3407, 20260804),
    randomization_seed: int = 20260804,
) -> dict[str, Any]:
    conditions = tuple(conditions or REGISTERED_CONDITIONS)
    if split not in {"tune", "locked"}:
        raise ValueError("split must be tune or locked")
    if not conditions or len(conditions) != len(set(conditions)):
        raise ValueError("conditions must be nonempty and unique")
    if not seeds or len(seeds) != len({int(seed) for seed in seeds}):
        raise ValueError("seeds must be nonempty and unique")
    selection = load_json(scene_selection_path)
    scenes = [str(scene) for scene in selection["selection"][split]]
    rng = np.random.default_rng(randomization_seed)
    runs: list[dict[str, Any]] = []
    sequence = 0
    for scene_id in sorted(scenes):
        for run_seed in sorted(int(seed) for seed in seeds):
            order = rng.permutation(len(conditions)).tolist()
            for within_block_order, condition_index in enumerate(order):
                runs.append(
                    {
                        "sequence": sequence,
                        "block": f"{scene_id}/seed-{run_seed}",
                        "scene_id": scene_id,
                        "run_seed": run_seed,
                        "within_block_order": within_block_order,
                        "condition": str(conditions[condition_index]),
                    }
                )
                sequence += 1
    payload = {
        "schema_version": "1.0",
        "kind": "run_schedule",
        "split": f"val-{split}",
        "randomization": "condition order shuffled within scene/seed blocks",
        "randomization_seed": randomization_seed,
        "scene_selection_sha256": sha256_file(scene_selection_path),
        "conditions": list(conditions),
        "seeds": sorted(int(seed) for seed in seeds),
        "runs": runs,
    }
    payload["content_sha256"] = hash_json(payload)
    return payload
