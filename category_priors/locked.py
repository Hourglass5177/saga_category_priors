from __future__ import annotations

import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import OVERLAPS, PROTOCOL_VERSION
from .io import load_json
from .mapping import REGISTERED_CONDITIONS, validate_mapping_config
from .priors import validate_priors
from .taxonomy import load_taxonomy


SMALL_CATEGORIES = (
    "cup",
    "switch",
    "book",
    "phone",
    "speaker",
    "lamp",
    "trash can",
)

FACTORIAL_BITS = {
    "P000-B2": (0, 0, 0),
    "P001-small": (0, 0, 1),
    "P010-smooth": (0, 1, 0),
    "P011-smooth-small": (0, 1, 1),
    "P100-size": (1, 0, 0),
    "P101-size-small": (1, 0, 1),
    "P110-size-smooth": (1, 1, 0),
    "P111-combined": (1, 1, 1),
}


def _display_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def summarize_priors(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_priors(payload)
    return {
        "datasets": list(payload["provenance"]["datasets"]),
        "splits": list(payload["provenance"]["splits"]),
        "row_count": int(payload["provenance"]["row_count"]),
        "normalization": dict(payload["normalization"]),
        "fit_config": dict(payload["fit_config"]),
        # The plan freezes the small, readable prior table itself.  Category
        # names alone would not detect a changed size/smoothness statistic.
        "categories": deepcopy(dict(payload["categories"])),
        "fallback": dict(payload["fallback"]),
    }


def summarize_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_mapping_config(payload)
    return {
        "baseline": dict(payload["baseline"]),
        "coefficients": dict(payload["coefficients"]),
        "fixed": dict(payload["fixed"]),
    }


def assess_seed_sensitivity(
    rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str] = ("P000-B2", "P111-combined"),
    seeds: Sequence[int] = (42, 3407, 20260804),
    maximum_range: float = 0.002,
) -> dict[str, Any]:
    """Apply the preregistered tune-only rule for the locked seed policy."""
    expected_conditions = tuple(str(value) for value in conditions)
    expected_seeds = tuple(sorted(int(value) for value in seeds))
    by_key: dict[tuple[str, int], float] = {}
    protocol_versions = {str(row.get("protocol_version", "")) for row in rows}
    if protocol_versions != {PROTOCOL_VERSION}:
        raise ValueError(
            "Seed-sensitivity metrics must all use "
            f"{PROTOCOL_VERSION}; found={sorted(protocol_versions)}"
        )
    for row in rows:
        if str(row.get("split", "")).lower() != "val-tune":
            raise ValueError("Seed sensitivity accepts val-tune metrics only")
        condition = str(row["condition"])
        run_seed = int(row["run_seed"])
        key = (condition, run_seed)
        if key in by_key:
            raise ValueError(f"Duplicate seed-sensitivity metric: {key}")
        value = float(row["map_50_95"])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite seed-sensitivity metric: {key}")
        by_key[key] = value

    expected = {
        (condition, run_seed)
        for condition in expected_conditions
        for run_seed in expected_seeds
    }
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    if missing or extra:
        raise ValueError(
            f"Seed-sensitivity grid mismatch; missing={missing}, extra={extra}"
        )

    ranges = {
        condition: max(by_key[(condition, seed)] for seed in expected_seeds)
        - min(by_key[(condition, seed)] for seed in expected_seeds)
        for condition in expected_conditions
    }
    reference, treatment = expected_conditions
    deltas = {
        str(seed): by_key[(treatment, seed)] - by_key[(reference, seed)]
        for seed in expected_seeds
    }
    nonzero_signs = {int(np.sign(value)) for value in deltas.values()}
    direction_consistent = len(nonzero_signs) == 1 and 0 not in nonzero_signs
    stable = all(value <= maximum_range for value in ranges.values())
    selected_seeds = [42] if stable and direction_consistent else list(expected_seeds)
    return {
        "schema_version": "1.0",
        "kind": "seed_sensitivity_decision",
        "split": "val-tune",
        "conditions": list(expected_conditions),
        "audited_seeds": list(expected_seeds),
        "maximum_allowed_range": maximum_range,
        "ranges": ranges,
        "treatment_minus_reference": deltas,
        "direction_consistent": direction_consistent,
        "stable": stable,
        "selected_locked_seeds": selected_seeds,
        "decision": "single-seed" if len(selected_seeds) == 1 else "three-seed",
    }


def build_locked_plan(
    locked_scenes_path: str | Path,
    priors_path: str | Path,
    mapping_path: str | Path,
    taxonomy_path: str | Path | None,
    code_commit: str,
    seeds: Sequence[int],
    *,
    conditions: Sequence[str] = REGISTERED_CONDITIONS,
    randomization_seed: int = 20260804,
) -> dict[str, Any]:
    """Build the small, human-readable source of truth for locked evaluation."""
    selection = load_json(locked_scenes_path)
    if selection.get("kind") != "locked_evaluation_scenes":
        raise ValueError("Expected locked_evaluation_scenes")
    mapping = load_json(mapping_path)
    validate_mapping_config(mapping)
    priors = load_json(priors_path)
    validate_priors(priors)
    taxonomy = load_taxonomy(taxonomy_path)
    scene_items = [
        {
            "scene_id": str(item["scene_id"]),
            "physical_scene_id": str(item["physical_scene_id"]),
        }
        for item in selection["scenes"]
    ]
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "locked_plan",
        "status": "frozen",
        "split": "val-locked",
        "code_commit": code_commit.strip(),
        "scenes": scene_items,
        "conditions": [str(value) for value in conditions],
        "seeds": sorted(int(value) for value in seeds),
        "randomization_seed": int(randomization_seed),
        "inputs": {
            "locked_scenes": _display_path(locked_scenes_path),
            "category_priors": _display_path(priors_path),
            "prior_mapping": _display_path(mapping_path),
            "taxonomy": _display_path(
                taxonomy_path or Path(__file__).with_name("default_taxonomy.json")
            ),
        },
        "parameters": summarize_mapping(mapping),
        "priors": summarize_priors(priors),
        "taxonomy": {
            "benchmark_name": taxonomy.benchmark_name,
            "canonical_classes": list(taxonomy.canonical_classes),
        },
        "analysis": {
            "primary_comparison": {
                "reference": "P000-B2",
                "treatment": "P111-combined",
            },
            "primary_metric": "map_50_95",
            "overlaps": [round(value, 2) for value in np.arange(0.50, 0.96, 0.05)],
            "also_report": [
                "map_0.25",
                "map_0.50",
                "runtime_seconds",
                "first_attempt_failure_rate",
                "final_failure_rate",
            ],
            "min_region_size": 100,
            "radius_m": 0.05,
            "minimum_mapped_fraction": 0.90,
            "bootstrap": {"method": "exp1_scene_weights", "samples": 10_000},
            "permutation": {"method": "paired_scene_swap", "samples": 50_000},
            "alpha": 0.05,
            "factorial_bits": {name: list(bits) for name, bits in FACTORIAL_BITS.items()},
            "factorial_multiplicity": "holm-7",
            "small_categories": list(SMALL_CATEGORIES),
        },
    }
    validate_locked_plan(payload)
    return payload


def validate_locked_plan(payload: Mapping[str, Any]) -> None:
    if payload.get("kind") != "locked_plan" or payload.get("status") != "frozen":
        raise ValueError("Expected a frozen locked_plan")
    if payload.get("split") != "val-locked":
        raise ValueError("Locked plan split must be val-locked")
    if not str(payload.get("code_commit", "")).strip():
        raise ValueError("Locked plan requires a Git commit")
    scenes = list(payload.get("scenes", []))
    scene_ids = [str(item["scene_id"]) for item in scenes]
    physical_ids = [str(item["physical_scene_id"]) for item in scenes]
    if len(scene_ids) != 48 or len(set(scene_ids)) != 48:
        raise ValueError("Locked plan requires exactly 48 unique scans")
    if len(set(physical_ids)) != 48:
        raise ValueError("Locked plan requires 48 unique physical scenes")
    conditions = [str(value) for value in payload.get("conditions", [])]
    if conditions != list(REGISTERED_CONDITIONS):
        raise ValueError("Locked plan must contain the 12 registered conditions")
    seeds = [int(value) for value in payload.get("seeds", [])]
    if seeds not in ([42], [42, 3407, 20260804]):
        raise ValueError("Locked seeds must follow the preregistered decision rule")
    taxonomy = payload.get("taxonomy", {})
    classes = [str(value) for value in taxonomy.get("canonical_classes", [])]
    registered_taxonomy = load_taxonomy()
    if (
        taxonomy.get("benchmark_name") != registered_taxonomy.benchmark_name
        or classes != list(registered_taxonomy.canonical_classes)
    ):
        raise ValueError("Locked plan requires the registered ordered SAGA20 taxonomy")
    if payload.get("priors", {}).get("splits") != ["train"]:
        raise ValueError("Locked plan priors must be fitted from train only")
    configured_overlaps = tuple(
        float(value) for value in payload.get("analysis", {}).get("overlaps", [])
    )
    if configured_overlaps != tuple(float(value) for value in OVERLAPS):
        raise ValueError("Locked plan overlaps must match the official protocol")


def expand_locked_runs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_locked_plan(payload)
    rng = np.random.default_rng(int(payload["randomization_seed"]))
    conditions = [str(value) for value in payload["conditions"]]
    runs: list[dict[str, Any]] = []
    sequence = 0
    for scene in sorted(payload["scenes"], key=lambda item: str(item["scene_id"])):
        scene_id = str(scene["scene_id"])
        for run_seed in sorted(int(value) for value in payload["seeds"]):
            order = rng.permutation(len(conditions)).tolist()
            for within_block_order, condition_index in enumerate(order):
                condition = conditions[condition_index]
                runs.append(
                    {
                        "sequence": sequence,
                        "run_id": f"{scene_id}/seed-{run_seed}/{condition}",
                        "block": f"{scene_id}/seed-{run_seed}",
                        "scene_id": scene_id,
                        "physical_scene_id": str(scene["physical_scene_id"]),
                        "run_seed": run_seed,
                        "within_block_order": within_block_order,
                        "condition": condition,
                    }
                )
                sequence += 1
    return runs
