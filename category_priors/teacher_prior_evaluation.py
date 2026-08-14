from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .class_first_evaluation import evaluate_class_first_runs
from .io import load_json, write_json
from .taxonomy import Taxonomy
from .teacher_prior_runner import (
    TEACHER_PRIOR_CONDITIONS,
    TEACHER_PRIOR_EXPERIMENT_CONDITIONS,
)


def _teacher_structural_diagnostics(
    diagnostics: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for condition, by_seed in diagnostics.items():
        result[condition] = {}
        for seed, summary in by_seed.items():
            numeric = summary.get("numeric_fallback", {})
            teacher_values = {
                name.removeprefix("teacher_prior."): value
                for name, value in numeric.items()
                if name.startswith("teacher_prior.")
            }
            if not teacher_values:
                teacher_values = {
                    name: value
                    for name, value in numeric.items()
                    if not name.startswith(("run.", "runner."))
                }
            result[condition][seed] = {
                "teacher_prior": teacher_values,
                "runner": summary.get("runner", {}),
                "alignment": summary.get("alignment", {}),
            }
    return result


def partition_change_fraction(reference: Sequence[int], treatment: Sequence[int]) -> float:
    """Compare instance partitions after optimally renaming treatment IDs."""
    left = np.asarray(reference, dtype=np.int64)
    right = np.asarray(treatment, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("Point-label arrays must have the same shape")
    left_ids = np.unique(left[left >= 0])
    right_ids = np.unique(right[right >= 0])
    unmatched = np.iinfo(np.int64).min
    renamed = np.full_like(right, unmatched)
    renamed[right < 0] = -1
    if len(left_ids) and len(right_ids):
        valid = (left >= 0) & (right >= 0)
        overlap = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
        left_positions = np.searchsorted(left_ids, left[valid])
        right_positions = np.searchsorted(right_ids, right[valid])
        np.add.at(overlap, (left_positions, right_positions), 1)
        rows, columns = linear_sum_assignment(overlap, maximize=True)
        for row, column in zip(rows, columns, strict=True):
            renamed[right == right_ids[column]] = left_ids[row]
    return float(np.mean(left != renamed)) if len(left) else 0.0


def _intervention_diagnostics(
    output_root: str | Path,
    conditions: Sequence[str],
    seeds: Sequence[int],
    scenes: Sequence[Mapping[str, Any]],
    reference: str | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    root = Path(output_root).resolve()
    result: dict[str, Any] = {}
    for condition in conditions:
        if condition == reference:
            continue
        rows = []
        for seed in seeds:
            for scene in scenes:
                scene_id = str(scene["scene_id"])
                left = load_json(
                    root / reference / scene_id / f"seed-{seed}" / "output.json"
                )
                right = load_json(
                    root / condition / scene_id / f"seed-{seed}" / "output.json"
                )
                left_labels = np.asarray(left["point_labels"], dtype=np.int64)
                right_labels = np.asarray(right["point_labels"], dtype=np.int64)
                left_assigned = int(np.count_nonzero(left_labels >= 0))
                right_assigned = int(np.count_nonzero(right_labels >= 0))
                reference_instances = len(left["instances"])
                treatment_instances = len(right["instances"])
                instance_ratio = (
                    treatment_instances / reference_instances
                    if reference_instances
                    else (1.0 if treatment_instances == 0 else None)
                )
                rows.append(
                    {
                        "scene_id": scene_id,
                        "seed": int(seed),
                        "gaussian_partition_change_fraction": partition_change_fraction(
                            left_labels, right_labels
                        ),
                        "reference_instance_count": reference_instances,
                        "treatment_instance_count": treatment_instances,
                        "instance_count_delta": treatment_instances
                        - reference_instances,
                        "instance_count_ratio": instance_ratio,
                        "reference_coverage": (
                            left_assigned / len(left_labels) if len(left_labels) else 0.0
                        ),
                        "treatment_coverage": (
                            right_assigned / len(right_labels) if len(right_labels) else 0.0
                        ),
                    }
                )
        result[condition] = {
            "runs": rows,
            "mean_partition_change_fraction": float(
                np.mean([row["gaussian_partition_change_fraction"] for row in rows])
            ),
            "max_partition_change_fraction": float(
                np.max([row["gaussian_partition_change_fraction"] for row in rows])
            ),
            "mean_coverage_delta": float(
                np.mean(
                    [
                        row["treatment_coverage"] - row["reference_coverage"]
                        for row in rows
                    ]
                )
            ),
            "mean_instance_count_delta": float(
                np.mean([row["instance_count_delta"] for row in rows])
            ),
            "mean_instance_ratio": (
                float(
                    np.mean(
                        [
                            row["instance_count_ratio"]
                            for row in rows
                            if row["instance_count_ratio"] is not None
                        ]
                    )
                )
                if any(row["instance_count_ratio"] is not None for row in rows)
                else None
            ),
        }
    return result


def evaluate_teacher_prior_runs(
    scene_manifest_path: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    taxonomy: Taxonomy,
    *,
    metrics_path: str | Path,
    analysis_path: str | Path,
    conditions: Sequence[str] | None = None,
    seeds: Sequence[int] = (42,),
    scene_ids: Sequence[str] | None = None,
    scene_list_path: str | Path | None = None,
    selection_path: str | Path | None = None,
    selection_split: str = "tune",
    reference: str | None = None,
    treatment: str | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260804,
    radius_m: float = 0.05,
    minimum_mapped_fraction: float = 0.90,
    min_region_size: int = 100,
    split: str = "teacher-prior",
) -> dict[str, Any]:
    payload = evaluate_class_first_runs(
        scene_manifest_path=scene_manifest_path,
        gt_dir=gt_dir,
        output_root=output_root,
        taxonomy=taxonomy,
        metrics_path=metrics_path,
        analysis_path=analysis_path,
        conditions=conditions or TEACHER_PRIOR_EXPERIMENT_CONDITIONS,
        seeds=seeds,
        scene_ids=scene_ids,
        scene_list_path=scene_list_path,
        selection_path=selection_path,
        selection_split=selection_split,
        reference=reference,
        treatment=treatment,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        radius_m=radius_m,
        minimum_mapped_fraction=minimum_mapped_fraction,
        min_region_size=min_region_size,
        split=split,
        supported_conditions=tuple(TEACHER_PRIOR_CONDITIONS),
    )
    payload["kind"] = "teacher_prior_analysis"
    payload["condition_modes"] = {
        condition: TEACHER_PRIOR_CONDITIONS[condition]
        for condition in payload["conditions"]
    }
    payload["technical_replicate_aggregation"] = (
        "Seeds are averaged within each physical scene and bootstrap resample."
    )
    payload["structural_diagnostics"] = _teacher_structural_diagnostics(
        payload["diagnostics"]
    )
    payload["intervention_diagnostics"] = _intervention_diagnostics(
        output_root,
        payload["conditions"],
        payload["technical_replicates"],
        payload["scenes"],
        reference,
    )
    payload["best_median_worst"] = payload.get("qualitative_cases")
    write_json(analysis_path, payload)
    return payload
