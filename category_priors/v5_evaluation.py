from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .class_first_evaluation import evaluate_class_first_runs
from .taxonomy import Taxonomy
from .v5_candidate import CONDITIONS


V5_OUTPUT_CONDITIONS = ("B1-original", *CONDITIONS)


def evaluate_v5_runs(
    *, scene_manifest: str | Path, gt_dir: str | Path, output_root: str | Path,
    taxonomy: Taxonomy, metrics_output: str | Path, analysis_output: str | Path,
    conditions: Sequence[str], seeds: Sequence[int], scene_ids: Sequence[str],
    reference: str | None = None, treatment: str | None = None,
    bootstrap_samples: int = 10_000, bootstrap_seed: int = 20260804,
    radius_m: float = 0.05, minimum_mapped_fraction: float = 0.90,
    min_region_size: int = 100, split: str = "v5",
) -> dict[str, Any]:
    selected = tuple(str(value) for value in conditions)
    unknown = sorted(set(selected) - set(V5_OUTPUT_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown V5 output conditions: {unknown}")
    payload = evaluate_class_first_runs(
        scene_manifest_path=scene_manifest, gt_dir=gt_dir, output_root=output_root,
        taxonomy=taxonomy, metrics_path=metrics_output, analysis_path=analysis_output,
        conditions=selected, seeds=seeds, scene_ids=scene_ids, reference=reference,
        treatment=treatment, bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed, radius_m=radius_m,
        minimum_mapped_fraction=minimum_mapped_fraction,
        min_region_size=min_region_size, split=split,
        supported_conditions=V5_OUTPUT_CONDITIONS,
    )
    payload["kind"] = "v5_analysis"
    payload["candidate_score_conditions"] = list(CONDITIONS)
    return payload
