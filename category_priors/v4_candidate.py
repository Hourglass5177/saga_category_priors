from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .legacy_prior import category_geometry, empirical_scale_quantile
from .io import write_json


MODES = ("uniform", "class-scale", "class-core", "combined")


@dataclass(frozen=True)
class V4CandidateConfig:
    semantic_threshold: float = 0.7
    min_cluster_size: int = 5
    min_samples: int = 3
    sample_fraction: float = 0.03
    sample_cap: int = 5000
    support_multiplier: int = 4
    core_alpha: float = 0.05
    min_cluster_low: int = 3
    min_cluster_high: int = 20


def uses_class_scale(mode: str) -> bool:
    if mode not in MODES:
        raise ValueError(f"unsupported V4 candidate mode: {mode}")
    return mode in {"class-scale", "combined"}


def uses_class_core(mode: str) -> bool:
    if mode not in MODES:
        raise ValueError(f"unsupported V4 candidate mode: {mode}")
    return mode in {"class-core", "combined"}


def nested_sample_count(
    candidate_count: int,
    min_cluster_size: int,
    config: V4CandidateConfig,
) -> int:
    return int(
        min(
            int(candidate_count),
            config.sample_cap,
            max(
                round(config.sample_fraction * int(candidate_count)),
                config.support_multiplier * int(min_cluster_size),
            ),
        )
    )


def resolve_v4_candidate_parameters(
    priors: Mapping[str, Any],
    mode: str,
    class_name: str,
    candidate_count: int,
    surface_density: float,
    mask_scales: Sequence[float],
    config: V4CandidateConfig = V4CandidateConfig(),
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported V4 candidate mode: {mode}")
    geometry = category_geometry(priors, class_name)
    supported = geometry is not None
    d_c = float(geometry["d_c_m"]) if supported else None
    area = float(geometry["A_c_m2"]) if supported else None

    min_cluster_size = config.min_cluster_size
    if supported and uses_class_core(mode):
        expected_full_support = max(float(surface_density), 0.0) * area
        min_cluster_size = config.min_cluster_low
        for _ in range(2):
            sampled = nested_sample_count(candidate_count, min_cluster_size, config)
            sampling_rate = sampled / max(int(candidate_count), 1)
            min_cluster_size = int(
                np.clip(
                    round(config.core_alpha * expected_full_support * sampling_rate),
                    config.min_cluster_low,
                    config.min_cluster_high,
                )
            )

    sample_count = nested_sample_count(candidate_count, min_cluster_size, config)
    gate_input = (
        empirical_scale_quantile(mask_scales, d_c)
        if supported and uses_class_scale(mode)
        else 1.0
    )
    return {
        "mode": mode,
        "supported": supported,
        "semantic_threshold": config.semantic_threshold,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": config.min_samples,
        "sample_count": int(sample_count),
        "scale_gate_input": float(gate_input),
        "d_c_m": d_c,
        "A_c_m2": area,
        "expected_full_support": (
            float(max(surface_density, 0.0) * area) if area is not None else None
        ),
    }


def class_seed(seed: int, class_name: str) -> int:
    return int(seed) + sum(
        (index + 1) * ord(character)
        for index, character in enumerate(str(class_name))
    )


def nested_permutation(length: int, seed: int, class_name: str) -> np.ndarray:
    return np.random.default_rng(class_seed(seed, class_name)).permutation(int(length))


def write_v4_candidate_capture(
    *, json_path: str | Path, labels_path: str | Path, scene_id: str, seed: int,
    mode: str, git_commit: str, class_names: Sequence[str], affinity_gate: Any,
    branch_labels: Any, semantic_top1: Any, semantic_top1_score: Any,
    semantic_margin: Any, sam_covered: Any, candidates: Sequence[Mapping[str, Any]],
    class_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported V4 candidate mode: {mode}")
    labels = np.asarray(branch_labels, dtype=np.int32)
    semantic = np.asarray(semantic_top1, dtype=np.int16)
    scores = np.asarray(semantic_top1_score, dtype=np.float32)
    margins = np.asarray(semantic_margin, dtype=np.float32)
    covered = np.asarray(sam_covered, dtype=bool)
    if not (labels.shape == semantic.shape == scores.shape == margins.shape == covered.shape):
        raise ValueError("all per-Gaussian V4 candidate arrays must share one shape")
    target = Path(labels_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, branch_labels=labels, semantic_top1=semantic,
        semantic_top1_score=scores, semantic_margin=margins,
        sam_covered_packed=np.packbits(covered),
        point_count=np.asarray([len(labels)], dtype=np.int64),
    )
    payload = {
        "kind": "v4_candidate_capture", "schema_version": "1.0",
        "git_commit": str(git_commit), "scene_id": str(scene_id),
        "seed": int(seed), "mode": mode,
        "class_names": [str(value) for value in class_names],
        "affinity_gate": np.asarray(affinity_gate, dtype=np.float32).reshape(-1).tolist(),
        "labels_npz": str(target), "point_count": len(labels),
        "candidate_count": len(candidates),
        "sam_covered_fraction": float(covered.mean()) if len(covered) else 0.0,
        "candidates": [dict(row) for row in candidates],
        "class_diagnostics": dict(class_diagnostics),
    }
    write_json(json_path, payload)
    return payload
