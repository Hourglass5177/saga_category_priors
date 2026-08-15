from __future__ import annotations

"""V5 proposal-bank primitives.

The module deliberately contains no final-label mutation.  Candidate generation
may be expensive and GPU-backed, but all acceptance and category-prior effects
are expressed as compact, CPU-replayable proposal records.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import write_json


SOURCES = ("codebook", "multiview")
CONDITIONS = ("U00-uniform", "D10-size", "D01-core", "D11-combined")
# Keep the candidate set identical to the 32-label SAGA assets and the teacher
# branch; the ScanNet taxonomy mapping is applied only at official evaluation.
SAGA20 = (
    "chair", "table", "plant", "tv", "painting", "sofa", "cabinet", "bed",
    "socket", "book", "switch", "door", "window", "lamp", "speaker", "fan",
    "refrigerator", "cup", "phone", "trash can",
)


@dataclass(frozen=True)
class V5CandidateConfig:
    semantic_threshold: float = 0.7
    min_cluster_size: int = 5
    min_samples: int = 5
    sample_cap: int = 5000
    feature_ratio: float = 0.5
    spatial_ratio: float = 0.3
    semantic_ratio: float = 0.2
    epsilon: float = 0.01
    assignment_threshold: float = 0.3
    core_assignment_threshold: float = 0.70
    min_multiview_observations: int = 3
    min_multiview_ratio: float = 0.60
    min_multiview_margin: float = 0.10

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def validate_source(source: str) -> str:
    if source not in SOURCES:
        raise ValueError(f"unsupported V5 candidate source: {source}")
    return source


def validate_condition(condition: str) -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported V5 replay condition: {condition}")
    return condition


def uses_size(condition: str) -> bool:
    return validate_condition(condition) in {"D10-size", "D11-combined"}


def uses_core(condition: str) -> bool:
    return validate_condition(condition) in {"D01-core", "D11-combined"}


def class_seed(seed: int, class_name: str) -> int:
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(class_name))


def nested_permutation(length: int, seed: int, class_name: str) -> np.ndarray:
    """One deterministic order per scene seed/class; every arm uses its prefix."""
    return np.random.default_rng(class_seed(seed, class_name)).permutation(int(length))


def resolve_v5_candidate_parameters(
    candidate_count: int, config: V5CandidateConfig = V5CandidateConfig(),
) -> dict[str, Any]:
    """The V5 bank has one frozen candidate pool across every replay arm."""
    return {
        "semantic_threshold": float(config.semantic_threshold),
        "min_cluster_size": int(config.min_cluster_size),
        "min_samples": int(config.min_samples),
        "sample_count": int(min(int(candidate_count), int(config.sample_cap))),
        "core_assignment_threshold": float(config.core_assignment_threshold),
        "feature_ratio": float(config.feature_ratio),
        "spatial_ratio": float(config.spatial_ratio),
        "semantic_ratio": float(config.semantic_ratio),
        "epsilon": float(config.epsilon),
        "assignment_threshold": float(config.assignment_threshold),
    }


def normalized_top1(
    point_features: np.ndarray,
    label_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full-codebook cosine winner, confidence, and winner-minus-runner-up margin."""
    points = np.asarray(point_features, dtype=np.float64)
    labels = np.asarray(label_features, dtype=np.float64)
    if points.ndim != 2 or labels.ndim != 2 or points.shape[1] != labels.shape[1]:
        raise ValueError("point and label features must be compatible matrices")
    point_norm = np.linalg.norm(points, axis=1, keepdims=True)
    label_norm = np.linalg.norm(labels, axis=1, keepdims=True)
    similarities = (points / np.maximum(point_norm, 1e-12)) @ (labels / np.maximum(label_norm, 1e-12)).T
    winner = similarities.argmax(axis=1).astype(np.int16)
    score = similarities[np.arange(len(winner)), winner].astype(np.float32)
    if similarities.shape[1] < 2:
        margin = np.zeros(len(winner), dtype=np.float32)
    else:
        top_two = np.partition(similarities, -2, axis=1)[:, -2:]
        margin = (top_two[:, 1] - top_two[:, 0]).astype(np.float32)
    return winner, score, margin


def source_masks(
    *, source: str, winner: np.ndarray, score: np.ndarray,
    class_indices: Mapping[str, int], multiview_views: np.ndarray | None = None,
    multiview_ratio: np.ndarray | None = None, multiview_margin: np.ndarray | None = None,
    config: V5CandidateConfig = V5CandidateConfig(),
) -> dict[int, np.ndarray]:
    """Exclusive SAGA20 masks after an all-codebook winner has been fixed."""
    validate_source(source)
    winner = np.asarray(winner)
    score = np.asarray(score, dtype=np.float64)
    if winner.shape != score.shape:
        raise ValueError("winner and score must share one shape")
    if source == "multiview":
        if multiview_views is None or multiview_ratio is None or multiview_margin is None:
            raise ValueError("multiview source requires view counts, ratio, and margin")
        eligible = (
            (np.asarray(multiview_views) >= config.min_multiview_observations)
            & (np.asarray(multiview_ratio, dtype=np.float64) >= config.min_multiview_ratio)
            & (np.asarray(multiview_margin, dtype=np.float64) >= config.min_multiview_margin)
        )
    else:
        eligible = score >= config.semantic_threshold
    return {
        int(index): eligible & (winner == int(index))
        for name, index in class_indices.items()
        if name in SAGA20
    }


def _q(node: Mapping[str, Any], field: str, quantile: str) -> float | None:
    try:
        value = float(node["shrunk"]["geometry"][field][quantile])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def prior_geometry(priors: Mapping[str, Any], class_name: str) -> dict[str, float | None]:
    """Return log extent and area statistics, falling back to frozen global values."""
    categories = priors.get("categories", {})
    node = categories.get(class_name) if isinstance(categories, Mapping) else None
    if not isinstance(node, Mapping):
        node = priors.get("global", {})
    if not isinstance(node, Mapping):
        return {key: None for key in ("diag_q50", "diag_q75", "area_q50")}
    return {
        "diag_q50": _q(node, "log_bbox_diag_m", "q50"),
        "diag_q75": _q(node, "log_bbox_diag_m", "q75"),
        "area_q50": _q(node, "log_surface_area_m2", "q50"),
    }


def _extent_stats(priors: Mapping[str, Any], class_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Frozen sorted long/mid/short train extents, with global fallback."""
    categories = priors.get("categories", {})
    node = categories.get(class_name) if isinstance(categories, Mapping) else None
    if not isinstance(node, Mapping):
        node = priors.get("global", {})
    if not isinstance(node, Mapping):
        return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
    fields = ("log_extent_short_m", "log_extent_mid_m", "log_extent_long_m")
    q50 = [_q(node, field, "q50") for field in fields]
    q75 = [_q(node, field, "q75") for field in fields]
    if any(value is None for value in q50 + q75):
        geometry = prior_geometry(priors, class_name)
        diagonal_q50 = geometry["diag_q50"]
        diagonal_q75 = geometry["diag_q75"]
        if diagonal_q50 is None or diagonal_q75 is None:
            return np.zeros(3, dtype=np.float64), np.ones(3, dtype=np.float64)
        return (
            np.full(3, diagonal_q50, dtype=np.float64),
            np.full(3, diagonal_q75, dtype=np.float64),
        )
    return np.asarray(q50, dtype=np.float64), np.asarray(q75, dtype=np.float64)


def size_compatibility(
    metric_extents_m: Sequence[float], priors: Mapping[str, Any], class_name: str,
    *, global_only: bool = False,
) -> float:
    name = "__missing__" if global_only else class_name
    if global_only:
        node = priors.get("global", {})
        wrapped = {"categories": {name: node}, "global": node}
    else:
        wrapped = priors
    q50, q75 = _extent_stats(wrapped, name)
    extents = np.sort(np.maximum(np.asarray(metric_extents_m, dtype=np.float64), 1e-12))
    if extents.shape != (3,):
        raise ValueError("metric extents must have exactly three values")
    log_extents = np.log(extents)
    z = np.maximum(0.0, log_extents - q50) / np.maximum(q75 - q50, 1e-6)
    return float(np.exp(-0.5 * np.mean(np.minimum(z * z, 25.0))))


def core_compatibility(
    core_point_count: int, local_surface_density: float | None,
    priors: Mapping[str, Any], class_name: str, *, global_only: bool = False,
) -> float:
    geometry = prior_geometry(priors, "__missing__" if global_only else class_name)
    if global_only:
        global_node = priors.get("global", {})
        geometry = {
            "diag_q50": _q(global_node, "log_bbox_diag_m", "q50"),
            "diag_q75": _q(global_node, "log_bbox_diag_m", "q75"),
            "area_q50": _q(global_node, "log_surface_area_m2", "q50"),
        }
    area_q50 = geometry["area_q50"]
    density = float(local_surface_density or 0.0)
    if area_q50 is None or density <= 0.0 or not math.isfinite(density):
        return 1.0
    expected = max(3.0, 0.05 * density * math.exp(area_q50))
    return float(np.clip(int(core_point_count) / expected, 0.0, 1.0))


def evidence_score(candidate: Mapping[str, Any]) -> float:
    vote = candidate.get("vote", {})
    return float(np.clip(
        float(vote.get("branch_class_ratio", 0.0))
        * float(candidate.get("assignment_confidence_mean", 0.0))
        * float(candidate.get("hdbscan_membership_mean", 0.0) or 0.0)
        * (0.5 + 0.5 * float(candidate.get("hdbscan_persistence", 0.0) or 0.0)),
        0.0, 1.0,
    ))


def score_candidate(
    candidate: Mapping[str, Any], priors: Mapping[str, Any], condition: str,
) -> dict[str, float]:
    use_global_size = not uses_size(condition)
    use_global_core = not uses_core(condition)
    class_name = str(candidate["branch_class"])
    g = size_compatibility(
        candidate.get("metric_extents_m", [0.0, 0.0, 0.0]), priors, class_name,
        global_only=use_global_size,
    )
    c = core_compatibility(
        int(candidate.get("core_assignment_points", 0)),
        candidate.get("local_surface_density"), priors, class_name,
        global_only=use_global_core,
    )
    e = evidence_score(candidate)
    return {"E": e, "G": g, "C": c, "score": float(e * (g if uses_size(condition) else 1.0) * (c if uses_core(condition) else 1.0))}


def base_evidence_reason(candidate: Mapping[str, Any]) -> str | None:
    vote = candidate.get("vote", {})
    if float(vote.get("branch_class_ratio", 0.0)) < 0.60:
        return "weak_branch_vote"
    if float(vote.get("branch_class_ratio", 0.0)) <= float(vote.get("background_ratio", 1.0)):
        return "background_vote"
    if float(candidate.get("assignment_confidence_mean", 0.0)) < 0.30:
        return "weak_assignment"
    if float(candidate.get("hdbscan_membership_mean", 0.0) or 0.0) < 0.70:
        return "weak_membership"
    if float(candidate.get("hdbscan_persistence", 0.0) or 0.0) < 0.03:
        return "weak_persistence"
    if int(candidate.get("core_assignment_points", 0)) < 100:
        return "insufficient_core"
    return None


def write_v5_proposal_capture(
    *, json_path: str | Path, labels_path: str | Path, scene_id: str, seed: int,
    source: str, git_commit: str, class_names: Sequence[str], branch_labels: Any,
    core_labels: Any, assignment_confidence: Any, semantic_winner: Any,
    semantic_score: Any, semantic_margin: Any, source_view_count: Any,
    source_vote_ratio: Any, source_vote_margin: Any,
    candidates: Sequence[Mapping[str, Any]], class_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    validate_source(source)
    arrays = {
        "branch_labels": np.asarray(branch_labels, dtype=np.int32),
        "core_labels": np.asarray(core_labels, dtype=np.int32),
        "assignment_confidence": np.asarray(assignment_confidence, dtype=np.float32),
        "semantic_winner": np.asarray(semantic_winner, dtype=np.int16),
        "semantic_score": np.asarray(semantic_score, dtype=np.float32),
        "semantic_margin": np.asarray(semantic_margin, dtype=np.float32),
        "source_view_count": np.asarray(source_view_count, dtype=np.int16),
        "source_vote_ratio": np.asarray(source_vote_ratio, dtype=np.float32),
        "source_vote_margin": np.asarray(source_vote_margin, dtype=np.float32),
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all V5 per-Gaussian arrays must share one shape")
    target = Path(labels_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    payload = {
        "kind": "v5_proposal_bank", "schema_version": "1.0",
        "git_commit": str(git_commit), "scene_id": str(scene_id), "seed": int(seed),
        "source": source, "class_names": [str(value) for value in class_names],
        "point_count": int(len(arrays["branch_labels"])), "labels_npz": str(target),
        "candidate_count": int(len(candidates)), "candidates": [dict(row) for row in candidates],
        "class_diagnostics": dict(class_diagnostics),
    }
    write_json(json_path, payload)
    return payload
