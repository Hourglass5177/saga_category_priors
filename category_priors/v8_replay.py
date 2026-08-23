from __future__ import annotations

"""Pure category-prior scoring and deterministic replay for V8 banks."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .v8_objects import CandidateBank


CONDITIONS = ("U00", "D10", "D01", "D11")


@dataclass(frozen=True)
class ReplayResult:
    point_labels: np.ndarray
    instances: dict[str, dict[str, str]]
    instance_metadata: dict[str, dict[str, Any]]
    candidate_scores: tuple[dict[str, Any], ...]
    accepted_candidate_ids: tuple[int, ...]
    rejected_candidate_ids: tuple[int, ...]
    suppressed_candidate_ids: tuple[int, ...]
    dropped_small_candidate_ids: tuple[int, ...]


def _global_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise ValueError("category priors are missing a global shrunk node")
    return node


def _prior_node(
    priors: Mapping[str, Any], class_name: str, use_class: bool
) -> Mapping[str, Any]:
    if use_class:
        categories = priors.get("categories", {})
        node = categories.get(class_name) if isinstance(categories, Mapping) else None
        if isinstance(node, Mapping) and isinstance(node.get("shrunk"), Mapping):
            return node
    return _global_node(priors)


def _geometry(node: Mapping[str, Any]) -> Mapping[str, Any]:
    shrunk = node.get("shrunk")
    geometry = shrunk.get("geometry") if isinstance(shrunk, Mapping) else None
    if not isinstance(geometry, Mapping):
        raise ValueError("prior node has invalid shrunk geometry")
    return geometry


def size_compatibility(candidate: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    extents = np.sort(
        np.maximum(np.asarray(candidate["metric_extents_m"], dtype=np.float64), 1e-9)
    )
    if extents.shape != (3,):
        raise ValueError("candidate metric_extents_m must contain three values")
    geometry = _geometry(node)
    fields = ("log_extent_short_m", "log_extent_mid_m", "log_extent_long_m")
    z: list[float] = []
    for extent, field in zip(extents, fields):
        summary = geometry.get(field)
        if not isinstance(summary, Mapping):
            return 1.0
        q50 = float(summary["q50"])
        q75 = float(summary["q75"])
        z.append(max(0.0, math.log(float(extent)) - q50) / max(q75 - q50, 1e-6))
    return float(math.exp(-0.5 * float(np.mean(np.minimum(np.square(z), 25.0)))))


def core_compatibility(candidate: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    area = _geometry(node).get("log_surface_area_m2")
    if not isinstance(area, Mapping):
        return 1.0
    density = max(float(candidate.get("local_surface_density", 0.0)), 0.0)
    minimum = max(3.0, 0.05 * density * math.exp(float(area["q50"])))
    return float(min(1.0, float(candidate["core_point_count"]) / minimum))


def score_candidate(
    candidate: Mapping[str, Any],
    priors: Mapping[str, Any],
    condition: str,
) -> dict[str, float]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V8 replay condition: {condition}")
    class_name = str(candidate["branch_class"])
    use_class_size = condition in {"D10", "D11"}
    use_class_core = condition in {"D01", "D11"}
    g = size_compatibility(
        candidate, _prior_node(priors, class_name, use_class_size)
    )
    c = core_compatibility(
        candidate, _prior_node(priors, class_name, use_class_core)
    )
    q = float(candidate["base_score"])
    return {
        "Q": q,
        "G": g,
        "C": c,
        "score": float(np.clip(q * g * c, 0.0, 1.0)),
    }


def _point_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_ids = np.unique(np.asarray(left, dtype=np.int64))
    right_ids = np.unique(np.asarray(right, dtype=np.int64))
    if not len(left_ids) and not len(right_ids):
        return 0.0
    intersection = len(np.intersect1d(left_ids, right_ids, assume_unique=True))
    return float(intersection / (len(left_ids) + len(right_ids) - intersection))


def greedy_same_class_core_nms(
    candidates: Sequence[Mapping[str, Any]],
    score_by_id: Mapping[int, float],
    core_ids_by_candidate: Mapping[int, np.ndarray],
    *,
    iou_threshold: float = 0.50,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Apply score-ordered NMS only between candidates of the same class."""
    rows = {int(row["candidate_id"]): row for row in candidates}
    if len(rows) != len(candidates):
        raise ValueError("candidate_id values must be unique")
    ordered = sorted(score_by_id, key=lambda cid: (-float(score_by_id[cid]), int(cid)))
    kept: list[int] = []
    suppressed: list[int] = []
    for candidate_id in ordered:
        if candidate_id not in rows or candidate_id not in core_ids_by_candidate:
            raise ValueError(f"missing candidate or core mask for id {candidate_id}")
        class_name = str(rows[candidate_id]["branch_class"])
        duplicate = any(
            str(rows[kept_id]["branch_class"]) == class_name
            and _point_iou(
                core_ids_by_candidate[candidate_id], core_ids_by_candidate[kept_id]
            )
            >= float(iou_threshold)
            for kept_id in kept
        )
        if duplicate:
            suppressed.append(candidate_id)
        else:
            kept.append(candidate_id)
    return tuple(kept), tuple(suppressed)


def assign_unique_gaussians(
    point_count: int,
    accepted_candidate_ids: Sequence[int],
    score_by_id: Mapping[int, float],
    full_ids_by_candidate: Mapping[int, np.ndarray],
    *,
    min_points: int = 10,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Assign overlaps to the highest score, then remove undersized instances."""
    if point_count < 0:
        raise ValueError("point_count must be non-negative")
    owner = np.full(point_count, -1, dtype=np.int32)
    ordered = sorted(
        map(int, accepted_candidate_ids),
        key=lambda cid: (-float(score_by_id[cid]), cid),
    )
    for candidate_id in ordered:
        ids = np.unique(np.asarray(full_ids_by_candidate[candidate_id], dtype=np.int64))
        if np.any(ids < 0) or np.any(ids >= point_count):
            raise ValueError(f"candidate {candidate_id} has an out-of-range Gaussian id")
        unowned = ids[owner[ids] < 0]
        owner[unowned] = candidate_id
    kept: list[int] = []
    dropped: list[int] = []
    for candidate_id in ordered:
        if np.count_nonzero(owner == candidate_id) < int(min_points):
            owner[owner == candidate_id] = -1
            dropped.append(candidate_id)
        else:
            kept.append(candidate_id)
    return owner, tuple(kept), tuple(dropped)


def replay_candidates(
    bank: CandidateBank,
    priors: Mapping[str, Any],
    condition: str,
    *,
    acceptance_threshold: float = 0.20,
    nms_core_iou: float = 0.50,
    min_points: int = 10,
) -> ReplayResult:
    """Replay one frozen bank without changing any candidate construction."""
    core_labels = np.asarray(bank.core_candidate_id, dtype=np.int32)
    if core_labels.shape != (int(bank.point_count),):
        raise ValueError("core candidate labels must match bank.point_count")
    rows = tuple(dict(row) for row in bank.candidates)
    row_by_id = {int(row["candidate_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("candidate_id values must be unique")
    expected_ids = set(range(len(rows)))
    if set(row_by_id) != expected_ids:
        raise ValueError("V8 bank candidate_id values must be contiguous from zero")
    if len(bank.full_ids) != len(rows) or len(bank.core_ids) != len(rows):
        raise ValueError("V8 bank ragged masks must match candidate rows")
    full_ids = {
        candidate_id: np.asarray(bank.full_ids[candidate_id], dtype=np.int32)
        for candidate_id in row_by_id
    }
    core_ids = {
        candidate_id: np.asarray(bank.core_ids[candidate_id], dtype=np.int32)
        for candidate_id in row_by_id
    }
    score_parts = {
        candidate_id: score_candidate(row, priors, condition)
        for candidate_id, row in row_by_id.items()
    }
    passing_scores = {
        candidate_id: parts["score"]
        for candidate_id, parts in score_parts.items()
        if parts["score"] >= float(acceptance_threshold)
    }
    rejected = tuple(
        sorted(set(row_by_id).difference(passing_scores))
    )
    accepted, suppressed = greedy_same_class_core_nms(
        rows,
        passing_scores,
        core_ids,
        iou_threshold=float(nms_core_iou),
    )
    owner, kept, dropped = assign_unique_gaussians(
        int(bank.point_count),
        accepted,
        passing_scores,
        full_ids,
        min_points=int(min_points),
    )

    output = np.full(len(owner), -1, dtype=np.int32)
    instances: dict[str, dict[str, str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    # Instance IDs are a serialization detail, not part of the prior score.
    # Keep them stable by candidate identity so a score-only rank swap between
    # disjoint candidates cannot masquerade as an ownership intervention.
    stable_kept = tuple(sorted(map(int, kept)))
    for instance_id, candidate_id in enumerate(stable_kept):
        output[owner == candidate_id] = instance_id
        row = row_by_id[candidate_id]
        class_name = str(row["branch_class"])
        instances[str(instance_id)] = {"class": class_name}
        metadata[str(instance_id)] = {
            "class": class_name,
            "score": float(passing_scores[candidate_id]),
            "candidate_id": int(candidate_id),
            "source": "v8_object_bank",
        }
    candidate_scores = tuple(
        {
            "candidate_id": candidate_id,
            "class": str(row_by_id[candidate_id]["branch_class"]),
            **score_parts[candidate_id],
        }
        for candidate_id in sorted(row_by_id)
    )
    return ReplayResult(
        point_labels=output,
        instances=instances,
        instance_metadata=metadata,
        candidate_scores=candidate_scores,
        accepted_candidate_ids=stable_kept,
        rejected_candidate_ids=rejected,
        suppressed_candidate_ids=suppressed,
        dropped_small_candidate_ids=dropped,
    )
