from __future__ import annotations

"""Frozen-bank V9 category-prior scoring and strict prediction export."""

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .prediction_contract import (
    normalize_prediction,
    normalize_score,
    validate_prediction_contract,
)
from .v9_objectbank import CandidateBank, SAGA20


CONDITION_FACTORS: dict[str, tuple[bool, bool, bool]] = {
    "U000": (False, False, False),
    "D100": (True, False, False),
    "D010": (False, True, False),
    "D001": (False, False, True),
    "D110": (True, True, False),
    "D101": (True, False, True),
    "D011": (False, True, True),
    "D111": (True, True, True),
}


@dataclass(frozen=True)
class FinalPrediction:
    point_labels: np.ndarray
    instances: dict[str, dict[str, Any]]
    instance_metadata: dict[str, dict[str, Any]]

    def output_payload(self) -> dict[str, Any]:
        return {
            "point_labels": self.point_labels.tolist(),
            "instances": {key: dict(value) for key, value in self.instances.items()},
        }

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "kind": "saga_instance_metadata",
            "instances": {
                key: dict(value) for key, value in self.instance_metadata.items()
            },
        }


@dataclass(frozen=True)
class ReplayResult:
    prediction: FinalPrediction
    candidate_scores: tuple[dict[str, Any], ...]
    accepted_candidate_ids: tuple[int, ...]
    rejected_candidate_ids: tuple[int, ...]
    suppressed_candidate_ids: tuple[int, ...]
    dropped_small_candidate_ids: tuple[int, ...]

    @property
    def point_labels(self) -> np.ndarray:
        return self.prediction.point_labels

    @property
    def instances(self) -> dict[str, dict[str, Any]]:
        return self.prediction.instances

    @property
    def instance_metadata(self) -> dict[str, dict[str, Any]]:
        return self.prediction.instance_metadata


def _global_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise ValueError("category priors are missing a global shrunk node")
    return node


def _prior_node(
    priors: Mapping[str, Any], class_name: str, use_class: bool
) -> Mapping[str, Any]:
    if use_class:
        categories = priors.get("categories")
        if isinstance(categories, Mapping):
            node = categories.get(class_name)
            if isinstance(node, Mapping) and isinstance(node.get("shrunk"), Mapping):
                return node
    return _global_node(priors)


def _section(node: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    shrunk = node.get("shrunk")
    section = shrunk.get(name) if isinstance(shrunk, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError(f"prior node has invalid shrunk {name}")
    return section


def size_compatibility(candidate: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    extents = np.sort(
        np.maximum(np.asarray(candidate["metric_extents_m"], dtype=np.float64), 1e-9)
    )
    if extents.shape != (3,):
        raise ValueError("candidate metric_extents_m must contain three values")
    geometry = _section(node, "geometry")
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


def support_compatibility(
    candidate: Mapping[str, Any], node: Mapping[str, Any]
) -> float:
    area = _section(node, "geometry").get("log_surface_area_m2")
    if not isinstance(area, Mapping):
        return 1.0
    density = max(float(candidate.get("local_surface_density", 0.0)), 0.0)
    minimum = max(3.0, 0.05 * density * math.exp(float(area["q50"])))
    return float(min(1.0, float(candidate["core_point_count"]) / minimum))


def smoothness_compatibility(
    candidate: Mapping[str, Any], node: Mapping[str, Any]
) -> float:
    summary = _section(node, "neighborhood").get("boundary_fixed:0.05")
    if not isinstance(summary, Mapping):
        return 1.0
    boundary = float(candidate["boundary_ratio_5cm"])
    if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
        raise ValueError("candidate boundary_ratio_5cm must be finite and in [0, 1]")
    q50 = float(summary["q50"])
    q75 = float(summary["q75"])
    z = max(0.0, boundary - q50) / max(q75 - q50, 1e-6)
    return float(math.exp(-0.5 * min(z * z, 25.0)))


def score_candidate(
    candidate: Mapping[str, Any],
    priors: Mapping[str, Any],
    condition: str,
) -> dict[str, float]:
    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown V9 replay condition: {condition}")
    use_size, use_support, use_smoothness = CONDITION_FACTORS[condition]
    class_name = str(candidate["branch_class"])
    q = normalize_score(candidate["base_score"], context="candidate base_score")
    g = size_compatibility(candidate, _prior_node(priors, class_name, use_size))
    c = support_compatibility(
        candidate, _prior_node(priors, class_name, use_support)
    )
    b = smoothness_compatibility(
        candidate, _prior_node(priors, class_name, use_smoothness)
    )
    return {
        "Q": q,
        "G": g,
        "C": c,
        "B": b,
        "score": float(np.clip(q * g * c * b, 0.0, 1.0)),
    }


def _point_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_ids = np.unique(np.asarray(left, dtype=np.int64))
    right_ids = np.unique(np.asarray(right, dtype=np.int64))
    intersection = len(np.intersect1d(left_ids, right_ids, assume_unique=True))
    union = len(left_ids) + len(right_ids) - intersection
    return float(intersection / union) if union else 0.0


def same_class_core_nms(
    candidates: Sequence[Mapping[str, Any]],
    score_by_id: Mapping[int, float],
    core_ids_by_candidate: Mapping[int, np.ndarray],
    *,
    iou_threshold: float = 0.50,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
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
            suppressed.append(int(candidate_id))
        else:
            kept.append(int(candidate_id))
    return tuple(kept), tuple(suppressed)


def assign_unique_gaussians(
    point_count: int,
    accepted_candidate_ids: Sequence[int],
    score_by_id: Mapping[int, float],
    full_ids_by_candidate: Mapping[int, np.ndarray],
    *,
    min_points: int = 10,
) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Assign score-ordered ownership and re-evaluate after small-mask removal."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    active = set(map(int, accepted_candidate_ids))
    dropped: set[int] = set()
    owner = np.full(point_count, -1, dtype=np.int32)
    while active:
        owner.fill(-1)
        ordered = sorted(active, key=lambda cid: (-float(score_by_id[cid]), cid))
        for candidate_id in ordered:
            ids = np.unique(
                np.asarray(full_ids_by_candidate[candidate_id], dtype=np.int64)
            )
            if np.any(ids < 0) or np.any(ids >= point_count):
                raise ValueError(f"candidate {candidate_id} has an out-of-range Gaussian id")
            unowned = ids[owner[ids] < 0]
            owner[unowned] = candidate_id
        too_small = {
            candidate_id
            for candidate_id in active
            if np.count_nonzero(owner == candidate_id) < int(min_points)
        }
        if not too_small:
            break
        active.difference_update(too_small)
        dropped.update(too_small)
    if not active:
        owner.fill(-1)
    kept = tuple(sorted(active))
    return owner, kept, tuple(sorted(dropped))


def materialize_prediction(
    owner_candidate_id: np.ndarray,
    candidates: Sequence[Mapping[str, Any]],
    score_by_id: Mapping[int, float],
    *,
    source: str = "v9_object_bank",
) -> FinalPrediction:
    owner = np.asarray(owner_candidate_id, dtype=np.int32)
    if owner.ndim != 1 or np.any(owner < -1):
        raise ValueError("owner_candidate_id must be one-dimensional and >= -1")
    rows = {int(row["candidate_id"]): row for row in candidates}
    candidate_ids = tuple(sorted(int(value) for value in np.unique(owner[owner >= 0])))
    labels = np.full(len(owner), -1, dtype=np.int32)
    instances: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for instance_id, candidate_id in enumerate(candidate_ids):
        if candidate_id not in rows or candidate_id not in score_by_id:
            raise ValueError(f"owner references unknown candidate {candidate_id}")
        labels[owner == candidate_id] = instance_id
        row = rows[candidate_id]
        class_name = str(row["branch_class"])
        score = normalize_score(
            score_by_id[candidate_id], context=f"candidate {candidate_id} score"
        )
        instances[str(instance_id)] = {"class": class_name, "score": score}
        metadata[str(instance_id)] = {
            "class": class_name,
            "score": score,
            "candidate_id": candidate_id,
            "point_count": int(np.count_nonzero(owner == candidate_id)),
            "source": source,
        }
    contracted = normalize_prediction(labels, instances)
    validate_prediction_contract(contracted.point_labels, contracted.instances)
    return FinalPrediction(contracted.point_labels, contracted.instances, metadata)


def replay_candidate_bank(
    bank: CandidateBank,
    priors: Mapping[str, Any],
    condition: str,
    *,
    acceptance_threshold: float,
    nms_core_iou: float = 0.50,
    min_points: int = 10,
) -> ReplayResult:
    """Replay one 2^3 condition without changing the frozen candidate bank."""
    if condition not in CONDITION_FACTORS:
        raise ValueError(f"unknown V9 replay condition: {condition}")
    if not np.isfinite(acceptance_threshold) or not 0 <= acceptance_threshold <= 1:
        raise ValueError("acceptance_threshold must be in [0, 1]")
    rows = tuple(dict(row) for row in bank.candidates)
    row_by_id = {int(row["candidate_id"]): row for row in rows}
    score_parts = {
        candidate_id: score_candidate(row, priors, condition)
        for candidate_id, row in row_by_id.items()
    }
    # Non-SAGA20/abstained late classifications stay in the immutable geometry
    # bank for association evaluation.  They are filtered only at final replay
    # so semantics cannot alter fragment, graph, track, or candidate identity.
    saga_eligible = {
        candidate_id
        for candidate_id, row in row_by_id.items()
        if str(row.get("branch_class", "")) in SAGA20
        and bool(row.get("classification_eligible", True))
    }
    passing = {
        candidate_id: parts["score"]
        for candidate_id, parts in score_parts.items()
        if candidate_id in saga_eligible
        and parts["score"] >= float(acceptance_threshold)
    }
    rejected = tuple(sorted(set(row_by_id).difference(passing)))
    core_ids = {
        candidate_id: bank.core_ids[candidate_id] for candidate_id in row_by_id
    }
    full_ids = {
        candidate_id: bank.full_ids[candidate_id] for candidate_id in row_by_id
    }
    accepted, suppressed = same_class_core_nms(
        rows, passing, core_ids, iou_threshold=float(nms_core_iou)
    )
    owner, kept, dropped = assign_unique_gaussians(
        int(bank.point_count),
        accepted,
        passing,
        full_ids,
        min_points=int(min_points),
    )
    prediction = materialize_prediction(owner, rows, passing)
    candidate_scores = tuple(
        {
            "candidate_id": candidate_id,
            "class": str(row_by_id[candidate_id]["branch_class"]),
            **score_parts[candidate_id],
        }
        for candidate_id in sorted(row_by_id)
    )
    return ReplayResult(
        prediction=prediction,
        candidate_scores=candidate_scores,
        accepted_candidate_ids=tuple(sorted(kept)),
        rejected_candidate_ids=rejected,
        suppressed_candidate_ids=suppressed,
        dropped_small_candidate_ids=dropped,
    )
