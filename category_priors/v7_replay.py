from __future__ import annotations

"""Offline category-prior replay over an immutable V7 object bank."""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json


CONDITIONS = ("U00-uniform", "D10-size", "D01-core", "D11-combined")


def _prior_node(priors: Mapping[str, Any], class_name: str, use_class: bool) -> Mapping[str, Any]:
    if use_class:
        node = priors.get("categories", {}).get(class_name)
        if isinstance(node, Mapping) and isinstance(node.get("shrunk"), Mapping):
            return node
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise ValueError("category priors are missing a global shrunk node")
    return node


def _geometry(node: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]]:
    shrunk = node.get("shrunk", {})
    geometry = shrunk.get("geometry", {}) if isinstance(shrunk, Mapping) else {}
    if not isinstance(geometry, Mapping):
        raise ValueError("prior node has invalid shrunk geometry")
    return geometry


def size_compatibility(candidate: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    extents = np.sort(np.maximum(np.asarray(candidate["metric_extents_m"], dtype=float), 1e-9))
    geometry = _geometry(node)
    fields = ("log_extent_short_m", "log_extent_mid_m", "log_extent_long_m")
    z: list[float] = []
    for extent, field in zip(extents, fields):
        summary = geometry.get(field)
        if not isinstance(summary, Mapping):
            return 1.0
        q50, q75 = float(summary["q50"]), float(summary["q75"])
        z.append(max(0.0, math.log(float(extent)) - q50) / max(q75 - q50, 1e-6))
    return float(math.exp(-0.5 * np.mean(np.minimum(np.square(z), 25.0))))


def core_compatibility(candidate: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    geometry = _geometry(node)
    area = geometry.get("log_surface_area_m2")
    if not isinstance(area, Mapping):
        return 1.0
    density = max(float(candidate.get("local_surface_density", 0.0)), 0.0)
    minimum = max(3.0, 0.05 * density * math.exp(float(area["q50"])))
    return float(min(1.0, float(candidate["core_point_count"]) / minimum))


def score_candidate(
    candidate: Mapping[str, Any], priors: Mapping[str, Any], condition: str,
) -> dict[str, float]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown V7 replay condition: {condition}")
    size_class = condition in {"D10-size", "D11-combined"}
    core_class = condition in {"D01-core", "D11-combined"}
    class_name = str(candidate["branch_class"])
    g = size_compatibility(candidate, _prior_node(priors, class_name, size_class))
    c = core_compatibility(candidate, _prior_node(priors, class_name, core_class))
    q = float(candidate["base_score"])
    return {"Q": q, "G": g, "C": c, "score": float(np.clip(q * g * c, 0.0, 1.0))}


def replay_v7_scene(
    *, bank_dir: str | Path, output_root: str | Path, condition: str,
    category_priors: str | Path, acceptance_threshold: float = 0.20,
) -> dict[str, Any]:
    bank_path = Path(bank_dir).resolve()
    bank = load_json(bank_path / "object_bank.json")
    if bank.get("schema") != "saga-v7-object-bank-v1":
        raise ValueError(f"{bank_path}: not a V7 object bank")
    arrays_path = Path(str(bank["arrays_npz"]))
    if not arrays_path.is_absolute():
        arrays_path = bank_path / arrays_path
    with np.load(arrays_path, allow_pickle=False) as arrays:
        candidate_labels = np.asarray(arrays["candidate_labels"], dtype=np.int32)
    priors = load_json(category_priors)
    scored: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
    for row in bank.get("candidates", []):
        candidate = dict(row)
        candidate_id = int(candidate["candidate_id"])
        parts = score_candidate(candidate, priors, condition)
        scored.append((parts["score"], candidate_id, candidate, parts))
    scored.sort(key=lambda item: (-item[0], item[1]))

    labels = np.full(len(candidate_labels), -1, dtype=np.int32)
    instances: dict[str, dict[str, str]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for score, candidate_id, candidate, parts in scored:
        if score < float(acceptance_threshold):
            rejected.append({"candidate_id": candidate_id, "score_parts": parts})
            continue
        instance_id = len(instances)
        mask = candidate_labels == candidate_id
        labels[mask] = instance_id
        class_name = str(candidate["branch_class"])
        instances[str(instance_id)] = {"class": class_name}
        metadata[str(instance_id)] = {
            "class": class_name,
            "score": float(score),
            "candidate_id": candidate_id,
            "source": "v7_object_bank",
        }
        accepted.append(
            {
                "candidate_id": candidate_id,
                "instance_id": instance_id,
                "class": class_name,
                "point_count": int(np.count_nonzero(mask)),
                "score_parts": parts,
            }
        )

    scene_id = str(bank["scene_id"])
    target = Path(output_root).resolve() / condition / scene_id
    target.mkdir(parents=True, exist_ok=True)
    output = {"point_labels": labels.tolist(), "instances": instances}
    diagnostics = {
        "kind": "v7_replay_diagnostics",
        "schema_version": "1.0",
        "scene_id": scene_id,
        "condition": condition,
        "point_count": int(len(labels)),
        "candidate_count": int(len(scored)),
        "accepted_count": int(len(accepted)),
        "assigned_points": int(np.count_nonzero(labels >= 0)),
        "coverage": float(np.mean(labels >= 0)) if len(labels) else 0.0,
        "accepted": accepted,
        "rejected": rejected,
        "instances": metadata,
    }
    write_json(target / "output.json", output)
    write_json(target / "diagnostics.json", diagnostics)
    return diagnostics


def replay_v7_priors(
    *, bank_root: str | Path, output_root: str | Path,
    scene_ids: Sequence[str], conditions: Sequence[str],
    category_priors: str | Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        for condition in conditions:
            records.append(
                replay_v7_scene(
                    bank_dir=Path(bank_root) / scene_id,
                    output_root=output_root,
                    condition=str(condition),
                    category_priors=category_priors,
                )
            )
    return {"kind": "v7_prior_replay", "runs": records}
