from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json
from .v5_candidate import (
    CONDITIONS,
    base_evidence_reason,
    score_candidate,
    validate_condition,
)
from .v5_candidate_runner import v5_candidate_run_paths


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    return intersection / union if union else 0.0


def _candidate_masks(labels: np.ndarray, candidate_id: int) -> np.ndarray:
    return np.asarray(labels == int(candidate_id), dtype=bool)


def _base_instances(output: Mapping[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for raw_id, details in output.get("instances", {}).items():
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        class_name = details.get("class") if isinstance(details, Mapping) else None
        if isinstance(class_name, str):
            result[instance_id] = class_name
    return result


def replay_v5_scene(
    *, candidate_root: str | Path, output_root: str | Path, source: str,
    condition: str, scene_id: str, seed: int, category_priors: str | Path,
    calibrator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a frozen bank over B1 without touching any non-background B1 point."""
    validate_condition(condition)
    paths = v5_candidate_run_paths(candidate_root, source, scene_id, seed)
    output = load_json(paths["output"])
    diagnostics = load_json(paths["diagnostics"])
    bank = load_json(paths["proposals"])
    priors = load_json(category_priors)
    with np.load(paths["proposal_labels"], allow_pickle=False) as arrays:
        branch_labels = np.asarray(arrays["branch_labels"], dtype=np.int64)
        core_labels = np.asarray(arrays["core_labels"], dtype=np.int64)
    labels = np.asarray(output["point_labels"], dtype=np.int64)
    if labels.shape != branch_labels.shape or labels.shape != core_labels.shape:
        raise ValueError(f"{scene_id}: B1 output and V5 bank have incompatible point counts")

    instances = {str(key): dict(value) for key, value in output.get("instances", {}).items()}
    metadata_instances = {
        str(key): dict(value)
        for key, value in diagnostics.get("instances", {}).items()
        if isinstance(value, Mapping)
    }
    base_classes = _base_instances(output)
    next_instance = max(base_classes, default=-1) + 1
    accepted_full_masks: dict[str, list[np.ndarray]] = defaultdict(list)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    candidates = [dict(item) for item in bank.get("candidates", [])]
    scored: list[tuple[float, int, dict[str, Any], dict[str, float]]] = []
    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        evidence_reason = base_evidence_reason(candidate)
        if evidence_reason is not None:
            rejected[evidence_reason] += 1
            continue
        score_parts = score_candidate(candidate, priors, condition)
        score = float(score_parts["score"])
        if calibrator is not None:
            score = calibrated_probability(score_parts, calibrator)
            score_parts = {**score_parts, "calibrated_probability": score}
        if score < 0.20:
            rejected["low_score"] += 1
            continue
        scored.append((score, candidate_id, candidate, score_parts))
    scored.sort(key=lambda item: (-item[0], item[1]))

    for score, candidate_id, candidate, score_parts in scored:
        full_mask = _candidate_masks(branch_labels, candidate_id)
        core_mask = _candidate_masks(core_labels, candidate_id)
        class_name = str(candidate["branch_class"])
        if not bool(core_mask.any()):
            rejected["empty_core"] += 1
            continue
        if any(_iou(full_mask, prior) >= 0.50 for prior in accepted_full_masks[class_name]):
            rejected["candidate_nms"] += 1
            continue
        overlaps: list[tuple[float, int, str]] = []
        for instance_id, instance_class in base_classes.items():
            overlap = _iou(full_mask, labels == int(instance_id))
            if overlap > 0:
                overlaps.append((overlap, instance_id, instance_class))
        if any(overlap > 0.25 and instance_class != class_name for overlap, _, instance_class in overlaps):
            rejected["conflicts_with_other_class"] += 1
            continue
        same = [item for item in overlaps if item[2] == class_name and item[0] >= 0.25]
        background_core = core_mask & (labels < 0)
        if same:
            _, destination, _ = max(same, key=lambda item: (item[0], -item[1]))
            changed = int(background_core.sum())
            if not changed:
                rejected["no_background_to_complete"] += 1
                continue
            labels[background_core] = destination
            fusion = "same_class_completion"
            destination_id = destination
        else:
            # All B1 overlaps are <= .25 here; do not carve out or relabel points.
            changed = int(background_core.sum())
            if not changed:
                rejected["no_background_for_new_instance"] += 1
                continue
            destination_id = next_instance
            labels[background_core] = destination_id
            instances[str(destination_id)] = {"class": class_name}
            metadata_instances[str(destination_id)] = {
                "class": class_name, "score": float(score), "source": "v5_proposal_replay",
                "candidate_id": candidate_id,
            }
            base_classes[destination_id] = class_name
            next_instance += 1
            fusion = "new_instance"
        accepted_full_masks[class_name].append(full_mask)
        accepted.append({
            "candidate_id": candidate_id, "instance_id": int(destination_id),
            "class": class_name, "score": float(score), "score_parts": score_parts,
            "fusion": fusion, "core_points": int(core_mask.sum()),
            "added_background_points": changed,
        })

    output["point_labels"] = labels.tolist()
    output["instances"] = instances
    diagnostics["instances"] = metadata_instances
    diagnostics["status"] = "complete"
    diagnostics["run"] = {
        "condition": condition, "scene_id": str(scene_id), "seed": int(seed),
    }
    diagnostics["runner"] = {
        **dict(diagnostics.get("runner", {})), "point_count": int(len(labels)),
        "instance_count": int(len(instances)),
    }
    diagnostics["v5_proposal_replay"] = {
        "source": source, "condition": condition, "accepted_count": len(accepted),
        "accepted_core_points": int(sum(item["core_points"] for item in accepted)),
        "added_background_points": int(sum(item["added_background_points"] for item in accepted)),
        "rejection_reasons": dict(sorted(rejected.items())), "accepted": accepted,
    }
    target = Path(output_root).resolve() / condition / scene_id / f"seed-{int(seed)}"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "output.json", output)
    write_json(target / "diagnostics.json", diagnostics)
    return diagnostics["v5_proposal_replay"]


def materialize_v5_b1_baseline(
    *, candidate_root: str | Path, output_root: str | Path, source: str,
    scene_ids: Sequence[str], seeds: Sequence[int], condition: str = "B1-original",
) -> dict[str, Any]:
    """Expose frozen B1 files under the common replay layout without rerunning it."""
    result: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        for seed in seeds:
            source_paths = v5_candidate_run_paths(candidate_root, source, str(scene_id), int(seed))
            output = load_json(source_paths["output"])
            diagnostics = load_json(source_paths["diagnostics"])
            labels = np.asarray(output["point_labels"], dtype=np.int64)
            diagnostics["status"] = "complete"
            diagnostics["run"] = {
                "condition": condition, "scene_id": str(scene_id), "seed": int(seed),
            }
            diagnostics["runner"] = {
                **dict(diagnostics.get("runner", {})), "point_count": int(len(labels)),
                "instance_count": int(len(output.get("instances", {}))),
            }
            target = Path(output_root).resolve() / condition / str(scene_id) / f"seed-{int(seed)}"
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "output.json", output)
            write_json(target / "diagnostics.json", diagnostics)
            result.append({"scene_id": str(scene_id), "seed": int(seed), "condition": condition})
    return {"kind": "v5_b1_materialization", "source": source, "condition": condition, "runs": result}


def calibrated_probability(score_parts: Mapping[str, float], calibrator: Mapping[str, Any]) -> float:
    """Apply a persisted three-feature logistic model without loading any GT."""
    feature_names = list(calibrator.get("feature_names", ("E", "G", "C")))
    coefficients = np.asarray(calibrator["coefficients"], dtype=np.float64)
    mean = np.asarray(calibrator.get("mean", np.zeros(len(feature_names))), dtype=np.float64)
    scale = np.asarray(calibrator.get("scale", np.ones(len(feature_names))), dtype=np.float64)
    values = np.asarray([float(score_parts[name]) for name in feature_names], dtype=np.float64)
    logit = float(calibrator["intercept"]) + float(np.dot(coefficients, (values - mean) / np.maximum(scale, 1e-12)))
    return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))


def replay_v5_proposals(
    *, candidate_root: str | Path, output_root: str | Path, source: str,
    conditions: Sequence[str], scene_ids: Sequence[str], seeds: Sequence[int],
    category_priors: str | Path, calibrators: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"kind": "v5_proposal_replay", "source": source, "runs": []}
    for condition in conditions:
        validate_condition(condition)
        for scene_id in scene_ids:
            for seed in seeds:
                result["runs"].append({
                    "condition": condition, "scene_id": str(scene_id), "seed": int(seed),
                    "result": replay_v5_scene(
                        candidate_root=candidate_root, output_root=output_root, source=source,
                        condition=condition, scene_id=str(scene_id), seed=int(seed),
                        category_priors=category_priors,
                        calibrator=(calibrators or {}).get(condition),
                    ),
                })
    return result
