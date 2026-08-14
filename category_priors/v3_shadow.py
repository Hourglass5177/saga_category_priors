from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import write_json


SHADOW_MODES = ("exact", "exclusive")


def all_class_top1_labels(
    semantic_features: Any,
    label_features: Any,
    *,
    threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 32-way top-1 labels, scores and margins after cosine normalization."""
    semantic = np.asarray(semantic_features, dtype=np.float64)
    labels = np.asarray(label_features, dtype=np.float64)
    if semantic.ndim != 2 or labels.ndim != 2 or semantic.shape[1] != labels.shape[1]:
        raise ValueError("semantic_features and label_features must be compatible matrices")
    semantic_norm = np.linalg.norm(semantic, axis=1, keepdims=True)
    label_norm = np.linalg.norm(labels, axis=1, keepdims=True)
    semantic = semantic / np.maximum(semantic_norm, 1e-12)
    labels = labels / np.maximum(label_norm, 1e-12)
    scores = semantic @ labels.T
    winners = scores.argmax(axis=1).astype(np.int16)
    winner_scores = scores[np.arange(len(scores)), winners]
    if scores.shape[1] > 1:
        second = np.partition(scores, -2, axis=1)[:, -2]
    else:
        second = np.full(len(scores), -1.0, dtype=np.float64)
    margins = winner_scores - second
    winners[winner_scores < float(threshold)] = -1
    return winners, winner_scores.astype(np.float32), margins.astype(np.float32)


def target_top1_masks(
    semantic_features: Any,
    label_features: Any,
    target_indices: Sequence[int],
    *,
    threshold: float = 0.7,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Compete over the complete codebook, then expose masks for target classes only."""
    winners, scores, margins = all_class_top1_labels(
        semantic_features, label_features, threshold=threshold
    )
    masks = {int(index): winners == int(index) for index in target_indices}
    return masks, winners, scores, margins


def label_overlap(labels: Any, mask: Any) -> dict[str, Any]:
    source = np.asarray(labels, dtype=np.int64)
    selected = np.asarray(mask, dtype=bool)
    values = source[selected]
    values = values[values >= 0]
    if values.size == 0:
        return {"instance_id": None, "point_count": 0, "fraction": 0.0}
    ids, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    return {
        "instance_id": int(ids[best]),
        "point_count": int(counts[best]),
        "fraction": float(counts[best] / max(int(selected.sum()), 1)),
    }


def candidate_survival(
    candidate_mask: Any,
    merged_instance_id: int,
    after_knn_labels: Any,
    after_filter_labels: Any,
) -> dict[str, Any]:
    mask = np.asarray(candidate_mask, dtype=bool)
    after_knn = np.asarray(after_knn_labels, dtype=np.int64)
    after_filter = np.asarray(after_filter_labels, dtype=np.int64)
    total = int(mask.sum())
    knn_count = int(np.sum(mask & (after_knn == int(merged_instance_id))))
    filter_count = int(np.sum(mask & (after_filter == int(merged_instance_id))))
    return {
        "after_knn_points": knn_count,
        "after_knn_survival_rate": knn_count / total if total else 0.0,
        "after_filter_points": filter_count,
        "after_filter_survival_rate": filter_count / total if total else 0.0,
    }


def vote_summary(
    class_ratios: Any,
    class_names: Sequence[str],
    branch_class: str,
) -> dict[str, Any]:
    ratios = np.asarray(class_ratios, dtype=np.float64)
    if ratios.shape != (len(class_names),):
        raise ValueError("class vote ratios do not match class_names")
    background = float(max(0.0, 1.0 - ratios.sum()))
    winner_index = int(np.argmax(ratios)) if ratios.size else -1
    winner_score = float(ratios[winner_index]) if winner_index >= 0 else 0.0
    winner = class_names[winner_index] if winner_score >= background else "background"
    branch_index = class_names.index(branch_class)
    return {
        "branch_class_ratio": float(ratios[branch_index]),
        "winner": winner,
        "winner_ratio": max(winner_score, background),
        "background_ratio": background,
        "winner_matches_branch": winner == branch_class,
    }


def write_shadow_capture(
    *,
    json_path: str | Path,
    labels_path: str | Path,
    scene_id: str,
    seed: int,
    mode: str,
    git_commit: str,
    class_names: Sequence[str],
    affinity_gate: Any,
    branch_labels: Any,
    semantic_top1: Any,
    semantic_top1_score: Any,
    semantic_margin: Any,
    sam_covered: Any,
    candidates: Sequence[Mapping[str, Any]],
    class_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if mode not in SHADOW_MODES:
        raise ValueError(f"unsupported V3 shadow mode: {mode}")
    labels = np.asarray(branch_labels, dtype=np.int32)
    semantic = np.asarray(semantic_top1, dtype=np.int16)
    scores = np.asarray(semantic_top1_score, dtype=np.float32)
    margins = np.asarray(semantic_margin, dtype=np.float32)
    covered = np.asarray(sam_covered, dtype=bool)
    if not (labels.shape == semantic.shape == scores.shape == margins.shape == covered.shape):
        raise ValueError("all per-Gaussian V3 shadow arrays must share one shape")
    target = Path(labels_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        branch_labels=labels,
        semantic_top1=semantic,
        semantic_top1_score=scores,
        semantic_margin=margins,
        sam_covered_packed=np.packbits(covered),
        point_count=np.asarray([len(labels)], dtype=np.int64),
    )
    payload = {
        "kind": "v3_shadow_capture",
        "schema_version": "1.0",
        "git_commit": git_commit,
        "scene_id": scene_id,
        "seed": int(seed),
        "mode": mode,
        "class_names": [str(value) for value in class_names],
        "affinity_gate": np.asarray(affinity_gate, dtype=np.float32).reshape(-1).tolist(),
        "labels_npz": str(target),
        "point_count": len(labels),
        "candidate_count": len(candidates),
        "sam_covered_fraction": float(covered.mean()) if len(covered) else 0.0,
        "candidates": [dict(row) for row in candidates],
        "class_diagnostics": dict(class_diagnostics),
    }
    write_json(json_path, payload)
    return payload


def load_shadow_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        point_count = int(np.asarray(payload["point_count"]).reshape(-1)[0])
        covered = np.unpackbits(payload["sam_covered_packed"])[:point_count].astype(bool)
        return {
            "branch_labels": np.asarray(payload["branch_labels"], dtype=np.int32),
            "semantic_top1": np.asarray(payload["semantic_top1"], dtype=np.int16),
            "semantic_top1_score": np.asarray(payload["semantic_top1_score"], dtype=np.float32),
            "semantic_margin": np.asarray(payload["semantic_margin"], dtype=np.float32),
            "sam_covered": covered,
        }
