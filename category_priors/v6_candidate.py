from __future__ import annotations

"""Deterministic affinity-first proposal-bank primitives for V6.

This module intentionally knows nothing about ground truth or final output
mutation.  It first builds instance-shaped components from local affinity
edges, then lets a separate multiview pass decide whether a component has a
credible ScanNet class.  Category priors are deliberately absent here.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .io import write_json


SAGA20 = (
    "chair", "table", "plant", "tv", "painting", "sofa", "cabinet", "bed",
    "socket", "book", "switch", "door", "window", "lamp", "speaker", "fan",
    "refrigerator", "cup", "phone", "trash can",
)


@dataclass(frozen=True)
class V6GraphConfig:
    physical_neighbors: int = 24
    affinity_neighbors: int = 4
    core_degree: int = 3
    min_core_points: int = 10
    min_effective_views: int = 3
    min_vote_ratio: float = 0.60
    min_vote_margin: float = 0.10

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def normalized_top1(
    point_features: np.ndarray, label_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all-codebook cosine winner, score and winner/runner-up margin."""
    points = np.asarray(point_features, dtype=np.float64)
    labels = np.asarray(label_features, dtype=np.float64)
    if points.ndim != 2 or labels.ndim != 2 or points.shape[1] != labels.shape[1]:
        raise ValueError("point and label features must be compatible matrices")
    points = points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    labels = labels / np.maximum(np.linalg.norm(labels, axis=1, keepdims=True), 1e-12)
    scores = points @ labels.T
    winner = scores.argmax(axis=1).astype(np.int16)
    top = scores[np.arange(len(winner)), winner].astype(np.float32)
    if labels.shape[0] < 2:
        margin = np.zeros(len(winner), dtype=np.float32)
    else:
        two = np.partition(scores, -2, axis=1)[:, -2:]
        margin = (two[:, 1] - two[:, 0]).astype(np.float32)
    return winner, top, margin


def _normalise_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("affinity features must be a non-empty matrix")
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def build_affinity_components(
    metric_xyz: np.ndarray, affinity_features: np.ndarray,
    config: V6GraphConfig = V6GraphConfig(),
) -> dict[str, Any]:
    """Build mutual-top-affinity components in a fixed physical 24-NN graph."""
    xyz = np.asarray(metric_xyz, dtype=np.float64)
    features = _normalise_features(affinity_features)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(features):
        raise ValueError("xyz and affinity features must have matching N x 3/N x D shapes")
    n_points = len(xyz)
    if n_points < 2:
        empty = np.full(n_points, -1, dtype=np.int32)
        return {"full_labels": empty, "core_labels": empty.copy(), "degree": np.zeros(n_points, dtype=np.int16), "candidates": [], "edge_left": np.empty(0, dtype=np.int32), "edge_right": np.empty(0, dtype=np.int32), "edge_affinity": np.empty(0, dtype=np.float32)}
    neighbor_count = min(int(config.physical_neighbors), n_points - 1)
    tree = cKDTree(xyz)
    _, neighbors = tree.query(xyz, k=neighbor_count + 1, workers=-1)
    neighbors = np.asarray(neighbors, dtype=np.int64)[:, 1:]
    similarities = np.einsum("nd,nkd->nk", features, features[neighbors])
    keep_count = min(int(config.affinity_neighbors), neighbor_count)
    directed_rows: list[np.ndarray] = []
    directed_cols: list[np.ndarray] = []
    for index in range(n_points):
        # lexsort gives deterministic point-index tie breaking after affinity.
        order = np.lexsort((neighbors[index], -similarities[index]))[:keep_count]
        directed_rows.append(np.full(len(order), index, dtype=np.int64))
        directed_cols.append(neighbors[index, order])
    directed = coo_matrix(
        (np.ones(sum(map(len, directed_rows)), dtype=np.uint8),
         (np.concatenate(directed_rows), np.concatenate(directed_cols))),
        shape=(n_points, n_points),
    ).tocsr()
    mutual = directed.multiply(directed.T).astype(np.uint8).tocsr()
    mutual_coo = mutual.tocoo()
    select = mutual_coo.row < mutual_coo.col
    left, right = mutual_coo.row[select], mutual_coo.col[select]
    degree = np.asarray(mutual.sum(axis=1)).ravel().astype(np.int16)
    full_labels = np.full(n_points, -1, dtype=np.int32)
    core_labels = np.full(n_points, -1, dtype=np.int32)
    candidates: list[dict[str, Any]] = []
    if not len(left):
        return {"full_labels": full_labels, "core_labels": core_labels, "degree": degree, "candidates": candidates, "edge_left": left.astype(np.int32), "edge_right": right.astype(np.int32), "edge_affinity": np.einsum("nd,nd->n", features[left], features[right]).astype(np.float32)}
    graph = coo_matrix((np.ones(len(left) * 2, dtype=np.uint8), (np.concatenate((left, right)), np.concatenate((right, left)))), shape=(n_points, n_points)).tocsr()
    _, components = connected_components(graph, directed=False, return_labels=True)
    # ``connected_components`` can legitimately return thousands of tiny
    # components.  Re-scanning the complete label vector for every component
    # turns this otherwise sparse graph operation into O(N * components).
    # Group the labels once instead, while retaining the exact same members
    # for every connected component.
    component_order = np.argsort(components, kind="stable")
    ordered_components = components[component_order]
    component_starts = np.r_[
        0,
        np.flatnonzero(np.diff(ordered_components)) + 1,
        n_points,
    ]
    candidate_id = 0
    for start, end in zip(component_starts[:-1], component_starts[1:]):
        members = component_order[start:end]
        core = members[degree[members] >= int(config.core_degree)]
        if len(core) < int(config.min_core_points):
            continue
        full_labels[members] = candidate_id
        core_labels[core] = candidate_id
        extents = np.ptp(xyz[members], axis=0)
        local_k = min(16, n_points - 1)
        if local_k:
            distances, _ = tree.query(xyz[core], k=local_k + 1, workers=-1)
            radii = np.maximum(np.asarray(distances)[:, -1], 1e-6)
            density = float(np.median(local_k / (np.pi * radii * radii)))
        else:
            density = 0.0
        contained = mutual[members][:, members].tocoo()
        edge_select = contained.row < contained.col
        edge_values = np.einsum(
            "nd,nd->n", features[members][contained.row[edge_select]],
            features[members][contained.col[edge_select]],
        )
        candidates.append({
            "candidate_id": candidate_id,
            "full_point_count": int(len(members)),
            "core_point_count": int(len(core)),
            "metric_extent_xyz_m": [float(value) for value in extents],
            "metric_extents_m": [float(value) for value in np.sort(extents)],
            "local_surface_density": density,
            "internal_affinity_mean": float(edge_values.mean()) if len(edge_values) else 0.0,
            "internal_affinity_min": float(edge_values.min()) if len(edge_values) else 0.0,
        })
        candidate_id += 1
    return {"full_labels": full_labels, "core_labels": core_labels, "degree": degree, "candidates": candidates, "edge_left": left.astype(np.int32), "edge_right": right.astype(np.int32), "edge_affinity": np.einsum("nd,nd->n", features[left], features[right]).astype(np.float32)}


def finalise_multiview_candidates(
    components: Mapping[str, Any], candidate_frame_votes: np.ndarray,
    class_names: Sequence[str], config: V6GraphConfig = V6GraphConfig(),
) -> dict[str, Any]:
    """Keep only components whose post-hoc multiview class evidence is sound."""
    full = np.asarray(components["full_labels"], dtype=np.int32).copy()
    core = np.asarray(components["core_labels"], dtype=np.int32).copy()
    votes = np.asarray(candidate_frame_votes, dtype=np.int64)
    candidates = [dict(row) for row in components["candidates"]]
    if votes.shape != (len(candidates), len(class_names)):
        raise ValueError("candidate-frame vote matrix has incompatible shape")
    retained: list[dict[str, Any]] = []
    # Candidate IDs are dense indices into ``candidates``.  Keep their
    # remapping in an array so labels can be remapped in one indexed pass.
    # Repeated ``full == old_id`` scans are quadratic when the affinity graph
    # yields thousands of small components.
    remap = np.full(len(candidates), -1, dtype=np.int32)
    for row in candidates:
        old_id = int(row["candidate_id"])
        counts = votes[old_id]
        total = int(counts.sum())
        winner = int(counts.argmax()) if total else -1
        winner_count = int(counts[winner]) if winner >= 0 else 0
        runner_up = int(np.partition(counts, -2)[-2]) if len(counts) > 1 else 0
        ratio = winner_count / total if total else 0.0
        margin = (winner_count - runner_up) / total if total else 0.0
        class_name = str(class_names[winner]) if winner >= 0 else "background"
        accepted = (
            class_name in SAGA20
            and total >= int(config.min_effective_views)
            and ratio >= float(config.min_vote_ratio)
            and margin >= float(config.min_vote_margin)
        )
        row.update({
            "branch_class": class_name,
            "effective_view_count": total,
            "vote": {
                "winner": class_name,
                "winner_ratio": float(ratio),
                "winner_margin": float(margin),
                "background_ratio": 0.0,
                "winner_matches_branch": bool(accepted),
            },
            "accepted_semantic": bool(accepted),
        })
        if accepted:
            new_id = len(retained)
            remap[old_id] = new_id
            row["candidate_id"] = new_id
            retained.append(row)
    new_full = np.full_like(full, -1)
    new_core = np.full_like(core, -1)
    valid_full = full >= 0
    valid_core = core >= 0
    new_full[valid_full] = remap[full[valid_full]]
    new_core[valid_core] = remap[core[valid_core]]
    return {
        "full_labels": new_full,
        "core_labels": new_core,
        "degree": np.asarray(components["degree"], dtype=np.int16),
        "candidates": retained,
        "candidate_frame_votes": votes,
        "edge_left": np.asarray(components.get("edge_left", []), dtype=np.int32),
        "edge_right": np.asarray(components.get("edge_right", []), dtype=np.int32),
        "edge_affinity": np.asarray(components.get("edge_affinity", []), dtype=np.float32),
    }


def write_v6_proposal_bank(
    *, json_path: str | Path, labels_path: str | Path, scene_id: str, seed: int,
    git_commit: str, class_names: Sequence[str], finalised: Mapping[str, Any],
    codebook_winner: np.ndarray, codebook_score: np.ndarray, codebook_margin: np.ndarray,
    point_view_count: np.ndarray, point_vote_winner: np.ndarray,
    point_vote_ratio: np.ndarray, point_vote_margin: np.ndarray,
) -> dict[str, Any]:
    arrays = {
        "full_labels": np.asarray(finalised["full_labels"], dtype=np.int32),
        "core_labels": np.asarray(finalised["core_labels"], dtype=np.int32),
        "mutual_degree": np.asarray(finalised["degree"], dtype=np.int16),
        "codebook_winner": np.asarray(codebook_winner, dtype=np.int16),
        "codebook_score": np.asarray(codebook_score, dtype=np.float32),
        "codebook_margin": np.asarray(codebook_margin, dtype=np.float32),
        "point_view_count": np.asarray(point_view_count, dtype=np.int16),
        "point_vote_winner": np.asarray(point_vote_winner, dtype=np.int16),
        "point_vote_ratio": np.asarray(point_vote_ratio, dtype=np.float32),
        "point_vote_margin": np.asarray(point_vote_margin, dtype=np.float32),
    }
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("all V6 per-Gaussian arrays must have the same shape")
    target = Path(labels_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, **arrays,
        edge_left=np.asarray(finalised.get("edge_left", []), dtype=np.int32),
        edge_right=np.asarray(finalised.get("edge_right", []), dtype=np.int32),
        edge_affinity=np.asarray(finalised.get("edge_affinity", []), dtype=np.float32),
    )
    payload = {
        "kind": "v6_affinity_proposal_bank", "schema_version": "1.0",
        "git_commit": str(git_commit), "scene_id": str(scene_id), "seed": int(seed),
        "class_names": [str(name) for name in class_names],
        "config": V6GraphConfig().as_json(), "point_count": int(len(arrays["full_labels"])),
        "labels_npz": str(target), "candidate_count": int(len(finalised["candidates"])),
        "candidates": [dict(row) for row in finalised["candidates"]],
    }
    write_json(json_path, payload)
    return payload
