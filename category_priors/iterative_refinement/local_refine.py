from __future__ import annotations

"""Deterministic local graph refinement, size control, and constrained B0 fusion."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_flow
from scipy.spatial import cKDTree

from ..geometry import pca_sorted_extents_m
from .contracts import (
    CandidateSeed,
    GaussianEvidence,
    ObjectState,
    RefinementConfig,
    RefinementProfile,
)


@dataclass(frozen=True)
class SizePrior:
    diagonal_q50_m: float
    extents_q95_m: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class LocalRefinementResult:
    state: ObjectState
    roi_point_ids: np.ndarray
    graph_edges: np.ndarray
    graph_too_large: bool
    no_hard_evidence: bool
    size_trimmed_count: int
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class FusionResult:
    labels: np.ndarray
    object_raw_labels: Mapping[int, int]
    accepted_object_ids: tuple[int, ...]
    rejected_object_ids: tuple[int, ...]
    diagnostics: Mapping[str, Any]


def _summary_value(node: Mapping[str, Any], field: str, quantile: str) -> float:
    value = float(node["shrunk"]["geometry"][field][quantile])
    if not np.isfinite(value):
        raise ValueError(f"invalid prior value {field}.{quantile}")
    return value


def size_prior_from_payload(
    payload: Mapping[str, Any],
    class_name: str | None,
) -> SizePrior:
    global_node = payload.get("global")
    if not isinstance(global_node, Mapping):
        raise ValueError("prior payload has no global node")
    source = "global"
    node = global_node
    if class_name is not None:
        candidate = payload.get("categories", {}).get(class_name)
        if isinstance(candidate, Mapping) and candidate.get("active", False):
            node = candidate
            source = str(class_name)
    diagonal = float(np.exp(_summary_value(node, "log_bbox_diag_m", "q50")))
    extents = tuple(
        float(np.exp(_summary_value(node, field, "q95")))
        for field in ("log_extent_short_m", "log_extent_mid_m", "log_extent_long_m")
    )
    return SizePrior(diagonal, extents, source)


def local_roi_point_ids(
    xyz_m: Any,
    seed: CandidateSeed,
    evidence: GaussianEvidence,
    prior: SizePrior,
    config: RefinementConfig = RefinementConfig(),
    *,
    scene_tree: cKDTree | None = None,
) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("xyz_m must be a finite Nx3 matrix")
    base = np.unique(np.concatenate((seed.seed_support, evidence.point_ids))).astype(np.int64)
    if not len(base):
        return base
    if int(base.max()) >= len(xyz):
        raise ValueError("candidate/evidence point IDs exceed xyz")
    radius = float(
        np.clip(
            config.graph_radius_fraction * prior.diagonal_q50_m,
            config.graph_radius_min_m,
            config.graph_radius_max_m,
        )
    )
    tree = scene_tree if scene_tree is not None else cKDTree(xyz)
    if int(tree.n) != len(xyz):
        raise ValueError("scene_tree is not aligned with xyz_m")
    discovered: set[int] = set(int(value) for value in base)
    # Chunked queries avoid one giant Python list for large candidate supports.
    for start in range(0, len(base), 4096):
        for neighbors in tree.query_ball_point(xyz[base[start : start + 4096]], r=radius):
            discovered.update(int(value) for value in neighbors)
            if len(discovered) > config.graph_node_limit:
                return np.asarray(sorted(discovered), dtype=np.int64)
    return np.asarray(sorted(discovered), dtype=np.int64)


def mutual_local_edges(
    xyz_m: Any,
    affinity: Any,
    roi_point_ids: Any,
    profile: RefinementProfile,
    config: RefinementConfig = RefinementConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    features = np.asarray(affinity, dtype=np.float64)
    ids = np.asarray(roi_point_ids, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(xyz):
        raise ValueError("affinity must be an NxD matrix aligned to xyz")
    if not len(ids):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float64)
    local_xyz = xyz[ids]
    local_features = features[ids]
    norms = np.linalg.norm(local_features, axis=1, keepdims=True)
    local_features = np.divide(local_features, norms, out=np.zeros_like(local_features), where=norms > 0)
    count = len(ids)
    k = min(config.graph_neighbors + 1, count)
    distances, neighbors = cKDTree(local_xyz).query(
        local_xyz,
        k=k,
        distance_upper_bound=config.graph_edge_radius_m,
    )
    if k == 1:
        distances = distances[:, None]
        neighbors = neighbors[:, None]
    directed: dict[tuple[int, int], float] = {}
    for left in range(count):
        for distance, right in zip(distances[left, 1:], neighbors[left, 1:]):
            right_id = int(right)
            if right_id >= count or not np.isfinite(distance) or left == right_id:
                continue
            directed[(left, right_id)] = float(distance)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for (left, right), distance in sorted(directed.items()):
        if left >= right or (right, left) not in directed:
            continue
        cosine = max(float(local_features[left] @ local_features[right]), 0.0)
        weight = profile.pairwise_weight * np.exp(
            -(distance * distance) / (2.0 * config.graph_edge_radius_m**2)
        ) * cosine**2
        if weight > 0:
            edges.append((left, right))
            weights.append(float(weight))
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(weights, dtype=np.float64)


def _evidence_on_roi(
    seed: CandidateSeed,
    evidence: GaussianEvidence,
    roi: np.ndarray,
    b0_labels: np.ndarray,
    profile: RefinementProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = {int(point_id): index for index, point_id in enumerate(roi)}
    positive = np.zeros(len(roi), dtype=np.float64)
    negative = np.zeros(len(roi), dtype=np.float64)
    hard_count = np.zeros(len(roi), dtype=np.float64)
    support = np.zeros(len(roi), dtype=bool)
    anchor = np.zeros(len(roi), dtype=bool)
    for point_id in seed.seed_support:
        if int(point_id) in position:
            support[position[int(point_id)]] = True
    for point_id in seed.seed_anchor:
        if int(point_id) in position:
            anchor[position[int(point_id)]] = True
    for offset, point_id in enumerate(evidence.point_ids):
        target = position.get(int(point_id))
        if target is None:
            continue
        hp = float(evidence.hard_positive_views[offset])
        hn = float(evidence.hard_negative_views[offset])
        hard_count[target] = hp
        positive[target] += 2.0 * hp + profile.alpha_weight * float(evidence.alpha_soft_support[offset])
        negative[target] += 2.0 * hn
    positive += 0.5 * support + 1.0 * anchor
    raw_owner = (
        seed.diagnostics.get("matched_stage_instance_id")
        if seed.anchor_stage == "exported_prediction"
        else None
    )
    own_owner = None if raw_owner is None else int(raw_owner)
    foreign_b0 = (b0_labels[roi] >= 0) & (
        True if own_owner is None else (b0_labels[roi] != own_owner)
    )
    negative += 0.5 * foreign_b0
    hard_positive = hard_count >= 2
    hard_negative = (negative >= 4.0) & (hard_count == 0)
    return positive, negative, hard_positive, hard_negative, hard_count


def binary_graph_cut(
    positive: Any,
    negative: Any,
    hard_positive: Any,
    hard_negative: Any,
    edges: Any,
    edge_weights: Any,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(positive, dtype=np.float64)
    n = np.asarray(negative, dtype=np.float64)
    hp = np.asarray(hard_positive, dtype=bool)
    hn = np.asarray(hard_negative, dtype=bool)
    pairs = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    pair_weights = np.asarray(edge_weights, dtype=np.float64)
    if not (p.shape == n.shape == hp.shape == hn.shape) or len(pairs) != len(pair_weights):
        raise ValueError("graph unary and pairwise arrays are inconsistent")
    if np.any(hp & hn) or np.any(p < 0) or np.any(n < 0):
        raise ValueError("hard evidence conflicts or finite evidence is negative")
    probability = (p + 1.0) / (p + n + 2.0)
    foreground_cost = -np.log(np.clip(probability, 1e-12, 1.0))
    background_cost = -np.log(np.clip(1.0 - probability, 1e-12, 1.0))
    scale = 1000.0
    foreground_cap = np.rint(foreground_cost * scale).astype(np.int64)
    background_cap = np.rint(background_cost * scale).astype(np.int64)
    pair_cap = np.maximum(1, np.rint(pair_weights * scale).astype(np.int64))
    finite_total = int(foreground_cap.sum() + background_cap.sum() + 2 * pair_cap.sum())
    infinity = finite_total + 1
    foreground_cap[hp] = 0
    background_cap[hp] = infinity
    foreground_cap[hn] = infinity
    background_cap[hn] = 0
    count = len(p)
    source, sink = count, count + 1
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []
    for index in range(count):
        rows.extend((source, index))
        cols.extend((index, sink))
        data.extend((int(background_cap[index]), int(foreground_cap[index])))
    for (left, right), capacity in zip(pairs, pair_cap):
        rows.extend((int(left), int(right)))
        cols.extend((int(right), int(left)))
        data.extend((int(capacity), int(capacity)))
    capacity_matrix = coo_matrix(
        (np.asarray(data, dtype=np.int64), (rows, cols)),
        shape=(count + 2, count + 2),
        dtype=np.int64,
    ).tocsr()
    result = maximum_flow(capacity_matrix, source, sink)
    residual = (capacity_matrix - result.flow).tocsr()
    reachable = np.zeros(count + 2, dtype=bool)
    stack = [source]
    reachable[source] = True
    while stack:
        node = stack.pop()
        start, stop = residual.indptr[node], residual.indptr[node + 1]
        for neighbor, value in zip(residual.indices[start:stop], residual.data[start:stop]):
            if value > 0 and not reachable[int(neighbor)]:
                reachable[int(neighbor)] = True
                stack.append(int(neighbor))
    margin = p - n
    return reachable[:count], margin


def _within_extent_limit(points_m: np.ndarray, limit: tuple[float, float, float]) -> bool:
    if not len(points_m):
        return True
    extents = pca_sorted_extents_m(points_m, 1.0)
    return bool(np.all(extents <= np.asarray(limit, dtype=np.float64) + 1e-9))


def trim_oversize_additions(
    selected_ids: np.ndarray,
    original_ids: np.ndarray,
    hard_positive_ids: np.ndarray,
    margins: np.ndarray,
    xyz_m: np.ndarray,
    limit: tuple[float, float, float],
) -> tuple[np.ndarray, int]:
    selected = np.asarray(selected_ids, dtype=np.int64)
    if _within_extent_limit(xyz_m[selected], limit):
        return selected, 0
    protected = set(int(value) for value in original_ids) | set(int(value) for value in hard_positive_ids)
    margin_by_id = {int(point_id): float(value) for point_id, value in zip(selected, margins)}
    removable = sorted(
        (int(value) for value in selected if int(value) not in protected),
        key=lambda point_id: (margin_by_id[point_id], point_id),
    )
    retained = set(int(value) for value in selected)
    for point_id in removable:
        retained.remove(point_id)
        candidate = np.asarray(sorted(retained), dtype=np.int64)
        if _within_extent_limit(xyz_m[candidate], limit):
            return candidate, len(selected) - len(candidate)
    return np.asarray(sorted(retained), dtype=np.int64), len(selected) - len(retained)


def refine_candidate_local(
    *,
    seed: CandidateSeed,
    evidence: GaussianEvidence,
    xyz_m: Any,
    affinity: Any,
    b0_labels: Any,
    prior: SizePrior,
    profile: RefinementProfile,
    round_index: int,
    review_class: str | None,
    reliable_review_class: bool,
    config: RefinementConfig = RefinementConfig(),
    scene_tree: cKDTree | None = None,
) -> LocalRefinementResult:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    b0 = np.asarray(b0_labels, dtype=np.int64)
    # A candidate cannot change without at least one Gaussian supported by two
    # independent views.  Previously we discovered this only after building a
    # potentially 100k-node ROI and local k-NN graph.  The graph is unused on
    # this path, so skip that mathematically dead work before any spatial
    # query.  The predicate is profile-independent and identical to the hard
    # foreground predicate produced by _evidence_on_roi below.
    if not bool(np.any(np.asarray(evidence.hard_positive_views) >= 2)):
        points = seed.seed_support.copy()
        state = ObjectState(
            object_id=seed.candidate_id,
            parent_candidate_ids=seed.parent_candidate_ids,
            point_ids=points,
            anchor_ids=np.intersect1d(seed.seed_anchor, points, assume_unique=True),
            hard_positive_ids=np.empty(0, dtype=np.int64),
            hard_positive_counts=np.zeros(len(points), dtype=np.float64),
            evidence_margin=np.zeros(len(points), dtype=np.float64),
            review_class=review_class,
            reliable_review_class=reliable_review_class,
            round_index=int(round_index),
            changed=False,
        )
        return LocalRefinementResult(
            state,
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            False,
            True,
            0,
            {"reason": "no_two_view_hard_support", "roi_points": 0, "edge_count": 0},
        )
    roi = local_roi_point_ids(xyz, seed, evidence, prior, config, scene_tree=scene_tree)
    too_large = len(roi) > config.graph_node_limit
    if too_large:
        points = seed.seed_support
        state = ObjectState(
            seed.candidate_id,
            seed.parent_candidate_ids,
            points,
            np.intersect1d(seed.seed_anchor, points, assume_unique=True),
            np.empty(0, dtype=np.int64),
            np.zeros(len(points)),
            np.zeros(len(points)),
            review_class,
            reliable_review_class,
            round_index,
            False,
        )
        return LocalRefinementResult(state, roi, np.empty((0, 2), np.int64), True, False, 0, {"reason": "graph_node_limit"})
    edges, edge_weights = mutual_local_edges(xyz, affinity, roi, profile, config)
    p, n, hp, hn, hard_count = _evidence_on_roi(seed, evidence, roi, b0, profile)
    no_hard = not bool(np.any(hp))
    if no_hard:
        selected = seed.seed_support.copy()
        margin = np.zeros(len(selected), dtype=np.float64)
        hard_ids = np.empty(0, dtype=np.int64)
        hard_counts = np.zeros(len(selected), dtype=np.float64)
        trimmed = 0
    else:
        foreground, roi_margin = binary_graph_cut(p, n, hp, hn, edges, edge_weights)
        selected = roi[foreground]
        hard_ids = roi[hp & foreground]
        selected_margin = roi_margin[foreground]
        selected_hard_count = hard_count[foreground]
        selected, trimmed = trim_oversize_additions(
            selected,
            # The full pre-KNN support is deliberately *not* immutable: it is
            # precisely where legacy centre assignment can have introduced
            # pollution.  Only the last surviving in-support anchor and
            # two-view hard positives are protected from size trimming.
            seed.seed_anchor,
            hard_ids,
            selected_margin,
            xyz,
            prior.extents_q95_m,
        )
        index = {int(point_id): offset for offset, point_id in enumerate(roi)}
        margin = np.asarray([roi_margin[index[int(point_id)]] for point_id in selected])
        hard_counts = np.asarray([hard_count[index[int(point_id)]] for point_id in selected])
        hard_ids = selected[hard_counts >= 2]
    state = ObjectState(
        object_id=seed.candidate_id,
        parent_candidate_ids=seed.parent_candidate_ids,
        point_ids=selected,
        anchor_ids=np.intersect1d(seed.seed_anchor, selected, assume_unique=True),
        hard_positive_ids=hard_ids,
        hard_positive_counts=hard_counts,
        evidence_margin=margin,
        review_class=review_class,
        reliable_review_class=reliable_review_class,
        round_index=int(round_index),
        changed=not np.array_equal(selected, seed.seed_support),
    )
    return LocalRefinementResult(
        state,
        roi,
        edges,
        too_large,
        no_hard,
        int(trimmed),
        {
            "roi_points": int(len(roi)),
            "edge_count": int(len(edges)),
            "hard_positive_points": int(np.count_nonzero(hp)),
            "hard_negative_points": int(np.count_nonzero(hn)),
            "selected_points": int(len(selected)),
            "prior_conflict": not _within_extent_limit(xyz[selected], prior.extents_q95_m),
        },
    )


def object_components(
    state: ObjectState,
    xyz_m: Any,
    config: RefinementConfig = RefinementConfig(),
) -> tuple[np.ndarray, ...]:
    ids = state.point_ids
    if not len(ids):
        return ()
    local_xyz = np.asarray(xyz_m, dtype=np.float64)[ids]
    tree = cKDTree(local_xyz)
    parent = np.arange(len(ids), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Radius connectivity, not mutual-kNN connectivity: density differences
    # must not manufacture a physical split.  Chunking avoids materialising a
    # potentially huge global pair matrix.
    for start in range(0, len(ids), 2048):
        neighborhoods = tree.query_ball_point(local_xyz[start : start + 2048], config.graph_edge_radius_m)
        for offset, neighbors in enumerate(neighborhoods):
            left = start + offset
            for right in neighbors:
                if int(right) > left:
                    union(left, int(right))
    roots = np.asarray([find(index) for index in range(len(ids))], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    count = int(labels.max()) + 1
    components = [ids[labels == index] for index in range(count)]
    return tuple(sorted(components, key=lambda values: (-len(values), int(values.min()))))


def fuse_objects_with_b0(
    b0_labels: Any,
    objects: Sequence[ObjectState],
    xyz_m: Any,
    *,
    config: RefinementConfig = RefinementConfig(),
) -> FusionResult:
    """Apply deterministic point ownership and conservative B0 carve guards.

    Semantic export is deliberately outside this function.  Callers must roll
    back an object's changes if final 33-way voting cannot export that object.
    """
    b0 = np.asarray(b0_labels, dtype=np.int64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if b0.ndim != 1 or len(b0) != len(xyz):
        raise ValueError("B0 labels and xyz must share a point axis")
    claims: dict[int, list[tuple[float, float, int, int, ObjectState]]] = {}
    for state in objects:
        margin_by_id = {int(point_id): float(value) for point_id, value in zip(state.point_ids, state.evidence_margin)}
        hard_by_id = {int(point_id): float(value) for point_id, value in zip(state.point_ids, state.hard_positive_counts)}
        anchors = set(state.anchor_ids.tolist())
        for point_id in state.point_ids:
            pid = int(point_id)
            claims.setdefault(pid, []).append((hard_by_id[pid], margin_by_id[pid], int(pid in anchors), state.object_id, state))
    owner: dict[int, ObjectState] = {}
    for point_id, rows in claims.items():
        rows.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        best = rows[0]
        # Background may only be claimed by two-view hard support.  Existing
        # B0 ownership additionally permits the candidate's own stable anchor;
        # anchor status is not misreported as new 2D evidence.
        if b0[point_id] < 0 and best[0] < 2:
            continue
        if b0[point_id] >= 0 and best[0] < 2 and not best[2]:
            continue
        if len(rows) > 1 and rows[0][:3] == rows[1][:3] and b0[point_id] >= 0:
            continue
        owner[point_id] = best[4]

    accepted = {state.object_id: [] for state in objects}
    rejected: set[int] = set()
    carve_rows: list[dict[str, Any]] = []
    for state in objects:
        proposed = np.asarray(sorted(point_id for point_id, value in owner.items() if value.object_id == state.object_id), dtype=np.int64)
        if len(proposed) < config.minimum_new_object_points or len(state.hard_positive_ids) == 0:
            rejected.add(state.object_id)
            continue
        veto = False
        for b0_id in sorted(int(value) for value in np.unique(b0[proposed]) if int(value) >= 0):
            original = np.flatnonzero(b0 == b0_id)
            carved = proposed[b0[proposed] == b0_id]
            fraction = len(carved) / max(len(original), 1)
            hard_counts = state.hard_positive_counts[np.isin(state.point_ids, carved)]
            if fraction > config.b0_max_carve_fraction:
                veto = True
            elif fraction > config.b0_two_view_carve_fraction and (not len(hard_counts) or hard_counts.min() < 3):
                veto = True
            remaining = np.setdiff1d(original, carved, assume_unique=True)
            if len(remaining) >= 2:
                local = cKDTree(xyz[remaining])
                pairs = local.query_pairs(config.graph_edge_radius_m, output_type="ndarray")
                if len(pairs):
                    graph = csr_matrix(
                        (
                            np.ones(2 * len(pairs), np.int8),
                            (
                                np.concatenate((pairs[:, 0], pairs[:, 1])),
                                np.concatenate((pairs[:, 1], pairs[:, 0])),
                            ),
                        ),
                        shape=(len(remaining), len(remaining)),
                    )
                    count, labels = connected_components(graph, directed=False)
                    sizes = np.bincount(labels, minlength=count)
                    if len(sizes) > 1 and np.sort(sizes)[-2] / len(remaining) >= config.b0_disconnect_fraction:
                        veto = True
            carve_rows.append({"object_id": state.object_id, "b0_id": b0_id, "fraction": fraction, "veto": veto})
        if veto:
            rejected.add(state.object_id)
        else:
            accepted[state.object_id] = proposed.tolist()

    labels = b0.copy()
    maximum = int(labels[labels >= 0].max()) if np.any(labels >= 0) else -1
    raw_by_object: dict[int, int] = {}
    for ordinal, state in enumerate(sorted(objects, key=lambda row: row.object_id)):
        point_ids = accepted.get(state.object_id, [])
        if not point_ids:
            continue
        raw_label = maximum + ordinal + 1
        labels[np.asarray(point_ids, dtype=np.int64)] = raw_label
        raw_by_object[state.object_id] = raw_label
    labels.setflags(write=False)
    return FusionResult(
        labels,
        raw_by_object,
        tuple(sorted(raw_by_object)),
        tuple(sorted(rejected)),
        {"carves": tuple(carve_rows), "claimed_point_count": int(sum(map(len, accepted.values())))},
    )


__all__ = [
    "FusionResult",
    "LocalRefinementResult",
    "SizePrior",
    "binary_graph_cut",
    "fuse_objects_with_b0",
    "local_roi_point_ids",
    "mutual_local_edges",
    "object_components",
    "refine_candidate_local",
    "size_prior_from_payload",
    "trim_oversize_additions",
]
