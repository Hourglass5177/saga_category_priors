from __future__ import annotations

"""Class-constrained assembly of frozen raw HDBSCAN fragments.

This module implements the pure, ground-truth-free core registered in section
33 of :mod:`TEACHER_PRIOR_V3_EXPERIMENT_PLAN.md`.  A ``FragmentGraph`` is built
exactly once from raw fragment membership, physical neighbourhoods and the
affinity feature.  Global (U) and class-shrunk (D) replay consume that same
graph; the only arm-dependent values are the size/support statistics used to
decide whether a proposed union improves the prior compatibility.

The code deliberately has no renderer, evaluator, GT or ScanNet imports.  It
also performs no full assignment, KNN fill, filtering, halo or rescue.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .category_candidate_prior_v2 import (
    size_platform_compatibility,
    trusted_core_support_threshold,
)
from .category_denoise import CandidateBank, pca_sorted_extents_m

FRAGMENT_GRAPH_SCHEMA = "saga-category-fragment-graph-v1"
FRAGMENT_MERGE_SCHEMA = "saga-category-fragment-merge-v1"
PHYSICAL_K = 24
AFFINITY_TOP_K = 4
MIN_CROSS_EDGES = 3
MAX_EDGE_DIAG_FRACTION = 0.1
PRIOR_IMPROVEMENT_EPSILON = 1e-6


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    result.setflags(write=False)
    return result


def _normalize_rows(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("affinity_features must be a matrix")
    if not np.isfinite(array).all():
        raise ValueError("affinity_features must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def _exact_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
        exact = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not np.isfinite(exact) or exact != float(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class FragmentNode:
    """One immutable raw non-noise HDBSCAN fragment."""

    fragment_id: int
    source_fragment_id: int
    point_ids: np.ndarray
    class_index: int
    class_name: str
    membership_mean: float
    semantic_score_mean: float

    def __post_init__(self) -> None:
        fragment_id = _exact_nonnegative_int(self.fragment_id, "fragment_id")
        source_id = _exact_nonnegative_int(
            self.source_fragment_id, "source_fragment_id"
        )
        class_index = _exact_nonnegative_int(self.class_index, "class_index")
        points = np.unique(np.asarray(self.point_ids, dtype=np.int64))
        if points.ndim != 1 or not len(points) or np.any(points < 0):
            raise ValueError("fragment point_ids must be non-empty and non-negative")
        membership = float(self.membership_mean)
        semantic = float(self.semantic_score_mean)
        if not np.isfinite((membership, semantic)).all():
            raise ValueError("fragment evidence must be finite")
        if not 0.0 <= membership <= 1.0 or not -1.0 <= semantic <= 1.0:
            raise ValueError("fragment evidence is outside its valid range")
        if not str(self.class_name):
            raise ValueError("fragment class_name must be non-empty")
        object.__setattr__(self, "fragment_id", fragment_id)
        object.__setattr__(self, "source_fragment_id", source_id)
        object.__setattr__(self, "class_index", class_index)
        object.__setattr__(self, "point_ids", _readonly(points, np.int64))
        object.__setattr__(self, "membership_mean", membership)
        object.__setattr__(self, "semantic_score_mean", semantic)


@dataclass(frozen=True)
class FragmentEdge:
    """Public evidence connecting two same-class raw fragments."""

    left_fragment_id: int
    right_fragment_id: int
    cross_edge_count: int
    affinity_cosine_median: float
    min_distance_m: float

    def __post_init__(self) -> None:
        left = _exact_nonnegative_int(self.left_fragment_id, "left_fragment_id")
        right = _exact_nonnegative_int(self.right_fragment_id, "right_fragment_id")
        if left == right:
            raise ValueError("fragment edges cannot be self edges")
        left, right = sorted((left, right))
        count = _exact_nonnegative_int(self.cross_edge_count, "cross_edge_count")
        cosine = float(self.affinity_cosine_median)
        distance = float(self.min_distance_m)
        if count < MIN_CROSS_EDGES:
            raise ValueError("fragment edge has too few cross-fragment point edges")
        if not np.isfinite((cosine, distance)).all() or distance < 0.0:
            raise ValueError("fragment edge evidence must be finite")
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("affinity cosine must be in [-1, 1]")
        object.__setattr__(self, "left_fragment_id", left)
        object.__setattr__(self, "right_fragment_id", right)
        object.__setattr__(self, "cross_edge_count", count)
        object.__setattr__(self, "affinity_cosine_median", cosine)
        object.__setattr__(self, "min_distance_m", distance)


def _edge_sort_key(
    edge: FragmentEdge, node_by_fragment: Mapping[int, FragmentNode]
) -> tuple[Any, ...]:
    endpoint_lineage = tuple(
        sorted(
            (
                node_by_fragment[edge.left_fragment_id].source_fragment_id,
                node_by_fragment[edge.right_fragment_id].source_fragment_id,
            )
        )
    )
    return (
        -edge.cross_edge_count,
        -edge.affinity_cosine_median,
        endpoint_lineage,
    )


@dataclass(frozen=True)
class FragmentGraph:
    """One canonical graph shared exactly by U and D replay."""

    nodes: tuple[FragmentNode, ...]
    edges: tuple[FragmentEdge, ...]
    point_count: int
    scene_scale_m_per_unit: float
    global_typical_diag_m: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema: str = FRAGMENT_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != FRAGMENT_GRAPH_SCHEMA:
            raise ValueError(f"unsupported fragment graph schema: {self.schema}")
        point_count = _exact_nonnegative_int(self.point_count, "point_count")
        scale = float(self.scene_scale_m_per_unit)
        diagonal = float(self.global_typical_diag_m)
        if not np.isfinite((scale, diagonal)).all() or scale <= 0.0 or diagonal <= 0.0:
            raise ValueError("scene scale and global typical diagonal must be positive")
        nodes = tuple(
            sorted(
                self.nodes,
                key=lambda node: (
                    node.class_index,
                    node.source_fragment_id,
                    node.fragment_id,
                ),
            )
        )
        fragment_ids = [node.fragment_id for node in nodes]
        source_ids = [node.source_fragment_id for node in nodes]
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("fragment_id values must be unique")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_fragment_id values must be unique")
        claimed = np.zeros(point_count, dtype=bool)
        classes: dict[int, int] = {}
        node_by_fragment: dict[int, FragmentNode] = {}
        for node in nodes:
            if np.any(node.point_ids >= point_count):
                raise ValueError("fragment point_id exceeds graph point axis")
            if np.any(claimed[node.point_ids]):
                raise ValueError("raw fragments must not overlap")
            claimed[node.point_ids] = True
            classes[node.fragment_id] = node.class_index
            node_by_fragment[node.fragment_id] = node
        edges = tuple(
            sorted(
                self.edges,
                key=lambda edge: _edge_sort_key(edge, node_by_fragment),
            )
        )
        edge_pairs: set[tuple[int, int]] = set()
        for edge in edges:
            pair = (edge.left_fragment_id, edge.right_fragment_id)
            if pair in edge_pairs:
                raise ValueError("fragment graph contains duplicate edge endpoints")
            edge_pairs.add(pair)
            if pair[0] not in classes or pair[1] not in classes:
                raise ValueError("fragment edge references an unknown node")
            if classes[pair[0]] != classes[pair[1]]:
                raise ValueError("fragment graph cannot connect different classes")
            if edge.min_distance_m > MAX_EDGE_DIAG_FRACTION * diagonal + 1e-12:
                raise ValueError("fragment edge exceeds the registered physical radius")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "point_count", point_count)
        object.__setattr__(self, "scene_scale_m_per_unit", scale)
        object.__setattr__(self, "global_typical_diag_m", diagonal)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def identity(self) -> tuple[Any, ...]:
        """Return a value identity suitable for exact U/D graph auditing."""

        node_identity = tuple(
            (
                node.source_fragment_id,
                tuple(int(value) for value in node.point_ids),
                node.class_index,
                node.class_name,
                node.membership_mean,
                node.semantic_score_mean,
            )
            for node in self.nodes
        )
        node_by_fragment = {node.fragment_id: node for node in self.nodes}
        edge_identity = tuple(
            (
                tuple(
                    sorted(
                        (
                            node_by_fragment[
                                edge.left_fragment_id
                            ].source_fragment_id,
                            node_by_fragment[
                                edge.right_fragment_id
                            ].source_fragment_id,
                        )
                    )
                ),
                edge.cross_edge_count,
                edge.affinity_cosine_median,
                edge.min_distance_m,
            )
            for edge in self.edges
        )
        return (
            self.schema,
            self.point_count,
            self.scene_scale_m_per_unit,
            self.global_typical_diag_m,
            node_identity,
            edge_identity,
        )


@dataclass(frozen=True)
class FragmentMergeDecision:
    round_index: int
    left_source_fragment_ids: tuple[int, ...]
    right_source_fragment_ids: tuple[int, ...]
    union_source_fragment_ids: tuple[int, ...]
    left_prior_score: float
    right_prior_score: float
    union_prior_score: float
    prior_eligible: bool
    mutual_best: bool
    accepted: bool
    reason: str
    cross_edge_count: int
    affinity_cosine_median: float


@dataclass(frozen=True)
class FragmentObject:
    """One final component, including support-rejected components for audit."""

    source_fragment_ids: tuple[int, ...]
    point_ids: np.ndarray
    class_index: int
    class_name: str
    metric_extents_m: tuple[float, float, float]
    n_raw: int
    G: float
    C: float
    P: float
    support_threshold: int
    base_score: float
    accepted: bool
    output_instance_id: int | None

    def __post_init__(self) -> None:
        lineage = tuple(
            sorted(
                _exact_nonnegative_int(value, "source_fragment_id")
                for value in self.source_fragment_ids
            )
        )
        if not lineage or len(lineage) != len(set(lineage)):
            raise ValueError("object lineage must be non-empty and unique")
        points = np.unique(np.asarray(self.point_ids, dtype=np.int64))
        if points.ndim != 1 or not len(points) or np.any(points < 0):
            raise ValueError("object point_ids must be non-empty and non-negative")
        object.__setattr__(self, "source_fragment_ids", lineage)
        object.__setattr__(self, "point_ids", _readonly(points, np.int64))


@dataclass(frozen=True)
class FragmentMergeResult:
    """Deterministic replay output for one prior arm."""

    mode: str
    graph: FragmentGraph
    objects: tuple[FragmentObject, ...]
    point_labels: np.ndarray
    decisions: tuple[FragmentMergeDecision, ...]
    diagnostics: Mapping[str, Any]
    schema: str = FRAGMENT_MERGE_SCHEMA

    def __post_init__(self) -> None:
        if self.mode not in {"global", "class"}:
            raise ValueError("merge mode must be 'global' or 'class'")
        labels = np.asarray(self.point_labels, dtype=np.int64)
        if labels.shape != (self.graph.point_count,) or np.any(labels < -1):
            raise ValueError("point_labels do not match the graph point axis")
        object.__setattr__(self, "point_labels", _readonly(labels, np.int64))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def _candidate_rows_by_id(bank: CandidateBank) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for row in bank.candidates:
        candidate_id = _exact_nonnegative_int(row.get("candidate_id"), "candidate_id")
        if candidate_id in result:
            raise ValueError("candidate bank contains duplicate candidate IDs")
        result[candidate_id] = row
    return result


def _raw_fragment_nodes(bank: CandidateBank) -> tuple[FragmentNode, ...]:
    labels = np.asarray(bank.branch_core_labels, dtype=np.int64)
    membership = np.asarray(bank.assignment_confidence, dtype=np.float64)
    semantic_class = np.asarray(bank.semantic_top1, dtype=np.int64)
    semantic_score = np.asarray(bank.semantic_top1_score, dtype=np.float64)
    arrays = (labels, membership, semantic_class, semantic_score)
    if any(value.shape != (bank.point_count,) for value in arrays):
        raise ValueError("candidate bank arrays do not share one point axis")
    if not np.isfinite(membership).all() or np.any((membership < 0.0) | (membership > 1.0)):
        raise ValueError("raw HDBSCAN membership must be finite and in [0, 1]")
    rows = _candidate_rows_by_id(bank)
    nodes: list[FragmentNode] = []
    observed_source_ids: set[int] = set()
    for fragment_id in sorted(int(value) for value in np.unique(labels) if value >= 0):
        if fragment_id not in rows:
            raise ValueError(f"raw fragment {fragment_id} has no candidate metadata")
        point_ids = np.flatnonzero(labels == fragment_id)
        row = rows[fragment_id]
        class_values = np.unique(semantic_class[point_ids])
        if len(class_values) != 1 or int(class_values[0]) < 0:
            raise ValueError("raw fragment members must share one predicted class")
        class_index = int(class_values[0])
        declared_index = int(row.get("branch_class_index", class_index))
        if declared_index != class_index:
            raise ValueError("raw fragment metadata disagrees with semantic routing")
        if class_index >= len(bank.class_names):
            raise ValueError("raw fragment class index is outside class_names")
        class_name = str(row.get("branch_class", bank.class_names[class_index]))
        if class_name != bank.class_names[class_index]:
            raise ValueError("raw fragment class name/index disagree")
        source_id = _exact_nonnegative_int(
            row.get("stable_source_id", fragment_id), "stable_source_id"
        )
        if source_id in observed_source_ids:
            raise ValueError("stable_source_id values must be unique")
        observed_source_ids.add(source_id)
        nodes.append(
            FragmentNode(
                fragment_id=fragment_id,
                source_fragment_id=source_id,
                point_ids=point_ids,
                class_index=class_index,
                class_name=class_name,
                membership_mean=float(np.mean(membership[point_ids])),
                semantic_score_mean=float(np.mean(semantic_score[point_ids])),
            )
        )
    extra_rows = set(rows).difference(node.fragment_id for node in nodes)
    if extra_rows:
        raise ValueError(
            "candidate metadata declares empty raw fragments: "
            f"{sorted(extra_rows)[:5]}"
        )
    return tuple(nodes)


def _mutual_affinity_pairs(
    point_ids: np.ndarray,
    xyz_m: np.ndarray,
    affinity: np.ndarray,
    *,
    physical_k: int,
    affinity_top_k: int,
) -> tuple[tuple[int, int, float, float], ...]:
    """Return canonical mutual top-affinity pairs within one predicted class."""

    from scipy.spatial import cKDTree

    indices = np.asarray(point_ids, dtype=np.int64)
    if len(indices) < 2:
        return ()
    width = min(max(int(physical_k), 1), len(indices) - 1)
    query_distance, query_neighbor = cKDTree(xyz_m[indices]).query(
        xyz_m[indices], k=width + 1, workers=1
    )
    query_distance = np.asarray(query_distance, dtype=np.float64)
    query_neighbor = np.asarray(query_neighbor, dtype=np.int64)
    if query_distance.ndim == 1:
        query_distance = query_distance[:, None]
        query_neighbor = query_neighbor[:, None]
    selected: set[tuple[int, int]] = set()
    for local_source, global_source in enumerate(indices):
        local_targets = query_neighbor[local_source]
        distances = query_distance[local_source]
        keep = local_targets != local_source
        local_targets = local_targets[keep]
        distances = distances[keep]
        global_targets = indices[local_targets]
        physical_order = np.lexsort((global_targets, distances))[:width]
        global_targets = global_targets[physical_order]
        similarities = affinity[global_targets] @ affinity[int(global_source)]
        affinity_order = np.lexsort((global_targets, -similarities))[
            : min(max(int(affinity_top_k), 1), len(global_targets))
        ]
        selected.update(
            (int(global_source), int(target))
            for target in global_targets[affinity_order]
        )

    mutual: list[tuple[int, int, float, float]] = []
    for source, target in sorted(selected):
        if source >= target or (target, source) not in selected:
            continue
        distance = float(np.linalg.norm(xyz_m[source] - xyz_m[target]))
        cosine = float(affinity[source] @ affinity[target])
        mutual.append((source, target, cosine, distance))
    return tuple(mutual)


def build_fragment_graph(
    bank: CandidateBank,
    xyz_scene: Any,
    affinity_features: Any,
    global_typical_diag_m: float,
    *,
    physical_k: int = PHYSICAL_K,
    affinity_top_k: int = AFFINITY_TOP_K,
    min_cross_edges: int = MIN_CROSS_EDGES,
    max_edge_diag_fraction: float = MAX_EDGE_DIAG_FRACTION,
) -> FragmentGraph:
    """Build the registered same-class physical/affinity fragment graph.

    ``branch_core_labels`` is interpreted as the raw non-noise HDBSCAN label
    vector.  The worker does not read ``global_pre_knn``, priors or GT.
    """

    if int(physical_k) != PHYSICAL_K or int(affinity_top_k) != AFFINITY_TOP_K:
        raise ValueError("fragment graph requires registered physical-24/top-4")
    if int(min_cross_edges) != MIN_CROSS_EDGES:
        raise ValueError("fragment graph requires at least three cross edges")
    if float(max_edge_diag_fraction) != MAX_EDGE_DIAG_FRACTION:
        raise ValueError("fragment graph requires the registered 0.1 diagonal radius")
    xyz = np.asarray(xyz_scene, dtype=np.float64)
    affinity = _normalize_rows(affinity_features)
    if xyz.shape != (bank.point_count, 3) or affinity.shape[0] != bank.point_count:
        raise ValueError("bank, xyz_scene and affinity_features must share a point axis")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite")
    scale = float(bank.scene_scale_m_per_unit)
    diagonal = float(global_typical_diag_m)
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("global_typical_diag_m must be finite and positive")
    nodes = _raw_fragment_nodes(bank)
    node_by_fragment = {node.fragment_id: node for node in nodes}
    point_fragment = np.full(bank.point_count, -1, dtype=np.int64)
    for node in nodes:
        point_fragment[node.point_ids] = node.fragment_id
    xyz_m = xyz * scale
    maximum_distance = MAX_EDGE_DIAG_FRACTION * diagonal
    grouped: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for node in nodes:
        by_class[node.class_index].append(node.point_ids)
    mutual_count = 0
    radius_rejected = 0
    for class_index in sorted(by_class):
        class_points = np.sort(np.concatenate(by_class[class_index]))
        for source, target, cosine, distance in _mutual_affinity_pairs(
            class_points,
            xyz_m,
            affinity,
            physical_k=PHYSICAL_K,
            affinity_top_k=AFFINITY_TOP_K,
        ):
            mutual_count += 1
            left = int(point_fragment[source])
            right = int(point_fragment[target])
            if left == right:
                continue
            if distance > maximum_distance:
                radius_rejected += 1
                continue
            pair = tuple(sorted((left, right)))
            grouped[pair].append((cosine, distance))

    edges: list[FragmentEdge] = []
    for (left, right), evidence in grouped.items():
        if len(evidence) < MIN_CROSS_EDGES:
            continue
        if node_by_fragment[left].class_index != node_by_fragment[right].class_index:
            raise AssertionError("class-local graph construction crossed classes")
        cosines = np.asarray([item[0] for item in evidence], dtype=np.float64)
        distances = np.asarray([item[1] for item in evidence], dtype=np.float64)
        edges.append(
            FragmentEdge(
                left_fragment_id=left,
                right_fragment_id=right,
                cross_edge_count=len(evidence),
                affinity_cosine_median=float(np.median(cosines)),
                min_distance_m=float(np.min(distances)),
            )
        )
    return FragmentGraph(
        nodes=nodes,
        edges=tuple(edges),
        point_count=bank.point_count,
        scene_scale_m_per_unit=scale,
        global_typical_diag_m=diagonal,
        diagnostics={
            "gt_used": False,
            "category_specific_prior_used": False,
            "physical_knn_k": PHYSICAL_K,
            "affinity_top_k": AFFINITY_TOP_K,
            "minimum_cross_edges": MIN_CROSS_EDGES,
            "maximum_cross_edge_distance_m": maximum_distance,
            "raw_fragment_count": len(nodes),
            "mutual_point_edge_count": mutual_count,
            "radius_rejected_cross_edge_count": radius_rejected,
            "fragment_edge_count": len(edges),
        },
    )


@dataclass(frozen=True)
class _Component:
    fragment_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ComponentScore:
    metric_extents_m: tuple[float, float, float]
    n_raw: int
    G: float
    C: float
    P: float
    support_threshold: int
    base_score: float


@dataclass(frozen=True)
class _AggregatedEdge:
    left: tuple[int, ...]
    right: tuple[int, ...]
    cross_edge_count: int
    affinity_cosine_median: float
    min_distance_m: float


def _prior_node(
    priors: Mapping[str, Any], class_name: str, mode: str
) -> Mapping[str, Any]:
    global_node = priors.get("global")
    if not isinstance(global_node, Mapping) or not isinstance(global_node.get("shrunk"), Mapping):
        raise TypeError("category priors are missing a global shrunk node")
    if mode == "global":
        return global_node
    categories = priors.get("categories")
    class_node = categories.get(class_name) if isinstance(categories, Mapping) else None
    if isinstance(class_node, Mapping) and isinstance(class_node.get("shrunk"), Mapping):
        return class_node
    return global_node


def _component_points(
    component: _Component, node_by_id: Mapping[int, FragmentNode]
) -> np.ndarray:
    return np.sort(
        np.concatenate([node_by_id[value].point_ids for value in component.fragment_ids])
    )


def _component_lineage(
    component: _Component, node_by_id: Mapping[int, FragmentNode]
) -> tuple[int, ...]:
    return tuple(sorted(node_by_id[value].source_fragment_id for value in component.fragment_ids))


def _score_component(
    component: _Component,
    *,
    node_by_id: Mapping[int, FragmentNode],
    xyz_scene: np.ndarray,
    scene_scale_m_per_unit: float,
    priors: Mapping[str, Any],
    mode: str,
) -> _ComponentScore:
    points = _component_points(component, node_by_id)
    ordered_fragment_ids = tuple(
        sorted(
            component.fragment_ids,
            key=lambda value: node_by_id[value].source_fragment_id,
        )
    )
    first = node_by_id[ordered_fragment_ids[0]]
    if any(node_by_id[value].class_index != first.class_index for value in component.fragment_ids):
        raise AssertionError("merge component crossed predicted classes")
    extents = pca_sorted_extents_m(xyz_scene[points], scene_scale_m_per_unit)
    node = _prior_node(priors, first.class_name, mode)
    g_value = size_platform_compatibility({"metric_extents_m": extents}, node)
    threshold = trusted_core_support_threshold(
        priors, first.class_name, "uniform" if mode == "global" else "class"
    )
    c_value = min(1.0, len(points) / threshold)
    weights = np.asarray(
        [len(node_by_id[value].point_ids) for value in ordered_fragment_ids],
        dtype=np.float64,
    )
    semantic = float(
        np.average(
            [node_by_id[value].semantic_score_mean for value in ordered_fragment_ids],
            weights=weights,
        )
    )
    membership = float(
        np.average(
            [node_by_id[value].membership_mean for value in ordered_fragment_ids],
            weights=weights,
        )
    )
    base_score = math.sqrt(max(semantic, 0.0) * max(membership, 0.0))
    return _ComponentScore(
        metric_extents_m=tuple(float(value) for value in extents),
        n_raw=len(points),
        G=g_value,
        C=c_value,
        P=g_value * c_value,
        support_threshold=threshold,
        base_score=base_score,
    )


def _weighted_median(values: Sequence[float], weights: Sequence[int]) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.int64)
    order = np.argsort(value, kind="mergesort")
    value = value[order]
    weight = weight[order]
    cutoff = (int(np.sum(weight)) + 1) / 2.0
    return float(value[int(np.searchsorted(np.cumsum(weight), cutoff, side="left"))])


def _aggregate_component_edges(
    components: Sequence[_Component],
    graph: FragmentGraph,
    node_by_id: Mapping[int, FragmentNode],
) -> tuple[_AggregatedEdge, ...]:
    owner: dict[int, tuple[int, ...]] = {}
    for component in components:
        for fragment_id in component.fragment_ids:
            owner[fragment_id] = component.fragment_ids
    grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[FragmentEdge]] = defaultdict(list)
    for edge in graph.edges:
        left = owner[edge.left_fragment_id]
        right = owner[edge.right_fragment_id]
        if left == right:
            continue
        pair = tuple(
            sorted(
                (left, right),
                key=lambda ids: _component_lineage(
                    _Component(ids), node_by_id
                ),
            )
        )
        grouped[pair].append(edge)
    output: list[_AggregatedEdge] = []
    for (left, right), edges in grouped.items():
        output.append(
            _AggregatedEdge(
                left=left,
                right=right,
                cross_edge_count=sum(edge.cross_edge_count for edge in edges),
                affinity_cosine_median=_weighted_median(
                    [edge.affinity_cosine_median for edge in edges],
                    [edge.cross_edge_count for edge in edges],
                ),
                min_distance_m=min(edge.min_distance_m for edge in edges),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda edge: (
                -edge.cross_edge_count,
                -edge.affinity_cosine_median,
                _component_lineage(_Component(edge.left), node_by_id),
                _component_lineage(_Component(edge.right), node_by_id),
            ),
        )
    )


def _best_neighbors(
    components: Sequence[_Component],
    edges: Sequence[_AggregatedEdge],
    node_by_id: Mapping[int, FragmentNode],
) -> dict[tuple[int, ...], tuple[int, ...]]:
    incident: dict[tuple[int, ...], list[_AggregatedEdge]] = defaultdict(list)
    for edge in edges:
        incident[edge.left].append(edge)
        incident[edge.right].append(edge)
    result: dict[tuple[int, ...], tuple[int, ...]] = {}
    for component in components:
        candidates = incident.get(component.fragment_ids, [])
        if not candidates:
            continue
        candidates.sort(
            key=lambda edge: (
                -edge.cross_edge_count,
                -edge.affinity_cosine_median,
                _component_lineage(_Component(edge.left), node_by_id),
                _component_lineage(_Component(edge.right), node_by_id),
            )
        )
        if len(candidates) > 1:
            first_evidence = (
                candidates[0].cross_edge_count,
                candidates[0].affinity_cosine_median,
            )
            second_evidence = (
                candidates[1].cross_edge_count,
                candidates[1].affinity_cosine_median,
            )
            if first_evidence == second_evidence:
                continue
        edge = candidates[0]
        result[component.fragment_ids] = (
            edge.right if edge.left == component.fragment_ids else edge.left
        )
    return result


def merge_category_fragments(
    graph: FragmentGraph,
    xyz_scene: Any,
    priors: Mapping[str, Any],
    mode: str,
) -> FragmentMergeResult:
    """Replay deterministic mutual-best assembly with global or class priors."""

    if mode not in {"global", "class"}:
        raise ValueError("mode must be 'global' or 'class'")
    xyz = np.asarray(xyz_scene, dtype=np.float64)
    if xyz.shape != (graph.point_count, 3) or not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite and match the graph point axis")
    node_by_id = {node.fragment_id: node for node in graph.nodes}
    components = [
        _Component((node.fragment_id,))
        for node in sorted(graph.nodes, key=lambda item: item.source_fragment_id)
    ]
    decisions: list[FragmentMergeDecision] = []
    round_index = 0
    while True:
        component_by_ids = {component.fragment_ids: component for component in components}
        aggregated = _aggregate_component_edges(components, graph, node_by_id)
        if not aggregated:
            break
        component_scores = {
            ids: _score_component(
                component,
                node_by_id=node_by_id,
                xyz_scene=xyz,
                scene_scale_m_per_unit=graph.scene_scale_m_per_unit,
                priors=priors,
                mode=mode,
            )
            for ids, component in component_by_ids.items()
        }
        edge_scores: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            tuple[_ComponentScore, bool],
        ] = {}
        eligible_edges: list[_AggregatedEdge] = []
        for edge in aggregated:
            pair = (edge.left, edge.right)
            union = _Component(tuple(sorted(edge.left + edge.right)))
            union_score = _score_component(
                union,
                node_by_id=node_by_id,
                xyz_scene=xyz,
                scene_scale_m_per_unit=graph.scene_scale_m_per_unit,
                priors=priors,
                mode=mode,
            )
            eligible = bool(
                union_score.P
                > max(
                    component_scores[edge.left].P,
                    component_scores[edge.right].P,
                )
                + PRIOR_IMPROVEMENT_EPSILON
            )
            edge_scores[pair] = (union_score, eligible)
            if eligible:
                eligible_edges.append(edge)

        # The prior participates in neighbour selection: an attractive public
        # edge whose union makes the object less plausible cannot block a
        # weaker edge whose union improves it.
        best = _best_neighbors(components, eligible_edges, node_by_id)
        mutual_pairs = {
            tuple(
                sorted(
                    (left, right),
                    key=lambda ids: _component_lineage(
                        _Component(ids), node_by_id
                    ),
                )
            )
            for left, right in best.items()
            if best.get(right) == left
        }
        accepted_pairs: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        ] = []
        for edge in aggregated:
            left_ids, right_ids = edge.left, edge.right
            left = component_by_ids[left_ids]
            right = component_by_ids[right_ids]
            union_ids = tuple(sorted(left_ids + right_ids))
            union = _Component(union_ids)
            left_score = component_scores[left_ids]
            right_score = component_scores[right_ids]
            union_score, prior_eligible = edge_scores[(left_ids, right_ids)]
            mutual_best = (left_ids, right_ids) in mutual_pairs
            accepted = prior_eligible and mutual_best
            if accepted:
                reason = "prior_eligible_mutual_best"
            elif not prior_eligible:
                reason = "prior_not_improved"
            else:
                reason = "not_mutual_best"
            decisions.append(
                FragmentMergeDecision(
                    round_index=round_index,
                    left_source_fragment_ids=_component_lineage(left, node_by_id),
                    right_source_fragment_ids=_component_lineage(right, node_by_id),
                    union_source_fragment_ids=_component_lineage(union, node_by_id),
                    left_prior_score=left_score.P,
                    right_prior_score=right_score.P,
                    union_prior_score=union_score.P,
                    prior_eligible=prior_eligible,
                    mutual_best=mutual_best,
                    accepted=accepted,
                    reason=reason,
                    cross_edge_count=edge.cross_edge_count,
                    affinity_cosine_median=edge.affinity_cosine_median,
                )
            )
            if accepted:
                accepted_pairs.append((left_ids, right_ids, union_ids))
        if not accepted_pairs:
            break
        replaced = {ids for left, right, _ in accepted_pairs for ids in (left, right)}
        next_components = [
            component for component in components if component.fragment_ids not in replaced
        ]
        next_components.extend(_Component(union) for _, _, union in accepted_pairs)
        components = sorted(
            next_components,
            key=lambda item: _component_lineage(item, node_by_id),
        )
        round_index += 1

    object_rows: list[FragmentObject] = []
    labels = np.full(graph.point_count, -1, dtype=np.int64)
    accepted_output_id = 0
    for component in sorted(
        components, key=lambda item: _component_lineage(item, node_by_id)
    ):
        score = _score_component(
            component,
            node_by_id=node_by_id,
            xyz_scene=xyz,
            scene_scale_m_per_unit=graph.scene_scale_m_per_unit,
            priors=priors,
            mode=mode,
        )
        points = _component_points(component, node_by_id)
        first = node_by_id[component.fragment_ids[0]]
        accepted = score.n_raw >= score.support_threshold
        output_id = accepted_output_id if accepted else None
        if accepted:
            labels[points] = accepted_output_id
            accepted_output_id += 1
        object_rows.append(
            FragmentObject(
                source_fragment_ids=_component_lineage(component, node_by_id),
                point_ids=points,
                class_index=first.class_index,
                class_name=first.class_name,
                metric_extents_m=score.metric_extents_m,
                n_raw=score.n_raw,
                G=score.G,
                C=score.C,
                P=score.P,
                support_threshold=score.support_threshold,
                base_score=score.base_score,
                accepted=accepted,
                output_instance_id=output_id,
            )
        )

    covered = np.zeros(graph.point_count, dtype=bool)
    final_lineage = {
        source_id
        for row in object_rows
        for source_id in row.source_fragment_ids
    }
    expected_lineage = {node.source_fragment_id for node in graph.nodes}
    for row in object_rows:
        if row.accepted:
            covered[row.point_ids] = True
    if not np.array_equal(covered, labels >= 0):
        raise AssertionError("fragment output contains orphan labels")
    if final_lineage != expected_lineage:
        raise AssertionError("fragment output lost or duplicated lineage")
    return FragmentMergeResult(
        mode=mode,
        graph=graph,
        objects=tuple(object_rows),
        point_labels=labels,
        decisions=tuple(decisions),
        diagnostics={
            "gt_used": False,
            "graph_identity_equal_to_input": True,
            "raw_fragment_count": len(graph.nodes),
            "final_component_count": len(object_rows),
            "accepted_object_count": accepted_output_id,
            "merge_proposal_count": len(decisions),
            "accepted_merge_count": sum(row.accepted for row in decisions),
            "orphan_count": 0,
            "negative_metadata_count": 0,
            "core_full_contract_violation_count": 0,
            "overlap_ownership_violation_count": 0,
            "lineage_missing_fragment_count": 0,
            "deterministic_algorithm": True,
        },
    )


def merge_same_graph_both_modes(
    graph: FragmentGraph,
    xyz_scene: Any,
    priors: Mapping[str, Any],
) -> tuple[FragmentMergeResult, FragmentMergeResult]:
    """Replay U/D and assert that both results retain the exact input graph."""

    uniform = merge_category_fragments(graph, xyz_scene, priors, "global")
    class_shrunk = merge_category_fragments(graph, xyz_scene, priors, "class")
    if uniform.graph is not graph or class_shrunk.graph is not graph:
        raise AssertionError("U/D replay did not preserve the shared graph object")
    if uniform.graph.identity() != class_shrunk.graph.identity():
        raise AssertionError("U/D replay graph identity differs")
    return uniform, class_shrunk


__all__ = [
    "AFFINITY_TOP_K",
    "FRAGMENT_GRAPH_SCHEMA",
    "FRAGMENT_MERGE_SCHEMA",
    "MAX_EDGE_DIAG_FRACTION",
    "MIN_CROSS_EDGES",
    "PHYSICAL_K",
    "PRIOR_IMPROVEMENT_EPSILON",
    "FragmentEdge",
    "FragmentGraph",
    "FragmentMergeDecision",
    "FragmentMergeResult",
    "FragmentNode",
    "FragmentObject",
    "build_fragment_graph",
    "merge_category_fragments",
    "merge_same_graph_both_modes",
]
