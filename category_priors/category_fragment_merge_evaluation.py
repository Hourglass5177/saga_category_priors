from __future__ import annotations

"""GT-aware offline evaluation for the section-33 fragment merge study.

The graph builder and merger must not import this module.  Ground truth enters
only through :class:`ClusterEvaluationScene`, after a graph and both paired
merge results have already been materialised.  Scene, rather than fragment,
edge, or candidate, is the independent experimental unit.
"""

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .category_candidate_prior_evaluation import binary_average_precision
from .category_cluster_evaluation import ClusterEvaluationScene

SCHEMA = "saga-category-fragment-merge-evaluation-v1"
DEV2_SCENE_IDS = ("scene0645_00", "scene0025_01")
DEV8_SCENE_IDS = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)

GRAPH_IOU050_MIN = 6
GRAPH_TINY_SMALL_RECALL025_MIN = 0.20
MECHANICAL_SCORE_DELTA_MIN = 0.01
MECHANICAL_SCORE_FRACTION_MIN = 0.10
MECHANICAL_DECISION_MIN = 5
MECHANICAL_CLASS_MIN = 2
MECHANICAL_SCENE_MIN = 2

DEV2_D_IOU050_MIN = 4
DEV2_D_PRECISION025_MIN = 0.10
DEV2_PRECISION_RELATIVE_GAIN_MIN = 0.25
DEV2_UNSUPPORTED_DROP_MIN = 0.10
DEV2_CANDIDATE_RATIO_MAX = 1.25
DEV2_NONWITNESS_GT_IOU_DROP_MAX = 0.05

DEV8_D_IOU050_MIN = 12
DEV8_D_IOU050_SCENE_MIN = 4
DEV8_D_PRECISION025_MIN = 0.10
DEV8_TINY_SMALL_RECALL025_MIN = 0.20
DEV8_AP025_GAIN_MIN = 0.002
DEV8_AP050_DROP_MAX = 0.002
DEV8_POSITIVE_SCENE_MIN = 5
DEV8_FP_TP_WORSENING_MAX = 0.20

_EPS = 1e-12


def _field(
    value: Any,
    *names: str,
    required: bool = True,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if required:
        raise ValueError(f"record is missing one of {names}")
    return default


def _records(value: Any, *names: str) -> tuple[Any, ...]:
    rows = _field(value, *names)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{names[0]} must be a sequence")
    return tuple(rows)


def _readonly_ids(values: Any, *, name: str, upper: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(result < 0) or (upper is not None and np.any(result >= upper)):
        raise ValueError(f"{name} contains an invalid index")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices")
    result = np.array(np.sort(result), dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _fraction(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _class_id(scene: ClusterEvaluationScene, class_name: str) -> int:
    try:
        return int(scene.class_name_to_id[str(class_name)])
    except KeyError as exc:
        raise ValueError(f"scene taxonomy does not define {class_name!r}") from exc


@dataclass(frozen=True)
class _Node:
    fragment_id: int
    source_fragment_id: int
    point_ids: np.ndarray
    class_name: str


@dataclass(frozen=True)
class _Edge:
    left: int
    right: int


@dataclass(frozen=True)
class _Object:
    source_fragment_ids: tuple[int, ...]
    point_ids: np.ndarray
    class_name: str
    base_score: float
    accepted: bool
    output_instance_id: int | None


def _normalise_graph(graph: Any) -> tuple[dict[int, _Node], tuple[_Edge, ...], int]:
    point_count = int(_field(graph, "point_count"))
    if point_count <= 0:
        raise ValueError("fragment graph point_count must be positive")
    nodes: dict[int, _Node] = {}
    source_ids: set[int] = set()
    owner = np.full(point_count, -1, dtype=np.int64)
    for row in _records(graph, "nodes", "fragments"):
        fragment_id = int(_field(row, "fragment_id", "source_fragment_id"))
        source_fragment_id = int(
            _field(row, "source_fragment_id", required=False, default=fragment_id)
        )
        if fragment_id < 0 or fragment_id in nodes:
            raise ValueError("fragment ids must be unique non-negative integers")
        if source_fragment_id < 0 or source_fragment_id in source_ids:
            raise ValueError(
                "source_fragment_id values must be unique non-negative integers"
            )
        source_ids.add(source_fragment_id)
        point_ids = _readonly_ids(
            _field(row, "point_ids", "gaussian_ids", "member_ids"),
            name="fragment point_ids",
            upper=point_count,
        )
        if not len(point_ids):
            raise ValueError("raw fragments must not be empty")
        if np.any(owner[point_ids] >= 0):
            raise ValueError("raw fragment masks must be disjoint")
        owner[point_ids] = fragment_id
        class_name = str(_field(row, "class_name", "branch_class"))
        if not class_name:
            raise ValueError("fragment class name must not be empty")
        nodes[fragment_id] = _Node(
            fragment_id, source_fragment_id, point_ids, class_name
        )
    if not nodes:
        raise ValueError("fragment graph must contain at least one node")
    edges: list[_Edge] = []
    seen: set[tuple[int, int]] = set()
    for row in _records(graph, "edges"):
        left = int(_field(row, "left_fragment_id", "left"))
        right = int(_field(row, "right_fragment_id", "right"))
        if left == right or left not in nodes or right not in nodes:
            raise ValueError("fragment edge has an invalid endpoint")
        if nodes[left].class_name != nodes[right].class_name:
            raise ValueError("public fragment graph contains a cross-class edge")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError("fragment graph contains duplicate edges")
        seen.add(key)
        edges.append(_Edge(*key))
    edges.sort(key=lambda row: (row.left, row.right))
    return nodes, tuple(edges), point_count


def _normalise_objects(result: Any, point_count: int) -> tuple[_Object, ...]:
    objects: list[_Object] = []
    for row in _records(result, "objects", "candidates"):
        lineage = tuple(
            sorted(
                int(value)
                for value in _field(row, "source_fragment_ids", "fragment_ids")
            )
        )
        if not lineage or len(set(lineage)) != len(lineage) or min(lineage) < 0:
            raise ValueError("merge object has an invalid fragment lineage")
        point_ids = _readonly_ids(
            _field(row, "point_ids", "gaussian_ids", "full_ids"),
            name="merge object point_ids",
            upper=point_count,
        )
        score = float(_field(row, "base_score", "score", "Q"))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("merge object base score must be finite and in [0, 1]")
        output_value = _field(
            row,
            "output_instance_id",
            "candidate_id",
            required=False,
            default=None,
        )
        objects.append(
            _Object(
                source_fragment_ids=lineage,
                point_ids=point_ids,
                class_name=str(_field(row, "class_name", "branch_class")),
                base_score=score,
                accepted=bool(_field(row, "accepted", required=False, default=True)),
                output_instance_id=(
                    None if output_value is None else int(output_value)
                ),
            )
        )
    return tuple(objects)


@dataclass(frozen=True)
class _ProjectionIndex:
    object_point_counts: np.ndarray
    gaussian_object_point_counts: csr_matrix
    gaussian_to_gt_object_indices: np.ndarray


def _projection_index(scene: ClusterEvaluationScene) -> _ProjectionIndex:
    object_count = len(scene.gt_object_class_ids)
    point_objects = np.asarray(scene.gt_point_object_indices, dtype=np.int64)
    gaussians = np.asarray(scene.gt_to_gaussian_indices, dtype=np.int64)
    valid = (point_objects >= 0) & (gaussians >= 0)
    counts = csr_matrix(
        (
            np.ones(int(np.count_nonzero(valid)), dtype=np.int64),
            (gaussians[valid], point_objects[valid]),
        ),
        shape=(len(scene.gaussian_to_gt_object_indices), object_count),
        dtype=np.int64,
    )
    return _ProjectionIndex(
        object_point_counts=np.bincount(
            point_objects[point_objects >= 0], minlength=object_count
        ).astype(np.int64, copy=False),
        gaussian_object_point_counts=counts,
        gaussian_to_gt_object_indices=np.asarray(
            scene.gaussian_to_gt_object_indices, dtype=np.int64
        ),
    )


def _candidate_iou(
    projection: _ProjectionIndex,
    point_ids: np.ndarray,
) -> np.ndarray:
    """IoU on mapped GT points, with unsupported Gaussians retained as FP."""

    intersections = np.asarray(
        projection.gaussian_object_point_counts[point_ids].sum(axis=0)
    ).ravel()
    predicted_point_count = int(np.sum(intersections))
    unsupported = int(
        np.count_nonzero(projection.gaussian_to_gt_object_indices[point_ids] < 0)
    )
    union = (
        projection.object_point_counts
        + predicted_point_count
        + unsupported
        - intersections
    )
    return np.divide(
        intersections,
        union,
        out=np.zeros(len(projection.object_point_counts), dtype=np.float64),
        where=union > 0,
    )


def _dominant_gt(scene: ClusterEvaluationScene, point_ids: np.ndarray) -> int | None:
    mapped = np.asarray(scene.gaussian_to_gt_object_indices)[point_ids]
    mapped = mapped[mapped >= 0]
    if not len(mapped):
        return None
    counts = np.bincount(mapped, minlength=len(scene.gt_object_class_ids))
    return int(np.flatnonzero(counts == counts.max())[0])


@dataclass(frozen=True)
class FragmentGraphOracleMetrics:
    scene_id: str
    graph_node_count: int
    graph_edge_count: int
    same_gt_edge_count: int
    different_gt_edge_count: int
    unknown_gt_edge_count: int
    same_class_iou_025_count: int
    same_class_iou_050_count: int
    tiny_small_gt_count: int
    tiny_small_iou_025_count: int
    best_iou_by_gt: tuple[float, ...]
    candidate_rows: tuple[Mapping[str, Any], ...]

    @property
    def tiny_small_recall_025(self) -> float:
        return _fraction(self.tiny_small_iou_025_count, self.tiny_small_gt_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in self.__dict__.items()
                if key != "candidate_rows"
            },
            "best_iou_by_gt": list(self.best_iou_by_gt),
            "tiny_small_recall_025": self.tiny_small_recall_025,
            "candidate_rows": [dict(row) for row in self.candidate_rows],
        }


def evaluate_fragment_graph_oracle(
    scene: ClusterEvaluationScene,
    graph: Any,
) -> FragmentGraphOracleMetrics:
    """Measure the best dominant-fragment component allowed by same-GT edges."""

    nodes, edges, point_count = _normalise_graph(graph)
    if point_count != len(scene.gaussian_to_gt_object_indices):
        raise ValueError("fragment graph and GT projection differ in point count")
    projection = _projection_index(scene)
    dominant = {
        fragment_id: _dominant_gt(scene, node.point_ids)
        for fragment_id, node in nodes.items()
    }
    adjacency: dict[int, set[int]] = {fragment_id: set() for fragment_id in nodes}
    same_gt = different_gt = unknown_gt = 0
    for edge in edges:
        left_gt, right_gt = dominant[edge.left], dominant[edge.right]
        if left_gt is None or right_gt is None:
            unknown_gt += 1
        elif left_gt == right_gt:
            same_gt += 1
            adjacency[edge.left].add(edge.right)
            adjacency[edge.right].add(edge.left)
        else:
            different_gt += 1

    best = np.zeros(len(scene.gt_object_class_ids), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for object_index, gt_class_id in enumerate(scene.gt_object_class_ids):
        eligible = [
            fragment_id
            for fragment_id, node in nodes.items()
            if dominant[fragment_id] == object_index
            and _class_id(scene, node.class_name) == int(gt_class_id)
        ]
        if not eligible:
            continue
        allowed = set(eligible)
        components: list[tuple[int, ...]] = []
        unseen = set(allowed)
        while unseen:
            root = min(unseen)
            component: set[int] = set()
            queue: deque[int] = deque((root,))
            while queue:
                current = queue.popleft()
                if current in component:
                    continue
                component.add(current)
                queue.extend(
                    sorted(adjacency[current].intersection(allowed - component))
                )
            unseen.difference_update(component)
            components.append(tuple(sorted(component)))
        component_rows: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        for component in components:
            points = np.unique(
                np.concatenate([nodes[item].point_ids for item in component])
            )
            component_rows.append(
                (
                    float(_candidate_iou(projection, points)[object_index]),
                    component,
                    points,
                )
            )
        # This is an evaluation-only upper bound: choose the connected
        # component with the highest true IoU.  Stable fragment lineage breaks
        # exact ties, so node iteration order cannot change the oracle.
        iou, component, point_ids = min(
            component_rows, key=lambda row: (-row[0], row[1])
        )
        best[object_index] = iou
        rows.append(
            {
                "scene_id": scene.scene_id,
                "gt_object_index": object_index,
                "gt_instance_id": int(scene.gt_object_instance_ids[object_index]),
                "class_id": int(gt_class_id),
                "size_bin": scene.gt_object_size_bins[object_index],
                "root_fragment_id": int(component[0]),
                "eligible_component_count": len(components),
                "source_fragment_ids": list(component),
                "point_count": len(point_ids),
                "same_class_iou": iou,
            }
        )
    tiny = np.asarray(
        [value in {"tiny", "small"} for value in scene.gt_object_size_bins],
        dtype=bool,
    )
    return FragmentGraphOracleMetrics(
        scene_id=scene.scene_id,
        graph_node_count=len(nodes),
        graph_edge_count=len(edges),
        same_gt_edge_count=same_gt,
        different_gt_edge_count=different_gt,
        unknown_gt_edge_count=unknown_gt,
        same_class_iou_025_count=int(np.count_nonzero(best >= 0.25)),
        same_class_iou_050_count=int(np.count_nonzero(best >= 0.50)),
        tiny_small_gt_count=int(np.count_nonzero(tiny)),
        tiny_small_iou_025_count=int(np.count_nonzero(tiny & (best >= 0.25))),
        best_iou_by_gt=tuple(map(float, best)),
        candidate_rows=tuple(rows),
    )


@dataclass(frozen=True)
class FragmentMergeSceneMetrics:
    scene_id: str
    mode: str
    candidate_count: int
    candidate_point_count: int
    unsupported_point_count: int
    unsupported_candidate_count: int
    same_class_iou_025_count: int
    same_class_iou_050_count: int
    tiny_small_gt_count: int
    tiny_small_iou_025_count: int
    tiny_small_iou_050_count: int
    best_iou_by_gt: tuple[float, ...]
    lineage_violation_count: int
    overlap_ownership_violation_count: int
    orphan_count: int
    negative_metadata_count: int
    core_full_contract_violation_count: int
    determinism_violation_count: int
    candidate_rows: tuple[Mapping[str, Any], ...]

    @property
    def candidate_precision_025(self) -> float:
        return _fraction(self.same_class_iou_025_count, self.candidate_count)

    @property
    def candidate_precision_050(self) -> float:
        return _fraction(self.same_class_iou_050_count, self.candidate_count)

    @property
    def unsupported_fraction(self) -> float:
        return _fraction(self.unsupported_point_count, self.candidate_point_count)

    @property
    def tiny_small_recall_025(self) -> float:
        return _fraction(self.tiny_small_iou_025_count, self.tiny_small_gt_count)

    @property
    def tiny_small_recall_050(self) -> float:
        return _fraction(self.tiny_small_iou_050_count, self.tiny_small_gt_count)

    @property
    def output_contract_violation_count(self) -> int:
        return (
            self.lineage_violation_count
            + self.overlap_ownership_violation_count
            + self.orphan_count
            + self.negative_metadata_count
            + self.core_full_contract_violation_count
            + self.determinism_violation_count
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **{
                key: value
                for key, value in self.__dict__.items()
                if key != "candidate_rows"
            },
            "best_iou_by_gt": list(self.best_iou_by_gt),
            "candidate_precision_025": self.candidate_precision_025,
            "candidate_precision_050": self.candidate_precision_050,
            "unsupported_fraction": self.unsupported_fraction,
            "tiny_small_recall_025": self.tiny_small_recall_025,
            "tiny_small_recall_050": self.tiny_small_recall_050,
            "output_contract_violation_count": self.output_contract_violation_count,
            "candidate_rows": [dict(row) for row in self.candidate_rows],
        }


def evaluate_fragment_merge_scene(
    scene: ClusterEvaluationScene,
    graph: Any,
    result: Any,
) -> FragmentMergeSceneMetrics:
    """Evaluate one already-materialised U or D object partition."""

    nodes, _, point_count = _normalise_graph(graph)
    if point_count != len(scene.gaussian_to_gt_object_indices):
        raise ValueError("fragment graph and GT projection differ in point count")
    projection = _projection_index(scene)
    mode = str(_field(result, "mode", "condition"))
    if mode not in {"global", "class"}:
        raise ValueError("merge result mode must be 'global' or 'class'")
    all_objects = _normalise_objects(result, point_count)
    objects = tuple(row for row in all_objects if row.accepted)
    raw_point_labels = _field(result, "point_labels", required=False, default=None)
    point_labels = None
    if raw_point_labels is not None:
        point_labels = np.asarray(raw_point_labels, dtype=np.int64)
        if point_labels.shape != (point_count,) or np.any(point_labels < -1):
            raise ValueError("merge result point_labels violate the output contract")
    source_nodes = {row.source_fragment_id: row for row in nodes.values()}
    source_owner: dict[int, int] = {}
    point_owner = np.full(point_count, -1, dtype=np.int64)
    lineage_violations = 0
    overlap_violations = 0
    rows: list[dict[str, Any]] = []
    best_by_gt = np.zeros(len(scene.gt_object_class_ids), dtype=np.float64)
    unsupported_points = 0
    unsupported_candidates = 0
    candidate_points = 0
    negative_metadata = int(
        sum(
            (
                item.accepted
                and item.output_instance_id is not None
                and item.output_instance_id < 0
            )
            or (not item.accepted and item.output_instance_id is not None)
            for item in all_objects
        )
    )
    expected_labels = np.full(point_count, -1, dtype=np.int64)
    for object_index, item in enumerate(objects):
        negative_metadata += int(any(value < 0 for value in item.source_fragment_ids))
        missing = [
            value for value in item.source_fragment_ids if value not in source_nodes
        ]
        duplicate_lineage = [
            value for value in item.source_fragment_ids if value in source_owner
        ]
        lineage_violations += len(missing) + len(duplicate_lineage)
        for value in item.source_fragment_ids:
            source_owner.setdefault(value, object_index)
        known_lineage = [
            value for value in item.source_fragment_ids if value in source_nodes
        ]
        expected = (
            np.unique(
                np.concatenate(
                    [source_nodes[value].point_ids for value in known_lineage]
                )
            )
            if known_lineage
            else np.empty(0, dtype=np.int64)
        )
        if not np.array_equal(expected, item.point_ids):
            lineage_violations += 1
        overlap = point_owner[item.point_ids] >= 0
        overlap_violations += int(np.count_nonzero(overlap))
        point_owner[item.point_ids[~overlap]] = object_index
        if point_labels is not None:
            if item.output_instance_id is None or item.output_instance_id < 0:
                negative_metadata += 1
            else:
                expected_labels[item.point_ids] = item.output_instance_id
        class_id = _class_id(scene, item.class_name)
        same_gt = np.flatnonzero(scene.gt_object_class_ids == class_id)
        all_iou = _candidate_iou(projection, item.point_ids)
        if len(same_gt):
            local = int(np.argmax(all_iou[same_gt]))
            best_object = int(same_gt[local])
            best_iou = float(all_iou[best_object])
            best_by_gt[same_gt] = np.maximum(best_by_gt[same_gt], all_iou[same_gt])
        else:
            best_object = None
            best_iou = 0.0
        unsupported = int(
            np.count_nonzero(scene.gaussian_to_gt_object_indices[item.point_ids] < 0)
        )
        unsupported_points += unsupported
        unsupported_candidates += int(unsupported == len(item.point_ids))
        candidate_points += len(item.point_ids)
        rows.append(
            {
                "scene_id": scene.scene_id,
                "mode": mode,
                "candidate_index": object_index,
                "source_fragment_ids": list(item.source_fragment_ids),
                "class_name": item.class_name,
                "class_id": class_id,
                "base_score": item.base_score,
                "point_count": len(item.point_ids),
                "best_same_class_object_index": best_object,
                "best_same_class_instance_id": (
                    int(scene.gt_object_instance_ids[best_object])
                    if best_object is not None
                    else None
                ),
                "best_same_class_size_bin": (
                    scene.gt_object_size_bins[best_object]
                    if best_object is not None
                    else None
                ),
                "best_same_class_iou": best_iou,
                "unsupported_point_count": unsupported,
                "unsupported_fraction": _fraction(unsupported, len(item.point_ids)),
            }
        )
    diagnostics = _field(result, "diagnostics", required=False, default={})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    orphan = int(diagnostics.get("orphan_count", 0))
    if point_labels is not None:
        orphan += int(np.count_nonzero(point_labels != expected_labels))
    lineage_violations += int(diagnostics.get("lineage_missing_fragment_count", 0))
    overlap_violations += int(diagnostics.get("overlap_ownership_violation_count", 0))
    negative_metadata += int(diagnostics.get("negative_metadata_count", 0))
    core_full = int(diagnostics.get("core_full_contract_violation_count", 0))
    determinism = int(
        diagnostics.get(
            "determinism_violation_count",
            0 if diagnostics.get("deterministic_algorithm", True) else 1,
        )
    )
    if min(orphan, negative_metadata, core_full, determinism) < 0:
        raise ValueError("merge diagnostics contain negative contract counts")
    tiny = np.asarray(
        [value in {"tiny", "small"} for value in scene.gt_object_size_bins],
        dtype=bool,
    )
    return FragmentMergeSceneMetrics(
        scene_id=scene.scene_id,
        mode=mode,
        candidate_count=len(objects),
        candidate_point_count=candidate_points,
        unsupported_point_count=unsupported_points,
        unsupported_candidate_count=unsupported_candidates,
        same_class_iou_025_count=int(
            sum(row["best_same_class_iou"] >= 0.25 for row in rows)
        ),
        same_class_iou_050_count=int(
            sum(row["best_same_class_iou"] >= 0.50 for row in rows)
        ),
        tiny_small_gt_count=int(np.count_nonzero(tiny)),
        tiny_small_iou_025_count=int(np.count_nonzero(tiny & (best_by_gt >= 0.25))),
        tiny_small_iou_050_count=int(np.count_nonzero(tiny & (best_by_gt >= 0.50))),
        best_iou_by_gt=tuple(map(float, best_by_gt)),
        lineage_violation_count=lineage_violations,
        overlap_ownership_violation_count=overlap_violations,
        orphan_count=orphan,
        negative_metadata_count=negative_metadata,
        core_full_contract_violation_count=core_full,
        determinism_violation_count=determinism,
        candidate_rows=tuple(rows),
    )


def _decision_round(row: Any) -> int:
    return int(_field(row, "round_index", "round", default=0, required=False))


def _is_merge_proposal(row: Any) -> bool:
    """Return whether a trace row is an actual mutual-best proposal.

    Early section-33 producers persisted only proposals.  The final producer
    persists every component edge and marks the proposal subset explicitly;
    accepting the absent field as ``True`` keeps both audit formats readable
    without treating non-mutual diagnostic edges as interventions.
    """

    value = _field(row, "mutual_best", required=False, default=True)
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("merge decision mutual_best must be boolean")
    return bool(value)


def _decision_pair(row: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = tuple(
        sorted(map(int, _field(row, "left_source_fragment_ids", "left_lineage")))
    )
    right = tuple(
        sorted(map(int, _field(row, "right_source_fragment_ids", "right_lineage")))
    )
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _decision_score(row: Any) -> float:
    value = float(_field(row, "union_prior_score", "prior_score", "union_P", "P"))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("merge decision prior score must be finite and in [0, 1]")
    return value


def _fragment_outcomes(
    graph: Any, result: Any
) -> tuple[dict[int, tuple[int, ...] | None], dict[int, str]]:
    nodes, _, point_count = _normalise_graph(graph)
    source_nodes = {row.source_fragment_id: row for row in nodes.values()}
    outcomes: dict[int, tuple[int, ...] | None] = {
        value: None for value in source_nodes
    }
    for item in _normalise_objects(result, point_count):
        if not item.accepted:
            continue
        for fragment_id in item.source_fragment_ids:
            if fragment_id not in outcomes:
                raise ValueError("merge result lineage references an unknown fragment")
            if outcomes[fragment_id] is not None:
                raise ValueError("accepted merge lineages overlap")
            outcomes[fragment_id] = item.source_fragment_ids
    return outcomes, {value: row.class_name for value, row in source_nodes.items()}


def evaluate_fragment_merge_mechanical_effect(
    graphs: Mapping[str, Any],
    uniform_results: Mapping[str, Any],
    class_results: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the registered first-round-score OR final-decision gate."""

    scene_ids = sorted(graphs)
    if set(uniform_results) != set(scene_ids) or set(class_results) != set(scene_ids):
        raise ValueError("graphs and paired merge results must share scene ids")
    proposal_count = changed_proposals = 0
    unmatched_proposals = 0
    changed_fragments: list[tuple[str, int, str]] = []
    per_scene: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        uniform_decisions = {
            _decision_pair(row): _decision_score(row)
            for row in _records(uniform_results[scene_id], "decisions")
            if _decision_round(row) == 0 and _is_merge_proposal(row)
        }
        class_decisions = {
            _decision_pair(row): _decision_score(row)
            for row in _records(class_results[scene_id], "decisions")
            if _decision_round(row) == 0 and _is_merge_proposal(row)
        }
        common = sorted(set(uniform_decisions).intersection(class_decisions))
        changed_here = sum(
            abs(class_decisions[key] - uniform_decisions[key])
            >= MECHANICAL_SCORE_DELTA_MIN
            for key in common
        )
        proposal_count += len(common)
        changed_proposals += changed_here
        unmatched = len(set(uniform_decisions).symmetric_difference(class_decisions))
        unmatched_proposals += unmatched
        uniform_outcome, classes = _fragment_outcomes(
            graphs[scene_id], uniform_results[scene_id]
        )
        class_outcome, _ = _fragment_outcomes(graphs[scene_id], class_results[scene_id])
        changed_ids = [
            fragment_id
            for fragment_id in sorted(uniform_outcome)
            if uniform_outcome[fragment_id] != class_outcome[fragment_id]
        ]
        changed_fragments.extend(
            (scene_id, fragment_id, classes[fragment_id]) for fragment_id in changed_ids
        )
        per_scene.append(
            {
                "scene_id": scene_id,
                "common_first_round_proposal_count": len(common),
                "score_changed_first_round_proposal_count": int(changed_here),
                "unmatched_first_round_proposal_count": unmatched,
                "final_fragment_decision_changed_count": len(changed_ids),
            }
        )
    fraction = _fraction(changed_proposals, proposal_count)
    changed_classes = sorted({row[2] for row in changed_fragments})
    changed_scenes = sorted({row[0] for row in changed_fragments})
    score_gate = fraction >= MECHANICAL_SCORE_FRACTION_MIN
    decision_gate = (
        len(changed_fragments) >= MECHANICAL_DECISION_MIN
        and len(changed_classes) >= MECHANICAL_CLASS_MIN
        and len(changed_scenes) >= MECHANICAL_SCENE_MIN
    )
    return {
        "common_first_round_proposal_count": proposal_count,
        "score_changed_first_round_proposal_count": int(changed_proposals),
        "score_changed_first_round_proposal_fraction": fraction,
        "unmatched_first_round_proposal_count": unmatched_proposals,
        "score_change_gate_passed": score_gate,
        "final_fragment_decision_changed_count": len(changed_fragments),
        "final_decision_changed_classes": changed_classes,
        "final_decision_changed_scenes": changed_scenes,
        "final_decision_gate_passed": decision_gate,
        "mechanically_effective": score_gate or decision_gate,
        "per_scene": per_scene,
    }


def _aggregate(rows: Sequence[FragmentMergeSceneMetrics]) -> dict[str, Any]:
    candidates = sum(row.candidate_count for row in rows)
    points = sum(row.candidate_point_count for row in rows)
    unsupported = sum(row.unsupported_point_count for row in rows)
    tiny_gt = sum(row.tiny_small_gt_count for row in rows)
    tiny_025 = sum(row.tiny_small_iou_025_count for row in rows)
    tiny_050 = sum(row.tiny_small_iou_050_count for row in rows)
    return {
        "scene_count": len(rows),
        "candidate_count": candidates,
        "candidate_point_count": points,
        "same_class_iou_025_count": sum(row.same_class_iou_025_count for row in rows),
        "same_class_iou_050_count": sum(row.same_class_iou_050_count for row in rows),
        "same_class_iou_050_scene_count": sum(
            row.same_class_iou_050_count > 0 for row in rows
        ),
        "candidate_precision_025": _fraction(
            sum(row.same_class_iou_025_count for row in rows), candidates
        ),
        "candidate_precision_050": _fraction(
            sum(row.same_class_iou_050_count for row in rows), candidates
        ),
        "unsupported_fraction": _fraction(unsupported, points),
        "tiny_small_gt_count": tiny_gt,
        "tiny_small_recall_025": _fraction(tiny_025, tiny_gt),
        "tiny_small_recall_050": _fraction(tiny_050, tiny_gt),
        "output_contract_violation_count": sum(
            row.output_contract_violation_count for row in rows
        ),
        "per_scene": [row.as_dict() for row in rows],
    }


def _aggregate_oracle(rows: Sequence[FragmentGraphOracleMetrics]) -> dict[str, Any]:
    tiny_gt = sum(row.tiny_small_gt_count for row in rows)
    tiny_hit = sum(row.tiny_small_iou_025_count for row in rows)
    return {
        "scene_count": len(rows),
        "same_class_iou_025_count": sum(row.same_class_iou_025_count for row in rows),
        "same_class_iou_050_count": sum(row.same_class_iou_050_count for row in rows),
        "tiny_small_gt_count": tiny_gt,
        "tiny_small_iou_025_count": tiny_hit,
        "tiny_small_recall_025": _fraction(tiny_hit, tiny_gt),
        "per_scene": [row.as_dict() for row in rows],
    }


def _validate_scene_sets(
    expected: Sequence[str],
    oracle: Sequence[FragmentGraphOracleMetrics],
    uniform: Sequence[FragmentMergeSceneMetrics],
    class_rows: Sequence[FragmentMergeSceneMetrics],
) -> None:
    target = set(expected)
    for label, rows in (
        ("oracle", oracle),
        ("uniform", uniform),
        ("class", class_rows),
    ):
        ids = [row.scene_id for row in rows]
        if set(ids) != target or len(ids) != len(target):
            raise ValueError(f"{label} rows must contain exactly the registered scenes")


def _graph_gate(aggregate: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "same_class_iou050_at_least_6": int(aggregate["same_class_iou_050_count"])
        >= GRAPH_IOU050_MIN,
        "tiny_small_recall025_at_least_0.20": float(aggregate["tiny_small_recall_025"])
        >= GRAPH_TINY_SMALL_RECALL025_MIN,
    }


def analyze_fragment_merge_dev2(
    *,
    oracle_rows: Sequence[FragmentGraphOracleMetrics],
    uniform_rows: Sequence[FragmentMergeSceneMetrics],
    class_rows: Sequence[FragmentMergeSceneMetrics],
    mechanical_effect: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply Stage-A graph, mechanism, paired quality, and safety gates."""

    _validate_scene_sets(DEV2_SCENE_IDS, oracle_rows, uniform_rows, class_rows)
    oracle = _aggregate_oracle(oracle_rows)
    uniform = _aggregate(uniform_rows)
    class_aggregate = _aggregate(class_rows)
    graph_checks = _graph_gate(oracle)
    graph_passed = all(graph_checks.values())
    by_u = {row.scene_id: row for row in uniform_rows}
    by_d = {row.scene_id: row for row in class_rows}
    improved: list[str] = []
    max_drop_by_scene: dict[str, float] = {}
    for scene_id in DEV2_SCENE_IDS:
        u, d = by_u[scene_id], by_d[scene_id]
        u_key = (
            u.same_class_iou_050_count,
            u.same_class_iou_025_count,
            u.candidate_precision_025,
            -u.unsupported_fraction,
        )
        d_key = (
            d.same_class_iou_050_count,
            d.same_class_iou_025_count,
            d.candidate_precision_025,
            -d.unsupported_fraction,
        )
        if d_key > u_key:
            improved.append(scene_id)
        max_drop_by_scene[scene_id] = max(
            (left - right for left, right in zip(u.best_iou_by_gt, d.best_iou_by_gt)),
            default=0.0,
        )
    safe_witness = next(
        (
            scene_id
            for scene_id in improved
            if max(
                (
                    drop
                    for other, drop in max_drop_by_scene.items()
                    if other != scene_id
                ),
                default=0.0,
            )
            <= DEV2_NONWITNESS_GT_IOU_DROP_MAX + _EPS
        ),
        None,
    )
    u_precision = float(uniform["candidate_precision_025"])
    d_precision = float(class_aggregate["candidate_precision_025"])
    relative_precision = (
        (d_precision - u_precision) / u_precision
        if u_precision > 0
        else (math.inf if d_precision > 0 else 0.0)
    )
    unsupported_drop = float(uniform["unsupported_fraction"]) - float(
        class_aggregate["unsupported_fraction"]
    )
    quality_checks = {
        "mechanically_effective": bool(
            mechanical_effect.get("mechanically_effective", False)
        ),
        "iou025_not_lower": int(class_aggregate["same_class_iou_025_count"])
        >= int(uniform["same_class_iou_025_count"]),
        "iou050_not_lower": int(class_aggregate["same_class_iou_050_count"])
        >= int(uniform["same_class_iou_050_count"]),
        "precision_gain_or_unsupported_drop": relative_precision + _EPS
        >= DEV2_PRECISION_RELATIVE_GAIN_MIN
        or unsupported_drop + _EPS >= DEV2_UNSUPPORTED_DROP_MIN,
        "tiny_small_recall025_not_lower": float(
            class_aggregate["tiny_small_recall_025"]
        )
        + _EPS
        >= float(uniform["tiny_small_recall_025"]),
        "candidate_count_at_most_1.25x": int(class_aggregate["candidate_count"])
        <= DEV2_CANDIDATE_RATIO_MAX * max(int(uniform["candidate_count"]), 1),
        "one_scene_improved_other_safe": safe_witness is not None,
        "class_iou050_at_least_4": int(class_aggregate["same_class_iou_050_count"])
        >= DEV2_D_IOU050_MIN,
        "class_precision025_at_least_0.10": d_precision + _EPS
        >= DEV2_D_PRECISION025_MIN,
        "output_contract_zero": int(class_aggregate["output_contract_violation_count"])
        == 0,
    }
    passed = graph_passed and all(quality_checks.values())
    if not graph_passed:
        conclusion = "graph-upper-bound-failed-category-prior-not-evaluable"
    elif not bool(mechanical_effect.get("mechanically_effective", False)):
        conclusion = "class-prior-not-mechanically-effective-in-merge-interface"
    elif passed:
        conclusion = "dev2-passed-proceed-to-dev8"
    else:
        conclusion = "class-prior-changed-merges-without-registered-dev2-benefit"
    return {
        "schema": SCHEMA,
        "phase": "dev2",
        "scene_ids": list(DEV2_SCENE_IDS),
        "graph_oracle": oracle,
        "uniform": uniform,
        "class": class_aggregate,
        "mechanical_effect": dict(mechanical_effect),
        "graph_checks": graph_checks,
        "graph_passed": graph_passed,
        "quality_checks": quality_checks,
        "relative_precision_gain_025": relative_precision,
        "unsupported_fraction_drop": unsupported_drop,
        "improved_scene_ids": improved,
        "maximum_gt_iou_drop_by_scene": max_drop_by_scene,
        "safe_witness_scene": safe_witness,
        "passed": passed,
        "conclusion": conclusion,
        "independent_experimental_unit": "physical_scene",
    }


def _scene_ap(row: FragmentMergeSceneMetrics, threshold: float) -> float:
    return binary_average_precision(
        [float(item["base_score"]) for item in row.candidate_rows],
        [
            float(item["best_same_class_iou"]) >= threshold
            for item in row.candidate_rows
        ],
    )


def _fp_tp_ratio(rows: Sequence[FragmentMergeSceneMetrics]) -> float:
    tp = sum(row.same_class_iou_025_count for row in rows)
    fp = sum(row.candidate_count - row.same_class_iou_025_count for row in rows)
    return fp / max(tp, 1)


def analyze_fragment_merge_dev8(
    *,
    oracle_rows: Sequence[FragmentGraphOracleMetrics],
    uniform_rows: Sequence[FragmentMergeSceneMetrics],
    class_rows: Sequence[FragmentMergeSceneMetrics],
    mechanical_effect: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the Stage-B absolute-health and scene-equal paired gates."""

    _validate_scene_sets(DEV8_SCENE_IDS, oracle_rows, uniform_rows, class_rows)
    oracle = _aggregate_oracle(oracle_rows)
    uniform = _aggregate(uniform_rows)
    class_aggregate = _aggregate(class_rows)
    by_u = {row.scene_id: row for row in uniform_rows}
    by_d = {row.scene_id: row for row in class_rows}
    per_scene: list[dict[str, Any]] = []
    for scene_id in DEV8_SCENE_IDS:
        u, d = by_u[scene_id], by_d[scene_id]
        u25, d25 = _scene_ap(u, 0.25), _scene_ap(d, 0.25)
        u50, d50 = _scene_ap(u, 0.50), _scene_ap(d, 0.50)
        per_scene.append(
            {
                "scene_id": scene_id,
                "uniform_candidate_ap_025": u25,
                "class_candidate_ap_025": d25,
                "delta_candidate_ap_025": d25 - u25,
                "uniform_candidate_ap_050": u50,
                "class_candidate_ap_050": d50,
                "delta_candidate_ap_050": d50 - u50,
            }
        )
    delta25 = float(np.mean([row["delta_candidate_ap_025"] for row in per_scene]))
    delta50 = float(np.mean([row["delta_candidate_ap_050"] for row in per_scene]))
    positive = [
        row["scene_id"] for row in per_scene if row["delta_candidate_ap_025"] > 0
    ]
    u_ratio, d_ratio = _fp_tp_ratio(uniform_rows), _fp_tp_ratio(class_rows)
    if u_ratio == 0.0:
        fp_tp_worsening = 0.0 if d_ratio == 0.0 else math.inf
    else:
        fp_tp_worsening = (d_ratio - u_ratio) / u_ratio
    graph_checks = _graph_gate(oracle)
    health_checks = {
        "class_iou050_at_least_12": int(class_aggregate["same_class_iou_050_count"])
        >= DEV8_D_IOU050_MIN,
        "class_iou050_at_least_4_scenes": int(
            class_aggregate["same_class_iou_050_scene_count"]
        )
        >= DEV8_D_IOU050_SCENE_MIN,
        "class_precision025_at_least_0.10": float(
            class_aggregate["candidate_precision_025"]
        )
        + _EPS
        >= DEV8_D_PRECISION025_MIN,
        "class_tiny_small_recall025_at_least_0.20": float(
            class_aggregate["tiny_small_recall_025"]
        )
        + _EPS
        >= DEV8_TINY_SMALL_RECALL025_MIN,
        "candidate_count_at_most_1.25x": int(class_aggregate["candidate_count"])
        <= DEV2_CANDIDATE_RATIO_MAX * max(int(uniform["candidate_count"]), 1),
        "output_contract_zero": int(class_aggregate["output_contract_violation_count"])
        == 0,
    }
    relative_checks = {
        "mechanically_effective": bool(
            mechanical_effect.get("mechanically_effective", False)
        ),
        "scene_equal_ap025_gain_at_least_0.002": delta25 + _EPS >= DEV8_AP025_GAIN_MIN,
        "scene_equal_ap050_drop_at_most_0.002": delta50 + _EPS >= -DEV8_AP050_DROP_MAX,
        "positive_scene_count_at_least_5": len(positive) >= DEV8_POSITIVE_SCENE_MIN,
        "tiny_small_recall025_not_lower": float(
            class_aggregate["tiny_small_recall_025"]
        )
        + _EPS
        >= float(uniform["tiny_small_recall_025"]),
        "fp_tp_worsening_at_most_20pct": fp_tp_worsening
        <= DEV8_FP_TP_WORSENING_MAX + _EPS,
    }
    passed = (
        all(graph_checks.values())
        and all(health_checks.values())
        and all(relative_checks.values())
    )
    return {
        "schema": SCHEMA,
        "phase": "dev8",
        "scene_ids": list(DEV8_SCENE_IDS),
        "graph_oracle": oracle,
        "uniform": uniform,
        "class": class_aggregate,
        "mechanical_effect": dict(mechanical_effect),
        "graph_checks": graph_checks,
        "absolute_health_checks": health_checks,
        "relative_benefit_checks": relative_checks,
        "scene_equal_candidate_ap_025_delta": delta25,
        "scene_equal_candidate_ap_050_delta": delta50,
        "positive_scene_ids": positive,
        "positive_scene_count": len(positive),
        "uniform_fp_tp_ratio": u_ratio,
        "class_fp_tp_ratio": d_ratio,
        "fp_tp_relative_worsening": fp_tp_worsening,
        "per_scene": per_scene,
        "passed": passed,
        "conclusion": (
            "dev8-passed-category-prior-helps-fragment-assembly"
            if passed
            else "dev8-registered-gates-failed"
        ),
        "independent_experimental_unit": "physical_scene",
        "candidate_rows_are_not_independent_replicates": True,
    }


def evaluate_category_fragment_merge(
    *,
    scenes: Mapping[str, ClusterEvaluationScene],
    graphs: Mapping[str, Any],
    uniform_results: Mapping[str, Any],
    class_results: Mapping[str, Any],
    phase: str,
) -> dict[str, Any]:
    """Evaluate materialised paired results without exposing GT to runtime code."""

    expected = (
        DEV2_SCENE_IDS
        if phase == "dev2"
        else DEV8_SCENE_IDS
        if phase == "dev8"
        else None
    )
    if expected is None:
        raise ValueError("phase must be 'dev2' or 'dev8'")
    if any(
        set(rows) != set(expected)
        for rows in (scenes, graphs, uniform_results, class_results)
    ):
        raise ValueError(f"{phase} inputs must contain exactly the registered scenes")
    oracle = tuple(
        evaluate_fragment_graph_oracle(scenes[key], graphs[key]) for key in expected
    )
    uniform = tuple(
        evaluate_fragment_merge_scene(scenes[key], graphs[key], uniform_results[key])
        for key in expected
    )
    class_rows = tuple(
        evaluate_fragment_merge_scene(scenes[key], graphs[key], class_results[key])
        for key in expected
    )
    mechanical = evaluate_fragment_merge_mechanical_effect(
        graphs, uniform_results, class_results
    )
    if phase == "dev2":
        return analyze_fragment_merge_dev2(
            oracle_rows=oracle,
            uniform_rows=uniform,
            class_rows=class_rows,
            mechanical_effect=mechanical,
        )
    return analyze_fragment_merge_dev8(
        oracle_rows=oracle,
        uniform_rows=uniform,
        class_rows=class_rows,
        mechanical_effect=mechanical,
    )


__all__ = [
    "DEV2_SCENE_IDS",
    "DEV8_SCENE_IDS",
    "FragmentGraphOracleMetrics",
    "FragmentMergeSceneMetrics",
    "analyze_fragment_merge_dev2",
    "analyze_fragment_merge_dev8",
    "evaluate_category_fragment_merge",
    "evaluate_fragment_graph_oracle",
    "evaluate_fragment_merge_mechanical_effect",
    "evaluate_fragment_merge_scene",
]
