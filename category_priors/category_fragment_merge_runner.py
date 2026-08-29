from __future__ import annotations

"""Ground-truth-free I/O runner for the section-33 fragment experiment.

The worker reconstructs only the registered ``native-2k-grounded`` /
``predicted-32-top1`` raw HDBSCAN arm, freezes one :class:`FragmentGraph`, and
replays both global and class-shrunk priors from that exact graph.  Evaluation
and ScanNet ground truth deliberately live in a separate module.
"""

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise import CandidateBank, load_candidate_bank, save_candidate_bank
from .category_feature_routing_factorial import (
    SAMPLE_CAP,
    SEED,
    build_raw_cluster_bank,
    load_feature_ply,
    predicted_route,
)
from .category_fragment_merge import (
    FRAGMENT_GRAPH_SCHEMA,
    FRAGMENT_MERGE_SCHEMA,
    FragmentEdge,
    FragmentGraph,
    FragmentMergeDecision,
    FragmentMergeResult,
    FragmentNode,
    FragmentObject,
    build_fragment_graph,
    merge_category_fragments,
)
from .evaluator import apply_transform, load_ply_xyz
from .io import load_json, write_json
from .prompt_prior import materialize_prompt_priors
from .runner import load_scene_runtime_manifest

SCHEMA = "saga-category-fragment-merge-runner-v1"
FEATURE_SOURCE = "native-2k-grounded"
SEMANTIC_ROUTE = "predicted-32-top1"
MODES = ("global", "class")
POINT_AXIS_ATOL = 1e-5


@dataclass(frozen=True)
class FragmentScenePaths:
    scene_root: Path
    raw_bank: Path
    graph_root: Path
    global_result: Path
    class_result: Path
    completion: Path

    @classmethod
    def at(cls, output_root: str | Path, scene_id: str) -> FragmentScenePaths:
        root = Path(output_root) / str(scene_id)
        return cls(
            scene_root=root,
            raw_bank=root / "raw_bank",
            graph_root=root / "fragment_graph",
            global_result=root / "replay" / "global",
            class_result=root / "replay" / "class",
            completion=root / "scene_complete.json",
        )

    def result_root(self, mode: str) -> Path:
        if mode == "global":
            return self.global_result
        if mode == "class":
            return self.class_result
        raise ValueError("mode must be 'global' or 'class'")


@dataclass(frozen=True)
class FragmentSceneArtifacts:
    raw_bank: CandidateBank
    graph: FragmentGraph
    xyz_scene: np.ndarray
    uniform: FragmentMergeResult
    class_shrunk: FragmentMergeResult
    metadata: Mapping[str, Any]


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Atomically replace one NPZ without creating a lock or cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def _pack_ragged(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    indptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    values = (
        np.concatenate([np.asarray(row, dtype=np.int64) for row in rows])
        if int(indptr[-1])
        else np.empty(0, dtype=np.int64)
    )
    return indptr, values


def _unpack_ragged(
    indptr: np.ndarray, values: np.ndarray, *, expected_rows: int, name: str
) -> tuple[np.ndarray, ...]:
    pointers = np.asarray(indptr, dtype=np.int64)
    data = np.asarray(values, dtype=np.int64)
    if pointers.shape != (expected_rows + 1,):
        raise ValueError(f"{name} indptr has the wrong length")
    if pointers[0] != 0 or pointers[-1] != len(data) or np.any(np.diff(pointers) < 0):
        raise ValueError(f"{name} ragged encoding is invalid")
    return tuple(
        data[pointers[index] : pointers[index + 1]].copy()
        for index in range(expected_rows)
    )


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _scene_asset(scene: Mapping[str, Any], keys: Sequence[str], fallback: str) -> Path:
    base = Path(str(scene["base_path"])).resolve()
    for key in keys:
        value = scene.get(key)
        if value:
            path = Path(str(value))
            return (path if path.is_absolute() else base / path).resolve()
    return (base / fallback).resolve()


def _gaussian_axis_asset(
    scene: Mapping[str, Any],
) -> tuple[Path, tuple[tuple[float, ...], ...]]:
    """Resolve the evaluation Gaussian PLY and transform without opening GT.

    The two private helpers are the canonical path/axis contract already used
    by the GT-aware evaluator.  Importing them does not read a ground-truth
    file; it only prevents this runner from inventing a second path fallback.
    """

    from .v9_metrics import _gaussian_ply, _transform

    path = Path(_gaussian_ply(scene)).resolve()
    transform = np.asarray(_transform(scene), dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("gaussian_to_gt_transform must be a finite 4x4 matrix")
    return path, tuple(tuple(float(value) for value in row) for row in transform)


def _validate_feature_gaussian_axis(
    feature_xyz: Any,
    gaussian_xyz: Any,
    transform: Sequence[Sequence[float]],
    *,
    scene_id: str,
) -> dict[str, Any]:
    """Prove that feature point IDs address the evaluator's Gaussian axis."""

    feature = np.asarray(feature_xyz, dtype=np.float64)
    gaussian = np.asarray(gaussian_xyz, dtype=np.float64)
    if feature.ndim != 2 or feature.shape[1:] != (3,):
        raise ValueError(f"{scene_id}: feature XYZ must have shape (N, 3)")
    if gaussian.ndim != 2 or gaussian.shape[1:] != (3,):
        raise ValueError(f"{scene_id}: Gaussian XYZ must have shape (N, 3)")
    if len(feature) != len(gaussian):
        raise ValueError(
            f"{scene_id}: feature/Gaussian point-count mismatch "
            f"({len(feature)} != {len(gaussian)})"
        )
    if not np.isfinite(feature).all() or not np.isfinite(gaussian).all():
        raise ValueError(f"{scene_id}: feature/Gaussian XYZ must be finite")
    feature_eval = apply_transform(feature, transform)
    gaussian_eval = apply_transform(gaussian, transform)
    if not np.isfinite(feature_eval).all() or not np.isfinite(gaussian_eval).all():
        raise ValueError(f"{scene_id}: transformed feature/Gaussian XYZ is non-finite")
    absolute = np.abs(feature_eval - gaussian_eval)
    maximum = float(absolute.max()) if absolute.size else 0.0
    if maximum > POINT_AXIS_ATOL:
        raise ValueError(
            f"{scene_id}: feature/Gaussian point axis differs after the registered "
            f"transform (max_abs_error={maximum:.9g}, atol={POINT_AXIS_ATOL:g})"
        )
    return {
        "passed": True,
        "feature_point_count": len(feature),
        "gaussian_point_count": len(gaussian),
        "max_abs_error_after_transform": maximum,
        "atol": POINT_AXIS_ATOL,
        "gaussian_transform_applied_to_both_axes": True,
        "gt_used": False,
    }


def _scene_identity(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    feature_path: Path,
    gaussian_path: Path,
    gaussian_transform: Sequence[Sequence[float]],
    label_path: Path,
    category_priors_path: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scene_id": str(scene_id),
        "feature_source": FEATURE_SOURCE,
        "semantic_route": SEMANTIC_ROUTE,
        "seed": int(seed),
        "sample_cap": SAMPLE_CAP,
        "scene_scale_m_per_unit": float(scene["scene_scale_m_per_unit"]),
        "coordinate_contract": (
            "feature_xyz_multiplied_by_scene_scale_before_raw_clustering; "
            "persisted raw bank and graph therefore use metric XYZ with scale=1"
        ),
        "feature_ply": _file_record(feature_path),
        "gaussian_ply": _file_record(gaussian_path),
        "gaussian_to_gt_transform": [
            [float(value) for value in row] for row in gaussian_transform
        ],
        "label_features": _file_record(label_path),
        "category_priors": _file_record(category_priors_path),
    }


def _load_label_features(path: Path) -> np.ndarray:
    import torch

    value = torch.load(path, map_location="cpu")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def save_fragment_graph(
    graph: FragmentGraph,
    xyz_scene: Any,
    root: str | Path,
    *,
    input_identity: Mapping[str, Any],
) -> None:
    """Persist a replayable graph and its aligned XYZ axis."""

    destination = Path(root)
    xyz = np.asarray(xyz_scene, dtype=np.float64)
    if xyz.shape != (graph.point_count, 3) or not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite and match the graph point axis")
    indptr, point_ids = _pack_ragged([node.point_ids for node in graph.nodes])
    _atomic_savez(
        destination / "fragment_graph.npz",
        xyz_scene=xyz,
        node_point_indptr=indptr,
        node_point_ids=point_ids,
    )
    write_json(
        destination / "fragment_graph.json",
        {
            "schema": graph.schema,
            "input_identity": dict(input_identity),
            "point_count": graph.point_count,
            "scene_scale_m_per_unit": graph.scene_scale_m_per_unit,
            "global_typical_diag_m": graph.global_typical_diag_m,
            "nodes": [
                {
                    "fragment_id": node.fragment_id,
                    "source_fragment_id": node.source_fragment_id,
                    "class_index": node.class_index,
                    "class_name": node.class_name,
                    "membership_mean": node.membership_mean,
                    "semantic_score_mean": node.semantic_score_mean,
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "left_fragment_id": edge.left_fragment_id,
                    "right_fragment_id": edge.right_fragment_id,
                    "cross_edge_count": edge.cross_edge_count,
                    "affinity_cosine_median": edge.affinity_cosine_median,
                    "min_distance_m": edge.min_distance_m,
                }
                for edge in graph.edges
            ],
            "diagnostics": dict(graph.diagnostics),
        },
    )


def load_fragment_graph(
    root: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[FragmentGraph, np.ndarray, dict[str, Any]]:
    source = Path(root)
    metadata = load_json(source / "fragment_graph.json")
    if metadata.get("schema") != FRAGMENT_GRAPH_SCHEMA:
        raise ValueError("fragment graph metadata has the wrong schema")
    if expected_identity is not None and metadata.get("input_identity") != dict(
        expected_identity
    ):
        raise ValueError("fragment graph input identity differs")
    nodes_meta = metadata.get("nodes")
    edges_meta = metadata.get("edges")
    if not isinstance(nodes_meta, list) or not isinstance(edges_meta, list):
        raise TypeError("fragment graph metadata lacks nodes or edges")
    with np.load(source / "fragment_graph.npz", allow_pickle=False) as archive:
        required = {"xyz_scene", "node_point_indptr", "node_point_ids"}
        if not required.issubset(archive.files):
            raise ValueError("fragment graph NPZ is incomplete")
        xyz = np.asarray(archive["xyz_scene"], dtype=np.float64).copy()
        rows = _unpack_ragged(
            archive["node_point_indptr"],
            archive["node_point_ids"],
            expected_rows=len(nodes_meta),
            name="node points",
        )
    nodes = tuple(
        FragmentNode(point_ids=points, **dict(row))
        for row, points in zip(nodes_meta, rows)
    )
    edges = tuple(FragmentEdge(**dict(row)) for row in edges_meta)
    graph = FragmentGraph(
        nodes=nodes,
        edges=edges,
        point_count=int(metadata["point_count"]),
        scene_scale_m_per_unit=float(metadata["scene_scale_m_per_unit"]),
        global_typical_diag_m=float(metadata["global_typical_diag_m"]),
        diagnostics=dict(metadata.get("diagnostics", {})),
        schema=str(metadata["schema"]),
    )
    if xyz.shape != (graph.point_count, 3) or not np.isfinite(xyz).all():
        raise ValueError("persisted graph XYZ is invalid")
    xyz.setflags(write=False)
    return graph, xyz, dict(metadata)


def _decision_json(row: FragmentMergeDecision) -> dict[str, Any]:
    return {
        "round_index": row.round_index,
        "left_source_fragment_ids": list(row.left_source_fragment_ids),
        "right_source_fragment_ids": list(row.right_source_fragment_ids),
        "union_source_fragment_ids": list(row.union_source_fragment_ids),
        "left_prior_score": row.left_prior_score,
        "right_prior_score": row.right_prior_score,
        "union_prior_score": row.union_prior_score,
        "prior_eligible": row.prior_eligible,
        "mutual_best": row.mutual_best,
        "accepted": row.accepted,
        "reason": row.reason,
        "cross_edge_count": row.cross_edge_count,
        "affinity_cosine_median": row.affinity_cosine_median,
    }


def _graph_replay_identity(graph: FragmentGraph) -> dict[str, Any]:
    """Exact, unhashed graph values that can change replay decisions."""

    return {
        "nodes": [
            [
                node.fragment_id,
                node.source_fragment_id,
                node.class_index,
                node.class_name,
                node.membership_mean,
                node.semantic_score_mean,
                len(node.point_ids),
            ]
            for node in graph.nodes
        ],
        "edges": [
            [
                edge.left_fragment_id,
                edge.right_fragment_id,
                edge.cross_edge_count,
                edge.affinity_cosine_median,
                edge.min_distance_m,
            ]
            for edge in graph.edges
        ],
    }


def save_fragment_merge_result(
    result: FragmentMergeResult,
    root: str | Path,
    *,
    input_identity: Mapping[str, Any],
) -> None:
    destination = Path(root)
    point_indptr, point_ids = _pack_ragged([row.point_ids for row in result.objects])
    _atomic_savez(
        destination / "merge_result.npz",
        point_labels=result.point_labels,
        object_point_indptr=point_indptr,
        object_point_ids=point_ids,
    )
    write_json(
        destination / "merge_result.json",
        {
            "schema": result.schema,
            "mode": result.mode,
            "input_identity": dict(input_identity),
            "graph_schema": result.graph.schema,
            "graph_point_count": result.graph.point_count,
            "graph_replay_identity": _graph_replay_identity(result.graph),
            "objects": [
                {
                    "source_fragment_ids": list(row.source_fragment_ids),
                    "class_index": row.class_index,
                    "class_name": row.class_name,
                    "metric_extents_m": list(row.metric_extents_m),
                    "n_raw": row.n_raw,
                    "G": row.G,
                    "C": row.C,
                    "P": row.P,
                    "support_threshold": row.support_threshold,
                    "base_score": row.base_score,
                    "accepted": row.accepted,
                    "output_instance_id": row.output_instance_id,
                }
                for row in result.objects
            ],
            "decisions": [_decision_json(row) for row in result.decisions],
            "diagnostics": dict(result.diagnostics),
        },
    )


def _validate_result_against_graph(result: FragmentMergeResult) -> None:
    nodes = {node.source_fragment_id: node for node in result.graph.nodes}
    seen: set[int] = set()
    reconstructed = np.full(result.graph.point_count, -1, dtype=np.int64)
    expected_output = 0
    for row in result.objects:
        lineage = set(row.source_fragment_ids)
        if not lineage or seen.intersection(lineage) or not lineage.issubset(nodes):
            raise ValueError("merge object lineage is incomplete or overlapping")
        seen.update(lineage)
        points = np.sort(
            np.concatenate([nodes[item].point_ids for item in sorted(lineage)])
        )
        if not np.array_equal(points, row.point_ids):
            raise ValueError("merge object points differ from fragment lineage")
        classes = {nodes[item].class_index for item in lineage}
        if classes != {row.class_index}:
            raise ValueError("merge object crosses predicted classes")
        if row.accepted:
            if row.output_instance_id != expected_output:
                raise ValueError("accepted object IDs must be contiguous")
            reconstructed[row.point_ids] = expected_output
            expected_output += 1
        elif row.output_instance_id is not None:
            raise ValueError("rejected object cannot declare an output ID")
    if seen != set(nodes):
        raise ValueError("merge output does not preserve every fragment lineage")
    if not np.array_equal(reconstructed, result.point_labels):
        raise ValueError("merge point_labels disagree with accepted objects")


def load_fragment_merge_result(
    root: str | Path,
    graph: FragmentGraph,
    *,
    expected_mode: str,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[FragmentMergeResult, dict[str, Any]]:
    source = Path(root)
    metadata = load_json(source / "merge_result.json")
    if metadata.get("schema") != FRAGMENT_MERGE_SCHEMA:
        raise ValueError("fragment merge metadata has the wrong schema")
    if metadata.get("mode") != expected_mode:
        raise ValueError("fragment merge mode differs")
    if expected_identity is not None and metadata.get("input_identity") != dict(
        expected_identity
    ):
        raise ValueError("fragment merge input identity differs")
    if (
        metadata.get("graph_schema") != graph.schema
        or int(metadata.get("graph_point_count", -1)) != graph.point_count
    ):
        raise ValueError("fragment merge references a different graph")
    if metadata.get("graph_replay_identity") != _graph_replay_identity(graph):
        raise ValueError("fragment merge references different graph values")
    objects_meta = metadata.get("objects")
    decisions_meta = metadata.get("decisions")
    if not isinstance(objects_meta, list) or not isinstance(decisions_meta, list):
        raise TypeError("fragment merge metadata lacks objects or decisions")
    with np.load(source / "merge_result.npz", allow_pickle=False) as archive:
        required = {"point_labels", "object_point_indptr", "object_point_ids"}
        if not required.issubset(archive.files):
            raise ValueError("fragment merge NPZ is incomplete")
        labels = np.asarray(archive["point_labels"], dtype=np.int64).copy()
        points = _unpack_ragged(
            archive["object_point_indptr"],
            archive["object_point_ids"],
            expected_rows=len(objects_meta),
            name="object points",
        )
    objects = tuple(
        FragmentObject(
            point_ids=point_ids,
            source_fragment_ids=tuple(row["source_fragment_ids"]),
            class_index=int(row["class_index"]),
            class_name=str(row["class_name"]),
            metric_extents_m=tuple(row["metric_extents_m"]),
            n_raw=int(row["n_raw"]),
            G=float(row["G"]),
            C=float(row["C"]),
            P=float(row["P"]),
            support_threshold=int(row["support_threshold"]),
            base_score=float(row["base_score"]),
            accepted=bool(row["accepted"]),
            output_instance_id=(
                None
                if row.get("output_instance_id") is None
                else int(row["output_instance_id"])
            ),
        )
        for row, point_ids in zip(objects_meta, points)
    )
    decisions = tuple(
        FragmentMergeDecision(
            round_index=int(row["round_index"]),
            left_source_fragment_ids=tuple(row["left_source_fragment_ids"]),
            right_source_fragment_ids=tuple(row["right_source_fragment_ids"]),
            union_source_fragment_ids=tuple(row["union_source_fragment_ids"]),
            left_prior_score=float(row["left_prior_score"]),
            right_prior_score=float(row["right_prior_score"]),
            union_prior_score=float(row["union_prior_score"]),
            prior_eligible=bool(row["prior_eligible"]),
            mutual_best=bool(row["mutual_best"]),
            accepted=bool(row["accepted"]),
            reason=str(row["reason"]),
            cross_edge_count=int(row["cross_edge_count"]),
            affinity_cosine_median=float(row["affinity_cosine_median"]),
        )
        for row in decisions_meta
    )
    result = FragmentMergeResult(
        mode=str(metadata["mode"]),
        graph=graph,
        objects=objects,
        point_labels=labels,
        decisions=decisions,
        diagnostics=dict(metadata.get("diagnostics", {})),
        schema=str(metadata["schema"]),
    )
    _validate_result_against_graph(result)
    return result, dict(metadata)


def _validate_graph_against_raw_bank(graph: FragmentGraph, bank: CandidateBank) -> None:
    if graph.point_count != bank.point_count:
        raise ValueError("raw bank and fragment graph point counts differ")
    expected_ids = {
        int(value) for value in np.unique(bank.branch_core_labels) if value >= 0
    }
    observed_ids = {node.fragment_id for node in graph.nodes}
    if expected_ids != observed_ids:
        raise ValueError("fragment graph does not preserve every raw bank fragment")
    rows = {int(row["candidate_id"]): row for row in bank.candidates}
    for node in graph.nodes:
        points = np.flatnonzero(bank.branch_core_labels == node.fragment_id)
        if not np.array_equal(points, node.point_ids):
            raise ValueError("fragment graph membership differs from raw bank")
        row = rows.get(node.fragment_id)
        if (
            row is None
            or int(row.get("stable_source_id", -1)) != node.source_fragment_id
        ):
            raise ValueError(
                "fragment graph stable source identity differs from raw bank"
            )
        classes = np.unique(bank.semantic_top1[points])
        if (
            classes.tolist() != [node.class_index]
            or bank.class_names[node.class_index] != node.class_name
        ):
            raise ValueError("fragment graph class identity differs from raw bank")
        if not np.isclose(
            node.membership_mean,
            float(np.mean(bank.assignment_confidence[points])),
            rtol=0.0,
            atol=1e-12,
        ) or not np.isclose(
            node.semantic_score_mean,
            float(np.mean(bank.semantic_top1_score[points])),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("fragment graph evidence differs from raw bank")


def _load_raw_bank(root: Path, identity: Mapping[str, Any]) -> CandidateBank:
    bank = load_candidate_bank(root)
    diagnostics = bank.diagnostics
    if diagnostics.get("fragment_merge_runner_identity") != dict(identity):
        raise ValueError("raw bank input identity differs")
    if diagnostics.get("feature_source") != FEATURE_SOURCE:
        raise ValueError("raw bank is not the native-2k feature arm")
    if diagnostics.get("semantic_route") != SEMANTIC_ROUTE:
        raise ValueError("raw bank is not the predicted-32-top1 route")
    if diagnostics.get("gt_used") is not False:
        raise ValueError("raw bank does not prove a GT-free worker path")
    axis = diagnostics.get("point_axis_validation")
    if (
        not isinstance(axis, Mapping)
        or axis.get("passed") is not True
        or axis.get("gt_used") is not False
        or int(axis.get("feature_point_count", -1)) != len(bank.global_pre_knn)
        or int(axis.get("gaussian_point_count", -1)) != len(bank.global_pre_knn)
        or float(axis.get("atol", float("nan"))) != POINT_AXIS_ATOL
    ):
        raise ValueError("raw bank lacks a valid feature/Gaussian point-axis proof")
    if not np.array_equal(bank.branch_full_labels, bank.branch_core_labels):
        raise ValueError("raw bank contains a forbidden full-assignment stage")
    return bank


def load_category_fragment_scene(scene_root: str | Path) -> FragmentSceneArtifacts:
    """Load one complete scene for the independent GT-aware evaluator."""

    root = Path(scene_root)
    completion = load_json(root / "scene_complete.json")
    if completion.get("schema") != SCHEMA or completion.get("status") != "complete":
        raise ValueError("scene completion marker is missing or invalid")
    identity = completion.get("input_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("scene completion marker lacks input identity")
    bank = _load_raw_bank(root / "raw_bank", identity)
    graph, xyz, _ = load_fragment_graph(
        root / "fragment_graph", expected_identity=identity
    )
    _validate_graph_against_raw_bank(graph, bank)
    uniform, _ = load_fragment_merge_result(
        root / "replay" / "global",
        graph,
        expected_mode="global",
        expected_identity=identity,
    )
    class_shrunk, _ = load_fragment_merge_result(
        root / "replay" / "class",
        graph,
        expected_mode="class",
        expected_identity=identity,
    )
    return FragmentSceneArtifacts(bank, graph, xyz, uniform, class_shrunk, completion)


def _try_load(call: Callable[[], Any]) -> Any | None:
    try:
        return call()
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _bind_raw_identity(
    bank: CandidateBank,
    identity: Mapping[str, Any],
    point_axis_validation: Mapping[str, Any],
) -> CandidateBank:
    if not np.isclose(bank.scene_scale_m_per_unit, 1.0, rtol=0.0, atol=0.0):
        raise ValueError("raw fragment bank must persist metric XYZ with scene scale 1")
    rows = tuple(
        {**dict(row), "stable_source_id": int(row["candidate_id"])}
        for row in bank.candidates
    )
    diagnostics = {
        **dict(bank.diagnostics),
        "fragment_merge_runner_identity": dict(identity),
        "gt_used": False,
        "feature_source": FEATURE_SOURCE,
        "semantic_route": SEMANTIC_ROUTE,
        "coordinate_contract": ("xyz_m_input; candidate_bank_scene_scale_m_per_unit=1"),
        "point_axis_validation": dict(point_axis_validation),
    }
    return replace(bank, candidates=rows, diagnostics=diagnostics)


def _ensure_scene_graph(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    category_priors_payload: Mapping[str, Any],
    category_priors_path: str | Path,
    output_root: str | Path,
    seed: int,
    feature_loader: Callable[[str | Path], tuple[np.ndarray, np.ndarray, np.ndarray]],
    gaussian_loader: Callable[[str | Path], np.ndarray],
    label_loader: Callable[[Path], np.ndarray],
    raw_bank_builder: Callable[..., CandidateBank],
    graph_builder: Callable[..., FragmentGraph],
) -> tuple[
    FragmentScenePaths,
    dict[str, Any],
    CandidateBank,
    FragmentGraph,
    np.ndarray,
    list[str],
]:
    """Ensure only the raw bank and graph stages for one GT-free scene."""

    if int(seed) != SEED:
        raise ValueError(f"fragment experiment requires frozen seed {SEED}")
    priors_path = Path(category_priors_path).resolve()
    feature_path = _scene_asset(
        scene,
        ("contrastive_feature_point_cloud_path", "feature_ply"),
        "saga/contrastive_feature_point_cloud.ply",
    )
    gaussian_path, gaussian_transform = _gaussian_axis_asset(scene)
    label_path = _scene_asset(
        scene,
        ("grounded_label_features_path", "label_features_path"),
        "saga/labels/label_features.pt",
    )
    identity = _scene_identity(
        scene_id=scene_id,
        scene=scene,
        feature_path=feature_path,
        gaussian_path=gaussian_path,
        gaussian_transform=gaussian_transform,
        label_path=label_path,
        category_priors_path=priors_path,
        seed=seed,
    )
    paths = FragmentScenePaths.at(output_root, scene_id)
    rebuilt: list[str] = []
    bank = _try_load(lambda: _load_raw_bank(paths.raw_bank, identity))
    raw_rebuilt = bank is None
    xyz_scene: np.ndarray | None = None
    affinity: np.ndarray | None = None
    if bank is None:
        xyz, affinity, semantic = feature_loader(feature_path)
        point_axis_validation = _validate_feature_gaussian_axis(
            xyz,
            gaussian_loader(gaussian_path),
            gaussian_transform,
            scene_id=scene_id,
        )
        label_features = label_loader(label_path)
        top_class, route_score, branch_class = predicted_route(semantic, label_features)
        xyz_scene = np.asarray(xyz, dtype=np.float64) * float(
            scene["scene_scale_m_per_unit"]
        )
        global_diagonal = materialize_prompt_priors(
            category_priors_payload
        ).global_typical_diag_m
        bank = raw_bank_builder(
            affinity=affinity,
            xyz_m=xyz_scene,
            top_class=top_class,
            route_score=route_score,
            branch_class=branch_class,
            global_typical_diag_m=global_diagonal,
            scene_id=scene_id,
            feature_source=FEATURE_SOURCE,
            route=SEMANTIC_ROUTE,
            seed=seed,
            sample_cap=SAMPLE_CAP,
        )
        bank = _bind_raw_identity(bank, identity, point_axis_validation)
        save_candidate_bank(bank, paths.raw_bank)
        rebuilt.append("raw_bank")

    loaded_graph = (
        None
        if raw_rebuilt
        else _try_load(
            lambda: load_fragment_graph(paths.graph_root, expected_identity=identity)
        )
    )
    if loaded_graph is None:
        if xyz_scene is None or affinity is None:
            xyz, affinity, _ = feature_loader(feature_path)
            xyz_scene = np.asarray(xyz, dtype=np.float64) * float(
                scene["scene_scale_m_per_unit"]
            )
        graph = graph_builder(
            bank,
            xyz_scene,
            affinity,
            materialize_prompt_priors(category_priors_payload).global_typical_diag_m,
        )
        _validate_graph_against_raw_bank(graph, bank)
        save_fragment_graph(graph, xyz_scene, paths.graph_root, input_identity=identity)
        rebuilt.append("fragment_graph")
    else:
        graph, xyz_scene, _ = loaded_graph
        _validate_graph_against_raw_bank(graph, bank)
    if rebuilt:
        # A previous completion marker no longer describes the current graph.
        paths.completion.unlink(missing_ok=True)
    return paths, identity, bank, graph, xyz_scene, rebuilt


def _load_existing_scene_graph(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    category_priors_path: str | Path,
    output_root: str | Path,
    seed: int,
) -> tuple[
    FragmentScenePaths, dict[str, Any], CandidateBank, FragmentGraph, np.ndarray
]:
    priors_path = Path(category_priors_path).resolve()
    feature_path = _scene_asset(
        scene,
        ("contrastive_feature_point_cloud_path", "feature_ply"),
        "saga/contrastive_feature_point_cloud.ply",
    )
    gaussian_path, gaussian_transform = _gaussian_axis_asset(scene)
    label_path = _scene_asset(
        scene,
        ("grounded_label_features_path", "label_features_path"),
        "saga/labels/label_features.pt",
    )
    identity = _scene_identity(
        scene_id=scene_id,
        scene=scene,
        feature_path=feature_path,
        gaussian_path=gaussian_path,
        gaussian_transform=gaussian_transform,
        label_path=label_path,
        category_priors_path=priors_path,
        seed=seed,
    )
    paths = FragmentScenePaths.at(output_root, scene_id)
    bank = _load_raw_bank(paths.raw_bank, identity)
    graph, xyz_scene, _ = load_fragment_graph(
        paths.graph_root, expected_identity=identity
    )
    _validate_graph_against_raw_bank(graph, bank)
    return paths, identity, bank, graph, xyz_scene


def _replay_scene_modes(
    *,
    paths: FragmentScenePaths,
    identity: Mapping[str, Any],
    graph: FragmentGraph,
    xyz_scene: np.ndarray,
    category_priors_payload: Mapping[str, Any],
    modes: Sequence[str],
    merge_builder: Callable[..., FragmentMergeResult],
    force_rebuild: bool = False,
) -> tuple[dict[str, FragmentMergeResult], list[str]]:
    requested = tuple(map(str, modes))
    if (
        not requested
        or len(requested) != len(set(requested))
        or not set(requested).issubset(MODES)
    ):
        raise ValueError(
            "modes must be a non-empty unique subset of ('global', 'class')"
        )
    results: dict[str, FragmentMergeResult] = {}
    rebuilt: list[str] = []
    for mode in requested:
        loaded = (
            None
            if force_rebuild
            else _try_load(
                lambda mode=mode: load_fragment_merge_result(
                    paths.result_root(mode),
                    graph,
                    expected_mode=mode,
                    expected_identity=identity,
                )[0]
            )
        )
        if loaded is None:
            loaded = merge_builder(graph, xyz_scene, category_priors_payload, mode)
            if loaded.graph is not graph or loaded.graph.identity() != graph.identity():
                raise ValueError(
                    "U/D replay did not preserve the shared fragment graph"
                )
            _validate_result_against_graph(loaded)
            save_fragment_merge_result(
                loaded, paths.result_root(mode), input_identity=identity
            )
            rebuilt.append(f"replay-{mode}")
        results[mode] = loaded
    if rebuilt:
        paths.completion.unlink(missing_ok=True)
    return results, rebuilt


def _write_completion_if_both_modes_exist(
    *,
    scene_id: str,
    paths: FragmentScenePaths,
    identity: Mapping[str, Any],
    graph: FragmentGraph,
    rebuilt: Sequence[str],
) -> dict[str, Any] | None:
    results: dict[str, FragmentMergeResult] = {}
    for mode in MODES:
        loaded = _try_load(
            lambda mode=mode: load_fragment_merge_result(
                paths.result_root(mode),
                graph,
                expected_mode=mode,
                expected_identity=identity,
            )[0]
        )
        if loaded is None:
            return None
        results[mode] = loaded
    completion = {
        "schema": SCHEMA,
        "status": "complete",
        "scene_id": scene_id,
        "input_identity": dict(identity),
        "feature_source": FEATURE_SOURCE,
        "semantic_route": SEMANTIC_ROUTE,
        "raw_fragment_count": len(graph.nodes),
        "fragment_edge_count": len(graph.edges),
        "global_object_count": len(results["global"].objects),
        "class_object_count": len(results["class"].objects),
        "rebuilt": list(rebuilt),
        "gt_used": False,
    }
    write_json(paths.completion, completion)
    return completion


def run_category_fragment_scene(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    category_priors_payload: Mapping[str, Any],
    category_priors_path: str | Path,
    output_root: str | Path,
    seed: int = SEED,
    feature_loader: Callable[
        [str | Path], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = load_feature_ply,
    gaussian_loader: Callable[[str | Path], np.ndarray] = load_ply_xyz,
    label_loader: Callable[[Path], np.ndarray] = _load_label_features,
    raw_bank_builder: Callable[..., CandidateBank] = build_raw_cluster_bank,
    graph_builder: Callable[..., FragmentGraph] = build_fragment_graph,
    merge_builder: Callable[..., FragmentMergeResult] = merge_category_fragments,
) -> dict[str, Any]:
    """Build/replay one scene without accepting or loading any GT input."""

    paths, identity, _, graph, xyz_scene, graph_rebuilt = _ensure_scene_graph(
        scene_id=scene_id,
        scene=scene,
        category_priors_payload=category_priors_payload,
        category_priors_path=category_priors_path,
        output_root=output_root,
        seed=seed,
        feature_loader=feature_loader,
        gaussian_loader=gaussian_loader,
        label_loader=label_loader,
        raw_bank_builder=raw_bank_builder,
        graph_builder=graph_builder,
    )
    _, replay_rebuilt = _replay_scene_modes(
        paths=paths,
        identity=identity,
        graph=graph,
        xyz_scene=xyz_scene,
        category_priors_payload=category_priors_payload,
        modes=MODES,
        merge_builder=merge_builder,
        force_rebuild=bool(graph_rebuilt),
    )
    rebuilt = [*graph_rebuilt, *replay_rebuilt]
    completion = _write_completion_if_both_modes_exist(
        scene_id=scene_id,
        paths=paths,
        identity=identity,
        graph=graph,
        rebuilt=rebuilt,
    )
    if completion is None:  # pragma: no cover - both modes were requested above
        raise AssertionError("both replay modes completed without a completion marker")
    if not rebuilt:
        return {**completion, "status": "reused"}
    return completion


def _runner_inputs(
    *,
    runtime_manifest: str | Path,
    category_priors: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str] | str | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    tuple[str, ...],
    Path,
    Mapping[str, Any],
    Path,
]:
    scenes = load_scene_runtime_manifest(runtime_manifest)
    if scene_ids is None:
        selected = tuple(sorted(scenes))
    elif isinstance(scene_ids, str):
        selected = tuple(item.strip() for item in scene_ids.split(",") if item.strip())
    else:
        selected = tuple(map(str, scene_ids))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("scene_ids must be a non-empty unique sequence")
    unknown = sorted(set(selected).difference(scenes))
    if unknown:
        raise KeyError(f"runtime manifest lacks scenes: {unknown}")
    prior_path = Path(category_priors).resolve()
    payload = load_json(prior_path)
    if not isinstance(payload, Mapping):
        raise TypeError("category priors must be a JSON object")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return scenes, selected, prior_path, payload, root


def build_category_fragment_graphs(
    *,
    runtime_manifest: str | Path,
    category_priors: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str] | str | None = None,
    seed: int = SEED,
    feature_loader: Callable[
        [str | Path], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = load_feature_ply,
    gaussian_loader: Callable[[str | Path], np.ndarray] = load_ply_xyz,
    label_loader: Callable[[Path], np.ndarray] = _load_label_features,
    raw_bank_builder: Callable[..., CandidateBank] = build_raw_cluster_bank,
    graph_builder: Callable[..., FragmentGraph] = build_fragment_graph,
) -> dict[str, Any]:
    """Build only the frozen raw banks and public fragment graphs."""

    scenes, selected, prior_path, payload, root = _runner_inputs(
        runtime_manifest=runtime_manifest,
        category_priors=category_priors,
        output_root=output_root,
        scene_ids=scene_ids,
    )
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        _, _, _, graph, _, rebuilt = _ensure_scene_graph(
            scene_id=scene_id,
            scene=scenes[scene_id],
            category_priors_payload=payload,
            category_priors_path=prior_path,
            output_root=root,
            seed=seed,
            feature_loader=feature_loader,
            gaussian_loader=gaussian_loader,
            label_loader=label_loader,
            raw_bank_builder=raw_bank_builder,
            graph_builder=graph_builder,
        )
        rows.append(
            {
                "scene_id": scene_id,
                "status": "completed" if rebuilt else "reused",
                "rebuilt": rebuilt,
                "raw_fragment_count": len(graph.nodes),
                "fragment_edge_count": len(graph.edges),
                "gt_used": False,
            }
        )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "stage": "build-fragment-graph",
        "scene_ids": list(selected),
        "feature_source": FEATURE_SOURCE,
        "semantic_route": SEMANTIC_ROUTE,
        "seed": int(seed),
        "scenes": rows,
    }
    write_json(root / "build_summary.json", summary)
    return summary


def merge_category_fragment_graphs(
    *,
    runtime_manifest: str | Path,
    category_priors: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str] | str | None = None,
    modes: Sequence[str] = MODES,
    seed: int = SEED,
    merge_builder: Callable[..., FragmentMergeResult] = merge_category_fragments,
) -> dict[str, Any]:
    """Replay selected U/D modes from already persisted shared graphs."""

    if int(seed) != SEED:
        raise ValueError(f"fragment experiment requires frozen seed {SEED}")
    requested_modes = tuple(map(str, modes))
    if (
        not requested_modes
        or len(requested_modes) != len(set(requested_modes))
        or not set(requested_modes).issubset(MODES)
    ):
        raise ValueError(
            "modes must be a non-empty unique subset of ('global', 'class')"
        )
    scenes, selected, prior_path, payload, root = _runner_inputs(
        runtime_manifest=runtime_manifest,
        category_priors=category_priors,
        output_root=output_root,
        scene_ids=scene_ids,
    )
    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        paths, identity, _, graph, xyz_scene = _load_existing_scene_graph(
            scene_id=scene_id,
            scene=scenes[scene_id],
            category_priors_path=prior_path,
            output_root=root,
            seed=seed,
        )
        results, rebuilt = _replay_scene_modes(
            paths=paths,
            identity=identity,
            graph=graph,
            xyz_scene=xyz_scene,
            category_priors_payload=payload,
            modes=requested_modes,
            merge_builder=merge_builder,
        )
        completion = _write_completion_if_both_modes_exist(
            scene_id=scene_id,
            paths=paths,
            identity=identity,
            graph=graph,
            rebuilt=rebuilt,
        )
        rows.append(
            {
                "scene_id": scene_id,
                "status": "completed" if rebuilt else "reused",
                "modes": list(requested_modes),
                "rebuilt": rebuilt,
                "result_object_counts": {
                    mode: len(result.objects) for mode, result in results.items()
                },
                "scene_complete": completion is not None,
                "gt_used": False,
            }
        )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "stage": "merge-fragment-graph",
        "scene_ids": list(selected),
        "modes": list(requested_modes),
        "feature_source": FEATURE_SOURCE,
        "semantic_route": SEMANTIC_ROUTE,
        "seed": int(seed),
        "scenes": rows,
    }
    write_json(root / "merge_summary.json", summary)
    return summary


def run_category_fragment_merge(
    *,
    runtime_manifest: str | Path,
    category_priors: str | Path,
    output_root: str | Path,
    scene_ids: Sequence[str] | str | None = None,
    seed: int = SEED,
    **worker_dependencies: Any,
) -> dict[str, Any]:
    """Convenience composition of the explicit build and both-mode stages."""

    allowed = {
        "feature_loader",
        "gaussian_loader",
        "label_loader",
        "raw_bank_builder",
        "graph_builder",
        "merge_builder",
    }
    unknown = sorted(set(worker_dependencies).difference(allowed))
    if unknown:
        raise TypeError(f"unknown worker dependencies: {unknown}")
    build_dependencies = {
        key: value
        for key, value in worker_dependencies.items()
        if key != "merge_builder"
    }
    build_category_fragment_graphs(
        runtime_manifest=runtime_manifest,
        category_priors=category_priors,
        output_root=output_root,
        scene_ids=scene_ids,
        seed=seed,
        **build_dependencies,
    )
    return merge_category_fragment_graphs(
        runtime_manifest=runtime_manifest,
        category_priors=category_priors,
        output_root=output_root,
        scene_ids=scene_ids,
        modes=MODES,
        seed=seed,
        **(
            {"merge_builder": worker_dependencies["merge_builder"]}
            if "merge_builder" in worker_dependencies
            else {}
        ),
    )


__all__ = [
    "FEATURE_SOURCE",
    "MODES",
    "SCHEMA",
    "SEMANTIC_ROUTE",
    "FragmentSceneArtifacts",
    "FragmentScenePaths",
    "build_category_fragment_graphs",
    "load_category_fragment_scene",
    "load_fragment_graph",
    "load_fragment_merge_result",
    "merge_category_fragment_graphs",
    "run_category_fragment_merge",
    "run_category_fragment_scene",
    "save_fragment_graph",
    "save_fragment_merge_result",
]
