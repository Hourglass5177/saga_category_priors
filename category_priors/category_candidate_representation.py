from __future__ import annotations

"""Offline representation controls for a raw-clustering failure.

Ground truth is deliberately confined to this module.  The candidate worker,
repair code, and replay code never import it.  The two registered diagnostics
answer separate questions:

* do local affinity edges rank same-instance neighbours above different
  instances; and
* if an evaluator supplies perfect raw-cluster seeds, can the frozen mixed
  distance recover a useful candidate at all?
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_evaluation import _scene_context
from .category_candidate_trace import load_candidate_formation_trace
from .category_denoise import load_candidate_bank
from .evaluator import apply_transform
from .io import write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norm, out=np.zeros_like(array), where=norm > 0)


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUROC without a sklearn dependency."""

    from scipy.stats import rankdata

    truth = np.asarray(labels, dtype=bool)
    value = np.asarray(scores, dtype=np.float64)
    positives = int(np.count_nonzero(truth))
    negatives = int(len(truth) - positives)
    if not positives or not negatives:
        return 0.5
    ranks = rankdata(value, method="average")
    return float(
        (ranks[truth].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def load_affinity_feature_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the registered 32-D affinity feature and its point order."""

    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    fields = [f"f_{index}" for index in range(32)]
    missing = [name for name in fields if name not in names]
    if missing:
        raise ValueError(f"{path}: missing affinity fields {missing[:3]}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64, copy=False
    )
    feature = np.column_stack([vertex[name] for name in fields]).astype(
        np.float64, copy=False
    )
    if not np.isfinite(xyz).all() or not np.isfinite(feature).all():
        raise ValueError(f"{path}: non-finite affinity feature")
    return xyz, _normalize_rows(feature)


def local_affinity_edge_auc(
    xyz: np.ndarray,
    affinity: np.ndarray,
    instance_key: np.ndarray,
    selected_class: np.ndarray,
    *,
    k: int = 24,
) -> dict[str, Any]:
    """Measure local same-instance ranking within each semantic branch."""

    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    affinity = _normalize_rows(affinity)
    key = np.asarray(instance_key, dtype=np.int64)
    branch = np.asarray(selected_class, dtype=np.int64)
    if xyz.shape != (len(key), 3) or affinity.shape[0] != len(key) or branch.shape != key.shape:
        raise ValueError("representation arrays must share one point axis")
    edge_labels: list[np.ndarray] = []
    edge_scores: list[np.ndarray] = []
    class_rows: list[dict[str, Any]] = []
    for class_index in sorted(int(value) for value in np.unique(branch) if value >= 0):
        indices = np.flatnonzero((branch == class_index) & (key >= 0))
        if len(indices) < 3:
            continue
        width = min(max(int(k), 1), len(indices) - 1)
        _, neighbor = cKDTree(xyz[indices]).query(xyz[indices], k=width + 1)
        neighbor = np.asarray(neighbor, dtype=np.int64)[:, 1:]
        source = np.repeat(indices, width)
        target = indices[neighbor.reshape(-1)]
        truth = key[source] == key[target]
        score = np.sum(affinity[source] * affinity[target], axis=1)
        edge_labels.append(truth)
        edge_scores.append(score)
        class_rows.append(
            {
                "class_index": class_index,
                "point_count": len(indices),
                "edge_count": len(truth),
                "positive_edge_count": int(np.count_nonzero(truth)),
                "affinity_edge_auroc": _binary_auc(truth, score),
            }
        )
    if edge_labels:
        all_labels = np.concatenate(edge_labels)
        all_scores = np.concatenate(edge_scores)
        auc = _binary_auc(all_labels, all_scores)
    else:
        all_labels = np.empty(0, dtype=bool)
        auc = 0.5
    return {
        "affinity_edge_auroc": auc,
        "edge_count": int(len(all_labels)),
        "positive_edge_count": int(np.count_nonzero(all_labels)),
        "per_class": class_rows,
    }


def _scaled(values: np.ndarray, maximum: float) -> np.ndarray:
    return values / (maximum + 1e-8) if maximum > 0 else values


def oracle_seed_candidate_mask(
    *,
    selected_indices: np.ndarray,
    sampled_object_indices: np.ndarray,
    xyz_scene: np.ndarray,
    affinity: np.ndarray,
    semantic_score: np.ndarray,
    instance_distance_max: float,
    spatial_distance_max: float,
) -> np.ndarray:
    """Use GT only to supply a perfect seed; distance/radius remain frozen."""

    selected = np.asarray(selected_indices, dtype=np.int64)
    seed_global = np.asarray(sampled_object_indices, dtype=np.int64)
    output = np.zeros(len(xyz_scene), dtype=bool)
    if len(seed_global) < 3:
        return output
    local_lookup = {int(value): index for index, value in enumerate(selected)}
    try:
        seed_local = np.asarray([local_lookup[int(value)] for value in seed_global])
    except KeyError as exc:
        raise ValueError("oracle seeds must be selected semantic points") from exc
    selected_feature = _normalize_rows(np.asarray(affinity)[selected])
    minimum = np.asarray(xyz_scene).min(axis=0)
    span = np.asarray(xyz_scene).max(axis=0) - minimum
    selected_xyz = (np.asarray(xyz_scene)[selected] - minimum) / np.where(span > 0, span, 1.0)
    selected_score = np.asarray(semantic_score, dtype=np.float64)[selected]
    seed_feature = selected_feature[seed_local]
    seed_xyz = selected_xyz[seed_local]
    seed_score = selected_score[seed_local]
    instance = np.maximum(1.0 - seed_feature @ seed_feature.T, 0.0)
    spatial = np.linalg.norm(seed_xyz[:, None, :] - seed_xyz[None, :, :], axis=2)
    semantic = np.clip(1.0 - np.outer(seed_score, seed_score), 0.0, 1.0)
    distance = 0.5 * _scaled(instance, float(instance_distance_max)) + 0.3 * _scaled(
        spatial, float(spatial_distance_max)
    ) + 0.2 * semantic
    totals = distance.sum(axis=1)
    medoid_order = np.lexsort((seed_global, totals))
    medoid = int(seed_local[int(medoid_order[0])])
    seed_distance = distance[:, int(medoid_order[0])]
    radius = float(np.quantile(seed_distance, 0.95, method="linear"))
    query_instance = np.maximum(
        1.0 - selected_feature @ selected_feature[medoid], 0.0
    )
    query_spatial = np.linalg.norm(selected_xyz - selected_xyz[medoid], axis=1)
    query_semantic = np.clip(
        1.0 - selected_score * selected_score[medoid], 0.0, 1.0
    )
    query = 0.5 * _scaled(query_instance, float(instance_distance_max)) + 0.3 * _scaled(
        query_spatial, float(spatial_distance_max)
    ) + 0.2 * query_semantic
    output[selected[query <= radius]] = True
    return output


def _feature_path(scene: Mapping[str, Any]) -> Path:
    for key in (
        "contrastive_feature_point_cloud_path",
        "feature_point_cloud_path",
        "feature_ply_path",
    ):
        value = scene.get(key)
        if value:
            path = Path(str(value))
            return path if path.is_absolute() else Path(str(scene["base_path"])) / path
    base_path = scene.get("base_path")
    if base_path:
        # The canonical tune/final runtime manifests intentionally keep only
        # the scene root.  Match run_pipeline's native asset convention rather
        # than requiring an experiment-specific manifest rewrite.
        native = Path(str(base_path)) / "saga" / "contrastive_feature_point_cloud.ply"
        if native.is_file():
            return native
    raise KeyError("runtime scene is missing its affinity feature PLY")


def _mapped_object_gaussians(mapping: Any, point_indices: Any) -> np.ndarray:
    """Return the GT->Gaussian nearest indices for one GT object's points."""

    return np.asarray(mapping.gt_to_gaussian.indices, dtype=np.int64)[
        np.asarray(point_indices, dtype=np.int64)
    ]


def evaluate_candidate_representation(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    run_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Run the registered AUROC and perfect-seed controls on traced scenes."""

    from .io import load_json

    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins is not None else None
    rows: list[dict[str, Any]] = []
    scene_edge_rows: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        scene = scenes[scene_id]
        trace = load_candidate_formation_trace(run_root / "candidate_trace" / scene_id)
        c0 = load_candidate_bank(run_root / "bank" / scene_id / "C0-legacy")
        feature_xyz, affinity = load_affinity_feature_ply(_feature_path(scene))
        feature_xyz = apply_transform(
            feature_xyz,
            scene.get(
                "gaussian_to_gt_transform",
                np.eye(4, dtype=np.float64).tolist(),
            ),
        )
        context = _scene_context(
            scene_id=scene_id,
            scene=scene,
            gt_dir=gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        gaussian_xyz = np.asarray(context["gaussian_xyz"], dtype=np.float64)
        if feature_xyz.shape != gaussian_xyz.shape or not np.allclose(
            feature_xyz, gaussian_xyz, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"{scene_id}: affinity and Gaussian PLY point orders differ")
        reverse_semantic = np.asarray(context["reverse_semantic"], dtype=np.int64)
        reverse_instance = np.asarray(context["reverse_instance"], dtype=np.int64)
        instance_key = np.where(
            np.asarray(context["reverse_evaluable"], dtype=bool),
            reverse_semantic * 1_000_000 + reverse_instance,
            -1,
        )
        edge = local_affinity_edge_auc(
            gaussian_xyz,
            affinity,
            instance_key,
            np.asarray(trace.semantic_selected_class_index, dtype=np.int64),
        )
        scene_edge_rows.append(
            {
                "scene_id": scene_id,
                "affinity_edge_auroc": float(edge["affinity_edge_auroc"]),
                "edge_count": int(edge["edge_count"]),
                "positive_edge_count": int(edge["positive_edge_count"]),
            }
        )
        class_diagnostics = {
            int(row["branch_class_index"]): row.get("capture_diagnostics", {})
            for row in trace.class_rows
        }
        oracle_iou: list[float] = []
        for item in context["objects"]:
            selected = np.flatnonzero(
                np.asarray(trace.semantic_selected_class_index) == item.class_id
            )
            if not len(selected):
                continue
            mapped = _mapped_object_gaussians(
                context["mapping"], item.point_indices
            )
            valid_mapped = mapped >= 0
            semantic_coverage = float(
                np.count_nonzero(
                    valid_mapped
                    & (
                        np.asarray(trace.semantic_selected_class_index)[
                            np.maximum(mapped, 0)
                        ]
                        == item.class_id
                    )
                )
            ) / max(item.point_count, 1)
            if semantic_coverage < 0.25:
                continue
            target_key = item.class_id * 1_000_000 + item.instance_id
            # The oracle supplies only the instance identity of seeds that are
            # actually available to this class branch.  A GT point can map to
            # a Gaussian sampled by another semantic branch; passing such an
            # index to the class-local envelope is both impossible at runtime
            # and used to abort the diagnostic.
            seed = np.flatnonzero(
                (np.asarray(trace.sample_rank) >= 0)
                & (
                    np.asarray(trace.semantic_selected_class_index, dtype=np.int64)
                    == item.class_id
                )
                & (instance_key == target_key)
            )
            diagnostic = class_diagnostics.get(item.class_id, {})
            mask = oracle_seed_candidate_mask(
                selected_indices=selected,
                sampled_object_indices=seed,
                xyz_scene=gaussian_xyz,
                affinity=affinity,
                semantic_score=np.asarray(c0.semantic_top1_score),
                instance_distance_max=float(diagnostic.get("instance_distance_max", 0.0)),
                spatial_distance_max=float(diagnostic.get("spatial_distance_max", 0.0)),
            )
            target = instance_key == target_key
            intersection = int(np.count_nonzero(mask & target))
            union = int(np.count_nonzero(mask | target))
            value = intersection / union if union else 0.0
            oracle_iou.append(value)
            rows.append(
                {
                    "scene_id": scene_id,
                    "gt_class": item.class_name,
                    "gt_instance_id": item.instance_id,
                    "size_bin": item.size_bin,
                    "sampled_seed_count": len(seed),
                    "oracle_seed_iou": value,
                    "affinity_edge_auroc": edge["affinity_edge_auroc"],
                }
            )
    scene_aucs = {
        str(row["scene_id"]): float(row["affinity_edge_auroc"])
        for row in scene_edge_rows
    }
    oracle_recall = float(np.mean([row["oracle_seed_iou"] >= 0.25 for row in rows])) if rows else 0.0
    mean_auc = float(np.mean(list(scene_aucs.values()))) if scene_aucs else 0.5
    insufficient = mean_auc < 0.60 and oracle_recall < 0.20
    analysis = {
        "schema": "saga-candidate-representation-diagnostic-v1",
        "scene_count": len(scene_aucs),
        "diagnosable_object_count": len(rows),
        "mean_local_affinity_edge_auroc": mean_auc,
        "per_scene_affinity": scene_edge_rows,
        "oracle_seed_recall_025": oracle_recall,
        "representation_bottleneck_triggered": insufficient,
        "next_action": (
            "run-two-scene-10k-feature-control"
            if insufficient
            else "evaluate-C1-C2-full-assignment-repair"
        ),
        "gt_boundary": "offline_diagnosis_only",
    }
    write_rows(metrics_output, rows)
    write_json(analysis_output, analysis)
    return analysis


__all__ = [
    "evaluate_candidate_representation",
    "load_affinity_feature_ply",
    "local_affinity_edge_auc",
    "oracle_seed_candidate_mask",
]
