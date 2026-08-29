from __future__ import annotations

"""Two-scene raw-cluster factorial for representation and semantic routing.

This diagnostic intentionally stops at sampled HDBSCAN clusters.  It contains
no full-point expansion, KNN, filter, category prior, or final prediction
writer.  Ground truth is used only by the ``gt-class`` positive-control route
and by offline evaluation.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_clustering import cluster_metric_hdbscan
from .category_candidate_evaluation import _scene_context
from .category_candidate_representation import load_affinity_feature_ply
from .category_cluster_evaluation import evaluate_cluster_scene
from .category_cluster_scene_evaluation import _evaluation_scene
from .category_denoise import (
    CandidateBank,
    _normalize_rows,
    _readonly,
    _validate_bank,
    normalized_top1_32,
    stable_class_seed,
)
from .evaluator import apply_transform
from .io import load_json, write_json, write_rows
from .prompt_prior import materialize_prompt_priors
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .teacher_prior import SAGA20_CLASSES


SCHEMA = "saga-feature-routing-raw-cluster-factorial-v1"
DEV2 = ("scene0645_00", "scene0025_01")
RUNTIME_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)
FEATURE_SOURCES = ("native-2k-grounded", "v9-10k-dual-source")
ROUTES = ("predicted-32-top1", "gt-class-oracle")
SAMPLE_CAP = 5_000
SEED = 42


def load_feature_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned XYZ, affinity, and semantic vectors from one feature PLY."""

    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = set(vertex.dtype.names or ())
    affinity_fields = [f"f_{index}" for index in range(32)]
    semantic_fields = [f"sf_{index}" for index in range(32)]
    missing = [
        name for name in (*affinity_fields, *semantic_fields) if name not in names
    ]
    if missing:
        raise ValueError(f"{path}: missing feature fields {missing[:3]}")
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64, copy=False
    )
    affinity = np.column_stack([vertex[name] for name in affinity_fields]).astype(
        np.float64, copy=False
    )
    semantic = np.column_stack([vertex[name] for name in semantic_fields]).astype(
        np.float64, copy=False
    )
    if not all(np.isfinite(value).all() for value in (xyz, affinity, semantic)):
        raise ValueError(f"{path}: non-finite feature values")
    return xyz, _normalize_rows(affinity), _normalize_rows(semantic)


def predicted_route(
    semantic: np.ndarray, label_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top1 = normalized_top1_32(
        semantic,
        label_features,
        RUNTIME_CLASSES,
        SAGA20_CLASSES,
        threshold=0.7,
    )
    return (
        np.asarray(top1.top_class_index, dtype=np.int64),
        np.asarray(top1.top_score, dtype=np.float64),
        np.asarray(top1.branch_class_index, dtype=np.int64),
    )


def gt_class_route(
    gaussian_to_object: np.ndarray,
    object_class_ids: np.ndarray,
    taxonomy: Taxonomy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Route by GT class only; GT instance identity is deliberately ignored."""

    object_index = np.asarray(gaussian_to_object, dtype=np.int64)
    object_classes = np.asarray(object_class_ids, dtype=np.int64)
    runtime_lookup = {name: index for index, name in enumerate(RUNTIME_CLASSES)}
    top = np.zeros(len(object_index), dtype=np.int64)
    branch = np.full(len(object_index), -1, dtype=np.int64)
    valid = object_index >= 0
    for gaussian_id in np.flatnonzero(valid):
        canonical_id = int(object_classes[object_index[gaussian_id]])
        class_name = taxonomy.canonical_classes[canonical_id]
        if class_name in SAGA20_CLASSES:
            runtime_id = runtime_lookup[class_name]
            top[gaussian_id] = runtime_id
            branch[gaussian_id] = runtime_id
    score = np.where(branch >= 0, 1.0, 0.0)
    return top, score.astype(np.float64), branch


def build_raw_cluster_bank(
    *,
    affinity: np.ndarray,
    xyz_m: np.ndarray,
    top_class: np.ndarray,
    route_score: np.ndarray,
    branch_class: np.ndarray,
    global_typical_diag_m: float,
    scene_id: str,
    feature_source: str,
    route: str,
    seed: int = SEED,
    sample_cap: int = SAMPLE_CAP,
    hdbscan_factory: Any | None = None,
) -> CandidateBank:
    """Build sampled raw HDBSCAN clusters without any expansion or denoising."""

    point_count = len(xyz_m)
    arrays = (affinity, top_class, route_score, branch_class)
    if affinity.ndim != 2 or any(len(value) != point_count for value in arrays):
        raise ValueError("factorial arrays must share one Gaussian axis")
    full = np.full(point_count, -1, dtype=np.int64)
    core = np.full(point_count, -1, dtype=np.int64)
    confidence = np.zeros(point_count, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    next_id = 0
    for class_name in sorted(SAGA20_CLASSES):
        class_index = RUNTIME_CLASSES.index(class_name)
        selected = np.flatnonzero(branch_class == class_index)
        sample_count = min(len(selected), int(sample_cap))
        if sample_count < 3:
            class_rows.append(
                {"class": class_name, "selected": len(selected), "sampled": sample_count,
                 "raw_clusters": 0}
            )
            continue
        rng = np.random.default_rng(stable_class_seed(seed, class_name))
        sampled = selected[rng.permutation(len(selected))[:sample_count]]
        raw = cluster_metric_hdbscan(
            affinity[sampled],
            xyz_m[sampled],
            global_typical_diag_m,
            hdbscan_factory=hdbscan_factory,
        )
        labels = np.asarray(raw.labels, dtype=np.int64)
        membership = np.asarray(raw.membership, dtype=np.float64)
        for raw_id in raw.raw_cluster_ids:
            member_local = np.flatnonzero(labels == raw_id)
            member_global = sampled[member_local]
            candidate_id = next_id
            next_id += 1
            full[member_global] = candidate_id
            core[member_global] = candidate_id
            confidence[member_global] = membership[member_local]
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "branch_class": class_name,
                    "branch_class_index": class_index,
                    "full_point_count": int(len(member_global)),
                    "core_point_count": int(len(member_global)),
                    "raw_cluster_id": int(raw_id),
                }
            )
        class_rows.append(
            {"class": class_name, "selected": len(selected), "sampled": sample_count,
             "raw_clusters": len(raw.raw_cluster_ids)}
        )
    bank = CandidateBank(
        class_names=RUNTIME_CLASSES,
        saga20_names=tuple(SAGA20_CLASSES),
        scene_scale_m_per_unit=1.0,
        seed=int(seed),
        global_pre_knn=_readonly(np.full(point_count, -1, dtype=np.int64)),
        semantic_top1=_readonly(np.asarray(top_class, dtype=np.int64)),
        semantic_top1_score=_readonly(np.asarray(route_score, dtype=np.float64)),
        branch_full_labels=_readonly(full),
        branch_core_labels=_readonly(core),
        assignment_confidence=_readonly(confidence),
        candidates=tuple(rows),
        diagnostics={
            "schema": SCHEMA,
            "scene_id": scene_id,
            "feature_source": feature_source,
            "semantic_route": route,
            "raw_clusters_only": True,
            "full_expansion_used": False,
            "knn_used": False,
            "filter_used": False,
            "category_prior_used": False,
            "gt_route_uses_instance_identity": False,
            "class_diagnostics": class_rows,
            "raw_member_count": int(np.count_nonzero(full >= 0)),
            "raw_member_retained_count": int(np.count_nonzero(full >= 0)),
            "orphan_count": 0,
            "negative_metadata_count": 0,
        },
    )
    _validate_bank(bank)
    return bank


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_count = sum(int(row["candidate_count"]) for row in rows)
    count025 = sum(int(row["same_class_iou_025_count"]) for row in rows)
    count050 = sum(int(row["same_class_iou_050_count"]) for row in rows)
    tiny_count = sum(int(row["tiny_small_gt_count"]) for row in rows)
    tiny025 = sum(int(row["tiny_small_iou_025_count"]) for row in rows)
    return {
        "scene_count": len(rows),
        "candidate_count": candidate_count,
        "same_class_iou_025_count": count025,
        "same_class_iou_050_count": count050,
        "candidate_precision_025": count025 / candidate_count if candidate_count else 0.0,
        "tiny_small_recall_025": tiny025 / tiny_count if tiny_count else 0.0,
        "per_scene": [dict(row) for row in rows],
    }


def material_gain(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        int(candidate["candidate_count"]) <= 1.5 * max(int(baseline["candidate_count"]), 1)
        and (
            int(candidate["same_class_iou_025_count"])
            >= int(baseline["same_class_iou_025_count"]) + 2
            or float(candidate["tiny_small_recall_025"])
            >= float(baseline["tiny_small_recall_025"]) + 0.10
        )
    )


def infer_root_cause(conditions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    f2p = conditions["native-2k-grounded__predicted-32-top1"]
    f2g = conditions["native-2k-grounded__gt-class-oracle"]
    f10p = conditions["v9-10k-dual-source__predicted-32-top1"]
    f10g = conditions["v9-10k-dual-source__gt-class-oracle"]
    routing_gain = material_gain(f2g, f2p) or material_gain(f10g, f10p)
    representation_gain = material_gain(f10p, f2p) or material_gain(f10g, f2g)
    real_healthy = any(
        int(row["same_class_iou_050_count"]) >= 6
        and float(row["tiny_small_recall_025"]) >= 0.20
        for key, row in conditions.items()
        if key.endswith("__predicted-32-top1")
    )
    if routing_gain and representation_gain:
        conclusion = "both-semantic-routing-and-representation"
    elif routing_gain:
        conclusion = "semantic-routing-dominant"
    elif representation_gain:
        conclusion = "representation-version-dominant"
    else:
        conclusion = "affinity-objective-or-raw-clustering-dominant"
    if not real_healthy and routing_gain:
        conclusion = (
            "semantic-routing-contributor-but-raw-affinity-clustering-still-insufficient"
        )
    elif not real_healthy and representation_gain:
        conclusion = (
            "representation-contributor-but-raw-affinity-clustering-still-insufficient"
        )
    return {
        "semantic_routing_material_gain": routing_gain,
        "representation_version_material_gain": representation_gain,
        "root_cause": conclusion,
        "real_runtime_arm_healthy": real_healthy,
        "category_prior_tested": False,
        "interpretation_boundary": (
            "The 10k arm changes both training budget and affinity mask source; "
            "it identifies a representation-version effect, not a pure iteration effect."
        ),
    }


def run_feature_routing_factorial(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    category_priors: Path,
    feature_10k_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    output_dir: Path,
    size_bins: Path | None = None,
) -> dict[str, Any]:
    requested = tuple(map(str, scene_ids))
    if set(requested) != set(DEV2) or len(requested) != 2:
        raise ValueError(f"factorial requires exactly {DEV2}")
    scenes = load_scene_runtime_manifest(runtime_manifest)
    priors = materialize_prompt_priors(load_json(category_priors))
    size_spec = load_json(size_bins) if size_bins else None
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_rows: dict[str, list[dict[str, Any]]] = {}
    identities: list[dict[str, Any]] = []
    for scene_id in DEV2:
        scene = scenes[scene_id]
        evaluation = _evaluation_scene(
            scene_id=scene_id,
            scene=scene,
            gt_dir=gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=0.05,
            min_region_size=100,
        )
        context = _scene_context(
            scene_id=scene_id,
            scene=scene,
            gt_dir=gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=0.05,
            min_region_size=100,
        )
        base = Path(str(scene["base_path"]))
        feature_paths = {
            "native-2k-grounded": base / "saga/contrastive_feature_point_cloud.ply",
            "v9-10k-dual-source": feature_10k_root / scene_id / "contrastive_feature_point_cloud_10k.ply",
        }
        label_path = base / "saga/labels/label_features.pt"
        import torch
        label_features = torch.load(label_path, map_location="cpu").detach().cpu().numpy()
        for feature_source, feature_path in feature_paths.items():
            xyz, affinity, semantic = load_feature_ply(feature_path)
            xyz_eval = apply_transform(xyz, scene.get("gaussian_to_gt_transform", np.eye(4).tolist()))
            gaussian_xyz = np.asarray(context["gaussian_xyz"], dtype=np.float64)
            if xyz_eval.shape != gaussian_xyz.shape or not np.allclose(
                xyz_eval, gaussian_xyz, atol=1e-5, rtol=0.0
            ):
                raise ValueError(f"{scene_id}: {feature_source} point order differs")
            identities.append(
                {"scene_id": scene_id, "feature_source": feature_source,
                 "path": str(feature_path), "point_count": len(xyz),
                 "size_bytes": feature_path.stat().st_size}
            )
            routes = {
                "predicted-32-top1": predicted_route(semantic, label_features),
                "gt-class-oracle": gt_class_route(
                    evaluation.gaussian_to_gt_object_indices,
                    evaluation.gt_object_class_ids,
                    taxonomy,
                ),
            }
            for route, (top, score, branch) in routes.items():
                bank = build_raw_cluster_bank(
                    affinity=affinity,
                    xyz_m=xyz * float(scene["scene_scale_m_per_unit"]),
                    top_class=top,
                    route_score=score,
                    branch_class=branch,
                    global_typical_diag_m=priors.global_typical_diag_m,
                    scene_id=scene_id,
                    feature_source=feature_source,
                    route=route,
                )
                metrics = evaluate_cluster_scene(evaluation, bank).as_dict()
                metrics.pop("candidate_rows", None)
                key = f"{feature_source}__{route}"
                condition_rows.setdefault(key, []).append(metrics)
    conditions = {key: _aggregate(rows) for key, rows in condition_rows.items()}
    decision = infer_root_cause(conditions)
    rows = [
        {"condition": key, **row}
        for key, aggregate in conditions.items()
        for row in aggregate["per_scene"]
    ]
    analysis = {
        "schema": SCHEMA,
        "status": "complete",
        "scene_ids": list(DEV2),
        "factors": {"feature_source": list(FEATURE_SOURCES), "semantic_route": list(ROUTES)},
        "fixed_mechanics": {
            "raw_hdbscan_only": True, "sample_cap": SAMPLE_CAP, "seed": SEED,
            "semantic_threshold_predicted": 0.7, "min_cluster_size": 3,
            "min_samples": 3, "cluster_selection_epsilon": 0.01,
        },
        "feature_identities": identities,
        "conditions": conditions,
        "decision": decision,
    }
    write_rows(output_dir / "feature_routing_factorial_dev2.parquet", rows)
    write_json(output_dir / "feature_routing_factorial_analysis.json", analysis)
    return analysis


__all__ = [
    "DEV2", "FEATURE_SOURCES", "ROUTES", "RUNTIME_CLASSES",
    "build_raw_cluster_bank", "gt_class_route", "infer_root_cause",
    "load_feature_ply", "material_gain", "predicted_route",
    "run_feature_routing_factorial",
]
