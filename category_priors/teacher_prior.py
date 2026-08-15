from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODES = ("off", "original", "all-uniform", "size", "smooth", "small", "combined")
DATA_MODES = ("all-uniform", "size", "smooth", "small", "combined")
SAGA20_CLASSES = (
    "chair",
    "table",
    "plant",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "socket",
    "book",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "fan",
    "refrigerator",
    "cup",
    "phone",
    "trash can",
)

# These are the unchanged source/a800 branch mechanics.  The materialized table
# contains only category statistics and the parameters derived from them.
SEMANTIC_THRESHOLD = 0.7
SAMPLE_NUM = 5000
FEATURE_RATIO = 0.5
SPATIAL_RATIO = 0.3
SEMANTIC_RATIO = 0.2
ASSIGNMENT_THRESHOLD = 0.3
GLOBAL_MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3
MIN_CLUSTER_LOW = 3
MIN_CLUSTER_HIGH = 10
SMALL_EXPONENT = 0.5
GLOBAL_KNN_K = 256
KNN_LOW = 16
KNN_HIGH = 256
BOUNDARY_FLOOR = 0.02
RESCUE_RATIO = 0.1


def _q50(node: Mapping[str, Any], section: str, field: str) -> float:
    return float(node["shrunk"][section][field]["q50"])


def _statistics(node: Mapping[str, Any]) -> tuple[float, float, float]:
    diagonal_m = math.exp(_q50(node, "geometry", "log_bbox_diag_m"))
    surface_area_m2 = math.exp(_q50(node, "geometry", "log_surface_area_m2"))
    boundary = _q50(node, "neighborhood", "boundary_fixed:0.05")
    return diagonal_m, surface_area_m2, boundary


def _materialized_row(
    diagonal_m: float,
    surface_area_m2: float,
    boundary: float,
    global_area_m2: float,
    global_boundary: float,
) -> dict[str, Any]:
    min_cluster_size = int(
        np.clip(
            round(
                GLOBAL_MIN_CLUSTER_SIZE
                * (surface_area_m2 / global_area_m2) ** SMALL_EXPONENT
            ),
            MIN_CLUSTER_LOW,
            MIN_CLUSTER_HIGH,
        )
    )
    knn_k = int(
        np.clip(
            round(GLOBAL_KNN_K * global_boundary / max(boundary, BOUNDARY_FLOOR)),
            KNN_LOW,
            KNN_HIGH,
        )
    )
    return {
        "typical_diag_m": float(diagonal_m),
        "surface_area_m2": float(surface_area_m2),
        "boundary_ratio_5cm": float(boundary),
        "min_cluster_size": min_cluster_size,
        "min_samples": MIN_SAMPLES,
        "knn_k": knn_k,
        "rescue_radius_m": float(RESCUE_RATIO * diagonal_m),
    }


def materialize_teacher_prior(
    priors: Mapping[str, Any], *, branch_preservation: bool = False,
    restore_after_global_filter: bool = False,
) -> dict[str, Any]:
    """Materialize one mode-independent table from train-only shrunk priors."""
    global_diag_m, global_area_m2, global_boundary = _statistics(priors["global"])
    global_row = _materialized_row(
        global_diag_m,
        global_area_m2,
        global_boundary,
        global_area_m2,
        global_boundary,
    )
    # The uniform condition is the teacher's original all-class setting.
    global_row["min_cluster_size"] = GLOBAL_MIN_CLUSTER_SIZE
    global_row["min_samples"] = MIN_SAMPLES
    global_row["knn_k"] = GLOBAL_KNN_K

    classes: dict[str, dict[str, Any]] = {}
    for class_name, node in sorted(priors.get("categories", {}).items()):
        if not isinstance(node, Mapping):
            continue
        diagonal_m, surface_area_m2, boundary = _statistics(node)
        classes[str(class_name)] = _materialized_row(
            diagonal_m,
            surface_area_m2,
            boundary,
            global_area_m2,
            global_boundary,
        )
    return {
        "kind": "teacher_category_params",
        "branch_preservation": bool(branch_preservation),
        "restore_after_global_filter": bool(restore_after_global_filter),
        "global": global_row,
        "classes": classes,
    }


def load_teacher_category_params(path: str | Path) -> dict[str, Any]:
    """Load a materialized table; raw train priors are accepted for convenience."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("kind") == "teacher_category_params":
        if not isinstance(payload.get("global"), Mapping):
            raise ValueError("teacher_category_params is missing global parameters")
        if not isinstance(payload.get("classes"), Mapping):
            raise ValueError("teacher_category_params is missing class parameters")
        return payload
    if "global" in payload and "categories" in payload:
        return materialize_teacher_prior(payload)
    raise ValueError("expected teacher_category_params or train-only category_priors")


def resolve_teacher_parameters(
    table: Mapping[str, Any], class_name: str, mode: str
) -> dict[str, Any]:
    """Select the one factor controlled by ``mode`` from a shared class table."""
    if mode not in DATA_MODES:
        raise ValueError(f"mode {mode!r} does not use teacher category parameters")
    global_row = table["global"]
    class_row = table.get("classes", {}).get(class_name, global_row)
    use_size = mode in {"size", "combined"}
    use_smooth = mode in {"smooth", "combined"}
    use_small = mode in {"small", "combined"}
    return {
        "semantic_threshold": SEMANTIC_THRESHOLD,
        "sample_num": SAMPLE_NUM,
        "feature_ratio": FEATURE_RATIO,
        "spatial_ratio": SPATIAL_RATIO,
        "semantic_ratio": SEMANTIC_RATIO,
        "assignment_threshold": ASSIGNMENT_THRESHOLD,
        "spatial_scale_m": float(
            class_row["typical_diag_m"] if use_size else global_row["typical_diag_m"]
        ),
        "min_cluster_size": int(
            class_row["min_cluster_size"] if use_small else global_row["min_cluster_size"]
        ),
        "min_samples": MIN_SAMPLES,
        "knn_k": int(class_row["knn_k"] if use_smooth else global_row["knn_k"]),
        "rescue_radius_m": (
            float(class_row["rescue_radius_m"]) if use_small else None
        ),
        # Evidence protection is a shared structural factor.  Uniform uses the
        # global train-only radius; small/combined use the class radius.
        "protection_radius_m": float(
            class_row["rescue_radius_m"] if use_small else global_row["rescue_radius_m"]
        ),
        "typical_diag_m": float(class_row["typical_diag_m"]),
        "surface_area_m2": float(class_row["surface_area_m2"]),
        "boundary_ratio_5cm": float(class_row["boundary_ratio_5cm"]),
    }


def exclusive_top1_masks(
    semantic_features: Any,
    label_features: Any,
    class_indices: Any,
    threshold: float = SEMANTIC_THRESHOLD,
) -> dict[int, np.ndarray]:
    """Return order-independent masks after top-1 competition within SAGA20."""
    semantic = np.asarray(semantic_features, dtype=np.float64)
    labels = np.asarray(label_features, dtype=np.float64)
    indices = np.asarray(class_indices, dtype=np.int64)
    similarity = semantic @ labels[indices].T
    top_local = similarity.argmax(axis=1)
    top_class = indices[top_local]
    top_score = similarity[np.arange(len(similarity)), top_local]
    return {
        int(class_index): (top_class == int(class_index)) & (top_score >= float(threshold))
        for class_index in indices
    }


def saga20_branch_classes(class_to_idx: Mapping[str, int]) -> tuple[str, ...]:
    return tuple(name for name in SAGA20_CLASSES if name in class_to_idx)


def build_teacher_hdbscan(
    clusterer_type: Any, min_cluster_size: int, min_samples: int = MIN_SAMPLES
) -> Any:
    """Build branch HDBSCAN with min_samples explicit and independent from m_c."""
    return clusterer_type(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        cluster_selection_epsilon=0.01,
        allow_single_cluster=False,
        metric="precomputed",
    )


def merge_branch_labels(
    fallback_labels: Any,
    branch_labels: Any,
    branch_classes: Mapping[int, str],
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Overlay successful branches while leaving unclustered points on fallback."""
    merged = np.asarray(fallback_labels, dtype=np.int64).copy()
    branch = np.asarray(branch_labels, dtype=np.int64)
    preservation = np.full_like(merged, -1)
    next_label = int(merged[merged >= 0].max()) + 1 if np.any(merged >= 0) else 0
    preserved_classes: dict[int, str] = {}
    ordered_branches = sorted(
        ((int(label), str(class_name)) for label, class_name in branch_classes.items()),
        key=lambda item: (item[1], item[0]),
    )
    for branch_label, branch_class in ordered_branches:
        mask = branch == branch_label
        if not np.any(mask):
            continue
        merged[mask] = next_label
        preservation[mask] = next_label
        preserved_classes[next_label] = branch_class
        next_label += 1
    return merged, preservation, preserved_classes


def restore_preserved_branch(labels: Any, preservation: Any) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int64).copy()
    preserved = np.asarray(preservation, dtype=np.int64)
    mask = preserved >= 0
    result[mask] = preserved[mask]
    return result


def restore_surviving_branches(
    labels: Any, branch_membership: Any
) -> tuple[np.ndarray, int]:
    """Restore the extent only for branches accepted by global filtering."""
    result = np.asarray(labels, dtype=np.int64).copy()
    membership = np.asarray(branch_membership, dtype=np.int64)
    restored = 0
    for branch_id in np.unique(membership[membership >= 0]):
        branch_mask = membership == branch_id
        if np.any(result[branch_mask] == branch_id):
            result[branch_mask] = branch_id
            restored += 1
    return result, restored


def protect_multi_anchor_halo(
    labels: Any,
    branch_membership: Any,
    xyz_m: Any,
    branch_classes: Mapping[int, str],
    branch_parameters: Mapping[int, Mapping[str, Any]],
    vote_ratios: Mapping[int, Any],
    class_to_idx: Mapping[str, int],
    label_threshold: float,
    anchor_neighbors: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore only same-candidate halo supported by several surviving anchors.

    The global legacy KNN/filter remains authoritative.  A lost candidate point
    is restored only when its branch retains at least ``min_cluster_size``
    anchors, the 2D vote winner (including background) agrees with the branch
    class, and the point has ``anchor_neighbors`` anchors within the registered
    physical radius.  One surviving point can therefore never restore a whole
    proposal.
    """
    from scipy.spatial import cKDTree

    result = np.asarray(labels, dtype=np.int64).copy()
    membership = np.asarray(branch_membership, dtype=np.int64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if result.shape != membership.shape or xyz.shape != (len(result), 3):
        raise ValueError("labels, branch membership and xyz must describe the same points")
    neighbor_count = max(int(anchor_neighbors), 2)
    reasons: dict[str, int] = {}
    branches: dict[str, Any] = {}
    restored_points = 0
    accepted_branches = 0

    def reject(branch_id: int, reason: str, anchors: int, candidates: int) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1
        branches[str(branch_id)] = {
            "status": reason,
            "anchor_points": int(anchors),
            "candidate_points": int(candidates),
            "restored_points": 0,
        }

    for branch_id, branch_class in sorted(branch_classes.items()):
        branch_id = int(branch_id)
        candidate_mask = membership == branch_id
        candidate_count = int(candidate_mask.sum())
        anchor_indices = np.flatnonzero(candidate_mask & (result == branch_id))
        parameters = branch_parameters.get(branch_id, {})
        core = int(parameters.get("min_cluster_size", 0))
        if len(anchor_indices) < max(core, neighbor_count):
            reject(branch_id, "insufficient_anchors", len(anchor_indices), candidate_count)
            continue
        ratio = np.asarray(vote_ratios.get(branch_id, ()), dtype=np.float64)
        class_index = class_to_idx.get(str(branch_class))
        if class_index is None or ratio.shape != (len(class_to_idx),):
            reject(branch_id, "missing_vote", len(anchor_indices), candidate_count)
            continue
        class_ratio = float(ratio[class_index])
        background_ratio = max(0.0, 1.0 - float(ratio.sum()))
        foreground_winner = int(np.argmax(ratio)) if ratio.size else -1
        if (
            foreground_winner != int(class_index)
            or class_ratio < float(label_threshold)
            or class_ratio < background_ratio
        ):
            reject(branch_id, "vote_rejected", len(anchor_indices), candidate_count)
            continue
        radius_m = float(parameters.get("protection_radius_m", 0.0))
        if not np.isfinite(radius_m) or radius_m <= 0:
            reject(branch_id, "invalid_radius", len(anchor_indices), candidate_count)
            continue
        halo_indices = np.flatnonzero(candidate_mask & (result != branch_id))
        if len(halo_indices):
            distances, _ = cKDTree(xyz[anchor_indices]).query(
                xyz[halo_indices], k=neighbor_count, workers=-1
            )
            distances = np.asarray(distances).reshape(len(halo_indices), neighbor_count)
            accepted = distances[:, -1] <= radius_m
            restored = halo_indices[accepted]
            result[restored] = branch_id
        else:
            restored = np.empty(0, dtype=np.int64)
        restored_count = int(len(restored))
        restored_points += restored_count
        accepted_branches += 1
        branches[str(branch_id)] = {
            "status": "accepted",
            "class": str(branch_class),
            "anchor_points": int(len(anchor_indices)),
            "candidate_points": candidate_count,
            "restored_points": restored_count,
            "vote_ratio": class_ratio,
            "background_vote_ratio": background_ratio,
            "radius_m": radius_m,
            "min_cluster_size": core,
        }
    return result, {
        "mode": "multi-anchor",
        "anchor_neighbors": neighbor_count,
        "candidate_branches": len(branch_classes),
        "accepted_branches": accepted_branches,
        "restored_points": restored_points,
        "rejection_reasons": reasons,
        "branches": branches,
    }


def teacher_spatial_distance(xyz_m: Any, scale_m: float) -> np.ndarray:
    """D_xyz=min(||x_i-x_j||/d,1), with no per-matrix max normalization."""
    xyz = np.asarray(xyz_m, dtype=np.float64)
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    return np.minimum(distances / max(float(scale_m), 1e-12), 1.0)


def class_local_knn(labels: Any, xyz_m: Any, k: int) -> np.ndarray:
    """Apply deterministic KNN voting inside one semantic-class branch."""
    from scipy.spatial import cKDTree

    source = np.asarray(labels, dtype=np.int64)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    if len(source) == 0:
        return source.copy()
    k_eff = min(max(int(k), 1), len(source))
    _, indices = cKDTree(xyz).query(xyz, k=k_eff, workers=-1)
    indices = np.asarray(indices).reshape(len(source), k_eff)
    result = np.empty_like(source)
    for index, neighbors in enumerate(indices):
        values, counts = np.unique(source[neighbors], return_counts=True)
        result[index] = values[int(np.argmax(counts))]
    return result


def filter_small_class_clusters(labels: Any, min_cluster_size: int) -> np.ndarray:
    source = np.asarray(labels, dtype=np.int64)
    result = source.copy()
    values, counts = np.unique(source[source >= 0], return_counts=True)
    for value, count in zip(values, counts):
        if int(count) < int(min_cluster_size):
            result[source == value] = -1
    return result


def rescue_same_class_noise(
    labels: Any, xyz_m: Any, radius_m: float | None
) -> tuple[np.ndarray, int]:
    """Restore noise from the nearest assigned anchor in this same class branch."""
    from scipy.spatial import cKDTree

    result = np.asarray(labels, dtype=np.int64).copy()
    if radius_m is None:
        return result, 0
    xyz = np.asarray(xyz_m, dtype=np.float64)
    anchors = np.flatnonzero(result >= 0)
    noise = np.flatnonzero(result < 0)
    if not len(anchors) or not len(noise):
        return result, 0
    distances, indices = cKDTree(xyz[anchors]).query(xyz[noise], k=1, workers=-1)
    accepted = np.asarray(distances) <= float(radius_m)
    result[noise[accepted]] = result[anchors[np.asarray(indices)[accepted]]]
    return result, int(np.count_nonzero(accepted))
