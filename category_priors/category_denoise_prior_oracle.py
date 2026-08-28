from __future__ import annotations

"""Pure CPU oracle diagnostics for category-denoising priors.

The routines in this module deliberately construct evaluation-only Gaussian
objects from ScanNet ground truth.  They never create a deployable prediction.
Their only purpose is to ask whether the frozen global/class prior formulas can
rank a complete object above deterministic fragment and merge counterexamples.
"""

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .category_denoise import (
    _boundary_fixed_ratio_with_tree,
    pca_sorted_extents_m,
    size_compatibility,
    smoothness_compatibility,
    support_threshold,
)
from .evaluator import apply_transform

SCORE_FLOOR = math.exp(-12.5)
DEFAULT_RADII_M = (0.02, 0.05, 0.10)
SCORE_NAMES = (
    "U_size_score",
    "D_size_score",
    "U_smooth_score",
    "D_smooth_score",
    "U_combined_score",
    "D_combined_score",
    "U_combined_support_score",
    "D_combined_support_score",
)
PRIOR_SCORE_PAIRS = {
    "size": ("U_size_score", "D_size_score"),
    "smooth": ("U_smooth_score", "D_smooth_score"),
    "combined": ("U_combined_score", "D_combined_score"),
    "combined_support": (
        "U_combined_support_score",
        "D_combined_support_score",
    ),
}


def _array(value: Any, dtype: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _normalize_radii(
    radii_m: Sequence[float], main_radius_m: float
) -> tuple[tuple[float, ...], float]:
    radii = tuple(sorted({float(value) for value in radii_m}))
    main = float(main_radius_m)
    if not radii or any(not math.isfinite(value) or value <= 0 for value in radii):
        raise ValueError("radii_m must contain positive finite values")
    if not math.isfinite(main) or main <= 0:
        raise ValueError("main_radius_m must be positive and finite")
    if not any(
        math.isclose(main, value, rel_tol=0.0, abs_tol=1e-12) for value in radii
    ):
        raise ValueError("main_radius_m must be present in radii_m")
    return radii, main


def _size_bin(diagonal_m: float, spec: Mapping[str, Any] | None) -> str | None:
    if spec is None:
        return None
    boundaries = spec.get("boundaries_m")
    if not isinstance(boundaries, Mapping):
        raise TypeError("size_bins must contain boundaries_m")
    if diagonal_m <= float(boundaries["tiny_max_m"]):
        return "tiny"
    if diagonal_m <= float(boundaries["small_max_m"]):
        return "small"
    if diagonal_m <= float(boundaries["medium_max_m"]):
        return "medium"
    return "large"


def _prior_node(priors: Mapping[str, Any], class_name: str | None) -> Mapping[str, Any]:
    global_node = priors.get("global")
    if not isinstance(global_node, Mapping) or not isinstance(
        global_node.get("shrunk"), Mapping
    ):
        raise TypeError("priors are missing global.shrunk")
    if class_name is None:
        return global_node
    categories = priors.get("categories")
    node = categories.get(class_name) if isinstance(categories, Mapping) else None
    if isinstance(node, Mapping) and isinstance(node.get("shrunk"), Mapping):
        return node
    return global_node


def _official_gt_objects(
    gt_xyz_m: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_names: Sequence[str],
    *,
    min_region_size: int,
    size_bins: Mapping[str, Any] | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    valid = (semantic >= 0) & (semantic < len(class_names)) & (instance >= 0)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for class_id, instance_id in sorted(
        set(zip(semantic[valid].tolist(), instance[valid].tolist()))
    ):
        mask = valid & (semantic == class_id) & (instance == instance_id)
        point_ids = np.flatnonzero(mask).astype(np.int64)
        if len(point_ids) < int(min_region_size):
            continue
        points = gt_xyz_m[point_ids]
        diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        key = (int(class_id), int(instance_id))
        result[key] = {
            "class_id": int(class_id),
            "class_name": str(class_names[int(class_id)]),
            "instance_id": int(instance_id),
            "gt_point_ids": point_ids,
            "gt_point_count": len(point_ids),
            "gt_centroid_m": points.mean(axis=0),
            "gt_bbox_diagonal_m": diagonal,
            "size_bin": _size_bin(diagonal, size_bins),
        }
    return result


def gaussian_to_official_gt_assignments(
    gaussian_xyz_m: np.ndarray,
    gt_xyz_m: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    official_objects: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    radius_m: float,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    """Map every Gaussian to all-GT first, then retain official object matches.

    Building the KD tree from an already-filtered GT subset would incorrectly
    attract Gaussians whose true nearest point belongs to void, a non-protocol
    class, or a region below the official minimum size.
    """

    gaussians = _array(gaussian_xyz_m, np.float64, name="gaussian_xyz_m")
    gt = _array(gt_xyz_m, np.float64, name="gt_xyz_m")
    semantic = np.asarray(gt_semantic, dtype=np.int64)
    instance = np.asarray(gt_instance, dtype=np.int64)
    if gaussians.ndim != 2 or gaussians.shape[1:] != (3,):
        raise ValueError("gaussian_xyz_m must have shape (N, 3)")
    if gt.ndim != 2 or gt.shape[1:] != (3,):
        raise ValueError("gt_xyz_m must have shape (M, 3)")
    if semantic.shape != (len(gt),) or instance.shape != (len(gt),):
        raise ValueError("GT arrays must have matching lengths")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius_m must be positive and finite")
    if not len(gt):
        raise ValueError("at least one GT point is required")

    distances, nearest = cKDTree(gt).query(gaussians, k=1, workers=-1)
    within = np.isfinite(distances) & (distances <= radius) & (nearest < len(gt))
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    nearest_official = np.zeros(len(gaussians), dtype=bool)
    for gaussian_id in np.flatnonzero(within):
        gt_id = int(nearest[gaussian_id])
        key = (int(semantic[gt_id]), int(instance[gt_id]))
        if key in official_objects:
            groups[key].append(int(gaussian_id))
            nearest_official[gaussian_id] = True
    assignments = {
        key: np.asarray(groups.get(key, ()), dtype=np.int64)
        for key in official_objects
    }
    diagnostics = {
        "gaussian_count": len(gaussians),
        "within_radius_count": int(np.count_nonzero(within)),
        "within_radius_fraction": float(np.mean(within)) if len(within) else 0.0,
        "official_mapped_count": int(np.count_nonzero(nearest_official)),
        "official_mapped_fraction": float(np.mean(nearest_official))
        if len(nearest_official)
        else 0.0,
        "nearest_nonofficial_or_void_count": int(
            np.count_nonzero(within & ~nearest_official)
        ),
        "outside_radius_count": int(np.count_nonzero(~within)),
        "unsupported_definition": (
            "Gaussian is excluded when its nearest point among all GT points is "
            "outside the radius or is not in an official-valid SAGA20 object"
        ),
    }
    return assignments, diagnostics


def deterministic_pca_half_fragment(
    gaussian_ids: np.ndarray | Sequence[int], gaussian_xyz_m: np.ndarray
) -> np.ndarray:
    """Return a stable PCA half-object fragment without random sampling."""

    ids = np.unique(np.asarray(gaussian_ids, dtype=np.int64))
    xyz = _array(gaussian_xyz_m, np.float64, name="gaussian_xyz_m")
    if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(xyz)):
        raise ValueError("gaussian_ids are outside gaussian_xyz_m")
    if len(ids) < 3:
        return np.empty(0, dtype=np.int64)
    points = xyz[ids]
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    top = float(eigenvalues[order[0]])
    second = float(eigenvalues[order[1]])
    scale = max(abs(top), 1.0)
    if abs(top - second) <= 1e-10 * scale:
        ranges = np.ptp(points, axis=0)
        axis_index = int(np.argmax(ranges))
        axis = np.eye(3, dtype=np.float64)[axis_index]
    else:
        axis = np.asarray(eigenvectors[:, order[0]], dtype=np.float64)
        sign_index = int(np.argmax(np.abs(axis)))
        if axis[sign_index] < 0:
            axis = -axis
    projection = centered @ axis
    stable_order = np.lexsort((ids, projection))
    count = int(math.ceil(0.5 * len(ids)))
    fragment = ids[stable_order[:count]]
    return np.sort(fragment).astype(np.int64, copy=False)


def nearest_merge_partner(
    target_key: tuple[int, int],
    eligible_keys: Sequence[tuple[int, int]],
    official_objects: Mapping[tuple[int, int], Mapping[str, Any]],
) -> tuple[tuple[int, int] | None, float | None]:
    """Choose the nearest other eligible GT object without score cherry-picking."""

    if target_key not in official_objects:
        raise KeyError(target_key)
    target = np.asarray(official_objects[target_key]["gt_centroid_m"], dtype=np.float64)
    candidates: list[tuple[float, int, int]] = []
    for class_id, instance_id in eligible_keys:
        key = (int(class_id), int(instance_id))
        if key == target_key:
            continue
        centroid = np.asarray(official_objects[key]["gt_centroid_m"], dtype=np.float64)
        candidates.append(
            (float(np.linalg.norm(centroid - target)), key[0], key[1])
        )
    if not candidates:
        return None, None
    distance, class_id, instance_id = min(candidates)
    return (class_id, instance_id), distance


def _variant_metrics(
    gaussian_ids: np.ndarray,
    gaussian_xyz_m: np.ndarray,
    boundary_tree: cKDTree,
    priors: Mapping[str, Any],
    target_class: str,
) -> dict[str, Any]:
    ids = np.unique(np.asarray(gaussian_ids, dtype=np.int64))
    if len(ids) < 3:
        raise ValueError("oracle scoring requires at least three Gaussians")
    mask = np.zeros(len(gaussian_xyz_m), dtype=bool)
    mask[ids] = True
    extents = pca_sorted_extents_m(gaussian_xyz_m[ids], 1.0)
    boundary = _boundary_fixed_ratio_with_tree(gaussian_xyz_m, mask, boundary_tree)
    candidate = {
        "metric_extents_m": extents,
        "boundary_ratio_5cm": boundary,
    }
    global_node = _prior_node(priors, None)
    class_node = _prior_node(priors, target_class)
    u_size = size_compatibility(candidate, global_node)
    d_size = size_compatibility(candidate, class_node)
    u_smooth = smoothness_compatibility(candidate, global_node)
    d_smooth = smoothness_compatibility(candidate, class_node)
    u_support = support_threshold(priors, target_class, "uniform")
    d_support = support_threshold(priors, target_class, "class")
    u_pass = len(ids) >= u_support
    d_pass = len(ids) >= d_support
    u_combined = float(u_size * u_smooth)
    d_combined = float(d_size * d_smooth)
    return {
        "gaussian_count": len(ids),
        "metric_extent_short_m": float(extents[0]),
        "metric_extent_mid_m": float(extents[1]),
        "metric_extent_long_m": float(extents[2]),
        "boundary_ratio_5cm": float(boundary),
        "U_G_size": float(u_size),
        "D_G_size": float(d_size),
        "U_B_smooth": float(u_smooth),
        "D_B_smooth": float(d_smooth),
        "U_support_threshold": int(u_support),
        "D_support_threshold": int(d_support),
        "U_support_pass": bool(u_pass),
        "D_support_pass": bool(d_pass),
        "U_size_score": float(u_size),
        "D_size_score": float(d_size),
        "U_smooth_score": float(u_smooth),
        "D_smooth_score": float(d_smooth),
        "U_combined_score": u_combined,
        "D_combined_score": d_combined,
        "U_combined_support_score": u_combined if u_pass else 0.0,
        "D_combined_support_score": d_combined if d_pass else 0.0,
        "U_size_floor": bool(np.isclose(u_size, SCORE_FLOOR, rtol=1e-10, atol=1e-15)),
        "D_size_floor": bool(np.isclose(d_size, SCORE_FLOOR, rtol=1e-10, atol=1e-15)),
        "U_smooth_floor": bool(
            np.isclose(u_smooth, SCORE_FLOOR, rtol=1e-10, atol=1e-15)
        ),
        "D_smooth_floor": bool(
            np.isclose(d_smooth, SCORE_FLOOR, rtol=1e-10, atol=1e-15)
        ),
    }


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _comparison(full: float, negative: float) -> tuple[int, float]:
    if np.isclose(full, negative, rtol=1e-9, atol=1e-12):
        return 0, 0.5
    return (1, 1.0) if full > negative else (-1, 0.0)


def build_oracle_scene(
    *,
    scene_id: str,
    physical_scene_id: str,
    gaussian_xyz: np.ndarray,
    gaussian_to_gt_transform: Sequence[Sequence[float]],
    gt_xyz_m: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    class_names: Sequence[str],
    priors: Mapping[str, Any],
    size_bins: Mapping[str, Any] | None = None,
    radii_m: Sequence[float] = DEFAULT_RADII_M,
    main_radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Construct full/fragment/merge oracle rows for one scene.

    GT and the Gaussian-to-GT transform are explicit inputs.  The returned
    rows are evaluation-only and contain no deployable prediction labels.
    """

    radii, main = _normalize_radii(radii_m, main_radius_m)
    scene_name = str(scene_id)
    physical_name = str(physical_scene_id)
    if not scene_name or not physical_name:
        raise ValueError("scene identifiers must be non-empty")
    gt_xyz = _array(gt_xyz_m, np.float64, name="gt_xyz_m")
    semantic = np.asarray(gt_semantic, dtype=np.int64)
    instance = np.asarray(gt_instance, dtype=np.int64)
    if gt_xyz.ndim != 2 or gt_xyz.shape[1:] != (3,):
        raise ValueError("gt_xyz_m must have shape (N, 3)")
    if semantic.shape != (len(gt_xyz),) or instance.shape != (len(gt_xyz),):
        raise ValueError("GT arrays must have matching lengths")
    gaussian_xyz_m = _array(
        apply_transform(
            _array(gaussian_xyz, np.float64, name="gaussian_xyz"),
            gaussian_to_gt_transform,
        ),
        np.float64,
        name="transformed gaussian_xyz",
    )
    official = _official_gt_objects(
        gt_xyz,
        semantic,
        instance,
        tuple(map(str, class_names)),
        min_region_size=int(min_region_size),
        size_bins=size_bins,
    )
    boundary_tree = cKDTree(gaussian_xyz_m)
    object_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    radius_diagnostics: dict[str, Any] = {}

    for radius in radii:
        assignments, mapping = gaussian_to_official_gt_assignments(
            gaussian_xyz_m,
            gt_xyz,
            semantic,
            instance,
            official,
            radius_m=radius,
        )
        eligible_keys = sorted(key for key, ids in assignments.items() if len(ids) >= 3)
        full_metrics: dict[tuple[int, int], dict[str, Any]] = {}
        for key, gt_object in official.items():
            ids = assignments[key]
            eligible = len(ids) >= 3
            row = {
                "scene_id": scene_name,
                "physical_scene_id": physical_name,
                "radius_m": float(radius),
                "is_main_radius": bool(math.isclose(radius, main, abs_tol=1e-12)),
                "class_id": int(gt_object["class_id"]),
                "class_name": str(gt_object["class_name"]),
                "instance_id": int(gt_object["instance_id"]),
                "size_bin": gt_object["size_bin"],
                "gt_point_count": int(gt_object["gt_point_count"]),
                "gt_bbox_diagonal_m": float(gt_object["gt_bbox_diagonal_m"]),
                "gaussian_count": len(ids),
                "eligible": bool(eligible),
                "oracle_purity_by_construction": 1.0 if len(ids) else None,
                "oracle_unsupported_fraction_by_construction": (
                    0.0 if len(ids) else None
                ),
                "unsupported_definition": mapping["unsupported_definition"],
            }
            if eligible:
                metrics = _variant_metrics(
                    ids,
                    gaussian_xyz_m,
                    boundary_tree,
                    priors,
                    str(gt_object["class_name"]),
                )
                full_metrics[key] = metrics
                row.update(metrics)
            object_rows.append(row)

        for key in eligible_keys:
            gt_object = official[key]
            target_class = str(gt_object["class_name"])
            full_ids = assignments[key]
            full = full_metrics[key]

            fragment_ids = deterministic_pca_half_fragment(full_ids, gaussian_xyz_m)
            if len(fragment_ids) >= 3:
                negative = _variant_metrics(
                    fragment_ids,
                    gaussian_xyz_m,
                    boundary_tree,
                    priors,
                    target_class,
                )
                pair = {
                    "scene_id": scene_name,
                    "physical_scene_id": physical_name,
                    "radius_m": float(radius),
                    "class_id": key[0],
                    "class_name": target_class,
                    "instance_id": key[1],
                    "size_bin": gt_object["size_bin"],
                    "negative_type": "fragment",
                    "negative_class_id": key[0],
                    "negative_class_name": target_class,
                    "negative_instance_id": key[1],
                    "merge_centroid_distance_m": None,
                    "same_class_merge": None,
                }
                pair.update(_prefixed("full_", full))
                pair.update(_prefixed("negative_", negative))
                for score_name in SCORE_NAMES:
                    comparison, auc = _comparison(
                        float(full[score_name]), float(negative[score_name])
                    )
                    pair[f"{score_name}_comparison"] = comparison
                    pair[f"{score_name}_paired_auc"] = auc
                pair_rows.append(pair)

            partner, distance = nearest_merge_partner(key, eligible_keys, official)
            if partner is not None:
                merge_ids = np.union1d(full_ids, assignments[partner])
                negative = _variant_metrics(
                    merge_ids,
                    gaussian_xyz_m,
                    boundary_tree,
                    priors,
                    target_class,
                )
                partner_object = official[partner]
                pair = {
                    "scene_id": scene_name,
                    "physical_scene_id": physical_name,
                    "radius_m": float(radius),
                    "class_id": key[0],
                    "class_name": target_class,
                    "instance_id": key[1],
                    "size_bin": gt_object["size_bin"],
                    "negative_type": "merge",
                    "negative_class_id": partner[0],
                    "negative_class_name": str(partner_object["class_name"]),
                    "negative_instance_id": partner[1],
                    "merge_centroid_distance_m": float(distance),
                    "same_class_merge": bool(partner[0] == key[0]),
                }
                pair.update(_prefixed("full_", full))
                pair.update(_prefixed("negative_", negative))
                for score_name in SCORE_NAMES:
                    comparison, auc = _comparison(
                        float(full[score_name]), float(negative[score_name])
                    )
                    pair[f"{score_name}_comparison"] = comparison
                    pair[f"{score_name}_paired_auc"] = auc
                pair_rows.append(pair)

        radius_diagnostics[f"{radius:.2f}"] = {
            **mapping,
            "official_valid_gt_count": len(official),
            "eligible_object_count": len(eligible_keys),
        }

    return {
        "schema": "saga-category-prior-oracle-scene-v1",
        "scene_id": scene_name,
        "physical_scene_id": physical_name,
        "main_radius_m": main,
        "radii_m": list(radii),
        "gaussian_count": len(gaussian_xyz_m),
        "gt_point_count": len(gt_xyz),
        "official_valid_gt_count": len(official),
        "radius_diagnostics": radius_diagnostics,
        "objects": object_rows,
        "pairs": pair_rows,
    }


def scan_physical_equal_summary(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> dict[str, Any]:
    """Average rows within scan, then scans within physical scene, then scenes."""

    by_scan: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        raw = row.get(value_key)
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        by_scan[(str(row["physical_scene_id"]), str(row["scene_id"]))].append(value)
    scan_values = {
        key: float(np.mean(values)) for key, values in by_scan.items() if values
    }
    by_physical: dict[str, list[float]] = defaultdict(list)
    for (physical_id, _scene_id), value in scan_values.items():
        by_physical[physical_id].append(value)
    physical_values = {
        key: float(np.mean(values)) for key, values in sorted(by_physical.items())
    }
    return {
        "row_count": int(sum(len(values) for values in by_scan.values())),
        "scan_count": len(scan_values),
        "physical_scene_count": len(physical_values),
        "mean": float(np.mean(list(physical_values.values())))
        if physical_values
        else None,
        "per_scan": {
            f"{physical}/{scene}": value
            for (physical, scene), value in sorted(scan_values.items())
        },
        "per_physical_scene": physical_values,
    }


def summarize_score_domain(
    rows: Sequence[Mapping[str, Any]], score_name: str
) -> dict[str, Any]:
    values = np.asarray(
        [float(row[score_name]) for row in rows if row.get(score_name) is not None],
        dtype=np.float64,
    )
    if not len(values):
        return {"count": 0, "status": "score_domain_mixed"}
    quantiles = np.quantile(values, (0.0, 0.25, 0.50, 0.75, 1.0))
    floor_fraction = float(
        np.mean(np.isclose(values, SCORE_FLOOR, rtol=1e-10, atol=1e-15))
    )
    median = float(quantiles[2])
    status = (
        "score_domain_collapsed"
        if floor_fraction >= 0.5
        else "score_domain_usable"
        if median > 0.01
        else "score_domain_mixed"
    )
    return {
        "count": len(values),
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": median,
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "floor_fraction": floor_fraction,
        "below_1e-6_fraction": float(np.mean(values < 1e-6)),
        "below_1e-4_fraction": float(np.mean(values < 1e-4)),
        "below_1e-2_fraction": float(np.mean(values < 1e-2)),
        "status": status,
    }


def _pair_summary(
    rows: Sequence[Mapping[str, Any]], negative_type: str
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("negative_type") == negative_type]
    result: dict[str, Any] = {"pair_count": len(selected), "scores": {}}
    for score_name in SCORE_NAMES:
        comparison_key = f"{score_name}_comparison"
        auc_key = f"{score_name}_paired_auc"
        comparisons = [int(row[comparison_key]) for row in selected]
        result["scores"][score_name] = {
            "full_higher_fraction": float(np.mean(np.asarray(comparisons) > 0))
            if comparisons
            else None,
            "tie_fraction": float(np.mean(np.asarray(comparisons) == 0))
            if comparisons
            else None,
            "negative_higher_fraction": float(np.mean(np.asarray(comparisons) < 0))
            if comparisons
            else None,
            "scene_equal_paired_auc": scan_physical_equal_summary(selected, auc_key),
        }
    return result


def _paired_prior_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    u_score_name: str,
    d_score_name: str,
) -> dict[str, Any]:
    """Compare class-shrunk and global scores on the same paired examples."""

    delta_rows: list[dict[str, Any]] = []
    for row in rows:
        u_key = f"{u_score_name}_paired_auc"
        d_key = f"{d_score_name}_paired_auc"
        if row.get(u_key) is None or row.get(d_key) is None:
            continue
        delta_rows.append(
            {
                "scene_id": row["scene_id"],
                "physical_scene_id": row["physical_scene_id"],
                "delta": float(row[d_key]) - float(row[u_key]),
            }
        )
    summary = scan_physical_equal_summary(delta_rows, "delta")
    physical = list(summary["per_physical_scene"].values())
    tolerance = 1e-12
    return {
        "scene_equal_D_minus_U": summary,
        "positive_physical_scene_count": sum(value > tolerance for value in physical),
        "tie_physical_scene_count": sum(abs(value) <= tolerance for value in physical),
        "negative_physical_scene_count": sum(value < -tolerance for value in physical),
    }


def _full_score_prior_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    u_score_name: str,
    d_score_name: str,
) -> dict[str, Any]:
    delta_rows = [
        {
            "scene_id": row["scene_id"],
            "physical_scene_id": row["physical_scene_id"],
            "delta": float(row[d_score_name]) - float(row[u_score_name]),
        }
        for row in rows
        if row.get(u_score_name) is not None and row.get(d_score_name) is not None
    ]
    return scan_physical_equal_summary(delta_rows, "delta")


def _prior_effects(
    *,
    object_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        factor: {
            "full_object_D_minus_U": _full_score_prior_effect(
                object_rows,
                u_score_name=u_name,
                d_score_name=d_name,
            ),
            "full_vs_fragment": _paired_prior_effect(
                [row for row in pair_rows if row.get("negative_type") == "fragment"],
                u_score_name=u_name,
                d_score_name=d_name,
            ),
            "full_vs_merge": _paired_prior_effect(
                [row for row in pair_rows if row.get("negative_type") == "merge"],
                u_score_name=u_name,
                d_score_name=d_name,
            ),
        }
        for factor, (u_name, d_name) in PRIOR_SCORE_PAIRS.items()
    }


def _stratum_summary(
    object_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "eligible_object_count": len(object_rows),
        "pair_count": len(pair_rows),
        "full_object_score_domain": {
            name: summarize_score_domain(object_rows, name) for name in SCORE_NAMES
        },
        "full_vs_fragment": _pair_summary(pair_rows, "fragment"),
        "full_vs_merge": _pair_summary(pair_rows, "merge"),
        "prior_effects": _prior_effects(
            object_rows=object_rows,
            pair_rows=pair_rows,
        ),
    }


def _combined_ranking(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[
        tuple[str, float, int, int], dict[str, Mapping[str, Any]]
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["scene_id"]),
            float(row["radius_m"]),
            int(row["class_id"]),
            int(row["instance_id"]),
        )
        grouped[key][str(row["negative_type"])] = row
    result: dict[str, Any] = {}
    for score_name in SCORE_NAMES:
        ranking_rows: list[dict[str, Any]] = []
        for pair_types in grouped.values():
            if not {"fragment", "merge"}.issubset(pair_types):
                continue
            fragment = pair_types["fragment"]
            merge = pair_types["merge"]
            auc_values = [
                float(fragment[f"{score_name}_paired_auc"]),
                float(merge[f"{score_name}_paired_auc"]),
            ]
            comparisons = [
                int(fragment[f"{score_name}_comparison"]),
                int(merge[f"{score_name}_comparison"]),
            ]
            full_rank = 1.0 + sum(value < 0 for value in comparisons) + 0.5 * sum(
                value == 0 for value in comparisons
            )
            ranking_rows.append(
                {
                    "scene_id": fragment["scene_id"],
                    "physical_scene_id": fragment["physical_scene_id"],
                    "paired_auc": float(np.mean(auc_values)),
                    "full_average_rank": float(full_rank),
                }
            )
        result[score_name] = {
            "object_count": len(ranking_rows),
            "scene_equal_auc": scan_physical_equal_summary(
                ranking_rows, "paired_auc"
            ),
            "scene_equal_full_rank": scan_physical_equal_summary(
                ranking_rows, "full_average_rank"
            ),
        }
    return result


def summarize_prior_oracle(
    scene_results: Sequence[Mapping[str, Any]], *, main_radius_m: float = 0.05
) -> dict[str, Any]:
    """Aggregate oracle diagnostics without treating objects as replicates."""

    objects = [dict(row) for result in scene_results for row in result["objects"]]
    pairs = [dict(row) for result in scene_results for row in result["pairs"]]
    radii = sorted({float(row["radius_m"]) for row in objects})
    main = float(main_radius_m)
    per_radius: dict[str, Any] = {}
    for radius in radii:
        radius_objects = [
            row
            for row in objects
            if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
            and bool(row["eligible"])
        ]
        radius_pairs = [
            row
            for row in pairs
            if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
        ]
        per_radius[f"{radius:.2f}"] = {
            "eligible_object_count": len(radius_objects),
            "physical_scene_count": len(
                {str(row["physical_scene_id"]) for row in radius_objects}
            ),
            "full_object_score_domain": {
                name: summarize_score_domain(radius_objects, name)
                for name in SCORE_NAMES
            },
            "full_vs_fragment": _pair_summary(radius_pairs, "fragment"),
            "full_vs_merge": _pair_summary(radius_pairs, "merge"),
            "combined_ranking": _combined_ranking(radius_pairs),
            "prior_effects": _prior_effects(
                object_rows=radius_objects,
                pair_rows=radius_pairs,
            ),
        }

    eligible_sets: list[set[tuple[str, int, int]]] = []
    for radius in radii:
        eligible_sets.append(
            {
                (str(row["scene_id"]), int(row["class_id"]), int(row["instance_id"]))
                for row in objects
                if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
                and bool(row["eligible"])
            }
        )
    common_objects = set.intersection(*eligible_sets) if eligible_sets else set()
    common_pair_sets: list[set[tuple[Any, ...]]] = []
    for radius in radii:
        common_pair_sets.append(
            {
                (
                    str(row["scene_id"]),
                    int(row["class_id"]),
                    int(row["instance_id"]),
                    str(row["negative_type"]),
                    int(row["negative_class_id"]),
                    int(row["negative_instance_id"]),
                )
                for row in pairs
                if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
            }
        )
    common_pairs = set.intersection(*common_pair_sets) if common_pair_sets else set()
    common_radius_sensitivity: dict[str, Any] = {}
    for radius in radii:
        common_radius_objects = [
            row
            for row in objects
            if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
            and (
                str(row["scene_id"]),
                int(row["class_id"]),
                int(row["instance_id"]),
            )
            in common_objects
        ]
        common_radius_pairs = [
            row
            for row in pairs
            if math.isclose(float(row["radius_m"]), radius, abs_tol=1e-12)
            and (
                str(row["scene_id"]),
                int(row["class_id"]),
                int(row["instance_id"]),
                str(row["negative_type"]),
                int(row["negative_class_id"]),
                int(row["negative_instance_id"]),
            )
            in common_pairs
        ]
        common_radius_sensitivity[f"{radius:.2f}"] = _stratum_summary(
            common_radius_objects,
            common_radius_pairs,
        )
    main_key = min(radii, key=lambda value: abs(value - main)) if radii else main
    main_objects = [
        row
        for row in objects
        if math.isclose(float(row["radius_m"]), main_key, abs_tol=1e-12)
        and bool(row["eligible"])
    ]
    main_pairs = [
        row
        for row in pairs
        if math.isclose(float(row["radius_m"]), main_key, abs_tol=1e-12)
    ]
    main_merge = _pair_summary(main_pairs, "merge")
    u_auc = main_merge["scores"]["U_combined_support_score"][
        "scene_equal_paired_auc"
    ]
    d_auc = main_merge["scores"]["D_combined_support_score"][
        "scene_equal_paired_auc"
    ]
    u_value = u_auc["mean"]
    d_value = d_auc["mean"]
    collapsed = any(
        summarize_score_domain(main_objects, name)["status"]
        == "score_domain_collapsed"
        for name in ("U_size_score", "D_size_score", "U_smooth_score", "D_smooth_score")
    )
    main_prior_effects = _prior_effects(
        object_rows=main_objects,
        pair_rows=main_pairs,
    )
    merge_effect = main_prior_effects["combined_support"]["full_vs_merge"]
    mapping_signs: set[int] = set()
    for radius_summary in common_radius_sensitivity.values():
        value = radius_summary["prior_effects"]["combined_support"][
            "full_vs_merge"
        ]["scene_equal_D_minus_U"]["mean"]
        if value is not None and not math.isclose(value, 0.0, abs_tol=1e-12):
            mapping_signs.add(1 if value > 0 else -1)
    mapping_sensitive = len(mapping_signs) > 1
    if collapsed:
        interpretation = "oracle-full-score-domain-collapsed"
    elif u_value is None or d_value is None:
        interpretation = "insufficient-oracle-pairs"
    elif mapping_sensitive:
        interpretation = "oracle-ranking-is-radius-sensitive"
    elif (
        d_value > u_value
        and merge_effect["positive_physical_scene_count"]
        > merge_effect["negative_physical_scene_count"]
    ):
        interpretation = "class-shrunk-prior-has-oracle-ranking-potential"
    elif d_value < u_value:
        interpretation = "class-shrunk-prior-underperforms-global-on-oracle-merges"
    elif math.isclose(u_value, 0.5, abs_tol=1e-12) and math.isclose(
        d_value, 0.5, abs_tol=1e-12
    ):
        interpretation = "global-and-class-priors-do-not-separate-oracle-merges"
    else:
        interpretation = "oracle-ranking-evidence-is-mixed"

    by_class: dict[str, Any] = {}
    for class_name in sorted({str(row["class_name"]) for row in main_objects}):
        class_objects = [
            row for row in main_objects if str(row["class_name"]) == class_name
        ]
        class_pairs = [
            row for row in main_pairs if str(row["class_name"]) == class_name
        ]
        by_class[class_name] = _stratum_summary(class_objects, class_pairs)
    by_scene: dict[str, Any] = {}
    for scene_id in sorted({str(row["scene_id"]) for row in main_objects}):
        scene_objects = [
            row for row in main_objects if str(row["scene_id"]) == scene_id
        ]
        scene_pairs = [row for row in main_pairs if str(row["scene_id"]) == scene_id]
        by_scene[scene_id] = _stratum_summary(scene_objects, scene_pairs)
    tiny_small_objects = [
        row for row in main_objects if row.get("size_bin") in {"tiny", "small"}
    ]
    tiny_small_pairs = [
        row for row in main_pairs if row.get("size_bin") in {"tiny", "small"}
    ]
    full_score_domain = {
        name: summarize_score_domain(main_objects, name) for name in SCORE_NAMES
    }

    return {
        "schema": "saga-category-prior-oracle-analysis-v1",
        "scene_count": len({str(row["scene_id"]) for row in objects}),
        "physical_scene_count": len(
            {str(row["physical_scene_id"]) for row in objects}
        ),
        "main_radius_m": main_key,
        "eligible_object_count": len(main_objects),
        "radius_sensitivity": per_radius,
        "common_eligible_object_count": len(common_objects),
        "common_pair_count": len(common_pairs),
        "common_subset_radius_sensitivity": common_radius_sensitivity,
        "mapping_radius_sensitive": mapping_sensitive,
        "full_object_score_domain": full_score_domain,
        "floor_saturation": {
            name: {
                "floor_fraction": domain.get("floor_fraction"),
                "status": domain["status"],
            }
            for name, domain in full_score_domain.items()
        },
        "full_vs_fragment": _pair_summary(main_pairs, "fragment"),
        "full_vs_merge": main_merge,
        "combined_ranking": _combined_ranking(main_pairs),
        "prior_effects": main_prior_effects,
        "scene_equal": {
            factor: effect["full_vs_merge"]["scene_equal_D_minus_U"]
            for factor, effect in main_prior_effects.items()
        },
        "per_scene": by_scene,
        "per_class": by_class,
        "tiny_small": _stratum_summary(tiny_small_objects, tiny_small_pairs),
        "interpretation": interpretation,
        "interpretation_details": {
            "oracle_full_score_domain_collapsed": collapsed,
            "mapping_radius_sensitive": mapping_sensitive,
            "combined_support_merge_U_scene_equal_auc": u_value,
            "combined_support_merge_D_scene_equal_auc": d_value,
            "combined_support_merge_D_minus_U": merge_effect,
            "oracle_purity_and_unsupported_are_by_construction": True,
        },
        "conclusion_boundary": (
            "This GT-oracle diagnostic tests only score-domain calibration and paired "
            "ranking on synthetic complete/fragment/merge masks. It does not measure "
            "candidate formation, deployable AP, or category-prior effectiveness in a "
            "complete instance-segmentation pipeline."
        ),
    }


__all__ = [
    "DEFAULT_RADII_M",
    "SCORE_FLOOR",
    "SCORE_NAMES",
    "build_oracle_scene",
    "deterministic_pca_half_fragment",
    "gaussian_to_official_gt_assignments",
    "nearest_merge_partner",
    "scan_physical_equal_summary",
    "summarize_prior_oracle",
    "summarize_score_domain",
]
