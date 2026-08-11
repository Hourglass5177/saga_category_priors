from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json


CLASS_FIRST_MODES = ("uniform", "size", "smooth", "small", "combined")


def validate_class_first_priors(payload: Mapping[str, Any]) -> None:
    """Check only the structural/train-only contract; do not hash artifacts."""

    if payload.get("kind") != "category_priors":
        raise ValueError("class-first requires kind=category_priors")
    if payload.get("provenance", {}).get("splits") != ["train"]:
        raise ValueError("class-first priors must be fit from the train split only")
    if not isinstance(payload.get("global"), Mapping) or not isinstance(
        payload.get("categories"), Mapping
    ):
        raise ValueError("class-first priors require global and categories objects")


@dataclass(frozen=True)
class ClassFirstConfig:
    """Global parameters for the independent class-first postprocessor.

    Defaults are the source/refactor class-first settings.  Category priors do
    not pass through the former mapping/gating runtime; each factor below has
    one orthogonal effect.
    """

    semantic_threshold: float = 0.5
    sample_fraction: float = 0.03
    class_sample_max: int = 5000
    min_cluster_size: int = 10
    min_samples: int = 10
    knn_k: int = 256
    instance_feature_ratio: float = 0.3
    semantic_feature_ratio: float = 0.3
    xyz_feature_ratio: float = 0.4
    instance_threshold: float = 0.25
    cluster_selection_epsilon: float = 0.01
    allow_single_cluster: bool = False
    use_sor: bool = True
    sor_nb_neighbors: int = 50
    sor_std_ratio: float = 0.05
    opacity_threshold: float = 0.01
    scale_threshold: float = 0.8
    rescue_radius_ratio: float = 0.10
    small_area_exponent: float = 0.50

    def __post_init__(self) -> None:
        weights = (
            self.instance_feature_ratio,
            self.semantic_feature_ratio,
            self.xyz_feature_ratio,
        )
        if any(value < 0 for value in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-6
        ):
            raise ValueError("class-first distance weights must be nonnegative and sum to 1")
        if not 0.0 <= self.semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be in [0, 1]")
        if not 0.0 < self.sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if self.class_sample_max <= 0:
            raise ValueError("class_sample_max must be positive")
        if self.min_cluster_size <= 0 or self.min_samples <= 0 or self.knn_k <= 0:
            raise ValueError("min_cluster_size, min_samples and knn_k must be positive")
        if not 0.0 <= self.instance_threshold <= 1.0:
            raise ValueError("instance_threshold must be in [0, 1]")
        if self.sor_nb_neighbors <= 0 or self.sor_std_ratio < 0:
            raise ValueError("invalid SOR parameters")
        if self.rescue_radius_ratio not in (0.10, 0.20):
            raise ValueError("rescue_radius_ratio must be exactly 0.10 or 0.20")
        if self.small_area_exponent not in (0.25, 0.50):
            raise ValueError("small_area_exponent must be exactly 0.25 or 0.50")

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({"schema_version": "1.0", "kind": "class_first_config"})
        return payload


@dataclass
class ClassFirstResult:
    labels: np.ndarray
    assignment_confidence: np.ndarray
    instances: dict[int, dict[str, Any]]
    diagnostics: dict[str, Any]


def load_class_first_config(path: str | Path) -> ClassFirstConfig:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("class-first config must be a JSON object")
    if payload.get("kind") != "class_first_config":
        raise ValueError("class-first config kind must be 'class_first_config'")
    if payload.get("schema_version", "1.0") != "1.0":
        raise ValueError("unsupported class-first config schema")
    values = {
        key: value
        for key, value in payload.items()
        if key not in {"kind", "schema_version"}
    }
    known = set(ClassFirstConfig.__dataclass_fields__)
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown class-first config fields: {unknown}")
    return ClassFirstConfig(**values)


def _as_numpy(value: Any, dtype: np.dtype | type | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _normalize_rows(values: Any) -> np.ndarray:
    array = _as_numpy(values, np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def robust_scale_coordinates(xyz: Any) -> np.ndarray:
    """Median/IQR transform used by source/refactor's uniform path."""

    array = _as_numpy(xyz, np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    if not len(array):
        return array.copy()
    median = np.median(array, axis=0)
    q25, q75 = np.quantile(array, (0.25, 0.75), axis=0)
    scale = np.where(q75 - q25 > 1e-12, q75 - q25, 1.0)
    return (array - median) / scale


def build_class_masks(
    point_semantic_features: Any,
    label_features: Any,
    classes: Sequence[str],
    threshold: float,
    valid_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build mutually-exclusive top-1 semantic candidate sets."""

    semantic = _normalize_rows(point_semantic_features)
    labels = _normalize_rows(label_features)
    if semantic.shape[1] != labels.shape[1]:
        raise ValueError("semantic and label feature dimensions differ")
    if labels.shape[0] != len(classes):
        raise ValueError("label feature count must equal len(classes)")
    similarity = semantic @ labels.T
    top_class = similarity.argmax(axis=1)
    top_similarity = similarity[np.arange(len(similarity)), top_class]
    valid = (
        np.ones(len(semantic), dtype=bool)
        if valid_mask is None
        else _as_numpy(valid_mask, bool).reshape(-1)
    )
    if len(valid) != len(semantic):
        raise ValueError("valid_mask length differs from point count")
    masks = {
        str(name): (top_class == index) & (top_similarity >= threshold) & valid
        for index, name in enumerate(classes)
    }
    return top_class, top_similarity, masks


def class_local_knn(labels: Any, xyz_m: Any, k: int) -> np.ndarray:
    """Vectorized KNN majority vote, called separately for every class."""

    from scipy.spatial import cKDTree
    from scipy.stats import mode

    result = _as_numpy(labels, np.int64).reshape(-1)
    points = _as_numpy(xyz_m, np.float64)
    if len(result) != len(points):
        raise ValueError("labels and xyz lengths differ")
    if not len(result):
        return result.copy()
    neighbors = min(max(int(k), 1), len(result))
    tree = cKDTree(points)
    voted = np.empty(len(result), dtype=np.int64)
    query_chunk = 32_768
    for start in range(0, len(result), query_chunk):
        stop = min(start + query_chunk, len(result))
        _, indices = tree.query(points[start:stop], k=neighbors, workers=-1)
        if neighbors == 1:
            indices = indices[:, None]
        # scipy.stats.mode chooses the smallest value on a tie, as sklearn does.
        voted[start:stop] = mode(
            result[indices], axis=1, keepdims=False
        ).mode
    return voted


def statistical_outlier_mask(
    points: Any, nb_neighbors: int = 50, std_ratio: float = 0.05
) -> np.ndarray:
    """Statistical outlier removal without importing Open3D."""

    from scipy.spatial import cKDTree

    xyz = _as_numpy(points, np.float64)
    if len(xyz) < 3:
        return np.ones(len(xyz), dtype=bool)
    neighbors = min(int(nb_neighbors), len(xyz) - 1)
    distances, _ = cKDTree(xyz).query(xyz, k=neighbors + 1, workers=-1)
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = float(mean_distance.mean() + float(std_ratio) * mean_distance.std())
    return mean_distance <= threshold


def apply_sor_to_clusters(
    labels: Any,
    xyz_m: Any,
    nb_neighbors: int,
    std_ratio: float,
    sor_filter: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Mark each cluster's SOR outliers -1; never restore a whole cluster."""

    result = _as_numpy(labels, np.int64).reshape(-1).copy()
    points = _as_numpy(xyz_m, np.float64)
    if len(result) != len(points):
        raise ValueError("labels and xyz lengths differ")
    filter_fn = sor_filter or statistical_outlier_mask
    removed = 0
    for cluster_id in np.unique(result[result >= 0]):
        indices = np.flatnonzero(result == cluster_id)
        if len(indices) < 3:
            continue
        neighbors = min(int(nb_neighbors), len(indices) - 1)
        inlier = _as_numpy(
            filter_fn(points[indices], neighbors, float(std_ratio)), bool
        ).reshape(-1)
        if len(inlier) != len(indices):
            raise ValueError("SOR filter returned a mask with the wrong length")
        outliers = indices[~inlier]
        result[outliers] = -1
        removed += len(outliers)
    return result, int(removed)


def rescue_noise_by_anchor(
    labels: Any, xyz_m: Any, max_distance_m: float
) -> tuple[np.ndarray, int]:
    """Rescue noise from its nearest surviving anchor in the same class branch."""

    from scipy.spatial import cKDTree

    result = _as_numpy(labels, np.int64).reshape(-1).copy()
    points = _as_numpy(xyz_m, np.float64)
    if len(result) != len(points):
        raise ValueError("labels and xyz lengths differ")
    noise = np.flatnonzero(result < 0)
    anchors = np.flatnonzero(result >= 0)
    if not len(noise) or not len(anchors) or max_distance_m <= 0:
        return result, 0
    distance, nearest = cKDTree(points[anchors]).query(
        points[noise], k=1, workers=-1
    )
    accepted = np.isfinite(distance) & (distance <= float(max_distance_m))
    result[noise[accepted]] = result[anchors[nearest[accepted]]]
    return result, int(accepted.sum())


def _q50(stats: Mapping[str, Any], section: str, metric: str) -> float:
    try:
        value = float(stats[section][metric]["q50"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"prior statistic is missing: {section}.{metric}.q50") from exc
    if not math.isfinite(value):
        raise ValueError(f"prior statistic is non-finite: {section}.{metric}.q50")
    return value


def _prior_geometry(stats: Mapping[str, Any]) -> dict[str, float]:
    return {
        "d_c_m": math.exp(_q50(stats, "geometry", "log_bbox_diag_m")),
        "A_c_m2": math.exp(_q50(stats, "geometry", "log_surface_area_m2")),
        "b_c": _q50(stats, "neighborhood", "boundary_fixed:0.05"),
    }


def resolve_class_parameters(
    priors: Mapping[str, Any],
    config: ClassFirstConfig,
    class_name: str,
    mode: str,
) -> dict[str, Any]:
    """Resolve direct train-prior formulas, with strict uniform fallback."""

    if mode not in CLASS_FIRST_MODES:
        raise ValueError(f"unknown class-first mode: {mode}")
    global_geometry = _prior_geometry(priors["global"]["shrunk"])
    node = priors.get("categories", {}).get(class_name)
    geometry: dict[str, float | None]
    try:
        geometry = _prior_geometry(node["shrunk"])
        supported = isinstance(node, Mapping)
    except (KeyError, TypeError, ValueError):
        supported = False
        geometry = {
        "d_c_m": None,
        "A_c_m2": None,
        "b_c": None,
        }
    min_cluster = int(config.min_cluster_size)
    knn_k = int(config.knn_k)
    if supported and mode in {"small", "combined"}:
        min_cluster = int(
            np.clip(
                round(
                    config.min_cluster_size
                    * (geometry["A_c_m2"] / global_geometry["A_c_m2"])
                    ** config.small_area_exponent
                ),
                3,
                20,
            )
        )
    if supported and mode in {"smooth", "combined"}:
        knn_k = int(
            np.clip(
                round(
                    config.knn_k
                    * global_geometry["b_c"]
                    / max(float(geometry["b_c"]), 0.02)
                ),
                16,
                256,
            )
        )
    size_enabled = bool(supported and mode in {"size", "combined"})
    rescue_enabled = bool(supported and mode in {"small", "combined"})
    return {
        "class": class_name,
        "supported": supported,
        "min_cluster_size": min_cluster,
        "min_samples": int(config.min_samples),
        "knn_k": knn_k,
        "coordinate_mode": "metric_divide_d_c" if size_enabled else "robust_scale",
        "spatial_scale_m": geometry["d_c_m"] if size_enabled else None,
        "rescue_enabled": rescue_enabled,
        "rescue_radius_m": (
            config.rescue_radius_ratio * float(geometry["d_c_m"])
            if rescue_enabled
            else None
        ),
        **geometry,
        "d_global_m": global_geometry["d_c_m"],
        "A_global_m2": global_geometry["A_c_m2"],
        "b_global": global_geometry["b_c"],
    }


def sample_class_indices(
    count: int,
    min_cluster_size: int,
    config: ClassFirstConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    quota = min(
        int(count),
        int(config.class_sample_max),
        max(int(round(config.sample_fraction * count)), 4 * int(min_cluster_size)),
    )
    if quota <= 0:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(count, size=quota, replace=False)).astype(
        np.int64, copy=False
    )


def _hybrid_distance(
    instance_features: np.ndarray,
    semantic_features: np.ndarray,
    xyz: np.ndarray,
    config: ClassFirstConfig,
) -> np.ndarray:
    from scipy.spatial.distance import cdist

    instance_distance = np.clip(
        1.0 - instance_features @ instance_features.T, 0.0, None
    )
    semantic_distance = np.clip(
        1.0 - semantic_features @ semantic_features.T, 0.0, None
    )
    xyz_distance = cdist(xyz, xyz, metric="euclidean")
    return (
        config.instance_feature_ratio * instance_distance
        + config.semantic_feature_ratio * semantic_distance
        + config.xyz_feature_ratio * xyz_distance
    ).astype(np.float64, copy=False)


def _cluster_sample(
    distance: np.ndarray,
    min_cluster_size: int,
    config: ClassFirstConfig,
    clusterer_factory: Callable[..., Any] | None,
) -> np.ndarray:
    if clusterer_factory is None:
        from sklearn.cluster import HDBSCAN

        clusterer_factory = HDBSCAN
    clusterer = clusterer_factory(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(config.min_samples),
        cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        allow_single_cluster=bool(config.allow_single_cluster),
        metric="precomputed",
    )
    return _as_numpy(clusterer.fit_predict(distance), np.int64).reshape(-1)


def _assign_from_centers(
    sample_labels: np.ndarray,
    sample_instance: np.ndarray,
    sample_semantic: np.ndarray,
    sample_xyz: np.ndarray,
    instance_features: np.ndarray,
    semantic_features: np.ndarray,
    xyz: np.ndarray,
    config: ClassFirstConfig,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    from scipy.spatial.distance import cdist

    cluster_ids = np.unique(sample_labels[sample_labels >= 0])
    if not len(cluster_ids):
        return (
            np.full(len(instance_features), -1, dtype=np.int64),
            np.zeros(len(instance_features), dtype=np.float32),
            0,
            np.empty((len(instance_features), 0), dtype=np.float32),
        )
    instance_centers = _normalize_rows(
        np.stack(
            [sample_instance[sample_labels == cid].mean(axis=0) for cid in cluster_ids]
        )
    )
    semantic_centers = _normalize_rows(
        np.stack(
            [sample_semantic[sample_labels == cid].mean(axis=0) for cid in cluster_ids]
        )
    )
    xyz_centers = np.stack(
        [sample_xyz[sample_labels == cid].mean(axis=0) for cid in cluster_ids]
    )
    instance_similarity = (
        np.clip(instance_features @ instance_centers.T, -1.0, 1.0) + 1.0
    ) / 2.0
    semantic_similarity = (
        np.clip(semantic_features @ semantic_centers.T, -1.0, 1.0) + 1.0
    ) / 2.0
    xyz_similarity = np.exp(-cdist(xyz, xyz_centers, metric="euclidean") / 3.0)
    score_matrix = (
        config.instance_feature_ratio * instance_similarity
        + config.semantic_feature_ratio * semantic_similarity
        + config.xyz_feature_ratio * xyz_similarity
    )
    labels = score_matrix.argmax(axis=1).astype(np.int64)
    confidence = score_matrix[np.arange(len(score_matrix)), labels].astype(np.float32)
    labels[confidence <= config.instance_threshold] = -1
    return labels, confidence, len(cluster_ids), score_matrix.astype(
        np.float32, copy=False
    )


def run_class_first(
    point_features: Any,
    point_semantic_features: Any,
    point_xyz: Any,
    label_features: Any,
    classes: Sequence[str],
    priors: Mapping[str, Any],
    config: ClassFirstConfig,
    mode: str,
    scene_scale_m_per_unit: float,
    seed: int = 42,
    valid_mask: Any | None = None,
    selected_classes: Sequence[str] | None = None,
    clusterer_factory: Callable[..., Any] | None = None,
    sor_filter: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
) -> ClassFirstResult:
    """Class-first clustering with no camera, 2-D vote, or legacy overlay state."""

    if mode not in CLASS_FIRST_MODES:
        raise ValueError(f"unknown class-first mode: {mode}")
    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    validate_class_first_priors(priors)
    started = time.perf_counter()
    instance = _normalize_rows(point_features)
    semantic = _normalize_rows(point_semantic_features)
    xyz_units = _as_numpy(point_xyz, np.float64)
    if not (len(instance) == len(semantic) == len(xyz_units)):
        raise ValueError("point arrays have different lengths")
    xyz_m = xyz_units * float(scene_scale_m_per_unit)
    _, _, class_masks = build_class_masks(
        semantic, label_features, classes, config.semantic_threshold, valid_mask
    )
    selected = tuple(classes if selected_classes is None else selected_classes)
    unknown_selected = sorted(set(selected) - set(classes))
    if unknown_selected:
        raise ValueError(f"selected_classes contains unknown classes: {unknown_selected}")
    if len(selected) != len(set(selected)):
        raise ValueError("selected_classes contains duplicates")
    labels = np.full(len(instance), -1, dtype=np.int64)
    confidence = np.zeros(len(instance), dtype=np.float32)
    instances: dict[int, dict[str, Any]] = {}
    classes_diagnostic: dict[str, Any] = {}
    next_instance = 0
    totals = {
        "candidate_points": 0,
        "sampled_points": 0,
        "hdbscan_noise_points": 0,
        "sor_removed_points": 0,
        "rescued_points": 0,
        "final_instances": 0,
        "assigned_points": 0,
    }
    for class_name in selected:
        indices = np.flatnonzero(class_masks[str(class_name)])
        count = len(indices)
        totals["candidate_points"] += count
        diagnostic: dict[str, Any] = {"candidate_points": int(count)}
        parameters = resolve_class_parameters(priors, config, str(class_name), mode)
        diagnostic["parameters"] = parameters
        m_c = int(parameters["min_cluster_size"])
        if count < m_c:
            diagnostic.update(
                {"status": "below_min_cluster_size", "sampled_points": 0, "final_instances": 0}
            )
            classes_diagnostic[str(class_name)] = diagnostic
            continue
        class_instance = instance[indices]
        class_semantic = semantic[indices]
        class_xyz_m = xyz_m[indices]
        uniform_xyz = robust_scale_coordinates(class_xyz_m)
        clustering_xyz = (
            class_xyz_m / max(float(parameters["spatial_scale_m"]), 1e-12)
            if parameters["coordinate_mode"] == "metric_divide_d_c"
            else uniform_xyz
        )
        class_seed = (
            int(seed)
            ^ int.from_bytes(str(class_name).encode("utf-8"), "little")
        ) % (2**31 - 1)
        sampled = sample_class_indices(
            count, m_c, config, np.random.default_rng(class_seed)
        )
        diagnostic["sampled_points"] = int(len(sampled))
        totals["sampled_points"] += len(sampled)
        if len(sampled) < m_c or len(sampled) < config.min_samples:
            diagnostic.update({"status": "insufficient_sample", "final_instances": 0})
            classes_diagnostic[str(class_name)] = diagnostic
            continue
        distance = _hybrid_distance(
            class_instance[sampled],
            class_semantic[sampled],
            clustering_xyz[sampled],
            config,
        )
        sample_labels = _cluster_sample(distance, m_c, config, clusterer_factory)
        sample_noise = int((sample_labels < 0).sum())
        diagnostic["hdbscan_noise_points"] = sample_noise
        totals["hdbscan_noise_points"] += sample_noise
        local_labels, local_confidence, cluster_count, local_score_matrix = (
            _assign_from_centers(
            sample_labels,
            class_instance[sampled],
            class_semantic[sampled],
            uniform_xyz[sampled],
            class_instance,
            class_semantic,
            uniform_xyz,
            config,
            )
        )
        if not cluster_count:
            diagnostic.update({"status": "no_clusters", "final_instances": 0})
            classes_diagnostic[str(class_name)] = diagnostic
            continue
        # Preserve source/refactor's robust-scaled KNN geometry.  Smooth changes
        # only K; size affects only the HDBSCAN coordinates above.
        local_labels = class_local_knn(
            local_labels,
            uniform_xyz,
            int(parameters["knn_k"]),
        )
        sor_removed = 0
        if config.use_sor:
            local_labels, sor_removed = apply_sor_to_clusters(
                local_labels,
                class_xyz_m,
                config.sor_nb_neighbors,
                config.sor_std_ratio,
                sor_filter,
            )
        rescued = 0
        if parameters["rescue_enabled"]:
            local_labels, rescued = rescue_noise_by_anchor(
                local_labels,
                class_xyz_m,
                float(parameters["rescue_radius_m"]),
            )
        diagnostic["sor_removed_points"] = sor_removed
        diagnostic["rescued_points"] = rescued
        totals["sor_removed_points"] += sor_removed
        totals["rescued_points"] += rescued
        local_ids = np.unique(local_labels[local_labels >= 0])
        for local_id in local_ids:
            mask = local_labels == local_id
            point_indices = indices[mask]
            final_confidence = local_score_matrix[mask, int(local_id)]
            score = float(np.clip(np.mean(final_confidence), 0.0, 1.0))
            labels[point_indices] = next_instance
            confidence[point_indices] = final_confidence
            instances[next_instance] = {
                "class": str(class_name),
                "score": score,
                "point_count": int(mask.sum()),
                "mean_assignment_confidence": score,
            }
            next_instance += 1
        assigned = int((local_labels >= 0).sum())
        diagnostic.update(
            {
                "status": "complete",
                "assigned_points": assigned,
                "final_instances": int(len(local_ids)),
            }
        )
        totals["assigned_points"] += assigned
        totals["final_instances"] += len(local_ids)
        classes_diagnostic[str(class_name)] = diagnostic
    totals["coverage"] = float(totals["assigned_points"] / max(len(instance), 1))
    diagnostics = {
        "mode": mode,
        "seed": int(seed),
        "scene_scale_m_per_unit": float(scene_scale_m_per_unit),
        "elapsed_seconds": float(time.perf_counter() - started),
        "totals": totals,
        "classes": classes_diagnostic,
    }
    return ClassFirstResult(labels, confidence, instances, diagnostics)


def build_class_first_metadata(
    result: ClassFirstResult, run: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "kind": "saga_instance_metadata",
        "run": dict(run or {}),
        "instances": {
            str(instance_id): dict(values)
            for instance_id, values in sorted(result.instances.items())
        },
        "class_first": result.diagnostics,
    }


def build_class_first_params(
    priors: Mapping[str, Any], config: ClassFirstConfig, mode: str = "combined"
) -> dict[str, Any]:
    """Make the 20-class d/A/b/m/K/rescue table explicit, without hashes."""

    validate_class_first_priors(priors)
    categories: dict[str, Any] = {}
    for class_name in sorted(priors["categories"]):
        parameters = resolve_class_parameters(priors, config, class_name, mode)
        categories[class_name] = {
            "supported": bool(parameters["supported"]),
            "d_c_m": parameters["d_c_m"],
            "b_c": parameters["b_c"],
            "A_c_m2": parameters["A_c_m2"],
            "m_c": int(parameters["min_cluster_size"]),
            "K_c": int(parameters["knn_k"]),
            "rescue_radius_m": parameters["rescue_radius_m"],
        }
    return {
        "schema_version": "1.0",
        "kind": "class_first_params",
        "mode": mode,
        "global_config": config.as_json(),
        "formulas": {
            "size": "HDBSCAN spatial coordinates = xyz_m / d_c",
            "smooth": "clip(round(256*b_global/max(b_c,0.02)),16,256)",
            "small": "clip(round(10*(A_c/A_global)^p),3,20)",
            "sample_quota": "min(N,5000,max(round(0.03*N),4*m_c))",
            "rescue_radius": "rho*d_c",
        },
        "categories": categories,
    }


def write_class_first_params(
    priors_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    mode: str = "combined",
) -> dict[str, Any]:
    priors = load_json(priors_path)
    config = load_class_first_config(config_path)
    payload = build_class_first_params(priors, config, mode)
    write_json(output_path, payload)
    return payload
