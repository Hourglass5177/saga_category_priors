from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODES = ("uniform", "size", "smooth", "small", "combined")
BOUNDARY_RADII_M = (0.02, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class LegacyPriorConfig:
    kind: str = "legacy_prior_config"
    semantic_threshold: float = 0.2
    alpha: float = 0.05
    boundary_beta: float = 0.10
    min_samples: int = 3
    sample_fraction: float = 0.03
    sample_cap: int = 5000
    support_multiplier: int = 4
    min_cluster_low: int = 3
    min_cluster_high: int = 20
    knn_max: int = 64
    halo_neighbors: int = 8
    halo_min_agreement: int = 3
    assignment_threshold: float = 0.25

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def load_legacy_prior_config(path: str | Path) -> LegacyPriorConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("legacy-prior config must be a JSON object")
    known = set(LegacyPriorConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown legacy-prior config fields: {unknown}")
    config = LegacyPriorConfig(**payload)
    if config.kind != "legacy_prior_config":
        raise ValueError("legacy-prior config has the wrong kind")
    if not 0.0 <= config.semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be in [0, 1]")
    if config.alpha <= 0 or config.boundary_beta <= 0:
        raise ValueError("alpha and boundary_beta must be positive")
    if config.min_samples <= 0 or config.sample_cap <= 0:
        raise ValueError("min_samples and sample_cap must be positive")
    if config.halo_min_agreement > config.halo_neighbors:
        raise ValueError("halo_min_agreement cannot exceed halo_neighbors")
    return config


def _q50(node: Mapping[str, Any], section: str, field: str) -> float:
    return float(node["shrunk"][section][field]["q50"])


def category_geometry(
    priors: Mapping[str, Any], class_name: str
) -> dict[str, float] | None:
    node = priors.get("categories", {}).get(class_name)
    if not isinstance(node, Mapping):
        return None
    return {
        "d_c_m": math.exp(_q50(node, "geometry", "log_bbox_diag_m")),
        "A_c_m2": math.exp(_q50(node, "geometry", "log_surface_area_m2")),
    }


def choose_smoothing_radius(
    priors: Mapping[str, Any], class_name: str, beta: float
) -> float | None:
    node = priors.get("categories", {}).get(class_name)
    if not isinstance(node, Mapping):
        return None
    neighborhood = node["shrunk"]["neighborhood"]
    eligible = [
        radius
        for radius in BOUNDARY_RADII_M
        if float(neighborhood[f"boundary_fixed:{radius:.2f}"]["q50"]) <= beta
    ]
    return max(eligible) if eligible else min(BOUNDARY_RADII_M)


def empirical_scale_quantile(mask_scales: Sequence[float], physical_size_m: float) -> float:
    values = np.asarray(mask_scales, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if not len(values):
        return 1.0
    return float(np.clip(np.mean(values <= physical_size_m), 0.0, 1.0))


def estimate_surface_density(xyz_m: np.ndarray, sample_cap: int = 50000) -> float:
    from scipy.spatial import cKDTree

    points = np.asarray(xyz_m, dtype=np.float64)
    if len(points) < 17:
        return 1.0
    if len(points) > sample_cap:
        indices = np.linspace(0, len(points) - 1, sample_cap, dtype=np.int64)
        query = points[indices]
    else:
        query = points
    distances, _ = cKDTree(points).query(query, k=17, workers=-1)
    radius = float(np.median(distances[:, -1]))
    return 16.0 / max(math.pi * radius * radius, 1e-12)


def resolve_class_parameters(
    priors: Mapping[str, Any],
    config: LegacyPriorConfig,
    class_name: str,
    mode: str,
    candidate_count: int,
    surface_density: float,
    mask_scales: Sequence[float],
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown legacy-prior mode: {mode}")
    geometry = category_geometry(priors, class_name)
    supported = geometry is not None
    use_size = supported and mode in {"size", "combined"}
    use_smooth = supported and mode in {"smooth", "combined"}
    use_small = supported and mode in {"small", "combined"}
    d_c = float(geometry["d_c_m"]) if supported else None
    area = float(geometry["A_c_m2"]) if supported else None
    if use_small:
        expected_support = area * max(surface_density, 0.0)
        # Sampling support and the sampled-domain cluster threshold depend on
        # one another.  Two fixed-point updates are sufficient because m_c is
        # clipped to a small integer interval.
        m_c = config.min_cluster_low
        for _ in range(2):
            provisional_sample_count = min(
                candidate_count,
                config.sample_cap,
                max(
                    round(config.sample_fraction * candidate_count),
                    config.support_multiplier * m_c,
                ),
            )
            m_c = int(
                np.clip(
                    round(
                        config.alpha
                        * expected_support
                        * provisional_sample_count
                        / max(candidate_count, 1)
                    ),
                    config.min_cluster_low,
                    config.min_cluster_high,
                )
            )
    else:
        m_c = 5
    sample_count = min(
        candidate_count,
        config.sample_cap,
        max(round(config.sample_fraction * candidate_count), config.support_multiplier * m_c),
    )
    radius = (
        choose_smoothing_radius(priors, class_name, config.boundary_beta)
        if use_smooth
        else None
    )
    return {
        "supported": supported,
        "semantic_threshold": config.semantic_threshold,
        "min_cluster_size": m_c,
        "min_samples": config.min_samples,
        "sample_count": int(sample_count),
        "spatial_scale_m": d_c if use_size else None,
        "scale_gate_input": empirical_scale_quantile(mask_scales, d_c)
        if use_size
        else 1.0,
        "smoothing_radius_m": radius,
        "knn_max": config.knn_max,
        "rescue_enabled": bool(use_small),
        "rescue_radius_m": radius if radius is not None else (0.2 * d_c if use_small else None),
        "halo_neighbors": config.halo_neighbors,
        "halo_min_agreement": config.halo_min_agreement,
        "assignment_threshold": config.assignment_threshold,
        "d_c_m": d_c,
        "A_c_m2": area,
    }


def radius_vote_labels(
    labels: np.ndarray,
    xyz_m: np.ndarray,
    radius_m: float | None,
    k_max: int,
) -> np.ndarray:
    """Smooth assigned labels without allowing noise to outvote an instance."""
    from scipy.spatial import cKDTree

    source = np.asarray(labels, dtype=np.int64).copy()
    result = source.copy()
    if radius_m is None or len(result) < 2:
        return result
    tree = cKDTree(np.asarray(xyz_m, dtype=np.float64))
    neighborhoods = tree.query_ball_point(xyz_m, radius_m, workers=-1)
    for index, neighbors in enumerate(neighborhoods):
        neighbor_indices = np.asarray(neighbors, dtype=np.int64)
        if len(neighbor_indices) > k_max:
            delta = np.asarray(xyz_m)[neighbor_indices] - np.asarray(xyz_m)[index]
            order = np.argsort(np.einsum("ij,ij->i", delta, delta), kind="stable")
            neighbor_indices = neighbor_indices[order[:k_max]]
        valid = source[neighbor_indices]
        valid = valid[valid >= 0]
        if len(valid):
            values, counts = np.unique(valid, return_counts=True)
            result[index] = values[int(np.argmax(counts))]
    return result


def rescue_halo(
    labels: np.ndarray,
    xyz_m: np.ndarray,
    radius_m: float | None,
    neighbors: int,
    min_agreement: int,
) -> tuple[np.ndarray, int]:
    """Recover noise from several nearby anchors; never use a single nearest anchor."""
    from scipy.spatial import cKDTree

    result = np.asarray(labels, dtype=np.int64).copy()
    if radius_m is None:
        return result, 0
    anchors = np.flatnonzero(result >= 0)
    noise = np.flatnonzero(result < 0)
    if not len(anchors) or not len(noise):
        return result, 0
    k = min(neighbors, len(anchors))
    distances, indices = cKDTree(np.asarray(xyz_m)[anchors]).query(
        np.asarray(xyz_m)[noise], k=k, workers=-1
    )
    distances = np.asarray(distances).reshape(len(noise), k)
    indices = np.asarray(indices).reshape(len(noise), k)
    recovered = 0
    for row, point_index in enumerate(noise):
        valid = distances[row] <= radius_m
        votes = result[anchors[indices[row, valid]]]
        if len(votes) < min_agreement:
            continue
        values, counts = np.unique(votes, return_counts=True)
        best = int(np.argmax(counts))
        if int(counts[best]) >= min_agreement:
            result[point_index] = int(values[best])
            recovered += 1
    return result, recovered


def write_legacy_prior_params(
    priors: Mapping[str, Any],
    config: LegacyPriorConfig,
    output: str | Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.0",
        "kind": "legacy_prior_params",
        "config": config.as_json(),
        "categories": {
            name: {
                "geometry": category_geometry(priors, name),
                "smoothing_radius_m": choose_smoothing_radius(
                    priors, name, config.boundary_beta
                ),
            }
            for name in sorted(priors.get("categories", {}))
        },
    }
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
