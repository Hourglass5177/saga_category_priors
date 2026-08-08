from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import hash_json, load_json, sha256_file, write_json
from .mapping import load_mapping_config
from .priors import validate_priors

MODE_FACTORS = {
    "off": (False, False, False),
    "global": (False, False, False),
    "size": (True, False, False),
    "smooth": (False, True, False),
    "small": (False, False, True),
    "size-smooth": (True, True, False),
    "size-small": (True, False, True),
    "smooth-small": (False, True, True),
    "combined": (True, True, True),
}


@dataclass
class OverlayResult:
    labels: Any
    assignment_confidence: Any
    fallback_labels: Any
    fallback_assignment_confidence: Any
    branch_instances: dict[int, dict[str, Any]]
    diagnostics: dict[str, Any]


class PriorResolver:
    def __init__(self, priors: Mapping[str, Any], mapping: Mapping[str, Any]):
        validate_priors(priors)
        self.priors = dict(priors)
        self.mapping = dict(mapping)

    @classmethod
    def from_paths(
        cls, priors_path: str | Path, mapping_path: str | Path
    ) -> PriorResolver:
        priors = load_json(priors_path)
        validate_priors(priors)
        mapping = load_mapping_config(mapping_path)
        expected = mapping["provenance"]["category_priors_sha256"]
        actual = sha256_file(priors_path)
        if expected != actual:
            raise ValueError(
                "Mapping config was tuned for a different category_priors.json"
            )
        return cls(priors, mapping)

    def _node(self, class_name: str) -> Mapping[str, Any] | None:
        node = self.priors["categories"].get(class_name)
        return node if node and node.get("active") else None

    def _stats(
        self, class_name: str, use_shrink: bool
    ) -> tuple[Mapping[str, Any], float, bool]:
        node = self._node(class_name)
        if node is None:
            return self.priors["global"]["shrunk"], 0.0, False
        key = "shrunk" if use_shrink else "raw"
        return node[key], float(node["reliability"]), True

    @staticmethod
    def _median(
        stats: Mapping[str, Any],
        section: str,
        metric: str,
        fallback: Mapping[str, Any] | None = None,
    ) -> float:
        try:
            return float(stats[section][metric]["q50"])
        except (KeyError, TypeError, ValueError):
            if fallback is None:
                raise ValueError(f"Prior statistic is missing: {section}.{metric}")
            try:
                return float(fallback[section][metric]["q50"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Global prior statistic is missing: {section}.{metric}"
                ) from exc

    def class_parameters(
        self,
        class_name: str,
        mode: str,
        semantic_similarity: float,
        surface_density: float,
        sample_fraction: float,
        gate_enabled: bool = True,
        shrink_enabled: bool = True,
    ) -> dict[str, Any]:
        if mode not in MODE_FACTORS:
            raise ValueError(f"Unknown prior mode: {mode}")
        size_on, smooth_on, small_on = MODE_FACTORS[mode]
        category_stats, reliability, active = self._stats(class_name, shrink_enabled)
        global_stats = self.priors["global"]["shrunk"]
        baseline = self.mapping["baseline"]
        coefficients = self.mapping["coefficients"]
        fixed = self.mapping["fixed"]
        threshold = float(baseline["semantic_threshold"])
        semantic_confidence = float(
            np.clip(
                (semantic_similarity - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
            )
        )
        gate = (
            (reliability * semantic_confidence)
            if gate_enabled
            else (1.0 if active else 0.0)
        )

        global_diag = math.exp(
            self._median(global_stats, "geometry", "log_bbox_diag_m")
        )
        class_diag = math.exp(
            self._median(category_stats, "geometry", "log_bbox_diag_m", global_stats)
        )
        spatial_target = float(coefficients["alpha_r"]) * class_diag
        spatial_scale = (
            (1.0 - gate) * global_diag + gate * spatial_target
            if size_on
            else global_diag
        )

        class_node = self.priors["categories"].get(class_name, {})
        small_score = float(class_node.get("small_score", 0.0)) if active else 0.0
        class_area = math.exp(
            self._median(
                category_stats, "geometry", "log_surface_area_m2", global_stats
            )
        )
        expected_support = max(class_area * surface_density * sample_fraction, 0.0)
        low_m, high_m = (int(value) for value in fixed["min_cluster_bounds"])
        target_min_cluster = int(
            np.clip(
                round(
                    float(coefficients["alpha_m"])
                    * expected_support ** float(coefficients["support_exponent"])
                    * (1.0 - float(fixed["small_cluster_discount"]) * small_score)
                ),
                low_m,
                high_m,
            )
        )
        min_cluster = (
            round(
                (1.0 - gate) * int(baseline["min_cluster_size"])
                + gate * target_min_cluster
            )
            if small_on
            else int(baseline["min_cluster_size"])
        )

        global_consistency = self._median(
            global_stats, "neighborhood", "same_instance_relative:0.05"
        )
        class_consistency = self._median(
            category_stats,
            "neighborhood",
            "same_instance_relative:0.05",
            global_stats,
        )
        ratio_low, ratio_high = (
            float(value) for value in fixed["consistency_ratio_bounds"]
        )
        consistency_ratio = float(
            np.clip(
                class_consistency / max(global_consistency, 1e-6), ratio_low, ratio_high
            )
        )
        if smooth_on and gate > 1e-8:
            target_radius = (
                float(coefficients["alpha_k"]) * class_diag * consistency_ratio
            )
            low_k, high_k = (int(value) for value in fixed["knn_k_bounds"])
            target_k = int(
                np.clip(
                    round(surface_density * math.pi * target_radius**2),
                    low_k,
                    high_k,
                )
            )
            knn_k = int(
                np.clip(
                    round((1.0 - gate) * int(baseline["knn_k"]) + gate * target_k),
                    low_k,
                    high_k,
                )
            )
            # A weak gate relaxes the category radius toward the unrestricted
            # legacy neighborhood; gate=1 uses the mapped category radius.
            radius = target_radius / gate
        else:
            radius = math.inf
            knn_k = int(baseline["knn_k"])

        return {
            "class": class_name,
            "active": active,
            "reliability": reliability,
            "semantic_similarity": float(semantic_similarity),
            "semantic_confidence": semantic_confidence,
            "gate": float(gate),
            "spatial_scale_m": float(max(spatial_scale, 1e-6)),
            "expected_sample_support": float(expected_support),
            "min_cluster_size": min_cluster,
            "knn_radius_m": radius,
            "knn_k": knn_k,
            "small_score": small_score,
            "same_instance_relative_005": class_consistency,
            "factors": {"size": size_on, "smooth": smooth_on, "small": small_on},
        }


def allocate_class_quotas(
    candidate_counts: Mapping[str, int],
    small_scores: Mapping[str, float],
    total_budget: int,
    minimum: int,
    maximum: int,
) -> dict[str, int]:
    active = {
        name: int(count) for name, count in candidate_counts.items() if int(count) > 0
    }
    if not active:
        return {}
    minimums = {name: min(count, minimum) for name, count in active.items()}
    if sum(minimums.values()) >= total_budget:
        order = sorted(active, key=lambda name: (-small_scores.get(name, 0.0), name))
        quotas = {name: 0 for name in active}
        remaining = total_budget
        for name in order:
            quota = min(active[name], minimum, remaining)
            quotas[name] = quota
            remaining -= quota
            if remaining <= 0:
                break
        return quotas
    weights = {
        name: math.sqrt(count) * (1.0 + float(small_scores.get(name, 0.0)))
        for name, count in active.items()
    }
    quotas = dict(minimums)
    remaining = total_budget - sum(quotas.values())
    while remaining > 0:
        eligible = [
            name for name in active if quotas[name] < min(active[name], maximum)
        ]
        if not eligible:
            break
        total_weight = sum(weights[name] for name in eligible)
        progress = 0
        for name in sorted(eligible):
            share = max(
                1, math.floor(remaining * weights[name] / max(total_weight, 1e-12))
            )
            capacity = min(active[name], maximum) - quotas[name]
            addition = min(share, capacity, remaining)
            if addition:
                quotas[name] += addition
                remaining -= addition
                progress += addition
            if remaining <= 0:
                break
        if progress == 0:
            break
    return quotas


def estimate_surface_density(
    points_m: np.ndarray, k: int = 16, sample_cap: int = 50_000, seed: int = 42
) -> float:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Surface density estimation requires scipy") from exc
    if len(points_m) <= k:
        return 0.0
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(
        len(points_m), size=min(len(points_m), sample_cap), replace=False
    )
    tree = cKDTree(points_m)
    distances, _ = tree.query(points_m[sample_indices], k=k + 1, workers=-1)
    kth = distances[:, -1]
    valid = np.isfinite(kth) & (kth > 0)
    if not np.any(valid):
        return 0.0
    return float(np.median(k / (math.pi * kth[valid] ** 2)))


def _softmax_max(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    labels = probabilities.argmax(axis=1)
    return probabilities[np.arange(len(probabilities)), labels], labels


def apply_prior_overlay(
    point_features: Any,
    point_semantic_features: Any,
    point_xyz: Any,
    label_features: Any,
    classes: Sequence[str],
    fallback_labels: Any,
    fallback_assignment_confidence: Any,
    resolver: PriorResolver,
    mode: str,
    scene_scale_m_per_unit: float,
    seed: int,
    gate_enabled: bool,
    shrink_enabled: bool,
) -> OverlayResult:
    import torch
    from hdbscan import HDBSCAN

    started = time.perf_counter()
    features = point_features.detach().cpu().numpy().astype(np.float32, copy=False)
    semantic = (
        point_semantic_features.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    xyz_m = point_xyz.detach().cpu().numpy().astype(np.float64, copy=False) * float(
        scene_scale_m_per_unit
    )
    labels_matrix = label_features.detach().cpu().numpy().astype(np.float32, copy=False)
    similarities = semantic @ labels_matrix.T
    top_class = similarities.argmax(axis=1)
    top_similarity = similarities[np.arange(len(similarities)), top_class]
    threshold = float(resolver.mapping["baseline"]["semantic_threshold"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    candidate_masks: dict[str, np.ndarray] = {}
    candidate_counts: dict[str, int] = {}
    small_scores: dict[str, float] = {}
    if mode not in MODE_FACTORS:
        raise ValueError(f"Unknown prior mode: {mode}")
    _, _, small_on = MODE_FACTORS[mode]
    for class_name, node in resolver.priors["categories"].items():
        class_index = class_to_index.get(class_name)
        if class_index is None or not node.get("active"):
            continue
        mask = (top_class == class_index) & (top_similarity >= threshold)
        count = int(mask.sum())
        if count:
            candidate_masks[class_name] = mask
            candidate_counts[class_name] = count
            mean_similarity = float(np.mean(top_similarity[mask]))
            semantic_confidence = float(
                np.clip(
                    (mean_similarity - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
                )
            )
            quota_gate = (
                float(node.get("reliability", 0.0)) * semantic_confidence
                if gate_enabled
                else 1.0
            )
            small_scores[class_name] = (
                float(node.get("small_score", 0.0)) * quota_gate if small_on else 0.0
            )

    fixed = resolver.mapping["fixed"]
    quotas = allocate_class_quotas(
        candidate_counts,
        small_scores,
        int(fixed["total_sample_budget"]),
        int(fixed["class_sample_min"]),
        int(fixed["class_sample_max"]),
    )
    density = estimate_surface_density(
        xyz_m,
        int(fixed["density_k"]),
        int(fixed["density_sample_cap"]),
        seed,
    )
    result_labels = fallback_labels.detach().cpu().numpy().astype(np.int64, copy=True)
    result_confidence = (
        fallback_assignment_confidence.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    next_instance_id = (
        int(result_labels[result_labels >= 0].max() + 1)
        if np.any(result_labels >= 0)
        else 0
    )
    branch_instances: dict[int, dict[str, Any]] = {}
    class_diagnostics: dict[str, Any] = {}

    for class_name in sorted(candidate_masks):
        mask = candidate_masks[class_name]
        indices = np.flatnonzero(mask)
        quota = min(quotas.get(class_name, 0), len(indices))
        mean_similarity = float(np.mean(top_similarity[indices]))
        parameters = resolver.class_parameters(
            class_name,
            mode,
            mean_similarity,
            density,
            quota / max(len(indices), 1),
            gate_enabled,
            shrink_enabled,
        )
        if quota < parameters["min_cluster_size"]:
            class_diagnostics[class_name] = {
                "status": "insufficient_sample",
                "quota": quota,
                **parameters,
            }
            continue
        rng = np.random.default_rng(seed ^ (sum(class_name.encode("utf-8")) << 8))
        sampled_local = np.sort(rng.choice(len(indices), size=quota, replace=False))
        sampled_indices = indices[sampled_local]
        sampled_features = features[sampled_indices]
        sampled_xyz = xyz_m[sampled_indices] / parameters["spatial_scale_m"]
        feature_distance = np.clip(
            1.0 - sampled_features @ sampled_features.T, 0.0, None
        )
        spatial_distance = np.linalg.norm(
            sampled_xyz[:, None, :] - sampled_xyz[None, :, :], axis=-1
        )
        spatial_distance = np.clip(spatial_distance, 0.0, 1.0)
        feature_ratio = float(resolver.mapping["baseline"]["feature_ratio"])
        distance = (
            feature_ratio * feature_distance + (1.0 - feature_ratio) * spatial_distance
        )
        clusterer = HDBSCAN(
            min_cluster_size=int(parameters["min_cluster_size"]),
            cluster_selection_epsilon=float(
                resolver.mapping["baseline"]["cluster_selection_epsilon"]
            ),
            allow_single_cluster=False,
            metric="precomputed",
        )
        cluster_labels = clusterer.fit_predict(distance.astype(np.float64, copy=False))
        valid_cluster_ids = [
            int(value) for value in np.unique(cluster_labels) if value >= 0
        ]
        if not valid_cluster_ids:
            class_diagnostics[class_name] = {
                "status": "no_clusters",
                "quota": quota,
                **parameters,
            }
            continue
        centers_feature = np.stack(
            [
                sampled_features[cluster_labels == cluster_id].mean(axis=0)
                for cluster_id in valid_cluster_ids
            ]
        )
        centers_feature /= np.maximum(
            np.linalg.norm(centers_feature, axis=1, keepdims=True), 1e-12
        )
        centers_xyz = np.stack(
            [
                sampled_xyz[cluster_labels == cluster_id].mean(axis=0)
                for cluster_id in valid_cluster_ids
            ]
        )
        selected_features = features[indices]
        selected_xyz = xyz_m[indices] / parameters["spatial_scale_m"]
        all_assignments: list[np.ndarray] = []
        all_confidences: list[np.ndarray] = []
        for start in range(0, len(indices), 50_000):
            stop = min(start + 50_000, len(indices))
            feature_similarity = np.clip(
                selected_features[start:stop] @ centers_feature.T, -1.0, 1.0
            )
            xyz_similarity = np.exp(
                -np.linalg.norm(
                    selected_xyz[start:stop, None, :] - centers_xyz[None, :, :], axis=-1
                )
            )
            hybrid = (
                feature_ratio * feature_similarity
                + (1.0 - feature_ratio) * xyz_similarity
            )
            confidence, assignment = _softmax_max(hybrid * 10.0)
            all_assignments.append(assignment)
            all_confidences.append(confidence)
        assignments = np.concatenate(all_assignments)
        confidences = np.concatenate(all_confidences)
        valid_assignment = confidences >= float(
            resolver.mapping["baseline"]["instance_threshold"]
        )
        created = 0
        for local_cluster in range(len(valid_cluster_ids)):
            cluster_mask = valid_assignment & (assignments == local_cluster)
            if int(cluster_mask.sum()) < int(parameters["min_cluster_size"]):
                continue
            point_indices = indices[cluster_mask]
            instance_id = next_instance_id
            next_instance_id += 1
            result_labels[point_indices] = instance_id
            result_confidence[point_indices] = confidences[cluster_mask]
            branch_instances[instance_id] = {
                "branch_class": class_name,
                "parameters": parameters,
                "point_count": len(point_indices),
                "mean_assignment_confidence": float(np.mean(confidences[cluster_mask])),
            }
            created += 1
        class_diagnostics[class_name] = {
            "status": "ok" if created else "clusters_rejected",
            "candidate_count": len(indices),
            "quota": quota,
            "created_instances": created,
            **parameters,
        }

    diagnostics = {
        "surface_density_points_per_m2": density,
        "candidate_counts": candidate_counts,
        "gated_small_sampling_scores": small_scores,
        "quotas": quotas,
        "classes": class_diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return OverlayResult(
        labels=torch.from_numpy(result_labels),
        assignment_confidence=torch.from_numpy(result_confidence),
        fallback_labels=fallback_labels.detach().cpu().clone(),
        fallback_assignment_confidence=fallback_assignment_confidence.detach()
        .cpu()
        .clone(),
        branch_instances=branch_instances,
        diagnostics=diagnostics,
    )


def validate_overlay(
    overlay: OverlayResult,
    instance_ratios: Mapping[int, np.ndarray],
    classes: Sequence[str],
    label_threshold: float,
) -> tuple[Any, Any, list[int]]:
    labels = overlay.labels.detach().cpu().numpy().astype(np.int64, copy=True)
    confidence = (
        overlay.assignment_confidence.detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=True)
    )
    fallback_labels = overlay.fallback_labels.detach().cpu().numpy()
    fallback_confidence = overlay.fallback_assignment_confidence.detach().cpu().numpy()
    rejected: list[int] = []
    for instance_id, metadata in overlay.branch_instances.items():
        ratio = instance_ratios.get(instance_id)
        predicted_class = (
            classes[int(np.argmax(ratio))]
            if ratio is not None and len(ratio)
            else "background"
        )
        semantic_confidence = (
            float(np.max(ratio)) if ratio is not None and len(ratio) else 0.0
        )
        if (
            semantic_confidence < label_threshold
            or predicted_class != metadata["branch_class"]
        ):
            mask = labels == instance_id
            labels[mask] = fallback_labels[mask]
            confidence[mask] = fallback_confidence[mask]
            metadata["rejected_by_vote"] = True
            metadata["vote_class"] = predicted_class
            metadata["vote_confidence"] = semantic_confidence
            rejected.append(instance_id)
        else:
            metadata["rejected_by_vote"] = False
            metadata["vote_class"] = predicted_class
            metadata["vote_confidence"] = semantic_confidence
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Overlay validation requires torch") from exc
    return torch.from_numpy(labels), torch.from_numpy(confidence), rejected


def smooth_labels(
    point_xyz: Any,
    point_labels: Any,
    instance_ratios: Mapping[int, np.ndarray],
    classes: Sequence[str],
    resolver: PriorResolver,
    mode: str,
    scene_scale_m_per_unit: float,
    surface_density: float,
    gate_enabled: bool,
    shrink_enabled: bool,
    semantic_similarity_by_instance: Mapping[int, float],
    chunk_size: int = 20_000,
) -> Any:
    try:
        import torch
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Class-conditioned smoothing requires scipy and torch"
        ) from exc
    xyz_m = point_xyz.detach().cpu().numpy().astype(np.float64, copy=False) * float(
        scene_scale_m_per_unit
    )
    labels = point_labels.detach().cpu().numpy().astype(np.int64, copy=False)
    tree = cKDTree(xyz_m)
    per_instance: dict[int, dict[str, Any]] = {}
    max_k = int(resolver.mapping["baseline"]["knn_k"])
    for instance_id in np.unique(labels):
        if instance_id < 0:
            continue
        ratio = instance_ratios.get(int(instance_id))
        class_name = (
            classes[int(np.argmax(ratio))] if ratio is not None and len(ratio) else ""
        )
        params = resolver.class_parameters(
            class_name,
            mode,
            float(semantic_similarity_by_instance.get(int(instance_id), 0.0)),
            surface_density,
            1.0,
            gate_enabled,
            shrink_enabled,
        )
        per_instance[int(instance_id)] = params
        max_k = max(max_k, int(params["knn_k"]))
    max_k = min(max_k, int(resolver.mapping["fixed"]["knn_k_bounds"][1]))
    max_k = min(max_k, len(labels))
    if max_k <= 0:
        return torch.from_numpy(labels.copy())
    smoothed = labels.copy()
    for start in range(0, len(labels), chunk_size):
        stop = min(start + chunk_size, len(labels))
        distances, indices = tree.query(xyz_m[start:stop], k=max_k, workers=-1)
        if max_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        current_labels = labels[start:stop]
        k_by_row = np.full(
            len(current_labels),
            int(resolver.mapping["baseline"]["knn_k"]),
            dtype=np.int64,
        )
        radius_by_row = np.full(len(current_labels), math.inf, dtype=np.float64)
        for current in np.unique(current_labels):
            params = per_instance.get(current)
            if params:
                rows = current_labels == current
                k_by_row[rows] = min(int(params["knn_k"]), max_k)
                radius_by_row[rows] = float(params["knn_radius_m"])
        smoothed[start:stop] = _majority_neighbor_labels(
            labels,
            current_labels,
            distances,
            indices,
            np.minimum(k_by_row, max_k),
            radius_by_row,
        )
    return torch.from_numpy(smoothed)


def _majority_neighbor_labels(
    all_labels: np.ndarray,
    current_labels: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    k_by_row: np.ndarray,
    radius_by_row: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of per-point ``np.unique(..., return_counts=True)``.

    Encoded ``(row, label)`` pairs let NumPy count all neighbor votes in one
    operation. Sorting of the encoded values preserves the legacy tie break:
    when counts match, the numerically smallest label wins.
    """
    result = current_labels.copy()
    if not len(result):
        return result
    columns = np.arange(indices.shape[1], dtype=np.int64)[None, :]
    valid = (
        (columns < k_by_row[:, None])
        & np.isfinite(distances)
        & (indices < len(all_labels))
        & (distances <= radius_by_row[:, None])
    )
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return result
    neighbor_labels = all_labels[indices[rows, columns]]
    minimum_label = int(all_labels.min())
    maximum_label = int(all_labels.max())
    label_span = maximum_label - minimum_label + 1
    encoded = rows.astype(np.int64) * label_span + neighbor_labels - minimum_label
    unique_encoded, counts = np.unique(encoded, return_counts=True)
    counted_rows = unique_encoded // label_span
    counted_labels = unique_encoded % label_span + minimum_label
    starts = np.r_[0, np.flatnonzero(np.diff(counted_rows)) + 1]
    stops = np.r_[starts[1:], len(counted_rows)]
    maximum_counts = np.maximum.reduceat(counts, starts)
    maximum_per_pair = np.repeat(maximum_counts, stops - starts)
    winning_labels = np.minimum.reduceat(
        np.where(counts == maximum_per_pair, counted_labels, maximum_label + 1),
        starts,
    )
    result[counted_rows[starts]] = winning_labels
    return result


def filter_small_clusters(
    labels: Any,
    branch_instances: Mapping[int, Mapping[str, Any]],
    default_min: int = 10,
) -> Any:
    import torch

    array = labels.detach().cpu().numpy().astype(np.int64, copy=True)
    values, counts = np.unique(array, return_counts=True)
    for instance_id, count in zip(values, counts):
        if instance_id < 0:
            continue
        minimum = int(
            branch_instances.get(int(instance_id), {})
            .get("parameters", {})
            .get("min_cluster_size", default_min)
        )
        if int(count) < minimum:
            array[array == instance_id] = -1
    return torch.from_numpy(array)


def _json_safe_metadata(value: Any) -> Any:
    """Convert runtime diagnostics to strict-JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_instance_metadata(
    labels: Any,
    instance_ratios: Mapping[int, np.ndarray],
    assignment_confidence: Any,
    classes: Sequence[str],
    overlay: OverlayResult | None,
    run_info: Mapping[str, Any],
    include_content_hash: bool = True,
) -> dict[str, Any]:
    label_array = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    confidence = (
        assignment_confidence.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    instances: dict[str, Any] = {}
    for instance_id in np.unique(label_array):
        if instance_id < 0:
            continue
        ratio = instance_ratios.get(int(instance_id))
        semantic_conf = (
            float(np.max(ratio)) if ratio is not None and len(ratio) else 0.0
        )
        class_name = (
            classes[int(np.argmax(ratio))]
            if ratio is not None and len(ratio)
            else "background"
        )
        mask = label_array == instance_id
        assignment_conf = float(np.mean(confidence[mask])) if np.any(mask) else 0.0
        branch = overlay.branch_instances.get(int(instance_id), {}) if overlay else {}
        instances[str(int(instance_id))] = {
            "class": class_name,
            "semantic_confidence": semantic_conf,
            "mean_assignment_confidence": assignment_conf,
            "score": float(np.clip(semantic_conf * assignment_conf, 0.0, 1.0)),
            "point_count": int(mask.sum()),
            "prior": branch,
        }
    payload = _json_safe_metadata({
        "schema_version": "1.0",
        "kind": "saga_instance_metadata",
        "run": dict(run_info),
        "instances": instances,
        "overlay_diagnostics": overlay.diagnostics if overlay else {},
    })
    if include_content_hash:
        payload["content_sha256"] = hash_json(payload)
    return payload


def write_instance_metadata(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)
