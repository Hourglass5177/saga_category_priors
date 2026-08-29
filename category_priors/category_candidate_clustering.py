from __future__ import annotations

"""Pure instance-formation algorithms for the category candidate experiment.

The historical category branch mixed three independent operations: construction
of a pairwise distance, HDBSCAN, and an all-point centre assignment.  This
module keeps those responsibilities separate so that the clustering mechanism
can be audited without category priors, semantic classes, ground truth, scene
I/O, or final post-processing.

Inputs to every public function are already class-local.  Semantic routing is
therefore a caller responsibility and cannot leak into the instance distance.
All coordinates passed here are metres.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


R1_METRIC_HDBSCAN = "R1-corrected-distance-legacy-expand"
R2_ANCHORED_HDBSCAN = "R2-corrected-distance-anchored-expand"
G1_MUTUAL_LOCAL_GRAPH = "G1-mutual-local-graph"

AFFINITY_WEIGHT = 0.625
SPATIAL_WEIGHT = 0.375
MIN_CLUSTER_SIZE = 3
MIN_SAMPLES = 3
CLUSTER_SELECTION_EPSILON = 0.01
ENVELOPE_QUANTILE = 0.95
PHYSICAL_NEIGHBORS = 24
AFFINITY_NEIGHBORS = 4
DEFAULT_QUERY_CHUNK_SIZE = 1_024
_EPSILON = 1e-8


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _readonly(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _normalise_affinity(value: Any, *, name: str) -> np.ndarray:
    array = _as_numpy(value, np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{name} rows must have non-zero norm")
    return array / norms


def _validate_xyz_m(value: Any, *, row_count: int, name: str) -> np.ndarray:
    xyz = _as_numpy(value, np.float64)
    if xyz.shape != (row_count, 3):
        raise ValueError(f"{name} must have shape ({row_count}, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{name} must be finite")
    return xyz


def _validate_typical_diag(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("global_typical_diag_m must be finite and positive")
    return result


def _validate_weights(affinity_weight: float, spatial_weight: float) -> tuple[float, float]:
    affinity = float(affinity_weight)
    spatial = float(spatial_weight)
    if not np.isfinite(affinity) or not np.isfinite(spatial):
        raise ValueError("distance weights must be finite")
    if affinity < 0 or spatial < 0:
        raise ValueError("distance weights must be non-negative")
    if not np.isclose(affinity + spatial, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("distance weights must sum to one")
    return affinity, spatial


def _cross_corrected_distance(
    left_affinity: np.ndarray,
    left_xyz_m: np.ndarray,
    right_affinity: np.ndarray,
    right_xyz_m: np.ndarray,
    global_typical_diag_m: float,
    affinity_weight: float,
    spatial_weight: float,
) -> np.ndarray:
    cosine = np.clip(left_affinity @ right_affinity.T, -1.0, 1.0)
    affinity_distance = np.arccos(cosine) / np.pi
    delta = left_xyz_m[:, None, :] - right_xyz_m[None, :, :]
    spatial_distance = np.sqrt(np.sum(delta * delta, axis=2))
    spatial_distance = np.minimum(spatial_distance / global_typical_diag_m, 1.0)
    distance = affinity_weight * affinity_distance + spatial_weight * spatial_distance
    if not np.isfinite(distance).all():
        raise ValueError("corrected distance produced non-finite values")
    return np.clip(distance, 0.0, 1.0)


def corrected_pairwise_distance(
    affinity_features: Any,
    xyz_m: Any,
    global_typical_diag_m: float,
    *,
    affinity_weight: float = AFFINITY_WEIGHT,
    spatial_weight: float = SPATIAL_WEIGHT,
) -> np.ndarray:
    """Return the registered angular-affinity plus physical-space distance.

    Unlike the legacy matrix, this metric has a strict zero diagonal, does not
    use a sample-dependent maximum, and does not contain semantic confidence.
    ``global_typical_diag_m`` is one frozen global physical scale shared by all
    classes.
    """

    features = _normalise_affinity(affinity_features, name="affinity_features")
    xyz = _validate_xyz_m(xyz_m, row_count=len(features), name="xyz_m")
    typical_diag = _validate_typical_diag(global_typical_diag_m)
    affinity, spatial = _validate_weights(affinity_weight, spatial_weight)
    result = _cross_corrected_distance(
        features,
        xyz,
        features,
        xyz,
        typical_diag,
        affinity,
        spatial,
    )
    # The explicit symmetrisation protects the precomputed-metric contract
    # against insignificant BLAS asymmetry.  HDBSCAN requires an exact zero
    # diagonal.
    result = (result + result.T) * 0.5
    np.fill_diagonal(result, 0.0)
    return _readonly(result, np.float64)


@dataclass(frozen=True)
class MetricHDBSCANConfig:
    min_cluster_size: int = MIN_CLUSTER_SIZE
    min_samples: int = MIN_SAMPLES
    cluster_selection_epsilon: float = CLUSTER_SELECTION_EPSILON
    allow_single_cluster: bool = False
    affinity_weight: float = AFFINITY_WEIGHT
    spatial_weight: float = SPATIAL_WEIGHT

    def validate(self) -> None:
        if int(self.min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be at least two")
        if int(self.min_samples) < 1:
            raise ValueError("min_samples must be positive")
        epsilon = float(self.cluster_selection_epsilon)
        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("cluster_selection_epsilon must be finite and non-negative")
        _validate_weights(self.affinity_weight, self.spatial_weight)


@dataclass(frozen=True)
class RawClusterResult:
    """Raw sampled HDBSCAN result before any full-point expansion."""

    labels: np.ndarray
    membership: np.ndarray
    distance_matrix: np.ndarray
    raw_cluster_ids: tuple[int, ...]
    diagnostics: Mapping[str, Any]


def _default_hdbscan_factory(**kwargs: Any) -> Any:
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:  # pragma: no cover - environment specific
        raise RuntimeError("metric HDBSCAN requires the hdbscan package") from exc
    return HDBSCAN(**kwargs)


def cluster_metric_hdbscan(
    affinity_features: Any,
    xyz_m: Any,
    global_typical_diag_m: float,
    *,
    config: MetricHDBSCANConfig = MetricHDBSCANConfig(),
    hdbscan_factory: Callable[..., Any] | None = None,
) -> RawClusterResult:
    """Cluster one already-routed class using the corrected precomputed metric."""

    config.validate()
    features = _normalise_affinity(affinity_features, name="affinity_features")
    xyz = _validate_xyz_m(xyz_m, row_count=len(features), name="xyz_m")
    distance = corrected_pairwise_distance(
        features,
        xyz,
        global_typical_diag_m,
        affinity_weight=config.affinity_weight,
        spatial_weight=config.spatial_weight,
    )
    distance_contract = {
        "distance_finite": bool(np.isfinite(distance).all()),
        "distance_symmetry_max_abs": float(
            np.max(np.abs(distance - distance.T)) if distance.size else 0.0
        ),
        "distance_diagonal_max_abs": float(
            np.max(np.abs(np.diag(distance))) if distance.size else 0.0
        ),
        "distance_min": float(np.min(distance) if distance.size else 0.0),
        "distance_max": float(np.max(distance) if distance.size else 0.0),
    }
    count = len(features)
    if count < int(config.min_cluster_size):
        labels = np.full(count, -1, dtype=np.int64)
        membership = np.zeros(count, dtype=np.float64)
        return RawClusterResult(
            labels=_readonly(labels),
            membership=_readonly(membership),
            distance_matrix=distance,
            raw_cluster_ids=(),
            diagnostics={
                "method": R1_METRIC_HDBSCAN,
                "hdbscan_ran": False,
                "sample_count": int(count),
                "raw_cluster_count": 0,
                "raw_member_count": 0,
                "noise_point_count": int(count),
                "min_cluster_size": int(config.min_cluster_size),
                "min_samples": int(config.min_samples),
                "cluster_selection_epsilon": float(config.cluster_selection_epsilon),
                "affinity_weight": float(config.affinity_weight),
                "spatial_weight": float(config.spatial_weight),
                "global_typical_diag_m": float(global_typical_diag_m),
                **distance_contract,
            },
        )

    factory = hdbscan_factory or _default_hdbscan_factory
    clusterer = factory(
        min_cluster_size=int(config.min_cluster_size),
        min_samples=int(config.min_samples),
        cluster_selection_epsilon=float(config.cluster_selection_epsilon),
        allow_single_cluster=bool(config.allow_single_cluster),
        metric="precomputed",
    )
    labels = _as_numpy(clusterer.fit_predict(np.asarray(distance)), np.int64)
    if labels.shape != (count,):
        raise ValueError("HDBSCAN returned an invalid label vector")
    if np.any(labels < -1):
        raise ValueError("HDBSCAN labels may only use -1 as a negative value")
    raw_membership = getattr(clusterer, "probabilities_", None)
    if raw_membership is None:
        membership = np.where(labels >= 0, 1.0, 0.0).astype(np.float64)
    else:
        membership = _as_numpy(raw_membership, np.float64)
        if membership.shape != (count,):
            raise ValueError("HDBSCAN returned invalid membership probabilities")
        if not np.isfinite(membership).all() or np.any(
            (membership < 0) | (membership > 1)
        ):
            raise ValueError("HDBSCAN membership must be finite and in [0, 1]")
        membership = np.where(labels >= 0, membership, 0.0)
    raw_ids = tuple(int(value) for value in np.unique(labels) if int(value) >= 0)
    return RawClusterResult(
        labels=_readonly(labels, np.int64),
        membership=_readonly(membership, np.float64),
        distance_matrix=distance,
        raw_cluster_ids=raw_ids,
        diagnostics={
            "method": R1_METRIC_HDBSCAN,
            "hdbscan_ran": True,
            "sample_count": int(count),
            "raw_cluster_count": int(len(raw_ids)),
            "raw_member_count": int(np.count_nonzero(labels >= 0)),
            "noise_point_count": int(np.count_nonzero(labels < 0)),
            "min_cluster_size": int(config.min_cluster_size),
            "min_samples": int(config.min_samples),
            "cluster_selection_epsilon": float(config.cluster_selection_epsilon),
            "affinity_weight": float(config.affinity_weight),
            "spatial_weight": float(config.spatial_weight),
            "global_typical_diag_m": float(global_typical_diag_m),
            **distance_contract,
        },
    )


@dataclass(frozen=True)
class AnchoredExpansionConfig:
    envelope_quantile: float = ENVELOPE_QUANTILE
    minimum_candidate_points: int = MIN_CLUSTER_SIZE
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE
    affinity_weight: float = AFFINITY_WEIGHT
    spatial_weight: float = SPATIAL_WEIGHT

    def validate(self) -> None:
        quantile = float(self.envelope_quantile)
        if not np.isfinite(quantile) or not 0 <= quantile <= 1:
            raise ValueError("envelope_quantile must be in [0, 1]")
        if int(self.minimum_candidate_points) < 1:
            raise ValueError("minimum_candidate_points must be positive")
        if int(self.query_chunk_size) < 1:
            raise ValueError("query_chunk_size must be positive")
        _validate_weights(self.affinity_weight, self.spatial_weight)


@dataclass(frozen=True)
class AnchoredExpansionResult:
    """Class-local raw/core/full labels on the complete selected-point axis."""

    raw_cluster_labels_selected: np.ndarray
    trusted_core_labels: np.ndarray
    full_candidate_labels: np.ndarray
    assignment_confidence: np.ndarray
    raw_cluster_ids: tuple[int, ...]
    candidates: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]


def _validate_sample_trace(
    selected_count: int,
    sampled_local_indices: Any,
    raw_cluster_labels: Any,
    raw_membership: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled = _as_numpy(sampled_local_indices, np.int64)
    labels = _as_numpy(raw_cluster_labels, np.int64)
    membership = _as_numpy(raw_membership, np.float64)
    if sampled.ndim != 1 or labels.shape != sampled.shape or membership.shape != sampled.shape:
        raise ValueError("sample indices, labels and membership must share one vector axis")
    if len(sampled) and (int(sampled.min()) < 0 or int(sampled.max()) >= selected_count):
        raise ValueError("sampled_local_indices are outside the selected-point axis")
    if len(np.unique(sampled)) != len(sampled):
        raise ValueError("sampled_local_indices must be unique")
    if np.any(labels < -1):
        raise ValueError("raw_cluster_labels may only use -1 as a negative value")
    if not np.isfinite(membership).all() or np.any((membership < 0) | (membership > 1)):
        raise ValueError("raw_membership must be finite and in [0, 1]")
    membership = np.where(labels >= 0, membership, 0.0)
    return sampled, labels, membership


def _validate_sample_distance(value: Any, sample_count: int) -> np.ndarray:
    distance = _as_numpy(value, np.float64)
    if distance.shape != (sample_count, sample_count):
        raise ValueError("sample_distance_matrix has an invalid shape")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("sample_distance_matrix must be finite and non-negative")
    if not np.allclose(distance, distance.T, rtol=0.0, atol=1e-12):
        raise ValueError("sample_distance_matrix must be symmetric")
    if not np.all(np.diag(distance) == 0):
        raise ValueError("sample_distance_matrix must have a zero diagonal")
    return distance


def expand_anchored_clusters(
    affinity_features: Any,
    xyz_m: Any,
    sampled_local_indices: Any,
    raw_cluster_labels: Any,
    raw_membership: Any,
    global_typical_diag_m: float,
    *,
    sample_distance_matrix: Any | None = None,
    config: AnchoredExpansionConfig = AnchoredExpansionConfig(),
) -> AnchoredExpansionResult:
    """Expand raw clusters from their nearest immutable sampled members.

    The attachment envelope for a raw cluster is the 95th percentile of its
    members' leave-one-out nearest-member distances.  Non-raw points attach
    only when their nearest raw member lies inside that member's cluster
    envelope.  An exact nearest-distance tie across clusters remains
    background.  Raw members are never reassigned.
    """

    config.validate()
    features = _normalise_affinity(affinity_features, name="affinity_features")
    xyz = _validate_xyz_m(xyz_m, row_count=len(features), name="xyz_m")
    typical_diag = _validate_typical_diag(global_typical_diag_m)
    affinity_weight, spatial_weight = _validate_weights(
        config.affinity_weight, config.spatial_weight
    )
    sampled, raw_labels, membership = _validate_sample_trace(
        len(features), sampled_local_indices, raw_cluster_labels, raw_membership
    )
    if sample_distance_matrix is None:
        sample_distance = corrected_pairwise_distance(
            features[sampled],
            xyz[sampled],
            typical_diag,
            affinity_weight=affinity_weight,
            spatial_weight=spatial_weight,
        )
    else:
        sample_distance = _validate_sample_distance(
            sample_distance_matrix, len(sampled)
        )

    all_raw_ids = tuple(int(value) for value in np.unique(raw_labels) if int(value) >= 0)
    retained_raw_ids = tuple(
        raw_id
        for raw_id in all_raw_ids
        if int(np.count_nonzero(raw_labels == raw_id))
        >= int(config.minimum_candidate_points)
    )
    raw_to_candidate = {raw_id: index for index, raw_id in enumerate(retained_raw_ids)}
    selected_count = len(features)
    raw_selected = np.full(selected_count, -1, dtype=np.int64)
    trusted = np.full(selected_count, -1, dtype=np.int64)
    full = np.full(selected_count, -1, dtype=np.int64)
    confidence = np.zeros(selected_count, dtype=np.float64)

    envelope_radius = np.zeros(len(retained_raw_ids), dtype=np.float64)
    anchor_sample_rows: list[int] = []
    anchor_candidate_ids: list[int] = []
    for raw_id, candidate_id in raw_to_candidate.items():
        member_rows = np.flatnonzero(raw_labels == raw_id)
        within = np.array(sample_distance[np.ix_(member_rows, member_rows)], copy=True)
        if len(member_rows) > 1:
            np.fill_diagonal(within, np.inf)
            nearest_other = np.min(within, axis=1)
            radius = float(
                np.quantile(
                    nearest_other,
                    float(config.envelope_quantile),
                    method="linear",
                )
            )
        else:  # Defensive; HDBSCAN should not emit this under the fixed config.
            radius = 0.0
        envelope_radius[candidate_id] = radius
        for sample_row in member_rows:
            selected_row = int(sampled[sample_row])
            raw_selected[selected_row] = candidate_id
            trusted[selected_row] = candidate_id
            full[selected_row] = candidate_id
            confidence[selected_row] = float(membership[sample_row])
            anchor_sample_rows.append(int(sample_row))
            anchor_candidate_ids.append(candidate_id)

    anchor_sample = np.asarray(anchor_sample_rows, dtype=np.int64)
    anchor_candidate = np.asarray(anchor_candidate_ids, dtype=np.int64)
    # Only unsampled semantic points are eligible for expansion.  HDBSCAN
    # sampled noise is an observed rejection, not an ordinary unobserved point
    # that may be recovered by the full-assignment step.
    sampled_mask = np.zeros(selected_count, dtype=bool)
    sampled_mask[sampled] = True
    query_rows = np.flatnonzero(~sampled_mask)
    exact_tie_count = 0
    attached_count = 0
    if len(anchor_sample):
        anchor_selected = sampled[anchor_sample]
        anchor_features = features[anchor_selected]
        anchor_xyz = xyz[anchor_selected]
        for start in range(0, len(query_rows), int(config.query_chunk_size)):
            rows = query_rows[start : start + int(config.query_chunk_size)]
            distance = _cross_corrected_distance(
                features[rows],
                xyz[rows],
                anchor_features,
                anchor_xyz,
                typical_diag,
                affinity_weight,
                spatial_weight,
            )
            best_anchor = np.argmin(distance, axis=1)
            minimum = distance[np.arange(len(rows)), best_anchor]
            best_candidate = anchor_candidate[best_anchor]
            tied = distance == minimum[:, None]
            cross_cluster_tie = np.any(
                tied & (anchor_candidate[None, :] != best_candidate[:, None]), axis=1
            )
            within_envelope = minimum <= envelope_radius[best_candidate]
            attach = ~cross_cluster_tie & within_envelope
            selected_rows = rows[attach]
            selected_candidate = best_candidate[attach]
            selected_distance = minimum[attach]
            selected_radius = envelope_radius[selected_candidate]
            full[selected_rows] = selected_candidate
            confidence[selected_rows] = np.exp(
                -selected_distance / np.maximum(selected_radius, _EPSILON)
            )
            exact_tie_count += int(np.count_nonzero(cross_cluster_tie))
            attached_count += int(np.count_nonzero(attach))

    candidates: list[dict[str, Any]] = []
    for raw_id, candidate_id in raw_to_candidate.items():
        core_mask = trusted == candidate_id
        full_mask = full == candidate_id
        candidates.append(
            {
                "candidate_id": int(candidate_id),
                "raw_cluster_id": int(raw_id),
                "raw_member_count": int(np.count_nonzero(core_mask)),
                "raw_member_retained_count": int(np.count_nonzero(core_mask)),
                "core_point_count": int(np.count_nonzero(core_mask)),
                "full_point_count": int(np.count_nonzero(full_mask)),
                "envelope_radius": float(envelope_radius[candidate_id]),
                "expanded_point_count": int(
                    np.count_nonzero(full_mask) - np.count_nonzero(core_mask)
                ),
                "assignment_confidence_mean": float(
                    np.mean(confidence[full_mask]) if np.any(full_mask) else 0.0
                ),
            }
        )
    core_outside_full = int(np.count_nonzero((trusted >= 0) & (trusted != full)))
    if core_outside_full:
        raise AssertionError("anchored expansion produced core outside full")
    return AnchoredExpansionResult(
        raw_cluster_labels_selected=_readonly(raw_selected, np.int64),
        trusted_core_labels=_readonly(trusted, np.int64),
        full_candidate_labels=_readonly(full, np.int64),
        assignment_confidence=_readonly(confidence, np.float64),
        raw_cluster_ids=retained_raw_ids,
        candidates=tuple(candidates),
        diagnostics={
            "method": R2_ANCHORED_HDBSCAN,
            "selected_point_count": int(selected_count),
            "sampled_point_count": int(len(sampled)),
            "raw_cluster_count": int(len(all_raw_ids)),
            "raw_member_count": int(np.count_nonzero(raw_labels >= 0)),
            "raw_member_retained_count": int(np.count_nonzero(raw_selected >= 0)),
            "candidate_count": int(len(retained_raw_ids)),
            "dropped_raw_cluster_ids": tuple(
                raw_id for raw_id in all_raw_ids if raw_id not in raw_to_candidate
            ),
            "raw_member_reassigned_count": 0,
            "core_outside_full_count": core_outside_full,
            "expanded_point_count": int(attached_count),
            "sampled_noise_point_count": int(np.count_nonzero(raw_labels < 0)),
            "exact_cross_cluster_tie_count": int(exact_tie_count),
            "envelope_quantile": float(config.envelope_quantile),
            "query_chunk_size": int(config.query_chunk_size),
            "affinity_weight": float(affinity_weight),
            "spatial_weight": float(spatial_weight),
            "global_typical_diag_m": float(typical_diag),
        },
    )


@dataclass(frozen=True)
class MutualGraphConfig:
    physical_neighbors: int = PHYSICAL_NEIGHBORS
    affinity_neighbors: int = AFFINITY_NEIGHBORS
    minimum_candidate_points: int = MIN_CLUSTER_SIZE

    def validate(self) -> None:
        if int(self.physical_neighbors) < 1:
            raise ValueError("physical_neighbors must be positive")
        if int(self.affinity_neighbors) < 1:
            raise ValueError("affinity_neighbors must be positive")
        if int(self.affinity_neighbors) > int(self.physical_neighbors):
            raise ValueError("affinity_neighbors cannot exceed physical_neighbors")
        if int(self.minimum_candidate_points) < 1:
            raise ValueError("minimum_candidate_points must be positive")


@dataclass(frozen=True)
class MutualGraphResult:
    full_candidate_labels: np.ndarray
    trusted_core_labels: np.ndarray
    assignment_confidence: np.ndarray
    edges: np.ndarray
    edge_similarity: np.ndarray
    candidates: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]


def build_mutual_local_graph(
    affinity_features: Any,
    xyz_m: Any,
    *,
    config: MutualGraphConfig = MutualGraphConfig(),
) -> MutualGraphResult:
    """Build deterministic mutual-top4 components inside physical 24-NN.

    This is a class-local fallback, not a learned graph and not a candidate
    rescue path.  Components below the fixed minimum point count remain
    background.
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    config.validate()
    features = _normalise_affinity(affinity_features, name="affinity_features")
    xyz = _validate_xyz_m(xyz_m, row_count=len(features), name="xyz_m")
    count = len(features)
    empty_edges = np.empty((0, 2), dtype=np.int64)
    if count == 0:
        empty = np.empty(0, dtype=np.int64)
        return MutualGraphResult(
            full_candidate_labels=_readonly(empty),
            trusted_core_labels=_readonly(empty),
            assignment_confidence=_readonly(np.empty(0, dtype=np.float64)),
            edges=_readonly(empty_edges),
            edge_similarity=_readonly(np.empty(0, dtype=np.float64)),
            candidates=(),
            diagnostics={
                "method": G1_MUTUAL_LOCAL_GRAPH,
                "point_count": 0,
                "edge_count": 0,
                "candidate_count": 0,
                "raw_member_count": 0,
                "raw_member_retained_count": 0,
                "physical_neighbors_effective": 0,
                "affinity_neighbors_effective": 0,
            },
        )
    physical_k = min(int(config.physical_neighbors), count - 1)
    affinity_k = min(int(config.affinity_neighbors), physical_k)
    if physical_k == 0:
        labels = np.full(count, -1, dtype=np.int64)
        return MutualGraphResult(
            full_candidate_labels=_readonly(labels),
            trusted_core_labels=_readonly(labels),
            assignment_confidence=_readonly(np.zeros(count, dtype=np.float64)),
            edges=_readonly(empty_edges),
            edge_similarity=_readonly(np.empty(0, dtype=np.float64)),
            candidates=(),
            diagnostics={
                "method": G1_MUTUAL_LOCAL_GRAPH,
                "point_count": int(count),
                "edge_count": 0,
                "candidate_count": 0,
                "raw_member_count": 0,
                "raw_member_retained_count": 0,
                "physical_neighbors_effective": 0,
                "affinity_neighbors_effective": 0,
            },
        )

    query_k = min(count, physical_k + 1)
    physical_distance, physical_neighbor = cKDTree(xyz).query(xyz, k=query_k)
    physical_distance = np.asarray(physical_distance, dtype=np.float64)
    physical_neighbor = np.asarray(physical_neighbor, dtype=np.int64)
    if physical_distance.ndim == 1:
        physical_distance = physical_distance[:, None]
        physical_neighbor = physical_neighbor[:, None]
    row_ids = np.arange(count, dtype=np.int64)[:, None]
    physical_distance = physical_distance.copy()
    physical_distance[physical_neighbor == row_ids] = np.inf
    # Distance is primary and point ID is the deterministic tie-break.
    physical_order = np.lexsort((physical_neighbor, physical_distance), axis=1)
    physical_neighbor = np.take_along_axis(
        physical_neighbor, physical_order[:, :physical_k], axis=1
    )

    similarities = np.einsum(
        "nd,nkd->nk", features, features[physical_neighbor], optimize=True
    )
    # Affinity is primary and point ID is the deterministic tie-break.
    affinity_order = np.lexsort((physical_neighbor, -similarities), axis=1)
    selected = np.take_along_axis(
        physical_neighbor, affinity_order[:, :affinity_k], axis=1
    )
    selected_similarity = np.take_along_axis(
        similarities, affinity_order[:, :affinity_k], axis=1
    )

    sources = np.repeat(np.arange(count, dtype=np.int64), affinity_k)
    targets = selected.reshape(-1)
    directed_similarity = selected_similarity.reshape(-1)
    codes = sources * count + targets
    reverse_codes = targets * count + sources
    sorted_codes = np.sort(codes)
    positions = np.searchsorted(sorted_codes, reverse_codes)
    mutual = positions < len(sorted_codes)
    mutual[mutual] &= sorted_codes[positions[mutual]] == reverse_codes[mutual]
    keep = mutual & (sources < targets)
    edges = np.column_stack((sources[keep], targets[keep])).astype(np.int64)
    edge_similarity = directed_similarity[keep].astype(np.float64)

    if len(edges):
        adjacency = coo_matrix(
            (
                np.ones(len(edges), dtype=np.uint8),
                (edges[:, 0], edges[:, 1]),
            ),
            shape=(count, count),
        )
        _, component = connected_components(adjacency, directed=False)
        component = np.asarray(component, dtype=np.int64)
    else:
        component = np.arange(count, dtype=np.int64)

    full = np.full(count, -1, dtype=np.int64)
    confidence = np.zeros(count, dtype=np.float64)
    components = []
    for component_id in np.unique(component):
        members = np.flatnonzero(component == component_id)
        if len(members) >= int(config.minimum_candidate_points):
            components.append(members)
    components.sort(key=lambda members: int(members[0]))
    candidate_rows: list[dict[str, Any]] = []
    for candidate_id, members in enumerate(components):
        full[members] = candidate_id
        # G1 has a hard graph-membership decision but no registered assignment
        # probability.  Use one for retained core points instead of silently
        # introducing affinity cosine as a new scoring factor.  Edge affinity
        # remains available as a separate diagnostic below.
        confidence[members] = 1.0
        member_mask = np.zeros(count, dtype=bool)
        member_mask[members] = True
        internal_edge = member_mask[edges[:, 0]] & member_mask[edges[:, 1]]
        candidate_rows.append(
            {
                "candidate_id": int(candidate_id),
                "core_point_count": int(len(members)),
                "full_point_count": int(len(members)),
                "minimum_selected_point_index": int(members[0]),
                "assignment_confidence_mean": float(np.mean(confidence[members])),
                "internal_edge_similarity_mean": float(
                    np.mean(edge_similarity[internal_edge])
                    if np.any(internal_edge)
                    else 0.0
                ),
            }
        )
    confidence[full < 0] = 0.0
    return MutualGraphResult(
        full_candidate_labels=_readonly(full, np.int64),
        trusted_core_labels=_readonly(full, np.int64),
        assignment_confidence=_readonly(confidence, np.float64),
        edges=_readonly(edges, np.int64),
        edge_similarity=_readonly(edge_similarity, np.float64),
        candidates=tuple(candidate_rows),
        diagnostics={
            "method": G1_MUTUAL_LOCAL_GRAPH,
            "point_count": int(count),
            "edge_count": int(len(edges)),
            "candidate_count": int(len(candidate_rows)),
            "raw_member_count": 0,
            "raw_member_retained_count": 0,
            "background_point_count": int(np.count_nonzero(full < 0)),
            "physical_neighbors": int(config.physical_neighbors),
            "affinity_neighbors": int(config.affinity_neighbors),
            "physical_neighbors_effective": int(physical_k),
            "affinity_neighbors_effective": int(affinity_k),
            "minimum_candidate_points": int(config.minimum_candidate_points),
        },
    )


__all__ = [
    "AFFINITY_NEIGHBORS",
    "AFFINITY_WEIGHT",
    "AnchoredExpansionConfig",
    "AnchoredExpansionResult",
    "CLUSTER_SELECTION_EPSILON",
    "DEFAULT_QUERY_CHUNK_SIZE",
    "ENVELOPE_QUANTILE",
    "G1_MUTUAL_LOCAL_GRAPH",
    "MIN_CLUSTER_SIZE",
    "MIN_SAMPLES",
    "MetricHDBSCANConfig",
    "MutualGraphConfig",
    "MutualGraphResult",
    "PHYSICAL_NEIGHBORS",
    "R1_METRIC_HDBSCAN",
    "R2_ANCHORED_HDBSCAN",
    "RawClusterResult",
    "SPATIAL_WEIGHT",
    "build_mutual_local_graph",
    "cluster_metric_hdbscan",
    "corrected_pairwise_distance",
    "expand_anchored_clusters",
]
