from __future__ import annotations

"""Build the registered R0/R1/R2/G1 all-category candidate banks.

This module is the narrow integration layer between the pure class-local
clustering algorithms and the existing :class:`CandidateBank` persistence
contract.  It performs semantic routing once, uses one deterministic sample
prefix per class, and never reads GT or class-specific prior values.  The only
prior value accepted here is the single train-only *global* typical diagonal
used by every class in the corrected physical distance.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_clustering import (
    G1_MUTUAL_LOCAL_GRAPH,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    build_mutual_local_graph,
    cluster_metric_hdbscan,
    expand_anchored_clusters,
)
from .category_denoise import (
    ASSIGNMENT_TEMPERATURE,
    CLUSTER_SELECTION_EPSILON,
    INSTANCE_WEIGHT,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    SAMPLE_CAP,
    SEMANTIC_WEIGHT,
    SPATIAL_WEIGHT,
    CandidateBank,
    _boundary_fixed_ratio_with_tree,
    _default_hdbscan_factory,
    _normalize_rows,
    _readonly,
    _scaled_distance,
    _softmax,
    _validate_bank,
    _validate_names,
    build_candidate_bank,
    normalized_top1_32,
    pca_sorted_extents_m,
    stable_class_seed,
)


R0_LEGACY = "R0-legacy"
CLUSTER_CONDITIONS = (
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    G1_MUTUAL_LOCAL_GRAPH,
)


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


@dataclass(frozen=True)
class ClusterBankFamily:
    """One exact legacy bank and registered corrected candidate arms."""

    banks: Mapping[str, CandidateBank]
    global_typical_diag_m: float
    r0_sample_rank: np.ndarray
    r0_sample_class_index: np.ndarray
    r0_hdbscan_labels: np.ndarray
    r0_hdbscan_membership: np.ndarray
    r0_diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if R0_LEGACY not in self.banks:
            raise ValueError("cluster bank family requires R0-legacy")
        unknown = set(self.banks).difference(CLUSTER_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown cluster conditions: {sorted(unknown)}")
        point_count = self.banks[R0_LEGACY].point_count
        arrays = {
            "r0_sample_rank": np.asarray(self.r0_sample_rank),
            "r0_sample_class_index": np.asarray(self.r0_sample_class_index),
            "r0_hdbscan_labels": np.asarray(self.r0_hdbscan_labels),
            "r0_hdbscan_membership": np.asarray(self.r0_hdbscan_membership),
        }
        if any(value.shape != (point_count,) for value in arrays.values()):
            raise ValueError("R0 raw-audit arrays must share the bank point axis")
        sampled = arrays["r0_sample_rank"] >= 0
        if np.any((arrays["r0_sample_class_index"] >= 0) != sampled):
            raise ValueError("R0 sampled rows must have exactly one class index")
        if np.any(arrays["r0_hdbscan_labels"][~sampled] != -1):
            raise ValueError("unsampled R0 rows must have the noise-label sentinel")
        membership = arrays["r0_hdbscan_membership"]
        if not np.isfinite(membership).all() or np.any(
            (membership < 0.0) | (membership > 1.0)
        ):
            raise ValueError("R0 membership must be finite and in [0, 1]")
        if np.any(membership[~sampled] != 0.0):
            raise ValueError("unsampled R0 rows must have zero membership")


_DETERMINISM_DIAGNOSTIC_KEYS = frozenset(
    {
        "determinism_measured",
        "determinism_measured_this_scene",
        "determinism_contract_verified",
        "determinism_algorithm_contract_reference",
        "determinism_check",
        "determinism_violation_count",
        "determinism_point_violation_count",
        "determinism_candidate_violation_count",
        "determinism_diagnostic_violation_count",
        "determinism_raw_trace_violation_count",
    }
)


def _without_determinism_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in _DETERMINISM_DIAGNOSTIC_KEYS
    }


def _candidate_difference_count(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> int:
    paired = sum(
        dict(left_row) != dict(right_row)
        for left_row, right_row in zip(left, right)
    )
    return int(paired + abs(len(left) - len(right)))


def _pointwise_bank_difference_count(left: CandidateBank, right: CandidateBank) -> int:
    """Count points changed in any persisted bank array.

    Floating evidence is compared exactly: the registered requirement is a
    repeated run with identical pointwise output, not merely a numerically
    close rerun.
    """

    attributes = (
        "global_pre_knn",
        "semantic_top1",
        "semantic_top1_score",
        "branch_full_labels",
        "branch_core_labels",
        "assignment_confidence",
    )
    if left.point_count != right.point_count:
        return max(left.point_count, right.point_count)
    changed = np.zeros(left.point_count, dtype=bool)
    for attribute in attributes:
        first = np.asarray(getattr(left, attribute))
        second = np.asarray(getattr(right, attribute))
        if first.shape != second.shape or first.shape != (left.point_count,):
            return max(left.point_count, right.point_count)
        changed |= first != second
    return int(np.count_nonzero(changed))


def measure_cluster_family_determinism(
    family: ClusterBankFamily, repeated: ClusterBankFamily
) -> ClusterBankFamily:
    """Attach measured repeatability diagnostics to a cluster-bank family.

    Both families must be independently reconstructed from the same immutable
    inputs.  This function does not stamp a successful default: every emitted
    condition receives counts measured from persisted point arrays, candidate
    metadata, structural diagnostics and (for R0) the raw HDBSCAN trace.
    """

    if tuple(family.banks) != tuple(repeated.banks):
        raise ValueError("determinism rerun used a different condition set")
    if family.global_typical_diag_m != repeated.global_typical_diag_m:
        raise ValueError("determinism rerun used a different global metric scale")

    raw_trace_violation = 0
    for attribute in (
        "r0_sample_rank",
        "r0_sample_class_index",
        "r0_hdbscan_labels",
        "r0_hdbscan_membership",
    ):
        first = np.asarray(getattr(family, attribute))
        second = np.asarray(getattr(repeated, attribute))
        if first.shape != second.shape:
            raw_trace_violation += max(first.size, second.size)
        else:
            raw_trace_violation += int(np.count_nonzero(first != second))
    if _without_determinism_diagnostics(
        family.r0_diagnostics
    ) != _without_determinism_diagnostics(repeated.r0_diagnostics):
        raw_trace_violation += 1

    measured_banks: dict[str, CandidateBank] = {}
    for condition, bank in family.banks.items():
        rerun = repeated.banks[condition]
        point_violations = _pointwise_bank_difference_count(bank, rerun)
        candidate_violations = _candidate_difference_count(
            bank.candidates, rerun.candidates
        )
        diagnostic_violations = int(
            _without_determinism_diagnostics(bank.diagnostics)
            != _without_determinism_diagnostics(rerun.diagnostics)
        )
        condition_raw_violations = (
            int(raw_trace_violation) if condition == R0_LEGACY else 0
        )
        total = (
            point_violations
            + candidate_violations
            + diagnostic_violations
            + condition_raw_violations
        )
        diagnostics = {
            **dict(bank.diagnostics),
            "determinism_measured": True,
            "determinism_measured_this_scene": True,
            "determinism_contract_verified": total == 0,
            "determinism_check": "independent-in-process-family-rebuild",
            "determinism_violation_count": int(total),
            "determinism_point_violation_count": int(point_violations),
            "determinism_candidate_violation_count": int(candidate_violations),
            "determinism_diagnostic_violation_count": int(
                diagnostic_violations
            ),
            "determinism_raw_trace_violation_count": int(
                condition_raw_violations
            ),
        }
        measured_banks[condition] = replace(bank, diagnostics=diagnostics)

    return replace(
        family,
        banks=measured_banks,
        r0_diagnostics={
            **dict(family.r0_diagnostics),
            "determinism_measured": True,
            "determinism_measured_this_scene": True,
            "determinism_contract_verified": raw_trace_violation == 0,
            "determinism_check": "independent-in-process-family-rebuild",
            "determinism_violation_count": int(raw_trace_violation),
        },
    )


def save_cluster_raw_audit(
    family: ClusterBankFamily, destination: str | Path
) -> Path:
    """Persist the standalone R0 raw-clustering audit sidecar (not a cache)."""

    path = Path(destination)
    if path.suffix.lower() != ".npz":
        path = path / "r0_raw_trace.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray("saga-category-cluster-r0-raw-audit-v2"),
            global_typical_diag_m=np.asarray(family.global_typical_diag_m),
            sample_rank=family.r0_sample_rank,
            sample_class_index=family.r0_sample_class_index,
            hdbscan_labels=family.r0_hdbscan_labels,
            hdbscan_membership=family.r0_hdbscan_membership,
            diagnostics_json=np.asarray(
                json.dumps(
                    dict(family.r0_diagnostics),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )
    return path


def load_cluster_raw_audit(path: str | Path) -> dict[str, Any]:
    location = Path(path)
    if location.is_dir():
        location = location / "r0_raw_trace.npz"
    with np.load(location, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]).item())
        if schema != "saga-category-cluster-r0-raw-audit-v2":
            raise ValueError(f"unsupported cluster raw audit schema: {schema}")
        result = {
            "schema": schema,
            "global_typical_diag_m": float(
                np.asarray(archive["global_typical_diag_m"]).item()
            ),
            "sample_rank": np.asarray(archive["sample_rank"], dtype=np.int64),
            "sample_class_index": np.asarray(
                archive["sample_class_index"], dtype=np.int64
            ),
            "hdbscan_labels": np.asarray(
                archive["hdbscan_labels"], dtype=np.int64
            ),
            "hdbscan_membership": np.asarray(
                archive["hdbscan_membership"], dtype=np.float64
            ),
            "diagnostics": json.loads(str(archive["diagnostics_json"].item())),
        }
    shapes = {
        np.asarray(result[name]).shape
        for name in (
            "sample_rank",
            "sample_class_index",
            "hdbscan_labels",
            "hdbscan_membership",
        )
    }
    if len(shapes) != 1:
        raise ValueError("R0 raw-audit arrays do not share one point axis")
    sampled = result["sample_rank"] >= 0
    if np.any((result["sample_class_index"] >= 0) != sampled):
        raise ValueError("R0 raw-audit sampled/class sentinels disagree")
    if np.any(result["hdbscan_labels"][~sampled] != -1):
        raise ValueError("R0 raw-audit has labels on unsampled points")
    membership = result["hdbscan_membership"]
    if not np.isfinite(membership).all() or np.any(
        (membership < 0.0) | (membership > 1.0)
    ):
        raise ValueError("R0 raw-audit membership is invalid")
    return result


def _measure_distance_matrix(distance: np.ndarray) -> dict[str, Any]:
    """Return measured, not declarative, precomputed-metric diagnostics."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("HDBSCAN distance must be a square matrix")
    return {
        "distance_finite": bool(np.isfinite(matrix).all()),
        "distance_symmetry_max_abs": float(
            np.max(np.abs(matrix - matrix.T)) if matrix.size else 0.0
        ),
        "distance_diagonal_max_abs": float(
            np.max(np.abs(np.diag(matrix))) if matrix.size else 0.0
        ),
        "distance_min": float(np.min(matrix) if matrix.size else 0.0),
        "distance_max": float(np.max(matrix) if matrix.size else 0.0),
    }


def _aggregate_distance_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]], *, require_zero_diagonal: bool
) -> dict[str, Any]:
    """Aggregate per-class measured matrices without claiming unmeasured facts."""

    rows = tuple(diagnostics)
    result = {
        "distance_matrix_count": int(len(rows)),
        "distance_all_finite": bool(
            all(bool(row.get("distance_finite", False)) for row in rows)
        ),
        "distance_symmetry_max_abs": float(
            max(
                (float(row.get("distance_symmetry_max_abs", np.inf)) for row in rows),
                default=0.0,
            )
        ),
        "distance_diagonal_max_abs": float(
            max(
                (float(row.get("distance_diagonal_max_abs", np.inf)) for row in rows),
                default=0.0,
            )
        ),
        "distance_min": float(
            min((float(row.get("distance_min", np.inf)) for row in rows), default=0.0)
        ),
        "distance_max": float(
            max((float(row.get("distance_max", -np.inf)) for row in rows), default=0.0)
        ),
    }
    result["distance_contract_passed"] = bool(
        bool(rows)
        and result["distance_all_finite"]
        and result["distance_symmetry_max_abs"] <= 1e-12
        and result["distance_min"] >= -1e-12
        and (
            not require_zero_diagonal
            or result["distance_diagonal_max_abs"] <= 1e-12
        )
    )
    return result


def _legacy_raw_hdbscan(
    sampled_features: np.ndarray,
    sampled_standardized_xyz: np.ndarray,
    sampled_scores: np.ndarray,
    *,
    hdbscan_factory: Callable[..., Any] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reproduce the section-30 sampled raw labels for the Stage-0 sidecar."""

    from scipy.spatial.distance import cdist

    instance_distance = np.maximum(
        1.0 - sampled_features @ sampled_features.T, 0.0
    )
    spatial_distance = cdist(
        sampled_standardized_xyz,
        sampled_standardized_xyz,
        metric="euclidean",
    )
    semantic_distance = np.clip(
        1.0 - np.outer(sampled_scores, sampled_scores), 0.0, 1.0
    )
    hybrid = (
        INSTANCE_WEIGHT * _scaled_distance(instance_distance)
        + SPATIAL_WEIGHT * _scaled_distance(spatial_distance)
        + SEMANTIC_WEIGHT * semantic_distance
    )
    distance_diagnostics = _measure_distance_matrix(hybrid)
    factory = hdbscan_factory or _default_hdbscan_factory
    clusterer = factory(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        allow_single_cluster=False,
        metric="precomputed",
    )
    labels = np.asarray(clusterer.fit_predict(hybrid), dtype=np.int64)
    if labels.shape != (len(sampled_features),):
        raise ValueError("legacy HDBSCAN returned an invalid label vector")
    probabilities = getattr(clusterer, "probabilities_", None)
    membership = (
        np.where(labels >= 0, 1.0, 0.0).astype(np.float64)
        if probabilities is None
        else np.asarray(probabilities, dtype=np.float64)
    )
    if membership.shape != labels.shape or not np.isfinite(membership).all():
        raise ValueError("legacy HDBSCAN returned invalid membership")
    membership = np.where(labels >= 0, membership, 0.0)
    return labels, membership, {
        "sample_count": int(len(labels)),
        "raw_cluster_count": int(len(np.unique(labels[labels >= 0]))),
        "raw_member_count": int(np.count_nonzero(labels >= 0)),
        "noise_point_count": int(np.count_nonzero(labels < 0)),
        "min_cluster_size": int(MIN_CLUSTER_SIZE),
        "min_samples": int(MIN_SAMPLES),
        "cluster_selection_epsilon": float(CLUSTER_SELECTION_EPSILON),
        **distance_diagnostics,
    }


@dataclass(frozen=True)
class _ClassConditionResult:
    full: np.ndarray
    core: np.ndarray
    confidence: np.ndarray
    candidates: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]


def _legacy_center_expansion(
    selected_features: np.ndarray,
    selected_standardized_xyz: np.ndarray,
    sampled_local: np.ndarray,
    raw_labels: np.ndarray,
) -> _ClassConditionResult:
    """Apply the historical mean-centre assignment to corrected raw labels."""

    from scipy.spatial.distance import cdist

    raw_ids = tuple(int(value) for value in np.unique(raw_labels) if value >= 0)
    count = len(selected_features)
    full = np.full(count, -1, dtype=np.int64)
    core = np.full(count, -1, dtype=np.int64)
    confidence = np.zeros(count, dtype=np.float64)
    if not raw_ids:
        return _ClassConditionResult(
            full=full,
            core=core,
            confidence=confidence,
            candidates=(),
            diagnostics={
                "method": R1_METRIC_HDBSCAN,
                "raw_cluster_count": 0,
                "candidate_count": 0,
                "raw_member_count": 0,
                "raw_member_retained_count": 0,
                "raw_member_reassigned_count": 0,
                "core_outside_full_count": 0,
            },
        )

    sampled_features = selected_features[sampled_local]
    sampled_xyz = selected_standardized_xyz[sampled_local]
    feature_centres: list[np.ndarray] = []
    xyz_centres: list[np.ndarray] = []
    for raw_id in raw_ids:
        mask = raw_labels == raw_id
        feature_centres.append(
            _normalize_rows(sampled_features[mask].mean(axis=0, keepdims=True))[0]
        )
        xyz_centres.append(sampled_xyz[mask].mean(axis=0))
    feature_centres_array = np.asarray(feature_centres, dtype=np.float64)
    xyz_centres_array = np.asarray(xyz_centres, dtype=np.float64)
    feature_similarity = np.clip(
        selected_features @ feature_centres_array.T, -1.0, 1.0
    )
    xyz_similarity = np.exp(
        -cdist(selected_standardized_xyz, xyz_centres_array, metric="euclidean")
    )
    hybrid_similarity = 0.5 * feature_similarity + 0.5 * xyz_similarity
    probability = _softmax(hybrid_similarity * ASSIGNMENT_TEMPERATURE)
    assigned_centre = np.argmax(probability, axis=1)
    assigned_confidence = probability[np.arange(count), assigned_centre]
    assigned_centre[assigned_confidence < 0.3] = -1

    rows: list[dict[str, Any]] = []
    raw_member_count = int(np.count_nonzero(raw_labels >= 0))
    retained_raw_member_count = 0
    for centre_index, raw_id in enumerate(raw_ids):
        full_mask = assigned_centre == centre_index
        if int(np.count_nonzero(full_mask)) < MIN_CLUSTER_SIZE:
            continue
        candidate_id = len(rows)
        raw_member_rows = sampled_local[raw_labels == raw_id]
        full[full_mask] = candidate_id
        core[raw_member_rows] = candidate_id
        confidence[full_mask] = assigned_confidence[full_mask]
        retained_raw_member_count += int(
            np.count_nonzero(full[raw_member_rows] == candidate_id)
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "raw_cluster_id": int(raw_id),
                "raw_member_count": int(len(raw_member_rows)),
                "core_point_count": int(len(raw_member_rows)),
                "full_point_count": int(np.count_nonzero(full_mask)),
                "assignment_confidence_mean": float(
                    assigned_confidence[full_mask].mean()
                ),
                "expansion_kind": "legacy-normalized-mean-centre-softmax",
            }
        )
    core_outside = int(np.count_nonzero((core >= 0) & (core != full)))
    return _ClassConditionResult(
        full=full,
        core=core,
        confidence=confidence,
        candidates=tuple(rows),
        diagnostics={
            "method": R1_METRIC_HDBSCAN,
            "raw_cluster_count": len(raw_ids),
            "candidate_count": len(rows),
            "raw_member_count": int(raw_member_count),
            "raw_member_retained_count": int(retained_raw_member_count),
            "raw_member_reassigned_count": int(
                raw_member_count - retained_raw_member_count
            ),
            "core_outside_full_count": core_outside,
        },
    )


def _result_from_anchored(value: Any) -> _ClassConditionResult:
    return _ClassConditionResult(
        full=np.asarray(value.full_candidate_labels, dtype=np.int64),
        core=np.asarray(value.trusted_core_labels, dtype=np.int64),
        confidence=np.asarray(value.assignment_confidence, dtype=np.float64),
        candidates=tuple(dict(row) for row in value.candidates),
        diagnostics=dict(value.diagnostics),
    )


def _result_from_graph(value: Any) -> _ClassConditionResult:
    return _ClassConditionResult(
        full=np.asarray(value.full_candidate_labels, dtype=np.int64),
        core=np.asarray(value.trusted_core_labels, dtype=np.int64),
        confidence=np.asarray(value.assignment_confidence, dtype=np.float64),
        candidates=tuple(dict(row) for row in value.candidates),
        diagnostics=dict(value.diagnostics),
    )


def build_cluster_bank_family(
    instance_features: Any,
    semantic_features: Any,
    xyz_scene: Any,
    label_features: Any,
    class_names: Sequence[str],
    saga20_names: Sequence[str],
    global_pre_knn: Any,
    scene_scale_m_per_unit: float,
    global_typical_diag_m: float,
    *,
    scene_id: str,
    seed: int = 42,
    conditions: Sequence[str] = (
        R0_LEGACY,
        R1_METRIC_HDBSCAN,
        R2_ANCHORED_HDBSCAN,
    ),
    hdbscan_factory: Callable[..., Any] | None = None,
) -> ClusterBankFamily:
    """Build registered clustering arms from one immutable semantic routing.

    ``R0`` calls the pre-existing legacy builder verbatim.  R1 and R2 share
    corrected sampled HDBSCAN labels; only their full-point expansion differs.
    G1 is constructed only when explicitly requested by the stage controller.
    """

    from scipy.spatial import cKDTree

    requested = tuple(dict.fromkeys(map(str, conditions)))
    if R0_LEGACY not in requested:
        raise ValueError("R0-legacy must be included for identity and safety gates")
    unknown = set(requested).difference(CLUSTER_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown cluster conditions: {sorted(unknown)}")
    scale = float(scene_scale_m_per_unit)
    global_diag = float(global_typical_diag_m)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    if not np.isfinite(global_diag) or global_diag <= 0:
        raise ValueError("global_typical_diag_m must be finite and positive")
    if not str(scene_id):
        raise ValueError("scene_id must not be empty")

    classes, branches = _validate_names(class_names, saga20_names)
    instance = _as_numpy(instance_features, np.float64)
    semantic = _as_numpy(semantic_features, np.float64)
    xyz = _as_numpy(xyz_scene, np.float64)
    global_labels = _as_numpy(global_pre_knn, np.int64)
    count = len(xyz)
    if (
        instance.ndim != 2
        or semantic.ndim != 2
        or xyz.shape != (count, 3)
        or global_labels.shape != (count,)
        or len(instance) != count
        or len(semantic) != count
    ):
        raise ValueError("features, xyz_scene and global_pre_knn must share a point axis")
    if not np.isfinite(instance).all() or not np.isfinite(xyz).all():
        raise ValueError("instance features and xyz_scene must be finite")
    if np.any(global_labels < -1):
        raise ValueError("global_pre_knn may only use -1 as its negative label")

    top1 = normalized_top1_32(
        semantic, label_features, classes, branches, threshold=0.7
    )
    normed_instance = _normalize_rows(instance)
    xyz_m = xyz * scale
    minimum = xyz.min(axis=0)
    span = xyz.max(axis=0) - minimum
    standardized_xyz = (xyz - minimum) / np.where(span > 0, span, 1.0)

    # R0 is deliberately not reconstructed through the new helpers.  This is
    # the byte-level behavioural anchor for the previous frozen candidate bank.
    legacy = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        branches,
        global_labels,
        scale,
        seed=seed,
        hdbscan_factory=hdbscan_factory,
    )
    legacy.diagnostics["scene_id"] = str(scene_id)
    legacy.diagnostics["candidate_cluster_condition"] = R0_LEGACY
    legacy_core = np.asarray(legacy.branch_core_labels, dtype=np.int64)
    legacy_full = np.asarray(legacy.branch_full_labels, dtype=np.int64)

    nonlegacy = tuple(condition for condition in requested if condition != R0_LEGACY)
    full_by_condition = {
        condition: np.full(count, -1, dtype=np.int64) for condition in nonlegacy
    }
    core_by_condition = {
        condition: np.full(count, -1, dtype=np.int64) for condition in nonlegacy
    }
    confidence_by_condition = {
        condition: np.zeros(count, dtype=np.float64) for condition in nonlegacy
    }
    rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in nonlegacy
    }
    class_diagnostics: dict[str, dict[str, dict[str, Any]]] = {
        condition: {} for condition in nonlegacy
    }
    next_candidate = {condition: 0 for condition in nonlegacy}
    r0_sample_rank = np.full(count, -1, dtype=np.int64)
    r0_sample_class_index = np.full(count, -1, dtype=np.int64)
    r0_hdbscan_labels = np.full(count, -1, dtype=np.int64)
    r0_hdbscan_membership = np.zeros(count, dtype=np.float64)
    r0_class_diagnostics: dict[str, dict[str, Any]] = {}

    for class_name in sorted(branches):
        class_index = classes.index(class_name)
        selected_indices = np.flatnonzero(
            top1.eligible_mask & (top1.branch_class_index == class_index)
        )
        selected_count = len(selected_indices)
        for condition in nonlegacy:
            class_diagnostics[condition][class_name] = {
                "class_index": class_index,
                "selected_points": selected_count,
                "sampled_points": 0,
                "candidate_count": 0,
            }
        if selected_count < MIN_CLUSTER_SIZE:
            continue
        selected_features = normed_instance[selected_indices]
        selected_xyz_m = xyz_m[selected_indices]
        selected_standardized_xyz = standardized_xyz[selected_indices]
        sample_count = min(selected_count, SAMPLE_CAP)
        rng = np.random.default_rng(stable_class_seed(seed, class_name))
        sampled_local = rng.permutation(selected_count)[:sample_count]
        sampled_global = selected_indices[sampled_local]
        r0_sample_rank[sampled_global] = np.arange(sample_count, dtype=np.int64)
        r0_sample_class_index[sampled_global] = int(class_index)
        (
            legacy_raw_labels,
            legacy_raw_membership,
            legacy_raw_diagnostics,
        ) = _legacy_raw_hdbscan(
            selected_features[sampled_local],
            selected_standardized_xyz[sampled_local],
            np.asarray(top1.top_score[selected_indices][sampled_local], dtype=np.float64),
            hdbscan_factory=hdbscan_factory,
        )
        r0_hdbscan_labels[sampled_global] = legacy_raw_labels
        r0_hdbscan_membership[sampled_global] = legacy_raw_membership
        legacy_expansion_audit = _legacy_center_expansion(
            selected_features,
            selected_standardized_xyz,
            sampled_local,
            legacy_raw_labels,
        )
        r0_class_diagnostics[class_name] = {
            "class_index": int(class_index),
            "selected_points": int(selected_count),
            "sampled_points": int(sample_count),
            **legacy_raw_diagnostics,
            "raw_member_retained_count": int(
                legacy_expansion_audit.diagnostics["raw_member_retained_count"]
            ),
            "raw_member_reassigned_count": int(
                legacy_expansion_audit.diagnostics["raw_member_reassigned_count"]
            ),
            "core_outside_full_count": int(
                legacy_expansion_audit.diagnostics["core_outside_full_count"]
            ),
            "candidate_count": int(
                legacy_expansion_audit.diagnostics["candidate_count"]
            ),
        }

        need_hdbscan = any(
            condition in {R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN}
            for condition in nonlegacy
        )
        raw = None
        if need_hdbscan:
            raw = cluster_metric_hdbscan(
                selected_features[sampled_local],
                selected_xyz_m[sampled_local],
                global_diag,
                hdbscan_factory=hdbscan_factory,
            )

        results: dict[str, _ClassConditionResult] = {}
        if R1_METRIC_HDBSCAN in nonlegacy:
            assert raw is not None
            results[R1_METRIC_HDBSCAN] = _legacy_center_expansion(
                selected_features,
                selected_standardized_xyz,
                sampled_local,
                np.asarray(raw.labels, dtype=np.int64),
            )
        if R2_ANCHORED_HDBSCAN in nonlegacy:
            assert raw is not None
            results[R2_ANCHORED_HDBSCAN] = _result_from_anchored(
                expand_anchored_clusters(
                    selected_features,
                    selected_xyz_m,
                    sampled_local,
                    raw.labels,
                    raw.membership,
                    global_diag,
                    sample_distance_matrix=raw.distance_matrix,
                )
            )
        if G1_MUTUAL_LOCAL_GRAPH in nonlegacy:
            results[G1_MUTUAL_LOCAL_GRAPH] = _result_from_graph(
                build_mutual_local_graph(selected_features, selected_xyz_m)
            )

        for condition, result in results.items():
            offset = next_candidate[condition]
            valid_full = result.full >= 0
            valid_core = result.core >= 0
            full_by_condition[condition][selected_indices[valid_full]] = (
                result.full[valid_full] + offset
            )
            core_by_condition[condition][selected_indices[valid_core]] = (
                result.core[valid_core] + offset
            )
            confidence_by_condition[condition][selected_indices] = result.confidence
            for source_row in result.candidates:
                local_id = int(source_row["candidate_id"])
                candidate_id = offset + local_id
                full_indices = selected_indices[result.full == local_id]
                core_indices = selected_indices[result.core == local_id]
                row = dict(source_row)
                if "minimum_selected_point_index" in row:
                    local_minimum = int(row.pop("minimum_selected_point_index"))
                    row["minimum_class_local_index"] = local_minimum
                    row["minimum_global_point_index"] = int(
                        selected_indices[local_minimum]
                    )
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "branch_class": class_name,
                        "branch_class_index": class_index,
                        "semantic_selected_point_count": selected_count,
                        "sampled_point_count": sample_count,
                        "core_point_count": int(len(core_indices)),
                        "trusted_core_point_count": int(len(core_indices)),
                        "full_point_count": int(len(full_indices)),
                        "metric_extents_m": pca_sorted_extents_m(
                            xyz[full_indices], scale
                        ).tolist(),
                    }
                )
                rows_by_condition[condition].append(row)
            next_candidate[condition] += len(result.candidates)
            diagnostics = dict(result.diagnostics)
            if raw is not None and condition in {
                R1_METRIC_HDBSCAN,
                R2_ANCHORED_HDBSCAN,
            }:
                diagnostics["raw_hdbscan"] = dict(raw.diagnostics)
            if condition == R2_ANCHORED_HDBSCAN:
                # The pure R2 result counts every raw member, including raw
                # clusters that could be dropped before candidate emission.
                # Do not reconstruct this denominator from candidate rows.
                diagnostics["raw_member_count"] = int(
                    result.diagnostics["raw_member_count"]
                )
                diagnostics["raw_member_retained_count"] = int(
                    result.diagnostics["raw_member_retained_count"]
                )
                diagnostics["raw_member_reassigned_count"] = int(
                    result.diagnostics["raw_member_reassigned_count"]
                )
                diagnostics["core_outside_full_count"] = int(
                    result.diagnostics["core_outside_full_count"]
                )
            elif condition == G1_MUTUAL_LOCAL_GRAPH:
                diagnostics["raw_member_count"] = 0
                diagnostics["raw_member_retained_count"] = 0
                diagnostics["raw_member_reassigned_count"] = 0
                diagnostics["core_outside_full_count"] = 0
            diagnostics["sampled_points"] = int(sample_count)
            diagnostics["candidate_count"] = len(result.candidates)
            class_diagnostics[condition][class_name].update(diagnostics)

    r0_distance = _aggregate_distance_diagnostics(
        tuple(r0_class_diagnostics.values()), require_zero_diagonal=False
    )
    r0_raw_member_count = int(
        sum(int(row["raw_member_count"]) for row in r0_class_diagnostics.values())
    )
    r0_retained_count = int(
        sum(
            int(row["raw_member_retained_count"])
            for row in r0_class_diagnostics.values()
        )
    )
    r0_core_outside = int(
        sum(
            int(row["core_outside_full_count"])
            for row in r0_class_diagnostics.values()
        )
    )
    legacy.diagnostics.update(
        {
            "raw_member_count": r0_raw_member_count,
            "raw_member_retained_count": r0_retained_count,
            "raw_member_reassigned_count": (
                r0_raw_member_count - r0_retained_count
            ),
            "core_outside_full_count": r0_core_outside,
            "exported_core_point_count": int(np.count_nonzero(legacy_core >= 0)),
            "exported_core_in_full_count": int(
                np.count_nonzero((legacy_core >= 0) & (legacy_core == legacy_full))
            ),
            "r0_raw_class_diagnostics": r0_class_diagnostics,
            "r0_legacy_distance_measurements": r0_distance,
            "orphan_count": 0,
            "negative_metadata_count": 0,
            "determinism_measured": False,
            "determinism_measured_this_scene": False,
            "determinism_contract_verified": False,
        }
    )

    boundary_tree = cKDTree(xyz_m)
    banks: dict[str, CandidateBank] = {R0_LEGACY: legacy}
    for condition in nonlegacy:
        rows: list[dict[str, Any]] = []
        for source_row in rows_by_condition[condition]:
            row = dict(source_row)
            candidate_id = int(row["candidate_id"])
            row["boundary_ratio_5cm"] = _boundary_fixed_ratio_with_tree(
                xyz_m, full_by_condition[condition] == candidate_id, boundary_tree
            )
            rows.append(row)
        if condition == R1_METRIC_HDBSCAN:
            raw_member_count = int(
                sum(
                    int(values.get("raw_member_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
            raw_member_retained_count = int(
                sum(
                    int(values.get("raw_member_retained_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
            core_outside_full_count = int(
                sum(
                    int(values.get("core_outside_full_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
        elif condition == R2_ANCHORED_HDBSCAN:
            raw_member_count = int(
                sum(
                    int(values.get("raw_member_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
            raw_member_retained_count = int(
                sum(
                    int(values.get("raw_member_retained_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
            core_outside_full_count = int(
                sum(
                    int(values.get("core_outside_full_count", 0))
                    for values in class_diagnostics[condition].values()
                )
            )
        else:
            # G1 has no HDBSCAN raw-member concept.  Its components are hard
            # graph cores and must not be mislabeled as retained raw members.
            raw_member_count = 0
            raw_member_retained_count = 0
            core_outside_full_count = 0
        if condition in {R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN}:
            distance_diagnostics = _aggregate_distance_diagnostics(
                tuple(
                    values["raw_hdbscan"]
                    for values in class_diagnostics[condition].values()
                    if "raw_hdbscan" in values
                ),
                require_zero_diagonal=True,
            )
            distance_diagnostics["corrected_distance_contract_passed"] = bool(
                distance_diagnostics["distance_contract_passed"]
                and distance_diagnostics["distance_matrix_count"] > 0
            )
            distance_diagnostics["corrected_distance_contract_measured"] = bool(
                distance_diagnostics["distance_matrix_count"] > 0
            )
            if (
                distance_diagnostics["corrected_distance_contract_measured"]
                and not distance_diagnostics["corrected_distance_contract_passed"]
            ):
                raise AssertionError(
                    f"{condition} violated the measured corrected-distance contract"
                )
        else:
            distance_diagnostics = {
                "distance_matrix_count": 0,
                "corrected_distance_contract_passed": True,
                "corrected_distance_contract_measured": False,
            }
        bank = CandidateBank(
            class_names=classes,
            saga20_names=tuple(sorted(branches)),
            scene_scale_m_per_unit=scale,
            seed=int(seed),
            global_pre_knn=_readonly(global_labels.astype(np.int64, copy=True)),
            semantic_top1=_readonly(top1.top_class_index.astype(np.int64, copy=True)),
            semantic_top1_score=_readonly(top1.top_score.astype(np.float64, copy=True)),
            branch_full_labels=_readonly(full_by_condition[condition]),
            branch_core_labels=_readonly(core_by_condition[condition]),
            assignment_confidence=_readonly(confidence_by_condition[condition]),
            candidates=tuple(rows),
            diagnostics={
                "scene_id": str(scene_id),
                "candidate_cluster_condition": condition,
                "semantic_threshold": 0.7,
                "sample_cap": SAMPLE_CAP,
                "global_typical_diag_m": global_diag,
                **distance_diagnostics,
                "class_diagnostics": class_diagnostics[condition],
                "raw_member_count": raw_member_count,
                "raw_member_retained_count": raw_member_retained_count,
                "raw_member_reassigned_count": (
                    raw_member_count - raw_member_retained_count
                ),
                "core_outside_full_count": core_outside_full_count,
                "orphan_count": 0,
                "negative_metadata_count": 0,
                "determinism_measured": False,
                "determinism_measured_this_scene": False,
                "determinism_contract_verified": False,
                "gt_used": False,
                "category_specific_prior_used": False,
            },
        )
        _validate_bank(bank)
        banks[condition] = bank
    return ClusterBankFamily(
        banks=banks,
        global_typical_diag_m=global_diag,
        r0_sample_rank=_readonly(r0_sample_rank),
        r0_sample_class_index=_readonly(r0_sample_class_index),
        r0_hdbscan_labels=_readonly(r0_hdbscan_labels),
        r0_hdbscan_membership=_readonly(r0_hdbscan_membership),
        r0_diagnostics={
            "scene_id": str(scene_id),
            "condition": R0_LEGACY,
            "sampled_point_count": int(np.count_nonzero(r0_sample_rank >= 0)),
            "raw_member_count": int(r0_raw_member_count),
            "raw_member_retained_count": int(r0_retained_count),
            "raw_member_reassigned_count": int(
                r0_raw_member_count - r0_retained_count
            ),
            "core_outside_full_count": int(r0_core_outside),
            "legacy_distance_measurements": r0_distance,
            "class_diagnostics": r0_class_diagnostics,
            "determinism_measured": False,
            "determinism_measured_this_scene": False,
            "determinism_contract_verified": False,
        },
    )


__all__ = [
    "CLUSTER_CONDITIONS",
    "ClusterBankFamily",
    "G1_MUTUAL_LOCAL_GRAPH",
    "R0_LEGACY",
    "R1_METRIC_HDBSCAN",
    "R2_ANCHORED_HDBSCAN",
    "build_cluster_bank_family",
    "load_cluster_raw_audit",
    "measure_cluster_family_determinism",
    "save_cluster_raw_audit",
]
