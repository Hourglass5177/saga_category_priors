from __future__ import annotations

"""Lossless construction trace for a frozen category-denoise candidate bank.

The v1 :class:`~category_priors.category_denoise.CandidateBank` intentionally
keeps only retained sampled-cluster membership and retained full assignment.
That is sufficient for replay, but it cannot explain clusters discarded after
full assignment.  This module defines a separate trace artifact; it neither
changes the bank schema nor participates in deployable prediction.

``hdbscan_labels`` below are the hard labels returned by ``fit_predict``.
They describe sampled cluster membership, not HDBSCAN's stricter notion of
core samples.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise import (
    ASSIGNMENT_THRESHOLD,
    MIN_CLUSTER_SIZE,
    SAMPLE_CAP,
    SEMANTIC_THRESHOLD,
    CandidateBank,
    stable_class_seed,
)

TRACE_SCHEMA = "saga-category-denoise-formation-trace-v2"
RAW_CLUSTERS_SCHEMA = "saga-category-denoise-raw-clusters-v1"
TRACE_DIAGNOSTICS_SCHEMA = "saga-category-denoise-trace-diagnostics-v1"
IDENTITY_ATOL = 1e-6


def _array(value: Any, dtype: np.dtype[Any] | type) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("trace metadata must contain only finite numbers")
        return number
    raise TypeError(f"trace metadata contains unsupported value {type(value)!r}")


@dataclass(frozen=True)
class CandidateFormationClassCapture:
    """Construction-time intermediates for one semantic branch.

    ``sampled_local_indices`` are positions in ``selected_indices`` in the
    exact RNG permutation order.  ``prethreshold_argmax_center`` contains
    positions in ``raw_cluster_ids`` before the 0.3 confidence gate is
    applied.  For a branch without a raw cluster it must be all ``-1`` and
    confidence must be zero.
    """

    branch_class: str
    branch_class_index: int
    selected_indices: np.ndarray
    sampled_local_indices: np.ndarray
    hdbscan_labels: np.ndarray
    hdbscan_membership: np.ndarray
    raw_cluster_ids: tuple[int, ...]
    prethreshold_argmax_center: np.ndarray
    prethreshold_assignment_confidence: np.ndarray
    legacy_assignment_chosen_center: np.ndarray
    legacy_assignment_feature_similarity: np.ndarray
    legacy_assignment_feature_center_norm: np.ndarray
    legacy_assignment_spatial_distance_standardized: np.ndarray
    legacy_assignment_spatial_similarity: np.ndarray
    legacy_assignment_hybrid_similarity: np.ndarray
    legacy_assignment_xyz_denominator: np.ndarray
    legacy_assignment_softmax_temperature: float
    sampled_raw_medoid_local_index: np.ndarray
    sampled_medoid_instance_distance: np.ndarray
    sampled_medoid_spatial_distance: np.ndarray
    sampled_medoid_semantic_distance: np.ndarray
    sampled_medoid_hybrid_distance: np.ndarray
    diagnostics: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CandidateFormationTrace:
    """Self-validating per-point trace kept separate from CandidateBank v1."""

    scene_id: str
    point_count: int
    class_names: tuple[str, ...]
    saga20_names: tuple[str, ...]
    scene_scale_m_per_unit: float
    seed: int
    semantic_threshold: float
    sample_cap: int
    min_cluster_size: int
    assignment_threshold: float
    semantic_selected_class_index: np.ndarray
    sample_rank: np.ndarray
    hdbscan_labels: np.ndarray
    hdbscan_membership: np.ndarray
    raw_cluster_membership: np.ndarray
    prethreshold_argmax_raw_cluster: np.ndarray
    prethreshold_assignment_confidence: np.ndarray
    legacy_assignment_chosen_raw_cluster: np.ndarray
    legacy_assignment_feature_similarity: np.ndarray
    legacy_assignment_feature_center_norm: np.ndarray
    legacy_assignment_spatial_distance_standardized: np.ndarray
    legacy_assignment_spatial_similarity: np.ndarray
    legacy_assignment_hybrid_similarity: np.ndarray
    raw_medoid_point_index: np.ndarray
    raw_medoid_instance_distance: np.ndarray
    raw_medoid_spatial_distance: np.ndarray
    raw_medoid_semantic_distance: np.ndarray
    raw_medoid_hybrid_distance: np.ndarray
    class_rows: tuple[dict[str, Any], ...]
    raw_cluster_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    schema: str = TRACE_SCHEMA


@dataclass(frozen=True)
class CandidateBankIdentityComparison:
    """Exact-label and 1e-6 numeric comparison of two frozen banks."""

    matches: bool
    atol: float
    mismatches: tuple[str, ...]
    max_abs_differences: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": self.matches,
            "atol": self.atol,
            "mismatches": list(self.mismatches),
            "max_abs_differences": dict(self.max_abs_differences),
        }


def _validate_tolerance(atol: float) -> float:
    value = float(atol)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("atol must be finite and non-negative")
    return value


def _max_abs_difference(left: Any, right: Any) -> float | None:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or not first.size:
        return None
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return math.inf
    return float(np.max(np.abs(first - second)))


def _compare_nested(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    mismatches: list[str],
    differences: dict[str, float | None],
) -> None:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            mismatches.append(path)
            return
        if set(left) != set(right):
            mismatches.append(f"{path}.keys")
        for key in sorted(set(left).intersection(right), key=str):
            _compare_nested(
                left[key],
                right[key],
                path=f"{path}.{key}",
                atol=atol,
                mismatches=mismatches,
                differences=differences,
            )
        return
    sequence_types = (list, tuple, np.ndarray)
    if isinstance(left, sequence_types) or isinstance(right, sequence_types):
        if not isinstance(left, sequence_types) or not isinstance(
            right, sequence_types
        ):
            mismatches.append(path)
            return
        first = list(np.asarray(left, dtype=object).reshape(-1))
        second = list(np.asarray(right, dtype=object).reshape(-1))
        if len(first) != len(second):
            mismatches.append(f"{path}.length")
            return
        for index, (first_item, second_item) in enumerate(zip(first, second)):
            _compare_nested(
                first_item,
                second_item,
                path=f"{path}[{index}]",
                atol=atol,
                mismatches=mismatches,
                differences=differences,
            )
        return
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        if not isinstance(left, (bool, np.bool_)) or not isinstance(
            right, (bool, np.bool_)
        ) or bool(left) != bool(right):
            mismatches.append(path)
        return
    if isinstance(left, Real) or isinstance(right, Real):
        if not isinstance(left, Real) or not isinstance(right, Real):
            mismatches.append(path)
            return
        first_number = float(left)
        second_number = float(right)
        difference = abs(first_number - second_number)
        differences[path] = difference
        if (
            not math.isfinite(first_number)
            or not math.isfinite(second_number)
            or difference > atol
        ):
            mismatches.append(path)
        return
    if left != right:
        mismatches.append(path)


def compare_candidate_bank_identity(
    reference: CandidateBank,
    observed: CandidateBank,
    *,
    atol: float = IDENTITY_ATOL,
) -> CandidateBankIdentityComparison:
    """Compare a shadow rerun with a frozen bank.

    Integer labels and structural identity are exact.  Semantic scores,
    candidate numeric evidence (including ``base_score``/Q), metric geometry,
    vote ratios, and assignment confidence use absolute tolerance only.
    """

    tolerance = _validate_tolerance(atol)
    mismatches: list[str] = []
    differences: dict[str, float | None] = {}

    exact_scalars = {
        "schema": (reference.schema, observed.schema),
        "point_count": (reference.point_count, observed.point_count),
        "class_names": (reference.class_names, observed.class_names),
        "saga20_names": (reference.saga20_names, observed.saga20_names),
        "seed": (reference.seed, observed.seed),
    }
    for name, (left, right) in exact_scalars.items():
        if left != right:
            mismatches.append(name)
    scene_left = reference.diagnostics.get("scene_id")
    scene_right = observed.diagnostics.get("scene_id")
    if scene_left != scene_right:
        mismatches.append("diagnostics.scene_id")

    scale_difference = abs(
        float(reference.scene_scale_m_per_unit)
        - float(observed.scene_scale_m_per_unit)
    )
    differences["scene_scale_m_per_unit"] = scale_difference
    if scale_difference > tolerance:
        mismatches.append("scene_scale_m_per_unit")

    exact_arrays = (
        "global_pre_knn",
        "semantic_top1",
        "branch_full_labels",
        "branch_core_labels",
    )
    for name in exact_arrays:
        left = np.asarray(getattr(reference, name))
        right = np.asarray(getattr(observed, name))
        if left.shape != right.shape or not np.array_equal(left, right):
            mismatches.append(name)

    numeric_arrays = ("semantic_top1_score", "assignment_confidence")
    for name in numeric_arrays:
        left = np.asarray(getattr(reference, name), dtype=np.float64)
        right = np.asarray(getattr(observed, name), dtype=np.float64)
        differences[name] = _max_abs_difference(left, right)
        if (
            left.shape != right.shape
            or not np.isfinite(left).all()
            or not np.isfinite(right).all()
            or not np.allclose(left, right, rtol=0.0, atol=tolerance)
        ):
            mismatches.append(name)

    if len(reference.candidates) != len(observed.candidates):
        mismatches.append("candidates.length")
    else:
        for index, (left, right) in enumerate(
            zip(reference.candidates, observed.candidates)
        ):
            _compare_nested(
                left,
                right,
                path=f"candidates[{index}]",
                atol=tolerance,
                mismatches=mismatches,
                differences=differences,
            )

    stable_parameter_keys = (
        "semantic_threshold",
        "sample_cap",
        "min_cluster_size",
        "min_samples",
        "weights",
        "assignment_threshold",
    )
    for key in stable_parameter_keys:
        if key in reference.diagnostics or key in observed.diagnostics:
            _compare_nested(
                reference.diagnostics.get(key),
                observed.diagnostics.get(key),
                path=f"diagnostics.{key}",
                atol=tolerance,
                mismatches=mismatches,
                differences=differences,
            )

    unique_mismatches = tuple(dict.fromkeys(mismatches))
    return CandidateBankIdentityComparison(
        matches=not unique_mismatches,
        atol=tolerance,
        mismatches=unique_mismatches,
        max_abs_differences=differences,
    )


def assert_candidate_bank_identity(
    reference: CandidateBank,
    observed: CandidateBank,
    *,
    atol: float = IDENTITY_ATOL,
) -> CandidateBankIdentityComparison:
    """Raise unless a shadow bank reproduces its frozen reference."""

    result = compare_candidate_bank_identity(reference, observed, atol=atol)
    if not result.matches:
        details = ", ".join(result.mismatches[:8])
        if len(result.mismatches) > 8:
            details += ", ..."
        raise ValueError(f"candidate bank identity mismatch: {details}")
    return result


def _expected_selected_class(bank: CandidateBank, threshold: float) -> np.ndarray:
    saga_indices = np.asarray(
        [bank.class_names.index(name) for name in bank.saga20_names],
        dtype=np.int64,
    )
    semantic = np.asarray(bank.semantic_top1, dtype=np.int64)
    score = np.asarray(bank.semantic_top1_score, dtype=np.float64)
    eligible = np.isin(semantic, saga_indices) & (score >= float(threshold))
    return np.where(eligible, semantic, -1).astype(np.int64, copy=False)


def _candidate_by_raw_key(
    bank: CandidateBank,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for candidate in bank.candidates:
        if "hdbscan_cluster_id" not in candidate:
            raise ValueError(
                "trace construction requires hdbscan_cluster_id on every candidate"
            )
        key = (
            str(candidate["branch_class"]),
            int(candidate["hdbscan_cluster_id"]),
        )
        if key in result:
            raise ValueError(f"candidate bank repeats raw cluster key {key}")
        result[key] = candidate
    return result


def build_candidate_formation_trace(
    *,
    scene_id: str,
    bank: CandidateBank,
    class_captures: Sequence[CandidateFormationClassCapture],
    semantic_threshold: float = SEMANTIC_THRESHOLD,
    sample_cap: int = SAMPLE_CAP,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    assignment_threshold: float = ASSIGNMENT_THRESHOLD,
    diagnostics: Mapping[str, Any] | None = None,
) -> CandidateFormationTrace:
    """Materialize a trace from the exact class-loop intermediates.

    One capture is required for every ``bank.saga20_names`` branch, including
    classes that did not have enough selected points to run HDBSCAN.
    """

    scene = str(scene_id)
    if not scene:
        raise ValueError("scene_id must not be empty")
    cap = int(sample_cap)
    minimum = int(min_cluster_size)
    threshold = float(assignment_threshold)
    semantic_gate = float(semantic_threshold)
    if cap <= 0:
        raise ValueError("sample_cap must be positive")
    if minimum <= 0:
        raise ValueError("min_cluster_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("assignment_threshold must be in [0, 1]")
    if not math.isfinite(semantic_gate):
        raise ValueError("semantic_threshold must be finite")
    recorded_scene = bank.diagnostics.get("scene_id")
    if recorded_scene not in {None, scene}:
        raise ValueError("scene_id does not match the candidate bank")

    captures: dict[str, CandidateFormationClassCapture] = {}
    for capture in class_captures:
        name = str(capture.branch_class)
        if name in captures:
            raise ValueError(f"duplicate class capture: {name}")
        captures[name] = capture
    if set(captures) != set(bank.saga20_names):
        missing = sorted(set(bank.saga20_names) - set(captures))
        extra = sorted(set(captures) - set(bank.saga20_names))
        raise ValueError(
            f"class captures differ from bank branches: {missing=}, {extra=}"
        )

    point_count = bank.point_count
    selected_class = np.full(point_count, -1, dtype=np.int64)
    sample_rank = np.full(point_count, -1, dtype=np.int64)
    hdbscan_labels = np.full(point_count, -1, dtype=np.int64)
    hdbscan_membership = np.zeros(point_count, dtype=np.float64)
    raw_membership = np.full(point_count, -1, dtype=np.int64)
    prethreshold_argmax = np.full(point_count, -1, dtype=np.int64)
    prethreshold_confidence = np.zeros(point_count, dtype=np.float64)
    legacy_chosen_cluster = np.full(point_count, -1, dtype=np.int64)
    legacy_feature_similarity = np.zeros(point_count, dtype=np.float64)
    legacy_feature_center_norm = np.zeros(point_count, dtype=np.float64)
    legacy_spatial_distance = np.zeros(point_count, dtype=np.float64)
    legacy_spatial_similarity = np.zeros(point_count, dtype=np.float64)
    legacy_hybrid_similarity = np.zeros(point_count, dtype=np.float64)
    raw_medoid_point = np.full(point_count, -1, dtype=np.int64)
    medoid_instance = np.zeros(point_count, dtype=np.float64)
    medoid_spatial = np.zeros(point_count, dtype=np.float64)
    medoid_semantic = np.zeros(point_count, dtype=np.float64)
    medoid_hybrid = np.zeros(point_count, dtype=np.float64)
    expected_selected = _expected_selected_class(bank, semantic_gate)
    retained_by_key = _candidate_by_raw_key(bank)
    used_retained_keys: set[tuple[str, int]] = set()
    class_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    next_raw_cluster_id = 0

    for class_name in sorted(bank.saga20_names):
        capture = captures[class_name]
        class_index = int(capture.branch_class_index)
        expected_index = bank.class_names.index(class_name)
        if class_index != expected_index:
            raise ValueError(f"{class_name}: capture class index is inconsistent")
        selected = _array(capture.selected_indices, np.int64)
        if selected.ndim != 1:
            raise ValueError(f"{class_name}: selected_indices must be one-dimensional")
        if len(selected) and (
            np.any((selected < 0) | (selected >= point_count))
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError(
                f"{class_name}: selected_indices must be sorted unique point indices"
            )
        expected_indices = np.flatnonzero(expected_selected == class_index)
        if not np.array_equal(selected, expected_indices):
            raise ValueError(
                f"{class_name}: selected_indices do not reproduce semantic selection"
            )
        selected_class[selected] = class_index

        sampled_local = _array(capture.sampled_local_indices, np.int64)
        if sampled_local.ndim != 1:
            raise ValueError(
                f"{class_name}: sampled_local_indices must be one-dimensional"
            )
        if len(sampled_local) and (
            np.any((sampled_local < 0) | (sampled_local >= len(selected)))
            or len(np.unique(sampled_local)) != len(sampled_local)
        ):
            raise ValueError(f"{class_name}: sampled_local_indices are invalid")
        expected_sample_count = (
            min(len(selected), cap) if len(selected) >= minimum else 0
        )
        if len(sampled_local) != expected_sample_count:
            raise ValueError(
                f"{class_name}: sampled point count differs from construction contract"
            )
        expected_sampled_local = (
            np.random.default_rng(
                stable_class_seed(bank.seed, class_name)
            ).permutation(len(selected))[:expected_sample_count]
            if expected_sample_count
            else np.asarray([], dtype=np.int64)
        )
        if not np.array_equal(sampled_local, expected_sampled_local):
            raise ValueError(
                f"{class_name}: sample order does not reproduce the deterministic RNG"
            )
        sampled_global = selected[sampled_local]
        sample_rank[sampled_global] = np.arange(len(sampled_global), dtype=np.int64)

        labels = _array(capture.hdbscan_labels, np.int64)
        membership = _array(capture.hdbscan_membership, np.float64)
        if labels.shape != (len(sampled_global),):
            raise ValueError(f"{class_name}: HDBSCAN labels have an invalid shape")
        if membership.shape != (len(sampled_global),):
            raise ValueError(
                f"{class_name}: HDBSCAN membership has an invalid shape"
            )
        if np.any(labels < -1):
            raise ValueError(f"{class_name}: HDBSCAN labels may only use -1 as noise")
        if (
            not np.isfinite(membership).all()
            or np.any((membership < 0.0) | (membership > 1.0))
            or np.any(membership[labels < 0] != 0.0)
        ):
            raise ValueError(f"{class_name}: HDBSCAN membership is invalid")
        raw_local_ids = tuple(int(value) for value in capture.raw_cluster_ids)
        observed_local_ids = tuple(
            int(value) for value in np.unique(labels) if int(value) >= 0
        )
        if raw_local_ids != observed_local_ids:
            raise ValueError(
                f"{class_name}: raw_cluster_ids do not match HDBSCAN labels"
            )
        hdbscan_labels[sampled_global] = labels
        hdbscan_membership[sampled_global] = membership

        medoid_local = _array(
            capture.sampled_raw_medoid_local_index, np.int64
        )
        distance_arrays = {
            "instance": _array(
                capture.sampled_medoid_instance_distance, np.float64
            ),
            "spatial": _array(
                capture.sampled_medoid_spatial_distance, np.float64
            ),
            "semantic": _array(
                capture.sampled_medoid_semantic_distance, np.float64
            ),
            "hybrid": _array(
                capture.sampled_medoid_hybrid_distance, np.float64
            ),
        }
        if medoid_local.shape != (len(sampled_global),):
            raise ValueError(f"{class_name}: raw-medoid index has an invalid shape")
        for component, values in distance_arrays.items():
            if values.shape != (len(sampled_global),) or not np.isfinite(values).all():
                raise ValueError(
                    f"{class_name}: {component} medoid distance is invalid"
                )
            if np.any(values < 0.0):
                raise ValueError(
                    f"{class_name}: {component} medoid distance is negative"
                )
        clustered_sample = labels >= 0
        if np.any(clustered_sample & ((medoid_local < 0) | (medoid_local >= len(sampled_global)))):
            raise ValueError(f"{class_name}: clustered sample lacks a valid medoid")
        if np.any((~clustered_sample) & (medoid_local != -1)):
            raise ValueError(f"{class_name}: noise sample cannot name a raw medoid")
        if np.any((~clustered_sample) & np.logical_or.reduce(tuple(values != 0.0 for values in distance_arrays.values()))):
            raise ValueError(f"{class_name}: noise sample has non-zero medoid distance")
        raw_medoid_point[sampled_global[clustered_sample]] = sampled_global[
            medoid_local[clustered_sample]
        ]
        medoid_instance[sampled_global] = distance_arrays["instance"]
        medoid_spatial[sampled_global] = distance_arrays["spatial"]
        medoid_semantic[sampled_global] = distance_arrays["semantic"]
        medoid_hybrid[sampled_global] = distance_arrays["hybrid"]

        center = _array(capture.prethreshold_argmax_center, np.int64)
        confidence = _array(
            capture.prethreshold_assignment_confidence, np.float64
        )
        if center.shape != (len(selected),) or confidence.shape != (len(selected),):
            raise ValueError(
                f"{class_name}: pre-threshold assignment arrays have invalid shapes"
            )
        if not np.isfinite(confidence).all() or np.any(
            (confidence < 0.0) | (confidence > 1.0)
        ):
            raise ValueError(
                f"{class_name}: pre-threshold assignment confidence is invalid"
            )
        if raw_local_ids:
            if np.any((center < 0) | (center >= len(raw_local_ids))):
                raise ValueError(
                    f"{class_name}: pre-threshold center index is invalid"
                )
        elif np.any(center != -1) or np.any(confidence != 0.0):
            raise ValueError(
                f"{class_name}: a branch without raw clusters cannot assign centers"
            )

        legacy_center = _array(
            capture.legacy_assignment_chosen_center, np.int64
        )
        legacy_components = {
            "feature_similarity": _array(
                capture.legacy_assignment_feature_similarity, np.float64
            ),
            "feature_center_norm": _array(
                capture.legacy_assignment_feature_center_norm, np.float64
            ),
            "spatial_distance_standardized": _array(
                capture.legacy_assignment_spatial_distance_standardized,
                np.float64,
            ),
            "spatial_similarity": _array(
                capture.legacy_assignment_spatial_similarity, np.float64
            ),
            "hybrid_similarity": _array(
                capture.legacy_assignment_hybrid_similarity, np.float64
            ),
        }
        if legacy_center.shape != (len(selected),):
            raise ValueError(
                f"{class_name}: legacy assignment center has an invalid shape"
            )
        for component, values in legacy_components.items():
            if values.shape != (len(selected),) or not np.isfinite(values).all():
                raise ValueError(
                    f"{class_name}: legacy {component} has an invalid shape or values"
                )
        xyz_denominator = _array(
            capture.legacy_assignment_xyz_denominator, np.float64
        )
        if (
            xyz_denominator.shape != (3,)
            or not np.isfinite(xyz_denominator).all()
            or np.any(xyz_denominator <= 0.0)
        ):
            raise ValueError(
                f"{class_name}: legacy XYZ denominator must be three positive values"
            )
        temperature = float(capture.legacy_assignment_softmax_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"{class_name}: legacy assignment temperature must be positive"
            )
        if not np.array_equal(legacy_center, center):
            raise ValueError(
                f"{class_name}: legacy chosen center differs from pre-threshold argmax"
            )
        if raw_local_ids:
            if (
                np.any(
                    (legacy_components["feature_similarity"] < -1.0)
                    | (legacy_components["feature_similarity"] > 1.0)
                )
                or np.any(legacy_components["feature_center_norm"] < 0.0)
                or np.any(
                    legacy_components["spatial_distance_standardized"] < 0.0
                )
                or np.any(
                    (legacy_components["spatial_similarity"] < 0.0)
                    | (legacy_components["spatial_similarity"] > 1.0)
                )
            ):
                raise ValueError(
                    f"{class_name}: legacy assignment components are outside their domains"
                )
            expected_spatial_similarity = np.exp(
                -legacy_components["spatial_distance_standardized"]
            )
            if np.max(
                np.abs(
                    expected_spatial_similarity
                    - legacy_components["spatial_similarity"]
                ),
                initial=0.0,
            ) > IDENTITY_ATOL:
                raise ValueError(
                    f"{class_name}: legacy spatial similarity disagrees with its distance"
                )
            expected_hybrid_similarity = (
                0.5 * legacy_components["feature_similarity"]
                + 0.5 * legacy_components["spatial_similarity"]
            )
            if np.max(
                np.abs(
                    expected_hybrid_similarity
                    - legacy_components["hybrid_similarity"]
                ),
                initial=0.0,
            ) > IDENTITY_ATOL:
                raise ValueError(
                    f"{class_name}: legacy hybrid similarity disagrees with its components"
                )
        elif np.any(legacy_center != -1) or any(
            np.any(values != 0.0) for values in legacy_components.values()
        ):
            raise ValueError(
                f"{class_name}: a branch without raw clusters has legacy assignment evidence"
            )

        local_to_global: dict[int, int] = {}
        for local_id in raw_local_ids:
            local_to_global[local_id] = next_raw_cluster_id
            next_raw_cluster_id += 1
        for sample_index, local_id in zip(sampled_global, labels):
            if int(local_id) >= 0:
                raw_membership[int(sample_index)] = local_to_global[int(local_id)]
        if raw_local_ids:
            global_centers = np.asarray(
                [local_to_global[raw_local_ids[int(value)]] for value in center],
                dtype=np.int64,
            )
            prethreshold_argmax[selected] = global_centers
            prethreshold_confidence[selected] = confidence
            legacy_chosen_cluster[selected] = global_centers
            legacy_feature_similarity[selected] = legacy_components[
                "feature_similarity"
            ]
            legacy_feature_center_norm[selected] = legacy_components[
                "feature_center_norm"
            ]
            legacy_spatial_distance[selected] = legacy_components[
                "spatial_distance_standardized"
            ]
            legacy_spatial_similarity[selected] = legacy_components[
                "spatial_similarity"
            ]
            legacy_hybrid_similarity[selected] = legacy_components[
                "hybrid_similarity"
            ]

        for local_id in raw_local_ids:
            raw_id = local_to_global[local_id]
            member_mask = raw_membership == raw_id
            argmax_mask = prethreshold_argmax == raw_id
            thresholded_mask = argmax_mask & (
                prethreshold_confidence >= threshold
            )
            retained = retained_by_key.get((class_name, local_id))
            if retained is not None:
                used_retained_keys.add((class_name, local_id))
            same_argmax = member_mask & argmax_mask
            cross_argmax = member_mask & (prethreshold_argmax >= 0) & ~argmax_mask
            rejected = member_mask & (prethreshold_confidence < threshold)
            member_strength = hdbscan_membership[member_mask]
            raw_rows.append(
                {
                    "raw_cluster_id": raw_id,
                    "branch_class": class_name,
                    "branch_class_index": class_index,
                    "hdbscan_cluster_id": local_id,
                    "sampled_member_count": int(np.count_nonzero(member_mask)),
                    "hdbscan_membership_mean": float(member_strength.mean()),
                    "prethreshold_argmax_count": int(np.count_nonzero(argmax_mask)),
                    "thresholded_full_count": int(
                        np.count_nonzero(thresholded_mask)
                    ),
                    "raw_member_same_argmax_count": int(
                        np.count_nonzero(same_argmax)
                    ),
                    "raw_member_cross_argmax_count": int(
                        np.count_nonzero(cross_argmax)
                    ),
                    "raw_member_threshold_rejected_count": int(
                        np.count_nonzero(rejected)
                    ),
                    "retained_candidate_id": int(retained["candidate_id"])
                    if retained is not None
                    else None,
                    "retention_status": "retained"
                    if retained is not None
                    else "discarded_full_below_min_cluster_size",
                }
            )

        class_rows.append(
            {
                "branch_class": class_name,
                "branch_class_index": class_index,
                "semantic_selected_point_count": len(selected),
                "sampled_point_count": len(sampled_global),
                "hdbscan_ran": len(selected) >= minimum,
                "hdbscan_noise_point_count": int(np.count_nonzero(labels < 0)),
                "raw_cluster_count": len(raw_local_ids),
                "retained_candidate_count": int(
                    sum(
                        row["branch_class"] == class_name
                        and row["retained_candidate_id"] is not None
                        for row in raw_rows
                    )
                ),
                "capture_diagnostics": _jsonable(capture.diagnostics or {}),
                "legacy_assignment_xyz_denominator": xyz_denominator.tolist(),
                "legacy_assignment_softmax_temperature": temperature,
                "legacy_assignment_feature_weight": 0.5,
                "legacy_assignment_spatial_weight": 0.5,
            }
        )

    if used_retained_keys != set(retained_by_key):
        missing = sorted(set(retained_by_key) - used_retained_keys)
        raise ValueError(f"candidate bank contains uncaptured raw clusters: {missing}")

    trace = CandidateFormationTrace(
        scene_id=scene,
        point_count=point_count,
        class_names=tuple(bank.class_names),
        saga20_names=tuple(bank.saga20_names),
        scene_scale_m_per_unit=float(bank.scene_scale_m_per_unit),
        seed=int(bank.seed),
        semantic_threshold=semantic_gate,
        sample_cap=cap,
        min_cluster_size=minimum,
        assignment_threshold=threshold,
        semantic_selected_class_index=_readonly(selected_class),
        sample_rank=_readonly(sample_rank),
        hdbscan_labels=_readonly(hdbscan_labels),
        hdbscan_membership=_readonly(hdbscan_membership),
        raw_cluster_membership=_readonly(raw_membership),
        prethreshold_argmax_raw_cluster=_readonly(prethreshold_argmax),
        prethreshold_assignment_confidence=_readonly(
            prethreshold_confidence
        ),
        legacy_assignment_chosen_raw_cluster=_readonly(
            legacy_chosen_cluster
        ),
        legacy_assignment_feature_similarity=_readonly(
            legacy_feature_similarity
        ),
        legacy_assignment_feature_center_norm=_readonly(
            legacy_feature_center_norm
        ),
        legacy_assignment_spatial_distance_standardized=_readonly(
            legacy_spatial_distance
        ),
        legacy_assignment_spatial_similarity=_readonly(
            legacy_spatial_similarity
        ),
        legacy_assignment_hybrid_similarity=_readonly(
            legacy_hybrid_similarity
        ),
        raw_medoid_point_index=_readonly(raw_medoid_point),
        raw_medoid_instance_distance=_readonly(medoid_instance),
        raw_medoid_spatial_distance=_readonly(medoid_spatial),
        raw_medoid_semantic_distance=_readonly(medoid_semantic),
        raw_medoid_hybrid_distance=_readonly(medoid_hybrid),
        class_rows=tuple(class_rows),
        raw_cluster_rows=tuple(raw_rows),
        diagnostics=dict(_jsonable(diagnostics or {})),
    )
    validate_candidate_formation_trace(trace, bank=bank)
    return trace


def _rows_by_raw_id(trace: CandidateFormationTrace) -> dict[int, Mapping[str, Any]]:
    rows: dict[int, Mapping[str, Any]] = {}
    for row in trace.raw_cluster_rows:
        raw_id = int(row["raw_cluster_id"])
        if raw_id in rows:
            raise ValueError(f"trace repeats raw_cluster_id {raw_id}")
        rows[raw_id] = row
    if sorted(rows) != list(range(len(rows))):
        raise ValueError("trace raw cluster IDs must be contiguous")
    return rows


def _derived_bank_arrays(
    trace: CandidateFormationTrace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_rows = _rows_by_raw_id(trace)
    raw_to_candidate = np.full(len(raw_rows), -1, dtype=np.int64)
    for raw_id, row in raw_rows.items():
        candidate_id = row.get("retained_candidate_id")
        if candidate_id is not None:
            raw_to_candidate[raw_id] = int(candidate_id)
    core = np.full(trace.point_count, -1, dtype=np.int64)
    full = np.full(trace.point_count, -1, dtype=np.int64)
    confidence = np.zeros(trace.point_count, dtype=np.float64)
    membership = np.asarray(trace.raw_cluster_membership, dtype=np.int64)
    argmax = np.asarray(trace.prethreshold_argmax_raw_cluster, dtype=np.int64)
    preconfidence = np.asarray(
        trace.prethreshold_assignment_confidence, dtype=np.float64
    )
    core_valid = membership >= 0
    if np.any(core_valid):
        mapped = raw_to_candidate[membership[core_valid]]
        indices = np.flatnonzero(core_valid)
        kept = mapped >= 0
        core[indices[kept]] = mapped[kept]
    full_valid = (argmax >= 0) & (
        preconfidence >= float(trace.assignment_threshold)
    )
    if np.any(full_valid):
        mapped = raw_to_candidate[argmax[full_valid]]
        indices = np.flatnonzero(full_valid)
        kept = mapped >= 0
        full[indices[kept]] = mapped[kept]
        confidence[indices[kept]] = preconfidence[indices[kept]]
    return core, full, confidence


def validate_candidate_formation_trace(
    trace: CandidateFormationTrace,
    *,
    bank: CandidateBank | None = None,
    atol: float = IDENTITY_ATOL,
) -> None:
    """Validate trace invariants and, optionally, its exact derived bank."""

    tolerance = _validate_tolerance(atol)
    if trace.schema != TRACE_SCHEMA:
        raise ValueError(f"unsupported candidate formation trace: {trace.schema}")
    if not trace.scene_id:
        raise ValueError("trace scene_id must not be empty")
    if int(trace.point_count) < 0:
        raise ValueError("trace point_count must be non-negative")
    if len(set(trace.class_names)) != len(trace.class_names):
        raise ValueError("trace class_names must be unique")
    if len(set(trace.saga20_names)) != len(trace.saga20_names):
        raise ValueError("trace saga20_names must be unique")
    if not set(trace.saga20_names).issubset(trace.class_names):
        raise ValueError("trace contains an unknown SAGA branch class")
    if int(trace.sample_cap) <= 0 or int(trace.min_cluster_size) <= 0:
        raise ValueError("trace sampling and cluster limits must be positive")
    if not math.isfinite(float(trace.semantic_threshold)):
        raise ValueError("trace semantic_threshold must be finite")
    if not 0.0 <= float(trace.assignment_threshold) <= 1.0:
        raise ValueError("trace assignment_threshold must be in [0, 1]")
    if (
        not math.isfinite(float(trace.scene_scale_m_per_unit))
        or float(trace.scene_scale_m_per_unit) <= 0.0
    ):
        raise ValueError("trace scene scale must be finite and positive")

    count = int(trace.point_count)
    integer_arrays = {
        "semantic_selected_class_index": trace.semantic_selected_class_index,
        "sample_rank": trace.sample_rank,
        "hdbscan_labels": trace.hdbscan_labels,
        "raw_cluster_membership": trace.raw_cluster_membership,
        "prethreshold_argmax_raw_cluster": (
            trace.prethreshold_argmax_raw_cluster
        ),
        "legacy_assignment_chosen_raw_cluster": (
            trace.legacy_assignment_chosen_raw_cluster
        ),
        "raw_medoid_point_index": trace.raw_medoid_point_index,
    }
    for name, value in integer_arrays.items():
        array = np.asarray(value)
        if array.shape != (count,) or array.dtype.kind not in {"i", "u"}:
            raise ValueError(f"trace array {name} has an invalid shape")
    float_arrays = {
        "hdbscan_membership": trace.hdbscan_membership,
        "prethreshold_assignment_confidence": (
            trace.prethreshold_assignment_confidence
        ),
        "legacy_assignment_feature_similarity": (
            trace.legacy_assignment_feature_similarity
        ),
        "legacy_assignment_feature_center_norm": (
            trace.legacy_assignment_feature_center_norm
        ),
        "legacy_assignment_spatial_distance_standardized": (
            trace.legacy_assignment_spatial_distance_standardized
        ),
        "legacy_assignment_spatial_similarity": (
            trace.legacy_assignment_spatial_similarity
        ),
        "legacy_assignment_hybrid_similarity": (
            trace.legacy_assignment_hybrid_similarity
        ),
        "raw_medoid_instance_distance": trace.raw_medoid_instance_distance,
        "raw_medoid_spatial_distance": trace.raw_medoid_spatial_distance,
        "raw_medoid_semantic_distance": trace.raw_medoid_semantic_distance,
        "raw_medoid_hybrid_distance": trace.raw_medoid_hybrid_distance,
    }
    for name, value in float_arrays.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError(f"trace array {name} has invalid shape or values")

    selected_class = np.asarray(
        trace.semantic_selected_class_index, dtype=np.int64
    )
    ranks = np.asarray(trace.sample_rank, dtype=np.int64)
    labels = np.asarray(trace.hdbscan_labels, dtype=np.int64)
    strengths = np.asarray(trace.hdbscan_membership, dtype=np.float64)
    raw_membership = np.asarray(trace.raw_cluster_membership, dtype=np.int64)
    raw_medoid = np.asarray(trace.raw_medoid_point_index, dtype=np.int64)
    argmax = np.asarray(
        trace.prethreshold_argmax_raw_cluster, dtype=np.int64
    )
    confidence = np.asarray(
        trace.prethreshold_assignment_confidence, dtype=np.float64
    )
    legacy_chosen = np.asarray(
        trace.legacy_assignment_chosen_raw_cluster, dtype=np.int64
    )
    legacy_feature = np.asarray(
        trace.legacy_assignment_feature_similarity, dtype=np.float64
    )
    legacy_center_norm = np.asarray(
        trace.legacy_assignment_feature_center_norm, dtype=np.float64
    )
    legacy_spatial_distance = np.asarray(
        trace.legacy_assignment_spatial_distance_standardized,
        dtype=np.float64,
    )
    legacy_spatial = np.asarray(
        trace.legacy_assignment_spatial_similarity, dtype=np.float64
    )
    legacy_hybrid = np.asarray(
        trace.legacy_assignment_hybrid_similarity, dtype=np.float64
    )
    distance_components = (
        np.asarray(trace.raw_medoid_instance_distance, dtype=np.float64),
        np.asarray(trace.raw_medoid_spatial_distance, dtype=np.float64),
        np.asarray(trace.raw_medoid_semantic_distance, dtype=np.float64),
        np.asarray(trace.raw_medoid_hybrid_distance, dtype=np.float64),
    )
    branch_indices = {
        trace.class_names.index(name) for name in trace.saga20_names
    }
    if np.any(
        (selected_class < -1)
        | ((selected_class >= 0) & ~np.isin(selected_class, list(branch_indices)))
    ):
        raise ValueError("trace semantic selection contains an invalid class")
    selected = selected_class >= 0
    sampled = ranks >= 0
    if np.any(ranks < -1) or np.any(sampled & ~selected):
        raise ValueError("trace sample ranks are invalid")
    if np.any(labels < -1) or np.any((~sampled) & (labels != -1)):
        raise ValueError("trace HDBSCAN labels are invalid")
    if (
        np.any((strengths < 0.0) | (strengths > 1.0))
        or np.any((~sampled) & (strengths != 0.0))
        or np.any(sampled & (labels < 0) & (strengths != 0.0))
    ):
        raise ValueError("trace HDBSCAN membership is invalid")
    if np.any(raw_membership < -1) or np.any((~sampled) & (raw_membership != -1)):
        raise ValueError("trace raw cluster membership is invalid")
    clustered = raw_membership >= 0
    if np.any((~clustered) & (raw_medoid != -1)) or np.any(
        clustered & ((raw_medoid < 0) | (raw_medoid >= count))
    ):
        raise ValueError("trace raw medoid indices are invalid")
    if np.any(clustered & (raw_membership[raw_medoid] != raw_membership)):
        raise ValueError("trace raw medoid crosses raw clusters")
    for component in distance_components:
        if np.any(component < 0.0) or np.any((~clustered) & (component != 0.0)):
            raise ValueError("trace raw-medoid distance components are invalid")
    reconstructed_hybrid = (
        0.5 * distance_components[0]
        + 0.3 * distance_components[1]
        + 0.2 * distance_components[2]
    )
    if np.max(np.abs(reconstructed_hybrid - distance_components[3]), initial=0.0) > tolerance:
        raise ValueError("trace hybrid distance disagrees with its components")
    if np.any(argmax < -1) or np.any((~selected) & (argmax != -1)):
        raise ValueError("trace pre-threshold argmax is invalid")
    if (
        np.any((confidence < 0.0) | (confidence > 1.0))
        or np.any((~selected) & (confidence != 0.0))
        or np.any((argmax < 0) & (confidence != 0.0))
    ):
        raise ValueError("trace pre-threshold confidence is invalid")
    if not np.array_equal(legacy_chosen, argmax):
        raise ValueError(
            "trace legacy chosen centers differ from pre-threshold argmax"
        )
    unassigned = legacy_chosen < 0
    if (
        np.any((legacy_feature < -1.0) | (legacy_feature > 1.0))
        or np.any(legacy_center_norm < 0.0)
        or np.any(legacy_spatial_distance < 0.0)
        or np.any((legacy_spatial < 0.0) | (legacy_spatial > 1.0))
        or np.any(unassigned & (legacy_feature != 0.0))
        or np.any(unassigned & (legacy_center_norm != 0.0))
        or np.any(unassigned & (legacy_spatial_distance != 0.0))
        or np.any(unassigned & (legacy_spatial != 0.0))
        or np.any(unassigned & (legacy_hybrid != 0.0))
    ):
        raise ValueError("trace legacy assignment components are invalid")
    assigned = ~unassigned
    reconstructed_spatial = np.exp(-legacy_spatial_distance[assigned])
    if np.max(
        np.abs(reconstructed_spatial - legacy_spatial[assigned]),
        initial=0.0,
    ) > tolerance:
        raise ValueError(
            "trace legacy spatial similarity disagrees with its distance"
        )
    reconstructed_assignment_hybrid = (
        0.5 * legacy_feature[assigned] + 0.5 * legacy_spatial[assigned]
    )
    if np.max(
        np.abs(reconstructed_assignment_hybrid - legacy_hybrid[assigned]),
        initial=0.0,
    ) > tolerance:
        raise ValueError(
            "trace legacy hybrid similarity disagrees with its components"
        )

    raw_rows = _rows_by_raw_id(trace)
    candidate_ids = sorted(
        int(row["retained_candidate_id"])
        for row in raw_rows.values()
        if row.get("retained_candidate_id") is not None
    )
    if candidate_ids != list(range(len(candidate_ids))):
        raise ValueError("trace retained candidate IDs must be contiguous")
    raw_keys: set[tuple[str, int]] = set()
    for raw_id, row in raw_rows.items():
        required = {
            "raw_cluster_id",
            "branch_class",
            "branch_class_index",
            "hdbscan_cluster_id",
            "sampled_member_count",
            "hdbscan_membership_mean",
            "prethreshold_argmax_count",
            "thresholded_full_count",
            "raw_member_same_argmax_count",
            "raw_member_cross_argmax_count",
            "raw_member_threshold_rejected_count",
            "retained_candidate_id",
            "retention_status",
        }
        if not required.issubset(row):
            raise ValueError(f"raw cluster row {raw_id} is missing required fields")
        name = str(row["branch_class"])
        class_index = int(row["branch_class_index"])
        local_id = int(row["hdbscan_cluster_id"])
        if (
            name not in trace.saga20_names
            or not 0 <= class_index < len(trace.class_names)
            or trace.class_names[class_index] != name
            or local_id < 0
        ):
            raise ValueError(f"raw cluster row {raw_id} has invalid class identity")
        key = (name, local_id)
        if key in raw_keys:
            raise ValueError(f"trace repeats raw HDBSCAN cluster {key}")
        raw_keys.add(key)
        class_mask = selected_class == class_index
        member_mask = raw_membership == raw_id
        if np.any(member_mask & ~class_mask):
            raise ValueError(f"raw cluster {raw_id} crosses semantic classes")
        expected_local = sampled & class_mask & (labels == local_id)
        if not np.array_equal(member_mask, expected_local):
            raise ValueError(f"raw cluster {raw_id} membership is inconsistent")
        argmax_mask = argmax == raw_id
        if np.any(argmax_mask & ~class_mask):
            raise ValueError(f"raw cluster {raw_id} argmax crosses semantic classes")
        thresholded = argmax_mask & (
            confidence >= float(trace.assignment_threshold)
        )
        same_argmax = member_mask & argmax_mask
        cross_argmax = member_mask & (argmax >= 0) & ~argmax_mask
        rejected = member_mask & (
            confidence < float(trace.assignment_threshold)
        )
        strength_mean = float(strengths[member_mask].mean())
        expected_values: dict[str, int | float] = {
            "sampled_member_count": int(np.count_nonzero(member_mask)),
            "hdbscan_membership_mean": strength_mean,
            "prethreshold_argmax_count": int(np.count_nonzero(argmax_mask)),
            "thresholded_full_count": int(np.count_nonzero(thresholded)),
            "raw_member_same_argmax_count": int(
                np.count_nonzero(same_argmax)
            ),
            "raw_member_cross_argmax_count": int(
                np.count_nonzero(cross_argmax)
            ),
            "raw_member_threshold_rejected_count": int(
                np.count_nonzero(rejected)
            ),
        }
        for field, expected in expected_values.items():
            actual = row[field]
            if isinstance(expected, float):
                if abs(float(actual) - expected) > tolerance:
                    raise ValueError(f"raw cluster {raw_id} {field} is inconsistent")
            elif int(actual) != expected:
                raise ValueError(f"raw cluster {raw_id} {field} is inconsistent")
        retained = row.get("retained_candidate_id") is not None
        expected_retained = int(row["thresholded_full_count"]) >= int(
            trace.min_cluster_size
        )
        if retained != expected_retained:
            raise ValueError(f"raw cluster {raw_id} retention is inconsistent")
        expected_status = (
            "retained"
            if retained
            else "discarded_full_below_min_cluster_size"
        )
        if row["retention_status"] != expected_status:
            raise ValueError(f"raw cluster {raw_id} retention status is inconsistent")

    observed_raw_keys = {
        (
            trace.class_names[int(selected_class[index])],
            int(labels[index]),
        )
        for index in np.flatnonzero(sampled & (labels >= 0))
    }
    if observed_raw_keys != raw_keys:
        raise ValueError("trace raw cluster rows do not cover all HDBSCAN labels")

    declared_raw_ids = np.asarray(sorted(raw_rows), dtype=np.int64)
    for name, array in (
        ("raw_cluster_membership", raw_membership),
        ("prethreshold_argmax_raw_cluster", argmax),
        ("legacy_assignment_chosen_raw_cluster", legacy_chosen),
    ):
        observed = np.unique(array[array >= 0])
        if not np.all(np.isin(observed, declared_raw_ids)):
            raise ValueError(f"trace {name} contains an undeclared raw cluster")

    class_rows: dict[str, Mapping[str, Any]] = {}
    for row in trace.class_rows:
        name = str(row.get("branch_class"))
        if name in class_rows:
            raise ValueError(f"trace repeats class row {name}")
        class_rows[name] = row
    if set(class_rows) != set(trace.saga20_names):
        raise ValueError("trace class rows differ from SAGA branch classes")
    for name in trace.saga20_names:
        row = class_rows[name]
        class_index = trace.class_names.index(name)
        class_mask = selected_class == class_index
        class_sampled = sampled & class_mask
        class_raw = [
            raw_id
            for raw_id, raw_row in raw_rows.items()
            if int(raw_row["branch_class_index"]) == class_index
        ]
        ranks_in_class = np.sort(ranks[class_sampled])
        if not np.array_equal(
            ranks_in_class, np.arange(len(ranks_in_class), dtype=np.int64)
        ):
            raise ValueError(f"{name}: sample ranks must be contiguous")
        expected_sample_count = (
            min(int(np.count_nonzero(class_mask)), int(trace.sample_cap))
            if int(np.count_nonzero(class_mask)) >= int(trace.min_cluster_size)
            else 0
        )
        expected = {
            "branch_class_index": class_index,
            "semantic_selected_point_count": int(np.count_nonzero(class_mask)),
            "sampled_point_count": expected_sample_count,
            "hdbscan_ran": int(np.count_nonzero(class_mask))
            >= int(trace.min_cluster_size),
            "hdbscan_noise_point_count": int(
                np.count_nonzero(class_sampled & (labels < 0))
            ),
            "raw_cluster_count": len(class_raw),
            "retained_candidate_count": int(
                sum(
                    raw_rows[raw_id].get("retained_candidate_id") is not None
                    for raw_id in class_raw
                )
            ),
            "legacy_assignment_feature_weight": 0.5,
            "legacy_assignment_spatial_weight": 0.5,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{name}: class row {field} is inconsistent")
        if int(np.count_nonzero(class_sampled)) != expected_sample_count:
            raise ValueError(f"{name}: sampled membership count is inconsistent")
        denominator = np.asarray(
            row.get("legacy_assignment_xyz_denominator"), dtype=np.float64
        )
        if (
            denominator.shape != (3,)
            or not np.isfinite(denominator).all()
            or np.any(denominator <= 0.0)
        ):
            raise ValueError(
                f"{name}: legacy assignment XYZ denominator is invalid"
            )
        temperature = float(
            row.get("legacy_assignment_softmax_temperature", math.nan)
        )
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"{name}: legacy assignment softmax temperature is invalid"
            )
        selected_indices = np.flatnonzero(class_mask)
        expected_local = (
            np.random.default_rng(
                stable_class_seed(trace.seed, name)
            ).permutation(len(selected_indices))[:expected_sample_count]
            if expected_sample_count
            else np.asarray([], dtype=np.int64)
        )
        expected_global = selected_indices[expected_local]
        actual_global = np.flatnonzero(class_sampled)
        actual_global = actual_global[np.argsort(ranks[actual_global])]
        if not np.array_equal(actual_global, expected_global):
            raise ValueError(f"{name}: sample ranks do not reproduce deterministic RNG")
        if class_raw:
            if np.any(class_mask & (argmax < 0)):
                raise ValueError(f"{name}: selected point lacks pre-threshold argmax")
        elif np.any(class_mask & ((argmax >= 0) | (confidence != 0.0))):
            raise ValueError(f"{name}: branch without raw clusters has assignment")

    if bank is None:
        return
    if trace.point_count != bank.point_count:
        raise ValueError("trace point count does not match candidate bank")
    if trace.class_names != bank.class_names or trace.saga20_names != bank.saga20_names:
        raise ValueError("trace class tables do not match candidate bank")
    if trace.seed != bank.seed:
        raise ValueError("trace seed does not match candidate bank")
    if (
        abs(trace.scene_scale_m_per_unit - bank.scene_scale_m_per_unit)
        > tolerance
    ):
        raise ValueError("trace scene scale does not match candidate bank")
    recorded_scene = bank.diagnostics.get("scene_id")
    if recorded_scene not in {None, trace.scene_id}:
        raise ValueError("trace scene_id does not match candidate bank")
    parameter_checks: tuple[tuple[str, int | float], ...] = (
        ("semantic_threshold", trace.semantic_threshold),
        ("sample_cap", trace.sample_cap),
        ("min_cluster_size", trace.min_cluster_size),
        ("assignment_threshold", trace.assignment_threshold),
    )
    for name, expected in parameter_checks:
        if name not in bank.diagnostics:
            continue
        actual = bank.diagnostics[name]
        if isinstance(expected, int):
            matches = int(actual) == expected
        else:
            matches = abs(float(actual) - float(expected)) <= tolerance
        if not matches:
            raise ValueError(f"trace {name} does not match candidate bank")
    expected_selected = _expected_selected_class(bank, trace.semantic_threshold)
    if not np.array_equal(selected_class, expected_selected):
        raise ValueError("trace semantic selection does not match candidate bank")

    derived_core, derived_full, derived_confidence = _derived_bank_arrays(trace)
    if not np.array_equal(derived_core, np.asarray(bank.branch_core_labels)):
        raise ValueError("trace does not reproduce candidate bank core labels")
    if not np.array_equal(derived_full, np.asarray(bank.branch_full_labels)):
        raise ValueError("trace does not reproduce candidate bank full labels")
    if not np.allclose(
        derived_confidence,
        np.asarray(bank.assignment_confidence, dtype=np.float64),
        rtol=0.0,
        atol=tolerance,
    ):
        raise ValueError("trace does not reproduce candidate bank confidence")

    candidates = {int(row["candidate_id"]): row for row in bank.candidates}
    if sorted(candidates) != list(range(len(candidates))):
        raise ValueError("candidate bank IDs are not contiguous")
    mapped_candidate_ids = {
        int(row["retained_candidate_id"])
        for row in raw_rows.values()
        if row.get("retained_candidate_id") is not None
    }
    if mapped_candidate_ids != set(candidates):
        raise ValueError("trace raw-to-candidate mapping is incomplete")
    for raw_id, raw_row in raw_rows.items():
        candidate_id = raw_row.get("retained_candidate_id")
        if candidate_id is None:
            continue
        candidate = candidates[int(candidate_id)]
        exact_fields = {
            "branch_class": raw_row["branch_class"],
            "branch_class_index": int(raw_row["branch_class_index"]),
            "hdbscan_cluster_id": int(raw_row["hdbscan_cluster_id"]),
            "core_point_count": int(raw_row["sampled_member_count"]),
            "full_point_count": int(raw_row["thresholded_full_count"]),
        }
        for field, value in exact_fields.items():
            if candidate.get(field) != value:
                raise ValueError(
                    "raw cluster "
                    f"{raw_id} disagrees with candidate {candidate_id} {field}"
                )
        full_mask = derived_full == int(candidate_id)
        actual_mean = float(derived_confidence[full_mask].mean())
        if (
            abs(float(candidate["assignment_confidence_mean"]) - actual_mean)
            > tolerance
        ):
            raise ValueError(
                f"raw cluster {raw_id} disagrees with candidate confidence mean"
            )
        class_row = class_rows[str(raw_row["branch_class"])]
        for candidate_field, class_field in (
            ("semantic_selected_point_count", "semantic_selected_point_count"),
            ("sampled_point_count", "sampled_point_count"),
        ):
            if candidate_field in candidate and int(candidate[candidate_field]) != int(
                class_row[class_field]
            ):
                raise ValueError(
                    f"candidate {candidate_id} {candidate_field} is inconsistent"
                )


def _trace_metadata(trace: CandidateFormationTrace) -> dict[str, Any]:
    return {
        "schema": trace.schema,
        "scene_id": trace.scene_id,
        "point_count": trace.point_count,
        "class_names": list(trace.class_names),
        "saga20_names": list(trace.saga20_names),
        "scene_scale_m_per_unit": trace.scene_scale_m_per_unit,
        "seed": trace.seed,
        "semantic_threshold": trace.semantic_threshold,
        "sample_cap": trace.sample_cap,
        "min_cluster_size": trace.min_cluster_size,
        "assignment_threshold": trace.assignment_threshold,
        "class_rows": [dict(row) for row in trace.class_rows],
        "raw_cluster_rows": [dict(row) for row in trace.raw_cluster_rows],
        "diagnostics": dict(trace.diagnostics),
    }


def _resolve_trace_paths(
    path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    location = Path(path)
    if location.suffix.lower() == ".npz":
        return location, Path(json_path) if json_path is not None else None
    sidecar = (
        Path(json_path)
        if json_path is not None
        else location / "formation_trace.json"
    )
    return location / "formation_trace.npz", sidecar


def _write_canonical_trace_views(
    trace: CandidateFormationTrace,
    directory: Path,
) -> None:
    """Write the canonical section-30 read-only diagnostic projections."""

    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "sample_rank.npz",
        semantic_selected_class_index=trace.semantic_selected_class_index,
        sample_rank=trace.sample_rank,
    )
    np.savez_compressed(
        directory / "raw_hdbscan_labels.npz",
        hdbscan_labels=trace.hdbscan_labels,
        hdbscan_membership=trace.hdbscan_membership,
    )
    np.savez_compressed(
        directory / "raw_membership.npz",
        raw_cluster_membership=trace.raw_cluster_membership,
    )
    np.savez_compressed(
        directory / "prethreshold_assignment.npz",
        prethreshold_argmax_raw_cluster=(
            trace.prethreshold_argmax_raw_cluster
        ),
        prethreshold_assignment_confidence=(
            trace.prethreshold_assignment_confidence
        ),
        legacy_assignment_chosen_raw_cluster=(
            trace.legacy_assignment_chosen_raw_cluster
        ),
    )
    np.savez_compressed(
        directory / "distance_components.npz",
        raw_cluster_membership=trace.raw_cluster_membership,
        raw_medoid_point_index=trace.raw_medoid_point_index,
        raw_medoid_instance_distance=trace.raw_medoid_instance_distance,
        raw_medoid_spatial_distance=trace.raw_medoid_spatial_distance,
        raw_medoid_semantic_distance=trace.raw_medoid_semantic_distance,
        raw_medoid_hybrid_distance=trace.raw_medoid_hybrid_distance,
        legacy_assignment_chosen_raw_cluster=(
            trace.legacy_assignment_chosen_raw_cluster
        ),
        legacy_assignment_feature_similarity=(
            trace.legacy_assignment_feature_similarity
        ),
        legacy_assignment_feature_center_norm=(
            trace.legacy_assignment_feature_center_norm
        ),
        legacy_assignment_spatial_distance_standardized=(
            trace.legacy_assignment_spatial_distance_standardized
        ),
        legacy_assignment_spatial_similarity=(
            trace.legacy_assignment_spatial_similarity
        ),
        legacy_assignment_hybrid_similarity=(
            trace.legacy_assignment_hybrid_similarity
        ),
    )
    raw_clusters = {
        "schema": RAW_CLUSTERS_SCHEMA,
        "trace_schema": trace.schema,
        "read_only": True,
        "scene_id": trace.scene_id,
        "raw_cluster_count": len(trace.raw_cluster_rows),
        "raw_clusters": [dict(row) for row in trace.raw_cluster_rows],
    }
    (directory / "raw_clusters.json").write_text(
        json.dumps(_jsonable(raw_clusters), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trace_diagnostics = {
        "schema": TRACE_DIAGNOSTICS_SCHEMA,
        "trace_schema": trace.schema,
        "read_only": True,
        "scene_id": trace.scene_id,
        "point_count": trace.point_count,
        "class_names": list(trace.class_names),
        "saga20_names": list(trace.saga20_names),
        "scene_scale_m_per_unit": trace.scene_scale_m_per_unit,
        "seed": trace.seed,
        "semantic_threshold": trace.semantic_threshold,
        "sample_cap": trace.sample_cap,
        "min_cluster_size": trace.min_cluster_size,
        "assignment_threshold": trace.assignment_threshold,
        "class_rows": [dict(row) for row in trace.class_rows],
        "diagnostics": dict(trace.diagnostics),
    }
    (directory / "trace_diagnostics.json").write_text(
        json.dumps(_jsonable(trace_diagnostics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_candidate_formation_trace(
    trace: CandidateFormationTrace,
    path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Save the main trace and all canonical section-30 projections."""

    validate_candidate_formation_trace(trace)
    destination, sidecar = _resolve_trace_paths(path, json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = _jsonable(_trace_metadata(trace))
    encoded = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            semantic_selected_class_index=trace.semantic_selected_class_index,
            sample_rank=trace.sample_rank,
            hdbscan_labels=trace.hdbscan_labels,
            hdbscan_membership=trace.hdbscan_membership,
            raw_cluster_membership=trace.raw_cluster_membership,
            prethreshold_argmax_raw_cluster=(
                trace.prethreshold_argmax_raw_cluster
            ),
            prethreshold_assignment_confidence=(
                trace.prethreshold_assignment_confidence
            ),
            legacy_assignment_chosen_raw_cluster=(
                trace.legacy_assignment_chosen_raw_cluster
            ),
            legacy_assignment_feature_similarity=(
                trace.legacy_assignment_feature_similarity
            ),
            legacy_assignment_feature_center_norm=(
                trace.legacy_assignment_feature_center_norm
            ),
            legacy_assignment_spatial_distance_standardized=(
                trace.legacy_assignment_spatial_distance_standardized
            ),
            legacy_assignment_spatial_similarity=(
                trace.legacy_assignment_spatial_similarity
            ),
            legacy_assignment_hybrid_similarity=(
                trace.legacy_assignment_hybrid_similarity
            ),
            raw_medoid_point_index=trace.raw_medoid_point_index,
            raw_medoid_instance_distance=trace.raw_medoid_instance_distance,
            raw_medoid_spatial_distance=trace.raw_medoid_spatial_distance,
            raw_medoid_semantic_distance=trace.raw_medoid_semantic_distance,
            raw_medoid_hybrid_distance=trace.raw_medoid_hybrid_distance,
            metadata_json=np.asarray(encoded),
        )
    if sidecar is not None:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _write_canonical_trace_views(trace, destination.parent)
    return destination, sidecar


def validate_candidate_formation_trace_views(
    trace: CandidateFormationTrace, path: str | Path
) -> None:
    """Reject stale canonical projections beside an otherwise valid trace."""

    directory = Path(path)
    if directory.suffix.lower() == ".npz":
        directory = directory.parent

    def check_npz(name: str, expected: Mapping[str, np.ndarray]) -> None:
        with np.load(directory / name, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError(f"trace view {name} has unexpected arrays")
            for key, value in expected.items():
                if not np.array_equal(archive[key], np.asarray(value)):
                    raise ValueError(f"trace view {name}:{key} is stale")

    check_npz(
        "sample_rank.npz",
        {
            "semantic_selected_class_index": trace.semantic_selected_class_index,
            "sample_rank": trace.sample_rank,
        },
    )
    check_npz(
        "raw_hdbscan_labels.npz",
        {
            "hdbscan_labels": trace.hdbscan_labels,
            "hdbscan_membership": trace.hdbscan_membership,
        },
    )
    check_npz(
        "raw_membership.npz",
        {"raw_cluster_membership": trace.raw_cluster_membership},
    )
    check_npz(
        "prethreshold_assignment.npz",
        {
            "prethreshold_argmax_raw_cluster": trace.prethreshold_argmax_raw_cluster,
            "prethreshold_assignment_confidence": trace.prethreshold_assignment_confidence,
            "legacy_assignment_chosen_raw_cluster": trace.legacy_assignment_chosen_raw_cluster,
        },
    )
    check_npz(
        "distance_components.npz",
        {
            "raw_cluster_membership": trace.raw_cluster_membership,
            "raw_medoid_point_index": trace.raw_medoid_point_index,
            "raw_medoid_instance_distance": trace.raw_medoid_instance_distance,
            "raw_medoid_spatial_distance": trace.raw_medoid_spatial_distance,
            "raw_medoid_semantic_distance": trace.raw_medoid_semantic_distance,
            "raw_medoid_hybrid_distance": trace.raw_medoid_hybrid_distance,
            "legacy_assignment_chosen_raw_cluster": trace.legacy_assignment_chosen_raw_cluster,
            "legacy_assignment_feature_similarity": trace.legacy_assignment_feature_similarity,
            "legacy_assignment_feature_center_norm": trace.legacy_assignment_feature_center_norm,
            "legacy_assignment_spatial_distance_standardized": trace.legacy_assignment_spatial_distance_standardized,
            "legacy_assignment_spatial_similarity": trace.legacy_assignment_spatial_similarity,
            "legacy_assignment_hybrid_similarity": trace.legacy_assignment_hybrid_similarity,
        },
    )
    raw_view = json.loads((directory / "raw_clusters.json").read_text(encoding="utf-8"))
    if raw_view.get("raw_clusters") != [dict(row) for row in trace.raw_cluster_rows]:
        raise ValueError("raw_clusters.json is stale")
    diagnostics_view = json.loads(
        (directory / "trace_diagnostics.json").read_text(encoding="utf-8")
    )
    if diagnostics_view.get("diagnostics") != dict(trace.diagnostics):
        raise ValueError("trace_diagnostics.json is stale")
    sidecar = json.loads(
        (directory / "formation_trace.json").read_text(encoding="utf-8")
    )
    if sidecar != _jsonable(_trace_metadata(trace)):
        raise ValueError("formation_trace.json is stale")


def load_candidate_formation_trace(path: str | Path) -> CandidateFormationTrace:
    """Load and fully validate a candidate formation trace."""

    source, _ = _resolve_trace_paths(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "semantic_selected_class_index",
            "sample_rank",
            "hdbscan_labels",
            "hdbscan_membership",
            "raw_cluster_membership",
            "prethreshold_argmax_raw_cluster",
            "prethreshold_assignment_confidence",
            "legacy_assignment_chosen_raw_cluster",
            "legacy_assignment_feature_similarity",
            "legacy_assignment_feature_center_norm",
            "legacy_assignment_spatial_distance_standardized",
            "legacy_assignment_spatial_similarity",
            "legacy_assignment_hybrid_similarity",
            "raw_medoid_point_index",
            "raw_medoid_instance_distance",
            "raw_medoid_spatial_distance",
            "raw_medoid_semantic_distance",
            "raw_medoid_hybrid_distance",
            "metadata_json",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"candidate formation trace is missing arrays: {missing}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        trace = CandidateFormationTrace(
            scene_id=str(metadata["scene_id"]),
            point_count=int(metadata["point_count"]),
            class_names=tuple(map(str, metadata["class_names"])),
            saga20_names=tuple(map(str, metadata["saga20_names"])),
            scene_scale_m_per_unit=float(metadata["scene_scale_m_per_unit"]),
            seed=int(metadata["seed"]),
            semantic_threshold=float(metadata["semantic_threshold"]),
            sample_cap=int(metadata["sample_cap"]),
            min_cluster_size=int(metadata["min_cluster_size"]),
            assignment_threshold=float(metadata["assignment_threshold"]),
            semantic_selected_class_index=_readonly(
                archive["semantic_selected_class_index"].astype(np.int64)
            ),
            sample_rank=_readonly(archive["sample_rank"].astype(np.int64)),
            hdbscan_labels=_readonly(
                archive["hdbscan_labels"].astype(np.int64)
            ),
            hdbscan_membership=_readonly(
                archive["hdbscan_membership"].astype(np.float64)
            ),
            raw_cluster_membership=_readonly(
                archive["raw_cluster_membership"].astype(np.int64)
            ),
            prethreshold_argmax_raw_cluster=_readonly(
                archive["prethreshold_argmax_raw_cluster"].astype(np.int64)
            ),
            prethreshold_assignment_confidence=_readonly(
                archive["prethreshold_assignment_confidence"].astype(np.float64)
            ),
            legacy_assignment_chosen_raw_cluster=_readonly(
                archive["legacy_assignment_chosen_raw_cluster"].astype(np.int64)
            ),
            legacy_assignment_feature_similarity=_readonly(
                archive["legacy_assignment_feature_similarity"].astype(np.float64)
            ),
            legacy_assignment_feature_center_norm=_readonly(
                archive["legacy_assignment_feature_center_norm"].astype(np.float64)
            ),
            legacy_assignment_spatial_distance_standardized=_readonly(
                archive[
                    "legacy_assignment_spatial_distance_standardized"
                ].astype(np.float64)
            ),
            legacy_assignment_spatial_similarity=_readonly(
                archive["legacy_assignment_spatial_similarity"].astype(np.float64)
            ),
            legacy_assignment_hybrid_similarity=_readonly(
                archive["legacy_assignment_hybrid_similarity"].astype(np.float64)
            ),
            raw_medoid_point_index=_readonly(
                archive["raw_medoid_point_index"].astype(np.int64)
            ),
            raw_medoid_instance_distance=_readonly(
                archive["raw_medoid_instance_distance"].astype(np.float64)
            ),
            raw_medoid_spatial_distance=_readonly(
                archive["raw_medoid_spatial_distance"].astype(np.float64)
            ),
            raw_medoid_semantic_distance=_readonly(
                archive["raw_medoid_semantic_distance"].astype(np.float64)
            ),
            raw_medoid_hybrid_distance=_readonly(
                archive["raw_medoid_hybrid_distance"].astype(np.float64)
            ),
            class_rows=tuple(dict(row) for row in metadata["class_rows"]),
            raw_cluster_rows=tuple(
                dict(row) for row in metadata["raw_cluster_rows"]
            ),
            diagnostics=dict(metadata.get("diagnostics", {})),
            schema=str(metadata["schema"]),
        )
    validate_candidate_formation_trace(trace)
    return trace


__all__ = [
    "CandidateBankIdentityComparison",
    "CandidateFormationClassCapture",
    "CandidateFormationTrace",
    "IDENTITY_ATOL",
    "RAW_CLUSTERS_SCHEMA",
    "TRACE_SCHEMA",
    "TRACE_DIAGNOSTICS_SCHEMA",
    "assert_candidate_bank_identity",
    "build_candidate_formation_trace",
    "compare_candidate_bank_identity",
    "load_candidate_formation_trace",
    "save_candidate_formation_trace",
    "validate_candidate_formation_trace",
    "validate_candidate_formation_trace_views",
]
