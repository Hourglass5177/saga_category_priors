from __future__ import annotations

"""Pure candidate-assignment repairs for the category-denoising experiment.

This module deliberately starts *after* semantic routing and HDBSCAN.  It does
not know about classes, category priors, ground truth, rendering, or the final
prediction format.  A caller supplies one class' selected Gaussian indices and
the exact HDBSCAN trace; the module returns class-local candidate labels which
can then be offset and scattered by a scene runner.

Three frozen modes are supported:

``legacy``
    Byte-for-byte passthrough of the labels/confidence supplied by the trace.
``consistent-envelope``
    Reassign every selected point with the same 0.5/0.3/0.2 distance used by
    HDBSCAN.  A raw member is a trusted core only when it returns to its own
    cluster with confidence at least 0.3.
``raw-anchored-envelope``
    Use the same envelope assignment, but pin every non-noise raw HDBSCAN
    member to its original cluster.  Its recorded confidence is its *own*
    cluster probability, not the probability of a competing winner.

The repair modes use a raw-member medoid and the 95th percentile of member to
medoid distance as the absolute attachment envelope.  There is no radius
multiplier or validation-time parameter search.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


LEGACY = "legacy"
CONSISTENT_ENVELOPE = "consistent-envelope"
RAW_ANCHORED_ENVELOPE = "raw-anchored-envelope"
REPAIR_MODES = frozenset(
    {LEGACY, CONSISTENT_ENVELOPE, RAW_ANCHORED_ENVELOPE}
)

INSTANCE_WEIGHT = 0.5
SPATIAL_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.2
ASSIGNMENT_THRESHOLD = 0.3
ASSIGNMENT_TEMPERATURE = 10.0
ENVELOPE_QUANTILE = 0.95
MIN_CANDIDATE_POINTS = 3
_DISTANCE_EPSILON = 1e-8


@dataclass(frozen=True)
class CandidateRepairScene:
    """Scene arrays needed by the class-local repair.

    ``xyz_scene`` must contain the complete scene, not only the selected class.
    The historical branch standardises XYZ using the complete scene min/span;
    retaining that axis is part of the frozen experiment.
    """

    instance_features: Any
    xyz_scene: Any
    semantic_top1_score: Any


@dataclass(frozen=True)
class CandidateRepairTrace:
    """One class' HDBSCAN trace on a selected-point axis.

    ``selected_global_indices`` maps the selected axis into the scene arrays.
    ``sampled_local_indices`` maps the HDBSCAN sample axis into the selected
    axis.  ``raw_cluster_labels`` is on the sample axis and uses ``-1`` for
    HDBSCAN noise.

    The three ``legacy_*`` arrays are required only for ``legacy`` mode and are
    all on the selected axis.  Optional distance maxima let a persisted trace
    provide the exact maxima used by the original sampled distance matrices;
    otherwise they are recomputed deterministically from the supplied arrays.
    """

    selected_global_indices: Any
    sampled_local_indices: Any
    raw_cluster_labels: Any
    legacy_full_labels: Any | None = None
    legacy_core_labels: Any | None = None
    legacy_assignment_confidence: Any | None = None
    instance_distance_max: float | None = None
    spatial_distance_max: float | None = None


@dataclass(frozen=True)
class CandidateRepairResult:
    """Class-local candidate repair output, all arrays on the selected axis."""

    mode: str
    selected_global_indices: np.ndarray
    raw_seed_cluster_index: np.ndarray
    trusted_core_labels: np.ndarray
    full_candidate_labels: np.ndarray
    assignment_confidence: np.ndarray
    raw_seed_own_probability: np.ndarray
    raw_cluster_ids: tuple[int, ...]
    candidates: tuple[dict[str, Any], ...]
    diagnostics: Mapping[str, Any]

    def scatter_labels(self, point_count: int, which: str = "full") -> np.ndarray:
        """Scatter selected-axis labels back onto the complete scene axis."""

        count = int(point_count)
        if count < 0:
            raise ValueError("point_count must be non-negative")
        if which == "full":
            selected_labels = self.full_candidate_labels
        elif which == "trusted_core":
            selected_labels = self.trusted_core_labels
        elif which == "raw_seed":
            selected_labels = self.raw_seed_cluster_index
        else:
            raise ValueError("which must be 'full', 'trusted_core', or 'raw_seed'")
        if len(self.selected_global_indices) and int(self.selected_global_indices.max()) >= count:
            raise ValueError("point_count does not cover selected_global_indices")
        output = np.full(count, -1, dtype=np.int64)
        output[self.selected_global_indices] = selected_labels
        return output


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
    # Always own the returned buffer.  ``np.ascontiguousarray`` is allowed to
    # return its input unchanged, after which setting ``write=False`` would
    # unexpectedly freeze a caller-owned trace array.
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _normalise_rows(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)


def _euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    difference = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def _softmax_negative_distance(distance: np.ndarray) -> np.ndarray:
    if distance.shape[1] == 0:
        return np.empty_like(distance)
    logits = -np.asarray(distance, dtype=np.float64) * ASSIGNMENT_TEMPERATURE
    logits -= np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(logits)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _distance_max(value: np.ndarray, supplied: float | None, name: str) -> float:
    if supplied is None:
        maximum = float(np.max(value)) if value.size else 0.0
    else:
        maximum = float(supplied)
        if not np.isfinite(maximum) or maximum < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    return maximum


def _scale_distance(value: np.ndarray, maximum: float) -> np.ndarray:
    if maximum <= 0:
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64) / (maximum + _DISTANCE_EPSILON)


def _validate_inputs(
    scene: CandidateRepairScene,
    trace: CandidateRepairTrace,
    mode: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if mode not in REPAIR_MODES:
        raise ValueError(f"mode must be one of {sorted(REPAIR_MODES)}, got {mode!r}")

    instance = _as_numpy(scene.instance_features, np.float64)
    xyz = _as_numpy(scene.xyz_scene, np.float64)
    semantic_score = _as_numpy(scene.semantic_top1_score, np.float64)
    if instance.ndim != 2:
        raise ValueError("instance_features must be a matrix")
    point_count = len(instance)
    if xyz.shape != (point_count, 3) or semantic_score.shape != (point_count,):
        raise ValueError("scene arrays must share one complete point axis")
    if not (
        np.isfinite(instance).all()
        and np.isfinite(xyz).all()
        and np.isfinite(semantic_score).all()
    ):
        raise ValueError("scene arrays must be finite")

    selected = _as_numpy(trace.selected_global_indices, np.int64)
    sampled = _as_numpy(trace.sampled_local_indices, np.int64)
    raw = _as_numpy(trace.raw_cluster_labels, np.int64)
    if selected.ndim != 1 or sampled.ndim != 1 or raw.shape != sampled.shape:
        raise ValueError("trace indices and raw_cluster_labels must be vectors")
    if len(selected) and (int(selected.min()) < 0 or int(selected.max()) >= point_count):
        raise ValueError("selected_global_indices are outside the scene point axis")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_global_indices must be unique")
    if len(sampled) and (int(sampled.min()) < 0 or int(sampled.max()) >= len(selected)):
        raise ValueError("sampled_local_indices are outside the selected axis")
    if len(np.unique(sampled)) != len(sampled):
        raise ValueError("sampled_local_indices must be unique")
    if np.any(raw < -1):
        raise ValueError("raw_cluster_labels may only use -1 as a negative label")
    return instance, xyz, semantic_score, selected, sampled, raw


def _legacy_result(
    trace: CandidateRepairTrace,
    selected: np.ndarray,
    sampled: np.ndarray,
    raw: np.ndarray,
) -> CandidateRepairResult:
    if (
        trace.legacy_full_labels is None
        or trace.legacy_core_labels is None
        or trace.legacy_assignment_confidence is None
    ):
        raise ValueError("legacy mode requires all three legacy selected-axis arrays")
    full = _as_numpy(trace.legacy_full_labels, np.int64)
    core = _as_numpy(trace.legacy_core_labels, np.int64)
    confidence = _as_numpy(trace.legacy_assignment_confidence, np.float64)
    expected = (len(selected),)
    if full.shape != expected or core.shape != expected or confidence.shape != expected:
        raise ValueError("legacy arrays must use the selected-point axis")
    if np.any(full < -1) or np.any(core < -1):
        raise ValueError("legacy labels may only use -1 as a negative label")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("legacy_assignment_confidence must be finite and in [0, 1]")

    raw_seed = np.full(len(selected), -1, dtype=np.int64)
    raw_cluster_ids = tuple(int(value) for value in np.unique(raw) if value >= 0)
    raw_to_index = {raw_id: index for index, raw_id in enumerate(raw_cluster_ids)}
    for sample_row, raw_id in enumerate(raw):
        if int(raw_id) >= 0:
            raw_seed[int(sampled[sample_row])] = raw_to_index[int(raw_id)]
    own_probability = np.full(len(selected), np.nan, dtype=np.float64)

    candidate_ids = sorted(
        {int(value) for value in np.concatenate((full, core)) if int(value) >= 0}
    )
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        full_mask = full == candidate_id
        core_mask = core == candidate_id
        rows.append(
            {
                "candidate_id": candidate_id,
                "raw_cluster_id": None,
                "raw_seed_count": int(np.count_nonzero(core_mask)),
                "trusted_core_count": int(np.count_nonzero(core_mask)),
                "full_point_count": int(np.count_nonzero(full_mask)),
                "assignment_confidence_mean": (
                    float(np.mean(confidence[full_mask]))
                    if np.any(full_mask)
                    else 0.0
                ),
            }
        )
    violations = int(np.count_nonzero((core >= 0) & (core != full)))
    return CandidateRepairResult(
        mode=LEGACY,
        selected_global_indices=_readonly(selected, np.int64),
        raw_seed_cluster_index=_readonly(raw_seed, np.int64),
        trusted_core_labels=_readonly(core.copy(), np.int64),
        full_candidate_labels=_readonly(full.copy(), np.int64),
        assignment_confidence=_readonly(confidence.copy(), np.float64),
        raw_seed_own_probability=_readonly(own_probability, np.float64),
        raw_cluster_ids=raw_cluster_ids,
        candidates=tuple(rows),
        diagnostics={
            "legacy_passthrough": True,
            "selected_point_count": int(len(selected)),
            "sampled_point_count": int(len(sampled)),
            "raw_cluster_count": int(len(raw_cluster_ids)),
            "candidate_count": int(len(candidate_ids)),
            "trusted_core_outside_full_count": violations,
            "core_contract_enforced": False,
        },
    )


def _stable_medoid(
    pair_distance: np.ndarray,
    member_rows: np.ndarray,
    sampled_global_indices: np.ndarray,
) -> int:
    within = pair_distance[np.ix_(member_rows, member_rows)]
    totals = np.sum(within, axis=1)
    global_ids = sampled_global_indices[member_rows]
    order = np.lexsort((global_ids, totals))
    return int(member_rows[int(order[0])])


def _remap_emitted_candidates(
    full_cluster_index: np.ndarray,
    trusted_cluster_index: np.ndarray,
    raw_cluster_ids: tuple[int, ...],
    minimum_points: int = MIN_CANDIDATE_POINTS,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    emitted_old = [
        index
        for index in range(len(raw_cluster_ids))
        if int(np.count_nonzero(full_cluster_index == index)) >= int(minimum_points)
    ]
    remap = {old: new for new, old in enumerate(emitted_old)}
    full = np.full(len(full_cluster_index), -1, dtype=np.int64)
    trusted = np.full(len(trusted_cluster_index), -1, dtype=np.int64)
    for old, new in remap.items():
        full[full_cluster_index == old] = new
        trusted[trusted_cluster_index == old] = new
    return full, trusted, remap


def repair_class_candidates(
    scene: CandidateRepairScene,
    trace: CandidateRepairTrace,
    mode: str,
) -> CandidateRepairResult:
    """Repair one class' raw-cluster to full-candidate assignment.

    The function has no class, prior, GT, evaluator, or file-system argument.
    Returned candidate IDs are class-local and contiguous in repair modes,
    ordered by sorted raw HDBSCAN cluster ID.  A scene runner may add a stable
    per-class offset before scattering them into a complete candidate bank.
    """

    instance, xyz, semantic_score, selected, sampled, raw = _validate_inputs(
        scene, trace, mode
    )
    if mode == LEGACY:
        return _legacy_result(trace, selected, sampled, raw)

    raw_cluster_ids = tuple(int(value) for value in np.unique(raw) if value >= 0)
    selected_count = len(selected)
    if not raw_cluster_ids:
        empty_labels = _readonly(np.full(selected_count, -1, dtype=np.int64))
        empty_probability = _readonly(
            np.full(selected_count, np.nan, dtype=np.float64)
        )
        return CandidateRepairResult(
            mode=mode,
            selected_global_indices=_readonly(selected, np.int64),
            raw_seed_cluster_index=empty_labels,
            trusted_core_labels=empty_labels,
            full_candidate_labels=empty_labels,
            assignment_confidence=_readonly(
                np.zeros(selected_count, dtype=np.float64)
            ),
            raw_seed_own_probability=empty_probability,
            raw_cluster_ids=(),
            candidates=(),
            diagnostics={
                "legacy_passthrough": False,
                "selected_point_count": int(selected_count),
                "sampled_point_count": int(len(sampled)),
                "raw_cluster_count": 0,
                "candidate_count": 0,
                "trusted_core_outside_full_count": 0,
                "core_contract_enforced": True,
            },
        )

    normed_instance = _normalise_rows(instance)
    minimum = np.min(xyz, axis=0)
    span = np.max(xyz, axis=0) - minimum
    standardised_xyz = (xyz - minimum) / np.where(span > 0, span, 1.0)
    selected_features = normed_instance[selected]
    selected_xyz = standardised_xyz[selected]
    selected_scores = semantic_score[selected]
    sampled_features = selected_features[sampled]
    sampled_xyz = selected_xyz[sampled]
    sampled_scores = selected_scores[sampled]

    sample_instance_distance = np.maximum(
        1.0 - sampled_features @ sampled_features.T, 0.0
    )
    sample_spatial_distance = _euclidean(sampled_xyz, sampled_xyz)
    sample_semantic_distance = np.clip(
        1.0 - np.outer(sampled_scores, sampled_scores), 0.0, 1.0
    )
    instance_max = _distance_max(
        sample_instance_distance, trace.instance_distance_max, "instance_distance_max"
    )
    spatial_max = _distance_max(
        sample_spatial_distance, trace.spatial_distance_max, "spatial_distance_max"
    )
    sampled_hybrid = (
        INSTANCE_WEIGHT * _scale_distance(sample_instance_distance, instance_max)
        + SPATIAL_WEIGHT * _scale_distance(sample_spatial_distance, spatial_max)
        + SEMANTIC_WEIGHT * sample_semantic_distance
    )

    sampled_global = selected[sampled]
    medoid_sample_rows: list[int] = []
    envelope_radius: list[float] = []
    cluster_members: list[np.ndarray] = []
    raw_to_index = {raw_id: index for index, raw_id in enumerate(raw_cluster_ids)}
    raw_seed_cluster_index = np.full(selected_count, -1, dtype=np.int64)
    for raw_id in raw_cluster_ids:
        members = np.flatnonzero(raw == raw_id).astype(np.int64, copy=False)
        cluster_members.append(members)
        medoid_row = _stable_medoid(sampled_hybrid, members, sampled_global)
        medoid_sample_rows.append(medoid_row)
        distances = sampled_hybrid[members, medoid_row]
        radius = float(np.quantile(distances, ENVELOPE_QUANTILE, method="linear"))
        envelope_radius.append(radius)
        raw_seed_cluster_index[sampled[members]] = raw_to_index[raw_id]

    medoid_selected_local = sampled[np.asarray(medoid_sample_rows, dtype=np.int64)]
    medoid_features = selected_features[medoid_selected_local]
    medoid_xyz = selected_xyz[medoid_selected_local]
    medoid_scores = selected_scores[medoid_selected_local]
    query_instance = np.maximum(1.0 - selected_features @ medoid_features.T, 0.0)
    query_spatial = _euclidean(selected_xyz, medoid_xyz)
    query_semantic = np.clip(
        1.0 - selected_scores[:, None] * medoid_scores[None, :], 0.0, 1.0
    )
    query_hybrid = (
        INSTANCE_WEIGHT * _scale_distance(query_instance, instance_max)
        + SPATIAL_WEIGHT * _scale_distance(query_spatial, spatial_max)
        + SEMANTIC_WEIGHT * query_semantic
    )
    probability = _softmax_negative_distance(query_hybrid)
    best_cluster = np.argmin(query_hybrid, axis=1).astype(np.int64, copy=False)
    minimum_distance = query_hybrid[np.arange(selected_count), best_cluster]
    exact_ties = np.sum(query_hybrid == minimum_distance[:, None], axis=1) != 1
    best_confidence = probability[np.arange(selected_count), best_cluster]
    radius_array = np.asarray(envelope_radius, dtype=np.float64)
    attach = (
        ~exact_ties
        & (best_confidence >= ASSIGNMENT_THRESHOLD)
        & (minimum_distance <= radius_array[best_cluster])
    )
    full_cluster_index = np.where(attach, best_cluster, -1).astype(np.int64)
    assignment_confidence = np.where(attach, best_confidence, 0.0).astype(np.float64)

    raw_seed_own_probability = np.full(selected_count, np.nan, dtype=np.float64)
    raw_selected_rows = np.flatnonzero(raw_seed_cluster_index >= 0)
    if len(raw_selected_rows):
        own = raw_seed_cluster_index[raw_selected_rows]
        raw_seed_own_probability[raw_selected_rows] = probability[
            raw_selected_rows, own
        ]

    if mode == RAW_ANCHORED_ENVELOPE:
        full_cluster_index[raw_selected_rows] = raw_seed_cluster_index[raw_selected_rows]
        assignment_confidence[raw_selected_rows] = raw_seed_own_probability[
            raw_selected_rows
        ]
        trusted_cluster_index = raw_seed_cluster_index.copy()
    else:
        trusted_cluster_index = np.where(
            (raw_seed_cluster_index >= 0)
            & (full_cluster_index == raw_seed_cluster_index)
            & (raw_seed_own_probability >= ASSIGNMENT_THRESHOLD),
            raw_seed_cluster_index,
            -1,
        ).astype(np.int64)

    pre_remap_full = full_cluster_index.copy()
    full, trusted, emitted_remap = _remap_emitted_candidates(
        full_cluster_index, trusted_cluster_index, raw_cluster_ids
    )
    assignment_confidence[full < 0] = 0.0

    rows: list[dict[str, Any]] = []
    for old_index, candidate_id in emitted_remap.items():
        raw_id = raw_cluster_ids[old_index]
        raw_mask = raw_seed_cluster_index == old_index
        trusted_mask = trusted == candidate_id
        full_mask = full == candidate_id
        raw_final = full[raw_mask]
        retained = int(np.count_nonzero(raw_final == candidate_id))
        reassigned = int(np.count_nonzero((raw_final >= 0) & (raw_final != candidate_id)))
        rejected = int(np.count_nonzero(raw_final < 0))
        rows.append(
            {
                "candidate_id": int(candidate_id),
                "raw_cluster_id": int(raw_id),
                "raw_cluster_index": int(old_index),
                "medoid_sample_row": int(medoid_sample_rows[old_index]),
                "medoid_selected_local_index": int(medoid_selected_local[old_index]),
                "medoid_global_index": int(selected[medoid_selected_local[old_index]]),
                "envelope_radius": float(radius_array[old_index]),
                "raw_seed_count": int(np.count_nonzero(raw_mask)),
                "trusted_core_count": int(np.count_nonzero(trusted_mask)),
                "full_point_count": int(np.count_nonzero(full_mask)),
                "raw_seed_retained_count": retained,
                "raw_seed_reassigned_count": reassigned,
                "raw_seed_rejected_count": rejected,
                "raw_seed_own_probability_mean": float(
                    np.mean(raw_seed_own_probability[raw_mask])
                ),
                "assignment_confidence_mean": float(
                    np.mean(assignment_confidence[full_mask])
                ),
            }
        )

    core_violations = int(np.count_nonzero((trusted >= 0) & (trusted != full)))
    if core_violations:
        raise AssertionError("repair produced trusted core outside its full candidate")
    diagnostics: dict[str, Any] = {
        "legacy_passthrough": False,
        "selected_point_count": int(selected_count),
        "sampled_point_count": int(len(sampled)),
        "raw_cluster_count": int(len(raw_cluster_ids)),
        "candidate_count": int(len(rows)),
        "dropped_raw_cluster_count": int(len(raw_cluster_ids) - len(rows)),
        "exact_tie_point_count": int(np.count_nonzero(exact_ties)),
        "instance_distance_max": float(instance_max),
        "spatial_distance_max": float(spatial_max),
        "weights": {
            "instance": INSTANCE_WEIGHT,
            "spatial": SPATIAL_WEIGHT,
            "semantic": SEMANTIC_WEIGHT,
        },
        "assignment_threshold": ASSIGNMENT_THRESHOLD,
        "assignment_temperature": ASSIGNMENT_TEMPERATURE,
        "envelope_quantile": ENVELOPE_QUANTILE,
        "trusted_core_outside_full_count": core_violations,
        "core_contract_enforced": True,
        "pre_remap_assigned_point_count": int(
            np.count_nonzero(pre_remap_full >= 0)
        ),
        "final_assigned_point_count": int(np.count_nonzero(full >= 0)),
    }
    return CandidateRepairResult(
        mode=mode,
        selected_global_indices=_readonly(selected, np.int64),
        raw_seed_cluster_index=_readonly(raw_seed_cluster_index, np.int64),
        trusted_core_labels=_readonly(trusted, np.int64),
        full_candidate_labels=_readonly(full, np.int64),
        assignment_confidence=_readonly(assignment_confidence, np.float64),
        raw_seed_own_probability=_readonly(raw_seed_own_probability, np.float64),
        raw_cluster_ids=raw_cluster_ids,
        candidates=tuple(rows),
        diagnostics=diagnostics,
    )


__all__ = [
    "ASSIGNMENT_THRESHOLD",
    "ASSIGNMENT_TEMPERATURE",
    "CONSISTENT_ENVELOPE",
    "CandidateRepairResult",
    "CandidateRepairScene",
    "CandidateRepairTrace",
    "ENVELOPE_QUANTILE",
    "INSTANCE_WEIGHT",
    "LEGACY",
    "RAW_ANCHORED_ENVELOPE",
    "REPAIR_MODES",
    "SEMANTIC_WEIGHT",
    "SPATIAL_WEIGHT",
    "repair_class_candidates",
]
