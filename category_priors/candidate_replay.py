from __future__ import annotations

"""Replay a frozen candidate selection through the teacher denoiser.

This module starts from the exact ``global_pre_knn`` partition and a frozen
candidate pool. Experimental conditions pass different accepted candidate IDs
but otherwise execute the same path:

1. accepted candidate masks overwrite ``global_pre_knn`` before KNN;
2. conflicts are resolved by frozen Q, total trusted-core support, and stable
   candidate ID, in that order;
3. the complete scene is passed to the existing historical KNN and count
   filter implementation;
4. nothing is protected or inserted after those operations.

Candidate classes and AP scores are frozen branch metadata.  There is no
second semantic vote in this replay, and there is no GT, prior, evaluator, or
file-system input.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .legacy_candidate_replay import (
    GLOBAL_KNN_K,
    GLOBAL_MIN_COUNT,
    LegacyKNNFilterResult,
    legacy_knn_filter,
)


@dataclass(frozen=True)
class LegacyReplayCandidate:
    """One candidate on the complete scene point axis.

    ``candidate_id`` is also the stable final conflict tie-break key.  The
    point collections contain scene point indices, not masks and not positions
    in a class-local selected axis.  ``trusted_core_indices`` must be a subset
    of ``full_point_indices``.
    """

    candidate_id: int
    branch_class: str
    q_score: float
    full_point_indices: Any
    trusted_core_indices: Any


@dataclass(frozen=True)
class CandidateSurvivalDiagnostic:
    """One candidate's complete pre-KNN to post-filter survival funnel."""

    candidate_id: int
    accepted: bool
    branch_class: str
    q_score: float
    raw_label: int | None
    full_point_count: int
    trusted_core_count: int
    overlap_point_count: int
    pre_knn_owned_count: int
    pre_knn_owned_trusted_core_count: int
    pre_knn_conflict_lost_count: int
    post_knn_total_count: int
    post_knn_retained_owned_count: int
    post_knn_gained_outside_count: int
    post_filter_total_count: int
    post_filter_retained_owned_count: int
    post_filter_gained_outside_count: int
    label_present_post_knn: bool
    label_present_post_filter: bool
    survived_post_knn: bool
    survived_post_filter: bool
    final_id: int | None
    final_class: str | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CandidateLegacyReplayResult:
    """Raw-label replay result before any outer strict-output projection."""

    source_labels: np.ndarray
    after_knn: np.ndarray
    after_filter: np.ndarray
    accepted_candidate_ids: tuple[int, ...]
    candidate_raw_labels: Mapping[int, int]
    candidate_class_by_raw_label: Mapping[int, str]
    candidate_score_by_raw_label: Mapping[int, float]
    candidates: tuple[CandidateSurvivalDiagnostic, ...]
    legacy: LegacyKNNFilterResult
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _ValidatedCandidate:
    candidate_id: int
    branch_class: str
    q_score: float
    full: np.ndarray
    core: np.ndarray


def _as_numpy(value: Any, dtype: Any) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _readonly(value: Any, dtype: Any = np.int64) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _integer_id(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _point_indices(value: Any, point_count: int, name: str) -> np.ndarray:
    indices = _as_numpy(value, np.int64)
    if indices.ndim != 1:
        raise ValueError(f"{name} must be a vector of scene point indices")
    if len(indices) and (
        int(indices.min()) < 0 or int(indices.max()) >= int(point_count)
    ):
        raise ValueError(f"{name} contains an index outside the scene point axis")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not repeat a point index")
    return np.sort(indices).astype(np.int64, copy=False)


def _validate_candidates(
    candidates: Sequence[LegacyReplayCandidate], point_count: int
) -> tuple[_ValidatedCandidate, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence")
    validated: list[_ValidatedCandidate] = []
    observed: set[int] = set()
    for row in candidates:
        if not isinstance(row, LegacyReplayCandidate):
            raise TypeError("candidates must contain LegacyReplayCandidate values")
        candidate_id = _integer_id(row.candidate_id, "candidate_id")
        if candidate_id in observed:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        observed.add(candidate_id)
        branch_class = str(row.branch_class)
        if not branch_class:
            raise ValueError(f"candidate {candidate_id} has an empty branch_class")
        q_score = float(row.q_score)
        if not np.isfinite(q_score) or not 0.0 <= q_score <= 1.0:
            raise ValueError(f"candidate {candidate_id} q_score must be in [0, 1]")
        full = _point_indices(
            row.full_point_indices, point_count, "full_point_indices"
        )
        core = _point_indices(
            row.trusted_core_indices, point_count, "trusted_core_indices"
        )
        if not len(full):
            raise ValueError(f"candidate {candidate_id} has an empty full mask")
        if len(core) and not np.all(np.isin(core, full)):
            raise ValueError(
                f"candidate {candidate_id} trusted core is not a subset of full"
            )
        validated.append(
            _ValidatedCandidate(candidate_id, branch_class, q_score, full, core)
        )
    return tuple(sorted(validated, key=lambda row: row.candidate_id))


def _accepted_ids(
    accepted_candidate_ids: Sequence[int], known: Mapping[int, _ValidatedCandidate]
) -> tuple[int, ...]:
    if isinstance(accepted_candidate_ids, (str, bytes)):
        raise TypeError("accepted_candidate_ids must be an integer sequence")
    normalized = tuple(
        _integer_id(value, "accepted candidate ID")
        for value in accepted_candidate_ids
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError("accepted_candidate_ids contains duplicates")
    unknown = sorted(set(normalized) - set(known))
    if unknown:
        raise ValueError(f"accepted_candidate_ids contains unknown IDs: {unknown}")
    return tuple(sorted(normalized))


def _candidate_ownership(
    point_count: int,
    accepted: tuple[int, ...],
    rows: Mapping[int, _ValidatedCandidate],
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Return owner candidate ID, overlap count, and conflict losses.

    The frozen priority is descending Q, then descending total trusted-core
    count of the candidate, then ascending stable candidate ID.  The second
    key is candidate-level evidence; it must not depend on whether the current
    overlap point itself happens to be in the core.
    """

    owner = np.full(point_count, -1, dtype=np.int64)
    owner_q = np.full(point_count, -np.inf, dtype=np.float64)
    owner_core_count = np.full(point_count, -1, dtype=np.int64)
    membership_count = np.zeros(point_count, dtype=np.int32)
    full_masks: dict[int, np.ndarray] = {}
    core_masks: dict[int, np.ndarray] = {}
    for candidate_id in accepted:
        row = rows[candidate_id]
        full_mask = np.zeros(point_count, dtype=np.bool_)
        core_mask = np.zeros(point_count, dtype=np.bool_)
        full_mask[row.full] = True
        core_mask[row.core] = True
        full_masks[candidate_id] = full_mask
        core_masks[candidate_id] = core_mask
        membership_count[row.full] += 1

    for candidate_id in accepted:
        row = rows[candidate_id]
        full_mask = full_masks[candidate_id]
        core_count = int(len(row.core))
        wins = full_mask & (
            (row.q_score > owner_q)
            | (
                (row.q_score == owner_q)
                & (
                    (core_count > owner_core_count)
                    | (
                        (core_count == owner_core_count)
                        & ((owner < 0) | (candidate_id < owner))
                    )
                )
            )
        )
        owner[wins] = candidate_id
        owner_q[wins] = row.q_score
        owner_core_count[wins] = core_count

    conflict_losses = {
        candidate_id: int(
            np.count_nonzero(
                full_masks[candidate_id]
                & (membership_count > 1)
                & (owner != candidate_id)
            )
        )
        for candidate_id in accepted
    }
    return owner, membership_count, conflict_losses


def replay_candidates_through_legacy(
    *,
    xyz_scene: Any,
    global_pre_knn: Any,
    candidates: Sequence[LegacyReplayCandidate],
    accepted_candidate_ids: Sequence[int],
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    chunk_size: int = 8_192,
) -> CandidateLegacyReplayResult:
    """Inject accepted candidates, then run the unchanged legacy KNN/filter.

    Rejected candidates do not modify ``global_pre_knn``.  Accepted candidates
    receive unique raw labels above the maximum global label.  These labels go
    through KNN and filtering with every other point; no point is protected and
    no candidate is inserted again afterward.
    """

    xyz = _as_numpy(xyz_scene, np.float64)
    global_labels = _as_numpy(global_pre_knn, np.int64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("xyz_scene must be an N x 3 matrix")
    if global_labels.shape != (len(xyz),):
        raise ValueError("global_pre_knn must use the scene point axis")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite")
    if np.any(global_labels < -1):
        raise ValueError("global_pre_knn may only use -1 as its negative label")

    validated = _validate_candidates(candidates, len(xyz))
    rows = {row.candidate_id: row for row in validated}
    accepted = _accepted_ids(accepted_candidate_ids, rows)
    owner, membership_count, conflict_losses = _candidate_ownership(
        len(xyz), accepted, rows
    )

    source = global_labels.copy()
    maximum_global = (
        int(global_labels[global_labels >= 0].max())
        if np.any(global_labels >= 0)
        else -1
    )
    candidate_raw_labels = {
        candidate_id: maximum_global + ordinal + 1
        for ordinal, candidate_id in enumerate(accepted)
    }
    for candidate_id, raw_label in candidate_raw_labels.items():
        source[owner == candidate_id] = raw_label

    legacy = legacy_knn_filter(
        xyz,
        source,
        k=k,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    diagnostics_rows: list[CandidateSurvivalDiagnostic] = []
    for row in validated:
        accepted_row = row.candidate_id in candidate_raw_labels
        raw_label = candidate_raw_labels.get(row.candidate_id)
        owned = owner == row.candidate_id if accepted_row else np.zeros(len(xyz), bool)
        if raw_label is None:
            after_knn_mask = np.zeros(len(xyz), dtype=np.bool_)
            after_filter_mask = np.zeros(len(xyz), dtype=np.bool_)
        else:
            after_knn_mask = np.asarray(legacy.after_knn) == raw_label
            after_filter_mask = np.asarray(legacy.after_filter) == raw_label
        retained_knn = int(np.count_nonzero(owned & after_knn_mask))
        retained_filter = int(np.count_nonzero(owned & after_filter_mask))
        post_knn_total = int(np.count_nonzero(after_knn_mask))
        post_filter_total = int(np.count_nonzero(after_filter_mask))
        full_mask = np.zeros(len(xyz), dtype=np.bool_)
        core_mask = np.zeros(len(xyz), dtype=np.bool_)
        full_mask[row.full] = True
        core_mask[row.core] = True
        diagnostics_rows.append(
            CandidateSurvivalDiagnostic(
                candidate_id=row.candidate_id,
                accepted=accepted_row,
                branch_class=row.branch_class,
                q_score=row.q_score,
                raw_label=raw_label,
                full_point_count=len(row.full),
                trusted_core_count=len(row.core),
                overlap_point_count=int(
                    np.count_nonzero(full_mask & (membership_count > 1))
                )
                if accepted_row
                else 0,
                pre_knn_owned_count=int(np.count_nonzero(owned)),
                pre_knn_owned_trusted_core_count=int(
                    np.count_nonzero(owned & core_mask)
                ),
                pre_knn_conflict_lost_count=conflict_losses.get(
                    row.candidate_id, 0
                ),
                post_knn_total_count=post_knn_total,
                post_knn_retained_owned_count=retained_knn,
                post_knn_gained_outside_count=post_knn_total - retained_knn,
                post_filter_total_count=post_filter_total,
                post_filter_retained_owned_count=retained_filter,
                post_filter_gained_outside_count=post_filter_total - retained_filter,
                label_present_post_knn=post_knn_total > 0,
                label_present_post_filter=post_filter_total > 0,
                survived_post_knn=retained_knn > 0,
                survived_post_filter=retained_filter > 0,
                final_id=raw_label if post_filter_total else None,
                final_class=row.branch_class if post_filter_total else None,
            )
        )

    candidate_class = {
        raw_label: rows[candidate_id].branch_class
        for candidate_id, raw_label in candidate_raw_labels.items()
    }
    candidate_score = {
        raw_label: rows[candidate_id].q_score
        for candidate_id, raw_label in candidate_raw_labels.items()
    }
    b0_exact = not accepted and np.array_equal(source, global_labels)
    if not accepted:
        # This is an executable identity gate, not merely a reported intent.
        baseline = legacy_knn_filter(
            xyz,
            global_labels,
            k=k,
            min_count=min_count,
            chunk_size=chunk_size,
        )
        b0_exact = bool(
            np.array_equal(legacy.after_knn, baseline.after_knn)
            and np.array_equal(legacy.after_filter, baseline.after_filter)
        )
        if not b0_exact:
            raise RuntimeError("zero-candidate replay is not pointwise equal to B0")

    result_diagnostics = MappingProxyType(
        {
            "candidate_count": len(validated),
            "accepted_candidate_count": len(accepted),
            "accepted_candidate_ids": accepted,
            "overlap_point_count": int(np.count_nonzero(membership_count > 1)),
            "pre_knn_branch_point_count": int(np.count_nonzero(owner >= 0)),
            "post_knn_surviving_candidate_count": int(
                sum(row.survived_post_knn for row in diagnostics_rows)
            ),
            "post_filter_surviving_candidate_count": int(
                sum(row.survived_post_filter for row in diagnostics_rows)
            ),
            "knn_k_effective": legacy.k_effective,
            "filter_min_count": legacy.min_count,
            "protected_or_reinserted_point_count": 0,
            "secondary_class_vote_applied": False,
            "zero_candidate_b0_pointwise_exact": b0_exact if not accepted else None,
        }
    )
    return CandidateLegacyReplayResult(
        source_labels=_readonly(source),
        after_knn=legacy.after_knn,
        after_filter=legacy.after_filter,
        accepted_candidate_ids=accepted,
        candidate_raw_labels=MappingProxyType(dict(candidate_raw_labels)),
        candidate_class_by_raw_label=MappingProxyType(candidate_class),
        candidate_score_by_raw_label=MappingProxyType(candidate_score),
        candidates=tuple(diagnostics_rows),
        legacy=legacy,
        diagnostics=result_diagnostics,
    )


__all__ = [
    "CandidateLegacyReplayResult",
    "CandidateSurvivalDiagnostic",
    "LegacyReplayCandidate",
    "replay_candidates_through_legacy",
]
