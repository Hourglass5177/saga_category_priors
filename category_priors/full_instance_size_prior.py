from __future__ import annotations

"""Pure algorithms for the full-instance size-prior experiment.

This module starts from the frozen pre-KNN ``merged_partition`` snapshot.  It
does not create clusters and it does not inspect evaluation annotations.  Its
only responsibilities are to:

* materialize every raw foreground instance as one immutable candidate;
* apply the registered 33-channel pre-KNN vote eligibility contract;
* score the exact same candidates with no, global, or class size statistics;
* verify that the two size arms differ only in their derived fields; and
* restore selected candidates into a copy of ``post_filter``.

In particular, no branch-specific confidence array is accepted here.  The
only common score is the directly comparable 2D vote ratio ``Q``.
"""

import copy
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .category_candidate_prior_v2 import size_platform_compatibility
from .category_denoise import pca_sorted_extents_m
from .teacher_prior import SAGA20_CLASSES


EXPECTED_FOREGROUND_CLASSES = 32
EXPECTED_VOTE_CHANNELS = 33
BACKGROUND_CHANNEL = 32
MIN_WINNER_RATIO = 0.30

ScoreMode = Literal["q-only", "global-size", "class-size"]

DERIVED_SCORE_FIELDS = frozenset(
    {
        "score_mode",
        "size_lookup_class",
        "size_fallback_global",
        "prior_applied",
        "G",
        "S",
    }
)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def _integer_labels(value: Any, name: str) -> np.ndarray:
    raw = _as_numpy(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError(f"{name} must contain integer labels")
    labels = np.asarray(raw, dtype=np.int64)
    if np.any(labels < -1):
        raise ValueError(f"{name} may only use -1 as a negative label")
    return labels


def _validated_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in class_names)
    if len(names) != EXPECTED_FOREGROUND_CLASSES:
        raise ValueError(
            "pre-KNN voting requires exactly "
            f"{EXPECTED_FOREGROUND_CLASSES} foreground classes"
        )
    if len(set(names)) != len(names):
        raise ValueError("class_names must be unique")
    return names


def _normalized_integer_keys(
    values: Mapping[Any, Any] | None, name: str
) -> dict[int, Any]:
    normalized: dict[int, Any] = {}
    if values is None:
        return normalized
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for raw_key, value in values.items():
        if isinstance(raw_key, (bool, np.bool_)):
            raise TypeError(f"{name} keys must be non-negative integers")
        try:
            key = int(raw_key)
            exact = float(raw_key)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} keys must be non-negative integers") from exc
        if not np.isfinite(exact) or exact != float(key) or key < 0:
            raise ValueError(f"{name} keys must be non-negative integers")
        if key in normalized:
            raise ValueError(f"{name} contains duplicate normalized key {key}")
        normalized[key] = value
    return normalized


@dataclass(frozen=True)
class PreVoteDecision:
    """Deterministic interpretation of one raw 33-channel vote histogram."""

    vote_histogram: tuple[float, ...]
    predicted_class_index: int | None
    predicted_class: str | None
    winner_ratio: float
    background_ratio: float
    eligible: bool
    eligibility_reason: str

    @property
    def Q(self) -> float:
        return self.winner_ratio


@dataclass(frozen=True)
class FullInstanceCandidate:
    """One raw instance in the frozen pre-KNN partition."""

    scene_id: str
    candidate_id: int
    raw_instance_id: int
    source: str
    member_indices: np.ndarray
    point_count: int
    metric_extents_m: tuple[float, float, float]
    vote_histogram: tuple[float, ...]
    predicted_class_index: int | None
    predicted_class: str | None
    winner_ratio: float
    background_ratio: float
    eligible: bool
    eligibility_reason: str

    @property
    def Q(self) -> float:
        return self.winner_ratio

    def to_row(self, *, include_members: bool = False) -> dict[str, Any]:
        """Return a stable row; omit the large index vector by default."""

        row: dict[str, Any] = {
            "scene_id": self.scene_id,
            "candidate_id": self.candidate_id,
            "raw_instance_id": self.raw_instance_id,
            "source": self.source,
            "point_count": self.point_count,
            "metric_extents_m": self.metric_extents_m,
            "vote_histogram": self.vote_histogram,
            "predicted_class_index": self.predicted_class_index,
            "predicted_class": self.predicted_class,
            "winner_ratio": self.winner_ratio,
            "background_ratio": self.background_ratio,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "Q": self.Q,
        }
        if include_members:
            row["member_indices"] = self.member_indices
        return row


@dataclass(frozen=True)
class FullInstanceSnapshot:
    """All foreground candidates and stable construction diagnostics."""

    candidates: tuple[FullInstanceCandidate, ...]
    diagnostics: Mapping[str, Any]

    def rows(self, *, include_members: bool = False) -> tuple[dict[str, Any], ...]:
        return tuple(
            candidate.to_row(include_members=include_members)
            for candidate in self.candidates
        )


@dataclass(frozen=True)
class SameBankIdentityAudit:
    candidate_ids: tuple[int, ...]
    raw_instance_ids: tuple[int, ...]
    q_values: tuple[float, ...]
    member_point_count: int
    compared_row_count: int
    bank_identity_equal: bool = True
    q_unchanged: bool = True


@dataclass(frozen=True)
class SameBankFullInstanceScores:
    q_only: tuple[dict[str, Any], ...]
    global_size: tuple[dict[str, Any], ...]
    class_size: tuple[dict[str, Any], ...]
    identity: SameBankIdentityAudit


@dataclass(frozen=True)
class RestorationResult:
    point_labels: np.ndarray
    diagnostics: Mapping[str, Any]


def evaluate_pre_vote(
    vote_histogram: Sequence[float],
    class_names: Sequence[str],
    *,
    saga20_classes: Collection[str] = SAGA20_CLASSES,
    minimum_winner_ratio: float = MIN_WINNER_RATIO,
) -> PreVoteDecision:
    """Apply the frozen 32-foreground-plus-background eligibility contract.

    ``Q`` is the unique winning foreground count divided by the sum over all
    33 channels.  The candidate remains materialized when it is ineligible.
    """

    names = _validated_class_names(class_names)
    saga20 = frozenset(str(name) for name in saga20_classes)
    unknown = sorted(saga20.difference(names))
    if unknown:
        raise ValueError(f"SAGA20 classes absent from class_names: {unknown}")
    threshold = float(minimum_winner_ratio)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_winner_ratio must be finite and in [0, 1]")

    votes = np.asarray(vote_histogram, dtype=np.float64)
    if votes.shape != (EXPECTED_VOTE_CHANNELS,):
        raise ValueError(
            f"vote_histogram must contain {EXPECTED_VOTE_CHANNELS} channels"
        )
    if not np.isfinite(votes).all() or np.any(votes < 0.0):
        raise ValueError("vote_histogram must be finite and non-negative")
    total = float(votes.sum())
    if not np.isfinite(total):
        raise ValueError("vote_histogram total must be finite")
    if total <= 0.0:
        return PreVoteDecision(
            vote_histogram=tuple(float(value) for value in votes),
            predicted_class_index=None,
            predicted_class=None,
            winner_ratio=0.0,
            background_ratio=0.0,
            eligible=False,
            eligibility_reason="no_votes",
        )

    foreground = votes[:EXPECTED_FOREGROUND_CLASSES]
    winner_count = float(foreground.max())
    winners = np.flatnonzero(foreground == winner_count)
    q_value = winner_count / total
    background_ratio = float(votes[BACKGROUND_CHANNEL] / total)
    if len(winners) != 1:
        predicted_index: int | None = None
        predicted_class: str | None = None
        eligible = False
        reason = "foreground_tie"
    else:
        predicted_index = int(winners[0])
        predicted_class = names[predicted_index]
        if predicted_class not in saga20:
            eligible = False
            reason = "winner_not_saga20"
        elif q_value < threshold:
            eligible = False
            reason = "winner_ratio_below_threshold"
        elif q_value <= background_ratio:
            eligible = False
            reason = "background_not_lower"
        else:
            eligible = True
            reason = "eligible"

    return PreVoteDecision(
        vote_histogram=tuple(float(value) for value in votes),
        predicted_class_index=predicted_index,
        predicted_class=predicted_class,
        winner_ratio=float(q_value),
        background_ratio=background_ratio,
        eligible=eligible,
        eligibility_reason=reason,
    )


def _vote_histogram_for_id(
    vote_histograms: Mapping[Any, Sequence[float]] | np.ndarray,
    raw_instance_id: int,
) -> Sequence[float]:
    if isinstance(vote_histograms, Mapping):
        normalized = _normalized_integer_keys(vote_histograms, "vote_histograms")
        if raw_instance_id not in normalized:
            raise ValueError(
                f"vote_histograms is missing raw instance {raw_instance_id}"
            )
        return normalized[raw_instance_id]
    array = _as_numpy(vote_histograms)
    if array.ndim != 2 or array.shape[1] != EXPECTED_VOTE_CHANNELS:
        raise ValueError("vote_histograms array must have shape (N, 33)")
    if raw_instance_id >= len(array):
        raise ValueError(
            f"vote_histograms is missing raw instance {raw_instance_id}"
        )
    return array[raw_instance_id]


def build_full_instance_candidates(
    merged_partition: Any,
    xyz_scene: Any,
    scene_scale_m_per_unit: float,
    vote_histograms: Mapping[Any, Sequence[float]] | np.ndarray,
    class_names: Sequence[str],
    *,
    scene_id: str,
    branch_instance_classes: Mapping[Any, Any] | None = None,
    saga20_classes: Collection[str] = SAGA20_CLASSES,
) -> FullInstanceSnapshot:
    """Materialize every non-negative raw instance from ``merged_partition``.

    Candidate IDs intentionally equal raw instance IDs.  This preserves the
    frozen snapshot identity and makes restoration auditable without a second
    translation table.
    """

    labels = _integer_labels(merged_partition, "merged_partition")
    xyz = _as_numpy(xyz_scene, np.float64)
    if xyz.ndim != 2 or xyz.shape != (len(labels), 3):
        raise ValueError("xyz_scene must have shape (point_count, 3)")
    if not np.isfinite(xyz).all():
        raise ValueError("xyz_scene must be finite")
    scale = float(scene_scale_m_per_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scene_scale_m_per_unit must be finite and positive")
    scene = str(scene_id)
    if not scene:
        raise ValueError("scene_id must not be empty")
    names = _validated_class_names(class_names)
    branch_map = _normalized_integer_keys(
        branch_instance_classes, "branch_instance_classes"
    )

    foreground_indices = np.flatnonzero(labels >= 0)
    if len(foreground_indices):
        # Group once rather than scanning the million-point scene separately
        # for every raw instance.  Stable sorting also keeps member indices in
        # their original point order inside each candidate.
        order = np.argsort(labels[foreground_indices], kind="stable")
        grouped_indices = foreground_indices[order]
        grouped_labels = labels[grouped_indices]
        unique_ids, starts, counts = np.unique(
            grouped_labels, return_index=True, return_counts=True
        )
        raw_ids = tuple(int(value) for value in unique_ids)
        members_by_id = {
            int(raw_id): _readonly(
                grouped_indices[start : start + count].astype(np.int64, copy=False)
            )
            for raw_id, start, count in zip(unique_ids, starts, counts)
        }
    else:
        raw_ids = ()
        members_by_id = {}
    candidates: list[FullInstanceCandidate] = []
    eligibility_counts: dict[str, int] = {}
    for raw_id in raw_ids:
        members = members_by_id[raw_id]
        decision = evaluate_pre_vote(
            _vote_histogram_for_id(vote_histograms, raw_id),
            names,
            saga20_classes=saga20_classes,
        )
        extents = pca_sorted_extents_m(xyz[members], scale)
        candidate = FullInstanceCandidate(
            scene_id=scene,
            candidate_id=raw_id,
            raw_instance_id=raw_id,
            source="other_classes" if raw_id in branch_map else "global",
            member_indices=members,
            point_count=len(members),
            metric_extents_m=tuple(float(value) for value in extents),
            vote_histogram=decision.vote_histogram,
            predicted_class_index=decision.predicted_class_index,
            predicted_class=decision.predicted_class,
            winner_ratio=decision.winner_ratio,
            background_ratio=decision.background_ratio,
            eligible=decision.eligible,
            eligibility_reason=decision.eligibility_reason,
        )
        candidates.append(candidate)
        eligibility_counts[decision.eligibility_reason] = (
            eligibility_counts.get(decision.eligibility_reason, 0) + 1
        )

    unused_branch_ids = tuple(sorted(set(branch_map).difference(raw_ids)))
    foreground_count = int(np.count_nonzero(labels >= 0))
    if sum(candidate.point_count for candidate in candidates) != foreground_count:
        raise AssertionError("candidate members do not partition foreground labels")
    diagnostics: dict[str, Any] = {
        "scene_id": scene,
        "point_count": len(labels),
        "foreground_point_count": foreground_count,
        "background_point_count": int(np.count_nonzero(labels == -1)),
        "candidate_count": len(candidates),
        "eligible_candidate_count": sum(
            int(candidate.eligible) for candidate in candidates
        ),
        "global_candidate_count": sum(
            int(candidate.source == "global") for candidate in candidates
        ),
        "other_classes_candidate_count": sum(
            int(candidate.source == "other_classes") for candidate in candidates
        ),
        "raw_instance_ids": raw_ids,
        "unused_branch_instance_ids": unused_branch_ids,
        "eligibility_reason_counts": {
            key: eligibility_counts[key] for key in sorted(eligibility_counts)
        },
        "candidate_members_mutually_exclusive": True,
    }
    return FullInstanceSnapshot(tuple(candidates), diagnostics)


def _candidate_row(
    candidate: FullInstanceCandidate | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(candidate, FullInstanceCandidate):
        return candidate.to_row(include_members=True)
    if not isinstance(candidate, Mapping):
        raise TypeError("candidates must be FullInstanceCandidate objects or mappings")
    return copy.deepcopy(dict(candidate))


def _global_size_node(priors: Mapping[str, Any]) -> Mapping[str, Any]:
    node = priors.get("global")
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise TypeError("category priors are missing a global shrunk node")
    return node


def _class_size_node(
    priors: Mapping[str, Any], class_name: str
) -> tuple[Mapping[str, Any], bool]:
    categories = priors.get("categories")
    if not isinstance(categories, Mapping) or class_name not in categories:
        return _global_size_node(priors), True
    node = categories[class_name]
    if not isinstance(node, Mapping) or not isinstance(node.get("shrunk"), Mapping):
        raise TypeError(f"category prior {class_name!r} has no shrunk node")
    return node, False


def _validated_candidate_base(row: Mapping[str, Any]) -> tuple[int, float, bool]:
    try:
        raw_id = int(row["raw_instance_id"])
        candidate_id = int(row["candidate_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("candidate_id and raw_instance_id must be integers") from exc
    if raw_id < 0 or candidate_id != raw_id:
        raise ValueError("candidate_id must equal a non-negative raw_instance_id")
    try:
        q_value = float(row["Q"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("candidate Q must be numeric") from exc
    if not np.isfinite(q_value) or not 0.0 <= q_value <= 1.0:
        raise ValueError("candidate Q must be finite and in [0, 1]")
    eligible = row.get("eligible")
    if not isinstance(eligible, (bool, np.bool_)):
        raise TypeError("candidate eligible must be boolean")
    return raw_id, q_value, bool(eligible)


def score_full_instance_candidates(
    candidates: Sequence[FullInstanceCandidate | Mapping[str, Any]],
    priors: Mapping[str, Any],
    mode: ScoreMode,
) -> tuple[dict[str, Any], ...]:
    """Score one frozen bank with Q-only, global, or predicted-class size.

    Ineligible candidates are deliberately retained but receive ``G=1`` and
    ``S=Q`` in every arm.  Consequently the prior cannot alter baseline
    handling for background winners, ties, low-confidence winners, or classes
    outside SAGA20.
    """

    if mode not in {"q-only", "global-size", "class-size"}:
        raise ValueError("mode must be 'q-only', 'global-size', or 'class-size'")
    if not isinstance(priors, Mapping):
        raise TypeError("priors must be a mapping")
    global_node = _global_size_node(priors)

    rows: list[dict[str, Any]] = []
    observed_ids: set[int] = set()
    for candidate in candidates:
        row = _candidate_row(candidate)
        collisions = DERIVED_SCORE_FIELDS.intersection(row)
        if collisions:
            raise ValueError(
                "candidate already contains derived score fields: "
                + ", ".join(sorted(collisions))
            )
        raw_id, q_value, eligible = _validated_candidate_base(row)
        if raw_id in observed_ids:
            raise ValueError("candidate raw_instance_id values must be unique")
        observed_ids.add(raw_id)

        lookup_class: str | None = None
        fallback = False
        prior_applied = False
        if mode == "q-only" or not eligible:
            g_value = 1.0
        elif mode == "global-size":
            lookup_class = "global"
            prior_applied = True
            g_value = size_platform_compatibility(row, global_node)
        else:
            predicted_class = row.get("predicted_class")
            if not isinstance(predicted_class, str) or not predicted_class:
                raise ValueError("eligible candidate requires predicted_class")
            node, fallback = _class_size_node(priors, predicted_class)
            lookup_class = "global" if fallback else predicted_class
            prior_applied = True
            g_value = size_platform_compatibility(row, node)

        scored = copy.deepcopy(row)
        scored.update(
            {
                "score_mode": mode,
                "size_lookup_class": lookup_class,
                "size_fallback_global": fallback,
                "prior_applied": prior_applied,
                "G": float(g_value),
                "S": float(q_value * g_value),
            }
        )
        rows.append(scored)
    rows.sort(key=lambda item: int(item["raw_instance_id"]))
    return tuple(rows)


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            return False
        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            return False
        return all(_values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, float) or not isinstance(right, float):
            return False
        if np.isnan(left) and np.isnan(right):
            return True
    try:
        return bool(type(left) is type(right) and left == right)
    except (TypeError, ValueError):
        return False


def verify_same_bank_size_scores(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
) -> SameBankIdentityAudit:
    """Require exact bank/Q identity after removing arm-derived fields."""

    if len(left_rows) != len(right_rows):
        raise ValueError("score arms have different candidate row counts")
    left_sorted = sorted(left_rows, key=lambda row: int(row["raw_instance_id"]))
    right_sorted = sorted(right_rows, key=lambda row: int(row["raw_instance_id"]))
    candidate_ids: list[int] = []
    raw_ids: list[int] = []
    q_values: list[float] = []
    member_count = 0
    for index, (left, right) in enumerate(zip(left_sorted, right_sorted)):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError("score rows must be mappings")
        if not DERIVED_SCORE_FIELDS.issubset(left) or not DERIVED_SCORE_FIELDS.issubset(
            right
        ):
            raise ValueError("score rows are missing derived fields")
        left_base = {
            key: value for key, value in left.items() if key not in DERIVED_SCORE_FIELDS
        }
        right_base = {
            key: value for key, value in right.items() if key not in DERIVED_SCORE_FIELDS
        }
        if not _values_equal(left_base, right_base):
            differing = sorted(
                key
                for key in set(left_base).union(right_base)
                if key not in left_base
                or key not in right_base
                or not _values_equal(left_base.get(key), right_base.get(key))
            )
            raise ValueError(
                f"score-arm bank identity differs at row {index}: {differing}"
            )
        raw_id, q_value, _ = _validated_candidate_base(left_base)
        members = np.asarray(left_base.get("member_indices", ()), dtype=np.int64)
        candidate_ids.append(int(left_base["candidate_id"]))
        raw_ids.append(raw_id)
        q_values.append(q_value)
        member_count += len(members)
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError("score rows contain duplicate raw_instance_id values")
    return SameBankIdentityAudit(
        candidate_ids=tuple(candidate_ids),
        raw_instance_ids=tuple(raw_ids),
        q_values=tuple(q_values),
        member_point_count=member_count,
        compared_row_count=len(raw_ids),
    )


def score_same_bank_size_priors(
    candidates: Sequence[FullInstanceCandidate | Mapping[str, Any]],
    priors: Mapping[str, Any],
) -> SameBankFullInstanceScores:
    """Materialize all three score arms and immediately prove bank identity."""

    q_only = score_full_instance_candidates(candidates, priors, "q-only")
    global_size = score_full_instance_candidates(candidates, priors, "global-size")
    class_size = score_full_instance_candidates(candidates, priors, "class-size")
    verify_same_bank_size_scores(q_only, global_size)
    identity = verify_same_bank_size_scores(global_size, class_size)
    return SameBankFullInstanceScores(
        q_only=q_only,
        global_size=global_size,
        class_size=class_size,
        identity=identity,
    )


def _candidate_restore_fields(
    candidate: FullInstanceCandidate | Mapping[str, Any],
) -> tuple[int, bool, np.ndarray]:
    row = _candidate_row(candidate)
    raw_id, _, eligible = _validated_candidate_base(row)
    if "member_indices" not in row:
        raise ValueError(f"candidate {raw_id} is missing member_indices")
    raw_members = _as_numpy(row["member_indices"])
    if raw_members.ndim != 1:
        raise ValueError("candidate member_indices must be one-dimensional")
    if np.issubdtype(raw_members.dtype, np.bool_) or not np.issubdtype(
        raw_members.dtype, np.integer
    ):
        raise TypeError("candidate member_indices must contain integers")
    members = np.asarray(raw_members, dtype=np.int64)
    if len(np.unique(members)) != len(members):
        raise ValueError(f"candidate {raw_id} contains duplicate member indices")
    return raw_id, eligible, members


def restore_selected_instances(
    post_filter: Any,
    candidates: Sequence[FullInstanceCandidate | Mapping[str, Any]],
    selected_raw_instance_ids: Collection[int],
) -> RestorationResult:
    """Restore selected eligible raw masks into a copy of ``post_filter``.

    All candidate masks are validated together before any write.  Since they
    must be mutually exclusive, the result is independent of selection order.
    """

    labels = _integer_labels(post_filter, "post_filter")
    point_count = len(labels)
    owner = np.full(point_count, -1, dtype=np.int64)
    candidate_by_id: dict[int, tuple[bool, np.ndarray]] = {}
    for candidate in candidates:
        raw_id, eligible, members = _candidate_restore_fields(candidate)
        if raw_id in candidate_by_id:
            raise ValueError(f"duplicate candidate raw instance {raw_id}")
        if np.any(members < 0) or np.any(members >= point_count):
            raise ValueError(f"candidate {raw_id} has out-of-range member indices")
        overlap = members[owner[members] != -1]
        if len(overlap):
            other_ids = tuple(int(value) for value in np.unique(owner[overlap]))
            raise ValueError(
                f"candidate {raw_id} overlaps candidate masks {other_ids}"
            )
        owner[members] = raw_id
        candidate_by_id[raw_id] = (eligible, members)

    selected: set[int] = set()
    for value in selected_raw_instance_ids:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("selected raw instance IDs must be integers")
        try:
            raw_id = int(value)
            exact = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("selected raw instance IDs must be integers") from exc
        if not np.isfinite(exact) or exact != float(raw_id) or raw_id < 0:
            raise ValueError("selected raw instance IDs must be non-negative integers")
        selected.add(raw_id)
    unknown = tuple(sorted(selected.difference(candidate_by_id)))
    if unknown:
        raise ValueError(f"selected raw instance IDs are absent from bank: {unknown}")
    ineligible = tuple(
        raw_id
        for raw_id in sorted(selected)
        if not candidate_by_id[raw_id][0]
    )
    if ineligible:
        raise ValueError(f"ineligible candidates cannot be restored: {ineligible}")

    output = labels.copy()
    selected_mask = np.zeros(point_count, dtype=bool)
    per_candidate: list[dict[str, Any]] = []
    for raw_id in sorted(selected):
        members = candidate_by_id[raw_id][1]
        before = labels[members]
        selected_mask[members] = True
        output[members] = raw_id
        per_candidate.append(
            {
                "raw_instance_id": raw_id,
                "member_point_count": len(members),
                "changed_point_count": int(np.count_nonzero(before != raw_id)),
                "restored_from_background_count": int(np.count_nonzero(before == -1)),
                "already_same_label_count": int(np.count_nonzero(before == raw_id)),
                "overwritten_other_foreground_count": int(
                    np.count_nonzero((before >= 0) & (before != raw_id))
                ),
            }
        )

    changed = output != labels
    diagnostics: dict[str, Any] = {
        "point_count": point_count,
        "candidate_count": len(candidate_by_id),
        "selected_candidate_count": len(selected),
        "selected_raw_instance_ids": tuple(sorted(selected)),
        "selected_member_point_count": int(np.count_nonzero(selected_mask)),
        "changed_point_count": int(np.count_nonzero(changed)),
        "restored_from_background_count": int(
            np.count_nonzero(selected_mask & (labels == -1))
        ),
        "overwritten_other_foreground_count": int(
            np.count_nonzero(
                selected_mask & (labels >= 0) & (output != labels)
            )
        ),
        "outside_selected_changed_count": int(
            np.count_nonzero((~selected_mask) & changed)
        ),
        "candidate_members_mutually_exclusive": True,
        "selection_order_independent": True,
        "per_candidate": tuple(per_candidate),
    }
    return RestorationResult(_readonly(output), diagnostics)
