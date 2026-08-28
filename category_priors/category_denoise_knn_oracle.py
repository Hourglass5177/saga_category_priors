from __future__ import annotations

"""Pure CPU mechanics for the category-denoise KNN oracle experiment.

Ground truth is admitted only by :func:`prepare_knn_oracle_scene`, which fixes
the evaluation-only candidate selection and its matched GT target.  Every
replay function accepts only frozen candidate IDs, Gaussian coordinates, bank
arrays, and B0 metadata.  In particular, changing GT-derived fields in a plan
cannot alter the KNN/filter mechanics once its candidate IDs are fixed.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .category_denoise import (
    GLOBAL_KNN_K,
    GLOBAL_MIN_COUNT,
    CandidateBank,
    LegacyKNNFilterResult,
    build_strict_prediction_metadata,
    legacy_knn_filter,
    replay_protected_denoise,
)
from .prediction_contract import validate_prediction_contract


PLAN_SCHEMA = "saga-category-denoise-knn-oracle-plan-v1"


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


@dataclass(frozen=True)
class OracleCandidateSelection:
    """One official-point-space oracle-positive candidate and fixed target."""

    scene_id: str
    candidate_id: int
    branch_class: str
    branch_class_index: int
    same_class_iou: float
    matched_gt_class_id: int
    matched_gt_class: str
    matched_gt_instance_id: int
    matched_gt_point_count: int
    full_point_count: int
    core_point_count: int
    base_score: float | None
    official_gt_coverage: float
    gaussian_target_precision: float
    gaussian_supported_purity: float
    gaussian_unsupported_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "candidate_id": self.candidate_id,
            "branch_class": self.branch_class,
            "branch_class_index": self.branch_class_index,
            "same_class_iou": self.same_class_iou,
            "matched_gt_class_id": self.matched_gt_class_id,
            "matched_gt_class": self.matched_gt_class,
            "matched_gt_instance_id": self.matched_gt_instance_id,
            "matched_gt_point_count": self.matched_gt_point_count,
            "full_point_count": self.full_point_count,
            "core_point_count": self.core_point_count,
            "Q": self.base_score,
            "official_gt_coverage": self.official_gt_coverage,
            "gaussian_target_precision": self.gaussian_target_precision,
            "gaussian_supported_purity": self.gaussian_supported_purity,
            "gaussian_unsupported_fraction": self.gaussian_unsupported_fraction,
        }


@dataclass(frozen=True)
class KNNOracleScenePlan:
    """Evaluation-only oracle selection for one scene."""

    scene_id: str
    point_count: int
    bank_schema: str
    bank_seed: int
    iou_threshold: float
    radius_m: float
    min_region_size: int
    candidates: tuple[OracleCandidateSelection, ...]
    schema: str = PLAN_SCHEMA
    evaluation_only: bool = True

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(row.candidate_id for row in self.candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evaluation_only": self.evaluation_only,
            "scene_id": self.scene_id,
            "point_count": self.point_count,
            "bank_schema": self.bank_schema,
            "bank_seed": self.bank_seed,
            "iou_threshold": self.iou_threshold,
            "radius_m": self.radius_m,
            "min_region_size": self.min_region_size,
            "candidates": [row.to_dict() for row in self.candidates],
        }


def _nearest_gt_indices(
    gaussian_xyz: np.ndarray, gt_xyz: np.ndarray, radius_m: float
) -> np.ndarray:
    from scipy.spatial import cKDTree

    result = np.full(len(gaussian_xyz), -1, dtype=np.int64)
    if not len(gaussian_xyz) or not len(gt_xyz):
        return result
    distances, indices = cKDTree(gt_xyz).query(
        gaussian_xyz, k=1, distance_upper_bound=radius_m, workers=-1
    )
    valid = np.isfinite(distances) & (indices < len(gt_xyz))
    result[valid] = indices[valid]
    return result


def prepare_knn_oracle_scene(
    *,
    scene_id: str,
    bank: CandidateBank,
    gaussian_xyz_metric: Any,
    gt_xyz: Any,
    gt_semantic: Any,
    gt_instance: Any,
    canonical_classes: Sequence[str],
    iou_threshold: float = 0.50,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> KNNOracleScenePlan:
    """Select same-class full candidates at a frozen official IoU threshold.

    This is the only function in this module whose inputs contain GT.  The
    matched GT identity is frozen in the returned plan so later evaluation
    cannot silently switch targets after KNN expansion or erosion.
    """

    # Keep the GT projection dependency out of the replay import path.  The
    # public replay functions below neither import nor accept evaluator data.
    from .evaluator import map_gaussians_to_gt

    gaussian_xyz = _array(gaussian_xyz_metric, np.float64)
    gt_points = _array(gt_xyz, np.float64)
    semantic = _array(gt_semantic, np.int64)
    instances = _array(gt_instance, np.int64)
    classes = tuple(map(str, canonical_classes))
    threshold = float(iou_threshold)
    radius = float(radius_m)
    minimum = int(min_region_size)
    if gaussian_xyz.shape != (bank.point_count, 3):
        raise ValueError("gaussian_xyz_metric does not match the candidate bank")
    if gt_points.ndim != 2 or gt_points.shape[1:] != (3,):
        raise ValueError("gt_xyz must have shape [M, 3]")
    if semantic.shape != (len(gt_points),) or instances.shape != (len(gt_points),):
        raise ValueError("GT coordinates, semantic, and instance arrays must align")
    if not np.isfinite(gaussian_xyz).all() or not np.isfinite(gt_points).all():
        raise ValueError("Gaussian and GT coordinates must be finite")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if minimum <= 0:
        raise ValueError("min_region_size must be positive")
    if len(set(classes)) != len(classes):
        raise ValueError("canonical_classes must be unique")

    # Official point-space projection: every GT point queries its nearest
    # Gaussian.  This deliberately matches the repository evaluator.
    mapped_full, _ = map_gaussians_to_gt(
        gt_points,
        gaussian_xyz,
        np.asarray(bank.branch_full_labels, dtype=np.int64),
        radius,
    )
    gaussian_to_gt = _nearest_gt_indices(gaussian_xyz, gt_points, radius)

    valid_gt_rows: list[tuple[int, int, str, np.ndarray]] = []
    valid = (
        (semantic >= 0)
        & (semantic < len(classes))
        & (instances >= 0)
    )
    pairs = sorted(
        set(zip(semantic[valid].tolist(), instances[valid].tolist()))
    )
    for class_id, instance_id in pairs:
        mask = valid & (semantic == class_id) & (instances == instance_id)
        if int(np.count_nonzero(mask)) < minimum:
            continue
        valid_gt_rows.append((int(class_id), int(instance_id), classes[class_id], mask))

    candidate_by_id = {
        int(row["candidate_id"]): row for row in bank.candidates
    }
    selected: list[OracleCandidateSelection] = []
    for candidate_id in sorted(candidate_by_id):
        candidate = candidate_by_id[candidate_id]
        branch_class = str(candidate["branch_class"])
        if branch_class not in classes:
            continue
        prediction = mapped_full == candidate_id
        best_iou = -1.0
        best_target: tuple[int, int, str, np.ndarray] | None = None
        best_intersection = 0
        for gt_row in valid_gt_rows:
            class_id, _, class_name, gt_mask = gt_row
            if class_name != branch_class:
                continue
            intersection = int(np.count_nonzero(prediction & gt_mask))
            union = int(np.count_nonzero(prediction | gt_mask))
            iou = intersection / union if union else 0.0
            # valid_gt_rows is sorted by (class_id, instance_id), so strict
            # improvement gives a deterministic smallest-ID tie break.
            if iou > best_iou:
                best_iou = float(iou)
                best_target = gt_row
                best_intersection = intersection
        if best_target is None or best_iou < threshold:
            continue

        class_id, instance_id, class_name, gt_mask = best_target
        full_mask = np.asarray(bank.branch_full_labels) == candidate_id
        core_mask = np.asarray(bank.branch_core_labels) == candidate_id
        candidate_indices = np.flatnonzero(full_mask)
        nearest = gaussian_to_gt[candidate_indices]
        has_neighbor = nearest >= 0
        evaluable = np.zeros(len(candidate_indices), dtype=bool)
        if np.any(has_neighbor):
            supported_gt = nearest[has_neighbor]
            evaluable[has_neighbor] = (
                (semantic[supported_gt] >= 0)
                & (semantic[supported_gt] < len(classes))
                & (instances[supported_gt] >= 0)
            )
        correct = np.zeros(len(candidate_indices), dtype=bool)
        if np.any(evaluable):
            correct[evaluable] = (
                (semantic[nearest[evaluable]] == class_id)
                & (instances[nearest[evaluable]] == instance_id)
            )
        full_count = len(candidate_indices)
        supported_count = int(np.count_nonzero(evaluable))
        correct_count = int(np.count_nonzero(correct))
        raw_q = candidate.get("base_score")
        base_score = float(raw_q) if raw_q is not None else None
        selected.append(
            OracleCandidateSelection(
                scene_id=str(scene_id),
                candidate_id=candidate_id,
                branch_class=branch_class,
                branch_class_index=int(candidate["branch_class_index"]),
                same_class_iou=best_iou,
                matched_gt_class_id=class_id,
                matched_gt_class=class_name,
                matched_gt_instance_id=instance_id,
                matched_gt_point_count=int(np.count_nonzero(gt_mask)),
                full_point_count=int(np.count_nonzero(full_mask)),
                core_point_count=int(np.count_nonzero(core_mask)),
                base_score=base_score,
                official_gt_coverage=(
                    best_intersection / int(np.count_nonzero(gt_mask))
                    if np.any(gt_mask)
                    else 0.0
                ),
                gaussian_target_precision=(
                    correct_count / full_count if full_count else 0.0
                ),
                gaussian_supported_purity=(
                    correct_count / supported_count if supported_count else 0.0
                ),
                gaussian_unsupported_fraction=(
                    1.0 - supported_count / full_count if full_count else 0.0
                ),
            )
        )

    return KNNOracleScenePlan(
        scene_id=str(scene_id),
        point_count=bank.point_count,
        bank_schema=bank.schema,
        bank_seed=bank.seed,
        iou_threshold=threshold,
        radius_m=radius,
        min_region_size=minimum,
        candidates=tuple(selected),
    )


class ExactB0MappingError(ValueError):
    """Raised when raw legacy labels cannot be mapped exactly to strict B0."""


@dataclass(frozen=True)
class ExactB0Mapping:
    """Exact raw-global label to strict B0 metadata identity."""

    raw_to_b0_instance: dict[int, int]
    class_by_raw: dict[int, str]
    score_by_raw: dict[int, float]
    baseline_raw_labels: np.ndarray
    b0_point_labels: np.ndarray
    baseline_projected_labels: np.ndarray
    b0_instance_count: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "exact": True,
            "point_count": int(len(self.b0_point_labels)),
            "mapped_raw_instance_count": len(self.raw_to_b0_instance),
            "b0_instance_count": self.b0_instance_count,
            "baseline_point_difference_count": int(
                np.count_nonzero(
                    self.baseline_projected_labels != self.b0_point_labels
                )
            ),
        }


def recover_exact_b0_mapping(
    baseline_raw_labels: Any,
    b0_point_labels: Any,
    b0_instances: Mapping[str | int, Mapping[str, Any]],
) -> ExactB0Mapping:
    """Recover raw-global identity only when every B0 mask is exactly equal.

    A one-way containment check is insufficient: a raw label containing the B0
    mask plus additional B0-background points would silently corrupt later
    counterfactual metadata.  This function therefore requires equality in both
    directions and verifies that the complete declared projection reproduces
    B0 point for point.
    """

    raw = _array(baseline_raw_labels, np.int64)
    b0_value = b0_point_labels
    if hasattr(b0_value, "detach"):
        b0_value = b0_value.detach()
    if hasattr(b0_value, "cpu"):
        b0_value = b0_value.cpu()
    if hasattr(b0_value, "numpy"):
        b0_value = b0_value.numpy()
    b0_untyped = np.asarray(b0_value)
    if raw.ndim != 1 or b0_untyped.shape != raw.shape:
        raise ExactB0MappingError("baseline raw labels and B0 labels must align")
    if np.any(raw < -1):
        raise ExactB0MappingError("baseline raw labels may only use -1 as negative")
    try:
        validate_prediction_contract(b0_untyped, b0_instances)
    except (TypeError, ValueError) as exc:
        raise ExactB0MappingError(f"B0 prediction is not strict: {exc}") from exc
    b0 = b0_untyped.astype(np.int64, copy=False)

    raw_to_b0: dict[int, int] = {}
    class_by_raw: dict[int, str] = {}
    score_by_raw: dict[int, float] = {}
    for b0_id in range(len(b0_instances)):
        b0_mask = b0 == b0_id
        raw_values = np.unique(raw[b0_mask])
        if len(raw_values) != 1 or int(raw_values[0]) < 0:
            raise ExactB0MappingError(
                f"B0 instance {b0_id} is not contained in one non-negative raw label"
            )
        raw_id = int(raw_values[0])
        if raw_id in raw_to_b0:
            raise ExactB0MappingError(
                f"raw label {raw_id} maps to multiple B0 instances"
            )
        if not np.array_equal(raw == raw_id, b0_mask):
            raise ExactB0MappingError(
                f"raw label {raw_id} and B0 instance {b0_id} masks differ"
            )
        metadata = b0_instances.get(str(b0_id), b0_instances.get(b0_id))
        if not isinstance(metadata, Mapping):  # guarded by strict validation
            raise ExactB0MappingError(f"B0 instance {b0_id} metadata is missing")
        raw_to_b0[raw_id] = b0_id
        class_by_raw[raw_id] = str(metadata["class"])
        score_by_raw[raw_id] = float(metadata["score"])

    projected = np.full(raw.shape, -1, dtype=np.int64)
    for raw_id, b0_id in raw_to_b0.items():
        projected[raw == raw_id] = b0_id
    if not np.array_equal(projected, b0):
        difference = int(np.count_nonzero(projected != b0))
        raise ExactB0MappingError(
            f"raw declared projection differs from B0 at {difference} points"
        )
    return ExactB0Mapping(
        raw_to_b0_instance=raw_to_b0,
        class_by_raw=class_by_raw,
        score_by_raw=score_by_raw,
        baseline_raw_labels=_readonly(raw.copy()),
        b0_point_labels=_readonly(b0.copy()),
        baseline_projected_labels=_readonly(projected),
        b0_instance_count=len(b0_instances),
    )


@dataclass(frozen=True)
class CandidateReplayDiagnostic:
    candidate_id: int
    raw_label: int
    original_point_count: int
    retained_original_after_knn: int
    retained_original_after_filter: int
    retained_original_fraction_after_knn: float
    retained_original_fraction_after_filter: float
    gained_outside_after_knn: int
    gained_outside_after_filter: int
    label_present_after_knn: bool
    label_present_after_filter: bool
    survived_after_knn: bool
    survived_after_filter: bool

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        # Frozen E2 report aliases refer to the final post-filter state; the
        # explicit stage-specific fields remain available for funnel analysis.
        payload.update(
            {
                "retained_original_point_count": self.retained_original_after_filter,
                "retained_original_fraction": self.retained_original_fraction_after_filter,
                "gained_outside_point_count": self.gained_outside_after_filter,
            }
        )
        return payload


def _replay_diagnostic(
    candidate_id: int,
    raw_label: int,
    original_mask: np.ndarray,
    after_knn: np.ndarray,
    after_filter: np.ndarray,
) -> CandidateReplayDiagnostic:
    knn_mask = after_knn == raw_label
    filter_mask = after_filter == raw_label
    original_count = int(np.count_nonzero(original_mask))
    retained_knn = int(np.count_nonzero(original_mask & knn_mask))
    retained_filter = int(np.count_nonzero(original_mask & filter_mask))
    return CandidateReplayDiagnostic(
        candidate_id=int(candidate_id),
        raw_label=int(raw_label),
        original_point_count=original_count,
        retained_original_after_knn=retained_knn,
        retained_original_after_filter=retained_filter,
        retained_original_fraction_after_knn=(
            retained_knn / original_count if original_count else 0.0
        ),
        retained_original_fraction_after_filter=(
            retained_filter / original_count if original_count else 0.0
        ),
        gained_outside_after_knn=int(np.count_nonzero(knn_mask & ~original_mask)),
        gained_outside_after_filter=int(
            np.count_nonzero(filter_mask & ~original_mask)
        ),
        label_present_after_knn=bool(np.any(knn_mask)),
        label_present_after_filter=bool(np.any(filter_mask)),
        survived_after_knn=retained_knn > 0,
        survived_after_filter=retained_filter > 0,
    )


def _candidate_rows(
    bank: CandidateBank, candidate_ids: Sequence[int]
) -> tuple[tuple[int, ...], dict[int, Mapping[str, Any]]]:
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be an integer sequence")
    normalized: list[int] = []
    for raw_id in candidate_ids:
        if isinstance(raw_id, (bool, np.bool_)) or not isinstance(
            raw_id, (int, np.integer)
        ):
            raise TypeError("candidate IDs must be integers")
        candidate_id = int(raw_id)
        normalized.append(candidate_id)
    if len(normalized) != len(set(normalized)):
        raise ValueError("candidate_ids contains duplicates")
    rows = {int(row["candidate_id"]): row for row in bank.candidates}
    unknown = sorted(set(normalized) - set(rows))
    if unknown:
        raise ValueError(f"unknown oracle candidate IDs: {unknown}")
    selected = tuple(sorted(normalized))
    if selected:
        # branch_full_labels is a partition, but keep the assertion local to
        # E2 so corrupt hand-authored banks cannot create overlapping oracles.
        membership = np.zeros(bank.point_count, dtype=np.int8)
        for candidate_id in selected:
            mask = np.asarray(bank.branch_full_labels) == candidate_id
            if not np.any(mask):
                raise ValueError(f"oracle candidate {candidate_id} has an empty full mask")
            membership[mask] += 1
        if np.any(membership > 1):
            raise ValueError("oracle candidate full masks overlap")
    return selected, rows


@dataclass(frozen=True)
class UnprotectedOracleReplay:
    source_labels: np.ndarray
    after_knn: np.ndarray
    after_filter: np.ndarray
    candidate_raw_labels: dict[int, int]
    candidates: tuple[CandidateReplayDiagnostic, ...]
    legacy: LegacyKNNFilterResult


def replay_unprotected_oracle(
    *,
    xyz_scene: Any,
    bank: CandidateBank,
    candidate_ids: Sequence[int],
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    chunk_size: int = 8_192,
) -> UnprotectedOracleReplay:
    """Inject fixed candidate IDs, then expose them to legacy KNN/filter.

    No GT-derived metric or target identity is accepted by this function.
    """

    xyz = _array(xyz_scene, np.float64)
    if xyz.shape != (bank.point_count, 3):
        raise ValueError("xyz_scene does not match the candidate bank")
    selected, _ = _candidate_rows(bank, candidate_ids)
    source = np.asarray(bank.global_pre_knn, dtype=np.int64).copy()
    maximum = int(source[source >= 0].max()) if np.any(source >= 0) else -1
    candidate_raw = {
        candidate_id: maximum + ordinal + 1
        for ordinal, candidate_id in enumerate(selected)
    }
    for candidate_id, raw_label in candidate_raw.items():
        source[np.asarray(bank.branch_full_labels) == candidate_id] = raw_label
    legacy = legacy_knn_filter(
        xyz,
        source,
        k=k,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    diagnostics = tuple(
        _replay_diagnostic(
            candidate_id,
            candidate_raw[candidate_id],
            np.asarray(bank.branch_full_labels) == candidate_id,
            legacy.after_knn,
            legacy.after_filter,
        )
        for candidate_id in selected
    )
    return UnprotectedOracleReplay(
        source_labels=_readonly(source),
        after_knn=legacy.after_knn,
        after_filter=legacy.after_filter,
        candidate_raw_labels=candidate_raw,
        candidates=diagnostics,
        legacy=legacy,
    )


@dataclass(frozen=True)
class ProtectedOracleReplay:
    after_filter: np.ndarray
    candidate_raw_labels: dict[int, int]
    candidate_class_by_raw: dict[int, str]
    candidate_score_by_raw: dict[int, float]
    candidates: tuple[CandidateReplayDiagnostic, ...]
    diagnostics: dict[str, Any]


def replay_protected_oracle(
    *,
    xyz_scene: Any,
    bank: CandidateBank,
    candidate_ids: Sequence[int],
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    chunk_size: int = 8_192,
) -> ProtectedOracleReplay:
    """Protect fixed full masks through the existing early-exclusion replay."""

    selected, rows = _candidate_rows(bank, candidate_ids)
    accepted: list[dict[str, Any]] = []
    for candidate_id in selected:
        raw_q = rows[candidate_id].get("base_score")
        if raw_q is None:
            raise ValueError(
                f"oracle candidate {candidate_id} is missing its frozen Q/base_score"
            )
        accepted.append(
            {
                "candidate_id": candidate_id,
                "accepted": True,
                "ap_score": float(raw_q),
            }
        )
    labels, class_by_raw, score_by_raw, replay_diagnostics = replay_protected_denoise(
        xyz_scene,
        bank,
        accepted,
        k=k,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    inserted = {
        int(candidate_id): int(raw_label)
        for candidate_id, raw_label in replay_diagnostics[
            "inserted_candidate_to_instance"
        ].items()
    }
    diagnostics: list[CandidateReplayDiagnostic] = []
    for candidate_id in selected:
        raw_label = inserted[candidate_id]
        expected = np.asarray(bank.branch_full_labels) == candidate_id
        observed = np.asarray(labels) == raw_label
        if not np.array_equal(observed, expected):
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} was not preserved exactly"
            )
        expected_class = str(rows[candidate_id]["branch_class"])
        if class_by_raw.get(raw_label) != expected_class:
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} changed class"
            )
        expected_score = float(rows[candidate_id]["base_score"])
        if score_by_raw.get(raw_label) != expected_score:
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} changed score"
            )
        diagnostics.append(
            _replay_diagnostic(
                candidate_id,
                raw_label,
                expected,
                np.asarray(labels),
                np.asarray(labels),
            )
        )
    return ProtectedOracleReplay(
        after_filter=labels,
        candidate_raw_labels=inserted,
        candidate_class_by_raw=class_by_raw,
        candidate_score_by_raw=score_by_raw,
        candidates=tuple(diagnostics),
        diagnostics=dict(replay_diagnostics),
    )


@dataclass(frozen=True)
class StrictOraclePrediction:
    """Strict prediction plus stable pre-contract IDs for causal diagnostics."""

    stable_internal_labels: np.ndarray
    point_labels: np.ndarray
    instances: dict[str, dict[str, Any]]
    prediction_contract: dict[str, Any]
    raw_to_stable: dict[int, int]
    stable_to_strict: dict[int, int]
    candidate_to_stable: dict[int, int]
    candidate_to_strict: dict[int, int | None]
    unmapped_raw_instance_ids: tuple[int, ...]
    unmapped_raw_gaussian_count: int


def project_oracle_prediction(
    *,
    xyz_scene: Any,
    raw_labels: Any,
    bank: CandidateBank,
    candidate_raw_labels: Mapping[int, int],
    b0_mapping: ExactB0Mapping,
) -> StrictOraclePrediction:
    """Project a raw replay through frozen B0 metadata and strict output truth.

    Global raw labels without an exact B0 declaration become background.
    Candidate IDs take precedence over raw-global IDs; this is required because
    ``replay_protected_denoise`` may reuse the numeric ID of a raw global label
    that disappeared from the active partition before candidate insertion.
    """

    xyz = _array(xyz_scene, np.float64)
    raw = _array(raw_labels, np.int64)
    if xyz.shape != (bank.point_count, 3) or raw.shape != (bank.point_count,):
        raise ValueError("xyz_scene, raw_labels, and candidate bank must align")
    selected, rows = _candidate_rows(bank, tuple(candidate_raw_labels))
    normalized_candidate_raw: dict[int, int] = {}
    for candidate_id in selected:
        raw_value = candidate_raw_labels[candidate_id]
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, np.integer)
        ):
            raise TypeError("candidate raw labels must be integers")
        raw_id = int(raw_value)
        if raw_id < 0:
            raise ValueError("candidate raw labels must be non-negative")
        normalized_candidate_raw[candidate_id] = raw_id
    if len(set(normalized_candidate_raw.values())) != len(normalized_candidate_raw):
        raise ValueError("candidate raw labels must be unique")

    stable = np.full(raw.shape, -1, dtype=np.int64)
    raw_to_stable: dict[int, int] = {}
    class_by_stable: dict[int, str] = {}
    score_by_stable: dict[int, float] = {}
    # Stable global IDs are the original strict B0 IDs.  This avoids the old
    # global-rank comparison bug when oracle instances are inserted.
    for raw_id, b0_id in sorted(b0_mapping.raw_to_b0_instance.items()):
        stable[raw == raw_id] = b0_id
        raw_to_stable[raw_id] = b0_id
        class_by_stable[b0_id] = b0_mapping.class_by_raw[raw_id]
        score_by_stable[b0_id] = b0_mapping.score_by_raw[raw_id]

    candidate_to_stable: dict[int, int] = {}
    for ordinal, candidate_id in enumerate(selected):
        raw_id = normalized_candidate_raw[candidate_id]
        stable_id = b0_mapping.b0_instance_count + ordinal
        # Candidate overwrite is deliberate for the disappeared-raw-ID
        # collision described in the docstring.
        stable[raw == raw_id] = stable_id
        raw_to_stable[raw_id] = stable_id
        candidate_to_stable[candidate_id] = stable_id
        candidate = rows[candidate_id]
        raw_q = candidate.get("base_score")
        if raw_q is None:
            raise ValueError(
                f"oracle candidate {candidate_id} is missing its frozen Q/base_score"
            )
        class_by_stable[stable_id] = str(candidate["branch_class"])
        score_by_stable[stable_id] = float(raw_q)

    known_raw = set(b0_mapping.raw_to_b0_instance) | set(
        normalized_candidate_raw.values()
    )
    present_raw = set(int(value) for value in np.unique(raw[raw >= 0]))
    unmapped = tuple(sorted(present_raw - known_raw))
    unmapped_count = int(np.count_nonzero(np.isin(raw, np.asarray(unmapped)))) if unmapped else 0

    strict = build_strict_prediction_metadata(
        stable,
        xyz,
        class_by_stable,
        score_by_stable,
    )
    stable_to_strict: dict[int, int] = {}
    for stable_id in np.unique(stable[stable >= 0]):
        values = np.unique(strict.point_labels[stable == stable_id])
        if len(values) != 1 or int(values[0]) < 0:
            raise RuntimeError("strict normalization split or removed a declared instance")
        stable_to_strict[int(stable_id)] = int(values[0])
    candidate_to_strict = {
        candidate_id: stable_to_strict.get(stable_id)
        for candidate_id, stable_id in candidate_to_stable.items()
    }
    return StrictOraclePrediction(
        stable_internal_labels=_readonly(stable),
        point_labels=strict.point_labels,
        instances=strict.instances,
        prediction_contract=strict.audit,
        raw_to_stable=raw_to_stable,
        stable_to_strict=stable_to_strict,
        candidate_to_stable=candidate_to_stable,
        candidate_to_strict=candidate_to_strict,
        unmapped_raw_instance_ids=unmapped,
        unmapped_raw_gaussian_count=unmapped_count,
    )


@dataclass(frozen=True)
class KNNOracleReplay:
    unprotected: UnprotectedOracleReplay
    protected: ProtectedOracleReplay
    o1_prediction: StrictOraclePrediction
    o2_prediction: StrictOraclePrediction
    diagnostics: dict[str, Any]


def replay_knn_oracle_scene(
    *,
    xyz_scene: Any,
    bank: CandidateBank,
    candidate_ids: Sequence[int],
    b0_mapping: ExactB0Mapping,
    k: int = GLOBAL_KNN_K,
    min_count: int = GLOBAL_MIN_COUNT,
    chunk_size: int = 8_192,
) -> KNNOracleReplay:
    """Run O1/O2 from fixed candidate IDs and build strict predictions.

    This public replay entry point deliberately has no GT, IoU, radius, or
    matched-target parameter.
    """

    unprotected = replay_unprotected_oracle(
        xyz_scene=xyz_scene,
        bank=bank,
        candidate_ids=candidate_ids,
        k=k,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    protected = replay_protected_oracle(
        xyz_scene=xyz_scene,
        bank=bank,
        candidate_ids=candidate_ids,
        k=k,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    o1 = project_oracle_prediction(
        xyz_scene=xyz_scene,
        raw_labels=unprotected.after_filter,
        bank=bank,
        candidate_raw_labels=unprotected.candidate_raw_labels,
        b0_mapping=b0_mapping,
    )
    o2 = project_oracle_prediction(
        xyz_scene=xyz_scene,
        raw_labels=protected.after_filter,
        bank=bank,
        candidate_raw_labels=protected.candidate_raw_labels,
        b0_mapping=b0_mapping,
    )
    selected, rows = _candidate_rows(bank, candidate_ids)
    for candidate_id in selected:
        strict_id = o2.candidate_to_strict.get(candidate_id)
        if strict_id is None:
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} vanished in strict projection"
            )
        expected_mask = np.asarray(bank.branch_full_labels) == candidate_id
        if not np.array_equal(o2.point_labels == strict_id, expected_mask):
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} changed during strict projection"
            )
        metadata = o2.instances[str(strict_id)]
        if metadata["class"] != str(rows[candidate_id]["branch_class"]):
            raise RuntimeError(
                f"protected oracle candidate {candidate_id} class changed in strict output"
            )
    oracle_union = np.isin(
        np.asarray(bank.branch_full_labels), np.asarray(selected, dtype=np.int64)
    ) if selected else np.zeros(bank.point_count, dtype=bool)
    outside = ~oracle_union
    b0 = b0_mapping.b0_point_labels
    diagnostics = {
        "candidate_count": len(selected),
        "oracle_gaussian_count": int(np.count_nonzero(oracle_union)),
        "o1_outside_oracle_changed_vs_b0_count": int(
            np.count_nonzero(outside & (o1.stable_internal_labels != b0))
        ),
        "o2_outside_oracle_changed_vs_b0_count": int(
            np.count_nonzero(outside & (o2.stable_internal_labels != b0))
        ),
        "o2_outside_oracle_changed_vs_o1_count": int(
            np.count_nonzero(
                outside
                & (o2.stable_internal_labels != o1.stable_internal_labels)
            )
        ),
        "o1_outside_oracle_raw_changed_vs_b0_count": int(
            np.count_nonzero(
                outside
                & (
                    unprotected.after_filter
                    != b0_mapping.baseline_raw_labels
                )
            )
        ),
        "o2_outside_oracle_raw_changed_vs_b0_count": int(
            np.count_nonzero(
                outside
                & (protected.after_filter != b0_mapping.baseline_raw_labels)
            )
        ),
        "o2_outside_oracle_raw_changed_vs_o1_count": int(
            np.count_nonzero(
                outside
                & (protected.after_filter != unprotected.after_filter)
            )
        ),
        "outside_oracle_gaussian_count": int(np.count_nonzero(outside)),
        "o1_unmapped_raw_gaussian_count": o1.unmapped_raw_gaussian_count,
        "o2_unmapped_raw_gaussian_count": o2.unmapped_raw_gaussian_count,
    }
    denominator = max(diagnostics["outside_oracle_gaussian_count"], 1)
    for key in (
        "o1_outside_oracle_changed_vs_b0_count",
        "o2_outside_oracle_changed_vs_b0_count",
        "o2_outside_oracle_changed_vs_o1_count",
        "o1_outside_oracle_raw_changed_vs_b0_count",
        "o2_outside_oracle_raw_changed_vs_b0_count",
        "o2_outside_oracle_raw_changed_vs_o1_count",
    ):
        diagnostics[key.removesuffix("_count") + "_fraction"] = (
            diagnostics[key] / denominator
        )
    return KNNOracleReplay(
        unprotected=unprotected,
        protected=protected,
        o1_prediction=o1,
        o2_prediction=o2,
        diagnostics=diagnostics,
    )


__all__ = [
    "PLAN_SCHEMA",
    "CandidateReplayDiagnostic",
    "ExactB0Mapping",
    "ExactB0MappingError",
    "KNNOracleReplay",
    "KNNOracleScenePlan",
    "OracleCandidateSelection",
    "ProtectedOracleReplay",
    "StrictOraclePrediction",
    "UnprotectedOracleReplay",
    "prepare_knn_oracle_scene",
    "project_oracle_prediction",
    "recover_exact_b0_mapping",
    "replay_knn_oracle_scene",
    "replay_protected_oracle",
    "replay_unprotected_oracle",
]
