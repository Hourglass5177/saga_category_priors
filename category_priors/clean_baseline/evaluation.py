from __future__ import annotations

"""Evaluation and export contracts for the clean alpha-mask baseline.

This module is deliberately independent of the evidence builder and consensus
implementation.  Runtime code operates on Gaussian IDs; evaluation projects
those IDs into the official GT point space before computing any IoU.  GT is
therefore confined to this module and never becomes an input to consensus.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluator import (
    GroundTruthScene,
    PredictedInstance,
    apply_transform,
    evaluate_instances,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from ..io import hash_json, load_json, sha256_file, write_json
from ..prediction_contract import (
    normalize_prediction,
    validate_prediction_contract,
)

CLEAN_PREDICTION_SCHEMA = "saga-clean-alpha-mask-prediction-v2"
CLEAN_EVALUATION_SCHEMA = "saga-clean-alpha-mask-evaluation-v2"
RUN_IDENTITY_SCHEMA = "saga-clean-alpha-mask-run-identity-v1"
EVALUATION_IDENTITY_SCHEMA = "saga-clean-alpha-mask-evaluation-identity-v1"
FORMAL_CONDITIONS = frozenset(
    {"C0-no-prior", "U-global", "D-predicted"}
)
ORACLE_CONDITION = "D-oracle-class"


def validate_embedded_identity(
    value: Mapping[str, Any], *, expected_schema: str
) -> dict[str, Any]:
    """Validate one self-contained identity embedded in an artifact.

    Identities are deliberately embedded instead of being written as sidecar
    SHA files.  This keeps the output layout small while preventing a valid
    file from a different bank, taxonomy, prior, configuration, or commit from
    being mistaken for a resumable result.
    """

    if not isinstance(value, Mapping):
        raise TypeError("artifact identity must be a mapping")
    result = dict(value)
    if result.get("schema") != expected_schema:
        raise ValueError("artifact identity schema mismatch")
    expected = result.get("content_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("artifact identity is missing content_sha256")
    unsigned = dict(result)
    unsigned.pop("content_sha256", None)
    if hash_json(unsigned) != expected:
        raise ValueError("artifact identity content hash mismatch")
    return result


def _stable_id_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _ids(
    values: Sequence[int] | np.ndarray,
    *,
    name: str,
    upper_bound: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype == np.bool_:
        result = np.flatnonzero(raw).astype(np.int64, copy=False)
    else:
        try:
            result = raw.astype(np.int64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must contain integers") from exc
        if not np.array_equal(raw, result):
            raise TypeError(f"{name} must contain integers")
        result = np.unique(result)
    if np.any(result < 0):
        raise ValueError(f"{name} cannot contain negative IDs")
    if upper_bound is not None and np.any(result >= int(upper_bound)):
        raise ValueError(f"{name} contains an out-of-range ID")
    result = np.asarray(result, dtype=np.int64)
    result.setflags(write=False)
    return result


def _unit_interval(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


@dataclass(frozen=True)
class GroundTruthObject:
    object_id: int
    class_id: str | int
    point_ids: np.ndarray
    official_valid: bool = True
    is_tiny_small: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.object_id, bool) or int(self.object_id) != self.object_id:
            raise TypeError("GT object_id must be an integer")
        if int(self.object_id) < 0:
            raise ValueError("GT object_id must be non-negative")
        object.__setattr__(
            self,
            "point_ids",
            _ids(self.point_ids, name="GT point_ids"),
        )
        if len(self.point_ids) == 0:
            raise ValueError("GT objects cannot be empty")


@dataclass(frozen=True)
class CleanCandidate:
    object_id: int | str
    gaussian_ids: np.ndarray
    class_id: str | int | None
    winner_probability: float
    view_consensus: float
    detection_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gaussian_ids",
            _ids(self.gaussian_ids, name="candidate gaussian_ids"),
        )
        if len(self.gaussian_ids) == 0:
            raise ValueError("candidate gaussian_ids cannot be empty")
        for field in (
            "winner_probability",
            "view_consensus",
            "detection_ratio",
        ):
            object.__setattr__(
                self,
                field,
                _unit_interval(getattr(self, field), name=field),
            )

    @property
    def score(self) -> float:
        return float(
            self.winner_probability
            * math.sqrt(self.view_consensus * self.detection_ratio)
        )


def ground_truth_objects_from_arrays(
    semantic: Sequence[int] | np.ndarray,
    instance: Sequence[int] | np.ndarray,
    *,
    class_names: Sequence[str] | None = None,
    min_region_size: int = 100,
    tiny_small_instance_ids: set[int] | frozenset[int] | None = None,
) -> list[GroundTruthObject]:
    semantic_array = np.asarray(semantic)
    instance_array = np.asarray(instance)
    if (
        semantic_array.ndim != 1
        or instance_array.ndim != 1
        or semantic_array.shape != instance_array.shape
    ):
        raise ValueError("GT semantic and instance arrays must be aligned vectors")
    if isinstance(min_region_size, bool) or int(min_region_size) != min_region_size:
        raise TypeError("min_region_size must be an integer")
    if int(min_region_size) <= 0:
        raise ValueError("min_region_size must be positive")
    semantic_array = semantic_array.astype(np.int64, copy=False)
    instance_array = instance_array.astype(np.int64, copy=False)
    tiny_small = tiny_small_instance_ids or set()
    result: list[GroundTruthObject] = []
    for instance_id in np.unique(instance_array[instance_array >= 0]):
        instance_mask = instance_array == instance_id
        class_values, class_counts = np.unique(
            semantic_array[instance_mask], return_counts=True
        )
        valid = class_values >= 0
        if not np.any(valid):
            continue
        class_values = class_values[valid]
        class_counts = class_counts[valid]
        maximum = int(class_counts.max())
        class_index = int(class_values[class_counts == maximum].min())
        if class_names is not None:
            if not 0 <= class_index < len(class_names):
                continue
            class_id: str | int = str(class_names[class_index])
        else:
            class_id = class_index
        point_ids = np.flatnonzero(instance_mask & (semantic_array == class_index))
        result.append(
            GroundTruthObject(
                object_id=int(instance_id),
                class_id=class_id,
                point_ids=point_ids,
                official_valid=len(point_ids) >= int(min_region_size),
                is_tiny_small=int(instance_id) in tiny_small,
            )
        )
    return result


def gt_point_to_gaussian_mapping(
    gt_coords: np.ndarray,
    gaussian_coords: np.ndarray,
    *,
    radius_m: float = 0.05,
) -> tuple[np.ndarray, dict[str, float]]:
    """Map each official GT point to its nearest Gaussian or ``-1``."""

    gaussian_ids = np.arange(len(gaussian_coords), dtype=np.int64)
    return map_gaussians_to_gt(
        np.asarray(gt_coords, dtype=np.float64),
        np.asarray(gaussian_coords, dtype=np.float64),
        gaussian_ids,
        radius_m=float(radius_m),
    )


def project_gaussian_support_to_gt_points(
    gaussian_ids: Sequence[int] | np.ndarray,
    gt_point_to_gaussian: Sequence[int] | np.ndarray,
    *,
    gaussian_count: int | None = None,
    include_unmapped_fp: bool = False,
) -> np.ndarray:
    """Project support to official points, optionally retaining unsupported FP.

    The official evaluator uses only real GT point IDs.  Candidate/oracle
    diagnostics additionally append one synthetic point ID per Gaussian that
    no GT point maps to, so unsupported prediction mass cannot disappear from
    a diagnostic IoU union.  Synthetic IDs always start after the GT domain.
    """

    mapping = np.asarray(gt_point_to_gaussian)
    if mapping.ndim != 1:
        raise ValueError("gt_point_to_gaussian must be one-dimensional")
    mapping = mapping.astype(np.int64, copy=False)
    if np.any(mapping < -1):
        raise ValueError("gt_point_to_gaussian may only use -1 as its sentinel")
    if gaussian_count is None:
        valid_mapping = mapping[mapping >= 0]
        inferred = int(valid_mapping.max() + 1) if len(valid_mapping) else 0
        raw = np.asarray(gaussian_ids)
        if raw.dtype == np.bool_:
            inferred = max(inferred, len(raw))
        elif raw.size:
            try:
                inferred = max(inferred, int(np.max(raw)) + 1)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError("gaussian support must contain integers") from exc
        gaussian_count = inferred
    support = _ids(
        gaussian_ids,
        name="gaussian support",
        upper_bound=int(gaussian_count),
    )
    if len(support) == 0:
        return np.empty(0, dtype=np.int64)
    point_ids = np.flatnonzero(np.isin(mapping, support, assume_unique=False))
    if include_unmapped_fp:
        represented = np.unique(mapping[mapping >= 0])
        unsupported = np.setdiff1d(
            support, represented, assume_unique=True
        )
        if len(unsupported):
            sentinels = len(mapping) + unsupported
            point_ids = np.concatenate((point_ids, sentinels))
    point_ids = np.asarray(point_ids, dtype=np.int64)
    point_ids.setflags(write=False)
    return point_ids


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _best_single(
    supports: Sequence[np.ndarray], mask_ids: Sequence[Any], gt: np.ndarray
) -> tuple[float, Any | None, np.ndarray]:
    best_iou = 0.0
    best_id: Any | None = None
    best_support = np.empty(0, dtype=np.int64)
    for mask_id, support in zip(mask_ids, supports):
        value = _iou(support, gt)
        if value > best_iou + 1e-12 or (
            abs(value - best_iou) <= 1e-12
            and value > 0.0
            and (best_id is None or _stable_id_key(mask_id) < _stable_id_key(best_id))
        ):
            best_iou = value
            best_id = mask_id
            best_support = support
    return best_iou, best_id, best_support


def _greedy_perfect_association(
    supports: Sequence[np.ndarray],
    mask_ids: Sequence[Any],
    gt: np.ndarray,
    *,
    frame_ids: Sequence[int] | None = None,
) -> tuple[float, list[Any]]:
    best_iou, best_id, current = _best_single(supports, mask_ids, gt)
    if best_id is None:
        return 0.0, []
    selected = [best_id]
    best_index = next(
        index for index, mask_id in enumerate(mask_ids) if mask_id == best_id
    )
    used_frames = set() if frame_ids is None else {int(frame_ids[best_index])}
    remaining = [
        index
        for index, mask_id in enumerate(mask_ids)
        if mask_id != best_id
        and (frame_ids is None or int(frame_ids[index]) not in used_frames)
    ]
    while remaining:
        proposals: list[tuple[float, tuple[str, str], int, np.ndarray]] = []
        for index in remaining:
            merged = np.union1d(current, supports[index])
            proposals.append(
                (_iou(merged, gt), _stable_id_key(mask_ids[index]), index, merged)
            )
        proposals.sort(key=lambda row: (-row[0], row[1]))
        value, _, index, merged = proposals[0]
        if value <= best_iou + 1e-12:
            break
        best_iou = value
        current = merged
        selected.append(mask_ids[index])
        if frame_ids is None:
            remaining.remove(index)
        else:
            used_frames.add(int(frame_ids[index]))
            remaining = [
                candidate
                for candidate in remaining
                if int(frame_ids[candidate]) not in used_frames
            ]
    return best_iou, selected


def _aggregate_oracle_rows(
    rows: Sequence[Mapping[str, Any]], selector: str | None = None
) -> dict[str, Any]:
    selected = list(rows) if selector is None else [row for row in rows if bool(row[selector])]
    result: dict[str, Any] = {"gt_count": len(selected)}
    for metric in ("best_single", "perfect_association", "perfect_trim"):
        values = [float(row[metric]) for row in selected]
        result[metric] = {
            "mean_iou": float(np.mean(values)) if values else 0.0,
            "match_025_count": sum(value >= 0.25 for value in values),
            "match_050_count": sum(value >= 0.50 for value in values),
            "recall_025": (
                sum(value >= 0.25 for value in values) / len(values)
                if values
                else 0.0
            ),
            "recall_050": (
                sum(value >= 0.50 for value in values) / len(values)
                if values
                else 0.0
            ),
        }
    return result


def _validate_gt_objects(
    gt_objects: Sequence[GroundTruthObject], *, point_count: int
) -> None:
    seen: set[int] = set()
    for gt in gt_objects:
        if int(gt.object_id) in seen:
            raise ValueError(f"duplicate GT object_id: {gt.object_id}")
        seen.add(int(gt.object_id))
        if np.any(gt.point_ids >= int(point_count)):
            raise ValueError("GT object point_ids exceed the mapping domain")


def evaluate_geometry_oracles(
    mask_gaussian_ids: Sequence[Sequence[int] | np.ndarray],
    gt_objects: Sequence[GroundTruthObject],
    gt_point_to_gaussian: Sequence[int] | np.ndarray,
    *,
    mask_ids: Sequence[Any] | None = None,
    mask_frame_ids: Sequence[int] | None = None,
    gaussian_count: int | None = None,
) -> dict[str, Any]:
    """Evaluate class-agnostic mask support ceilings in official point space.

    ``perfect_association`` is a deterministic monotonic GT-only greedy union.
    ``perfect_trim`` additionally assumes every false-positive point can be
    removed and therefore measures support/coverage rather than deployable IoU.
    """

    mapping = np.asarray(gt_point_to_gaussian)
    if mapping.ndim != 1:
        raise ValueError("gt_point_to_gaussian must be one-dimensional")
    _validate_gt_objects(gt_objects, point_count=len(mapping))
    if mask_ids is None:
        mask_ids = list(range(len(mask_gaussian_ids)))
    if len(mask_ids) != len(mask_gaussian_ids):
        raise ValueError("mask_ids must contain one entry per mask")
    if len(set(map(str, mask_ids))) != len(mask_ids):
        raise ValueError("mask_ids must be unique")
    if mask_frame_ids is not None:
        if len(mask_frame_ids) != len(mask_gaussian_ids):
            raise ValueError("mask_frame_ids must contain one entry per mask")
        if any(int(frame_id) < 0 for frame_id in mask_frame_ids):
            raise ValueError("mask_frame_ids must be non-negative")
    projected = [
        project_gaussian_support_to_gt_points(
            support,
            mapping,
            gaussian_count=gaussian_count,
            include_unmapped_fp=True,
        )
        for support in mask_gaussian_ids
    ]
    rows: list[dict[str, Any]] = []
    for gt in gt_objects:
        single, single_id, _ = _best_single(projected, mask_ids, gt.point_ids)
        associated, associated_ids = _greedy_perfect_association(
            projected,
            mask_ids,
            gt.point_ids,
            frame_ids=mask_frame_ids,
        )
        covered = np.empty(0, dtype=np.int64)
        contributing: list[Any] = []
        for mask_id, support in zip(mask_ids, projected):
            overlap = np.intersect1d(support, gt.point_ids, assume_unique=True)
            if len(overlap):
                covered = np.union1d(covered, overlap)
                contributing.append(mask_id)
        perfect_trim = len(covered) / len(gt.point_ids)
        rows.append(
            {
                "gt_instance_id": int(gt.object_id),
                "gt_class_id": gt.class_id,
                "gt_point_count": len(gt.point_ids),
                "official_valid": bool(gt.official_valid),
                "is_tiny_small": bool(gt.is_tiny_small),
                "best_single": float(single),
                "best_single_mask_id": single_id,
                "perfect_association": float(associated),
                "perfect_association_mask_ids": associated_ids,
                "perfect_trim": float(perfect_trim),
                "perfect_trim_mask_ids": contributing,
            }
        )
    return {
        "schema": "saga-clean-alpha-mask-geometry-oracle-v1",
        "mask_count": len(mask_gaussian_ids),
        "per_gt": rows,
        "aggregate": {
            "all": _aggregate_oracle_rows(rows),
            "official_valid": _aggregate_oracle_rows(rows, "official_valid"),
            "tiny_small_official_valid": _aggregate_oracle_rows(
                [
                    {
                        **row,
                        "tiny_small_official_valid": bool(row["official_valid"])
                        and bool(row["is_tiny_small"]),
                    }
                    for row in rows
                ],
                "tiny_small_official_valid",
            ),
        },
    }


_MISSING = object()


def _candidate_value(
    value: Mapping[str, Any] | Any,
    name: str,
    *aliases: str,
    default: Any = _MISSING,
) -> Any:
    names = (name, *aliases)
    if isinstance(value, Mapping):
        for key in names:
            if key in value:
                return value[key]
    else:
        for key in names:
            if hasattr(value, key):
                return getattr(value, key)
    if default is not _MISSING:
        return default
    raise TypeError(f"candidate is missing required field {name!r}")


def _candidate_from_any(value: CleanCandidate | Mapping[str, Any] | Any) -> CleanCandidate:
    if isinstance(value, CleanCandidate):
        return value
    return CleanCandidate(
        object_id=_candidate_value(value, "object_id"),
        gaussian_ids=np.asarray(_candidate_value(value, "gaussian_ids")),
        class_id=_candidate_value(
            value, "class_id", "class_name", default=None
        ),
        winner_probability=_candidate_value(
            value, "winner_probability", default=0.0
        ),
        view_consensus=_candidate_value(
            value, "view_consensus", "mean_view_consensus", default=0.0
        ),
        detection_ratio=_candidate_value(
            value, "detection_ratio", "mean_detection_ratio", default=0.0
        ),
    )


def _require_unique_candidate_ids(candidates: Sequence[CleanCandidate]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        identifier = str(candidate.object_id)
        if identifier in seen:
            raise ValueError(f"duplicate candidate object_id: {identifier}")
        seen.add(identifier)


def evaluate_candidates(
    candidates: Sequence[CleanCandidate | Mapping[str, Any]],
    gt_objects: Sequence[GroundTruthObject],
    gt_point_to_gaussian: Sequence[int] | np.ndarray,
    *,
    gaussian_count: int | None = None,
) -> dict[str, Any]:
    normalized = [_candidate_from_any(candidate) for candidate in candidates]
    _require_unique_candidate_ids(normalized)
    mapping = np.asarray(gt_point_to_gaussian)
    if mapping.ndim != 1:
        raise ValueError("gt_point_to_gaussian must be one-dimensional")
    _validate_gt_objects(gt_objects, point_count=len(mapping))
    official = [gt for gt in gt_objects if gt.official_valid]
    rows: list[dict[str, Any]] = []
    best_by_gt = {
        int(gt.object_id): {"geometry": 0.0, "same_class": 0.0}
        for gt in official
    }
    for candidate in normalized:
        points = project_gaussian_support_to_gt_points(
            candidate.gaussian_ids,
            mapping,
            gaussian_count=gaussian_count,
            include_unmapped_fp=True,
        )
        geometry_matches = [(_iou(points, gt.point_ids), gt) for gt in official]
        same_class_matches = [
            (value, gt)
            for value, gt in geometry_matches
            if candidate.class_id is not None
            and str(candidate.class_id) == str(gt.class_id)
        ]
        geometry_value, geometry_gt = max(
            geometry_matches,
            key=lambda row: (row[0], -int(row[1].object_id)),
            default=(0.0, None),
        )
        same_value, same_gt = max(
            same_class_matches,
            key=lambda row: (row[0], -int(row[1].object_id)),
            default=(0.0, None),
        )
        for value, gt in geometry_matches:
            best_by_gt[int(gt.object_id)]["geometry"] = max(
                best_by_gt[int(gt.object_id)]["geometry"], value
            )
            if candidate.class_id is not None and str(candidate.class_id) == str(
                gt.class_id
            ):
                best_by_gt[int(gt.object_id)]["same_class"] = max(
                    best_by_gt[int(gt.object_id)]["same_class"], value
                )
        rows.append(
            {
                "candidate_id": candidate.object_id,
                "class_id": candidate.class_id,
                "gaussian_count": len(candidate.gaussian_ids),
                "official_point_count": int(np.count_nonzero(points < len(mapping))),
                "unsupported_gaussian_count": int(
                    np.count_nonzero(points >= len(mapping))
                ),
                "score": candidate.score,
                "best_geometry_iou": float(geometry_value),
                "best_geometry_gt_instance_id": (
                    None if geometry_gt is None else int(geometry_gt.object_id)
                ),
                "best_same_class_iou": float(same_value),
                "best_same_class_gt_instance_id": (
                    None if same_gt is None else int(same_gt.object_id)
                ),
            }
        )
    same_025 = sum(row["best_same_class_iou"] >= 0.25 for row in rows)
    same_050 = sum(row["best_same_class_iou"] >= 0.50 for row in rows)
    geometry_025 = sum(row["best_geometry_iou"] >= 0.25 for row in rows)
    geometry_050 = sum(row["best_geometry_iou"] >= 0.50 for row in rows)
    gt_rows = []
    for gt in official:
        values = best_by_gt[int(gt.object_id)]
        gt_rows.append(
            {
                "gt_instance_id": int(gt.object_id),
                "gt_class_id": gt.class_id,
                "is_tiny_small": bool(gt.is_tiny_small),
                "best_geometry_iou": float(values["geometry"]),
                "best_same_class_iou": float(values["same_class"]),
            }
        )
    tiny = [row for row in gt_rows if row["is_tiny_small"]]
    return {
        "schema": "saga-clean-alpha-mask-candidate-evaluation-v1",
        "candidate_rows": rows,
        "gt_rows": gt_rows,
        "aggregate": {
            "candidate_count": len(rows),
            "official_gt_count": len(official),
            "geometry_iou_025_count": geometry_025,
            "geometry_iou_050_count": geometry_050,
            "same_class_iou_025_count": same_025,
            "same_class_iou_050_count": same_050,
            "candidate_precision_025": same_025 / len(rows) if rows else 0.0,
            "candidate_precision_050": same_050 / len(rows) if rows else 0.0,
            "same_class_recall_025": (
                sum(row["best_same_class_iou"] >= 0.25 for row in gt_rows)
                / len(gt_rows)
                if gt_rows
                else 0.0
            ),
            "same_class_recall_050": (
                sum(row["best_same_class_iou"] >= 0.50 for row in gt_rows)
                / len(gt_rows)
                if gt_rows
                else 0.0
            ),
            "tiny_small_recall_025": (
                sum(row["best_same_class_iou"] >= 0.25 for row in tiny) / len(tiny)
                if tiny
                else 0.0
            ),
            "tiny_small_recall_050": (
                sum(row["best_same_class_iou"] >= 0.50 for row in tiny) / len(tiny)
                if tiny
                else 0.0
            ),
            "fp_tp_ratio_025": (
                (len(rows) - same_025) / same_025 if same_025 else None
            ),
        },
    }


def build_prediction_payload(
    *,
    scene_id: str,
    condition: str,
    gaussian_count: int,
    candidates: Sequence[CleanCandidate | Mapping[str, Any]],
    allowed_classes: Sequence[str],
    run_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one strict formal prediction; oracle-class output is forbidden."""

    if condition not in FORMAL_CONDITIONS:
        if condition == ORACLE_CONDITION:
            raise ValueError("D-oracle-class is evaluation-only and cannot be exported")
        raise ValueError(f"unregistered clean-baseline condition: {condition}")
    if isinstance(gaussian_count, bool) or int(gaussian_count) != gaussian_count:
        raise TypeError("gaussian_count must be an integer")
    if int(gaussian_count) <= 0:
        raise ValueError("gaussian_count must be positive")
    identity = validate_embedded_identity(
        run_identity, expected_schema=RUN_IDENTITY_SCHEMA
    )
    if identity.get("scene_id") != str(scene_id):
        raise ValueError("run identity scene_id mismatch")
    if identity.get("condition") != condition:
        raise ValueError("run identity condition mismatch")
    allowed = {str(value) for value in allowed_classes}
    labels = np.full(int(gaussian_count), -1, dtype=np.int64)
    instances: dict[str, dict[str, Any]] = {}
    normalized = sorted(
        (_candidate_from_any(candidate) for candidate in candidates),
        key=lambda candidate: _stable_id_key(candidate.object_id),
    )
    _require_unique_candidate_ids(normalized)
    skipped_unclassified: list[Any] = []
    for candidate in normalized:
        if candidate.class_id is None or str(candidate.class_id) not in allowed:
            skipped_unclassified.append(candidate.object_id)
            continue
        point_ids = _ids(
            candidate.gaussian_ids,
            name="candidate gaussian_ids",
            upper_bound=int(gaussian_count),
        )
        occupied = point_ids[labels[point_ids] >= 0]
        if len(occupied):
            raise ValueError(
                "consensus candidates overlap; ownership must be resolved before export"
            )
        export_id = len(instances)
        labels[point_ids] = export_id
        instances[str(export_id)] = {
            "class": str(candidate.class_id),
            "score": candidate.score,
            "point_count": len(point_ids),
            "source_object_id": str(candidate.object_id),
            "winner_probability": candidate.winner_probability,
            "view_consensus": candidate.view_consensus,
            "detection_ratio": candidate.detection_ratio,
        }
    contracted = normalize_prediction(labels, instances)
    if contracted.audit["orphan_gaussian_count"] != 0:
        raise AssertionError("clean prediction unexpectedly produced orphan labels")
    payload = {
        "schema": CLEAN_PREDICTION_SCHEMA,
        "scene_id": str(scene_id),
        "condition": condition,
        "run_identity": identity,
        "point_labels": contracted.point_labels.tolist(),
        "instances": contracted.instances,
    }
    diagnostics = {
        "schema": "saga-clean-alpha-mask-prediction-diagnostics-v1",
        "scene_id": str(scene_id),
        "condition": condition,
        "run_identity": identity,
        "gaussian_count": int(gaussian_count),
        "candidate_count": len(normalized),
        "exported_instance_count": len(contracted.instances),
        "skipped_unclassified_object_ids": [str(value) for value in skipped_unclassified],
        "contract": contracted.audit,
        "oracle_class_used": False,
    }
    return payload, diagnostics


def prediction_is_complete(
    path: str | Path,
    *,
    expected_scene_id: str | None = None,
    expected_condition: str | None = None,
    expected_gaussian_count: int | None = None,
    expected_run_identity: Mapping[str, Any] | None = None,
) -> bool:
    try:
        payload = load_json(path)
        if payload.get("schema") != CLEAN_PREDICTION_SCHEMA:
            return False
        if expected_scene_id is not None and payload.get("scene_id") != expected_scene_id:
            return False
        if expected_condition is not None and payload.get("condition") != expected_condition:
            return False
        if payload.get("condition") not in FORMAL_CONDITIONS:
            return False
        identity = validate_embedded_identity(
            payload.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        if expected_run_identity is not None:
            expected_identity = validate_embedded_identity(
                expected_run_identity, expected_schema=RUN_IDENTITY_SCHEMA
            )
            if identity != expected_identity:
                return False
        labels = np.asarray(payload["point_labels"])
        if expected_gaussian_count is not None and len(labels) != int(
            expected_gaussian_count
        ):
            return False
        validate_prediction_contract(labels, payload["instances"])
        return True
    except (FileNotFoundError, OSError, KeyError, TypeError, ValueError):
        return False


def _payload_predictions(
    scene_id: str,
    payload: Mapping[str, Any],
    gt_point_to_gaussian: np.ndarray,
    class_names: Sequence[str],
) -> list[PredictedInstance]:
    labels = np.asarray(payload["point_labels"], dtype=np.int64)
    validate_prediction_contract(labels, payload["instances"])
    mapping = np.asarray(gt_point_to_gaussian, dtype=np.int64)
    mapped = np.full(len(mapping), -1, dtype=np.int64)
    valid = (mapping >= 0) & (mapping < len(labels))
    mapped[valid] = labels[mapping[valid]]
    class_to_id = {str(name): index for index, name in enumerate(class_names)}
    predictions: list[PredictedInstance] = []
    for raw_id, metadata in payload["instances"].items():
        instance_id = int(raw_id)
        class_name = str(metadata["class"])
        if class_name not in class_to_id:
            raise ValueError(f"unknown formal prediction class: {class_name}")
        predictions.append(
            PredictedInstance(
                scene_id=scene_id,
                instance_id=instance_id,
                class_id=class_to_id[class_name],
                score=float(metadata["score"]),
                mask=mapped == instance_id,
            )
        )
    return predictions


def evaluate_prediction_payload(
    *,
    scene_id: str,
    payload: Mapping[str, Any],
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    gt_point_to_gaussian: np.ndarray,
    class_names: Sequence[str],
    min_region_size: int = 100,
) -> dict[str, Any]:
    if payload.get("scene_id") != scene_id:
        raise ValueError("prediction scene_id mismatch")
    if payload.get("condition") not in FORMAL_CONDITIONS:
        raise ValueError("only formal C0/U/D predictions can enter official evaluation")
    ground_truth = GroundTruthScene(
        scene_id=scene_id,
        semantic=np.asarray(gt_semantic, dtype=np.int64),
        instance=np.asarray(gt_instance, dtype=np.int64),
    )
    predictions = _payload_predictions(
        scene_id,
        payload,
        np.asarray(gt_point_to_gaussian, dtype=np.int64),
        class_names,
    )
    return evaluate_instances(
        [ground_truth],
        predictions,
        class_names,
        min_region_size=int(min_region_size),
    )


def evaluate_ground_truth_parity(
    ground_truth: Sequence[GroundTruthScene],
    class_names: Sequence[str],
    *,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Run the official evaluator with GT-as-prediction as a protocol check."""

    predictions: list[PredictedInstance] = []
    for scene in ground_truth:
        for class_id in range(len(class_names)):
            class_mask = scene.semantic == class_id
            for instance_id in np.unique(scene.instance[class_mask]):
                if instance_id < 0:
                    continue
                mask = class_mask & (scene.instance == instance_id)
                if int(mask.sum()) < int(min_region_size):
                    continue
                predictions.append(
                    PredictedInstance(
                        scene_id=scene.scene_id,
                        instance_id=int(instance_id),
                        class_id=class_id,
                        score=1.0,
                        mask=mask,
                    )
                )
    result = evaluate_instances(
        ground_truth,
        predictions,
        class_names,
        min_region_size=int(min_region_size),
    )
    supported = [
        values
        for values in result["per_class"].values()
        if int(values["gt_instances"]) > 0
    ]
    if supported and any(
        not math.isclose(float(values["ap_0.50"]), 1.0, abs_tol=1e-12)
        for values in supported
    ):
        raise AssertionError("GT-as-prediction parity failed")
    result["gt_as_prediction_parity"] = True
    return result


def evaluate_clean_baseline_manifest(
    manifest_path: str | Path,
    *,
    class_names: Sequence[str],
    output_path: str | Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate paired C0/U/D outputs from a GT-only manifest adapter."""

    manifest = load_json(manifest_path)
    if manifest.get("kind") != "clean_baseline_evaluation_manifest":
        raise ValueError("expected a clean_baseline_evaluation_manifest")
    minimum_mapped_fraction = float(
        manifest.get("minimum_mapped_fraction", 0.90)
    )
    if not 0.0 < minimum_mapped_fraction <= 1.0:
        raise ValueError("minimum_mapped_fraction must be in (0, 1]")
    base = Path(manifest_path).resolve().parent
    conditions = tuple(str(value) for value in manifest.get("conditions", []))
    if not conditions or any(value not in FORMAL_CONDITIONS for value in conditions):
        raise ValueError("evaluation conditions must be a non-empty subset of C0/U/D")
    if len(set(conditions)) != len(conditions):
        raise ValueError("evaluation conditions must be unique")
    if ORACLE_CONDITION in conditions:
        raise ValueError("D-oracle-class cannot enter formal evaluation")
    gt_scenes: list[GroundTruthScene] = []
    predictions = {condition: [] for condition in conditions}
    diagnostics: dict[str, Any] = {}
    input_identities: dict[str, Any] = {}
    seen: set[str] = set()
    scenes = manifest.get("scenes", [])
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("evaluation manifest must contain at least one scene")
    for item in scenes:
        scene_id = str(item["scene_id"])
        if scene_id in seen:
            raise ValueError(f"duplicate evaluation scene: {scene_id}")
        seen.add(scene_id)
        gt_coords, gt_scene = load_ground_truth_npz(base / item["gt_npz"], scene_id)
        gt_path = base / item["gt_npz"]
        gaussian_path = base / item["gaussian_ply"]
        gaussian_xyz = apply_transform(
            load_ply_xyz(gaussian_path),
            item["gaussian_to_gt_transform"],
        )
        mapping, mapping_diagnostics = gt_point_to_gaussian_mapping(
            gt_coords,
            gaussian_xyz,
            radius_m=float(radius_m),
        )
        if (
            float(mapping_diagnostics["median_nn_distance_m"]) > float(radius_m)
            or float(mapping_diagnostics["mapped_fraction"])
            < minimum_mapped_fraction
        ):
            raise ValueError(f"{scene_id}: coordinate alignment gate failed")
        gt_scenes.append(gt_scene)
        diagnostics[scene_id] = mapping_diagnostics
        output_map = item.get("outputs", {})
        for condition in conditions:
            if condition not in output_map:
                raise ValueError(f"{scene_id}: missing {condition} output")
            output_spec = output_map[condition]
            output_path_value = (
                output_spec["output_json"]
                if isinstance(output_spec, Mapping)
                else output_spec
            )
            payload = load_json(base / output_path_value)
            if payload.get("scene_id") != scene_id or payload.get("condition") != condition:
                raise ValueError(f"{scene_id}/{condition}: prediction identity mismatch")
            if not prediction_is_complete(
                base / output_path_value,
                expected_scene_id=scene_id,
                expected_condition=condition,
                expected_gaussian_count=len(gaussian_xyz),
            ):
                raise ValueError(f"{scene_id}/{condition}: incomplete prediction")
            predictions[condition].extend(
                _payload_predictions(scene_id, payload, mapping, class_names)
            )
            identity = validate_embedded_identity(
                payload.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
            )
            input_identities.setdefault(scene_id, {}).setdefault(
                "predictions", {}
            )[condition] = identity["content_sha256"]
        input_identities[scene_id].update(
            {
                "gt_sha256": sha256_file(gt_path),
                "gaussian_sha256": sha256_file(gaussian_path),
                "gaussian_to_gt_transform": item["gaussian_to_gt_transform"],
            }
        )
    evaluation_identity: dict[str, Any] = {
        "schema": EVALUATION_IDENTITY_SCHEMA,
        "manifest": hash_json(manifest),
        "class_names": list(map(str, class_names)),
        "conditions": list(conditions),
        "radius_m": float(radius_m),
        "minimum_mapped_fraction": minimum_mapped_fraction,
        "min_region_size": int(min_region_size),
        "inputs": input_identities,
    }
    evaluation_identity["content_sha256"] = hash_json(evaluation_identity)
    if output_path is not None and evaluation_is_complete(
        output_path,
        expected_scene_ids=sorted(seen),
        expected_conditions=conditions,
        expected_evaluation_identity=evaluation_identity,
    ):
        return {**load_json(output_path), "runner_status": "skipped-complete"}
    metrics = {
        condition: evaluate_instances(
            gt_scenes,
            predictions[condition],
            class_names,
            min_region_size=int(min_region_size),
        )
        for condition in conditions
    }
    result = {
        "schema": CLEAN_EVALUATION_SCHEMA,
        "scene_ids": sorted(seen),
        "conditions": list(conditions),
        "radius_m": float(radius_m),
        "minimum_mapped_fraction": minimum_mapped_fraction,
        "min_region_size": int(min_region_size),
        "metrics": metrics,
        "mapping_diagnostics": diagnostics,
        "oracle_class_in_formal_metrics": False,
        "evaluation_identity": evaluation_identity,
    }
    if output_path is not None:
        write_json(output_path, result)
    return {**result, "runner_status": "complete"}


def evaluation_is_complete(
    path: str | Path,
    *,
    expected_scene_ids: Sequence[str] | None = None,
    expected_conditions: Sequence[str] | None = None,
    expected_evaluation_identity: Mapping[str, Any] | None = None,
) -> bool:
    try:
        payload = load_json(path)
        if payload.get("schema") != CLEAN_EVALUATION_SCHEMA:
            return False
        if payload.get("oracle_class_in_formal_metrics") is not False:
            return False
        identity = validate_embedded_identity(
            payload.get("evaluation_identity"),
            expected_schema=EVALUATION_IDENTITY_SCHEMA,
        )
        if expected_evaluation_identity is not None:
            expected_identity = validate_embedded_identity(
                expected_evaluation_identity,
                expected_schema=EVALUATION_IDENTITY_SCHEMA,
            )
            if identity != expected_identity:
                return False
        conditions = payload.get("conditions")
        if not isinstance(conditions, list) or any(
            value not in FORMAL_CONDITIONS for value in conditions
        ):
            return False
        if expected_scene_ids is not None and sorted(payload.get("scene_ids", [])) != sorted(
            map(str, expected_scene_ids)
        ):
            return False
        if expected_conditions is not None and conditions != list(expected_conditions):
            return False
        return set(conditions) == set(payload.get("metrics", {}))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False


__all__ = [
    "CLEAN_EVALUATION_SCHEMA",
    "CLEAN_PREDICTION_SCHEMA",
    "FORMAL_CONDITIONS",
    "ORACLE_CONDITION",
    "CleanCandidate",
    "GroundTruthObject",
    "build_prediction_payload",
    "evaluate_candidates",
    "evaluate_clean_baseline_manifest",
    "evaluate_geometry_oracles",
    "evaluate_ground_truth_parity",
    "evaluate_prediction_payload",
    "evaluation_is_complete",
    "ground_truth_objects_from_arrays",
    "gt_point_to_gaussian_mapping",
    "prediction_is_complete",
    "project_gaussian_support_to_gt_points",
]
