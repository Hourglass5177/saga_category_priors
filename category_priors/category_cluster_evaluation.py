from __future__ import annotations

"""Offline evaluation and preregistered gates for repaired cluster banks.

This module deliberately has no scene loader, renderer, clustering routine, or
candidate writer.  Candidate construction can therefore import neither this
module nor validation ground truth by accident.  The public entry points accept
an already-built, bank-like object and an explicit evaluation-only projection
of ScanNet GT onto the Gaussian axis.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .category_candidate_clustering import (
    G1_MUTUAL_LOCAL_GRAPH,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
from .category_denoise_diagnostics import _stage_iou


SCHEMA = "saga-category-cluster-evaluation-v1"
R0_LEGACY = "R0-legacy"
REGISTERED_CONDITIONS = (
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    G1_MUTUAL_LOCAL_GRAPH,
)

_PRIMARY_REPAIRS = (R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN)
_TIE_PRIORITY = {
    # R1 changes only the distance and is therefore structurally simpler than
    # R2 when every registered quality metric ties exactly.
    R1_METRIC_HDBSCAN: 3,
    R2_ANCHORED_HDBSCAN: 2,
    G1_MUTUAL_LOCAL_GRAPH: 1,
}
_EPS = 1e-12


def _readonly_int(values: Any, *, name: str, ndim: int = 1) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D integer array")
    result.setflags(write=False)
    return result


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


@runtime_checkable
class ClusterCandidateBankLike(Protocol):
    """Minimum common bank surface used by the offline evaluator.

    The active integration uses :class:`category_denoise.CandidateBank`.
    ``evaluate_cluster_scene`` also accepts pure algorithm results exposing
    ``full_candidate_labels`` and ``trusted_core_labels`` instead of the two
    historical ``branch_*`` names.
    """

    candidates: Sequence[Mapping[str, Any]]
    branch_full_labels: Any
    branch_core_labels: Any
    class_names: Sequence[str]


@dataclass(frozen=True)
class ClusterEvaluationScene:
    """Evaluation-only GT projection for one physical scan.

    ``gt_to_gaussian_indices`` projects every GT point to its nearest supported
    Gaussian.  ``gt_point_object_indices`` identifies the official GT object at
    each GT point.  ``gaussian_to_gt_object_indices`` performs the reverse 5 cm
    mapping and uses ``-1`` for unsupported/non-evaluable Gaussians.
    """

    scene_id: str
    gt_to_gaussian_indices: np.ndarray
    gt_point_object_indices: np.ndarray
    gt_object_class_ids: np.ndarray
    gt_object_size_bins: tuple[str, ...]
    gaussian_to_gt_object_indices: np.ndarray
    class_name_to_id: Mapping[str, int] = field(default_factory=dict)
    gt_object_instance_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        gt_to_gaussian = _readonly_int(
            self.gt_to_gaussian_indices, name="gt_to_gaussian_indices"
        )
        point_objects = _readonly_int(
            self.gt_point_object_indices, name="gt_point_object_indices"
        )
        object_classes = _readonly_int(
            self.gt_object_class_ids, name="gt_object_class_ids"
        )
        gaussian_objects = _readonly_int(
            self.gaussian_to_gt_object_indices,
            name="gaussian_to_gt_object_indices",
        )
        if gt_to_gaussian.shape != point_objects.shape:
            raise ValueError(
                "gt_to_gaussian_indices and gt_point_object_indices differ in length"
            )
        object_count = len(object_classes)
        if len(self.gt_object_size_bins) != object_count:
            raise ValueError("gt_object_size_bins and gt_object_class_ids differ in length")
        if np.any((point_objects < -1) | (point_objects >= object_count)):
            raise ValueError("gt_point_object_indices contains an invalid object id")
        if np.any((gaussian_objects < -1) | (gaussian_objects >= object_count)):
            raise ValueError(
                "gaussian_to_gt_object_indices contains an invalid object id"
            )
        gaussian_count = len(gaussian_objects)
        if np.any((gt_to_gaussian < -1) | (gt_to_gaussian >= gaussian_count)):
            raise ValueError("gt_to_gaussian_indices contains an invalid Gaussian id")
        object_counts = np.bincount(
            point_objects[point_objects >= 0], minlength=object_count
        )
        if object_count and np.any(object_counts == 0):
            raise ValueError("every official GT object must own at least one GT point")
        class_lookup = {str(key): int(value) for key, value in self.class_name_to_id.items()}
        if self.gt_object_instance_ids is None:
            object_instances = np.arange(object_count, dtype=np.int64)
        else:
            object_instances = _readonly_int(
                self.gt_object_instance_ids, name="gt_object_instance_ids"
            )
            if len(object_instances) != object_count:
                raise ValueError(
                    "gt_object_instance_ids and gt_object_class_ids differ in length"
                )
            if np.any(object_instances < 0):
                raise ValueError("gt_object_instance_ids must be non-negative")
        object_instances.setflags(write=False)
        object.__setattr__(self, "scene_id", str(self.scene_id))
        object.__setattr__(self, "gt_to_gaussian_indices", gt_to_gaussian)
        object.__setattr__(self, "gt_point_object_indices", point_objects)
        object.__setattr__(self, "gt_object_class_ids", object_classes)
        object.__setattr__(self, "gt_object_size_bins", tuple(map(str, self.gt_object_size_bins)))
        object.__setattr__(self, "gaussian_to_gt_object_indices", gaussian_objects)
        object.__setattr__(self, "class_name_to_id", class_lookup)
        object.__setattr__(self, "gt_object_instance_ids", object_instances)


@dataclass(frozen=True)
class ClusterSceneMetrics:
    """Count-preserving metrics for one condition in one physical scan."""

    scene_id: str
    candidate_count: int
    candidate_point_count: int
    unsupported_point_count: int
    same_class_iou_025_count: int
    same_class_iou_050_count: int
    tiny_small_gt_count: int
    tiny_small_iou_025_count: int
    tiny_small_iou_050_count: int
    core_subset_full_violation_count: int
    best_iou_by_gt: tuple[float, ...]
    raw_member_count: int = 0
    raw_member_retained_count: int = 0
    orphan_count: int = 0
    negative_metadata_count: int = 0
    determinism_violation_count: int = 0
    determinism_measured_this_scene: bool = False
    determinism_algorithm_contract_reference: bool = False
    candidate_rows: tuple[Mapping[str, Any], ...] = ()

    @property
    def candidate_precision_025(self) -> float:
        return _fraction(self.same_class_iou_025_count, self.candidate_count)

    @property
    def candidate_precision_050(self) -> float:
        return _fraction(self.same_class_iou_050_count, self.candidate_count)

    @property
    def unsupported_fraction(self) -> float:
        return _fraction(self.unsupported_point_count, self.candidate_point_count)

    @property
    def tiny_small_recall_025(self) -> float:
        return _fraction(self.tiny_small_iou_025_count, self.tiny_small_gt_count)

    @property
    def tiny_small_recall_050(self) -> float:
        return _fraction(self.tiny_small_iou_050_count, self.tiny_small_gt_count)

    @property
    def raw_member_retention(self) -> float:
        return (
            _fraction(self.raw_member_retained_count, self.raw_member_count)
            if self.raw_member_count
            else 1.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "candidate_count": self.candidate_count,
            "candidate_point_count": self.candidate_point_count,
            "unsupported_point_count": self.unsupported_point_count,
            "same_class_iou_025_count": self.same_class_iou_025_count,
            "same_class_iou_050_count": self.same_class_iou_050_count,
            "candidate_precision_025": self.candidate_precision_025,
            "candidate_precision_050": self.candidate_precision_050,
            "unsupported_fraction": self.unsupported_fraction,
            "tiny_small_gt_count": self.tiny_small_gt_count,
            "tiny_small_iou_025_count": self.tiny_small_iou_025_count,
            "tiny_small_iou_050_count": self.tiny_small_iou_050_count,
            "tiny_small_recall_025": self.tiny_small_recall_025,
            "tiny_small_recall_050": self.tiny_small_recall_050,
            "core_subset_full_violation_count": self.core_subset_full_violation_count,
            "raw_member_count": self.raw_member_count,
            "raw_member_retained_count": self.raw_member_retained_count,
            "raw_member_retention": self.raw_member_retention,
            "orphan_count": self.orphan_count,
            "negative_metadata_count": self.negative_metadata_count,
            "determinism_violation_count": self.determinism_violation_count,
            "determinism_measured_this_scene": self.determinism_measured_this_scene,
            "determinism_algorithm_contract_reference": (
                self.determinism_algorithm_contract_reference
            ),
            "best_iou_by_gt": list(self.best_iou_by_gt),
            "candidate_rows": [dict(row) for row in self.candidate_rows],
        }


def _bank_array(bank: Any, *names: str) -> np.ndarray:
    for name in names:
        if hasattr(bank, name):
            return np.asarray(getattr(bank, name), dtype=np.int64)
    raise TypeError(f"candidate bank lacks required array ({' or '.join(names)})")


def _candidate_class_identity(
    row: Mapping[str, Any], bank: Any, scene: ClusterEvaluationScene
) -> tuple[str, int]:
    # ``branch_class_index`` belongs to the runtime's full 32-class codebook,
    # whereas GT object class IDs belong to the canonical ScanNet/SAGA table.
    # The class name is therefore the authoritative bridge between ID spaces.
    if "branch_class" in row:
        class_name = str(row["branch_class"])
        if class_name not in scene.class_name_to_id:
            raise ValueError(
                f"scene taxonomy does not define branch class {class_name!r}"
            )
        return class_name, int(scene.class_name_to_id[class_name])
    for key in ("branch_class_id", "class_id"):
        if key in row:
            value = int(row[key])
            if value < 0:
                raise ValueError("candidate canonical class id must be non-negative")
            names = sorted(
                name
                for name, class_id in scene.class_name_to_id.items()
                if int(class_id) == value
            )
            if len(names) != 1:
                raise ValueError(
                    "a numeric canonical candidate class id requires exactly one "
                    "class_name_to_id reverse match"
                )
            return names[0], value
    raise ValueError(
        "candidate metadata needs branch_class or a canonical branch_class_id/class_id; "
        "branch_class_index alone is the unsafe runtime-32 ID space"
    )


def _compact_candidate_axis(
    labels: np.ndarray, candidate_ids: Sequence[int], *, name: str
) -> np.ndarray:
    if np.any(labels < -1):
        raise ValueError(f"{name} may only use -1 as a negative label")
    output = np.full(labels.shape, -1, dtype=np.int64)
    known = labels < 0
    for compact_id, candidate_id in enumerate(candidate_ids):
        mask = labels == candidate_id
        output[mask] = compact_id
        known |= mask
    if np.any(~known):
        undeclared = np.unique(labels[~known]).tolist()
        raise ValueError(f"{name} contains undeclared candidate ids: {undeclared}")
    return output


def _bank_contract_counts(
    bank: Any,
) -> tuple[int, int, int, int, int, bool, bool]:
    """Extract registered structural-contract counts from bank diagnostics."""

    raw_diagnostics = getattr(bank, "diagnostics", {})
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, Mapping) else {}
    class_diagnostics = diagnostics.get("class_diagnostics")
    if isinstance(class_diagnostics, Mapping):
        blocks = [
            value for value in class_diagnostics.values() if isinstance(value, Mapping)
        ]
    else:
        blocks = [diagnostics]
    class_raw_count = sum(
        int(block.get("raw_member_count", 0)) for block in blocks
    )
    class_retained = sum(
        int(
            block.get(
                "raw_member_retained_count",
                int(block.get("raw_member_count", 0))
                - int(block.get("raw_member_reassigned_count", 0)),
            )
        )
        for block in blocks
    )
    # Integrated banks publish scene-level totals after remapping and candidate
    # filtering.  Those are the authoritative contract values; per-class
    # blocks remain a diagnostic fallback for pure algorithm results.
    raw_count = int(diagnostics.get("raw_member_count", class_raw_count))
    retained = int(
        diagnostics.get(
            "raw_member_retained_count",
            raw_count - int(diagnostics.get("raw_member_reassigned_count", 0))
            if "raw_member_count" in diagnostics
            else class_retained,
        )
    )
    if retained < 0 or retained > raw_count:
        raise ValueError("bank diagnostics contain invalid raw-member retention")
    # A validated CandidateBank has no label without candidate metadata and no
    # negative candidate ID.  Explicit diagnostics may make a later replay
    # failure visible without weakening those construction-time checks.
    orphan = int(diagnostics.get("orphan_count", 0))
    negative_metadata = int(diagnostics.get("negative_metadata_count", 0))
    measured = bool(
        diagnostics.get(
            "determinism_measured_this_scene",
            diagnostics.get("determinism_measured", False),
        )
    )
    reference = diagnostics.get("determinism_algorithm_contract_reference")
    condition = str(diagnostics.get("candidate_cluster_condition", ""))
    reference_valid = bool(
        isinstance(reference, Mapping)
        and reference.get("schema")
        == "saga-category-cluster-determinism-reference-v1"
        and reference.get("source_phase") == "dev2"
        and isinstance(reference.get("artifact"), Mapping)
        and isinstance(reference.get("conditions"), Mapping)
        and condition in reference["conditions"]
        and int(reference["conditions"][condition].get("scene_count", 0)) >= 2
        and int(
            reference["conditions"][condition].get(
                "measured_this_scene_count", -1
            )
        )
        == int(reference["conditions"][condition].get("scene_count", 0))
        and int(reference["conditions"][condition].get("violation_count", -1))
        == 0
    )
    contract_verified = diagnostics.get("determinism_contract_verified") is True
    if measured:
        if "determinism_violation_count" not in diagnostics:
            raise ValueError("measured determinism lacks an observed violation count")
        determinism = int(diagnostics["determinism_violation_count"])
        if reference_valid:
            raise ValueError("a scene cannot be both directly measured and reference-only")
    elif reference_valid and contract_verified:
        # Zero is derived from the verified DEV2 witness, not silently stamped
        # onto the current scene.  The separate reference flag preserves that
        # distinction through aggregation and gates.
        determinism = 0
    else:
        determinism = 0
    if min(orphan, negative_metadata, determinism) < 0:
        raise ValueError("bank contract violation counts must be non-negative")
    return (
        raw_count,
        retained,
        orphan,
        negative_metadata,
        determinism,
        measured and contract_verified,
        reference_valid and contract_verified,
    )


def evaluate_cluster_scene(
    scene: ClusterEvaluationScene,
    bank: ClusterCandidateBankLike | Any,
) -> ClusterSceneMetrics:
    """Evaluate one immutable candidate bank against one scene's offline GT."""

    candidates = tuple(bank.candidates)
    candidate_ids = tuple(int(row["candidate_id"]) for row in candidates)
    if len(set(candidate_ids)) != len(candidate_ids) or any(item < 0 for item in candidate_ids):
        raise ValueError("candidate ids must be unique non-negative integers")
    full_raw = _bank_array(bank, "branch_full_labels", "full_candidate_labels")
    core_raw = _bank_array(bank, "branch_core_labels", "trusted_core_labels")
    if full_raw.ndim != 1 or core_raw.shape != full_raw.shape:
        raise ValueError("candidate full/core labels must be aligned 1-D arrays")
    if len(full_raw) != len(scene.gaussian_to_gt_object_indices):
        raise ValueError("candidate bank and Gaussian-to-GT projection differ in length")
    full = _compact_candidate_axis(full_raw, candidate_ids, name="full labels")
    core = _compact_candidate_axis(core_raw, candidate_ids, name="core labels")

    projected = np.full(len(scene.gt_to_gaussian_indices), -1, dtype=np.int64)
    supported = scene.gt_to_gaussian_indices >= 0
    projected[supported] = full[scene.gt_to_gaussian_indices[supported]]
    object_counts = np.bincount(
        scene.gt_point_object_indices[scene.gt_point_object_indices >= 0],
        minlength=len(scene.gt_object_class_ids),
    ).astype(np.int64, copy=False)
    iou, _, _ = _stage_iou(
        projected,
        len(candidates),
        scene.gt_point_object_indices,
        object_counts,
    )
    candidate_class_identities = tuple(
        _candidate_class_identity(row, bank, scene) for row in candidates
    )
    candidate_classes = np.asarray(
        [identity[1] for identity in candidate_class_identities], dtype=np.int64
    )
    best_by_gt = np.zeros(len(scene.gt_object_class_ids), dtype=np.float64)
    candidate_rows: list[dict[str, Any]] = []
    unsupported_total = 0
    candidate_point_total = 0
    violations = 0
    for compact_id, (candidate_id, (candidate_class_name, candidate_class)) in enumerate(
        zip(candidate_ids, candidate_class_identities)
    ):
        same_gt = np.flatnonzero(scene.gt_object_class_ids == candidate_class)
        if len(same_gt):
            same_iou = iou[compact_id, same_gt]
            local_best = int(np.argmax(same_iou))
            best_object = int(same_gt[local_best])
            best_iou = float(same_iou[local_best])
            best_by_gt[same_gt] = np.maximum(best_by_gt[same_gt], same_iou)
        else:
            best_object = None
            best_iou = 0.0
        full_mask = full == compact_id
        core_mask = core == compact_id
        full_count = int(np.count_nonzero(full_mask))
        core_count = int(np.count_nonzero(core_mask))
        unsupported = int(
            np.count_nonzero(
                full_mask & (scene.gaussian_to_gt_object_indices < 0)
            )
        )
        correct = (
            int(
                np.count_nonzero(
                    full_mask
                    & (scene.gaussian_to_gt_object_indices == best_object)
                )
            )
            if best_object is not None
            else 0
        )
        core_subset_full = bool(np.all(~core_mask | full_mask))
        violations += int(not core_subset_full)
        candidate_point_total += full_count
        unsupported_total += unsupported
        candidate_rows.append(
            {
                "scene_id": scene.scene_id,
                "candidate_id": candidate_id,
                "branch_class": candidate_class_name,
                "branch_class_id": int(candidate_class),
                "full_point_count": full_count,
                "trusted_core_point_count": core_count,
                "best_same_class_object_index": best_object,
                "best_same_class_instance_id": (
                    int(scene.gt_object_instance_ids[best_object])
                    if best_object is not None
                    else None
                ),
                "best_same_class_size_bin": (
                    scene.gt_object_size_bins[best_object]
                    if best_object is not None
                    else None
                ),
                "best_same_class_iou": best_iou,
                "gaussian_precision": _fraction(correct, full_count),
                "unsupported_point_count": unsupported,
                "unsupported_fraction": _fraction(unsupported, full_count),
                "core_subset_full": core_subset_full,
            }
        )
    tiny_small = np.asarray(
        [size in {"tiny", "small"} for size in scene.gt_object_size_bins],
        dtype=bool,
    )
    (
        raw_count,
        retained,
        orphan,
        negative_metadata,
        determinism,
        determinism_measured,
        determinism_reference,
    ) = (
        _bank_contract_counts(bank)
    )
    return ClusterSceneMetrics(
        scene_id=scene.scene_id,
        candidate_count=len(candidates),
        candidate_point_count=candidate_point_total,
        unsupported_point_count=unsupported_total,
        same_class_iou_025_count=int(
            sum(row["best_same_class_iou"] >= 0.25 for row in candidate_rows)
        ),
        same_class_iou_050_count=int(
            sum(row["best_same_class_iou"] >= 0.50 for row in candidate_rows)
        ),
        tiny_small_gt_count=int(np.count_nonzero(tiny_small)),
        tiny_small_iou_025_count=int(
            np.count_nonzero(tiny_small & (best_by_gt >= 0.25))
        ),
        tiny_small_iou_050_count=int(
            np.count_nonzero(tiny_small & (best_by_gt >= 0.50))
        ),
        core_subset_full_violation_count=violations,
        best_iou_by_gt=tuple(map(float, best_by_gt)),
        raw_member_count=raw_count,
        raw_member_retained_count=retained,
        orphan_count=orphan,
        negative_metadata_count=negative_metadata,
        determinism_violation_count=determinism,
        determinism_measured_this_scene=determinism_measured,
        determinism_algorithm_contract_reference=determinism_reference,
        candidate_rows=tuple(candidate_rows),
    )


def _aggregate_condition(
    condition: str, scene_metrics: Sequence[ClusterSceneMetrics]
) -> dict[str, Any]:
    candidate_count = sum(row.candidate_count for row in scene_metrics)
    point_count = sum(row.candidate_point_count for row in scene_metrics)
    unsupported = sum(row.unsupported_point_count for row in scene_metrics)
    count_025 = sum(row.same_class_iou_025_count for row in scene_metrics)
    count_050 = sum(row.same_class_iou_050_count for row in scene_metrics)
    tiny_count = sum(row.tiny_small_gt_count for row in scene_metrics)
    tiny_025 = sum(row.tiny_small_iou_025_count for row in scene_metrics)
    tiny_050 = sum(row.tiny_small_iou_050_count for row in scene_metrics)
    raw_count = sum(row.raw_member_count for row in scene_metrics)
    retained = sum(row.raw_member_retained_count for row in scene_metrics)
    return {
        "condition": condition,
        "scene_count": len(scene_metrics),
        "candidate_count": candidate_count,
        "candidate_point_count": point_count,
        "unsupported_point_count": unsupported,
        "same_class_iou_025_count": count_025,
        "same_class_iou_050_count": count_050,
        "same_class_iou_050_scene_count": sum(
            row.same_class_iou_050_count > 0 for row in scene_metrics
        ),
        "candidate_precision_025": _fraction(count_025, candidate_count),
        "candidate_precision_050": _fraction(count_050, candidate_count),
        "unsupported_fraction": _fraction(unsupported, point_count),
        "tiny_small_gt_count": tiny_count,
        "tiny_small_iou_025_count": tiny_025,
        "tiny_small_iou_050_count": tiny_050,
        "tiny_small_recall_025": _fraction(tiny_025, tiny_count),
        "tiny_small_recall_050": _fraction(tiny_050, tiny_count),
        "core_subset_full_violation_count": sum(
            row.core_subset_full_violation_count for row in scene_metrics
        ),
        "raw_member_count": raw_count,
        "raw_member_retained_count": retained,
        "raw_member_retention": _fraction(retained, raw_count)
        if raw_count
        else 1.0,
        "orphan_count": sum(row.orphan_count for row in scene_metrics),
        "negative_metadata_count": sum(
            row.negative_metadata_count for row in scene_metrics
        ),
        "determinism_violation_count": sum(
            row.determinism_violation_count for row in scene_metrics
        ),
        "determinism_measured_this_scene_count": sum(
            row.determinism_measured_this_scene for row in scene_metrics
        ),
        "determinism_reference_scene_count": sum(
            row.determinism_algorithm_contract_reference for row in scene_metrics
        ),
        "determinism_unverified_scene_count": sum(
            not (
                row.determinism_measured_this_scene
                or row.determinism_algorithm_contract_reference
            )
            for row in scene_metrics
        ),
        "per_scene": [row.as_dict() for row in scene_metrics],
    }


def _scene_quality_key(row: ClusterSceneMetrics) -> tuple[int, int, float, float]:
    return (
        row.same_class_iou_050_count,
        row.same_class_iou_025_count,
        row.candidate_precision_025,
        -row.unsupported_fraction,
    )


def _quality_key(condition: str, aggregate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(aggregate["same_class_iou_050_count"]),
        int(aggregate["same_class_iou_025_count"]),
        float(aggregate["candidate_precision_025"]),
        -float(aggregate["unsupported_fraction"]),
        -int(aggregate["candidate_count"]),
        _TIE_PRIORITY.get(condition, 0),
    )


def _determinism_contract_passed(
    aggregate: Mapping[str, Any], *, require_direct_measurement: bool
) -> bool:
    scene_count = int(aggregate["scene_count"])
    measured = int(aggregate["determinism_measured_this_scene_count"])
    referenced = int(aggregate["determinism_reference_scene_count"])
    unverified = int(aggregate["determinism_unverified_scene_count"])
    if int(aggregate["determinism_violation_count"]) != 0 or unverified != 0:
        return False
    if require_direct_measurement:
        return measured == scene_count and referenced == 0
    return measured + referenced == scene_count


def _compare_to_r0(
    condition_rows: Sequence[ClusterSceneMetrics],
    r0_rows: Sequence[ClusterSceneMetrics],
) -> tuple[
    list[dict[str, Any]],
    dict[str, float],
    tuple[str, ...],
    int,
    int,
]:
    by_scene = {row.scene_id: row for row in condition_rows}
    r0_by_scene = {row.scene_id: row for row in r0_rows}
    gt_rows: list[dict[str, Any]] = []
    per_scene_maximum_drop: dict[str, float] = {}
    improved_scene_ids: list[str] = []
    positive = 0
    negative = 0
    for scene_id in sorted(r0_by_scene):
        baseline = r0_by_scene[scene_id]
        observed = by_scene[scene_id]
        if len(baseline.best_iou_by_gt) != len(observed.best_iou_by_gt):
            raise ValueError(f"{scene_id}: conditions disagree on GT object count")
        observed_key = _scene_quality_key(observed)
        baseline_key = _scene_quality_key(baseline)
        if observed_key > baseline_key:
            improved_scene_ids.append(scene_id)
            positive += 1
        negative += int(observed_key < baseline_key)
        scene_drops: list[float] = []
        for object_index, (left, right) in enumerate(
            zip(baseline.best_iou_by_gt, observed.best_iou_by_gt)
        ):
            drop = float(left - right)
            scene_drops.append(drop)
            gt_rows.append(
                {
                    "scene_id": scene_id,
                    "gt_object_index": object_index,
                    "r0_best_same_class_iou": float(left),
                    "condition_best_same_class_iou": float(right),
                    "best_iou_drop_vs_r0": drop,
                }
            )
        per_scene_maximum_drop[scene_id] = max(scene_drops, default=0.0)
    return (
        gt_rows,
        per_scene_maximum_drop,
        tuple(improved_scene_ids),
        positive,
        negative,
    )


def _dev2_gate(
    arm: Mapping[str, Any],
    r0: Mapping[str, Any],
    *,
    per_scene_maximum_drop: Mapping[str, float],
    improved_scene_ids: Sequence[str],
    improved_scene_count: int,
) -> dict[str, Any]:
    baseline_precision = float(r0["candidate_precision_025"])
    arm_precision = float(arm["candidate_precision_025"])
    precision_relative = (
        (arm_precision - baseline_precision) / baseline_precision
        if baseline_precision > 0.0
        else (1.0 if arm_precision > 0.0 else 0.0)
    )
    unsupported_drop = float(r0["unsupported_fraction"]) - float(
        arm["unsupported_fraction"]
    )
    witness_trials: list[tuple[str, float]] = []
    for witness_scene in sorted(map(str, improved_scene_ids)):
        nonwitness_maximum_drop = max(
            (
                float(drop)
                for scene_id, drop in per_scene_maximum_drop.items()
                if scene_id != witness_scene
            ),
            default=0.0,
        )
        witness_trials.append((witness_scene, nonwitness_maximum_drop))
    passing_witnesses = [
        trial for trial in witness_trials if trial[1] <= 0.05 + _EPS
    ]
    if passing_witnesses:
        witness_scene, nonwitness_maximum_drop = passing_witnesses[0]
    else:
        witness_scene = None
        nonwitness_maximum_drop = (
            min((trial[1] for trial in witness_trials), default=None)
        )
    maximum_drop = max(per_scene_maximum_drop.values(), default=0.0)
    checks = {
        "iou025_not_lower": int(arm["same_class_iou_025_count"])
        >= int(r0["same_class_iou_025_count"]),
        "iou050_not_lower": int(arm["same_class_iou_050_count"])
        >= int(r0["same_class_iou_050_count"]),
        "precision_or_unsupported_improved": precision_relative + _EPS >= 0.25
        or unsupported_drop + _EPS >= 0.10,
        "tiny_small_recall025_not_lower": float(arm["tiny_small_recall_025"])
        + _EPS
        >= float(r0["tiny_small_recall_025"]),
        "candidate_count_within_1.25x": int(arm["candidate_count"])
        <= 1.25 * max(int(r0["candidate_count"]), 1),
        "at_least_one_scene_improved": improved_scene_count >= 1,
        # The preregistration treats the improved scene as the causal witness.
        # The 0.05 safety bound therefore applies only to every *other* scene;
        # a local trade-off inside the witness must not veto the improvement.
        "per_gt_drop_at_most_0.05": witness_scene is not None,
        "core_subset_full": int(arm["core_subset_full_violation_count"]) == 0,
        "raw_members_retained_100pct": int(arm["raw_member_retained_count"])
        == int(arm["raw_member_count"]),
        "orphan_count_zero": int(arm["orphan_count"]) == 0,
        "negative_metadata_count_zero": int(arm["negative_metadata_count"]) == 0,
        "deterministic": _determinism_contract_passed(
            arm, require_direct_measurement=True
        )
        and _determinism_contract_passed(r0, require_direct_measurement=True),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_precision_relative_change": precision_relative,
        "unsupported_fraction_drop": unsupported_drop,
        "improved_scene_count": improved_scene_count,
        "improved_scene_ids": list(map(str, improved_scene_ids)),
        "witness_scene": witness_scene,
        "per_scene_maximum_gt_best_iou_drop": {
            str(scene_id): float(drop)
            for scene_id, drop in sorted(per_scene_maximum_drop.items())
        },
        "nonwitness_maximum_gt_best_iou_drop": nonwitness_maximum_drop,
        "maximum_gt_best_iou_drop": maximum_drop,
    }


def _dev8_gate(
    arm: Mapping[str, Any],
    r0: Mapping[str, Any],
    *,
    positive: int,
    negative: int,
) -> dict[str, Any]:
    checks = {
        "iou050_at_least_12": int(arm["same_class_iou_050_count"]) >= 12,
        "iou050_at_least_4_scenes": int(arm["same_class_iou_050_scene_count"]) >= 4,
        "candidate_precision025_at_least_5pct": float(
            arm["candidate_precision_025"]
        )
        + _EPS
        >= 0.05,
        "tiny_small_recall025_at_least_0.20": float(arm["tiny_small_recall_025"])
        + _EPS
        >= 0.20,
        "positive_scenes_more_than_negative": positive > negative,
        "candidate_count_within_1.25x": int(arm["candidate_count"])
        <= 1.25 * max(int(r0["candidate_count"]), 1),
        "core_subset_full": int(arm["core_subset_full_violation_count"]) == 0,
        "raw_members_retained_100pct": int(arm["raw_member_retained_count"])
        == int(arm["raw_member_count"]),
        "orphan_count_zero": int(arm["orphan_count"]) == 0,
        "negative_metadata_count_zero": int(arm["negative_metadata_count"]) == 0,
        "deterministic": _determinism_contract_passed(
            arm, require_direct_measurement=False
        )
        and _determinism_contract_passed(r0, require_direct_measurement=False),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "positive_scene_count": positive,
        "negative_scene_count": negative,
    }


def analyze_cluster_metrics(
    metrics_by_condition: Mapping[str, Sequence[ClusterSceneMetrics]],
    *,
    phase: Literal["dev2", "dev8"],
    selected_condition: str | None = None,
) -> dict[str, Any]:
    """Aggregate physical-scene metrics and apply the frozen stage gates."""

    if phase not in {"dev2", "dev8"}:
        raise ValueError("phase must be 'dev2' or 'dev8'")
    conditions = tuple(metrics_by_condition)
    if R0_LEGACY not in conditions:
        raise ValueError(f"{R0_LEGACY} is required as the structural baseline")
    unknown = sorted(set(conditions) - set(REGISTERED_CONDITIONS))
    if unknown:
        raise ValueError(f"unregistered cluster conditions: {unknown}")
    challengers = tuple(condition for condition in conditions if condition != R0_LEGACY)
    if not challengers:
        raise ValueError("at least one repaired cluster condition is required")

    normalized: dict[str, tuple[ClusterSceneMetrics, ...]] = {}
    expected_scenes: tuple[str, ...] | None = None
    for condition in conditions:
        rows = tuple(sorted(metrics_by_condition[condition], key=lambda row: row.scene_id))
        scene_ids = tuple(row.scene_id for row in rows)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError(f"{condition}: duplicate physical scene metrics")
        if expected_scenes is None:
            expected_scenes = scene_ids
        elif scene_ids != expected_scenes:
            raise ValueError("all conditions must evaluate exactly the same physical scenes")
        normalized[condition] = rows
    if not expected_scenes:
        raise ValueError("at least one physical scene is required")

    aggregates = {
        condition: _aggregate_condition(condition, rows)
        for condition, rows in normalized.items()
    }
    r0 = aggregates[R0_LEGACY]
    comparisons: dict[str, dict[str, Any]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for condition in challengers:
        (
            gt_rows,
            per_scene_maximum_drop,
            improved_scene_ids,
            positive,
            negative,
        ) = _compare_to_r0(
            normalized[condition], normalized[R0_LEGACY]
        )
        maximum_drop = max(per_scene_maximum_drop.values(), default=0.0)
        comparisons[condition] = {
            "per_gt": gt_rows,
            "maximum_gt_best_iou_drop": maximum_drop,
            "per_scene_maximum_gt_best_iou_drop": {
                scene_id: float(drop)
                for scene_id, drop in sorted(per_scene_maximum_drop.items())
            },
            "improved_scene_ids": list(improved_scene_ids),
            "positive_scene_count": positive,
            "negative_scene_count": negative,
        }
        if phase == "dev2":
            gates[condition] = _dev2_gate(
                aggregates[condition],
                r0,
                per_scene_maximum_drop=per_scene_maximum_drop,
                improved_scene_ids=improved_scene_ids,
                improved_scene_count=positive,
            )
        else:
            gates[condition] = _dev8_gate(
                aggregates[condition], r0, positive=positive, negative=negative
            )

    ranking = sorted(
        challengers,
        key=lambda condition: (
            bool(gates[condition]["passed"]),
            *_quality_key(condition, aggregates[condition]),
        ),
        reverse=True,
    )
    if phase == "dev2":
        if selected_condition is not None:
            raise ValueError("selected_condition must be omitted during DEV2 selection")
        primary = [
            condition
            for condition in _PRIMARY_REPAIRS
            if condition in challengers and gates[condition]["passed"]
        ]
        if primary:
            selected = max(
                primary,
                key=lambda condition: _quality_key(condition, aggregates[condition]),
            )
            selection_tier = "primary_hdbscan_repair"
        elif G1_MUTUAL_LOCAL_GRAPH in challengers and gates[
            G1_MUTUAL_LOCAL_GRAPH
        ]["passed"]:
            selected = G1_MUTUAL_LOCAL_GRAPH
            selection_tier = "registered_graph_fallback"
        else:
            selected = None
            selection_tier = None
    else:
        if selected_condition is None:
            if len(challengers) != 1:
                raise ValueError(
                    "DEV8 selected_condition is required when multiple repairs are supplied"
                )
            selected = challengers[0]
        else:
            selected = str(selected_condition)
            if selected not in challengers:
                raise ValueError("selected_condition is absent from DEV8 metrics")
        selection_tier = "frozen_dev2_selection"

    return {
        "schema": SCHEMA,
        "phase": phase,
        "scene_ids": list(expected_scenes),
        "conditions": aggregates,
        "comparisons_vs_r0": comparisons,
        "gates": gates,
        "ranking": ranking,
        "selected_condition": selected,
        "selection_tier": selection_tier,
        "selected_gate": gates.get(selected) if selected is not None else None,
        "category_prior_tested": False,
        "gt_boundary": "offline_evaluation_only",
    }


def evaluate_cluster_candidate_banks(
    scenes: Mapping[str, ClusterEvaluationScene],
    banks_by_condition: Mapping[str, Mapping[str, ClusterCandidateBankLike | Any]],
    *,
    phase: Literal["dev2", "dev8"],
    selected_condition: str | None = None,
) -> dict[str, Any]:
    """Evaluate condition/scene banks and return a JSON-serializable analysis."""

    scene_ids = tuple(sorted(scenes))
    if not scene_ids:
        raise ValueError("at least one evaluation scene is required")
    metrics: dict[str, tuple[ClusterSceneMetrics, ...]] = {}
    for condition, banks in banks_by_condition.items():
        if set(banks) != set(scene_ids):
            raise ValueError(
                f"{condition}: bank scenes must exactly match evaluation scenes"
            )
        metrics[condition] = tuple(
            evaluate_cluster_scene(scenes[scene_id], banks[scene_id])
            for scene_id in scene_ids
        )
    return analyze_cluster_metrics(
        metrics, phase=phase, selected_condition=selected_condition
    )


__all__ = [
    "SCHEMA",
    "R0_LEGACY",
    "R1_METRIC_HDBSCAN",
    "R2_ANCHORED_HDBSCAN",
    "G1_MUTUAL_LOCAL_GRAPH",
    "REGISTERED_CONDITIONS",
    "ClusterCandidateBankLike",
    "ClusterEvaluationScene",
    "ClusterSceneMetrics",
    "evaluate_cluster_scene",
    "analyze_cluster_metrics",
    "evaluate_cluster_candidate_banks",
]
