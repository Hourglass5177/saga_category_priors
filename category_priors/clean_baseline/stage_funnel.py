"""Read-only reconstruction of the clean-baseline production stages.

The clean baseline historically persisted the immutable evidence bank, the
final condition diagnostics, and ``output.json``.  Those artifacts are enough
to reconstruct the deterministic transformations after mask lifting without
running the expensive observer/supporter search again:

``complete masks -> association masks -> undersegmentation removal ->``
``accepted-edge components -> detection filtering -> physical filtering ->``
``unique ownership -> export``.

This module deliberately has no ground-truth dependency.  A caller can attach
an evaluation callback to any reconstructed stage, but the reconstruction
itself only consumes production evidence and production diagnostics.  If an
older artifact omitted information needed by a stage, that stage and every
dependent stage are marked unavailable rather than inferred from the final
answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..io import load_json
from .consensus import (
    ConsensusConfig,
    ConsensusEdge,
    ConsensusObject,
    MaskObservation,
    remove_contained_objects,
    split_disconnected_support,
)
from .evidence import load_evidence_bank
from .models import AlphaMaskEvidenceBank


MetricCallback = Callable[
    [str, tuple["FunnelObject", ...]], Mapping[str, Any] | None
]


STAGE_NAMES = (
    "complete_mask_support",
    "association_support",
    "undersegmentation_filtered",
    "accepted_edge_components",
    "detection_ratio_filtered",
    "physical_split_and_deduplicated",
    "unique_gaussian_ownership",
    "final_export",
)


@dataclass(frozen=True)
class FunnelObject:
    """One immutable object hypothesis at a named production stage."""

    stable_id: str
    gaussian_ids: np.ndarray
    mask_ids: tuple[int, ...] = ()
    frame_ids: tuple[int, ...] = ()
    class_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.stable_id):
            raise ValueError("stable_id must be non-empty")
        raw = np.asarray(self.gaussian_ids)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise TypeError("gaussian_ids must be a one-dimensional integer array")
        ids = np.unique(raw.astype(np.int64, copy=False))
        if np.any(ids < 0):
            raise ValueError("gaussian_ids must be non-negative")
        ids.setflags(write=False)
        masks = tuple(sorted({int(value) for value in self.mask_ids}))
        frames = tuple(sorted({int(value) for value in self.frame_ids}))
        if any(value < 0 for value in masks + frames):
            raise ValueError("mask_ids and frame_ids must be non-negative")
        object.__setattr__(self, "stable_id", str(self.stable_id))
        object.__setattr__(self, "gaussian_ids", ids)
        object.__setattr__(self, "mask_ids", masks)
        object.__setattr__(self, "frame_ids", frames)
        object.__setattr__(
            self,
            "class_name",
            None if self.class_name is None else str(self.class_name),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class FunnelStage:
    """A reconstructed stage or an explicit unavailable marker."""

    name: str
    available: bool
    objects: tuple[FunnelObject, ...] = ()
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in STAGE_NAMES:
            raise ValueError(f"unknown stage name: {self.name}")
        if self.available and self.reason is not None:
            raise ValueError("an available stage cannot have an unavailable reason")
        if not self.available and self.objects:
            raise ValueError("an unavailable stage cannot contain reconstructed objects")
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def summary(self) -> dict[str, int | bool | str | None]:
        if not self.available:
            return {
                "available": False,
                "reason": self.reason,
                "object_count": 0,
                "assignment_count": 0,
                "unique_gaussian_count": 0,
                "duplicate_assignment_count": 0,
            }
        assignment_count = sum(len(item.gaussian_ids) for item in self.objects)
        if self.objects and assignment_count:
            unique_count = int(
                len(np.unique(np.concatenate([item.gaussian_ids for item in self.objects])))
            )
        else:
            unique_count = 0
        return {
            "available": True,
            "reason": None,
            "object_count": len(self.objects),
            "assignment_count": assignment_count,
            "unique_gaussian_count": unique_count,
            "duplicate_assignment_count": assignment_count - unique_count,
        }


@dataclass(frozen=True)
class PartitionEquivalence:
    equivalent: bool
    changed_points: int
    reconstructed_instance_count: int
    frozen_instance_count: int
    class_exact: bool
    point_count_exact: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalent": bool(self.equivalent),
            "changed_points": int(self.changed_points),
            "reconstructed_instance_count": int(self.reconstructed_instance_count),
            "frozen_instance_count": int(self.frozen_instance_count),
            "class_exact": bool(self.class_exact),
            "point_count_exact": bool(self.point_count_exact),
        }


@dataclass(frozen=True)
class CleanStageFunnel:
    scene_id: str
    condition: str
    point_count: int
    stages: tuple[FunnelStage, ...]
    final_equivalence: PartitionEquivalence | None
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(stage.name for stage in self.stages)
        if names != STAGE_NAMES:
            raise ValueError("stages must contain every registered stage in order")

    def stage(self, name: str) -> FunnelStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": "saga-clean-stage-funnel-v1",
            "scene_id": self.scene_id,
            "condition": self.condition,
            "point_count": self.point_count,
            "stages": [
                {
                    "name": stage.name,
                    **stage.summary,
                    "details": dict(stage.details),
                    "metrics": dict(stage.metrics),
                }
                for stage in self.stages
            ],
            "final_equivalence": (
                None
                if self.final_equivalence is None
                else self.final_equivalence.to_dict()
            ),
            "issues": list(self.issues),
        }


def _stage(
    name: str,
    objects: Sequence[FunnelObject],
    *,
    details: Mapping[str, Any] | None,
    metric_callback: MetricCallback | None,
) -> FunnelStage:
    frozen = tuple(objects)
    metrics: Mapping[str, Any] = {}
    if metric_callback is not None:
        returned = metric_callback(name, frozen)
        if returned is not None:
            if not isinstance(returned, Mapping):
                raise TypeError("metric callback must return a mapping or None")
            metrics = dict(returned)
    return FunnelStage(
        name=name,
        available=True,
        objects=frozen,
        details={} if details is None else details,
        metrics=metrics,
    )


def _unavailable(name: str, reason: str) -> FunnelStage:
    return FunnelStage(name=name, available=False, reason=str(reason))


def _observations(bank: AlphaMaskEvidenceBank) -> tuple[MaskObservation, ...]:
    rows: list[MaskObservation] = []
    for metadata in bank.masks:
        ids, _, _, ambiguity = bank.support_for_mask(
            metadata.global_mask_id, include_ambiguous=True
        )
        rows.append(
            MaskObservation(
                mask_id=metadata.global_mask_id,
                frame_id=metadata.frame_id,
                gaussian_ids=ids,
                ambiguous_ids=ids[ambiguity],
            )
        )
    return tuple(rows)


def _parse_config(diagnostics: Mapping[str, Any]) -> ConsensusConfig:
    raw = diagnostics.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("condition diagnostics omitted the frozen consensus config")
    allowed = {
        "mask_visible_threshold",
        "undersegment_filter_threshold",
        "view_consensus_threshold",
        "contained_threshold",
        "point_filter_threshold",
        "dbscan_eps_m",
        "dbscan_min_samples",
        "min_views",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"condition diagnostics contain unknown config fields: {sorted(unknown)}")
    config = ConsensusConfig(**{key: raw[key] for key in allowed if key in raw})
    config.validate()
    return config


def _parse_edges(rows: object) -> tuple[ConsensusEdge, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("accepted_edges must be a sequence")
    result: list[ConsensusEdge] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise TypeError("accepted edge rows must be mappings")
        result.append(
            ConsensusEdge(
                tuple(map(int, raw["left_mask_ids"])),
                tuple(map(int, raw["right_mask_ids"])),
                int(raw["observer_count"]),
                int(raw["supporter_count"]),
                float(raw["consensus"]),
                int(raw["observer_level"]),
            )
        )
    return tuple(result)


def _component_mask_ids(
    active_mask_ids: Sequence[int], accepted_edges: Sequence[ConsensusEdge]
) -> tuple[tuple[int, ...], ...]:
    active = tuple(sorted(map(int, active_mask_ids)))
    active_set = set(active)
    parent = {mask_id: mask_id for mask_id in active}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep

    for edge in accepted_edges:
        members = tuple(edge.left_mask_ids) + tuple(edge.right_mask_ids)
        if not members or any(mask_id not in active_set for mask_id in members):
            raise ValueError("accepted edge references a rejected or unknown mask")
        anchor = int(members[0])
        for mask_id in members[1:]:
            union(anchor, int(mask_id))
    groups: dict[int, list[int]] = {}
    for mask_id in active:
        groups.setdefault(find(mask_id), []).append(mask_id)
    return tuple(sorted((tuple(values) for values in groups.values()), key=lambda row: row))


def _component_full_ids(
    component: Sequence[int], by_mask: Mapping[int, MaskObservation]
) -> np.ndarray:
    arrays = [by_mask[mask_id].gaussian_ids for mask_id in component]
    if not arrays or not any(len(row) for row in arrays):
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(arrays)).astype(np.int64, copy=False)


def _detection_profile(
    component: Sequence[int],
    *,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    config: ConsensusConfig,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Return the frozen component full union and its per-point ratios.

    The ratio is computed for *every* point in the component full union.  This
    is intentionally separated from the historical hard threshold so a
    diagnostic replay can remove that threshold without changing the evidence
    it measures.  ``min_views`` and ``empty_full`` remain preconditions in both
    paths and are reported independently.
    """

    full_ids = _component_full_ids(component, by_mask)
    frame_ids = tuple(sorted({by_mask[mask_id].frame_id for mask_id in component}))
    if len(frame_ids) < config.min_views:
        return full_ids, np.zeros(full_ids.size, dtype=np.float64), "min_views"
    if full_ids.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), "empty_full"
    visible_counts = np.zeros(full_ids.size, dtype=np.int64)
    detection_counts = np.zeros(full_ids.size, dtype=np.int64)
    member_by_frame: dict[int, list[int]] = {}
    for mask_id in component:
        member_by_frame.setdefault(by_mask[mask_id].frame_id, []).append(mask_id)
    for frame in bank.frames:
        if frame.geometry_abstained:
            continue
        visible_ids, _ = bank.visibility_for_frame(frame.frame_id)
        _, positions, _ = np.intersect1d(
            full_ids, visible_ids, assume_unique=True, return_indices=True
        )
        if positions.size == 0:
            continue
        ambiguous = bank.ambiguous_for_frame(frame.frame_id)
        if ambiguous.size:
            positions = positions[
                ~np.isin(full_ids[positions], ambiguous, assume_unique=True)
            ]
        if positions.size == 0:
            continue
        eligible = np.zeros(full_ids.size, dtype=bool)
        eligible[positions] = True
        visible_counts += eligible.astype(np.int64)
        frame_masks = member_by_frame.get(frame.frame_id, ())
        if not frame_masks:
            continue
        detected = np.zeros(full_ids.size, dtype=bool)
        for mask_id in frame_masks:
            detected |= np.isin(
                full_ids, by_mask[mask_id].association_ids, assume_unique=True
            )
        detection_counts += (detected & eligible).astype(np.int64)
    ratios = np.divide(
        detection_counts,
        visible_counts,
        out=np.zeros(full_ids.size, dtype=np.float64),
        where=visible_counts > 0,
    )
    return full_ids, ratios, None


def _detection_filter(
    component: Sequence[int],
    *,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    config: ConsensusConfig,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Apply the historical hard detection-ratio threshold."""

    full_ids, ratios, dropped_reason = _detection_profile(
        component,
        bank=bank,
        by_mask=by_mask,
        config=config,
    )
    if dropped_reason is not None:
        return full_ids, ratios, dropped_reason
    keep = ratios >= config.point_filter_threshold
    if not np.any(keep):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), "detection_ratio"
    return full_ids[keep], ratios[keep], None


def _part_metadata(
    part: np.ndarray,
    component: Sequence[int],
    *,
    by_mask: Mapping[int, MaskObservation],
    edges: Sequence[ConsensusEdge],
    filtered_ids: np.ndarray,
    filtered_ratios: np.ndarray,
    config: ConsensusConfig,
) -> ConsensusObject | None:
    supporting = tuple(
        mask_id
        for mask_id in sorted(component)
        if np.intersect1d(
            part, by_mask[mask_id].association_ids, assume_unique=True
        ).size
        > 0
    )
    frames = tuple(sorted({by_mask[mask_id].frame_id for mask_id in supporting}))
    if len(frames) < config.min_views:
        return None
    supporting_set = set(supporting)
    relevant = [
        edge
        for edge in edges
        if set(edge.left_mask_ids + edge.right_mask_ids).issubset(supporting_set)
    ]
    mean_consensus = (
        float(np.mean([edge.consensus for edge in relevant])) if relevant else 0.0
    )
    positions = np.searchsorted(filtered_ids, part)
    mean_detection = float(np.mean(filtered_ratios[positions]))
    quality = float(np.sqrt(max(0.0, mean_consensus) * max(0.0, mean_detection)))
    return ConsensusObject(
        object_id=-1,
        mask_ids=supporting,
        frame_ids=frames,
        gaussian_ids=part,
        mean_view_consensus=mean_consensus,
        mean_detection_ratio=mean_detection,
        geometric_quality=quality,
    )


def _final_supported_masks(
    gaussian_ids: np.ndarray,
    mask_ids: Sequence[int],
    *,
    by_mask: Mapping[int, MaskObservation],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    masks = tuple(
        mask_id
        for mask_id in sorted(map(int, mask_ids))
        if np.intersect1d(
            gaussian_ids,
            by_mask[mask_id].association_ids,
            assume_unique=True,
        ).size
        > 0
    )
    frames = tuple(sorted({by_mask[mask_id].frame_id for mask_id in masks}))
    return masks, frames


def _unique_ownership(
    objects: Sequence[ConsensusObject],
    *,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    config: ConsensusConfig,
) -> tuple[FunnelObject, ...]:
    occupied = np.zeros(bank.point_count, dtype=np.bool_)
    result: list[FunnelObject] = []
    ranked = sorted(
        objects,
        key=lambda item: (
            -item.geometric_quality,
            -len(item.gaussian_ids),
            item.mask_ids,
            item.object_id,
        ),
    )
    for item in ranked:
        available = item.gaussian_ids[~occupied[item.gaussian_ids]]
        if len(available) < config.dbscan_min_samples:
            continue
        for part in split_disconnected_support(
            available,
            bank.xyz_m,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        ):
            masks, frames = _final_supported_masks(
                part, item.mask_ids, by_mask=by_mask
            )
            if len(frames) < config.min_views:
                continue
            occupied[part] = True
            result.append(
                FunnelObject(
                    stable_id=f"object:{len(result)}",
                    gaussian_ids=part,
                    mask_ids=masks,
                    frame_ids=frames,
                    metadata={
                        "source_geometric_quality": float(item.geometric_quality),
                    },
                )
            )
    return tuple(result)


def _class_for_masks(
    mask_ids: Sequence[int], bank: AlphaMaskEvidenceBank
) -> tuple[str | None, float]:
    by_frame: dict[int, list[np.ndarray]] = {}
    for mask_id in mask_ids:
        row = bank.mask_position(mask_id)
        if bool(bank.semantic_abstained[row]):
            continue
        posterior = np.asarray(bank.semantic_posteriors[row], dtype=np.float64)
        total = float(posterior.sum())
        if total <= 0:
            continue
        by_frame.setdefault(bank.masks[row].frame_id, []).append(posterior / total)
    frame_rows = []
    for frame_id in sorted(by_frame):
        posterior = np.mean(by_frame[frame_id], axis=0)
        total = float(posterior.sum())
        if total > 0:
            frame_rows.append(posterior / total)
    if not frame_rows:
        return None, 0.0
    posterior = np.mean(frame_rows, axis=0)
    posterior /= float(posterior.sum())
    winner = int(np.flatnonzero(posterior == posterior.max())[0])
    return bank.class_names[winner], float(posterior[winner])


def _export_reconstruction(
    unique_objects: Sequence[FunnelObject],
    *,
    bank: AlphaMaskEvidenceBank,
    allowed_classes: Sequence[str],
) -> tuple[FunnelObject, ...]:
    allowed = {str(value) for value in allowed_classes}
    rows: list[FunnelObject] = []
    for item in unique_objects:
        class_name, winner_probability = _class_for_masks(item.mask_ids, bank)
        if class_name is None or class_name not in allowed:
            continue
        rows.append(
            FunnelObject(
                stable_id=f"export:{len(rows)}",
                gaussian_ids=item.gaussian_ids,
                mask_ids=item.mask_ids,
                frame_ids=item.frame_ids,
                class_name=class_name,
                metadata={"winner_probability": winner_probability},
            )
        )
    return tuple(rows)


def _objects_from_output(
    output: Mapping[str, Any], *, point_count: int
) -> tuple[FunnelObject, ...]:
    labels = np.asarray(output.get("point_labels"))
    if labels.shape != (point_count,) or labels.dtype.kind not in "iuf":
        raise ValueError("output point_labels have the wrong shape or type")
    if labels.dtype.kind == "f" and (
        np.any(~np.isfinite(labels)) or np.any(labels != np.floor(labels))
    ):
        raise TypeError("output point_labels must contain integers")
    labels = labels.astype(np.int64, copy=False)
    instances = output.get("instances")
    if not isinstance(instances, Mapping):
        raise TypeError("output instances must be a mapping")
    rows: list[FunnelObject] = []
    observed = {int(value) for value in np.unique(labels) if int(value) >= 0}
    declared = {int(value) for value in instances}
    if observed != declared:
        raise ValueError("output has orphan labels or empty metadata instances")
    for instance_id in sorted(declared):
        metadata = instances[str(instance_id)]
        if not isinstance(metadata, Mapping) or "class" not in metadata:
            raise ValueError("output instance metadata omitted class")
        ids = np.flatnonzero(labels == instance_id).astype(np.int64)
        rows.append(
            FunnelObject(
                stable_id=f"frozen-export:{instance_id}",
                gaussian_ids=ids,
                class_name=str(metadata["class"]),
                metadata={"instance_id": instance_id},
            )
        )
    return tuple(rows)


def _partition_equivalence(
    reconstructed: Sequence[FunnelObject],
    frozen: Sequence[FunnelObject],
    *,
    point_count: int,
) -> PartitionEquivalence:
    reconstructed_rows = tuple(reconstructed)
    frozen_rows = tuple(frozen)
    point_count_exact = all(
        item.gaussian_ids.size == 0 or int(item.gaussian_ids[-1]) < point_count
        for item in reconstructed_rows + frozen_rows
    )
    reconstructed_labels = np.full(point_count, -1, dtype=np.int64)
    frozen_labels = np.full(point_count, -1, dtype=np.int64)
    for index, item in enumerate(reconstructed_rows):
        if np.any(reconstructed_labels[item.gaussian_ids] >= 0):
            raise ValueError("reconstructed export contains duplicate ownership")
        reconstructed_labels[item.gaussian_ids] = index
    for index, item in enumerate(frozen_rows):
        if np.any(frozen_labels[item.gaussian_ids] >= 0):
            raise ValueError("frozen export contains duplicate ownership")
        frozen_labels[item.gaussian_ids] = index

    overlap = np.zeros(
        (len(reconstructed_rows), len(frozen_rows)), dtype=np.int64
    )
    for index, item in enumerate(reconstructed_rows):
        values = frozen_labels[item.gaussian_ids]
        values = values[values >= 0]
        if values.size:
            overlap[index] = np.bincount(
                values, minlength=len(frozen_rows)
            )
    mapping: dict[int, int] = {}
    if overlap.size:
        from scipy.optimize import linear_sum_assignment

        left, right = linear_sum_assignment(-overlap)
        mapping = {
            int(reconstructed_id): int(frozen_id)
            for reconstructed_id, frozen_id in zip(left, right)
        }
    mapped_labels = np.full(point_count, -1, dtype=np.int64)
    next_unmatched = len(frozen_rows)
    for index, item in enumerate(reconstructed_rows):
        mapped_id = mapping.get(index)
        if mapped_id is None:
            mapped_id = next_unmatched
            next_unmatched += 1
        mapped_labels[item.gaussian_ids] = mapped_id
    changed_points = int(np.count_nonzero(mapped_labels != frozen_labels))
    class_exact = (
        len(reconstructed_rows) == len(frozen_rows)
        and len(mapping) == len(reconstructed_rows)
        and all(
            reconstructed_rows[left].class_name == frozen_rows[right].class_name
            for left, right in mapping.items()
        )
    )
    equivalent = (
        point_count_exact
        and class_exact
        and len(reconstructed_rows) == len(frozen_rows)
        and changed_points == 0
    )
    return PartitionEquivalence(
        equivalent=equivalent,
        changed_points=changed_points,
        reconstructed_instance_count=len(reconstructed_rows),
        frozen_instance_count=len(frozen_rows),
        class_exact=class_exact,
        point_count_exact=point_count_exact,
    )


def reconstruct_clean_stage_funnel(
    *,
    bank: AlphaMaskEvidenceBank,
    diagnostics: Mapping[str, Any],
    output: Mapping[str, Any],
    allowed_classes: Sequence[str],
    metric_callback: MetricCallback | None = None,
) -> CleanStageFunnel:
    """Reconstruct every available clean-baseline stage without GT.

    ``diagnostics`` must be the condition-level ``diagnostics.json``.  Older
    payloads that omit accepted edges or rejected-mask identities still yield
    the first two stages and the frozen final export, but dependent middle
    stages are explicitly unavailable.
    """

    if not isinstance(diagnostics, Mapping) or not isinstance(output, Mapping):
        raise TypeError("diagnostics and output must be mappings")
    scene_id = str(diagnostics.get("scene_id", bank.scene_id))
    condition = str(diagnostics.get("condition", output.get("condition", "")))
    if scene_id != bank.scene_id or str(output.get("scene_id")) != bank.scene_id:
        raise ValueError("bank, diagnostics, and output scene identities differ")
    if str(output.get("condition")) != condition:
        raise ValueError("diagnostics and output condition identities differ")
    config = _parse_config(diagnostics)
    observations = _observations(bank)
    by_mask = {item.mask_id: item for item in observations}
    stages: list[FunnelStage] = []
    issues: list[str] = []

    complete = tuple(
        FunnelObject(
            stable_id=f"mask:{item.mask_id}",
            gaussian_ids=item.gaussian_ids,
            mask_ids=(item.mask_id,),
            frame_ids=(item.frame_id,),
        )
        for item in observations
        if item.gaussian_ids.size
    )
    stages.append(
        _stage(
            "complete_mask_support",
            complete,
            details={"input_mask_count": len(observations)},
            metric_callback=metric_callback,
        )
    )
    association = tuple(
        FunnelObject(
            stable_id=f"mask:{item.mask_id}",
            gaussian_ids=item.association_ids,
            mask_ids=(item.mask_id,),
            frame_ids=(item.frame_id,),
            metadata={"ambiguous_removed_count": len(item.ambiguous_ids)},
        )
        for item in observations
        if item.association_ids.size
    )
    stages.append(
        _stage(
            "association_support",
            association,
            details={
                "empty_after_ambiguity_count": len(observations) - len(association),
                "ambiguous_assignment_count": int(
                    sum(len(item.ambiguous_ids) for item in observations)
                ),
            },
            metric_callback=metric_callback,
        )
    )

    if "rejected_undersegmented_mask_ids" not in diagnostics:
        reason = "condition diagnostics omitted rejected_undersegmented_mask_ids"
        issues.append(reason)
        stages.extend(_unavailable(name, reason) for name in STAGE_NAMES[2:7])
        frozen_export = _objects_from_output(output, point_count=bank.point_count)
        stages.append(
            _stage(
                "final_export",
                frozen_export,
                details={"source": "frozen-output-only"},
                metric_callback=metric_callback,
            )
        )
        return CleanStageFunnel(
            scene_id,
            condition,
            bank.point_count,
            tuple(stages),
            None,
            tuple(issues),
        )

    rejected = tuple(sorted(map(int, diagnostics["rejected_undersegmented_mask_ids"])))
    unknown_rejected = set(rejected) - set(by_mask)
    if unknown_rejected:
        raise ValueError("undersegmentation diagnostics reference unknown masks")
    active = tuple(
        item for item in observations if item.mask_id not in set(rejected)
    )
    underseg = tuple(
        FunnelObject(
            stable_id=f"mask:{item.mask_id}",
            gaussian_ids=item.association_ids,
            mask_ids=(item.mask_id,),
            frame_ids=(item.frame_id,),
        )
        for item in active
        if item.association_ids.size
    )
    stages.append(
        _stage(
            "undersegmentation_filtered",
            underseg,
            details={
                "rejected_mask_count": len(rejected),
                "active_mask_count": len(active),
            },
            metric_callback=metric_callback,
        )
    )

    if "accepted_edges" not in diagnostics:
        reason = "condition diagnostics omitted accepted_edges"
        issues.append(reason)
        stages.extend(_unavailable(name, reason) for name in STAGE_NAMES[3:7])
        frozen_export = _objects_from_output(output, point_count=bank.point_count)
        stages.append(
            _stage(
                "final_export",
                frozen_export,
                details={"source": "frozen-output-only"},
                metric_callback=metric_callback,
            )
        )
        return CleanStageFunnel(
            scene_id,
            condition,
            bank.point_count,
            tuple(stages),
            None,
            tuple(issues),
        )

    edges = _parse_edges(diagnostics["accepted_edges"])
    components = _component_mask_ids([item.mask_id for item in active], edges)
    consensus_diagnostics = diagnostics.get("consensus")
    expected_component_count = (
        consensus_diagnostics.get("component_count_before_output_filters")
        if isinstance(consensus_diagnostics, Mapping)
        else None
    )
    if expected_component_count is not None and int(expected_component_count) != len(components):
        reason = (
            "accepted-edge lineage cannot reconstruct the persisted component count "
            f"({len(components)} != {int(expected_component_count)})"
        )
        issues.append(reason)
        stages.extend(_unavailable(name, reason) for name in STAGE_NAMES[3:7])
        frozen_export = _objects_from_output(output, point_count=bank.point_count)
        stages.append(
            _stage(
                "final_export",
                frozen_export,
                details={"source": "frozen-output-only"},
                metric_callback=metric_callback,
            )
        )
        return CleanStageFunnel(
            scene_id,
            condition,
            bank.point_count,
            tuple(stages),
            None,
            tuple(issues),
        )

    component_objects = tuple(
        FunnelObject(
            stable_id="component:" + ",".join(map(str, component)),
            gaussian_ids=_component_full_ids(component, by_mask),
            mask_ids=component,
            frame_ids=tuple(sorted({by_mask[value].frame_id for value in component})),
            metadata={
                "association_gaussian_count": int(
                    len(
                        np.unique(
                            np.concatenate(
                                [by_mask[value].association_ids for value in component]
                            )
                        )
                    )
                ),
            },
        )
        for component in components
        if _component_full_ids(component, by_mask).size
    )
    stages.append(
        _stage(
            "accepted_edge_components",
            component_objects,
            details={
                "accepted_edge_count": len(edges),
                "component_count": len(components),
            },
            metric_callback=metric_callback,
        )
    )

    detection_rows: list[FunnelObject] = []
    filtered_by_component: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
    dropped_reasons: dict[str, int] = {}
    for component in components:
        ids, ratios, dropped_reason = _detection_filter(
            component, bank=bank, by_mask=by_mask, config=config
        )
        if dropped_reason is not None:
            dropped_reasons[dropped_reason] = dropped_reasons.get(dropped_reason, 0) + 1
            continue
        filtered_by_component[component] = (ids, ratios)
        detection_rows.append(
            FunnelObject(
                stable_id="component:" + ",".join(map(str, component)),
                gaussian_ids=ids,
                mask_ids=component,
                frame_ids=tuple(sorted({by_mask[value].frame_id for value in component})),
                metadata={"mean_detection_ratio": float(np.mean(ratios))},
            )
        )
    stages.append(
        _stage(
            "detection_ratio_filtered",
            detection_rows,
            details={"dropped_component_reasons": dropped_reasons},
            metric_callback=metric_callback,
        )
    )

    provisional: list[ConsensusObject] = []
    dropped_connectivity = 0
    dropped_part_views = 0
    for component, (ids, ratios) in filtered_by_component.items():
        parts = split_disconnected_support(
            ids,
            bank.xyz_m,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        )
        if not parts:
            dropped_connectivity += 1
            continue
        for part in parts:
            row = _part_metadata(
                part,
                component,
                by_mask=by_mask,
                edges=edges,
                filtered_ids=ids,
                filtered_ratios=ratios,
                config=config,
            )
            if row is None:
                dropped_part_views += 1
            else:
                provisional.append(row)
    deduplicated = remove_contained_objects(
        provisional, contained_threshold=config.contained_threshold
    )
    physical_rows = tuple(
        FunnelObject(
            stable_id=f"physical:{index}",
            gaussian_ids=item.gaussian_ids,
            mask_ids=item.mask_ids,
            frame_ids=item.frame_ids,
            metadata={
                "geometric_quality": float(item.geometric_quality),
                "mean_view_consensus": float(item.mean_view_consensus),
                "mean_detection_ratio": float(item.mean_detection_ratio),
            },
        )
        for index, item in enumerate(deduplicated)
    )
    stages.append(
        _stage(
            "physical_split_and_deduplicated",
            physical_rows,
            details={
                "provisional_count": len(provisional),
                "contained_duplicate_count": len(provisional) - len(deduplicated),
                "dropped_by_connectivity": dropped_connectivity,
                "dropped_part_by_min_views": dropped_part_views,
            },
            metric_callback=metric_callback,
        )
    )

    unique_rows = _unique_ownership(
        deduplicated, bank=bank, by_mask=by_mask, config=config
    )
    stages.append(
        _stage(
            "unique_gaussian_ownership",
            unique_rows,
            details={
                "overlap_removed_count": int(
                    sum(len(item.gaussian_ids) for item in physical_rows)
                    - sum(len(item.gaussian_ids) for item in unique_rows)
                )
            },
            metric_callback=metric_callback,
        )
    )

    reconstructed_export = _export_reconstruction(
        unique_rows, bank=bank, allowed_classes=allowed_classes
    )
    frozen_export = _objects_from_output(output, point_count=bank.point_count)
    equivalence = _partition_equivalence(
        reconstructed_export, frozen_export, point_count=bank.point_count
    )
    if not equivalence.equivalent:
        issues.append(
            "reconstructed final partition differs from frozen output: "
            f"changed_points={equivalence.changed_points}"
        )
    stages.append(
        _stage(
            "final_export",
            reconstructed_export,
            details={
                "frozen_instance_count": len(frozen_export),
                "equivalence": equivalence.to_dict(),
            },
            metric_callback=metric_callback,
        )
    )
    return CleanStageFunnel(
        scene_id,
        condition,
        bank.point_count,
        tuple(stages),
        equivalence,
        tuple(issues),
    )


def audit_frozen_clean_scene(
    *,
    bank_dir: str | Path,
    diagnostics_path: str | Path,
    output_path: str | Path,
    allowed_classes: Sequence[str],
    metric_callback: MetricCallback | None = None,
) -> CleanStageFunnel:
    """Path-oriented wrapper for frozen cloud artifacts."""

    bank = load_evidence_bank(bank_dir)
    diagnostics = load_json(diagnostics_path)
    output = load_json(output_path)
    if not isinstance(diagnostics, Mapping) or not isinstance(output, Mapping):
        raise TypeError("frozen diagnostics and output must contain JSON objects")
    return reconstruct_clean_stage_funnel(
        bank=bank,
        diagnostics=diagnostics,
        output=output,
        allowed_classes=allowed_classes,
        metric_callback=metric_callback,
    )


def write_funnel_summary(path: str | Path, funnel: CleanStageFunnel) -> None:
    """Write a compact summary; point memberships remain in the immutable bank."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.write_text(
        json.dumps(funnel.to_summary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


__all__ = [
    "CleanStageFunnel",
    "FunnelObject",
    "FunnelStage",
    "MetricCallback",
    "PartitionEquivalence",
    "STAGE_NAMES",
    "audit_frozen_clean_scene",
    "reconstruct_clean_stage_funnel",
    "write_funnel_summary",
]
