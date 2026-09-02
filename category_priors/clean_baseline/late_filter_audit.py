"""Pure replay of the clean baseline's late filtering decisions.

The expensive SAM generation, alpha lifting, and view-consensus search are
outside this module.  The replay starts from the immutable evidence bank and
the accepted-edge lineage persisted in a condition ``diagnostics.json``.  It
therefore cannot create a new mask, edge, component, semantic label, or prior.

Two factors are replayed on the same accepted components:

``A1``
    Historical per-Gaussian detection-ratio hard filtering.
``A0``
    Keep the complete member-mask full union while computing exactly the same
    per-Gaussian detection ratios for diagnostics and scoring.
``B1``
    Historical terminal ownership and semantic export.
``B0``
    Stop after historical physical splitting and containment deduplication.
    The resulting candidates may overlap and are diagnostic only; no formal
    ``output.json`` partition exists for this arm.

Ground truth and category priors are deliberately absent from every public
signature.  An optional frozen prediction can be supplied only to verify that
``A1B1`` reconstructs the already persisted output exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ..io import load_json
from .consensus import (
    ConsensusEdge,
    ConsensusObject,
    MaskObservation,
    remove_contained_objects,
    split_disconnected_support,
)
from .evidence import load_evidence_bank
from .evaluation import CleanCandidate
from .models import AlphaMaskEvidenceBank
from .pipeline import _object_class_distribution, _resolve_unique_ownership
from .stage_funnel import (
    FunnelObject,
    PartitionEquivalence,
    _component_full_ids,
    _component_mask_ids,
    _detection_profile,
    _final_supported_masks,
    _objects_from_output,
    _observations,
    _parse_config,
    _parse_edges,
    _part_metadata,
    _partition_equivalence,
    _unique_ownership,
)


ARM_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "A1B1": "A1B1-current",
        "A0B1": "A0B1-no-detection-hard-filter",
        "A1B0": "A1B0-pre-late-pruning",
        "A0B0": "A0B0-both-relaxed",
    }
)


_DROP_REASON_KEYS = (
    "min_views",
    "empty_full",
    "detection_ratio",
    "dbscan_no_valid_part",
    "split_part_min_views",
    "contained_duplicate",
    "ownership_residual_below_min_samples",
    "ownership_dbscan_no_valid_part",
    "ownership_part_min_views",
    "semantic_abstain",
    "non_allowed_class",
)


def _readonly_ids(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be a one-dimensional integer array")
    result = np.unique(raw.astype(np.int64, copy=False))
    if np.any(result < 0):
        raise ValueError(f"{name} must be non-negative")
    result.setflags(write=False)
    return result


def _readonly_ratios(value: object, *, expected: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (expected,) or not np.isfinite(result).all():
        raise ValueError("detection_ratios must be a finite vector aligned to full_ids")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("detection_ratios must lie in [0, 1]")
    result = result.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ComponentDetectionEvidence:
    """Shared per-component evidence consumed by both A arms."""

    stable_id: str
    mask_ids: tuple[int, ...]
    frame_ids: tuple[int, ...]
    full_ids: np.ndarray
    detection_ratios: np.ndarray
    historical_ids: np.ndarray
    dropped_reason: str | None = None

    def __post_init__(self) -> None:
        stable_id = str(self.stable_id)
        if not stable_id:
            raise ValueError("stable_id must be non-empty")
        mask_ids = tuple(sorted({int(value) for value in self.mask_ids}))
        frame_ids = tuple(sorted({int(value) for value in self.frame_ids}))
        if any(value < 0 for value in mask_ids + frame_ids):
            raise ValueError("mask_ids and frame_ids must be non-negative")
        full_ids = _readonly_ids(self.full_ids, name="full_ids")
        ratios = _readonly_ratios(self.detection_ratios, expected=len(full_ids))
        historical_ids = _readonly_ids(self.historical_ids, name="historical_ids")
        if np.setdiff1d(historical_ids, full_ids, assume_unique=True).size:
            raise ValueError("historical_ids must be a subset of full_ids")
        if self.dropped_reason not in (None, "min_views", "empty_full", "detection_ratio"):
            raise ValueError(f"unknown component drop reason: {self.dropped_reason}")
        object.__setattr__(self, "stable_id", stable_id)
        object.__setattr__(self, "mask_ids", mask_ids)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "full_ids", full_ids)
        object.__setattr__(self, "detection_ratios", ratios)
        object.__setattr__(self, "historical_ids", historical_ids)

    def selected(self, *, detection_hard_filter: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return membership and aligned ratios for one A condition."""

        if self.dropped_reason in {"min_views", "empty_full"}:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float64),
            )
        if detection_hard_filter:
            selected = self.historical_ids
            if selected.size == 0:
                return selected, np.empty(0, dtype=np.float64)
            positions = np.searchsorted(self.full_ids, selected)
            return selected, self.detection_ratios[positions]
        return self.full_ids, self.detection_ratios


@dataclass(frozen=True)
class LateFilterArmReplay:
    """One fixed arm of the A x B late-filter factorial."""

    code: str
    name: str
    detection_hard_filter: bool
    strict_late_export: bool
    detection_objects: tuple[FunnelObject, ...]
    physical_objects: tuple[FunnelObject, ...]
    ownership_objects: tuple[FunnelObject, ...]
    formal_output: tuple[FunnelObject, ...] | None
    drop_reasons: Mapping[str, int] = field(default_factory=dict)
    final_equivalence: PartitionEquivalence | None = None

    def __post_init__(self) -> None:
        if self.code not in ARM_NAMES or self.name != ARM_NAMES[self.code]:
            raise ValueError("late-filter arm name/code do not match the frozen registry")
        if self.strict_late_export != self.code.endswith("B1"):
            raise ValueError("strict_late_export disagrees with the arm code")
        if self.detection_hard_filter != self.code.startswith("A1"):
            raise ValueError("detection_hard_filter disagrees with the arm code")
        if self.strict_late_export and self.formal_output is None:
            raise ValueError("B1 must expose its reconstructed formal output")
        if not self.strict_late_export and self.formal_output is not None:
            raise ValueError("B0 is diagnostic-only and cannot expose formal output")
        reasons = {key: int(self.drop_reasons.get(key, 0)) for key in _DROP_REASON_KEYS}
        unknown = set(self.drop_reasons) - set(_DROP_REASON_KEYS)
        if unknown:
            raise ValueError(f"unknown drop-reason counters: {sorted(unknown)}")
        if any(value < 0 for value in reasons.values()):
            raise ValueError("drop-reason counters must be non-negative")
        object.__setattr__(self, "detection_objects", tuple(self.detection_objects))
        object.__setattr__(self, "physical_objects", tuple(self.physical_objects))
        object.__setattr__(self, "ownership_objects", tuple(self.ownership_objects))
        if self.formal_output is not None:
            object.__setattr__(self, "formal_output", tuple(self.formal_output))
        object.__setattr__(self, "drop_reasons", MappingProxyType(reasons))

    @property
    def formal_output_allowed(self) -> bool:
        return bool(self.strict_late_export)

    @property
    def diagnostic_candidates(self) -> tuple[FunnelObject, ...]:
        """Candidates that an offline one-to-one evaluator may inspect."""

        return self.physical_objects

    def to_summary(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "detection_hard_filter": self.detection_hard_filter,
            "strict_late_export": self.strict_late_export,
            "formal_output_allowed": self.formal_output_allowed,
            "detection_object_count": len(self.detection_objects),
            "physical_object_count": len(self.physical_objects),
            "ownership_object_count": len(self.ownership_objects),
            "formal_output_count": (
                None if self.formal_output is None else len(self.formal_output)
            ),
            "drop_reasons": dict(self.drop_reasons),
            "final_equivalence": (
                None
                if self.final_equivalence is None
                else self.final_equivalence.to_dict()
            ),
        }


@dataclass(frozen=True)
class LateFilterFactorialReplay:
    """All four arms reconstructed from one immutable accepted-component set."""

    scene_id: str
    condition: str
    point_count: int
    accepted_components: tuple[FunnelObject, ...]
    detection_evidence: tuple[ComponentDetectionEvidence, ...]
    arms: Mapping[str, LateFilterArmReplay]
    shared_identity: Mapping[str, Any]
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        arms = dict(self.arms)
        if tuple(arms) != tuple(ARM_NAMES):
            raise ValueError("arms must contain A1B1/A0B1/A1B0/A0B0 in registry order")
        object.__setattr__(self, "accepted_components", tuple(self.accepted_components))
        object.__setattr__(self, "detection_evidence", tuple(self.detection_evidence))
        object.__setattr__(self, "arms", MappingProxyType(arms))
        object.__setattr__(self, "shared_identity", MappingProxyType(dict(self.shared_identity)))
        object.__setattr__(self, "issues", tuple(map(str, self.issues)))

    def arm(self, code: str) -> LateFilterArmReplay:
        return self.arms[str(code)]

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": "saga-clean-late-filter-factorial-v1",
            "scene_id": self.scene_id,
            "condition": self.condition,
            "point_count": self.point_count,
            "accepted_component_count": len(self.accepted_components),
            "component_detection_evidence_count": len(self.detection_evidence),
            "shared_identity": dict(self.shared_identity),
            "arms": {code: arm.to_summary() for code, arm in self.arms.items()},
            "issues": list(self.issues),
        }


def _component_stable_id(component: Sequence[int]) -> str:
    return "component:" + ",".join(map(str, component))


def _build_detection_evidence(
    components: Sequence[tuple[int, ...]],
    *,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    config: Any,
) -> tuple[ComponentDetectionEvidence, ...]:
    rows: list[ComponentDetectionEvidence] = []
    for component in components:
        full_ids, ratios, precondition = _detection_profile(
            component,
            bank=bank,
            by_mask=by_mask,
            config=config,
        )
        if precondition is None:
            keep = ratios >= config.point_filter_threshold
            historical_ids = full_ids[keep]
            dropped_reason = None if historical_ids.size else "detection_ratio"
        else:
            historical_ids = np.empty(0, dtype=np.int64)
            dropped_reason = precondition
        rows.append(
            ComponentDetectionEvidence(
                stable_id=_component_stable_id(component),
                mask_ids=component,
                frame_ids=tuple(
                    sorted({by_mask[mask_id].frame_id for mask_id in component})
                ),
                full_ids=full_ids,
                detection_ratios=ratios,
                historical_ids=historical_ids,
                dropped_reason=dropped_reason,
            )
        )
    return tuple(rows)


def _candidate_with_posterior(
    item: ConsensusObject,
    *,
    index: int,
    bank: AlphaMaskEvidenceBank,
    allowed_classes: Sequence[str],
) -> FunnelObject:
    class_name, winner_probability = _production_class_for_masks(
        item.mask_ids, bank
    )
    allowed = {str(value) for value in allowed_classes}
    return FunnelObject(
        stable_id=f"physical:{index}",
        gaussian_ids=item.gaussian_ids,
        mask_ids=item.mask_ids,
        frame_ids=item.frame_ids,
        class_name=class_name,
        metadata={
            "geometric_quality": float(item.geometric_quality),
            "mean_view_consensus": float(item.mean_view_consensus),
            "mean_detection_ratio": float(item.mean_detection_ratio),
            "winner_probability": float(winner_probability),
            "semantic_abstain": class_name is None,
            "class_allowed": class_name in allowed if class_name is not None else False,
        },
    )


def _production_class_for_masks(
    mask_ids: Sequence[int], bank: AlphaMaskEvidenceBank
) -> tuple[str | None, float]:
    """Use the production pipeline's per-view semantic reducer exactly."""

    posterior = _object_class_distribution(mask_ids, bank)
    if posterior.sum() <= 0:
        return None, 0.0
    winner = int(np.flatnonzero(posterior == posterior.max())[0])
    return bank.class_names[winner], float(posterior[winner])


def _physical_replay(
    evidence: Sequence[ComponentDetectionEvidence],
    *,
    detection_hard_filter: bool,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    edges: Sequence[ConsensusEdge],
    config: Any,
    allowed_classes: Sequence[str],
) -> tuple[
    tuple[FunnelObject, ...],
    tuple[ConsensusObject, ...],
    tuple[FunnelObject, ...],
    dict[str, int],
]:
    reasons = {key: 0 for key in _DROP_REASON_KEYS}
    detection_objects: list[FunnelObject] = []
    provisional: list[ConsensusObject] = []
    for row in evidence:
        if row.dropped_reason in {"min_views", "empty_full"}:
            reasons[row.dropped_reason] += 1
            continue
        ids, ratios = row.selected(detection_hard_filter=detection_hard_filter)
        if ids.size == 0:
            reasons["detection_ratio"] += 1
            continue
        detection_objects.append(
            FunnelObject(
                stable_id=row.stable_id,
                gaussian_ids=ids,
                mask_ids=row.mask_ids,
                frame_ids=row.frame_ids,
                metadata={
                    "mean_detection_ratio": float(np.mean(ratios)),
                    "zero_detection_ratio_count": int(np.count_nonzero(ratios == 0.0)),
                    "full_union_count": int(len(row.full_ids)),
                    "hard_filter_applied": bool(detection_hard_filter),
                },
            )
        )
        parts = split_disconnected_support(
            ids,
            bank.xyz_m,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        )
        if not parts:
            reasons["dbscan_no_valid_part"] += 1
            continue
        for part in parts:
            candidate = _part_metadata(
                part,
                row.mask_ids,
                by_mask=by_mask,
                edges=edges,
                filtered_ids=ids,
                filtered_ratios=ratios,
                config=config,
            )
            if candidate is None:
                reasons["split_part_min_views"] += 1
            else:
                provisional.append(candidate)
    deduplicated = remove_contained_objects(
        provisional,
        contained_threshold=config.contained_threshold,
    )
    reasons["contained_duplicate"] = len(provisional) - len(deduplicated)
    physical = tuple(
        _candidate_with_posterior(
            item,
            index=index,
            bank=bank,
            allowed_classes=allowed_classes,
        )
        for index, item in enumerate(deduplicated)
    )
    return tuple(detection_objects), tuple(deduplicated), physical, reasons


def _historical_ownership_with_reasons(
    objects: Sequence[ConsensusObject],
    *,
    bank: AlphaMaskEvidenceBank,
    by_mask: Mapping[int, MaskObservation],
    config: Any,
    edges: Sequence[ConsensusEdge],
    rejected_mask_ids: Sequence[int],
    observations: Sequence[MaskObservation],
) -> tuple[tuple[FunnelObject, ...], dict[str, int]]:
    """Run the terminal ownership logic while exposing each hard drop."""

    reasons = {key: 0 for key in _DROP_REASON_KEYS}
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
            reasons["ownership_residual_below_min_samples"] += 1
            continue
        parts = split_disconnected_support(
            available,
            bank.xyz_m,
            eps_m=config.dbscan_eps_m,
            min_samples=config.dbscan_min_samples,
        )
        if not parts:
            reasons["ownership_dbscan_no_valid_part"] += 1
            continue
        for part in parts:
            masks, frames = _final_supported_masks(
                part,
                item.mask_ids,
                by_mask=by_mask,
            )
            if len(frames) < config.min_views:
                reasons["ownership_part_min_views"] += 1
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
    simplified = _unique_ownership(
        objects,
        bank=bank,
        by_mask=by_mask,
        config=config,
    )
    if len(result) != len(simplified) or any(
        left.mask_ids != right.mask_ids
        or left.frame_ids != right.frame_ids
        or not np.array_equal(left.gaussian_ids, right.gaussian_ids)
        for left, right in zip(result, simplified)
    ):
        raise RuntimeError("ownership diagnostics drifted from the historical terminal")
    canonical = _resolve_unique_ownership(
        objects,
        bank,
        accepted_edges=edges,
        rejected_undersegmented_mask_ids=rejected_mask_ids,
        config=config,
        observations=observations,
        visibility=None,
    )
    if len(result) != len(canonical) or any(
        left.mask_ids != right.mask_ids
        or left.frame_ids != right.frame_ids
        or not np.array_equal(left.gaussian_ids, right.gaussian_ids)
        for left, right in zip(result, canonical)
    ):
        raise RuntimeError("ownership replay drifted from the production terminal")
    exact = tuple(
        FunnelObject(
            stable_id=f"object:{index}",
            gaussian_ids=item.gaussian_ids,
            mask_ids=item.mask_ids,
            frame_ids=item.frame_ids,
            metadata={
                "source_geometric_quality": float(item.geometric_quality),
                "mean_view_consensus": float(item.mean_view_consensus),
                "mean_detection_ratio": float(item.mean_detection_ratio),
                "geometric_quality": float(item.geometric_quality),
            },
        )
        for index, item in enumerate(canonical)
    )
    return exact, reasons


def _export_with_reasons(
    ownership: Sequence[FunnelObject],
    *,
    bank: AlphaMaskEvidenceBank,
    allowed_classes: Sequence[str],
) -> tuple[tuple[FunnelObject, ...], dict[str, int]]:
    reasons = {key: 0 for key in _DROP_REASON_KEYS}
    allowed = {str(value) for value in allowed_classes}
    rows: list[FunnelObject] = []
    for item in ownership:
        class_name, winner_probability = _production_class_for_masks(
            item.mask_ids, bank
        )
        if class_name is None:
            reasons["semantic_abstain"] += 1
        elif class_name not in allowed:
            reasons["non_allowed_class"] += 1
        else:
            mean_consensus = float(item.metadata["mean_view_consensus"])
            mean_detection = float(item.metadata["mean_detection_ratio"])
            production_candidate = CleanCandidate(
                object_id=item.stable_id,
                gaussian_ids=item.gaussian_ids,
                class_id=class_name,
                winner_probability=winner_probability,
                view_consensus=mean_consensus,
                detection_ratio=mean_detection,
            )
            geometric_quality = float(item.metadata["geometric_quality"])
            rows.append(
                FunnelObject(
                    stable_id=f"export:{len(rows)}",
                    gaussian_ids=item.gaussian_ids,
                    mask_ids=item.mask_ids,
                    frame_ids=item.frame_ids,
                    class_name=class_name,
                    metadata={
                        "winner_probability": float(winner_probability),
                        "mean_view_consensus": mean_consensus,
                        "mean_detection_ratio": mean_detection,
                        "geometric_quality": geometric_quality,
                        # This is the exact production score implementation,
                        # not a numerically-close replay of its formula.
                        "score": production_candidate.score,
                    },
                )
            )
    return tuple(rows), reasons


def _merge_reason_counts(*rows: Mapping[str, int]) -> dict[str, int]:
    result = {key: 0 for key in _DROP_REASON_KEYS}
    for row in rows:
        for key, value in row.items():
            if key not in result:
                raise ValueError(f"unknown drop reason: {key}")
            result[key] += int(value)
    return result


def replay_late_filter_factorial(
    *,
    bank: AlphaMaskEvidenceBank,
    diagnostics: Mapping[str, Any],
    allowed_classes: Sequence[str],
    frozen_output: Mapping[str, Any] | None = None,
) -> LateFilterFactorialReplay:
    """Replay A1B1/A0B1/A1B0/A0B0 from one frozen bank and lineage.

    ``frozen_output`` is optional and is never used to construct an arm.  When
    present, it is used only for the A1B1 partition-equivalence assertion.
    """

    if not isinstance(diagnostics, Mapping):
        raise TypeError("diagnostics must be a mapping")
    if frozen_output is not None and not isinstance(frozen_output, Mapping):
        raise TypeError("frozen_output must be a mapping or None")
    scene_id = str(diagnostics.get("scene_id", bank.scene_id))
    condition = str(diagnostics.get("condition", ""))
    if scene_id != bank.scene_id:
        raise ValueError("bank and diagnostics scene identities differ")
    if frozen_output is not None:
        if str(frozen_output.get("scene_id")) != bank.scene_id:
            raise ValueError("bank and frozen output scene identities differ")
        if str(frozen_output.get("condition")) != condition:
            raise ValueError("diagnostics and frozen output condition identities differ")

    config = _parse_config(diagnostics)
    observations = _observations(bank)
    by_mask = {item.mask_id: item for item in observations}
    if "rejected_undersegmented_mask_ids" not in diagnostics:
        raise ValueError("diagnostics omitted rejected_undersegmented_mask_ids")
    if "accepted_edges" not in diagnostics:
        raise ValueError("diagnostics omitted accepted_edges")
    rejected = tuple(
        sorted(map(int, diagnostics["rejected_undersegmented_mask_ids"]))
    )
    unknown_rejected = set(rejected) - set(by_mask)
    if unknown_rejected:
        raise ValueError("undersegmentation diagnostics reference unknown masks")
    rejected_set = set(rejected)
    active_mask_ids = tuple(
        item.mask_id for item in observations if item.mask_id not in rejected_set
    )
    edges = _parse_edges(diagnostics["accepted_edges"])
    components = _component_mask_ids(active_mask_ids, edges)
    consensus = diagnostics.get("consensus")
    expected = (
        consensus.get("component_count_before_output_filters")
        if isinstance(consensus, Mapping)
        else None
    )
    if expected is not None and int(expected) != len(components):
        raise ValueError(
            "accepted-edge lineage cannot reconstruct persisted component count "
            f"({len(components)} != {int(expected)})"
        )

    accepted_components = tuple(
        FunnelObject(
            stable_id=_component_stable_id(component),
            gaussian_ids=_component_full_ids(component, by_mask),
            mask_ids=component,
            frame_ids=tuple(
                sorted({by_mask[mask_id].frame_id for mask_id in component})
            ),
        )
        for component in components
    )
    evidence = _build_detection_evidence(
        components,
        bank=bank,
        by_mask=by_mask,
        config=config,
    )

    frozen_objects = (
        None
        if frozen_output is None
        else _objects_from_output(frozen_output, point_count=bank.point_count)
    )
    arms: dict[str, LateFilterArmReplay] = {}
    issues: list[str] = []
    physical_cache: dict[bool, tuple[Any, ...]] = {}
    for hard in (True, False):
        physical_cache[hard] = _physical_replay(
            evidence,
            detection_hard_filter=hard,
            bank=bank,
            by_mask=by_mask,
            edges=edges,
            config=config,
            allowed_classes=allowed_classes,
        )

    for code in ARM_NAMES:
        hard = code.startswith("A1")
        strict = code.endswith("B1")
        detection_objects, consensus_objects, physical_objects, physical_reasons = (
            physical_cache[hard]
        )
        ownership: tuple[FunnelObject, ...] = ()
        formal_output: tuple[FunnelObject, ...] | None = None
        terminal_reasons = {key: 0 for key in _DROP_REASON_KEYS}
        equivalence: PartitionEquivalence | None = None
        if strict:
            ownership, ownership_reasons = _historical_ownership_with_reasons(
                consensus_objects,
                bank=bank,
                by_mask=by_mask,
                config=config,
                edges=edges,
                rejected_mask_ids=rejected,
                observations=observations,
            )
            formal_output, export_reasons = _export_with_reasons(
                ownership,
                bank=bank,
                allowed_classes=allowed_classes,
            )
            terminal_reasons = _merge_reason_counts(
                ownership_reasons,
                export_reasons,
            )
            if code == "A1B1" and frozen_objects is not None:
                equivalence = _partition_equivalence(
                    formal_output,
                    frozen_objects,
                    point_count=bank.point_count,
                )
                if not equivalence.equivalent:
                    issues.append(
                        "A1B1 differs from frozen output: "
                        f"changed_points={equivalence.changed_points}"
                    )
        arms[code] = LateFilterArmReplay(
            code=code,
            name=ARM_NAMES[code],
            detection_hard_filter=hard,
            strict_late_export=strict,
            detection_objects=detection_objects,
            physical_objects=physical_objects,
            ownership_objects=ownership,
            formal_output=formal_output,
            drop_reasons=_merge_reason_counts(physical_reasons, terminal_reasons),
            final_equivalence=equivalence,
        )

    shared_identity = {
        "active_mask_ids": active_mask_ids,
        "accepted_edge_count": len(edges),
        "component_mask_ids": components,
        "component_count": len(components),
        "component_identity_shared": True,
        "detection_ratio_identity_shared": True,
    }
    return LateFilterFactorialReplay(
        scene_id=scene_id,
        condition=condition,
        point_count=bank.point_count,
        accepted_components=accepted_components,
        detection_evidence=evidence,
        arms=arms,
        shared_identity=shared_identity,
        issues=tuple(issues),
    )


def audit_frozen_late_filters(
    *,
    bank_dir: str | Path,
    diagnostics_path: str | Path,
    allowed_classes: Sequence[str],
    output_path: str | Path | None = None,
) -> LateFilterFactorialReplay:
    """Path-oriented read-only wrapper for frozen cloud artifacts."""

    bank = load_evidence_bank(bank_dir)
    diagnostics = load_json(diagnostics_path)
    output = None if output_path is None else load_json(output_path)
    if not isinstance(diagnostics, Mapping):
        raise TypeError("frozen diagnostics must contain a JSON object")
    if output is not None and not isinstance(output, Mapping):
        raise TypeError("frozen output must contain a JSON object")
    return replay_late_filter_factorial(
        bank=bank,
        diagnostics=diagnostics,
        allowed_classes=allowed_classes,
        frozen_output=output,
    )


__all__ = [
    "ARM_NAMES",
    "ComponentDetectionEvidence",
    "LateFilterArmReplay",
    "LateFilterFactorialReplay",
    "audit_frozen_late_filters",
    "replay_late_filter_factorial",
]
