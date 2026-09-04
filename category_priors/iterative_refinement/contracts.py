from __future__ import annotations

"""Immutable contracts shared by the iterative-refinement runtime and replay."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


SCHEMA = "saga-iterative-refinement-v1"


def readonly_int(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1 or np.any(result < 0):
        raise ValueError("point IDs must be a one-dimensional non-negative array")
    if len(result) and len(np.unique(result)) != len(result):
        raise ValueError("point IDs must be unique")
    result.sort()
    result.setflags(write=False)
    return result


def readonly_float(values: Any) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("evidence arrays must be one-dimensional and finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class RefinementProfile:
    name: str
    alpha_weight: float
    pairwise_weight: float

    def __post_init__(self) -> None:
        if self.name not in {"stable", "balanced", "coverage"}:
            raise ValueError(f"unknown refinement profile: {self.name}")
        if self.alpha_weight <= 0 or self.pairwise_weight <= 0:
            raise ValueError("profile weights must be positive")


PROFILES = MappingProxyType(
    {
        "stable": RefinementProfile("stable", 0.5, 2.0),
        "balanced": RefinementProfile("balanced", 1.0, 1.0),
        "coverage": RefinementProfile("coverage", 2.0, 0.5),
    }
)


@dataclass(frozen=True)
class RefinementConfig:
    min_projected_pixels: int = 4
    max_views: int = 6
    views_per_round: int = 3
    min_ray_angle_deg: float = 15.0
    min_baseline_depth_ratio: float = 0.05
    box_threshold: float = 0.35
    text_threshold: float = 0.35
    nms_threshold: float = 0.80
    max_boxes_per_view: int = 2
    max_masks_per_view: int = 2
    mask_jaccard_min: float = 0.30
    hard_opacity_min: float = 0.50
    hard_contributor_fraction_min: float = 0.50
    hard_negative_inside_ratio_max: float = 0.10
    alpha_opacity_min: float = 0.05
    alpha_inside_mass_min: float = 0.50
    alpha_inside_ratio_min: float = 0.50
    graph_radius_fraction: float = 0.15
    graph_radius_min_m: float = 0.03
    graph_radius_max_m: float = 0.15
    graph_edge_radius_m: float = 0.05
    graph_neighbors: int = 16
    graph_node_limit: int = 100_000
    merge_mask_iou_min: float = 0.80
    merge_seed_coverage_min: float = 0.50
    merge_distance_m: float = 0.05
    final_label_threshold: float = 0.30
    minimum_new_object_points: int = 3
    b0_two_view_carve_fraction: float = 0.20
    b0_max_carve_fraction: float = 0.50
    b0_disconnect_fraction: float = 0.10
    round_change_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.max_views != 2 * self.views_per_round:
            raise ValueError("the registered runtime requires two equal view rounds")
        probability_fields = (
            self.box_threshold,
            self.text_threshold,
            self.nms_threshold,
            self.mask_jaccard_min,
            self.hard_opacity_min,
            self.hard_contributor_fraction_min,
            self.hard_negative_inside_ratio_max,
            self.alpha_opacity_min,
            self.alpha_inside_ratio_min,
            self.merge_mask_iou_min,
            self.merge_seed_coverage_min,
            self.final_label_threshold,
            self.b0_two_view_carve_fraction,
            self.b0_max_carve_fraction,
            self.b0_disconnect_fraction,
            self.round_change_fraction,
        )
        if any(not 0 <= value <= 1 for value in probability_fields):
            raise ValueError("registered probability/fraction thresholds must be in [0,1]")
        if self.graph_neighbors < 1 or self.graph_node_limit < 1:
            raise ValueError("graph limits must be positive")


@dataclass(frozen=True)
class CandidateSeed:
    candidate_id: int
    parent_candidate_ids: tuple[int, ...]
    branch_class: str
    seed_support: np.ndarray
    seed_anchor: np.ndarray
    anchor_stage: str | None
    q_score: float
    reachable: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        support = readonly_int(self.seed_support)
        anchor = readonly_int(self.seed_anchor)
        if not set(anchor.tolist()).issubset(set(support.tolist())):
            raise ValueError("seed_anchor must be a subset of seed_support")
        if self.candidate_id < 0 or not self.parent_candidate_ids:
            raise ValueError("candidate IDs must be non-negative and have lineage")
        if not np.isfinite(self.q_score) or not 0 <= self.q_score <= 1:
            raise ValueError("candidate Q must be in [0,1]")
        object.__setattr__(self, "seed_support", support)
        object.__setattr__(self, "seed_anchor", anchor)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class ViewObservation:
    candidate_id: int
    camera_index: int
    image_name: str
    pixel_count: int
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    quality: float
    median_depth: float
    view_ray: tuple[float, float, float]
    camera_center: tuple[float, float, float]
    independent: bool = True


@dataclass(frozen=True)
class MaskHypothesis:
    hypothesis_id: str
    candidate_id: int
    round_index: int
    camera_index: int
    image_name: str
    crop_kind: str
    detected_class: str
    detection_score: float
    sam_score: float
    box_xyxy: tuple[float, float, float, float]
    seed_coverage: float
    seed_occupancy: float
    mask_area: int
    packed_mask: np.ndarray
    mask_shape: tuple[int, int]
    stable_ordinal: int

    def unpack_mask(self) -> np.ndarray:
        count = int(self.mask_shape[0] * self.mask_shape[1])
        values = np.unpackbits(np.asarray(self.packed_mask, dtype=np.uint8))[:count]
        return values.reshape(self.mask_shape).astype(bool, copy=False)


@dataclass(frozen=True)
class GaussianEvidence:
    candidate_id: int
    point_ids: np.ndarray
    hard_positive_views: np.ndarray
    hard_negative_views: np.ndarray
    alpha_soft_support: np.ndarray
    independent_positive_views: int
    independent_negative_views: int
    selected_hypothesis_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = readonly_int(self.point_ids)
        hp = readonly_float(self.hard_positive_views)
        hn = readonly_float(self.hard_negative_views)
        soft = readonly_float(self.alpha_soft_support)
        if not (len(ids) == len(hp) == len(hn) == len(soft)):
            raise ValueError("Gaussian evidence arrays must share a point axis")
        if np.any(hp < 0) or np.any(hn < 0) or np.any(soft < 0):
            raise ValueError("Gaussian evidence must be non-negative")
        object.__setattr__(self, "point_ids", ids)
        object.__setattr__(self, "hard_positive_views", hp)
        object.__setattr__(self, "hard_negative_views", hn)
        object.__setattr__(self, "alpha_soft_support", soft)


@dataclass(frozen=True)
class ObjectState:
    object_id: int
    parent_candidate_ids: tuple[int, ...]
    point_ids: np.ndarray
    anchor_ids: np.ndarray
    hard_positive_ids: np.ndarray
    hard_positive_counts: np.ndarray
    evidence_margin: np.ndarray
    review_class: str | None
    reliable_review_class: bool
    round_index: int
    changed: bool

    def __post_init__(self) -> None:
        points = readonly_int(self.point_ids)
        anchors = readonly_int(self.anchor_ids)
        hard = readonly_int(self.hard_positive_ids)
        hard_counts = readonly_float(self.hard_positive_counts)
        margin = readonly_float(self.evidence_margin)
        if not (len(points) == len(margin) == len(hard_counts)):
            raise ValueError("object points, hard counts, and margins must share an axis")
        if not set(hard.tolist()).issubset(set(points.tolist())) or not set(anchors.tolist()).issubset(set(points.tolist())):
            raise ValueError("anchors and hard positives must belong to the object")
        object.__setattr__(self, "point_ids", points)
        object.__setattr__(self, "anchor_ids", anchors)
        object.__setattr__(self, "hard_positive_ids", hard)
        object.__setattr__(self, "hard_positive_counts", hard_counts)
        object.__setattr__(self, "evidence_margin", margin)


@dataclass(frozen=True)
class LineageRecord:
    node_id: str
    parent_node_ids: tuple[str, ...]
    candidate_ids: tuple[int, ...]
    affected_b0_ids: tuple[int, ...]
    round_index: int
    operation: str
    added_point_ids: tuple[int, ...]
    removed_point_ids: tuple[int, ...]
    hypothesis_ids: tuple[str, ...]
    export_id: int | None = None

    def __post_init__(self) -> None:
        if self.operation not in {"seed", "refine", "merge", "split", "delete", "export"}:
            raise ValueError(f"unsupported lineage operation: {self.operation}")
