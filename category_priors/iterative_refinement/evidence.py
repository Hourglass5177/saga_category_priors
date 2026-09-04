from __future__ import annotations

"""Hybrid max-contributor and normalized all-alpha Gaussian evidence."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import GaussianEvidence, MaskHypothesis, RefinementConfig


@dataclass(frozen=True)
class AlphaMass:
    inside_mass: np.ndarray
    visible_mass: np.ndarray
    valid_pixel_count: int

    def __post_init__(self) -> None:
        inside = np.asarray(self.inside_mass, dtype=np.float64)
        visible = np.asarray(self.visible_mass, dtype=np.float64)
        if inside.ndim != 2 or visible.ndim != 1 or inside.shape[1] != len(visible):
            raise ValueError("alpha mass must be MxN inside and N visible")
        if not np.isfinite(inside).all() or not np.isfinite(visible).all():
            raise ValueError("alpha mass must be finite")
        if np.any(inside < -1e-7) or np.any(visible < -1e-7):
            raise ValueError("alpha mass must be non-negative")
        tolerance = 5e-5 * np.maximum(visible[None, :], 1.0)
        if inside.size and np.any(inside - visible[None, :] > tolerance):
            raise ValueError("inside alpha mass exceeds visible mass")
        inside = np.minimum(np.maximum(inside, 0.0), np.maximum(visible[None, :], 0.0))
        visible = np.maximum(visible, 0.0)
        inside.setflags(write=False)
        visible.setflags(write=False)
        object.__setattr__(self, "inside_mass", inside)
        object.__setattr__(self, "visible_mass", visible)


@dataclass(frozen=True)
class AlphaObjective:
    mask_indices: tuple[int, ...]
    inside_coefficients: np.ndarray
    visible_coefficient: np.ndarray
    valid_pixels: np.ndarray


@dataclass(frozen=True)
class HypothesisGaussianEvidence:
    hypothesis_id: str
    camera_index: int
    positive_ids: np.ndarray
    negative_ids: np.ndarray
    soft_ids: np.ndarray
    soft_ratios: np.ndarray


def iter_three_channel_masks(masks: Any) -> tuple[tuple[tuple[int, ...], np.ndarray], ...]:
    array = np.asarray(masks, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("masks must have shape MxHxW")
    rows = []
    for start in range(0, len(array), 3):
        stop = min(start + 3, len(array))
        target = np.zeros((3, *array.shape[1:]), dtype=np.float32)
        target[: stop - start] = array[start:stop]
        rows.append((tuple(range(start, stop)), target))
    return tuple(rows)


def build_alpha_objective(
    mask_indices: Sequence[int],
    mask_targets: Any,
    opacity: Any,
    *,
    min_opacity: float = 0.05,
) -> AlphaObjective:
    targets = np.asarray(mask_targets, dtype=np.float64)
    alpha = np.asarray(opacity, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[0] != 3 or targets.shape[1:] != alpha.shape:
        raise ValueError("three mask channels and opacity must share image shape")
    valid = np.isfinite(alpha) & (alpha >= float(min_opacity))
    inverse = np.zeros_like(alpha)
    inverse[valid] = 1.0 / alpha[valid]
    coefficients = targets * inverse[None, :, :]
    coefficients[:, ~valid] = 0.0
    return AlphaObjective(tuple(int(value) for value in mask_indices), coefficients, inverse, valid)


def alpha_mass_from_gradients(
    visible_gradient: Any,
    inside_batches: Sequence[tuple[Sequence[int], Any]],
    mask_count: int,
    valid_pixel_count: int,
) -> AlphaMass:
    visible_raw = np.asarray(visible_gradient, dtype=np.float64)
    visible = visible_raw[:, 0] if visible_raw.ndim == 2 else visible_raw
    if visible.ndim != 1:
        raise ValueError("visible gradient must be N or NxC")
    inside = np.zeros((int(mask_count), len(visible)), dtype=np.float64)
    seen = np.zeros(int(mask_count), dtype=bool)
    for indices, gradient in inside_batches:
        values = np.asarray(gradient, dtype=np.float64)
        if values.shape != (len(visible), 3):
            raise ValueError("inside gradients must have shape Nx3")
        for channel, mask_index in enumerate(indices):
            index = int(mask_index)
            if not 0 <= index < mask_count or seen[index]:
                raise ValueError("alpha gradient mask indices are invalid or repeated")
            inside[index] = values[:, channel]
            seen[index] = True
    if mask_count and not seen.all():
        raise ValueError("one or more mask gradients are missing")
    return AlphaMass(inside, visible, int(valid_pixel_count))


def normalized_soft_membership(
    alpha_mass: AlphaMass,
    mask_index: int,
    config: RefinementConfig = RefinementConfig(),
) -> dict[int, float]:
    inside = alpha_mass.inside_mass[int(mask_index)]
    ratio = np.divide(
        inside,
        alpha_mass.visible_mass,
        out=np.zeros_like(inside),
        where=alpha_mass.visible_mass > 0,
    )
    selected = (inside >= config.alpha_inside_mass_min) & (ratio >= config.alpha_inside_ratio_min)
    return {int(index): float(ratio[index]) for index in np.flatnonzero(selected)}


def _box_mask(shape: tuple[int, int], box_xyxy: Sequence[float]) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    left = max(0, min(width, int(np.floor(x0))))
    top = max(0, min(height, int(np.floor(y0))))
    right = max(left, min(width, int(np.ceil(x1))))
    bottom = max(top, min(height, int(np.ceil(y1))))
    result = np.zeros(shape, dtype=bool)
    result[top:bottom, left:right] = True
    return result


def hypothesis_gaussian_evidence(
    hypothesis: MaskHypothesis,
    *,
    contributor_ids: Any,
    max_weights: Any,
    opacity: Any,
    alpha_mass: AlphaMass,
    alpha_mask_index: int,
    point_count: int,
    config: RefinementConfig = RefinementConfig(),
) -> HypothesisGaussianEvidence:
    ids = np.asarray(contributor_ids, dtype=np.int64)
    weights = np.asarray(max_weights, dtype=np.float64)
    alpha = np.asarray(opacity, dtype=np.float64)
    mask = hypothesis.unpack_mask()
    if ids.shape != mask.shape or weights.shape != ids.shape or alpha.shape != ids.shape:
        raise ValueError("hypothesis and rendered images must share HxW")
    valid = (
        (ids >= 0)
        & (ids < int(point_count))
        & np.isfinite(weights)
        & np.isfinite(alpha)
        & (alpha >= config.hard_opacity_min)
    )
    fraction = np.divide(weights, alpha, out=np.zeros_like(weights), where=alpha > 0)
    reliable = valid & (fraction >= config.hard_contributor_fraction_min)
    box = _box_mask(ids.shape, hypothesis.box_xyxy)
    visible = reliable & box
    inside = visible & mask
    outside = visible & ~mask
    visible_counts = np.bincount(ids[visible], minlength=point_count)
    inside_counts = np.bincount(ids[inside], minlength=point_count)
    outside_counts = np.bincount(ids[outside], minlength=point_count)
    inside_ratio = np.divide(
        inside_counts,
        visible_counts,
        out=np.zeros(point_count, dtype=np.float64),
        where=visible_counts > 0,
    )
    positive = np.flatnonzero((inside_counts >= 1) & (inside_ratio >= 0.50)).astype(np.int64)
    negative = np.flatnonzero(
        (visible_counts >= 2)
        & (outside_counts >= 2)
        & (inside_ratio <= config.hard_negative_inside_ratio_max)
    ).astype(np.int64)
    soft = normalized_soft_membership(alpha_mass, alpha_mask_index, config)
    soft_ids = np.asarray(sorted(soft), dtype=np.int64)
    soft_ratios = np.asarray([soft[index] for index in soft_ids], dtype=np.float64)
    for array in (positive, negative, soft_ids, soft_ratios):
        array.setflags(write=False)
    return HypothesisGaussianEvidence(
        hypothesis.hypothesis_id,
        hypothesis.camera_index,
        positive,
        negative,
        soft_ids,
        soft_ratios,
    )


def aggregate_gaussian_evidence(
    candidate_id: int,
    selected_hypotheses: Sequence[MaskHypothesis],
    per_hypothesis: Mapping[str, HypothesisGaussianEvidence],
) -> GaussianEvidence:
    if len({item.camera_index for item in selected_hypotheses}) != len(selected_hypotheses):
        raise ValueError("selected hypotheses must use distinct camera views")
    point_set: set[int] = set()
    rows = []
    for hypothesis in selected_hypotheses:
        row = per_hypothesis[hypothesis.hypothesis_id]
        rows.append(row)
        point_set.update(int(value) for value in row.positive_ids)
        point_set.update(int(value) for value in row.negative_ids)
        point_set.update(int(value) for value in row.soft_ids)
    point_ids = np.asarray(sorted(point_set), dtype=np.int64)
    position = {int(point_id): index for index, point_id in enumerate(point_ids)}
    positive = np.zeros(len(point_ids), dtype=np.float64)
    negative = np.zeros(len(point_ids), dtype=np.float64)
    soft = np.zeros(len(point_ids), dtype=np.float64)
    positive_view_count = 0
    negative_view_count = 0
    for row in rows:
        if len(row.positive_ids):
            positive_view_count += 1
        if len(row.negative_ids):
            negative_view_count += 1
        for point_id in row.positive_ids:
            positive[position[int(point_id)]] += 1.0
        for point_id in row.negative_ids:
            negative[position[int(point_id)]] += 1.0
        for point_id, ratio in zip(row.soft_ids, row.soft_ratios):
            soft[position[int(point_id)]] += float(ratio)
    return GaussianEvidence(
        candidate_id=int(candidate_id),
        point_ids=point_ids,
        hard_positive_views=positive,
        hard_negative_views=negative,
        alpha_soft_support=soft,
        independent_positive_views=positive_view_count,
        independent_negative_views=negative_view_count,
        selected_hypothesis_ids=tuple(item.hypothesis_id for item in selected_hypotheses),
    )


__all__ = [
    "AlphaMass",
    "AlphaObjective",
    "HypothesisGaussianEvidence",
    "aggregate_gaussian_evidence",
    "alpha_mass_from_gradients",
    "build_alpha_objective",
    "hypothesis_gaussian_evidence",
    "iter_three_channel_masks",
    "normalized_soft_membership",
]
