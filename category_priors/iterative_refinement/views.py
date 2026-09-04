from __future__ import annotations

"""Deterministic view selection, crop geometry, and mask-hypothesis consensus."""

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import MaskHypothesis, RefinementConfig, ViewObservation


@dataclass(frozen=True)
class CropSpec:
    kind: str
    side: int
    left: int
    top: int
    requested_side: float
    capped: bool
    padding_fraction: float


def pack_mask(mask: Any) -> tuple[np.ndarray, tuple[int, int]]:
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError("mask must be a two-dimensional image")
    packed = np.packbits(array.reshape(-1)).astype(np.uint8, copy=False)
    packed.setflags(write=False)
    return packed, (int(array.shape[0]), int(array.shape[1]))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else np.zeros_like(vector)


def build_view_observation(
    *,
    candidate_id: int,
    camera_index: int,
    image_name: str,
    candidate_point_ids: Any,
    candidate_xyz: Any,
    contributor_ids: Any,
    contribution_weights: Any,
    opacity: Any,
    camera_center: Any,
    config: RefinementConfig = RefinementConfig(),
) -> ViewObservation | None:
    ids = np.asarray(contributor_ids, dtype=np.int64)
    weights = np.asarray(contribution_weights, dtype=np.float64)
    alpha = np.asarray(opacity, dtype=np.float64)
    support = np.asarray(candidate_point_ids, dtype=np.int64)
    xyz = np.asarray(candidate_xyz, dtype=np.float64)
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    if ids.ndim != 2 or weights.shape != ids.shape or alpha.shape != ids.shape:
        raise ValueError("contributor, weight, and opacity images must share HxW")
    if xyz.shape != (len(support), 3) or not np.isfinite(xyz).all():
        raise ValueError("candidate_xyz must align with candidate_point_ids")
    valid = (
        (ids >= 0)
        & np.isfinite(weights)
        & (weights > 0)
        & np.isfinite(alpha)
        & (alpha >= config.alpha_opacity_min)
        & np.isin(ids, support)
    )
    ys, xs = np.nonzero(valid)
    if len(xs) < config.min_projected_pixels:
        return None
    ratios = np.divide(weights[valid], alpha[valid], out=np.zeros(len(xs)), where=alpha[valid] > 0)
    candidate_center = np.median(xyz, axis=0)
    ray = _unit(candidate_center - center)
    depth = float(np.median(np.linalg.norm(xyz - center[None, :], axis=1)))
    quality = float(math.log1p(len(xs)) * np.clip(np.mean(ratios), 0.0, 1.0))
    return ViewObservation(
        candidate_id=int(candidate_id),
        camera_index=int(camera_index),
        image_name=str(image_name),
        pixel_count=int(len(xs)),
        bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        centroid_xy=(float(xs.mean()), float(ys.mean())),
        quality=quality,
        median_depth=depth,
        view_ray=tuple(float(value) for value in ray),
        camera_center=tuple(float(value) for value in center),
    )


def observations_are_independent(
    first: ViewObservation,
    second: ViewObservation,
    config: RefinementConfig = RefinementConfig(),
) -> bool:
    a = _unit(np.asarray(first.view_ray, dtype=np.float64))
    b = _unit(np.asarray(second.view_ray, dtype=np.float64))
    angle = math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))
    baseline = float(
        np.linalg.norm(
            np.asarray(first.camera_center, dtype=np.float64)
            - np.asarray(second.camera_center, dtype=np.float64)
        )
    )
    depth = max(min(first.median_depth, second.median_depth), 1e-8)
    return angle >= config.min_ray_angle_deg or baseline / depth >= config.min_baseline_depth_ratio


def select_diverse_views(
    observations: Sequence[ViewObservation],
    config: RefinementConfig = RefinementConfig(),
) -> tuple[ViewObservation, ...]:
    if not observations:
        return ()
    ordered = sorted(observations, key=lambda row: (-row.quality, -row.pixel_count, row.image_name))
    selected: list[ViewObservation] = [replace(ordered[0], independent=True)]
    remaining = ordered[1:]
    while remaining and len(selected) < config.max_views:
        scored: list[tuple[int, float, int, str, ViewObservation]] = []
        for row in remaining:
            independent = all(observations_are_independent(row, item, config) for item in selected)
            # Prefer a genuinely new view; quality breaks ties deterministically.
            scored.append((int(independent), row.quality, row.pixel_count, row.image_name, row))
        scored.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        choice = scored[0][4]
        independent = all(observations_are_independent(choice, item, config) for item in selected)
        selected.append(replace(choice, independent=independent))
        remaining = [row for row in remaining if row.camera_index != choice.camera_index]
    return tuple(selected)


def make_crop_spec(
    observation: ViewObservation,
    image_shape: tuple[int, int],
    *,
    kind: str,
    focal_geometric_mean: float,
    prior_diagonal_m: float,
    config: RefinementConfig = RefinementConfig(),
) -> CropSpec:
    height, width = (int(image_shape[0]), int(image_shape[1]))
    x0, y0, x1, y1 = observation.bbox_xyxy
    current_side = float(max(x1 - x0, y1 - y0))
    if kind == "tight":
        requested = 1.5 * current_side
    elif kind == "prior":
        expected = float(focal_geometric_mean) * float(prior_diagonal_m) / max(
            float(observation.median_depth), 1e-4
        )
        requested = 1.5 * max(current_side, expected)
    else:
        raise ValueError("crop kind must be tight or prior")
    requested = max(64.0, requested)
    side = int(math.ceil(min(requested, float(max(height, width)))))
    cx = int(math.floor(observation.centroid_xy[0] + 0.5))
    cy = int(math.floor(observation.centroid_xy[1] + 0.5))
    left = cx - side // 2
    top = cy - side // 2
    source_x0, source_y0 = max(left, 0), max(top, 0)
    source_x1, source_y1 = min(left + side, width), min(top + side, height)
    inside = max(0, source_x1 - source_x0) * max(0, source_y1 - source_y0)
    padding = 1.0 - inside / float(side * side)
    return CropSpec(kind, side, left, top, requested, requested > max(height, width), padding)


def extract_crop(image: Any, mask: Any, spec: CropSpec) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(image, dtype=np.uint8)
    source_mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or source_mask.shape != rgb.shape[:2]:
        raise ValueError("RGB and mask must share HxW")
    crop = np.full((spec.side, spec.side, 3), 127, dtype=np.uint8)
    crop_mask = np.zeros((spec.side, spec.side), dtype=bool)
    sx0, sy0 = max(spec.left, 0), max(spec.top, 0)
    sx1, sy1 = min(spec.left + spec.side, rgb.shape[1]), min(spec.top + spec.side, rgb.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        tx0, ty0 = sx0 - spec.left, sy0 - spec.top
        tx1, ty1 = tx0 + sx1 - sx0, ty0 + sy1 - sy0
        crop[ty0:ty1, tx0:tx1] = rgb[sy0:sy1, sx0:sx1]
        crop_mask[ty0:ty1, tx0:tx1] = source_mask[sy0:sy1, sx0:sx1]
    return crop, crop_mask


def crop_box_to_image(box_xyxy: Sequence[float], spec: CropSpec) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in box_xyxy)
    return x0 + spec.left, y0 + spec.top, x1 + spec.left, y1 + spec.top


def weighted_jaccard(first: Mapping[int, float], second: Mapping[int, float]) -> float:
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    numerator = sum(min(float(first.get(key, 0.0)), float(second.get(key, 0.0))) for key in keys)
    denominator = sum(max(float(first.get(key, 0.0)), float(second.get(key, 0.0))) for key in keys)
    return float(numerator / denominator) if denominator > 0 else 0.0


def select_consistent_hypotheses(
    hypotheses: Sequence[MaskHypothesis],
    soft_membership: Mapping[str, Mapping[int, float]],
    independent_camera_pairs: Mapping[tuple[int, int], bool],
    hypothesis_penalties: Mapping[str, float] | None = None,
    config: RefinementConfig = RefinementConfig(),
) -> tuple[MaskHypothesis, ...]:
    grouped: dict[int, list[MaskHypothesis]] = {}
    for row in hypotheses:
        grouped.setdefault(row.camera_index, []).append(row)
    cameras = sorted(grouped)
    best: tuple[Any, tuple[MaskHypothesis, ...]] | None = None
    for count in range(2, len(cameras) + 1):
        for selected_cameras in itertools.combinations(cameras, count):
            for choice in itertools.product(*(grouped[camera] for camera in selected_cameras)):
                pair_scores: list[float] = []
                valid = True
                for left, right in itertools.combinations(choice, 2):
                    pair = tuple(sorted((left.camera_index, right.camera_index)))
                    if not independent_camera_pairs.get(pair, False):
                        valid = False
                        break
                    pair_scores.append(
                        weighted_jaccard(
                            soft_membership.get(left.hypothesis_id, {}),
                            soft_membership.get(right.hypothesis_id, {}),
                        )
                    )
                if not valid or not pair_scores or min(pair_scores) < config.mask_jaccard_min:
                    continue
                penalties = hypothesis_penalties or {}
                score = (
                    len(choice),
                    float(np.mean(pair_scores)),
                    float(np.mean([item.seed_coverage for item in choice])),
                    float(np.mean([item.sam_score * float(penalties.get(item.hypothesis_id, 1.0)) for item in choice])),
                    float(np.mean([item.detection_score for item in choice])),
                    tuple(-item.stable_ordinal for item in choice),
                )
                stable_choice = tuple(sorted(choice, key=lambda item: (item.camera_index, item.stable_ordinal)))
                if best is None or score > best[0]:
                    best = (score, stable_choice)
    return () if best is None else best[1]


__all__ = [
    "CropSpec",
    "build_view_observation",
    "crop_box_to_image",
    "extract_crop",
    "make_crop_spec",
    "observations_are_independent",
    "pack_mask",
    "select_consistent_hypotheses",
    "select_diverse_views",
    "weighted_jaccard",
]
