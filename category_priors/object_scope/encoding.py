"""Shared RGB/region geometry for the frozen 336px region classifier.

This module has no model imports. A crop is an effective source rectangle;
letterboxing happens once, identically for RGB, alpha and valid pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image, __version__ as pillow_version

from .artifacts import array_digest, digest, readonly, file_digest

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
RGB_PADDING = (123, 117, 104)
ENCODING_VERSION = "scope-region-letterbox336-v1"


def _array(value, dtype=None):
    return readonly(value, dtype)


def _boolean(value, shape):
    result = np.asarray(value)
    if result.dtype != np.bool_ or result.shape != tuple(shape):
        raise ValueError("require a boolean mask in the original RGB shape")
    return result


def _box(mask):
    ys, xs = np.nonzero(mask)
    return None if not len(xs) else [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


@dataclass(frozen=True)
class CropEncodingPlan:
    image_shape: tuple[int, int]
    box_xyxy: tuple[int, int, int, int]
    kind: str
    requested_side: int | None
    provenance: dict

    def __post_init__(self):
        shape = tuple(self.image_shape)
        box = tuple(self.box_xyxy)
        if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("invalid RGB shape")
        if len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError("crop uses integer half-open pixel edges")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
            raise ValueError("crop must be a nonempty effective RGB rectangle")
        if self.kind not in {"detail", "context", "full"}:
            raise ValueError("unknown encoding branch")
        if self.requested_side is not None and (type(self.requested_side) is not int or self.requested_side < 1):
            raise ValueError("invalid requested crop size")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "box_xyxy", box)


def _rectangle(image_shape, bbox, side):
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    crop_w, crop_h = min(side, width), min(side, height)
    left = min(max(math.floor(center_x - crop_w / 2), 0), width - crop_w)
    top = min(max(math.floor(center_y - crop_h / 2), 0), height - crop_h)
    return (left, top, left + crop_w, top + crop_h)


def make_crop_pair(*, image_shape, bbox_xyxy, fx, fy, optical_depths,
                   global_d50, class_d50=None, condition="U", crop_class=None,
                   statistics_active=False):
    """No GT/category filtering: condition affects context D50 only.

    optical_depths are the CURRENT region's reliable contributor optical depths,
    supplied by the independent measurement adapter, never an old support proxy.
    """
    shape = tuple(image_shape)
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("invalid image shape")
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("invalid current-region bbox")
    if not (0 <= bbox[0] < bbox[2] <= shape[1] and 0 <= bbox[1] < bbox[3] <= shape[0]):
        raise ValueError("bbox must describe visible pixels of the current region")
    condition_group = {"U": "U", "D": "D", "U1": "U", "U2-open": "U",
                       "U2-feedback": "U", "D2-feedback": "D"}.get(condition)
    if condition_group is None or type(statistics_active) is not bool:
        raise ValueError("freeze U/D condition and statistics activity")
    if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in (fx, fy, global_d50)):
        raise ValueError("invalid focal length or global statistic")
    depths = np.asarray(optical_depths, dtype=np.float64)
    if depths.ndim != 1 or not np.isfinite(depths).all():
        raise ValueError("depth measurement must be a finite vector")
    positive = depths[depths > 0]
    z = float(np.median(positive)) if len(positive) else None
    longest = float(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    detail_side = max(1, math.ceil(1.5 * longest))
    detail = CropEncodingPlan(shape, _rectangle(shape, bbox, detail_side), "detail", detail_side,
                              {"bbox_xyxy": bbox.tolist(), "rule": "1.5-current-bbox", "version": ENCODING_VERSION})
    d50 = float(global_d50)
    fallbacks = []
    prior_source = "global"
    if condition_group == "D":
        if not crop_class:
            fallbacks.append("crop_class_unknown")
        elif not statistics_active:
            fallbacks.append("statistics_inactive")
        else:
            if class_d50 is None or isinstance(class_d50, bool) or not np.isfinite(class_d50) or class_d50 <= 0:
                raise ValueError("active class statistic is missing or invalid")
            d50, prior_source = float(class_d50), "class"
    if z is None:
        context_side = detail_side
        context_requested = detail_side
        prior_source = "global_fallback"
        fallbacks.extend(["global_fallback", "depth_unavailable"])
    else:
        requested = math.ceil(1.5 * max(longest, math.sqrt(fx * fy) * d50 / z))
        context_requested = requested
        context_side = min(max(requested, 64), max(shape))
    context = CropEncodingPlan(shape, _rectangle(shape, bbox, context_side), "context", context_requested,
        {"bbox_xyxy": bbox.tolist(), "condition": condition, "crop_class": crop_class,
         "statistics_active": statistics_active, "d50": d50, "prior_source": prior_source,
         "d50_applied": z is not None, "d50_used": d50 if z is not None else None,
         "positive_optical_depth_median": z, "depths_sha256": array_digest(depths),
         "fallback_reasons": fallbacks, "clamped_side": context_side,
         "fx": float(fx), "fy": float(fy), "version": ENCODING_VERSION})
    return detail, context


def full_image_plan(image_shape):
    height, width = image_shape
    return CropEncodingPlan(tuple(image_shape), (0, 0, width, height), "full", None,
                            {"rule": "effective-full-RGB", "version": ENCODING_VERSION})


@dataclass(frozen=True)
class EncodedRegion:
    rgb_tensor: np.ndarray
    alpha_tensor: np.ndarray
    rgb_uint8: np.ndarray
    target_mask: np.ndarray
    valid_mask: np.ndarray
    trace: dict

    @property
    def valid(self):
        return self.trace["status"] == "complete"


def encode_region(image_rgb, mask, valid, plan: CropEncodingPlan, *, alpha_mode="object"):
    rgb = np.asarray(image_rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*plan.image_shape, 3):
        raise ValueError("require original uint8 RGB identity")
    mask = _boolean(mask, plan.image_shape)
    valid = _boolean(valid, plan.image_shape)
    if alpha_mode not in {"object", "valid"}:
        raise ValueError("unknown diagnostic alpha condition")
    x0, y0, x1, y1 = plan.box_xyxy
    region_mask, region_valid = mask[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    height, width = region_mask.shape
    scale = 336 / max(height, width)
    # Fixed round-half-up dimensions, extra odd padding goes right/bottom.
    resize_w = max(1, min(336, math.floor(width * scale + .5)))
    resize_h = max(1, min(336, math.floor(height * scale + .5)))
    left, top = (336 - resize_w) // 2, (336 - resize_h) // 2
    ys, xs = slice(top, top + resize_h), slice(left, left + resize_w)
    encoded_rgb = np.empty((336, 336, 3), dtype=np.uint8)
    encoded_rgb[:] = RGB_PADDING
    encoded_rgb[ys, xs] = np.asarray(Image.fromarray(rgb[y0:y1, x0:x1]).resize(
        (resize_w, resize_h), resample=Image.Resampling.BICUBIC))
    target = np.zeros((336, 336), dtype=bool)
    effective = np.zeros_like(target)
    target[ys, xs] = np.asarray(Image.fromarray(region_mask).resize((resize_w, resize_h), Image.Resampling.NEAREST))
    effective[ys, xs] = np.asarray(Image.fromarray(region_valid).resize((resize_w, resize_h), Image.Resampling.NEAREST))
    target &= effective
    alpha = target if alpha_mode == "object" else effective
    rgb_tensor = ((encoded_rgb.astype(np.float32) / np.float32(255) - np.array(CLIP_MEAN, np.float32))
                  / np.array(CLIP_STD, np.float32)).transpose(2, 0, 1)
    alpha_tensor = ((alpha.astype(np.float32) - np.float32(.5)) / np.float32(.26))[None]
    original_count = int(mask.sum())
    outside_crop = original_count - int(region_mask.sum())
    invalid_original = int((mask & ~valid).sum())
    reasons = []
    if not original_count:
        reasons.append("empty_original_region")
    if outside_crop:
        reasons.append("foreground_truncated_by_crop")
    if invalid_original:
        reasons.append("foreground_outside_valid_RGB")
    if not target.any():
        reasons.append("empty_encoded_region")
    trace = {"schema": ENCODING_VERSION, "encoding_source_sha256": file_digest(__file__),
        "numpy_version": np.__version__, "pillow_version": pillow_version,
        "plan": {"image_shape": list(plan.image_shape),
             "box_xyxy": list(plan.box_xyxy), "kind": plan.kind, "requested_side": plan.requested_side,
             "provenance": plan.provenance}, "alpha_mode": alpha_mode,
        "original_rgb_sha256": array_digest(rgb), "original_mask_sha256": array_digest(mask),
        "original_valid_sha256": array_digest(valid), "original_target_count": original_count,
        "original_target_bbox": _box(mask), "foreground_pixels_lost_to_crop": outside_crop,
        "foreground_invalid_pixels": invalid_original, "resize_wh": [resize_w, resize_h],
        "scale_xy": [resize_w / width, resize_h / height],
        "padding_ltrb": [left, top, 336 - left - resize_w, 336 - top - resize_h],
        "encoded_target_count": int(target.sum()), "encoded_target_bbox": _box(target),
        "encoded_valid_count": int(effective.sum()), "rgb_padding_uint8": list(RGB_PADDING),
        "rgb_interpolation": "Pillow-bicubic", "mask_interpolation": "Pillow-nearest",
        "rgb_mean": list(CLIP_MEAN), "rgb_std": list(CLIP_STD), "alpha_mean": .5, "alpha_std": .26,
        "rgb_uint8_sha256": array_digest(encoded_rgb), "target_sha256": array_digest(target),
        "valid_sha256": array_digest(effective), "rgb_tensor_sha256": array_digest(rgb_tensor),
        "alpha_tensor_sha256": array_digest(alpha_tensor), "tensor_dtype": "float32",
        "status": "unknown" if reasons else "complete", "unknown_reasons": reasons}
    trace["encoding_sha256"] = digest(trace)
    return EncodedRegion(*[_array(a) for a in (rgb_tensor, alpha_tensor, encoded_rgb, target, effective)], trace)


def validate_encoding(value: EncodedRegion):
    trace = dict(value.trace)
    stored = trace.pop("encoding_sha256")
    if digest(trace) != stored:
        raise ValueError("encoding trace hash mismatch")
    for name, array in (("rgb_tensor", value.rgb_tensor), ("alpha_tensor", value.alpha_tensor),
                        ("rgb_uint8", value.rgb_uint8), ("target", value.target_mask), ("valid", value.valid_mask)):
        if array_digest(array) != trace[name + "_sha256"]:
            raise ValueError("encoding tensor content mismatch: " + name)
    if (value.rgb_uint8.shape != (336, 336, 3) or value.rgb_uint8.dtype != np.uint8
            or value.target_mask.shape != (336, 336) or value.valid_mask.shape != (336, 336)
            or value.target_mask.dtype != np.bool_ or value.valid_mask.dtype != np.bool_
            or np.any(value.target_mask & ~value.valid_mask)):
        raise ValueError("invalid encoding arrays or target validity")
    expected_rgb = ((value.rgb_uint8.astype(np.float32) / np.float32(255) - np.array(CLIP_MEAN, np.float32))
                    / np.array(CLIP_STD, np.float32)).transpose(2, 0, 1)
    alpha = value.target_mask if trace["alpha_mode"] == "object" else value.valid_mask
    expected_alpha = ((alpha.astype(np.float32) - np.float32(.5)) / np.float32(.26))[None]
    if not np.array_equal(expected_rgb, value.rgb_tensor) or not np.array_equal(expected_alpha, value.alpha_tensor):
        raise ValueError("actual normalized tensors differ from frozen RGB/alpha formula")
    return stored
