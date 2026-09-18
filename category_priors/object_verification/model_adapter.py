"""Injected RGB model adapters for object verification v1.

The caller owns model loading and GPU leases. This module imports neither torch
nor the old reviewer. DINO must return raw query logits and caption-token offsets;
the legacy phrase/first-substring result is deliberately not an accepted input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import digest
from .observation import CameraView, SemanticObservation, _boolean, _readonly


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    header = f"{array.dtype.str}:{array.shape}:".encode("ascii")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


@dataclass(frozen=True)
class CropTransform:
    image_shape: tuple[int, int]
    left: int
    top: int
    width: int
    height: int
    requested_side: float | None = None

    def __post_init__(self):
        shape = tuple(int(v) for v in self.image_shape)
        if len(shape) != 2 or min(shape) <= 0 or min(self.width, self.height) <= 0:
            raise ValueError("crop and RGB dimensions must be positive")
        if any(int(v) != v for v in (self.left, self.top, self.width, self.height)):
            raise ValueError("crop pixel coordinates must be integers")
        if self.requested_side is not None and (not np.isfinite(self.requested_side) or self.requested_side <= 0):
            raise ValueError("invalid requested crop size")
        object.__setattr__(self, "image_shape", shape)
        for field in ("left", "top", "width", "height"):
            object.__setattr__(self, field, int(getattr(self, field)))

    def image_to_crop_points(self, points: Any, *, encoded_shape=None) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 2) - [self.left, self.top]
        if encoded_shape is not None:
            height, width = encoded_shape
            if min(height, width) <= 0:
                raise ValueError("encoded geometry must be positive")
            value *= [width / self.width, height / self.height]
        return value

    def crop_to_image_points(self, points: Any, *, encoded_shape=None) -> np.ndarray:
        value = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
        if encoded_shape is not None:
            height, width = encoded_shape
            if min(height, width) <= 0:
                raise ValueError("encoded geometry must be positive")
            value *= [self.width / width, self.height / height]
        return value + [self.left, self.top]

    def image_to_crop_box(self, box: Sequence[float], *, clip=False) -> tuple[float, ...]:
        value = self.image_to_crop_points(np.asarray(box).reshape(2, 2))
        if clip:
            value[:, 0] = np.clip(value[:, 0], 0, self.width)
            value[:, 1] = np.clip(value[:, 1], 0, self.height)
        return tuple(float(v) for v in value.reshape(-1))

    def crop_to_image_box(self, box: Sequence[float]) -> tuple[float, ...]:
        return tuple(float(v) for v in self.crop_to_image_points(np.asarray(box).reshape(2, 2)).reshape(-1))

    def _slices(self):
        x0, y0 = max(0, self.left), max(0, self.top)
        x1, y1 = min(self.image_shape[1], self.left + self.width), min(self.image_shape[0], self.top + self.height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (slice(y0, y1), slice(x0, x1)), (slice(y0 - self.top, y1 - self.top), slice(x0 - self.left, x1 - self.left))

    def extract(self, image_rgb: Any) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8 or image.shape != self.image_shape + (3,):
            raise ValueError("actual RGB must be uint8 and match frozen crop geometry")
        crop = np.full((self.height, self.width, 3), 127, dtype=np.uint8)
        valid = np.zeros((self.height, self.width), dtype=bool)
        slices = self._slices()
        if slices:
            source, target = slices
            crop[target] = image[source]
            valid[target] = True
        return crop, valid

    def mask_to_crop(self, mask: Any) -> np.ndarray:
        source = _boolean(mask, self.image_shape)
        result = np.zeros((self.height, self.width), dtype=bool)
        slices = self._slices()
        if slices:
            result[slices[1]] = source[slices[0]]
        return result

    def mask_to_image(self, mask: Any) -> np.ndarray:
        source = _boolean(mask, (self.height, self.width))
        result = np.zeros(self.image_shape, dtype=bool)
        slices = self._slices()
        if slices:
            result[slices[0]] = source[slices[1]]
        return result


def prior_crop(*, image_shape, bbox_xyxy, focal_geometric_mean: float,
               prior_diagonal_m: float, positive_optical_z: float) -> CropTransform:
    box = np.asarray(bbox_xyxy, dtype=np.float64)
    if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
            or any(not np.isfinite(v) or v <= 0 for v in
                   (focal_geometric_mean, prior_diagonal_m, positive_optical_z))):
        raise ValueError("crop requires a finite box and positive focal/diagonal/optical-z")
    requested = 1.5 * max(float(max(box[2:] - box[:2])), focal_geometric_mean * prior_diagonal_m / positive_optical_z)
    side = min(max(math.ceil(requested), 64), max(image_shape))
    center = np.floor((box[:2] + box[2:]) / 2 + .5).astype(int)
    return CropTransform(tuple(image_shape), int(center[0]) - side // 2,
                         int(center[1]) - side // 2, side, side, requested)


def point_prior_crop(*, image_shape, point_xy, focal_geometric_mean: float,
                     prior_diagonal_m: float, positive_optical_z: float) -> CropTransform:
    """A metric-scale observation centered on a measured point, independent of the old box."""
    point = np.asarray(point_xy, dtype=np.float64)
    values = (focal_geometric_mean, prior_diagonal_m, positive_optical_z)
    if point.shape != (2,) or not np.isfinite(point).all() or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError("point crop needs a finite point and positive focal/size/optical depth")
    requested = 1.5 * focal_geometric_mean * prior_diagonal_m / positive_optical_z
    side = min(max(math.ceil(requested), 64), max(image_shape))
    center = np.floor(point + .5).astype(int)
    return CropTransform(tuple(image_shape), int(center[0]) - side // 2,
                         int(center[1]) - side // 2, side, side, requested)


@dataclass(frozen=True)
class FrozenLocator:
    bbox_xyxy: tuple[float, float, float, float]
    anchor_ids: tuple[int, ...]
    support_ids: tuple[int, ...]
    origin: str
    source_sha256: str

    def __post_init__(self):
        box = np.asarray(self.bbox_xyxy, dtype=np.float64)
        if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
                or self.origin not in {"frozen_source", "verified_direct"} or not self.source_sha256):
            raise ValueError("locator must come from frozen source or verified direct evidence")
        anchor, support = tuple(sorted(set(self.anchor_ids))), tuple(sorted(set(self.support_ids)))
        if any(int(v) != v or v < 0 for v in anchor + support) or not set(anchor) <= set(support):
            raise ValueError("locator anchors must be nonnegative source/verified support IDs")
        object.__setattr__(self, "anchor_ids", anchor)
        object.__setattr__(self, "support_ids", support)
        object.__setattr__(self, "bbox_xyxy", tuple(float(v) for v in box))


@dataclass(frozen=True)
class UnavailableLocator:
    """A measured absence of usable geometry, not a synthetic full-image box.

    Missing/corrupt assets must raise at their reader and cannot be represented
    here. Source and receipt membership survive an unobservable camera.
    """
    anchor_ids: tuple[int, ...]
    support_ids: tuple[int, ...]
    origin: str
    source_sha256: str
    reason: str

    def __post_init__(self):
        if (self.origin not in {"frozen_source", "verified_direct"} or not self.source_sha256
                or self.reason not in {"no_visible_seed_projection", "no_positive_optical_depth"}):
            raise ValueError("unavailable locator requires measured geometry reason and actual source identity")
        values = tuple(self.anchor_ids) + tuple(self.support_ids)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0 for v in values):
            raise ValueError("unavailable locator members must remain nonnegative integer IDs")
        anchors, support = tuple(sorted(set(self.anchor_ids))), tuple(sorted(set(self.support_ids)))
        if not set(anchors) <= set(support):
            raise ValueError("unavailable locator anchors must remain within source support")
        object.__setattr__(self, "anchor_ids", anchors)
        object.__setattr__(self, "support_ids", support)


def reliable_prompt_point(*, locator: FrozenLocator, contributor_ids: Any,
                          max_contribution: Any, opacity: Any, valid_pixels: Any) -> dict[str, Any] | None:
    valid = _boolean(valid_pixels)
    ids = np.asarray(contributor_ids)
    weight, alpha = np.asarray(max_contribution, dtype=np.float64), np.asarray(opacity, dtype=np.float64)
    if ids.shape != valid.shape or ids.dtype.kind not in "iu" or weight.shape != valid.shape or alpha.shape != valid.shape:
        raise ValueError("prompt contributor arrays must share registered RGB geometry")
    if not np.isfinite(weight).all() or not np.isfinite(alpha).all() or np.any(weight < 0) or np.any(alpha < 0):
        raise ValueError("invalid contributor measurement")
    ratio = np.divide(weight, alpha, out=np.zeros_like(weight), where=alpha > 0)
    yy, xx = np.indices(valid.shape)
    x0, y0, x1, y1 = locator.bbox_xyxy
    inside_box = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    reliable = valid & inside_box & (ids >= 0) & (alpha >= .50) & (ratio >= .50)
    for kind, selected in (("anchor", locator.anchor_ids), ("support", locator.support_ids)):
        pixels = np.flatnonzero(reliable & np.isin(ids, selected))
        if len(pixels):
            ordered = sorted((int(v) for v in pixels),
                             key=lambda v: (-float(ratio.flat[v]), -float(weight.flat[v]), v))
            index = ordered[0]
            y, x = np.unravel_index(index, valid.shape)
            return dict(point_image_xy=(float(x), float(y)), pixel_id=index,
                        gaussian_id=int(ids.flat[index]), source=kind,
                        reliability=float(ratio.flat[index]), max_contribution=float(weight.flat[index]))
    return None


def class_caption(classes32: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    classes = tuple(str(value) for value in classes32)
    if len(classes) != 32 or len({v.lower().strip() for v in classes}) != 32 or any(not v.strip() for v in classes):
        raise ValueError("the DINO caption requires the frozen 32 unique class names")
    caption, spans = "", []
    for name in classes:
        name = name.lower().strip()
        start = len(caption)
        caption += name
        spans.append((start, len(caption)))
        caption += ". "
    return caption.strip(), tuple(spans)


def class_token_spans(classes32: Sequence[str], offset_mapping: Any,
                      special_tokens_mask: Any) -> tuple[tuple[int, ...], ...]:
    """Use exact character offsets, not phrase substring or token string search.

    Every alphanumeric class character must be covered. Truncated captions fail
    closed instead of quietly removing late classes from competition.
    """
    caption, spans = class_caption(classes32)
    offsets = np.asarray(offset_mapping)
    specials = np.asarray(special_tokens_mask, dtype=bool)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or offsets.dtype.kind not in "iu" or specials.shape != (len(offsets),):
        raise ValueError("raw tokenizer offsets and special mask are required")
    if np.any(offsets < 0) or np.any(offsets[:, 1] < offsets[:, 0]) or np.any(offsets > len(caption)):
        raise ValueError("invalid caption token offsets")
    output, used = [], set()
    for start, end in spans:
        needed = {i for i in range(start, end) if caption[i].isalnum()}
        covered, tokens = set(), []
        for token, ((left, right), special) in enumerate(zip(offsets, specials)):
            if special or left == right:
                continue
            content = {i for i in range(int(left), int(right)) if caption[i].isalnum()}
            if content & needed:
                if not content <= needed or token in used:
                    raise ValueError("token crosses class boundaries")
                tokens.append(token)
                covered |= content
                used.add(token)
        if not tokens or covered != needed:
            raise ValueError("truncated or incomplete 32-class token spans")
        output.append(tuple(tokens))
    return tuple(output)


def decode_dino_queries(*, classes32: Sequence[str], raw_logits: Any, boxes_crop_xyxy: Any,
                        offset_mapping: Any, special_tokens_mask: Any) -> dict[str, Any]:
    logits = np.asarray(raw_logits, dtype=np.float64)
    boxes = np.asarray(boxes_crop_xyxy, dtype=np.float64)
    tokens = class_token_spans(classes32, offset_mapping, special_tokens_mask)
    if (logits.ndim != 2 or logits.shape[1] != len(offset_mapping) or boxes.shape != (len(logits), 4)
            or not np.isfinite(logits).all() or not np.isfinite(boxes).all()):
        raise ValueError("DINO must supply finite raw query logits and absolute crop boxes")
    probabilities = np.exp(-np.logaddexp(0., -logits))
    proposals = []
    for index, box in enumerate(boxes):
        scores = {name: float(probabilities[index, list(span)].max()) for name, span in zip(classes32, tokens)}
        qualified = [name for name in classes32 if scores[name] > .35]
        valid_box = bool(np.all(box[2:] > box[:2]))
        proposals.append(dict(query_id=index, box_crop_xyxy=box.tolist(), class_scores=scores,
                              qualified_classes=qualified, status="eligible" if qualified and valid_box else "rejected",
                              reason="class_token_threshold_pass" if qualified and valid_box else
                              "invalid_box" if not valid_box else "below_box_and_text_threshold"))
    caption, spans = class_caption(classes32)
    return dict(caption=caption, classes32=list(classes32), class_character_spans=spans,
                class_token_spans=tokens, offset_mapping=np.asarray(offset_mapping).tolist(),
                special_tokens_mask=np.asarray(special_tokens_mask, dtype=bool).tolist(),
                raw_logits=logits.tolist(), boxes_crop_xyxy=boxes.tolist(), proposals=proposals,
                box_threshold=.35, text_threshold=.35, class_score_rule="sigmoid_max_over_class_content_token_span")


@dataclass(frozen=True)
class SamMask:
    observation_uid: str
    mask_image: np.ndarray
    mask_crop: np.ndarray
    sam_quality: float
    stable_ordinal: int

    def __post_init__(self):
        if not self.observation_uid or not np.isfinite(self.sam_quality) or self.stable_ordinal not in (0, 1, 2):
            raise ValueError("SAM mask requires a stable raw ordinal and finite quality")
        object.__setattr__(self, "mask_image", _boolean(self.mask_image))
        object.__setattr__(self, "mask_crop", _boolean(self.mask_crop))


@dataclass(frozen=True)
class GeometryResult:
    status: str
    reason: str
    masks: tuple[SamMask, ...]
    trace: Mapping[str, Any]


class InjectedModelAdapter:
    """SAM is a predictor with set_image/predict; DINO is a raw-query callable.

    dino(image_uint8, caption) returns raw_logits, boxes_crop_xyxy (absolute,
    not normalized), offset_mapping and special_tokens_mask. Returning old
    confidence/class_ids is an interface error, not an empty scientific result.
    """
    def __init__(self, *, sam_predictor, dino_raw=None):
        self.sam = sam_predictor
        self.dino_raw = dino_raw

    def _sam_masks(self, *, image, crop, box_crop, point_crop, uid):
        self.sam.set_image(image)
        kwargs = dict(box=None if box_crop is None else np.asarray(box_crop, dtype=np.float32),
                      multimask_output=True)
        if point_crop is not None:
            points = np.asarray(point_crop, dtype=np.float32).reshape(-1, 2)
            if not len(points) or not np.isfinite(points).all():
                raise ValueError("SAM positive points must be finite and nonempty")
            kwargs.update(point_coords=points,
                          point_labels=np.ones(len(points), dtype=np.int64))
        else:
            kwargs.update(point_coords=None, point_labels=None)
        masks, quality, _ = self.sam.predict(**kwargs)
        masks, quality = np.asarray(masks), np.asarray(quality, dtype=np.float64)
        if masks.dtype != np.bool_ or masks.shape != (3, crop.height, crop.width) or quality.shape != (3,) or not np.isfinite(quality).all():
            raise ValueError("SAM must return all three masks in the actual encoded crop geometry")
        return tuple(SamMask(f"{uid}:sam:{i}", crop.mask_to_image(mask), mask, float(score), i)
                     for i, (mask, score) in enumerate(zip(masks, quality)))

    def geometry(self, *, image_rgb, crop: CropTransform, locator: FrozenLocator,
                 contributor_ids, max_contribution, opacity, observation_uid: str,
                 role: str = "construction", valid_pixels=None) -> GeometryResult:
        if role not in {"construction", "prepass", "online_verification", "diagnostic"}:
            raise ValueError("manual/GT observations cannot enter the model adapter")
        encoded, crop_valid = crop.extract(image_rgb)
        valid = crop.mask_to_image(crop_valid)
        if valid_pixels is not None:
            valid &= _boolean(valid_pixels, crop.image_shape)
        point = reliable_prompt_point(locator=locator, contributor_ids=contributor_ids,
                                      max_contribution=max_contribution, opacity=opacity, valid_pixels=valid)
        box = crop.image_to_crop_box(locator.bbox_xyxy)
        clipped = crop.image_to_crop_box(locator.bbox_xyxy, clip=True)
        trace = dict(observation_uid=observation_uid, role=role, encoding="actual_crop", crop=crop.__dict__,
                     image_sha256=array_sha256(image_rgb), encoded_rgb_sha256=array_sha256(encoded),
                     encoded_shape=list(encoded.shape), valid_pixels_sha256=array_sha256(valid),
                     padding_pixels=int((~crop_valid).sum()), locator=locator.__dict__, point=point,
                     box_crop_xyxy=clipped, box_before_clip=box, bbox_truncated=box != clipped,
                     prompt_type="box_plus_one_positive", negative_point_count=0,
                     dino_gating=False, sam_multimask_count=3, calls=dict(sam=0, dino=0))
        if point is None or clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return GeometryResult("unknown", "no_reliable_prompt" if point is None else "crop_insufficient", (), trace)
        point_crop = crop.image_to_crop_points([point["point_image_xy"]])
        masks = self._sam_masks(image=encoded, crop=crop, box_crop=clipped, point_crop=point_crop, uid=observation_uid)
        trace["calls"]["sam"] = 1
        trace["point_crop_xy"] = point_crop[0].tolist()
        trace["masks"] = [dict(observation_uid=m.observation_uid, stable_ordinal=m.stable_ordinal,
                               sam_quality=m.sam_quality, mask_sha256=array_sha256(m.mask_image),
                               crop_mask_sha256=array_sha256(m.mask_crop), pixels=int(m.mask_image.sum())) for m in masks]
        return GeometryResult("unknown" if box != clipped else "observed",
                              "crop_insufficient" if box != clipped else "three_raw_masks_retained", masks, trace)

    def independent_verification_masks(self, *, image_rgb, crop: CropTransform, locator: FrozenLocator,
                                       contributor_ids, max_contribution, opacity, observation_uid: str,
                                       valid_pixels=None) -> GeometryResult:
        """No pending proposal/mask/member input exists on this API.

        The caller must bind locator provenance to a frozen source or already
        verified identity, and run proposal P/R evaluation only after this call.
        """
        return self.geometry(image_rgb=image_rgb, crop=crop, locator=locator,
                             contributor_ids=contributor_ids, max_contribution=max_contribution,
                             opacity=opacity, observation_uid=observation_uid,
                             role="online_verification", valid_pixels=valid_pixels)

    def semantic(self, *, image_rgb, crop: CropTransform, object_mask, classes32: Sequence[str],
                 camera: CameraView, observation_uid: str, role="construction", valid_pixels=None):
        if self.dino_raw is None:
            raise RuntimeError("raw-token DINO adapter is required for semantic observation")
        if role not in {"construction", "prepass", "online_verification", "diagnostic"}:
            raise ValueError("GT/human semantics cannot enter a model input")
        encoded, crop_valid = crop.extract(image_rgb)
        valid = crop.mask_to_image(crop_valid)
        if valid_pixels is not None:
            valid &= _boolean(valid_pixels, crop.image_shape)
        target = _boolean(object_mask, crop.image_shape)
        caption, _ = class_caption(classes32)
        raw = self.dino_raw(encoded.copy(), caption)
        required = {"raw_logits", "boxes_crop_xyxy", "offset_mapping", "special_tokens_mask"}
        if not isinstance(raw, Mapping) or not required <= raw.keys():
            raise ValueError("DINO raw logits/offsets are missing; legacy phrase scores are not supported")
        decoded = decode_dino_queries(classes32=classes32, **{key: raw[key] for key in required})
        trace = dict(observation_uid=observation_uid, role=role, encoding="actual_crop", crop=crop.__dict__,
                     image_sha256=array_sha256(image_rgb), encoded_rgb_sha256=array_sha256(encoded),
                     object_mask_sha256=array_sha256(target), valid_pixels_sha256=array_sha256(valid),
                     dino=decoded, calls=dict(dino=1, sam=0), masks=[])
        if "runtime_record" in raw:
            # Includes exact model input token IDs, native padding and resize
            # provenance. These are measurement records, never semantic scores.
            trace["dino"]["runtime_record"] = raw["runtime_record"]
        matched = {}
        all_masks = []
        incomplete = bool(np.any(target & ~valid))
        for proposal in decoded["proposals"]:
            proposal["mask_results"] = []
            if proposal["status"] != "eligible":
                continue
            box = np.asarray(proposal["box_crop_xyxy"], dtype=np.float64).reshape(2, 2)
            box[:, 0] = np.clip(box[:, 0], 0, crop.width)
            box[:, 1] = np.clip(box[:, 1], 0, crop.height)
            clipped = box.reshape(-1)
            proposal["actual_box_crop_xyxy"] = clipped.tolist()
            proposal["bbox_truncated"] = not np.array_equal(clipped, proposal["box_crop_xyxy"])
            if np.any(box[1] <= box[0]):
                proposal.update(status="rejected", reason="box_outside_crop")
                continue
            masks = self._sam_masks(image=encoded, crop=crop, box_crop=clipped, point_crop=None,
                                    uid=f"{observation_uid}:query:{proposal['query_id']}")
            trace["calls"]["sam"] += 1
            all_masks.extend(masks)
            for mask in masks:
                intersection = int((mask.mask_image & target & valid).sum())
                union = int(((mask.mask_image | target) & valid).sum())
                iou = intersection / union if union else None
                passed = not incomplete and not proposal["bbox_truncated"] and iou is not None and iou >= .80
                proposal["mask_results"].append(dict(observation_uid=mask.observation_uid,
                    sam_quality=mask.sam_quality, stable_ordinal=mask.stable_ordinal, intersection=intersection,
                    union=union, iou=iou, mask_sha256=array_sha256(mask.mask_image),
                    status="matched" if passed else "unknown" if incomplete or proposal["bbox_truncated"] or not union else "rejected",
                    reason="mask_iou_pass" if passed else "crop_insufficient" if incomplete or proposal["bbox_truncated"] else
                    "empty_valid_union" if not union else "object_mask_iou_below_threshold"))
            if any(r["status"] == "matched" for r in proposal["mask_results"]):
                proposal.update(status="matched", reason="box_only_sam_matches_object")
                for name in proposal["qualified_classes"]:
                    matched[name] = max(matched.get(name, 0.), proposal["class_scores"][name])
            else:
                proposal.update(status="unknown" if incomplete or proposal["bbox_truncated"] else "rejected",
                                reason="crop_insufficient" if incomplete or proposal["bbox_truncated"] else "no_grounded_object_mask")
        reason = None if matched else "crop_insufficient" if incomplete else "no_qualified_semantic_proposal"
        observation = SemanticObservation(observation_uid, camera, role, matched,
                                          observable=bool(np.any(target & valid)), unknown_reason=reason)
        trace["semantic_observation"] = dict(class_scores=matched, observable=observation.observable, unknown_reason=reason)
        trace["payload_sha256"] = digest(trace)
        return observation, tuple(all_masks), trace
