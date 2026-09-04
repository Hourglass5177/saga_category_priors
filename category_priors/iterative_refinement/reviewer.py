from __future__ import annotations

"""GroundingDINO/SAM review with one SAM image encoding per camera pass."""

import re
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import MaskHypothesis, RefinementConfig, ViewObservation
from .views import CropSpec, crop_box_to_image, extract_crop, pack_mask


@dataclass(frozen=True)
class DetectionProposal:
    class_name: str
    score: float
    box_crop_xyxy: tuple[float, float, float, float]
    box_image_xyxy: tuple[float, float, float, float]
    seed_coverage: float
    seed_occupancy: float
    stable_ordinal: int


def _rectangle_mask(shape: tuple[int, int], box: Sequence[float]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = (float(value) for value in box)
    left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    right, bottom = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    result = np.zeros(shape, dtype=bool)
    if right > left and bottom > top:
        result[top:bottom, left:right] = True
    return result


def rank_detection_proposals(
    *,
    boxes_xyxy: Any,
    scores: Any,
    class_ids: Any,
    classes: Sequence[str],
    seed_mask: Any,
    crop_spec: CropSpec,
    limit: int = 2,
) -> tuple[DetectionProposal, ...]:
    """Filter detections by seed intersection and rank away oversized boxes."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    confidence = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(class_ids, dtype=object).reshape(-1)
    seed = np.asarray(seed_mask, dtype=bool)
    if not (len(boxes) == len(confidence) == len(labels)) or seed.ndim != 2:
        raise ValueError("detections and seed crop are inconsistent")
    ys, xs = np.nonzero(seed)
    if not len(xs):
        return ()
    centroid = (float(xs.mean()), float(ys.mean()))
    rows: list[tuple[float, int, DetectionProposal]] = []
    for ordinal, (box, score, raw_class) in enumerate(zip(boxes, confidence, labels)):
        if raw_class is None or not np.isfinite(score):
            continue
        class_id = int(raw_class)
        if not 0 <= class_id < len(classes):
            continue
        rectangle = _rectangle_mask(seed.shape, box)
        intersection = int(np.count_nonzero(rectangle & seed))
        if intersection == 0:
            continue
        cx, cy = centroid
        contains_center = box[0] <= cx < box[2] and box[1] <= cy < box[3]
        # A high-contribution seed pixel is represented by the seed crop.  In
        # the absence of a centre hit, intersection is still required.
        if not contains_center and intersection == 0:
            continue
        seed_area = int(seed.sum())
        box_area = int(rectangle.sum())
        coverage = intersection / max(seed_area, 1)
        occupancy = intersection / max(box_area, 1)
        evidence = float(score) * float(np.sqrt(coverage * occupancy))
        proposal = DetectionProposal(
            class_name=str(classes[class_id]),
            score=float(score),
            box_crop_xyxy=tuple(float(value) for value in box),
            box_image_xyxy=crop_box_to_image(box, crop_spec),
            seed_coverage=float(coverage),
            seed_occupancy=float(occupancy),
            stable_ordinal=int(ordinal),
        )
        rows.append((evidence, -ordinal, proposal))
    rows.sort(key=lambda row: (-row[0], -row[1]))
    return tuple(row[2] for row in rows[: int(limit)])


def dispersed_prompt_points(mask: Any, limit: int = 4) -> np.ndarray:
    """Choose stable, spatially spread prompt pixels from a binary mask."""

    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(xs):
        return np.empty((0, 2), dtype=np.float32)
    points = np.column_stack((xs, ys)).astype(np.float64)
    chosen = [int(np.lexsort((points[:, 0], points[:, 1]))[0])]
    while len(chosen) < min(int(limit), len(points)):
        distances = np.min(
            np.sum((points[:, None, :] - points[np.asarray(chosen)][None, :, :]) ** 2, axis=2),
            axis=1,
        )
        distances[np.asarray(chosen)] = -1
        chosen.append(int(np.flatnonzero(distances == distances.max())[0]))
    return points[np.asarray(chosen)].astype(np.float32)


def rank_sam_masks(
    *,
    masks: Any,
    sam_scores: Any,
    proposal: DetectionProposal,
    seed_mask_full: Any,
    negative_mask_full: Any,
    candidate_id: int,
    round_index: int,
    camera_index: int,
    image_name: str,
    crop_kind: str,
    stable_base: int,
    limit: int = 2,
) -> tuple[MaskHypothesis, ...]:
    array = np.asarray(masks, dtype=bool)
    quality = np.asarray(sam_scores, dtype=np.float64).reshape(-1)
    seed = np.asarray(seed_mask_full, dtype=bool)
    negative = np.asarray(negative_mask_full, dtype=bool)
    if array.ndim != 3 or len(array) != len(quality) or array.shape[1:] != seed.shape:
        raise ValueError("SAM masks do not share the full-image geometry")
    seed_area = int(seed.sum())
    rows: list[tuple[tuple[float, ...], MaskHypothesis]] = []
    for ordinal, (mask, score) in enumerate(zip(array, quality)):
        area = int(mask.sum())
        if area == 0 or not np.isfinite(score):
            continue
        seed_coverage = int(np.count_nonzero(mask & seed)) / max(seed_area, 1)
        negative_fraction = int(np.count_nonzero(mask & negative)) / max(int(negative.sum()), 1)
        area_ratio = area / max(seed_area, 1)
        packed, shape = pack_mask(mask)
        stable = int(stable_base + ordinal)
        hypothesis = MaskHypothesis(
            hypothesis_id=f"c{candidate_id}:r{round_index}:v{camera_index}:{crop_kind}:{stable}",
            candidate_id=int(candidate_id),
            round_index=int(round_index),
            camera_index=int(camera_index),
            image_name=str(image_name),
            crop_kind=str(crop_kind),
            detected_class=proposal.class_name,
            detection_score=float(proposal.score),
            sam_score=float(score),
            box_xyxy=proposal.box_image_xyxy,
            seed_coverage=float(seed_coverage),
            seed_occupancy=float(proposal.seed_occupancy),
            mask_area=area,
            packed_mask=packed,
            mask_shape=shape,
            stable_ordinal=stable,
        )
        # Lexicographic order follows the preregistration: seed inclusion,
        # competitor exclusion, SAM quality, detector quality, smaller masks.
        rank = (
            float(seed_coverage),
            float(1.0 - negative_fraction),
            float(score),
            float(proposal.score),
            float(-area_ratio),
            float(-stable),
        )
        rows.append((rank, hypothesis))
    rows.sort(key=lambda row: tuple(-value for value in row[0]))
    return tuple(row[1] for row in rows[: int(limit)])


class GroundedSamFullImageReviewer:
    """One detector/model pair, with a full-image SAM embedding per camera."""

    def __init__(self, *, dino_config: str, dino_checkpoint: str, sam_checkpoint: str) -> None:
        import torch

        if "regex" not in sys.modules:
            try:
                __import__("regex")
            except ImportError:
                sys.modules["regex"] = re
        try:
            __import__("pycocotools")
        except ImportError:
            visualizer = types.ModuleType("groundingdino.util.visualizer")
            visualizer.COCOVisualizer = type("COCOVisualizer", (), {})
            sys.modules.setdefault("groundingdino.util.visualizer", visualizer)
        from groundingdino.util.inference import Model
        from segment_anything import SamPredictor, sam_model_registry

        for value in (dino_config, dino_checkpoint, sam_checkpoint):
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        self._torch = torch
        self._dino = Model(model_config_path=dino_config, model_checkpoint_path=dino_checkpoint)
        self._sam = SamPredictor(sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda"))
        self._image: np.ndarray | None = None
        self._image_name: str | None = None
        self.sam_encoding_count = 0

    def begin_camera(self, image_name: str, image_rgb: Any) -> None:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("full camera image must be RGB")
        self._sam.set_image(image)
        self._image = image
        self._image_name = str(image_name)
        self.sam_encoding_count += 1

    def review_crop(
        self,
        *,
        candidate_id: int,
        round_index: int,
        camera_index: int,
        crop_spec: CropSpec,
        seed_mask_full: Any,
        negative_mask_full: Any,
        seed_prompt_mask_full: Any | None = None,
        classes: Sequence[str],
        config: RefinementConfig = RefinementConfig(),
    ) -> tuple[MaskHypothesis, ...]:
        import torchvision

        if self._image is None or self._image_name is None:
            raise RuntimeError("begin_camera must be called before review_crop")
        crop, seed_crop = extract_crop(self._image, seed_mask_full, crop_spec)
        detection = self._dino.predict_with_classes(
            image=crop[..., ::-1].copy(),
            classes=list(classes),
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
        )
        if len(detection.xyxy) == 0:
            return ()
        keep = torchvision.ops.nms(
            self._torch.as_tensor(detection.xyxy, dtype=self._torch.float32),
            self._torch.as_tensor(detection.confidence, dtype=self._torch.float32),
            config.nms_threshold,
        ).cpu().numpy()
        proposals = rank_detection_proposals(
            boxes_xyxy=np.asarray(detection.xyxy)[keep],
            scores=np.asarray(detection.confidence)[keep],
            class_ids=np.asarray(detection.class_id, dtype=object)[keep],
            classes=classes,
            seed_mask=seed_crop,
            crop_spec=crop_spec,
            limit=config.max_boxes_per_view,
        )
        crop_region = np.zeros(self._image.shape[:2], dtype=bool)
        x0, y0 = max(crop_spec.left, 0), max(crop_spec.top, 0)
        x1 = min(crop_spec.left + crop_spec.side, self._image.shape[1])
        y1 = min(crop_spec.top + crop_spec.side, self._image.shape[0])
        if x1 > x0 and y1 > y0:
            crop_region[y0:y1, x0:x1] = True
        prompt_source = seed_mask_full if seed_prompt_mask_full is None else seed_prompt_mask_full
        positive = dispersed_prompt_points(np.asarray(prompt_source, dtype=bool) & crop_region, 4)
        if not len(positive):
            positive = dispersed_prompt_points(np.asarray(seed_mask_full, dtype=bool) & crop_region, 4)
        negative = dispersed_prompt_points(np.asarray(negative_mask_full, dtype=bool) & crop_region, 4)
        points = np.concatenate((positive, negative), axis=0)
        labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative)))).astype(np.int32)
        hypotheses: list[MaskHypothesis] = []
        for box_ordinal, proposal in enumerate(proposals):
            x0, y0, x1, y1 = proposal.box_image_xyxy
            clipped_box = (
                float(np.clip(x0, 0, self._image.shape[1])),
                float(np.clip(y0, 0, self._image.shape[0])),
                float(np.clip(x1, 0, self._image.shape[1])),
                float(np.clip(y1, 0, self._image.shape[0])),
            )
            if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
                continue
            proposal = replace(proposal, box_image_xyxy=clipped_box)
            masks, scores, _ = self._sam.predict(
                point_coords=points if len(points) else None,
                point_labels=labels if len(points) else None,
                box=np.asarray(clipped_box, dtype=np.float32),
                multimask_output=True,
            )
            hypotheses.extend(
                rank_sam_masks(
                    masks=masks,
                    sam_scores=scores,
                    proposal=proposal,
                    seed_mask_full=seed_mask_full,
                    negative_mask_full=negative_mask_full,
                    candidate_id=candidate_id,
                    round_index=round_index,
                    camera_index=camera_index,
                    image_name=self._image_name,
                    crop_kind=crop_spec.kind,
                    stable_base=box_ordinal * 10,
                    limit=config.max_masks_per_view,
                )
            )
        hypotheses.sort(
            key=lambda row: (
                -row.seed_coverage,
                -row.sam_score,
                -row.detection_score,
                row.mask_area,
                row.stable_ordinal,
            )
        )
        return tuple(hypotheses[: config.max_masks_per_view])


__all__ = [
    "DetectionProposal",
    "GroundedSamFullImageReviewer",
    "dispersed_prompt_points",
    "rank_detection_proposals",
    "rank_sam_masks",
]
