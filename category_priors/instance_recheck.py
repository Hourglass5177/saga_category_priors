from __future__ import annotations

"""Project frozen 3D candidates into RGB views and verify them in 2D.

The runtime path in this module deliberately contains no ground truth.  It
reads one immutable CandidateBank, reviews every projectable candidate under
the global- and class-sized crop conditions, and replays the accepted IDs
through the same legacy KNN/filter/finalization path.
"""

import argparse
import json
import math
import os
import re
import sys
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from .candidate_bank import CandidateBank, gaussian_xyz_sha256, load_candidate_bank
from .candidate_replay import (
    candidate_bank_to_replay_candidates,
    candidate_export_ids,
    replay_candidates_through_legacy,
)
from .instance_projection import bounded_recheck_crop_side
from .prediction_finalization import (
    finalize_prediction,
    prediction_output_payload,
    write_prediction_output_atomic,
)


BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.35
NMS_THRESHOLD = 0.80
MIN_PROJECTED_PIXELS = 4
MAX_VIEWS = 3
MIN_COVERAGE = 0.50
DEFAULT_CLASSES = (
    "chair", "table", "plant", "flower", "foliage", "tv", "painting",
    "sofa", "cabinet", "bed", "wall", "floor", "ceiling", "person",
    "socket", "remote", "key", "book", "lighting", "switch", "door",
    "window", "lamp", "speaker", "computer", "fan", "refrigerator",
    "robot", "cup", "vase", "phone", "trash can",
)


@dataclass(frozen=True)
class ProjectedCandidateView:
    candidate_id: int
    camera_index: int
    image_name: str
    pixel_count: int
    x0: int
    y0: int
    x1: int
    y1: int
    centroid_x: float
    centroid_y: float
    image_width: int
    image_height: int


@dataclass(frozen=True)
class CropGeometry:
    side: int
    left: int
    top: int
    crop_capped: bool
    requested_side_px: float
    prior_scaled_side_px: float


@dataclass(frozen=True)
class ViewReview:
    image_name: str
    projected_pixels: int
    crop_side: int
    crop_capped: bool
    detected_class: str | None
    detection_confidence: float
    coverage: float
    mask_iou: float
    mask_candidate_area_ratio: float
    confirmed: bool


@dataclass(frozen=True)
class CandidateReview:
    candidate_id: int
    branch_class: str
    condition: str
    prior_diagonal_m: float
    usable_view_count: int
    confirming_view_count: int
    required_votes: int
    accepted: bool
    views: tuple[ViewReview, ...]


class CropReviewer(Protocol):
    def review(
        self,
        image_rgb: np.ndarray,
        candidate_mask: np.ndarray,
        class_names: Sequence[str],
        branch_class: str,
    ) -> tuple[str | None, float, np.ndarray | None]: ...


def candidate_projection_rows(
    *,
    candidate_by_gaussian: Any,
    contributor_ids: Any,
    contribution_weights: Any,
    camera_index: int,
    image_name: str,
) -> tuple[ProjectedCandidateView, ...]:
    """Return projectable candidate observations for one rendered view."""

    membership = np.asarray(candidate_by_gaussian, dtype=np.int64)
    contributor = np.asarray(contributor_ids, dtype=np.int64)
    weights = np.asarray(contribution_weights, dtype=np.float64)
    if membership.ndim != 1 or contributor.ndim != 2 or weights.shape != contributor.shape:
        raise ValueError("candidate membership and contributor maps have invalid shapes")
    valid = (
        (contributor >= 0)
        & (contributor < len(membership))
        & np.isfinite(weights)
        & (weights > 0)
    )
    pixel_candidate = np.full(contributor.shape, -1, dtype=np.int64)
    pixel_candidate[valid] = membership[contributor[valid]]
    rows: list[ProjectedCandidateView] = []
    height, width = contributor.shape
    for candidate_id in np.unique(pixel_candidate[pixel_candidate >= 0]):
        ys, xs = np.nonzero(pixel_candidate == candidate_id)
        if len(xs) < MIN_PROJECTED_PIXELS:
            continue
        rows.append(
            ProjectedCandidateView(
                candidate_id=int(candidate_id),
                camera_index=int(camera_index),
                image_name=str(image_name),
                pixel_count=int(len(xs)),
                x0=int(xs.min()),
                y0=int(ys.min()),
                x1=int(xs.max()) + 1,
                y1=int(ys.max()) + 1,
                centroid_x=float(xs.mean()),
                centroid_y=float(ys.mean()),
                image_width=int(width),
                image_height=int(height),
            )
        )
    return tuple(rows)


def select_candidate_views(
    rows: Sequence[ProjectedCandidateView], max_views: int = MAX_VIEWS
) -> dict[int, tuple[ProjectedCandidateView, ...]]:
    grouped: dict[int, list[ProjectedCandidateView]] = defaultdict(list)
    for row in rows:
        grouped[int(row.candidate_id)].append(row)
    return {
        candidate_id: tuple(
            sorted(values, key=lambda row: (-row.pixel_count, row.image_name))[
                : int(max_views)
            ]
        )
        for candidate_id, values in sorted(grouped.items())
    }


def projection_mask(
    candidate_by_gaussian: Any,
    contributor_ids: Any,
    contribution_weights: Any,
    candidate_id: int,
) -> np.ndarray:
    membership = np.asarray(candidate_by_gaussian, dtype=np.int64)
    contributor = np.asarray(contributor_ids, dtype=np.int64)
    weights = np.asarray(contribution_weights, dtype=np.float64)
    valid = (
        (contributor >= 0)
        & (contributor < len(membership))
        & np.isfinite(weights)
        & (weights > 0)
    )
    safe = contributor.clip(0, max(len(membership) - 1, 0))
    return valid & (membership[safe] == int(candidate_id))


def crop_candidate_view(
    *,
    image_rgb: np.ndarray,
    projected_mask: np.ndarray,
    candidate_diagonal_m: float,
    prior_diagonal_m: float,
) -> tuple[np.ndarray, np.ndarray, CropGeometry]:
    """Create the frozen square crop with neutral-grey out-of-frame padding."""

    image = np.asarray(image_rgb, dtype=np.uint8)
    mask = np.asarray(projected_mask, dtype=bool)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
        raise ValueError("RGB image and projected mask must share H x W")
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PROJECTED_PIXELS:
        raise ValueError("candidate projection has fewer than four valid pixels")
    side_px = float(max(int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1))
    scale = bounded_recheck_crop_side(
        candidate_side_px=side_px,
        candidate_diagonal_m=float(candidate_diagonal_m),
        prior_diagonal_m=float(prior_diagonal_m),
        image_width=image.shape[1],
        image_height=image.shape[0],
    )
    side = max(1, int(math.ceil(scale.crop_side_px)))
    center_x = int(math.floor(float(xs.mean()) + 0.5))
    center_y = int(math.floor(float(ys.mean()) + 0.5))
    left = center_x - side // 2
    top = center_y - side // 2
    crop = np.full((side, side, 3), 127, dtype=np.uint8)
    crop_mask = np.zeros((side, side), dtype=bool)
    source_x0 = max(left, 0)
    source_y0 = max(top, 0)
    source_x1 = min(left + side, image.shape[1])
    source_y1 = min(top + side, image.shape[0])
    if source_x1 > source_x0 and source_y1 > source_y0:
        target_x0 = source_x0 - left
        target_y0 = source_y0 - top
        target_x1 = target_x0 + source_x1 - source_x0
        target_y1 = target_y0 + source_y1 - source_y0
        crop[target_y0:target_y1, target_x0:target_x1] = image[
            source_y0:source_y1, source_x0:source_x1
        ]
        crop_mask[target_y0:target_y1, target_x0:target_x1] = mask[
            source_y0:source_y1, source_x0:source_x1
        ]
    return crop, crop_mask, CropGeometry(
        side=side,
        left=left,
        top=top,
        crop_capped=scale.crop_capped,
        requested_side_px=scale.requested_side_px,
        prior_scaled_side_px=scale.prior_scaled_side_px,
    )


def review_geometry(
    *,
    candidate_mask: np.ndarray,
    detected_mask: np.ndarray | None,
) -> tuple[float, float]:
    candidate = np.asarray(candidate_mask, dtype=bool)
    if detected_mask is None:
        return 0.0, 0.0
    detected = np.asarray(detected_mask, dtype=bool)
    if detected.shape != candidate.shape:
        raise ValueError("detected and candidate masks must share a crop")
    candidate_area = int(candidate.sum())
    detected_area = int(detected.sum())
    if candidate_area == 0:
        raise ValueError("candidate mask is empty")
    intersection = int(np.count_nonzero(candidate & detected))
    union = candidate_area + detected_area - intersection
    coverage = intersection / candidate_area
    iou = intersection / union if union else 0.0
    return float(coverage), float(iou)


def required_confirming_votes(view_count: int) -> int:
    count = int(view_count)
    if count <= 0:
        return 1
    return 1 if count == 1 else 2


def aggregate_candidate_review(
    *,
    candidate_id: int,
    branch_class: str,
    condition: str,
    prior_diagonal_m: float,
    views: Sequence[ViewReview],
) -> CandidateReview:
    required = required_confirming_votes(len(views))
    confirming = sum(bool(view.confirmed) for view in views)
    return CandidateReview(
        candidate_id=int(candidate_id),
        branch_class=str(branch_class),
        condition=str(condition),
        prior_diagonal_m=float(prior_diagonal_m),
        usable_view_count=len(views),
        confirming_view_count=int(confirming),
        required_votes=required,
        accepted=bool(views) and confirming >= required,
        views=tuple(views),
    )


def _prior_diagonal(priors: dict[str, Any], class_name: str | None) -> float:
    node = priors["global"] if class_name is None else priors.get("categories", {}).get(class_name)
    if not isinstance(node, dict) or not node.get("active", False):
        node = priors["global"]
    try:
        return float(math.exp(node["shrunk"]["geometry"]["log_bbox_diag_m"]["q50"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("train-only priors lack the frozen bbox-diagonal median") from exc


def _camera_rgb(camera: Any) -> np.ndarray:
    image = camera.original_image.detach().cpu().clamp(0, 1).numpy()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{camera.image_name}: camera RGB tensor has an invalid shape")
    return np.rint(np.moveaxis(image, 0, 2) * 255.0).astype(np.uint8)


def _render_contributor(camera: Any, model: Any, args: Any, background: Any) -> tuple[np.ndarray, np.ndarray]:
    import diff_gaussian_rasterization_max_contributor as contributor_module
    from gaussian_renderer import render_with_max_contributor

    package_dir = (
        Path(__file__).resolve().parents[1]
        / "submodules"
        / "diff-gaussian-rasterization-max-contributor"
        / "diff_gaussian_rasterization_max_contributor"
    ).resolve()
    expected = (package_dir / "__init__.py").resolve()
    actual = Path(contributor_module.__file__).resolve()
    if actual != expected:
        raise RuntimeError(
            "wrong max-contributor extension imported: "
            f"expected {expected}, got {actual}"
        )
    binary = Path(contributor_module._C.__file__).resolve()
    if binary.parent != package_dir:
        raise RuntimeError(
            "wrong max-contributor CUDA binary imported: "
            f"expected it below {package_dir}, got {binary}"
        )
    rendered = render_with_max_contributor(camera, model, args, background)
    return (
        rendered["max_contributor"].detach().cpu().numpy(),
        rendered["max_contribute"].detach().cpu().numpy(),
    )


class GroundedSamCropReviewer:
    """One model pair per scene; complete 32-class competition per crop."""

    def __init__(self, *, dino_config: str, dino_checkpoint: str, sam_checkpoint: str) -> None:
        import torch

        # The persisted offline GroundingDINO environment uses Hugging Face's
        # BERT tokenizer.  Its pure-Python path only needs the stdlib regular
        # expression API, while the optional visualizer (and pycocotools) is
        # irrelevant to inference.  Keep those optional packages out of the
        # runtime contract rather than downloading them on a fresh GPU host.
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

        for path in (dino_config, dino_checkpoint, sam_checkpoint):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        self._torch = torch
        self._dino = Model(
            model_config_path=dino_config,
            model_checkpoint_path=dino_checkpoint,
        )
        self._sam = SamPredictor(
            sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda")
        )

    def review(
        self,
        image_rgb: np.ndarray,
        candidate_mask: np.ndarray,
        class_names: Sequence[str],
        branch_class: str,
    ) -> tuple[str | None, float, np.ndarray | None]:
        import torchvision

        image = np.asarray(image_rgb, dtype=np.uint8)
        candidate = np.asarray(candidate_mask, dtype=bool)
        detection = self._dino.predict_with_classes(
            image=image[..., ::-1].copy(),
            classes=list(class_names),
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )
        if len(detection.xyxy) == 0:
            return None, 0.0, None
        keep = torchvision.ops.nms(
            self._torch.as_tensor(detection.xyxy, dtype=self._torch.float32),
            self._torch.as_tensor(detection.confidence, dtype=self._torch.float32),
            NMS_THRESHOLD,
        ).cpu().numpy()
        boxes = np.asarray(detection.xyxy)[keep]
        confidence = np.asarray(detection.confidence)[keep]
        class_ids = np.asarray(detection.class_id, dtype=object)[keep]
        ys, xs = np.nonzero(candidate)
        candidate_box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        self._sam.set_image(image)
        best: tuple[float, int, str, float, np.ndarray] | None = None
        for ordinal, (box, score, raw_class_id) in enumerate(zip(boxes, confidence, class_ids)):
            if raw_class_id is None:
                continue
            class_id = int(raw_class_id)
            if not 0 <= class_id < len(class_names):
                continue
            x0, y0, x1, y1 = (float(value) for value in box)
            if x1 <= candidate_box[0] or x0 >= candidate_box[2] or y1 <= candidate_box[1] or y0 >= candidate_box[3]:
                continue
            masks, mask_scores, _ = self._sam.predict(box=np.asarray(box), multimask_output=True)
            chosen = int(np.argmax(mask_scores))
            mask = np.asarray(masks[chosen], dtype=bool)
            coverage, _ = review_geometry(candidate_mask=candidate, detected_mask=mask)
            evidence = float(score) * coverage
            row = (evidence, -ordinal, str(class_names[class_id]), float(score), mask)
            if best is None or row[:2] > best[:2]:
                best = row
        if best is None:
            return None, 0.0, None
        return best[2], best[3], best[4]


def _load_cameras(args: Any) -> list[Any]:
    from postprocess import _load_cameras as load_cameras

    return load_cameras(args)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _write_projection_overlay(
    path: Path,
    *,
    camera: Any,
    candidate_mask: np.ndarray,
) -> None:
    """Save one human-checkable alignment image before model review starts."""

    from PIL import Image

    image = _camera_rgb(camera).copy()
    mask = np.asarray(candidate_mask, dtype=bool)
    edge = mask & ~(
        np.roll(mask, 1, axis=0)
        & np.roll(mask, -1, axis=0)
        & np.roll(mask, 1, axis=1)
        & np.roll(mask, -1, axis=1)
    )
    image[edge] = np.asarray([0, 255, 0], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part" + path.suffix)
    Image.fromarray(image).save(temporary)
    os.replace(temporary, path)


def _review_condition(
    *,
    condition: str,
    bank: CandidateBank,
    selected_views: dict[int, tuple[ProjectedCandidateView, ...]],
    rendered_by_camera: dict[int, tuple[np.ndarray, np.ndarray]],
    cameras: Sequence[Any],
    reviewer: CropReviewer,
    priors: dict[str, Any],
) -> tuple[CandidateReview, ...]:
    rows: list[CandidateReview] = []
    metadata = {int(row["candidate_id"]): row for row in bank.candidates}
    for candidate_id in range(len(bank.candidates)):
        candidate = metadata[candidate_id]
        branch_class = str(candidate["branch_class"])
        prior_diagonal = _prior_diagonal(
            priors, branch_class if condition == "class" else None
        )
        extents = np.asarray(candidate["metric_extents_m"], dtype=np.float64)
        candidate_diagonal = float(np.linalg.norm(extents))
        view_results: list[ViewReview] = []
        for observation in selected_views.get(candidate_id, ()):
            contributor, weights = rendered_by_camera[observation.camera_index]
            full_mask = projection_mask(
                bank.branch_full_labels, contributor, weights, candidate_id
            )
            crop, crop_mask, crop_geometry = crop_candidate_view(
                image_rgb=_camera_rgb(cameras[observation.camera_index]),
                projected_mask=full_mask,
                candidate_diagonal_m=candidate_diagonal,
                prior_diagonal_m=prior_diagonal,
            )
            detected_class, confidence, detected_mask = reviewer.review(
                crop, crop_mask, bank.class_names, branch_class
            )
            coverage, iou = review_geometry(
                candidate_mask=crop_mask, detected_mask=detected_mask
            )
            detected_area = int(np.count_nonzero(detected_mask)) if detected_mask is not None else 0
            candidate_area = int(np.count_nonzero(crop_mask))
            view_results.append(
                ViewReview(
                    image_name=observation.image_name,
                    projected_pixels=observation.pixel_count,
                    crop_side=crop_geometry.side,
                    crop_capped=crop_geometry.crop_capped,
                    detected_class=detected_class,
                    detection_confidence=float(confidence),
                    coverage=coverage,
                    mask_iou=iou,
                    mask_candidate_area_ratio=(detected_area / candidate_area if candidate_area else 0.0),
                    confirmed=detected_class == branch_class and coverage >= MIN_COVERAGE,
                )
            )
        rows.append(
            aggregate_candidate_review(
                candidate_id=candidate_id,
                branch_class=branch_class,
                condition=condition,
                prior_diagonal_m=prior_diagonal,
                views=view_results,
            )
        )
    return tuple(rows)


def run_scene(args: Any) -> dict[str, Any]:
    import torch
    from scene import FeatureGaussianModel, GaussianModel

    from .semantic_voting import (
        compute_instance_vote_evidence,
        summarize_mask_label_assets,
    )

    if tuple(args.classes) != DEFAULT_CLASSES:
        raise ValueError("the recheck runtime requires the frozen 32-class order")
    bank = load_candidate_bank(args.candidate_bank)
    gs_model = GaussianModel(args.sh_degree)
    gs_model.load_ply(args.point_cloud_path)
    feature_model = FeatureGaussianModel(args.feature_dim, args.semantic_feature_dim)
    feature_model.load_ply(args.contrastive_feature_point_cloud_path)
    xyz = feature_model.get_xyz.detach().cpu()
    if len(xyz) != bank.point_count or not torch.allclose(
        gs_model.get_xyz.detach().cpu(), xyz, rtol=0.0, atol=1e-6
    ):
        raise ValueError("candidate bank, RGB Gaussians and feature Gaussians disagree")
    if bank.gaussian_xyz_sha256 is None:
        raise ValueError("instance recheck requires a fresh bank with a Gaussian XYZ fingerprint")
    if gaussian_xyz_sha256(xyz) != bank.gaussian_xyz_sha256:
        raise ValueError("candidate bank Gaussian XYZ fingerprint does not match the scene")
    point_scales = feature_model.get_scaling.detach().cpu()
    max_scale = point_scales.max(dim=-1).values
    is_big = max_scale > max_scale.median() * args.scale_threshold
    is_transparent = feature_model.get_opacity.detach().cpu().squeeze() < args.opcity_threshold
    cameras = _load_cameras(args)
    semantic_vote_assets = summarize_mask_label_assets(args, cameras)
    background = torch.tensor(
        [1, 1, 1] if args.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    all_observations: list[ProjectedCandidateView] = []
    for camera_index, camera in enumerate(cameras):
        contributor, weights = _render_contributor(camera, gs_model, args, background)
        all_observations.extend(
            candidate_projection_rows(
                candidate_by_gaussian=bank.branch_full_labels,
                contributor_ids=contributor,
                contribution_weights=weights,
                camera_index=camera_index,
                image_name=camera.image_name,
            )
        )
    selected = select_candidate_views(all_observations)
    # Contributor images are large.  The discovery pass deliberately discards
    # them, then renders only cameras that survived deterministic top-view
    # selection.  This bounds host memory without changing any observation.
    selected_camera_indices = sorted({
        observation.camera_index
        for observations in selected.values()
        for observation in observations
    })
    rendered_by_camera = {
        camera_index: _render_contributor(
            cameras[camera_index], gs_model, args, background
        )
        for camera_index in selected_camera_indices
    }
    first_observation = next(
        (
            observation
            for candidate_id in sorted(selected)
            for observation in selected[candidate_id][:1]
        ),
        None,
    )
    output_dir = Path(args.output_dir)
    if first_observation is not None:
        contributor, weights = rendered_by_camera[first_observation.camera_index]
        _write_projection_overlay(
            output_dir / "projection_overlay.png",
            camera=cameras[first_observation.camera_index],
            candidate_mask=projection_mask(
                bank.branch_full_labels,
                contributor,
                weights,
                first_observation.candidate_id,
            ),
        )
    reviewer = GroundedSamCropReviewer(
        dino_config=args.groundingdino_config_path,
        dino_checkpoint=args.groundingdino_checkpoint_path,
        sam_checkpoint=args.sam_checkpoint_path,
    )
    priors = json.loads(Path(args.priors).read_text(encoding="utf-8"))
    global_reviews = _review_condition(
        condition="global", bank=bank, selected_views=selected,
        rendered_by_camera=rendered_by_camera, cameras=cameras,
        reviewer=reviewer, priors=priors,
    )
    class_reviews = _review_condition(
        condition="class", bank=bank, selected_views=selected,
        rendered_by_camera=rendered_by_camera, cameras=cameras,
        reviewer=reviewer, priors=priors,
    )
    replay_input = candidate_bank_to_replay_candidates(bank)
    accepted = {
        "raw": tuple(range(len(bank.candidates))),
        "global": tuple(row.candidate_id for row in global_reviews if row.accepted),
        "class": tuple(row.candidate_id for row in class_reviews if row.accepted),
    }
    replays = {
        name: replay_candidates_through_legacy(
            xyz_scene=xyz,
            global_pre_knn=bank.global_pre_knn,
            candidates=replay_input.candidates,
            accepted_candidate_ids=ids,
        )
        for name, ids in accepted.items()
    }
    ratios, raw_votes = compute_instance_vote_evidence(
        label_sets={name: replay.after_filter for name, replay in replays.items()},
        camera_list=cameras,
        gs_model=gs_model,
        args=args,
        bg_color=background,
        update_progress=True,
    )
    conditions: dict[str, Any] = {}
    for name, replay in replays.items():
        finalized = finalize_prediction(
            point_labels=replay.after_filter,
            xyz_scene=xyz,
            is_big_gaussian=is_big,
            vote_ratios_by_raw=ratios[name],
            class_names=args.classes,
            selected_classes=bank.saga20_names,
            label_threshold=args.label_threshold,
        )
        payload = prediction_output_payload(
            finalized,
            is_big_gaussian=is_big,
            is_transparent_gaussian=is_transparent,
        )
        source_ids = candidate_export_ids(replay, finalized.contracted)
        payload["candidate_export_ids"] = {
            str(key): int(value) for key, value in source_ids.items()
        }
        payload["condition"] = name
        write_prediction_output_atomic(output_dir / name / "output.json", payload)
        conditions[name] = {
            "accepted_candidate_ids": list(accepted[name]),
            "exported_candidate_ids": list(source_ids),
            "instance_count": len(finalized.contracted.instances),
            "replay": dict(replay.diagnostics),
            "candidate_survival": [row.to_dict() for row in replay.candidates],
            "vote_histogram_33": {
                str(key): value.tolist() for key, value in raw_votes[name].items()
            },
        }
    audit = {
        "schema": "saga-instance-recheck-scene-v1",
        "candidate_bank": str(Path(args.candidate_bank).resolve()),
        "candidate_count": len(bank.candidates),
        "projectable_candidate_count": len(selected),
        "selected_camera_count": len(selected_camera_indices),
        "semantic_vote_assets": semantic_vote_assets,
        "gaussian_xyz_sha256": bank.gaussian_xyz_sha256,
        "candidate_bank_replay": dict(replay_input.diagnostics),
        "global_reviews": [asdict(row) for row in global_reviews],
        "class_reviews": [asdict(row) for row in class_reviews],
        "conditions": conditions,
    }
    _write_json_atomic(output_dir / "instance_recheck.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    from arguments import ModelParams, PipelineParams

    parser = argparse.ArgumentParser(description="Frozen-candidate 3D-to-2D recheck")
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-path", required=True)
    parser.add_argument("--sam-checkpoint-path", required=True)
    parser.add_argument("--groundingdino-checkpoint-path", required=True)
    parser.add_argument("--groundingdino-config-path", required=True)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--label-threshold", type=float, default=0.3)
    parser.add_argument("--scale-threshold", type=float, default=0.8)
    parser.add_argument("--opcity-threshold", type=float, default=0.005)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_scene(args)
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "projectable_candidate_count": result["projectable_candidate_count"],
        "conditions": {
            key: value["instance_count"] for key, value in result["conditions"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CandidateReview", "CropGeometry", "GroundedSamCropReviewer",
    "ProjectedCandidateView", "ViewReview", "aggregate_candidate_review",
    "candidate_projection_rows", "crop_candidate_view", "projection_mask",
    "required_confirming_votes", "review_geometry", "run_scene",
    "select_candidate_views",
]
