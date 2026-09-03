from __future__ import annotations

"""Shared corrected-contributor 2D semantic voting for final predictions."""

import os
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import render_with_max_contributor


def load_mask_and_labels(args, camera):
    mask_path = os.path.join(args.masks_path, f"{camera.image_name}.pt")
    label_path = os.path.join(args.labels_path, f"{camera.image_name}.pt")
    has_mask = os.path.isfile(mask_path)
    has_label = os.path.isfile(label_path)
    if not has_mask and not has_label:
        return None
    if has_mask != has_label:
        raise FileNotFoundError(
            f"mask/label pair is incomplete for frame {camera.image_name}"
        )
    masks = torch.load(mask_path, map_location="cpu")
    labels = torch.load(label_path, map_location="cpu")
    if masks.ndim != 3:
        raise ValueError(f"{mask_path}: masks must have shape [M,H,W]")
    if masks.shape[-2:] != (camera.image_height, camera.image_width):
        masks = F.interpolate(
            masks.float().unsqueeze(1),
            mode="bilinear",
            size=(camera.image_height, camera.image_width),
            align_corners=False,
        ).squeeze(1) > 0.5
    else:
        masks = masks > 0.5
    labels = torch.as_tensor(labels).reshape(-1)
    if len(labels) != len(masks):
        raise ValueError(f"{camera.image_name}: mask and label counts differ")
    invalid = (labels < 0) | (labels >= len(args.classes))
    if bool(invalid.any()):
        values = sorted({int(value) for value in labels[invalid].tolist()})
        raise ValueError(
            f"{camera.image_name}: mask labels fall outside the 32-class table: {values}"
        )
    return masks, labels


def compute_instance_vote_evidence(
    *,
    label_sets: Mapping[str, torch.Tensor | np.ndarray],
    camera_list,
    gs_model,
    args,
    bg_color: torch.Tensor,
    update_progress: bool,
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, np.ndarray]],
]:
    """Return normalized foreground ratios and raw 32-class-plus-background votes."""

    prepared: dict[str, torch.Tensor] = {}
    instance_ids: dict[str, list[int]] = {}
    votes: dict[str, np.ndarray] = {}
    for name, raw_labels in label_sets.items():
        labels = torch.as_tensor(raw_labels, dtype=torch.long).detach().cpu()
        prepared[name] = labels
        ids = [int(value) for value in torch.unique(labels) if int(value) >= 0]
        instance_ids[name] = ids
        width = max(ids) + 1 if ids else 0
        votes[name] = np.zeros((width, len(args.classes) + 1), dtype=np.int64)

    for index, camera in tqdm(list(enumerate(camera_list))):
        if update_progress:
            with open(args.progress_path, "w", encoding="utf-8") as handle:
                handle.write(str((index + 1) * 100 // len(camera_list)))
        loaded = load_mask_and_labels(args, camera)
        if loaded is None:
            continue
        masks, labels_2d = loaded
        render = render_with_max_contributor(camera, gs_model, args, bg_color)
        contributor = render["max_contributor"].detach().cpu().long()
        contribution = render["max_contribute"].detach().cpu()
        expected_shape = (camera.image_height, camera.image_width)
        if contributor.shape != expected_shape or contribution.shape != expected_shape:
            raise ValueError(
                f"{camera.image_name}: contributor render has an invalid shape"
            )
        valid = (
            (contributor >= 0)
            & (contributor < int(gs_model.get_xyz.shape[0]))
            & torch.isfinite(contribution)
            & (contribution > 0)
        )
        gaussian_count = int(gs_model.get_xyz.shape[0])
        if gaussian_count <= 0:
            raise ValueError("RGB Gaussian model is empty")
        safe_contributor = contributor.clamp(min=0, max=gaussian_count - 1)
        background = torch.ones(expected_shape, dtype=torch.bool)
        for mask in masks:
            background &= ~mask

        for name, point_labels in prepared.items():
            if not instance_ids[name]:
                continue
            if len(point_labels) != gaussian_count:
                raise ValueError(f"{name}: label count differs from RGB Gaussian count")
            pixel_instances = point_labels[safe_contributor].clone()
            pixel_instances[~valid] = -1
            maximum_id = votes[name].shape[0] - 1
            for label_2d, mask in zip(labels_2d, masks):
                label_index = int(label_2d)
                values = pixel_instances[mask]
                values = values[(values >= 0) & (values <= maximum_id)]
                if values.numel():
                    votes[name][:, label_index] += torch.bincount(
                        values, minlength=maximum_id + 1
                    ).numpy()
            values = pixel_instances[background]
            values = values[(values >= 0) & (values <= maximum_id)]
            if values.numel():
                votes[name][:, len(args.classes)] += torch.bincount(
                    values, minlength=maximum_id + 1
                ).numpy()

    result: dict[str, dict[int, np.ndarray]] = {}
    raw_result: dict[str, dict[int, np.ndarray]] = {}
    for name, ids in instance_ids.items():
        rows: dict[int, np.ndarray] = {}
        raw_rows: dict[int, np.ndarray] = {}
        for instance_id in ids:
            raw_row = votes[name][instance_id].astype(np.int64, copy=True)
            raw_rows[instance_id] = raw_row
            row = raw_row.astype(np.float64, copy=False)
            denominator = float(row.sum())
            rows[instance_id] = (
                row[:-1] / denominator
                if denominator > 0
                else np.zeros(len(args.classes), dtype=np.float64)
            )
        result[name] = rows
        raw_result[name] = raw_rows
    return result, raw_result


__all__ = ["compute_instance_vote_evidence", "load_mask_and_labels"]
