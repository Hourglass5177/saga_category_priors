from __future__ import annotations

"""Correct max-contributor and streaming normalized-alpha rendering wrappers."""

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import RefinementConfig
from .evidence import AlphaMass, alpha_mass_from_gradients, build_alpha_objective, iter_three_channel_masks


def validate_max_contributor_extension() -> None:
    import diff_gaussian_rasterization_max_contributor as module

    package = (
        Path(__file__).resolve().parents[2]
        / "submodules" / "diff-gaussian-rasterization-max-contributor"
        / "diff_gaussian_rasterization_max_contributor"
    ).resolve()
    if Path(module.__file__).resolve() != (package / "__init__.py").resolve():
        raise RuntimeError("wrong max-contributor Python extension imported")
    if Path(module._C.__file__).resolve().parent != package:
        raise RuntimeError("wrong max-contributor CUDA binary imported")


def render_camera_maps(camera: Any, model: Any, pipeline: Any, background: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from gaussian_renderer import render_mask, render_with_max_contributor

    validate_max_contributor_extension()
    rendered = render_with_max_contributor(camera, model, pipeline, background)
    ids = rendered["max_contributor"].detach().cpu().numpy().astype(np.int64, copy=False)
    weights = rendered["max_contribute"].detach().cpu().numpy().astype(np.float64, copy=False)
    probe = torch.ones((int(model.get_xyz.shape[0]), 3), dtype=model.get_xyz.dtype, device=model.get_xyz.device)
    # Probe colors encode accumulated alpha.  A white raster background would
    # make empty pixels look opaque, so attribution always uses zero background
    # independently of the RGB scene's rendering convention.
    zero_background = torch.zeros_like(background)
    opacity_render = render_mask(camera, model, pipeline, zero_background, precomputed_mask=probe)["mask"]
    opacity = opacity_render.detach()[0].float().cpu().numpy().astype(np.float64, copy=False)
    empty = (~np.isfinite(weights)) | (weights <= 0) | (~np.isfinite(opacity)) | (opacity <= 0)
    ids = ids.copy()
    weights = weights.copy()
    ids[empty] = -1
    weights[empty] = 0.0
    return ids, weights, np.maximum(opacity, 0.0)


def render_alpha_mass(
    camera: Any,
    model: Any,
    pipeline: Any,
    background: Any,
    masks: Sequence[np.ndarray],
    *,
    config: RefinementConfig = RefinementConfig(),
) -> AlphaMass:
    """Accumulate per-Gaussian normalized alpha mass, max three masks/render."""

    import torch
    from gaussian_renderer import render_mask

    mask_array = np.asarray(masks, dtype=np.float32)
    shape = (int(camera.image_height), int(camera.image_width))
    if mask_array.ndim != 3 or mask_array.shape[1:] != shape:
        raise ValueError("alpha masks must share the camera image shape")
    point_count = int(model.get_xyz.shape[0])
    visible_gradient: np.ndarray | None = None
    inside_batches: list[tuple[Sequence[int], np.ndarray]] = []
    valid_count = 0
    zero_background = torch.zeros_like(background)
    for batch_number, (indices, targets) in enumerate(iter_three_channel_masks(mask_array)):
        probe = torch.ones((point_count, 3), dtype=model.get_xyz.dtype, device=model.get_xyz.device, requires_grad=True)
        image = render_mask(camera, model, pipeline, zero_background, precomputed_mask=probe)["mask"]
        opacity = image.detach()[0].float().cpu().numpy()
        objective = build_alpha_objective(indices, targets, opacity, min_opacity=config.alpha_opacity_min)
        coefficients = torch.as_tensor(objective.inside_coefficients, dtype=image.dtype, device=image.device)
        if batch_number == 0:
            visible_coeff = torch.as_tensor(objective.visible_coefficient, dtype=image.dtype, device=image.device)
            visible = torch.sum(image[0] * visible_coeff)
            visible_gradient = torch.autograd.grad(visible, probe, retain_graph=bool(indices))[0].detach().cpu().numpy()
            valid_count = int(np.count_nonzero(objective.valid_pixels))
        if indices:
            inside = torch.sum(image * coefficients)
            gradient = torch.autograd.grad(inside, probe, retain_graph=False)[0].detach().cpu().numpy()
            inside_batches.append((indices, gradient))
    if visible_gradient is None:
        raise RuntimeError("alpha mass rendering requires at least one mask")
    return alpha_mass_from_gradients(visible_gradient, inside_batches, len(mask_array), valid_count)


__all__ = ["render_alpha_mass", "render_camera_maps", "validate_max_contributor_extension"]
