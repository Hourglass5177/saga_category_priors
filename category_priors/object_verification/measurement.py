"""Numerical attribution only. No object identity, graph, vote or scene mutation.

Reference objective preserves normalized alpha*T_previous pixel contributions.
The analytic path is an independent CPU formula check, not a model experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import file_digest, digest


@dataclass(frozen=True)
class AlphaMeasurement:
    inside_mass: np.ndarray
    visible_mass: np.ndarray
    valid_pixels: int

    def __post_init__(self):
        inside = np.array(self.inside_mass, dtype=np.float64, copy=True)
        visible = np.array(self.visible_mass, dtype=np.float64, copy=True)
        if inside.ndim != 2 or visible.ndim != 1 or inside.shape[1] != len(visible):
            raise ValueError("expected inside[M,N], visible[N]")
        if not np.isfinite(inside).all() or not np.isfinite(visible).all():
            raise ValueError("nonfinite reference mass")
        if np.any(inside < -1e-7) or np.any(visible < -1e-7):
            raise ValueError("negative reference mass")
        if np.any(inside - visible[None, :] > 5e-5 * np.maximum(visible[None, :], 1.0)):
            raise ValueError("reference inside exceeds canonical visible denominator")
        # Preserve raw accepted numeric values, including first cached drift.
        # Clipping is only performed when classifying a member, never in storage.
        # Bytes-backed arrays cannot regain write access through setflags().
        inside = np.frombuffer(inside.tobytes(), dtype=inside.dtype).reshape(inside.shape)
        visible = np.frombuffer(visible.tobytes(), dtype=visible.dtype).reshape(visible.shape)
        object.__setattr__(self, "inside_mass", inside)
        object.__setattr__(self, "visible_mass", visible)

    def qualified(self, mask_index: int) -> tuple[int, ...]:
        inside = self.inside_mass[mask_index]
        ratio = np.divide(inside, self.visible_mass, out=np.zeros_like(inside),
                          where=self.visible_mass > 0)
        return tuple(int(index) for index in np.flatnonzero((inside >= .5) & (ratio >= .5)))


def normalized_coefficients(opacity: Any, masks: Any, *, valid_pixels: Any = None) -> tuple[np.ndarray, np.ndarray]:
    opacity = np.asarray(opacity, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    if opacity.ndim != 2 or masks.ndim != 3 or masks.shape[1:] != opacity.shape:
        raise ValueError("opacity[H,W] and masks[M,H,W] required")
    if not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1)):
        raise ValueError("mask values outside [0,1]")
    valid = np.isfinite(opacity) & (opacity >= .05)
    if valid_pixels is not None:
        rgb_valid = np.asarray(valid_pixels)
        if rgb_valid.dtype != np.bool_ or rgb_valid.shape != opacity.shape:
            raise ValueError("valid_pixels must be a boolean mask in actual RGB coordinates")
        # The same observed domain controls inside AND visible coefficients.
        valid &= rgb_valid
    inverse = np.zeros_like(opacity)
    inverse[valid] = 1.0 / opacity[valid]
    return inverse, masks * inverse[None, :, :]


def analytic_alpha(contributions: Any, masks: Any, *, valid_pixels: Any = None) -> AlphaMeasurement:
    """A[p,g] = alpha_pg*T_previous_pg, shape H,W,N; no GT inputs."""
    matrix = np.asarray(contributions, dtype=np.float64)
    if matrix.ndim != 3 or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("expected nonnegative H,W,N contribution matrix")
    opacity = matrix.sum(axis=2)
    inverse, coefficients = normalized_coefficients(opacity, masks, valid_pixels=valid_pixels)
    visible = np.einsum("hwn,hw->n", matrix, inverse)
    inside = np.einsum("hwn,mhw->mn", matrix, coefficients)
    return AlphaMeasurement(inside, visible, int(np.count_nonzero(inverse)))


def gradient_reference_from_graph(image, probe, masks, *, valid_pixels=None) -> AlphaMeasurement:
    """Differentiate one real render graph; save these exact first values.

    SceneAccess must pass its RGB/crop validity domain here. Omitting this value
    explicitly measures the whole image, as in the original reference formula.
    """
    import torch
    mask_array = np.asarray(masks, dtype=np.float32)
    opacity = image.detach()[0].float().cpu().numpy()
    inverse, _ = normalized_coefficients(opacity, mask_array, valid_pixels=valid_pixels)
    if tuple(image.shape) != (3, *opacity.shape) or probe.ndim != 2 or probe.shape[1] != 3:
        raise ValueError("reference renderer image/probe axis contract")
    visible_gradient = torch.autograd.grad(
        torch.sum(image[0] * torch.as_tensor(inverse, dtype=image.dtype, device=image.device)),
        probe, retain_graph=bool(len(mask_array)))[0]
    visible = visible_gradient.detach().cpu().numpy()[:, 0].astype(np.float64)
    inside = np.zeros((len(mask_array), len(visible)), dtype=np.float64)
    for start in range(0, len(mask_array), 3):
        stop = min(start + 3, len(mask_array))
        targets = np.zeros((3, *opacity.shape), dtype=np.float32)
        targets[:stop-start] = mask_array[start:stop]
        _, coefficients = normalized_coefficients(opacity, targets, valid_pixels=valid_pixels)
        gradient = torch.autograd.grad(
            torch.sum(image * torch.as_tensor(coefficients, dtype=image.dtype, device=image.device)),
            probe, retain_graph=stop < len(mask_array))[0].detach().cpu().numpy()
        inside[start:stop] = gradient[:, :stop-start].T
    return AlphaMeasurement(inside, visible, int(np.count_nonzero(inverse)))


def render_reference(camera, gaussians, pipeline, background, masks, *, valid_pixels=None) -> AlphaMeasurement:
    import torch
    from gaussian_renderer import render_mask
    probe = torch.ones((len(gaussians.get_xyz), 3), dtype=gaussians.get_xyz.dtype,
                       device=gaussians.get_xyz.device, requires_grad=True)
    image = render_mask(camera, gaussians, pipeline, torch.zeros_like(background),
                        precomputed_mask=probe)["mask"]
    return gradient_reference_from_graph(image, probe, masks, valid_pixels=valid_pixels)


def loaded_backend_identity() -> dict[str, Any]:
    import gaussian_renderer
    import diff_gaussian_rasterization as rasterizer
    import diff_gaussian_rasterization_max_contributor as contributor
    modules = {}
    binaries = {}
    for name, module in (("renderer", gaussian_renderer), ("rasterizer", rasterizer),
                         ("contributor", contributor)):
        modules[name] = {"path": str(Path(module.__file__).resolve()),
                         "sha256": file_digest(module.__file__)}
        if hasattr(module, "_C"):
            binaries[name] = {"path": str(Path(module._C.__file__).resolve()),
                              "sha256": file_digest(module._C.__file__)}
    if set(binaries) != {"rasterizer", "contributor"}:
        raise RuntimeError("cannot identify both actually loaded rasterizer binaries")
    modules["measurement"] = {"path": __file__, "sha256": file_digest(__file__)}
    return {"backend": "gradient-reference", "module_sha256": digest(modules),
            "binary_sha256": digest(binaries), "modules": modules, "binaries": binaries,
            "formula": "normalized-alpha-t-prev-v1"}


def render_contributors(camera, gaussians, pipeline, background):
    import torch
    from gaussian_renderer import render_mask, render_with_max_contributor
    rendered = render_with_max_contributor(camera, gaussians, pipeline, background)
    point_ids = rendered["max_contributor"].detach().cpu().numpy().astype(np.int64)
    weights = rendered["max_contribute"].detach().cpu().numpy().astype(np.float64)
    probe = torch.ones((len(gaussians.get_xyz), 3), dtype=gaussians.get_xyz.dtype,
                       device=gaussians.get_xyz.device)
    opacity = render_mask(camera, gaussians, pipeline, torch.zeros_like(background),
                          precomputed_mask=probe)["mask"].detach()[0].float().cpu().numpy()
    empty = ~np.isfinite(weights) | (weights <= 0) | ~np.isfinite(opacity) | (opacity <= 0)
    point_ids[empty] = -1
    weights[empty] = 0
    return point_ids, weights, opacity
