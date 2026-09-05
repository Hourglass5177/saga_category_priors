from __future__ import annotations

"""Correct max-contributor and streaming normalized-alpha rendering wrappers."""

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .contracts import RefinementConfig
from .alpha_backend import AlphaEvidenceCache
from .evidence import AlphaMass


def expected_max_contributor_package() -> Path:
    configured = os.environ.get("SAGA_MAX_CONTRIBUTOR_PACKAGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(__file__).resolve().parents[2]
        / "submodules" / "diff-gaussian-rasterization-max-contributor"
        / "diff_gaussian_rasterization_max_contributor"
    ).resolve()


def validate_max_contributor_extension() -> None:
    import diff_gaussian_rasterization_max_contributor as module

    package = expected_max_contributor_package()
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
    backend: str = "gradient-reference",
    cache: AlphaEvidenceCache | None = None,
    camera_index: int = -1,
) -> AlphaMass:
    """Accumulate normalized all-contributor mass through an explicit backend."""
    from .alpha_backend import _render_backend
    if cache is not None:
        return cache.get(
            camera_index, camera, model, pipeline, background, masks,
            backend=backend, config=config,
        )
    return _render_backend(backend, camera, model, pipeline, background, masks, config)


__all__ = [
    "expected_max_contributor_package",
    "render_alpha_mass",
    "render_camera_maps",
    "validate_max_contributor_extension",
]
