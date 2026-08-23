from __future__ import annotations

"""Pure attribution-to-fragment primitives for the V8 lifting audit.

The renderer is deliberately outside this module.  Maximum-contributor (M1)
lifting is reduced directly from its ID/weight images.  Alpha-mass (AM)
lifting is represented by pixel objective coefficients and point-feature
gradients, so a worker can use the existing differentiable renderer without
materialising a dense pixel-by-Gaussian contributor cache.
"""

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class V8FragmentConfig:
    full_min_inside_mass: float = 0.5
    core_min_inside_mass: float = 2.0
    core_min_inside_ratio: float = 0.50
    fragment_min_core: int = 5
    fragment_min_full: int = 10


@dataclass(frozen=True)
class AttributionMass:
    """Per-mask inside mass and shared per-frame visible mass.

    ``inside_mass`` is shaped ``(mask_count, point_count)`` and
    ``visible_mass`` is shaped ``(point_count,)``.  Missing masks are encoded
    as ``abstained=True`` rather than as background evidence.
    """

    source: str
    inside_mass: np.ndarray
    visible_mass: np.ndarray
    valid_pixel_count: int
    abstained: bool = False

    @property
    def mask_count(self) -> int:
        return int(self.inside_mass.shape[0])

    @property
    def point_count(self) -> int:
        return int(self.visible_mass.shape[0])

    @property
    def outside_mass(self) -> np.ndarray:
        """Mass visible outside each mask, including unsupported foreground."""
        return np.maximum(
            self.visible_mass[None, :] - self.inside_mass,
            0.0,
        )


@dataclass(frozen=True)
class AttributionFragment:
    fragment_id: int
    frame_id: int
    mask_index: int
    full_ids: np.ndarray
    core_ids: np.ndarray
    full_inside_mass: np.ndarray
    core_inside_mass: np.ndarray
    core_inside_ratio: np.ndarray


@dataclass(frozen=True)
class LiftedAttributionFrame:
    frame_id: int
    source: str
    fragments: tuple[AttributionFragment, ...]
    visible_ids: np.ndarray
    valid_pixel_count: int
    abstained: bool


@dataclass(frozen=True)
class AMChannelBatch:
    """Up to three masks packed into the renderer's RGB channels."""

    mask_indices: tuple[int, ...]
    targets: np.ndarray

    @property
    def active_channels(self) -> int:
        return len(self.mask_indices)


@dataclass(frozen=True)
class AMObjectiveTargets:
    """Detached pixel coefficients for differentiable alpha-mass lifting.

    If the renderer output for a point-feature channel is ``rendered[c]``, the
    inside objective is ``sum(rendered * inside_coefficients)``.  Its gradient
    with respect to the per-Gaussian channel equals the accumulated normalized
    alpha/transmittance mass for that mask.  ``visible_coefficient`` provides
    the corresponding all-valid-pixel objective for one channel.
    """

    mask_indices: tuple[int, ...]
    inside_coefficients: np.ndarray
    visible_coefficient: np.ndarray
    valid_pixels: np.ndarray

    @property
    def active_channels(self) -> int:
        return len(self.mask_indices)


def _image_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array


def _mask_stack(
    masks: np.ndarray | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, bool]:
    if masks is None:
        return np.zeros((0, *shape), dtype=bool), True
    array = np.asarray(masks, dtype=bool)
    if array.ndim != 3 or array.shape[1:] != shape:
        raise ValueError(f"masks must have shape Mx{shape[0]}x{shape[1]}")
    return array, False


def _validated_mass(
    source: str,
    inside_mass: np.ndarray,
    visible_mass: np.ndarray,
    valid_pixel_count: int,
    *,
    abstained: bool,
) -> AttributionMass:
    inside = np.asarray(inside_mass, dtype=np.float64)
    visible = np.asarray(visible_mass, dtype=np.float64)
    if visible.ndim != 1 or inside.ndim != 2 or inside.shape[1] != len(visible):
        raise ValueError("inside_mass must be MxN and visible_mass must be N")
    if not np.all(np.isfinite(inside)) or not np.all(np.isfinite(visible)):
        raise ValueError("attribution masses must be finite")
    if np.any(inside < -1e-7) or np.any(visible < -1e-7):
        raise ValueError("attribution masses must be non-negative")
    inside = np.maximum(inside, 0.0)
    visible = np.maximum(visible, 0.0)
    excess = inside - visible[None, :]
    if excess.size and float(np.max(excess)) > 1e-5:
        raise ValueError("inside mass cannot exceed visible mass")
    inside = np.minimum(inside, visible[None, :])
    return AttributionMass(
        source=str(source),
        inside_mass=inside,
        visible_mass=visible,
        valid_pixel_count=int(valid_pixel_count),
        abstained=bool(abstained),
    )


def mass_from_max_contributor(
    max_id: np.ndarray,
    max_weight: np.ndarray,
    masks: np.ndarray | None,
    point_count: int,
    *,
    valid_pixels: np.ndarray | None = None,
) -> AttributionMass:
    """Reduce corrected M1 images to per-Gaussian attribution mass.

    Only pixels with an in-range ID and finite positive weight contribute.
    ``masks=None`` means the detector/mask source abstained for this frame; it
    does not turn the visible mass into negative or background evidence.
    """
    ids = np.asarray(max_id)
    weights = np.asarray(max_weight, dtype=np.float64)
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("max_id and max_weight must be matching HxW images")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    mask_array, abstained = _mask_stack(masks, ids.shape)
    allowed = (
        np.ones(ids.shape, dtype=bool)
        if valid_pixels is None
        else _image_mask(valid_pixels, ids.shape, "valid_pixels")
    )
    flat_ids = ids.reshape(-1).astype(np.int64, copy=False)
    flat_weights = weights.reshape(-1)
    valid = (
        allowed.reshape(-1)
        & (flat_ids >= 0)
        & (flat_ids < int(point_count))
        & np.isfinite(flat_weights)
        & (flat_weights > 0)
    )
    # M1 is the one-hot member of the same per-pixel normalized attribution
    # family as AM: the corrected alpha*T weight chooses the winner and gates
    # empty pixels, then that pixel contributes unit mass to its winner.  Using
    # the raw winner weight here would confound lifting source with a different
    # mass scale while applying identical fragment thresholds.
    visible = np.bincount(flat_ids[valid], minlength=point_count).astype(
        np.float64, copy=False
    )
    inside = np.zeros((len(mask_array), point_count), dtype=np.float64)
    for mask_index, mask in enumerate(mask_array):
        selected = valid & mask.reshape(-1)
        inside[mask_index] = np.bincount(
            flat_ids[selected], minlength=point_count
        )
    return _validated_mass(
        "M1", inside, visible, int(np.count_nonzero(valid)), abstained=abstained
    )


def iter_three_channel_mask_batches(masks: np.ndarray) -> Iterator[AMChannelBatch]:
    """Pack mask targets into deterministic, zero-padded RGB batches."""
    array = np.asarray(masks, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("masks must be MxHxW")
    for start in range(0, len(array), 3):
        stop = min(start + 3, len(array))
        targets = np.zeros((3, *array.shape[1:]), dtype=np.float32)
        targets[: stop - start] = array[start:stop]
        yield AMChannelBatch(
            mask_indices=tuple(range(start, stop)),
            targets=targets,
        )


def build_am_objective_targets(
    batch: AMChannelBatch,
    pixel_opacity: np.ndarray,
    *,
    valid_pixels: np.ndarray | None = None,
    min_opacity: float = 1e-8,
) -> AMObjectiveTargets:
    """Build per-pixel coefficients for normalized alpha-mass gradients.

    The caller must detach these coefficients from autograd.  The renderer
    should use zero background so the point-feature gradient contains only
    Gaussian alpha/transmittance contributions.
    """
    targets = np.asarray(batch.targets, dtype=np.float64)
    if targets.ndim != 3 or targets.shape[0] != 3:
        raise ValueError("AM batch targets must be 3xHxW")
    opacity = np.asarray(pixel_opacity, dtype=np.float64)
    if opacity.shape != targets.shape[1:]:
        raise ValueError("pixel_opacity must match the batch image shape")
    allowed = (
        np.ones(opacity.shape, dtype=bool)
        if valid_pixels is None
        else _image_mask(valid_pixels, opacity.shape, "valid_pixels")
    )
    valid = allowed & np.isfinite(opacity) & (opacity > float(min_opacity))
    inverse_opacity = np.zeros_like(opacity)
    inverse_opacity[valid] = 1.0 / opacity[valid]
    inside = targets * inverse_opacity[None, :, :]
    inside[:, ~valid] = 0.0
    return AMObjectiveTargets(
        mask_indices=batch.mask_indices,
        inside_coefficients=inside,
        visible_coefficient=inverse_opacity,
        valid_pixels=valid,
    )


def _gradient_mass(gradient: np.ndarray) -> np.ndarray:
    values = np.asarray(gradient, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("point-feature gradient must be finite")
    if np.any(values < -1e-7):
        raise ValueError("point-feature gradient mass must be non-negative")
    return np.maximum(values, 0.0)


def attribution_from_am_gradients(
    visible_gradient: np.ndarray,
    inside_gradient_batches: Sequence[tuple[AMChannelBatch, np.ndarray]],
    mask_count: int,
    valid_pixel_count: int,
    *,
    abstained: bool = False,
) -> AttributionMass:
    """Assemble AM attribution from renderer point-feature gradients.

    ``visible_gradient`` may be ``N`` or ``NxC``; the first channel is used.
    Each inside gradient is ``Nx3`` and corresponds to its channel batch.
    """
    visible_array = _gradient_mass(visible_gradient)
    if visible_array.ndim == 2:
        if visible_array.shape[1] < 1:
            raise ValueError("visible gradient must have at least one channel")
        visible = visible_array[:, 0]
    elif visible_array.ndim == 1:
        visible = visible_array
    else:
        raise ValueError("visible gradient must be N or NxC")
    if mask_count < 0:
        raise ValueError("mask_count must be non-negative")
    inside = np.zeros((int(mask_count), len(visible)), dtype=np.float64)
    filled = np.zeros(int(mask_count), dtype=bool)
    for batch, gradient in inside_gradient_batches:
        values = _gradient_mass(gradient)
        if values.ndim != 2 or values.shape != (len(visible), 3):
            raise ValueError("inside point-feature gradients must be Nx3")
        for channel, mask_index in enumerate(batch.mask_indices):
            if not 0 <= mask_index < mask_count:
                raise ValueError(f"mask index {mask_index} is out of range")
            if filled[mask_index]:
                raise ValueError(f"mask index {mask_index} was supplied twice")
            inside[mask_index] = values[:, channel]
            filled[mask_index] = True
    if mask_count and not np.all(filled):
        missing = np.flatnonzero(~filled).tolist()
        raise ValueError(f"missing AM gradients for mask indices {missing}")
    return _validated_mass(
        "AM",
        inside,
        visible,
        valid_pixel_count,
        abstained=abstained,
    )


def fragments_from_attribution(
    attribution: AttributionMass,
    frame_id: int,
    *,
    fragment_id_start: int = 0,
    config: V8FragmentConfig = V8FragmentConfig(),
) -> tuple[AttributionFragment, ...]:
    """Apply the shared V8 full/core thresholds to either lifting source."""
    if attribution.abstained:
        return ()
    fragments: list[AttributionFragment] = []
    next_id = int(fragment_id_start)
    for mask_index, inside in enumerate(attribution.inside_mass):
        full_ids = np.flatnonzero(
            inside >= float(config.full_min_inside_mass)
        ).astype(np.int32)
        if len(full_ids) < int(config.fragment_min_full):
            continue
        ratios = np.divide(
            inside,
            attribution.visible_mass,
            out=np.zeros_like(inside),
            where=attribution.visible_mass > 0,
        )
        core_ids = np.flatnonzero(
            (inside >= float(config.core_min_inside_mass))
            & (ratios >= float(config.core_min_inside_ratio))
        ).astype(np.int32)
        if len(core_ids) < int(config.fragment_min_core):
            continue
        fragments.append(
            AttributionFragment(
                fragment_id=next_id,
                frame_id=int(frame_id),
                mask_index=int(mask_index),
                full_ids=full_ids,
                core_ids=core_ids,
                full_inside_mass=inside[full_ids].astype(np.float32),
                core_inside_mass=inside[core_ids].astype(np.float32),
                core_inside_ratio=ratios[core_ids].astype(np.float32),
            )
        )
        next_id += 1
    return tuple(fragments)


def lift_max_contributor(
    max_id: np.ndarray,
    max_weight: np.ndarray,
    masks: np.ndarray | None,
    point_count: int,
    frame_id: int,
    *,
    valid_pixels: np.ndarray | None = None,
    fragment_id_start: int = 0,
    config: V8FragmentConfig = V8FragmentConfig(),
) -> LiftedAttributionFrame:
    """Convenience wrapper for the complete pure M1 lifting path."""
    attribution = mass_from_max_contributor(
        max_id,
        max_weight,
        masks,
        point_count,
        valid_pixels=valid_pixels,
    )
    return LiftedAttributionFrame(
        frame_id=int(frame_id),
        source=attribution.source,
        fragments=fragments_from_attribution(
            attribution,
            frame_id,
            fragment_id_start=fragment_id_start,
            config=config,
        ),
        visible_ids=np.flatnonzero(attribution.visible_mass > 0).astype(np.int32),
        valid_pixel_count=attribution.valid_pixel_count,
        abstained=attribution.abstained,
    )
