"""Minimal prompt-style category prior primitives.

This module intentionally contains no SAGA runtime, CUDA, evaluator, or I/O
code.  It implements the smallest comparison needed for a prompt-conditioned
experiment:

* materialize a global and per-class typical physical scale from train priors;
* convert that physical scale to the scene-local scale-gate input with the
  training-time quantile transform; and
* apply the native gate -> L2 normalization -> cosine threshold operation.

Uniform and class-conditioned runs therefore execute the same code path.  The
only possible difference is the typical diagonal selected from the prior table.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _positive_finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _typical_diag_from_node(node: Mapping[str, Any], *, name: str) -> float:
    try:
        log_diag = node["shrunk"]["geometry"]["log_bbox_diag_m"]["q50"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{name} is missing shrunk log_bbox_diag_m.q50") from exc
    try:
        typical_diag = math.exp(float(log_diag))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} has an invalid log_bbox_diag_m.q50") from exc
    return _positive_finite(typical_diag, name=f"{name} typical diagonal")


@dataclass(frozen=True)
class PromptPriorTable:
    """Physical scales used by the uniform and class-conditioned arms."""

    global_typical_diag_m: float
    class_typical_diag_m: Mapping[str, float]

    def typical_diag_m(self, class_name: str | None, *, mode: str) -> float:
        """Return the arm's typical diagonal, falling back to global.

        ``uniform`` always uses the global value.  ``class`` uses a materialized
        class value when available; an unknown or missing class uses exactly the
        same global value as ``uniform``.
        """

        if mode == "uniform":
            return self.global_typical_diag_m
        if mode != "class":
            raise ValueError("mode must be 'uniform' or 'class'")
        key = str(class_name).strip().lower() if class_name is not None else ""
        return self.class_typical_diag_m.get(key, self.global_typical_diag_m)


def materialize_prompt_priors(priors: Mapping[str, Any]) -> PromptPriorTable:
    """Materialize typical diagonals from a train-only category-prior payload.

    The stored values are logarithms because the prior fitter operates in log
    space.  This function exponentiates the shrunk median once.  Categories
    without a complete shrunk value are deliberately omitted and consequently
    use the global fallback at inference time.
    """

    splits = priors.get("provenance", {}).get("splits")
    if splits != ["train"]:
        raise ValueError("prompt priors must contain exactly the train split")

    global_node = priors.get("global")
    if not isinstance(global_node, Mapping):
        raise ValueError("category priors are missing the global node")
    global_diag = _typical_diag_from_node(global_node, name="global prior")

    categories = priors.get("categories", {})
    if not isinstance(categories, Mapping):
        raise ValueError("category priors categories must be a mapping")

    materialized: dict[str, float] = {}
    for raw_name, raw_node in categories.items():
        if not isinstance(raw_node, Mapping):
            continue
        name = str(raw_name).strip().lower()
        if not name:
            continue
        try:
            materialized[name] = _typical_diag_from_node(
                raw_node, name=f"category prior {name!r}"
            )
        except ValueError:
            # Inactive/unsupported categories in the v1 prior schema have no
            # shrunk node.  They are intentionally handled by global fallback.
            continue

    return PromptPriorTable(
        global_typical_diag_m=global_diag,
        class_typical_diag_m=MappingProxyType(materialized),
    )


def materialize_prompt_prior(priors: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, plain-dict representation of prompt scale priors."""

    table = materialize_prompt_priors(priors)
    return {
        "global_typical_diag_m": table.global_typical_diag_m,
        "class_typical_diag_m": dict(table.class_typical_diag_m),
    }


def scene_mask_scale_ecdf(mask_scales_m: ArrayLike, target_scale_m: float) -> float:
    """Map a physical scale to the scene-local uniform scale-gate input.

    This is the right-continuous empirical CDF ``P(mask_scale <= target)``.
    It mirrors the uniform quantile normalization used when training SAGA's
    adaptive scale gate, without depending on scikit-learn or fitted runtime
    state.  Non-finite observed scales are ignored; negative scales are invalid.
    """

    target = _positive_finite(target_scale_m, name="target scale")
    observed = np.asarray(mask_scales_m, dtype=np.float64).reshape(-1)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("mask_scales_m must contain at least one finite value")
    if np.any(observed < 0.0):
        raise ValueError("mask scales cannot be negative")
    ordered = np.sort(observed, kind="mergesort")
    return float(np.searchsorted(ordered, target, side="right") / ordered.size)


def empirical_cdf(values: ArrayLike, query: float) -> float:
    """Stable public name for the scene mask-scale empirical CDF."""

    return scene_mask_scale_ecdf(values, query)


def training_quantile_uniform(values: ArrayLike, query: float) -> float:
    """Reconstruct SAGA's one-dimensional training quantile transform.

    ``train_contrastive_feature.get_quantile_func`` uses scikit-learn's
    ``QuantileTransformer(n_quantiles=1000, subsample=10000,
    output_distribution='uniform')``.  The registered prompt experiment has
    fewer than 1,000 mask scales per scene, so there is no subsampling and the
    transform is exactly the linear interpolation implemented here.  Larger
    inputs are rejected because the original training did not save the random
    subsample and an "exact" reconstruction would then be unknowable.
    """

    target = _positive_finite(query, name="query scale")
    observed = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("values must contain at least one finite scale")
    if np.any(observed < 0.0):
        raise ValueError("mask scales cannot be negative")
    if observed.size > 10_000:
        raise ValueError(
            "the original QuantileTransformer subsampled this scene; "
            "its exact training transform cannot be reconstructed"
        )
    quantile_count = min(1000, int(observed.size))
    references = np.linspace(0.0, 1.0, quantile_count)
    quantiles = np.quantile(observed, references, method="linear")
    forward = np.interp(target, quantiles, references)
    reverse = np.interp(-target, -quantiles[::-1], -references[::-1])
    return float(np.clip(0.5 * (forward - reverse), 0.0, 1.0))


def l2_normalize(rows: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """L2-normalize finite arrays while leaving zero rows as zero."""

    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("features must be finite")
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)


def gated_prompt_cosine(
    point_features: np.ndarray,
    rendered_query_feature: np.ndarray,
    gate_vector: ArrayLike,
) -> np.ndarray:
    """Apply the native gate and return point-to-query cosine similarities.

    ``rendered_query_feature`` is the raw feature rendered with
    ``norm_point_features=True``.  The same gate is applied exactly once to
    that query and exactly once to every raw 3D point feature, matching the
    public prompt notebook.  Both sides are then L2-normalized before the dot
    product.  A zero point feature gets cosine zero and a zero gated query is
    invalid.
    """

    points = np.asarray(point_features, dtype=np.float64)
    query = np.asarray(rendered_query_feature, dtype=np.float64)
    gate = np.asarray(gate_vector, dtype=np.float64).reshape(-1)
    if points.ndim != 2:
        raise ValueError("point_features must have shape (N, D)")
    if query.ndim != 1:
        raise ValueError("rendered_query_feature must have shape (D,)")
    if points.shape[1] != query.shape[0] or query.shape[0] != gate.shape[0]:
        raise ValueError("point, query, and gate feature dimensions must match")
    if not np.all(np.isfinite(gate)) or np.any(gate < 0.0):
        raise ValueError("gate_vector must contain finite non-negative values")

    gated_points = l2_normalize(points * gate[None, :])
    normalized_query = l2_normalize(query * gate)
    if not np.any(normalized_query):
        raise ValueError("the rendered query feature must be non-zero")
    return gated_points @ normalized_query


def gated_prompt_mask(
    point_features: np.ndarray,
    rendered_query_feature: np.ndarray,
    gate_vector: ArrayLike,
    threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, similarities)`` for the native prompt operation."""

    cutoff = float(threshold)
    if not math.isfinite(cutoff) or not -1.0 <= cutoff <= 1.0:
        raise ValueError("threshold must be finite and within [-1, 1]")
    similarities = gated_prompt_cosine(
        point_features, rendered_query_feature, gate_vector
    )
    return similarities > cutoff, similarities


@dataclass(frozen=True)
class PromptSelection:
    """Result of one uniform or class-conditioned prompt selection."""

    typical_diag_m: float
    gate_input: float
    gate_vector: np.ndarray
    cosine: np.ndarray
    selected: np.ndarray


def select_prompt_with_prior(
    point_features: np.ndarray,
    prompt_feature: np.ndarray,
    *,
    priors: PromptPriorTable,
    class_name: str | None,
    mode: str,
    scene_mask_scales_m: ArrayLike,
    gate_from_scale: Callable[[float], ArrayLike],
    cosine_threshold: float = 0.75,
) -> PromptSelection:
    """Run the shared U/D prompt-selection path.

    ``gate_from_scale`` injects the already-trained native SAGA scale gate.  It
    receives one scene-normalized scalar in ``[0, 1]`` and returns one gate
    vector.  The selection threshold is strict, matching ``cosine > 0.75``.
    """

    threshold = float(cosine_threshold)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("cosine_threshold must be finite and within [-1, 1]")
    typical_diag = priors.typical_diag_m(class_name, mode=mode)
    gate_input = training_quantile_uniform(scene_mask_scales_m, typical_diag)
    gate = np.asarray(gate_from_scale(gate_input), dtype=np.float64).reshape(-1)
    selected, cosine = gated_prompt_mask(
        point_features, prompt_feature, gate, threshold=threshold
    )
    return PromptSelection(
        typical_diag_m=typical_diag,
        gate_input=gate_input,
        gate_vector=gate.copy(),
        cosine=cosine,
        selected=selected,
    )


__all__ = [
    "PromptPriorTable",
    "PromptSelection",
    "empirical_cdf",
    "gated_prompt_mask",
    "gated_prompt_cosine",
    "l2_normalize",
    "materialize_prompt_prior",
    "materialize_prompt_priors",
    "scene_mask_scale_ecdf",
    "training_quantile_uniform",
    "select_prompt_with_prior",
]
