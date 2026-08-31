"""Metric size vetoes for clean-baseline mask consensus.

Runtime modes accept either no prior, the train-global upper envelope, or a
soft predicted-class mixture.  The GT/oracle helper is deliberately separate
and cannot be selected through :func:`make_size_merge_veto`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from ..priors import validate_priors


RuntimePriorMode = Literal["none", "global", "predicted"]


def _triplet(values: object, *, name: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True)
class SizePriorTable:
    """Upper log-extent envelopes fitted on ScanNet-train only."""

    global_log_q95: tuple[float, float, float]
    class_log_q95: Mapping[str, tuple[float, float, float]]

    def __post_init__(self) -> None:
        global_values = _triplet(self.global_log_q95, name="global_log_q95")
        classes: dict[str, tuple[float, float, float]] = {}
        for class_name, values in self.class_log_q95.items():
            if not isinstance(class_name, str) or not class_name:
                raise ValueError("class prior names must be non-empty strings")
            classes[class_name] = _triplet(
                values, name=f"class_log_q95[{class_name!r}]"
            )
        object.__setattr__(self, "global_log_q95", global_values)
        object.__setattr__(self, "class_log_q95", classes)

    @classmethod
    def from_category_priors(cls, payload: Mapping[str, object]) -> "SizePriorTable":
        """Read q95 values from the repository's train-only priors schema."""

        # This is a runtime data boundary, not a permissive parser.  In
        # particular, a hand-written JSON file with plausible q95 values must
        # never be accepted as a substitute for the frozen ScanNet-train
        # artifact.
        validate_priors(payload)

        fields = (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        )

        def read_geometry(node: object, *, label: str) -> tuple[float, float, float]:
            if not isinstance(node, Mapping):
                raise ValueError(f"{label} prior node must be a mapping")
            geometry = node.get("geometry")
            if not isinstance(geometry, Mapping):
                raise ValueError(f"{label} prior is missing geometry")
            values: list[float] = []
            for field in fields:
                summary = geometry.get(field)
                if not isinstance(summary, Mapping) or "q95" not in summary:
                    raise ValueError(f"{label} prior is missing {field}.q95")
                try:
                    value = float(summary["q95"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{label} {field}.q95 must be numeric"
                    ) from exc
                if not np.isfinite(value):
                    raise ValueError(f"{label} {field}.q95 must be finite")
                values.append(value)
            return tuple(values)  # type: ignore[return-value]

        global_node = payload.get("global")
        if not isinstance(global_node, Mapping):
            raise ValueError("global prior node must be a mapping")
        global_shrunk = global_node.get("shrunk")
        if global_shrunk is None:
            raise ValueError("global prior is missing shrunk statistics")
        global_values = read_geometry(global_shrunk, label="global.shrunk")
        categories = payload.get("categories", {})
        if not isinstance(categories, Mapping):
            raise ValueError("categories must be a mapping")
        class_values: dict[str, tuple[float, float, float]] = {}
        for class_name, category_node in categories.items():
            if not isinstance(class_name, str) or not isinstance(category_node, Mapping):
                continue
            shrunk = category_node.get("shrunk")
            if shrunk is None:
                continue
            class_values[class_name] = read_geometry(
                shrunk, label=f"category {class_name!r}"
            )
        return cls(global_values, class_values)


def pca_sorted_extents_m(points_m: object) -> np.ndarray:
    """Return short/mid/long PCA extents in meters.

    The calculation is translation- and rotation-invariant.  Degenerate point
    sets are valid and produce zero extents along unsupported axes.
    """

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("points_m must have shape [points, 3]")
    if not np.isfinite(points).all():
        raise ValueError("points_m must be finite")
    if len(points) == 0:
        raise ValueError("at least one point is required")
    centered = points - points.mean(axis=0, keepdims=True)
    if len(points) == 1:
        result = np.zeros(3, dtype=np.float64)
    else:
        _, _, axes = np.linalg.svd(centered, full_matrices=True)
        projected = centered @ axes.T
        result = np.sort(np.ptp(projected, axis=0))
    result = np.asarray(result, dtype=np.float64)
    result.setflags(write=False)
    return result


def extent_log_m(extents_m: object) -> np.ndarray:
    extents = np.asarray(extents_m, dtype=np.float64)
    if extents.shape != (3,) or not np.isfinite(extents).all() or np.any(extents < 0.0):
        raise ValueError("extents_m must contain three finite non-negative values")
    values = np.log(np.maximum(np.sort(extents), 1e-9))
    values.setflags(write=False)
    return values


def extent_within_q95(extents_m: object, log_q95: object) -> bool:
    """Return whether all sorted metric extents are under the q95 envelope."""

    logs = extent_log_m(extents_m)
    upper = np.asarray(_triplet(log_q95, name="log_q95"), dtype=np.float64)
    return bool(np.all(logs <= upper))


def global_size_compatibility(
    extents_m: object, priors: SizePriorTable
) -> float:
    """Binary global size compatibility used by ``U-global``."""

    return float(extent_within_q95(extents_m, priors.global_log_q95))


def _normalised_probabilities(
    class_probabilities: Mapping[str, float],
) -> dict[str, float]:
    if not class_probabilities:
        raise ValueError("predicted mode requires class probabilities")
    clean: dict[str, float] = {}
    for class_name, probability in class_probabilities.items():
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("class probability names must be non-empty strings")
        value = float(probability)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("class probabilities must be finite and non-negative")
        clean[class_name] = clean.get(class_name, 0.0) + value
    total = float(sum(clean.values()))
    if total <= 0.0:
        raise ValueError("class probabilities must have positive mass")
    return {name: value / total for name, value in clean.items()}


def predicted_size_compatibility(
    extents_m: object,
    priors: SizePriorTable,
    class_probabilities: Mapping[str, float],
) -> float:
    """Posterior-weighted q95 compatibility used by ``D-predicted``.

    Missing class statistics use the global envelope, as required by the
    frozen plan.  The returned value is always in ``[0, 1]``.
    """

    probabilities = _normalised_probabilities(class_probabilities)
    value = 0.0
    for class_name, probability in probabilities.items():
        upper = priors.class_log_q95.get(class_name, priors.global_log_q95)
        value += probability * float(extent_within_q95(extents_m, upper))
    return float(value)


def oracle_class_size_compatibility(
    extents_m: object,
    priors: SizePriorTable,
    oracle_class: str,
) -> float:
    """Evaluation-only class oracle, intentionally absent from runtime modes."""

    if not isinstance(oracle_class, str) or not oracle_class:
        raise ValueError("oracle_class must be a non-empty string")
    upper = priors.class_log_q95.get(oracle_class, priors.global_log_q95)
    return float(extent_within_q95(extents_m, upper))


def _average_mask_probabilities(
    mask_ids: Sequence[int],
    mask_class_probabilities: Mapping[int, Mapping[str, float]],
) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    for mask_id in mask_ids:
        probabilities = mask_class_probabilities.get(int(mask_id))
        if probabilities is None:
            continue
        normalised = _normalised_probabilities(probabilities)
        for class_name, value in normalised.items():
            sums[class_name] = sums.get(class_name, 0.0) + value
        count += 1
    if count == 0:
        return {"__global_fallback__": 1.0}
    return {class_name: value / count for class_name, value in sums.items()}


def make_size_merge_veto(
    mode: RuntimePriorMode,
    xyz_m: object,
    priors: SizePriorTable,
    mask_class_probabilities: Mapping[int, Mapping[str, float]] | None = None,
    *,
    acceptance_threshold: float = 0.50,
) -> Callable[[tuple[int, ...], np.ndarray], bool]:
    """Build a runtime-only C0/U/D merge veto.

    ``oracle`` is not a valid mode.  The predicted veto combines per-mask soft
    32-class posteriors only after a tentative geometric component exists.
    """

    if mode not in ("none", "global", "predicted"):
        raise ValueError("mode must be 'none', 'global', or 'predicted'")
    points = np.asarray(xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("xyz_m must be a finite [gaussians, 3] array")
    if not np.isfinite(acceptance_threshold) or not 0.0 <= acceptance_threshold <= 1.0:
        raise ValueError("acceptance_threshold must be in [0, 1]")
    if mode == "predicted" and mask_class_probabilities is None:
        raise ValueError("predicted mode requires mask_class_probabilities")

    def veto(mask_ids: tuple[int, ...], gaussian_ids: np.ndarray) -> bool:
        ids = np.asarray(gaussian_ids, dtype=np.int64)
        if ids.ndim != 1 or ids.size == 0:
            return False
        if np.any(ids < 0) or int(ids.max()) >= len(points):
            raise ValueError("merge veto received an unknown Gaussian ID")
        extents = pca_sorted_extents_m(points[ids])
        if mode == "none":
            compatibility = 1.0
        elif mode == "global":
            compatibility = global_size_compatibility(extents, priors)
        else:
            assert mask_class_probabilities is not None
            probabilities = _average_mask_probabilities(
                mask_ids, mask_class_probabilities
            )
            compatibility = predicted_size_compatibility(
                extents, priors, probabilities
            )
        return bool(compatibility >= acceptance_threshold)

    return veto


def base_ap_score(
    winner_probability: float,
    view_consensus: float,
    detection_ratio: float,
) -> float:
    """Official-AP ordering score, intentionally independent of any prior."""

    values = np.asarray(
        [winner_probability, view_consensus, detection_ratio], dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("score inputs must be finite values in [0, 1]")
    return float(values[0] * np.sqrt(values[1] * values[2]))


__all__ = [
    "RuntimePriorMode",
    "SizePriorTable",
    "base_ap_score",
    "extent_log_m",
    "extent_within_q95",
    "global_size_compatibility",
    "make_size_merge_veto",
    "oracle_class_size_compatibility",
    "pca_sorted_extents_m",
    "predicted_size_compatibility",
]
