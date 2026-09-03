from __future__ import annotations

"""Shared final semantic vote and strict export for teacher-compatible runs.

The legacy replay deliberately ends after KNN/filter.  This module owns the
single path from those raw labels to an exported prediction so the baseline and
future recheck conditions cannot drift in class selection, score, bounding box,
or instance reindexing.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .prediction_contract import PredictionContractResult, normalize_prediction


@dataclass(frozen=True)
class FinalizedPrediction:
    contracted: PredictionContractResult
    raw_instances: Mapping[int, Mapping[str, Any]]
    class_by_raw: Mapping[int, str]
    score_by_raw: Mapping[int, float]
    bbox_by_raw: Mapping[int, tuple[float, ...]]


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    converted = value
    if hasattr(converted, "detach"):
        converted = converted.detach()
    if hasattr(converted, "cpu"):
        converted = converted.cpu()
    if hasattr(converted, "numpy"):
        converted = converted.numpy()
    return np.asarray(converted, dtype=dtype)


def oriented_bbox_by_raw(
    point_labels: Any,
    xyz_scene: Any,
    is_big_gaussian: Any,
) -> dict[int, tuple[float, ...]]:
    """Preserve the handoff's yaw-oriented 3D bounding-box calculation."""

    from trimesh.bounds import oriented_bounds_2D

    labels = _as_numpy(point_labels, np.int64)
    # Preserve the source floating dtype.  The historical path feeds float32
    # Gaussian XYZ into trimesh; an eager float64 cast can create needless bbox
    # drift even though point membership is unchanged.
    xyz = _as_numpy(xyz_scene)
    is_big = _as_numpy(is_big_gaussian, bool)
    if labels.ndim != 1 or xyz.shape != (len(labels), 3) or is_big.shape != labels.shape:
        raise ValueError("labels, XYZ and large-Gaussian mask must share the point axis")
    if not np.isfinite(xyz).all():
        raise ValueError("Gaussian XYZ must be finite")

    result: dict[int, tuple[float, ...]] = {}
    for raw_id in sorted(int(value) for value in np.unique(labels) if int(value) >= 0):
        points = xyz[(labels == raw_id) & ~is_big]
        if len(points) == 0:
            result[raw_id] = (0.0,) * 24
            continue
        if len(points) < 3:
            minimum = points.min(axis=0)
            maximum = points.max(axis=0)
            corners = np.asarray(
                [
                    [maximum[0], maximum[1], maximum[2]],
                    [maximum[0], maximum[1], minimum[2]],
                    [maximum[0], minimum[1], minimum[2]],
                    [maximum[0], minimum[1], maximum[2]],
                    [minimum[0], maximum[1], maximum[2]],
                    [minimum[0], maximum[1], minimum[2]],
                    [minimum[0], minimum[1], minimum[2]],
                    [minimum[0], minimum[1], maximum[2]],
                ],
                dtype=np.float64,
            )
            result[raw_id] = tuple(float(value) for value in corners.reshape(-1))
            continue

        transform_2d, extents_2d = oriented_bounds_2D(points[:, [0, 2]])
        transform_3d = np.eye(4)
        transform_3d[0, 0] = transform_2d[0, 0]
        transform_3d[0, 2] = transform_2d[0, 1]
        transform_3d[0, 3] = transform_2d[0, 2]
        transform_3d[2, 0] = transform_2d[1, 0]
        transform_3d[2, 2] = transform_2d[1, 1]
        transform_3d[2, 3] = transform_2d[1, 2]
        homogeneous = np.column_stack((points, np.ones(len(points))))
        transformed = (transform_3d @ homogeneous.T).T[:, :3]
        ymin = float(transformed[:, 1].min())
        ymax = float(transformed[:, 1].max())
        half_x, half_z = np.asarray(extents_2d) / 2.0
        local = np.asarray(
            [
                [half_x, ymax, half_z],
                [half_x, ymax, -half_z],
                [half_x, ymin, -half_z],
                [half_x, ymin, half_z],
                [-half_x, ymax, half_z],
                [-half_x, ymax, -half_z],
                [-half_x, ymin, -half_z],
                [-half_x, ymin, half_z],
            ]
        )
        local_h = np.column_stack((local, np.ones(8)))
        world = (np.linalg.inv(transform_3d) @ local_h.T).T[:, :3]
        result[raw_id] = tuple(float(value) for value in world.reshape(-1))
    return result


def _vote_row(
    vote_ratios_by_raw: Mapping[int | str, Sequence[float] | np.ndarray],
    raw_id: int,
    class_count: int,
) -> np.ndarray:
    raw = vote_ratios_by_raw.get(raw_id, vote_ratios_by_raw.get(str(raw_id)))
    if raw is None:
        raise ValueError(f"missing final vote ratios for raw instance {raw_id}")
    ratio = _as_numpy(raw, np.float64)
    if ratio.shape != (class_count,) or not np.isfinite(ratio).all():
        raise ValueError(f"raw instance {raw_id} has invalid final vote ratios")
    if np.any(ratio < 0) or float(ratio.sum()) > 1.0 + 1e-6:
        raise ValueError(f"raw instance {raw_id} final vote ratios are not probabilities")
    return ratio


def finalize_prediction(
    *,
    point_labels: Any,
    xyz_scene: Any,
    is_big_gaussian: Any,
    vote_ratios_by_raw: Mapping[int | str, Sequence[float] | np.ndarray],
    class_names: Sequence[str],
    selected_classes: Sequence[str],
    label_threshold: float,
) -> FinalizedPrediction:
    """Apply the one final class/score/bbox/export path to raw labels."""

    labels = _as_numpy(point_labels, np.int64)
    classes = tuple(str(value) for value in class_names)
    selected = frozenset(str(value) for value in selected_classes)
    threshold = float(label_threshold)
    if labels.ndim != 1 or np.any(labels < -1):
        raise ValueError("point_labels must be one-dimensional and use only -1 as background")
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("class_names must be non-empty and unique")
    if not selected.issubset(classes):
        raise ValueError("selected_classes must be a subset of class_names")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("label_threshold must be in [0, 1]")

    bbox = oriented_bbox_by_raw(labels, xyz_scene, is_big_gaussian)
    raw_instances: dict[int, dict[str, Any]] = {}
    class_by_raw: dict[int, str] = {}
    score_by_raw: dict[int, float] = {}
    for raw_id in bbox:
        ratio = _vote_row(vote_ratios_by_raw, raw_id, len(classes))
        score = float(ratio.max()) if ratio.size else 0.0
        class_name = classes[int(ratio.argmax())] if score >= threshold else "background"
        class_by_raw[raw_id] = class_name
        score_by_raw[raw_id] = score
        if class_name in selected:
            raw_instances[raw_id] = {
                "bbox": list(bbox[raw_id]),
                "class": class_name,
                "score": score,
            }

    contracted = normalize_prediction(labels, raw_instances)
    return FinalizedPrediction(
        contracted=contracted,
        raw_instances=MappingProxyType(raw_instances),
        class_by_raw=MappingProxyType(class_by_raw),
        score_by_raw=MappingProxyType(score_by_raw),
        bbox_by_raw=MappingProxyType(bbox),
    )


def prediction_output_payload(
    finalized: FinalizedPrediction,
    *,
    is_big_gaussian: Any,
    is_transparent_gaussian: Any,
) -> dict[str, Any]:
    """Build the teacher-compatible JSON payload from a finalized prediction."""

    big = _as_numpy(is_big_gaussian, bool)
    transparent = _as_numpy(is_transparent_gaussian, bool)
    labels = finalized.contracted.point_labels
    if big.shape != labels.shape or transparent.shape != labels.shape:
        raise ValueError("Gaussian diagnostic masks must share the exported point axis")
    return {
        "point_labels": labels.tolist(),
        "is_big_gaussian": big.tolist(),
        "is_transparent_gaissian": transparent.tolist(),
        "instances": finalized.contracted.instances,
        "prediction_contract": finalized.contracted.audit,
    }


def write_prediction_output_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_text(json.dumps(dict(payload)), encoding="utf-8")
    os.replace(temporary, destination)


__all__ = [
    "FinalizedPrediction",
    "finalize_prediction",
    "oriented_bbox_by_raw",
    "prediction_output_payload",
    "write_prediction_output_atomic",
]
