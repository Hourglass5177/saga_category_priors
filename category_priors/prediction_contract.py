from __future__ import annotations

"""Strict, write-time prediction contract shared by active SAGA pipelines."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionContractResult:
    point_labels: np.ndarray
    instances: dict[str, dict[str, Any]]
    audit: dict[str, Any]


def remap_instance_metadata_to_export(
    raw_instances: Mapping[str | int, Mapping[str, Any]],
    export_id_by_raw: Mapping[int, int],
    contracted: PredictionContractResult,
) -> dict[str, dict[str, Any]]:
    """Project auxiliary metadata into the sole exported instance-ID space."""

    result: dict[str, dict[str, Any]] = {}
    labels = np.asarray(contracted.point_labels, dtype=np.int64)
    for raw_id, export_id in sorted(
        ((int(raw), int(export)) for raw, export in export_id_by_raw.items()),
        key=lambda item: item[1],
    ):
        values = raw_instances.get(str(raw_id), raw_instances.get(raw_id))
        exported = contracted.instances.get(str(export_id))
        if not isinstance(values, Mapping) or not isinstance(exported, Mapping):
            raise ValueError("raw/export instance metadata mapping is incomplete")
        normalized = dict(values)
        normalized.update(dict(exported))
        normalized["point_count"] = int(np.count_nonzero(labels == export_id))
        result[str(export_id)] = normalized
    if set(result) != set(contracted.instances):
        raise ValueError("metadata export IDs do not match the prediction contract")
    return result


def normalize_score(value: Any, *, context: str = "score") -> float:
    """Return the one canonical representation of an instance score.

    The prediction contract deliberately does not clip scores: clipping would
    silently turn a broken scoring pipeline into an apparently valid one.
    """

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{context} must be numeric")
    score = float(value)
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{context} must be finite and in [0, 1]")
    return score


def validate_prediction_contract(
    point_labels: Sequence[int] | np.ndarray,
    instances: Mapping[str | int, Mapping[str, Any]],
) -> None:
    """Validate the sole exported truth without changing it."""

    raw = np.asarray(point_labels)
    if raw.ndim != 1:
        raise ValueError("point_labels must be one-dimensional")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype, np.integer
    ):
        raise TypeError("point_labels must use an integer dtype")
    labels = raw.astype(np.int64, copy=False)
    if not np.array_equal(raw, labels) or np.any(labels < -1):
        raise TypeError("point_labels must contain integers >= -1")
    if not isinstance(instances, Mapping):
        raise TypeError("instances must be a mapping")

    declared: list[int] = []
    for raw_id in instances:
        if isinstance(raw_id, bool):
            raise TypeError("instance IDs must be integers")
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid instance ID: {raw_id!r}") from exc
        if str(instance_id) != str(raw_id).strip() or instance_id < 0:
            raise ValueError("instance IDs must be canonical non-negative integers")
        declared.append(instance_id)
    declared_tuple = tuple(sorted(declared))
    if declared_tuple != tuple(range(len(declared_tuple))):
        raise ValueError("final instance IDs must be contiguous from zero")
    present = tuple(int(value) for value in np.unique(labels[labels >= 0]))
    if present != declared_tuple:
        raise ValueError("non-negative labels and declared instances must agree exactly")

    for instance_id in declared_tuple:
        metadata = instances.get(str(instance_id), instances.get(instance_id))
        if not isinstance(metadata, Mapping):
            raise TypeError(f"instance {instance_id} metadata must be a mapping")
        class_name = metadata.get("class")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError(f"instance {instance_id} must declare a class")
        normalize_score(metadata.get("score"), context=f"instance {instance_id} score")
        if not np.any(labels == instance_id):
            raise ValueError(f"instance {instance_id} has an empty mask")


def normalize_prediction(
    point_labels: Sequence[int] | np.ndarray,
    instances: Mapping[str | int, Mapping[str, Any]],
) -> PredictionContractResult:
    """Return a contiguous, declared-only prediction without mutating inputs.

    Negative metadata and empty instances are removed.  Every undeclared
    non-negative point label becomes background.  Remaining masks are sorted by
    their old numeric ID and reindexed to ``0..K-1``.
    """

    raw = np.asarray(point_labels)
    if raw.ndim != 1:
        raise ValueError("point_labels must be one-dimensional")
    try:
        labels = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("point_labels must contain integers") from exc
    if not np.array_equal(raw, labels):
        raise TypeError("point_labels must contain integers")
    if not isinstance(instances, Mapping):
        raise TypeError("instances must be a mapping")

    valid: list[tuple[int, dict[str, Any], np.ndarray]] = []
    ignored_negative: list[int] = []
    empty: list[int] = []
    for raw_id, raw_metadata in instances.items():
        if isinstance(raw_id, bool):
            raise TypeError("instance IDs must be integers")
        try:
            instance_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"invalid instance ID: {raw_id!r}") from exc
        if str(instance_id) != str(raw_id).strip():
            raise TypeError(f"non-canonical instance ID: {raw_id!r}")
        if instance_id < 0:
            ignored_negative.append(instance_id)
            continue
        if not isinstance(raw_metadata, Mapping):
            raise TypeError(f"instance {instance_id} metadata must be a mapping")
        class_name = raw_metadata.get("class")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(f"instance {instance_id} must declare a class")
        score_value = normalize_score(
            raw_metadata.get("score"), context=f"instance {instance_id} score"
        )
        mask = labels == instance_id
        if not bool(mask.any()):
            empty.append(instance_id)
            continue
        metadata = dict(raw_metadata)
        metadata["score"] = score_value
        valid.append((instance_id, metadata, mask))

    valid.sort(key=lambda row: row[0])
    output_labels = np.full(labels.shape, -1, dtype=np.int64)
    output_instances: dict[str, dict[str, Any]] = {}
    declared_old_ids: list[int] = []
    for new_id, (old_id, metadata, mask) in enumerate(valid):
        output_labels[mask] = new_id
        output_instances[str(new_id)] = metadata
        declared_old_ids.append(old_id)

    raw_nonnegative = labels >= 0
    retained = output_labels >= 0
    orphan_mask = raw_nonnegative & ~np.isin(labels, declared_old_ids)
    orphan_ids, orphan_counts = np.unique(labels[orphan_mask], return_counts=True)
    validate_prediction_contract(output_labels, output_instances)
    output_labels.setflags(write=False)
    return PredictionContractResult(
        point_labels=output_labels,
        instances=output_instances,
        audit={
            "schema": "saga-strict-prediction-contract-v1",
            "point_count": int(len(labels)),
            "instance_count": len(output_instances),
            "declared_gaussian_count": int(np.count_nonzero(retained)),
            "background_gaussian_count": int(np.count_nonzero(~retained)),
            "ignored_negative_metadata_ids": sorted(ignored_negative),
            "empty_declared_instance_ids": sorted(empty),
            "orphan_instance_ids": [int(value) for value in orphan_ids],
            "orphan_counts": {
                str(int(instance_id)): int(count)
                for instance_id, count in zip(orphan_ids, orphan_counts)
            },
            "orphan_gaussian_count": int(np.count_nonzero(orphan_mask)),
        },
    )
