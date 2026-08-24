from __future__ import annotations

"""Read-only projection from raw cluster labels to exported instances.

Some historical post-processing outputs retain non-negative cluster IDs in
``point_labels`` after those clusters have been removed from ``instances``.
Those IDs are not exported predictions.  This module makes the existing
evaluation semantics explicit: labels declared by ``instances`` are retained,
undeclared non-negative labels are projected to background, and the raw payload
is never mutated.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeclaredInstanceProjection:
    """Projected labels and an audit of discarded undeclared cluster IDs."""

    point_labels: np.ndarray
    declared_instance_ids: tuple[int, ...]
    orphan_instance_ids: tuple[int, ...]
    orphan_counts: tuple[tuple[int, int], ...]
    empty_declared_instance_ids: tuple[int, ...]
    raw_background_gaussian_count: int
    declared_gaussian_count: int
    orphan_gaussian_count: int

    @property
    def point_count(self) -> int:
        return int(len(self.point_labels))

    def stats(self) -> dict[str, Any]:
        """Return JSON-safe projection diagnostics, including orphan IDs."""

        denominator = max(self.point_count, 1)
        return {
            "semantics": "undeclared non-negative point labels project to -1",
            "point_count": self.point_count,
            "declared_instance_count": len(self.declared_instance_ids),
            "nonempty_declared_instance_count": (
                len(self.declared_instance_ids)
                - len(self.empty_declared_instance_ids)
            ),
            "empty_declared_instance_ids": list(self.empty_declared_instance_ids),
            "declared_gaussian_count": self.declared_gaussian_count,
            "declared_gaussian_fraction": self.declared_gaussian_count / denominator,
            "orphan_instance_count": len(self.orphan_instance_ids),
            "orphan_instance_ids": list(self.orphan_instance_ids),
            "orphan_counts": {
                str(instance_id): count
                for instance_id, count in self.orphan_counts
            },
            "orphan_gaussian_count": self.orphan_gaussian_count,
            "orphan_gaussian_fraction": self.orphan_gaussian_count / denominator,
            "raw_background_gaussian_count": self.raw_background_gaussian_count,
            "raw_background_gaussian_fraction": (
                self.raw_background_gaussian_count / denominator
            ),
            "projected_background_gaussian_count": (
                self.raw_background_gaussian_count + self.orphan_gaussian_count
            ),
            "projected_background_gaussian_fraction": (
                self.raw_background_gaussian_count + self.orphan_gaussian_count
            )
            / denominator,
        }

    def numeric_stats(self) -> dict[str, float]:
        """Return scalar diagnostics suitable for evaluator alignment records."""

        stats = self.stats()
        keys = (
            "point_count",
            "declared_instance_count",
            "nonempty_declared_instance_count",
            "declared_gaussian_count",
            "declared_gaussian_fraction",
            "orphan_instance_count",
            "orphan_gaussian_count",
            "orphan_gaussian_fraction",
            "raw_background_gaussian_count",
            "raw_background_gaussian_fraction",
            "projected_background_gaussian_count",
            "projected_background_gaussian_fraction",
        )
        return {key: float(stats[key]) for key in keys}


def _declared_ids(instances_metadata: Mapping[str | int, Any]) -> tuple[int, ...]:
    result: set[int] = set()
    for raw_instance_id in instances_metadata:
        if isinstance(raw_instance_id, bool):
            raise TypeError("instance metadata IDs must be integers")
        try:
            instance_id = int(raw_instance_id)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"instance metadata ID is not an integer: {raw_instance_id!r}"
            ) from exc
        if str(instance_id) != str(raw_instance_id).strip():
            raise TypeError(
                f"instance metadata ID is not canonical: {raw_instance_id!r}"
            )
        if instance_id < 0:
            raise ValueError("declared instance IDs must be non-negative")
        result.add(instance_id)
    return tuple(sorted(result))


def project_declared_instances(
    point_labels: Sequence[int] | np.ndarray,
    instances_metadata: Mapping[str | int, Any],
) -> DeclaredInstanceProjection:
    """Project undeclared cluster labels to background without mutating input."""

    if not isinstance(instances_metadata, Mapping):
        raise TypeError("instances metadata must be a mapping")
    raw = np.asarray(point_labels)
    if raw.ndim != 1:
        raise ValueError("point_labels must be one-dimensional")
    try:
        labels = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("point_labels must contain integers") from exc
    if not np.array_equal(raw, labels):
        raise TypeError("point_labels must contain integers")

    declared_ids = _declared_ids(instances_metadata)
    nonnegative = labels >= 0
    declared_mask = np.isin(labels, declared_ids) if declared_ids else np.zeros(
        labels.shape, dtype=bool
    )
    orphan_mask = nonnegative & ~declared_mask
    orphan_ids, orphan_counts = np.unique(labels[orphan_mask], return_counts=True)
    labels[orphan_mask] = -1
    labels.setflags(write=False)

    present_declared = set(int(value) for value in np.unique(labels[declared_mask]))
    empty_declared = tuple(
        instance_id
        for instance_id in declared_ids
        if instance_id not in present_declared
    )
    return DeclaredInstanceProjection(
        point_labels=labels,
        declared_instance_ids=declared_ids,
        orphan_instance_ids=tuple(int(value) for value in orphan_ids),
        orphan_counts=tuple(
            (int(instance_id), int(count))
            for instance_id, count in zip(orphan_ids, orphan_counts)
        ),
        empty_declared_instance_ids=empty_declared,
        raw_background_gaussian_count=int(np.count_nonzero(~nonnegative)),
        declared_gaussian_count=int(np.count_nonzero(declared_mask)),
        orphan_gaussian_count=int(np.count_nonzero(orphan_mask)),
    )
