from __future__ import annotations

"""Frozen instance-level small and class-level tail evaluation strata."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .candidate_bank import SAGA20_CLASSES


DEFAULT_STRATA_PATH = Path(__file__).with_name("evaluation_strata.json")


@dataclass(frozen=True)
class EvaluationStrata:
    small_diagonal_threshold_m: float
    tail_classes: tuple[str, ...]
    training_instance_counts: Mapping[str, int]
    source: Mapping[str, Any]

    def is_small(self, gt_bbox_diagonal_m: float) -> bool:
        value = float(gt_bbox_diagonal_m)
        if not math.isfinite(value) or value < 0:
            raise ValueError("GT bounding-box diagonal must be finite and non-negative")
        return value <= self.small_diagonal_threshold_m

    def is_tail(self, class_name: str) -> bool:
        return str(class_name) in self.tail_classes


def load_evaluation_strata(path: str | Path = DEFAULT_STRATA_PATH) -> EvaluationStrata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "saga-instance-recheck-strata-v1":
        raise ValueError("unsupported evaluation-strata schema")
    small = payload.get("small_instance", {})
    threshold = float(small.get("threshold_m", float("nan")))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("small-instance threshold must be finite and positive")
    counts = {
        str(name): int(count)
        for name, count in dict(payload.get("training_instance_counts", {})).items()
    }
    expected_names = set(SAGA20_CLASSES)
    if set(counts) != expected_names or any(count < 0 for count in counts.values()):
        raise ValueError("training counts must cover the exact SAGA20 class table")
    tail = tuple(str(name) for name in payload.get("tail_classes", {}).get("names", ()))
    tail_count = int(payload.get("tail_classes", {}).get("count", -1))
    expected_tail = tuple(
        name for name, _ in sorted(counts.items(), key=lambda item: (item[1], item[0]))
    )[: math.ceil(len(counts) / 3)]
    if tail_count != len(expected_tail) or tail != expected_tail:
        raise ValueError("tail-class list does not follow the frozen count/tie rule")
    return EvaluationStrata(
        small_diagonal_threshold_m=threshold,
        tail_classes=tail,
        training_instance_counts=MappingProxyType(counts),
        source=MappingProxyType(dict(payload.get("source", {}))),
    )


__all__ = ["DEFAULT_STRATA_PATH", "EvaluationStrata", "load_evaluation_strata"]
