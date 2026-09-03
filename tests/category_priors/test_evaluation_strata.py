from __future__ import annotations

import json

import pytest

from category_priors.evaluation_strata import (
    DEFAULT_STRATA_PATH,
    load_evaluation_strata,
)


def test_frozen_small_and_tail_strata_are_instance_and_class_level() -> None:
    strata = load_evaluation_strata()
    assert strata.small_diagonal_threshold_m == pytest.approx(0.886095877588466)
    assert strata.is_small(strata.small_diagonal_threshold_m)
    assert not strata.is_small(strata.small_diagonal_threshold_m + 1e-6)
    assert strata.tail_classes == (
        "socket",
        "speaker",
        "switch",
        "fan",
        "refrigerator",
        "cup",
        "phone",
    )
    assert strata.is_tail("socket")
    assert not strata.is_tail("chair")


def test_tail_list_is_derived_by_count_then_name_and_is_not_a_viewer_list() -> None:
    payload = json.loads(DEFAULT_STRATA_PATH.read_text(encoding="utf-8"))
    counts = payload["training_instance_counts"]
    expected = tuple(
        name for name, _ in sorted(counts.items(), key=lambda item: (item[1], item[0]))
    )[:7]
    assert tuple(payload["tail_classes"]["names"]) == expected
    assert "book" not in expected


def test_small_threshold_rejects_invalid_measurements() -> None:
    strata = load_evaluation_strata()
    with pytest.raises(ValueError, match="finite and non-negative"):
        strata.is_small(float("nan"))
