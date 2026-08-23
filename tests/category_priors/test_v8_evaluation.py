from __future__ import annotations

import json

import numpy as np
import pytest

from category_priors.v8_evaluation import evaluate_fragment_oracles


def test_exact_fragment_has_unit_oracles() -> None:
    result = evaluate_fragment_oracles(
        [np.array([0, 1, 2])],
        ["chair"],
        [np.array([0, 1, 2])],
        ["chair"],
        fragment_ids=[17],
        gt_instance_ids=[31],
        gt_valid=[True],
        gt_is_tiny_small=[True],
    )

    row = result["per_gt"][0]
    for metric in (
        "geometric_single",
        "semantic_single",
        "geometric_greedy_upper_bound",
        "semantic_greedy_upper_bound",
        "geometric_perfect_trim_support_ceiling",
        "semantic_perfect_trim_support_ceiling",
    ):
        assert row[metric] == pytest.approx(1.0)
    assert row["geometric_single_fragment_id"] == 17
    assert row["geometric_greedy_fragment_ids"] == [17]
    assert result["aggregate"]["tiny_small_official_valid"][
        "geometric_single"
    ]["match_050_count"] == 1
    json.dumps(result, allow_nan=False)


def test_semantic_labels_do_not_change_geometric_oracles() -> None:
    kwargs = {
        "fragment_point_ids": [np.array([0, 1]), np.array([2, 3])],
        "gt_point_ids": [np.array([0, 1, 2, 3])],
        "gt_class_ids": ["chair"],
    }
    correct = evaluate_fragment_oracles(
        fragment_class_ids=["chair", "chair"], **kwargs
    )
    perturbed = evaluate_fragment_oracles(
        fragment_class_ids=["table", "table"], **kwargs
    )

    correct_row = correct["per_gt"][0]
    perturbed_row = perturbed["per_gt"][0]
    for metric in (
        "geometric_single",
        "geometric_greedy_upper_bound",
        "geometric_perfect_trim_support_ceiling",
    ):
        assert perturbed_row[metric] == correct_row[metric]
    assert correct_row["semantic_greedy_upper_bound"] == pytest.approx(1.0)
    assert perturbed_row["semantic_single"] == 0.0
    assert perturbed_row["semantic_greedy_upper_bound"] == 0.0
    assert perturbed_row["semantic_perfect_trim_support_ceiling"] == 0.0


def test_greedy_upper_bound_is_monotonic_and_rejects_harmful_union() -> None:
    result = evaluate_fragment_oracles(
        [
            np.array([0, 1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9, 10, 11]),
            np.arange(20, 120),
        ],
        [1, 1, 1],
        [np.arange(10)],
        [1],
        fragment_ids=["left", "right", "noise"],
    )

    row = result["per_gt"][0]
    assert row["geometric_single"] == pytest.approx(0.6)
    assert row["geometric_greedy_upper_bound"] == pytest.approx(10 / 12)
    assert row["geometric_greedy_upper_bound"] >= row["geometric_single"]
    assert row["geometric_greedy_fragment_ids"] == ["left", "right"]
    assert "noise" not in row["geometric_greedy_fragment_ids"]


def test_perfect_trim_measures_support_without_false_positive_penalty() -> None:
    result = evaluate_fragment_oracles(
        [np.array([0, 1, 10, 11]), np.array([2, 3, 12, 13])],
        ["chair", "table"],
        [np.array([0, 1, 2, 3])],
        ["chair"],
    )

    row = result["per_gt"][0]
    assert row["geometric_greedy_upper_bound"] == pytest.approx(0.5)
    assert row["geometric_perfect_trim_support_ceiling"] == pytest.approx(1.0)
    assert row["semantic_perfect_trim_support_ceiling"] == pytest.approx(0.5)


def test_aggregates_separate_official_and_tiny_small_ground_truth() -> None:
    result = evaluate_fragment_oracles(
        [np.array([0, 1]), np.array([2, 3])],
        [1, 1],
        [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])],
        [1, 1, 1],
        gt_valid=[True, True, False],
        gt_is_tiny_small=[True, False, True],
    )

    aggregates = result["aggregate"]
    assert aggregates["all"]["gt_count"] == 3
    assert aggregates["official_valid"]["gt_count"] == 2
    assert aggregates["tiny_small_official_valid"]["gt_count"] == 1
    assert aggregates["official_valid"]["geometric_single"]["recall_050"] == 1.0
    assert aggregates["all"]["geometric_single"]["recall_050"] == pytest.approx(2 / 3)


def test_boolean_memberships_and_invalid_metadata_are_checked() -> None:
    result = evaluate_fragment_oracles(
        [np.array([True, False, True])],
        [0],
        [np.array([True, False, True])],
        [0],
    )
    assert result["per_gt"][0]["geometric_single"] == 1.0

    with pytest.raises(ValueError, match="one entry per fragment"):
        evaluate_fragment_oracles([np.array([0])], [], [np.array([0])], [0])
