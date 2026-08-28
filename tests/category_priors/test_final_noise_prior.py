from __future__ import annotations

import json

import numpy as np

from category_priors.final_noise_prior import (
    class_threshold,
    derive_thresholds_by_id,
    replay_class_prior,
    replay_filter,
    replay_u10,
    summarize_replay,
    u10_parity,
)


def _node(area: float) -> dict:
    return {
        "shrunk": {
            "geometry": {
                "log_surface_area_m2": {"q50": float(np.log(area))},
            }
        }
    }


def _priors() -> dict:
    return {
        "global": _node(4.0),
        "categories": {
            "cup": _node(0.04),
            "chair": _node(1.0),
            "wall": _node(16.0),
        },
    }


def test_replay_filter_removes_only_ids_strictly_below_threshold() -> None:
    labels = np.array([-1] + [0] * 2 + [1] * 3 + [2] * 4, dtype=np.int32)
    result = replay_filter(labels, {0: 3, "1": 3}, default=5)

    assert np.all(result[labels == 0] == -1)
    assert np.all(result[labels == 1] == 1)  # equal to threshold is retained
    assert np.all(result[labels == 2] == -1)
    assert result[0] == -1
    assert result.dtype == labels.dtype


def test_u10_helper_matches_a_frozen_uniform_reference_exactly() -> None:
    labels = np.array([0] * 9 + [1] * 10 + [-1], dtype=np.int64)
    expected = np.array([-1] * 9 + [1] * 10 + [-1], dtype=np.int64)

    assert np.array_equal(replay_u10(labels), expected)
    assert u10_parity(labels, expected)
    assert not u10_parity(labels, labels)


def test_class_threshold_uses_shrunk_area_formula_and_uniform_fallback() -> None:
    priors = _priors()

    # cup: round(10*sqrt(.04/4))=1, clipped to 3
    assert class_threshold(priors, "cup") == 3
    # chair: round(10*sqrt(1/4))=5
    assert class_threshold(priors, "chair") == 5
    # wall: round(10*sqrt(16/4))=20, clipped to 10
    assert class_threshold(priors, "wall") == 10
    assert class_threshold(priors, "missing") == 10


def test_data_condition_changes_only_branch_instance_thresholds() -> None:
    thresholds = derive_thresholds_by_id(
        instance_classes={0: "cup", 1: "cup", 2: "chair", 3: "missing"},
        branch_instance_ids={0, 2, 3},
        priors=_priors(),
    )

    assert thresholds == {0: 3, 1: 10, 2: 5, 3: 10}


def test_class_replay_and_summary_are_mechanical_and_json_serializable() -> None:
    labels = np.array([0] * 4 + [1] * 4 + [2] * 5 + [-1] * 2, dtype=np.int32)
    result, thresholds = replay_class_prior(
        labels,
        instance_classes={0: "cup", 1: "cup", 2: "chair"},
        branch_instance_ids={0, 2},
        priors=_priors(),
    )

    # Branch cup (m=3) and chair (m=5) survive; non-branch cup still uses U10.
    assert np.all(result[labels == 0] == 0)
    assert np.all(result[labels == 1] == -1)
    assert np.all(result[labels == 2] == 2)
    assert thresholds == {0: 3, 1: 10, 2: 5}

    summary = summarize_replay(
        labels,
        result,
        threshold_by_id=thresholds,
        branch_instance_ids={0, 2},
    )
    assert summary["candidate_count_before"] == 3
    assert summary["candidate_count_after"] == 2
    assert summary["candidate_count_removed"] == 1
    assert summary["changed_point_count"] == 4
    assert summary["removed_branch_instance_ids"] == []
    assert summary["removed_nonbranch_instance_ids"] == [1]
    json.dumps(summary)
