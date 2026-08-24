from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from category_priors.v9_objectbank import CandidateBank
from category_priors.v9_replay import (
    assign_unique_gaussians,
    replay_candidate_bank,
    score_candidate,
    size_compatibility,
    smoothness_compatibility,
    support_compatibility,
    validate_prediction_contract,
)


def _node(
    short: float,
    middle: float,
    long: float,
    area: float,
    boundary_q50: float,
    boundary_q75: float,
) -> dict:
    return {
        "shrunk": {
            "geometry": {
                "log_extent_short_m": {
                    "q50": np.log(short),
                    "q75": np.log(short * 1.5),
                },
                "log_extent_mid_m": {
                    "q50": np.log(middle),
                    "q75": np.log(middle * 1.5),
                },
                "log_extent_long_m": {
                    "q50": np.log(long),
                    "q75": np.log(long * 1.5),
                },
                "log_surface_area_m2": {"q50": np.log(area)},
            },
            "neighborhood": {
                "boundary_fixed:0.05": {
                    "q50": boundary_q50,
                    "q75": boundary_q75,
                }
            },
        }
    }


def _priors() -> dict:
    return {
        "global": _node(0.5, 0.7, 1.0, 2.0, 0.5, 0.6),
        "categories": {
            "chair": _node(0.1, 0.2, 0.3, 0.1, 0.1, 0.2),
        },
    }


def _candidate(
    candidate_id: int,
    class_name: str = "chair",
    score: float = 0.8,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "branch_class": class_name,
        "base_score": score,
        "metric_extents_m": [0.5, 0.7, 1.0],
        "local_surface_density": 1000.0,
        "core_point_count": 3,
        "boundary_ratio_5cm": 0.5,
    }


def _bank(rows: tuple[dict, ...], full: tuple[np.ndarray, ...]) -> CandidateBank:
    core = tuple(np.asarray(ids[: min(3, len(ids))], dtype=np.int32) for ids in full)
    labels = np.full(max(int(max(map(np.max, full))) + 1, 1), -1, dtype=np.int32)
    for candidate_id, ids in enumerate(core):
        assert np.all(labels[ids] < 0)
        labels[ids] = candidate_id
    return CandidateBank(len(labels), "A1", labels, full, core, rows)


def test_registered_factors_change_only_their_compatibility_term() -> None:
    candidate = _candidate(0)
    scores = {
        condition: score_candidate(candidate, _priors(), condition)
        for condition in ("U000", "D100", "D010", "D001", "D111")
    }
    assert {parts["Q"] for parts in scores.values()} == {0.8}
    assert scores["D100"]["G"] < scores["U000"]["G"]
    assert scores["D010"]["C"] > scores["U000"]["C"]
    assert scores["D001"]["B"] < scores["U000"]["B"]
    assert np.isclose(
        scores["D111"]["score"],
        scores["D111"]["Q"]
        * scores["D111"]["G"]
        * scores["D111"]["C"]
        * scores["D111"]["B"],
    )


def test_compatibilities_are_one_sided_and_support_is_density_calibrated() -> None:
    global_node = _priors()["global"]
    small = _candidate(0)
    small["metric_extents_m"] = [0.1, 0.2, 0.3]
    assert size_compatibility(small, global_node) == 1.0
    assert size_compatibility(_candidate(1), global_node) == 1.0
    assert support_compatibility(_candidate(0), global_node) < 1.0
    smooth = _candidate(0)
    smooth["boundary_ratio_5cm"] = 0.05
    assert smoothness_compatibility(smooth, _priors()["categories"]["chair"]) == 1.0


def test_unknown_class_falls_back_to_global_for_all_three_factors() -> None:
    candidate = _candidate(0, "unknown")
    assert score_candidate(candidate, _priors(), "U000") == score_candidate(
        candidate, _priors(), "D111"
    )


def test_small_winner_is_removed_then_lower_candidate_recovers_points() -> None:
    owner, kept, dropped = assign_unique_gaussians(
        10,
        [0, 1],
        {0: 0.9, 1: 0.8},
        {0: np.arange(2), 1: np.arange(10)},
        min_points=3,
    )
    assert kept == (1,)
    assert dropped == (0,)
    assert np.all(owner == 1)


def test_replay_does_not_mutate_frozen_bank_and_exports_strict_contract() -> None:
    rows = (_candidate(0, score=1.0), _candidate(1, "table", score=0.9))
    full = (np.arange(0, 10), np.arange(10, 20))
    bank = _bank(rows, full)
    rows_before = deepcopy(rows)
    full_before = tuple(ids.copy() for ids in bank.full_ids)
    result = replay_candidate_bank(
        bank,
        _priors(),
        "U000",
        acceptance_threshold=0.0,
        min_points=3,
    )
    assert rows == rows_before
    assert all(np.array_equal(a, b) for a, b in zip(bank.full_ids, full_before))
    validate_prediction_contract(result.point_labels, result.instances)
    assert set(result.instances) == {"0", "1"}
    assert all("score" in value for value in result.instances.values())
    assert set(np.unique(result.point_labels)) == {0, 1}


def test_class_prior_score_can_change_overlapping_ownership_without_bank_change() -> None:
    rows = (_candidate(0, "chair", 0.9), _candidate(1, "table", 0.8))
    full = (np.arange(0, 15), np.arange(3, 18))
    # Disjoint cores make both candidates survive same-class NMS and satisfy
    # the immutable-bank contract while their full masks remain overlapping.
    core = (np.arange(0, 3), np.arange(15, 18))
    labels = np.full(18, -1, dtype=np.int32)
    labels[core[0]] = 0
    labels[core[1]] = 1
    bank = CandidateBank(18, "A1", labels, full, core, rows)
    uniform = replay_candidate_bank(
        bank, _priors(), "U000", acceptance_threshold=0.0, min_points=1
    )
    data = replay_candidate_bank(
        bank, _priors(), "D100", acceptance_threshold=0.0, min_points=1
    )
    uniform_owner = uniform.instance_metadata[str(uniform.point_labels[5])]["candidate_id"]
    data_owner = data.instance_metadata[str(data.point_labels[5])]["candidate_id"]
    assert uniform_owner == 0
    assert data_owner == 1
    assert np.array_equal(bank.core_candidate_id, labels)


@pytest.mark.parametrize(
    ("labels", "instances"),
    [
        ([0, 1], {"0": {"class": "chair", "score": 0.5}}),
        ([-1, 0], {"-1": {"class": "wall", "score": 0.5}, "0": {"class": "chair", "score": 0.5}}),
        ([0], {"0": {"class": "chair"}}),
        ([1], {"1": {"class": "chair", "score": 0.5}}),
    ],
)
def test_prediction_contract_rejects_orphan_negative_missing_score_and_noncontiguous(
    labels: list[int], instances: dict,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_prediction_contract(labels, instances)


def test_prediction_contract_rejects_integral_float_labels_and_boolean_score() -> None:
    with pytest.raises(TypeError):
        validate_prediction_contract(
            np.array([0.0]), {"0": {"class": "chair", "score": 0.5}}
        )
    with pytest.raises(ValueError):
        validate_prediction_contract(
            np.array([0]), {"0": {"class": "chair", "score": True}}
        )
