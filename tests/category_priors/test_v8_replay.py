from __future__ import annotations

from copy import deepcopy

import numpy as np

from category_priors.v8_objects import CandidateBank
from category_priors.v8_replay import (
    assign_unique_gaussians,
    core_compatibility,
    greedy_same_class_core_nms,
    replay_candidates,
    score_candidate,
    size_compatibility,
)


def _geometry(short: float, middle: float, long: float, area: float) -> dict:
    return {
        "log_extent_short_m": {"q50": np.log(short), "q75": np.log(short * 1.5)},
        "log_extent_mid_m": {"q50": np.log(middle), "q75": np.log(middle * 1.5)},
        "log_extent_long_m": {"q50": np.log(long), "q75": np.log(long * 1.5)},
        "log_surface_area_m2": {"q50": np.log(area)},
    }


def _priors() -> dict:
    return {
        "global": {"shrunk": {"geometry": _geometry(0.5, 0.7, 1.0, 2.0)}},
        "categories": {
            "chair": {"shrunk": {"geometry": _geometry(0.1, 0.2, 0.3, 0.1)}}
        },
    }


def _candidate(candidate_id: int, class_name: str = "chair", score: float = 0.8) -> dict:
    return {
        "candidate_id": candidate_id,
        "branch_class": class_name,
        "base_score": score,
        "metric_extents_m": [0.5, 0.7, 1.0],
        "local_surface_density": 100.0,
        "core_point_count": 20,
    }


def test_size_prior_only_penalizes_abnormally_large_candidates() -> None:
    global_node = _priors()["global"]
    small = _candidate(0)
    small["metric_extents_m"] = [0.1, 0.2, 0.3]
    large = _candidate(1)
    assert size_compatibility(small, global_node) == 1.0
    assert size_compatibility(large, global_node) == 1.0
    assert size_compatibility(large, _priors()["categories"]["chair"]) < 1.0


def test_core_compatibility_uses_density_area_and_core_count() -> None:
    candidate = _candidate(0)
    candidate["core_point_count"] = 3
    candidate["local_surface_density"] = 1000.0
    compatibility = core_compatibility(candidate, _priors()["global"])
    assert 0.0 < compatibility < 1.0


def test_four_conditions_change_only_g_and_c_not_q() -> None:
    candidate = _candidate(0)
    scores = {
        condition: score_candidate(candidate, _priors(), condition)
        for condition in ("U00", "D10", "D01", "D11")
    }
    assert {parts["Q"] for parts in scores.values()} == {0.8}
    assert scores["D10"]["G"] < scores["U00"]["G"]
    assert scores["D01"]["C"] >= scores["U00"]["C"]
    assert np.isclose(
        scores["D11"]["score"],
        scores["D11"]["Q"] * scores["D11"]["G"] * scores["D11"]["C"],
    )


def test_unknown_class_falls_back_to_global_for_both_terms() -> None:
    candidate = _candidate(0, "unknown")
    assert score_candidate(candidate, _priors(), "U00") == score_candidate(
        candidate, _priors(), "D11"
    )


def test_nms_suppresses_only_same_class_core_overlap() -> None:
    rows = [_candidate(0, "chair"), _candidate(1, "chair"), _candidate(2, "table")]
    scores = {0: 0.9, 1: 0.8, 2: 0.7}
    cores = {
        0: np.arange(10),
        1: np.arange(1, 11),
        2: np.arange(1, 11),
    }
    kept, suppressed = greedy_same_class_core_nms(rows, scores, cores)
    assert kept == (0, 2)
    assert suppressed == (1,)


def test_unique_ownership_prefers_score_and_uses_stable_id_tie_break() -> None:
    owner, kept, dropped = assign_unique_gaussians(
        20,
        [2, 1, 0],
        {0: 0.9, 1: 0.9, 2: 0.8},
        {
            0: np.arange(0, 10),
            1: np.arange(5, 15),
            2: np.arange(10, 20),
        },
        min_points=3,
    )
    assert np.all(owner[5:10] == 0)
    assert kept == (0, 1, 2)
    assert dropped == ()


def test_small_final_instance_is_removed_without_reassignment() -> None:
    owner, kept, dropped = assign_unique_gaussians(
        12,
        [0, 1],
        {0: 0.9, 1: 0.8},
        {0: np.arange(10), 1: np.arange(8, 12)},
        min_points=3,
    )
    assert kept == (0,)
    assert dropped == (1,)
    assert np.all(owner[10:] == -1)


def test_replay_does_not_mutate_bank_and_materializes_metadata() -> None:
    full = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    core = full.copy()
    rows = (_candidate(0, score=1.0), _candidate(1, "table", score=0.9))
    bank = CandidateBank(
        point_count=20,
        core_candidate_id=core,
        full_ids=(np.arange(10), np.arange(10, 20)),
        core_ids=(np.arange(10), np.arange(10, 20)),
        candidates=rows,
    )
    full_before = full.copy()
    rows_before = deepcopy(rows)
    result = replay_candidates(
        bank, _priors(), "U00", acceptance_threshold=0.0, min_points=10
    )
    assert len(result.instances) == 2
    assert result.instance_metadata["0"]["source"] == "v8_object_bank"
    assert np.array_equal(full, full_before)
    assert rows == rows_before
    assert set(np.unique(result.point_labels)) == {0, 1}


def test_prior_score_can_change_overlapping_full_mask_ownership() -> None:
    rows = (_candidate(0, "chair", 0.9), _candidate(1, "table", 0.8))
    bank = CandidateBank(
        point_count=20,
        core_candidate_id=np.array([0] * 10 + [-1] * 5 + [1] * 5, dtype=np.int32),
        full_ids=(np.arange(15), np.arange(10, 20)),
        core_ids=(np.arange(10), np.arange(15, 20)),
        candidates=rows,
    )
    uniform = replay_candidates(
        bank, _priors(), "U00", acceptance_threshold=0.0, min_points=1
    )
    data = replay_candidates(
        bank, _priors(), "D10", acceptance_threshold=0.0, min_points=1
    )
    assert uniform.accepted_candidate_ids == (0, 1)
    assert data.accepted_candidate_ids == (0, 1)
    uniform_owner = uniform.instance_metadata[str(uniform.point_labels[12])]["candidate_id"]
    data_owner = data.instance_metadata[str(data.point_labels[12])]["candidate_id"]
    assert uniform_owner == 0
    assert data_owner == 1


def test_disjoint_score_rank_swap_does_not_renumber_instances() -> None:
    rows = (_candidate(0, "chair", 0.9), _candidate(1, "table", 0.8))
    bank = CandidateBank(
        point_count=20,
        core_candidate_id=np.array([0] * 10 + [1] * 10, dtype=np.int32),
        full_ids=(np.arange(10), np.arange(10, 20)),
        core_ids=(np.arange(10), np.arange(10, 20)),
        candidates=rows,
    )
    uniform = replay_candidates(
        bank, _priors(), "U00", acceptance_threshold=0.0, min_points=1
    )
    data = replay_candidates(
        bank, _priors(), "D10", acceptance_threshold=0.0, min_points=1
    )

    assert uniform.accepted_candidate_ids == data.accepted_candidate_ids == (0, 1)
    assert np.array_equal(uniform.point_labels, data.point_labels)
