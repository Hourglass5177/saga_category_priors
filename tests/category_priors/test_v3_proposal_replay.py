from __future__ import annotations

from category_priors.v3_proposal_replay import candidate_acceptance, proposal_score


def _candidate(**overrides):
    value = {
        "active_branch_points": 300,
        "assignment_confidence_mean": 0.9,
        "hdbscan_membership_mean": 0.9,
        "hdbscan_persistence": 0.1,
        "global_final_overlap": {"fraction": 0.2},
        "vote": {
            "winner_matches_branch": True,
            "branch_class_ratio": 0.8,
            "background_ratio": 0.1,
        },
    }
    value.update(overrides)
    return value


def test_accepts_strong_background_proposal():
    accepted, reason = candidate_acceptance(_candidate(), background_points=240)
    assert accepted
    assert reason == "accepted"


def test_rejects_proposal_that_overlaps_b1():
    accepted, reason = candidate_acceptance(
        _candidate(global_final_overlap={"fraction": 0.8}), background_points=150
    )
    assert not accepted
    assert reason == "overlaps_b1"


def test_rejects_weak_or_wrong_vote():
    accepted, reason = candidate_acceptance(
        _candidate(
            vote={
                "winner_matches_branch": False,
                "branch_class_ratio": 0.9,
                "background_ratio": 0.0,
            }
        ),
        background_points=250,
    )
    assert not accepted
    assert reason == "vote_class_mismatch"


def test_score_is_finite_probability():
    assert 0.0 <= proposal_score(_candidate()) <= 1.0
