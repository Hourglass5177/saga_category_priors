import json

import numpy as np

from category_priors.effective_repair_core import g0_members, g2_members, mean32, merge_scene
from category_priors.prediction_contract import validate_prediction_contract


def _b0(n=12):
    return {"point_labels": [0] * n,
            "instances": {"0": {"class": "table", "score": .8, "original_note": "keep"}}}


def _proposal(uid, members, label="phone", score=.7, ratio=None):
    row = {"uid": uid, "members": np.asarray(members, dtype=np.int64), "class": label, "score": score}
    if ratio is not None:
        row["ratio"] = ratio
    return row


def test_g0_keeps_alpha_alpha_and_any_independent_pair_but_subtracts_all_negatives():
    observations = [
        {"camera_uid": "I1", "hard_ids": [], "alpha_ids": [0, 1, 2, 3, 4, 5]},
        {"camera_uid": "I2", "hard_ids": [], "alpha_ids": [0, 1, 2, 3, 4, 5]},
        {"camera_uid": "I3", "hard_ids": [6], "alpha_ids": [], "negative_ids": [5]},
    ]
    assert g0_members(observations[:2], [("I1", "I2")]).tolist() == list(range(6))
    assert g0_members(observations, [("I2", "I1"), ("I1", "I3")]).tolist() == list(range(5))
    assert not len(g0_members(observations, [("I1", "I3")]))


def test_g2_pools_weak_views_with_existing_mass_ratio_and_minimum_size_rules():
    inside = np.array([[.25, .5, .4, .1], [.25, .0, .4, .1]], dtype=np.float32)
    visible = np.array([[.5, .5, .5, .5], [.5, .5, .5, .5]], dtype=np.float32)
    result = g2_members(inside, visible)
    assert result["members"].tolist() == [0, 1, 2]
    assert np.allclose(result["ratio"], [.5, .5, .8, .2])
    assert not len(g2_members(inside[:, :2], visible[:, :2])["members"])


def test_mean32_reports_disagreement_without_veto_and_exact_mean_tie_is_unknown():
    classes = [f"class{i}" for i in range(32)]
    first = np.zeros(32); first[:2] = [.8, .1]
    second = np.zeros(32); second[:2] = [.2, .6]
    result = mean32([first, None, second], classes)
    assert result["class"] == "class0"
    assert result["score"] == .75
    assert result["valid_view_count"] == 2
    assert result["per_view_top1"] == ["class0", None, "class1"]
    assert result["disagreement"] is True
    assert np.isclose(result["margin"], .15)
    assert mean32([np.zeros(32)], classes)["class"] is None
    assert mean32([None], classes)["valid_view_count"] == 0


def test_ab_actual_semantic_rejection_restores_only_owned_points_without_redistribution():
    baseline = _b0()
    calls = []

    def classify(uid, members):
        calls.append((uid, members.tolist()))
        return {"class": "out_of_taxonomy" if uid == "a" else "phone", "score": .9}

    result = merge_scene(baseline, [
        _proposal("a", [0, 1, 2, 3]),
        _proposal("b", [3, 4, 5, 6]),
        _proposal("unknown", [4, 5, 6, 7], None, None),
        _proposal("outside", [4, 5, 6, 8], "bottle"),
    ], ["phone", "table"], policy="B", classify_actual=classify)
    assert calls == [("a", [0, 1, 2]), ("b", [4, 5, 6])]
    records = result["proposals"]
    assert records["a"]["status"] == "actual_non_saga20_or_unknown"
    assert records["a"]["assigned_members"].tolist() == [0, 1, 2]
    assert not len(records["a"]["exported_members"])
    assert records["b"]["exported_members"].tolist() == [4, 5, 6]
    # Point 3 remains B0 after a is rejected: no second competition pass.
    assert result["payload"]["point_labels"] == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert result["payload"]["instances"]["0"]["class"] == "table"
    assert result["payload"]["instances"]["0"]["score"] == .8
    assert result["payload"]["instances"]["0"]["original_note"] == "keep"
    assert records["b"]["exported_source_transfers"] == {"0": 3}
    assert baseline == _b0()
    json.dumps(result["payload"], allow_nan=False)
    validate_prediction_contract(np.asarray(result["payload"]["point_labels"]), result["payload"]["instances"])


def test_ab_too_small_after_competition_restores_b0_without_callback():
    result = merge_scene(_b0(), [
        _proposal("a", [0, 1, 2]), _proposal("b", [2, 3, 4]),
    ], ["phone", "table"], policy="A")
    assert result["payload"]["point_labels"] == _b0()["point_labels"]
    assert {row["status"] for row in result["proposals"].values()} == {"assigned_fewer_than_three_members"}


def test_c_ratio_winners_exact_ties_and_actual_classification_is_diagnostic_only():
    calls = []

    def classify(uid, members):
        calls.append((uid, members.tolist()))
        return {"class": None, "score": None}

    result = merge_scene(_b0(), [
        _proposal("a", [4, 3, 2, 1, 0], ratio=[.5, .6, .7, .8, .9]),
        _proposal("b", [3, 4, 5, 6, 7], "cup", ratio=[.4, .5, .8, .8, .8]),
    ], ["phone", "cup", "table"], policy="C", classify_actual=classify)
    assert calls == [("a", [0, 1, 2, 3]), ("b", [5, 6, 7])]
    assert result["summary"]["contested_point_count"] == 2
    assert result["summary"]["ratio_tie_point_count"] == 1
    assert result["payload"]["point_labels"][4] == 0
    assert result["proposals"]["a"]["final_class"] == "phone"
    assert result["proposals"]["b"]["final_class"] == "cup"
    assert result["proposals"]["a"]["actual_semantics_diagnostic"]["class"] is None


def test_exact_aliases_share_actual_export_for_feedback_and_do_not_compete():
    result = merge_scene(_b0(), [
        _proposal("z", [2, 1, 0], score=.95),
        _proposal("a", [0, 1, 2], score=.65),
    ], ["phone", "table"], policy="A")
    assert result["alias_to_representative"] == {"a": "a", "z": "a"}
    assert result["summary"]["contested_point_count"] == 0
    assert result["export_id_by_uid"]["z"] == result["export_id_by_uid"]["a"]
    assert result["proposals"]["z"]["exported_members"].tolist() == [0, 1, 2]
    assert result["proposals"]["z"]["final_score"] == .65
    assert result["proposals"]["z"]["alias_of"] == "a"
    assert result["payload"]["candidate_export_lineage"] == {"1": [0, 1]}
    assert result["payload"]["candidate_export_ids"] == {"0": [1], "1": [1]}
    assert result["payload"]["refined_export_ids"] == [1]


def test_distinct_classes_do_not_alias_even_with_identical_members():
    result = merge_scene(_b0(), [
        _proposal("a", [0, 1, 2]), _proposal("b", [0, 1, 2], "cup"),
    ], ["phone", "cup", "table"], policy="A")
    assert result["summary"]["alias_count"] == 0
    assert result["summary"]["contested_point_count"] == 3
    assert result["payload"]["point_labels"] == _b0()["point_labels"]
