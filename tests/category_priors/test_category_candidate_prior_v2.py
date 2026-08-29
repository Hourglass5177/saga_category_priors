from __future__ import annotations

import copy
import inspect
import math

import numpy as np
import pytest

from category_priors.category_candidate_prior_v2 import (
    DEV2_THRESHOLD_GRID,
    candidate_average_precision,
    candidate_f1_at_threshold,
    score_candidate_prior_v2,
    score_same_bank_candidate_priors,
    select_dev2_threshold,
    size_platform_compatibility,
    trusted_core_support_threshold,
    verify_same_bank_scores,
)


def _node(
    *,
    area: float = 1.0,
    extent_triplets: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 2.0, 4.0),) * 3,
) -> dict[str, object]:
    geometry: dict[str, object] = {
        "log_surface_area_m2": {"q50": math.log(area)}
    }
    for field, values in zip(
        (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        ),
        extent_triplets,
    ):
        geometry[field] = {
            "q25": math.log(values[0]),
            "q50": math.log(values[1]),
            "q75": math.log(values[2]),
        }
    # There is intentionally no neighborhood/smoothness section.
    return {"shrunk": {"geometry": geometry}}


def _priors() -> dict[str, object]:
    return {
        "global": _node(area=4.0),
        "categories": {
            "chair": _node(
                area=1.0,
                extent_triplets=((0.25, 0.50, 1.0),) * 3,
            ),
            "cabinet": _node(area=5.76),
            "huge": _node(area=64.0),
        },
    }


def _candidate(
    candidate_id: int = 0,
    *,
    class_name: str = "chair",
    extents: tuple[float, float, float] = (0.5, 0.5, 0.5),
    q_value: float = 0.8,
    trusted_count: int = 3,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "branch_class": class_name,
        "branch_class_index": 7,
        "metric_extents_m": list(extents),
        "base_score": q_value,
        "trusted_core_point_count": trusted_count,
        "full_ids": np.asarray([1, 2, 3, 4], dtype=np.int32),
        "trusted_core_ids": (1, 2, 3),
        "sample_rank": np.asarray([4, 1, 3], dtype=np.int64),
        # A deliberately unusable old smoothness feature proves it is ignored.
        "boundary_ratio_5cm": float("nan"),
    }


def test_two_sided_iqr_platform_is_inclusive_and_uses_sorted_pca_extents() -> None:
    node = _node(
        extent_triplets=(
            (0.5, 1.0, 2.0),
            (2.0, 3.0, 4.0),
            (5.0, 6.0, 7.0),
        )
    )
    candidate = {"metric_extents_m": [6.0, 1.0, 3.0]}
    assert size_platform_compatibility(candidate, node) == pytest.approx(1.0)


def test_two_sided_iqr_penalizes_low_and_high_by_their_own_half_iqr() -> None:
    node = _node()
    low = {"metric_extents_m": [0.5, 2.0, 4.0]}
    high = {"metric_extents_m": [1.0, 2.0, 8.0]}
    expected = math.exp(-0.5 / 3.0)
    assert size_platform_compatibility(low, node) == pytest.approx(expected)
    assert size_platform_compatibility(high, node) == pytest.approx(expected)


def test_two_sided_iqr_caps_each_squared_z_at_25() -> None:
    node = _node()
    candidate = {"metric_extents_m": [1e-100, 1e-100, 1e-100]}
    assert size_platform_compatibility(candidate, node) == pytest.approx(
        math.exp(-12.5)
    )


def test_size_platform_rejects_bad_extents_and_quantile_order() -> None:
    node = _node()
    with pytest.raises(ValueError, match="finite and non-negative"):
        size_platform_compatibility({"metric_extents_m": [-1.0, 2.0, 3.0]}, node)

    broken = _node()
    summary = broken["shrunk"]["geometry"]["log_extent_short_m"]
    summary["q25"] = summary["q75"] + 1.0
    with pytest.raises(ValueError, match="q25 <= q50 <= q75"):
        size_platform_compatibility({"metric_extents_m": [1.0, 2.0, 3.0]}, broken)


def test_trusted_core_support_is_fixed_for_u_scaled_for_d_and_clipped() -> None:
    priors = _priors()
    assert trusted_core_support_threshold(priors, "chair", "uniform") == 5
    # sqrt(1 / 4) * 5 = 2.5, rounded then clipped to the lower bound.
    assert trusted_core_support_threshold(priors, "chair", "class") == 3
    # sqrt(5.76 / 4) * 5 = 6 exactly.
    assert trusted_core_support_threshold(priors, "cabinet", "class") == 6
    assert trusted_core_support_threshold(priors, "huge", "class") == 10


def test_missing_class_falls_back_to_global_shrunk_statistics() -> None:
    priors = _priors()
    candidate = _candidate(
        class_name="unknown", extents=(1.0, 2.0, 4.0), trusted_count=5
    )
    uniform = score_candidate_prior_v2((candidate,), priors, "uniform")[0]
    class_shrunk = score_candidate_prior_v2((candidate,), priors, "class")[0]
    assert class_shrunk["G"] == pytest.approx(uniform["G"])
    assert class_shrunk["support_threshold"] == 5
    assert class_shrunk["support_pass"] is True


def test_score_is_q_times_g_with_trusted_core_hard_gate_and_no_smoothness() -> None:
    candidate = _candidate()
    original = copy.deepcopy(candidate)
    result = score_same_bank_candidate_priors((candidate,), _priors())
    uniform = result.uniform[0]
    class_shrunk = result.class_shrunk[0]

    assert uniform["G"] == pytest.approx(math.exp(-0.5))
    assert uniform["S"] == pytest.approx(uniform["Q"] * uniform["G"])
    assert uniform["support_threshold"] == 5
    assert uniform["support_pass"] is False
    assert class_shrunk["G"] == pytest.approx(1.0)
    assert class_shrunk["S"] == pytest.approx(0.8)
    assert class_shrunk["support_threshold"] == 3
    assert class_shrunk["support_pass"] is True
    assert "B" not in uniform and "B_smooth" not in uniform
    assert "accepted" not in uniform
    assert uniform["branch_class_index"] == class_shrunk["branch_class_index"] == 7
    assert result.identity.q_values == (0.8,)
    assert result.identity.bank_identity_equal
    assert result.identity.q_unchanged
    assert candidate.keys() == original.keys()
    assert np.array_equal(candidate["full_ids"], original["full_ids"])


def test_q_field_is_supported_but_must_agree_with_base_score() -> None:
    candidate = _candidate()
    candidate["Q"] = 0.8
    row = score_candidate_prior_v2((candidate,), _priors(), "uniform")[0]
    assert row["Q"] == 0.8
    candidate["Q"] = 0.7
    with pytest.raises(ValueError, match="disagree"):
        score_candidate_prior_v2((candidate,), _priors(), "uniform")


def test_score_requires_trusted_core_and_rejects_duplicate_or_derived_rows() -> None:
    missing = _candidate()
    del missing["trusted_core_point_count"]
    with pytest.raises(ValueError, match="trusted_core_point_count"):
        score_candidate_prior_v2((missing,), _priors(), "uniform")

    duplicate = (_candidate(0), _candidate(0))
    with pytest.raises(ValueError, match="unique"):
        score_candidate_prior_v2(duplicate, _priors(), "uniform")

    rescored = _candidate()
    rescored["S"] = 0.1
    with pytest.raises(ValueError, match="derived score fields"):
        score_candidate_prior_v2((rescored,), _priors(), "uniform")


def test_same_bank_verification_compares_every_field_except_arm_derivatives() -> None:
    result = score_same_bank_candidate_priors((_candidate(),), _priors())
    uniform = [copy.deepcopy(result.uniform[0])]
    class_shrunk = [copy.deepcopy(result.class_shrunk[0])]

    # All registered arm-dependent values may differ.
    class_shrunk[0]["G"] = 0.123
    class_shrunk[0]["S"] = 0.045
    class_shrunk[0]["support_threshold"] = 10
    class_shrunk[0]["support_pass"] = False
    assert verify_same_bank_scores(uniform, class_shrunk).candidate_ids == (0,)

    class_shrunk[0]["full_ids"][0] = 99
    with pytest.raises(ValueError, match="full_ids"):
        verify_same_bank_scores(uniform, class_shrunk)


def test_same_bank_verification_detects_q_and_optional_class_index_changes() -> None:
    result = score_same_bank_candidate_priors((_candidate(),), _priors())
    uniform = [copy.deepcopy(result.uniform[0])]
    changed_q = [copy.deepcopy(result.class_shrunk[0])]
    changed_q[0]["Q"] = np.nextafter(0.8, 1.0)
    with pytest.raises(ValueError, match="Q"):
        verify_same_bank_scores(uniform, changed_q)

    changed_class = [copy.deepcopy(result.class_shrunk[0])]
    changed_class[0]["branch_class_index"] = 8
    with pytest.raises(ValueError, match="branch_class_index"):
        verify_same_bank_scores(uniform, changed_class)


def test_score_apis_expose_no_gt_or_iou_parameters() -> None:
    for function in (score_candidate_prior_v2, score_same_bank_candidate_priors):
        names = set(inspect.signature(function).parameters)
        assert not any("gt" in name.lower() or "iou" in name.lower() for name in names)
        assert "positives" not in names


def test_candidate_ap_is_threshold_free_grouped_at_ties_and_support_aware() -> None:
    assert candidate_average_precision(
        [0.9, 0.8, 0.7], [True, False, True]
    ) == pytest.approx(5.0 / 6.0)
    assert candidate_average_precision(
        [0.9, 0.8, 0.7],
        [True, False, True],
        eligible=[True, False, True],
    ) == pytest.approx(1.0)
    assert candidate_average_precision(
        [0.9, 0.8, 0.7],
        [True, False, True],
        eligible=[True, True, False],
    ) == pytest.approx(0.5)
    assert candidate_average_precision([0.5, 0.5], [True, False]) == pytest.approx(
        0.5
    )
    assert candidate_average_precision([0.5, 0.5], [False, True]) == pytest.approx(
        0.5
    )


def test_candidate_ap_and_f1_can_use_equal_scene_weighting() -> None:
    scores = [0.9, 0.8, 0.9, 0.8]
    positives = [True, False, False, True]
    scenes = ["a", "a", "b", "b"]
    assert candidate_average_precision(
        scores, positives, scene_ids=scenes
    ) == pytest.approx(0.75)
    # At 0.85 scene a is perfect and scene b has one FP plus one FN.
    assert candidate_f1_at_threshold(
        scores, positives, 0.85, scene_ids=scenes
    ) == pytest.approx(0.5)


def test_candidate_f1_uses_inclusive_threshold_and_support_gate() -> None:
    scores = [0.9, 0.2, 0.1]
    positives = [True, False, True]
    assert candidate_f1_at_threshold(scores, positives, 0.2) == pytest.approx(0.5)
    assert candidate_f1_at_threshold(
        scores,
        positives,
        0.2,
        eligible=[True, False, True],
    ) == pytest.approx(2.0 / 3.0)


def test_dev2_threshold_selection_uses_fixed_grid_and_breaks_ties_higher() -> None:
    tied = select_dev2_threshold([0.9], [True])
    assert tied.selected_threshold == 0.25
    assert tied.selected_f1 == 1.0
    assert tuple(point.threshold for point in tied.grid) == DEV2_THRESHOLD_GRID

    unique = select_dev2_threshold([0.18, 0.12], [True, False])
    assert unique.selected_threshold == 0.15
    assert unique.selected_f1 == 1.0
    with pytest.raises(ValueError, match="fixed DEV2 grid"):
        select_dev2_threshold([0.9], [True], thresholds=(0.1, 0.2))
