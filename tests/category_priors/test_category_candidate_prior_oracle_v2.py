from __future__ import annotations

import math

import pytest

from category_priors.category_candidate_prior_oracle_v2 import (
    SIZE_G_FLOOR,
    evaluate_prior_oracle_v2_rows,
    prior_oracle_v2_gate_checks,
)


def _node(
    *,
    q25: float,
    q50: float,
    q75: float,
    area: float = 1.0,
) -> dict[str, object]:
    geometry: dict[str, object] = {
        "log_surface_area_m2": {"q50": math.log(area)}
    }
    for field in (
        "log_extent_short_m",
        "log_extent_mid_m",
        "log_extent_long_m",
    ):
        geometry[field] = {
            "q25": math.log(q25),
            "q50": math.log(q50),
            "q75": math.log(q75),
        }
    return {"shrunk": {"geometry": geometry}}


def _priors(
    *,
    global_extent: tuple[float, float, float] = (0.1, 1.0, 10.0),
    class_extent: tuple[float, float, float] = (1.0, 2.0, 4.0),
    global_area: float = 1.0,
    class_area: float = 1.0,
    include_class: bool = True,
) -> dict[str, object]:
    priors: dict[str, object] = {
        "global": _node(
            q25=global_extent[0],
            q50=global_extent[1],
            q75=global_extent[2],
            area=global_area,
        ),
        "categories": {},
    }
    if include_class:
        priors["categories"]["chair"] = _node(
            q25=class_extent[0],
            q50=class_extent[1],
            q75=class_extent[2],
            area=class_area,
        )
    return priors


def _object(
    *,
    extents: tuple[float, float, float] = (2.0, 2.0, 2.0),
    count: int = 8,
    scene_id: str = "scene0000_00",
    physical_scene_id: str = "scene0000",
    radius_m: float = 0.05,
    eligible: bool = True,
    class_name: str = "chair",
) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "physical_scene_id": physical_scene_id,
        "radius_m": radius_m,
        "eligible": eligible,
        "class_name": class_name,
        "gaussian_count": count,
        "metric_extent_short_m": extents[0],
        "metric_extent_mid_m": extents[1],
        "metric_extent_long_m": extents[2],
    }


def _pair(
    negative_type: str,
    *,
    full_extents: tuple[float, float, float] = (2.0, 2.0, 2.0),
    negative_extents: tuple[float, float, float] | None = None,
    full_count: int = 8,
    negative_count: int = 8,
    scene_id: str = "scene0000_00",
    physical_scene_id: str = "scene0000",
    radius_m: float = 0.05,
    class_name: str = "chair",
) -> dict[str, object]:
    if negative_extents is None:
        negative_extents = (
            (0.5, 2.0, 2.0)
            if negative_type == "fragment"
            else (2.0, 2.0, 8.0)
        )
    return {
        "scene_id": scene_id,
        "physical_scene_id": physical_scene_id,
        "radius_m": radius_m,
        "class_name": class_name,
        "negative_type": negative_type,
        "full_gaussian_count": full_count,
        "full_metric_extent_short_m": full_extents[0],
        "full_metric_extent_mid_m": full_extents[1],
        "full_metric_extent_long_m": full_extents[2],
        "negative_gaussian_count": negative_count,
        "negative_metric_extent_short_m": negative_extents[0],
        "negative_metric_extent_mid_m": negative_extents[1],
        "negative_metric_extent_long_m": negative_extents[2],
    }


def _healthy_result() -> dict[str, object]:
    return evaluate_prior_oracle_v2_rows(
        object_rows=(_object(),),
        pair_rows=(_pair("fragment"), _pair("merge")),
        priors=_priors(),
    )


def test_full_object_low_g_uses_preregistered_001_threshold() -> None:
    # z=4 on all axes produces exp(-8), which is below the registered .001
    # limit even though it has not reached the formula's clipped floor.
    near_floor = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(extents=(1 / 16, 1 / 16, 1 / 16)),),
        pair_rows=(),
        priors=_priors(global_extent=(1.0, 2.0, 4.0)),
    )
    assert near_floor["medians"]["full_D_G"] == pytest.approx(math.exp(-8.0))
    assert near_floor["full_D_G_le_0.001_fraction"] == 1.0
    assert not near_floor["checks"]["full_G_le_0.001_fraction_at_most_0.10"]

    clipped = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(extents=(1e-100, 1e-100, 1e-100)),),
        pair_rows=(),
        priors=_priors(),
    )
    assert clipped["G_formula_floor"] == pytest.approx(SIZE_G_FLOOR)
    assert clipped["medians"]["full_D_G"] == pytest.approx(SIZE_G_FLOOR)
    assert clipped["full_D_G_le_0.001_fraction"] == 1.0
    assert not clipped["checks"]["full_G_le_0.001_fraction_at_most_0.10"]


def test_full_vs_fragment_and_merge_use_support_gated_paired_auc() -> None:
    result = _healthy_result()
    for negative_type in ("fragment", "merge"):
        effect = result["effects"][negative_type]
        assert effect["pair_count"] == 1
        assert effect["U_scene_equal_accuracy"] == pytest.approx(0.5)
        assert effect["D_scene_equal_accuracy"] == pytest.approx(1.0)
        assert effect["D_minus_U"] == pytest.approx(0.5)
    assert result["checks"]["full_median_G_above_fragment_and_merge"]
    assert result["checks"][
        "class_improves_one_discrimination_without_harming_other"
    ]
    assert result["passed"]


def test_support_pass_uses_all_oracle_gaussians_as_trusted_core() -> None:
    result = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(count=4),),
        pair_rows=(
            _pair("fragment", full_count=4, negative_count=3),
            _pair("merge", full_count=4, negative_count=8),
        ),
        priors=_priors(class_area=0.25),
    )
    assert result["U_full_support_scene_equal_rate"] == 0.0
    assert result["D_full_support_scene_equal_rate"] == 1.0
    assert result["checks"][
        "class_full_support_rate_not_lower_by_more_than_0.05"
    ]
    assert result["effects"]["fragment"]["D_minus_U"] == pytest.approx(0.5)
    assert result["effects"]["merge"]["D_minus_U"] == pytest.approx(1.0)


def test_class_support_drop_is_a_hard_failure() -> None:
    result = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(count=5),),
        pair_rows=(_pair("fragment"), _pair("merge")),
        priors=_priors(class_area=4.0),
    )
    assert result["U_full_support_scene_equal_rate"] == 1.0
    assert result["D_full_support_scene_equal_rate"] == 0.0
    name = "class_full_support_rate_not_lower_by_more_than_0.05"
    assert not result["checks"][name]
    assert name in result["failed_checks"]
    assert not result["passed"]


def test_missing_class_uses_global_fallback_and_cannot_fake_an_effect() -> None:
    result = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(class_name="unknown"),),
        pair_rows=(
            _pair("fragment", class_name="unknown"),
            _pair("merge", class_name="unknown"),
        ),
        priors=_priors(include_class=False),
    )
    assert result["effects"]["fragment"]["D_minus_U"] == 0.0
    assert result["effects"]["merge"]["D_minus_U"] == 0.0
    assert result["U_full_support_scene_equal_rate"] == result[
        "D_full_support_scene_equal_rate"
    ]
    assert not result["checks"][
        "class_improves_one_discrimination_without_harming_other"
    ]
    assert not result["passed"]


def test_full_median_must_exceed_both_registered_negative_types() -> None:
    same_extent = (2.0, 2.0, 2.0)
    result = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(extents=same_extent, count=4),),
        pair_rows=(
            _pair(
                "fragment",
                full_extents=same_extent,
                negative_extents=same_extent,
                full_count=4,
                negative_count=3,
            ),
            _pair(
                "merge",
                full_extents=same_extent,
                negative_extents=same_extent,
                full_count=4,
                negative_count=8,
            ),
        ),
        priors=_priors(
            global_extent=(1.0, 2.0, 4.0),
            class_extent=(1.0, 2.0, 4.0),
            class_area=0.64,
        ),
    )
    assert result["effects"]["fragment"]["D_minus_U"] == pytest.approx(0.5)
    assert result["effects"]["merge"]["D_minus_U"] == pytest.approx(0.5)
    assert not result["checks"]["full_median_G_above_fragment_and_merge"]
    assert not result["passed"]


def test_registered_gate_boundaries_are_inclusive() -> None:
    checks = prior_oracle_v2_gate_checks(
        full_g_low_fraction=0.10,
        fragment_d_minus_u=0.02,
        merge_d_minus_u=-0.02,
        uniform_full_support_rate=0.80,
        class_full_support_rate=0.75,
        full_median_above_fragment_and_merge=True,
    )
    assert all(checks.values())

    reverse = prior_oracle_v2_gate_checks(
        full_g_low_fraction=0.10,
        fragment_d_minus_u=-0.02,
        merge_d_minus_u=0.02,
        uniform_full_support_rate=0.80,
        class_full_support_rate=0.75,
        full_median_above_fragment_and_merge=True,
    )
    assert all(reverse.values())


@pytest.mark.parametrize(
    ("overrides", "failed_name"),
    (
        (
            {"full_g_low_fraction": 0.101},
            "full_G_le_0.001_fraction_at_most_0.10",
        ),
        (
            {"fragment_d_minus_u": 0.019},
            "class_improves_one_discrimination_without_harming_other",
        ),
        (
            {"merge_d_minus_u": -0.021},
            "class_improves_one_discrimination_without_harming_other",
        ),
        (
            {"class_full_support_rate": 0.749},
            "class_full_support_rate_not_lower_by_more_than_0.05",
        ),
    ),
)
def test_registered_gate_fails_just_beyond_each_limit(
    overrides: dict[str, float], failed_name: str
) -> None:
    values = {
        "full_g_low_fraction": 0.10,
        "fragment_d_minus_u": 0.02,
        "merge_d_minus_u": -0.02,
        "uniform_full_support_rate": 0.80,
        "class_full_support_rate": 0.75,
        "full_median_above_fragment_and_merge": True,
    }
    values.update(overrides)
    checks = prior_oracle_v2_gate_checks(**values)
    assert not checks[failed_name]


def test_gate_rejects_out_of_domain_aggregate_values() -> None:
    checks = prior_oracle_v2_gate_checks(
        full_g_low_fraction=-0.1,
        fragment_d_minus_u=2.0,
        merge_d_minus_u=0.0,
        uniform_full_support_rate=1.1,
        class_full_support_rate=1.0,
        full_median_above_fragment_and_merge=False,
    )
    assert not any(checks.values())


def test_missing_pair_type_and_empty_input_fail_closed() -> None:
    fragment_only = evaluate_prior_oracle_v2_rows(
        object_rows=(_object(),),
        pair_rows=(_pair("fragment"),),
        priors=_priors(),
    )
    assert fragment_only["effects"]["merge"]["D_minus_U"] is None
    assert not fragment_only["passed"]

    empty = evaluate_prior_oracle_v2_rows(
        object_rows=(), pair_rows=(), priors=_priors()
    )
    assert empty["object_count"] == 0
    assert empty["pair_count"] == 0
    assert empty["full_D_G_le_0.001_fraction"] == 1.0
    assert not empty["passed"]
    assert len(empty["failed_checks"]) == 4


def test_only_main_radius_and_eligible_complete_objects_are_scored() -> None:
    result = evaluate_prior_oracle_v2_rows(
        object_rows=(
            _object(),
            _object(scene_id="scene0001_00", radius_m=0.02),
            _object(scene_id="scene0002_00", eligible=False),
        ),
        pair_rows=(
            _pair("fragment"),
            _pair("merge"),
            _pair("fragment", scene_id="scene0001_00", radius_m=0.02),
        ),
        priors=_priors(),
    )
    assert result["object_count"] == 1
    assert result["pair_count"] == 2
    assert result["main_radius_m"] == 0.05


def test_invalid_main_radius_and_negative_type_are_rejected() -> None:
    with pytest.raises(ValueError, match="main_radius_m"):
        evaluate_prior_oracle_v2_rows(
            object_rows=(), pair_rows=(), priors=_priors(), main_radius_m=0.0
        )
    with pytest.raises(ValueError, match="negative_type"):
        evaluate_prior_oracle_v2_rows(
            object_rows=(_object(),),
            pair_rows=(_pair("adversarial"),),
            priors=_priors(),
        )
