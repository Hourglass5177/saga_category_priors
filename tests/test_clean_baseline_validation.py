from __future__ import annotations

import random

import pytest

from category_priors.clean_baseline.validation import (
    BOOTSTRAP_SCHEMA,
    DATA_CONDITION,
    FINAL48_RESULT_SCHEMA,
    HOLDOUT5,
    HOLDOUT5_RESULT_SCHEMA,
    PHYSICAL_PAIR_ROW_SCHEMA,
    TUNE24_RESULT_SCHEMA,
    UNIFORM_CONDITION,
    ValidationObservation,
    evaluate_final48,
    evaluate_holdout5,
    evaluate_tune24,
    paired_scene_bootstrap,
    physical_scene_pair_rows,
    validate_final48_scene_ids,
)


def _paired_rows(
    scene_ids: list[str] | tuple[str, ...],
    *,
    delta_map: float = 0.003,
    delta_tiny: float = 0.02,
    base_map: float = 0.10,
    base_tiny: float = 0.20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene_id in scene_ids:
        rows.extend(
            [
                {
                    "scene_id": scene_id,
                    "condition": UNIFORM_CONDITION,
                    "map_50_95": base_map,
                    "tiny_small_recall_050": base_tiny,
                },
                {
                    "scene_id": scene_id,
                    "condition": DATA_CONDITION,
                    "map_50_95": base_map + delta_map,
                    "tiny_small_recall_050": base_tiny + delta_tiny,
                },
            ]
        )
    return rows


def _set_delta(rows: list[dict[str, object]], scene_id: str, delta: float) -> None:
    uniform = next(
        row
        for row in rows
        if row["scene_id"] == scene_id and row["condition"] == UNIFORM_CONDITION
    )
    data = next(
        row
        for row in rows
        if row["scene_id"] == scene_id and row["condition"] == DATA_CONDITION
    )
    data["map_50_95"] = float(uniform["map_50_95"]) + delta


def _tune24_scenes() -> list[str]:
    # Eleven physical environments have two scans; two have one scan.
    result = [
        f"scene{index:04d}_{scan:02d}"
        for index in range(11)
        for scan in (0, 1)
    ]
    result.extend(("scene0011_00", "scene0012_00"))
    assert len(result) == 24
    return result


def _final48_scenes() -> list[str]:
    return [f"scene{index:04d}_00" for index in range(48)]


def _assert_parquet_friendly(rows: list[dict[str, object]]) -> None:
    scalar_types = (str, int, float, bool, type(None))
    assert rows
    assert all(
        isinstance(value, scalar_types)
        for row in rows
        for value in row.values()
    )


def test_holdout5_is_the_frozen_canonical_set() -> None:
    assert HOLDOUT5 == (
        "scene0231_00",
        "scene0608_00",
        "scene0356_00",
        "scene0011_00",
        "scene0593_00",
    )
    assert len({scene.rsplit("_", 1)[0] for scene in HOLDOUT5}) == 5


def test_observation_validates_identity_condition_and_ranges() -> None:
    observation = ValidationObservation.from_row(
        {
            "scene_id": "scene0001_02",
            "condition": UNIFORM_CONDITION,
            "map_50_95": 0.1,
            "tiny_small_recall_050": 0.2,
        }
    )
    assert observation.physical_scene_id == "scene0001"
    assert observation.to_row()["map_50_95"] == 0.1
    with pytest.raises(ValueError, match="physical_scene_id"):
        ValidationObservation(
            "scene0001_02", "scene9999", UNIFORM_CONDITION, 0.1, 0.2
        )
    with pytest.raises(ValueError, match="condition"):
        ValidationObservation(
            "scene0001_02", "scene0001", "C0-no-prior", 0.1, 0.2
        )
    with pytest.raises(ValueError, match="map_50_95"):
        ValidationObservation(
            "scene0001_02", "scene0001", UNIFORM_CONDITION, float("nan"), 0.2
        )


def test_physical_rows_pair_then_average_repeated_scans() -> None:
    rows = _paired_rows(("scene0001_00", "scene0001_01", "scene0002_00"))
    _set_delta(rows, "scene0001_00", 0.01)
    _set_delta(rows, "scene0001_01", -0.002)
    physical = physical_scene_pair_rows(rows, stage="test")
    first = next(row for row in physical if row["physical_scene_id"] == "scene0001")
    assert first["scan_count"] == 2
    assert first["scene_ids"] == "scene0001_00|scene0001_01"
    assert first["delta_map_50_95"] == pytest.approx(0.004)
    assert first["schema"] == PHYSICAL_PAIR_ROW_SCHEMA
    _assert_parquet_friendly(physical)


def test_pairing_rejects_missing_or_duplicate_condition() -> None:
    rows = _paired_rows(("scene0001_00",))
    with pytest.raises(ValueError, match="duplicate"):
        physical_scene_pair_rows(rows + [dict(rows[0])], stage="test")
    with pytest.raises(ValueError, match="missing"):
        physical_scene_pair_rows(rows[:1], stage="test")


def test_holdout5_gate_passes_all_three_registered_checks() -> None:
    rows = _paired_rows(HOLDOUT5, delta_map=0.003, delta_tiny=0.01)
    result = evaluate_holdout5(rows)
    assert result["schema"] == HOLDOUT5_RESULT_SCHEMA
    assert result["passed"] is True
    assert result["macro"]["physical_scene_count"] == 5
    assert result["macro"]["positive_map_scene_count"] == 5
    assert all(result["checks"].values())
    _assert_parquet_friendly(result["rows"])


@pytest.mark.parametrize(
    ("deltas", "tiny_delta", "failed_check"),
    [
        ((-0.01, -0.01, 0.001, 0.001, 0.001), 0.01, "mean_delta_map_positive"),
        ((0.01, 0.01, -0.001, -0.001, -0.001), 0.01, "at_least_3_of_5_positive"),
        ((0.003,) * 5, -0.01, "tiny_small_delta_positive"),
    ],
)
def test_holdout5_gate_fails_each_registered_condition(
    deltas: tuple[float, ...], tiny_delta: float, failed_check: str
) -> None:
    rows = _paired_rows(HOLDOUT5, delta_map=0.0, delta_tiny=tiny_delta)
    for scene_id, delta in zip(HOLDOUT5, deltas):
        _set_delta(rows, scene_id, delta)
    result = evaluate_holdout5(rows)
    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_holdout5_requires_the_exact_registered_scans() -> None:
    rows = _paired_rows(HOLDOUT5[:-1] + ("scene9999_00",))
    with pytest.raises(ValueError, match="scene set differs"):
        evaluate_holdout5(rows)


def test_tune24_groups_scans_into_thirteen_equal_physical_units() -> None:
    rows = _paired_rows(_tune24_scenes(), delta_map=0.003)
    result = evaluate_tune24(rows)
    assert result["schema"] == TUNE24_RESULT_SCHEMA
    assert result["passed"] is True
    assert result["macro"]["scan_count"] == 24
    assert result["macro"]["physical_scene_count"] == 13
    assert result["macro"]["mean_delta_map_50_95"] == pytest.approx(0.003)
    assert len(result["rows"]) == 13
    _assert_parquet_friendly(result["rows"])


def test_tune24_uses_physical_macro_not_scan_weighting() -> None:
    scenes = _tune24_scenes()
    rows = _paired_rows(scenes, delta_map=0.0)
    # The first physical scene has two scans. Its mean delta is +0.10, while
    # each of the other 12 physical scenes is -0.005. Physical-scene macro is
    # positive even though every physical unit remains one vote.
    for scene_id in scenes:
        delta = 0.10 if scene_id.startswith("scene0000_") else -0.005
        _set_delta(rows, scene_id, delta)
    result = evaluate_tune24(rows)
    expected = (0.10 + 12 * -0.005) / 13
    assert result["macro"]["mean_delta_map_50_95"] == pytest.approx(expected)
    assert result["passed"] is True


def test_tune24_rejects_wrong_scan_or_physical_count() -> None:
    with pytest.raises(ValueError, match="24 scans"):
        evaluate_tune24(_paired_rows(_tune24_scenes()[:-1]))
    unique24 = [f"scene{index:04d}_00" for index in range(24)]
    with pytest.raises(ValueError, match="13 physical"):
        evaluate_tune24(_paired_rows(unique24))


def test_final48_scene_validation_rejects_repeated_physical_scene() -> None:
    valid = _final48_scenes()
    assert len(validate_final48_scene_ids(valid)) == 48
    invalid = list(valid)
    invalid[-1] = "scene0000_01"
    with pytest.raises(ValueError, match="repeated physical"):
        validate_final48_scene_ids(invalid)


def test_paired_bootstrap_is_deterministic_and_registered() -> None:
    rows = [
        {"delta_map_50_95": 0.001 + index * 0.0001}
        for index in range(48)
    ]
    first = paired_scene_bootstrap(rows)
    second = paired_scene_bootstrap(rows)
    assert first == second
    assert first["schema"] == BOOTSTRAP_SCHEMA
    assert first["samples"] == 10_000
    assert first["physical_scene_count"] == 48
    assert first["ci95_lower"] > 0.0
    with pytest.raises(ValueError, match="10,000"):
        paired_scene_bootstrap(rows, samples=9999)


def test_final48_passes_and_is_invariant_to_input_order() -> None:
    rows = _paired_rows(_final48_scenes(), delta_map=0.003, delta_tiny=0.01)
    first = evaluate_final48(rows)
    shuffled = list(rows)
    random.Random(123).shuffle(shuffled)
    second = evaluate_final48(shuffled)
    assert first == second
    assert first["schema"] == FINAL48_RESULT_SCHEMA
    assert first["passed"] is True
    assert first["macro"]["physical_scene_count"] == 48
    assert first["bootstrap"]["ci95_lower"] > 0.0
    assert all(first["checks"].values())
    _assert_parquet_friendly(first["rows"])


def test_final48_requires_effect_size_and_positive_ci() -> None:
    small = evaluate_final48(
        _paired_rows(_final48_scenes(), delta_map=0.001)
    )
    assert small["passed"] is False
    assert small["checks"]["delta_map_at_least_0.002"] is False

    rows = _paired_rows(_final48_scenes(), delta_map=0.0)
    for index, scene_id in enumerate(_final48_scenes()):
        _set_delta(rows, scene_id, 0.01 if index < 29 else -0.009)
    uncertain = evaluate_final48(rows)
    assert uncertain["macro"]["mean_delta_map_50_95"] >= 0.002
    assert uncertain["bootstrap"]["ci95_lower"] <= 0.0
    assert uncertain["passed"] is False
