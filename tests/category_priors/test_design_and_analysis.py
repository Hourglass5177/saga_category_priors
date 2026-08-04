from __future__ import annotations

import numpy as np
import pytest

from category_priors.analysis import factorial_bootstrap, holm_adjust
from category_priors.evaluator import GroundTruthScene, PredictedInstance
from category_priors.io import write_json
from category_priors.mapping import (
    build_run_schedule,
    choose_best_config,
    latin_hypercube_design,
)


def test_config_selection_rejects_locked_or_test_metrics() -> None:
    design = latin_hypercube_design("global", samples=2, seed=7)
    metrics = [
        {
            "config_id": design["configurations"][0]["config_id"],
            "split": "val-locked",
            "map_50_95": 0.5,
            "runtime_seconds": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="val-tune only"):
        choose_best_config(metrics, design)


def test_factorial_requires_all_eight_combinations() -> None:
    gt = GroundTruthScene(
        "scene", np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
    )
    prediction = PredictedInstance("scene", 1, 0, 1.0, np.ones(1, dtype=bool))
    bits = {
        f"condition-{index}": value
        for index, value in enumerate(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
            ]
        )
    }
    predictions = {name: [prediction] for name in bits}
    with pytest.raises(ValueError, match="eight"):
        factorial_bootstrap(
            [gt],
            predictions,
            bits,
            {"scene": "physical"},
            ["chair"],
            samples=1,
            min_region_size=1,
        )


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 1.0])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def test_schedule_is_seeded_and_block_complete(tmp_path) -> None:
    selection = tmp_path / "selection.json"
    write_json(
        selection, {"selection": {"tune": ["scene1"], "locked": ["scene2", "scene3"]}}
    )
    first = build_run_schedule(selection, "locked", ("A", "B", "C"), (7, 9), 11)
    second = build_run_schedule(selection, "locked", ("A", "B", "C"), (7, 9), 11)
    assert first == second
    assert len(first["runs"]) == 2 * 2 * 3
    for block in {row["block"] for row in first["runs"]}:
        assert {row["condition"] for row in first["runs"] if row["block"] == block} == {
            "A",
            "B",
            "C",
        }
