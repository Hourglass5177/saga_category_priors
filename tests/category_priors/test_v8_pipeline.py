from __future__ import annotations

import json

import pytest

from category_priors.v8_pipeline import (
    CAUSAL2,
    DEV8,
    HOLDOUT5,
    _b1_run_is_complete,
    _compare_b1_outputs,
    _copy_replay_case,
    _mechanical_prior_effect,
    _select_final_viewer_scenes,
    _validate_locked_manifest,
    _validate_tune_manifest,
)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_registered_scene_sets_are_fixed_and_nonoverlapping() -> None:
    assert CAUSAL2 == DEV8[:2]
    assert len(DEV8) == 8
    assert len(HOLDOUT5) == 5
    assert not set(DEV8) & set(HOLDOUT5)


def test_tune_manifest_requires_24_scans_and_13_physical_scenes() -> None:
    extra = (
        "scene0645_01", "scene0025_00", "scene0046_01", "scene0474_00",
        "scene0591_00", "scene0329_00", "scene0164_00", "scene0064_00",
        "scene0231_01", "scene0608_01", "scene0356_01",
    )
    scenes = DEV8 + HOLDOUT5 + extra
    assert len(scenes) == 24
    _validate_tune_manifest(scenes)
    with pytest.raises(ValueError, match="exactly 24 scans"):
        _validate_tune_manifest(scenes + ("scene0999_00",))


def test_locked_manifest_requires_exact_unique_physical_scene_set() -> None:
    expected = tuple(f"scene{index:04d}_01" for index in range(48))
    actual = expected
    _validate_locked_manifest(actual, expected)
    duplicate_physical = actual[:-1] + ("scene0000_02",)
    with pytest.raises(ValueError, match="exactly match|distinct physical"):
        _validate_locked_manifest(duplicate_physical, expected)


def test_b1_closeout_compares_labels_and_metadata(tmp_path) -> None:
    historical = tmp_path / "historical"
    fixed = tmp_path / "fixed"
    payload = {"point_labels": [0, -1, 1], "instances": {"0": {"class": "chair"}}}
    _write_json(historical / "scene0000_00/seed-42/output.json", payload)
    _write_json(fixed / "scene0000_00/output.json", payload)

    result = _compare_b1_outputs(
        historical, fixed, ("scene0000_00",)
    )

    assert result["status"] == "compared"
    assert result["all_equal"] is True


def test_b1_resume_requires_output_and_diagnostics(tmp_path) -> None:
    run = tmp_path / "run"
    output = {
        "point_labels": [0, -1, 0],
        "instances": {"0": {"class": "chair"}},
    }
    _write_json(run / "output.json", output)
    assert not _b1_run_is_complete(run)
    _write_json(run / "diagnostics.json", {"instances": output["instances"]})
    assert _b1_run_is_complete(run, 3)
    assert not _b1_run_is_complete(run, 4)
    _write_json(run / "diagnostics.json", {"instances": {}})
    assert _b1_run_is_complete(run)
    _write_json(run / "output.json", {
        "point_labels": [1], "instances": {"0": {"class": "chair"}}
    })
    assert not _b1_run_is_complete(run)


def test_mechanical_effect_ignores_disjoint_rank_only_renumbering(tmp_path) -> None:
    root = tmp_path / "replay"
    scene = "scene0000_00"
    uniform = {
        "accepted_candidate_ids": [0, 1],
        "candidate_scores": [
            {"candidate_id": 0, "score": 0.505},
            {"candidate_id": 1, "score": 0.500},
        ],
    }
    data = {
        "accepted_candidate_ids": [0, 1],
        "candidate_scores": [
            {"candidate_id": 0, "score": 0.500},
            {"candidate_id": 1, "score": 0.505},
        ],
    }
    output = {"point_labels": [0] * 10 + [1] * 10, "instances": {}}
    _write_json(root / "U00" / scene / "diagnostics.json", uniform)
    _write_json(root / "D10" / scene / "diagnostics.json", data)
    _write_json(root / "U00" / scene / "output.json", output)
    _write_json(root / "D10" / scene / "output.json", output)

    result = _mechanical_prior_effect(root, (scene,), "D10")

    assert not result["accepted_or_owner_changed"]
    assert result["score_difference_at_least_001_count"] == 0
    assert not result["passed"]


def test_final_viewer_selects_best_median_and_worst_delta() -> None:
    analysis = {
        "conditions": {
            "U00": {"per_scene": [
                {"scene_id": "scene0", "map_50_95": 0.10},
                {"scene_id": "scene1", "map_50_95": 0.20},
                {"scene_id": "scene2", "map_50_95": 0.30},
            ]},
            "D10": {"per_scene": [
                {"scene_id": "scene0", "map_50_95": 0.05},
                {"scene_id": "scene1", "map_50_95": 0.21},
                {"scene_id": "scene2", "map_50_95": 0.40},
            ]},
        }
    }

    assert _select_final_viewer_scenes(analysis, "D10") == {
        "worst": "scene0", "median": "scene1", "best": "scene2"
    }


def test_final_viewer_keeps_three_distinct_scenes_when_deltas_tie() -> None:
    per_scene = [
        {"scene_id": f"scene{index}", "map_50_95": 0.0}
        for index in range(4)
    ]
    analysis = {
        "conditions": {
            "U00": {"per_scene": per_scene},
            "D10": {"per_scene": per_scene},
        }
    }

    roles = _select_final_viewer_scenes(analysis, "D10")

    assert len(set(roles.values())) == 3


def test_copy_replay_case_synthesizes_missing_diagnostics(tmp_path) -> None:
    source = tmp_path / "source" / "output.json"
    output = {
        "point_labels": [0, -1],
        "instances": {"0": {"class": "chair"}},
    }
    _write_json(source, output)

    target = tmp_path / "target"
    _copy_replay_case(source, target)

    assert json.loads((target / "output.json").read_text()) == output
    diagnostics = json.loads((target / "diagnostics.json").read_text())
    assert diagnostics["instances"] == output["instances"]
