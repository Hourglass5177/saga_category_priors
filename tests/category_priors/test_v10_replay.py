from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import category_priors.v10_replay as v10_replay
import category_priors.v10_runner as v10_runner
from category_priors.prediction_contract import validate_prediction_contract
from category_priors.v10_replay import (
    replay_v10_candidate_bank,
    replay_v10_priors,
    replay_v10_scene,
    v10_replay_is_complete,
)
from category_priors.v10_runner import load_v10_candidate_bank, run_v10_banks
from category_priors.v9_lifting import V9_LIFTING_SCHEMA


def _node(short: float, middle: float, long: float, area: float, boundary: float) -> dict:
    return {
        "shrunk": {
            "geometry": {
                "log_extent_short_m": {
                    "q50": float(np.log(short)),
                    "q75": float(np.log(short * 1.5)),
                },
                "log_extent_mid_m": {
                    "q50": float(np.log(middle)),
                    "q75": float(np.log(middle * 1.5)),
                },
                "log_extent_long_m": {
                    "q50": float(np.log(long)),
                    "q75": float(np.log(long * 1.5)),
                },
                "log_surface_area_m2": {"q50": float(np.log(area))},
            },
            "neighborhood": {
                "boundary_fixed:0.05": {"q50": boundary, "q75": boundary + 0.1}
            },
        }
    }


def _priors(tmp_path: Path) -> Path:
    target = tmp_path / "priors.json"
    target.write_text(
        json.dumps(
            {
                "global": _node(0.5, 0.7, 1.0, 2.0, 0.5),
                "categories": {"book": _node(0.1, 0.2, 0.3, 0.1, 0.1)},
            }
        ),
        encoding="utf-8",
    )
    return target


def _bank(tmp_path: Path, monkeypatch, condition: str = "VC1") -> Path:
    scene_id = "scene0000_00"
    lifting_root = tmp_path / "lifting"
    lifting_dir = lifting_root / scene_id
    lifting_dir.mkdir(parents=True)
    (lifting_dir / "lifting_bank.json").write_text(
        json.dumps(
            {
                "schema": V9_LIFTING_SCHEMA,
                "scene_id": scene_id,
                "point_count": 13,
                "frame_count": 2,
                "identity": {
                    "schema": "test-lifting-identity",
                    "scene_id": scene_id,
                    "git_commit": "producer",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_load(source: Path):
        metadata = json.loads((source / "lifting_bank.json").read_text("utf-8"))
        return metadata, {"xyz_m": np.zeros((13, 3), dtype=np.float32)}

    def fake_builder(_metadata, _arrays, *, condition: str):
        stages = {
            stage: [
                {
                    "candidate_id": 0,
                    "class_name": "book" if stage == "final_candidate" else None,
                    "gaussian_ids": np.arange(10, dtype=np.int32),
                }
            ]
            for stage in v10_runner.V10_FUNNEL_STAGES
        }
        return {
            "point_count": 13,
            "fragments": [{"fragment_id": 1, "frame_id": 0}],
            "tracks": [{"track_id": 4, "fragment_ids": [1]}],
            "candidates": [
                {
                    "candidate_id": 0,
                    "track_id": 4,
                    "branch_class": "book",
                    "classification_eligible": True,
                    "full_point_count": 12,
                    "core_point_count": 10,
                    "base_score": 0.9,
                    "metric_extents_m": [0.5, 0.7, 1.0],
                    "local_surface_density": 100.0,
                    "boundary_ratio_5cm": 0.5,
                    "classifiers": {
                        classifier: {
                            "branch_class": "book",
                            "class_id": 0,
                            "semantic_ratio": 0.9,
                            "classification_eligible": True,
                        }
                        for classifier in ("mv-label", "codebook")
                    },
                }
            ],
            "full_ids": [np.arange(12, dtype=np.int32)],
            "core_ids": [np.arange(10, dtype=np.int32)],
            "accepted_edges": [],
            "stage_supports": stages,
        }

    monkeypatch.setattr(v10_runner, "load_lifting_bank", fake_load)
    output_root = tmp_path / "banks"
    run_v10_banks(
        lifting_root=lifting_root,
        output_root=output_root,
        scene_ids=[scene_id],
        conditions=[condition],
        git_commit="consumer",
        builder=fake_builder,
    )
    return output_root / condition / scene_id


def test_replay_reuses_v9_factorial_without_mutating_v10_bank(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _bank(tmp_path, monkeypatch)
    _, bank = load_v10_candidate_bank(source)
    full_before = tuple(ids.copy() for ids in bank.full_ids)
    core_before = tuple(ids.copy() for ids in bank.core_ids)
    fragments_before = tuple(dict(row) for row in bank.fragments)
    tracks_before = tuple(dict(row) for row in bank.tracks)
    priors = json.loads(_priors(tmp_path).read_text("utf-8"))

    uniform = replay_v10_candidate_bank(
        bank, priors, "U000", acceptance_threshold=0.0, min_points=3
    )
    data = replay_v10_candidate_bank(
        bank, priors, "D111", acceptance_threshold=0.0, min_points=3
    )
    validate_prediction_contract(uniform.point_labels, uniform.instances)
    validate_prediction_contract(data.point_labels, data.instances)
    assert uniform.candidate_scores[0]["score"] != data.candidate_scores[0]["score"]
    assert all(np.array_equal(left, right) for left, right in zip(bank.full_ids, full_before))
    assert all(np.array_equal(left, right) for left, right in zip(bank.core_ids, core_before))
    assert tuple(dict(row) for row in bank.fragments) == fragments_before
    assert tuple(dict(row) for row in bank.tracks) == tracks_before
    with pytest.raises(ValueError):
        bank.full_ids[0][0] = 12
    with pytest.raises(TypeError):
        bank.candidates[0]["branch_class"] = "chair"


def test_replay_outputs_are_condition_isolated_resumable_and_repairable(
    tmp_path: Path, monkeypatch,
) -> None:
    source = _bank(tmp_path, monkeypatch, "P1R1")
    priors = _priors(tmp_path)
    output_root = tmp_path / "replay"
    summary = replay_v10_priors(
        bank_root=tmp_path / "banks",
        output_root=output_root,
        scene_ids=["scene0000_00"],
        structure_conditions=["P1R1"],
        prior_conditions=["U000", "D111"],
        classifier="mv-label",
        category_priors=priors,
        acceptance_threshold=0.0,
        git_commit="replay-commit",
        min_points=3,
    )
    assert len(summary["runs"]) == 2
    uniform_dir = output_root / "P1R1/mv-label/U000/scene0000_00"
    data_dir = output_root / "P1R1/mv-label/D111/scene0000_00"
    assert uniform_dir != data_dir
    assert v10_replay_is_complete(
        uniform_dir,
        expected_structure_condition="P1R1",
        expected_prior_condition="U000",
        expected_classifier="mv-label",
        expected_category_priors=priors,
        expected_git_commit="replay-commit",
        expected_point_count=13,
    )
    assert v10_replay_is_complete(data_dir)

    original_replay = v10_replay.replay_v10_candidate_bank

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete replay output must be reused")

    monkeypatch.setattr(v10_replay, "replay_v10_candidate_bank", forbidden)
    resumed = replay_v10_scene(
        bank_dir=source,
        output_root=output_root,
        condition="U000",
        classifier="mv-label",
        category_priors=priors,
        acceptance_threshold=0.0,
        git_commit="replay-commit",
        min_points=3,
    )
    assert resumed["prior_condition"] == "U000"

    diagnostics_path = data_dir / "diagnostics.json"
    original_diagnostics = diagnostics_path.read_bytes()
    damaged_diagnostics = json.loads(original_diagnostics)
    damaged_diagnostics["candidate_scores"][0]["candidate_id"] = 1
    diagnostics_path.write_text(json.dumps(damaged_diagnostics), encoding="utf-8")
    assert not v10_replay_is_complete(data_dir)
    diagnostics_path.write_bytes(original_diagnostics)

    damaged_diagnostics = json.loads(original_diagnostics)
    damaged_diagnostics["candidate_scores"][0]["score"] = float("nan")
    diagnostics_path.write_text(json.dumps(damaged_diagnostics), encoding="utf-8")
    assert not v10_replay_is_complete(data_dir)
    diagnostics_path.write_bytes(original_diagnostics)

    damaged_diagnostics = json.loads(original_diagnostics)
    candidate_id = damaged_diagnostics["accepted_candidate_ids"][0]
    damaged_diagnostics["rejected_candidate_ids"].append(candidate_id)
    diagnostics_path.write_text(json.dumps(damaged_diagnostics), encoding="utf-8")
    assert not v10_replay_is_complete(data_dir)
    diagnostics_path.write_bytes(original_diagnostics)

    (data_dir / "output.json").write_text("{damaged", encoding="utf-8")
    repaired: list[str] = []

    def counted(*args, **kwargs):
        repaired.append("D111")
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(v10_replay, "replay_v10_candidate_bank", counted)
    replay_v10_scene(
        bank_dir=source,
        output_root=output_root,
        condition="D111",
        classifier="mv-label",
        category_priors=priors,
        acceptance_threshold=0.0,
        git_commit="replay-commit",
        min_points=3,
    )
    assert repaired == ["D111"]
    assert v10_replay_is_complete(data_dir)

    # The no-hash path/size/mtime identity must invalidate a replay even when
    # the priors path itself is unchanged.
    priors.write_text(priors.read_text("utf-8") + " ", encoding="utf-8")
    assert not v10_replay_is_complete(
        data_dir,
        expected_category_priors=priors,
        expected_git_commit="replay-commit",
    )
    repaired.clear()
    replay_v10_scene(
        bank_dir=source,
        output_root=output_root,
        condition="D111",
        classifier="mv-label",
        category_priors=priors,
        acceptance_threshold=0.0,
        git_commit="replay-commit",
        min_points=3,
    )
    assert repaired == ["D111"]
