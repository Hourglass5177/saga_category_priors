from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import category_priors.v9_runner as v9_runner
from category_priors.v9_objectbank import V9Config
from category_priors.v9_runner import (
    build_v9_object_bank,
    load_v9_candidate_bank,
    object_bank_is_complete,
    replay_v9_scene,
)


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": 0,
        "track_id": 2,
        "association_mode": "A1",
        "branch_class": "book",
        "class_id": 1,
        "classification_source": "mv-label",
        "full_point_count": 12,
        "core_point_count": 10,
        "halo_point_count": 2,
        "effective_view_count": 2,
        "semantic_ratio": 0.8,
        "semantic_margin": 0.4,
        "mean_core_positive_ratio": 0.8,
        "conflict_ratio": 0.0,
        "internal_affinity": 0.9,
        "median_track_overlap": 0.5,
        "base_score": 0.9,
        "metric_extents_m": [0.1, 0.2, 0.3],
        "local_surface_density": 10.0,
        "boundary_ratio_5cm": 0.2,
    }


def _bank(tmp_path: Path) -> Path:
    target = tmp_path / "A1" / "scene0000_00"
    target.mkdir(parents=True)
    candidate = _candidate()
    metadata = {
        "schema": "saga-v9-clean-object-bank-v1",
        "scene_id": "scene0000_00",
        "point_count": 13,
        "association_mode": "A1",
        "git_commit": "commit",
        "source_lifting_bank": str(tmp_path / "lifting"),
        "source_lifting_identity": {"git_commit": "commit", "scene_id": "scene0000_00"},
        "config": {},
        "classifiers": {
            "mv-label": {"candidate_count": 1, "candidates": [candidate]},
            "codebook": {"candidate_count": 1, "candidates": [candidate]},
        },
    }
    (target / "object_bank.json").write_text(json.dumps(metadata), encoding="utf-8")
    arrays: dict[str, np.ndarray] = {"xyz_m": np.zeros((13, 3), dtype=np.float32)}
    for suffix in ("mv_label", "codebook"):
        arrays[f"core_candidate_id_{suffix}"] = np.asarray([0] * 10 + [-1] * 3)
        arrays[f"full_candidate_indptr_{suffix}"] = np.asarray([0, 12])
        arrays[f"full_candidate_ids_{suffix}"] = np.arange(12)
        arrays[f"core_candidate_indptr_{suffix}"] = np.asarray([0, 10])
        arrays[f"core_candidate_ids_{suffix}"] = np.arange(10)
    np.savez_compressed(target / "object_bank.npz", **arrays)
    return target


def _priors(path: Path) -> Path:
    node = {
        "shrunk": {
            "geometry": {
                "log_extent_short_m": {"q50": -3.0, "q75": -1.0},
                "log_extent_mid_m": {"q50": -3.0, "q75": -1.0},
                "log_extent_long_m": {"q50": -3.0, "q75": -1.0},
                "log_surface_area_m2": {"q50": -2.0},
            },
            "neighborhood": {"boundary_fixed:0.05": {"q50": 0.2, "q75": 0.5}},
        }
    }
    output = path / "priors.json"
    output.write_text(
        json.dumps({"global": node, "categories": {"book": node}}),
        encoding="utf-8",
    )
    return output


def test_v9_bank_round_trip_and_contract(tmp_path: Path) -> None:
    target = _bank(tmp_path)
    lifting_identity = {"git_commit": "commit", "scene_id": "scene0000_00"}
    assert object_bank_is_complete(
        target,
        expected_scene_id="scene0000_00",
        expected_mode="A1",
        expected_source_lifting=tmp_path / "lifting",
        expected_config={},
        expected_git_commit="commit",
        expected_lifting_identity=lifting_identity,
    )
    assert not object_bank_is_complete(target, expected_git_commit="old")
    assert not object_bank_is_complete(target, expected_config={"changed": True})
    assert not object_bank_is_complete(
        target, expected_lifting_identity={"git_commit": "other"}
    )
    metadata, bank = load_v9_candidate_bank(target, "mv-label")
    assert metadata["association_mode"] == "A1"
    assert bank.core_ids[0].tolist() == list(range(10))
    assert bank.full_ids[0].tolist() == list(range(12))


def test_v9_replay_writes_one_strict_output(tmp_path: Path) -> None:
    bank = _bank(tmp_path)
    diagnostics = replay_v9_scene(
        bank_dir=bank,
        output_root=tmp_path / "replay",
        classifier="mv-label",
        condition="U000",
        category_priors=_priors(tmp_path),
        acceptance_threshold=0.05,
    )
    output_path = tmp_path / "replay/U000/scene0000_00/output.json"
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert output["point_labels"] == [0] * 12 + [-1]
    assert output["instances"]["0"]["class"] == "book"
    assert 0.05 <= output["instances"]["0"]["score"] <= 0.9
    assert diagnostics["coverage"] == 12 / 13


def test_complete_object_bank_resume_does_not_load_lifting_arrays(
    tmp_path: Path, monkeypatch,
) -> None:
    target = _bank(tmp_path)
    metadata_path = target / "object_bank.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config"] = V9Config().as_json()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    lifting = tmp_path / "lifting"
    lifting.mkdir()
    (lifting / "lifting_bank.json").write_text(
        json.dumps(
            {
                "schema": "saga-v9-native-lifting-bank-v1",
                "scene_id": "scene0000_00",
                "identity": metadata["source_lifting_identity"],
            }
        ),
        encoding="utf-8",
    )

    def forbidden_load(_source):
        raise AssertionError("complete-bank resume must not load lifting arrays")

    monkeypatch.setattr(v9_runner, "_load_lifting_bank", forbidden_load)
    resumed = build_v9_object_bank(lifting, target, association_mode="A1")
    assert resumed == metadata
