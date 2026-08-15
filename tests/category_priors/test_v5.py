from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.io import load_json, write_json
from category_priors.v5_calibrator import fit_v5_calibrator
from category_priors.v5_candidate import (
    V5CandidateConfig,
    normalized_top1,
    score_candidate,
    source_masks,
)
from category_priors.v5_candidate_runner import build_v5_candidate_command, v5_candidate_run_paths
from category_priors.v5_replay import materialize_v5_b1_baseline, replay_v5_proposals


def _priors() -> dict:
    geometry = {
        "log_extent_short_m": {"q50": -2.0, "q75": -1.0},
        "log_extent_mid_m": {"q50": -1.0, "q75": 0.0},
        "log_extent_long_m": {"q50": 0.0, "q75": 1.0},
        "log_bbox_diag_m": {"q50": 0.0, "q75": 1.0},
        "log_surface_area_m2": {"q50": 0.0, "q75": 1.0},
    }
    return {"global": {"shrunk": {"geometry": geometry}}, "categories": {"chair": {"shrunk": {"geometry": geometry}}}}


def _candidate(candidate_id: int, class_name: str = "chair") -> dict:
    return {
        "candidate_id": candidate_id, "branch_class": class_name,
        "assignment_confidence_mean": 0.8, "hdbscan_membership_mean": 0.8,
        "hdbscan_persistence": 0.2, "core_assignment_points": 120,
        "metric_extents_m": [0.2, 0.5, 1.0], "local_surface_density": 100.0,
        "vote": {"branch_class_ratio": 0.8, "background_ratio": 0.1, "winner_matches_branch": True},
    }


def _candidate_bank(root: Path, source: str, scene: str, seed: int, labels: list[int], core: list[int], candidates: list[dict]) -> None:
    paths = v5_candidate_run_paths(root, source, scene, seed)
    paths["run_dir"].mkdir(parents=True)
    write_json(paths["output"], {"point_labels": labels, "instances": {"0": {"class": "table"}}})
    write_json(paths["diagnostics"], {"status": "complete", "instances": {"0": {"class": "table", "score": 1.0}}})
    write_json(paths["proposals"], {"kind": "v5_proposal_bank", "source": source, "candidates": candidates})
    np.savez_compressed(
        paths["proposal_labels"], branch_labels=np.asarray(core, dtype=np.int32),
        core_labels=np.asarray(core, dtype=np.int32), assignment_confidence=np.ones(len(core)),
        semantic_winner=np.zeros(len(core), dtype=np.int16), semantic_score=np.ones(len(core)),
        semantic_margin=np.ones(len(core)), source_view_count=np.zeros(len(core), dtype=np.int16),
        source_vote_ratio=np.ones(len(core)), source_vote_margin=np.ones(len(core)),
    )


def test_normalized_top1_uses_full_codebook_and_multiview_filters() -> None:
    points = np.asarray([[10.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    winner, score, margin = normalized_top1(points, labels)
    assert winner.tolist() == [0, 1, 0]
    masks = source_masks(
        source="multiview", winner=winner, score=score,
        class_indices={"chair": 0, "wall": 1}, multiview_views=np.asarray([3, 3, 2]),
        multiview_ratio=np.asarray([0.7, 0.9, 0.9]), multiview_margin=np.asarray([0.2, 0.2, 0.2]),
        config=V5CandidateConfig(),
    )
    assert masks[0].tolist() == [True, False, False]
    assert 1 not in masks
    assert margin[0] > 0


def test_score_changes_only_score_components() -> None:
    candidate = _candidate(0)
    priors = _priors()
    uniform = score_candidate(candidate, priors, "U00-uniform")
    combined = score_candidate(candidate, priors, "D11-combined")
    assert uniform["E"] == combined["E"]
    assert uniform["G"] == combined["G"]
    assert uniform["C"] == combined["C"]
    assert uniform["score"] == uniform["E"]
    assert combined["score"] == pytest.approx(combined["E"] * combined["G"] * combined["C"])


def test_replay_never_overwrites_existing_or_other_class_points(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    outputs = tmp_path / "outputs"
    priors_path = tmp_path / "priors.json"
    write_json(priors_path, _priors())
    # A weak cross-class overlap cannot relabel the table point: the proposal is
    # allowed to create a new instance only on the remaining B1 background.
    labels = [0] + [-1] * 119
    branch = [0] * 120
    _candidate_bank(candidate_root, "codebook", "scene0000_00", 42, labels, branch, [_candidate(0, "chair")])
    replay_v5_proposals(
        candidate_root=candidate_root, output_root=outputs, source="codebook",
        conditions=["U00-uniform"], scene_ids=["scene0000_00"], seeds=[42], category_priors=priors_path,
    )
    replayed = load_json(outputs / "U00-uniform" / "scene0000_00" / "seed-42" / "output.json")
    assert replayed["point_labels"][0] == 0
    assert all(value == 1 for value in replayed["point_labels"][1:])
    diagnostics = load_json(outputs / "U00-uniform" / "scene0000_00" / "seed-42" / "diagnostics.json")
    assert diagnostics["v5_proposal_replay"]["accepted_count"] == 1


def test_materialized_b1_and_runner_paths_are_isolated(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    _candidate_bank(candidate_root, "codebook", "scene0000_00", 42, [-1] * 3, [-1] * 3, [])
    result = materialize_v5_b1_baseline(
        candidate_root=candidate_root, output_root=tmp_path / "output", source="codebook",
        scene_ids=["scene0000_00"], seeds=[42],
    )
    assert result["runs"]
    baseline = load_json(tmp_path / "output" / "B1-original" / "scene0000_00" / "seed-42" / "output.json")
    assert baseline["point_labels"] == [-1] * 3
    command, _ = build_v5_candidate_command(
        pipeline="/tmp/run_pipeline.sh", scene={"base_path": "/tmp/base", "python_bin": "/tmp/python", "scene_scale_m_per_unit": 1.0},
        output_root=tmp_path, source="codebook", scene_id="scene0000_00", seed=42, git_commit="abc",
    )
    assert "--v5-candidate-source" in command
    assert "--max-contributor-cache-path" not in command


def test_calibrator_reads_only_declared_development_scenes(tmp_path: Path) -> None:
    rows = []
    for index, scene in enumerate(("dev", "holdout")):
        for positive in (0, 1):
            rows.append({
                "row_type": "candidate", "source": "codebook", "scene_id": scene,
                "same_class_best_iou": 0.8 if positive else 0.0,
                "uniform_E": 0.1 + positive, "uniform_G": 0.2 + positive, "uniform_C": 0.3 + positive,
                "class_E": 0.1 + positive, "class_G": 0.2 + positive, "class_C": 0.3 + positive,
            })
    table = tmp_path / "rows.jsonl"
    table.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "calibrator.json"
    payload = fit_v5_calibrator(table, output, source="codebook", development_scenes=["dev"])
    assert payload["uniform"]["candidate_count"] == 2
    assert payload["development_scenes"] == ["dev"]
