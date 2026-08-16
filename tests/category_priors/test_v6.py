from __future__ import annotations

from pathlib import Path

import numpy as np

from category_priors.v6_candidate import (
    V6GraphConfig,
    build_affinity_components,
    finalise_multiview_candidates,
    normalized_top1,
)
from category_priors.v6_candidate_runner import build_v6_candidate_command, v6_candidate_run_paths


def _two_components() -> tuple[np.ndarray, np.ndarray]:
    first = np.column_stack((np.arange(12, dtype=float) * 0.01, np.zeros(12), np.zeros(12)))
    second = np.column_stack((10.0 + np.arange(12, dtype=float) * 0.01, np.zeros(12), np.zeros(12)))
    xyz = np.vstack((first, second))
    features = np.vstack((np.tile([1.0, 0.0], (12, 1)), np.tile([0.0, 1.0], (12, 1))))
    return xyz, features


def test_affinity_components_are_deterministic_and_have_no_semantic_input() -> None:
    xyz, features = _two_components()
    config = V6GraphConfig(physical_neighbors=4, affinity_neighbors=4, core_degree=1, min_core_points=3)
    first = build_affinity_components(xyz, features, config)
    second = build_affinity_components(xyz, features, config)
    assert np.array_equal(first["full_labels"], second["full_labels"])
    assert len(first["candidates"]) == 2
    assert np.all(first["full_labels"][:12] == first["full_labels"][0])
    assert np.all(first["full_labels"][12:] == first["full_labels"][12])
    assert first["full_labels"][0] != first["full_labels"][12]
    assert first["edge_left"].max() < len(xyz)


def test_multiview_finalisation_requires_saga20_winner_and_evidence() -> None:
    xyz, features = _two_components()
    config = V6GraphConfig(physical_neighbors=4, affinity_neighbors=4, core_degree=1, min_core_points=3)
    components = build_affinity_components(xyz, features, config)
    votes = np.asarray([[4, 0], [0, 4]], dtype=np.int16)
    finalised = finalise_multiview_candidates(components, votes, ["chair", "wall"], config)
    assert len(finalised["candidates"]) == 1
    assert finalised["candidates"][0]["branch_class"] == "chair"
    assert np.all(finalised["full_labels"][:12] == 0)
    assert np.all(finalised["full_labels"][12:] == -1)


def test_codebook_competes_over_all_classes_after_l2_normalisation() -> None:
    winner, score, margin = normalized_top1(
        np.asarray([[10.0, 0.0], [0.5, 0.5]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
    )
    assert winner.tolist() == [0, 0]
    assert score[0] == 1.0
    assert margin.shape == score.shape


def test_v6_runner_paths_are_seed_isolated_and_do_not_request_cache(tmp_path: Path) -> None:
    left = v6_candidate_run_paths(tmp_path, "scene0000_00", 42)
    right = v6_candidate_run_paths(tmp_path, "scene0000_00", 3407)
    assert left["output"] != right["output"]
    command, paths = build_v6_candidate_command(
        pipeline="/tmp/run_pipeline.sh",
        scene={"base_path": "/tmp/base", "python_bin": "/tmp/python", "scene_scale_m_per_unit": 1.0},
        output_root=tmp_path, scene_id="scene0000_00", seed=42, git_commit="abc",
    )
    assert "--v6-candidate-mode" in command
    assert "--max-contributor-cache-path" not in command
    assert paths["proposal_labels"].name == "v6_proposal_labels.npz"
