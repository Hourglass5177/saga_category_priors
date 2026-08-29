from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from category_priors.category_candidate_representation import (
    _feature_path,
    _mapped_object_gaussians,
    local_affinity_edge_auc,
    oracle_seed_candidate_mask,
)


def test_local_affinity_auc_separates_same_and_different_instances() -> None:
    xyz = np.asarray(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.asarray(
        [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64
    )
    result = local_affinity_edge_auc(
        xyz,
        affinity,
        np.asarray([1, 1, 2, 2]),
        np.asarray([0, 0, 0, 0]),
        k=3,
    )
    assert result["affinity_edge_auroc"] == 1.0
    assert result["positive_edge_count"] == 4


def test_oracle_seed_uses_frozen_q95_envelope_not_gt_mask() -> None:
    xyz = np.asarray(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [1, 0, 0]],
        dtype=np.float64,
    )
    affinity = np.asarray(
        [[1, 0], [1, 0], [1, 0], [0, 1]], dtype=np.float64
    )
    mask = oracle_seed_candidate_mask(
        selected_indices=np.arange(4),
        sampled_object_indices=np.asarray([0, 1, 2]),
        xyz_scene=xyz,
        affinity=affinity,
        semantic_score=np.ones(4),
        instance_distance_max=1.0,
        spatial_distance_max=1.0,
    )
    assert mask.tolist() == [True, True, True, False]


def test_feature_path_falls_back_to_native_scene_asset(tmp_path: Path) -> None:
    feature = tmp_path / "saga" / "contrastive_feature_point_cloud.ply"
    feature.parent.mkdir(parents=True)
    feature.touch()

    assert _feature_path({"base_path": str(tmp_path)}) == feature


def test_object_mapping_reads_nearest_point_indices_not_wrapper_object() -> None:
    mapping = SimpleNamespace(
        gt_to_gaussian=SimpleNamespace(indices=np.asarray([4, -1, 7, 2]))
    )

    np.testing.assert_array_equal(
        _mapped_object_gaussians(mapping, np.asarray([0, 2, 3])),
        np.asarray([4, 7, 2]),
    )
