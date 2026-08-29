from __future__ import annotations

import numpy as np

from category_priors.category_candidate_representation import (
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
