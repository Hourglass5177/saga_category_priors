from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from category_priors.category_feature_routing_factorial import (
    RUNTIME_CLASSES,
    build_raw_cluster_bank,
    gt_class_route,
    infer_root_cause,
    material_gain,
)
from category_priors.taxonomy import load_taxonomy


class _OneCluster:
    probabilities_ = np.array([0.7, 0.8, 0.9])

    def fit_predict(self, distance):
        assert distance.shape == (3, 3)
        return np.array([0, 0, 0])


def _factory(**kwargs):
    assert kwargs["min_cluster_size"] == 3
    assert kwargs["min_samples"] == 3
    assert kwargs["metric"] == "precomputed"
    return _OneCluster()


def test_raw_bank_stops_before_expansion_knn_filter_and_prior() -> None:
    count = 5
    chair = RUNTIME_CLASSES.index("chair")
    bank = build_raw_cluster_bank(
        affinity=np.eye(count, 32),
        xyz_m=np.arange(count * 3, dtype=float).reshape(count, 3) * 0.01,
        top_class=np.full(count, chair),
        route_score=np.ones(count),
        branch_class=np.full(count, chair),
        global_typical_diag_m=1.0,
        scene_id="scene0645_00",
        feature_source="native-2k-grounded",
        route="predicted-32-top1",
        sample_cap=3,
        hdbscan_factory=_factory,
    )
    assert np.count_nonzero(bank.branch_full_labels >= 0) == 3
    assert np.array_equal(bank.branch_full_labels, bank.branch_core_labels)
    assert bank.diagnostics["full_expansion_used"] is False
    assert bank.diagnostics["knn_used"] is False
    assert bank.diagnostics["filter_used"] is False
    assert bank.diagnostics["category_prior_used"] is False


def test_gt_route_uses_class_but_not_instance_identity() -> None:
    taxonomy = load_taxonomy()
    chair_id = taxonomy.canonical_classes.index("chair")
    table_id = taxonomy.canonical_classes.index("table")
    top, score, branch = gt_class_route(
        np.array([0, 0, 1, -1]),
        np.array([chair_id, table_id]),
        taxonomy,
    )
    assert branch.tolist() == [RUNTIME_CLASSES.index("chair")] * 2 + [
        RUNTIME_CLASSES.index("table"), -1
    ]
    assert score.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert top.shape == branch.shape


def test_material_gain_requires_quality_and_candidate_safety() -> None:
    baseline = {
        "candidate_count": 100,
        "same_class_iou_025_count": 2,
        "tiny_small_recall_025": 0.1,
    }
    assert material_gain(
        {"candidate_count": 120, "same_class_iou_025_count": 4,
         "tiny_small_recall_025": 0.1}, baseline
    )
    assert not material_gain(
        {"candidate_count": 151, "same_class_iou_025_count": 4,
         "tiny_small_recall_025": 0.3}, baseline
    )


def test_root_cause_decision_keeps_prior_uninterpreted() -> None:
    def row(count025, *, count=100, tiny=0.1, count050=0):
        return {
            "candidate_count": count,
            "same_class_iou_025_count": count025,
            "same_class_iou_050_count": count050,
            "tiny_small_recall_025": tiny,
        }

    result = infer_root_cause(
        {
            "native-2k-grounded__predicted-32-top1": row(1),
            "native-2k-grounded__gt-class-oracle": row(4),
            "v9-10k-dual-source__predicted-32-top1": row(1),
            "v9-10k-dual-source__gt-class-oracle": row(4),
        }
    )
    assert result["root_cause"] == "semantic-routing-dominant"
    assert result["category_prior_tested"] is False
