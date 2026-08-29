from __future__ import annotations

import numpy as np

from category_priors.category_candidate_trace import (
    assert_candidate_bank_identity,
    validate_candidate_formation_trace,
)
from category_priors.category_denoise import (
    build_candidate_bank,
    build_candidate_repair_family,
)


def _classes() -> tuple[str, ...]:
    return ("chair", "table", "wall") + tuple(
        f"class-{index}" for index in range(3, 32)
    )


class _TwoClusterer:
    def __init__(self, labels: np.ndarray) -> None:
        self._labels = np.asarray(labels, dtype=np.int64)
        self.probabilities_ = np.where(self._labels >= 0, 0.8, 0.0)

    def fit_predict(self, distance: np.ndarray) -> np.ndarray:
        assert distance.shape == (len(self._labels), len(self._labels))
        return self._labels.copy()


def _factory(**_: object) -> _TwoClusterer:
    return _TwoClusterer(np.asarray([0, 0, 0, 1, 1, 1, -1, -1]))


def test_one_hdbscan_trace_reproduces_c0_and_builds_contract_safe_repairs() -> None:
    classes = _classes()
    label_features = np.zeros((32, 2), dtype=np.float64)
    label_features[0] = [1.0, 0.0]
    label_features[1] = [0.0, 1.0]
    semantic = np.tile([1.0, 0.0], (8, 1))
    instance = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.7, 0.3],
            [0.3, 0.7],
        ],
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [np.arange(8, dtype=np.float64) * 0.01, np.zeros(8), np.zeros(8)]
    )
    global_labels = np.full(8, -1, dtype=np.int64)

    reference = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair", "table"),
        global_labels,
        1.0,
        hdbscan_factory=_factory,
    )
    reference.diagnostics["scene_id"] = "scene-test"
    family = build_candidate_repair_family(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair", "table"),
        global_labels,
        1.0,
        scene_id="scene-test",
        hdbscan_factory=_factory,
    )

    assert_candidate_bank_identity(reference, family.legacy)
    validate_candidate_formation_trace(
        family.formation_trace, bank=family.legacy
    )
    trace = family.formation_trace
    assigned = trace.legacy_assignment_chosen_raw_cluster >= 0
    np.testing.assert_array_equal(
        trace.legacy_assignment_chosen_raw_cluster,
        trace.prethreshold_argmax_raw_cluster,
    )
    np.testing.assert_allclose(
        trace.legacy_assignment_spatial_similarity[assigned],
        np.exp(
            -trace.legacy_assignment_spatial_distance_standardized[assigned]
        ),
    )
    np.testing.assert_allclose(
        trace.legacy_assignment_hybrid_similarity[assigned],
        0.5 * trace.legacy_assignment_feature_similarity[assigned]
        + 0.5 * trace.legacy_assignment_spatial_similarity[assigned],
    )
    chair_row = next(
        row for row in trace.class_rows if row["branch_class"] == "chair"
    )
    np.testing.assert_allclose(
        chair_row["legacy_assignment_xyz_denominator"],
        [0.07, 1.0, 1.0],
    )
    for repaired in (
        family.consistent_envelope,
        family.raw_anchored_envelope,
    ):
        core = np.asarray(repaired.branch_core_labels)
        full = np.asarray(repaired.branch_full_labels)
        assert np.all((core < 0) | (core == full))
        assert all(
            row["trusted_core_point_count"] == row["core_point_count"]
            for row in repaired.candidates
        )
        assert repaired.global_pre_knn.tolist() == global_labels.tolist()
