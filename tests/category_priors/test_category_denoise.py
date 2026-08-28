from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from category_priors.category_denoise import (
    ASSIGNMENT_THRESHOLD,
    CLUSTER_SELECTION_EPSILON,
    INSTANCE_WEIGHT,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    SAMPLE_CAP,
    SEMANTIC_THRESHOLD,
    SEMANTIC_WEIGHT,
    SPATIAL_WEIGHT,
    CandidateBank,
    attach_candidate_votes,
    boundary_fixed_ratio_5cm,
    build_candidate_bank,
    build_strict_prediction_metadata,
    load_candidate_bank,
    normalized_top1_32,
    pca_sorted_extents_m,
    replay_protected_denoise,
    save_candidate_bank,
    score_bank_candidates,
    size_compatibility,
    smoothness_compatibility,
    stable_class_seed,
    support_threshold,
)


def _box_corners(extents: tuple[float, float, float]) -> np.ndarray:
    half = np.asarray(extents, dtype=np.float64) / 2.0
    return np.asarray(
        [
            (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )


def _class_names() -> tuple[str, ...]:
    return ("chair", "table", "wall") + tuple(
        f"class-{index}" for index in range(3, 32)
    )


def _prior_node(
    *,
    area: float,
    extent_q50: float = 1.0,
    extent_q75: float = 2.0,
    boundary_q50: float = 0.10,
    boundary_q75: float = 0.20,
) -> dict[str, object]:
    return {
        "shrunk": {
            "geometry": {
                "log_extent_short_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_extent_mid_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_extent_long_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_surface_area_m2": {"q50": math.log(area)},
            },
            "neighborhood": {
                "boundary_fixed:0.05": {
                    "q50": boundary_q50,
                    "q75": boundary_q75,
                }
            },
        }
    }


def _priors() -> dict[str, object]:
    return {
        "global": _prior_node(area=1.0),
        "categories": {"chair": _prior_node(area=0.36)},
    }


def _manual_bank(*, core_count: int = 3) -> CandidateBank:
    class_names = _class_names()
    full_count = max(3, core_count)
    count = full_count + 1
    core_labels = np.full(count, -1, dtype=np.int64)
    core_labels[:core_count] = 0
    full_labels = np.full(count, -1, dtype=np.int64)
    full_labels[:full_count] = 0
    global_labels = np.full(count, -1, dtype=np.int64)
    global_labels[:full_count] = 7
    return CandidateBank(
        class_names=class_names,
        saga20_names=("chair", "table"),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=global_labels,
        semantic_top1=np.zeros(count, dtype=np.int64),
        semantic_top1_score=np.ones(count, dtype=np.float64),
        branch_full_labels=full_labels,
        branch_core_labels=core_labels,
        assignment_confidence=np.concatenate(
            [np.full(full_count, 0.8), np.asarray([0.0])]
        ),
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "branch_class_index": 0,
                "core_point_count": core_count,
                "full_point_count": full_count,
                "assignment_confidence_mean": 0.8,
                "metric_extents_m": [1.0, 1.0, 1.0],
                "boundary_ratio_5cm": 0.10,
            },
        ),
        diagnostics={},
    )


def test_normalized_top1_competes_over_all_32_before_saga20_filtering() -> None:
    classes = _class_names()
    label_features = np.zeros((32, 3), dtype=np.float64)
    label_features[0] = [10.0, 0.0, 0.0]
    label_features[1] = [0.0, 10.0, 0.0]
    label_features[2] = [1.0, 1.0, 0.0]  # non-SAGA20 winner for row 1
    semantic = np.asarray(
        [[4.0, 0.0, 0.0], [3.0, 3.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    forward = normalized_top1_32(
        semantic, label_features, classes, ("chair", "table")
    )
    reverse = normalized_top1_32(
        semantic, label_features, classes, ("table", "chair")
    )

    np.testing.assert_array_equal(forward.top_class_index, [0, 2, 0])
    np.testing.assert_array_equal(forward.eligible_mask, [True, False, False])
    np.testing.assert_array_equal(forward.branch_class_index, [0, -1, -1])
    np.testing.assert_array_equal(reverse.top_class_index, forward.top_class_index)
    np.testing.assert_array_equal(reverse.eligible_mask, forward.eligible_mask)


def test_candidate_bank_freezes_original_parameters_core_and_class_order() -> None:
    classes = _class_names()
    label_features = np.zeros((32, 2), dtype=np.float64)
    label_features[0] = [1.0, 0.0]
    label_features[1] = [0.0, 1.0]
    semantic = np.tile([1.0, 0.0], (6, 1))
    instance = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]] * 2,
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [np.arange(6, dtype=np.float64) * 0.01, np.zeros(6), np.zeros(6)]
    )
    calls: list[dict[str, object]] = []

    class FakeClusterer:
        def fit_predict(self, distance: np.ndarray) -> np.ndarray:
            assert distance.shape == (6, 6)
            return np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64)

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeClusterer()

    forward = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair", "table"),
        np.full(6, -1, dtype=np.int64),
        1.0,
        hdbscan_factory=factory,
    )
    reverse = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("table", "chair"),
        np.full(6, -1, dtype=np.int64),
        1.0,
        hdbscan_factory=factory,
    )

    assert calls[0] == {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_epsilon": CLUSTER_SELECTION_EPSILON,
        "allow_single_cluster": False,
        "metric": "precomputed",
    }
    assert SEMANTIC_THRESHOLD == 0.7
    assert SAMPLE_CAP == 5_000
    assert (INSTANCE_WEIGHT, SPATIAL_WEIGHT, SEMANTIC_WEIGHT) == (0.5, 0.3, 0.2)
    assert ASSIGNMENT_THRESHOLD == 0.3
    assert len(forward.candidates) == 1
    assert int(np.count_nonzero(forward.branch_core_labels == 0)) == 3
    assert int(np.count_nonzero(forward.branch_full_labels == 0)) == 6
    np.testing.assert_array_equal(reverse.branch_core_labels, forward.branch_core_labels)
    np.testing.assert_array_equal(reverse.branch_full_labels, forward.branch_full_labels)
    assert reverse.candidates == forward.candidates


def test_stable_class_seed_is_repeatable_and_class_specific() -> None:
    chair_first = stable_class_seed(42, "chair")
    chair_second = stable_class_seed(42, "chair")

    assert chair_first == chair_second
    assert chair_first != stable_class_seed(42, "cup")
    assert chair_first != stable_class_seed(43, "chair")


def test_pca_extents_use_metric_scale_and_are_rotation_invariant() -> None:
    points = _box_corners((2.0, 1.0, 0.5))
    angle = math.radians(37.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    original = pca_sorted_extents_m(points, scene_scale_m_per_unit=2.0)
    rotated = pca_sorted_extents_m(points @ rotation.T, scene_scale_m_per_unit=2.0)

    np.testing.assert_allclose(original, np.asarray([1.0, 2.0, 4.0]), atol=1e-10)
    np.testing.assert_allclose(rotated, original, atol=1e-10)


def test_boundary_fixed_ratio_matches_train_prior_boundary_point_definition() -> None:
    # Only candidate point 0 sees an outside point within 5 cm.  The expected
    # statistic is therefore one boundary *point* out of three candidate
    # points.  Counting crossing edges instead would produce a different value.
    xyz = np.asarray(
        [
            [0.000, 0.0, 0.0],
            [0.010, 0.0, 0.0],
            [0.020, 0.0, 0.0],
            [-0.049, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    candidate = np.asarray([True, True, True, False])

    ratio = boundary_fixed_ratio_5cm(
        xyz, candidate, scene_scale_m_per_unit=1.0
    )

    assert ratio == 1.0 / 3.0


def test_boundary_fixed_ratio_is_zero_without_an_outside_neighbor() -> None:
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [1.00, 0.0, 0.0]],
        dtype=np.float64,
    )
    candidate = np.asarray([True, True, False])

    assert boundary_fixed_ratio_5cm(
        xyz, candidate, scene_scale_m_per_unit=1.0
    ) == 0.0


def test_uniform_and_class_score_the_same_bank_without_mutating_it(tmp_path) -> None:
    bank = _manual_bank(core_count=3)
    ratios = np.zeros(32, dtype=np.float64)
    ratios[0] = 0.70
    voted = attach_candidate_votes(bank, {0: ratios}, bank.class_names)
    full_before = voted.branch_full_labels.copy()
    core_before = voted.branch_core_labels.copy()
    candidates_before = tuple(dict(row) for row in voted.candidates)

    uniform = score_bank_candidates(voted, _priors(), "uniform")
    per_class = score_bank_candidates(voted, _priors(), "class")

    assert uniform[0]["Q"] == per_class[0]["Q"] == pytest.approx(0.56)
    assert (
        uniform[0]["ap_score"]
        == per_class[0]["ap_score"]
        == pytest.approx(0.56)
    )
    assert uniform[0]["support_threshold"] == 5
    assert not uniform[0]["accepted"]
    assert per_class[0]["support_threshold"] == 3
    assert per_class[0]["accepted"]
    np.testing.assert_array_equal(voted.branch_full_labels, full_before)
    np.testing.assert_array_equal(voted.branch_core_labels, core_before)
    assert voted.candidates == candidates_before

    save_candidate_bank(voted, tmp_path / "bank")
    loaded = load_candidate_bank(tmp_path / "bank")
    np.testing.assert_array_equal(loaded.branch_full_labels, full_before)
    np.testing.assert_array_equal(loaded.branch_core_labels, core_before)
    assert loaded.candidates == candidates_before


def test_missing_class_statistics_fall_back_to_global_support() -> None:
    assert support_threshold(_priors(), "table", "class") == 5


def test_size_and_smoothness_are_one_sided_and_equal_exp_half_at_q75() -> None:
    node = _prior_node(area=1.0)
    at_median = {
        "metric_extents_m": [0.5, 0.75, 1.0],
        "boundary_ratio_5cm": 0.05,
    }
    at_upper_quartile = {
        "metric_extents_m": [2.0, 2.0, 2.0],
        "boundary_ratio_5cm": 0.20,
    }

    assert size_compatibility(at_median, node) == pytest.approx(1.0)
    assert smoothness_compatibility(at_median, node) == pytest.approx(1.0)
    assert size_compatibility(at_upper_quartile, node) == pytest.approx(
        math.exp(-0.5)
    )
    assert smoothness_compatibility(at_upper_quartile, node) == pytest.approx(
        math.exp(-0.5)
    )


@pytest.mark.parametrize(
    ("core_count", "expected"), ((4, False), (5, True))
)
def test_uniform_support_threshold_keeps_count_equal_to_five(
    core_count: int, expected: bool
) -> None:
    bank = _manual_bank(core_count=core_count)
    ratios = np.zeros(32, dtype=np.float64)
    ratios[0] = 0.70
    voted = attach_candidate_votes(bank, {0: ratios}, bank.class_names)

    assert score_bank_candidates(voted, _priors(), "uniform")[0]["accepted"] is expected


def test_background_must_not_lose_to_branch_for_candidate_acceptance() -> None:
    bank = _manual_bank(core_count=5)
    ratios = np.zeros(32, dtype=np.float64)
    ratios[0] = 0.35
    ratios[1] = 0.05
    # The unassigned/background share is 0.60, so background is the true
    # winner even though chair wins among the 32 foreground classes.
    voted = attach_candidate_votes(bank, {0: ratios}, bank.class_names)

    decision = score_bank_candidates(voted, _priors(), "uniform")[0]

    assert not decision["accepted"]


def test_protected_three_point_instance_bypasses_knn_and_global_count_filter() -> None:
    bank = _manual_bank(core_count=3)
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]],
        dtype=np.float64,
    )
    accepted = [{"candidate_id": 0, "accepted": True, "ap_score": 0.4}]

    labels, classes, scores, diagnostics = replay_protected_denoise(
        xyz, bank, accepted, k=256, min_count=10
    )

    np.testing.assert_array_equal(labels, [0, 0, 0, -1])
    assert classes == {0: "chair"}
    assert scores == {0: 0.4}
    assert diagnostics["protected_gaussian_count"] == 3
    assert diagnostics["knn_k_effective"] == 1
    assert diagnostics["protected_instance_survival_rate"] == 1.0
    assert diagnostics["protected_class_rewrite_rate"] == 0.0


def test_protected_points_are_removed_from_neighbor_votes_and_rejected_points_fallback() -> None:
    bank = _manual_bank(core_count=3)
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.03, 0.0, 0.0]],
        dtype=np.float64,
    )

    protected, _, _, _ = replay_protected_denoise(
        xyz,
        bank,
        [{"candidate_id": 0, "accepted": True, "ap_score": 0.4}],
        k=3,
        min_count=1,
    )
    fallback, _, _, _ = replay_protected_denoise(
        xyz, bank, [], k=3, min_count=1
    )

    # Point 3 stays background only if the three protected label-7 points are
    # absent from the KNN source tree, not merely overwritten after voting.
    assert protected[3] == -1
    np.testing.assert_array_equal(fallback, [7, 7, 7, 7])


def test_replay_handles_all_points_protected_without_constructing_a_knn() -> None:
    bank = _manual_bank(core_count=3)
    candidate = dict(bank.candidates[0])
    candidate["full_point_count"] = 4
    bank = replace(
        bank,
        branch_full_labels=np.zeros(4, dtype=np.int64),
        branch_core_labels=np.asarray([0, 0, 0, -1], dtype=np.int64),
        candidates=(candidate,),
    )
    xyz = np.column_stack(
        [np.arange(4, dtype=np.float64), np.zeros(4), np.zeros(4)]
    )

    labels, _, _, diagnostics = replay_protected_denoise(
        xyz,
        bank,
        [{"candidate_id": 0, "accepted": True, "ap_score": 0.4}],
    )

    np.testing.assert_array_equal(labels, [0, 0, 0, 0])
    assert diagnostics["knn_k_effective"] == 0


def test_strict_metadata_removes_sparse_internal_ids_without_orphans() -> None:
    labels = np.asarray([5, 5, -1, 9, 9], dtype=np.int64)
    xyz = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
        dtype=np.float64,
    )

    result = build_strict_prediction_metadata(
        labels,
        xyz,
        {5: "chair", 9: "table"},
        {5: 0.8, 9: 0.6},
    )

    np.testing.assert_array_equal(result.point_labels, [0, 0, -1, 1, 1])
    assert tuple(result.instances) == ("0", "1")
    assert result.audit["orphan_gaussian_count"] == 0
    assert all(len(row["bbox"]) == 24 for row in result.instances.values())
