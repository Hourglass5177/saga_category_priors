from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from category_priors.candidate_bank import (
    ASSIGNMENT_THRESHOLD,
    CLUSTER_SELECTION_EPSILON,
    INSTANCE_WEIGHT,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    SAMPLE_CAP,
    SCHEMA,
    SEMANTIC_THRESHOLD,
    SEMANTIC_WEIGHT,
    SPATIAL_WEIGHT,
    assert_candidate_bank_matches_inputs,
    attach_candidate_votes,
    build_candidate_bank,
    load_candidate_bank,
    normalized_top1_32,
    save_candidate_bank,
    stable_class_seed,
)
from category_priors.geometry import pca_sorted_extents_m
from category_priors.legacy_candidate_replay import legacy_knn_filter


def _class_names() -> tuple[str, ...]:
    return ("chair", "table", "wall") + tuple(
        f"class-{index}" for index in range(3, 32)
    )


def test_top1_competes_over_all_32_before_saga20_filtering() -> None:
    classes = _class_names()
    label_features = np.zeros((32, 3), dtype=np.float64)
    label_features[0] = [10.0, 0.0, 0.0]
    label_features[1] = [0.0, 10.0, 0.0]
    label_features[2] = [1.0, 1.0, 0.0]
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


def test_candidate_bank_freezes_registered_generation_contract() -> None:
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


def test_candidate_bank_side_path_does_not_change_prediction_or_global_rng() -> None:
    classes = _class_names()
    label_features = np.zeros((32, 2), dtype=np.float64)
    label_features[0] = [1.0, 0.0]
    semantic = np.tile([1.0, 0.0], (6, 1))
    instance = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]] * 2,
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [np.arange(6, dtype=np.float64) * 0.01, np.zeros(6), np.zeros(6)]
    )
    global_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)

    class FakeClusterer:
        def fit_predict(self, distance: np.ndarray) -> np.ndarray:
            return np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64)

    def factory(**_kwargs):
        return FakeClusterer()

    before = legacy_knn_filter(xyz, global_labels, k=3, min_count=1)
    frozen_labels = global_labels.copy()
    np.random.seed(20260903)
    expected_random = np.random.random(4)
    np.random.seed(20260903)

    bank = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair",),
        global_labels,
        1.0,
        hdbscan_factory=factory,
    )
    after = legacy_knn_filter(xyz, global_labels, k=3, min_count=1)
    observed_random = np.random.random(4)

    np.testing.assert_array_equal(global_labels, frozen_labels)
    np.testing.assert_array_equal(bank.global_pre_knn, frozen_labels)
    np.testing.assert_array_equal(after.after_knn, before.after_knn)
    np.testing.assert_array_equal(after.after_filter, before.after_filter)
    np.testing.assert_array_equal(observed_random, expected_random)


def test_candidate_bank_round_trip_and_votes_do_not_change_membership(tmp_path) -> None:
    classes = _class_names()
    label_features = np.zeros((32, 2), dtype=np.float64)
    label_features[0] = [1.0, 0.0]
    semantic = np.tile([1.0, 0.0], (6, 1))
    instance = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]] * 2,
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [np.arange(6, dtype=np.float64) * 0.01, np.zeros(6), np.zeros(6)]
    )

    class FakeClusterer:
        def fit_predict(self, _distance: np.ndarray) -> np.ndarray:
            return np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64)

    bank = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair",),
        np.full(6, -1, dtype=np.int64),
        1.0,
        hdbscan_factory=lambda **_kwargs: FakeClusterer(),
    )
    membership = bank.branch_full_labels.copy()
    ratios = np.zeros(32, dtype=np.float64)
    ratios[0] = 0.7
    voted = attach_candidate_votes(bank, {0: ratios}, classes)
    np.testing.assert_array_equal(voted.branch_full_labels, membership)
    assert voted.candidates[0]["branch_vote_ratio"] == 0.7

    save_candidate_bank(voted, tmp_path / "bank")
    loaded = load_candidate_bank(tmp_path / "bank")
    np.testing.assert_array_equal(loaded.branch_full_labels, membership)
    assert loaded.candidates == voted.candidates

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        save_candidate_bank(voted, tmp_path / "bank")


def test_candidate_bank_input_identity_includes_xyz_order() -> None:
    classes = _class_names()
    label_features = np.zeros((32, 2), dtype=np.float64)
    label_features[0] = [1.0, 0.0]
    semantic = np.tile([1.0, 0.0], (6, 1))
    instance = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]] * 2,
        dtype=np.float64,
    )
    xyz = np.column_stack(
        [np.arange(6, dtype=np.float64) * 0.01, np.zeros(6), np.zeros(6)]
    )
    global_labels = np.full(6, -1, dtype=np.int64)

    class FakeClusterer:
        def fit_predict(self, _distance: np.ndarray) -> np.ndarray:
            return np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64)

    bank = build_candidate_bank(
        instance,
        semantic,
        xyz,
        label_features,
        classes,
        ("chair",),
        global_labels,
        1.0,
        hdbscan_factory=lambda **_kwargs: FakeClusterer(),
    )
    assert_candidate_bank_matches_inputs(
        bank,
        xyz_scene=xyz,
        global_pre_knn=global_labels,
        instance_features=instance,
        semantic_features=semantic,
        label_features=label_features,
        class_names=classes,
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.0,
        seed=42,
    )
    with pytest.raises(ValueError, match="coordinates/order"):
        assert_candidate_bank_matches_inputs(
            bank,
            xyz_scene=xyz[::-1],
            global_pre_knn=global_labels,
            instance_features=instance,
            semantic_features=semantic,
            label_features=label_features,
            class_names=classes,
            saga20_names=("chair",),
            scene_scale_m_per_unit=1.0,
            seed=42,
        )
    with pytest.raises(ValueError, match="SAGA20 branch table"):
        assert_candidate_bank_matches_inputs(
            bank,
            xyz_scene=xyz,
            global_pre_knn=global_labels,
            instance_features=instance,
            semantic_features=semantic,
            label_features=label_features,
            class_names=classes,
            saga20_names=("chair", "table"),
            scene_scale_m_per_unit=1.0,
            seed=42,
        )
    modified_instance = instance.copy()
    modified_instance[0, 0] -= 0.01
    with pytest.raises(ValueError, match="feature/input fingerprints"):
        assert_candidate_bank_matches_inputs(
            bank,
            xyz_scene=xyz,
            global_pre_knn=global_labels,
            instance_features=modified_instance,
            semantic_features=semantic,
            label_features=label_features,
            class_names=classes,
            saga20_names=("chair",),
            scene_scale_m_per_unit=1.0,
            seed=42,
        )

    diagnostics = dict(bank.diagnostics)
    contract = dict(diagnostics["generation_contract"])
    contract["semantic_threshold"] = 0.6
    diagnostics["generation_contract"] = contract
    incompatible = replace(bank, diagnostics=diagnostics)
    with pytest.raises(ValueError, match="generation contract"):
        assert_candidate_bank_matches_inputs(
            incompatible,
            xyz_scene=xyz,
            global_pre_knn=global_labels,
            instance_features=instance,
            semantic_features=semantic,
            label_features=label_features,
            class_names=classes,
            saga20_names=("chair",),
            scene_scale_m_per_unit=1.0,
            seed=42,
        )


def test_class_seed_is_stable_and_class_specific() -> None:
    assert stable_class_seed(42, "chair") == stable_class_seed(42, "chair")
    assert stable_class_seed(42, "chair") != stable_class_seed(42, "cup")
    assert stable_class_seed(42, "chair") != stable_class_seed(43, "chair")


def test_retired_denoising_bank_schema_is_not_silently_reused() -> None:
    assert SCHEMA == "saga-instance-recheck-candidate-bank-v1"


def test_metric_pca_extents_are_rotation_invariant() -> None:
    half = np.asarray([1.0, 0.5, 0.25])
    points = np.asarray(
        [
            (sx * half[0], sy * half[1], sz * half[2])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    angle = math.radians(37.0)
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    original = pca_sorted_extents_m(points, scene_scale_m_per_unit=2.0)
    rotated = pca_sorted_extents_m(points @ rotation.T, scene_scale_m_per_unit=2.0)
    np.testing.assert_allclose(original, [1.0, 2.0, 4.0], atol=1e-10)
    np.testing.assert_allclose(rotated, original, atol=1e-10)
