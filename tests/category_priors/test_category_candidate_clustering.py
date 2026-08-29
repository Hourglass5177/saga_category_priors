from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from category_priors.category_candidate_clustering import (
    AFFINITY_NEIGHBORS,
    AFFINITY_WEIGHT,
    CLUSTER_SELECTION_EPSILON,
    MIN_CLUSTER_SIZE,
    MIN_SAMPLES,
    PHYSICAL_NEIGHBORS,
    SPATIAL_WEIGHT,
    AnchoredExpansionConfig,
    MetricHDBSCANConfig,
    MutualGraphConfig,
    build_mutual_local_graph,
    cluster_metric_hdbscan,
    corrected_pairwise_distance,
    expand_anchored_clusters,
)


def _xyz_1d(values: list[float]) -> np.ndarray:
    return np.column_stack(
        (
            np.asarray(values, dtype=np.float64),
            np.zeros(len(values), dtype=np.float64),
            np.zeros(len(values), dtype=np.float64),
        )
    )


def test_corrected_distance_is_angular_physical_symmetric_and_zero_diagonal() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 2.0], [-3.0, 0.0]])
    xyz = _xyz_1d([0.0, 1.0, 3.0])

    distance = corrected_pairwise_distance(features, xyz, 2.0)

    assert AFFINITY_WEIGHT == pytest.approx(0.625)
    assert SPATIAL_WEIGHT == pytest.approx(0.375)
    assert distance[0, 1] == pytest.approx(0.5)
    assert distance[0, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(distance, distance.T, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(np.diag(distance), np.zeros(3))
    assert np.isfinite(distance).all()
    assert np.all((distance >= 0) & (distance <= 1))
    assert distance.flags.writeable is False


def test_corrected_distance_uses_frozen_global_scale_not_sample_maximum() -> None:
    base_features = np.tile([1.0, 0.0], (2, 1))
    base_xyz = _xyz_1d([0.0, 0.5])
    base = corrected_pairwise_distance(base_features, base_xyz, 2.0)
    extended = corrected_pairwise_distance(
        np.vstack((base_features, [1.0, 0.0])),
        np.vstack((base_xyz, [1000.0, 0.0, 0.0])),
        2.0,
    )

    assert base[0, 1] == pytest.approx(extended[0, 1])
    assert base[0, 1] == pytest.approx(SPATIAL_WEIGHT * 0.25)


def test_corrected_distance_is_permutation_equivariant() -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.2, 0.8], [-0.5, 0.5], [0.0, -1.0]],
        dtype=np.float64,
    )
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    reference = corrected_pairwise_distance(features, xyz, 1.5)
    permuted = corrected_pairwise_distance(
        features[permutation], xyz[permutation], 1.5
    )

    np.testing.assert_allclose(
        permuted,
        reference[np.ix_(permutation, permutation)],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("features", "xyz", "diag", "message"),
    [
        (np.asarray([[0.0, 0.0]]), np.zeros((1, 3)), 1.0, "non-zero norm"),
        (np.asarray([[1.0, 0.0]]), np.zeros((1, 3)), 0.0, "positive"),
        (np.asarray([[1.0, np.nan]]), np.zeros((1, 3)), 1.0, "finite"),
    ],
)
def test_corrected_distance_rejects_undefined_inputs(
    features: np.ndarray, xyz: np.ndarray, diag: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        corrected_pairwise_distance(features, xyz, diag)


class _RecordedClusterer:
    def __init__(self, labels: np.ndarray, membership: np.ndarray) -> None:
        self.labels = labels
        self.probabilities_ = membership
        self.seen_distance: np.ndarray | None = None

    def fit_predict(self, distance: np.ndarray) -> np.ndarray:
        self.seen_distance = np.asarray(distance).copy()
        return self.labels.copy()


def test_metric_hdbscan_uses_explicit_fixed_precomputed_contract() -> None:
    recorded: dict[str, object] = {}
    clusterer = _RecordedClusterer(
        np.asarray([7, 7, 7, -1]), np.asarray([0.9, 0.8, 0.7, 0.6])
    )

    def factory(**kwargs: object) -> _RecordedClusterer:
        recorded.update(kwargs)
        return clusterer

    result = cluster_metric_hdbscan(
        np.tile([1.0, 0.0], (4, 1)),
        _xyz_1d([0.0, 0.1, 0.2, 3.0]),
        1.0,
        hdbscan_factory=factory,
    )

    assert recorded == {
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "min_samples": MIN_SAMPLES,
        "cluster_selection_epsilon": CLUSTER_SELECTION_EPSILON,
        "allow_single_cluster": False,
        "metric": "precomputed",
    }
    assert clusterer.seen_distance is not None
    np.testing.assert_allclose(clusterer.seen_distance, result.distance_matrix)
    np.testing.assert_array_equal(result.labels, [7, 7, 7, -1])
    np.testing.assert_allclose(result.membership, [0.9, 0.8, 0.7, 0.0])
    assert result.raw_cluster_ids == (7,)
    assert result.diagnostics["hdbscan_ran"] is True
    assert result.labels.flags.writeable is False
    assert result.membership.flags.writeable is False


def test_metric_hdbscan_short_input_is_all_noise_without_calling_factory() -> None:
    def forbidden_factory(**_: object) -> object:
        raise AssertionError("factory must not run below min_cluster_size")

    result = cluster_metric_hdbscan(
        np.tile([1.0, 0.0], (2, 1)),
        _xyz_1d([0.0, 0.1]),
        1.0,
        hdbscan_factory=forbidden_factory,
    )

    np.testing.assert_array_equal(result.labels, [-1, -1])
    np.testing.assert_array_equal(result.membership, [0.0, 0.0])
    assert result.raw_cluster_ids == ()
    assert result.diagnostics["hdbscan_ran"] is False


def test_anchored_expansion_keeps_raw_members_and_attaches_nearest_member() -> None:
    # Raw cluster 9 deliberately contains a member at x=.9, coincident with a
    # member of cluster 14.  It must remain in cluster 9.  The unsampled point
    # at .9 is an exact tie across raw clusters and must remain background.
    xyz = _xyz_1d([0.0, 0.1, 0.9, 0.9, 1.1, 1.2, 0.05, 0.9, 3.0])
    features = np.tile([1.0, 0.0], (len(xyz), 1))
    sampled = np.arange(6, dtype=np.int64)
    raw = np.asarray([9, 9, 9, 14, 14, 14], dtype=np.int64)
    membership = np.asarray([0.9, 0.8, 0.2, 0.95, 0.85, 0.75])

    result = expand_anchored_clusters(
        features,
        xyz,
        sampled,
        raw,
        membership,
        1.0,
        config=AnchoredExpansionConfig(query_chunk_size=1),
    )

    np.testing.assert_array_equal(
        result.raw_cluster_labels_selected,
        [0, 0, 0, 1, 1, 1, -1, -1, -1],
    )
    np.testing.assert_array_equal(
        result.trusted_core_labels,
        [0, 0, 0, 1, 1, 1, -1, -1, -1],
    )
    np.testing.assert_array_equal(
        result.full_candidate_labels,
        [0, 0, 0, 1, 1, 1, 0, -1, -1],
    )
    # Raw membership is evidence, not a forged probability of one.
    assert result.assignment_confidence[2] == pytest.approx(0.2)
    assert 0.0 < result.assignment_confidence[6] <= 1.0
    assert result.assignment_confidence[7] == 0.0
    assert result.diagnostics["raw_member_reassigned_count"] == 0
    # The intended x=.9 tie and the far, spatially-clipped point are both
    # exact cross-cluster ties; neither may attach.
    assert result.diagnostics["exact_cross_cluster_tie_count"] == 2
    assert result.diagnostics["core_outside_full_count"] == 0
    assert np.all(
        (result.trusted_core_labels < 0)
        | (result.trusted_core_labels == result.full_candidate_labels)
    )


def test_anchored_envelope_is_q95_of_leave_one_nearest_member_distance() -> None:
    xyz = _xyz_1d([0.0, 0.1, 0.2, 0.25, 0.31])
    features = np.tile([1.0, 0.0], (len(xyz), 1))
    result = expand_anchored_clusters(
        features,
        xyz,
        np.asarray([0, 1, 2]),
        np.asarray([3, 3, 3]),
        np.asarray([0.8, 0.8, 0.8]),
        1.0,
    )

    # All three leave-one nearest distances are .1 * spatial weight.
    assert result.candidates[0]["envelope_radius"] == pytest.approx(
        0.1 * SPATIAL_WEIGHT
    )
    assert result.full_candidate_labels[3] == 0
    assert result.full_candidate_labels[4] == -1
    expected = np.exp(
        -(0.05 * SPATIAL_WEIGHT) / (0.1 * SPATIAL_WEIGHT)
    )
    assert result.assignment_confidence[3] == pytest.approx(expected)


def test_anchored_expansion_is_chunk_invariant_and_read_only() -> None:
    xyz = _xyz_1d([0.0, 0.1, 0.2, 1.0, 1.1, 1.2, 0.05, 1.15, 5.0])
    features = np.tile([1.0, 0.0], (len(xyz), 1))
    args = (
        features,
        xyz,
        np.arange(6),
        np.asarray([0, 0, 0, 1, 1, 1]),
        np.asarray([0.9] * 6),
        1.0,
    )
    small = expand_anchored_clusters(
        *args, config=AnchoredExpansionConfig(query_chunk_size=1)
    )
    large = expand_anchored_clusters(
        *args, config=AnchoredExpansionConfig(query_chunk_size=1024)
    )

    for name in (
        "raw_cluster_labels_selected",
        "trusted_core_labels",
        "full_candidate_labels",
        "assignment_confidence",
    ):
        left = getattr(small, name)
        right = getattr(large, name)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)
        assert left.flags.writeable is False
    assert small.candidates == large.candidates


def test_anchored_expansion_never_recovers_sampled_hdbscan_noise() -> None:
    xyz = _xyz_1d([0.0, 0.1, 0.2, 0.05, 0.05])
    features = np.tile([1.0, 0.0], (len(xyz), 1))

    result = expand_anchored_clusters(
        features,
        xyz,
        np.asarray([0, 1, 2, 3]),
        np.asarray([0, 0, 0, -1]),
        np.asarray([0.9, 0.9, 0.9, 0.0]),
        1.0,
    )

    # Rows 3 and 4 are geometrically identical.  Row 3 was sampled and
    # rejected as HDBSCAN noise, so it remains background.  Row 4 was never
    # sampled and is eligible for the registered anchored expansion.
    assert result.full_candidate_labels[3] == -1
    assert result.full_candidate_labels[4] == 0
    assert result.diagnostics["sampled_noise_point_count"] == 1


def test_single_raw_member_has_zero_envelope_and_only_exact_support_attaches() -> None:
    xyz = _xyz_1d([0.0, 0.0, 0.01])
    features = np.tile([1.0, 0.0], (len(xyz), 1))
    result = expand_anchored_clusters(
        features,
        xyz,
        np.asarray([0]),
        np.asarray([4]),
        np.asarray([0.6]),
        1.0,
        config=AnchoredExpansionConfig(minimum_candidate_points=1),
    )

    assert result.candidates[0]["envelope_radius"] == 0.0
    np.testing.assert_array_equal(result.full_candidate_labels, [0, 0, -1])
    assert result.assignment_confidence[0] == pytest.approx(0.6)
    assert result.assignment_confidence[1] == pytest.approx(1.0)


def test_mutual_local_graph_uses_physical_neighbors_top4_and_mutual_edges() -> None:
    # Each five-point group supplies exactly four same-group affinity winners.
    # Although every point is within the effective physical neighbor set, the
    # orthogonal features prevent cross-group edges.
    xyz = _xyz_1d([0.00, 0.01, 0.02, 0.03, 0.04, 1.00, 1.01, 1.02, 1.03, 1.04])
    features = np.asarray([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)

    result = build_mutual_local_graph(features, xyz)
    repeated = build_mutual_local_graph(features, xyz)

    np.testing.assert_array_equal(result.full_candidate_labels, [0] * 5 + [1] * 5)
    np.testing.assert_array_equal(
        result.trusted_core_labels, result.full_candidate_labels
    )
    assert np.all(result.edges[:, 0] < result.edges[:, 1])
    assert all(
        (left < 5 and right < 5) or (left >= 5 and right >= 5)
        for left, right in result.edges
    )
    np.testing.assert_allclose(result.assignment_confidence, np.ones(10))
    np.testing.assert_array_equal(result.edges, repeated.edges)
    np.testing.assert_array_equal(
        result.full_candidate_labels, repeated.full_candidate_labels
    )
    assert result.diagnostics["physical_neighbors"] == PHYSICAL_NEIGHBORS
    assert result.diagnostics["affinity_neighbors"] == AFFINITY_NEIGHBORS
    assert result.diagnostics["candidate_count"] == 2


def test_mutual_local_graph_filters_components_below_three_points() -> None:
    result = build_mutual_local_graph(
        np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        _xyz_1d([0.0, 0.1]),
    )

    np.testing.assert_array_equal(result.full_candidate_labels, [-1, -1])
    np.testing.assert_array_equal(result.assignment_confidence, [0.0, 0.0])
    assert result.candidates == ()


def test_runtime_interfaces_have_no_gt_category_or_prior_inputs() -> None:
    public_functions = (
        corrected_pairwise_distance,
        cluster_metric_hdbscan,
        expand_anchored_clusters,
        build_mutual_local_graph,
    )
    dataclasses = (
        MetricHDBSCANConfig,
        AnchoredExpansionConfig,
        MutualGraphConfig,
    )
    names = {
        name
        for function in public_functions
        for name in inspect.signature(function).parameters
    }
    names.update(field.name for data_type in dataclasses for field in fields(data_type))

    assert not any(
        forbidden in name.lower()
        for name in names
        for forbidden in (
            "ground_truth",
            "gt_",
            "class_name",
            "class_id",
            "category",
            "prior",
            "semantic",
        )
    )


def test_configs_reject_parameter_drift_or_invalid_graph_shape() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        MetricHDBSCANConfig(affinity_weight=0.5, spatial_weight=0.4).validate()
    with pytest.raises(ValueError, match="cannot exceed"):
        MutualGraphConfig(physical_neighbors=3, affinity_neighbors=4).validate()
    with pytest.raises(ValueError, match="positive"):
        AnchoredExpansionConfig(query_chunk_size=0).validate()
