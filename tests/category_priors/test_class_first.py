from __future__ import annotations

import math

import numpy as np
import pytest

from category_priors.class_first import (
    ClassFirstConfig,
    _hybrid_distance,
    apply_sor_to_clusters,
    build_class_first_metadata,
    build_class_first_params,
    class_local_knn,
    rescue_noise_by_anchor,
    resolve_class_parameters,
    run_class_first,
    sample_class_indices,
)


def _summary(value: float) -> dict[str, float]:
    return {"q50": value}


def _stats(diameter: float, area: float, boundary: float) -> dict[str, object]:
    return {
        "geometry": {
            "log_bbox_diag_m": _summary(math.log(diameter)),
            "log_surface_area_m2": _summary(math.log(area)),
        },
        "neighborhood": {"boundary_fixed:0.05": _summary(boundary)},
    }


def _priors() -> dict[str, object]:
    return {
        "kind": "category_priors",
        "schema_version": "1.0",
        "provenance": {"splits": ["train"]},
        "global": {"shrunk": _stats(2.0, 4.0, 0.10)},
        "categories": {
            # active is deliberately false: shrunk train statistics still apply.
            "chair": {"active": False, "shrunk": _stats(1.0, 1.0, 0.20)},
        },
    }


def test_direct_factors_are_orthogonal_and_unknown_is_uniform() -> None:
    config = ClassFirstConfig()
    priors = _priors()
    uniform = resolve_class_parameters(priors, config, "chair", "uniform")
    size = resolve_class_parameters(priors, config, "chair", "size")
    smooth = resolve_class_parameters(priors, config, "chair", "smooth")
    small = resolve_class_parameters(priors, config, "chair", "small")
    combined = resolve_class_parameters(priors, config, "chair", "combined")
    unknown = resolve_class_parameters(priors, config, "remote", "combined")

    assert uniform["min_cluster_size"] == size["min_cluster_size"] == smooth["min_cluster_size"] == 10
    assert uniform["knn_k"] == size["knn_k"] == small["knn_k"] == 256
    assert size["coordinate_mode"] == "metric_divide_d_c"
    assert size["spatial_scale_m"] == pytest.approx(1.0)
    assert smooth["coordinate_mode"] == "robust_scale"
    assert smooth["knn_k"] == 128
    assert small["min_cluster_size"] == 5
    assert small["rescue_radius_m"] == pytest.approx(0.1)
    assert combined["min_cluster_size"] == 5
    assert combined["knn_k"] == 128
    assert unknown["supported"] is False
    assert unknown["min_cluster_size"] == 10
    assert unknown["knn_k"] == 256
    assert unknown["coordinate_mode"] == "robust_scale"
    assert unknown["rescue_enabled"] is False


def test_sample_quota_is_exact_and_deterministic() -> None:
    config = ClassFirstConfig()
    first = sample_class_indices(1000, 10, config, np.random.default_rng(7))
    second = sample_class_indices(1000, 10, config, np.random.default_rng(7))
    capped = sample_class_indices(9000, 20, config, np.random.default_rng(7))
    hard_capped = sample_class_indices(200_000, 20, config, np.random.default_rng(7))

    assert len(first) == 40
    assert np.array_equal(first, second)
    assert len(capped) == 270
    assert len(hard_capped) == 5000


def test_hybrid_distance_uses_all_three_registered_weights() -> None:
    instance = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    semantic = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    xyz = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    distance = _hybrid_distance(instance, semantic, xyz, ClassFirstConfig())
    assert distance[0, 1] == pytest.approx(0.3 * 1.0 + 0.3 * 2.0 + 0.4 * 2.0)


def test_knn_is_applied_to_each_class_branch_not_mixed_classes() -> None:
    # Two semantic branches may occupy identical coordinates.  Independent calls
    # cannot copy instance labels across that semantic boundary.
    xyz = np.asarray([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]], dtype=np.float64)
    chair = class_local_knn(np.asarray([0, 0, -1]), xyz, 3)
    cup = class_local_knn(np.asarray([7, 7, -1]), xyz, 3)
    assert chair.tolist() == [0, 0, 0]
    assert cup.tolist() == [7, 7, 7]


def test_sor_then_pointwise_rescue_obeys_metric_radius() -> None:
    labels = np.asarray([0, 0, 0, -1, -1], dtype=np.int64)
    xyz = np.asarray(
        [[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.025, 0, 0], [1.0, 0, 0]],
        dtype=np.float64,
    )

    def fake_sor(points: np.ndarray, neighbors: int, ratio: float) -> np.ndarray:
        assert neighbors == 2
        return np.asarray([True, True, False])

    after_sor, removed = apply_sor_to_clusters(labels, xyz, 50, 0.05, fake_sor)
    rescued, count = rescue_noise_by_anchor(after_sor, xyz, 0.03)

    assert removed == 1
    assert count == 2
    assert rescued.tolist() == [0, 0, 0, 0, -1]


def test_run_uses_selected_classes_and_explicit_min_samples() -> None:
    captured: list[dict[str, object]] = []

    class FakeClusterer:
        def __init__(self, **kwargs) -> None:
            captured.append(kwargs)

        def fit_predict(self, distance: np.ndarray) -> np.ndarray:
            return np.zeros(len(distance), dtype=np.int64)

    count = 80
    semantic = np.vstack((np.tile([1.0, 0.0], (40, 1)), np.tile([0.0, 1.0], (40, 1))))
    result = run_class_first(
        point_features=np.tile([1.0, 0.0], (count, 1)),
        point_semantic_features=semantic,
        point_xyz=np.column_stack((np.arange(count) / 100.0, np.zeros(count), np.zeros(count))),
        label_features=np.eye(2, dtype=np.float32),
        classes=("chair", "remote"),
        selected_classes=("chair",),
        priors=_priors(),
        config=ClassFirstConfig(use_sor=False, min_samples=7),
        mode="uniform",
        scene_scale_m_per_unit=1.0,
        clusterer_factory=FakeClusterer,
    )

    assert len(captured) == 1
    assert captured[0]["min_samples"] == 7
    assert set(result.diagnostics["classes"]) == {"chair"}
    assert np.all(result.labels[40:] == -1)
    assert result.instances[0]["class"] == "chair"
    assert 0.0 <= result.instances[0]["score"] <= 1.0
    metadata = build_class_first_metadata(result)
    assert metadata["kind"] == "saga_instance_metadata"
    assert metadata["instances"]["0"]["score"] == result.instances[0]["score"]


def test_params_table_has_no_hash_and_keeps_all_prior_categories() -> None:
    payload = build_class_first_params(_priors(), ClassFirstConfig(), "combined")
    assert payload["kind"] == "class_first_params"
    assert "content_sha256" not in payload
    assert set(payload["categories"]) == {"chair"}
    assert payload["categories"]["chair"]["m_c"] == 5


def test_config_rejects_unregistered_rho_and_area_exponent() -> None:
    with pytest.raises(ValueError, match="rescue_radius_ratio"):
        ClassFirstConfig(rescue_radius_ratio=0.15)
    with pytest.raises(ValueError, match="small_area_exponent"):
        ClassFirstConfig(small_area_exponent=0.75)
