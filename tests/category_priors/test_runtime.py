from __future__ import annotations

import math

from category_priors.runtime import PriorResolver, allocate_class_quotas


def test_quota_allocator_respects_budget_and_caps() -> None:
    quotas = allocate_class_quotas(
        {"chair": 1000, "cup": 300, "book": 20},
        {"chair": 0.0, "cup": 1.0, "book": 0.8},
        total_budget=500,
        minimum=32,
        maximum=300,
    )
    assert sum(quotas.values()) == 500
    assert quotas["book"] <= 20
    assert all(value <= 300 for value in quotas.values())


def test_gate_zero_is_exact_legacy_smoothing(fitted_priors, mapping_config) -> None:
    resolver = PriorResolver(fitted_priors, mapping_config)
    params = resolver.class_parameters(
        "chair",
        "smooth",
        semantic_similarity=0.0,
        surface_density=1000.0,
        sample_fraction=1.0,
        gate_enabled=True,
        shrink_enabled=True,
    )
    assert params["gate"] == 0.0
    assert params["knn_k"] == mapping_config["baseline"]["knn_k"]
    assert math.isinf(params["knn_radius_m"])


def test_unknown_class_always_falls_back_to_legacy(
    fitted_priors, mapping_config
) -> None:
    resolver = PriorResolver(fitted_priors, mapping_config)
    params = resolver.class_parameters(
        "not-in-taxonomy",
        "combined",
        semantic_similarity=1.0,
        surface_density=1000.0,
        sample_fraction=1.0,
    )
    assert params["active"] is False
    assert params["gate"] == 0.0
    assert params["min_cluster_size"] == mapping_config["baseline"]["min_cluster_size"]
    assert params["knn_k"] == mapping_config["baseline"]["knn_k"]
    assert math.isinf(params["knn_radius_m"])


def test_small_factor_reduces_target_for_cup(fitted_priors, mapping_config) -> None:
    resolver = PriorResolver(fitted_priors, mapping_config)
    cup = resolver.class_parameters("cup", "small", 1.0, 5000.0, 1.0)
    chair = resolver.class_parameters("chair", "small", 1.0, 5000.0, 1.0)
    assert cup["small_score"] > chair["small_score"]
    assert cup["min_cluster_size"] <= chair["min_cluster_size"]


def test_all_factorial_modes_expose_registered_bits(
    fitted_priors, mapping_config
) -> None:
    resolver = PriorResolver(fitted_priors, mapping_config)
    expected = {
        "global": {"size": False, "smooth": False, "small": False},
        "small": {"size": False, "smooth": False, "small": True},
        "smooth": {"size": False, "smooth": True, "small": False},
        "smooth-small": {"size": False, "smooth": True, "small": True},
        "size": {"size": True, "smooth": False, "small": False},
        "size-small": {"size": True, "smooth": False, "small": True},
        "size-smooth": {"size": True, "smooth": True, "small": False},
        "combined": {"size": True, "smooth": True, "small": True},
    }
    for mode, bits in expected.items():
        params = resolver.class_parameters("chair", mode, 1.0, 1000.0, 1.0)
        assert params["factors"] == bits
