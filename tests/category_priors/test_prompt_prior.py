from __future__ import annotations

import math

import numpy as np
import pytest

from category_priors.prompt_prior import (
    empirical_cdf,
    gated_prompt_mask,
    gated_prompt_cosine,
    materialize_prompt_prior,
    materialize_prompt_priors,
    scene_mask_scale_ecdf,
    select_prompt_with_prior,
    training_quantile_uniform,
)


def _node(diag_m: float) -> dict:
    return {
        "shrunk": {
            "geometry": {"log_bbox_diag_m": {"q50": math.log(diag_m)}}
        }
    }


def _priors(*, global_diag: float = 1.0, chair_diag: float = 0.5) -> dict:
    return {
        "provenance": {"splits": ["train"]},
        "global": _node(global_diag),
        "categories": {
            "chair": _node(chair_diag),
            # This is the shape used for an inactive category in schema v1.
            "phone": {"active": False},
        },
    }


def test_materializes_shrunk_log_diagonal_and_unknown_falls_back_global() -> None:
    table = materialize_prompt_priors(_priors(global_diag=2.0, chair_diag=0.25))

    assert table.global_typical_diag_m == pytest.approx(2.0)
    assert table.typical_diag_m("chair", mode="class") == pytest.approx(0.25)
    assert table.typical_diag_m("CHAIR", mode="class") == pytest.approx(0.25)
    assert table.typical_diag_m("phone", mode="class") == pytest.approx(2.0)
    assert table.typical_diag_m("unknown", mode="class") == pytest.approx(2.0)
    assert table.typical_diag_m("chair", mode="uniform") == pytest.approx(2.0)

    plain = materialize_prompt_prior(_priors(global_diag=2.0, chair_diag=0.25))
    assert plain == {
        "global_typical_diag_m": pytest.approx(2.0),
        "class_typical_diag_m": {"chair": pytest.approx(0.25)},
    }


def test_materialization_rejects_non_train_or_invalid_global_prior() -> None:
    payload = _priors()
    payload["provenance"]["splits"] = ["val"]
    with pytest.raises(ValueError, match="train split"):
        materialize_prompt_priors(payload)

    payload = _priors()
    payload["global"] = {"shrunk": {"geometry": {}}}
    with pytest.raises(ValueError, match="log_bbox_diag_m"):
        materialize_prompt_priors(payload)


def test_scene_mask_scale_ecdf_is_right_continuous_and_bounded() -> None:
    scales = np.array([0.1, 0.2, 0.2, 0.8, np.nan])

    assert scene_mask_scale_ecdf(scales, 0.05) == pytest.approx(0.0)
    assert scene_mask_scale_ecdf(scales, 0.2) == pytest.approx(0.75)
    assert scene_mask_scale_ecdf(scales, 2.0) == pytest.approx(1.0)
    assert empirical_cdf(scales, 0.2) == pytest.approx(0.75)

    with pytest.raises(ValueError, match="negative"):
        scene_mask_scale_ecdf([-0.1, 0.2], 0.1)


def test_training_quantile_reconstructs_linear_uniform_transform() -> None:
    scales = np.asarray([0.0, 1.0, 2.0])

    assert training_quantile_uniform(scales, 0.5) == pytest.approx(0.25)
    assert training_quantile_uniform(scales, 1.0) == pytest.approx(0.5)
    assert training_quantile_uniform(scales, 3.0) == pytest.approx(1.0)


def test_native_gate_normalize_and_strict_cosine_threshold() -> None:
    points = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],  # the asymmetric gate suppresses the second channel
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    prompt = np.array([1.0, 0.0])
    gate = np.array([1.0, 0.5])

    cosine = gated_prompt_cosine(points, prompt, gate)
    np.testing.assert_allclose(
        cosine,
        np.array([1.0, 0.8 / math.sqrt(0.8**2 + 0.3**2), 0.0, 0.0]),
    )

    # The raw rendered query and raw point are gated once on both sides.
    query = np.array([1.0, 1.0]) / math.sqrt(2.0)
    mask, similarities = gated_prompt_mask(
        np.array([[1.0, 1.0]]), query, np.array([1.0, 0.5]), threshold=0.0
    )
    np.testing.assert_allclose(similarities, np.array([1.0]))
    np.testing.assert_array_equal(mask, np.array([True]))

    table = materialize_prompt_priors(_priors(global_diag=1.0, chair_diag=1.0))
    result = select_prompt_with_prior(
        np.array([[1.0, 0.0], [0.75, math.sqrt(1.0 - 0.75**2)]]),
        prompt,
        priors=table,
        class_name="chair",
        mode="uniform",
        scene_mask_scales_m=[0.5, 1.0, 2.0],
        gate_from_scale=lambda _: np.ones(2),
    )
    # Exactly 0.75 is excluded because the native comparison is strict >.
    np.testing.assert_array_equal(result.selected, np.array([True, False]))


def test_uniform_and_class_are_pointwise_identical_when_parameters_match() -> None:
    table = materialize_prompt_priors(_priors(global_diag=0.5, chair_diag=0.5))
    points = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    prompt = np.array([1.0, 0.0])
    scales = [0.1, 0.3, 0.5, 1.0]

    def gate(scale: float) -> np.ndarray:
        return np.array([0.25 + scale, 1.25 - scale])

    uniform = select_prompt_with_prior(
        points,
        prompt,
        priors=table,
        class_name="chair",
        mode="uniform",
        scene_mask_scales_m=scales,
        gate_from_scale=gate,
    )
    data = select_prompt_with_prior(
        points,
        prompt,
        priors=table,
        class_name="chair",
        mode="class",
        scene_mask_scales_m=scales,
        gate_from_scale=gate,
    )

    assert uniform.gate_input == data.gate_input
    np.testing.assert_array_equal(uniform.gate_vector, data.gate_vector)
    np.testing.assert_array_equal(uniform.cosine, data.cosine)
    np.testing.assert_array_equal(uniform.selected, data.selected)


def test_unknown_class_data_arm_is_exact_uniform_fallback() -> None:
    table = materialize_prompt_priors(_priors(global_diag=0.7, chair_diag=0.2))
    points = np.array([[1.0, 1.0], [1.0, 0.0]])
    prompt = np.array([1.0, 1.0])
    kwargs = {
        "point_features": points,
        "prompt_feature": prompt,
        "priors": table,
        "class_name": "not-in-priors",
        "scene_mask_scales_m": [0.1, 0.7, 1.0],
        "gate_from_scale": lambda q: np.array([q + 0.1, 1.1 - q]),
    }

    uniform = select_prompt_with_prior(mode="uniform", **kwargs)
    data = select_prompt_with_prior(mode="class", **kwargs)

    assert uniform.typical_diag_m == data.typical_diag_m
    assert uniform.gate_input == data.gate_input
    np.testing.assert_array_equal(uniform.cosine, data.cosine)
    np.testing.assert_array_equal(uniform.selected, data.selected)
