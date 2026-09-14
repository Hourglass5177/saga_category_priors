import numpy as np
import pytest

from category_priors.object_verification.measurement import (
    AlphaMeasurement, analytic_alpha, gradient_reference_from_graph, normalized_coefficients,
)


def test_raw_alpha_is_deeply_immutable_without_clipping_first_values():
    inside = np.array([[-1e-8, 1. + 1e-8]])
    visible = np.array([0., 1.])
    result = AlphaMeasurement(inside, visible, 1)
    inside[:] = 0
    visible[:] = 0
    assert result.inside_mass[0, 0] == -1e-8
    assert result.inside_mass[0, 1] == 1. + 1e-8
    assert result.visible_mass[1] == 1.
    for array in (result.inside_mass, result.visible_mass):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert result.qualified(0) == (1,)


def test_unknown_domain_excluded_from_both_inside_and_visible_mass():
    contributions = np.array([[[.3, .2], [.1, .4]], [[.2, .3], [.25, .25]]])
    masks = np.array([[[1, 1], [1, 1]], [[1, 0], [0, 0]]], dtype=bool)
    valid = np.array([[True, False], [False, True]])
    result = analytic_alpha(contributions, masks, valid_pixels=valid)
    np.testing.assert_allclose(result.visible_mass, [1.1, .9])
    np.testing.assert_allclose(result.inside_mass, [[1.1, .9], [.6, .4]])
    assert result.valid_pixels == 2
    mask_only = analytic_alpha(contributions, masks & valid)
    assert not np.array_equal(mask_only.visible_mass, result.visible_mass)
    assert mask_only.qualified(1) == () and result.qualified(1) == (0,)


@pytest.mark.parametrize("empty", [False, True])
def test_CPU_autograd_reference_matches_formula_with_same_valid_domain(empty):
    torch = pytest.importorskip("torch")
    matrix = np.array([[[.3, .2], [.1, .4]], [[.2, .3], [.25, .25]]], dtype=np.float32)
    masks = np.array([[[1, 1], [1, 1]], [[1, 0], [0, 0]],
                      [[0, 1], [1, 0]], [[0, 0], [0, 1]]], dtype=bool)
    valid = np.zeros((2, 2), dtype=bool) if empty else np.array([[True, False], [False, True]])
    probe = torch.ones((2, 3), dtype=torch.float32, requires_grad=True, device="cpu")
    image = torch.einsum("hwn,nc->chw", torch.tensor(matrix, device="cpu"), probe)
    actual = gradient_reference_from_graph(image, probe, masks, valid_pixels=valid)
    expected = analytic_alpha(matrix, masks, valid_pixels=valid)
    np.testing.assert_allclose(actual.visible_mass, expected.visible_mass, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(actual.inside_mass, expected.inside_mass, rtol=1e-6, atol=1e-7)
    assert actual.valid_pixels == expected.valid_pixels == int(valid.sum())
    assert [actual.qualified(i) for i in range(4)] == [expected.qualified(i) for i in range(4)]


@pytest.mark.parametrize("valid", [np.ones((2, 2)), np.ones((1, 2), dtype=bool)])
def test_valid_domain_refuses_implicit_cast_or_geometry_change(valid):
    with pytest.raises(ValueError, match="actual RGB coordinates"):
        normalized_coefficients(np.ones((2, 2)), np.ones((1, 2, 2)), valid_pixels=valid)
