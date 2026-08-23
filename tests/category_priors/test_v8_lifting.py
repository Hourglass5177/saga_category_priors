from __future__ import annotations

import numpy as np

from category_priors.v8_lifting import (
    AttributionMass,
    V8FragmentConfig,
    attribution_from_am_gradients,
    build_am_objective_targets,
    fragments_from_attribution,
    iter_three_channel_mask_batches,
    lift_max_contributor,
    mass_from_max_contributor,
)


def test_m1_filters_empty_invalid_and_zero_weight_pixels() -> None:
    ids = np.array([[0, -1, 1], [3, 2, 0]])
    weights = np.array([[0.4, 0.9, 0.0], [0.8, np.nan, 0.6]])
    masks = np.ones((1, 2, 3), dtype=bool)
    mass = mass_from_max_contributor(ids, weights, masks, point_count=3)

    assert mass.valid_pixel_count == 2
    assert np.allclose(mass.visible_mass, [2.0, 0.0, 0.0])
    assert np.allclose(mass.inside_mass[0], mass.visible_mass)


def test_missing_mask_is_abstention_not_background_evidence() -> None:
    ids = np.array([[0, 1]])
    weights = np.ones((1, 2))
    mass = mass_from_max_contributor(ids, weights, None, point_count=2)

    assert mass.abstained
    assert mass.inside_mass.shape == (0, 2)
    assert np.allclose(mass.visible_mass, [1.0, 1.0])
    frame = lift_max_contributor(ids, weights, None, 2, frame_id=7)
    assert frame.fragments == ()
    assert frame.visible_ids.tolist() == [0, 1]


def test_m1_unit_weight_degenerates_to_pixel_count_lifting() -> None:
    ids = np.array([[0, 0, 1], [0, 1, 1]])
    weights = np.ones_like(ids, dtype=float)
    masks = np.array([[[1, 1, 0], [0, 1, 1]]], dtype=bool)
    mass = mass_from_max_contributor(ids, weights, masks, point_count=2)

    assert np.allclose(mass.visible_mass, [3.0, 3.0])
    assert np.allclose(mass.inside_mass[0], [2.0, 2.0])
    assert np.allclose(mass.outside_mass[0], [1.0, 1.0])


def test_m1_weight_selects_valid_pixels_but_does_not_rescale_pixel_mass() -> None:
    ids = np.array([[0, 0, 1]])
    weights = np.array([[0.01, 0.80, 0.20]])
    masks = np.ones((1, 1, 3), dtype=bool)

    mass = mass_from_max_contributor(ids, weights, masks, point_count=2)

    assert np.allclose(mass.visible_mass, [2.0, 1.0])
    assert np.allclose(mass.inside_mass[0], [2.0, 1.0])


def test_am_normalization_conserves_pixel_mass_and_splits_inside_outside() -> None:
    # Rows are pixels, columns are Gaussians; row sums are rendered opacity.
    contribution = np.array(
        [
            [0.2, 0.3, 0.0],
            [0.1, 0.2, 0.5],
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1],
        ],
        dtype=float,
    )
    opacity = contribution.sum(axis=1).reshape(2, 2)
    masks = np.array(
        [
            [[1, 1], [1, 0]],
            [[0, 1], [0, 1]],
        ],
        dtype=bool,
    )
    batch = next(iter_three_channel_mask_batches(masks))
    targets = build_am_objective_targets(batch, opacity)
    flat_contribution = contribution

    visible_gradient = flat_contribution.T @ targets.visible_coefficient.reshape(-1)
    inside_gradient = flat_contribution.T @ targets.inside_coefficients.reshape(3, -1).T
    mass = attribution_from_am_gradients(
        visible_gradient,
        [(batch, inside_gradient)],
        mask_count=2,
        valid_pixel_count=int(np.count_nonzero(targets.valid_pixels)),
    )

    assert np.isclose(mass.visible_mass.sum(), 3.0)
    assert np.allclose(mass.inside_mass.sum(axis=1), [2.0, 2.0])
    assert np.allclose(
        mass.inside_mass + mass.outside_mass,
        np.broadcast_to(mass.visible_mass, mass.inside_mass.shape),
    )


def test_am_three_channel_batches_are_deterministic_and_zero_padded() -> None:
    masks = np.arange(5 * 2 * 2).reshape(5, 2, 2) % 2
    batches = list(iter_three_channel_mask_batches(masks))

    assert [batch.mask_indices for batch in batches] == [(0, 1, 2), (3, 4)]
    assert batches[0].targets.shape == (3, 2, 2)
    assert np.array_equal(batches[1].targets[:2], masks[3:5])
    assert not np.any(batches[1].targets[2])


def test_fragment_rules_are_shared_mass_thresholds() -> None:
    visible = np.full(10, 2.0)
    inside = np.concatenate((np.full(5, 2.0), np.full(5, 0.5)))[None, :]
    attribution = AttributionMass(
        source="AM",
        inside_mass=inside,
        visible_mass=visible,
        valid_pixel_count=10,
    )
    fragments = fragments_from_attribution(attribution, frame_id=3)

    assert len(fragments) == 1
    assert fragments[0].full_ids.tolist() == list(range(10))
    assert fragments[0].core_ids.tolist() == list(range(5))
    assert np.allclose(fragments[0].core_inside_ratio, 1.0)


def test_core_ratio_rejects_visible_but_mask_ambiguous_gaussians() -> None:
    attribution = AttributionMass(
        source="M1",
        inside_mass=np.full((1, 10), 2.0),
        visible_mass=np.full(10, 5.0),
        valid_pixel_count=50,
    )
    config = V8FragmentConfig(fragment_min_core=1, fragment_min_full=1)

    assert fragments_from_attribution(attribution, 0, config=config) == ()
