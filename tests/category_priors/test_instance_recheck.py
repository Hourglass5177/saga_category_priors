from __future__ import annotations

import numpy as np

from category_priors.instance_recheck import (
    ViewReview,
    aggregate_candidate_review,
    candidate_projection_rows,
    crop_candidate_view,
    projection_mask,
    review_geometry,
    select_candidate_views,
)


def test_projection_rejects_empty_and_invalid_contributors() -> None:
    membership = np.asarray([3, 7, -1], dtype=np.int64)
    contributor = np.asarray([[0, 0, -1], [0, 0, 1]], dtype=np.int64)
    weights = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.float64)
    rows = candidate_projection_rows(
        candidate_by_gaussian=membership,
        contributor_ids=contributor,
        contribution_weights=weights,
        camera_index=2,
        image_name="frame",
    )
    assert len(rows) == 1
    assert rows[0].candidate_id == 3
    assert rows[0].pixel_count == 4
    assert not projection_mask(membership, contributor, weights, 7).any()


def test_top_views_use_pixel_count_then_image_name() -> None:
    rows = []
    for index, (name, count) in enumerate((('c', 5), ('a', 7), ('b', 7), ('d', 4))):
        contributor = np.zeros((1, count), dtype=np.int64)
        rows.extend(
            candidate_projection_rows(
                candidate_by_gaussian=np.asarray([0]),
                contributor_ids=contributor,
                contribution_weights=np.ones_like(contributor, dtype=float),
                camera_index=index,
                image_name=name,
            )
        )
    selected = select_candidate_views(rows)
    assert [row.image_name for row in selected[0]] == ['a', 'b', 'c']


def test_crop_preserves_projection_and_caps_at_image_long_side() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 1:3] = True
    crop, crop_mask, geometry = crop_candidate_view(
        image_rgb=image,
        projected_mask=mask,
        candidate_diagonal_m=0.01,
        prior_diagonal_m=2.0,
    )
    assert geometry.side == 10
    assert geometry.crop_capped
    assert crop.shape == (10, 10, 3)
    assert int(crop_mask.sum()) == int(mask.sum())


def test_review_uses_one_way_coverage_and_majority_rule() -> None:
    candidate = np.zeros((4, 4), dtype=bool)
    candidate[1:3, 1:3] = True
    detected = np.ones((4, 4), dtype=bool)
    coverage, iou = review_geometry(
        candidate_mask=candidate, detected_mask=detected
    )
    assert coverage == 1.0
    assert iou == 0.25

    view = ViewReview('a', 4, 8, False, 'cup', 0.9, 1.0, 0.25, 4.0, True)
    one = aggregate_candidate_review(
        candidate_id=0,
        branch_class='cup',
        condition='global',
        prior_diagonal_m=1.0,
        views=(view,),
    )
    assert one.accepted and one.required_votes == 1
    three = aggregate_candidate_review(
        candidate_id=0,
        branch_class='cup',
        condition='global',
        prior_diagonal_m=1.0,
        views=(view, view, ViewReview('b', 4, 8, False, 'chair', 0.9, 1.0, 0.25, 4.0, False)),
    )
    assert three.accepted and three.required_votes == 2
