from __future__ import annotations

import numpy as np
import pytest

from category_priors.gaussian_object_audit import evaluate_gaussian_object_precision
from category_priors.instance_projection import (
    bounded_recheck_crop_side,
    project_declared_instances,
)


def test_projection_preserves_declared_labels_and_does_not_mutate_input() -> None:
    raw = np.asarray([4, 8, 8, -1, 4, 13], dtype=np.int64)
    before = raw.copy()

    result = project_declared_instances(
        raw,
        {"4": {"class": "chair"}, "13": {"class": "book"}},
    )

    assert np.array_equal(raw, before)
    assert result.point_labels.tolist() == [4, -1, -1, -1, 4, 13]
    assert result.point_labels.flags.writeable is False
    assert result.declared_instance_ids == (4, 13)
    assert result.orphan_instance_ids == (8,)
    assert result.orphan_counts == ((8, 2),)
    assert result.stats()["orphan_gaussian_fraction"] == 2 / 6


def test_projection_reports_empty_declared_instances_and_all_orphans() -> None:
    result = project_declared_instances(
        [7, 7],
        {"3": {"class": "chair"}},
    )

    assert result.point_labels.tolist() == [-1, -1]
    assert result.empty_declared_instance_ids == (3,)
    assert result.stats()["declared_gaussian_count"] == 0
    assert result.stats()["orphan_gaussian_fraction"] == 1.0
    assert result.stats()["projected_background_gaussian_count"] == 2


def test_projection_leaves_declared_instance_precision_numerically_unchanged() -> None:
    gaussian_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    )
    raw_labels = np.asarray([4, 9, 4])
    instances = {"4": {"class": "chair"}}
    gt_semantic = np.asarray([0, 0, 0])
    gt_instance = np.asarray([1, 1, 1])
    projected = project_declared_instances(raw_labels, instances)

    def evaluate(labels: np.ndarray):
        return evaluate_gaussian_object_precision(
            gaussian_xyz,
            labels,
            instances,
            gaussian_xyz,
            gt_semantic,
            gt_instance,
            canonical_classes=("chair",),
        )

    raw_audit = evaluate(raw_labels)
    projected_audit = evaluate(projected.point_labels)

    assert projected_audit["instances"] == raw_audit["instances"]
    assert projected_audit["aggregate"] == raw_audit["aggregate"]


def test_projection_ignores_negative_metadata_ids_as_background() -> None:
    result = project_declared_instances(
        [-1, 4],
        {"-1": {"class": "cabinet"}, "4": {"class": "chair"}},
    )

    assert result.point_labels.tolist() == [-1, 4]
    assert result.declared_instance_ids == (4,)
    assert result.ignored_negative_metadata_ids == (-1,)
    assert result.stats()["ignored_negative_metadata_ids"] == [-1]


@pytest.mark.parametrize("instance_id", ["01", "not-an-id"])
def test_projection_rejects_invalid_declared_instance_ids(instance_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        project_declared_instances([0], {instance_id: {"class": "chair"}})


def test_recheck_crop_side_uses_frozen_formula_and_whole_image_cap() -> None:
    ordinary = bounded_recheck_crop_side(
        candidate_side_px=20,
        candidate_diagonal_m=0.5,
        prior_diagonal_m=1.0,
        image_width=640,
        image_height=480,
    )
    assert ordinary.prior_scaled_side_px == pytest.approx(40.0)
    assert ordinary.crop_side_px == pytest.approx(60.0)
    assert not ordinary.crop_capped

    tiny = bounded_recheck_crop_side(
        candidate_side_px=20,
        candidate_diagonal_m=0.00001,
        prior_diagonal_m=1.0,
        image_width=640,
        image_height=480,
    )
    assert tiny.requested_side_px > 640
    assert tiny.crop_side_px == 640
    assert tiny.crop_capped
