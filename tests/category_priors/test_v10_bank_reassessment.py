from __future__ import annotations

import numpy as np
import pytest

from category_priors.v10_bank_reassessment import (
    build_bidirectional_nearest,
    evaluate_candidate_three_spaces,
    official_projected_mask,
)
from category_priors.v10_metrics import V10GroundTruthObject


CLASSES = ("chair", "table")


def _evaluate(
    *,
    gaussian_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    candidate_ids: np.ndarray,
    gt_semantic: np.ndarray | None = None,
    gt_instance: np.ndarray | None = None,
    candidate_class: str | None = "chair",
    objects: list[V10GroundTruthObject] | None = None,
):
    semantic = (
        np.zeros(len(gt_xyz), dtype=np.int64)
        if gt_semantic is None
        else np.asarray(gt_semantic, dtype=np.int64)
    )
    instance = (
        np.ones(len(gt_xyz), dtype=np.int64)
        if gt_instance is None
        else np.asarray(gt_instance, dtype=np.int64)
    )
    truth = objects or [
        V10GroundTruthObject(
            "scene", "chair", 1, np.arange(len(gt_xyz)), True, "tiny"
        )
    ]
    nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
    return evaluate_candidate_three_spaces(
        scene_id="scene",
        candidate_id=7,
        candidate_gaussian_ids=candidate_ids,
        candidate_class_name=candidate_class,
        candidate_score=0.8,
        nearest=nearest,
        gt_semantic=semantic,
        gt_instance=instance,
        ground_truth=truth,
        canonical_classes=CLASSES,
        radii_m=(0.02, 0.05, 0.10),
    )


def _row(rows, radius: float):
    return next(row for row in rows if row["radius_m"] == pytest.approx(radius))


def test_dense_correct_gaussians_do_not_become_inverse_mapping_false_positives() -> None:
    gt_x = np.linspace(0.0, 0.099, 100)
    gt_xyz = np.column_stack((gt_x, np.zeros(100), np.zeros(100)))
    gaussian_x = np.linspace(0.0, 0.099, 1000)
    gaussian_xyz = np.column_stack(
        (gaussian_x, np.zeros(1000), np.zeros(1000))
    )

    rows = _evaluate(
        gaussian_xyz=gaussian_xyz,
        gt_xyz=gt_xyz,
        candidate_ids=np.arange(1000),
    )
    primary = _row(rows, 0.05)

    assert primary["official_geometric_iou"] == pytest.approx(1.0)
    assert primary["gaussian_to_gt_precision"] == pytest.approx(1.0)
    assert primary["gt_to_candidate_recall"] == pytest.approx(1.0)
    assert primary["gaussian_unsupported_count"] == 0


def test_far_extra_gaussian_only_reduces_gaussian_precision() -> None:
    rows = _evaluate(
        gaussian_xyz=np.asarray([[0.0, 0, 0], [0.02, 0, 0], [5.0, 0, 0]]),
        gt_xyz=np.asarray([[0.0, 0, 0], [0.02, 0, 0]]),
        candidate_ids=np.asarray([0, 1, 2]),
    )
    primary = _row(rows, 0.05)

    assert primary["official_geometric_iou"] == pytest.approx(1.0)
    assert primary["gt_to_candidate_recall"] == pytest.approx(1.0)
    assert primary["gaussian_to_gt_precision"] == pytest.approx(2 / 3)
    assert primary["gaussian_unsupported_count"] == 1


def test_missing_gaussian_reduces_gt_recall_not_gaussian_precision() -> None:
    rows = _evaluate(
        gaussian_xyz=np.asarray([[0.0, 0, 0], [3.0, 0, 0]]),
        gt_xyz=np.asarray([[0.0, 0, 0], [0.04, 0, 0], [0.08, 0, 0]]),
        candidate_ids=np.asarray([0]),
    )
    tight = _row(rows, 0.02)

    assert tight["gaussian_to_gt_precision"] == pytest.approx(1.0)
    assert tight["gt_to_candidate_recall"] == pytest.approx(1 / 3)
    assert tight["official_geometric_recall"] == pytest.approx(1 / 3)


def test_wrong_class_and_same_class_wrong_instance_are_separate() -> None:
    gt_xyz = np.asarray([[0.0, 0, 0], [0.02, 0, 0], [0.04, 0, 0]])
    objects = [
        V10GroundTruthObject("scene", "chair", 1, np.asarray([0]), True),
        V10GroundTruthObject("scene", "chair", 2, np.asarray([1]), True),
        V10GroundTruthObject("scene", "table", 3, np.asarray([2]), True),
    ]
    rows = _evaluate(
        gaussian_xyz=gt_xyz.copy(),
        gt_xyz=gt_xyz,
        candidate_ids=np.asarray([0, 1, 2]),
        gt_semantic=np.asarray([0, 0, 1]),
        gt_instance=np.asarray([1, 2, 3]),
        objects=objects,
    )
    primary = _row(rows, 0.05)

    assert primary["precision_target_instance_id"] in {1, 2}
    assert primary["gaussian_correct_count"] == 1
    assert primary["gaussian_same_class_wrong_instance_count"] == 1
    assert primary["gaussian_wrong_class_count"] == 1
    assert primary["gaussian_to_gt_precision"] == pytest.approx(1 / 3)


def test_asset_coverage_is_monotone_across_registered_radii() -> None:
    rows = _evaluate(
        gaussian_xyz=np.asarray([[0.0, 0, 0]]),
        gt_xyz=np.asarray([[0.0, 0, 0], [0.04, 0, 0], [0.09, 0, 0]]),
        candidate_ids=np.asarray([0]),
    )
    coverage = [row["scene_gt_asset_coverage"] for row in rows]
    assert coverage == sorted(coverage)
    assert coverage == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_projection_and_evaluation_do_not_mutate_candidate_ids() -> None:
    gaussian_xyz = np.asarray([[0.0, 0, 0], [0.02, 0, 0]])
    gt_xyz = gaussian_xyz.copy()
    ids = np.asarray([1, 0], dtype=np.int64)
    before = ids.copy()
    nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
    lookup = np.asarray([True, True])

    projected = official_projected_mask(lookup, nearest, 0.05)
    _evaluate(
        gaussian_xyz=gaussian_xyz,
        gt_xyz=gt_xyz,
        candidate_ids=ids,
    )

    assert projected.tolist() == [True, True]
    assert np.array_equal(ids, before)


def test_unclassified_candidate_has_no_same_class_match() -> None:
    rows = _evaluate(
        gaussian_xyz=np.asarray([[0.0, 0, 0]]),
        gt_xyz=np.asarray([[0.0, 0, 0]]),
        candidate_ids=np.asarray([0]),
        candidate_class=None,
    )
    primary = _row(rows, 0.05)

    assert primary["official_geometric_iou"] == pytest.approx(1.0)
    assert primary["official_same_class_iou"] == pytest.approx(0.0)
    assert primary["precision_target_instance_id"] is None
    assert primary["gaussian_to_gt_precision"] == pytest.approx(0.0)
