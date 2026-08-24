from __future__ import annotations

import numpy as np

from category_priors.gaussian_object_audit import (
    _export_viewer_case,
    evaluate_gaussian_object_precision,
)


CLASSES = ("chair", "cup")


def audit(
    gaussian_xyz: list[list[float]],
    labels: list[int],
    instances: dict[str, dict[str, str]],
    gt_xyz: list[list[float]],
    gt_semantic: list[int],
    gt_instance: list[int],
):
    return evaluate_gaussian_object_precision(
        np.asarray(gaussian_xyz, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        instances,
        np.asarray(gt_xyz, dtype=np.float64),
        np.asarray(gt_semantic, dtype=np.int64),
        np.asarray(gt_instance, dtype=np.int64),
        radius_m=0.05,
        canonical_classes=CLASSES,
    )


def test_perfect_object_has_unit_gaussian_precision_and_recall() -> None:
    result = audit(
        [[0, 0, 0], [0.02, 0, 0]],
        [10, 10],
        {"10": {"class": "chair"}},
        [[0, 0, 0], [0.02, 0, 0]],
        [0, 0],
        [1, 1],
    )
    row = result["instances"][0]
    assert row["point_precision"] == 1.0
    assert row["semantic_precision"] == 1.0
    assert row["gt_to_gaussian_recall"] == 1.0
    assert row["official_iou"] == 1.0


def test_unmapped_extra_gaussian_counts_as_false_positive() -> None:
    result = audit(
        [[0, 0, 0], [0.02, 0, 0], [5, 0, 0]],
        [10, 10, 10],
        {"10": {"class": "chair"}},
        [[0, 0, 0], [0.02, 0, 0]],
        [0, 0],
        [1, 1],
    )
    row = result["instances"][0]
    assert row["correct_gaussian_count"] == 2
    assert row["unsupported_count"] == 1
    assert row["point_precision"] == 2 / 3
    assert result["aggregate"]["micro_point_precision"] == 2 / 3


def test_missing_gaussians_reduce_recall_without_reducing_precision() -> None:
    result = audit(
        [[0, 0, 0]],
        [10],
        {"10": {"class": "chair"}},
        [[0, 0, 0], [0.20, 0, 0], [0.40, 0, 0]],
        [0, 0, 0],
        [1, 1, 1],
    )
    row = result["instances"][0]
    assert row["point_precision"] == 1.0
    assert row["gt_to_gaussian_recall"] == 1 / 3


def test_wrong_class_is_not_a_dominant_match() -> None:
    result = audit(
        [[0, 0, 0], [0.02, 0, 0]],
        [10, 10],
        {"10": {"class": "cup"}},
        [[0, 0, 0], [0.02, 0, 0]],
        [0, 0],
        [1, 1],
    )
    row = result["instances"][0]
    assert row["dominant_gt_instance"] is None
    assert row["wrong_class_count"] == 2
    assert row["point_precision"] == 0.0


def test_same_class_wrong_instance_is_separate_from_wrong_class() -> None:
    result = audit(
        [[0, 0, 0], [0.20, 0, 0], [0.22, 0, 0]],
        [10, 10, 10],
        {"10": {"class": "chair"}},
        [[0, 0, 0], [0.20, 0, 0], [0.22, 0, 0]],
        [0, 0, 0],
        [1, 2, 2],
    )
    row = result["instances"][0]
    assert row["dominant_gt_instance"] == 2
    assert row["correct_gaussian_count"] == 2
    assert row["same_class_wrong_instance_count"] == 1
    assert row["wrong_class_count"] == 0
    assert row["merge_candidate"] is True


def test_two_predictions_for_one_gt_mark_lower_quality_duplicate() -> None:
    result = audit(
        [[0, 0, 0], [0.02, 0, 0]],
        [10, 11],
        {"10": {"class": "chair"}, "11": {"class": "chair"}},
        [[0, 0, 0], [0.02, 0, 0]],
        [0, 0],
        [1, 1],
    )
    rows = {row["instance_id"]: row for row in result["instances"]}
    assert sum(int(row["duplicate_prediction"]) for row in rows.values()) == 1
    assert result["aggregate"]["duplicate_prediction_count"] == 1


def test_audit_does_not_mutate_prediction_labels() -> None:
    labels = np.asarray([10, -1], dtype=np.int64)
    before = labels.copy()
    evaluate_gaussian_object_precision(
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64),
        labels,
        {"10": {"class": "chair"}},
        np.asarray([[0, 0, 0]], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        canonical_classes=CLASSES,
    )
    assert np.array_equal(labels, before)


def test_negative_metadata_id_is_background_not_a_predicted_instance() -> None:
    result = audit(
        [[0, 0, 0], [0.02, 0, 0]],
        [-1, 10],
        {"-1": {"class": "cup"}, "10": {"class": "chair"}},
        [[0, 0, 0], [0.02, 0, 0]],
        [0, 0],
        [1, 1],
    )

    assert [row["instance_id"] for row in result["instances"]] == [10]
    assert result["aggregate"]["predicted_instance_count"] == 1
    assert result["aggregate"]["predicted_gaussian_count"] == 1


def test_negative_metadata_only_produces_no_prediction() -> None:
    result = audit(
        [[0, 0, 0]],
        [-1],
        {"-1": {"class": "chair"}},
        [[0, 0, 0]],
        [0],
        [1],
    )

    assert result["instances"] == []
    assert result["aggregate"]["predicted_instance_count"] == 0
    assert result["aggregate"]["predicted_gaussian_count"] == 0


def test_viewer_exports_predicted_gt_overlay_and_metrics(tmp_path) -> None:
    gaussian_xyz = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
    labels = np.asarray([10, -1], dtype=np.int64)
    gt_xyz = np.asarray([[0, 0, 0]], dtype=np.float64)
    gt_semantic = np.asarray([0], dtype=np.int64)
    gt_instance = np.asarray([1], dtype=np.int64)
    result = evaluate_gaussian_object_precision(
        gaussian_xyz,
        labels,
        {"10": {"class": "chair"}},
        gt_xyz,
        gt_semantic,
        gt_instance,
        canonical_classes=CLASSES,
    )
    case = {
        "role": "highest_precision",
        "scene_id": "scene",
        "condition": "B0-legacy",
        **result["instances"][0],
    }
    exported = _export_viewer_case(
        case,
        result,
        gt_xyz,
        gt_semantic,
        gt_instance,
        gaussian_xyz,
        labels,
        tmp_path,
    )
    directory = tmp_path / "scene" / "b0_legacy" / "highest_precision-instance-10"
    assert exported["directory"] == str(directory)
    assert (directory / "predicted_gaussians.ply").is_file()
    assert (directory / "matched_gt_points.ply").is_file()
    assert (directory / "overlay.ply").is_file()
    assert (directory / "metrics.json").is_file()
