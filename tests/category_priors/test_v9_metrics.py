from __future__ import annotations

import numpy as np
import pytest

from category_priors.evaluator import GroundTruthScene, PredictedInstance
from category_priors.v9_evaluation import GroundTruthSupport
from category_priors.v9_metrics import (
    ASSOCIATION_TRUTH_MIN_INTERSECTION,
    ASSOCIATION_TRUTH_MIN_IOU,
    ASSOCIATION_TRUTH_MIN_PURITY,
    _qualifying_association_gt,
    _unique_official_gt_coverage,
)


def test_unique_gt_coverage_counts_misses_once_and_uses_best_duplicate() -> None:
    # Two official chair instances plus one sub-threshold GT that must not enter
    # either denominator.
    gt = GroundTruthScene(
        "scene",
        np.asarray([0] * 200 + [1] * 50, dtype=np.int64),
        np.asarray([0] * 100 + [1] * 100 + [2] * 50, dtype=np.int64),
    )
    predictions = [
        PredictedInstance(
            "scene", 0, 0, 0.9,
            np.asarray([1] * 50 + [0] * 200, dtype=bool),
        ),
        # Duplicate of GT 0 with better coverage: only this 75-point support
        # may count.  It must not add to the first prediction's 50 points.
        PredictedInstance(
            "scene", 1, 0, 0.8,
            np.asarray([1] * 75 + [0] * 175, dtype=bool),
        ),
        # Full geometric coverage of GT 1 but the wrong class.  GT 1 therefore
        # remains a missed official instance and contributes recall zero.
        PredictedInstance(
            "scene", 2, 1, 0.7,
            np.asarray([0] * 100 + [1] * 100 + [0] * 50, dtype=bool),
        ),
    ]

    result = _unique_official_gt_coverage(
        gt, predictions, min_region_size=100
    )

    assert result["official_gt_instance_count"] == 2
    assert result["official_gt_point_count"] == 200
    assert result["best_covered_gt_point_count"] == 75
    assert result["gt_instance_macro_recall"] == pytest.approx(0.375)
    assert result["gt_point_micro_recall"] == pytest.approx(0.375)


def _gt_support() -> GroundTruthSupport:
    return GroundTruthSupport(
        scene_id="scene",
        instance_id=7,
        support_ids=np.arange(100, dtype=np.int64),
        class_name="chair",
        size_bin="small",
        support_count=100,
        official_valid=True,
    )


def test_association_truth_rejects_single_point_and_each_weak_support_case() -> None:
    gt = (_gt_support(),)
    assert ASSOCIATION_TRUTH_MIN_INTERSECTION == 3
    assert ASSOCIATION_TRUTH_MIN_IOU == pytest.approx(0.05)
    assert ASSOCIATION_TRUTH_MIN_PURITY == pytest.approx(0.50)

    # Accidental one-point overlap is never an association identity label.
    assert _qualifying_association_gt(
        np.asarray([0, 200, 201], dtype=np.int64), gt
    ) is None
    # Three pure points satisfy minimum intersection/purity but not IoU.
    assert _qualifying_association_gt(
        np.asarray([0, 1, 2], dtype=np.int64), gt
    ) is None
    # Ten GT points plus ninety foreign points satisfy IoU but not purity.
    assert _qualifying_association_gt(
        np.concatenate((np.arange(10), np.arange(200, 290))).astype(np.int64),
        gt,
    ) is None

    assert _qualifying_association_gt(
        np.arange(10, dtype=np.int64), gt
    ) == ("chair", 7)
