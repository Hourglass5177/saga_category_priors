from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.spatial import KDTree

from category_priors.category_denoise import (
    CandidateBank,
    legacy_knn_filter,
)
from category_priors.category_denoise_knn_oracle import (
    ExactB0MappingError,
    prepare_knn_oracle_scene,
    project_oracle_prediction,
    recover_exact_b0_mapping,
    replay_knn_oracle_scene,
    replay_protected_oracle,
    replay_unprotected_oracle,
)
from category_priors.prediction_contract import validate_prediction_contract


def _classes() -> tuple[str, ...]:
    return ("chair", "table", "sofa") + tuple(
        f"class-{index}" for index in range(3, 32)
    )


def _bank(
    *,
    point_count: int,
    full_labels: np.ndarray,
    global_labels: np.ndarray,
    candidates: tuple[dict[str, object], ...],
    core_labels: np.ndarray | None = None,
) -> CandidateBank:
    if core_labels is None:
        core_labels = np.asarray(full_labels, dtype=np.int64)
    semantic = np.zeros(point_count, dtype=np.int64)
    for row in candidates:
        candidate_id = int(row["candidate_id"])
        semantic[np.asarray(full_labels) == candidate_id] = int(
            row["branch_class_index"]
        )
        semantic[np.asarray(core_labels) == candidate_id] = int(
            row["branch_class_index"]
        )
    return CandidateBank(
        class_names=_classes(),
        saga20_names=("chair", "table", "sofa"),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.asarray(global_labels, dtype=np.int64),
        semantic_top1=semantic,
        semantic_top1_score=np.ones(point_count, dtype=np.float64),
        branch_full_labels=np.asarray(full_labels, dtype=np.int64),
        branch_core_labels=np.asarray(core_labels, dtype=np.int64),
        assignment_confidence=np.ones(point_count, dtype=np.float64),
        candidates=candidates,
        diagnostics={"scene_id": "toy"},
    )


def _candidate(
    candidate_id: int,
    class_name: str,
    full_count: int,
    core_count: int | None = None,
    base_score: float = 0.4,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "branch_class": class_name,
        "branch_class_index": _classes().index(class_name),
        "core_point_count": core_count if core_count is not None else full_count,
        "full_point_count": full_count,
        "assignment_confidence_mean": 1.0,
        "metric_extents_m": [0.1, 0.1, 0.1],
        "boundary_ratio_5cm": 0.0,
        "base_score": base_score,
    }


def _legacy_scalar_reference(
    xyz: np.ndarray, labels: np.ndarray, *, k: int, min_count: int
) -> tuple[np.ndarray, np.ndarray]:
    k_effective = min(max(int(k), 1), len(xyz))
    tree = KDTree(xyz)
    after_knn: list[int] = []
    for point in xyz:
        _, neighbor_ids = tree.query(point, k=k_effective)
        bins: list[int] = []
        counts: list[int] = []
        for value in labels[np.asarray(neighbor_ids).reshape(-1)]:
            label = int(value)
            if label in bins:
                counts[bins.index(label)] += 1
            else:
                bins.append(label)
                counts.append(1)
        after_knn.append(bins[counts.index(max(counts))])
    voted = np.asarray(after_knn, dtype=np.int64)
    filtered = voted.copy()
    values, counts = np.unique(voted[voted >= 0], return_counts=True)
    for value, count in zip(values, counts):
        if int(count) < min_count:
            filtered[voted == value] = -1
    return voted, filtered


def test_legacy_knn_filter_matches_scalar_historical_tie_rule() -> None:
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 1.0],
            [0.5, 0.5, -1.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([4, 7, -1, 7, 4, -1], dtype=np.int64)
    expected_knn, expected_filter = _legacy_scalar_reference(
        xyz, labels, k=4, min_count=2
    )

    result = legacy_knn_filter(xyz, labels, k=4, min_count=2, chunk_size=2)

    np.testing.assert_array_equal(result.after_knn, expected_knn)
    np.testing.assert_array_equal(result.after_filter, expected_filter)
    assert result.k_effective == 4
    assert not result.after_knn.flags.writeable
    assert not result.after_filter.flags.writeable


def test_prepare_selects_only_threshold_candidate_and_freezes_smallest_gt_tie() -> None:
    # Candidate 0 contains one point from each of two same-class GT instances.
    # Its IoU is exactly 0.50 with both; stable GT ordering must freeze id 10.
    xyz = np.column_stack(
        [np.arange(4, dtype=np.float64), np.zeros(4), np.zeros(4)]
    )
    full = np.asarray([0, 0, 1, 1], dtype=np.int64)
    bank = _bank(
        point_count=4,
        full_labels=full,
        global_labels=np.full(4, -1, dtype=np.int64),
        candidates=(
            _candidate(0, "chair", 2),
            _candidate(1, "table", 2),
        ),
    )
    semantic = np.asarray([0, 0, 1, 1], dtype=np.int64)
    instance = np.asarray([10, 11, 20, 20], dtype=np.int64)

    plan = prepare_knn_oracle_scene(
        scene_id="toy",
        bank=bank,
        gaussian_xyz_metric=xyz,
        gt_xyz=xyz,
        gt_semantic=semantic,
        gt_instance=instance,
        canonical_classes=("chair", "table", "sofa"),
        iou_threshold=0.50,
        radius_m=0.01,
        min_region_size=1,
    )

    assert plan.evaluation_only
    assert plan.candidate_ids == (0, 1)
    chair = plan.candidates[0]
    assert chair.same_class_iou == 0.50
    assert chair.matched_gt_instance_id == 10
    assert chair.matched_gt_class == "chair"
    assert chair.gaussian_target_precision == 0.50
    assert plan.to_dict()["min_region_size"] == 1


def test_prepare_does_not_fall_back_below_half_iou() -> None:
    xyz = np.column_stack(
        [np.arange(5, dtype=np.float64), np.zeros(5), np.zeros(5)]
    )
    full = np.asarray([0, 0, -1, -1, -1], dtype=np.int64)
    bank = _bank(
        point_count=5,
        full_labels=full,
        global_labels=np.full(5, -1, dtype=np.int64),
        candidates=(_candidate(0, "chair", 2),),
    )

    plan = prepare_knn_oracle_scene(
        scene_id="toy",
        bank=bank,
        gaussian_xyz_metric=xyz,
        gt_xyz=xyz,
        gt_semantic=np.zeros(5, dtype=np.int64),
        gt_instance=np.full(5, 10, dtype=np.int64),
        canonical_classes=("chair", "table", "sofa"),
        radius_m=0.01,
        min_region_size=1,
    )

    assert plan.candidate_ids == ()  # best IoU is 2/5 = 0.40


def test_recover_exact_b0_mapping_projects_unmapped_raw_labels_to_background() -> None:
    raw = np.asarray([7, 7, 9, 9, -1], dtype=np.int64)
    b0 = np.asarray([0, 0, -1, -1, -1], dtype=np.int64)
    instances = {"0": {"class": "chair", "score": 0.8}}

    mapping = recover_exact_b0_mapping(raw, b0, instances)

    assert mapping.raw_to_b0_instance == {7: 0}
    assert mapping.class_by_raw == {7: "chair"}
    np.testing.assert_array_equal(mapping.baseline_projected_labels, b0)
    assert mapping.diagnostics()["exact"]


def test_recover_exact_b0_mapping_does_not_coerce_noninteger_strict_labels() -> None:
    with pytest.raises(ExactB0MappingError, match="not strict"):
        recover_exact_b0_mapping(
            np.asarray([7, 7], dtype=np.int64),
            np.asarray([0.0, 0.0], dtype=np.float64),
            {"0": {"class": "chair", "score": 0.8}},
        )


@pytest.mark.parametrize(
    ("raw", "b0"),
    (
        ([3, 3, 4, -1], [0, 0, 0, -1]),  # one B0 mask spans two raw labels
        ([3, 3, 3, -1], [0, 0, -1, -1]),  # raw label has extra B0-background support
    ),
)
def test_recover_exact_b0_mapping_rejects_partial_or_ambiguous_masks(
    raw: list[int], b0: list[int]
) -> None:
    with pytest.raises(ExactB0MappingError):
        recover_exact_b0_mapping(
            np.asarray(raw, dtype=np.int64),
            np.asarray(b0, dtype=np.int64),
            {"0": {"class": "chair", "score": 0.8}},
        )


def test_o1_can_be_swallowed_while_o2_preserves_full_mask_and_strict_contract() -> None:
    xyz = np.column_stack(
        [np.arange(10, dtype=np.float64), np.zeros(10), np.zeros(10)]
    )
    full = np.asarray([0, 0, 0] + [-1] * 7, dtype=np.int64)
    bank = _bank(
        point_count=10,
        full_labels=full,
        global_labels=np.asarray([7, 7, 7] + [-1] * 7, dtype=np.int64),
        candidates=(_candidate(0, "chair", 3),),
    )
    baseline = legacy_knn_filter(xyz, bank.global_pre_knn, k=10, min_count=1)
    np.testing.assert_array_equal(baseline.after_filter, np.full(10, -1))
    b0_mapping = recover_exact_b0_mapping(
        baseline.after_filter,
        np.full(10, -1, dtype=np.int64),
        {},
    )

    replay = replay_knn_oracle_scene(
        xyz_scene=xyz,
        bank=bank,
        candidate_ids=(0,),
        b0_mapping=b0_mapping,
        k=10,
        min_count=1,
    )

    o1 = replay.unprotected.candidates[0]
    o2 = replay.protected.candidates[0]
    assert o1.retained_original_after_knn == 0
    assert not o1.survived_after_filter
    assert replay.o1_prediction.candidate_to_strict[0] is None
    assert o2.retained_original_after_filter == 3
    assert o2.gained_outside_after_filter == 0
    np.testing.assert_array_equal(
        replay.o2_prediction.point_labels,
        np.asarray([0, 0, 0] + [-1] * 7),
    )
    assert replay.o2_prediction.instances["0"]["class"] == "chair"
    validate_prediction_contract(
        replay.o2_prediction.point_labels, replay.o2_prediction.instances
    )


def test_o1_outside_expansion_is_counted() -> None:
    xyz = np.column_stack(
        [np.arange(5, dtype=np.float64), np.zeros(5), np.zeros(5)]
    )
    full = np.asarray([0, 0, 0, -1, -1], dtype=np.int64)
    bank = _bank(
        point_count=5,
        full_labels=full,
        global_labels=np.full(5, -1, dtype=np.int64),
        candidates=(_candidate(0, "chair", 3),),
    )

    replay = replay_unprotected_oracle(
        xyz_scene=xyz,
        bank=bank,
        candidate_ids=(0,),
        k=5,
        min_count=1,
    )

    assert replay.candidates[0].retained_original_after_filter == 3
    assert replay.candidates[0].gained_outside_after_filter == 2


def test_protected_inserted_id_wins_over_disappeared_raw_global_metadata() -> None:
    # Candidate points are raw label 1 in B0.  Removing them from the active
    # partition leaves max raw label 0, so protected replay reuses numeric ID 1
    # for the candidate.  It must inherit chair, not raw label 1's old sofa.
    xyz = np.column_stack(
        [np.arange(6, dtype=np.float64), np.zeros(6), np.zeros(6)]
    )
    full = np.asarray([0, 0, 0, -1, -1, -1], dtype=np.int64)
    global_labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    bank = _bank(
        point_count=6,
        full_labels=full,
        global_labels=global_labels,
        candidates=(_candidate(0, "chair", 3),),
    )
    b0_labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    b0_instances = {
        "0": {"class": "table", "score": 0.8},
        "1": {"class": "sofa", "score": 0.7},
    }
    b0_mapping = recover_exact_b0_mapping(
        global_labels, b0_labels, b0_instances
    )

    protected = replay_protected_oracle(
        xyz_scene=xyz,
        bank=bank,
        candidate_ids=(0,),
        k=1,
        min_count=1,
    )
    assert protected.candidate_raw_labels == {0: 1}
    prediction = project_oracle_prediction(
        xyz_scene=xyz,
        raw_labels=protected.after_filter,
        bank=bank,
        candidate_raw_labels=protected.candidate_raw_labels,
        b0_mapping=b0_mapping,
    )

    candidate_strict_id = prediction.candidate_to_strict[0]
    assert candidate_strict_id is not None
    assert prediction.instances[str(candidate_strict_id)]["class"] == "chair"
    assert set(row["class"] for row in prediction.instances.values()) == {
        "chair",
        "table",
    }
    validate_prediction_contract(prediction.point_labels, prediction.instances)


def test_replay_public_functions_have_no_gt_inputs() -> None:
    forbidden = {"gt", "iou", "radius", "semantic", "instance"}
    for function in (
        replay_unprotected_oracle,
        replay_protected_oracle,
        replay_knn_oracle_scene,
    ):
        names = set(inspect.signature(function).parameters)
        assert not any(
            token in name for token in forbidden for name in names
        )
