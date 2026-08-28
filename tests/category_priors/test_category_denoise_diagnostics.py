from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from category_priors.category_denoise import CandidateBank
from category_priors.category_denoise_diagnostics import (
    aggregate_candidate_funnel_results,
    build_bidirectional_mapping,
    build_official_gt_objects,
    diagnose_candidate_funnel_scene,
)
from category_priors.evaluator import GroundTruthScene
from category_priors.taxonomy import Taxonomy


def _class_names() -> tuple[str, ...]:
    return ("chair", "table", "wall") + tuple(
        f"class-{index}" for index in range(3, 32)
    )


def _taxonomy() -> Taxonomy:
    return Taxonomy(
        schema_version="test",
        benchmark_name="test",
        canonical_classes=("chair", "table"),
        dataset_mappings={},
        parents={},
        unsupported_saga_classes=(),
        content_hash="test",
    )


def _prior_node(*, area: float) -> dict[str, object]:
    geometry = {
        name: {"q50": math.log(1.0), "q75": math.log(2.0)}
        for name in (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        )
    }
    geometry["log_surface_area_m2"] = {"q50": math.log(area)}
    return {
        "shrunk": {
            "geometry": geometry,
            "neighborhood": {
                "boundary_fixed:0.05": {"q50": 0.10, "q75": 0.20}
            },
        }
    }


def _priors() -> dict[str, object]:
    return {
        "global": _prior_node(area=1.0),
        "categories": {
            "chair": _prior_node(area=0.36),
            "table": _prior_node(area=0.36),
        },
    }


def _candidate(
    candidate_id: int,
    branch_class: str,
    branch_class_index: int,
    *,
    core_count: int = 3,
    full_count: int = 3,
) -> dict[str, object]:
    ratios = {"chair": (0.70, 0.0), "table": (0.0, 0.70)}[branch_class]
    winner_index = 0 if branch_class == "chair" else 1
    return {
        "candidate_id": candidate_id,
        "branch_class": branch_class,
        "branch_class_index": branch_class_index,
        "core_point_count": core_count,
        "full_point_count": full_count,
        "assignment_confidence_mean": 0.8,
        "metric_extents_m": [0.1, 0.1, 0.1],
        "boundary_ratio_5cm": 0.10,
        "vote_winner_index": winner_index,
        "vote_winner": branch_class,
        "vote_winner_unique": True,
        "branch_vote_ratio": ratios[winner_index],
        "background_vote_ratio": 0.30,
        "base_score": 0.56,
    }


def _scene_fixture(scene_id: str = "scene0001_00") -> tuple[
    CandidateBank,
    np.ndarray,
    np.ndarray,
    GroundTruthScene,
]:
    # Six chair points form GT instance 10, three form instance 11, and three
    # form one table.  Candidate 0/1 full masks are duplicate fragments of
    # instance 10, while candidate 1's retained raw core is instance 11.
    xyz = np.asarray(
        [[index * 0.01, 0.0, 0.0] for index in range(6)]
        + [[1.0 + index * 0.01, 0.0, 0.0] for index in range(3)]
        + [[2.0 + index * 0.01, 0.0, 0.0] for index in range(3)],
        dtype=np.float64,
    )
    semantic = np.asarray([0] * 9 + [1] * 3, dtype=np.int64)
    instance = np.asarray([10] * 6 + [11] * 3 + [20] * 3, dtype=np.int64)
    core = np.full(12, -1, dtype=np.int64)
    core[[0, 1, 3]] = 0
    core[[6, 7, 8]] = 1
    core[[9, 10, 11]] = 2
    full = np.full(12, -1, dtype=np.int64)
    full[[0, 1, 2]] = 0
    full[[3, 4, 5]] = 1
    full[[9, 10, 11]] = 2
    top1 = np.asarray([0] * 9 + [1] * 3, dtype=np.int64)
    table_candidate = _candidate(2, "table", 1)
    table_candidate.update(
        {
            "vote_winner_index": -1,
            "vote_winner": "background",
            "branch_vote_ratio": 0.20,
            "background_vote_ratio": 0.80,
            "base_score": 0.16,
        }
    )
    bank = CandidateBank(
        class_names=_class_names(),
        saga20_names=("chair", "table"),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.full(12, -1, dtype=np.int64),
        semantic_top1=top1,
        semantic_top1_score=np.full(12, 0.9, dtype=np.float64),
        branch_full_labels=full,
        branch_core_labels=core,
        assignment_confidence=np.where(full >= 0, 0.8, 0.0),
        candidates=(
            _candidate(0, "chair", 0),
            _candidate(1, "chair", 0),
            table_candidate,
        ),
        diagnostics={"scene_id": scene_id},
    )
    ground_truth = GroundTruthScene(scene_id, semantic, instance)
    return bank, xyz, xyz.copy(), ground_truth


def _diagnose(scene_id: str = "scene0001_00"):
    bank, gaussian_xyz, gt_xyz, ground_truth = _scene_fixture(scene_id)
    size_spec = {
        "boundaries_m": {
            "tiny_max_m": 0.10,
            "small_max_m": 0.50,
            "medium_max_m": 1.0,
        }
    }
    return diagnose_candidate_funnel_scene(
        scene_id=scene_id,
        bank=bank,
        gaussian_xyz=gaussian_xyz,
        gt_xyz=gt_xyz,
        ground_truth=ground_truth,
        taxonomy=_taxonomy(),
        category_priors=_priors(),
        size_spec=size_spec,
        radius_m=0.05,
        min_region_size=1,
    )


def test_bidirectional_mapping_reuses_official_direction_and_is_not_inverse() -> None:
    gt_xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.02, 0.0, 0.0], [1.00, 0.0, 0.0]],
        dtype=np.float64,
    )
    gaussian_xyz = np.asarray(
        [[0.01, 0.0, 0.0], [2.00, 0.0, 0.0]], dtype=np.float64
    )

    mapping = build_bidirectional_mapping(gt_xyz, gaussian_xyz, radius_m=0.05)

    np.testing.assert_array_equal(mapping.gt_to_gaussian.indices, [0, 0, -1])
    np.testing.assert_array_equal(mapping.gaussian_to_gt.indices, [0, -1])
    assert mapping.gt_to_gaussian.indices.flags.writeable is False
    assert mapping.gaussian_to_gt.distances_m.flags.writeable is False
    assert mapping.gt_to_gaussian.diagnostics["mapped_fraction"] == pytest.approx(2 / 3)


def test_official_gt_objects_filter_small_regions_and_keep_compound_identity() -> None:
    scene_id = "scene0123_02"
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    ground_truth = GroundTruthScene(
        scene_id,
        semantic=np.asarray([0, 0, 1], dtype=np.int64),
        instance=np.asarray([7, 7, 7], dtype=np.int64),
    )
    size_spec = {
        "boundaries_m": {
            "tiny_max_m": 0.05,
            "small_max_m": 0.2,
            "medium_max_m": 1.0,
        }
    }

    objects = build_official_gt_objects(
        scene_id,
        xyz,
        ground_truth,
        _taxonomy(),
        min_region_size=2,
        size_spec=size_spec,
    )

    assert len(objects) == 1
    assert (objects[0].class_id, objects[0].instance_id) == (0, 7)
    assert objects[0].physical_scene_id == "scene0123"
    assert objects[0].size_bin == "small"


def test_funnel_audits_non_nested_core_and_mode_specific_support() -> None:
    result = _diagnose()
    by_candidate = {row["candidate_id"]: row for row in result.candidate_rows}

    first = by_candidate[0]
    assert first["core_actual_count"] == first["full_actual_count"] == 3
    assert first["core_intersection_full_count"] == 2
    assert first["core_only_count"] == first["full_only_count"] == 1
    assert first["core_subset_full"] is False
    assert first["core_best_same_class_iou"] == pytest.approx(0.5)
    assert first["full_best_same_class_iou"] == pytest.approx(0.5)

    # Uniform support is five, class-shrunk support is three.  P0/P4 must not
    # collapse these two experimental conditions into one ambiguous gate.
    assert first["U_support_threshold"] == 5
    assert first["D_support_threshold"] == 3
    assert first["P0_vote"] is True
    assert first["P0_U_support"] is False
    assert first["P0_D_support"] is True
    assert first["P4_U"] is first["U_accepted"] is False
    assert first["P4_D"] is first["D_accepted"] is True
    assert result.analysis["core_subset_violation_count"] == 2
    assert result.analysis["conclusion_boundary"]["S1_scope"] == (
        "retained_raw_core_only"
    )


def test_funnel_keeps_stage_best_and_same_candidate_deltas_separate() -> None:
    result = _diagnose()
    chair_second = next(
        row
        for row in result.gt_rows
        if row["gt_class"] == "chair" and row["gt_instance_id"] == 11
    )

    assert chair_second["S0_semantic_coverage"] == 1.0
    assert chair_second["S0_gaussian_purity"] == pytest.approx(1 / 3)
    assert chair_second["S0_gaussian_same_class_precision"] == 1.0
    assert chair_second["S1_best_same_candidate_id"] == 1
    assert chair_second["S1_best_same_iou_raw_core"] == 1.0
    assert chair_second["S2_best_same_candidate_id"] is None
    assert chair_second["S2_best_same_iou"] == 0.0
    assert chair_second["funnel_flag_full_assignment_harmful"] is True
    assert chair_second["funnel_status"] == "full_assignment_harmful"


def test_counterfactual_candidate_precision_does_not_double_count_gt_recall() -> None:
    result = _diagnose()
    layer = result.analysis["counterfactual_filters"]["P4_D"]

    # Both accepted chair fragments have IoU 0.5 with instance 10, but they
    # recover only one of the two chair GT objects.  Candidate precision and GT
    # recall therefore have different numerators and denominators.
    assert layer["same_class_iou_025_candidate_count"] == 2
    assert layer["gt_recall_025_count"] == 1
    assert layer["official_valid_gt_count"] == 3
    assert layer["gt_recall_025"] == pytest.approx(1 / 3)


def test_aggregate_is_scene_equal_and_preserves_score_domain_and_status() -> None:
    first = _diagnose("scene0001_00")
    second = _diagnose("scene0002_00")

    analysis = aggregate_candidate_funnel_results((second, first))

    assert analysis["scene_ids"] == ["scene0001_00", "scene0002_00"]
    assert analysis["scene_count"] == 2
    assert analysis["candidate_count"] == 6
    assert analysis["official_valid_gt_count"] == 6
    assert analysis["core_subset_violation_count"] == 4
    assert analysis["scene_equal"]["candidate_precision_025"] == pytest.approx(1.0)
    assert analysis["score_domain"]["Q"]["q50"] == pytest.approx(0.56)
    assert analysis["funnel_status_counts"]["full_assignment_harmful"] == 2
    assert (
        analysis["conclusion_boundary"]["candidate_rows_are_independent_replicates"]
        is False
    )


def test_funnel_rejects_point_count_and_scene_identity_mismatches() -> None:
    bank, gaussian_xyz, gt_xyz, ground_truth = _scene_fixture()
    kwargs = {
        "scene_id": "scene0001_00",
        "bank": bank,
        "gaussian_xyz": gaussian_xyz,
        "gt_xyz": gt_xyz,
        "ground_truth": ground_truth,
        "taxonomy": _taxonomy(),
        "category_priors": _priors(),
        "min_region_size": 1,
    }

    with pytest.raises(ValueError, match="candidate bank"):
        diagnose_candidate_funnel_scene(
            **{**kwargs, "gaussian_xyz": gaussian_xyz[:-1]}
        )
    with pytest.raises(ValueError, match="scene_id"):
        diagnose_candidate_funnel_scene(
            **{
                **kwargs,
                "ground_truth": replace(ground_truth, scene_id="scene9999_00"),
            }
        )
