from __future__ import annotations

import math

import numpy as np

from category_priors.category_denoise_prior_oracle import (
    SCORE_FLOOR,
    build_oracle_scene,
    deterministic_pca_half_fragment,
    gaussian_to_official_gt_assignments,
    scan_physical_equal_summary,
    summarize_prior_oracle,
    summarize_score_domain,
)


def _node(
    *,
    extent_q50: float,
    extent_q75: float,
    boundary_q50: float,
    boundary_q75: float,
    area: float,
) -> dict:
    return {
        "shrunk": {
            "geometry": {
                "log_extent_short_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_extent_mid_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_extent_long_m": {
                    "q50": math.log(extent_q50),
                    "q75": math.log(extent_q75),
                },
                "log_surface_area_m2": {"q50": math.log(area)},
            },
            "neighborhood": {
                "boundary_fixed:0.05": {
                    "q50": boundary_q50,
                    "q75": boundary_q75,
                }
            },
        }
    }


def _priors() -> dict:
    return {
        "global": _node(
            extent_q50=0.20,
            extent_q75=0.40,
            boundary_q50=0.10,
            boundary_q75=0.30,
            area=1.0,
        ),
        "categories": {
            "chair": _node(
                extent_q50=0.05,
                extent_q75=0.10,
                boundary_q50=0.00,
                boundary_q75=0.10,
                area=0.25,
            ),
            "table": _node(
                extent_q50=2.0,
                extent_q75=3.0,
                boundary_q50=0.00,
                boundary_q75=0.10,
                area=4.0,
            ),
        },
    }


def test_gaussian_mapping_queries_all_gt_before_official_filter() -> None:
    gt_xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],  # void and closest to Gaussian 0
            [0.04, 0.0, 0.0],
            [0.04, 0.01, 0.0],
            [0.04, 0.02, 0.0],
        ]
    )
    semantic = np.asarray([-1, 0, 0, 0])
    instance = np.asarray([-1, 7, 7, 7])
    gaussians = np.asarray([[0.001, 0.0, 0.0], [0.04, 0.01, 0.0]])
    official = {
        (0, 7): {
            "class_id": 0,
            "class_name": "chair",
            "instance_id": 7,
        }
    }

    assignments, diagnostics = gaussian_to_official_gt_assignments(
        gaussians,
        gt_xyz,
        semantic,
        instance,
        official,
        radius_m=0.05,
    )

    assert assignments[(0, 7)].tolist() == [1]
    assert diagnostics["nearest_nonofficial_or_void_count"] == 1


def test_pca_half_fragment_is_deterministic_and_uses_half() -> None:
    xyz = np.column_stack((np.arange(6, dtype=float), np.zeros(6), np.zeros(6)))
    ids = np.arange(6, dtype=np.int64)

    first = deterministic_pca_half_fragment(ids, xyz)
    second = deterministic_pca_half_fragment(ids[::-1], xyz)

    assert first.tolist() == [0, 1, 2]
    assert np.array_equal(first, second)


def test_oracle_uses_target_class_for_fragment_and_nearest_merge() -> None:
    chair = np.asarray(
        [
            [0.00, 0.00, 0.0],
            [0.02, 0.00, 0.0],
            [0.04, 0.00, 0.0],
            [0.00, 0.02, 0.0],
            [0.02, 0.02, 0.0],
            [0.04, 0.02, 0.0],
        ]
    )
    table = np.asarray(
        [
            [0.30, 0.00, 0.0],
            [0.32, 0.00, 0.0],
            [0.34, 0.00, 0.0],
            [0.30, 0.02, 0.0],
            [0.32, 0.02, 0.0],
            [0.34, 0.02, 0.0],
        ]
    )
    gt_xyz = np.concatenate((chair, table), axis=0)
    semantic = np.asarray([0] * 6 + [1] * 6)
    instance = np.asarray([10] * 6 + [20] * 6)
    transform = np.eye(4)

    result = build_oracle_scene(
        scene_id="scene0000_00",
        physical_scene_id="scene0000",
        gaussian_xyz=gt_xyz,
        gaussian_to_gt_transform=transform,
        gt_xyz_m=gt_xyz,
        gt_semantic=semantic,
        gt_instance=instance,
        class_names=("chair", "table"),
        priors=_priors(),
        radii_m=(0.05,),
        min_region_size=3,
    )

    chair_merge = next(
        row
        for row in result["pairs"]
        if row["class_name"] == "chair" and row["negative_type"] == "merge"
    )
    chair_fragment = next(
        row
        for row in result["pairs"]
        if row["class_name"] == "chair" and row["negative_type"] == "fragment"
    )
    assert chair_merge["negative_class_name"] == "table"
    assert math.isclose(chair_merge["merge_centroid_distance_m"], 0.30)
    assert chair_merge["full_D_size_score"] > chair_merge["negative_D_size_score"]
    # The registered size formula only penalizes objects that are too large.
    assert math.isclose(
        chair_fragment["full_D_size_score"],
        chair_fragment["negative_D_size_score"],
    )


def test_score_domain_floor_uses_numeric_tolerance() -> None:
    rows = [
        {"score": SCORE_FLOOR},
        {"score": np.nextafter(SCORE_FLOOR, math.inf)},
        {"score": 0.5},
    ]
    summary = summarize_score_domain(rows, "score")
    assert summary["floor_fraction"] == 2 / 3
    assert summary["status"] == "score_domain_collapsed"


def test_scan_then_physical_scene_equal_weighting() -> None:
    rows = [
        *(
            {
                "physical_scene_id": "p0",
                "scene_id": "p0_scan0",
                "value": 0.0,
            }
            for _ in range(100)
        ),
        {"physical_scene_id": "p0", "scene_id": "p0_scan1", "value": 1.0},
        {"physical_scene_id": "p1", "scene_id": "p1_scan0", "value": 1.0},
    ]
    summary = scan_physical_equal_summary(rows, "value")
    assert summary["scan_count"] == 3
    assert summary["physical_scene_count"] == 2
    assert math.isclose(summary["per_physical_scene"]["p0"], 0.5)
    assert math.isclose(summary["mean"], 0.75)


def test_radius_sensitivity_reports_common_eligible_objects() -> None:
    gt_xyz = np.asarray(
        [
            [0.00, 0.00, 0.0],
            [0.01, 0.00, 0.0],
            [0.00, 0.01, 0.0],
            [0.01, 0.01, 0.0],
            [0.02, 0.00, 0.0],
            [0.02, 0.01, 0.0],
            [0.20, 0.00, 0.0],
            [0.21, 0.00, 0.0],
            [0.20, 0.01, 0.0],
            [0.21, 0.01, 0.0],
            [0.22, 0.00, 0.0],
            [0.22, 0.01, 0.0],
        ]
    )
    result = build_oracle_scene(
        scene_id="scene0001_00",
        physical_scene_id="scene0001",
        gaussian_xyz=gt_xyz,
        gaussian_to_gt_transform=np.eye(4),
        gt_xyz_m=gt_xyz,
        gt_semantic=np.asarray([0] * 6 + [1] * 6),
        gt_instance=np.asarray([1] * 6 + [2] * 6),
        class_names=("chair", "table"),
        priors=_priors(),
        radii_m=(0.02, 0.05, 0.10),
        min_region_size=3,
    )
    analysis = summarize_prior_oracle([result])
    assert set(analysis["radius_sensitivity"]) == {"0.02", "0.05", "0.10"}
    assert analysis["eligible_object_count"] == 2
    assert analysis["common_eligible_object_count"] == 2
    assert analysis["common_pair_count"] == 4
    assert set(analysis["common_subset_radius_sensitivity"]) == {
        "0.02",
        "0.05",
        "0.10",
    }
    assert set(analysis["per_class"]) == {"chair", "table"}
    assert set(analysis["per_scene"]) == {"scene0001_00"}
    assert "combined_support" in analysis["prior_effects"]
    assert "U_size_score" in analysis["floor_saturation"]
    assert "deployable AP" in analysis["conclusion_boundary"]
