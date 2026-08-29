from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.category_cluster_evaluation import (
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    ClusterEvaluationScene,
    ClusterSceneMetrics,
    analyze_cluster_metrics,
    evaluate_cluster_candidate_banks,
    evaluate_cluster_scene,
)


def _bank(
    full: list[int],
    core: list[int],
    classes: list[int],
    *,
    ids: list[int] | None = None,
) -> SimpleNamespace:
    candidate_ids = ids if ids is not None else list(range(len(classes)))
    return SimpleNamespace(
        class_names=("chair", "cup"),
        branch_full_labels=np.asarray(full, dtype=np.int64),
        branch_core_labels=np.asarray(core, dtype=np.int64),
        candidates=tuple(
            {"candidate_id": candidate_id, "branch_class_id": class_id}
            for candidate_id, class_id in zip(candidate_ids, classes)
        ),
        diagnostics={
            "determinism_measured_this_scene": True,
            "determinism_contract_verified": True,
            "determinism_violation_count": 0,
        },
    )


def _scene(scene_id: str = "scene") -> ClusterEvaluationScene:
    # Official object 0 (chair/tiny) owns GT points 0..2; object 1
    # (cup/large) owns GT points 3..5.  Gaussian 6 is unsupported.
    return ClusterEvaluationScene(
        scene_id=scene_id,
        gt_to_gaussian_indices=np.arange(6),
        gt_point_object_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        gt_object_class_ids=np.asarray([0, 1]),
        gt_object_size_bins=("tiny", "large"),
        gaussian_to_gt_object_indices=np.asarray([0, 0, 0, 1, 1, 1, -1]),
        class_name_to_id={"chair": 0, "cup": 1},
        gt_object_instance_ids=np.asarray([17, 42]),
    )


def _metrics(
    scene_id: str,
    *,
    candidate_count: int,
    iou025: int,
    iou050: int,
    unsupported: float,
    tiny_recall: float,
    best: tuple[float, ...],
    violations: int = 0,
) -> ClusterSceneMetrics:
    points = 100 * candidate_count
    return ClusterSceneMetrics(
        scene_id=scene_id,
        candidate_count=candidate_count,
        candidate_point_count=points,
        unsupported_point_count=int(round(points * unsupported)),
        same_class_iou_025_count=iou025,
        same_class_iou_050_count=iou050,
        tiny_small_gt_count=1,
        tiny_small_iou_025_count=int(tiny_recall >= 0.25),
        tiny_small_iou_050_count=int(tiny_recall >= 0.50),
        core_subset_full_violation_count=violations,
        best_iou_by_gt=best,
        determinism_measured_this_scene=True,
    )


def test_scene_metrics_include_matches_precision_unsupported_and_tiny_recall() -> None:
    bank = _bank(
        # Candidate 0 exactly covers the chair. Candidate 1 covers the cup and
        # one unsupported Gaussian, so its point precision is 3/4.
        [0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, -1, -1],
        [0, 1],
    )

    metrics = evaluate_cluster_scene(_scene(), bank)

    assert metrics.candidate_count == 2
    assert metrics.same_class_iou_025_count == 2
    assert metrics.same_class_iou_050_count == 2
    assert metrics.candidate_precision_025 == 1.0
    assert metrics.unsupported_point_count == 1
    assert metrics.unsupported_fraction == pytest.approx(1 / 7)
    assert metrics.tiny_small_recall_025 == 1.0
    assert metrics.tiny_small_recall_050 == 1.0
    assert metrics.best_iou_by_gt == pytest.approx((1.0, 1.0))
    assert metrics.candidate_rows[1]["gaussian_precision"] == pytest.approx(0.75)
    assert metrics.candidate_rows[0]["branch_class"] == "chair"
    assert metrics.candidate_rows[0]["best_same_class_instance_id"] == 17
    assert metrics.candidate_rows[0]["best_same_class_size_bin"] == "tiny"
    assert metrics.candidate_rows[1]["best_same_class_instance_id"] == 42
    assert metrics.candidate_rows[1]["best_same_class_size_bin"] == "large"


def test_arbitrary_candidate_ids_are_compacted_without_changing_geometry() -> None:
    bank = _bank(
        [11, 11, 11, 42, 42, 42, -1],
        [11, 11, 11, 42, 42, 42, -1],
        [0, 1],
        ids=[11, 42],
    )

    metrics = evaluate_cluster_scene(_scene(), bank)

    assert metrics.same_class_iou_050_count == 2
    assert [row["candidate_id"] for row in metrics.candidate_rows] == [11, 42]


def test_bank_aliases_and_class_name_lookup_are_supported() -> None:
    bank = SimpleNamespace(
        class_names=(),
        full_candidate_labels=np.asarray([7, 7, 7, -1, -1, -1, -1]),
        trusted_core_labels=np.asarray([7, 7, 7, -1, -1, -1, -1]),
        candidates=({"candidate_id": 7, "branch_class": "chair"},),
    )

    metrics = evaluate_cluster_scene(_scene(), bank)

    assert metrics.same_class_iou_050_count == 1


def test_branch_class_name_bridges_runtime32_and_canonical_gt_id_spaces() -> None:
    bank = SimpleNamespace(
        class_names=tuple(f"runtime-{index}" for index in range(32)),
        branch_full_labels=np.asarray([5, 5, 5, -1, -1, -1, -1]),
        branch_core_labels=np.asarray([5, 5, 5, -1, -1, -1, -1]),
        candidates=(
            {
                "candidate_id": 5,
                "branch_class": "chair",
                # Deliberately different from canonical chair id 0.
                "branch_class_index": 17,
            },
        ),
    )

    metrics = evaluate_cluster_scene(_scene(), bank)

    assert metrics.same_class_iou_050_count == 1
    assert metrics.candidate_rows[0]["branch_class_id"] == 0
    assert metrics.candidate_rows[0]["branch_class"] == "chair"


def test_unmatched_candidate_preserves_class_and_has_no_gt_identity() -> None:
    bank = _bank(
        [-1, -1, -1, -1, -1, -1, 9],
        [-1, -1, -1, -1, -1, -1, 9],
        [0],
        ids=[9],
    )

    metrics = evaluate_cluster_scene(_scene(), bank)

    row = metrics.candidate_rows[0]
    assert row["branch_class"] == "chair"
    # There is a same-class GT object but no overlap, so its identity remains
    # explicit for IoU=0 diagnostics rather than being conflated with unknown.
    assert row["best_same_class_instance_id"] == 17
    assert row["best_same_class_size_bin"] == "tiny"
    assert row["best_same_class_iou"] == 0.0


def test_scene_instance_ids_are_validated_and_default_to_object_indices() -> None:
    defaulted = ClusterEvaluationScene(
        scene_id="defaulted",
        gt_to_gaussian_indices=np.asarray([0]),
        gt_point_object_indices=np.asarray([0]),
        gt_object_class_ids=np.asarray([0]),
        gt_object_size_bins=("tiny",),
        gaussian_to_gt_object_indices=np.asarray([0]),
        class_name_to_id={"chair": 0},
    )
    assert defaulted.gt_object_instance_ids.tolist() == [0]

    with pytest.raises(ValueError, match="instance_ids and gt_object_class_ids"):
        ClusterEvaluationScene(
            scene_id="broken",
            gt_to_gaussian_indices=np.asarray([0]),
            gt_point_object_indices=np.asarray([0]),
            gt_object_class_ids=np.asarray([0]),
            gt_object_size_bins=("tiny",),
            gaussian_to_gt_object_indices=np.asarray([0]),
            class_name_to_id={"chair": 0},
            gt_object_instance_ids=np.asarray([10, 11]),
        )


def test_core_outside_full_is_reported_and_undeclared_labels_are_rejected() -> None:
    violating = _bank(
        [0, 0, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, -1, -1, -1],
        [0],
    )
    assert evaluate_cluster_scene(
        _scene(), violating
    ).core_subset_full_violation_count == 1

    undeclared = _bank(
        [0, 0, 99, -1, -1, -1, -1],
        [0, 0, -1, -1, -1, -1, -1],
        [0],
    )
    with pytest.raises(ValueError, match="undeclared candidate ids"):
        evaluate_cluster_scene(_scene(), undeclared)


def test_scene_level_raw_member_totals_override_per_class_diagnostics() -> None:
    bank = _bank(
        [0, 0, 0, -1, -1, -1, -1],
        [0, 0, 0, -1, -1, -1, -1],
        [0],
    )
    bank.diagnostics = {
        "raw_member_count": 10,
        "raw_member_retained_count": 10,
        # These deliberately stale blocks must not override the registered
        # scene-level totals produced after integration/remapping.
        "class_diagnostics": {
            "chair": {
                "raw_member_count": 7,
                "raw_member_retained_count": 3,
            }
        },
    }

    metrics = evaluate_cluster_scene(_scene(), bank)

    assert metrics.raw_member_count == 10
    assert metrics.raw_member_retained_count == 10
    assert metrics.raw_member_retention == 1.0


def test_dev2_gate_records_every_gt_drop_and_selects_simpler_r1_on_tie() -> None:
    r0 = tuple(
        _metrics(
            scene,
            candidate_count=20,
            iou025=2,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.60, 0.20),
        )
        for scene in ("a", "b")
    )
    repaired = tuple(
        _metrics(
            scene,
            candidate_count=16,
            iou025=3,
            iou050=2,
            unsupported=0.08,
            tiny_recall=0.5,
            best=(0.62, 0.25),
        )
        for scene in ("a", "b")
    )

    result = analyze_cluster_metrics(
        {
            R0_LEGACY: r0,
            R1_METRIC_HDBSCAN: repaired,
            R2_ANCHORED_HDBSCAN: tuple(reversed(repaired)),
        },
        phase="dev2",
    )

    assert result["gates"][R1_METRIC_HDBSCAN]["passed"] is True
    assert result["gates"][R2_ANCHORED_HDBSCAN]["passed"] is True
    assert result["selected_condition"] == R1_METRIC_HDBSCAN
    assert result["ranking"][:2] == [
        R1_METRIC_HDBSCAN,
        R2_ANCHORED_HDBSCAN,
    ]
    per_gt = result["comparisons_vs_r0"][R2_ANCHORED_HDBSCAN]["per_gt"]
    assert len(per_gt) == 4
    assert max(row["best_iou_drop_vs_r0"] for row in per_gt) <= 0.0


def test_dev2_allows_a_local_drop_inside_the_improved_witness_scene() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.80,),
        ),
        _metrics(
            "b",
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.50,),
        ),
    )
    repair = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.01,
            tiny_recall=0.5,
            best=(0.74,),
        ),
        _metrics(
            "b",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.01,
            tiny_recall=0.5,
            best=(0.70,),
        ),
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, R1_METRIC_HDBSCAN: repair}, phase="dev2"
    )

    gate = result["gates"][R1_METRIC_HDBSCAN]
    assert gate["checks"]["per_gt_drop_at_most_0.05"] is True
    assert gate["maximum_gt_best_iou_drop"] == pytest.approx(0.06)
    assert gate["witness_scene"] == "a"
    assert gate["per_scene_maximum_gt_best_iou_drop"] == pytest.approx(
        {"a": 0.06, "b": -0.20}
    )
    assert gate["nonwitness_maximum_gt_best_iou_drop"] == pytest.approx(-0.20)
    assert gate["passed"] is True
    assert result["selected_condition"] == R1_METRIC_HDBSCAN


def test_dev2_rejects_drop_over_005_in_the_nonwitness_scene() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.50,),
        ),
        _metrics(
            "b",
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.50,),
        ),
    )
    repair = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.01,
            tiny_recall=0.5,
            best=(0.70,),
        ),
        # Scene b is not a quality witness: its registered quality tuple is
        # unchanged, while one GT object's best IoU drops by 0.06.
        _metrics(
            "b",
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.44,),
        ),
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, R1_METRIC_HDBSCAN: repair}, phase="dev2"
    )

    gate = result["gates"][R1_METRIC_HDBSCAN]
    assert gate["improved_scene_ids"] == ["a"]
    assert gate["witness_scene"] is None
    assert gate["per_scene_maximum_gt_best_iou_drop"] == pytest.approx(
        {"a": -0.20, "b": 0.06}
    )
    assert gate["nonwitness_maximum_gt_best_iou_drop"] == pytest.approx(0.06)
    assert gate["checks"]["per_gt_drop_at_most_0.05"] is False
    assert gate["passed"] is False
    assert result["selected_condition"] is None


def test_dev2_searches_all_improved_scenes_for_a_safe_witness() -> None:
    baseline = tuple(
        _metrics(
            scene,
            candidate_count=20,
            iou025=1,
            iou050=1,
            unsupported=0.20,
            tiny_recall=0.5,
            best=(0.50,),
        )
        for scene in ("a", "b")
    )
    repair = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.01,
            tiny_recall=0.5,
            best=(0.46,),
        ),
        _metrics(
            "b",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.01,
            tiny_recall=0.5,
            best=(0.44,),
        ),
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, R1_METRIC_HDBSCAN: repair}, phase="dev2"
    )

    gate = result["gates"][R1_METRIC_HDBSCAN]
    assert gate["improved_scene_ids"] == ["a", "b"]
    # Witness a would expose b's unsafe 0.06 drop.  The gate must also try b,
    # whose only non-witness scene a remains within the 0.05 safety bound.
    assert gate["witness_scene"] == "b"
    assert gate["nonwitness_maximum_gt_best_iou_drop"] == pytest.approx(0.04)
    assert gate["checks"]["per_gt_drop_at_most_0.05"] is True
    assert gate["passed"] is True


def test_g1_is_only_selected_after_both_hdbscan_repairs_fail() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=20,
            iou025=1,
            iou050=0,
            unsupported=0.30,
            tiny_recall=0.0,
            best=(0.0,),
        ),
    )
    failed_primary = (
        _metrics(
            "a",
            candidate_count=30,
            iou025=1,
            iou050=0,
            unsupported=0.30,
            tiny_recall=0.0,
            best=(0.0,),
        ),
    )
    passing_graph = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=2,
            iou050=1,
            unsupported=0.05,
            tiny_recall=0.5,
            best=(0.6,),
        ),
    )

    result = analyze_cluster_metrics(
        {
            R0_LEGACY: baseline,
            R1_METRIC_HDBSCAN: failed_primary,
            R2_ANCHORED_HDBSCAN: failed_primary,
            G1_MUTUAL_LOCAL_GRAPH: passing_graph,
        },
        phase="dev2",
    )

    assert result["gates"][R1_METRIC_HDBSCAN]["passed"] is False
    assert result["gates"][R2_ANCHORED_HDBSCAN]["passed"] is False
    assert result["gates"][G1_MUTUAL_LOCAL_GRAPH]["passed"] is True
    assert result["selected_condition"] == G1_MUTUAL_LOCAL_GRAPH
    assert result["selection_tier"] == "registered_graph_fallback"


def test_passing_primary_remains_authoritative_over_graph_fallback() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=20,
            iou025=1,
            iou050=0,
            unsupported=0.30,
            tiny_recall=0.0,
            best=(0.0,),
        ),
    )
    primary = (
        _metrics(
            "a",
            candidate_count=15,
            iou025=2,
            iou050=1,
            unsupported=0.05,
            tiny_recall=0.5,
            best=(0.6,),
        ),
    )
    graph = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=4,
            iou050=3,
            unsupported=0.0,
            tiny_recall=1.0,
            best=(0.9,),
        ),
    )

    result = analyze_cluster_metrics(
        {
            R0_LEGACY: baseline,
            R1_METRIC_HDBSCAN: primary,
            G1_MUTUAL_LOCAL_GRAPH: graph,
        },
        phase="dev2",
    )

    assert result["selected_condition"] == R1_METRIC_HDBSCAN
    assert result["selection_tier"] == "primary_hdbscan_repair"
    # Ranking remains a transparent quality report even though staged
    # selection correctly keeps G1 as a fallback only.
    assert result["ranking"][0] == G1_MUTUAL_LOCAL_GRAPH


def test_dev8_health_gate_passes_only_with_scene_support_and_positive_direction() -> None:
    scene_ids = tuple(f"s{index}" for index in range(8))
    baseline = tuple(
        _metrics(
            scene,
            candidate_count=20,
            iou025=1,
            iou050=0,
            unsupported=0.30,
            tiny_recall=0.0,
            best=(0.0,),
        )
        for scene in scene_ids
    )
    repair = tuple(
        _metrics(
            scene,
            candidate_count=20,
            iou025=2,
            iou050=2 if index < 6 else 0,
            unsupported=0.10,
            tiny_recall=1.0 if index < 2 else 0.0,
            best=(0.7 if index < 6 else 0.0,),
        )
        for index, scene in enumerate(scene_ids)
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, R2_ANCHORED_HDBSCAN: repair},
        phase="dev8",
        selected_condition=R2_ANCHORED_HDBSCAN,
    )

    gate = result["selected_gate"]
    assert gate["passed"] is True
    assert gate["positive_scene_count"] == 8
    assert gate["checks"]["iou050_at_least_12"] is True
    assert gate["checks"]["iou050_at_least_4_scenes"] is True
    assert gate["checks"]["tiny_small_recall025_at_least_0.20"] is True


def test_dev8_core_contract_violation_blocks_health_gate() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=0,
            iou050=0,
            unsupported=0.2,
            tiny_recall=0.0,
            best=(0.0,),
        ),
    )
    repair = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=10,
            iou050=12,
            unsupported=0.0,
            tiny_recall=1.0,
            best=(1.0,),
            violations=1,
        ),
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, G1_MUTUAL_LOCAL_GRAPH: repair}, phase="dev8"
    )

    assert result["selected_gate"]["checks"]["core_subset_full"] is False
    assert result["selected_gate"]["passed"] is False


def test_raw_member_loss_and_prediction_contract_counts_block_gates() -> None:
    baseline = (
        _metrics(
            "a",
            candidate_count=10,
            iou025=0,
            iou050=0,
            unsupported=0.2,
            tiny_recall=0.0,
            best=(0.0,),
        ),
    )
    row = _metrics(
        "a",
        candidate_count=5,
        iou025=2,
        iou050=1,
        unsupported=0.0,
        tiny_recall=1.0,
        best=(1.0,),
    )
    broken = ClusterSceneMetrics(
        **{
            **row.__dict__,
            "raw_member_count": 10,
            "raw_member_retained_count": 9,
            "orphan_count": 1,
            "negative_metadata_count": 1,
            "determinism_violation_count": 1,
        }
    )

    result = analyze_cluster_metrics(
        {R0_LEGACY: baseline, R2_ANCHORED_HDBSCAN: (broken,)}, phase="dev2"
    )

    checks = result["gates"][R2_ANCHORED_HDBSCAN]["checks"]
    assert checks["raw_members_retained_100pct"] is False
    assert checks["orphan_count_zero"] is False
    assert checks["negative_metadata_count_zero"] is False
    assert checks["deterministic"] is False
    assert result["selected_condition"] is None


def test_dev2_requires_direct_measurement_while_dev8_accepts_verified_reference() -> None:
    baseline = _metrics(
        "a",
        candidate_count=10,
        iou025=0,
        iou050=0,
        unsupported=0.2,
        tiny_recall=0.0,
        best=(0.0,),
    )
    healthy = _metrics(
        "a",
        candidate_count=10,
        iou025=10,
        iou050=12,
        unsupported=0.0,
        tiny_recall=1.0,
        best=(1.0,),
    )
    unmeasured = ClusterSceneMetrics(
        **{
            **healthy.__dict__,
            "determinism_measured_this_scene": False,
            "determinism_algorithm_contract_reference": False,
        }
    )
    dev2 = analyze_cluster_metrics(
        {R0_LEGACY: (baseline,), R1_METRIC_HDBSCAN: (unmeasured,)},
        phase="dev2",
    )
    assert dev2["gates"][R1_METRIC_HDBSCAN]["checks"]["deterministic"] is False

    referenced_baseline = ClusterSceneMetrics(
        **{
            **baseline.__dict__,
            "determinism_measured_this_scene": False,
            "determinism_algorithm_contract_reference": True,
        }
    )
    referenced_healthy = ClusterSceneMetrics(
        **{
            **healthy.__dict__,
            "determinism_measured_this_scene": False,
            "determinism_algorithm_contract_reference": True,
        }
    )
    dev8 = analyze_cluster_metrics(
        {
            R0_LEGACY: (referenced_baseline,),
            R1_METRIC_HDBSCAN: (referenced_healthy,),
        },
        phase="dev8",
        selected_condition=R1_METRIC_HDBSCAN,
    )
    assert dev8["selected_gate"]["checks"]["deterministic"] is True


def test_end_to_end_bank_evaluator_is_deterministic_under_mapping_order() -> None:
    scene_a = _scene("a")
    scene_b = _scene("b")
    r0 = _bank([0, 0, -1, 1, -1, -1, -1], [0, 0, -1, 1, -1, -1, -1], [0, 1])
    repaired = _bank(
        [0, 0, 0, 1, 1, 1, -1],
        [0, 0, 0, 1, 1, 1, -1],
        [0, 1],
    )

    forward = evaluate_cluster_candidate_banks(
        {"a": scene_a, "b": scene_b},
        {
            R0_LEGACY: {"a": r0, "b": r0},
            R1_METRIC_HDBSCAN: {"a": repaired, "b": repaired},
        },
        phase="dev2",
    )
    reverse = evaluate_cluster_candidate_banks(
        {"b": scene_b, "a": scene_a},
        {
            R1_METRIC_HDBSCAN: {"b": repaired, "a": repaired},
            R0_LEGACY: {"b": r0, "a": r0},
        },
        phase="dev2",
    )

    assert forward["scene_ids"] == reverse["scene_ids"] == ["a", "b"]
    assert forward["selected_condition"] == reverse["selected_condition"]
    assert (
        forward["conditions"][R1_METRIC_HDBSCAN]["same_class_iou_050_count"]
        == reverse["conditions"][R1_METRIC_HDBSCAN]["same_class_iou_050_count"]
    )


def test_scene_projection_validation_rejects_invalid_indices() -> None:
    with pytest.raises(ValueError, match="invalid Gaussian id"):
        ClusterEvaluationScene(
            scene_id="broken",
            gt_to_gaussian_indices=np.asarray([99]),
            gt_point_object_indices=np.asarray([0]),
            gt_object_class_ids=np.asarray([0]),
            gt_object_size_bins=("tiny",),
            gaussian_to_gt_object_indices=np.asarray([0]),
        )


def test_condition_scene_sets_must_be_identical() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_cluster_candidate_banks(
            {"a": _scene("a")},
            {
                R0_LEGACY: {"a": _bank([-1] * 7, [-1] * 7, [])},
                R1_METRIC_HDBSCAN: {},
            },
            phase="dev2",
        )
