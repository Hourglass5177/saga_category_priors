from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.category_cluster_evaluation import ClusterEvaluationScene
from category_priors.category_fragment_merge_evaluation import (
    DEV2_SCENE_IDS,
    DEV8_SCENE_IDS,
    FragmentGraphOracleMetrics,
    FragmentMergeSceneMetrics,
    analyze_fragment_merge_dev2,
    analyze_fragment_merge_dev8,
    evaluate_fragment_graph_oracle,
    evaluate_fragment_merge_mechanical_effect,
    evaluate_fragment_merge_scene,
)


def _scene(scene_id: str = "scene0645_00") -> ClusterEvaluationScene:
    return ClusterEvaluationScene(
        scene_id=scene_id,
        gt_to_gaussian_indices=np.asarray([0, 1, 2, 3, 4, 5]),
        gt_point_object_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        gt_object_class_ids=np.asarray([0, 1]),
        gt_object_size_bins=("tiny", "small"),
        gaussian_to_gt_object_indices=np.asarray([0, 0, 0, 1, 1, 1, -1, -1]),
        class_name_to_id={"chair": 0, "table": 1},
        gt_object_instance_ids=np.asarray([17, 23]),
    )


def _node(fragment_id: int, points: list[int], class_name: str) -> dict:
    return {
        "fragment_id": fragment_id,
        "source_fragment_id": fragment_id,
        "point_ids": np.asarray(points),
        "class_name": class_name,
    }


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        point_count=8,
        nodes=(
            _node(0, [0, 1], "chair"),
            _node(1, [2, 6], "chair"),
            _node(2, [3, 4, 5], "table"),
        ),
        edges=(
            {
                "left_fragment_id": 0,
                "right_fragment_id": 1,
                "cross_edge_count": 3,
            },
        ),
    )


def _decision(score: float) -> dict:
    return {
        "round_index": 0,
        "left_source_fragment_ids": [0],
        "right_source_fragment_ids": [1],
        "union_prior_score": score,
        "accepted": True,
    }


def _result(mode: str, merge_chair: bool, score: float = 0.5) -> SimpleNamespace:
    chair = (
        [
            {
                "source_fragment_ids": [0, 1],
                "point_ids": np.asarray([0, 1, 2, 6]),
                "class_name": "chair",
                "base_score": 0.8,
                "accepted": True,
            }
        ]
        if merge_chair
        else [
            {
                "source_fragment_ids": [0],
                "point_ids": np.asarray([0, 1]),
                "class_name": "chair",
                "base_score": 0.8,
                "accepted": True,
            },
            {
                "source_fragment_ids": [1],
                "point_ids": np.asarray([2, 6]),
                "class_name": "chair",
                "base_score": 0.7,
                "accepted": True,
            },
        ]
    )
    return SimpleNamespace(
        mode=mode,
        objects=tuple(
            chair
            + [
                {
                    "source_fragment_ids": [2],
                    "point_ids": np.asarray([3, 4, 5]),
                    "class_name": "table",
                    "base_score": 0.9,
                    "accepted": True,
                }
            ]
        ),
        decisions=(_decision(score),),
        diagnostics={
            "orphan_count": 0,
            "negative_metadata_count": 0,
            "core_full_contract_violation_count": 0,
            "determinism_violation_count": 0,
        },
    )


def test_graph_oracle_uses_only_same_gt_paths_and_counts_unmapped_gaussian_as_fp() -> (
    None
):
    observed = evaluate_fragment_graph_oracle(_scene(), _graph())

    assert observed.same_gt_edge_count == 1
    assert observed.different_gt_edge_count == 0
    assert observed.same_class_iou_050_count == 2
    assert observed.best_iou_by_gt == pytest.approx((0.75, 1.0))
    assert observed.tiny_small_recall_025 == 1.0
    assert observed.candidate_rows[0]["source_fragment_ids"] == [0, 1]


def test_graph_oracle_selects_best_of_all_disconnected_same_gt_components() -> None:
    scene = ClusterEvaluationScene(
        scene_id="disconnected",
        gt_to_gaussian_indices=np.asarray([0, 1, 2]),
        gt_point_object_indices=np.asarray([0, 0, 0]),
        gt_object_class_ids=np.asarray([0]),
        gt_object_size_bins=("tiny",),
        gaussian_to_gt_object_indices=np.asarray([0, 0, 0] + [-1] * 9),
        class_name_to_id={"chair": 0},
    )
    graph = SimpleNamespace(
        point_count=12,
        nodes=(
            _node(0, [0, 1, *range(3, 12)], "chair"),
            _node(1, [2], "chair"),
        ),
        edges=(),
    )

    observed = evaluate_fragment_graph_oracle(scene, graph)

    # Fragment 0 owns more correct Gaussians but nine unsupported FP.  The
    # disconnected singleton is the higher-IoU graph component and therefore
    # the proper evaluation-only upper-bound witness.
    assert observed.best_iou_by_gt == pytest.approx((1.0 / 3.0,))
    assert observed.candidate_rows[0]["source_fragment_ids"] == [1]
    assert observed.candidate_rows[0]["eligible_component_count"] == 2


def test_merge_scene_evaluates_lineage_and_unmapped_fp_without_mutating_result() -> (
    None
):
    result = _result("class", merge_chair=True)
    before = result.objects[0]["point_ids"].copy()

    observed = evaluate_fragment_merge_scene(_scene(), _graph(), result)

    assert observed.same_class_iou_050_count == 2
    assert observed.candidate_precision_025 == 1.0
    assert observed.candidate_rows[0]["best_same_class_iou"] == pytest.approx(0.75)
    assert observed.unsupported_point_count == 1
    assert observed.output_contract_violation_count == 0
    np.testing.assert_array_equal(result.objects[0]["point_ids"], before)


def test_lineage_mask_mismatch_is_reported_as_contract_failure() -> None:
    broken = _result("global", merge_chair=True)
    broken.objects[0]["point_ids"] = np.asarray([0, 1, 2])

    observed = evaluate_fragment_merge_scene(_scene(), _graph(), broken)

    assert observed.lineage_violation_count == 1
    assert observed.output_contract_violation_count == 1


def test_object_lineage_uses_stable_source_ids_not_compact_graph_ids() -> None:
    graph = _graph()
    for index, node in enumerate(graph.nodes):
        node["source_fragment_id"] = 100 + index
    result = _result("class", merge_chair=True)
    result.objects[0]["source_fragment_ids"] = [100, 101]
    result.objects[1]["source_fragment_ids"] = [102]
    result.decisions = (
        {
            **_decision(0.5),
            "left_source_fragment_ids": [100],
            "right_source_fragment_ids": [101],
        },
    )

    observed = evaluate_fragment_merge_scene(_scene(), graph, result)

    assert observed.lineage_violation_count == 0
    assert observed.same_class_iou_050_count == 2


def test_mechanical_gate_uses_common_first_round_scores_and_stable_lineage() -> None:
    graphs = {"a": _graph(), "b": _graph()}
    uniform = {
        "a": _result("global", merge_chair=False, score=0.40),
        "b": _result("global", merge_chair=False, score=0.40),
    }
    class_results = {
        "a": _result("class", merge_chair=True, score=0.42),
        "b": _result("class", merge_chair=True, score=0.42),
    }

    effect = evaluate_fragment_merge_mechanical_effect(graphs, uniform, class_results)

    assert effect["score_changed_first_round_proposal_fraction"] == 1.0
    assert effect["score_change_gate_passed"] is True
    assert effect["final_fragment_decision_changed_count"] == 4
    assert effect["mechanically_effective"] is True


def test_mechanical_gate_excludes_non_mutual_diagnostic_edges() -> None:
    graph = _graph()
    uniform = _result("global", merge_chair=False, score=0.40)
    class_result = _result("class", merge_chair=False, score=0.40)
    nonproposal_u = {**_decision(0.10), "mutual_best": False}
    nonproposal_d = {**_decision(0.90), "mutual_best": False}
    uniform.decisions = (*uniform.decisions, nonproposal_u)
    class_result.decisions = (*class_result.decisions, nonproposal_d)

    effect = evaluate_fragment_merge_mechanical_effect(
        {"a": graph}, {"a": uniform}, {"a": class_result}
    )

    assert effect["common_first_round_proposal_count"] == 1
    assert effect["score_changed_first_round_proposal_count"] == 0
    assert effect["mechanically_effective"] is False


def _oracle(
    scene_id: str, *, iou050: int = 3, tiny_hit: int = 1
) -> FragmentGraphOracleMetrics:
    return FragmentGraphOracleMetrics(
        scene_id=scene_id,
        graph_node_count=20,
        graph_edge_count=10,
        same_gt_edge_count=8,
        different_gt_edge_count=1,
        unknown_gt_edge_count=1,
        same_class_iou_025_count=iou050,
        same_class_iou_050_count=iou050,
        tiny_small_gt_count=5,
        tiny_small_iou_025_count=tiny_hit,
        best_iou_by_gt=tuple([0.6] * iou050 + [0.0] * (5 - iou050)),
        candidate_rows=(),
    )


def _metrics(
    scene_id: str,
    mode: str,
    *,
    candidate_count: int,
    iou025: int,
    iou050: int,
    unsupported_fraction: float,
    tiny_hit: int,
    best: tuple[float, ...],
    candidate_rows: tuple[dict, ...] = (),
    violations: int = 0,
) -> FragmentMergeSceneMetrics:
    points = candidate_count * 10
    return FragmentMergeSceneMetrics(
        scene_id=scene_id,
        mode=mode,
        candidate_count=candidate_count,
        candidate_point_count=points,
        unsupported_point_count=int(points * unsupported_fraction),
        unsupported_candidate_count=0,
        same_class_iou_025_count=iou025,
        same_class_iou_050_count=iou050,
        tiny_small_gt_count=5,
        tiny_small_iou_025_count=tiny_hit,
        tiny_small_iou_050_count=tiny_hit,
        best_iou_by_gt=best,
        lineage_violation_count=violations,
        overlap_ownership_violation_count=0,
        orphan_count=0,
        negative_metadata_count=0,
        core_full_contract_violation_count=0,
        determinism_violation_count=0,
        candidate_rows=candidate_rows,
    )


def test_dev2_gate_requires_graph_capacity_then_paired_class_benefit() -> None:
    oracle = tuple(_oracle(scene) for scene in DEV2_SCENE_IDS)
    uniform = tuple(
        _metrics(
            scene,
            "global",
            candidate_count=20,
            iou025=4,
            iou050=2,
            unsupported_fraction=0.20,
            tiny_hit=1,
            best=(0.5, 0.2),
        )
        for scene in DEV2_SCENE_IDS
    )
    class_rows = tuple(
        _metrics(
            scene,
            "class",
            candidate_count=16,
            iou025=5,
            iou050=2,
            unsupported_fraction=0.05,
            tiny_hit=1,
            best=(0.6, 0.2),
        )
        for scene in DEV2_SCENE_IDS
    )

    result = analyze_fragment_merge_dev2(
        oracle_rows=oracle,
        uniform_rows=uniform,
        class_rows=class_rows,
        mechanical_effect={"mechanically_effective": True},
    )

    assert result["graph_passed"] is True
    assert result["safe_witness_scene"] in DEV2_SCENE_IDS
    assert result["passed"] is True
    assert result["conclusion"] == "dev2-passed-proceed-to-dev8"

    failed_oracle = tuple(
        _oracle(scene, iou050=2, tiny_hit=0) for scene in DEV2_SCENE_IDS
    )
    failed = analyze_fragment_merge_dev2(
        oracle_rows=failed_oracle,
        uniform_rows=uniform,
        class_rows=class_rows,
        mechanical_effect={"mechanically_effective": True},
    )
    assert failed["passed"] is False
    assert (
        failed["conclusion"] == "graph-upper-bound-failed-category-prior-not-evaluable"
    )


def _ap_rows(improved: bool) -> tuple[dict, ...]:
    if improved:
        values = ((0.9, 0.6), (0.8, 0.6), (0.7, 0.0))
    else:
        values = ((0.9, 0.0), (0.8, 0.6), (0.7, 0.6))
    return tuple(
        {"base_score": score, "best_same_class_iou": iou} for score, iou in values
    )


def test_dev8_uses_scene_equal_ap_and_requires_five_positive_scenes() -> None:
    oracle = tuple(_oracle(scene, iou050=3, tiny_hit=1) for scene in DEV8_SCENE_IDS)
    uniform = tuple(
        _metrics(
            scene,
            "global",
            candidate_count=3,
            iou025=2,
            iou050=2,
            unsupported_fraction=0.0,
            tiny_hit=1,
            best=(0.6,),
            candidate_rows=_ap_rows(False),
        )
        for scene in DEV8_SCENE_IDS
    )
    class_rows = tuple(
        _metrics(
            scene,
            "class",
            candidate_count=3,
            iou025=2,
            iou050=2,
            unsupported_fraction=0.0,
            tiny_hit=1,
            best=(0.6,),
            candidate_rows=_ap_rows(index < 5),
        )
        for index, scene in enumerate(DEV8_SCENE_IDS)
    )

    result = analyze_fragment_merge_dev8(
        oracle_rows=oracle,
        uniform_rows=uniform,
        class_rows=class_rows,
        mechanical_effect={"mechanically_effective": True},
    )

    assert result["positive_scene_count"] == 5
    assert result["scene_equal_candidate_ap_025_delta"] > 0.002
    assert result["absolute_health_checks"]["class_iou050_at_least_12"] is True
    assert result["passed"] is True


def test_dev8_contract_violation_blocks_absolute_health() -> None:
    oracle = tuple(_oracle(scene, iou050=3, tiny_hit=1) for scene in DEV8_SCENE_IDS)
    uniform = tuple(
        _metrics(
            scene,
            "global",
            candidate_count=3,
            iou025=2,
            iou050=2,
            unsupported_fraction=0.0,
            tiny_hit=1,
            best=(0.6,),
            candidate_rows=_ap_rows(False),
        )
        for scene in DEV8_SCENE_IDS
    )
    class_rows = tuple(
        _metrics(
            scene,
            "class",
            candidate_count=3,
            iou025=2,
            iou050=2,
            unsupported_fraction=0.0,
            tiny_hit=1,
            best=(0.6,),
            candidate_rows=_ap_rows(True),
            violations=1 if index == 0 else 0,
        )
        for index, scene in enumerate(DEV8_SCENE_IDS)
    )

    result = analyze_fragment_merge_dev8(
        oracle_rows=oracle,
        uniform_rows=uniform,
        class_rows=class_rows,
        mechanical_effect={"mechanically_effective": True},
    )

    assert result["absolute_health_checks"]["output_contract_zero"] is False
    assert result["passed"] is False
