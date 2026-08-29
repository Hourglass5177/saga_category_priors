from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from category_priors.category_denoise import CandidateBank
from category_priors.category_fragment_merge import (
    FragmentEdge,
    FragmentGraph,
    FragmentNode,
    build_fragment_graph,
    merge_category_fragments,
    merge_same_graph_both_modes,
)


def _prior_node(
    *,
    long_platform: tuple[float, float, float] = (1e-9, 1.0, 1e9),
    area: float = 1.0,
) -> dict[str, object]:
    geometry: dict[str, object] = {
        "log_surface_area_m2": {"q50": math.log(area)},
    }
    platforms = (
        (1e-11, 1e-9, 1e-7),
        (1e-11, 1e-9, 1e-7),
        long_platform,
    )
    for field, values in zip(
        (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        ),
        platforms,
    ):
        geometry[field] = {
            "q25": math.log(values[0]),
            "q50": math.log(values[1]),
            "q75": math.log(values[2]),
        }
    return {"shrunk": {"geometry": geometry}}


def _priors(
    *,
    global_area: float = 1.0,
    class_area: float = 1.0,
    global_long: tuple[float, float, float] = (1e-9, 1.0, 1e9),
    class_long: tuple[float, float, float] = (1e-9, 1.0, 1e9),
) -> dict[str, object]:
    return {
        "global": _prior_node(long_platform=global_long, area=global_area),
        "categories": {
            "chair": _prior_node(long_platform=class_long, area=class_area)
        },
    }


def _node(fragment_id: int, points: list[int]) -> FragmentNode:
    return FragmentNode(
        fragment_id=fragment_id,
        source_fragment_id=fragment_id + 100,
        point_ids=np.asarray(points),
        class_index=0,
        class_name="chair",
        membership_mean=0.81,
        semantic_score_mean=0.90,
    )


def _graph(
    nodes: tuple[FragmentNode, ...], edges: tuple[FragmentEdge, ...], point_count: int
) -> FragmentGraph:
    return FragmentGraph(
        nodes=nodes,
        edges=edges,
        point_count=point_count,
        scene_scale_m_per_unit=1.0,
        global_typical_diag_m=2.0,
    )


def _raw_bank(
    labels: np.ndarray,
    class_indices: np.ndarray | None = None,
) -> CandidateBank:
    labels = np.asarray(labels, dtype=np.int64)
    count = len(labels)
    classes = tuple(["chair", "table"] + [f"class-{index}" for index in range(30)])
    semantic = (
        np.zeros(count, dtype=np.int64)
        if class_indices is None
        else np.asarray(class_indices, dtype=np.int64)
    )
    rows = []
    for candidate_id in sorted(int(value) for value in np.unique(labels) if value >= 0):
        points = np.flatnonzero(labels == candidate_id)
        class_index = int(np.unique(semantic[points]).item())
        rows.append(
            {
                "candidate_id": candidate_id,
                "stable_source_id": candidate_id + 100,
                "branch_class_index": class_index,
                "branch_class": classes[class_index],
            }
        )
    return CandidateBank(
        class_names=classes,
        saga20_names=("chair", "table"),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.full(count, -1, dtype=np.int64),
        semantic_top1=semantic,
        semantic_top1_score=np.full(count, 0.9),
        branch_full_labels=labels.copy(),
        branch_core_labels=labels.copy(),
        assignment_confidence=np.full(count, 0.8),
        candidates=tuple(rows),
        diagnostics={"raw_bank": True},
    )


def test_fragment_graph_builds_same_class_mutual_top4_edge() -> None:
    bank = _raw_bank(np.asarray([0, 0, 0, 1, 1, 1]))
    xyz = np.column_stack((np.arange(6) * 0.01, np.zeros(6), np.zeros(6)))
    affinity = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    graph = build_fragment_graph(bank, xyz, affinity, 1.0)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0].cross_edge_count >= 3
    assert graph.edges[0].min_distance_m <= 0.1
    assert graph.diagnostics["category_specific_prior_used"] is False


def test_fragment_graph_never_connects_different_predicted_classes() -> None:
    bank = _raw_bank(
        np.asarray([0, 0, 0, 1, 1, 1]),
        np.asarray([0, 0, 0, 1, 1, 1]),
    )
    xyz = np.column_stack((np.arange(6) * 0.01, np.zeros(6), np.zeros(6)))
    affinity = np.ones((6, 2), dtype=np.float64)

    graph = build_fragment_graph(bank, xyz, affinity, 1.0)

    assert not graph.edges


def test_fragment_graph_rejects_cross_edges_outside_global_radius() -> None:
    bank = _raw_bank(np.asarray([0, 0, 0, 1, 1, 1]))
    xyz = np.asarray(
        [[0.00, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0],
         [1.00, 0.0, 0.0], [1.01, 0.0, 0.0], [1.02, 0.0, 0.0]]
    )
    affinity = np.ones((6, 2), dtype=np.float64)

    graph = build_fragment_graph(bank, xyz, affinity, 1.0)

    assert not graph.edges


def test_support_prior_recovers_two_fragments_then_stops_chain() -> None:
    nodes = (_node(0, [0, 1, 2]), _node(1, [3, 4, 5]), _node(2, [6, 7, 8]))
    graph = _graph(
        nodes,
        (
            FragmentEdge(0, 1, 6, 0.99, 0.01),
            FragmentEdge(1, 2, 5, 0.98, 0.01),
        ),
        9,
    )
    xyz = np.column_stack((np.arange(9) * 0.01, np.zeros(9), np.zeros(9)))

    result = merge_category_fragments(graph, xyz, _priors(), "global")

    assert [row.source_fragment_ids for row in result.objects] == [(100, 101), (102,)]
    assert result.objects[0].accepted is True
    assert result.objects[1].accepted is False
    assert np.all(result.point_labels[:6] == 0)
    assert np.all(result.point_labels[6:] == -1)
    assert sum(row.accepted for row in result.decisions) == 1
    assert result.decisions[-1].accepted is False
    assert result.diagnostics["orphan_count"] == 0


def test_global_and_class_replay_share_graph_but_can_change_formation() -> None:
    graph = _graph(
        (_node(0, [0, 1, 2]), _node(1, [3, 4, 5])),
        (FragmentEdge(0, 1, 6, 0.99, 0.01),),
        6,
    )
    xyz = np.column_stack((np.arange(6) * 0.01, np.zeros(6), np.zeros(6)))
    priors = _priors(global_area=1.0, class_area=0.36)

    uniform, class_shrunk = merge_same_graph_both_modes(graph, xyz, priors)

    assert uniform.graph is graph and class_shrunk.graph is graph
    assert graph.identity() == uniform.graph.identity() == class_shrunk.graph.identity()
    assert [row.source_fragment_ids for row in uniform.objects] == [(100, 101)]
    assert [row.source_fragment_ids for row in class_shrunk.objects] == [(100,), (101,)]
    assert uniform.decisions[0].accepted is True
    assert class_shrunk.decisions[0].accepted is False


def test_two_sided_size_prior_recovers_100_point_objects_and_stops_wrong_union() -> None:
    first_half = np.linspace(0.0, 0.4, 25)
    second_half = np.linspace(0.6, 1.0, 25)
    third_half = np.linspace(2.0, 2.4, 25)
    fourth_half = np.linspace(2.6, 3.0, 25)
    graph = _graph(
        (
            _node(0, list(range(25))),
            _node(1, list(range(25, 50))),
            _node(2, list(range(50, 75))),
            _node(3, list(range(75, 100))),
        ),
        (
            FragmentEdge(0, 1, 8, 0.99, 0.01),
            FragmentEdge(2, 3, 8, 0.99, 0.01),
            FragmentEdge(1, 2, 3, 0.95, 0.05),
        ),
        100,
    )
    xyz = np.column_stack(
        (
            np.concatenate((first_half, second_half, third_half, fourth_half)),
            np.zeros(100),
            np.zeros(100),
        )
    )
    priors = _priors(
        global_long=(1e-9, 1.0, 1e9),
        class_long=(0.8, 1.0, 1.2),
    )

    uniform, class_shrunk = merge_same_graph_both_modes(graph, xyz, priors)

    assert len(uniform.objects) == 4
    assert [row.source_fragment_ids for row in class_shrunk.objects] == [
        (100, 101),
        (102, 103),
    ]
    assert all(len(row.point_ids) == 50 for row in class_shrunk.objects)
    assert class_shrunk.decisions[0].union_prior_score > max(
        class_shrunk.decisions[0].left_prior_score,
        class_shrunk.decisions[0].right_prior_score,
    )
    assert class_shrunk.decisions[-1].accepted is False


def test_prior_filters_bad_strong_edge_before_mutual_best_selection() -> None:
    graph = _graph(
        (
            _node(0, [0, 1, 2]),
            _node(1, [3, 4, 5]),
            _node(2, [6, 7, 8]),
        ),
        (
            # Public evidence favours 0--1, but that union is implausibly long.
            FragmentEdge(0, 1, 10, 0.99, 0.05),
            FragmentEdge(0, 2, 8, 0.98, 0.05),
        ),
        9,
    )
    xyz = np.column_stack(
        (
            np.asarray([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 0.81, 0.90, 1.0]),
            np.zeros(9),
            np.zeros(9),
        )
    )
    priors = _priors(
        global_long=(0.8, 1.0, 1.2),
        class_long=(0.8, 1.0, 1.2),
    )

    result = merge_category_fragments(graph, xyz, priors, "global")

    first_round = [row for row in result.decisions if row.round_index == 0]
    strong = next(row for row in first_round if row.union_source_fragment_ids == (100, 101))
    useful = next(row for row in first_round if row.union_source_fragment_ids == (100, 102))
    assert strong.prior_eligible is False
    assert strong.accepted is False
    assert useful.prior_eligible is True
    assert useful.mutual_best is True
    assert useful.accepted is True
    assert (100, 102) in [row.source_fragment_ids for row in result.objects]


def test_exact_public_evidence_tie_does_not_choose_a_best_neighbor() -> None:
    graph = _graph(
        (
            _node(0, [0, 1, 2]),
            _node(1, [3, 4, 5]),
            _node(2, [6, 7, 8]),
        ),
        (
            FragmentEdge(0, 1, 6, 0.99, 0.01),
            FragmentEdge(0, 2, 6, 0.99, 0.01),
        ),
        9,
    )
    xyz = np.column_stack(
        (
            np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, -0.03, -0.02, -0.01]),
            np.zeros(9),
            np.zeros(9),
        )
    )

    result = merge_category_fragments(graph, xyz, _priors(), "global")

    assert all(row.prior_eligible for row in result.decisions)
    assert not any(row.mutual_best for row in result.decisions)
    assert not any(row.accepted for row in result.decisions)
    assert [row.source_fragment_ids for row in result.objects] == [
        (100,),
        (101,),
        (102,),
    ]


def test_missing_class_statistics_fall_back_to_global_exactly() -> None:
    graph = _graph(
        (_node(0, [0, 1, 2]), _node(1, [3, 4, 5])),
        (FragmentEdge(0, 1, 6, 0.99, 0.01),),
        6,
    )
    xyz = np.column_stack((np.arange(6) * 0.01, np.zeros(6), np.zeros(6)))
    priors = {"global": _prior_node(), "categories": {}}

    uniform, class_shrunk = merge_same_graph_both_modes(graph, xyz, priors)

    assert np.array_equal(uniform.point_labels, class_shrunk.point_labels)
    assert uniform.objects[0].P == class_shrunk.objects[0].P
    assert uniform.decisions == class_shrunk.decisions


def test_lineage_and_partition_do_not_depend_on_node_or_edge_input_order() -> None:
    nodes = (_node(0, [0, 1, 2]), _node(1, [3, 4, 5]), _node(2, [6, 7, 8]))
    edges = (
        FragmentEdge(0, 1, 6, 0.99, 0.01),
        FragmentEdge(1, 2, 5, 0.98, 0.01),
    )
    first_graph = _graph(nodes, edges, 9)
    second_graph = _graph(tuple(reversed(nodes)), tuple(reversed(edges)), 9)
    xyz = np.column_stack((np.arange(9) * 0.01, np.zeros(9), np.zeros(9)))

    first = merge_category_fragments(first_graph, xyz, _priors(), "global")
    second = merge_category_fragments(second_graph, xyz, _priors(), "global")

    assert first_graph.identity() == second_graph.identity()
    assert np.array_equal(first.point_labels, second.point_labels)
    assert [row.source_fragment_ids for row in first.objects] == [
        row.source_fragment_ids for row in second.objects
    ]
    assert first.decisions == second.decisions


def test_raw_fragment_id_renumbering_preserves_identity_partition_and_lineage() -> None:
    point_ids = {
        100: [0, 1, 2],
        101: [3, 4, 5],
        102: [6, 7, 8],
    }

    def graph_for(raw_by_source: dict[int, int]) -> FragmentGraph:
        nodes = tuple(
            FragmentNode(
                fragment_id=raw_by_source[source_id],
                source_fragment_id=source_id,
                point_ids=np.asarray(point_ids[source_id]),
                class_index=0,
                class_name="chair",
                membership_mean=0.81,
                semantic_score_mean=0.90,
            )
            for source_id in (100, 101, 102)
        )
        edges = (
            FragmentEdge(raw_by_source[100], raw_by_source[101], 7, 0.99, 0.01),
            FragmentEdge(raw_by_source[101], raw_by_source[102], 6, 0.98, 0.01),
        )
        return _graph(nodes, edges, 9)

    first_graph = graph_for({100: 0, 101: 1, 102: 2})
    second_graph = graph_for({100: 2, 101: 0, 102: 1})
    xyz = np.column_stack((np.arange(9) * 0.01, np.zeros(9), np.zeros(9)))

    first = merge_category_fragments(first_graph, xyz, _priors(), "global")
    second = merge_category_fragments(second_graph, xyz, _priors(), "global")

    assert first_graph.identity() == second_graph.identity()
    assert np.array_equal(first.point_labels, second.point_labels)
    assert [row.source_fragment_ids for row in first.objects] == [
        row.source_fragment_ids for row in second.objects
    ]
    assert first.decisions == second.decisions


def test_worker_signatures_cannot_receive_gt() -> None:
    for function in (build_fragment_graph, merge_category_fragments):
        parameters = {name.lower() for name in inspect.signature(function).parameters}
        assert not any("gt" in name or "iou" in name for name in parameters)


def test_registered_graph_hyperparameters_cannot_be_silently_changed() -> None:
    bank = _raw_bank(np.asarray([0, 0, 0]))
    xyz = np.zeros((3, 3), dtype=np.float64)
    affinity = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="physical-24/top-4"):
        build_fragment_graph(bank, xyz, affinity, 1.0, physical_k=8)
