from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from category_priors.category_denoise import CandidateBank
from category_priors.category_feature_routing_factorial import RUNTIME_CLASSES
from category_priors.category_fragment_merge import (
    FragmentEdge,
    FragmentGraph,
    FragmentMergeDecision,
    FragmentMergeResult,
    FragmentNode,
    FragmentObject,
)
from category_priors.category_fragment_merge_runner import (
    FEATURE_SOURCE,
    POINT_AXIS_ATOL,
    SEMANTIC_ROUTE,
    _validate_feature_gaussian_axis,
    build_category_fragment_graphs,
    load_category_fragment_scene,
    load_fragment_graph,
    load_fragment_merge_result,
    merge_category_fragment_graphs,
    run_category_fragment_merge,
    run_category_fragment_scene,
    save_fragment_graph,
    save_fragment_merge_result,
)
from category_priors.teacher_prior import SAGA20_CLASSES


def _graph() -> FragmentGraph:
    return FragmentGraph(
        nodes=(
            FragmentNode(0, 0, np.asarray([0, 1, 2]), 0, "chair", 0.8, 1.0),
            FragmentNode(1, 1, np.asarray([3, 4, 5]), 0, "chair", 0.8, 1.0),
        ),
        edges=(FragmentEdge(0, 1, 3, 0.95, 0.01),),
        point_count=6,
        scene_scale_m_per_unit=1.0,
        global_typical_diag_m=1.0,
        diagnostics={"gt_used": False},
    )


def _result(graph: FragmentGraph, mode: str) -> FragmentMergeResult:
    return FragmentMergeResult(
        mode=mode,
        graph=graph,
        objects=(
            FragmentObject(
                source_fragment_ids=(0, 1),
                point_ids=np.arange(6),
                class_index=0,
                class_name="chair",
                metric_extents_m=(0.1, 0.2, 0.3),
                n_raw=6,
                G=0.8,
                C=1.0,
                P=0.8,
                support_threshold=5,
                base_score=0.85,
                accepted=True,
                output_instance_id=0,
            ),
        ),
        point_labels=np.zeros(6, dtype=np.int64),
        decisions=(
            FragmentMergeDecision(
                round_index=0,
                left_source_fragment_ids=(0,),
                right_source_fragment_ids=(1,),
                union_source_fragment_ids=(0, 1),
                left_prior_score=0.4,
                right_prior_score=0.5,
                union_prior_score=0.8,
                prior_eligible=True,
                mutual_best=True,
                accepted=True,
                reason="prior_eligible_mutual_best",
                cross_edge_count=3,
                affinity_cosine_median=0.95,
            ),
        ),
        diagnostics={
            "gt_used": False,
            "orphan_count": 0,
            "negative_metadata_count": 0,
        },
    )


def _raw_bank(
    top_class: np.ndarray, route_score: np.ndarray, seed: int
) -> CandidateBank:
    full = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return CandidateBank(
        class_names=RUNTIME_CLASSES,
        saga20_names=tuple(SAGA20_CLASSES),
        scene_scale_m_per_unit=1.0,
        seed=seed,
        global_pre_knn=np.full(6, -1, dtype=np.int64),
        semantic_top1=np.asarray(top_class, dtype=np.int64),
        semantic_top1_score=np.asarray(route_score, dtype=np.float64),
        branch_full_labels=full,
        branch_core_labels=full.copy(),
        assignment_confidence=np.full(6, 0.8, dtype=np.float64),
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "branch_class_index": 0,
                "full_point_count": 3,
                "core_point_count": 3,
                "raw_cluster_id": 0,
            },
            {
                "candidate_id": 1,
                "branch_class": "chair",
                "branch_class_index": 0,
                "full_point_count": 3,
                "core_point_count": 3,
                "raw_cluster_id": 1,
            },
        ),
        diagnostics={
            "feature_source": FEATURE_SOURCE,
            "semantic_route": SEMANTIC_ROUTE,
            "raw_clusters_only": True,
        },
    )


def _prior_payload() -> dict[str, Any]:
    geometry = {
        "log_bbox_diag_m": {"q50": 0.0},
        "log_extent_0_m": {"q25": -2.0, "q50": -1.5, "q75": -1.0},
        "log_extent_1_m": {"q25": -2.0, "q50": -1.5, "q75": -1.0},
        "log_extent_2_m": {"q25": -2.0, "q50": -1.5, "q75": -1.0},
        "log_surface_area_m2": {"q50": 0.0},
    }
    return {
        "provenance": {"splits": ["train"]},
        "global": {"shrunk": {"geometry": geometry}},
        "categories": {},
    }


def test_graph_and_merge_result_round_trip_without_hidden_identity_files(
    tmp_path: Path,
) -> None:
    graph = _graph()
    xyz = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
    identity = {"scene_id": "scene0645_00", "route": SEMANTIC_ROUTE}

    save_fragment_graph(graph, xyz, tmp_path / "graph", input_identity=identity)
    restored, restored_xyz, metadata = load_fragment_graph(
        tmp_path / "graph", expected_identity=identity
    )
    assert restored.identity() == graph.identity()
    assert np.array_equal(restored_xyz, xyz)
    assert metadata["input_identity"] == identity

    result = _result(restored, "global")
    save_fragment_merge_result(result, tmp_path / "result", input_identity=identity)
    replay, replay_metadata = load_fragment_merge_result(
        tmp_path / "result",
        restored,
        expected_mode="global",
        expected_identity=identity,
    )
    assert np.array_equal(replay.point_labels, result.point_labels)
    assert replay.objects[0].source_fragment_ids == (0, 1)
    assert replay.decisions[0].prior_eligible is True
    assert replay.decisions[0].mutual_best is True
    assert replay.graph is restored
    assert replay_metadata["input_identity"] == identity
    all_names = {path.name for path in tmp_path.rglob("*")}
    assert not any(
        "hash" in name or "lock" in name or "cache" in name for name in all_names
    )


def _runtime_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "scene"
    feature = base / "saga" / "contrastive_feature_point_cloud.ply"
    gaussian = (
        base
        / "output_models"
        / "point_cloud"
        / "iteration_30000"
        / "scene_point_cloud.ply"
    )
    labels = base / "saga" / "labels" / "label_features.pt"
    feature.parent.mkdir(parents=True)
    gaussian.parent.mkdir(parents=True)
    labels.parent.mkdir(parents=True)
    feature.write_bytes(b"native-feature")
    gaussian.write_bytes(b"registered-gaussian-axis")
    labels.write_bytes(b"label-features")
    priors = tmp_path / "priors.json"
    priors.write_text(json.dumps(_prior_payload()), encoding="utf-8")
    manifest = tmp_path / "runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "scene_runtime_manifest",
                "scenes": [
                    {
                        "scene_id": "scene0645_00",
                        "base_path": str(base),
                        "scene_scale_m_per_unit": 1.0,
                        # This deliberately invalid GT path proves the worker
                        # neither validates nor opens any evaluation asset.
                        "gt_path": str(tmp_path / "must-not-be-opened.gt"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, priors, base


def test_runner_uses_only_native_predicted_route_and_reuses_complete_scene(
    tmp_path: Path,
) -> None:
    manifest, priors, _ = _runtime_files(tmp_path)
    output = tmp_path / "runs"
    calls = {"feature": 0, "label": 0, "raw": 0, "graph": 0, "merge": []}
    graph_ids: list[int] = []

    def feature_loader(path):
        calls["feature"] += 1
        assert path.name == "contrastive_feature_point_cloud.ply"
        xyz = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        affinity = np.tile(np.eye(1, 32, 0), (6, 1))
        semantic = affinity.copy()
        return xyz, affinity, semantic

    def label_loader(path):
        calls["label"] += 1
        assert path.name == "label_features.pt"
        return np.eye(32, dtype=np.float64)

    def raw_builder(**kwargs):
        calls["raw"] += 1
        assert kwargs["feature_source"] == FEATURE_SOURCE
        assert kwargs["route"] == SEMANTIC_ROUTE
        assert kwargs["sample_cap"] == 5_000
        assert np.all(kwargs["branch_class"] == 0)
        return _raw_bank(kwargs["top_class"], kwargs["route_score"], kwargs["seed"])

    def graph_builder(bank, xyz_scene, affinity, global_diag):
        calls["graph"] += 1
        assert global_diag == 1.0
        return _graph()

    def merge_builder(graph, xyz_scene, payload, mode):
        calls["merge"].append(mode)
        graph_ids.append(id(graph))
        return _result(graph, mode)

    dependencies = {
        "feature_loader": feature_loader,
        "gaussian_loader": lambda path: (
            np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        ),
        "label_loader": label_loader,
        "raw_bank_builder": raw_builder,
        "graph_builder": graph_builder,
        "merge_builder": merge_builder,
    }
    first = run_category_fragment_merge(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        **dependencies,
    )
    assert first["scenes"][0]["gt_used"] is False
    assert calls == {
        "feature": 1,
        "label": 1,
        "raw": 1,
        "graph": 1,
        "merge": ["global", "class"],
    }
    assert graph_ids[0] == graph_ids[1]
    artifacts = load_category_fragment_scene(output / "scene0645_00")
    assert artifacts.uniform.graph is artifacts.class_shrunk.graph
    assert artifacts.raw_bank.diagnostics["gt_used"] is False
    axis = artifacts.raw_bank.diagnostics["point_axis_validation"]
    assert axis == {
        "passed": True,
        "feature_point_count": 6,
        "gaussian_point_count": 6,
        "max_abs_error_after_transform": 0.0,
        "atol": POINT_AXIS_ATOL,
        "gaussian_transform_applied_to_both_axes": True,
        "gt_used": False,
    }

    second = run_category_fragment_merge(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        **dependencies,
    )
    assert second["scenes"][0]["status"] == "reused"
    assert calls == {
        "feature": 1,
        "label": 1,
        "raw": 1,
        "graph": 1,
        "merge": ["global", "class"],
    }


def test_corrupt_replay_reruns_only_that_mode_from_the_same_graph(
    tmp_path: Path,
) -> None:
    manifest, priors, _ = _runtime_files(tmp_path)
    output = tmp_path / "runs"
    calls = {"feature": 0, "label": 0, "raw": 0, "graph": 0, "merge": []}

    def feature_loader(path):
        calls["feature"] += 1
        xyz = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        values = np.tile(np.eye(1, 32, 0), (6, 1))
        return xyz, values, values

    def label_loader(path):
        calls["label"] += 1
        return np.eye(32)

    def raw_builder(**kwargs):
        calls["raw"] += 1
        return _raw_bank(kwargs["top_class"], kwargs["route_score"], kwargs["seed"])

    def graph_builder(*args):
        calls["graph"] += 1
        return _graph()

    def merge_builder(graph, xyz_scene, payload, mode):
        calls["merge"].append(mode)
        return _result(graph, mode)

    dependencies = {
        "feature_loader": feature_loader,
        "gaussian_loader": lambda path: (
            np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        ),
        "label_loader": label_loader,
        "raw_bank_builder": raw_builder,
        "graph_builder": graph_builder,
        "merge_builder": merge_builder,
    }
    run_category_fragment_merge(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        **dependencies,
    )
    calls["merge"].clear()
    (output / "scene0645_00" / "replay" / "class" / "merge_result.npz").write_bytes(
        b"broken"
    )

    result = run_category_fragment_merge(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        **dependencies,
    )
    assert result["scenes"][0]["rebuilt"] == ["replay-class"]
    assert calls == {"feature": 1, "label": 1, "raw": 1, "graph": 1, "merge": ["class"]}


def test_explicit_build_and_mode_replay_stages_write_completion_only_after_both(
    tmp_path: Path,
) -> None:
    manifest, priors, _ = _runtime_files(tmp_path)
    output = tmp_path / "runs"
    calls: list[str] = []

    def feature_loader(path):
        calls.append("feature")
        xyz = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        values = np.tile(np.eye(1, 32, 0), (6, 1))
        return xyz, values, values

    def raw_builder(**kwargs):
        calls.append("raw")
        return _raw_bank(kwargs["top_class"], kwargs["route_score"], kwargs["seed"])

    build = build_category_fragment_graphs(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        feature_loader=feature_loader,
        gaussian_loader=lambda path: (
            np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
        ),
        label_loader=lambda path: np.eye(32),
        raw_bank_builder=raw_builder,
        graph_builder=lambda *args: _graph(),
    )
    scene_root = output / "scene0645_00"
    assert build["stage"] == "build-fragment-graph"
    assert build["scenes"][0]["point_axis_validation"]["passed"] is True
    assert (scene_root / "fragment_graph" / "fragment_graph.npz").is_file()
    assert not (scene_root / "replay").exists()
    assert not (scene_root / "scene_complete.json").exists()

    def merge_builder(graph, xyz_scene, payload, mode):
        calls.append(f"merge-{mode}")
        return _result(graph, mode)

    global_only = merge_category_fragment_graphs(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        modes=("global",),
        merge_builder=merge_builder,
    )
    assert global_only["scenes"][0]["scene_complete"] is False
    assert (scene_root / "replay" / "global" / "merge_result.npz").is_file()
    assert not (scene_root / "replay" / "class").exists()
    assert not (scene_root / "scene_complete.json").exists()

    class_only = merge_category_fragment_graphs(
        runtime_manifest=manifest,
        category_priors=priors,
        output_root=output,
        scene_ids=("scene0645_00",),
        modes=("class",),
        merge_builder=merge_builder,
    )
    assert class_only["scenes"][0]["scene_complete"] is True
    assert (scene_root / "scene_complete.json").is_file()
    assert calls == ["feature", "raw", "merge-global", "merge-class"]
    artifacts = load_category_fragment_scene(scene_root)
    assert artifacts.uniform.graph is artifacts.class_shrunk.graph


def test_worker_public_signatures_have_no_ground_truth_input() -> None:
    for function in (run_category_fragment_scene, run_category_fragment_merge):
        parameter_names = set(inspect.signature(function).parameters)
        assert not any(
            name.startswith("gt") or "ground_truth" in name for name in parameter_names
        )


def test_point_axis_validation_uses_registered_transform_and_rejects_reordering() -> (
    None
):
    feature = np.asarray(
        [[0.0, 0.0, 0.0], [0.25, 0.5, 0.75], [1.0, -2.0, 3.0]],
        dtype=np.float64,
    )
    transform = (
        (0.0, -1.0, 0.0, 2.0),
        (1.0, 0.0, 0.0, -3.0),
        (0.0, 0.0, 2.0, 1.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    proof = _validate_feature_gaussian_axis(
        feature, feature.copy(), transform, scene_id="scene0645_00"
    )
    assert proof["passed"] is True
    assert proof["max_abs_error_after_transform"] == 0.0

    with pytest.raises(ValueError, match="point axis differs"):
        _validate_feature_gaussian_axis(
            feature,
            feature[[1, 0, 2]],
            transform,
            scene_id="scene0645_00",
        )


def test_point_axis_validation_rejects_count_mismatch_before_candidate_build(
    tmp_path: Path,
) -> None:
    manifest, priors, _ = _runtime_files(tmp_path)
    called = {"raw": False}
    feature_xyz = np.arange(18, dtype=np.float64).reshape(6, 3) * 0.01
    values = np.tile(np.eye(1, 32, 0), (6, 1))

    def raw_builder(**kwargs):
        called["raw"] = True
        return _raw_bank(kwargs["top_class"], kwargs["route_score"], kwargs["seed"])

    with pytest.raises(ValueError, match="point-count mismatch"):
        build_category_fragment_graphs(
            runtime_manifest=manifest,
            category_priors=priors,
            output_root=tmp_path / "runs",
            scene_ids=("scene0645_00",),
            feature_loader=lambda path: (feature_xyz, values, values),
            gaussian_loader=lambda path: feature_xyz[:-1],
            label_loader=lambda path: np.eye(32),
            raw_bank_builder=raw_builder,
            graph_builder=lambda *args: _graph(),
        )
    assert called["raw"] is False
