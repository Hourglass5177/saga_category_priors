from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import category_priors.teacher_prior_runner as runner_module
from category_priors.io import load_json, write_json
from category_priors.teacher_prior import (
    build_teacher_hdbscan,
    class_local_knn,
    exclusive_top1_masks,
    materialize_teacher_prior,
    merge_branch_labels,
    protect_multi_anchor_halo,
    rescue_same_class_noise,
    resolve_teacher_parameters,
    restore_surviving_branches,
    restore_preserved_branch,
    saga20_branch_classes,
    teacher_spatial_distance,
)
from category_priors.teacher_prior_runner import (
    build_teacher_prior_command,
    execute_teacher_prior_runs,
    teacher_prior_run_paths,
)


def _summary(value: float) -> dict[str, float]:
    return {"q50": value}


def _stats(diagonal_m: float, area_m2: float, boundary: float) -> dict[str, object]:
    return {
        "geometry": {
            "log_bbox_diag_m": _summary(math.log(diagonal_m)),
            "log_surface_area_m2": _summary(math.log(area_m2)),
        },
        "neighborhood": {"boundary_fixed:0.05": _summary(boundary)},
    }


def _priors() -> dict[str, object]:
    return {
        "kind": "category_priors",
        "provenance": {"splits": ["train"]},
        "global": {"shrunk": _stats(2.0, 4.0, 0.10)},
        "categories": {
            "chair": {"shrunk": _stats(1.0, 1.0, 0.20)},
            "cup": {"shrunk": _stats(0.2, 0.04, 0.40)},
        },
    }


def _scene() -> dict[str, object]:
    return {
        "base_path": "assets/scene0000_00",
        "scene_scale_m_per_unit": 1.0,
        "python_bin": "env/bin/python",
    }


def _scene_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "scene_runtime.json"
    write_json(
        path,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": "scene0000_00",
                    "physical_scene_id": "scene0000",
                    **_scene(),
                }
            ],
        },
    )
    return path


def _runner_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pipeline = tmp_path / "run_pipeline.sh"
    pipeline.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    parameters = tmp_path / "teacher_params.json"
    write_json(
        parameters,
        materialize_teacher_prior(_priors()),
    )
    return pipeline, parameters


def _successful_subprocess(calls: list[list[str]]):
    def fake_run(command, **kwargs):
        calls.append(list(command))
        output = Path(command[command.index("--json-path") + 1])
        diagnostics = Path(
            command[command.index("--prior-metadata-path") + 1]
        )
        write_json(output, {"point_labels": [0, 0], "instances": {"0": {}}})
        write_json(diagnostics, {"kind": "teacher_prior_scores"})
        kwargs["stdout"].write("postprocess complete\n")
        return SimpleNamespace(returncode=0)

    return fake_run


def test_original_condition_is_parameter_free_and_selects_a800_mode(
    tmp_path: Path,
) -> None:
    category_params = tmp_path / "teacher_params.json"
    command, paths = build_teacher_prior_command(
        tmp_path / "run_pipeline.sh",
        _scene(),
        tmp_path / "runs",
        "original",
        "scene0000_00",
        42,
        category_params,
    )

    assert command[command.index("--teacher-prior-mode") + 1] == "original"
    assert "--teacher-category-params" not in command
    assert "--teacher-branch-preservation" not in command
    assert paths["run_dir"] == (
        tmp_path.resolve()
        / "runs"
        / "original"
        / "scene0000_00"
        / "seed-42"
    )
    pipeline_source = (
        Path(__file__).resolve().parents[2] / "run_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert 'teacher_prior_mode="original"' in pipeline_source


def test_all_uniform_and_data_modes_change_parameters_not_shared_mechanics() -> None:
    table = materialize_teacher_prior(_priors())
    shared = {
        "semantic_threshold",
        "sample_num",
        "feature_ratio",
        "spatial_ratio",
        "semantic_ratio",
        "assignment_threshold",
        "min_samples",
    }

    assert set(table["classes"]) == {"chair", "cup"}
    for class_name in table["classes"]:
        left = resolve_teacher_parameters(table, class_name, "all-uniform")
        right = resolve_teacher_parameters(table, class_name, "combined")
        assert {name: left[name] for name in shared} == {
            name: right[name] for name in shared
        }
    assert resolve_teacher_parameters(
        table, "cup", "all-uniform"
    )["spatial_scale_m"] == 2.0
    assert resolve_teacher_parameters(
        table, "cup", "combined"
    )["spatial_scale_m"] == 0.2


def test_hdbscan_min_samples_is_explicit_and_defaults_to_three() -> None:
    captured: dict[str, object] = {}

    class FakeClusterer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    table = materialize_teacher_prior(_priors())
    parameters = resolve_teacher_parameters(table, "cup", "all-uniform")
    build_teacher_hdbscan(
        FakeClusterer,
        parameters["min_cluster_size"],
        parameters["min_samples"],
    )

    assert parameters["min_samples"] == 3
    assert captured["min_samples"] == 3
    assert captured["min_cluster_size"] == 5
    assert captured["metric"] == "precomputed"


def test_typical_size_changes_spatial_distance_without_max_normalization() -> None:
    xyz_m = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    small_scale = teacher_spatial_distance(xyz_m, 0.5)
    large_scale = teacher_spatial_distance(xyz_m, 2.0)

    assert small_scale[0, 1] == 1.0
    assert large_scale[0, 1] == 0.25
    assert not np.array_equal(small_scale, large_scale)


def test_knn_and_noise_rescue_remain_inside_one_class_branch() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=np.float64,
    )
    chair = class_local_knn(np.asarray([3, 3, -1]), xyz, 3)
    cup = class_local_knn(np.asarray([9, 9, -1]), xyz, 3)
    assert chair.tolist() == [3, 3, 3]
    assert cup.tolist() == [9, 9, 9]

    labels = np.asarray([7, 7, -1, -1])
    rescue_xyz = np.asarray(
        [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.025, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    rescued, count = rescue_same_class_noise(labels, rescue_xyz, 0.05)
    assert rescued.tolist() == [7, 7, 7, -1]
    assert count == 1


def test_missing_category_falls_back_to_global_uniform_parameters() -> None:
    table = materialize_teacher_prior(_priors())
    unknown = resolve_teacher_parameters(table, "unknown-object", "combined")
    assert unknown["spatial_scale_m"] == table["global"]["typical_diag_m"]
    assert unknown["min_cluster_size"] == table["global"]["min_cluster_size"]
    assert unknown["knn_k"] == table["global"]["knn_k"]
    assert unknown["rescue_radius_m"] == table["global"]["rescue_radius_m"]


def test_restore_only_branches_that_survive_global_filtering() -> None:
    filtered = np.asarray([10, -1, 12, -1, 20, 20])
    membership = np.asarray([10, 10, 11, 11, -1, -1])
    restored, count = restore_surviving_branches(filtered, membership)
    assert restored.tolist() == [10, 10, 12, -1, 20, 20]
    assert count == 1


def test_multi_anchor_protection_restores_only_near_vote_confirmed_halo() -> None:
    labels = np.asarray([10, 10, 10, 99, -1, 20, -1])
    membership = np.asarray([10, 10, 10, 10, 10, 20, 20])
    xyz = np.asarray([
        [0.00, 0.0, 0.0],
        [0.02, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [0.03, 0.0, 0.0],
        [1.00, 0.0, 0.0],
        [2.00, 0.0, 0.0],
        [2.01, 0.0, 0.0],
    ])
    protected, diagnostics = protect_multi_anchor_halo(
        labels,
        membership,
        xyz,
        {10: "chair", 20: "cup"},
        {
            10: {"min_cluster_size": 3, "protection_radius_m": 0.10},
            20: {"min_cluster_size": 3, "protection_radius_m": 0.10},
        },
        {10: np.asarray([0.8, 0.1]), 20: np.asarray([0.1, 0.8])},
        {"chair": 0, "cup": 1},
        0.3,
    )

    assert protected.tolist() == [10, 10, 10, 10, -1, 20, -1]
    assert diagnostics["accepted_branches"] == 1
    assert diagnostics["restored_points"] == 1
    assert diagnostics["rejection_reasons"] == {"insufficient_anchors": 1}


def test_multi_anchor_protection_rejects_wrong_class_or_background_vote() -> None:
    labels = np.asarray([4, 4, 4, -1])
    membership = np.asarray([4, 4, 4, 4])
    xyz = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0],
                      [0.02, 0.0, 0.0], [0.015, 0.0, 0.0]])
    for ratio in (np.asarray([0.2, 0.7]), np.asarray([0.2, 0.1])):
        protected, diagnostics = protect_multi_anchor_halo(
            labels, membership, xyz, {4: "chair"},
            {4: {"min_cluster_size": 3, "protection_radius_m": 0.1}},
            {4: ratio}, {"chair": 0, "cup": 1}, 0.3,
        )
        assert protected.tolist() == labels.tolist()
        assert diagnostics["rejection_reasons"] == {"vote_rejected": 1}


def test_top1_and_preserved_branches_are_independent_of_category_order() -> None:
    assert materialize_teacher_prior(
        _priors(), branch_preservation=True
    )["branch_preservation"] is True
    assert materialize_teacher_prior(
        _priors(), restore_after_global_filter=True
    )["restore_after_global_filter"] is True
    semantic = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    label_features = np.eye(2)
    forward = exclusive_top1_masks(semantic, label_features, [0, 1], 0.5)
    reverse = exclusive_top1_masks(semantic, label_features, [1, 0], 0.5)
    for class_index in (0, 1):
        np.testing.assert_array_equal(forward[class_index], reverse[class_index])
    assert not np.any(forward[0] & forward[1])

    fallback = np.asarray([10, 10, 11, 11, -1])
    # Reversing class iteration also reverses the temporary branch IDs.  The
    # semantic partition, final IDs, and preserved class map must remain stable.
    merged_a = merge_branch_labels(
        fallback,
        np.asarray([0, 0, 1, 1, -1]),
        {0: "chair", 1: "cup"},
    )
    merged_b = merge_branch_labels(
        fallback,
        np.asarray([1, 1, 0, 0, -1]),
        {0: "cup", 1: "chair"},
    )
    for left, right in zip(merged_a, merged_b):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right
    damaged = np.zeros_like(merged_a[0])
    np.testing.assert_array_equal(
        restore_preserved_branch(damaged, merged_a[1]),
        np.asarray([12, 12, 13, 13, 0]),
    )


def test_branch_is_exact_saga20_and_top1_ignores_non_target_classes() -> None:
    taxonomy = load_json(
        Path(__file__).parents[2] / "category_priors" / "default_taxonomy.json"
    )
    class_to_idx = {
        name: index
        for index, name in enumerate(
            [*taxonomy["canonical_classes"], "flower", "robot"]
        )
    }
    assert list(saga20_branch_classes(class_to_idx)) == taxonomy["canonical_classes"]
    assert "flower" not in saga20_branch_classes(class_to_idx)
    assert "robot" not in saga20_branch_classes(class_to_idx)

    # The non-target third embedding is strongest, but it must not steal the
    # point from the top-1 competition among the requested SAGA20 branches.
    semantic = np.asarray([[0.8, 0.1, 1.0]])
    masks = exclusive_top1_masks(semantic, np.eye(3), [0, 1], threshold=0.7)
    assert masks[0].tolist() == [True]
    assert masks[1].tolist() == [False]


def test_runner_skips_complete_and_reruns_corrupt_output(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = _scene_manifest(tmp_path)
    pipeline, parameters = _runner_inputs(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        runner_module.subprocess, "run", _successful_subprocess(calls)
    )

    first = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        parameters,
        conditions=["U0-all-uniform"],
        seeds=[42],
    )
    second = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        parameters,
        conditions=["U0-all-uniform"],
        seeds=[42],
    )
    paths = teacher_prior_run_paths(
        tmp_path / "runs", "U0-all-uniform", "scene0000_00", 42
    )
    write_json(paths["output"], {})
    third = execute_teacher_prior_runs(
        manifest,
        tmp_path / "runs",
        pipeline,
        parameters,
        conditions=["U0-all-uniform"],
        seeds=[42],
    )

    assert first["complete"] == 1
    assert second["skipped"] == 1
    assert third["complete"] == 1
    assert len(calls) == 2
    assert load_json(paths["output"])["point_labels"] == [0, 0]


def test_runner_seed_paths_do_not_overlap(tmp_path: Path) -> None:
    seed_42 = teacher_prior_run_paths(
        tmp_path / "runs", "D-size", "scene0000_00", 42
    )
    seed_3407 = teacher_prior_run_paths(
        tmp_path / "runs", "D-size", "scene0000_00", 3407
    )

    assert seed_42["run_dir"].name == "seed-42"
    assert seed_3407["run_dir"].name == "seed-3407"
    for key in ("output", "pending_output", "diagnostics", "log"):
        assert seed_42[key] != seed_3407[key]
