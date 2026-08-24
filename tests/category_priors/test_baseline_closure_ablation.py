from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.baseline_closure import RuntimeScene
from category_priors.baseline_closure_ablation import (
    build_fixed_invocation,
    build_harness_invocation,
    canonical_point_partition,
    compare_partitions,
    evaluate_official_parity,
    parity_allows_structural_ablation,
    stage_trace_is_complete,
)


def _scene(tmp_path: Path) -> RuntimeScene:
    return RuntimeScene(
        scene_id="scene0064_01",
        base_path=tmp_path / "scene0064_01",
        python_bin=tmp_path / "python",
    )


def test_fixed_invocation_preserves_original_four_class_branch(tmp_path: Path) -> None:
    invocation = build_fixed_invocation(
        _scene(tmp_path),
        tmp_path / "closure",
        tmp_path / "fixed",
        condition="B1-original",
    )
    command = list(invocation.command)
    start = command.index("--other_classes") + 1
    assert command[start : start + 4] == ["socket", "book", "remote", "key"]
    assert "--v7-causal-ablation" not in command


def test_fixed_b0_uses_historical_no_match_sentinel(tmp_path: Path) -> None:
    command = list(
        build_fixed_invocation(
            _scene(tmp_path),
            tmp_path / "closure",
            tmp_path / "fixed",
            condition="B0-global",
        ).command
    )
    assert command[command.index("--other_classes") + 1] == "__disabled__"


def test_harness_invocation_is_explicit_and_has_no_prior(tmp_path: Path) -> None:
    command = list(
        build_harness_invocation(
            _scene(tmp_path),
            tmp_path / "closure",
            tmp_path / "current",
            level="L2",
            condition="B1-original",
            scene_scale_m_per_unit=1.0,
        ).command
    )
    assert command[command.index("--v7-causal-ablation") + 1] == "L2"
    assert command[command.index("--teacher-prior-mode") + 1] == "original"
    assert "--disable_other_classes" not in command
    assert "--category-priors" not in command
    assert "--prior_config" not in command
    assert "--max_contributor_cache_path" not in command


def test_harness_b0_disables_original_branch(tmp_path: Path) -> None:
    command = build_harness_invocation(
        _scene(tmp_path),
        tmp_path / "closure",
        tmp_path / "current",
        level="L0",
        condition="B0-global",
        scene_scale_m_per_unit=1.0,
    ).command
    assert "--disable_other_classes" in command


def test_harness_rejects_nonphysical_scale(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_harness_invocation(
            _scene(tmp_path),
            tmp_path / "closure",
            tmp_path / "current",
            level="L3",
            condition="B1-original",
            scene_scale_m_per_unit=0.0,
        )


def test_partition_comparison_ignores_raw_instance_ids(tmp_path: Path) -> None:
    left = {
        "point_labels": [8, 8, -1, 3, 3],
        "instances": {"8": {"class": "book"}, "3": {"class": "chair"}},
    }
    right = {
        "point_labels": [1, 1, -1, 9, 9],
        "instances": {"1": {"class": "book"}, "9": {"class": "chair"}},
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    comparison = compare_partitions(left_path, right_path)
    assert comparison["equivalent"] is True
    assert comparison["changed_points"] == 0
    assert comparison["raw_internal"]["geometry_changed_points"] == 0
    assert comparison["declared_exported"]["geometry_changed_points"] == 0


def test_partition_comparison_detects_class_or_membership_change(
    tmp_path: Path,
) -> None:
    left = {
        "point_labels": [0, 0, -1],
        "instances": {"0": {"class": "book"}},
    }
    right = {
        "point_labels": [0, -1, -1],
        "instances": {"0": {"class": "chair"}},
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    comparison = compare_partitions(left_path, right_path)
    assert comparison["equivalent"] is False
    assert comparison["changed_points"] == 2
    assert comparison["declared_exported"]["geometry_changed_points"] == 1
    assert comparison["declared_exported"]["class_changed_points"] == 1
    assert comparison["metadata"]["matched_class_difference_count"] == 1


def test_canonical_partition_projects_orphan_ids_to_background() -> None:
    assert canonical_point_partition(
        {"point_labels": [2, -1], "instances": {}}
    ) == (None, None)


def test_partition_comparison_exposes_orphan_projection_stats(tmp_path: Path) -> None:
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    payload = {
        "point_labels": [4, 9, -1],
        "instances": {"4": {"class": "book"}},
    }
    left_path.write_text(json.dumps(payload), encoding="utf-8")
    right_path.write_text(json.dumps(payload), encoding="utf-8")

    comparison = compare_partitions(left_path, right_path)

    assert comparison["equivalent"] is True
    assert comparison["left_projection"]["orphan_instance_ids"] == [9]
    assert comparison["left_projection"]["orphan_gaussian_count"] == 1


def test_partition_comparison_does_not_global_rank_shift_after_small_declaration(
    tmp_path: Path,
) -> None:
    small = 31
    shared_labels = [11] * small + [4] * 120 + [8] * 140 + [-1] * 9
    left = {
        "point_labels": shared_labels,
        "instances": {
            "4": {"class": "book", "score": 0.8},
            "8": {"class": "table", "score": 0.7},
        },
    }
    right = {
        "point_labels": shared_labels,
        "instances": {
            "11": {"class": "chair", "score": 0.9},
            "4": {"class": "book", "score": 0.8},
            "8": {"class": "table", "score": 0.7},
        },
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")

    comparison = compare_partitions(left_path, right_path)

    assert comparison["raw_geometry_changed_points"] == 0
    assert comparison["raw_internal"]["geometry_equivalent"] is True
    assert comparison["exported_geometry_changed_points"] == small
    assert comparison["changed_points"] == small
    assert comparison["declared_exported"]["unmatched_right_instances"] == [
        {"instance_id": 11, "size": small, "class": "chair"}
    ]
    assert comparison["metadata"]["right_only_declared_instance_ids"] == [11]
    assert comparison["metadata"]["equivalent"] is False

    left_canonical = canonical_point_partition(left)
    right_canonical = canonical_point_partition(right)
    assert left_canonical[small:] == right_canonical[small:]
    assert left_canonical[:small] == (None,) * small

    comparison["official_evaluator_parity"] = {
        "evaluated": True,
        "equal": True,
        "metrics": {"map": [0.1, 0.1], "ap50": [0.2, 0.2]},
    }
    gate = parity_allows_structural_ablation(comparison)
    assert gate == {
        "allowed": True,
        "raw_geometry_exact": True,
        "exported_boundary_points": 31,
        "exported_boundary_below_100": True,
        "matched_instance_classes_exact": True,
        "boundary_is_only_subregion_declarations": True,
        "official_evaluator_parity_evaluated": True,
        "official_evaluator_parity_equal": True,
        "blocked_without_official_evaluator_parity": False,
    }


def test_structural_gate_rejects_raw_or_large_export_changes() -> None:
    base = {
        "raw_geometry_changed_points": 0,
        "exported_geometry_changed_points": 31,
        "exported_class_changed_points": 0,
        "metadata": {"matched_class_difference_count": 0},
        "declared_exported": {
            "unmatched_left_instances": [],
            "unmatched_right_instances": [
                {"instance_id": 11, "size": 31, "class": "chair"}
            ],
        },
        "official_evaluator_parity": {"evaluated": True, "equal": True},
    }
    raw_changed = {**base, "raw_geometry_changed_points": 1}
    export_large = {**base, "exported_geometry_changed_points": 100}
    class_changed = {**base, "exported_class_changed_points": 1}
    assert not parity_allows_structural_ablation(raw_changed)["allowed"]
    assert not parity_allows_structural_ablation(export_large)["allowed"]
    assert not parity_allows_structural_ablation(class_changed)["allowed"]


def test_structural_gate_never_infers_ap_parity_from_gaussian_count() -> None:
    comparison = {
        "raw_geometry_changed_points": 0,
        "exported_geometry_changed_points": 31,
        "exported_class_changed_points": 0,
        "metadata": {"matched_class_difference_count": 0},
        "declared_exported": {
            "unmatched_left_instances": [],
            "unmatched_right_instances": [
                {"instance_id": 11, "size": 31, "class": "chair"}
            ],
        },
    }
    gate = parity_allows_structural_ablation(comparison)
    assert gate["allowed"] is False
    assert gate["blocked_without_official_evaluator_parity"] is True


def test_official_parity_evidence_is_computed_from_registered_evaluator(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "category_priors.baseline_closure_ablation.load_taxonomy",
        lambda: SimpleNamespace(canonical_classes=("chair",)),
    )
    monkeypatch.setattr(
        "category_priors.baseline_closure_ablation.load_ground_truth_npz",
        lambda path, scene_id: (np.zeros((1, 3)), object()),
    )
    monkeypatch.setattr(
        "category_priors.baseline_closure_ablation.saga_scene_predictions",
        lambda output_json, **kwargs: ([Path(output_json).stem], {"mapped": 1.0}),
    )

    def fake_evaluate(_gt, predictions, _classes, **_kwargs):
        changed = predictions[0] == "changed"
        value = 0.4 if changed else 0.5
        pred_instances = 2 if predictions[0] == "count-only" else 1
        aggregate = {"map_50_90": value}
        view = {
            "aggregate": aggregate,
            "per_class": {
                "chair": {
                    "ap_0.50": value,
                    "ap_50_90": value,
                    "gt_instances": 1,
                    "pred_instances": pred_instances,
                }
            },
        }
        return {
            "protocols": {
                "scannet_official_9": {
                    "full_saga20": view,
                    "predictable_intersection": view,
                }
            }
        }

    monkeypatch.setattr(
        "category_priors.baseline_closure_ablation.evaluate_baseline_closure",
        fake_evaluate,
    )
    scene = {"base_path": str(tmp_path), "gaussian_to_gt_transform": np.eye(4).tolist()}
    equal = evaluate_official_parity(
        scene_id="scene0000_00",
        reference_output=tmp_path / "same.json",
        candidate_output=tmp_path / "same.json",
        gt_dir=tmp_path,
        runtime_scene=scene,
    )
    unequal = evaluate_official_parity(
        scene_id="scene0000_00",
        reference_output=tmp_path / "same.json",
        candidate_output=tmp_path / "changed.json",
        gt_dir=tmp_path,
        runtime_scene=scene,
    )
    diagnostic_only = evaluate_official_parity(
        scene_id="scene0000_00",
        reference_output=tmp_path / "same.json",
        candidate_output=tmp_path / "count-only.json",
        gt_dir=tmp_path,
        runtime_scene=scene,
    )
    assert equal["evaluated"] is True and equal["equal"] is True
    assert unequal["equal"] is False
    assert diagnostic_only["equal"] is True
    assert diagnostic_only["metric_surface_equal"] is True
    assert diagnostic_only["diagnostic_protocol_equal"] is False
    assert equal["protocol"] == "scannet-official-instance-9-v1"


def test_partition_comparison_uses_internal_stage_trace(tmp_path: Path) -> None:
    internal = np.asarray([11] * 31 + [4] * 120 + [-1] * 3, dtype=np.int64)
    left = {
        "point_labels": internal.tolist(),
        "instances": {
            "11": {"class": "chair", "score": 0.8},
            "4": {"class": "book", "score": 0.7},
        },
    }
    right = {
        "point_labels": (np.where(internal == 4, 0, -1)).tolist(),
        "instances": {"0": {"class": "book", "score": 0.7}},
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    trace_path = tmp_path / "stage_trace.npz"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")
    np.savez_compressed(trace_path, final_internal_labels=internal)

    comparison = compare_partitions(
        left_path, right_path, right_internal_trace=trace_path
    )

    assert comparison["raw_geometry_changed_points"] == 0
    assert comparison["exported_geometry_changed_points"] == 31


def test_stage_trace_completion_requires_every_registered_partition(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.json"
    trace = tmp_path / "stage_trace.npz"
    output.write_text(
        json.dumps({"point_labels": [-1, 0], "instances": {}}),
        encoding="utf-8",
    )
    keys = (
        "global_sample_core",
        "global_full_assignment",
        "other_class_candidates",
        "branch_class_before_merge",
        "merged_partition",
        "post_global_knn",
        "post_filter",
        "post_attach",
        "final_internal_labels",
    )
    np.savez_compressed(trace, **{key: np.asarray([-1, 0]) for key in keys})
    trace.with_suffix(".json").write_text(
        json.dumps(
            {"schema": "saga-v9-legacy-stage-trace-v1", "point_count": 2}
        ),
        encoding="utf-8",
    )
    assert stage_trace_is_complete(trace, output)


def test_partition_comparison_reports_metadata_without_confusing_geometry(
    tmp_path: Path,
) -> None:
    left = {
        "point_labels": [5, 5, -1],
        "instances": {"5": {"class": "book", "score": 0.2}},
    }
    right = {
        "point_labels": [9, 9, -1],
        "instances": {"9": {"class": "book", "score": 0.8}},
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")

    comparison = compare_partitions(left_path, right_path)

    assert comparison["raw_geometry_changed_points"] == 0
    assert comparison["declared_exported"]["geometry_changed_points"] == 0
    assert comparison["declared_exported"]["class_changed_points"] == 0
    assert comparison["declared_exported"]["metadata_difference_count"] == 1
    assert comparison["declared_exported"]["matches"][0][
        "metadata_different_fields"
    ] == ["score"]
    assert comparison["metadata"]["matched_metadata_difference_count"] == 1
    assert comparison["equivalent"] is True
    assert comparison["metadata_equivalent"] is False
    assert comparison["fully_equivalent"] is False


def test_partition_comparison_reports_empty_declared_metadata(tmp_path: Path) -> None:
    left = {"point_labels": [-1, -1], "instances": {}}
    right = {
        "point_labels": [-1, -1],
        "instances": {"7": {"class": "chair", "score": 0.3}},
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")

    comparison = compare_partitions(left_path, right_path)

    assert comparison["raw_geometry_changed_points"] == 0
    assert comparison["exported_geometry_changed_points"] == 0
    assert comparison["metadata"]["right_only_declared_instance_ids"] == [7]
    assert comparison["metadata"]["equivalent"] is False
    assert comparison["equivalent"] is True
    assert comparison["fully_equivalent"] is False
