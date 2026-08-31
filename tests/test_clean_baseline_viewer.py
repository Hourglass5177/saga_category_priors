from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline.viewer import (
    CORRECT_COLOR,
    GT_COLOR,
    SAME_CLASS_WRONG_INSTANCE_COLOR,
    UNSUPPORTED_COLOR,
    WRONG_CLASS_COLOR,
    audit_clean_viewer_scene,
    build_clean_baseline_viewer,
    select_clean_viewer_cases,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "gaussian_xyz": np.asarray(
            [[index, 0.0, 0.0] for index in range(8)], dtype=np.float64
        ),
        "gt_xyz": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        "gt_semantic": np.asarray([0, 0, 0, 1], dtype=np.int64),
        "gt_instance": np.asarray([10, 10, 11, 12], dtype=np.int64),
        # Instance 0 sees: two correct, one same-class/wrong-instance,
        # one unsupported and one wrong class. Instance 1 is a pure FP.
        # Instance 2 is a split/duplicate prediction of GT instance 10.
        "gaussian_to_gt_point": np.asarray(
            [0, 1, 2, -1, 3, -1, -1, 0], dtype=np.int64
        ),
    }


def _prediction(condition: str) -> dict[str, object]:
    return {
        "scene_id": "scene-test",
        "condition": condition,
        "point_labels": [0, 0, 0, 0, 0, 1, 1, 2],
        "instances": {
            "0": {"class": "chair", "score": 0.8},
            "1": {"class": "chair", "score": 0.1},
            "2": {"class": "chair", "score": 0.7},
        },
    }


def _predictions() -> dict[str, dict[str, object]]:
    return {
        condition: _prediction(condition)
        for condition in ("C0-no-prior", "U-global", "D-predicted")
    }


def _audit() -> dict[str, object]:
    arrays = _arrays()
    return audit_clean_viewer_scene(
        scene_id="scene-test",
        predictions=_predictions(),
        class_names=("chair", "table"),
        tiny_small_instance_ids=(10,),
        **arrays,
    )


def _ply_colors(path: Path) -> list[tuple[int, int, int]]:
    lines = path.read_text(encoding="ascii").splitlines()
    end = lines.index("end_header")
    return [tuple(map(int, line.split()[-3:])) for line in lines[end + 1 :]]


def test_audit_counts_unmapped_gaussians_as_false_positive() -> None:
    audit = _audit()
    row = next(
        value
        for value in audit["objects"]
        if value["condition"] == "C0-no-prior" and value["instance_id"] == 0
    )
    assert row["predicted_gaussian_count"] == 5
    assert row["correct_gaussian_count"] == 2
    assert row["same_class_wrong_instance_count"] == 1
    assert row["wrong_class_count"] == 1
    assert row["unsupported_count"] == 1
    assert row["point_precision"] == pytest.approx(0.4)
    assert row["class_purity"] == pytest.approx(0.6)
    assert row["unsupported_fraction"] == pytest.approx(0.2)
    assert row["dominant_gt_instance"] == 10
    assert row["merge_candidate"] is True
    assert row["is_tiny_small"] is True
    assert row["matched_at_025"] is True


def test_audit_marks_pure_fp_split_and_condition_comparisons() -> None:
    audit = _audit()
    c0 = [row for row in audit["objects"] if row["condition"] == "C0-no-prior"]
    pure_fp = next(row for row in c0 if row["instance_id"] == 1)
    duplicate_group = [row for row in c0 if row["dominant_gt_instance"] == 10]
    assert pure_fp["pure_false_positive"] is True
    assert pure_fp["point_precision"] == 0.0
    assert len(duplicate_group) == 2
    assert all(row["split_candidate"] for row in duplicate_group)
    assert sum(bool(row["duplicate_prediction"]) for row in duplicate_group) == 1

    rows = audit["condition_comparison_rows"]
    target_rows = [row for row in rows if row["gt_instance_id"] == 10]
    assert [row["condition"] for row in target_rows] == [
        "C0-no-prior",
        "U-global",
        "D-predicted",
    ]
    assert all(row["same_class_iou"] == pytest.approx(0.5) for row in target_rows)
    encoded = json.dumps(rows).lower()
    assert "psnr" not in encoded
    assert "ssim" not in encoded
    assert "2d" not in encoded


def test_selection_is_deterministic_and_emits_available_roles() -> None:
    rows = list(_audit()["objects"])
    tiny_failure = dict(rows[0])
    tiny_failure.update(
        {
            "condition": "D-predicted",
            "instance_id": 99,
            "same_class_iou": 0.0,
            "matched_at_025": False,
            "point_precision": 0.0,
            "is_tiny_small": True,
        }
    )
    rows.append(tiny_failure)
    forward = select_clean_viewer_cases(rows)
    reverse = select_clean_viewer_cases(list(reversed(rows)))
    assert [(row["role"], row["condition"], row["instance_id"]) for row in forward] == [
        (row["role"], row["condition"], row["instance_id"]) for row in reverse
    ]
    assert {row["role"] for row in forward} == {
        "highest_precision",
        "median_precision",
        "lowest_precision",
        "pure_false_positive",
        "tiny_small_success",
        "tiny_small_failure",
        "merge_case",
        "split_case",
    }


def test_export_writes_fixed_color_object_plys_and_metrics(tmp_path: Path) -> None:
    arrays = _arrays()
    result = build_clean_baseline_viewer(
        scene_id="scene-test",
        predictions=_predictions(),
        class_names=("chair", "table"),
        tiny_small_instance_ids=(10,),
        output_dir=tmp_path,
        **arrays,
    )
    assert result["contains_2d_render_metrics"] is False
    roles = {case["role"]: Path(case["directory"]) for case in result["cases"]}
    assert {
        "highest_precision",
        "median_precision",
        "lowest_precision",
        "pure_false_positive",
        "tiny_small_success",
        "merge_case",
        "split_case",
    }.issubset(roles)

    merge_dir = roles["merge_case"]
    assert (merge_dir / "predicted_gaussians.ply").is_file()
    assert (merge_dir / "matched_gt_points.ply").is_file()
    assert (merge_dir / "overlay.ply").is_file()
    assert (merge_dir / "metrics.json").is_file()
    colors = _ply_colors(merge_dir / "predicted_gaussians.ply")
    assert colors == [
        tuple(CORRECT_COLOR),
        tuple(CORRECT_COLOR),
        tuple(SAME_CLASS_WRONG_INSTANCE_COLOR),
        tuple(UNSUPPORTED_COLOR),
        tuple(WRONG_CLASS_COLOR),
    ]
    assert set(_ply_colors(merge_dir / "matched_gt_points.ply")) == {
        tuple(GT_COLOR)
    }
    overlay_colors = _ply_colors(merge_dir / "overlay.ply")
    assert overlay_colors[:5] == colors
    assert set(overlay_colors[5:]) == {tuple(GT_COLOR)}

    metrics = json.loads((merge_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["contains_2d_render_metrics"] is False
    assert metrics["color_legend"]["correct_gt_instance"] == CORRECT_COLOR.tolist()
    comparison = json.loads(
        (tmp_path / "scene-test" / "condition_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert comparison["conditions"] == [
        "C0-no-prior",
        "U-global",
        "D-predicted",
    ]
    assert comparison["contains_2d_render_metrics"] is False


def test_viewer_rejects_prediction_contract_mismatch() -> None:
    arrays = _arrays()
    bad = _prediction("C0-no-prior")
    del bad["instances"]["2"]
    with pytest.raises(ValueError, match="disagree"):
        audit_clean_viewer_scene(
            scene_id="scene-test",
            predictions={"C0-no-prior": bad},
            class_names=("chair", "table"),
            **arrays,
        )
