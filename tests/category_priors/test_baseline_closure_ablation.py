from __future__ import annotations

import json
from pathlib import Path

import pytest

from category_priors.baseline_closure import RuntimeScene
from category_priors.baseline_closure_ablation import (
    build_fixed_invocation,
    build_harness_invocation,
    canonical_point_partition,
    compare_partitions,
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
