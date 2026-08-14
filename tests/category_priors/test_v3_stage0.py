from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from category_priors.io import load_json, read_rows, write_json, write_rows
from category_priors.taxonomy import load_taxonomy
from category_priors.v3_stage0 import (
    build_size_bin_spec,
    classify_physical_size,
    prepare_history_anchor,
    prepare_v3_stage0,
    select_diagnostic_scenes,
)


def test_stage0_cli_accepts_explicit_deployment_commit() -> None:
    from category_priors.cli import build_parser

    args = build_parser().parse_args(
        [
            "prepare-v3-stage0",
            "--locked-metrics", "locked.parquet",
            "--train-instance-stats", "train.parquet",
            "--tune-scene-manifest", "runtime.json",
            "--tune-gt-dir", "gt",
            "--history-output", "history.parquet",
            "--size-bins-output", "bins.json",
            "--diagnostic-scenes-output", "selection.json",
            "--git-commit", "abc123",
        ]
    )
    assert args.git_commit == "abc123"


def test_train_only_size_bins_ignore_invalid_and_nontrain_rows() -> None:
    rows = [
        {"split": "train", "canonical_class": "chair", "bbox_diag_m": value}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    rows.extend(
        [
            {"split": "val", "canonical_class": "chair", "bbox_diag_m": 100.0},
            {
                "split": "train",
                "canonical_class": "chair",
                "bbox_diag_m": 200.0,
                "quality_valid": False,
            },
            {"split": "train", "canonical_class": "not-saga20", "bbox_diag_m": 300.0},
        ]
    )
    spec = build_size_bin_spec(rows, ("chair", "table"))
    assert spec["training_instance_count"] == 4
    assert spec["boundaries_m"] == pytest.approx(
        {"tiny_max_m": 1.75, "small_max_m": 2.5, "medium_max_m": 3.25}
    )
    assert [classify_physical_size(value, spec) for value in (1.0, 2.0, 3.0, 4.0)] == [
        "tiny",
        "small",
        "medium",
        "large",
    ]
    assert spec["per_class_train"]["table"] == {"instance_count": 0}


def test_gt_only_selection_is_deterministic_and_physically_distinct() -> None:
    scenes = {
        "scene_a": {"physical_scene_id": "physical_a"},
        "scene_b": {"physical_scene_id": "physical_b"},
        "scene_c": {"physical_scene_id": "physical_c"},
        "scene_d": {"physical_scene_id": "physical_a"},
    }
    records = [
        {"scene_id": "scene_a", "canonical_class": "cup", "physical_size_bin": "tiny", "below_official_min_region_size": False},
        {"scene_id": "scene_b", "canonical_class": "phone", "physical_size_bin": "small", "below_official_min_region_size": False},
        {"scene_id": "scene_c", "canonical_class": "cup", "physical_size_bin": "small", "below_official_min_region_size": False},
        {"scene_id": "scene_c", "canonical_class": "chair", "physical_size_bin": "large", "below_official_min_region_size": False},
        {"scene_id": "scene_d", "canonical_class": "speaker", "physical_size_bin": "tiny", "below_official_min_region_size": True},
    ]
    result = select_diagnostic_scenes(records, scenes, budget=3, target_small_per_class=1)
    assert result["selected_scenes"] == ["scene_d", "scene_b", "scene_c"]
    selected_physical = {
        scenes[scene_id]["physical_scene_id"] for scene_id in result["selected_scenes"]
    }
    assert len(selected_physical) == 3
    assert result["selection_basis"] == "tune_gt_only"


def test_history_anchor_requires_complete_paired_three_seed_table() -> None:
    rows = [
        {
            "condition": condition,
            "run_seed": seed,
            "scene_count": 48,
            "protocol_version": "scannet-official-v1",
            "map_50_95": 0.05,
        }
        for condition in ("B0-legacy", "B1-other-classes")
        for seed in (42, 3407, 20260804)
    ]
    anchor = prepare_history_anchor(rows, git_commit="abc123")
    assert len(anchor) == 6
    assert {row["git_commit"] for row in anchor} == {"abc123"}
    with pytest.raises(ValueError, match="Expected exactly"):
        prepare_history_anchor(rows[:-1], git_commit="abc123")


def test_prepare_v3_stage0_writes_registered_outputs(tmp_path: Path) -> None:
    taxonomy = load_taxonomy()
    locked_metrics = tmp_path / "locked.parquet"
    write_rows(
        locked_metrics,
        [
            {
                "condition": condition,
                "run_seed": seed,
                "scene_count": 48,
                "protocol_version": "scannet-official-v1",
                "map_50_95": 0.05 + (condition == "B1-other-classes") * 0.001,
                "map_0.50": 0.1,
                "map_0.25": 0.2,
            }
            for condition in ("B0-legacy", "B1-other-classes")
            for seed in (42, 3407, 20260804)
        ],
    )
    train_stats = tmp_path / "train.parquet"
    write_rows(
        train_stats,
        [
            {"split": "train", "canonical_class": "chair", "bbox_diag_m": value}
            for value in (0.2, 0.4, 0.6, 0.8)
        ],
    )
    manifest = tmp_path / "runtime.json"
    write_json(
        manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "physical_scene_id": scene_id,
                    "base_path": str(tmp_path / scene_id),
                    "scene_scale_m_per_unit": 1.0,
                }
                for scene_id in ("scene0000_00", "scene0001_00")
            ],
        },
    )
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    for index, scene_id in enumerate(("scene0000_00", "scene0001_00")):
        coords = np.column_stack(
            (np.linspace(0.0, 0.3 + 0.4 * index, 120), np.zeros(120), np.zeros(120))
        )
        np.savez_compressed(
            gt_dir / f"{scene_id}.npz",
            coords=coords,
            semantic=np.zeros(120, dtype=np.int64),
            instance=np.zeros(120, dtype=np.int64),
        )
    history = tmp_path / "v3_history_anchor.parquet"
    bins = tmp_path / "v3_gt_size_bins.json"
    selection = tmp_path / "v3_diagnostic8_scenes.json"
    payload = prepare_v3_stage0(
        locked_metrics_path=locked_metrics,
        train_instance_stats_path=train_stats,
        tune_scene_manifest_path=manifest,
        tune_gt_dir=gt_dir,
        taxonomy=taxonomy,
        history_output=history,
        size_bins_output=bins,
        diagnostic_scenes_output=selection,
        diagnostic_budget=1,
        git_commit="abc123",
    )
    assert payload["status"] == "complete"
    assert len(read_rows(history)) == 6
    assert load_json(bins)["source_split"] == "train"
    selected = load_json(selection)
    assert len(selected["selected_scenes"]) == 1
    assert len(selected["remaining_scenes"]) == 1
    assert selected["selection_basis"] == "tune_gt_only"
