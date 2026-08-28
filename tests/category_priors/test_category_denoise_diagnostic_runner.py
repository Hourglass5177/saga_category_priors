from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.category_denoise import CandidateBank, save_candidate_bank
from category_priors.category_denoise_diagnostic_runner import (
    _normalize_scene_ids,
    _validate_knn_oracle_plan,
    replay_category_denoise_knn_oracle,
)
from category_priors.category_denoise_diagnostic_evaluation import _candidate_metrics
from category_priors.category_denoise_diagnostic_evaluation import (
    evaluate_category_denoise_knn_oracle,
)
from category_priors.category_denoise_knn_oracle import PLAN_SCHEMA
from category_priors.io import read_rows
from category_priors.taxonomy import load_taxonomy


SCENE = "scene0001_00"
CLASSES = (
    "chair",
    "table",
    "plant",
    "tv",
    "painting",
    "sofa",
    "cabinet",
    "bed",
    "socket",
    "book",
    "switch",
    "door",
    "window",
    "lamp",
    "speaker",
    "fan",
    "refrigerator",
    "cup",
    "phone",
    "trash can",
    *(f"extra-{index}" for index in range(12)),
)


def _write_ply(path: Path, xyz: np.ndarray) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
        *(" ".join(map(str, row)) for row in xyz.tolist()),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, ambiguous_b0: bool = False) -> tuple[Path, Path, Path]:
    point_count = 12
    xyz = np.asarray(
        [(index % 4, (index // 4) % 3, index / 100.0) for index in range(point_count)],
        dtype=np.float64,
    )
    gaussian = tmp_path / "scene" / "gaussian.ply"
    _write_ply(gaussian, xyz)
    manifest = tmp_path / "runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "scene_runtime_manifest",
                "scenes": [
                    {
                        "scene_id": SCENE,
                        "base_path": str(gaussian.parent),
                        "gaussian_ply": str(gaussian),
                        "scene_scale_m_per_unit": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    full = np.full(point_count, -1, dtype=np.int64)
    full[:3] = 0
    bank = CandidateBank(
        class_names=CLASSES,
        saga20_names=("chair",),
        scene_scale_m_per_unit=1.0,
        seed=42,
        global_pre_knn=np.zeros(point_count, dtype=np.int64),
        semantic_top1=np.zeros(point_count, dtype=np.int64),
        semantic_top1_score=np.ones(point_count, dtype=np.float64),
        branch_full_labels=full,
        branch_core_labels=full.copy(),
        assignment_confidence=np.where(full >= 0, 1.0, 0.0),
        candidates=(
            {
                "candidate_id": 0,
                "branch_class": "chair",
                "branch_class_index": 0,
                "full_point_count": 3,
                "core_point_count": 3,
                "base_score": 0.8,
            },
        ),
        diagnostics={"scene_id": SCENE},
    )
    bank_root = tmp_path / "bank"
    scene_root = bank_root / SCENE
    save_candidate_bank(bank, scene_root)
    if ambiguous_b0:
        labels = [0] * 6 + [1] * 6
        instances = {
            "0": {"class": "chair", "score": 0.5},
            "1": {"class": "table", "score": 0.5},
        }
    else:
        labels = [0] * point_count
        instances = {"0": {"class": "chair", "score": 0.5}}
    (scene_root / "output.json").write_text(
        json.dumps({"point_labels": labels, "instances": instances}), encoding="utf-8"
    )
    return manifest, bank_root, gaussian


def _plan(path: Path, *, target_instance: int = 7, candidates: bool = True) -> Path:
    selected = (
        [
            {
                "scene_id": SCENE,
                "candidate_id": 0,
                "matched_gt_class_id": 0,
                "matched_gt_instance_id": target_instance,
            }
        ]
        if candidates
        else []
    )
    payload = {
        "schema": PLAN_SCHEMA,
        "evaluation_only": True,
        "iou_threshold": 0.50,
        "radius_m": 0.05,
        "min_region_size": 100,
        "scene_ids": [SCENE],
        "candidate_count": len(selected),
        "scene_count_with_candidates": int(bool(selected)),
        "scenes": [
            {
                "schema": PLAN_SCHEMA,
                "evaluation_only": True,
                "scene_id": SCENE,
                "point_count": 12,
                "bank_schema": "saga-category-denoise-bank-v1",
                "bank_seed": 42,
                "iou_threshold": 0.50,
                "radius_m": 0.05,
                "min_region_size": 100,
                "candidates": selected,
            }
        ],
        "status": "complete",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scene_and_plan_validation_rejects_ambiguous_or_unsafe_inputs() -> None:
    scenes = {SCENE: {}}
    with pytest.raises(ValueError, match="at least one"):
        _normalize_scene_ids([], scenes)
    with pytest.raises(ValueError, match="duplicates"):
        _normalize_scene_ids([SCENE, SCENE], scenes)
    with pytest.raises(ValueError, match="invalid scene"):
        _normalize_scene_ids(["../scene0001_00"], {"../scene0001_00": {}})

    invalid = {
        "schema": PLAN_SCHEMA,
        "evaluation_only": True,
        "iou_threshold": 0.5,
        "radius_m": 0.05,
        "min_region_size": 100,
        "scene_ids": [],
        "candidate_count": 0,
        "scene_count_with_candidates": 0,
        "scenes": [],
    }
    with pytest.raises(ValueError, match="at least one"):
        _validate_knn_oracle_plan(invalid, scenes)


def test_replay_signature_and_artifacts_are_gt_free_and_plan_bound(tmp_path: Path) -> None:
    assert "gt_dir" not in inspect.signature(replay_category_denoise_knn_oracle).parameters
    manifest, bank_root, _ = _fixture(tmp_path)
    plan_path = _plan(tmp_path / "plan.json", target_instance=7)
    output = tmp_path / "replay"

    first = replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=plan_path,
        output_root=output,
    )
    assert first["gt_used_by_replay"] is False
    first_o2 = json.loads(
        (output / "O2-protected" / SCENE / "output.json").read_text(encoding="utf-8")
    )

    # Changing GT-derived labels cannot change mechanics, but must invalidate
    # the completion shortcut so the frozen evaluation selection is updated.
    _plan(plan_path, target_instance=999)
    second = replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=plan_path,
        output_root=output,
    )
    second_o2 = json.loads(
        (output / "O2-protected" / SCENE / "output.json").read_text(encoding="utf-8")
    )
    assert second["scenes"][0]["run_status"] == "complete"
    assert second["scenes"][0]["selection"][0]["matched_gt_instance_id"] == 999
    assert first_o2 == second_o2

    third = replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=plan_path,
        output_root=output,
    )
    assert third["scenes"][0]["run_status"] == "skipped_complete"


def test_mapping_failure_keeps_candidate_mechanics_without_fake_predictions(
    tmp_path: Path,
) -> None:
    manifest, bank_root, _ = _fixture(tmp_path, ambiguous_b0=True)
    result = replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=_plan(tmp_path / "plan.json"),
        output_root=tmp_path / "replay",
    )
    scene = result["scenes"][0]
    assert scene["full_prediction_available"] is False
    assert scene["mapping_error"]
    assert (tmp_path / "replay" / "scenes" / SCENE / "oracle_replay_labels.npz").is_file()
    assert not (tmp_path / "replay" / "O1-unprotected" / SCENE / "output.json").exists()

    gt = tmp_path / "gt"
    gt.mkdir()
    coords = np.asarray(
        [(index % 4, (index // 4) % 3, index / 100.0) for index in range(12)],
        dtype=np.float64,
    )
    np.savez_compressed(
        gt / f"{SCENE}.npz",
        coords=coords,
        semantic=np.zeros(12, dtype=np.int64),
        instance=np.full(12, 7, dtype=np.int64),
    )
    evaluation = tmp_path / "evaluation"
    analysis = evaluate_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        gt_dir=gt,
        prediction_root=tmp_path / "replay",
        oracle_plan=tmp_path / "plan.json",
        output_dir=evaluation,
        taxonomy=load_taxonomy(),
    )
    assert analysis["full_prediction_available"] is False
    metrics = {
        row["condition"]: row
        for row in read_rows(evaluation / "knn_oracle_metrics.parquet")
    }
    assert metrics["O1-unprotected"]["repository_map_50_95_10_thresholds"] is None
    assert metrics["O2-protected"]["AP50"] is None


def test_empty_oracle_selection_is_a_complete_deterministic_replay(tmp_path: Path) -> None:
    manifest, bank_root, _ = _fixture(tmp_path)
    result = replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=_plan(tmp_path / "empty.json", candidates=False),
        output_root=tmp_path / "empty-replay",
    )
    assert result["candidate_count"] == 0
    assert result["full_prediction_scene_count"] == 1
    assert result["scenes"][0]["candidate_raw_labels"] == {
        "O1-unprotected": {},
        "O2-protected": {},
    }


def test_gaussian_precision_counts_void_gt_as_unsupported() -> None:
    metrics = _candidate_metrics(
        candidate_mask=np.asarray([True, True]),
        raw_stage_labels=np.asarray([0, 0]),
        raw_label=0,
        gaussian_xyz_metric=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        gt_xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        gt_semantic=np.asarray([0, -1]),
        gt_instance=np.asarray([3, -1]),
        matched_class_id=0,
        matched_instance_id=3,
        radius_m=0.05,
        gaussian_to_gt_indices=np.asarray([0, 1]),
        class_count=20,
    )
    assert metrics["gaussian_target_precision"] == 0.5
    assert metrics["gaussian_supported_purity"] == 1.0
    assert metrics["gaussian_unsupported_fraction"] == 0.5


def test_evaluation_writes_consistent_schema_and_skips_complete_output(
    tmp_path: Path,
) -> None:
    manifest, bank_root, _ = _fixture(tmp_path)
    plan = _plan(tmp_path / "empty.json", candidates=False)
    replay = tmp_path / "replay"
    replay_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        bank_root=bank_root,
        b0_root=bank_root,
        oracle_plan=plan,
        output_root=replay,
    )
    coords = np.asarray(
        [(index % 4, (index // 4) % 3, index / 100.0) for index in range(12)],
        dtype=np.float64,
    )
    gt = tmp_path / "gt"
    gt.mkdir()
    np.savez_compressed(
        gt / f"{SCENE}.npz",
        coords=coords,
        semantic=np.zeros(12, dtype=np.int64),
        instance=np.zeros(12, dtype=np.int64),
    )
    output = tmp_path / "evaluation"
    first = evaluate_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        gt_dir=gt,
        prediction_root=replay,
        oracle_plan=plan,
        output_dir=output,
        taxonomy=load_taxonomy(),
    )
    assert first["status"] == "complete"
    rows = read_rows(output / "knn_oracle_metrics.parquet")
    assert {row["condition"] for row in rows} == {
        "B0",
        "O1-unprotected",
        "O2-protected",
    }
    assert all("repository_map_50_95_10_thresholds" in row for row in rows)
    second = evaluate_category_denoise_knn_oracle(
        runtime_manifest=manifest,
        gt_dir=gt,
        prediction_root=replay,
        oracle_plan=plan,
        output_dir=output,
        taxonomy=load_taxonomy(),
    )
    assert second["status"] == "skipped_complete"
