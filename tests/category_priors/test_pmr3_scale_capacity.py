from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors import pmr3_scale_capacity as pmr3


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _object_rows() -> list[dict]:
    rows: list[dict] = []
    for scene_id in pmr3.SCENES:
        for index in range(pmr3.EXPECTED_OBJECT_COUNTS[scene_id]):
            rows.append(
                {
                    "scene_id": scene_id,
                    "prompt_id": f"{scene_id}-p{index:02d}",
                    "grid_delta_iou": 0.0,
                    "grid_delta_precision": 0.0,
                }
            )
    return rows


def _analysis_payload(
    checkpoint: str,
    *,
    capacity: float,
    scene_capacities: tuple[float, float],
    object_positive_fraction: float,
    precision_delta: float,
) -> dict:
    rows = _object_rows()
    positive_count = round(object_positive_fraction * len(rows))
    for index, row in enumerate(rows):
        row["grid_delta_iou"] = 0.01 if index < positive_count else 0.0
        row["grid_delta_precision"] = precision_delta
    return {
        "kind": "pmr3_scale_checkpoint_analysis",
        "status": "complete",
        "checkpoint": checkpoint,
        "scene_equal_grid_delta_iou": capacity,
        "scene_equal_grid_delta_precision": precision_delta,
        "object_fraction_at_least_0p02": object_positive_fraction,
        "scene_results": [
            {
                "scene_id": scene_id,
                "grid_mean_delta_iou": value,
            }
            for scene_id, value in zip(pmr3.SCENES, scene_capacities, strict=True)
        ],
        "object_results": rows,
    }


def _analyze_pair(
    tmp_path: Path,
    native: dict,
    tenk: dict,
) -> dict:
    native_path = tmp_path / "native.json"
    tenk_path = tmp_path / "tenk.json"
    output_path = tmp_path / "comparison.json"
    _write_json(native_path, native)
    _write_json(tenk_path, tenk)
    return pmr3.analyze_pair(
        native_analysis=native_path,
        tenk_analysis=tenk_path,
        output=output_path,
    )


def test_training_command_freezes_one_10k_seed_zero_trajectory_without_gt(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "trajectory"
    command = pmr3.build_training_command(
        workspace=tmp_path / "workspace",
        python_bin=tmp_path / "python",
        scene_base=tmp_path / "scene",
        trajectory_root=trajectory,
    )

    assert command.count("--feature-iterations") == 1
    assert _option(command, "--feature-iterations") == "10000"
    assert command.count("--feature-seed") == 1
    assert _option(command, "--feature-seed") == "0"
    assert Path(_option(command, "--feature-snapshot-root")) == trajectory
    assert Path(_option(command, "--contrastive-feature-point-cloud-path")) == (
        trajectory / "iteration_10000" / "contrastive_feature_point_cloud.ply"
    )
    assert Path(_option(command, "--scale-gate-path")) == (
        trajectory / "iteration_10000" / "scale_gate.pt"
    )
    assert Path(_option(command, "--progress-path")) == trajectory / "progress"
    assert "--gt-dir" not in command
    assert not any("ground_truth" in token.lower() for token in command)


def test_training_and_segmentation_interfaces_cannot_receive_gt() -> None:
    assert "gt_dir" not in inspect.signature(pmr3.run_training_trajectories).parameters
    assert "gt_dir" not in inspect.signature(pmr3.segment_checkpoint).parameters
    assert "gt_dir" in inspect.signature(pmr3.evaluate_checkpoint).parameters

    parser = pmr3._parser()
    train_args = parser.parse_args(
        [
            "train",
            "--workspace",
            "workspace",
            "--python",
            "python",
            "--runtime-manifest",
            "scenes.json",
            "--training-root",
            "trajectory",
            "--scene",
            pmr3.SCENES[0],
        ]
    )
    segment_args = parser.parse_args(
        [
            "segment",
            "--runtime-manifest",
            "scenes.json",
            "--prompts-root",
            "prompts",
            "--parameters",
            "parameters.json",
            "--training-root",
            "trajectory",
            "--output-root",
            "conditions",
            "--scene",
            pmr3.SCENES[0],
            "--checkpoint",
            "native",
        ]
    )
    assert not hasattr(train_args, "gt_dir")
    assert not hasattr(segment_args, "gt_dir")


def test_analyze_pair_requires_exactly_the_same_34_registered_objects(
    tmp_path: Path,
) -> None:
    native = _analysis_payload(
        "native",
        capacity=0.0,
        scene_capacities=(0.0, 0.0),
        object_positive_fraction=0.0,
        precision_delta=0.0,
    )
    tenk = _analysis_payload(
        "10k",
        capacity=0.03,
        scene_capacities=(0.02, 0.02),
        object_positive_fraction=0.30,
        precision_delta=0.0,
    )

    result = _analyze_pair(tmp_path / "valid", native, tenk)
    assert result["object_count"] == 34
    assert sum(pmr3.EXPECTED_OBJECT_COUNTS.values()) == 34

    missing = json.loads(json.dumps(tenk))
    missing["object_results"].pop()
    with pytest.raises(ValueError, match="pair|object|34"):
        _analyze_pair(tmp_path / "missing", native, missing)

    renamed = json.loads(json.dumps(tenk))
    renamed["object_results"][0]["prompt_id"] = "unregistered-prompt"
    with pytest.raises(ValueError, match="pair|object|34"):
        _analyze_pair(tmp_path / "renamed", native, renamed)


def test_checkpoint_evaluation_uses_equal_scene_weight_not_object_pooling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_payloads: dict[str, dict[str, dict]] = {}
    evaluation_payloads: dict[str, dict[str, dict]] = {}
    for scene_id in pmr3.SCENES:
        count = pmr3.EXPECTED_OBJECT_COUNTS[scene_id]
        scene_payloads[scene_id] = {
            f"{scene_id}-p{index:02d}": {
                "prompt_id": f"{scene_id}-p{index:02d}",
                "class_name": "chair",
                "image_name": f"{index:06d}.jpg",
            }
            for index in range(count)
        }
        evaluation_payloads[scene_id] = {
            f"{scene_id}-p{index:02d}": {
                "prompt_id": f"{scene_id}-p{index:02d}",
                "class_id": 1,
                "gt_instance_id": index + 1,
                "class_name": "chair",
                "bbox_diagonal_m": 0.5,
            }
            for index in range(count)
        }

    monkeypatch.setattr(
        pmr3, "load_scene_runtime_manifest", lambda _path: {scene: {} for scene in pmr3.SCENES}
    )
    monkeypatch.setattr(pmr3, "_runtime_prompts", lambda _root, scene: scene_payloads[scene])
    monkeypatch.setattr(
        pmr3,
        "_evaluation_prompts",
        lambda _root, scene: evaluation_payloads[scene],
    )
    monkeypatch.setattr(
        pmr3,
        "_checkpoint_assets",
        lambda _scene, _root, _checkpoint: (
            {
                "feature_ply": tmp_path / "feature.ply",
                "point_cloud": tmp_path / "scene.ply",
                "scale_gate": tmp_path / "scale_gate.pt",
            },
            10000,
        ),
    )
    monkeypatch.setattr(pmr3, "load_ply_xyz", lambda _path: np.zeros((1, 3), dtype=np.float32))
    monkeypatch.setattr(
        pmr3,
        "load_ground_truth_npz",
        lambda _path, _scene_id: (
            np.zeros((1, 3), dtype=np.float32),
            SimpleNamespace(
                semantic=np.ones(1, dtype=np.int32),
                instance=np.ones(1, dtype=np.int32),
            ),
        ),
    )
    monkeypatch.setattr(
        pmr3,
        "_nearest",
        lambda _src, _dst: (np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.float32)),
    )
    monkeypatch.setattr(pmr3, "_transform", lambda _scene: np.eye(4, dtype=np.float64))
    monkeypatch.setattr(pmr3, "apply_transform", lambda xyz, _matrix: xyz)
    monkeypatch.setattr(
        pmr3,
        "_gt_path",
        lambda _scene, _gt_dir, _scene_id: tmp_path / "gt.npz",
    )
    monkeypatch.setattr(pmr3, "_result_complete", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pmr3, "write_rows", lambda *_args, **_kwargs: None)

    def fake_metrics(mask: np.ndarray, **_kwargs: object) -> dict:
        value = 0.04 if bool(mask[0]) else 0.0
        return {
            "iou": value,
            "gaussian_precision": value,
            "gt_recall": value,
        }

    monkeypatch.setattr(pmr3, "evaluate_prompt_pair_arrays", fake_metrics)

    masks_root = tmp_path / "conditions"
    for scene_id, prompts in scene_payloads.items():
        beneficial_scene = scene_id == pmr3.SCENES[0]
        for prompt in prompts.values():
            prompt_root = masks_root / "10k" / scene_id
            prompt_root.mkdir(parents=True, exist_ok=True)
            target = prompt_root / f"{prompt['prompt_id']}.npz"
            arrays = {"U_global": np.zeros(1, dtype=bool)}
            arrays.update(
                {
                    pmr3.scale_key(scale): np.full(1, beneficial_scene, dtype=bool)
                    for scale in pmr3.GRID9
                }
            )
            np.savez_compressed(target, **arrays)
            _write_json(
                target.with_suffix(".json"),
                {
                    "uniform_scale_input": 0.5,
                },
            )

    analysis = pmr3.evaluate_checkpoint(
        runtime_manifest=tmp_path / "scenes.json",
        prompts_root=tmp_path / "prompts",
        gt_dir=tmp_path / "gt",
        training_root=tmp_path / "trajectory",
        masks_root=masks_root,
        scene_ids=pmr3.SCENES,
        checkpoint="10k",
        table_output=tmp_path / "metrics.parquet",
        analysis_output=tmp_path / "analysis.json",
        size_bins=None,
    )

    pooled_object_mean = (
        pmr3.EXPECTED_OBJECT_COUNTS[pmr3.SCENES[0]] * 0.04
    ) / sum(pmr3.EXPECTED_OBJECT_COUNTS.values())
    assert analysis["scene_equal_grid_delta_iou"] == pytest.approx(0.02)
    assert analysis["scene_equal_grid_delta_iou"] != pytest.approx(pooled_object_mean)


@pytest.mark.parametrize(
    "override",
    [
        {"tenk_capacity": 0.02, "native_capacity": 0.0},
        {"scene_capacities": (0.01, 0.01)},
        {"object_positive_fraction": 0.25},
        {"tenk_capacity": 0.02, "native_capacity": 0.01},
        {"precision_delta": -0.01},
    ],
    ids=[
        "capacity-at-0.02",
        "both-scenes-at-0.01",
        "positive-fraction-at-0.25",
        "capacity-increment-at-0.01",
        "precision-loss-at-minus-0.01",
    ],
)
def test_all_five_preregistered_gate_boundaries_are_inclusive(
    tmp_path: Path,
    override: dict,
) -> None:
    values = {
        "tenk_capacity": 0.03,
        "native_capacity": 0.0,
        "scene_capacities": (0.02, 0.02),
        "object_positive_fraction": 0.30,
        "precision_delta": 0.0,
    }
    values.update(override)
    native = _analysis_payload(
        "native",
        capacity=values["native_capacity"],
        scene_capacities=(0.0, 0.0),
        object_positive_fraction=0.0,
        precision_delta=0.0,
    )
    tenk = _analysis_payload(
        "10k",
        capacity=values["tenk_capacity"],
        scene_capacities=values["scene_capacities"],
        object_positive_fraction=values["object_positive_fraction"],
        precision_delta=values["precision_delta"],
    )
    assert _analyze_pair(tmp_path, native, tenk)["passed"] is True


@pytest.mark.parametrize(
    ("override", "failed_check"),
    [
        (
            {"tenk_capacity": 0.019999},
            "tenk_scene_equal_capacity_at_least_0p02",
        ),
        (
            {"scene_capacities": (0.009999, 0.02)},
            "both_scenes_capacity_at_least_0p01",
        ),
        (
            {"object_positive_fraction": 0.249999},
            "object_fraction_at_least_25pct",
        ),
        (
            {"tenk_capacity": 0.03, "native_capacity": 0.020001},
            "tenk_minus_native_capacity_at_least_0p01",
        ),
        (
            {"precision_delta": -0.010001},
            "tenk_precision_loss_no_more_than_1pp",
        ),
    ],
    ids=[
        "capacity",
        "per-scene",
        "positive-fraction",
        "native-to-10k-gain",
        "precision-guardrail",
    ],
)
def test_each_preregistered_gate_fails_independently_below_its_boundary(
    tmp_path: Path,
    override: dict,
    failed_check: str,
) -> None:
    values = {
        "tenk_capacity": 0.03,
        "native_capacity": 0.0,
        "scene_capacities": (0.02, 0.02),
        "object_positive_fraction": 0.30,
        "precision_delta": 0.0,
    }
    values.update(override)
    native = _analysis_payload(
        "native",
        capacity=values["native_capacity"],
        scene_capacities=(0.0, 0.0),
        object_positive_fraction=0.0,
        precision_delta=0.0,
    )
    tenk = _analysis_payload(
        "10k",
        capacity=values["tenk_capacity"],
        scene_capacities=values["scene_capacities"],
        object_positive_fraction=values["object_positive_fraction"],
        precision_delta=values["precision_delta"],
    )
    result = _analyze_pair(tmp_path, native, tenk)
    assert result["passed"] is False
    assert result["checks"][failed_check] is False
    assert sum(not passed for passed in result["checks"].values()) == 1


def test_native_raw_snapshot_is_materialized_once_then_marked_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = tmp_path / "trajectory"
    native_root = trajectory / "iteration_native_2000"
    final_root = trajectory / "iteration_10000"
    native_root.mkdir(parents=True)
    final_root.mkdir(parents=True)
    raw_feature = native_root / "contrastive_feature_point_cloud.raw.ply"
    gate = native_root / "scale_gate.pt"
    raw_feature.write_bytes(b"raw-feature")
    gate.write_bytes(b"gate")
    _write_json(
        native_root / "snapshot.json",
        {
            "kind": "pmr3_scale_training_snapshot",
            "status": "raw_complete",
            "iteration": 2000,
            "seed": 0,
            "smooth_k": 16,
            "raw_feature_ply": str(raw_feature),
            "scale_gate": str(gate),
        },
    )
    _write_json(
        trajectory / "training_manifest.json",
        {
            "kind": "pmr3_scale_training_trajectory",
            "status": "training_complete",
            "seed": 0,
            "native_iteration": 2000,
            "final_iteration": 10000,
            "num_sampled_rays": 1000,
            "train_camera_count": 200,
            "smooth_k": 16,
            "native_snapshot": str(native_root),
            "final_snapshot": str(final_root),
        },
    )

    calls: list[dict] = []

    class FakeFeatureModel:
        def save_ply(self, path: str, **kwargs: object) -> None:
            calls.append({"path": path, **kwargs})
            Path(path).write_bytes(b"materialized-feature")

    monkeypatch.setattr(pmr3, "_feature_model", lambda path: FakeFeatureModel())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: calls.append({"empty_cache": True}))),
    )

    result = pmr3.materialize_native_snapshot(trajectory)
    feature = native_root / "contrastive_feature_point_cloud.ply"
    metadata = json.loads((native_root / "snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((trajectory / "training_manifest.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert feature.read_bytes() == b"materialized-feature"
    assert calls[0]["smooth_weights"] is None
    assert calls[0]["smooth_type"] == "traditional"
    assert calls[0]["smooth_K"] == 16
    assert metadata["status"] == "complete"
    assert Path(metadata["feature_ply"]) == feature
    assert Path(metadata["raw_feature_ply"]) == raw_feature
    assert manifest["status"] == "complete"
    assert manifest["native_materialization"] == "post-training-traditional-knn16"


def test_complete_native_snapshot_is_reused_without_rematerialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = tmp_path / "trajectory"
    native_root = trajectory / "iteration_native_2000"
    native_root.mkdir(parents=True)
    feature = native_root / "contrastive_feature_point_cloud.ply"
    gate = native_root / "scale_gate.pt"
    feature.write_bytes(b"feature")
    gate.write_bytes(b"gate")
    _write_json(
        native_root / "snapshot.json",
        {
            "kind": "pmr3_scale_training_snapshot",
            "status": "complete",
            "iteration": 2000,
            "seed": 0,
            "smooth_k": 16,
            "feature_ply": str(feature),
            "scale_gate": str(gate),
        },
    )
    _write_json(
        trajectory / "training_manifest.json",
        {
            "kind": "pmr3_scale_training_trajectory",
            "status": "complete",
            "seed": 0,
            "native_iteration": 2000,
            "final_iteration": 10000,
            "num_sampled_rays": 1000,
            "train_camera_count": 200,
            "smooth_k": 16,
            "native_snapshot": str(native_root),
        },
    )
    monkeypatch.setattr(
        pmr3,
        "_feature_model",
        lambda _path: pytest.fail("a complete native snapshot must be reused"),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )

    result = pmr3.materialize_native_snapshot(trajectory)
    assert result == {
        "status": "reused",
        "iteration": 2000,
        "feature_ply": str(feature.resolve()),
        "scale_gate": str(gate.resolve()),
    }


def test_trainer_manifest_hands_off_training_complete_not_final_complete() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "train_contrastive_feature.py"
    ).read_text(encoding="utf-8")
    match = re.search(
        r'training_manifest\.json"\s*\),\s*\{.*?"status"\s*:\s*"([^"]+)"',
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "trainer must write the PMR-3 trajectory manifest"
    assert match.group(1) == "training_complete", (
        "the trainer only produces the raw native checkpoint; the PMR-3 materializer "
        "is the sole state transition from training_complete to complete"
    )
