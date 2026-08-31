from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline import cli
from category_priors.clean_baseline import evaluation as clean_evaluation
from category_priors.clean_baseline.evaluation import (
    RUN_IDENTITY_SCHEMA,
    CleanCandidate,
    GroundTruthObject,
    build_prediction_payload,
    evaluate_candidates,
    evaluate_clean_baseline_manifest,
    evaluate_geometry_oracles,
    evaluate_ground_truth_parity,
    evaluate_prediction_payload,
    evaluation_is_complete,
    prediction_is_complete,
    project_gaussian_support_to_gt_points,
)
from category_priors.evaluator import GroundTruthScene
from category_priors.io import hash_json, write_json


def _run_identity(scene_id: str, condition: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": RUN_IDENTITY_SCHEMA,
        "consumer_commit": "test",
        "scene_id": scene_id,
        "condition": condition,
    }
    value["content_sha256"] = hash_json(value)
    return value


def _candidate(
    object_id: int,
    gaussian_ids: list[int],
    class_id: str = "chair",
) -> CleanCandidate:
    return CleanCandidate(
        object_id=object_id,
        gaussian_ids=np.asarray(gaussian_ids),
        class_id=class_id,
        winner_probability=1.0,
        view_consensus=1.0,
        detection_ratio=1.0,
    )


def test_geometry_oracles_keep_full_masks_and_monotonically_associate() -> None:
    mapping = np.arange(12, dtype=np.int64)
    gt = [
        GroundTruthObject(
            object_id=9,
            class_id="chair",
            point_ids=np.arange(6),
            official_valid=True,
            is_tiny_small=True,
        )
    ]
    result = evaluate_geometry_oracles(
        [np.asarray([0, 1, 2]), np.asarray([3, 4, 5]), np.arange(6, 12)],
        gt,
        mapping,
        mask_ids=["left", "right", "noise"],
        gaussian_count=12,
    )

    row = result["per_gt"][0]
    assert row["best_single"] == pytest.approx(0.5)
    assert row["perfect_association"] == pytest.approx(1.0)
    assert row["perfect_association_mask_ids"] == ["left", "right"]
    assert row["perfect_trim"] == pytest.approx(1.0)
    assert result["aggregate"]["tiny_small_official_valid"][
        "perfect_association"
    ]["recall_050"] == 1.0


def test_perfect_association_uses_at_most_one_mask_per_physical_view() -> None:
    result = evaluate_geometry_oracles(
        [np.asarray([0, 1]), np.asarray([2, 3])],
        [GroundTruthObject(1, "chair", np.arange(4))],
        np.arange(4, dtype=np.int64),
        mask_ids=[10, 11],
        mask_frame_ids=[0, 0],
        gaussian_count=4,
    )

    row = result["per_gt"][0]
    assert row["best_single"] == pytest.approx(0.5)
    assert row["perfect_association"] == pytest.approx(0.5)
    assert len(row["perfect_association_mask_ids"]) == 1


def test_perfect_trim_removes_false_positive_point_support() -> None:
    mapping = np.arange(6, dtype=np.int64)
    gt = [GroundTruthObject(1, "chair", np.arange(4))]
    result = evaluate_geometry_oracles(
        [np.asarray([0, 1, 4, 5])],
        gt,
        mapping,
        gaussian_count=6,
    )

    row = result["per_gt"][0]
    assert row["best_single"] == pytest.approx(2 / 6)
    assert row["perfect_association"] == pytest.approx(2 / 6)
    assert row["perfect_trim"] == pytest.approx(0.5)


def test_unmapped_gaussian_is_an_explicit_candidate_false_positive() -> None:
    result = evaluate_candidates(
        [_candidate(1, [0, 1])],
        [GroundTruthObject(1, "chair", np.asarray([0]))],
        np.asarray([0]),
        gaussian_count=2,
    )
    row = result["candidate_rows"][0]
    assert row["best_same_class_iou"] == pytest.approx(0.5)
    assert row["official_point_count"] == 1
    assert row["unsupported_gaussian_count"] == 1


def test_candidate_metrics_separate_geometry_from_same_class() -> None:
    mapping = np.arange(8, dtype=np.int64)
    ground_truth = [
        GroundTruthObject(1, "chair", np.arange(4), is_tiny_small=True),
        GroundTruthObject(2, "table", np.arange(4, 8)),
    ]
    result = evaluate_candidates(
        [
            _candidate(10, [0, 1, 2, 3], "table"),
            _candidate(11, [4, 5, 6, 7], "table"),
        ],
        ground_truth,
        mapping,
        gaussian_count=8,
    )

    rows = result["candidate_rows"]
    assert rows[0]["best_geometry_iou"] == 1.0
    assert rows[0]["best_same_class_iou"] == 0.0
    assert rows[1]["best_same_class_iou"] == 1.0
    aggregate = result["aggregate"]
    assert aggregate["geometry_iou_050_count"] == 2
    assert aggregate["same_class_iou_050_count"] == 1
    assert aggregate["candidate_precision_025"] == pytest.approx(0.5)
    assert aggregate["tiny_small_recall_025"] == 0.0


def test_candidate_adapter_accepts_consensus_style_attributes() -> None:
    class ClassifiedConsensusObject:
        object_id = 3
        gaussian_ids = np.asarray([0, 1])
        class_id = "chair"
        winner_probability = 0.8
        mean_view_consensus = 0.9
        mean_detection_ratio = 0.7

    result = evaluate_candidates(
        [ClassifiedConsensusObject()],
        [GroundTruthObject(1, "chair", np.asarray([0, 1]))],
        np.asarray([0, 1]),
    )
    assert result["aggregate"]["same_class_iou_050_count"] == 1

    with pytest.raises(ValueError, match="duplicate candidate"):
        evaluate_candidates(
            [_candidate(3, [0]), _candidate(3, [1])],
            [GroundTruthObject(1, "chair", np.asarray([0, 1]))],
            np.asarray([0, 1]),
        )


def test_projection_infers_unseen_gaussian_id_without_explicit_count() -> None:
    projected = project_gaussian_support_to_gt_points(
        np.asarray([5]), np.asarray([5, -1])
    )
    assert projected.tolist() == [0]


def test_strict_export_rejects_oracle_and_overlapping_ownership() -> None:
    with pytest.raises(ValueError, match="evaluation-only"):
        build_prediction_payload(
            scene_id="scene",
            condition="D-oracle-class",
            gaussian_count=4,
            candidates=[_candidate(1, [0, 1])],
            allowed_classes=["chair"],
            run_identity=_run_identity("scene", "D-oracle-class"),
        )
    with pytest.raises(ValueError, match="overlap"):
        build_prediction_payload(
            scene_id="scene",
            condition="U-global",
            gaussian_count=4,
            candidates=[_candidate(1, [0, 1]), _candidate(2, [1, 2])],
            allowed_classes=["chair"],
            run_identity=_run_identity("scene", "U-global"),
        )


def test_prediction_contract_and_official_evaluator_share_one_projection(
    tmp_path: Path,
) -> None:
    payload, diagnostics = build_prediction_payload(
        scene_id="scene0000_00",
        condition="C0-no-prior",
        gaussian_count=3,
        candidates=[_candidate(7, [0, 1, 2])],
        allowed_classes=["chair"],
        run_identity=_run_identity("scene0000_00", "C0-no-prior"),
    )
    output = tmp_path / "output.json"
    write_json(output, payload)
    assert diagnostics["contract"]["orphan_gaussian_count"] == 0
    assert prediction_is_complete(
        output,
        expected_scene_id="scene0000_00",
        expected_condition="C0-no-prior",
        expected_gaussian_count=3,
    )

    metrics = evaluate_prediction_payload(
        scene_id="scene0000_00",
        payload=payload,
        gt_semantic=np.asarray([0, 0, 0]),
        gt_instance=np.asarray([4, 4, 4]),
        gt_point_to_gaussian=np.asarray([0, 1, 2]),
        class_names=["chair"],
        min_region_size=1,
    )
    assert metrics["aggregate"]["map_50_95"] == pytest.approx(1.0)

    payload["point_labels"].append(-1)
    write_json(output, payload)
    assert not prediction_is_complete(output, expected_gaussian_count=3)


def test_gt_as_prediction_closes_official_evaluator_protocol() -> None:
    scene = GroundTruthScene(
        scene_id="scene0000_00",
        semantic=np.asarray([0, 0, 1, 1]),
        instance=np.asarray([4, 4, 8, 8]),
    )
    result = evaluate_ground_truth_parity(
        [scene], ["chair", "table"], min_region_size=1
    )
    assert result["gt_as_prediction_parity"] is True
    assert result["aggregate"]["map_50_95"] == pytest.approx(1.0)


def test_cli_exposes_only_formal_runtime_conditions_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "run"
    calls: list[str] = []

    def fake_runner(**kwargs: object) -> None:
        calls.append(str(kwargs["condition"]))
        payload, diagnostics = build_prediction_payload(
            scene_id=str(kwargs["scene_id"]),
            condition=str(kwargs["condition"]),
            gaussian_count=3,
            candidates=[_candidate(0, [0, 1])],
            allowed_classes=["chair"],
            run_identity=_run_identity(
                str(kwargs["scene_id"]), str(kwargs["condition"])
            ),
        )
        target = Path(kwargs["output_dir"])
        write_json(target / "output.json", payload)
        write_json(target / "diagnostics.json", diagnostics)

    monkeypatch.setattr(cli, "_resolve_callable", lambda _: fake_runner)
    argv = [
        "run-mask-consensus",
        "--scene-id",
        "scene0000_00",
        "--bank-dir",
        str(tmp_path / "bank"),
        "--output-dir",
        str(output_dir),
        "--condition",
        "C0-no-prior",
        "--gaussian-count",
        "3",
    ]
    args = cli.build_parser().parse_args(argv)
    first = args.handler(args)
    second = args.handler(args)
    assert first["status"] == "complete"
    assert second["status"] == "complete"
    assert calls == ["C0-no-prior", "C0-no-prior"]

    oracle_args = cli.build_parser().parse_args(
        [*argv[:-3], "D-oracle-class", *argv[-2:]]
    )
    with pytest.raises(ValueError, match="unregistered"):
        oracle_args.handler(oracle_args)


def test_evaluation_completeness_never_accepts_oracle_in_formal_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evaluation.json"
    evaluation_identity: dict[str, object] = {
        "schema": clean_evaluation.EVALUATION_IDENTITY_SCHEMA,
        "test": True,
    }
    evaluation_identity["content_sha256"] = hash_json(evaluation_identity)
    write_json(
        path,
        {
            "schema": clean_evaluation.CLEAN_EVALUATION_SCHEMA,
            "scene_ids": ["scene"],
            "conditions": ["C0-no-prior"],
            "metrics": {"C0-no-prior": {}},
            "oracle_class_in_formal_metrics": False,
            "evaluation_identity": evaluation_identity,
        },
    )
    assert evaluation_is_complete(
        path,
        expected_scene_ids=["scene"],
        expected_conditions=["C0-no-prior"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["conditions"] = ["D-oracle-class"]
    payload["metrics"] = {"D-oracle-class": {}}
    write_json(path, payload)
    assert not evaluation_is_complete(path)


def test_manifest_enforces_coordinate_alignment_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    write_json(
        manifest,
        {
            "kind": "clean_baseline_evaluation_manifest",
            "minimum_mapped_fraction": 0.9,
            "conditions": ["C0-no-prior"],
            "scenes": [
                {
                    "scene_id": "scene0000_00",
                    "gt_npz": "gt.npz",
                    "gaussian_ply": "gaussians.ply",
                    "gaussian_to_gt_transform": np.eye(4).tolist(),
                    "outputs": {"C0-no-prior": "output.json"},
                }
            ],
        },
    )
    scene = GroundTruthScene(
        scene_id="scene0000_00",
        semantic=np.asarray([0]),
        instance=np.asarray([1]),
    )
    monkeypatch.setattr(
        clean_evaluation,
        "load_ground_truth_npz",
        lambda *_: (np.zeros((1, 3)), scene),
    )
    monkeypatch.setattr(
        clean_evaluation,
        "load_ply_xyz",
        lambda *_: np.zeros((1, 3)),
    )
    monkeypatch.setattr(
        clean_evaluation,
        "gt_point_to_gaussian_mapping",
        lambda *_args, **_kwargs: (
            np.asarray([-1]),
            {
                "mapped_fraction": 0.5,
                "median_nn_distance_m": 0.01,
                "p95_nn_distance_m": 0.01,
            },
        ),
    )
    with pytest.raises(ValueError, match="coordinate alignment gate"):
        evaluate_clean_baseline_manifest(manifest, class_names=["chair"])
