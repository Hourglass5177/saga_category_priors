from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import category_priors.full_instance_size_experiment as experiment_module
from category_priors.full_instance_size_experiment import (
    DEV8,
    HOLDOUT5,
    ExperimentConfig,
    REGISTERED_HISTORICAL_T1_PRODUCER,
    _condition_output_paths,
    _condition_summary,
    _endpoint_gate,
    _final_gate,
    _holdout_gate,
    _load_t1_paths,
    _materialize_condition_output,
    _normalized_t1_command,
    _pooled_official_condition_summary,
    _resource_checkpoint,
    _registered_t1_identity_matches,
    _rehydrate_candidate_members,
    _score_snapshot_rows,
    _snapshot_paths,
    _teacher_bbox_corners,
    _tune_gate,
    _validate_registered_inputs,
)
from category_priors.evaluator import GroundTruthScene, PredictedInstance
from category_priors.io import hash_json, load_json, sha256_file, write_json, write_rows
from category_priors.taxonomy import load_taxonomy
from category_priors.full_instance_vote import (
    GaussianVoteEvidence,
    VOTE_EVIDENCE_SCHEMA,
    save_gaussian_vote_evidence,
)
from category_priors.v9_legacy_runner import CLASSES_32
from category_priors.v9_t1_runner import V9_FEATURE_SEED, V9_T1_SCHEMA


def _node(low: float, middle: float, high: float) -> dict[str, object]:
    geometry = {
        field: {
            "q25": math.log(low),
            "q50": math.log(middle),
            "q75": math.log(high),
        }
        for field in (
            "log_extent_short_m",
            "log_extent_mid_m",
            "log_extent_long_m",
        )
    }
    return {"shrunk": {"geometry": geometry}}


def _candidate(raw_id: int, point_count: int, exported: int) -> dict[str, object]:
    votes = [0.0] * 33
    votes[0] = 8.0
    votes[32] = 2.0
    return {
        "scene_id": "scene0645_00",
        "candidate_id": raw_id,
        "raw_instance_id": raw_id,
        "source": "global" if raw_id == 10 else "other_classes",
        "point_count": point_count,
        "metric_extents_m": [0.25, 0.30, 0.35],
        "vote_histogram": votes,
        "predicted_class_index": 0,
        "predicted_class": "chair",
        "winner_ratio": 0.8,
        "background_ratio": 0.2,
        "eligible": True,
        "eligibility_reason": "eligible",
        "Q": 0.8,
        "exported_instance_id": exported,
    }


def _config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        workspace=tmp_path,
        runtime_manifest=tmp_path / "runtime.json",
        locked_runtime_manifest=tmp_path / "locked-runtime.json",
        t1_root=tmp_path / "t1",
        rebuild_t1_root=tmp_path / "rebuild",
        gt_dir=tmp_path / "gt",
        locked_gt_dir=tmp_path / "locked-gt",
        train_stats=tmp_path / "train-stats.parquet",
        category_priors=tmp_path / "priors.json",
        size_bins=tmp_path / "size-bins.json",
        locked_evaluation_scenes=tmp_path / "final.json",
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
        taxonomy_path=None,
        git_commit="test-commit",
        disk_floor_gib=0.0,
    )


def _t1_command(prefix: str) -> list[str]:
    return [
        f"{prefix}/python",
        f"{prefix}/postprocess.py",
        "--progress_path",
        f"{prefix}/progress.txt",
        "--stage_trace_path",
        f"{prefix}/stage_trace.npz",
        "--json_path",
        f"{prefix}/output.json",
        "--prior_metadata_path",
        f"{prefix}/diagnostics.json",
        "--classes",
        "chair",
        "table",
        "--other_classes",
        "book",
        "cup",
        "--seed",
        "42",
    ]


def test_t1_command_normalization_changes_only_registered_paths() -> None:
    left = _normalized_t1_command(_t1_command("/historical"))
    right = _normalized_t1_command(_t1_command("/current"))
    assert left == right

    changed = _t1_command("/current")
    changed[changed.index("table")] = "sofa"
    assert _normalized_t1_command(changed) != left

    missing = _t1_command("/current")
    missing_index = missing.index("--stage_trace_path")
    del missing[missing_index : missing_index + 2]
    with pytest.raises(ValueError, match="stage_trace_path"):
        _normalized_t1_command(missing)


def test_registered_t1_requires_allowed_producer_full_command_source_and_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    postprocess = tmp_path / "postprocess.py"
    postprocess.write_bytes(b"registered postprocess\n")
    postprocess_sha256 = sha256_file(postprocess)
    postprocess_size = postprocess.stat().st_size
    feature = {"path": "/feature.ply", "size_bytes": 10, "mtime_ns": 1}
    scale = {"path": "/scale.pt", "size_bytes": 11, "mtime_ns": 2}
    labels = {"path": "/labels.pt", "size_bytes": 12, "mtime_ns": 3}
    expected_invocation = SimpleNamespace(
        identity={
            "command": _t1_command(str(tmp_path)),
            "feature_ply": feature,
            "scale_gate": scale,
            "label_features": labels,
        }
    )
    hashes = {
        "record_sha256": "1" * 64,
        "stage_trace_sha256": "2" * 64,
        "stage_trace_metadata_sha256": "3" * 64,
        "output_sha256": "4" * 64,
    }
    monkeypatch.setattr(experiment_module, "v9_t1_run_complete", lambda *_: True)
    monkeypatch.setattr(
        experiment_module,
        "_git_blob_identity",
        lambda *_: {
            "size_bytes": postprocess_size,
            "sha256": postprocess_sha256,
        },
    )
    monkeypatch.setattr(experiment_module, "_t1_artifact_hashes", lambda *_: hashes)

    identity = {
        "schema": V9_T1_SCHEMA,
        "git_commit": config.git_commit,
        "scene_id": "scene0645_00",
        "condition": "T1-B1",
        "seed": V9_FEATURE_SEED,
        "input_budget": "existing-scene-feature-2k",
        "contributor_weight": "alpha_times_t_prev",
        "teacher_prior_mode": "original",
        "causal_level": "L0",
        "command": _t1_command(str(tmp_path)),
        "postprocess": {
            "size_bytes": postprocess_size,
            "sha256": postprocess_sha256,
        },
        "feature_ply": feature,
        "scale_gate": scale,
        "label_features": labels,
    }
    record = {"identity": identity, "artifact_sha256": hashes}
    kwargs = {
        "config": config,
        "scene_id": "scene0645_00",
        "expected_invocation": expected_invocation,
        "paths": SimpleNamespace(),
        "record": record,
    }
    assert _registered_t1_identity_matches(**kwargs)

    identity["git_commit"] = REGISTERED_HISTORICAL_T1_PRODUCER
    identity["command"] = _t1_command(str(tmp_path / "removed-workspace"))
    assert _registered_t1_identity_matches(**kwargs)
    identity["git_commit"] = "unregistered-producer"
    assert not _registered_t1_identity_matches(**kwargs)
    identity["git_commit"] = config.git_commit
    identity["command"] = _t1_command(str(tmp_path))

    identity["command"] = [*_t1_command(str(tmp_path)), "--unexpected-flag"]
    assert not _registered_t1_identity_matches(**kwargs)
    identity["command"] = _t1_command(str(tmp_path))
    record["artifact_sha256"] = {**hashes, "output_sha256": "f" * 64}
    assert not _registered_t1_identity_matches(**kwargs)

    record["artifact_sha256"] = hashes
    postprocess.write_bytes(b"tampered postprocess\n")
    assert not _registered_t1_identity_matches(**kwargs)


def test_missing_t1_rebuild_uses_configured_python_bin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python_bin = tmp_path / "preflight-env/bin/python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_bytes(b"python")
    config = replace(
        _config(tmp_path),
        allow_rebuild_missing_traces=True,
        python_bin=python_bin,
    )
    monkeypatch.setattr(
        experiment_module,
        "build_v9_t1_invocation",
        lambda **_: SimpleNamespace(identity={}),
    )
    captured: dict[str, object] = {}

    class RebuildObserved(RuntimeError):
        pass

    def fake_execute(**kwargs: object) -> None:
        captured.update(kwargs)
        raise RebuildObserved

    monkeypatch.setattr(experiment_module, "execute_v9_t1_runs", fake_execute)

    with pytest.raises(RebuildObserved):
        _load_t1_paths(config, "scene0645_00", {})

    assert captured["python_bin"] == python_bin


def test_resource_checkpoint_enforces_disk_and_records_cgroup(tmp_path: Path) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024\n", encoding="utf-8")
    (cgroup / "memory.max").write_text("96636764160\n", encoding="utf-8")
    (cgroup / "memory.events").write_text("oom 0\n", encoding="utf-8")
    config = replace(_config(tmp_path), cgroup_root=cgroup)
    config.artifacts_root.mkdir(parents=True)
    snapshot = _resource_checkpoint(
        config, stage="test", scene_id="scene0000_00", status="running"
    )
    assert snapshot["memory.current"] == "1024"
    assert snapshot["memory.max"] == "96636764160"
    assert snapshot["memory.events"] == "oom 0"
    status = load_json(config.artifacts_root / "full_instance_size_status.json")
    assert status["resource_snapshot"] == snapshot

    impossible = replace(config, disk_floor_gib=10**9)
    with pytest.raises(RuntimeError, match="available disk"):
        _resource_checkpoint(impossible, stage="test", status="running")


def test_rehydrate_and_rescore_are_exact_and_idempotent() -> None:
    merged = np.asarray([10, 10, -1, 20, 20, 20], dtype=np.int64)
    rows = [_candidate(10, 2, 0), _candidate(20, 3, 1)]
    hydrated = _rehydrate_candidate_members(rows, merged)
    assert hydrated[0]["member_indices"].tolist() == [0, 1]
    assert hydrated[1]["member_indices"].tolist() == [3, 4, 5]

    priors = {
        "global": _node(0.5, 1.0, 2.0),
        "categories": {"chair": _node(0.2, 0.3, 0.5)},
    }
    snapshot = {
        "base_rows": rows,
        "base_rows_sha256": hash_json(rows),
        "rows": rows,
    }
    arrays = {"merged_partition": merged}
    first, first_identity = _score_snapshot_rows(snapshot, arrays, priors)
    second, second_identity = _score_snapshot_rows(
        {
            "base_rows": rows,
            "base_rows_sha256": hash_json(rows),
            "rows": first,
        },
        arrays,
        priors,
    )
    assert first == second
    assert first_identity == second_identity
    assert first_identity["member_point_count"] == 5
    assert "member_indices" not in first[0]
    assert "score_mode" not in first[0]


def test_controlled_baseline_preserves_partition_class_bbox_and_count(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    scene_id = "scene0645_00"
    snapshot_paths = _snapshot_paths(config, scene_id)
    snapshot_paths["root"].mkdir(parents=True)
    save_gaussian_vote_evidence(
        snapshot_paths["votes"],
        GaussianVoteEvidence(
            row_offsets=np.asarray([0, 2, 4, 4, 6, 8], dtype=np.int64),
            channels=np.asarray([0, 32, 0, 32, 1, 32, 1, 32], dtype=np.uint8),
            counts=np.asarray([4, 1, 4, 1, 3, 2, 3, 2], dtype=np.uint64),
            metadata={
                "schema": VOTE_EVIDENCE_SCHEMA,
                "point_count": 5,
                "class_names": list(CLASSES_32),
                "channel_count": 33,
                "background_index": 32,
                "total_vote_count": 20,
            },
        ),
    )
    write_json(config.category_priors, {"kind": "unit-test"})
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
         [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    point_cloud = (
        tmp_path / "output_models/point_cloud/iteration_30000/point_cloud.ply"
    )
    point_cloud.parent.mkdir(parents=True)
    point_cloud.write_text(
        "ply\nformat ascii 1.0\nelement vertex 5\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        + "\n".join(" ".join(map(str, row)) for row in xyz)
        + "\n",
        encoding="utf-8",
    )
    merged = np.asarray([10, 10, -1, 20, 20], dtype=np.int64)
    with snapshot_paths["arrays"].open("wb") as handle:
        np.savez_compressed(
            handle,
            merged_partition=merged,
            post_global_knn=merged,
            post_filter=merged,
            final_internal_labels=merged,
            exported_prediction=np.asarray([0, 0, -1, 1, 1], dtype=np.int64),
        )
    baseline_path = tmp_path / "frozen-output.json"
    baseline = {
        "point_labels": [0, 0, -1, 1, 1],
        "is_big_gaussian": [False] * 5,
        "instances": {
            "0": {
                "class": "chair",
                "score": 0.8,
                "bbox": _teacher_bbox_corners(xyz[:2]),
            },
            "1": {
                "class": "table",
                "score": 0.6,
                "bbox": _teacher_bbox_corners(xyz[3:]),
            },
        },
    }
    write_json(baseline_path, baseline)
    rows = [_candidate(10, 2, 0), _candidate(20, 2, 1)]
    rows[1]["predicted_class"] = "table"
    rows[1]["Q"] = 0.6
    write_json(
        snapshot_paths["metadata"],
        {
            "scene_id": scene_id,
            "baseline_output": str(baseline_path),
            "base_rows": rows,
            "base_rows_sha256": hash_json(rows),
            "rows": rows,
            "rows_sha256": hash_json(rows),
        },
    )

    diagnostics = _materialize_condition_output(
        config,
        "dev8",
        scene_id,
        {"base_path": str(tmp_path)},
        "controlled-baseline",
        0.55,
    )
    output = load_json(
        _condition_output_paths(
            config, "dev8", "controlled-baseline", scene_id
        )["output"]
    )
    assert output["point_labels"] == baseline["point_labels"]
    assert len(output["instances"]) == len(baseline["instances"])
    for key in baseline["instances"]:
        assert output["instances"][key]["class"] == baseline["instances"][key]["class"]
        assert output["instances"][key]["bbox"] == baseline["instances"][key]["bbox"]
    assert output["instances"]["0"]["score"] == 0.8
    assert output["instances"]["1"]["score"] == 0.6
    assert diagnostics["changed_export_points_vs_frozen_baseline"] == 0
    assert diagnostics["frozen_mask_class_bbox_instance_count_exact"]
    assert diagnostics["empty_restoration_replayed_from_post_filter"]
    assert diagnostics["empty_restoration_native_scores_exact"]

    # A valid-looking but corrupted cached result must not be reused.
    output_path = _condition_output_paths(
        config, "dev8", "controlled-baseline", scene_id
    )["output"]
    corrupted = load_json(output_path)
    corrupted["instances"]["0"]["score"] = 0.123
    write_json(output_path, corrupted)
    rerun = _materialize_condition_output(
        config,
        "dev8",
        scene_id,
        {"base_path": str(tmp_path)},
        "controlled-baseline",
        0.55,
    )
    assert load_json(output_path)["instances"]["0"]["score"] == 0.8
    assert rerun["output_sha256"] == sha256_file(output_path)


def test_scoring_rejects_tampered_immutable_base_rows() -> None:
    rows = [_candidate(10, 2, 0)]
    snapshot = {
        "base_rows": rows,
        "base_rows_sha256": hash_json(rows),
        "rows": rows,
    }
    snapshot["base_rows"][0]["Q"] = 0.1
    with pytest.raises(ValueError, match="base_rows content hash"):
        _score_snapshot_rows(
            snapshot,
            {"merged_partition": np.asarray([10, 10], dtype=np.int64)},
            {
                "global": _node(0.5, 1.0, 2.0),
                "categories": {"chair": _node(0.2, 0.3, 0.5)},
            },
        )


def test_registered_inputs_bind_train_stats_and_locked_replacement_gt(
    tmp_path: Path,
) -> None:
    taxonomy = load_taxonomy()
    config = _config(tmp_path)
    train_stats = config.train_stats
    write_rows(
        train_stats,
        [{"scene_id": "scene9000_00", "split": "train"}],
    )
    priors = {
        "schema_version": "1.0",
        "kind": "category_priors",
        "provenance": {
            "splits": ["train"],
            "source_table_sha256": sha256_file(train_stats),
            "taxonomy_sha256": taxonomy.content_hash,
        },
        "normalization": {"units": "meters"},
        "categories": {name: {} for name in taxonomy.canonical_classes},
    }
    priors["content_sha256"] = hash_json(priors)
    write_json(tmp_path / "priors.json", priors)

    tune24 = [*DEV8, *HOLDOUT5]
    tune24.extend(
        f"{scene_id[:9]}_99" for scene_id in (*DEV8, *HOLDOUT5)[:11]
    )
    final48 = [
        "scene0019_01",
        *(f"scene{1000 + index:04d}_00" for index in range(47)),
    ]
    tune_runtime = {scene_id: {"scene_id": scene_id} for scene_id in tune24}
    final_runtime = {scene_id: {"scene_id": scene_id} for scene_id in final48}
    tune_manifest = {
        "kind": "scene_runtime_manifest",
        "scenes": list(tune_runtime.values()),
    }
    tune_manifest["content_sha256"] = hash_json(tune_manifest)
    write_json(config.runtime_manifest, tune_manifest)
    final_manifest = {
        "kind": "scene_runtime_manifest",
        "scenes": list(final_runtime.values()),
    }
    final_manifest["content_sha256"] = hash_json(final_manifest)
    write_json(config.locked_runtime_manifest, final_manifest)
    write_json(
        config.size_bins,
        {
            "boundaries_m": {
                "tiny_max_m": 0.2,
                "small_max_m": 0.5,
                "medium_max_m": 1.0,
            }
        },
    )
    write_json(
        tmp_path / "final.json",
        {
            "kind": "locked_evaluation_scenes",
            "scenes": [
                {"scene_id": scene_id, "physical_scene_id": scene_id[:9]}
                for scene_id in final48
            ],
        },
    )
    (tmp_path / "gt").mkdir()
    (tmp_path / "locked-gt").mkdir()
    for scene_id in tune24:
        (tmp_path / "gt" / f"{scene_id}.npz").write_bytes(
            f"tune:{scene_id}".encode("utf-8")
        )
    for scene_id in final48:
        (tmp_path / "locked-gt" / f"{scene_id}.npz").write_bytes(
            f"final:{scene_id}".encode("utf-8")
        )
    result = _validate_registered_inputs(
        config,
        tune_scenes=tune_runtime,
        final_scenes=final_runtime,
        tune24=tune24,
        final48=final48,
        priors=priors,
        taxonomy=taxonomy,
    )
    assert result["physical_split_overlap"] is False
    assert result["train_stats_sha256"] == sha256_file(train_stats)
    assert result["scene0019_replacement_gt"].endswith("scene0019_01.npz")
    assert result["runtime_manifest_sha256"] == sha256_file(config.runtime_manifest)
    assert result["locked_runtime_manifest_sha256"] == sha256_file(
        config.locked_runtime_manifest
    )
    assert result["size_bins_sha256"] == sha256_file(config.size_bins)
    assert set(result["tune_gt"]) == set(tune24)
    assert set(result["final_gt"]) == set(final48)
    assert result["tune_gt"][DEV8[0]]["sha256"] == sha256_file(
        config.gt_dir / f"{DEV8[0]}.npz"
    )


def _endpoint_rows(changed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene_id in DEV8:
        for condition, map_value in (
            ("controlled-baseline", 0.100),
            ("global-size", 0.100),
            ("class-size", 0.103),
        ):
            rows.append(
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    "official_map_50_90": map_value,
                    "ap50": 0.2,
                    "ap25": 0.3,
                    "instance_count": 10.0,
                    "coverage": 0.5,
                    "gaussian_micro_precision": 0.8,
                    "fp_tp_ratio_025": 1.0,
                    "tiny_small_recall_025": 0.2,
                    "tiny_small_recall_050": 0.1,
                    "changed_points_class_vs_global": (
                        changed if condition == "class-size" else 0
                    ),
                }
            )
    return rows


def test_endpoint_gate_requires_real_class_vs_global_point_change() -> None:
    failed = _endpoint_gate(_endpoint_rows(0), threshold_dev_scenes=set(DEV8[:2]))
    passed = _endpoint_gate(_endpoint_rows(5), threshold_dev_scenes=set(DEV8[:2]))
    assert not failed["passed"]
    assert not failed["real_endpoint_change"]
    assert passed["passed"]
    assert passed["real_endpoint_change"]


def test_condition_summary_uses_true_pooled_counts_for_micro_guards() -> None:
    rows = [
        {
            "scene_id": "scene0001_00",
            "condition": "global-size",
            "official_map_50_90": 0.1,
            "ap50": 0.1,
            "ap25": 0.1,
            "historical_map_50_95": 0.1,
            "native_official_map_50_90": 0.1,
            "native_ap50": 0.1,
            "native_ap25": 0.1,
            "native_historical_map_50_95": 0.1,
            "instance_count": 1,
            "coverage": 1.0,
            "gt_point_count": 100,
            "gt_nearest_declared_count": 100,
            "gaussian_micro_precision": 1.0,
            "correct_gaussian_count": 1,
            "predicted_gaussian_count": 1,
            "fp_tp_ratio_025": 0.0,
            "true_positive_count_025": 1,
            "false_positive_count_025": 0,
            "tiny_small_gt_count": 1,
            "tiny_small_match_count_025": 1,
            "tiny_small_match_count_050": 1,
            "tiny_small_recall_025": 1.0,
            "tiny_small_recall_050": 1.0,
        },
        {
            "scene_id": "scene0002_00",
            "condition": "global-size",
            "official_map_50_90": 0.0,
            "ap50": 0.0,
            "ap25": 0.0,
            "historical_map_50_95": 0.0,
            "native_official_map_50_90": 0.0,
            "native_ap50": 0.0,
            "native_ap25": 0.0,
            "native_historical_map_50_95": 0.0,
            "instance_count": 9,
            "coverage": 0.0,
            "gt_point_count": 900,
            "gt_nearest_declared_count": 0,
            "gaussian_micro_precision": 0.0,
            "correct_gaussian_count": 0,
            "predicted_gaussian_count": 9,
            "fp_tp_ratio_025": 9.0,
            "true_positive_count_025": 0,
            "false_positive_count_025": 9,
            "tiny_small_gt_count": 9,
            "tiny_small_match_count_025": 0,
            "tiny_small_match_count_050": 0,
            "tiny_small_recall_025": 0.0,
            "tiny_small_recall_050": 0.0,
        },
    ]
    summary = _condition_summary(rows)["global-size"]
    assert summary["gaussian_micro_precision"] == pytest.approx(0.1)
    assert summary["coverage"] == pytest.approx(0.1)
    assert summary["fp_tp_ratio_025"] == pytest.approx(9.0)
    assert summary["tiny_small_recall_025"] == pytest.approx(0.1)
    assert summary["aggregation"] == "scene-equal-ap"


def test_pooled_official_summary_is_evaluated_as_one_dataset() -> None:
    taxonomy = load_taxonomy()
    class_id = taxonomy.canonical_classes.index("chair")
    gt = {
        scene_id: GroundTruthScene(
            scene_id,
            np.full(100, class_id, dtype=np.int64),
            np.zeros(100, dtype=np.int64),
        )
        for scene_id in ("scene0001_00", "scene0002_00")
    }
    predictions = {
        condition: [
            PredictedInstance(
                scene_id=scene_id,
                instance_id=0,
                class_id=class_id,
                score=0.9,
                mask=np.ones(100, dtype=bool),
            )
            for scene_id in gt
        ]
        for condition in ("global-size", "class-size")
    }
    structural = {
        condition: {
            "scene_count": 2,
            "aggregation": "scene-equal",
            "instance_count": 1.0,
            "coverage": 1.0,
            "gaussian_micro_precision": 1.0,
            "fp_tp_ratio_025": 0.0,
            "tiny_small_recall_025": 1.0,
            "tiny_small_recall_050": 1.0,
        }
        for condition in predictions
    }
    result = _pooled_official_condition_summary(
        ground_truth_by_scene=gt,
        predictions_by_condition=predictions,
        native_predictions_by_condition=predictions,
        taxonomy=taxonomy,
        structural_summaries=structural,
    )
    assert result["global-size"]["aggregation"] == "pooled-official-evaluator"
    assert result["global-size"]["official_map_50_90"] == pytest.approx(1.0)


def test_endpoint_gate_uses_supplied_pooled_official_metrics() -> None:
    rows = _endpoint_rows(5)
    summaries = _condition_summary(rows)
    summaries["controlled-baseline"]["aggregation"] = "pooled-official-evaluator"
    summaries["global-size"]["aggregation"] = "pooled-official-evaluator"
    summaries["class-size"]["aggregation"] = "pooled-official-evaluator"
    summaries["class-size"]["official_map_50_90"] = 0.09
    result = _endpoint_gate(
        rows,
        threshold_dev_scenes=set(DEV8[:2]),
        official_summaries=summaries,
    )
    assert not result["passed"]
    assert result["primary_ap_aggregation"] == "pooled-official-evaluator"


def test_tiny_small_guard_is_disabled_when_stage2_capacity_is_insufficient() -> None:
    rows = _endpoint_rows(5)
    for row in rows:
        if row["condition"] == "class-size":
            row["tiny_small_recall_025"] = 0.0
            row["tiny_small_recall_050"] = 0.0
    enabled = _endpoint_gate(
        rows,
        threshold_dev_scenes=set(DEV8[:2]),
        tiny_small_gate_enabled=True,
    )
    disabled = _endpoint_gate(
        rows,
        threshold_dev_scenes=set(DEV8[:2]),
        tiny_small_gate_enabled=False,
    )
    assert not enabled["passed"]
    assert not enabled["tiny_small_guard_passed"]
    assert disabled["passed"]
    assert disabled["tiny_small_guard_passed"]
    assert not disabled["tiny_small_exploratory_signal"]


def test_downstream_gates_require_real_class_vs_global_point_change() -> None:
    unchanged = _endpoint_rows(0)
    changed = _endpoint_rows(5)
    bootstrap = {"ci95": [0.001, 0.003]}

    assert not _holdout_gate(unchanged)["passed"]
    assert _holdout_gate(changed)["passed"]
    assert not _tune_gate(unchanged)["passed"]
    assert _tune_gate(changed)["passed"]
    assert not _final_gate(unchanged, bootstrap)["passed"]
    assert _final_gate(changed, bootstrap)["passed"]

    for gate in (
        _holdout_gate(changed),
        _tune_gate(changed),
        _final_gate(changed, bootstrap),
    ):
        assert gate["real_endpoint_change"]


def test_downstream_gates_skip_tiny_small_guard_only_when_disabled() -> None:
    rows = _endpoint_rows(5)
    for row in rows:
        if row["condition"] == "class-size":
            row["tiny_small_recall_025"] = 0.0
            row["tiny_small_recall_050"] = 0.0
    bootstrap = {"ci95": [0.001, 0.003]}

    for gate_fn in (_holdout_gate, _tune_gate):
        assert not gate_fn(rows, tiny_small_gate_enabled=True)["passed"]
        result = gate_fn(rows, tiny_small_gate_enabled=False)
        assert result["passed"]
        assert not result["tiny_small_gate_enabled"]
        assert result["tiny_small_guard_passed"]

    assert not _final_gate(
        rows, bootstrap, tiny_small_gate_enabled=True
    )["passed"]
    final = _final_gate(
        rows, bootstrap, tiny_small_gate_enabled=False
    )
    assert final["passed"]
    assert not final["tiny_small_gate_enabled"]
    assert final["tiny_small_guard_passed"]
