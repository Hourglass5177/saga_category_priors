from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from category_priors.io import write_json
from category_priors.v9_feature_training import (
    V9_FEATURE_ITERATIONS,
    V9_FEATURE_SCHEMA,
    V9_FEATURE_SEED,
    v9_feature_training_paths,
)
from category_priors.v9_legacy_runner import (
    CLASSES_32,
    TRACE_ARRAYS,
    V9LegacyInvocation,
    build_v9_legacy_invocation,
    execute_v9_legacy_runs,
    read_v9_legacy_resources,
    v9_legacy_paths,
    v9_legacy_run_complete,
)


def _touch(path: Path, payload: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    base = tmp_path / "scene"
    workspace = tmp_path / "workspace"
    python_bin = _touch(tmp_path / "env/bin/python", b"python")
    _touch(workspace / "postprocess.py", b"print('postprocess')\n")
    _touch(base / "fastRecon/dense/sparse/0/images/frame.jpg")
    (base / "fastRecon/dense/sparse/0").mkdir(parents=True, exist_ok=True)
    _touch(
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        b"ply\nformat ascii 1.0\nend_header\n",
    )
    for directory in (base / "saga/masks", base / "saga/labels", base / "saga/mask_scales"):
        _touch(directory / "frame.pt")
    _touch(base / "saga/labels/label_features.pt", b"labels")
    scene = {
        "scene_id": "scene0001_00",
        "base_path": str(base),
        "python_bin": str(python_bin),
        "scene_scale_m_per_unit": 1.25,
    }
    manifest = tmp_path / "runtime.json"
    write_json(manifest, {"kind": "scene_runtime_manifest", "scenes": [scene]})

    feature_root = tmp_path / "feature-10k-objectbank"
    paths = v9_feature_training_paths(feature_root, scene["scene_id"])
    _touch(paths.feature_ply, b"ply\nformat ascii 1.0\nend_header\n")
    _touch(paths.scale_gate, b"gate")
    paths.progress.write_text("100", encoding="utf-8")
    identity = {
        "schema": V9_FEATURE_SCHEMA,
        "scene_id": scene["scene_id"],
        "iterations": V9_FEATURE_ITERATIONS,
        "seed": V9_FEATURE_SEED,
        "outputs": {
            "feature_ply": str(paths.feature_ply),
            "scale_gate": str(paths.scale_gate),
        },
    }
    write_json(paths.record, {
        "kind": "v9_feature_training_run",
        "status": "complete",
        "git_commit": "feature-commit",
        "identity": identity,
    })
    return manifest, workspace, feature_root, scene


def _option(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _write_complete_artifacts(invocation: V9LegacyInvocation) -> None:
    paths = invocation.paths
    paths.root.mkdir(parents=True, exist_ok=True)
    write_json(paths.output, {
        "point_labels": [-1, 0, 0],
        "instances": {"0": {"class": "chair", "score": 0.75}},
        "prediction_contract": {
            "schema": "saga-strict-prediction-contract-v1",
            "point_count": 3,
        },
    })
    write_json(paths.diagnostics, {"kind": "diagnostics"})
    paths.progress.write_text("1.0\n", encoding="utf-8")
    trace_arrays = {
        name: np.full(3, -1, dtype=np.int32) for name in TRACE_ARRAYS
    }
    trace_arrays["final_internal_labels"] = np.asarray([-1, 0, 0], dtype=np.int32)
    trace_arrays["exported_prediction"] = np.asarray([-1, 0, 0], dtype=np.int32)
    np.savez_compressed(paths.stage_trace, **trace_arrays)
    write_json(paths.stage_trace_metadata, {
        "schema": "saga-v9-legacy-stage-trace-v1",
        "point_count": 3,
        "level": "L0",
        "raw_instances": {"0": {"class": "chair", "score": 0.75}},
    })


def test_build_invocation_uses_only_v9_features_and_frozen_legacy_conditions(
    tmp_path: Path,
) -> None:
    _, workspace, feature_root, scene = _fixture(tmp_path)
    b0 = build_v9_legacy_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        feature_root=feature_root,
        output_root=tmp_path / "runs",
        condition="F10k-B0",
        git_commit="code",
        feature_git_commit="feature-commit",
    )
    b1 = build_v9_legacy_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        feature_root=feature_root,
        output_root=tmp_path / "runs",
        condition="F10k-B1",
        git_commit="code",
        feature_git_commit="feature-commit",
    )
    feature_paths = v9_feature_training_paths(feature_root, scene["scene_id"])
    assert Path(_option(b0.command, "--contrastive_feature_point_cloud_path")) == feature_paths.feature_ply
    assert Path(_option(b0.command, "--scale_gate_path")) == feature_paths.scale_gate
    assert _option(b0.command, "--seed") == "42"
    assert _option(b0.command, "--v7-causal-ablation") == "L0"
    assert "--disable_other_classes" in b0.command
    assert "--disable_other_classes" not in b1.command
    class_start = b0.command.index("--classes") + 1
    selected_start = b0.command.index("--selected_classes")
    assert b0.command[class_start:selected_start] == CLASSES_32
    forbidden = {"--clean", "--max_contributor_cache_path", "--gt-dir", "--iterations"}
    assert forbidden.isdisjoint(b0.command)
    assert b0.paths.root != b1.paths.root


def test_complete_requires_strict_contract_and_every_stage_trace(
    tmp_path: Path,
) -> None:
    _, workspace, feature_root, scene = _fixture(tmp_path)
    invocation = build_v9_legacy_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        feature_root=feature_root,
        output_root=tmp_path / "runs",
        condition="F10k-B0",
        git_commit="code",
    )
    _write_complete_artifacts(invocation)
    write_json(invocation.paths.record, {
        "kind": "v9_f10k_legacy_run",
        "status": "complete",
        "identity": dict(invocation.identity),
    })
    assert v9_legacy_run_complete(invocation.paths, invocation.identity)

    write_json(invocation.paths.output, {
        "point_labels": [-1, 1, 1],
        "instances": {"1": {"class": "chair", "score": 0.75}},
        "prediction_contract": {
            "schema": "saga-strict-prediction-contract-v1", "point_count": 3
        },
    })
    assert not v9_legacy_run_complete(invocation.paths, invocation.identity)

    _write_complete_artifacts(invocation)
    with np.load(invocation.paths.stage_trace, allow_pickle=False) as arrays:
        incomplete = {
            name: np.asarray(arrays[name])
            for name in arrays.files
            if name != "post_filter"
        }
    np.savez_compressed(invocation.paths.stage_trace, **incomplete)
    assert not v9_legacy_run_complete(invocation.paths, invocation.identity)

    _write_complete_artifacts(invocation)
    with np.load(invocation.paths.stage_trace, allow_pickle=False) as arrays:
        mismatched = {name: np.asarray(arrays[name]) for name in arrays.files}
    mismatched["exported_prediction"] = np.asarray([-1, -1, 0], dtype=np.int32)
    np.savez_compressed(invocation.paths.stage_trace, **mismatched)
    assert not v9_legacy_run_complete(invocation.paths, invocation.identity)


def test_execute_is_sequential_resumable_and_does_not_train(
    tmp_path: Path,
) -> None:
    manifest, workspace, feature_root, scene = _fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def executor(invocation: V9LegacyInvocation) -> int:
        calls.append((invocation.scene_id, invocation.condition))
        _write_complete_artifacts(invocation)
        return 0

    first = execute_v9_legacy_runs(
        scene_manifest=manifest,
        feature_root=feature_root,
        output_root=tmp_path / "runs",
        workspace=workspace,
        git_commit="code",
        feature_git_commit="feature-commit",
        scene_ids=[scene["scene_id"]],
        cgroup_root=None,
        disk_floor_gib=0.0,
        executor=executor,
    )
    second = execute_v9_legacy_runs(
        scene_manifest=manifest,
        feature_root=feature_root,
        output_root=tmp_path / "runs",
        workspace=workspace,
        git_commit="code",
        feature_git_commit="feature-commit",
        scene_ids=[scene["scene_id"]],
        cgroup_root=None,
        disk_floor_gib=0.0,
        executor=executor,
    )
    assert calls == [
        (scene["scene_id"], "F10k-B0"),
        (scene["scene_id"], "F10k-B1"),
    ]
    assert first["complete"] == 2
    assert [row["status"] for row in second["runs"]] == [
        "skipped_complete", "skipped_complete"
    ]
    assert (tmp_path / "runs/execution_summary.json").is_file()


def test_feature_record_must_be_registered_10k_seed42(tmp_path: Path) -> None:
    _, workspace, feature_root, scene = _fixture(tmp_path)
    paths = v9_feature_training_paths(feature_root, scene["scene_id"])
    record = json.loads(paths.record.read_text(encoding="utf-8"))
    record["identity"]["iterations"] = 2000
    write_json(paths.record, record)
    with pytest.raises(ValueError, match="not complete/registered"):
        build_v9_legacy_invocation(
            workspace=workspace,
            scene=scene,
            scene_id=scene["scene_id"],
            feature_root=feature_root,
            output_root=tmp_path / "runs",
            condition="F10k-B0",
            git_commit="code",
        )


def test_cgroup_resource_check_uses_90_gib_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.current").write_text("1024\n", encoding="utf-8")
    (cgroup / "memory.max").write_text(str(90 * 1024**3), encoding="utf-8")
    (cgroup / "memory.events").write_text("low 0\noom 0\noom_kill 0\n", encoding="utf-8")
    monkeypatch.setattr(
        "category_priors.v9_legacy_runner.shutil.disk_usage",
        lambda _: type("Disk", (), {"free": 100 * 1024**3})(),
    )
    snapshot = read_v9_legacy_resources(tmp_path / "runs", cgroup_root=cgroup)
    assert snapshot["cgroup"]["max"] == 90 * 1024**3
    assert snapshot["disk_free_gib"] == 100.0

    (cgroup / "memory.max").write_text(str(80 * 1024**3), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 90 GiB"):
        read_v9_legacy_resources(tmp_path / "runs", cgroup_root=cgroup)
