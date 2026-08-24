from __future__ import annotations

from pathlib import Path

import numpy as np

from category_priors.io import write_json
from category_priors.v9_legacy_runner import TRACE_ARRAYS
from category_priors.v9_t1_runner import (
    V9T1Invocation,
    build_v9_t1_invocation,
    execute_v9_t1_runs,
    registered_v9_t1_batch,
    v9_t1_run_complete,
)


def _touch(path: Path, payload: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture(tmp_path: Path, scene_id: str = "scene0001_00") -> tuple[Path, Path, dict]:
    base = tmp_path / scene_id
    workspace = tmp_path / "workspace"
    python_bin = _touch(tmp_path / "env/bin/python", b"python")
    _touch(workspace / "postprocess.py", b"print('postprocess')\n")
    _touch(base / "fastRecon/dense/sparse/0/images/frame.jpg")
    (base / "fastRecon/dense/sparse/0").mkdir(parents=True, exist_ok=True)
    _touch(
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        b"ply\nformat ascii 1.0\nend_header\n",
    )
    for directory in (
        base / "saga/masks",
        base / "saga/labels",
        base / "saga/mask_scales",
    ):
        _touch(directory / "frame.pt")
    _touch(base / "saga/labels/label_features.pt", b"labels")
    _touch(
        base / "saga/contrastive_feature_point_cloud.ply",
        b"ply\nformat ascii 1.0\nend_header\n",
    )
    _touch(base / "saga/scale_gate.pt", b"gate")
    scene = {
        "scene_id": scene_id,
        "base_path": str(base),
        "python_bin": str(python_bin),
        "scene_scale_m_per_unit": 1.25,
    }
    manifest = tmp_path / f"{scene_id}.json"
    write_json(manifest, {"kind": "scene_runtime_manifest", "scenes": [scene]})
    return manifest, workspace, scene


def _option(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def _write_complete(invocation: V9T1Invocation) -> None:
    paths = invocation.paths
    paths.root.mkdir(parents=True, exist_ok=True)
    write_json(
        paths.output,
        {
            "point_labels": [-1, 0, 0],
            "instances": {"0": {"class": "chair", "score": 0.75}},
            "prediction_contract": {
                "schema": "saga-strict-prediction-contract-v1",
                "point_count": 3,
            },
        },
    )
    write_json(paths.diagnostics, {"kind": "diagnostics"})
    paths.progress.write_text("1.0\n", encoding="utf-8")
    trace_arrays = {
        name: np.full(3, -1, dtype=np.int32) for name in TRACE_ARRAYS
    }
    trace_arrays["final_internal_labels"] = np.asarray([-1, 0, 0], dtype=np.int32)
    trace_arrays["exported_prediction"] = np.asarray([-1, 0, 0], dtype=np.int32)
    np.savez_compressed(paths.stage_trace, **trace_arrays)
    write_json(
        paths.stage_trace_metadata,
        {
            "schema": "saga-v9-legacy-stage-trace-v1",
            "point_count": 3,
            "level": "L0",
            "raw_instances": {"0": {"class": "chair", "score": 0.75}},
        },
    )


def test_t1_uses_existing_scene_features_and_registered_teacher_structure(
    tmp_path: Path,
) -> None:
    _, workspace, scene = _fixture(tmp_path)
    b0 = build_v9_t1_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=tmp_path / "runs",
        condition="T1-B0",
        git_commit="fixed-commit",
    )
    b1 = build_v9_t1_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=tmp_path / "runs",
        condition="T1-B1",
        git_commit="fixed-commit",
    )
    base = Path(scene["base_path"])
    assert Path(_option(b0.command, "--contrastive_feature_point_cloud_path")) == (
        base / "saga/contrastive_feature_point_cloud.ply"
    ).resolve()
    assert Path(_option(b0.command, "--scale_gate_path")) == (
        base / "saga/scale_gate.pt"
    ).resolve()
    assert _option(b0.command, "--teacher-prior-mode") == "original"
    assert _option(b0.command, "--v7-causal-ablation") == "L0"
    assert "--disable_other_classes" in b0.command
    assert "--disable_other_classes" not in b1.command
    assert b1.identity["git_commit"] == "fixed-commit"
    assert b1.identity["contributor_weight"] == "alpha_times_t_prev"
    assert b1.identity["input_budget"] == "existing-scene-feature-2k"
    forbidden = {"--iterations", "--clean", "--gt-dir"}
    assert forbidden.isdisjoint(b1.command)


def test_t1_complete_requires_identity_strict_contract_and_full_trace(
    tmp_path: Path,
) -> None:
    _, workspace, scene = _fixture(tmp_path)
    invocation = build_v9_t1_invocation(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=tmp_path / "runs",
        condition="T1-B1",
        git_commit="fixed-commit",
    )
    _write_complete(invocation)
    write_json(
        invocation.paths.record,
        {
            "kind": "v9_t1_legacy_run",
            "status": "complete",
            "identity": dict(invocation.identity),
        },
    )
    assert v9_t1_run_complete(invocation.paths, invocation.identity)
    changed_identity = {**dict(invocation.identity), "git_commit": "other"}
    assert not v9_t1_run_complete(invocation.paths, changed_identity)

    with np.load(invocation.paths.stage_trace, allow_pickle=False) as arrays:
        partial = {
            name: np.asarray(arrays[name])
            for name in arrays.files
            if name != "post_global_knn"
        }
    np.savez_compressed(invocation.paths.stage_trace, **partial)
    assert not v9_t1_run_complete(invocation.paths, invocation.identity)


def test_t1_execution_is_ordered_resumable_and_never_trains(tmp_path: Path) -> None:
    manifest, workspace, scene = _fixture(tmp_path)
    calls: list[tuple[str, str]] = []

    def executor(invocation: V9T1Invocation) -> int:
        calls.append((invocation.scene_id, invocation.condition))
        _write_complete(invocation)
        return 0

    kwargs = {
        "scene_manifest": manifest,
        "output_root": tmp_path / "runs",
        "workspace": workspace,
        "git_commit": "fixed-commit",
        "scene_ids": [scene["scene_id"]],
        "cgroup_root": None,
        "disk_floor_gib": 0.0,
        "executor": executor,
    }
    first = execute_v9_t1_runs(**kwargs)
    second = execute_v9_t1_runs(**kwargs)
    recovered = registered_v9_t1_batch(
        tmp_path / "runs", scene_ids=[scene["scene_id"]]
    )
    assert calls == [
        (scene["scene_id"], "T1-B0"),
        (scene["scene_id"], "T1-B1"),
    ]
    assert first["complete"] == 2
    assert [row["status"] for row in second["runs"]] == [
        "skipped_complete",
        "skipped_complete",
    ]
    assert recovered is not None
    assert recovered["producer_git_commit"] == "fixed-commit"
    assert all(row["status"] == "registered_complete" for row in recovered["runs"])
    assert (tmp_path / "runs/execution_summary.json").is_file()
