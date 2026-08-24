from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.v9_feature_training import (
    V9_FEATURE_ITERATIONS,
    V9_FEATURE_SEED,
    build_v9_affinity_scale_command,
    build_v9_feature_training_command,
    execute_v9_feature_training,
    materialize_v9_sam_everything_masks,
    prepare_v9_affinity_inputs,
    resolve_v9_feature_inputs,
    validate_v8_sam_everything_source,
    v9_affinity_input_paths,
    v9_feature_training_paths,
)


def _touch(path: Path, payload: bytes = b"data") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _ply_payload(*, feature: bool, vertex_count: int = 1) -> bytes:
    properties = ["x", "y", "z"]
    if feature:
        properties.extend(f"f_{index}" for index in range(32))
        properties.extend(f"sf_{index}" for index in range(32))
        properties.append("opacity")
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {vertex_count}",
        *(f"property float {name}" for name in properties),
        "end_header",
    ]
    row = " ".join("0" for _ in properties)
    return ("\n".join(header + [row] * vertex_count) + "\n").encode("ascii")


def _scale_gate_zip(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scale_gate/data.pkl", b"state-dict")
        archive.writestr("scale_gate/data/0", b"weights")
        archive.writestr("scale_gate/data/1", b"bias")
    return path


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    base = tmp_path / "scene"
    _touch(base / "fastRecon/dense/sparse/0/images/frame.jpg")
    (base / "fastRecon/dense/sparse/0").mkdir(parents=True, exist_ok=True)
    _touch(
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        _ply_payload(feature=False),
    )
    python_bin = _touch(tmp_path / "env/bin/python", b"python")
    affinity_masks = base / "v9/sam_everything_masks"
    affinity_scales = base / "v9/sam_everything_mask_scales"
    grounded_masks = base / "saga/masks"
    grounded_labels = base / "saga/labels"
    grounded_scales = base / "saga/mask_scales"
    for directory in (
        affinity_masks,
        affinity_scales,
        grounded_masks,
        grounded_labels,
        grounded_scales,
    ):
        _touch(directory / "frame.pt")
    _touch(grounded_labels / "label_features.pt", b"features")
    workspace = tmp_path / "workspace"
    _touch(workspace / "train_contrastive_feature.py", b"print('trainer')\n")
    scene = {
        "scene_id": "scene0001_00",
        "base_path": str(base),
        "scene_scale_m_per_unit": 1.0,
        "python_bin": str(python_bin),
        "sam_everything_masks_path": str(affinity_masks),
        "sam_everything_mask_scales_path": str(affinity_scales),
    }
    return scene, workspace, base


def _manifest(path: Path, scene: dict) -> Path:
    path.write_text(
        json.dumps({"kind": "scene_runtime_manifest", "scenes": [scene]}),
        encoding="utf-8",
    )
    return path


class _FakeTensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def bool(self):
        return _FakeTensor(self.value.astype(np.bool_))

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeTorch:
    bool = np.dtype(np.bool_)

    @staticmethod
    def from_numpy(value):
        return _FakeTensor(value)

    @staticmethod
    def save(tensor, path):
        with Path(path).open("wb") as handle:
            np.save(handle, tensor.value, allow_pickle=False)

    @staticmethod
    def load(path, map_location=None):
        del map_location
        with Path(path).open("rb") as handle:
            return _FakeTensor(np.load(handle, allow_pickle=False))


def _packed_sam_source(root: Path, masks: np.ndarray) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    masks = np.asarray(masks, dtype=np.bool_)
    count, height, width = masks.shape
    np.savez_compressed(
        root / "frame.npz",
        packed=np.packbits(masks.reshape(count, height * width), axis=1),
        count=np.asarray(count, dtype=np.int32),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
    )
    (root / "summary.json").write_text(json.dumps({
        "schema": "saga-v8-segment-everything-v1",
        "image_count": 1,
        "mask_count": int(count),
        "images": [{
            "image": "frame.jpg",
            "height": int(height),
            "width": int(width),
            "mask_count": int(count),
        }],
    }), encoding="utf-8")
    return root


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_v9_command_separates_affinity_and_semantic_supervision(
    tmp_path: Path,
) -> None:
    scene, workspace, base = _fixture(tmp_path)
    command, paths, identity = build_v9_feature_training_command(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=tmp_path / "feature-10k-objectbank",
    )
    inputs = resolve_v9_feature_inputs(scene)
    assert Path(_option(command, "--masks_path")) == inputs.affinity_masks
    assert Path(_option(command, "--mask_scales_path")) == inputs.affinity_mask_scales
    assert Path(_option(command, "--semantic_masks_path")) == inputs.semantic_masks
    assert Path(_option(command, "--semantic_labels_path")) == inputs.semantic_labels
    assert Path(_option(command, "--semantic_mask_scales_path")) == inputs.semantic_mask_scales
    assert Path(_option(command, "--semantic_label_features_path")) == inputs.semantic_label_features
    assert _option(command, "--iterations") == str(V9_FEATURE_ITERATIONS)
    assert _option(command, "--seed") == str(V9_FEATURE_SEED)
    assert paths.feature_ply != base / "saga/contrastive_feature_point_cloud.ply"
    assert paths.scale_gate != base / "saga/scale_gate.pt"
    assert identity["inputs"]["frame_stems"] == ["frame"]
    assert "sha256" not in json.dumps(identity)


def test_v9_requires_explicit_sam_everything_source(tmp_path: Path) -> None:
    scene, _, _ = _fixture(tmp_path)
    scene.pop("sam_everything_masks_path")
    with pytest.raises(ValueError, match="sam_everything_masks_path"):
        resolve_v9_feature_inputs(scene)


def test_v9_rejects_frame_identity_mismatch(tmp_path: Path) -> None:
    scene, workspace, _ = _fixture(tmp_path)
    _touch(Path(scene["sam_everything_masks_path"]) / "extra.pt")
    with pytest.raises(ValueError, match="frame identity differs"):
        build_v9_feature_training_command(
            workspace=workspace,
            scene=scene,
            scene_id=scene["scene_id"],
            output_root=tmp_path / "feature-10k-objectbank",
        )


def test_v9_allows_grounded_sam_frame_abstention(tmp_path: Path) -> None:
    scene, workspace, base = _fixture(tmp_path)
    (base / "saga/masks/frame.pt").unlink()
    (base / "saga/labels/frame.pt").unlink()
    (base / "saga/mask_scales/frame.pt").unlink()
    _command, _paths, identity = build_v9_feature_training_command(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=tmp_path / "feature-10k-objectbank",
    )
    assert identity["inputs"]["frame_count"] == 1
    assert identity["inputs"]["semantic_frame_count"] == 0
    assert identity["inputs"]["semantic_abstention_frame_count"] == 1


def test_v9_rejects_partial_grounded_sam_triplet(tmp_path: Path) -> None:
    scene, workspace, base = _fixture(tmp_path)
    (base / "saga/labels/frame.pt").unlink()
    with pytest.raises(ValueError, match="Grounded-SAM masks"):
        build_v9_feature_training_command(
            workspace=workspace,
            scene=scene,
            scene_id=scene["scene_id"],
            output_root=tmp_path / "feature-10k-objectbank",
        )


def test_v9_complete_run_skips_only_matching_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, workspace, _ = _fixture(tmp_path)
    manifest = _manifest(tmp_path / "runtime.json", scene)
    output_root = tmp_path / "feature-10k-objectbank"
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        paths = v9_feature_training_paths(output_root, scene["scene_id"])
        _touch(paths.feature_ply, _ply_payload(feature=True))
        _scale_gate_zip(paths.scale_gate)
        paths.progress.write_text("100", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "category_priors.v9_feature_training.subprocess.run", fake_run
    )
    first = execute_v9_feature_training(
        scene_manifest=manifest,
        output_root=output_root,
        workspace=workspace,
        git_commit="abc",
        scene_ids=[scene["scene_id"]],
    )
    second = execute_v9_feature_training(
        scene_manifest=manifest,
        output_root=output_root,
        workspace=workspace,
        git_commit="abc",
        scene_ids=[scene["scene_id"]],
    )
    assert first["complete"] == 1
    assert second["runs"][0]["status"] == "skipped_complete"
    assert len(calls) == 1

    semantic_mask = Path(scene["base_path"]) / "saga/masks/frame.pt"
    semantic_mask.write_bytes(b"different-size")
    with pytest.raises(RuntimeError, match="different input or code identity"):
        execute_v9_feature_training(
            scene_manifest=manifest,
            output_root=output_root,
            workspace=workspace,
            git_commit="abc",
            scene_ids=[scene["scene_id"]],
        )


def test_v9_refuses_unattributed_existing_outputs(tmp_path: Path) -> None:
    scene, workspace, _ = _fixture(tmp_path)
    manifest = _manifest(tmp_path / "runtime.json", scene)
    output_root = tmp_path / "feature-10k-objectbank"
    paths = v9_feature_training_paths(output_root, scene["scene_id"])
    _touch(paths.feature_ply, b"ply\nformat ascii 1.0\nend_header\n")
    with pytest.raises(RuntimeError, match="without a run record"):
        execute_v9_feature_training(
            scene_manifest=manifest,
            output_root=output_root,
            workspace=workspace,
            git_commit="abc",
            scene_ids=[scene["scene_id"]],
            dry_run=True,
        )


def test_v9_never_marks_header_only_ply_or_arbitrary_scale_bytes_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, workspace, _ = _fixture(tmp_path)
    manifest = _manifest(tmp_path / "runtime.json", scene)
    output_root = tmp_path / "feature-10k-objectbank"

    def fake_run(command, **kwargs):
        del command, kwargs
        paths = v9_feature_training_paths(output_root, scene["scene_id"])
        _touch(paths.feature_ply, b"ply\nformat ascii 1.0\nend_header\n")
        _touch(paths.scale_gate, b"not-a-torch-container")
        paths.progress.write_text("100", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "category_priors.v9_feature_training.subprocess.run", fake_run
    )
    result = execute_v9_feature_training(
        scene_manifest=manifest,
        output_root=output_root,
        workspace=workspace,
        git_commit="abc",
        scene_ids=[scene["scene_id"]],
        continue_on_error=True,
    )
    assert result["failed"] == 1


def test_v9_materializes_v8_masks_losslessly_and_resumes(tmp_path: Path) -> None:
    masks = np.asarray([
        [[1, 0, 1], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 1]],
    ], dtype=np.bool_)
    source = _packed_sam_source(tmp_path / "packed", masks)
    target = tmp_path / "materialized"
    identity = validate_v8_sam_everything_source(source)
    first = materialize_v9_sam_everything_masks(
        source, target, torch_module=_FakeTorch
    )
    second = materialize_v9_sam_everything_masks(
        source, target, torch_module=_FakeTorch
    )
    observed = _FakeTorch.load(target / "frame.pt").numpy()
    assert identity["mask_count"] == 2
    assert observed.dtype == np.bool_
    assert np.array_equal(observed, masks)
    assert first["written"] == 1 and first["reused"] == 0
    assert second["written"] == 0 and second["reused"] == 1


def test_v9_packed_summary_count_must_match_payload(tmp_path: Path) -> None:
    source = _packed_sam_source(
        tmp_path / "packed", np.ones((2, 2, 2), dtype=np.bool_)
    )
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    summary["images"][0]["mask_count"] = 3
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="summary/frame mismatch"):
        validate_v8_sam_everything_source(source)


def test_v9_native_packed_summary_is_accepted(tmp_path: Path) -> None:
    source = _packed_sam_source(
        tmp_path / "packed", np.asarray([[[True, False]], [[False, True]]])
    )
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    summary["schema"] = "saga-v9-segment-everything-v1"
    (source / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    identity = validate_v8_sam_everything_source(source)
    assert identity["frame_count"] == 1
    assert identity["mask_count"] == 2


def test_v9_prepares_affinity_scales_in_isolated_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene, workspace, base = _fixture(tmp_path)
    _touch(workspace / "get_scale.py", b"print('scale')\n")
    source = _packed_sam_source(
        tmp_path / "packed",
        np.asarray([
            [[1, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 1]],
        ], dtype=np.bool_),
    )
    output_root = tmp_path / "feature-10k-objectbank"
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        commands.append(list(command))
        paths = v9_affinity_input_paths(output_root, scene["scene_id"])
        paths.mask_scales.mkdir(parents=True, exist_ok=True)
        _FakeTorch.save(_FakeTensor(np.asarray([0.1, 0.2])), paths.mask_scales / "frame.pt")
        paths.scale_progress.write_text("100", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "category_priors.v9_feature_training.subprocess.run", fake_run
    )
    command, paths = build_v9_affinity_scale_command(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        output_root=output_root,
    )
    assert command[1] == str((workspace / "get_scale.py").resolve())
    assert Path(_option(command, "--masks_path")) == paths.masks
    assert Path(_option(command, "--mask_scales_path")) == paths.mask_scales
    assert paths.mask_scales != base / "saga/mask_scales"

    first = prepare_v9_affinity_inputs(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        packed_masks_root=source,
        output_root=output_root,
        git_commit="abc",
        torch_module=_FakeTorch,
    )
    second = prepare_v9_affinity_inputs(
        workspace=workspace,
        scene=scene,
        scene_id=scene["scene_id"],
        packed_masks_root=source,
        output_root=output_root,
        git_commit="abc",
        torch_module=_FakeTorch,
    )
    assert first["status"] == "complete"
    assert second["status"] == "skipped_complete"
    assert len(commands) == 1
    assert first["scene_overrides"] == {
        "sam_everything_masks_path": str(paths.masks),
        "sam_everything_mask_scales_path": str(paths.mask_scales),
    }


def test_trainer_dual_source_plumbing_is_syntax_valid() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "train_contrastive_feature.py"
    ).read_text(encoding="utf-8")
    ast.parse(source)
    for option in (
        "--semantic_masks_path",
        "--semantic_labels_path",
        "--semantic_mask_scales_path",
        "--semantic_label_features_path",
        "--seed",
    ):
        assert option in source
