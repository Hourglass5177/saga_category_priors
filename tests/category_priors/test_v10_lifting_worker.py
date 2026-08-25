from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from category_priors.io import load_json, write_json
from category_priors.v10_lifting_worker import (
    V10_LIFTING_IDENTITY_SCHEMA,
    V10_LIFTING_SCHEMA,
    build_v10_lifting_identity,
    compatible_lifting_bank_is_complete,
    ensure_v10_lifting_banks,
    resolve_v10_lifting_inputs,
    run_v10_lifting_bank,
    v10_lifting_bank_is_complete,
)
from category_priors.v10_runner import _lifting_header
from category_priors.v9_feature_training import SAM_EVERYTHING_CONFIG
from category_priors.v9_lifting import FragmentConfig, V9_LIFTING_SCHEMA


def _touch(path: Path, value: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _inputs(tmp_path: Path, scene_id: str = "scene0000_00") -> tuple[Path, dict[str, Any]]:
    base = tmp_path / scene_id
    _touch(
        base / "output_models/point_cloud/iteration_30000/point_cloud.ply",
        b"rgb-ply",
    )
    feature = _touch(base / "trained/features.ply", b"trained-feature")
    _touch(base / "fastRecon/dense/sparse/0/cameras.bin", b"cameras")
    _touch(base / "fastRecon/dense/sparse/0/images.bin", b"poses")
    _touch(base / "fastRecon/dense/sparse/0/images/frame.jpg", b"image")
    _touch(base / "saga/masks/frame.pt", b"grounded-mask")
    _touch(base / "saga/labels/frame.pt", b"grounded-label")
    labels = _touch(base / "saga/labels/label_features.pt", b"codebook")
    sam = base / "sam-packed"
    sam.mkdir(parents=True)
    masks = np.asarray(
        [[True, False, False, True], [False, True, True, False]], dtype=np.bool_
    )
    np.savez_compressed(
        sam / "frame.npz",
        packed=np.packbits(masks, axis=1),
        count=np.asarray(2, dtype=np.int32),
        height=np.asarray(2, dtype=np.int32),
        width=np.asarray(2, dtype=np.int32),
    )
    write_json(
        sam / "summary.json",
        {
            "schema": "saga-v9-segment-everything-v1",
            "image_root": str((base / "fastRecon/dense/sparse/0/images").resolve()),
            "output_root": str(sam.resolve()),
            "sam_arch": "vit_h",
            "config": SAM_EVERYTHING_CONFIG,
            "image_count": 1,
            "mask_count": 2,
            "images": [
                {
                    "image": "frame.jpg",
                    "height": 2,
                    "width": 2,
                    "mask_count": 2,
                }
            ],
        },
    )
    scene = {
        "scene_id": scene_id,
        "base_path": str(base),
        "scene_scale_m_per_unit": 1.0,
        "feature_ply_path": str(feature),
        "label_features_path": str(labels),
        "segment_everything_root": str(sam),
    }
    manifest = tmp_path / f"{scene_id}-runtime.json"
    write_json(manifest, {"kind": "scene_runtime_manifest", "scenes": [scene]})
    return manifest, scene


def _arrays(path: Path) -> None:
    np.savez_compressed(
        path,
        xyz_m=np.zeros((3, 3), dtype=np.float32),
        affinity=np.ones((3, 2), dtype=np.float32),
        semantic=np.ones((3, 2), dtype=np.float32),
        label_features=np.asarray([[1.0, 0.0]], dtype=np.float32),
        fragment_full_indptr=np.asarray([0, 1], dtype=np.int64),
        fragment_full_ids=np.asarray([0], dtype=np.int32),
        fragment_full_mass=np.asarray([1.0], dtype=np.float32),
        fragment_core_indptr=np.asarray([0, 1], dtype=np.int64),
        fragment_core_ids=np.asarray([0], dtype=np.int32),
        fragment_core_mass=np.asarray([1.0], dtype=np.float32),
        fragment_id=np.asarray([0], dtype=np.int64),
        fragment_frame=np.asarray([0], dtype=np.int32),
        fragment_mask_index=np.asarray([0], dtype=np.int32),
        fragment_conflict_ratio=np.asarray([0.0], dtype=np.float32),
        frame_visible_indptr=np.asarray([0, 1], dtype=np.int64),
        frame_visible_ids=np.asarray([0], dtype=np.int32),
        frame_visible_mass=np.asarray([1.0], dtype=np.float32),
        frame_geometry_abstained=np.asarray([False]),
        frame_grounded_missing=np.asarray([False]),
        semantic_fragment_full_indptr=np.asarray([0, 1], dtype=np.int64),
        semantic_fragment_full_ids=np.asarray([0], dtype=np.int32),
        semantic_fragment_full_mass=np.asarray([1.0], dtype=np.float32),
        semantic_fragment_frame=np.asarray([0], dtype=np.int32),
        semantic_fragment_class=np.asarray([0], dtype=np.int16),
    )


def _write_bank(
    target: Path,
    *,
    schema: str,
    scene_id: str = "scene0000_00",
    identity: dict[str, Any] | None = None,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if identity is None:
        identity = {
            "schema": (
                "saga-v9-native-lifting-identity-v1"
                if schema == V9_LIFTING_SCHEMA
                else V10_LIFTING_IDENTITY_SCHEMA
            ),
            "scene_id": scene_id,
            "git_commit": "producer",
        }
    write_json(
        target / "lifting_bank.json",
        {
            "schema": schema,
            "scene_id": scene_id,
            "git_commit": str(identity["git_commit"]),
            "identity": identity,
            "lifting_source": "M1-core+AM-full",
            "mask_source": "SAM-everything",
            "feature_source": (
                "v9-10k-objectbank"
                if schema == V9_LIFTING_SCHEMA
                else "runtime-manifest-trained-feature"
            ),
            "config": vars(FragmentConfig()),
            "point_count": 3,
            "fragment_count": 1,
            "frame_count": 1,
            "semantic_fragment_count": 1,
            "classes": ["chair"],
        },
    )
    _arrays(target / "lifting_bank.npz")


def _contains_sha(value: Any) -> bool:
    if isinstance(value, dict):
        return any("sha" in str(key).lower() or _contains_sha(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sha(item) for item in value)
    return False


def test_v10_identity_records_paths_sizes_mtimes_without_training_record_or_hash(
    tmp_path: Path,
) -> None:
    _manifest, scene = _inputs(tmp_path)
    inputs = resolve_v10_lifting_inputs(scene)
    identity = build_v10_lifting_identity(
        scene_id=scene["scene_id"],
        git_commit="consumer",
        inputs=inputs,
        classes=("chair",),
    )

    assert identity["schema"] == V10_LIFTING_IDENTITY_SCHEMA
    assert identity["feature_ply"] == {
        "path": str(inputs.feature_ply),
        "size_bytes": inputs.feature_ply.stat().st_size,
        "mtime_ns": inputs.feature_ply.stat().st_mtime_ns,
    }
    assert identity["grounded_masks"]["files"][0]["relative_path"] == "frame.pt"
    assert "feature_record" not in json.dumps(identity)
    assert not _contains_sha(identity)


def test_v9_and_v10_lifting_headers_are_the_only_accepted_schemas(tmp_path: Path) -> None:
    v9 = tmp_path / "v9"
    v10 = tmp_path / "v10"
    _write_bank(v9, schema=V9_LIFTING_SCHEMA)
    _write_bank(v10, schema=V10_LIFTING_SCHEMA)

    assert compatible_lifting_bank_is_complete(v9, expected_scene_id="scene0000_00")
    assert compatible_lifting_bank_is_complete(v10, expected_scene_id="scene0000_00")
    assert _lifting_header(v9)["schema"] == V9_LIFTING_SCHEMA
    assert _lifting_header(v10)["schema"] == V10_LIFTING_SCHEMA

    metadata = json.loads((v10 / "lifting_bank.json").read_text("utf-8"))
    metadata["schema"] = "saga-v8-lifting-bank-v1"
    write_json(v10 / "lifting_bank.json", metadata)
    assert not compatible_lifting_bank_is_complete(v10)
    with pytest.raises(ValueError, match="V9/V10"):
        _lifting_header(v10)


def test_ensure_reuses_v9_read_only_without_resolving_or_training(tmp_path: Path) -> None:
    manifest, scene = _inputs(tmp_path)
    target = tmp_path / "lifting" / scene["scene_id"]
    _write_bank(target, schema=V9_LIFTING_SCHEMA, scene_id=scene["scene_id"])
    before = {
        name: (target / name).read_bytes()
        for name in ("lifting_bank.json", "lifting_bank.npz")
    }
    # Even missing current runtime inputs must not invalidate an already frozen,
    # structurally complete V9 producer artifact.
    Path(scene["feature_ply_path"]).unlink()

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("complete V9 lifting must not call a worker or trainer")

    result = ensure_v10_lifting_banks(
        runtime_manifest=manifest,
        scene_ids=[scene["scene_id"]],
        output_root=tmp_path / "lifting",
        git_commit="consumer",
        worker=forbidden,
    )

    assert result["runs"][0]["status"] == "reused"
    assert result["runs"][0]["schema"] == V9_LIFTING_SCHEMA
    assert {
        name: (target / name).read_bytes()
        for name in ("lifting_bank.json", "lifting_bank.npz")
    } == before


def test_ensure_writes_only_missing_scene_and_preserves_source_assets(tmp_path: Path) -> None:
    manifest, scene = _inputs(tmp_path)
    feature = Path(scene["feature_ply_path"])
    feature_before = feature.read_bytes()
    calls: list[str] = []

    def fake_worker(**kwargs: Any) -> dict[str, Any]:
        calls.append(str(kwargs["scene_id"]))
        inputs = resolve_v10_lifting_inputs(
            kwargs["scene"],
            segment_everything_root=kwargs["segment_everything_root"],
            feature_ply=kwargs["feature_ply"],
            label_features=kwargs["label_features"],
        )
        identity = build_v10_lifting_identity(
            scene_id=kwargs["scene_id"],
            git_commit=kwargs["git_commit"],
            inputs=inputs,
            classes=("chair",),
        )
        _write_bank(
            Path(kwargs["output_dir"]),
            schema=V10_LIFTING_SCHEMA,
            scene_id=kwargs["scene_id"],
            identity=identity,
        )
        return load_json(Path(kwargs["output_dir"]) / "lifting_bank.json")

    output = tmp_path / "isolated-lifting"
    result = ensure_v10_lifting_banks(
        runtime_manifest=manifest,
        scene_ids=[scene["scene_id"]],
        output_root=output,
        git_commit="consumer",
        worker=fake_worker,
    )

    assert calls == [scene["scene_id"]]
    assert result["runs"][0]["status"] == "completed"
    assert v10_lifting_bank_is_complete(output / scene["scene_id"])
    assert feature.read_bytes() == feature_before
    assert not (feature.parent / "train_10k.json").exists()


def test_missing_runtime_feature_fails_before_worker_is_called(tmp_path: Path) -> None:
    manifest, scene = _inputs(tmp_path)
    Path(scene["feature_ply_path"]).unlink()
    called = False

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("missing input must fail before GPU worker")

    with pytest.raises(FileNotFoundError, match="missing V10 lifting inputs"):
        ensure_v10_lifting_banks(
            runtime_manifest=manifest,
            scene_ids=[scene["scene_id"]],
            output_root=tmp_path / "new-lifting",
            git_commit="consumer",
            worker=forbidden,
        )
    assert called is False


def test_worker_atomically_promotes_a_complete_preserved_part(tmp_path: Path) -> None:
    _manifest, scene = _inputs(tmp_path)
    inputs = resolve_v10_lifting_inputs(scene)
    identity = build_v10_lifting_identity(
        scene_id=scene["scene_id"],
        git_commit="consumer",
        inputs=inputs,
    )
    output = tmp_path / "isolated-lifting" / scene["scene_id"]
    preserved_part = output.parent / f".{output.name}.part-complete"
    _write_bank(
        preserved_part,
        schema=V10_LIFTING_SCHEMA,
        scene_id=scene["scene_id"],
        identity=identity,
    )

    metadata = run_v10_lifting_bank(
        scene_id=scene["scene_id"],
        scene=scene,
        output_dir=output,
        git_commit="consumer",
    )

    assert metadata["identity"] == identity
    assert output.is_dir()
    assert not preserved_part.exists()
    assert v10_lifting_bank_is_complete(
        output,
        expected_scene_id=scene["scene_id"],
        expected_git_commit="consumer",
        expected_identity=identity,
    )
