from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.clean_baseline.mask_contract import (
    flatten_mask_stack,
    load_sam_mask_metadata,
    metadata_frame_from_sam_rows,
    save_sam_mask_metadata,
)
from category_priors.clean_baseline.mask_control import (
    MASK_CONTROL_STATE_SCHEMA,
    prepare_flat_mask_control_scene,
)
from category_priors.clean_baseline.sam_inputs import (
    SAM_EVERYTHING_CONFIG,
    ColmapFrameSpec,
    load_packed_mask_frame,
)
from category_priors.clean_baseline.worker import CleanSceneInputs


COMMIT = "1" * 40


def test_cv2_loader_uses_only_explicit_appended_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import mask_control

    package_root = tmp_path / "site-packages"
    (package_root / "cv2").mkdir(parents=True)
    (package_root / "cv2" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setenv(
        "SAGA_EXISTING_OPENCV_SITE_PACKAGES", str(package_root)
    )
    monkeypatch.setattr(mask_control.sys, "path", list(mask_control.sys.path))
    sentinel = object()
    calls: list[str] = []

    def fake_import(name: str):
        calls.append(name)
        if len(calls) == 1:
            raise ModuleNotFoundError("missing cv2", name="cv2")
        assert mask_control.sys.path[-1] == str(package_root.resolve())
        return sentinel

    assert mask_control._load_cv2(fake_import) is sentinel
    assert calls == ["cv2", "cv2"]


def _source_request(tmp_path: Path, checkpoint: Path) -> dict[str, object]:
    return {
        "scene": {
            "scene_id": "scene-test",
            "base_path": str(tmp_path / "base"),
            "segment_everything_root": str(tmp_path / "historical"),
        },
        "sam_generation": {
            "checkpoint": str(checkpoint),
            "sam_arch": "vit_h",
            "device": "cuda",
            "config": dict(SAM_EVERYTHING_CONFIG),
        },
    }


def _sam_rows(frame_index: int) -> list[dict[str, object]]:
    if frame_index == 0:
        return [
            {
                "segmentation": np.asarray(
                    [[1, 1, 0], [0, 0, 0]], dtype=np.bool_
                ),
                "predicted_iou": 0.8,
                "stability_score": 0.9,
                "area": 2,
            },
            {
                "segmentation": np.asarray(
                    [[0, 1, 1], [0, 0, 0]], dtype=np.bool_
                ),
                "predicted_iou": 0.9,
                "stability_score": 0.7,
                "area": 2,
            },
        ]
    return [
        {
            "segmentation": np.asarray(
                [[0, 0, 0], [1, 1, 0]], dtype=np.bool_
            ),
            "predicted_iou": 0.7,
            "stability_score": 0.95,
            "area": 2,
        }
    ]


def _save_packed(path: Path, dense: np.ndarray) -> None:
    array = np.asarray(dense, dtype=np.bool_)
    count, height, width = array.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed=np.packbits(array.reshape(count, height * width), axis=1),
            count=np.asarray(count, dtype=np.int32),
            height=np.asarray(height, dtype=np.int32),
            width=np.asarray(width, dtype=np.int32),
        )


def _install_fake_scene(tmp_path: Path, monkeypatch) -> tuple[
    dict[str, object], Path, list[int], list[Path], Path
]:
    from category_priors.clean_baseline import evidence, mask_control

    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint-v1")
    images = tmp_path / "images"
    historical = tmp_path / "historical"
    sparse = tmp_path / "sparse"
    for root in (images, historical, sparse):
        root.mkdir()
    (images / "frame0.jpg").write_bytes(b"image-0")
    (images / "frame1.jpg").write_bytes(b"image-1")
    frames = (
        ColmapFrameSpec("frame0", "frame0.jpg", 2, 3),
        ColmapFrameSpec("frame1", "frame1.jpg", 2, 3),
    )
    exact = metadata_frame_from_sam_rows(_sam_rows(0), height=2, width=3)
    _save_packed(historical / "frame0.npz", exact.dense())
    # Valid dimensions but deliberately different historical hierarchy.
    _save_packed(
        historical / "frame1.npz",
        np.asarray([[[1, 0, 0], [0, 0, 0]]], dtype=np.bool_),
    )
    inputs = CleanSceneInputs(
        base_path=tmp_path,
        rgb_ply=tmp_path / "unused.ply",
        sparse=sparse,
        images=images,
        sam_masks=historical,
        grounded_masks=tmp_path / "grounded-masks",
        grounded_labels=tmp_path / "grounded-labels",
    )
    monkeypatch.setattr(mask_control, "resolve_clean_scene_inputs", lambda _scene: inputs)
    monkeypatch.setattr(mask_control, "colmap_frame_specs", lambda _path: frames)

    def fake_evidence_source(*, scene_id: str, request: dict[str, object]):
        mask_root = Path(str(request["sam_masks"])).resolve()
        files = sorted(mask_root.glob("*.npz"), key=lambda path: path.name)
        return {
            "worker": "test-worker",
            "producer_commit": str(request["producer_commit"]),
            "sam_masks": str(mask_root),
            "mask_observation_mode": str(request["mask_observation_mode"]),
            "producer_inputs": {
                "sam_everything_masks": {
                    "path": str(mask_root),
                    "kind": "registered-sam-masks",
                    "file_count": len(files),
                    "relative_paths": [path.name for path in files],
                    "manifest_sha256": "f" * 64,
                }
            },
        }

    monkeypatch.setattr(evidence, "evidence_request_source", fake_evidence_source)

    factory_calls: list[int] = []
    loaded_images: list[Path] = []

    class Generator:
        def generate(self, image: np.ndarray):
            return _sam_rows(int(image[0, 0, 0]))

    def factory(*_args):
        factory_calls.append(1)
        return Generator()

    def image_loader(path: Path) -> np.ndarray:
        loaded_images.append(path)
        value = 0 if path.name == "frame0.jpg" else 1
        return np.full((2, 3, 3), value, dtype=np.uint8)

    request = _source_request(tmp_path, checkpoint)
    output = tmp_path / "control"
    return request, output, factory_calls, loaded_images, checkpoint


def _run(
    request: dict[str, object],
    output: Path,
    factory_calls: list[int],
    loaded_images: list[Path],
    *,
    commit: str = COMMIT,
) -> dict[str, object]:
    class Generator:
        def generate(self, image: np.ndarray):
            return _sam_rows(int(image[0, 0, 0]))

    def factory(*_args):
        factory_calls.append(1)
        return Generator()

    def loader(path: Path) -> np.ndarray:
        loaded_images.append(path)
        value = 0 if path.name == "frame0.jpg" else 1
        return np.full((2, 3, 3), value, dtype=np.uint8)

    return prepare_flat_mask_control_scene(
        scene_id="scene-test",
        source_request=request,
        output_root=output,
        producer_commit=commit,
        generator_factory=factory,
        image_loader=loader,
    )


def test_one_generation_derives_paired_hierarchy_and_flat_and_resumes(
    tmp_path: Path, monkeypatch,
) -> None:
    request, output, factories, images, _checkpoint = _install_fake_scene(
        tmp_path, monkeypatch
    )
    first = _run(request, output, factories, images)
    assert first["generated_frame_count"] == 2
    assert first["mechanical_contract_pass"] is True
    assert first["input_binding_pass"] is True
    assert set(first["input_bindings"]) == {"H-hierarchy", "P-flat"}
    assert first["historical_hierarchy_exact_frame_count"] == 1
    assert len(factories) == 1
    assert [path.name for path in images] == ["frame0.jpg", "frame1.jpg"]

    state_path = output / "sam-metadata/scene-test/generation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state) == {"schema", "request", "metadata_files"}
    assert state["schema"] == MASK_CONTROL_STATE_SCHEMA
    assert set(state["metadata_files"]) == {"frame0", "frame1"}
    assert not list(output.rglob("*.part"))

    for frame_name in ("frame0", "frame1"):
        metadata = load_sam_mask_metadata(
            output / f"sam-metadata/scene-test/{frame_name}.npz"
        )
        hierarchy = load_packed_mask_frame(
            output / f"masks/H-hierarchy/scene-test/{frame_name}.npz"
        )
        flat = flatten_mask_stack(metadata)
        packed_flat = load_packed_mask_frame(
            output / f"masks/P-flat/scene-test/{frame_name}.npz"
        )
        assert hierarchy.count == metadata.count
        assert np.array_equal(hierarchy.packed, metadata.packed)
        assert np.array_equal(packed_flat.packed, flat.frame.packed)
        with np.load(
            output / f"flat-maps/scene-test/{frame_name}.npz",
            allow_pickle=False,
        ) as mapping:
            assert np.array_equal(mapping["source_mask_ids"], flat.source_mask_ids)

    hierarchy_request = json.loads(
        Path(first["hierarchy_evidence_request"]).read_text(encoding="utf-8")
    )
    flat_request = json.loads(
        Path(first["flat_evidence_request"]).read_text(encoding="utf-8")
    )
    assert hierarchy_request["mask_observation_mode"] == "hierarchy"
    assert flat_request["mask_observation_mode"] == "flat-highest-quality"
    assert hierarchy_request["sam_masks"] != flat_request["sam_masks"]
    assert hierarchy_request["producer_commit"] == flat_request["producer_commit"]

    before_metadata = {
        path.name: path.read_bytes()
        for path in (output / "sam-metadata/scene-test").glob("*.npz")
    }
    second = _run(request, output, factories, images)
    assert second["generated_frame_count"] == 0
    assert len(factories) == 1
    assert len(images) == 2
    assert before_metadata == {
        path.name: path.read_bytes()
        for path in (output / "sam-metadata/scene-test").glob("*.npz")
    }


def test_shape_valid_but_wrong_derived_files_are_repaired_without_regeneration(
    tmp_path: Path, monkeypatch,
) -> None:
    request, output, factories, images, _checkpoint = _install_fake_scene(
        tmp_path, monkeypatch
    )
    _run(request, output, factories, images)
    metadata = load_sam_mask_metadata(
        output / "sam-metadata/scene-test/frame0.npz"
    )
    expected_flat = flatten_mask_stack(metadata)
    # Both payloads remain structurally valid and have the expected row count,
    # yet contain the wrong bits.  A shape-only resume check would miss this.
    wrong_h = np.zeros((metadata.count, metadata.height, metadata.width), dtype=np.bool_)
    wrong_h[:, 1, 2] = True
    _save_packed(output / "masks/H-hierarchy/scene-test/frame0.npz", wrong_h)
    wrong_p = np.zeros(
        (expected_flat.frame.count, metadata.height, metadata.width), dtype=np.bool_
    )
    wrong_p[:, 1, 1] = True
    _save_packed(output / "masks/P-flat/scene-test/frame0.npz", wrong_p)
    flat_map_path = output / "flat-maps/scene-test/frame0.npz"
    with flat_map_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            source_mask_ids=expected_flat.source_mask_ids[::-1].astype(np.int32),
        )

    resumed = _run(request, output, factories, images)
    assert resumed["generated_frame_count"] == 0
    assert len(factories) == 1
    assert len(images) == 2
    repaired_h = load_packed_mask_frame(
        output / "masks/H-hierarchy/scene-test/frame0.npz"
    )
    repaired_p = load_packed_mask_frame(
        output / "masks/P-flat/scene-test/frame0.npz"
    )
    assert np.array_equal(repaired_h.packed, metadata.packed)
    assert np.array_equal(repaired_p.packed, expected_flat.frame.packed)
    with np.load(flat_map_path, allow_pickle=False) as mapping:
        assert np.array_equal(
            mapping["source_mask_ids"], expected_flat.source_mask_ids
        )


def test_shape_valid_source_metadata_with_wrong_content_is_regenerated_by_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    request, output, factories, images, _checkpoint = _install_fake_scene(
        tmp_path, monkeypatch
    )
    _run(request, output, factories, images)
    metadata_path = output / "sam-metadata/scene-test/frame0.npz"
    wrong = metadata_frame_from_sam_rows(_sam_rows(1), height=2, width=3)
    save_sam_mask_metadata(metadata_path, wrong)
    resumed = _run(request, output, factories, images)
    assert resumed["generated_frame_count"] == 1
    assert len(factories) == 2
    assert [path.name for path in images].count("frame0.jpg") == 2
    repaired = load_sam_mask_metadata(metadata_path)
    expected = metadata_frame_from_sam_rows(_sam_rows(0), height=2, width=3)
    assert np.array_equal(repaired.packed, expected.packed)


@pytest.mark.parametrize("mutation", ["schema", "commit", "state-config"])
def test_state_schema_commit_and_config_identity_are_strict(
    tmp_path: Path, monkeypatch, mutation: str,
) -> None:
    request, output, factories, images, _checkpoint = _install_fake_scene(
        tmp_path, monkeypatch
    )
    _run(request, output, factories, images)
    state_path = output / "sam-metadata/scene-test/generation_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    kwargs: dict[str, object] = {}
    if mutation == "schema":
        state["schema"] = "wrong-schema"
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif mutation == "commit":
        kwargs["commit"] = "2" * 40
    else:
        state["request"]["config"]["points_per_side"] = 64
        state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schema|different request"):
        _run(request, output, factories, images, **kwargs)
    assert len(factories) == 1
    assert len(images) == 2


def test_nonfrozen_generation_config_is_rejected_before_generation(
    tmp_path: Path, monkeypatch,
) -> None:
    request, output, factories, images, _checkpoint = _install_fake_scene(
        tmp_path, monkeypatch
    )
    request["sam_generation"]["config"]["points_per_side"] = 64
    with pytest.raises(ValueError, match="frozen configuration"):
        _run(request, output, factories, images)
    assert factories == []
    assert images == []
