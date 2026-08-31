from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from category_priors.clean_baseline.sam_inputs import (
    SAM_EVERYTHING_CONFIG,
    ColmapFrameSpec,
    PackedMaskFrame,
    audit_scene_masks,
    colmap_frame_specs,
    ensure_scene_sam_masks,
    load_packed_mask_frame,
    packed_frame_is_valid,
)


def _packed(path: Path, *, count: int, height: int, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.zeros((count, height, width), dtype=np.bool_)
    np.savez_compressed(
        path,
        packed=np.packbits(masks.reshape(count, height * width), axis=1),
        count=np.asarray(count),
        height=np.asarray(height),
        width=np.asarray(width),
    )


def test_colmap_binary_camera_names_drive_packed_frame_audit(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse"
    sparse.mkdir()
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 7, 1, 5, 4))
        handle.write(struct.pack("<dddd", 1.0, 1.0, 2.0, 2.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(
            struct.pack("<idddddddi", 11, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7)
        )
        handle.write(b"frame0001.jpg\x00")
        handle.write(struct.pack("<Q", 0))
    frames = colmap_frame_specs(sparse)
    assert frames == (ColmapFrameSpec("frame0001", "frame0001.jpg", 4, 5),)
    sam, masks, labels = tmp_path / "sam", tmp_path / "masks", tmp_path / "labels"
    for root in (sam, masks, labels):
        root.mkdir()
    _packed(sam / "frame0001.npz", count=2, height=4, width=5)
    report = audit_scene_masks(
        frames=frames,
        sam_root=sam,
        grounded_masks_root=masks,
        grounded_labels_root=labels,
    )
    assert report["complete"] is True
    assert report["grounded_abstention_allowed"] is True
    # Same filename with the wrong camera shape is not a valid packed frame.
    _packed(sam / "frame0001.npz", count=2, height=5, width=4)
    assert audit_scene_masks(
        frames=frames,
        sam_root=sam,
        grounded_masks_root=masks,
        grounded_labels_root=labels,
    )["invalid_sam_frames"] == ["frame0001"]


def test_grounded_detection_may_abstain_but_cannot_be_one_sided(tmp_path: Path) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 2, 3)
    sam, masks, labels = tmp_path / "sam", tmp_path / "masks", tmp_path / "labels"
    for root in (sam, masks, labels):
        root.mkdir()
    _packed(sam / "frame.npz", count=0, height=2, width=3)
    assert audit_scene_masks(
        frames=(frame,), sam_root=sam,
        grounded_masks_root=masks, grounded_labels_root=labels,
    )["complete"] is True
    (masks / "frame.pt").write_bytes(b"mask")
    report = audit_scene_masks(
        frames=(frame,), sam_root=sam,
        grounded_masks_root=masks, grounded_labels_root=labels,
    )
    assert report["complete"] is False
    assert report["grounded_one_sided_frames"] == ["frame"]


def test_missing_packed_frame_is_generated_only_in_isolated_root(
    tmp_path: Path, monkeypatch,
) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 2, 3)
    images = tmp_path / "images"
    primary = tmp_path / "historical-sam"
    output = tmp_path / "clean-generated-sam"
    masks, labels = tmp_path / "masks", tmp_path / "labels"
    for root in (images, primary, masks, labels):
        root.mkdir()
    (images / "frame.jpg").write_bytes(b"image")
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"existing-weight")
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            IMREAD_COLOR=1,
            COLOR_BGR2RGB=2,
            imread=lambda *_args, **_kwargs: image.copy(),
            cvtColor=lambda value, _mode: value[..., ::-1],
        ),
    )

    class Generator:
        def generate(self, _image):
            return [{"segmentation": np.ones((2, 3), dtype=np.bool_)}]

    result = ensure_scene_sam_masks(
        frames=(frame,), images_root=images, primary_root=primary,
        grounded_masks_root=masks, grounded_labels_root=labels,
        generation={
            "output_root": str(output), "checkpoint": str(checkpoint),
            "sam_arch": "vit_h", "device": "cuda",
            "config": dict(SAM_EVERYTHING_CONFIG),
        },
        generator_factory=lambda *_args: Generator(),
    )
    assert result["status"] == "complete"
    assert result["source"] == "generated-isolated"
    assert not (primary / "frame.npz").exists()
    assert packed_frame_is_valid(output / "frame.npz", height=2, width=3)
    assert (output / "generation_manifest.json").is_file()


def test_generated_cache_is_bound_to_checkpoint_and_output_bytes(
    tmp_path: Path, monkeypatch,
) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 2, 3)
    images, primary, output, masks, labels = [
        tmp_path / value
        for value in ("images", "historical-sam", "generated", "masks", "labels")
    ]
    for root in (images, primary, masks, labels):
        root.mkdir()
    (images / "frame.jpg").write_bytes(b"image")
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"weight-v1")
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            IMREAD_COLOR=1,
            COLOR_BGR2RGB=2,
            imread=lambda *_args, **_kwargs: image.copy(),
            cvtColor=lambda value, _mode: value[..., ::-1],
        ),
    )
    calls: list[int] = []

    class Generator:
        def __init__(self, fill: bool):
            self.fill = fill

        def generate(self, _image):
            calls.append(1)
            return [{"segmentation": np.full((2, 3), self.fill, dtype=np.bool_)}]

    generation = {
        "output_root": str(output), "checkpoint": str(checkpoint),
        "sam_arch": "vit_h", "device": "cuda",
        "config": dict(SAM_EVERYTHING_CONFIG),
    }
    kwargs = dict(
        frames=(frame,), images_root=images, primary_root=primary,
        grounded_masks_root=masks, grounded_labels_root=labels,
        generation=generation,
    )
    first = ensure_scene_sam_masks(
        **kwargs, generator_factory=lambda *_args: Generator(True)
    )
    assert first["generation_attempted"] is True
    assert len(calls) == 1
    second = ensure_scene_sam_masks(
        **kwargs, generator_factory=lambda *_args: Generator(False)
    )
    assert second["generation_attempted"] is False
    assert len(calls) == 1

    checkpoint.write_bytes(b"weight-v2")
    third = ensure_scene_sam_masks(
        **kwargs, generator_factory=lambda *_args: Generator(False)
    )
    assert third["generation_attempted"] is True
    assert len(calls) == 2
    assert not load_packed_mask_frame(output / "frame.npz").dense_batch(0, 1).any()


def test_missing_checkpoint_is_explicitly_unavailable_without_download(tmp_path: Path) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 1, 1)
    roots = [tmp_path / value for value in ("images", "sam", "masks", "labels")]
    for root in roots:
        root.mkdir()
    result = ensure_scene_sam_masks(
        frames=(frame,), images_root=roots[0], primary_root=roots[1],
        grounded_masks_root=roots[2], grounded_labels_root=roots[3],
        generation={
            "output_root": str(tmp_path / "generated"),
            "checkpoint": str(tmp_path / "missing.pth"),
        },
    )
    assert result["status"] == "unavailable"
    assert result["generation_attempted"] is False
    assert result["download_attempted"] is False


@pytest.mark.parametrize("output_kind", ["empty", "same", "nested", "ancestor"])
def test_generation_output_must_be_explicit_and_disjoint_from_history(
    tmp_path: Path, output_kind: str,
) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 1, 1)
    images = tmp_path / "images"
    primary = tmp_path / "historical" / "sam"
    masks, labels = tmp_path / "masks", tmp_path / "labels"
    for root in (images, primary, masks, labels):
        root.mkdir(parents=True)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"weight")
    if output_kind == "empty":
        output: str | Path = ""
    elif output_kind == "same":
        output = primary
    elif output_kind == "nested":
        output = primary / "generated"
    else:
        output = primary.parent
    result = ensure_scene_sam_masks(
        frames=(frame,), images_root=images, primary_root=primary,
        grounded_masks_root=masks, grounded_labels_root=labels,
        generation={"output_root": output, "checkpoint": checkpoint},
    )
    assert result["status"] == "unavailable"
    assert result["generation_attempted"] is False
    assert "output_root" in result["reason"]


def test_packed_masks_reject_dtype_coercion_and_nonzero_padding(tmp_path: Path) -> None:
    wrong_dtype = tmp_path / "float.npz"
    np.savez_compressed(
        wrong_dtype,
        packed=np.asarray([[128.0]], dtype=np.float32),
        count=np.asarray(1), height=np.asarray(1), width=np.asarray(1),
    )
    assert not packed_frame_is_valid(wrong_dtype, height=1, width=1)
    with pytest.raises(ValueError, match="invalid packed"):
        load_packed_mask_frame(wrong_dtype, height=1, width=1)

    bad_padding = tmp_path / "padding.npz"
    np.savez_compressed(
        bad_padding,
        packed=np.asarray([[129]], dtype=np.uint8),
        count=np.asarray(1), height=np.asarray(1), width=np.asarray(1),
    )
    assert not packed_frame_is_valid(bad_padding, height=1, width=1)
    with pytest.raises(ValueError, match="count must be one integer"):
        PackedMaskFrame(
            packed=np.zeros((1, 1), dtype=np.uint8),
            count=1.5, height=1, width=1,
        )


def test_generation_rejects_nonregistered_sam_parameters(tmp_path: Path) -> None:
    frame = ColmapFrameSpec("frame", "frame.jpg", 1, 1)
    roots = [tmp_path / value for value in ("images", "sam", "masks", "labels")]
    for root in roots:
        root.mkdir()
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"weight")
    result = ensure_scene_sam_masks(
        frames=(frame,), images_root=roots[0], primary_root=roots[1],
        grounded_masks_root=roots[2], grounded_labels_root=roots[3],
        generation={
            "output_root": str(tmp_path / "generated"),
            "checkpoint": str(checkpoint),
            "config": {**SAM_EVERYTHING_CONFIG, "points_per_side": 16},
        },
    )
    assert result["status"] == "unavailable"
    assert result["generation_attempted"] is False
    assert "frozen configuration" in result["reason"]
