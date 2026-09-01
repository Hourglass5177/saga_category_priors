from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline import (
    build_alpha_mask_evidence,
    evidence_bank_is_complete,
    load_evidence_bank,
)
from category_priors.clean_baseline.worker import (
    CleanSceneInputs,
    RenderedFrameEvidence,
    RenderedMaskSupport,
)


def _inputs(base: Path, frame_count: int = 3) -> CleanSceneInputs:
    rgb = base / "points.ply"
    sparse = base / "sparse"
    images = base / "images"
    sam = base / "sam"
    grounded_masks = base / "grounded-masks"
    grounded_labels = base / "grounded-labels"
    rgb.write_bytes(b"immutable-ply")
    for directory in (sparse, images, sam, grounded_masks, grounded_labels):
        directory.mkdir(parents=True, exist_ok=True)
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 7, 1, 1, 1))
        handle.write(struct.pack("<dddd", 1.0, 1.0, 0.0, 0.0))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", frame_count))
        for index in range(frame_count):
            handle.write(
                struct.pack(
                    "<idddddddi",
                    index + 1,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    7,
                )
            )
            handle.write(f"frame-{index}.jpg".encode("ascii") + b"\x00")
            handle.write(struct.pack("<Q", 0))
            (images / f"frame-{index}.jpg").write_bytes(b"rgb")
            (sam / f"frame-{index}.npz").write_bytes(b"sam")
            (grounded_masks / f"frame-{index}.pt").write_bytes(b"mask")
            (grounded_labels / f"frame-{index}.pt").write_bytes(b"label")
    return CleanSceneInputs(
        base_path=base,
        rgb_ply=rgb,
        sparse=sparse,
        images=images,
        sam_masks=sam,
        grounded_masks=grounded_masks,
        grounded_labels=grounded_labels,
    )


def _record(frame_id: int) -> RenderedFrameEvidence:
    posterior = np.zeros(32, dtype=np.float32)
    posterior[0] = 1.0
    support = RenderedMaskSupport(
        mask_index=0,
        gaussian_ids=np.asarray([frame_id], dtype=np.int32),
        inside_mass=np.asarray([1.0], dtype=np.float32),
        inside_ratio=np.asarray([1.0], dtype=np.float32),
        ambiguous_ids=np.empty(0, dtype=np.int32),
        class_probabilities=posterior,
    )
    return RenderedFrameEvidence(
        frame_id=frame_id,
        image_name=f"frame-{frame_id}.jpg",
        visible_ids=np.asarray([frame_id], dtype=np.int32),
        visible_mass=np.asarray([1.0], dtype=np.float32),
        masks=(support,),
        grounded_abstained=False,
        valid_pixel_count=1,
    )


def test_interrupted_scene_resumes_and_only_corrupt_frame_is_rendered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import worker

    base = tmp_path / "scene"
    base.mkdir()
    inputs = _inputs(base)
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    calls: list[tuple[int, ...]] = []
    interrupt = {"enabled": True}

    def render(*_args, frame_ids=None, frame_callback=None, **_kwargs):
        selected = tuple(range(3)) if frame_ids is None else tuple(frame_ids)
        calls.append(selected)
        records = []
        for frame_id in selected:
            record = _record(frame_id)
            if frame_callback is not None:
                frame_callback(xyz, record)
            records.append(record)
            if interrupt["enabled"] and frame_id == 0:
                raise RuntimeError("simulated renderer interruption")
        return xyz, tuple(records)

    monkeypatch.setattr(worker, "resolve_clean_scene_inputs", lambda *_a, **_k: inputs)
    monkeypatch.setattr(worker, "render_scene_frames", render)
    request = {
        "producer_commit": "a" * 40,
        "scene": {"scene_id": "scene0000_00", "base_path": str(base)},
    }
    output = tmp_path / "bank"
    with pytest.raises(RuntimeError, match="simulated renderer interruption"):
        build_alpha_mask_evidence(
            scene_id="scene0000_00", request=request, output_dir=output
        )

    checkpoints = tmp_path / "bank.frame-evidence"
    assert (checkpoints / "scene.npz").is_file()
    assert (checkpoints / "frames" / "00000000.npz").is_file()
    assert not (output / "masks.json").exists()

    interrupt["enabled"] = False
    build_alpha_mask_evidence(
        scene_id="scene0000_00", request=request, output_dir=output
    )
    assert calls == [(0, 1, 2), (1, 2)]
    assert evidence_bank_is_complete(output, expected_scene_id="scene0000_00")
    assert {path.name for path in output.iterdir()} == {
        "evidence.npz",
        "masks.json",
        "diagnostics.json",
    }
    bank = load_evidence_bank(output)
    assert bank.frame_count == 3
    assert bank.mask_count == 3

    # Force a final-bank rebuild, then damage exactly one sparse frame.  The
    # intact scene and other frame checkpoints must remain reusable.
    (output / "masks.json").unlink()
    (checkpoints / "frames" / "00000001.npz").write_bytes(b"damaged")
    build_alpha_mask_evidence(
        scene_id="scene0000_00", request=request, output_dir=output
    )
    assert calls[-1] == (1,)
    assert evidence_bank_is_complete(output, expected_scene_id="scene0000_00")

    # A complete final bank remains the sole completion contract and skips
    # both checkpoint inspection and rendering.
    build_alpha_mask_evidence(
        scene_id="scene0000_00", request=request, output_dir=output
    )
    assert calls[-1] == (1,)

    with np.load(
        checkpoints / "frames" / "00000000.npz", allow_pickle=False
    ) as loaded:
        assert "header_json" in loaded.files
        assert not any("pixel" in name or "contributor" in name for name in loaded.files)


def test_changed_source_identity_does_not_reuse_old_frame_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from category_priors.clean_baseline import worker

    base = tmp_path / "scene"
    base.mkdir()
    inputs = _inputs(base)
    xyz = np.eye(3, dtype=np.float32)
    calls: list[tuple[int, ...]] = []

    def render(*_args, frame_ids=None, frame_callback=None, **_kwargs):
        selected = tuple(frame_ids)
        calls.append(selected)
        records = tuple(_record(frame_id) for frame_id in selected)
        for record in records:
            frame_callback(xyz, record)
        return xyz, records

    monkeypatch.setattr(worker, "resolve_clean_scene_inputs", lambda *_a, **_k: inputs)
    monkeypatch.setattr(worker, "render_scene_frames", render)
    output = tmp_path / "bank"
    request = {
        "producer_commit": "a" * 40,
        "scene": {"scene_id": "scene0000_00", "base_path": str(base)},
    }
    build_alpha_mask_evidence(
        scene_id="scene0000_00", request=request, output_dir=output
    )
    (output / "masks.json").unlink()
    changed = dict(request)
    changed["producer_commit"] = "b" * 40
    build_alpha_mask_evidence(
        scene_id="scene0000_00", request=changed, output_dir=output
    )
    assert calls == [(0, 1, 2), (0, 1, 2)]
