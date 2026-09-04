from __future__ import annotations

from types import SimpleNamespace

import pytest

from category_priors.semantic_voting import summarize_mask_label_assets


def _camera(name: str) -> SimpleNamespace:
    return SimpleNamespace(image_name=name)


def test_semantic_vote_roots_are_required(tmp_path) -> None:
    args = SimpleNamespace(masks_path="", labels_path="")
    with pytest.raises(ValueError, match="masks_path"):
        summarize_mask_label_assets(args, [_camera("frame-000000")])

    args = SimpleNamespace(
        masks_path=str(tmp_path / "missing-masks"),
        labels_path=str(tmp_path / "missing-labels"),
    )
    with pytest.raises(FileNotFoundError, match="masks_path"):
        summarize_mask_label_assets(args, [_camera("frame-000000")])


def test_semantic_vote_assets_require_at_least_one_pair(tmp_path) -> None:
    masks = tmp_path / "masks"
    labels = tmp_path / "labels"
    masks.mkdir()
    labels.mkdir()
    args = SimpleNamespace(masks_path=str(masks), labels_path=str(labels))
    with pytest.raises(ValueError, match="no paired mask/label"):
        summarize_mask_label_assets(args, [_camera("frame-000000")])


def test_semantic_vote_assets_allow_abstention_beside_valid_pair(tmp_path) -> None:
    masks = tmp_path / "masks"
    labels = tmp_path / "labels"
    masks.mkdir()
    labels.mkdir()
    (masks / "frame-000000.pt").write_bytes(b"mask")
    (labels / "frame-000000.pt").write_bytes(b"label")
    args = SimpleNamespace(masks_path=str(masks), labels_path=str(labels))

    summary = summarize_mask_label_assets(
        args, [_camera("frame-000000"), _camera("frame-000020")]
    )

    assert summary["paired_frame_count"] == 1
    assert summary["missing_pair_frame_count"] == 1


def test_semantic_vote_assets_reject_incomplete_pair(tmp_path) -> None:
    masks = tmp_path / "masks"
    labels = tmp_path / "labels"
    masks.mkdir()
    labels.mkdir()
    (masks / "frame-000000.pt").write_bytes(b"mask")
    args = SimpleNamespace(masks_path=str(masks), labels_path=str(labels))

    with pytest.raises(FileNotFoundError, match="incomplete"):
        summarize_mask_label_assets(args, [_camera("frame-000000")])
