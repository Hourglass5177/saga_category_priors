from __future__ import annotations

import json

import numpy as np
import pytest

from category_priors.full_instance_vote import (
    VOTE_EVIDENCE_SCHEMA,
    GaussianVoteEvidence,
    _checked_add_updates,
    _evidence_from_dense,
    _file_identity,
    accumulate_frame_votes,
    aggregate_instance_votes,
    load_gaussian_vote_evidence,
    save_gaussian_vote_evidence,
    validate_rgb_feature_order,
    vote_evidence_is_complete,
)


def _evidence(dense: np.ndarray, class_names: tuple[str, ...]) -> GaussianVoteEvidence:
    point_count = dense.shape[0]
    metadata = {
        "schema": VOTE_EVIDENCE_SCHEMA,
        "status": "complete",
        "scene_id": "scene0000_00",
        "point_count": point_count,
        "class_names": list(class_names),
        "channel_count": len(class_names) + 1,
        "background_index": len(class_names),
        "total_vote_count": int(dense.sum(dtype=np.uint64)),
        "input_identity": {"fixture": 1},
    }
    return _evidence_from_dense(
        dense.astype(np.uint64, copy=False).reshape(-1),
        point_count=point_count,
        class_names=class_names,
        metadata=metadata,
    )


def test_frame_votes_reproduce_overlap_and_background_contract() -> None:
    # Pixel (0,1) lies in two masks and therefore casts two foreground votes.
    # Pixel (1,2) is outside every mask and casts one background vote for g0.
    ids = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    weights = np.asarray([[0.2, 0.3, 0.4], [0.0, 0.5, 0.7]])
    masks = np.asarray(
        [
            [[1, 1, 0], [0, 0, 0]],
            [[0, 1, 1], [1, 0, 0]],
        ],
        dtype=bool,
    )
    labels = np.asarray([0, 1])
    accumulator = np.zeros(3 * 3, dtype=np.uint64)
    diagnostics = accumulate_frame_votes(
        accumulator,
        ids,
        weights,
        masks,
        labels,
        point_count=3,
        class_count=2,
    )
    table = accumulator.reshape(3, 3)
    assert table.tolist() == [
        [1, 0, 1],  # g0: class 0 and background
        [1, 1, 1],  # g1: overlap counts twice; its other pixel is background
        [0, 1, 0],  # g2: class 1; zero-weight occurrence is excluded
    ]
    assert diagnostics["valid_contributor_pixel_count"] == 5
    assert diagnostics["total_vote_count"] == 6


def test_invalid_label_mask_is_excluded_from_foreground_and_background() -> None:
    accumulator = np.zeros(2 * 3, dtype=np.uint64)
    diagnostics = accumulate_frame_votes(
        accumulator,
        np.asarray([[0, 1]]),
        np.asarray([[1.0, 1.0]]),
        np.asarray([[[1, 0]]], dtype=bool),
        np.asarray([99]),
        point_count=2,
        class_count=2,
    )
    assert accumulator.reshape(2, 3).tolist() == [[0, 0, 0], [0, 0, 1]]
    assert diagnostics["ignored_label_mask_count"] == 1


def test_zero_masks_vote_all_valid_pixels_as_background() -> None:
    accumulator = np.zeros(2 * 3, dtype=np.uint64)
    accumulate_frame_votes(
        accumulator,
        np.asarray([[0, -1], [1, 0]]),
        np.asarray([[1.0, 2.0], [3.0, 0.0]]),
        np.zeros((0, 2, 2), dtype=bool),
        np.empty(0, dtype=np.int64),
        point_count=2,
        class_count=2,
    )
    assert accumulator.reshape(2, 3).tolist() == [[0, 0, 1], [0, 0, 1]]


def test_mask_label_length_and_image_shape_are_strict() -> None:
    accumulator = np.zeros(2 * 3, dtype=np.uint64)
    with pytest.raises(ValueError, match="mask/label length"):
        accumulate_frame_votes(
            accumulator,
            np.zeros((2, 2), dtype=np.int64),
            np.ones((2, 2)),
            np.zeros((2, 2, 2), dtype=bool),
            np.asarray([0]),
            point_count=2,
            class_count=2,
        )
    with pytest.raises(ValueError, match="matching HxW arrays"):
        accumulate_frame_votes(
            accumulator,
            np.zeros((2, 2), dtype=np.int64),
            np.ones((1, 2)),
            np.zeros((0, 2, 2), dtype=bool),
            np.empty(0),
            point_count=2,
            class_count=2,
        )


def test_uint64_overflow_is_rejected_instead_of_wrapping() -> None:
    accumulator = np.asarray([np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(OverflowError, match="overflow"):
        _checked_add_updates(
            accumulator, np.asarray([0]), np.asarray([1], dtype=np.uint64)
        )


def test_sparse_roundtrip_is_atomic_complete_and_byte_deterministic(tmp_path) -> None:
    evidence = _evidence(
        np.asarray([[2, 0, 1], [0, 3, 0], [0, 0, 0]], dtype=np.uint64),
        ("chair", "table"),
    )
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    save_gaussian_vote_evidence(first, evidence)
    save_gaussian_vote_evidence(second, evidence)
    assert first.read_bytes() == second.read_bytes()
    assert vote_evidence_is_complete(first, expected_identity={"fixture": 1})
    assert not vote_evidence_is_complete(first, expected_identity={"fixture": 2})
    restored = load_gaussian_vote_evidence(first)
    assert restored.metadata == evidence.metadata
    assert np.array_equal(restored.row_offsets, evidence.row_offsets)
    assert np.array_equal(restored.channels, evidence.channels)
    assert np.array_equal(restored.counts, evidence.counts)
    assert not list(tmp_path.glob("*.tmp-*"))


def test_corrupt_or_incomplete_file_is_not_reusable(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    path.write_text(json.dumps({"not": "npz"}), encoding="utf-8")
    assert not vote_evidence_is_complete(path)


def test_input_file_identity_detects_same_size_and_mtime_content_change(
    tmp_path,
) -> None:
    path = tmp_path / "mask.pt"
    path.write_bytes(b"AAAA")
    before = _file_identity(path)
    path.write_bytes(b"BBBB")
    # Simulate a copied/replaced asset that preserves the coarse file metadata.
    import os

    os.utime(path, ns=(before["mtime_ns"], before["mtime_ns"]))
    after = _file_identity(path)
    assert before["size_bytes"] == after["size_bytes"]
    assert before["mtime_ns"] == after["mtime_ns"]
    assert before["sha256"] != after["sha256"]


def test_wrong_persisted_dtype_is_not_silently_coerced(tmp_path) -> None:
    evidence = _evidence(np.asarray([[1, 0, 0]], dtype=np.uint64), ("a", "b"))
    path = tmp_path / "wrong-dtype.npz"
    np.savez_compressed(
        path,
        channels=evidence.channels,
        counts=evidence.counts.astype(np.uint32),
        metadata_json=np.asarray(json.dumps(evidence.metadata)),
        row_offsets=evidence.row_offsets,
    )
    assert not vote_evidence_is_complete(path)


def test_instance_aggregation_supports_any_partition_and_keeps_background() -> None:
    evidence = _evidence(
        np.asarray(
            [
                [2, 0, 1],
                [0, 3, 0],
                [4, 5, 6],
                [0, 0, 7],
            ],
            dtype=np.uint64,
        ),
        ("chair", "table"),
    )
    result = aggregate_instance_votes(
        np.asarray([8, 8, -1, 3], dtype=np.int64), evidence
    )
    assert list(result) == [3, 8]
    assert result[3].tolist() == [0, 0, 7]
    assert result[8].tolist() == [2, 3, 1]
    empty = aggregate_instance_votes(np.full(4, -1), evidence)
    assert empty == {}


def test_rgb_and_feature_gaussian_order_must_match() -> None:
    xyz = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    validate_rgb_feature_order(xyz, xyz + 1e-7)
    with pytest.raises(ValueError, match="counts differ"):
        validate_rgb_feature_order(xyz, xyz[:1])
    with pytest.raises(ValueError, match="XYZ/order differ"):
        validate_rgb_feature_order(xyz, xyz[::-1])
