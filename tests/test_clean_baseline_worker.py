from __future__ import annotations

import numpy as np
import pytest

from category_priors.clean_baseline.sam_inputs import PackedMaskFrame
from category_priors.clean_baseline.worker import (
    CONTINUOUS_ALPHA_MASS_PERSISTED,
    EVIDENCE_VALUE_CONTRACT,
    decision_canonical_evidence,
    iter_mask_batches,
    mark_same_frame_ambiguity,
    mask_class_probabilities,
    normalized_alpha_objectives,
    sparse_support_from_mass,
)


def test_mask_batches_convert_only_the_current_three_masks_to_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from category_priors.clean_baseline import worker

    masks = np.zeros((7, 5, 6), dtype=bool)
    original_asarray = np.asarray

    def guarded_asarray(value, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else None)
        if value is masks and dtype is np.float32:
            raise AssertionError("the complete mask stack was converted to float32")
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(worker.np, "asarray", guarded_asarray)
    batches = iter_mask_batches(masks)
    assert not isinstance(batches, list)
    rows = list(batches)
    assert [row.indices for row in rows] == [(0, 1, 2), (3, 4, 5), (6,)]
    assert all(row.targets.shape == (3, 5, 6) for row in rows)
    assert all(row.targets.dtype == np.float32 for row in rows)


def test_packed_mask_batches_expand_at_most_three_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from category_priors.clean_baseline import sam_inputs

    dense = np.zeros((7, 5, 6), dtype=np.bool_)
    dense[:, 0, 0] = True
    packed = np.packbits(dense.reshape(7, -1), axis=1)
    frame = PackedMaskFrame(packed, 7, 5, 6)
    original = sam_inputs.np.unpackbits
    expanded_rows: list[int] = []

    def guarded(value, *args, **kwargs):
        expanded_rows.append(int(np.asarray(value).shape[0]))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(sam_inputs.np, "unpackbits", guarded)
    rows = list(iter_mask_batches(frame))
    assert [row.indices for row in rows] == [(0, 1, 2), (3, 4, 5), (6,)]
    assert expanded_rows == [3, 3, 1]


def test_normalized_alpha_objectives_conserve_valid_pixel_mass() -> None:
    targets = np.zeros((3, 1, 2), dtype=np.float32)
    targets[0, 0, 0] = 1.0
    targets[1, 0, 1] = 1.0
    opacity = np.asarray([[0.25, 0.50]], dtype=np.float32)

    inside, visible, valid = normalized_alpha_objectives(targets, opacity)

    assert valid.tolist() == [[True, True]]
    np.testing.assert_allclose(visible, [[4.0, 2.0]])
    np.testing.assert_allclose(inside[0], [[4.0, 0.0]])
    np.testing.assert_allclose(inside[1], [[0.0, 2.0]])


def test_sparse_support_uses_full_mask_thresholds() -> None:
    visible = np.asarray([1.0, 2.0, 0.6, 3.0])
    inside = np.asarray(
        [
            [0.5, 0.99, 0.4, 2.0],
            [0.0, 1.2, 0.5, 0.3],
        ]
    )

    rows = sparse_support_from_mass(inside, visible)

    assert rows[0][0].tolist() == [0, 3]
    assert rows[1][0].tolist() == [1, 2]
    assert rows[0][2].tolist() == pytest.approx([0.5, 2.0 / 3.0])


def test_decision_canonical_evidence_removes_only_unused_atomic_tail_bits() -> None:
    visible_a = np.asarray([0.50010, 2.0, 0.49990], dtype=np.float64)
    visible_b = np.asarray([0.50012, 2.03, 0.49988], dtype=np.float64)
    inside_a = np.asarray([[0.50001, 1.4, 0.2]], dtype=np.float64)
    inside_b = np.asarray([[0.50003, 1.42, 0.2]], dtype=np.float64)

    rows_a = sparse_support_from_mass(inside_a, visible_a)
    rows_b = sparse_support_from_mass(inside_b, visible_b)
    canonical_a = decision_canonical_evidence(rows_a, visible_a)
    canonical_b = decision_canonical_evidence(rows_b, visible_b)

    for left, right in zip(canonical_a[0], canonical_b[0], strict=True):
        assert all(np.array_equal(a, b) for a, b in zip(left, right, strict=True))
    assert np.array_equal(canonical_a[1], canonical_b[1])
    assert np.array_equal(canonical_a[2], canonical_b[2])
    assert canonical_a[0][0][1].tolist() == [1.0, 1.0]
    assert canonical_a[0][0][2].tolist() == [1.0, 1.0]
    assert canonical_a[2].tolist() == [1.0, 1.0]
    assert EVIDENCE_VALUE_CONTRACT == "thresholded-membership-indicator"
    assert CONTINUOUS_ALPHA_MASS_PERSISTED is False


def test_decision_canonical_evidence_cannot_hide_threshold_or_owner_changes() -> None:
    visible = np.ones(3, dtype=np.float64)
    kept = decision_canonical_evidence(
        sparse_support_from_mass(np.asarray([[0.50001, 0.0, 0.0]]), visible),
        visible,
    )[0]
    rejected = decision_canonical_evidence(
        sparse_support_from_mass(np.asarray([[0.49999, 0.0, 0.0]]), visible),
        visible,
    )[0]
    assert kept[0][0].tolist() == [0]
    assert rejected[0][0].tolist() == []

    from category_priors.clean_baseline.mask_contract import (
        make_sparse_support_exclusive,
    )

    ids = [np.asarray([1], dtype=np.int32), np.asarray([1], dtype=np.int32)]
    masses = [np.asarray([0.8]), np.asarray([0.7])]
    left_owner = make_sparse_support_exclusive(
        ids,
        masses,
        [np.asarray([0.8]), np.asarray([0.7])],
        point_count=3,
    )
    right_owner = make_sparse_support_exclusive(
        ids,
        masses[::-1],
        [np.asarray([0.7]), np.asarray([0.8])],
        point_count=3,
    )
    left = decision_canonical_evidence(left_owner, visible)[0]
    right = decision_canonical_evidence(right_owner, visible)[0]
    assert [row[0].tolist() for row in left] == [[1], []]
    assert [row[0].tolist() for row in right] == [[], [1]]


def test_decision_canonical_evidence_rejects_invalid_raw_values() -> None:
    with pytest.raises(ValueError, match="decision values"):
        decision_canonical_evidence(
            ((np.asarray([0]), np.asarray([np.nan]), np.asarray([0.8])),),
            np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="visible_threshold"):
        decision_canonical_evidence((), np.asarray([1.0]), visible_threshold=0.0)


def test_same_frame_hierarchy_is_marked_ambiguous() -> None:
    ambiguity = mark_same_frame_ambiguity(
        [np.asarray([0, 1, 2]), np.asarray([2, 3]), np.asarray([4])],
        point_count=5,
    )

    assert [row.tolist() for row in ambiguity] == [[2], [2], []]


def test_mask_semantics_are_soft_iou_evidence_and_abstain_cleanly() -> None:
    sam = np.zeros((2, 4, 4), dtype=bool)
    sam[0, :2, :2] = True
    sam[1, 2:, 2:] = True
    grounded = np.zeros((3, 4, 4), dtype=bool)
    grounded[0, :2, :2] = True
    grounded[1, :2, :3] = True
    grounded[2, 2:, 2:] = True

    probabilities, abstained = mask_class_probabilities(
        sam,
        grounded,
        np.asarray([0, 1, 1]),
        class_count=3,
    )

    assert abstained is False
    assert probabilities[0, 0] == pytest.approx(0.6)
    assert probabilities[0, 1] == pytest.approx(0.4)
    assert probabilities[1].tolist() == pytest.approx([0.0, 1.0, 0.0])

    empty, missing = mask_class_probabilities(
        sam, None, None, class_count=3
    )
    assert missing is True
    assert np.count_nonzero(empty) == 0

    with pytest.raises(ValueError, match="integer dtype"):
        mask_class_probabilities(
            sam, grounded[:1], np.asarray([0.5]), class_count=3
        )
