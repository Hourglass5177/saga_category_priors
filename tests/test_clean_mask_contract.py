from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from category_priors.clean_baseline.mask_contract import (
    SAM_METADATA_KEYS,
    SamMaskMetadataFrame,
    assign_gaussians_to_flat_masks,
    audit_flat_mask_contract,
    flatten_mask_stack,
    flatten_mask_stack_dense_reference,
    load_sam_mask_metadata,
    make_sparse_support_exclusive,
    metadata_frame_from_sam_rows,
    save_sam_mask_metadata,
)


def _rows() -> list[dict[str, object]]:
    # Masks 0/1 overlap at (0, 1); mask 2 loses all pixels to better masks.
    return [
        {
            "segmentation": np.asarray([[1, 1, 0], [0, 0, 0]], dtype=np.bool_),
            "predicted_iou": 0.8,
            "stability_score": 0.9,
            "area": 2,
        },
        {
            "segmentation": np.asarray([[0, 1, 1], [0, 0, 0]], dtype=np.bool_),
            "predicted_iou": 0.9,
            "stability_score": 0.7,
            "area": 2,
        },
        {
            "segmentation": np.asarray([[0, 1, 0], [0, 0, 0]], dtype=np.bool_),
            "predicted_iou": 0.5,
            "stability_score": 0.99,
            "area": 1,
        },
    ]


def test_metadata_round_trip_has_exact_schema_and_readonly_arrays(tmp_path: Path) -> None:
    frame = metadata_frame_from_sam_rows(_rows(), height=2, width=3)
    path = tmp_path / "frame.npz"
    save_sam_mask_metadata(path, frame)
    with np.load(path, allow_pickle=False) as payload:
        assert set(payload.files) == SAM_METADATA_KEYS
        assert payload["count"].dtype == np.int32
    loaded = load_sam_mask_metadata(path)
    assert np.array_equal(loaded.packed, frame.packed)
    assert np.array_equal(loaded.predicted_iou, frame.predicted_iou)
    assert np.array_equal(loaded.stability_score, frame.stability_score)
    assert np.array_equal(loaded.area, np.asarray([2, 2, 1]))
    assert not loaded.packed.flags.writeable
    assert not loaded.predicted_iou.flags.writeable


def test_raw_sam_iou_head_score_is_finite_but_not_probability_clipped() -> None:
    rows = _rows()
    rows[0]["predicted_iou"] = 1.05
    frame = metadata_frame_from_sam_rows(rows, height=2, width=3)
    assert frame.predicted_iou[0] == pytest.approx(1.05)
    flat = flatten_mask_stack(frame)
    # The overlapping pixel follows the preserved higher IoU-head score.
    assert flat.pixel_owner_source_id[0, 1] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows[0].pop("predicted_iou"), "lacks required"),
        (lambda rows: rows[0].__setitem__("area", 5), "area disagrees"),
        (lambda rows: rows[0].__setitem__("predicted_iou", np.nan), "finite"),
        (
            lambda rows: rows[0].__setitem__(
                "segmentation", np.asarray([[1, 0, 0]], dtype=np.bool_)
            ),
            "bool HxW",
        ),
    ],
)
def test_raw_sam_metadata_validation_is_strict(mutation, match: str) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        metadata_frame_from_sam_rows(rows, height=2, width=3)


def test_metadata_loader_rejects_extra_arrays_and_area_mismatch(tmp_path: Path) -> None:
    frame = metadata_frame_from_sam_rows(_rows(), height=2, width=3)
    extra = tmp_path / "extra.npz"
    np.savez_compressed(
        extra,
        packed=frame.packed,
        count=np.asarray(frame.count),
        height=np.asarray(frame.height),
        width=np.asarray(frame.width),
        predicted_iou=frame.predicted_iou,
        stability_score=frame.stability_score,
        area=frame.area,
        labels=np.zeros(frame.count),
    )
    with pytest.raises(ValueError, match="invalid SAM metadata"):
        load_sam_mask_metadata(extra)
    wrong_area = tmp_path / "wrong-area.npz"
    np.savez_compressed(
        wrong_area,
        packed=frame.packed,
        count=np.asarray(frame.count),
        height=np.asarray(frame.height),
        width=np.asarray(frame.width),
        predicted_iou=frame.predicted_iou,
        stability_score=frame.stability_score,
        area=np.asarray([1, 2, 1]),
    )
    with pytest.raises(ValueError, match="invalid SAM metadata"):
        load_sam_mask_metadata(wrong_area)


def test_flattening_uses_iou_then_stability_then_original_id() -> None:
    rows = _rows()
    # Add exact predicted-IoU ties: stability wins first, then original ID.
    rows.extend(
        [
            {
                "segmentation": np.asarray([[0, 0, 0], [1, 1, 0]], dtype=np.bool_),
                "predicted_iou": 0.7,
                "stability_score": 0.8,
                "area": 2,
            },
            {
                "segmentation": np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.bool_),
                "predicted_iou": 0.7,
                "stability_score": 0.9,
                "area": 3,
            },
            {
                "segmentation": np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.bool_),
                "predicted_iou": 0.7,
                "stability_score": 0.9,
                "area": 1,
            },
        ]
    )
    hierarchy = metadata_frame_from_sam_rows(rows, height=2, width=3)
    flat = flatten_mask_stack(hierarchy)
    # (0,1): mask 1 has better IoU; (1,0): mask 4 has better stability;
    # (1,2): masks 4/5 tie quality and lower original ID 4 wins.
    assert flat.pixel_owner_source_id.tolist() == [[0, 1, 1], [4, 4, 4]]
    # Fully shadowed masks 2, 3 and 5 are the only rows removed.  There is no
    # minimum remnant-area filter: one-pixel mask 0 is retained.
    assert flat.source_mask_ids.tolist() == [1, 0, 4]
    assert flat.frame.area.tolist() == [2, 1, 3]


def test_streaming_flatten_matches_independent_dense_reference_and_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    masks = rng.random((17, 7, 9)) < 0.25
    for row in range(len(masks)):
        masks[row, row % 7, row % 9] = True
    hierarchy = metadata_frame_from_sam_rows(
        [
            {
                "segmentation": mask,
                "predicted_iou": float((row % 5) / 4),
                "stability_score": float((row % 3) / 2),
                "area": int(mask.sum()),
            }
            for row, mask in enumerate(masks)
        ],
        height=7,
        width=9,
    )
    first = flatten_mask_stack(hierarchy)
    second = flatten_mask_stack(hierarchy)
    reference = flatten_mask_stack_dense_reference(hierarchy)
    assert np.array_equal(first.pixel_owner_source_id, second.pixel_owner_source_id)
    assert np.array_equal(first.frame.packed, second.frame.packed)
    assert np.array_equal(first.pixel_owner_source_id, reference.pixel_owner_source_id)
    assert np.array_equal(first.frame.packed, reference.frame.packed)
    audit = audit_flat_mask_contract(hierarchy, first)
    assert audit["mechanical_contract_pass"] is True
    assert audit["union_changed_pixel_count"] == 0
    assert audit["flat_overlap_pixel_count"] == 0


def test_flattening_keeps_empty_source_union_and_zero_masks_valid() -> None:
    hierarchy = SamMaskMetadataFrame(
        packed=np.zeros((0, 1), dtype=np.uint8),
        count=0,
        height=2,
        width=3,
        predicted_iou=np.zeros(0, dtype=np.float32),
        stability_score=np.zeros(0, dtype=np.float32),
        area=np.zeros(0, dtype=np.int64),
    )
    flat = flatten_mask_stack(hierarchy)
    assert flat.frame.count == 0
    assert np.all(flat.pixel_owner_source_id == -1)
    assert audit_flat_mask_contract(hierarchy, flat)["mechanical_contract_pass"]


def test_gaussian_assignment_uses_ratio_then_mask_priority() -> None:
    inside = np.asarray(
        [
            [0.7, 0.8, 0.4, 0.8],
            [0.8, 0.8, 0.7, 0.8],
            [0.8, 0.8, 0.7, 0.8],
        ]
    )
    visible = np.ones(4)
    result = assign_gaussians_to_flat_masks(
        inside,
        visible,
        predicted_iou=np.asarray([0.99, 0.8, 0.8]),
        stability_score=np.asarray([0.1, 0.7, 0.7]),
        source_mask_ids=np.asarray([10, 4, 3]),
    )
    # G0: higher ratio masks 1/2 beat mask 0, then source ID 3 wins.
    # G1/G3: all ratios tie, predicted IoU makes mask 0 win.
    # G2: only masks 1/2 qualify, then source ID 3 wins.
    assert result.owner_mask_index.tolist() == [2, 0, 2, 0]
    assert result.owner_source_mask_id.tolist() == [3, 10, 3, 10]
    assert result.qualifying_mask_count.tolist() == [3, 3, 2, 3]
    assert np.allclose(result.owner_ratio, [0.8, 0.8, 0.7, 0.8])
    assert np.all(result.qualifying_mask_count[result.owner_mask_index >= 0] >= 1)


def test_gaussian_assignment_has_one_owner_and_unassigned_sentinel() -> None:
    result = assign_gaussians_to_flat_masks(
        np.asarray([[0.49, 0.0], [0.3, 0.0]]),
        np.asarray([1.0, 0.0]),
        predicted_iou=np.asarray([0.5, 0.6]),
        stability_score=np.asarray([0.5, 0.5]),
    )
    assert result.owner_mask_index.tolist() == [-1, -1]
    assert result.owner_source_mask_id.tolist() == [-1, -1]
    assert result.owner_ratio.tolist() == [0.0, 0.0]
    assert result.qualifying_mask_count.tolist() == [0, 0]


def test_gaussian_assignment_rejects_alpha_mass_above_visibility() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        assign_gaussians_to_flat_masks(
            np.asarray([[1.1]]),
            np.asarray([1.0]),
            predicted_iou=np.asarray([0.5]),
            stability_score=np.asarray([0.5]),
        )


def test_sparse_worker_support_is_filtered_to_one_mask_per_gaussian() -> None:
    rows = make_sparse_support_exclusive(
        mask_gaussian_ids=[
            np.asarray([0, 1, 3], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
            np.asarray([0, 2, 3], dtype=np.int32),
        ],
        mask_inside_mass=[
            np.asarray([0.7, 0.8, 0.8]),
            np.asarray([0.8, 0.8, 0.7]),
            np.asarray([0.8, 0.7, 0.8]),
        ],
        mask_inside_ratio=[
            np.asarray([0.7, 0.8, 0.8]),
            np.asarray([0.8, 0.8, 0.7]),
            np.asarray([0.8, 0.7, 0.8]),
        ],
        point_count=4,
        predicted_iou=np.asarray([0.99, 0.8, 0.8]),
        stability_score=np.asarray([0.1, 0.7, 0.7]),
        source_mask_ids=np.asarray([10, 4, 3]),
    )
    # G0: rows 1/2 tie on ratio and source ID 3 wins.  G1/G3: row 0
    # wins the ratio tie by predicted IoU.  G2: rows 1/2 tie, row 2 wins.
    assert rows[0][0].tolist() == [1, 3]
    assert rows[1][0].tolist() == []
    assert rows[2][0].tolist() == [0, 2]
    assigned = np.concatenate([row[0] for row in rows])
    assert sorted(assigned.tolist()) == [0, 1, 2, 3]
    assert len(np.unique(assigned)) == len(assigned)


def test_sparse_worker_support_matches_dense_assignment() -> None:
    ids = [np.asarray([0, 2]), np.asarray([0, 1, 2])]
    mass = [np.asarray([0.6, 0.9]), np.asarray([0.8, 0.7, 0.9])]
    ratio = [np.asarray([0.6, 0.9]), np.asarray([0.8, 0.7, 0.9])]
    sparse = make_sparse_support_exclusive(
        ids,
        mass,
        ratio,
        point_count=3,
        predicted_iou=np.asarray([0.9, 0.8]),
        stability_score=np.asarray([0.5, 0.5]),
    )
    dense_inside = np.zeros((2, 3), dtype=np.float64)
    for row in range(2):
        dense_inside[row, ids[row]] = mass[row]
    assignment = assign_gaussians_to_flat_masks(
        dense_inside,
        np.ones(3),
        predicted_iou=np.asarray([0.9, 0.8]),
        stability_score=np.asarray([0.5, 0.5]),
        inside_min_mass=0.5,
        inside_min_ratio=0.5,
    )
    sparse_owner = np.full(3, -1, dtype=np.int32)
    for row, support in enumerate(sparse):
        sparse_owner[support[0]] = row
    assert sparse_owner.tolist() == assignment.owner_mask_index.tolist()


def test_sparse_worker_defaults_to_preordered_local_row_for_ratio_tie() -> None:
    rows = make_sparse_support_exclusive(
        [np.asarray([0, 1]), np.asarray([0, 1])],
        [np.asarray([0.6, 0.7]), np.asarray([0.6, 0.8])],
        [np.asarray([0.6, 0.7]), np.asarray([0.6, 0.8])],
        point_count=2,
    )
    # G0 ties and stays with the smaller local row (already higher priority);
    # G1 has genuinely larger ratio in row 1.
    assert rows[0][0].tolist() == [0]
    assert rows[1][0].tolist() == [1]
