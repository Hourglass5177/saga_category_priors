from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from category_priors.v8_lifting import (
    AttributionFragment,
    AttributionMass,
    V8FragmentConfig,
)
from category_priors.v8_worker import (
    _serialize_records,
    compare_contributor_images,
    load_segment_everything_payload,
    normalize_grounded_payload,
    render_alpha_mass_attribution,
    sparse_frame_lift_record,
    stable_fragments,
)
from category_priors.v8_masks import _save_packed_masks


def test_missing_grounded_output_is_abstention_not_background() -> None:
    payload = normalize_grounded_payload(None, None, 2, 3)
    assert payload.abstained
    assert payload.masks is None
    assert payload.mask_count == 0

    with pytest.raises(ValueError, match="both exist or both be absent"):
        normalize_grounded_payload(torch.zeros((1, 2, 3)), None, 2, 3)


def test_present_empty_grounded_output_is_not_abstention() -> None:
    payload = normalize_grounded_payload(
        torch.zeros((0, 2, 3), dtype=torch.bool),
        torch.zeros(0, dtype=torch.long),
        2,
        3,
    )
    assert not payload.abstained
    assert payload.masks is not None
    assert payload.masks.shape == (0, 2, 3)


def test_segment_everything_roundtrip_uses_packed_masks(tmp_path) -> None:
    masks = np.zeros((2, 3, 5), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 2, 4] = True
    _save_packed_masks(tmp_path / "frame.npz", masks)

    payload = load_segment_everything_payload(tmp_path, "frame", 3, 5)

    assert not payload.abstained
    np.testing.assert_array_equal(payload.masks, masks)


def test_fragment_id_comes_from_stable_source_mask_identity() -> None:
    mass = AttributionMass(
        source="AM",
        inside_mass=np.array([[0.0, 0.0], [3.0, 3.0]]),
        visible_mass=np.array([3.0, 3.0]),
        valid_pixel_count=2,
    )
    fragments = stable_fragments(
        mass,
        frame_id=4,
        stable_mask_offset=100,
        config=V8FragmentConfig(
            fragment_min_core=1,
            fragment_min_full=1,
        ),
    )
    assert len(fragments) == 1
    assert fragments[0].mask_index == 1
    assert fragments[0].fragment_id == 101


def test_am_gradient_worker_recovers_normalized_all_contributor_mass() -> None:
    # alpha*T masses for two pixels and two Gaussians.  Pixel opacity is the
    # row sum, so the expected attribution is normalized per pixel.
    weights = torch.tensor(
        [[[0.6, 0.3], [0.2, 0.4]]], dtype=torch.float64
    )

    class FakeGaussians:
        get_xyz = torch.zeros((2, 3), dtype=torch.float64)

    def fake_render(_camera, _pc, _pipeline, _background, *, precomputed_mask):
        return {"mask": torch.einsum("hwn,nc->chw", weights, precomputed_mask)}

    camera = SimpleNamespace(image_height=1, image_width=2)
    masks = np.array([[[True, False]]])
    result = render_alpha_mass_attribution(
        camera,
        FakeGaussians(),
        SimpleNamespace(),
        torch.zeros(3, dtype=torch.float64),
        masks,
        point_count=2,
        render_mask_fn=fake_render,
    )
    np.testing.assert_allclose(result.inside_mass[0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(result.visible_mass, [1.0, 1.0])
    assert result.valid_pixel_count == 2
    assert not result.abstained


def test_am_missing_masks_still_measures_visibility_but_abstains() -> None:
    weights = torch.tensor([[[0.4], [0.7]]], dtype=torch.float64)

    class FakeGaussians:
        get_xyz = torch.zeros((1, 3), dtype=torch.float64)

    def fake_render(_camera, _pc, _pipeline, _background, *, precomputed_mask):
        return {"mask": torch.einsum("hwn,nc->chw", weights, precomputed_mask)}

    result = render_alpha_mass_attribution(
        SimpleNamespace(image_height=1, image_width=2),
        FakeGaussians(),
        SimpleNamespace(),
        torch.zeros(3, dtype=torch.float64),
        None,
        point_count=1,
        render_mask_fn=fake_render,
    )
    assert result.abstained
    assert result.inside_mass.shape == (0, 1)
    np.testing.assert_allclose(result.visible_mass, [2.0])


def test_contributor_diff_retains_changed_pixel_evidence() -> None:
    old_id = np.array([[0, 2], [3, -1]])
    old_weight = np.array([[0.0, 0.2], [0.4, 0.0]])
    new_id = np.array([[-1, 1], [3, -1]])
    new_weight = np.array([[0.0, 0.3], [0.4, 0.0]])
    summary, evidence = compare_contributor_images(
        old_id, old_weight, new_id, new_weight
    )
    assert summary["pixel_count"] == 4
    assert summary["changed_pixel_count"] == 2
    assert summary["id_changed_pixel_count"] == 2
    np.testing.assert_array_equal(evidence["flat_pixel"], [0, 1])
    np.testing.assert_array_equal(evidence["historical_id"], [0, 2])
    np.testing.assert_array_equal(evidence["fixed_id"], [-1, 1])


def test_serialization_preserves_mass_and_grounded_abstention(tmp_path) -> None:
    visible = np.array([2.0, 1.0, 0.0])
    geometry_mass = AttributionMass(
        source="AM",
        inside_mass=np.array([[2.0, 0.5, 0.0]]),
        visible_mass=visible,
        valid_pixel_count=3,
    )
    fragment = AttributionFragment(
        fragment_id=7,
        frame_id=0,
        mask_index=0,
        full_ids=np.array([0, 1], dtype=np.int32),
        core_ids=np.array([0], dtype=np.int32),
        full_inside_mass=np.array([2.0, 0.5], dtype=np.float32),
        core_inside_mass=np.array([2.0], dtype=np.float32),
        core_inside_ratio=np.array([1.0], dtype=np.float32),
    )
    geometry = sparse_frame_lift_record(
        0,
        "frame",
        geometry_mass,
        (fragment,),
        np.array([-1], dtype=np.int16),
        retain_visibility=True,
    )
    semantic_mass = AttributionMass(
        source="AM",
        inside_mass=np.zeros((0, 3)),
        visible_mass=visible,
        valid_pixel_count=3,
        abstained=True,
    )
    semantic = sparse_frame_lift_record(
        0,
        "frame",
        semantic_mass,
        (),
        np.empty(0, dtype=np.int16),
        retain_visibility=False,
    )
    _serialize_records(
        tmp_path,
        np.zeros((3, 3)),
        np.ones(3),
        np.ones((3, 2)),
        np.ones((3, 2)),
        np.ones((32, 2)),
        [geometry],
        [semantic],
        [],
    )
    with np.load(tmp_path / "lifting_bank.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["fragment_full_ids"], [0, 1])
        np.testing.assert_allclose(arrays["fragment_full_mass"], [2.0, 0.5])
        np.testing.assert_array_equal(arrays["fragment_source_class"], [-1])
        np.testing.assert_array_equal(arrays["frame_visible_indptr"], [0, 2])
        np.testing.assert_array_equal(arrays["frame_visible_ids"], [0, 1])
        np.testing.assert_allclose(arrays["frame_visible_mass"], [2.0, 1.0])
        np.testing.assert_array_equal(arrays["frame_grounded_missing"], [True])


def test_sparse_frame_record_does_not_retain_dense_attribution() -> None:
    point_count = 100_000
    inside = np.zeros((3, point_count), dtype=np.float64)
    visible = np.zeros(point_count, dtype=np.float64)
    visible[[4, 90_000]] = [2.0, 1.0]
    attribution = AttributionMass(
        source="M1",
        inside_mass=inside,
        visible_mass=visible,
        valid_pixel_count=3,
    )

    record = sparse_frame_lift_record(
        5,
        "frame-5",
        attribution,
        (),
        np.arange(3, dtype=np.int16),
        retain_visibility=True,
    )

    assert not hasattr(record, "attribution")
    np.testing.assert_array_equal(record.visible_ids, [4, 90_000])
    np.testing.assert_allclose(record.visible_mass, [2.0, 1.0])
    assert record.mask_count == 3
    assert record.visible_ids.nbytes + record.visible_mass.nbytes == 16


def test_semantic_sparse_record_does_not_duplicate_visibility() -> None:
    attribution = AttributionMass(
        source="AM",
        inside_mass=np.zeros((0, 50_000), dtype=np.float64),
        visible_mass=np.ones(50_000, dtype=np.float64),
        valid_pixel_count=50_000,
        abstained=True,
    )

    record = sparse_frame_lift_record(
        0,
        "frame",
        attribution,
        (),
        np.empty(0, dtype=np.int16),
        retain_visibility=False,
    )

    assert record.visible_ids.size == 0
    assert record.visible_mass.size == 0
    assert record.abstained
