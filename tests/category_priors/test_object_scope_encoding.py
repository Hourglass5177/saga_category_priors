import numpy as np
import pytest

from category_priors.object_scope.encoding import (
    CropEncodingPlan, RGB_PADDING, encode_region, full_image_plan, make_crop_pair, validate_encoding)


def test_non_square_shared_letterbox_and_padding():
    image = np.zeros((50, 100, 3), np.uint8)
    mask = np.zeros((50, 100), bool)
    mask[:, :20] = True
    image[mask] = 255
    value = encode_region(image, mask, np.ones_like(mask), full_image_plan(mask.shape))
    assert value.valid
    assert value.trace["resize_wh"] == [336, 168]
    assert value.trace["padding_ltrb"] == [0, 84, 0, 84]
    assert value.target_mask[:84].sum() == 0
    assert np.array_equal(value.rgb_uint8[0, 0], RGB_PADDING)
    assert value.alpha_tensor[0, 0, 0] == np.float32(-.5) / np.float32(.26)
    assert np.all(value.rgb_uint8[value.target_mask][:, 0] > 100)
    validate_encoding(value)
    with pytest.raises(ValueError):
        value.rgb_tensor.setflags(write=True)


def test_translation_preserves_edge_target_not_an_unknown():
    mask = np.zeros((70, 100), bool)
    mask[:8, :8] = True
    detail, context = make_crop_pair(image_shape=mask.shape, bbox_xyxy=(0, 0, 8, 8),
        fx=100, fy=100, optical_depths=[1, 2, 3], global_d50=1, condition="U1")
    assert detail.box_xyxy == (0, 0, 12, 12)
    value = encode_region(np.zeros((*mask.shape, 3), np.uint8), mask, np.ones_like(mask), detail)
    assert value.valid and value.trace["foreground_pixels_lost_to_crop"] == 0


def test_context_only_intervention_and_missing_depth():
    args = dict(image_shape=(200, 300), bbox_xyxy=(40, 50, 60, 70), fx=100, fy=100,
                optical_depths=[2], global_d50=1, class_d50=4, crop_class="chair", statistics_active=True)
    du, cu = make_crop_pair(**args, condition="U2-feedback")
    dd, cd = make_crop_pair(**args, condition="D2-feedback")
    assert du == dd
    assert cu.box_xyxy != cd.box_xyxy
    args["optical_depths"] = [-1, 0]
    detail, context = make_crop_pair(**args, condition="D2-feedback")
    assert detail.box_xyxy == context.box_xyxy
    assert "depth_unavailable" in context.provenance["fallback_reasons"]
    assert context.provenance["positive_optical_depth_median"] is None


def test_no_mask_and_cropped_mask_measured_unknown():
    image = np.zeros((30, 50, 3), np.uint8)
    mask = np.zeros((30, 50), bool)
    empty = encode_region(image, mask, np.ones_like(mask), full_image_plan(mask.shape))
    assert not empty.valid
    mask[5:15, 5:15] = True
    crop = CropEncodingPlan(mask.shape, (10, 10, 30, 30), "detail", 20, {})
    result = encode_region(image, mask, np.ones_like(mask), crop)
    assert not result.valid and result.trace["foreground_pixels_lost_to_crop"] == 75


def test_actual_tensor_tampering_rejected():
    image = np.zeros((10, 10, 3), np.uint8)
    mask = np.ones((10, 10), bool)
    result = encode_region(image, mask, mask, full_image_plan(mask.shape))
    result.trace["rgb_tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_encoding(result)


@pytest.mark.parametrize("bad", [np.zeros((10, 10), np.float32), np.zeros((11, 10), bool)])
def test_continuous_alpha_and_wrong_axes_rejected(bad):
    with pytest.raises(ValueError):
        encode_region(np.zeros((10, 10, 3), np.uint8), bad, np.ones((10, 10), bool), full_image_plan((10, 10)))


def test_diagnostic_valid_alpha_preserves_target_for_evaluation():
    mask = np.zeros((20, 40), bool)
    mask[2:6, 4:8] = True
    args = (np.zeros((20, 40, 3), np.uint8), mask, np.ones_like(mask), full_image_plan(mask.shape))
    a, b = encode_region(*args), encode_region(*args, alpha_mode="valid")
    assert np.array_equal(a.target_mask, b.target_mask)
    assert np.array_equal(a.rgb_tensor, b.rgb_tensor)
    assert not np.array_equal(a.alpha_tensor, b.alpha_tensor)
