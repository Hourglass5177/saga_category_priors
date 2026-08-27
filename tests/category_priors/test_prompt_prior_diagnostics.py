from __future__ import annotations

import json

import numpy as np
import pytest

from category_priors.prompt_prior_diagnostics import (
    FORMULA_VERSION,
    SIMILARITY_THRESHOLD,
    _load_capacity_result,
    choose_grid_oracle,
    classify_direction_change,
    native_visible_mask_scale,
    scale_key,
)


def test_direction_audit_counts_helpful_and_harmful_changes() -> None:
    uniform = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    data = np.array([1, 0, 0, 1, 1, 0], dtype=np.uint8)
    # 0 target, 1 target, 2 same-class other, 3 target, 4 wrong class,
    # 5 unsupported.
    result = classify_direction_change(
        uniform,
        data,
        gaussian_to_gt_index=np.array([0, 0, 1, 0, 2, 2]),
        gaussian_to_gt_distance_m=np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.20]),
        gt_semantic=np.array([3, 3, 7]),
        gt_instance=np.array([8, 9, 1]),
        target_class_id=3,
        target_instance_id=8,
        radius_m=0.05,
    )
    assert result["added_target"] == 1
    assert result["added_wrong_class"] == 1
    assert result["removed_target"] == 1
    assert result["removed_same_class_other"] == 1
    assert result["helpful_count"] == 2
    assert result["harmful_count"] == 2
    assert result["help_ratio"] == pytest.approx(0.5)
    assert result["direction"] == pytest.approx(0.0)


def test_direction_audit_no_change_has_zero_direction_and_null_ratio() -> None:
    result = classify_direction_change(
        np.array([1, 0]),
        np.array([1, 0]),
        gaussian_to_gt_index=np.array([0, 0]),
        gaussian_to_gt_distance_m=np.array([0.01, 0.01]),
        gt_semantic=np.array([1]),
        gt_instance=np.array([2]),
        target_class_id=1,
        target_instance_id=2,
        radius_m=0.05,
    )
    assert result["changed_count"] == 0
    assert result["help_ratio"] is None
    assert result["direction"] == 0.0


def test_grid_oracle_tie_breaks_by_distance_then_smaller_scale() -> None:
    selected = choose_grid_oracle(
        [
            {"scale_input": 0.25, "iou": 0.7, "condition": "a"},
            {"scale_input": 0.75, "iou": 0.7, "condition": "b"},
            {"scale_input": 0.50, "iou": 0.6, "condition": "c"},
        ],
        uniform_scale=0.5,
    )
    assert selected["condition"] == "a"


def test_native_scale_reproduces_historical_axis_swap() -> None:
    depth = np.arange(1, 31, dtype=np.float64).reshape(5, 6)
    mask = np.ones_like(depth, dtype=bool)
    historical = native_visible_mask_scale(
        depth,
        mask,
        fov_x=1.1,
        fov_y=0.7,
        historical_axis_order=True,
    )
    corrected = native_visible_mask_scale(
        depth,
        mask,
        fov_x=1.1,
        fov_y=0.7,
        historical_axis_order=False,
    )
    assert historical["eligible"] is True
    assert corrected["eligible"] is True
    # The historical 3x3 sum>=5 rule removes the four image corners even for
    # an all-true mask because convolution uses zero padding.
    assert historical["selected_pixel_count"] == 26
    assert historical["raw_scale_scene_units"] != pytest.approx(
        corrected["raw_scale_scene_units"]
    )


def test_native_scale_matches_historical_torch_float32_formula() -> None:
    torch = pytest.importorskip("torch")
    depth = torch.arange(1, 31, dtype=torch.float32).reshape(5, 6)
    mask = torch.ones_like(depth, dtype=torch.float32)
    selected = (
        torch.nn.functional.conv2d(
            mask[None, None], torch.ones((1, 1, 3, 3)), padding=1
        )[0, 0]
        >= 5
    )
    y, x = torch.meshgrid(
        torch.arange(5, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = 3.0, 2.5
    fx = cx / np.tan(1.1 / 2.0)
    fy = cy / np.tan(0.7 / 2.0)
    xyz = torch.stack(((y - cx) * depth / fx, (x - cy) * depth / fy, depth), dim=-1)[
        selected
    ]
    reference = torch.linalg.vector_norm(2.0 * torch.std(xyz, dim=0)).item()
    actual = native_visible_mask_scale(
        depth.numpy(),
        np.ones((5, 6), dtype=bool),
        fov_x=1.1,
        fov_y=0.7,
        historical_axis_order=True,
    )
    assert actual["raw_scale_scene_units"] == pytest.approx(reference, abs=1e-5)


def test_native_scale_rejects_insufficient_or_nonfinite_support() -> None:
    depth = np.full((3, 3), np.nan)
    result = native_visible_mask_scale(
        depth,
        np.ones((3, 3), dtype=bool),
        fov_x=1.0,
        fov_y=1.0,
        historical_axis_order=True,
        require_valid_depth=True,
    )
    assert result == {
        "eligible": False,
        "selected_pixel_count": 0,
        "raw_scale_scene_units": None,
    }


def test_scale_keys_are_stable_and_reject_unregistered_precision() -> None:
    assert scale_key(0.0) == "s_0000"
    assert scale_key(0.125) == "s_0125"
    assert scale_key(1.0) == "s_1000"
    with pytest.raises(ValueError):
        scale_key(0.1234)


def test_capacity_loader_rejects_stale_threshold(tmp_path, monkeypatch) -> None:
    feature = tmp_path / "feature.ply"
    feature.write_bytes(b"feature")
    gate = tmp_path / "gate.pt"
    gate.write_bytes(b"gate")
    result = tmp_path / "result.npz"
    np.savez_compressed(result, s_0000=np.array([0, 1], dtype=np.uint8))
    metadata = {
        "kind": "prompt_prior_scale_capacity_masks",
        "status": "complete",
        "prompt": {"x": 1, "y": 2},
        "feature_ply": str(feature),
        "scale_gate": str(gate),
        "similarity_threshold": SIMILARITY_THRESHOLD - 0.01,
        "formula_version": FORMULA_VERSION,
        "grid_scales": {"s_0000": 0.0},
        "gates": {"s_0000": [0.5] * 32},
        "completed_keys": ["s_0000"],
    }
    result.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.delenv("SAGA_EXPERIMENT_COMMIT", raising=False)
    with pytest.raises(ValueError, match="threshold"):
        _load_capacity_result(
            result,
            expected_points=2,
            expected_prompt={"x": 1, "y": 2},
            expected_scales=[0.0],
            expect_o_instance=False,
            feature_ply=feature,
            scale_gate=gate,
            o_instance_scale_input=None,
        )
