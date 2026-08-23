from __future__ import annotations

import numpy as np

from category_priors.v8_masks import (
    SAM_EVERYTHING_CONFIG,
    _mask_file_is_complete,
    _save_packed_masks,
)


def test_sam_everything_configuration_is_frozen() -> None:
    assert SAM_EVERYTHING_CONFIG == {
        "points_per_side": 32,
        "pred_iou_thresh": 0.88,
        "stability_score_thresh": 0.95,
        "box_nms_thresh": 0.70,
        "crop_n_layers": 0,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 100,
    }


def test_mask_file_validation_rejects_corrupt_shape_and_dtype(tmp_path) -> None:
    path = tmp_path / "masks.npz"
    _save_packed_masks(path, np.zeros((2, 4, 5), dtype=bool))
    assert _mask_file_is_complete(path, 4, 5)

    _save_packed_masks(path, np.zeros((2, 5, 4), dtype=bool))
    assert not _mask_file_is_complete(path, 4, 5)

    path.write_bytes(b"broken")
    assert not _mask_file_is_complete(path, 4, 5)
