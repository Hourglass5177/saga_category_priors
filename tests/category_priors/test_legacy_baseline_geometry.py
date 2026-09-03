from __future__ import annotations

from pathlib import Path

import numpy as np

from category_priors.geometry import nonnegative_cluster_ids, unit_cube_coordinates


def test_cluster_ids_do_not_assume_noise_label_exists() -> None:
    assert nonnegative_cluster_ids(
        np.asarray([-1, 0, 0, 1, 1], dtype=np.int64)
    ) == (0, 1)
    assert nonnegative_cluster_ids(
        np.asarray([0, 0, 1, 1], dtype=np.int64)
    ) == (0, 1)


def test_unit_cube_coordinates_keep_zero_span_axis_finite() -> None:
    points = np.asarray(
        [[2.0, 4.0, -3.0], [2.0, 7.0, -1.0], [2.0, 10.0, 1.0]],
        dtype=np.float32,
    )

    normalized = unit_cube_coordinates(points)

    assert normalized.dtype == points.dtype
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 0], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(normalized[:, 1], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(normalized[:, 2], [0.0, 0.5, 1.0])


def test_teacher_stage_trace_uses_neutral_l0_schema() -> None:
    source = (Path(__file__).resolve().parents[2] / "postprocess.py").read_text(
        encoding="utf-8"
    )

    assert '"schema": "saga-teacher-stage-trace-v2"' in source
    assert '"level": "L0"' in source
    assert "saga-v9-legacy-stage-trace-v1" not in source


def test_teacher_postprocess_enforces_clean_input_and_output_contracts() -> None:
    source = (Path(__file__).resolve().parents[2] / "postprocess.py").read_text(
        encoding="utf-8"
    )

    assert "DEFAULT_SELECTED_CLASSES = SAGA20_CLASSES" in source
    assert "except Exception" not in source
    assert "masks = masks > 0.5" in source
    assert "mask labels fall outside the 32-class table" in source
    assert "vote_histogram_33" in source
    assert "vote_ratios_32" in source
