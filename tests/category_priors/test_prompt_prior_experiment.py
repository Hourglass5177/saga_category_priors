from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from category_priors.io import load_json, write_json
from category_priors.prompt_prior_experiment import (
    _condition_scale,
    _mark_mechanical,
    _materialize_parameters,
    _parser,
    choose_interior_prompt,
    evaluate_prompt_pair_arrays,
)


def _prior_node(diagonal_m: float) -> dict:
    return {
        "shrunk": {
            "geometry": {"log_bbox_diag_m": {"q50": math.log(diagonal_m)}}
        }
    }


def test_furthest_interior_prompt_uses_xy_json_order() -> None:
    footprint = np.zeros((7, 9), dtype=bool)
    footprint[1:6, 2:7] = True

    assert choose_interior_prompt(footprint) == (4, 3)


def test_segment_command_has_no_ground_truth_argument() -> None:
    args = _parser().parse_args(
        [
            "segment",
            "--runtime-manifest",
            "runtime.json",
            "--prompts-root",
            "prompts",
            "--parameters",
            "params.json",
            "--output-root",
            "masks",
            "--scene",
            "scene0000_00",
        ]
    )

    assert args.command == "segment"
    assert not hasattr(args, "gt_dir")


def test_parameter_materialization_uses_flat_prior_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior_path = tmp_path / "priors.json"
    output_path = tmp_path / "params.json"
    write_json(
        prior_path,
        {
            "provenance": {"splits": ["train"]},
            "global": _prior_node(1.0),
            "categories": {"chair": _prior_node(0.25)},
        },
    )
    monkeypatch.setattr(
        "category_priors.prompt_prior_experiment._scene_assets",
        lambda _scene: {
            "mask_scales": tmp_path / "unused",
            "feature_ply": tmp_path / "feature.ply",
            "scale_gate": tmp_path / "scale_gate.pt",
        },
    )
    (tmp_path / "feature.ply").write_bytes(b"feature")
    (tmp_path / "scale_gate.pt").write_bytes(b"gate")
    monkeypatch.setattr(
        "category_priors.prompt_prior_experiment._mask_scale_values",
        lambda _path: np.asarray([0.1, 0.25, 0.5, 1.0]),
    )

    payload = _materialize_parameters(
        priors_path=prior_path,
        scenes={"scene": {"base_path": str(tmp_path)}},
        scene_ids=["scene"],
        output=output_path,
    )

    assert payload["table"]["global_typical_diag_m"] == pytest.approx(1.0)
    assert payload["table"]["class_typical_diag_m"]["chair"] == pytest.approx(
        0.25
    )
    assert payload["scenes"]["scene"]["global_scale_input"] == pytest.approx(1.0)
    assert payload["scenes"]["scene"]["class_scale_inputs"][
        "chair"
    ] == pytest.approx(1.0 / 3.0)
    assert load_json(output_path) == payload


def test_unknown_category_is_exact_global_scale_fallback() -> None:
    parameters = {
        "scenes": {
            "scene": {
                "global_scale_input": 0.75,
                "class_scale_inputs": {"chair": 0.25},
            }
        }
    }

    assert _condition_scale(parameters, "scene", "chair", "U-global") == 0.75
    assert _condition_scale(parameters, "scene", "chair", "D-class") == 0.25
    assert _condition_scale(parameters, "scene", "unknown", "D-class") == 0.75


def test_gaussian_precision_and_gt_recall_use_opposite_nearest_directions() -> None:
    # Two GT points belong to the target. Gaussian 0 covers them both, while
    # Gaussian 1 is an unsupported false positive.
    result = evaluate_prompt_pair_arrays(
        mask=np.asarray([True, True]),
        target_class_id=0,
        target_instance_id=7,
        gt_semantic=np.asarray([0, 0]),
        gt_instance=np.asarray([7, 7]),
        gt_to_gaussian_index=np.asarray([0, 0]),
        gt_to_gaussian_distance_m=np.asarray([0.0, 0.01]),
        gaussian_to_gt_index=np.asarray([0, 1]),
        gaussian_to_gt_distance_m=np.asarray([0.0, 2.0]),
        radius_m=0.05,
    )

    assert result["iou"] == pytest.approx(1.0)
    assert result["gt_recall"] == pytest.approx(1.0)
    assert result["gaussian_precision"] == pytest.approx(0.5)
    assert result["unsupported_gaussian_count"] == 1


def test_mechanical_selection_keeps_two_small_and_two_large() -> None:
    rows = [
        {"bbox_diagonal_m": float(index), "class_name": f"class-{index}", "gt_instance_id": index}
        for index in range(1, 7)
    ]

    _mark_mechanical(rows)

    selected = [row for row in rows if row["mechanical_selected"]]
    assert len(selected) == 4
    assert {row["bbox_diagonal_m"] for row in selected} == {1.0, 2.0, 5.0, 6.0}
