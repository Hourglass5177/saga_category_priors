from __future__ import annotations

from pathlib import Path

from category_priors.category_denoise_experiment import (
    DEV8,
    _dev2_parity,
    _dev8_gate,
    _mechanical_effect,
)
from category_priors.io import write_json


def _prediction(path: Path, labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = sorted({value for value in labels if value >= 0})
    write_json(
        path,
        {
            "point_labels": labels,
            "instances": {
                str(instance): {"class": "chair", "score": 0.5}
                for instance in present
            },
        },
    )


def test_dev2_parity_compares_actual_labels_and_instances(tmp_path: Path) -> None:
    for scene_id in ("scene0645_00", "scene0025_01"):
        _prediction(tmp_path / "b0-off" / scene_id / "output.json", [-1, 0, 0])
        _prediction(tmp_path / "bank" / scene_id / "output.json", [-1, 0, 0])
    assert _dev2_parity(tmp_path)["passed"]

    _prediction(
        tmp_path / "bank" / "scene0025_01" / "output.json", [-1, -1, 0]
    )
    result = _dev2_parity(tmp_path)
    assert not result["passed"]
    assert result["scenes"][1]["changed_point_count"] == 1


def test_mechanical_effect_requires_real_score_or_acceptance_changes(
    tmp_path: Path,
) -> None:
    for index, scene_id in enumerate(DEV8):
        for mode, score, accepted in (
            ("uniform", 0.30, False),
            ("class", 0.32 if index < 4 else 0.30, index < 4),
        ):
            write_json(
                tmp_path / mode / scene_id / "diagnostics.json",
                {
                    "category_denoise": {
                        "decisions": [
                            {
                                "candidate_id": 0,
                                "branch_class": "chair" if index % 2 else "cup",
                                "score": score,
                                "accepted": accepted,
                            }
                        ]
                    }
                },
            )
    result = _mechanical_effect(tmp_path, DEV8)
    assert result["passed"]
    assert result["score_delta_ge_0.01_count"] == 4


def _analysis() -> dict:
    per_scene_u = []
    per_scene_d = []
    for index, scene_id in enumerate(DEV8):
        per_scene_u.append({"scene_id": scene_id, "map_50_95": 0.05})
        per_scene_d.append(
            {"scene_id": scene_id, "map_50_95": 0.053 if index < 5 else 0.05}
        )
    base = {
        "map_50_95": 0.050,
        "ap50": 0.10,
        "predicted_instance_count": 100,
        "prediction_coverage": 0.50,
        "tiny_small_recall_050": 0.20,
        "false_positive_count": 20,
        "true_positive_count": 20,
    }
    uniform = {**base, "map_50_95": 0.050, "ap50": 0.10}
    data = {
        **base,
        "map_50_95": 0.053,
        "ap50": 0.102,
        "tiny_small_recall_050": 0.22,
    }
    return {
        "candidate_bank": {
            "same_class_iou_050_count": 12,
            "same_class_iou_050_scene_count": 4,
        },
        "conditions": {
            "bank": {"metrics": base, "per_scene": per_scene_u},
            "uniform": {"metrics": uniform, "per_scene": per_scene_u},
            "class": {"metrics": data, "per_scene": per_scene_d},
        },
    }


def test_dev8_gate_keeps_structure_and_prior_questions_separate() -> None:
    passed = _dev8_gate(_analysis(), {"passed": True})
    assert passed["passed"]

    failed = _analysis()
    failed["candidate_bank"]["same_class_iou_050_count"] = 11
    result = _dev8_gate(failed, {"passed": True})
    assert not result["passed"]
    assert not result["checks"]["candidate_space"]
