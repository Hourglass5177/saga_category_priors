from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from category_priors.io import load_json, write_json
from category_priors.v9_pipeline import (
    V9_STAGE2_SCENES,
    V9Stage2Config,
    V9Stage2Hooks,
    run_v9_stage2,
    select_v9_association,
    select_v9_late_classifier,
)


def _config(tmp_path: Path) -> V9Stage2Config:
    manifest = tmp_path / "runtime.json"
    write_json(
        manifest,
        {
            "kind": "scene_runtime_manifest",
            "scenes": [
                {
                    "scene_id": scene_id,
                    "base_path": str(tmp_path / scene_id),
                    "python_bin": str(tmp_path / "python"),
                    "scene_scale_m_per_unit": 1.0,
                    "grounded_masks_path": "grounded/masks",
                    "grounded_labels_path": "grounded/labels",
                    "grounded_mask_scales_path": "grounded/scales",
                }
                for scene_id in V9_STAGE2_SCENES
            ],
        },
    )
    return V9Stage2Config(
        runtime_manifest=manifest,
        workspace=tmp_path / "workspace",
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
        gt_dir=tmp_path / "gt",
        sam_packed_root=tmp_path / "sam-packed",
        sam_checkpoint=tmp_path / "sam.pth",
        label_features=tmp_path / "label_features.pt",
        size_bins=tmp_path / "size_bins.json",
        git_commit="abc123",
    )


def _candidate_rows(count: int, *, correct: int, scene_id: str) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": scene_id,
            "candidate_id": index,
            "class": "chair" if index < correct else "table",
            "geometric_best_gt_class": "chair",
            "geometric_best_iou": 0.6,
        }
        for index in range(count)
    ]


def _hooks(
    events: list[str],
    *,
    oracle_passed: bool,
    matches: dict[str, int] | None = None,
) -> V9Stage2Hooks:
    matches = matches or {"A0": 1, "A1": 2, "A2": 3, "A3": 6}

    def ensure(**kwargs: Any) -> Path:
        scene_id = kwargs["scene_id"]
        events.append(f"ensure:{scene_id}")
        return Path(kwargs["reusable_root"]) / scene_id

    def prepare(**kwargs: Any) -> dict[str, Any]:
        scene_id = kwargs["scene_id"]
        events.append(f"prepare:{scene_id}")
        return {
            "scene_overrides": {
                "sam_everything_masks_path": str(Path(kwargs["output_root"]) / scene_id / "masks"),
                "sam_everything_mask_scales_path": str(Path(kwargs["output_root"]) / scene_id / "scales"),
            }
        }

    def train(**kwargs: Any) -> dict[str, Any]:
        scene_id = kwargs["scene_ids"][0]
        events.append(f"train:{scene_id}")
        return {"status": "complete"}

    def audit(**kwargs: Any) -> dict[str, Any]:
        events.append(f"audit:{kwargs['scene_id']}:{kwargs['action']}")
        return {
            "disk_free_gib": 100.0,
            "cgroup": {"current": 1, "max": 90 * 1024**3, "events": {}},
        }

    def lift(*args: Any, **kwargs: Any) -> dict[str, Any]:
        scene_id = args[1][0]
        assert "gt_dir" not in kwargs
        events.append(f"lift:{scene_id}")
        return {"status": "complete"}

    def oracle(**kwargs: Any) -> dict[str, Any]:
        events.append("oracle")
        return {
            "geometric_match_050_count": 6 if oracle_passed else 5,
            "geometric_tiny_small_recall_025": 0.20 if oracle_passed else 0.19,
            "gate": {"passed": oracle_passed, "checks": {}},
        }

    def banks(**kwargs: Any) -> dict[str, Any]:
        mode = kwargs["association_modes"][0]
        events.append(f"bank:{mode}")
        for scene_id in V9_STAGE2_SCENES:
            target = Path(kwargs["output_root"]) / mode / scene_id
            target.mkdir(parents=True, exist_ok=True)
            count = matches[mode]
            write_json(
                target / "object_bank.json",
                {
                    "classifiers": {
                        "mv-label": {
                            "candidates": [
                                {"candidate_id": index, "track_id": 100 + index}
                                for index in range(count)
                            ]
                        },
                        "codebook": {
                            "candidates": [
                                {"candidate_id": index, "track_id": 100 + index}
                                for index in range(count)
                            ]
                        },
                    }
                },
            )
        return {"status": "complete"}

    def evaluate(**kwargs: Any) -> dict[str, Any]:
        mode = kwargs["association_mode"]
        classifier = kwargs["classifier"]
        events.append(f"evaluate:{mode}:{classifier}")
        count = matches[mode]
        per_candidate: list[dict[str, Any]] = []
        for scene_index, scene_id in enumerate(V9_STAGE2_SCENES):
            scene_count = count if scene_index == 0 else 0
            correct = (count - 2) if classifier == "mv-label" else (count - 1)
            per_candidate.extend(
                _candidate_rows(scene_count, correct=max(correct, 0), scene_id=scene_id)
            )
        return {
            "geometric_match_050_count": count,
            "association_pair_precision": 0.8,
            "association_pair_f1": 0.7,
            "association_per_scene": [
                {
                    "predicted_pair_count": count,
                    "oracle_positive_pair_count": count,
                    "true_positive_pair_count": count,
                }
            ],
            "geometric": {"candidate_precision_025": 1.0},
            "per_candidate": per_candidate,
        }

    return V9Stage2Hooks(
        ensure_masks=ensure,
        prepare_affinity_inputs=prepare,
        train_features=train,
        run_lifting=lift,
        evaluate_oracle=oracle,
        run_banks=banks,
        evaluate_banks=evaluate,
        audit_resources=audit,
    )


def test_oracle_failure_stops_before_object_bank(tmp_path: Path) -> None:
    events: list[str] = []
    result = run_v9_stage2(_config(tmp_path), hooks=_hooks(events, oracle_passed=False))

    assert result["state"] == "stopped"
    assert result["checkpoint"] == "stage2-geometric-oracle-failed"
    assert not any(item.startswith("bank:") for item in events)
    assert events.index("oracle") > events.index(f"lift:{V9_STAGE2_SCENES[-1]}")
    augmented = load_json(tmp_path / "artifacts/v9_stage2_runtime_manifest.json")
    assert [row["scene_id"] for row in augmented["scenes"]] == list(V9_STAGE2_SCENES)
    assert all("sam_everything_masks_path" in row for row in augmented["scenes"])
    for scene_id in V9_STAGE2_SCENES:
        assert events.index(f"audit:{scene_id}:train-10k") < events.index(
            f"train:{scene_id}"
        )


def test_stage2_registered_feature_recovery_skips_prepare_and_training(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []

    def frozen(paths):
        return {
            "producer_git_commit": "feature-producer",
            "feature_ply": paths.feature_ply,
            "scale_gate": paths.scale_gate,
            "scene_overrides": {
                "sam_everything_masks_path": str(paths.root / "frozen-masks"),
                "sam_everything_mask_scales_path": str(paths.root / "frozen-scales"),
            },
        }

    monkeypatch.setattr(
        "category_priors.v9_pipeline.registered_v9_feature_source", frozen
    )
    result = run_v9_stage2(
        _config(tmp_path), hooks=_hooks(events, oracle_passed=False)
    )

    assert result["state"] == "stopped"
    assert not any(item.startswith("prepare:") for item in events)
    assert not any(item.startswith("train:") for item in events)
    assert not any(":train-10k" in item for item in events)
    assert [row["scene_id"] for row in result["progress"]["registered_feature_scenes"]] == list(
        V9_STAGE2_SCENES
    )


def test_a3_fallback_selects_only_association_mode(tmp_path: Path) -> None:
    events: list[str] = []
    result = run_v9_stage2(_config(tmp_path), hooks=_hooks(events, oracle_passed=True))

    assert result["state"] == "complete"
    assert result["selection"]["selected_association"] == "A3"
    assert "late_classifier" not in result["selection"]
    assert [item for item in events if item.startswith("bank:")] == [
        "bank:A0", "bank:A1", "bank:A2", "bank:A3"
    ]
    event_count = len(events)
    assert run_v9_stage2(_config(tmp_path), hooks=_hooks(events, oracle_passed=True)) == result
    assert len(events) == event_count


def test_primary_success_does_not_run_a3(tmp_path: Path) -> None:
    events: list[str] = []
    result = run_v9_stage2(
        _config(tmp_path),
        hooks=_hooks(
            events,
            oracle_passed=True,
            matches={"A0": 6, "A1": 5, "A2": 4, "A3": 9},
        ),
    )
    assert result["state"] == "complete"
    assert result["selection"]["selected_association"] == "A0"
    assert "bank:A3" not in events


def test_selection_ties_prefer_simpler_and_mv_within_two_points() -> None:
    selected = select_v9_association(
        [
            {
                "association_mode": mode,
                "geometric_match_050_count": 6,
                "association_pair_precision": 0.8,
                "association_pair_f1": 0.7,
                "merge_error_proxy_count": 1,
                "split_error_proxy_count": 1,
                "candidate_precision_025": 0.5,
            }
            for mode in ("A2", "A0", "A1")
        ]
    )
    assert selected["selected_association"] == "A0"

    mv = {"per_candidate": _candidate_rows(100, correct=79, scene_id="scene")}
    codebook = {"per_candidate": _candidate_rows(100, correct=81, scene_id="scene")}
    assert select_v9_late_classifier(mv, codebook)["selected_classifier"] == "mv-label"
