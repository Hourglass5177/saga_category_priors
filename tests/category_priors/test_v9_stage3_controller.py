from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import numpy as np

from category_priors.evaluator import (
    GroundTruthScene,
    PredictedInstance,
    evaluate_instances,
)
from category_priors.io import load_json, write_json
from category_priors.v9_feature_training import (
    v9_affinity_input_paths,
    v9_feature_training_paths,
)
from category_priors.v9_metrics import (
    paired_physical_scene_bootstrap,
    paired_scannet_scene_bootstrap,
    physical_scene_macro_delta,
    pooled_scannet_metrics_from_scene_weights,
    precompute_scannet_scene_ap_events,
)
from category_priors.v9_stage3_controller import (
    DEV8,
    V9ContinuationConfig,
    V9ContinuationHooks,
    _safe_cleanup_scene,
    _select_dev8_classifier,
    _select_threshold,
    _stage2_selection,
    _validate_t1_reference,
)


def _config(tmp_path: Path) -> V9ContinuationConfig:
    paths = {
        name: tmp_path / name
        for name in (
            "stage2.json",
            "runtime.json",
            "locked-runtime.json",
            "locked-scenes.json",
            "workspace",
            "runs",
            "artifacts",
            "gt",
            "locked-gt",
            "sam-packed",
            "sam.pth",
            "labels.pt",
            "sizes.json",
            "priors.json",
            "t1",
        )
    }
    return V9ContinuationConfig(
        stage2_status=paths["stage2.json"],
        runtime_manifest=paths["runtime.json"],
        locked_runtime_manifest=paths["locked-runtime.json"],
        locked_evaluation_scenes=paths["locked-scenes.json"],
        workspace=paths["workspace"],
        runs_root=paths["runs"],
        artifacts_root=paths["artifacts"],
        gt_dir=paths["gt"],
        locked_gt_dir=paths["locked-gt"],
        sam_packed_root=paths["sam-packed"],
        sam_checkpoint=paths["sam.pth"],
        label_features=paths["labels.pt"],
        size_bins=paths["sizes.json"],
        category_priors=paths["priors.json"],
        t1_b1_root=paths["t1"],
        git_commit="abc123",
    ).normalized()


def _analysis(condition: str, value: float, *, scenes: tuple[str, ...]) -> dict[str, Any]:
    return {
        "conditions": {
            condition: {
                "metrics": {
                    "condition": condition,
                    "map_50_95": value,
                    "ap50": 0.10,
                    "predicted_instance_count": 10,
                    "orphan_gaussian_count": 0,
                    "negative_metadata_count": 0,
                },
                "per_scene": [
                    {
                        "scene_id": scene,
                        "map_50_95": value,
                        "tiny_small_gt_count": 1,
                        "tiny_small_match_050_count": 0,
                        "false_positive_count": 1,
                        "true_positive_count": 1,
                    }
                    for scene in scenes
                ],
            }
        }
    }


def test_stage2_selection_contains_only_frozen_association(tmp_path: Path) -> None:
    config = _config(tmp_path)
    write_json(
        config.stage2_status,
        {
            "state": "complete",
            "checkpoint": "stage2-objectbank-selected",
            "selection": {
                "selected_association": "A2",
            },
        },
    )
    assert _stage2_selection(config) == "A2"

    write_json(config.stage2_status, {"state": "stopped"})
    with pytest.raises(ValueError, match="has not completed"):
        _stage2_selection(config)


def test_late_classifier_is_selected_on_all_dev8_without_worker_or_replay(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def evaluate_banks(**kwargs: Any) -> dict[str, Any]:
        classifier = str(kwargs["classifier"])
        scenes = tuple(map(str, kwargs["scene_ids"]))
        calls.append((classifier, scenes))
        assert scenes == DEV8
        rows = []
        for index, scene_id in enumerate(scenes):
            # The first two scenes prefer codebook; all eight together prefer
            # MV-label.  This fails if selection silently regresses to Stage 2.
            mv_correct = index >= 2
            correct = mv_correct if classifier == "mv-label" else not mv_correct
            rows.append(
                {
                    "scene_id": scene_id,
                    "candidate_id": 0,
                    "track_id": 1000 + index,
                    "class": "chair" if correct else "table",
                    "geometric_best_gt_class": "chair",
                    "geometric_best_iou": 0.60,
                }
            )
        return {"per_candidate": rows}

    def forbidden(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("classifier selection must not call a worker or replay")

    hooks = V9ContinuationHooks(
        evaluate_banks=evaluate_banks,
        run_banks=forbidden,
        replay=forbidden,
    )
    result = _select_dev8_classifier(config, hooks, association_mode="A2")

    assert result["selected_classifier"] == "mv-label"
    assert result["eligible_candidate_count"] == len(DEV8)
    assert calls == [("mv-label", DEV8), ("codebook", DEV8)]
    assert load_json(config.artifacts_root / "late_classifier_selection8.json") == result


def test_t1_reference_preserves_registered_producer_commit_and_strict_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    for scene_id in (
        "scene0645_00", "scene0025_01", "scene0046_00", "scene0474_01",
        "scene0591_02", "scene0329_02", "scene0164_03", "scene0064_01",
    ):
        target = config.t1_b1_root / config.t1_b1_condition / scene_id
        write_json(
            target / "output.json",
            {
                "point_labels": [-1],
                "instances": {},
                "prediction_contract": {
                    "schema": "saga-strict-prediction-contract-v1",
                    "point_count": 1,
                },
            },
        )
        write_json(
            target / "run.json",
            {
                "status": "complete",
                "identity": {
                    "schema": "saga-v9-t1-legacy-v1",
                    "git_commit": config.git_commit,
                    "scene_id": scene_id,
                    "condition": config.t1_b1_condition,
                    "seed": 42,
                    "input_budget": "existing-scene-feature-2k",
                    "contributor_weight": "alpha_times_t_prev",
                    "teacher_prior_mode": "original",
                    "causal_level": "L0",
                    "command": ["postprocess.py", "--teacher-prior-mode original"],
                },
            },
        )
    initial = _validate_t1_reference(config)
    assert initial["corrected_contributor"] is True

    bad = config.t1_b1_root / config.t1_b1_condition / "scene0645_00/run.json"
    payload = load_json(bad)
    payload["identity"]["git_commit"] = "old-producer"
    write_json(bad, payload)
    with pytest.raises(ValueError, match="mix producer commits"):
        _validate_t1_reference(config)

    for scene_id in (
        "scene0645_00", "scene0025_01", "scene0046_00", "scene0474_01",
        "scene0591_02", "scene0329_02", "scene0164_03", "scene0064_01",
    ):
        path = config.t1_b1_root / config.t1_b1_condition / scene_id / "run.json"
        payload = load_json(path)
        payload["identity"]["git_commit"] = "old-producer"
        write_json(path, payload)
    recovered = _validate_t1_reference(config)
    scene = next(
        row for row in recovered["scenes"] if row["scene_id"] == "scene0645_00"
    )
    assert scene["producer_git_commit"] == "old-producer"
    assert scene["consumer_git_commit"] == config.git_commit

    payload = load_json(bad)
    payload["identity"]["git_commit"] = ""
    write_json(bad, payload)
    with pytest.raises(ValueError, match="lacks its producer commit"):
        _validate_t1_reference(config)


def test_threshold_selection_rejects_unsafe_and_breaks_tie_higher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        stem = str(kwargs["stem"])
        calls.append(stem)
        if stem == "threshold2_f10k_b0":
            return _analysis("F10k-B0", 0.05, scenes=("a", "b"))
        threshold = int(stem.rsplit("_", 1)[1]) / 100
        value = {0.05: 0.051, 0.10: 0.052, 0.15: 0.052, 0.20: 0.049, 0.25: 0.048}[threshold]
        result = _analysis("U000", value, scenes=("a", "b"))
        # 0.20 and 0.25 violate the preregistered mAP safety floor.
        return result

    monkeypatch.setattr(
        "category_priors.v9_stage3_controller._evaluate", fake_evaluate
    )
    monkeypatch.setattr(
        "category_priors.v9_stage3_controller._replay_resume",
        lambda *args, **kwargs: None,
    )
    selected = _select_threshold(
        config,
        V9ContinuationHooks(),
        association_mode="A1",
        classifier="mv-label",
    )
    assert selected["passed"]
    assert selected["selected_threshold"] == pytest.approx(0.15)
    assert len(calls) == 6


def test_stream_cleanup_is_scoped_and_preserves_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    feature = v9_feature_training_paths(
        config.runs_root / "feature-10k-objectbank", "scene0000_00"
    )
    affinity = v9_affinity_input_paths(
        config.runs_root / "feature-10k-objectbank", "scene0000_00"
    )
    for path in (feature.model, affinity.masks, affinity.mask_scales, affinity.scale_model):
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.bin").write_bytes(b"x")
    for path in (feature.feature_ply, feature.scale_gate, feature.log, feature.record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    audited: list[bool] = []

    def audit(**kwargs: Any) -> dict[str, Any]:
        audited.append(feature.feature_ply.is_file())
        return {"disk_free_gib": 100.0, "cgroup": {"current": 1, "max": 90 * 1024**3}}

    monkeypatch.setattr(
        "category_priors.v9_stage3_controller._default_resource_audit", audit
    )

    result = _safe_cleanup_scene(config=config, scene_id="scene0000_00")

    assert len(result["removed"]) == 6
    assert not feature.model.exists()
    assert not feature.feature_ply.exists()
    assert not affinity.masks.exists()
    assert feature.log.is_file()
    assert feature.record.is_file()
    assert (feature.root / "cleanup.json").is_file()
    assert audited == [True]
    assert result["resources_before_cleanup"]["disk_free_gib"] == 100.0


def test_stream_cleanup_never_deletes_part_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    feature = v9_feature_training_paths(
        config.runs_root / "feature-10k-objectbank", "scene0000_00"
    )
    feature.model.mkdir(parents=True)
    part = feature.model / "checkpoint.part"
    part.write_bytes(b"resume")

    result = _safe_cleanup_scene(
        config=config,
        scene_id="scene0000_00",
        resource_audit={
            "disk_free_gib": 100.0,
            "cgroup": {"current": 1, "max": 90 * 1024**3},
        },
    )

    assert part.is_file()
    assert str(part.resolve()) in result["preserved_part_files"]


def test_physical_scene_macro_and_paired_bootstrap_group_scans_first() -> None:
    analysis = {
        "conditions": {
            "U000": {
                "metrics": {},
                "per_scene": [
                    {"scene_id": "scene0001_00", "map_50_95": 0.10},
                    {"scene_id": "scene0001_01", "map_50_95": 0.20},
                    {"scene_id": "scene0002_00", "map_50_95": 0.10},
                ],
            },
            "D100": {
                "metrics": {},
                "per_scene": [
                    {"scene_id": "scene0001_00", "map_50_95": 0.12},
                    {"scene_id": "scene0001_01", "map_50_95": 0.22},
                    {"scene_id": "scene0002_00", "map_50_95": 0.14},
                ],
            },
        }
    }
    macro = physical_scene_macro_delta(
        analysis, reference="U000", treatment="D100"
    )
    assert macro["physical_scene_count"] == 2
    assert macro["physical_scene_deltas"]["scene0001"] == pytest.approx(0.02)
    assert macro["macro_delta_map_50_95"] == pytest.approx(0.03)

    bootstrap = paired_physical_scene_bootstrap(
        analysis,
        reference="U000",
        treatment="D100",
        samples=500,
        seed=7,
    )
    assert bootstrap["delta_map_50_95"] == pytest.approx(0.03)
    assert bootstrap["paired_bootstrap_ci95"][0] > 0


def test_v9_scannet_events_reproduce_pooled_official_ap_and_paired_draws() -> None:
    scenes = [
        GroundTruthScene(
            "scene-a",
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        ),
        GroundTruthScene(
            "scene-b",
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
            np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
        ),
    ]
    predictions = [
        PredictedInstance(
            scene_id="scene-a",
            instance_id=0,
            class_id=0,
            score=0.9,
            mask=np.asarray([1, 1, 1, 0, 0, 0], dtype=bool),
        ),
        PredictedInstance(
            scene_id="scene-b",
            instance_id=0,
            class_id=1,
            score=0.8,
            mask=np.asarray([0, 0, 0, 1, 1, 1], dtype=bool),
        ),
    ]
    overlaps = (0.50, 0.75)
    direct = evaluate_instances(
        scenes, predictions, ("chair", "table"),
        overlaps=overlaps, min_region_size=2,
    )
    events = precompute_scannet_scene_ap_events(
        scenes, predictions, ("chair", "table"),
        overlaps=overlaps, min_region_size=2,
    )
    pooled = pooled_scannet_metrics_from_scene_weights(events)
    for key in ("map_0.50", "map_0.75", "map_50_95"):
        assert pooled["aggregate"][key] == pytest.approx(
            direct["aggregate"][key]
        )

    bootstrap = paired_scannet_scene_bootstrap(
        events,
        events,
        physical_scene_ids=("physical-a", "physical-b"),
        samples=129,
        seed=17,
        batch_size=31,
    )
    assert bootstrap["delta_map_50_95"] == pytest.approx(0.0)
    assert bootstrap["paired_bootstrap_ci95"] == pytest.approx([0.0, 0.0])
    assert bootstrap["finite_sample_count"] == 129
