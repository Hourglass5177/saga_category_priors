from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import category_priors.clean_baseline.experiment as experiment_module
from category_priors.clean_baseline.experiment import (
    CONFIG_KIND,
    DEV2,
    DEV8,
    HOLDOUT5,
    CleanExperimentConfig,
    CleanExperimentHooks,
    _validate_registered_inputs,
    _verify_metric_geometry_identity,
    run_clean_baseline_experiment,
)
from category_priors.clean_baseline.identity_control import (
    IDENTITY_CONTROL_REGISTRATION_SCHEMA,
    IdentityAssetPaths,
    IdentityControlConfig,
)
from category_priors.evaluator import GroundTruthScene
from category_priors.io import hash_json, load_json, sha256_file, write_json
from category_priors.clean_baseline.worker import DEFAULT_CLASSES
from category_priors.taxonomy import load_taxonomy


CLASSES = tuple(DEFAULT_CLASSES)
EVALUATION_CLASSES = tuple(load_taxonomy().canonical_classes)


def _tune24() -> tuple[str, ...]:
    base = DEV8 + HOLDOUT5
    repeats = tuple(
        f"{scene.rsplit('_', 1)[0]}_99" for scene in base[:11]
    )
    return base + repeats


def _final48() -> tuple[str, ...]:
    return tuple(f"scene{index:04d}_00" for index in range(1000, 1048))


def _write_scene_inputs(root: Path, scene_id: str) -> dict[str, object]:
    request = root / f"{scene_id}-request.json"
    gt = root / f"{scene_id}-gt.npz"
    ply = root / f"{scene_id}.ply"
    write_json(
        request,
        {
            "producer_commit": "a" * 40,
            "classes": list(CLASSES),
            "scene": {"scene_id": scene_id},
        },
    )
    np.savez_compressed(
        gt,
        coords=np.zeros((1, 3), dtype=np.float32),
        semantic=np.zeros(1, dtype=np.int64),
        instance=np.zeros(1, dtype=np.int64),
    )
    ply.write_text("ply\n", encoding="utf-8")
    return {
        "evidence_request": str(request),
        "gt_npz": str(gt),
        "gaussian_ply": str(ply),
        "gaussian_to_gt_transform": np.eye(4).tolist(),
        "tiny_small_instance_ids": [0],
    }


def _config(tmp_path: Path) -> CleanExperimentConfig:
    prior = tmp_path / "priors.json"
    prior_payload = {
        "kind": "category_priors",
        "schema_version": "1.0",
        "provenance": {"splits": ["train"]},
        "global": {},
        "categories": {},
    }
    prior_payload["content_sha256"] = hash_json(prior_payload)
    write_json(prior, prior_payload)
    size_bins = tmp_path / "size-bins.json"
    boundaries = {
        "tiny_max_m": 0.5,
        "small_max_m": 1.0,
        "medium_max_m": 2.0,
    }
    write_json(size_bins, {"boundaries_m": boundaries})
    tune24 = _tune24()
    final48 = _final48()
    scene_ids = tuple(sorted(set(tune24).union(final48)))
    scene_rows = {
        scene_id: _write_scene_inputs(tmp_path, scene_id) for scene_id in scene_ids
    }
    runtime_registration = {
        scene_id: {"scene_id": scene_id, "registered_for_test": True}
        for scene_id in scene_ids
    }
    for scene_id, row in scene_rows.items():
        write_json(
            row["evidence_request"],
            {
                "producer_commit": "a" * 40,
                "classes": list(CLASSES),
                "scene": {"scene_id": scene_id},
                "runtime_registration": runtime_registration[scene_id],
            },
        )
    payload = {
        "kind": CONFIG_KIND,
        "code_commit": "a" * 40,
        "repo_root": str(tmp_path),
        "run_root": str(tmp_path / "runs"),
        "artifact_root": str(tmp_path / "artifacts"),
        "category_priors": str(prior),
        "size_bins": str(size_bins),
        "size_bin_boundaries_m": boundaries,
        "class_names": list(CLASSES),
        "evidence_class_names": list(CLASSES),
        "evaluation_class_names": list(EVALUATION_CLASSES),
        "allowed_classes": list(EVALUATION_CLASSES),
        "b1_fixed_metrics": {"map_50_95": 0.05, "map_0.50": 0.06},
        "train_physical_scene_ids": ["scene0900"],
        "identity_control_registration": {
            "schema": IDENTITY_CONTROL_REGISTRATION_SCHEMA,
            "status": "unavailable",
            "train_scene_ids": list(DEV2),
            "validation_scene_id": "scene0046_00",
            "issues": ["synthetic identity assets are not registered"],
        },
        "dev2": list(DEV2),
        "dev8": list(DEV8),
        "holdout5": list(HOLDOUT5),
        "tune24": list(tune24),
        "final48": list(final48),
        "runtime_registration": runtime_registration,
        "scenes": scene_rows,
    }
    path = tmp_path / "experiment.json"
    write_json(path, payload)
    config = CleanExperimentConfig.from_json(path)
    for scene_id in scene_ids:
        write_json(
            config.scene_input_path(scene_id),
            {
                "schema": experiment_module.SCENE_INPUT_REGISTRATION_SCHEMA,
                "status": "complete",
                "scene_id": scene_id,
                "code_commit": config.code_commit,
                "prepared_request": str(config.scenes[scene_id].evidence_request),
                "gt_npz": str(config.scenes[scene_id].gt_npz),
                "gaussian_ply": str(config.scenes[scene_id].gaussian_ply),
                "content_identity": {
                    "prepared_request_sha256": sha256_file(
                        config.scenes[scene_id].evidence_request
                    ),
                    "gt_npz_sha256": sha256_file(config.scenes[scene_id].gt_npz),
                    "gaussian_ply_sha256": sha256_file(
                        config.scenes[scene_id].gaussian_ply
                    ),
                },
                "tiny_small_instance_ids": [0],
                "sam": {
                    "status": "complete",
                    "source": "synthetic-test",
                    "audit": {"complete": True},
                },
            },
        )
    return config


def _oracle_result(scene_id: str, *, passed: bool = True) -> dict[str, object]:
    matches = 3 if passed else 0
    return {
        "schema": "saga-clean-alpha-mask-geometry-oracle-v2",
        "scene_id": scene_id,
        "aggregate": {
            "official_valid": {
                "gt_count": 4,
                "perfect_trim": {"match_050_count": matches},
            },
            "tiny_small_official_valid": {
                "gt_count": 2,
                "perfect_trim": {
                    "match_025_count": 1 if passed else 0,
                },
            },
        },
    }


def test_prepared_scene_rejects_same_path_content_replacement(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scene_id = config.dev2[0]
    assert config.prepared_scene_spec(scene_id).scene_id == scene_id

    # A resume must not silently evaluate an old evidence bank against a new
    # GT payload merely because the filesystem path stayed the same.
    config.scenes[scene_id].gt_npz.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="input content changed"):
        config.prepared_scene_spec(scene_id)
    with pytest.raises(ValueError, match="registration changed or became incomplete"):
        experiment_module._prepare_registered_scene(config, scene_id)


def test_imported_evidence_is_byte_validated_and_never_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    scene_id = config.dev2[0]
    producer_commit = "b" * 40
    bank_dir = tmp_path / "imported-bank" / scene_id
    bank_dir.mkdir(parents=True)
    filenames = ("evidence.npz", "masks.json", "diagnostics.json")
    for name in filenames:
        (bank_dir / name).write_bytes(f"registered-{name}".encode("ascii"))

    request_path = config.scenes[scene_id].evidence_request
    request = load_json(request_path)
    request["producer_commit"] = producer_commit
    write_json(request_path, request)
    payload = load_json(config.config_path)
    payload["evidence_imports"] = {
        scene_id: {
            "schema": experiment_module.EVIDENCE_IMPORT_SCHEMA,
            "bank_dir": str(bank_dir),
            "producer_commit": producer_commit,
            "files": {
                name: sha256_file(bank_dir / name) for name in filenames
            },
        }
    }
    write_json(config.config_path, payload)
    imported_config = CleanExperimentConfig.from_json(config.config_path)
    assert imported_config.bank_dir(scene_id) == bank_dir.resolve()
    assert experiment_module._state_identity(imported_config)["evidence_imports"][
        scene_id
    ]["producer_commit"] == producer_commit

    build_calls: list[str] = []
    monkeypatch.setattr(
        experiment_module,
        "evidence_request_source",
        lambda **_kwargs: {"producer_commit": producer_commit},
    )
    monkeypatch.setattr(
        experiment_module,
        "evidence_bank_is_complete",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        experiment_module,
        "load_evidence_bank",
        lambda *_args, **_kwargs: SimpleNamespace(
            mask_count=7, source={"producer_commit": producer_commit}
        ),
    )
    monkeypatch.setattr(
        experiment_module,
        "build_alpha_mask_evidence",
        lambda **_kwargs: build_calls.append("built"),
    )
    result = experiment_module._default_build_evidence(
        imported_config,
        scene_id=scene_id,
        request_path=request_path,
        output_dir=bank_dir,
    )
    assert result["status"] == "reused-imported"
    assert result["producer_commit"] == producer_commit
    assert build_calls == []

    (bank_dir / "evidence.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="byte identity changed"):
        experiment_module._default_build_evidence(
            imported_config,
            scene_id=scene_id,
            request_path=request_path,
            output_dir=bank_dir,
        )
    assert build_calls == []


def test_geometry_gate_uses_true_support_ceiling_not_greedy_subset() -> None:
    rows = []
    for _scene in DEV2:
        rows.append(
            {
                "aggregate": {
                    "official_valid": {
                        "gt_count": 4,
                        "greedy_association": {"match_050_count": 0},
                        "perfect_trim": {"match_050_count": 3},
                    },
                    "tiny_small_official_valid": {
                        "gt_count": 2,
                        "greedy_association": {"match_025_count": 0},
                        "perfect_trim": {"match_025_count": 1},
                    },
                }
            }
        )
    gate = experiment_module._aggregate_oracle_gate(rows)
    assert gate["passed"] is True
    assert gate["gate_metric"] == "perfect_trim_support_ceiling"
    assert gate["greedy_association_used_for_gate"] is False


def test_evidence_and_official_evaluation_vocabularies_cannot_be_conflated(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assert config.evidence_class_names[3] == "flower"
    assert config.evaluation_class_names[3] == "tv"

    payload = load_json(config.config_path)
    payload["evaluation_class_names"] = payload["evidence_class_names"][:20]
    payload["allowed_classes"] = payload["evaluation_class_names"]
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="canonical SAGA20 evaluator order"):
        CleanExperimentConfig.from_json(config.config_path)


def test_old_config_without_explicit_vocabulary_split_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = load_json(config.config_path)
    payload.pop("evidence_class_names")
    payload.pop("evaluation_class_names")
    write_json(config.config_path, payload)
    with pytest.raises(KeyError):
        CleanExperimentConfig.from_json(config.config_path)


def test_nested_gt_field_cannot_reenter_formal_runtime_request(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = load_json(config.config_path)
    scene_id = DEV2[0]
    payload["runtime_registration"][scene_id]["nested"] = {
        "ground_truth_path": "forbidden.npz"
    }
    request = load_json(config.scenes[scene_id].evidence_request)
    request["runtime_registration"] = payload["runtime_registration"][scene_id]
    write_json(config.scenes[scene_id].evidence_request, request)
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="evaluation-only fields leaked"):
        CleanExperimentConfig.from_json(config.config_path)


def test_geometry_oracle_receives_complete_masks_including_ambiguity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, bool]] = []
    bank = SimpleNamespace(
        masks=(SimpleNamespace(global_mask_id=7, frame_id=3),),
        point_count=3,
        source={"identity": "bank"},
    )

    def support_for_mask(
        mask_id: int, *, include_ambiguous: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        calls.append((mask_id, include_ambiguous))
        ids = np.asarray([0, 1, 2] if include_ambiguous else [0, 1])
        return ids, np.ones(ids.size), np.ones(ids.size), np.asarray([2])

    bank.support_for_mask = support_for_mask
    captured: dict[str, object] = {}
    monkeypatch.setattr(experiment_module, "load_evidence_bank", lambda *_a, **_k: bank)
    identity = {
        "schema": experiment_module.OFFLINE_ORACLE_IDENTITY_SCHEMA,
        "artifact_kind": "geometry-oracle",
    }
    identity["content_sha256"] = experiment_module.hash_json(identity)
    monkeypatch.setattr(
        experiment_module,
        "_build_offline_oracle_identity",
        lambda *_a, **_k: identity,
    )
    monkeypatch.setattr(
        experiment_module,
        "_scene_gt_adapter",
        lambda *_a, **_k: (None, (), np.empty(0, dtype=np.int64), {}),
    )

    def capture(supports: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        captured["supports"] = supports
        return {}

    monkeypatch.setattr(experiment_module, "evaluate_geometry_oracles", capture)
    experiment_module._default_geometry_oracle(
        object(),
        scene_id="scene0000_00",
        bank_dir=tmp_path,
        output_path=tmp_path / "oracle.json",
    )

    assert calls == [(7, True)]
    assert [ids.tolist() for ids in captured["supports"]] == [[0, 1, 2]]


def test_offline_oracle_excludes_geometry_abstained_visibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    bank = SimpleNamespace(
        masks=(),
        frames=(
            SimpleNamespace(frame_id=0, geometry_abstained=False),
            SimpleNamespace(frame_id=1, geometry_abstained=True),
        ),
        frame_count=2,
        point_count=3,
        xyz_m=np.zeros((3, 3), dtype=np.float64),
        visibility_for_frame=lambda _frame_id: (
            np.arange(3, dtype=np.int64),
            np.ones(3),
        ),
    )
    captured: dict[str, np.ndarray] = {}
    monkeypatch.setattr(experiment_module, "load_evidence_bank", lambda *_a, **_k: bank)
    monkeypatch.setattr(
        experiment_module,
        "_scene_gt_adapter",
        lambda *_a, **_k: (None, (), np.empty(0, dtype=np.int64), {}),
    )
    monkeypatch.setattr(
        experiment_module.SizePriorTable,
        "from_category_priors",
        lambda _payload: object(),
    )

    def capture(_observations, visibility, _xyz, **_kwargs):
        captured["visibility"] = np.asarray(visibility)
        return SimpleNamespace(objects=(), accepted_edges=())

    monkeypatch.setattr(experiment_module, "run_mask_consensus", capture)
    candidates, decisions, rows = experiment_module._oracle_candidates(
        config, DEV2[0], tmp_path
    )
    assert candidates == [] and decisions == [] and rows == []
    np.testing.assert_array_equal(
        captured["visibility"],
        np.asarray([[True, True, True], [False, False, False]]),
    )


def _offline_identity_bank(
    bank_dir: Path, scene_id: str = DEV2[0]
) -> SimpleNamespace:
    bank_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("evidence.npz", b"evidence-content-v1"),
        ("masks.json", b'{"schema":"test"}\n'),
        ("diagnostics.json", b'{"schema":"test"}\n'),
    ):
        (bank_dir / name).write_bytes(payload)
    bank = SimpleNamespace(
        schema="saga-clean-alpha-mask-evidence-v2",
        scene_id=scene_id,
        masks=(SimpleNamespace(global_mask_id=7, frame_id=3),),
        point_count=3,
        frame_count=1,
        mask_count=1,
        class_names=CLASSES,
        source={"identity": "bank-source-v1"},
        thresholds=SimpleNamespace(to_dict=lambda: {"visible_mass": 0.5}),
    )
    bank.support_for_mask = lambda *_a, **_k: (
        np.asarray([0, 1, 2]),
        np.ones(3),
        np.ones(3),
        np.empty(0, dtype=np.int64),
    )
    return bank


def test_geometry_oracle_rejects_registered_gt_change_and_recomputes_bank_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    scene_id = DEV2[0]
    bank_dir = tmp_path / "bank"
    bank = _offline_identity_bank(bank_dir)
    calls = 0

    monkeypatch.setattr(
        experiment_module, "load_evidence_bank", lambda *_a, **_k: bank
    )
    monkeypatch.setattr(
        experiment_module,
        "_scene_gt_adapter",
        lambda *_a, **_k: (None, (), np.empty(0, dtype=np.int64), {}),
    )

    def evaluate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"aggregate": {}}

    monkeypatch.setattr(experiment_module, "evaluate_geometry_oracles", evaluate)
    output = tmp_path / "geometry-oracle.json"
    first = experiment_module._default_geometry_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    second = experiment_module._default_geometry_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    assert calls == 1
    assert first["runner_status"] == "complete"
    assert second["runner_status"] == "skipped-complete"

    # The path is unchanged.  Only its bytes change, so a path-only cache
    # would incorrectly reuse the old diagnostic.
    original_gt = config.scenes[scene_id].gt_npz.read_bytes()
    with config.scenes[scene_id].gt_npz.open("ab") as handle:
        handle.write(b"changed-gt-content")
    with pytest.raises(ValueError, match="input content changed"):
        experiment_module._default_geometry_oracle(
            config,
            scene_id=scene_id,
            bank_dir=bank_dir,
            output_path=output,
        )
    assert calls == 1
    config.scenes[scene_id].gt_npz.write_bytes(original_gt)
    with (bank_dir / "evidence.npz").open("ab") as handle:
        handle.write(b"changed-bank-content")
    third = experiment_module._default_geometry_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    assert calls == 2
    assert third["runner_status"] == "complete"
    assert (
        first["run_identity"]["content_sha256"]
        != third["run_identity"]["content_sha256"]
    )


def test_oracle_class_recomputes_when_same_prior_path_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    scene_id = DEV2[0]
    bank_dir = tmp_path / "bank"
    bank = _offline_identity_bank(bank_dir)
    calls = 0

    monkeypatch.setattr(
        experiment_module, "load_evidence_bank", lambda *_a, **_k: bank
    )

    def candidates(*_args: object, **_kwargs: object) -> tuple[list, list, list]:
        nonlocal calls
        calls += 1
        return [], [], []

    monkeypatch.setattr(experiment_module, "_oracle_candidates", candidates)
    monkeypatch.setattr(
        experiment_module,
        "_scene_gt_adapter",
        lambda *_a, **_k: (None, (), np.empty(0, dtype=np.int64), {}),
    )
    monkeypatch.setattr(
        experiment_module,
        "evaluate_candidates",
        lambda *_a, **_k: {"candidate_count": 0},
    )
    output = tmp_path / "oracle-class.json"
    first = experiment_module._default_run_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    second = experiment_module._default_run_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    assert calls == 1
    assert second["runner_status"] == "skipped-complete"

    write_json(config.category_priors, {"changed": True})
    third = experiment_module._default_run_oracle(
        config,
        scene_id=scene_id,
        bank_dir=bank_dir,
        output_path=output,
    )
    assert calls == 2
    assert third["runner_status"] == "complete"
    assert (
        first["run_identity"]["content_sha256"]
        != third["run_identity"]["content_sha256"]
    )


def _condition(
    *,
    candidate_count: int,
    geometry25: int,
    geometry50: int,
    same25: int,
    same50: int,
    tiny25: float,
    tiny50: float,
    map_value: float,
    map50: float,
    coverage: int,
) -> dict[str, object]:
    return {
        "official": {
            "map_50_95": map_value,
            "map_0.50": map50,
            "map_0.25": map50,
        },
        "candidate": {
            "candidate_count": candidate_count,
            "geometry_iou_025_count": geometry25,
            "geometry_iou_050_count": geometry50,
            "geometry_iou_050_scene_count": coverage,
            "same_class_iou_025_count": same25,
            "same_class_iou_050_count": same50,
            "same_class_iou_050_scene_count": coverage,
            "candidate_precision_025": same25 / candidate_count,
            "tiny_small_recall_025": tiny25,
            "tiny_small_recall_050": tiny50,
            "score_iou_spearman": 0.4,
            "fp_tp_ratio_025": (candidate_count - same25) / same25,
        },
        "contract": {
            "orphan_gaussian_count": 0,
            "negative_metadata_count": 0,
            "duplicate_ownership_count": 0,
        },
        "scenes": [],
    }


def _report(
    scene_ids: tuple[str, ...],
    conditions: tuple[str, ...],
    *,
    mechanical: bool = True,
    prior_passed: bool = True,
) -> dict[str, object]:
    count = 20 if len(scene_ids) == 8 else 8
    c0 = _condition(
        candidate_count=count,
        geometry25=14,
        geometry50=10,
        same25=10,
        same50=8,
        tiny25=0.4,
        tiny50=0.2,
        map_value=0.055,
        map50=0.065,
        coverage=min(5, len(scene_ids)),
    )
    uniform = _condition(
        candidate_count=count,
        geometry25=18,
        geometry50=16,
        same25=14,
        same50=12,
        tiny25=0.4,
        tiny50=0.2,
        map_value=0.055,
        map50=0.065,
        coverage=min(5, len(scene_ids)),
    )
    data = _condition(
        candidate_count=count,
        geometry25=18,
        geometry50=16,
        same25=14,
        same50=12,
        tiny25=0.4,
        tiny50=0.22,
        map_value=0.058 if prior_passed else 0.055,
        map50=0.067,
        coverage=min(5, len(scene_ids)),
    )
    values = {
        "C0-no-prior": c0,
        "U-global": uniform,
        "D-predicted": data,
    }
    for condition, value in values.items():
        condition_delta = 0.003 if condition == "D-predicted" else 0.0
        tiny_delta = 0.01 if condition == "D-predicted" else 0.0
        value["scenes"] = [
            {
                "scene_id": scene_id,
                "official": {"map_50_95": 0.05 + condition_delta},
                "candidate": {"tiny_small_recall_050": 0.20 + tiny_delta},
            }
            for scene_id in scene_ids
        ]
    return {
        "conditions": {name: values[name] for name in conditions},
        "prior_effect": {
            "merge_status_change_fraction": 0.2 if mechanical else 0.0,
            "final_merge_decision_change_count": 6 if mechanical else 0,
        },
        "data_minus_uniform": {
            "map_50_95_delta": 0.003 if prior_passed else 0.0,
            "tiny_small_recall_050_delta": 0.02 if prior_passed else 0.0,
            "positive_scene_count": 5 if prior_passed else 0,
            "fp_tp_degradation": 0.0,
        },
        "rows": [],
        "prior_rows": [],
    }


def _hooks(
    calls: list[tuple],
    *,
    geometry_passed: bool = True,
    mechanical: bool = True,
    prior_passed: bool = True,
    fail_build_once: bool = False,
    fail_validation_stage: str | None = None,
) -> CleanExperimentHooks:
    failed = False

    def build_evidence(**kwargs: object) -> dict[str, object]:
        nonlocal failed
        calls.append(("build", kwargs))
        assert "scene_spec" not in kwargs
        assert not any("gt" in key.lower() for key in kwargs)
        if fail_build_once and not failed:
            failed = True
            raise RuntimeError("simulated interruption")
        return {"scene_id": kwargs["scene_id"], "complete": True}

    def geometry_oracle(**kwargs: object) -> dict[str, object]:
        calls.append(("geometry", kwargs))
        return _oracle_result(str(kwargs["scene_id"]), passed=geometry_passed)

    def run_formal(**kwargs: object) -> dict[str, object]:
        calls.append(("formal", kwargs))
        assert "scene_spec" not in kwargs
        assert not any("gt" in key.lower() or "oracle" in key.lower() for key in kwargs)
        return {"complete": True}

    def run_oracle(**kwargs: object) -> dict[str, object]:
        calls.append(("oracle", kwargs))
        return {"evaluation_only": True, "formal_output_written": False}

    def evaluate_stage(**kwargs: object) -> dict[str, object]:
        calls.append(("evaluate", kwargs))
        report = _report(
            tuple(kwargs["scene_ids"]),
            tuple(kwargs["conditions"]),
            mechanical=mechanical,
            prior_passed=prior_passed,
        )
        stage = (
            "holdout5" if tuple(kwargs["scene_ids"]) == HOLDOUT5
            else "tune24" if tuple(kwargs["scene_ids"]) == _tune24()
            else "final48" if tuple(kwargs["scene_ids"]) == _final48()
            else None
        )
        if fail_validation_stage is not None and stage == fail_validation_stage:
            for row in report["conditions"]["D-predicted"]["scenes"]:
                row["official"]["map_50_95"] = 0.05
                row["candidate"]["tiny_small_recall_050"] = 0.20
        return report

    return CleanExperimentHooks(
        build_evidence=build_evidence,
        geometry_oracle=geometry_oracle,
        run_formal=run_formal,
        run_oracle=run_oracle,
        evaluate_stage=evaluate_stage,
    )


def test_happy_path_reaches_dev8_and_keeps_formal_runtime_gt_free(
    tmp_path: Path,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    # The default Stage-0 parity is tested in evaluation.py.  Stub only the
    # registered file validator here so this controller test stays synthetic.
    import category_priors.clean_baseline.experiment as module

    original = module._validate_registered_inputs
    module._validate_registered_inputs = lambda _: {"gt_as_prediction_parity": True}
    try:
        result = run_clean_baseline_experiment(config, _hooks(calls))
    finally:
        module._validate_registered_inputs = original

    assert result["status"] == "complete"
    assert result["checkpoint"] == "final48-complete"
    assert result["current_stage"] is None
    assert experiment_module._load_state(config)["status"] == "complete"
    assert result["candidate_prior_tested"] is True
    assert result["oracle_class_formal_output"] is False
    assert len([call for call in calls if call[0] == "formal"]) == 184
    assert load_json(config.analysis_path)["oracle_class_formal_output"] is False
    for name in (
        "size_prior_holdout5.parquet",
        "size_prior_tune24.parquet",
        "size_prior_final48.parquet",
        "size_prior_final48_bootstrap.json",
    ):
        assert (config.artifact_root / name).is_file()
    assert load_json(
        config.artifact_root / "size_prior_final48_bootstrap.json"
    )["samples"] == 10_000


def test_failed_geometry_gate_stops_before_any_formal_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    result = run_clean_baseline_experiment(
        config, _hooks(calls, geometry_passed=False)
    )
    assert result["status"] == "stopped"
    assert result["checkpoint"] == "dev2-geometry-gate-failed"
    assert result["candidate_prior_tested"] is False
    assert not [call for call in calls if call[0] == "formal"]


def test_healthy_geometry_and_failed_dev8_uniform_runs_only_offline_identity_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    identity_assets = {
        scene_id: IdentityAssetPaths(
            tmp_path / f"{scene_id}-feature.ply",
            tmp_path / f"{scene_id}-gaussian.ply",
        )
        for scene_id in (*DEV2, "scene0046_00")
    }
    config = replace(
        config,
        identity_control=IdentityControlConfig(assets=identity_assets),
        identity_control_registration={
            "schema": IDENTITY_CONTROL_REGISTRATION_SCHEMA,
            "status": "available",
            "train_scene_ids": list(DEV2),
            "validation_scene_id": "scene0046_00",
            "issues": [],
        },
    )
    hooks = replace(
        _hooks(calls),
        run_identity_control=lambda **kwargs: {
            "formal_method": False,
            "category_prior_tested": False,
            "gate": {"passed": True},
            "output_path": str(kwargs["output_path"]),
        },
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._dev8_health_gate",
        lambda *_: {"passed": False},
    )
    result = run_clean_baseline_experiment(config, hooks)
    assert result["status"] == "stopped"
    assert result["checkpoint"] == "dev8-uniform-gate-failed"
    assert result["identity_control_run"] is True
    assert result["identity_control_formal_method"] is False
    assert result["candidate_prior_tested"] is False
    identity_history = result["history"][-1]["identity_control"]
    assert identity_history["formal_method"] is False
    assert identity_history["gate"]["passed"] is True


def test_healthy_geometry_reports_unavailable_identity_assets_without_running_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._dev8_health_gate",
        lambda *_: {"passed": False},
    )
    result = run_clean_baseline_experiment(config, _hooks(calls))
    assert result["status"] == "stopped"
    assert result["identity_control_run"] is False
    assert "registered existing assets were unavailable" in result["stop_reason"]
    assert "synthetic identity assets are not registered" in result["stop_reason"]


def test_inactive_prior_is_not_reported_as_prior_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    result = run_clean_baseline_experiment(
        config, _hooks(calls, mechanical=False)
    )
    assert result["checkpoint"] == "dev2-prior-intervention-inactive"
    assert "not evidence" in result["stop_reason"]


def test_holdout_failure_stops_before_tune_or_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    result = run_clean_baseline_experiment(
        config, _hooks(calls, fail_validation_stage="holdout5")
    )
    assert result["status"] == "stopped"
    assert result["checkpoint"] == "holdout5-gate-failed"
    built_scenes = [
        str(call[1]["scene_id"]) for call in calls if call[0] == "build"
    ]
    assert not set(_final48()).intersection(built_scenes)
    assert not set(_tune24()).difference(set(DEV8 + HOLDOUT5)).intersection(
        built_scenes
    )
    assert not (config.artifact_root / "size_prior_tune24.parquet").exists()


def test_interrupted_stage_resumes_from_last_complete_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment._validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    hooks = _hooks(calls, fail_build_once=True)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_clean_baseline_experiment(config, hooks)
    partial = load_json(config.state_path)
    assert partial["checkpoint"] == "validated"
    assert partial["next_stage"] == "dev2-evidence"

    result = run_clean_baseline_experiment(config, hooks)
    assert result["status"] == "complete"
    assert [entry["stage"] for entry in result["history"]].count("validated") == 1


def test_state_content_and_stage_chain_reject_tampering(tmp_path: Path) -> None:
    config = _config(tmp_path)
    state = experiment_module._load_state(config)
    raw = load_json(config.state_path)
    raw["checkpoint"] = "tune24-evaluate"
    write_json(config.state_path, raw)
    with pytest.raises(ValueError, match="content identity mismatch"):
        experiment_module._load_state(config)

    # Even an internally re-hashed file cannot authorize a skipped stage.
    raw.pop("content_sha256", None)
    raw["content_sha256"] = hash_json(raw)
    write_json(config.state_path, raw)
    with pytest.raises(ValueError, match="history skips or reorders"):
        experiment_module._load_state(config)


def test_terminal_resume_revalidates_commit_and_dirty_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(
        experiment_module,
        "_validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    result = run_clean_baseline_experiment(
        config, _hooks([], geometry_passed=False)
    )
    assert result["status"] == "stopped"

    monkeypatch.setattr(
        experiment_module,
        "_validate_deployment_environment",
        lambda _config: (_ for _ in ()).throw(
            RuntimeError("deployment identity changed")
        ),
    )
    with pytest.raises(RuntimeError, match="deployment identity changed"):
        run_clean_baseline_experiment(config)


def test_interrupted_identity_control_rewinds_to_registered_dev8_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple] = []
    config = _config(tmp_path)
    identity_assets = {
        scene_id: IdentityAssetPaths(
            tmp_path / f"{scene_id}-feature.ply",
            tmp_path / f"{scene_id}-gaussian.ply",
        )
        for scene_id in (*DEV2, "scene0046_00")
    }
    config = replace(
        config,
        identity_control=IdentityControlConfig(assets=identity_assets),
        identity_control_registration={
            "schema": IDENTITY_CONTROL_REGISTRATION_SCHEMA,
            "status": "available",
            "train_scene_ids": list(DEV2),
            "validation_scene_id": "scene0046_00",
            "issues": [],
        },
    )
    hooks = replace(
        _hooks(calls),
        run_identity_control=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("interrupted identity control")
        ),
    )
    monkeypatch.setattr(
        experiment_module,
        "_validate_registered_inputs",
        lambda _: {"gt_as_prediction_parity": True},
    )
    monkeypatch.setattr(experiment_module, "_dev8_health_gate", lambda *_: {"passed": False})
    with pytest.raises(RuntimeError, match="interrupted identity control"):
        run_clean_baseline_experiment(config, hooks)
    persisted = load_json(config.state_path)
    assert persisted["checkpoint"] == "dev8-evidence"
    assert persisted["current_stage"] is None
    assert persisted["next_stage"] == "dev8-uniform"
    # The persisted state is loadable and will safely replay the parent stage.
    loaded = experiment_module._load_state(config)
    assert loaded["next_stage"] == "dev8-uniform"


def test_registered_splits_cannot_overlap(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = load_json(config.config_path)
    payload["train_physical_scene_ids"] = ["scene0645"]
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="ScanNet-train priors overlaps DEV8"):
        CleanExperimentConfig.from_json(config.config_path)

    payload["train_physical_scene_ids"] = ["scene8000"]
    payload["final48"][0] = "scene8000_00"
    payload["scenes"]["scene8000_00"] = payload["scenes"]["scene1000_00"]
    payload["runtime_registration"]["scene8000_00"] = payload[
        "runtime_registration"
    ].pop("scene1000_00")
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="ScanNet-train priors overlaps final"):
        CleanExperimentConfig.from_json(config.config_path)

    payload["train_physical_scene_ids"] = []
    payload["final48"][1] = "scene8000_99"
    payload["scenes"]["scene8000_99"] = payload["scenes"]["scene1001_00"]
    payload["runtime_registration"]["scene8000_99"] = payload[
        "runtime_registration"
    ].pop("scene1001_00")
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="repeated physical"):
        CleanExperimentConfig.from_json(config.config_path)


def test_train_prior_physical_scenes_must_be_explicitly_registered(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    payload = load_json(config.config_path)
    payload["train_physical_scene_ids"] = []
    write_json(config.config_path, payload)
    with pytest.raises(ValueError, match="must be explicitly registered"):
        CleanExperimentConfig.from_json(config.config_path)


def test_metric_geometry_identity_allows_rigid_transform_but_rejects_scale() -> None:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(100, 3))
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rigid = points @ rotation.T + np.asarray([3.0, -2.0, 1.0])
    audit = _verify_metric_geometry_identity(points, rigid, scene_id="scene")
    assert audit["sample_count"] == 64
    assert audit["translation_rotation_invariant"] is True
    with pytest.raises(ValueError, match="metric geometry differs"):
        _verify_metric_geometry_identity(points, rigid * 2.0, scene_id="scene")


def test_stage0_records_provenance_sai3d_skip_and_exact_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)

    def provenance(**kwargs: object) -> dict[str, object]:
        payload = {
            "schema": "saga-clean-baseline-provenance-v1",
            "current_commit": config.code_commit,
        }
        if kwargs.get("output_path") is not None:
            write_json(Path(kwargs["output_path"]), payload)
        return payload

    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment.build_clean_baseline_provenance",
        provenance,
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment.evidence_request_source",
        lambda **_: {
            "producer_commit": config.code_commit,
            "class_names": list(config.evidence_class_names),
        },
    )
    scene = GroundTruthScene(
        scene_id="ignored",
        semantic=np.zeros(100, dtype=np.int64),
        instance=np.zeros(100, dtype=np.int64),
    )
    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment.load_ground_truth_npz",
        lambda _path, scene_id: (
            np.zeros((100, 3), dtype=np.float64),
            GroundTruthScene(scene_id, scene.semantic, scene.instance),
        ),
    )
    result = _validate_registered_inputs(config)
    assert result["gt_as_prediction_parity"] == "deferred-to-dev2-input-preflight"
    assert result["sai3d"]["status"] == "skipped-missing-assets"
    assert result["sai3d"]["download_attempted"] is False
    assert (config.artifact_root / "clean_baseline_provenance.json").is_file()

    monkeypatch.setattr(
        "category_priors.clean_baseline.experiment.build_clean_baseline_provenance",
        lambda **_: {
            "schema": "saga-clean-baseline-provenance-v1",
            "current_commit": "different-commit",
        },
    )
    with pytest.raises(ValueError, match="code_commit"):
        _validate_registered_inputs(config)
