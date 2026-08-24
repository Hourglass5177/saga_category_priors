from __future__ import annotations

"""Recoverable V9 stages 3--6 controller.

Stage 2 freezes only the association structure.  This module expands that
structure to all eight development scenes, freezes the late classifier from
those eight offline evaluations, selects the one global uniform acceptance
threshold on the two registered development scenes, and then applies the
preregistered health/prior/holdout/final gates.  Ground truth is passed
exclusively to evaluator hooks.
"""

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json, write_rows
from .prediction_contract import validate_prediction_contract
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import load_taxonomy
from .v9_evaluation import SceneMethodMetrics, stage3_uniform_health_gate, stage4_prior_gate
from .v9_feature_training import (
    execute_v9_feature_training,
    prepare_v9_affinity_inputs,
    v9_affinity_input_paths,
    v9_feature_training_paths,
)
from .v9_legacy_runner import execute_v9_legacy_runs
from .v9_legacy_runner import read_v9_legacy_resources
from .v9_lifting import lifting_bank_is_complete, load_lifting_bank
from .v9_lifting_runner import ensure_v9_segment_everything, run_v9_lifting_banks
from .v9_objectbank import V9Config
from .v9_metrics import (
    evaluate_v9_candidate_banks,
    evaluate_v9_predictions,
    metrics_by_condition,
    paired_scannet_bootstrap_from_predictions,
    physical_scene_macro_delta,
    scene_metrics,
)
from .v9_pipeline import select_v9_late_classifier
from .v9_replay import CONDITION_FACTORS
from .v9_runner import object_bank_is_complete, replay_v9_priors, run_v9_banks


DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
THRESHOLD2 = DEV8[:2]
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
CONDITIONS = tuple(CONDITION_FACTORS)


@dataclass(frozen=True)
class V9ContinuationConfig:
    stage2_status: Path
    runtime_manifest: Path
    locked_runtime_manifest: Path
    locked_evaluation_scenes: Path
    workspace: Path
    runs_root: Path
    artifacts_root: Path
    gt_dir: Path
    locked_gt_dir: Path
    sam_packed_root: Path
    sam_checkpoint: Path
    label_features: Path
    size_bins: Path
    category_priors: Path
    t1_b1_root: Path
    git_commit: str
    t1_b1_condition: str = "T1-B1"
    taxonomy_path: Path | None = None
    sam_reusable_root: Path | None = None

    def normalized(self) -> "V9ContinuationConfig":
        values = asdict(self)
        for key, value in tuple(values.items()):
            if key.endswith("_path") or key in {
                "stage2_status",
                "runtime_manifest",
                "locked_runtime_manifest",
                "locked_evaluation_scenes",
                "workspace",
                "runs_root",
                "artifacts_root",
                "gt_dir",
                "locked_gt_dir",
                "sam_packed_root",
                "sam_checkpoint",
                "label_features",
                "size_bins",
                "category_priors",
                "t1_b1_root",
                "sam_reusable_root",
            }:
                if value is not None:
                    values[key] = Path(value).resolve()
        values["git_commit"] = str(values["git_commit"]).strip()
        if not values["git_commit"]:
            raise ValueError("git_commit must be non-empty")
        return V9ContinuationConfig(**values)


def _default_ensure_sam(**kwargs: Any) -> Path:
    scenes = load_scene_runtime_manifest(Path(kwargs["runtime_manifest"]))
    scene_id = str(kwargs["scene_id"])
    return ensure_v9_segment_everything(
        scene_id=scene_id,
        scene=scenes[scene_id],
        repo_root=Path(kwargs["workspace"]),
        output_root=Path(kwargs["sam_packed_root"]),
        sam_checkpoint=Path(kwargs["sam_checkpoint"]),
        reusable_root=(
            None
            if kwargs.get("sam_reusable_root") is None
            else Path(kwargs["sam_reusable_root"])
        ),
    )


def _default_resource_audit(**kwargs: Any) -> Mapping[str, Any]:
    cgroup_root = (
        "/sys/fs/cgroup" if Path("/sys/fs/cgroup/memory.max").is_file() else None
    )
    return read_v9_legacy_resources(
        kwargs["output_root"], cgroup_root=cgroup_root, disk_floor_gib=80.0
    )


@dataclass(frozen=True)
class V9ContinuationHooks:
    ensure_sam: Callable[..., Path] = _default_ensure_sam
    prepare_affinity: Callable[..., Mapping[str, Any]] = prepare_v9_affinity_inputs
    train_features: Callable[..., Mapping[str, Any]] = execute_v9_feature_training
    run_lifting: Callable[..., Mapping[str, Any]] = run_v9_lifting_banks
    run_legacy: Callable[..., Mapping[str, Any]] = execute_v9_legacy_runs
    run_banks: Callable[..., Mapping[str, Any]] = run_v9_banks
    evaluate_banks: Callable[..., Mapping[str, Any]] = evaluate_v9_candidate_banks
    replay: Callable[..., Mapping[str, Any]] = replay_v9_priors
    evaluate_predictions: Callable[..., Mapping[str, Any]] = evaluate_v9_predictions
    cleanup_scene: Callable[..., Mapping[str, Any]] | None = None
    audit_resources: Callable[..., Mapping[str, Any]] = _default_resource_audit


def _identity(config: V9ContinuationConfig) -> dict[str, Any]:
    return {
        "schema": "saga-v9-stage3-6-identity-v1",
        "git_commit": config.git_commit,
        "stage2_status": str(config.stage2_status),
        "runtime_manifest": str(config.runtime_manifest),
        "locked_runtime_manifest": str(config.locked_runtime_manifest),
        "locked_evaluation_scenes": str(config.locked_evaluation_scenes),
        "category_priors": str(config.category_priors),
        "t1_b1_root": str(config.t1_b1_root),
        "sam_packed_root": str(config.sam_packed_root),
        "sam_reusable_root": (
            None if config.sam_reusable_root is None else str(config.sam_reusable_root)
        ),
    }


def _status_path(config: V9ContinuationConfig) -> Path:
    return config.artifacts_root / "v9_stage3_6_status.json"


def _record(
    config: V9ContinuationConfig,
    *,
    state: str,
    checkpoint: str,
    **payload: Any,
) -> dict[str, Any]:
    result = {
        "schema": "saga-v9-stage3-6-status-v1",
        "identity": _identity(config),
        "state": state,
        "checkpoint": checkpoint,
        **payload,
    }
    write_json(_status_path(config), result)
    return result


def _terminal(config: V9ContinuationConfig) -> dict[str, Any] | None:
    path = _status_path(config)
    if not path.is_file():
        return None
    try:
        status = load_json(path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if status.get("identity") != _identity(config):
        return None
    return status if status.get("state") in {"complete", "stopped"} else None


def _stage2_selection(config: V9ContinuationConfig) -> str:
    status = load_json(config.stage2_status)
    if status.get("state") != "complete" or status.get("checkpoint") != "stage2-objectbank-selected":
        raise ValueError("V9 Stage 2 has not completed the registered object-bank selection")
    selection = status.get("selection", {})
    mode = str(selection.get("selected_association", ""))
    if mode not in {"A0", "A1", "A2", "A3"}:
        raise ValueError("Stage-2 association selection is invalid")
    return mode


def _select_dev8_classifier(
    config: V9ContinuationConfig,
    hooks: V9ContinuationHooks,
    *,
    association_mode: str,
) -> dict[str, Any]:
    """Freeze the late classifier from DEV8 without touching bank geometry.

    Both views are evaluated over the already-frozen selected-association
    banks.  Ground truth is supplied only to the evaluator hook; neither a
    worker nor proposal replay is invoked here.
    """

    evaluations: dict[str, dict[str, Any]] = {}
    taxonomy = load_taxonomy(config.taxonomy_path)
    for classifier in ("mv-label", "codebook"):
        stem = classifier.replace("-", "_")
        evaluations[classifier] = dict(
            hooks.evaluate_banks(
                runtime_manifest=config.runtime_manifest,
                gt_dir=config.gt_dir,
                bank_root=config.runs_root / "object-banks",
                scene_ids=DEV8,
                association_mode=association_mode,
                classifier=classifier,
                taxonomy=taxonomy,
                rows_output=config.artifacts_root
                / f"object_bank8_classifier_{stem}.parquet",
                analysis_output=config.artifacts_root
                / f"object_bank8_classifier_{stem}.json",
                size_bins=config.size_bins,
            )
        )
    selection = select_v9_late_classifier(
        evaluations["mv-label"], evaluations["codebook"]
    )
    result = {
        "schema": "saga-v9-late-classifier-selection8-v1",
        "association_mode": association_mode,
        "development_scenes": list(DEV8),
        **selection,
    }
    write_json(config.artifacts_root / "late_classifier_selection8.json", result)
    return result


def _scene_manifest(
    config: V9ContinuationConfig,
    *,
    source_manifest: Path,
    scene_id: str,
    scene: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Path:
    target = config.artifacts_root / "runtime" / f"{scene_id}.json"
    write_json(
        target,
        {
            "kind": "scene_runtime_manifest",
            "schema_version": "v9-streamed-scene",
            "source_manifest": str(source_manifest),
            "scenes": [{**dict(scene), **dict(overrides), "scene_id": scene_id}],
        },
    )
    return target


def _valid_prediction(path: Path) -> bool:
    try:
        payload = load_json(path)
        labels = np.asarray(payload.get("point_labels"))
        instances = payload.get("instances")
        if labels.ndim != 1 or not len(labels) or not isinstance(instances, Mapping):
            return False
        validate_prediction_contract(labels, instances)
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _validate_t1_reference(config: V9ContinuationConfig) -> dict[str, Any]:
    """Reject historical-contributor or non-contract teacher references."""

    rows: list[dict[str, Any]] = []
    for scene_id in DEV8:
        target = config.t1_b1_root / config.t1_b1_condition / scene_id
        if not (target / "output.json").is_file():
            target = target / "seed-42"
        output = load_json(target / "output.json")
        contract = output.get("prediction_contract")
        if not isinstance(contract, Mapping) or contract.get("schema") != "saga-strict-prediction-contract-v1":
            raise ValueError(f"{scene_id}: T1-B1 lacks the strict prediction contract")
        labels = np.asarray(output.get("point_labels"))
        instances = output.get("instances")
        if not isinstance(instances, Mapping):
            raise ValueError(f"{scene_id}: T1-B1 lacks instance metadata")
        validate_prediction_contract(labels, instances)
        if int(contract.get("point_count", -1)) != len(labels):
            raise ValueError(f"{scene_id}: T1-B1 contract point count is invalid")
        record = load_json(target / "run.json")
        identity = record.get("identity")
        if record.get("status") != "complete" or not isinstance(identity, Mapping):
            raise ValueError(f"{scene_id}: T1-B1 run record is not complete")
        if str(identity.get("git_commit", "")) != config.git_commit:
            raise ValueError(f"{scene_id}: T1-B1 was not produced by the fixed V9 commit")
        command = tuple(map(str, identity.get("command", ())))
        joined = " ".join(command).lower()
        if "historical" in joined or "--teacher-prior-mode original" not in joined:
            raise ValueError(f"{scene_id}: T1-B1 contributor/teacher path is not registered")
        rows.append({"scene_id": scene_id, "root": str(target), "git_commit": config.git_commit})
    result = {
        "schema": "saga-v9-t1-b1-reference-audit-v1",
        "condition": config.t1_b1_condition,
        "corrected_contributor": True,
        "strict_contract": True,
        "scenes": rows,
    }
    write_json(config.artifacts_root / "t1_b1_reference_audit.json", result)
    return result


def _legacy_complete(root: Path, scene_id: str, git_commit: str) -> bool:
    for condition in ("F10k-B0", "F10k-B1"):
        target = root / condition / scene_id / "seed-42"
        if not (
            _valid_prediction(target / "output.json")
            and (target / "diagnostics.json").is_file()
            and (target / "stage_trace.npz").is_file()
            and (target / "stage_trace.json").is_file()
        ):
            return False
        try:
            record = load_json(target / "run.json")
            identity = record.get("identity")
            if (
                record.get("status") != "complete"
                or not isinstance(identity, Mapping)
                or identity.get("git_commit") != git_commit
                or identity.get("condition") != condition
            ):
                return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
    return True


def _downstream_complete(
    config: V9ContinuationConfig,
    *,
    scene_id: str,
    association_mode: str,
    include_legacy: bool,
) -> bool:
    lifting = config.runs_root / "lifting" / "S-AM" / scene_id
    feature_record_path = v9_feature_training_paths(
        config.runs_root / "feature-10k-objectbank", scene_id
    ).record
    try:
        feature_record = load_json(feature_record_path)
        feature_identity = feature_record.get("identity")
        if (
            feature_record.get("status") != "complete"
            or feature_record.get("git_commit") != config.git_commit
            or not isinstance(feature_identity, Mapping)
            or not lifting_bank_is_complete(
                lifting,
                expected_scene_id=scene_id,
                expected_git_commit=config.git_commit,
                expected_feature_record_identity=feature_identity,
            )
        ):
            return False
        lifting_metadata, _ = load_lifting_bank(lifting)
        lifting_identity = lifting_metadata.get("identity")
        if not isinstance(lifting_identity, Mapping):
            return False
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    bank = config.runs_root / "object-banks" / association_mode / scene_id
    return object_bank_is_complete(
        bank,
        expected_scene_id=scene_id,
        expected_mode=association_mode,
        expected_source_lifting=lifting,
        expected_config=V9Config().as_json(),
        expected_git_commit=config.git_commit,
        expected_lifting_identity=lifting_identity,
    ) and (
        not include_legacy
        or _legacy_complete(
            config.runs_root / "f10k-legacy", scene_id, config.git_commit
        )
    )


def _safe_cleanup_scene(
    *,
    config: V9ContinuationConfig,
    scene_id: str,
    resource_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evict only explicitly registered, reproducible 10k intermediates."""

    audited = dict(
        resource_audit
        if resource_audit is not None
        else _default_resource_audit(output_root=config.runs_root)
    )
    root = config.runs_root.resolve()
    feature = v9_feature_training_paths(root / "feature-10k-objectbank", scene_id)
    affinity = v9_affinity_input_paths(root / "feature-10k-objectbank", scene_id)
    candidates = (
        feature.model,
        feature.feature_ply,
        feature.scale_gate,
        affinity.masks,
        affinity.mask_scales,
        affinity.scale_model,
    )
    removed: list[str] = []
    preserved_part_files: list[str] = []
    for candidate in candidates:
        target = candidate.resolve()
        if root not in target.parents or target.name.endswith(".part"):
            raise RuntimeError(f"refusing unsafe V9 cleanup target: {target}")
        if target.is_dir():
            part_files = sorted(
                path.resolve()
                for path in target.rglob("*")
                if path.name.endswith(".part")
            )
            if part_files:
                preserved_part_files.extend(map(str, part_files))
                continue
            shutil.rmtree(target)
            removed.append(str(target))
        elif target.is_file():
            target.unlink()
            removed.append(str(target))
    packed_scene = (config.sam_packed_root / scene_id).resolve()
    if packed_scene.is_dir():
        if root not in packed_scene.parents:
            raise RuntimeError(f"refusing to evict SAM masks outside V9 runs: {packed_scene}")
        for target in sorted(packed_scene.glob("*.npz")):
            resolved = target.resolve()
            if root not in resolved.parents or resolved.name.endswith(".part"):
                raise RuntimeError(f"refusing unsafe packed-mask cleanup: {resolved}")
            resolved.unlink()
            removed.append(str(resolved))
    record = {
        "schema": "saga-v9-reproducible-intermediate-cleanup-v1",
        "scene_id": scene_id,
        "removed": removed,
        "preserved_part_files": preserved_part_files,
        "preserved_logs_and_records": True,
        "resources_before_cleanup": audited,
    }
    write_json(feature.root / "cleanup.json", record)
    return record


def _process_scenes(
    config: V9ContinuationConfig,
    hooks: V9ContinuationHooks,
    *,
    source_manifest: Path,
    scene_ids: Sequence[str],
    association_mode: str,
    include_legacy: bool,
) -> list[dict[str, Any]]:
    scenes = load_scene_runtime_manifest(source_manifest)
    missing = sorted(set(map(str, scene_ids)).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    progress: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        if _downstream_complete(
            config,
            scene_id=scene_id,
            association_mode=association_mode,
            include_legacy=include_legacy,
        ):
            resources_before_cleanup = hooks.audit_resources(
                output_root=config.runs_root,
                scene_id=scene_id,
                action="cleanup-reused",
            )
            cleanup = (
                _safe_cleanup_scene(
                    config=config,
                    scene_id=scene_id,
                    resource_audit=resources_before_cleanup,
                )
                if hooks.cleanup_scene is None
                else hooks.cleanup_scene(config=config, scene_id=scene_id)
            )
            progress.append({
                "scene_id": scene_id,
                "status": "reused",
                "resources_before_cleanup": resources_before_cleanup,
                "cleanup": cleanup,
            })
            continue
        packed = hooks.ensure_sam(
            runtime_manifest=source_manifest,
            scene_id=scene_id,
            workspace=config.workspace,
            sam_packed_root=config.sam_packed_root,
            sam_checkpoint=config.sam_checkpoint,
            sam_reusable_root=config.sam_reusable_root,
        )
        prepared = hooks.prepare_affinity(
            workspace=config.workspace,
            scene=scenes[scene_id],
            scene_id=scene_id,
            packed_masks_root=packed,
            output_root=config.runs_root / "feature-10k-objectbank",
            git_commit=config.git_commit,
            resume=True,
        )
        manifest = _scene_manifest(
            config,
            source_manifest=source_manifest,
            scene_id=scene_id,
            scene=scenes[scene_id],
            overrides=prepared["scene_overrides"],
        )
        resources_before_training = hooks.audit_resources(
            output_root=config.runs_root,
            scene_id=scene_id,
            action="train-10k",
        )
        hooks.train_features(
            scene_manifest=manifest,
            output_root=config.runs_root / "feature-10k-objectbank",
            workspace=config.workspace,
            git_commit=config.git_commit,
            scene_ids=(scene_id,),
            resume=True,
            continue_on_error=False,
        )
        feature = v9_feature_training_paths(
            config.runs_root / "feature-10k-objectbank", scene_id
        ).feature_ply
        hooks.run_lifting(
            manifest,
            (scene_id,),
            config.runs_root / "lifting" / "S-AM",
            config.workspace,
            mask_source="S",
            lifting_source="AM",
            sam_masks_root=config.sam_packed_root,
            sam_checkpoint=config.sam_checkpoint,
            label_features=config.label_features,
            feature_ply_by_scene={scene_id: feature},
            sam_scene_roots={scene_id: packed},
            git_commit=config.git_commit,
            contributor_audit=False,
        )
        if include_legacy:
            hooks.run_legacy(
                scene_manifest=manifest,
                feature_root=config.runs_root / "feature-10k-objectbank",
                output_root=config.runs_root / "f10k-legacy",
                workspace=config.workspace,
                git_commit=config.git_commit,
                scene_ids=(scene_id,),
                feature_git_commit=config.git_commit,
                resume=True,
                continue_on_error=False,
            )
        hooks.run_banks(
            lifting_root=config.runs_root / "lifting" / "S-AM",
            output_root=config.runs_root / "object-banks",
            scene_ids=(scene_id,),
            association_modes=(association_mode,),
            git_commit=config.git_commit,
        )
        if not _downstream_complete(
            config,
            scene_id=scene_id,
            association_mode=association_mode,
            include_legacy=include_legacy,
        ):
            raise RuntimeError(f"{scene_id}: downstream V9 artifacts are incomplete")
        resources_before_cleanup = hooks.audit_resources(
            output_root=config.runs_root,
            scene_id=scene_id,
            action="cleanup-complete",
        )
        cleanup = (
            _safe_cleanup_scene(
                config=config,
                scene_id=scene_id,
                resource_audit=resources_before_cleanup,
            )
            if hooks.cleanup_scene is None
            else hooks.cleanup_scene(config=config, scene_id=scene_id)
        )
        progress.append({
            "scene_id": scene_id,
            "status": "complete",
            "resources_before_training": resources_before_training,
            "resources_before_cleanup": resources_before_cleanup,
            "cleanup": cleanup,
        })
    return progress


def _replay_complete(
    root: Path,
    *,
    condition: str,
    scene_id: str,
    classifier: str,
    threshold: float,
) -> bool:
    target = root / condition / scene_id
    try:
        diagnostics = load_json(target / "diagnostics.json")
        return (
            _valid_prediction(target / "output.json")
            and diagnostics.get("condition") == condition
            and diagnostics.get("classifier") == classifier
            and np.isclose(float(diagnostics.get("acceptance_threshold", -1)), threshold)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _replay_resume(
    hooks: V9ContinuationHooks,
    *,
    bank_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    classifier: str,
    conditions: Sequence[str],
    category_priors: Path,
    threshold: float,
) -> None:
    for scene_id in map(str, scene_ids):
        missing = [
            condition
            for condition in map(str, conditions)
            if not _replay_complete(
                output_root,
                condition=condition,
                scene_id=scene_id,
                classifier=classifier,
                threshold=threshold,
            )
        ]
        if missing:
            hooks.replay(
                bank_root=bank_root,
                output_root=output_root,
                scene_ids=(scene_id,),
                classifier=classifier,
                conditions=tuple(missing),
                category_priors=category_priors,
                acceptance_threshold=threshold,
            )


def _evaluate(
    config: V9ContinuationConfig,
    hooks: V9ContinuationHooks,
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    stem: str,
    viewer_output: Path | None = None,
) -> dict[str, Any]:
    return dict(
        hooks.evaluate_predictions(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            prediction_root=root,
            scene_ids=tuple(scene_ids),
            conditions=tuple(conditions),
            taxonomy=load_taxonomy(config.taxonomy_path),
            metrics_output=config.artifacts_root / f"{stem}.parquet",
            analysis_output=config.artifacts_root / f"{stem}.json",
            size_bins=config.size_bins,
            viewer_output=viewer_output,
        )
    )


def _threshold_safe(uniform: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return (
        float(uniform["map_50_95"]) >= float(baseline["map_50_95"]) - 0.001 - 1e-12
        and float(uniform["ap50"]) >= float(baseline["ap50"]) - 0.002 - 1e-12
        and int(uniform["predicted_instance_count"])
        <= 1.25 * max(int(baseline["predicted_instance_count"]), 1)
        and int(uniform["orphan_gaussian_count"]) == 0
        and int(uniform["negative_metadata_count"]) == 0
    )


def _select_threshold(
    config: V9ContinuationConfig,
    hooks: V9ContinuationHooks,
    *,
    association_mode: str,
    classifier: str,
) -> dict[str, Any]:
    legacy = _evaluate(
        config,
        hooks,
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
        root=config.runs_root / "f10k-legacy",
        scene_ids=THRESHOLD2,
        conditions=("F10k-B0",),
        stem="threshold2_f10k_b0",
    )
    baseline = metrics_by_condition(legacy)["F10k-B0"]
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        root = config.runs_root / "threshold-replay" / f"t-{int(round(threshold * 100)):02d}"
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / association_mode,
            output_root=root,
            scene_ids=THRESHOLD2,
            classifier=classifier,
            conditions=("U000",),
            category_priors=config.category_priors,
            threshold=threshold,
        )
        analysis = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=root,
            scene_ids=THRESHOLD2,
            conditions=("U000",),
            stem=f"threshold2_{int(round(threshold * 100)):02d}",
        )
        metrics = metrics_by_condition(analysis)["U000"]
        rows.append(
            {
                "threshold": threshold,
                "structure_safe": _threshold_safe(metrics, baseline),
                **metrics,
            }
        )
    safe = [row for row in rows if row["structure_safe"]]
    if not safe:
        return {"passed": False, "rows": rows, "reason": "no U000 threshold passed structural safety"}
    selected = max(safe, key=lambda row: (float(row["map_50_95"]), float(row["threshold"])))
    result = {"passed": True, "selected_threshold": float(selected["threshold"]), "rows": rows}
    write_rows(config.artifacts_root / "uniform_threshold2.parquet", rows)
    write_json(config.artifacts_root / "uniform_threshold_selection2.json", result)
    return result


def _bank_health_metrics(
    candidate: Mapping[str, Any], replay: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "geometric_match_050_count": candidate["geometric_match_050_count"],
        "geometric_match_050_scene_count": candidate["geometric_match_050_scene_count"],
        "same_class_match_050_count": candidate["same_class_match_050_count"],
        "same_class_match_050_scene_count": candidate["same_class_match_050_scene_count"],
        "same_class_candidate_precision_025": candidate["same_class_candidate_precision_025"],
        "tiny_small_recall_025": candidate["tiny_small_recall_025"],
        "score_iou_spearman": candidate["score_iou_spearman"],
        **{
            key: replay[key]
            for key in (
                "gaussian_micro_precision",
                "unsupported_instance_fraction",
                "gt_recall",
                "map_50_95",
                "ap50",
                "predicted_instance_count",
                "orphan_gaussian_count",
                "negative_metadata_count",
            )
        },
    }


def _scene_method_rows(analysis: Mapping[str, Any], condition: str) -> list[SceneMethodMetrics]:
    return [
        SceneMethodMetrics(
            scene_id=str(row["scene_id"]),
            map_50_95=float(row["map_50_95"]),
            tiny_small_match_050_count=int(row["tiny_small_match_050_count"]),
            tiny_small_gt_count=int(row["tiny_small_gt_count"]),
            false_positive_count=int(row["false_positive_count"]),
            true_positive_count=int(row["true_positive_count"]),
        )
        for row in scene_metrics(analysis, condition)
    ]


def _mechanical_effect(root: Path, scenes: Sequence[str], condition: str) -> dict[str, Any]:
    deltas: list[float] = []
    changed = False
    for scene_id in map(str, scenes):
        uniform = load_json(root / "U000" / scene_id / "diagnostics.json")
        data = load_json(root / condition / scene_id / "diagnostics.json")
        u_scores = {
            int(row["candidate_id"]): float(row["score"])
            for row in uniform["candidate_scores"]
        }
        d_scores = {
            int(row["candidate_id"]): float(row["score"])
            for row in data["candidate_scores"]
        }
        if set(u_scores) != set(d_scores):
            raise ValueError("prior replay changed the frozen candidate set")
        deltas.extend(d_scores[key] - u_scores[key] for key in sorted(u_scores))
        changed |= any(
            uniform[key] != data[key]
            for key in (
                "accepted_candidate_ids",
                "suppressed_candidate_ids",
                "dropped_small_candidate_ids",
            )
        )
        owners: list[np.ndarray] = []
        for current_condition, diagnostics in (("U000", uniform), (condition, data)):
            output = load_json(root / current_condition / scene_id / "output.json")
            labels = np.asarray(output["point_labels"], dtype=np.int64)
            owner = np.full(len(labels), -1, dtype=np.int32)
            metadata = diagnostics.get("instances", {})
            for raw_id, row in metadata.items():
                instance_id = int(raw_id)
                owner[labels == instance_id] = int(row["candidate_id"])
            owners.append(owner)
        changed |= not np.array_equal(owners[0], owners[1])
    return {"score_deltas": deltas, "accepted_or_ownership_changed": changed}


def _prior_selection(
    analysis: Mapping[str, Any], replay_root: Path
) -> dict[str, Any]:
    aggregates = metrics_by_condition(analysis)
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:]:
        effect = _mechanical_effect(replay_root, DEV8, condition)
        gate = stage4_prior_gate(
            _scene_method_rows(analysis, "U000"),
            _scene_method_rows(analysis, condition),
            candidate_score_deltas=effect["score_deltas"],
            accepted_or_ownership_changed=effect["accepted_or_ownership_changed"],
        )
        factors = sum(CONDITION_FACTORS[condition])
        rows.append(
            {
                "condition": condition,
                "factor_count": factors,
                "ap50": aggregates[condition]["ap50"],
                **gate,
            }
        )
    passed = [row for row in rows if row["passed"]]
    if not passed:
        return {"passed": False, "rows": rows}
    selected = max(
        passed,
        key=lambda row: (
            float(row["mean_map_delta"]),
            float(row["tiny_small_recall_050_delta"]),
            float(row["ap50"]),
            -int(row["factor_count"]),
        ),
    )
    return {"passed": True, "best_condition": selected["condition"], "rows": rows}


def _locked_scenes(config: V9ContinuationConfig) -> tuple[str, ...]:
    payload = load_json(config.locked_evaluation_scenes)
    rows = payload.get("scenes", payload)
    expected = tuple(
        str(row["scene_id"] if isinstance(row, Mapping) else row) for row in rows
    )
    actual = tuple(load_scene_runtime_manifest(config.locked_runtime_manifest))
    if len(expected) != 48 or set(expected) != set(actual):
        raise ValueError("locked runtime must exactly match the registered 48 scenes")
    if len({physical_scene_id(scene) for scene in actual}) != 48:
        raise ValueError("locked runtime must contain 48 distinct physical scenes")
    if "scene0019_01" not in actual or "scene0019_00" in actual:
        raise ValueError("locked runtime must use the scene0019_01 replacement")
    return actual


def run_v9_stage3_to_6(
    config: V9ContinuationConfig,
    *,
    hooks: V9ContinuationHooks | None = None,
) -> dict[str, Any]:
    config = config.normalized()
    hooks = hooks or V9ContinuationHooks()
    terminal = _terminal(config)
    if terminal is not None:
        return terminal
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    config.runs_root.mkdir(parents=True, exist_ok=True)
    mode = _stage2_selection(config)
    try:
        _validate_t1_reference(config)
        _record(
            config,
            state="running",
            checkpoint="stage3-preparing-dev8",
            association_mode=mode,
        )
        progress = _process_scenes(
            config,
            hooks,
            source_manifest=config.runtime_manifest,
            scene_ids=DEV8,
            association_mode=mode,
            include_legacy=True,
        )
        classifier_selection = _select_dev8_classifier(
            config, hooks, association_mode=mode
        )
        classifier = str(classifier_selection["selected_classifier"])
        _record(
            config,
            state="running",
            checkpoint="stage3-late-classifier-selected",
            association_mode=mode,
            classifier=classifier,
            classifier_selection=classifier_selection,
        )
        threshold = _select_threshold(
            config, hooks, association_mode=mode, classifier=classifier
        )
        if not threshold["passed"]:
            return _record(
                config,
                state="stopped",
                checkpoint="stage3-uniform-threshold-unsafe",
                progress=progress,
                threshold=threshold,
                stop_reason="no preregistered U000 threshold preserved structural safety",
            )
        selected_threshold = float(threshold["selected_threshold"])
        replay_root = config.runs_root / "replay"
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / mode,
            output_root=replay_root,
            scene_ids=DEV8,
            classifier=classifier,
            conditions=("U000",),
            category_priors=config.category_priors,
            threshold=selected_threshold,
        )
        bank = dict(
            hooks.evaluate_banks(
                runtime_manifest=config.runtime_manifest,
                gt_dir=config.gt_dir,
                bank_root=config.runs_root / "object-banks",
                scene_ids=DEV8,
                association_mode=mode,
                classifier=classifier,
                taxonomy=load_taxonomy(config.taxonomy_path),
                rows_output=config.artifacts_root / "object_bank8.parquet",
                analysis_output=config.artifacts_root / "object_bank8_analysis.json",
                size_bins=config.size_bins,
            )
        )
        uniform_analysis = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=replay_root,
            scene_ids=DEV8,
            conditions=("U000",),
            stem="uniform8_metrics",
            viewer_output=config.artifacts_root / "viewer" / "dev8" / "objectbank-uniform",
        )
        legacy_analysis = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=config.runs_root / "f10k-legacy",
            scene_ids=DEV8,
            conditions=("F10k-B0", "F10k-B1"),
            stem="f10k_legacy8_metrics",
            viewer_output=config.artifacts_root / "viewer" / "dev8" / "legacy10k",
        )
        t1_analysis = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=config.t1_b1_root,
            scene_ids=DEV8,
            conditions=(config.t1_b1_condition,),
            stem="t1_b1_8_metrics",
            viewer_output=config.artifacts_root / "viewer" / "dev8" / "t1",
        )
        uniform_metrics = metrics_by_condition(uniform_analysis)["U000"]
        legacy_metrics = metrics_by_condition(legacy_analysis)
        t1_metrics = metrics_by_condition(t1_analysis)[config.t1_b1_condition]
        health_input = _bank_health_metrics(bank, uniform_metrics)
        health = stage3_uniform_health_gate(
            health_input,
            t1_b1=t1_metrics,
            f10k_b0=legacy_metrics["F10k-B0"],
        )
        health_payload = {
            "schema": "saga-v9-uniform-health8-v1",
            "selected_threshold": selected_threshold,
            "association_mode": mode,
            "classifier": classifier,
            "gate": health,
            "bank": bank,
            "n0_uniform": uniform_metrics,
            "t1_b1": t1_metrics,
            "f10k_b0": legacy_metrics["F10k-B0"],
            "f10k_b1": legacy_metrics["F10k-B1"],
        }
        write_json(config.artifacts_root / "uniform_health8.json", health_payload)
        if not health["passed"]:
            return _record(
                config,
                state="stopped",
                checkpoint="stage3-uniform-health-failed",
                selected_threshold=selected_threshold,
                uniform_health=health_payload,
                stop_reason="Clean ObjectBank failed the preregistered eight-scene health gate",
            )

        _record(config, state="running", checkpoint="stage4-prior-factorial8", selected_threshold=selected_threshold)
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / mode,
            output_root=replay_root,
            scene_ids=DEV8,
            classifier=classifier,
            conditions=CONDITIONS,
            category_priors=config.category_priors,
            threshold=selected_threshold,
        )
        factorial = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=replay_root,
            scene_ids=DEV8,
            conditions=CONDITIONS,
            stem="category_prior_factorial8",
            viewer_output=config.artifacts_root / "viewer" / "dev8" / "prior-factorial",
        )
        prior = _prior_selection(factorial, replay_root)
        write_json(config.artifacts_root / "category_prior_factorial8_analysis.json", prior)
        if not prior["passed"]:
            return _record(
                config,
                state="stopped",
                checkpoint="stage4-category-prior-failed",
                selected_threshold=selected_threshold,
                prior_selection=prior,
                stop_reason="uniform bank was healthy but no data-driven prior passed the registered gate",
            )
        best = str(prior["best_condition"])

        _record(config, state="running", checkpoint="stage5-holdout5", best_condition=best)
        _process_scenes(
            config,
            hooks,
            source_manifest=config.runtime_manifest,
            scene_ids=HOLDOUT5,
            association_mode=mode,
            include_legacy=False,
        )
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / mode,
            output_root=replay_root,
            scene_ids=HOLDOUT5,
            classifier=classifier,
            conditions=("U000", best),
            category_priors=config.category_priors,
            threshold=selected_threshold,
        )
        holdout = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=replay_root,
            scene_ids=HOLDOUT5,
            conditions=("U000", best),
            stem="holdout5_metrics",
        )
        ref = {row.scene_id: row for row in _scene_method_rows(holdout, "U000")}
        trt = {row.scene_id: row for row in _scene_method_rows(holdout, best)}
        deltas = np.asarray([trt[key].map_50_95 - ref[key].map_50_95 for key in sorted(ref)])
        u_gt = sum(row.tiny_small_gt_count for row in ref.values())
        d_gt = sum(row.tiny_small_gt_count for row in trt.values())
        if u_gt != d_gt:
            raise ValueError("holdout tiny/small GT denominators differ")
        tiny_delta = (
            sum(row.tiny_small_match_050_count for row in trt.values())
            - sum(row.tiny_small_match_050_count for row in ref.values())
        ) / u_gt if u_gt else 0.0
        holdout_gate = {
            "passed": bool(np.mean(deltas) > 0 and np.count_nonzero(deltas > 0) >= 3 and tiny_delta > 0),
            "mean_delta_map_50_95": float(np.mean(deltas)),
            "positive_scene_count": int(np.count_nonzero(deltas > 0)),
            "negative_scene_count": int(np.count_nonzero(deltas < 0)),
            "tiny_small_recall_050_delta": float(tiny_delta),
        }
        if not holdout_gate["passed"]:
            return _record(
                config,
                state="stopped",
                checkpoint="stage5-holdout5-failed",
                best_condition=best,
                holdout_gate=holdout_gate,
                stop_reason="data-driven prior failed the independent five-scene holdout",
            )

        tune_scenes = tuple(load_scene_runtime_manifest(config.runtime_manifest))
        if len(tune_scenes) != 24 or len({physical_scene_id(scene) for scene in tune_scenes}) != 13:
            raise ValueError("tune manifest must contain 24 scans from 13 physical scenes")
        remaining = tuple(scene for scene in tune_scenes if scene not in set(DEV8) | set(HOLDOUT5))
        _process_scenes(
            config,
            hooks,
            source_manifest=config.runtime_manifest,
            scene_ids=remaining,
            association_mode=mode,
            include_legacy=False,
        )
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / mode,
            output_root=replay_root,
            scene_ids=remaining,
            classifier=classifier,
            conditions=("U000", best),
            category_priors=config.category_priors,
            threshold=selected_threshold,
        )
        tune = _evaluate(
            config,
            hooks,
            runtime_manifest=config.runtime_manifest,
            gt_dir=config.gt_dir,
            root=replay_root,
            scene_ids=tune_scenes,
            conditions=("U000", best),
            stem="tune24_metrics",
        )
        tune_macro = physical_scene_macro_delta(tune, reference="U000", treatment=best)
        tune_macro["passed"] = tune_macro["macro_delta_map_50_95"] >= 0.002 - 1e-12
        write_json(config.artifacts_root / "tune24_physical_macro.json", tune_macro)
        if not tune_macro["passed"]:
            return _record(
                config,
                state="stopped",
                checkpoint="stage5-tune24-failed",
                best_condition=best,
                tune_macro=tune_macro,
                stop_reason="data-driven prior failed physical-scene-weighted tune24",
            )

        _record(config, state="running", checkpoint="stage6-final48", best_condition=best)
        locked = _locked_scenes(config)
        _process_scenes(
            config,
            hooks,
            source_manifest=config.locked_runtime_manifest,
            scene_ids=locked,
            association_mode=mode,
            include_legacy=False,
        )
        final_replay = config.runs_root / "replay-final48"
        _replay_resume(
            hooks,
            bank_root=config.runs_root / "object-banks" / mode,
            output_root=final_replay,
            scene_ids=locked,
            classifier=classifier,
            conditions=("U000", best),
            category_priors=config.category_priors,
            threshold=selected_threshold,
        )
        final = _evaluate(
            config,
            hooks,
            runtime_manifest=config.locked_runtime_manifest,
            gt_dir=config.locked_gt_dir,
            root=final_replay,
            scene_ids=locked,
            conditions=("U000", best),
            stem="final48_metrics",
            viewer_output=config.artifacts_root / "viewer" / "final48",
        )
        bootstrap = paired_scannet_bootstrap_from_predictions(
            runtime_manifest=config.locked_runtime_manifest,
            gt_dir=config.locked_gt_dir,
            prediction_root=final_replay,
            scene_ids=locked,
            reference_condition="U000",
            treatment_condition=best,
            taxonomy=load_taxonomy(config.taxonomy_path),
            samples=10_000,
            seed=20_260_804,
        )
        final_metrics = metrics_by_condition(final)
        official_delta = float(
            final_metrics[best]["map_50_95"]
            - final_metrics["U000"]["map_50_95"]
        )
        if not np.isclose(
            official_delta,
            float(bootstrap["delta_map_50_95"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "paired bootstrap point estimate does not reproduce pooled official mAP"
            )
        supports = (
            official_delta >= 0.002 - 1e-12
            and bootstrap["paired_bootstrap_ci95"][0] > 0
        )
        summary = {
            "schema": "saga-v9-final-analysis-v1",
            "association_mode": mode,
            "classifier": classifier,
            "acceptance_threshold": selected_threshold,
            "best_condition": best,
            "official_delta_map_50_95": official_delta,
            "bootstrap": bootstrap,
            "supports_stable_category_prior": supports,
            "uniform": final_metrics["U000"],
            "data": final_metrics[best],
        }
        write_json(config.artifacts_root / "final_analysis.json", summary)
        return _record(
            config,
            state="complete",
            checkpoint="stage6-final48-complete",
            best_condition=best,
            final_analysis=summary,
            stop_reason=(
                None
                if supports
                else "V9 proposal-level category prior showed no stable improvement"
            ),
        )
    except BaseException as error:
        _record(
            config,
            state="failed",
            checkpoint="stage3-6-exception",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
