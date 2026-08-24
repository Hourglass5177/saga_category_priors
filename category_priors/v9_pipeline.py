from __future__ import annotations

"""Sequential Stage-2 controller for the V9 Clean ObjectBank experiment.

The controller deliberately keeps training/lifting workers and offline GT
evaluation on opposite sides of a narrow filesystem boundary.  It prepares
the registered dual-source 10k input, freezes one S-AM lifting bank per scene,
applies the geometric feasibility gate, and only then constructs and evaluates
the registered association modes.  It never passes a GT path to a worker.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy, load_taxonomy
from .v9_evaluation import stage2_oracle_gate
from .v9_feature_training import (
    execute_v9_feature_training,
    prepare_v9_affinity_inputs,
    v9_feature_training_paths,
)
from .v9_legacy_runner import read_v9_legacy_resources
from .v9_runner import run_v9_banks
from .v9_lifting_runner import ensure_v9_segment_everything, run_v9_lifting_banks
from .v9_oracle import (
    evaluate_fragment_oracle,
    gaussian_ply,
    load_lifting_bank,
    map_gt_to_gaussian,
    runtime_rows,
    transform,
    unpack_ragged,
)
from .v9_metrics import evaluate_v9_candidate_banks


V9_STAGE2_SCENES = ("scene0645_00", "scene0025_01")
V9_PRIMARY_ASSOCIATIONS = ("A0", "A1", "A2")
V9_FALLBACK_ASSOCIATION = "A3"
V9_CLASSIFIERS = ("mv-label", "codebook")


def _default_resource_audit(**kwargs: Any) -> Mapping[str, Any]:
    cgroup_root = (
        "/sys/fs/cgroup" if Path("/sys/fs/cgroup/memory.max").is_file() else None
    )
    return read_v9_legacy_resources(
        kwargs["output_root"], cgroup_root=cgroup_root, disk_floor_gib=80.0
    )


@dataclass(frozen=True)
class V9Stage2Config:
    runtime_manifest: Path
    workspace: Path
    runs_root: Path
    artifacts_root: Path
    gt_dir: Path
    sam_packed_root: Path
    sam_checkpoint: Path
    label_features: Path
    size_bins: Path
    git_commit: str
    taxonomy_path: Path | None = None

    def normalized(self) -> "V9Stage2Config":
        values = asdict(self)
        for field in (
            "runtime_manifest",
            "workspace",
            "runs_root",
            "artifacts_root",
            "gt_dir",
            "sam_packed_root",
            "sam_checkpoint",
            "label_features",
            "size_bins",
        ):
            values[field] = Path(values[field]).resolve()
        if values["taxonomy_path"] is not None:
            values["taxonomy_path"] = Path(values["taxonomy_path"]).resolve()
        values["git_commit"] = str(values["git_commit"]).strip()
        if not values["git_commit"]:
            raise ValueError("git_commit must be non-empty")
        return V9Stage2Config(**values)


@dataclass(frozen=True)
class V9Stage2Hooks:
    ensure_masks: Callable[..., Path] = ensure_v9_segment_everything
    prepare_affinity_inputs: Callable[..., Mapping[str, Any]] = (
        prepare_v9_affinity_inputs
    )
    train_features: Callable[..., Mapping[str, Any]] = execute_v9_feature_training
    run_lifting: Callable[..., Mapping[str, Any]] = run_v9_lifting_banks
    evaluate_oracle: Callable[..., Mapping[str, Any]] | None = None
    run_banks: Callable[..., Mapping[str, Any]] = run_v9_banks
    evaluate_banks: Callable[..., Mapping[str, Any]] = evaluate_v9_candidate_banks
    audit_resources: Callable[..., Mapping[str, Any]] = _default_resource_audit


def _config_identity(config: V9Stage2Config) -> dict[str, Any]:
    return {
        "runtime_manifest": str(config.runtime_manifest),
        "workspace": str(config.workspace),
        "runs_root": str(config.runs_root),
        "artifacts_root": str(config.artifacts_root),
        "gt_dir": str(config.gt_dir),
        "sam_packed_root": str(config.sam_packed_root),
        "sam_checkpoint": str(config.sam_checkpoint),
        "label_features": str(config.label_features),
        "size_bins": str(config.size_bins),
        "taxonomy_path": (
            None if config.taxonomy_path is None else str(config.taxonomy_path)
        ),
        "git_commit": config.git_commit,
        "scene_ids": list(V9_STAGE2_SCENES),
    }


def _status_path(config: V9Stage2Config) -> Path:
    return config.artifacts_root / "v9_status.json"


def _write_status(
    config: V9Stage2Config,
    *,
    state: str,
    checkpoint: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "schema": "saga-v9-stage2-status-v1",
        "stage": "stage2-two-scene-10k-objectbank",
        "state": str(state),
        "checkpoint": str(checkpoint),
        "identity": _config_identity(config),
        **extra,
    }
    write_json(_status_path(config), payload)
    return payload


def _existing_terminal_status(config: V9Stage2Config) -> dict[str, Any] | None:
    path = _status_path(config)
    if not path.is_file():
        return None
    try:
        payload = load_json(path)
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("schema") != "saga-v9-stage2-status-v1":
        return None
    if payload.get("identity") != _config_identity(config):
        return None
    return payload if payload.get("state") in {"complete", "stopped"} else None


def _selected_runtime_rows(config: V9Stage2Config) -> dict[str, dict[str, Any]]:
    scenes = load_scene_runtime_manifest(config.runtime_manifest)
    missing = sorted(set(V9_STAGE2_SCENES) - set(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing V9 Stage-2 scenes: {missing}")
    return {scene_id: dict(scenes[scene_id]) for scene_id in V9_STAGE2_SCENES}


def write_augmented_runtime_manifest(
    *,
    config: V9Stage2Config,
    scenes: Mapping[str, Mapping[str, Any]],
    affinity_overrides: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write the two-scene manifest consumed by both training and lifting."""

    rows: list[dict[str, Any]] = []
    for scene_id in V9_STAGE2_SCENES:
        if scene_id not in scenes or scene_id not in affinity_overrides:
            raise ValueError(f"missing augmented runtime data for {scene_id}")
        row = {**dict(scenes[scene_id]), **dict(affinity_overrides[scene_id])}
        row["scene_id"] = scene_id
        rows.append(row)
    target = config.artifacts_root / "v9_stage2_runtime_manifest.json"
    write_json(
        target,
        {
            "kind": "scene_runtime_manifest",
            "schema_version": "v9-stage2",
            "source_manifest": str(config.runtime_manifest),
            "scenes": rows,
        },
    )
    return target


def evaluate_stage2_geometric_oracle(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    lifting_root: Path,
    scene_ids: Sequence[str],
    size_bins: Path,
    rows_output: Path,
    analysis_output: Path,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate frozen S-AM fragments; this is the first function that reads GT."""

    runtime = runtime_rows(runtime_manifest)
    size_spec = load_json(size_bins)
    rows: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    for scene_id in map(str, scene_ids):
        scene = runtime[scene_id]
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(gaussian_ply(scene)), transform(scene)
        )
        nearest, diagnostics = map_gt_to_gaussian(
            gt_xyz, gaussian_xyz, radius_m
        )
        mapping[scene_id] = diagnostics
        metadata, arrays = load_lifting_bank(lifting_root / scene_id)
        if metadata.get("schema") != "saga-v9-native-lifting-bank-v1":
            raise ValueError(f"{scene_id}: Stage-2 oracle requires native V9 lifting")
        if int(metadata["point_count"]) != len(gaussian_xyz):
            raise ValueError(f"{scene_id}: lifting/Gaussian point counts differ")
        fragments = unpack_ragged(
            arrays["fragment_full_indptr"], arrays["fragment_full_ids"]
        )
        rows.append(
            evaluate_fragment_oracle(
                scene_id=scene_id,
                fragment_gaussian_ids=fragments,
                gt_nearest_gaussian=nearest,
                gt_xyz=gt_xyz,
                gt_semantic=gt.semantic,
                gt_instance=gt.instance,
                size_spec=size_spec,
                min_region_size=min_region_size,
            )
        )
    official_gt = sum(int(row["official_gt_count"]) for row in rows)
    matches_050 = sum(
        int(row["geometric_greedy_match_050_count"]) for row in rows
    )
    tiny_gt = sum(int(row["tiny_small_gt_count"]) for row in rows)
    tiny_matches = sum(
        int(row["tiny_small_geometric_match_025_count"]) for row in rows
    )
    aggregate = {
        "scene_count": len(rows),
        "official_valid_gt_count": official_gt,
        "geometric_match_050_count": matches_050,
        "geometric_recall_050": matches_050 / official_gt if official_gt else 0.0,
        "tiny_small_official_valid_gt_count": tiny_gt,
        "geometric_tiny_small_recall_025": (
            tiny_matches / tiny_gt if tiny_gt else 0.0
        ),
    }
    gate = stage2_oracle_gate(aggregate)
    result = {
        "schema": "saga-v9-stage2-geometric-oracle-v1",
        **aggregate,
        "gate": gate,
        "per_scene": rows,
        "mapping": mapping,
    }
    write_rows(
        rows_output,
        [
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list, tuple))
            }
            for row in rows
        ],
    )
    write_json(analysis_output, result)
    return result


def _association_error_counts(metrics: Mapping[str, Any]) -> tuple[int, int]:
    rows = metrics.get("association_per_scene", ())
    predicted = sum(int(row.get("predicted_pair_count", 0)) for row in rows)
    positives = sum(int(row.get("oracle_positive_pair_count", 0)) for row in rows)
    true_positive = sum(int(row.get("true_positive_pair_count", 0)) for row in rows)
    return max(predicted - true_positive, 0), max(positives - true_positive, 0)


def association_ranking_row(
    association_mode: str, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    merge_errors, split_errors = _association_error_counts(metrics)
    geometric = metrics["geometric"]
    return {
        "association_mode": association_mode,
        "geometric_match_050_count": int(metrics["geometric_match_050_count"]),
        "association_pair_precision": float(metrics["association_pair_precision"]),
        "association_pair_f1": float(metrics["association_pair_f1"]),
        "merge_error_proxy_count": merge_errors,
        "split_error_proxy_count": split_errors,
        "candidate_precision_025": float(geometric["candidate_precision_025"]),
    }


def _association_ranking_from_classifiers(
    association_mode: str,
    metrics_by_classifier: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse late-classifier views without letting semantics change geometry."""

    indexed: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {}
    for classifier, metrics in metrics_by_classifier.items():
        indexed[classifier] = {
            (str(row["scene_id"]), int(row.get("track_id", row["candidate_id"]))): row
            for row in metrics["per_candidate"]
        }
    keys = set(indexed["mv-label"])
    if any(set(rows) != keys for rows in indexed.values()):
        raise ValueError("late classifiers changed the frozen geometry track set")
    for key in keys:
        values = [float(rows[key]["geometric_best_iou"]) for rows in indexed.values()]
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise ValueError("late classifiers changed candidate geometry")
    representative = metrics_by_classifier["mv-label"]
    merge_errors, split_errors = _association_error_counts(representative)
    ious = [float(row["geometric_best_iou"]) for row in indexed["mv-label"].values()]
    return {
        "association_mode": association_mode,
        "geometric_match_050_count": sum(value >= 0.50 for value in ious),
        "association_pair_precision": float(
            representative["association_pair_precision"]
        ),
        "association_pair_f1": float(representative["association_pair_f1"]),
        "merge_error_proxy_count": merge_errors,
        "split_error_proxy_count": split_errors,
        "candidate_precision_025": (
            sum(value >= 0.25 for value in ious) / len(ious) if ious else 0.0
        ),
    }


def select_v9_association(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one association result is required")
    complexity = {"A0": 0, "A1": 1, "A2": 2, "A3": 3}
    by_mode = {str(row["association_mode"]): dict(row) for row in rows}
    if len(by_mode) != len(rows):
        raise ValueError("association results must have unique modes")
    selected = min(
        by_mode,
        key=lambda mode: (
            -int(by_mode[mode]["geometric_match_050_count"]),
            -float(by_mode[mode]["association_pair_precision"]),
            -float(by_mode[mode]["association_pair_f1"]),
            int(by_mode[mode]["merge_error_proxy_count"])
            + int(by_mode[mode]["split_error_proxy_count"]),
            -float(by_mode[mode]["candidate_precision_025"]),
            complexity[mode],
        ),
    )
    return {
        "selected_association": selected,
        "ranking_order": [
            "geometric_match_050_count",
            "association_pair_precision",
            "association_pair_f1",
            "merge_split_error",
            "candidate_precision_025",
            "simpler_association",
        ],
        "rows": [by_mode[mode] for mode in sorted(by_mode, key=complexity.get)],
    }


def select_v9_late_classifier(
    mv_metrics: Mapping[str, Any],
    codebook_metrics: Mapping[str, Any],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Choose MV within two points of codebook on geometrically eligible candidates."""

    def indexed(metrics: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
        return {
            (
                str(row["scene_id"]),
                int(row.get("track_id", row["candidate_id"])),
            ): row
            for row in metrics["per_candidate"]
        }

    mv = indexed(mv_metrics)
    codebook = indexed(codebook_metrics)
    keys = set(mv) | set(codebook)
    eligible = [
        key
        for key in keys
        if max(
            float(rows[key]["geometric_best_iou"])
            for rows in (mv, codebook)
            if key in rows
        )
        >= 0.25
    ]
    for key in eligible:
        if key in mv and key in codebook and not np.isclose(
            float(mv[key]["geometric_best_iou"]),
            float(codebook[key]["geometric_best_iou"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("late classifiers changed frozen candidate geometry")

    def correct(rows: Mapping[tuple[str, int], Mapping[str, Any]]) -> int:
        return sum(
            key in rows
            and str(rows[key]["class"])
            == str(rows[key]["geometric_best_gt_class"])
            for key in eligible
        )

    mv_correct = correct(mv)
    codebook_correct = correct(codebook)
    denominator = len(eligible)
    mv_accuracy = mv_correct / denominator if denominator else 0.0
    codebook_accuracy = codebook_correct / denominator if denominator else 0.0
    selected = (
        "mv-label"
        if mv_accuracy >= codebook_accuracy - float(tolerance) - 1e-12
        else "codebook"
    )
    return {
        "selected_classifier": selected,
        "eligible_candidate_count": denominator,
        "mv_label_correct_count": mv_correct,
        "codebook_correct_count": codebook_correct,
        "mv_label_accuracy": mv_accuracy,
        "codebook_accuracy": codebook_accuracy,
        "mv_tolerance": float(tolerance),
    }


def _evaluate_association_mode(
    *,
    config: V9Stage2Config,
    hooks: V9Stage2Hooks,
    augmented_manifest: Path,
    taxonomy: Taxonomy,
    mode: str,
) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for classifier in V9_CLASSIFIERS:
        stem = f"{mode}_{classifier.replace('-', '_')}"
        evaluated = dict(hooks.evaluate_banks(
            runtime_manifest=augmented_manifest,
            gt_dir=config.gt_dir,
            bank_root=config.runs_root / "object-banks",
            scene_ids=V9_STAGE2_SCENES,
            association_mode=mode,
            classifier=classifier,
            taxonomy=taxonomy,
            rows_output=config.artifacts_root / f"{stem}.parquet",
            analysis_output=config.artifacts_root / f"{stem}.json",
            size_bins=config.size_bins,
        ))
        # Candidate IDs remain local serialization IDs.  Add the immutable
        # geometry track ID explicitly so classifier comparisons never depend
        # on serialization order or class eligibility.
        by_scene: dict[str, dict[int, int]] = {}
        for scene_id in V9_STAGE2_SCENES:
            bank_metadata = load_json(
                config.runs_root
                / "object-banks"
                / mode
                / scene_id
                / "object_bank.json"
            )
            by_scene[scene_id] = {
                int(row["candidate_id"]): int(row["track_id"])
                for row in bank_metadata["classifiers"][classifier]["candidates"]
            }
        evaluated["per_candidate"] = [
            {
                **dict(row),
                "track_id": by_scene[str(row["scene_id"])][int(row["candidate_id"])],
            }
            for row in evaluated["per_candidate"]
        ]
        results[classifier] = evaluated
    return results


def run_v9_stage2(
    config: V9Stage2Config,
    *,
    hooks: V9Stage2Hooks | None = None,
) -> dict[str, Any]:
    """Run or resume the complete, preregistered two-scene V9 Stage 2."""

    config = config.normalized()
    hooks = hooks or V9Stage2Hooks()
    terminal = _existing_terminal_status(config)
    if terminal is not None:
        return terminal
    config.runs_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    scenes = _selected_runtime_rows(config)
    progress: dict[str, Any] = {"prepared_scenes": [], "trained_scenes": []}
    _write_status(config, state="running", checkpoint="preparing-affinity-inputs")
    try:
        overrides: dict[str, Mapping[str, Any]] = {}
        sam_scene_roots: dict[str, Path] = {}
        for scene_id in V9_STAGE2_SCENES:
            sam_scene_roots[scene_id] = hooks.ensure_masks(
                scene_id=scene_id,
                scene=scenes[scene_id],
                repo_root=config.workspace,
                output_root=config.runs_root / "sam-everything",
                sam_checkpoint=config.sam_checkpoint,
                reusable_root=config.sam_packed_root,
            )
            result = hooks.prepare_affinity_inputs(
                workspace=config.workspace,
                scene=scenes[scene_id],
                scene_id=scene_id,
                packed_masks_root=sam_scene_roots[scene_id],
                output_root=config.runs_root / "feature-10k-objectbank",
                git_commit=config.git_commit,
                resume=True,
            )
            overrides[scene_id] = dict(result["scene_overrides"])
            progress["prepared_scenes"].append(scene_id)
            _write_status(
                config,
                state="running",
                checkpoint="preparing-affinity-inputs",
                progress=progress,
            )
        augmented = write_augmented_runtime_manifest(
            config=config, scenes=scenes, affinity_overrides=overrides
        )
        progress["augmented_runtime_manifest"] = str(augmented)

        for scene_id in V9_STAGE2_SCENES:
            resource_audit = hooks.audit_resources(
                output_root=config.runs_root,
                scene_id=scene_id,
                action="train-10k",
            )
            hooks.train_features(
                scene_manifest=augmented,
                output_root=config.runs_root / "feature-10k-objectbank",
                workspace=config.workspace,
                git_commit=config.git_commit,
                scene_ids=(scene_id,),
                resume=True,
                continue_on_error=False,
            )
            progress["trained_scenes"].append(scene_id)
            progress.setdefault("resource_audits", []).append(
                {"scene_id": scene_id, "action": "train-10k", **dict(resource_audit)}
            )
            _write_status(
                config,
                state="running",
                checkpoint="training-10k-features",
                progress=progress,
            )

        feature_ply = {
            scene_id: v9_feature_training_paths(
                config.runs_root / "feature-10k-objectbank", scene_id
            ).feature_ply
            for scene_id in V9_STAGE2_SCENES
        }
        lifting_root = config.runs_root / "lifting" / "S-AM"
        for scene_id in V9_STAGE2_SCENES:
            hooks.run_lifting(
                augmented,
                (scene_id,),
                lifting_root,
                config.workspace,
                sam_masks_root=config.sam_packed_root,
                sam_checkpoint=config.sam_checkpoint,
                sam_scene_roots=sam_scene_roots,
                label_features=config.label_features,
                feature_ply_by_scene=feature_ply,
                git_commit=config.git_commit,
                contributor_audit=False,
            )
            _write_status(
                config,
                state="running",
                checkpoint="freezing-sam-alpha-lifting",
                progress=progress,
            )

        evaluate_oracle = hooks.evaluate_oracle or evaluate_stage2_geometric_oracle
        oracle = evaluate_oracle(
            runtime_manifest=augmented,
            gt_dir=config.gt_dir,
            lifting_root=lifting_root,
            scene_ids=V9_STAGE2_SCENES,
            size_bins=config.size_bins,
            rows_output=config.artifacts_root / "v9_stage2_oracle2.parquet",
            analysis_output=config.artifacts_root / "v9_stage2_oracle2.json",
        )
        oracle_gate = (
            dict(oracle["gate"])
            if "gate" in oracle
            else stage2_oracle_gate(oracle)
        )
        if not oracle_gate["passed"]:
            return _write_status(
                config,
                state="stopped",
                checkpoint="stage2-geometric-oracle-failed",
                progress=progress,
                oracle=oracle,
                stop_reason=(
                    "10k S-AM geometric support failed the preregistered Stage-2 "
                    "gate; ObjectBank association and category priors were not run"
                ),
            )

        taxonomy = load_taxonomy(config.taxonomy_path)
        evaluations: dict[str, dict[str, Mapping[str, Any]]] = {}
        association_rows: list[dict[str, Any]] = []
        for mode in V9_PRIMARY_ASSOCIATIONS:
            hooks.run_banks(
                lifting_root=lifting_root,
                output_root=config.runs_root / "object-banks",
                scene_ids=V9_STAGE2_SCENES,
                association_modes=(mode,),
                git_commit=config.git_commit,
            )
            evaluations[mode] = _evaluate_association_mode(
                config=config,
                hooks=hooks,
                augmented_manifest=augmented,
                taxonomy=taxonomy,
                mode=mode,
            )
            association_rows.append(
                _association_ranking_from_classifiers(mode, evaluations[mode])
            )
            _write_status(
                config,
                state="running",
                checkpoint=f"evaluated-{mode}",
                progress=progress,
                oracle=oracle,
                association_rows=association_rows,
            )

        if max(row["geometric_match_050_count"] for row in association_rows) < 6:
            mode = V9_FALLBACK_ASSOCIATION
            hooks.run_banks(
                lifting_root=lifting_root,
                output_root=config.runs_root / "object-banks",
                scene_ids=V9_STAGE2_SCENES,
                association_modes=(mode,),
                git_commit=config.git_commit,
            )
            evaluations[mode] = _evaluate_association_mode(
                config=config,
                hooks=hooks,
                augmented_manifest=augmented,
                taxonomy=taxonomy,
                mode=mode,
            )
            association_rows.append(
                _association_ranking_from_classifiers(mode, evaluations[mode])
            )

        selection = select_v9_association(association_rows)
        selected_mode = str(selection["selected_association"])
        write_rows(
            config.artifacts_root / "object_association_ablation2.parquet",
            association_rows,
        )
        write_json(config.artifacts_root / "object_association_selection2.json", selection)

        if max(row["geometric_match_050_count"] for row in association_rows) < 6:
            return _write_status(
                config,
                state="stopped",
                checkpoint="stage2-object-association-failed",
                progress=progress,
                oracle=oracle,
                selection=selection,
                stop_reason=(
                    "10k features plus A3 could not recover six geometric IoU>=0.50 "
                    "candidates; current SAGA identity/tracker representation is insufficient"
                ),
            )
        return _write_status(
            config,
            state="complete",
            checkpoint="stage2-objectbank-selected",
            progress=progress,
            oracle=oracle,
            selection=selection,
            next_stage="stage3-eight-scene-classifier-selection-and-uniform-health",
        )
    except BaseException as error:
        _write_status(
            config,
            state="failed",
            checkpoint="stage2-exception",
            progress=progress,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
