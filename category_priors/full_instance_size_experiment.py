from __future__ import annotations

"""Registered minimal full-instance category-size prior experiment.

The experiment consumes the corrected teacher-compatible ``T1-B1`` stage
trace at ``merged_partition``.  It never reclusters points.  Ground truth is
loaded only by the evaluation functions in this module, after the immutable
candidate snapshot and 2-D vote evidence have been materialized.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evaluator import (
    GroundTruthScene,
    PredictedInstance,
    apply_transform,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from .gaussian_object_audit import evaluate_gaussian_object_precision
from .io import hash_json, load_json, read_rows, sha256_file, write_json, write_rows
from .prediction_contract import normalize_prediction, validate_prediction_contract
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import Taxonomy, load_taxonomy
from .v9_legacy_runner import (
    CLASSES_32,
    OTHER_CLASSES_8,
    SELECTED_CLASSES_28,
    _default_point_cloud,
    _resolve_scene_path,
)
from .v9_t1_runner import (
    V9_FEATURE_SEED,
    V9_T1_SCHEMA,
    build_v9_t1_invocation,
    execute_v9_t1_runs,
    v9_t1_paths,
    v9_t1_run_complete,
)


SCHEMA = "saga-full-instance-size-prior-v1"
DEV2 = ("scene0645_00", "scene0025_01")
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
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)
CONDITIONS = ("controlled-baseline", "global-size", "class-size")
THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
MIN_REGION_SIZE = 100
RADIUS_M = 0.05
REGISTERED_HISTORICAL_T1_PRODUCER = (
    "271dc1834f62a148c49f63e21102d2741bf46690"
)
_T1_COMMAND_PATH_PLACEHOLDERS = {
    "--progress_path": "<PROGRESS>",
    "--stage_trace_path": "<STAGE_TRACE>",
    "--json_path": "<OUTPUT_JSON>",
    "--prior_metadata_path": "<DIAGNOSTICS_JSON>",
}


@dataclass(frozen=True)
class ExperimentConfig:
    workspace: Path
    runtime_manifest: Path
    locked_runtime_manifest: Path
    t1_root: Path
    rebuild_t1_root: Path
    gt_dir: Path
    locked_gt_dir: Path
    train_stats: Path
    category_priors: Path
    size_bins: Path
    locked_evaluation_scenes: Path
    runs_root: Path
    artifacts_root: Path
    taxonomy_path: Path | None
    git_commit: str
    allow_rebuild_missing_traces: bool = False
    disk_floor_gib: float = 80.0
    cgroup_root: Path = Path("/sys/fs/cgroup")
    python_bin: Path | None = None

    def validate(self) -> None:
        required_files = (
            self.runtime_manifest,
            self.locked_runtime_manifest,
            self.train_stats,
            self.category_priors,
            self.size_bins,
            self.locked_evaluation_scenes,
        )
        for path in required_files:
            if not path.is_file():
                raise FileNotFoundError(path)
        if self.taxonomy_path is not None and not self.taxonomy_path.is_file():
            raise FileNotFoundError(self.taxonomy_path)
        if self.python_bin is not None and not self.python_bin.is_file():
            raise FileNotFoundError(self.python_bin)
        if not self.workspace.is_dir():
            raise FileNotFoundError(self.workspace)
        if not self.gt_dir.is_dir():
            raise FileNotFoundError(self.gt_dir)
        if not self.locked_gt_dir.is_dir():
            raise FileNotFoundError(self.locked_gt_dir)
        if not self.git_commit.strip():
            raise ValueError("git_commit must be non-empty")
        if self.disk_floor_gib < 0:
            raise ValueError("disk_floor_gib must be non-negative")


def _git_commit(workspace: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("cannot resolve workspace git commit")
    return result.stdout.strip()


def _point_cloud(scene: Mapping[str, Any]) -> Path:
    base = Path(str(scene["base_path"])).resolve()
    if scene.get("point_cloud_path"):
        return _resolve_scene_path(scene, ("point_cloud_path",), "")
    return _default_point_cloud(base).resolve()


def _transform(scene: Mapping[str, Any]) -> Sequence[Sequence[float]]:
    return scene.get(
        "gaussian_to_gt_transform",
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )


def _resolve_relative(path_value: Any, base: Path) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _gt_path(
    scene: Mapping[str, Any],
    gt_dir: Path,
    scene_id: str,
    *,
    locked_row: Mapping[str, Any] | None = None,
    locked_spec_dir: Path | None = None,
) -> Path:
    if locked_row is not None:
        registration_base = (
            locked_spec_dir.resolve()
            if locked_spec_dir is not None
            else gt_dir.resolve()
        )
        for key in ("gt_npz", "replacement_gt_npz", "replacement_gt", "gt_path"):
            if locked_row.get(key) not in (None, ""):
                return _resolve_relative(locked_row[key], registration_base)
    for key in ("gt_npz", "replacement_gt_npz", "replacement_gt", "gt_path"):
        if scene.get(key) not in (None, ""):
            return _resolve_relative(scene[key], Path(str(scene["base_path"])))
    return (gt_dir / f"{scene_id}.npz").resolve()


def _resource_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    disk_root = config.runs_root.parent if config.runs_root.parent.exists() else config.workspace
    available = shutil.disk_usage(disk_root).free
    if available < config.disk_floor_gib * 1024**3:
        raise RuntimeError(
            f"available disk {available / 1024**3:.2f} GiB is below "
            f"{config.disk_floor_gib:.2f} GiB"
        )
    result: dict[str, Any] = {"disk_available_bytes": int(available)}
    if config.cgroup_root.is_dir():
        for name in ("memory.current", "memory.max", "memory.events"):
            path = config.cgroup_root / name
            if path.is_file():
                result[name] = path.read_text(encoding="utf-8").strip()
    return result


def _normalized_t1_command(command: Any) -> tuple[str, ...]:
    """Canonicalize only paths that legitimately differ between T1 producers.

    Everything else -- including the 32-class ordering, selected/other-class
    lists and every postprocess flag -- remains byte-for-byte comparable.
    """

    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        raise TypeError("T1 command must be a token sequence")
    tokens = [str(value) for value in command]
    if len(tokens) < 2:
        raise ValueError("T1 command must contain interpreter and postprocess path")
    tokens[0] = "<PYTHON>"
    tokens[1] = "<POSTPROCESS>"
    for option, placeholder in _T1_COMMAND_PATH_PLACEHOLDERS.items():
        positions = [index for index, value in enumerate(tokens) if value == option]
        if len(positions) != 1 or positions[0] + 1 >= len(tokens):
            raise ValueError(f"T1 command must contain exactly one {option}")
        tokens[positions[0] + 1] = placeholder
    return tuple(tokens)


def _git_blob_identity(workspace: Path, commit: str, relative_path: str) -> dict[str, Any]:
    """Return the exact Git-blob identity for one registered producer source."""

    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=workspace,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"cannot resolve registered producer source {commit}:{relative_path}: {detail}"
        )
    payload = bytes(result.stdout)
    return {
        "git_commit": commit,
        "path": relative_path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _recorded_postprocess_matches_git_blob(
    *,
    config: ExperimentConfig,
    producer: str,
    command: Any,
    recorded_postprocess: Any,
) -> dict[str, Any] | None:
    """Verify the producer's postprocessor against the immutable Git blob.

    Historical workspaces may no longer exist, so their authoritative source is
    read with ``git show`` from the allow-listed producer commit.  Whenever the
    command's original postprocess path is still present (and always for a run
    claiming the current consumer commit), its bytes must also equal that blob.
    """

    if not isinstance(recorded_postprocess, Mapping):
        return None
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return None
    tokens = [str(value) for value in command]
    if len(tokens) < 2:
        return None
    try:
        producer_source = _git_blob_identity(config.workspace, producer, "postprocess.py")
        if int(recorded_postprocess.get("size_bytes", -1)) != int(
            producer_source["size_bytes"]
        ):
            return None
    except (OSError, TypeError, ValueError):
        return None
    recorded_source_hash = recorded_postprocess.get("sha256")
    if recorded_source_hash not in (None, "") and str(recorded_source_hash) != str(
        producer_source["sha256"]
    ):
        return None

    command_source = Path(tokens[1])
    if command_source.is_file():
        if sha256_file(command_source) != str(producer_source["sha256"]):
            return None
    elif producer == config.git_commit:
        # A current-commit run must point at source that is still auditable in
        # this deployment.  Only the registered historical producer may rely
        # exclusively on its immutable Git object after its workspace is gone.
        return None
    return producer_source


def _t1_artifact_hashes(paths: Any) -> dict[str, str]:
    """Hash all four frozen T1 artifacts used by the consumer."""

    return {
        "record_sha256": sha256_file(paths.record),
        "stage_trace_sha256": sha256_file(paths.stage_trace),
        "stage_trace_metadata_sha256": sha256_file(paths.stage_trace_metadata),
        "output_sha256": sha256_file(paths.output),
    }


def _registered_t1_identity_matches(
    *,
    config: ExperimentConfig,
    scene_id: str,
    expected_invocation: Any,
    paths: Any,
    record: Mapping[str, Any],
) -> bool:
    """Validate one historical/current T1 producer without trusting self-labels."""

    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        return False
    producer = str(identity.get("git_commit", "")).strip()
    if producer not in {REGISTERED_HISTORICAL_T1_PRODUCER, config.git_commit}:
        return False
    expected = {
        "schema": V9_T1_SCHEMA,
        "scene_id": scene_id,
        "condition": "T1-B1",
        "seed": V9_FEATURE_SEED,
        "input_budget": "existing-scene-feature-2k",
        "contributor_weight": "alpha_times_t_prev",
        "teacher_prior_mode": "original",
        "causal_level": "L0",
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        return False
    if any(
        identity.get(key) != expected_invocation.identity.get(key)
        for key in ("feature_ply", "scale_gate", "label_features")
    ):
        return False
    try:
        if _normalized_t1_command(identity.get("command")) != _normalized_t1_command(
            expected_invocation.identity.get("command")
        ):
            return False
    except (OSError, TypeError, ValueError):
        return False
    producer_source = _recorded_postprocess_matches_git_blob(
        config=config,
        producer=producer,
        command=identity.get("command"),
        recorded_postprocess=identity.get("postprocess"),
    )
    if producer_source is None:
        return False
    if not v9_t1_run_complete(paths, identity):
        return False
    try:
        hashes_before = _t1_artifact_hashes(paths)
        declared_hashes = record.get("artifact_sha256")
        if declared_hashes is not None and dict(declared_hashes) != hashes_before:
            return False
        # Reject artifacts being concurrently replaced while admission is in
        # progress.  These exact hashes are persisted in snapshot_identity and
        # therefore become the immutable resume contract after first import.
        hashes_after = _t1_artifact_hashes(paths)
    except (OSError, TypeError, ValueError):
        return False
    return hashes_before == hashes_after and all(
        len(value) == 64 for value in hashes_before.values()
    )


def _load_t1_paths(
    config: ExperimentConfig,
    scene_id: str,
    scene: Mapping[str, Any],
    *,
    runtime_manifest: Path | None = None,
) -> Any:
    expected_invocation = build_v9_t1_invocation(
        workspace=config.workspace,
        scene=scene,
        scene_id=scene_id,
        output_root=config.rebuild_t1_root,
        condition="T1-B1",
        git_commit=config.git_commit,
    )

    for root in (config.t1_root, config.rebuild_t1_root):
        paths = v9_t1_paths(root, "T1-B1", scene_id)
        try:
            record = load_json(paths.record)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if _registered_t1_identity_matches(
            config=config,
            scene_id=scene_id,
            expected_invocation=expected_invocation,
            paths=paths,
            record=record,
        ):
            return paths
    if not config.allow_rebuild_missing_traces:
        raise FileNotFoundError(
            f"{scene_id}: no complete identity-compatible T1-B1 trace; "
            "rebuild is disabled"
        )
    execute_v9_t1_runs(
        scene_manifest=(runtime_manifest or config.runtime_manifest),
        output_root=config.rebuild_t1_root,
        workspace=config.workspace,
        git_commit=config.git_commit,
        scene_ids=(scene_id,),
        conditions=("T1-B1",),
        resume=True,
        cgroup_root=config.cgroup_root,
        disk_floor_gib=config.disk_floor_gib,
        python_bin=config.python_bin,
    )
    paths = v9_t1_paths(config.rebuild_t1_root, "T1-B1", scene_id)
    record = load_json(paths.record)
    if not _registered_t1_identity_matches(
        config=config,
        scene_id=scene_id,
        expected_invocation=expected_invocation,
        paths=paths,
        record=record,
    ):
        raise RuntimeError(f"{scene_id}: rebuilt T1-B1 trace is incomplete")
    return paths


def _load_trace(paths: Any) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    with np.load(paths.stage_trace, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    metadata = load_json(paths.stage_trace_metadata)
    output = load_json(paths.output)
    required = (
        "merged_partition",
        "post_global_knn",
        "post_filter",
        "final_internal_labels",
        "exported_prediction",
    )
    point_count = len(output["point_labels"])
    for name in required:
        values = arrays.get(name)
        if values is None or values.shape != (point_count,):
            raise ValueError(f"{paths.stage_trace}: invalid {name}")
    if not np.array_equal(
        arrays["exported_prediction"], np.asarray(output["point_labels"], dtype=np.int64)
    ):
        raise ValueError("frozen trace exported_prediction differs from output.json")
    return arrays, metadata, output


def _raw_to_export(metadata: Mapping[str, Any]) -> dict[int, int]:
    raw = metadata.get("raw_instances")
    if not isinstance(raw, Mapping):
        raise TypeError("stage trace metadata lacks raw_instances")
    return {raw_id: new_id for new_id, raw_id in enumerate(sorted(map(int, raw)))}


def _trace_identity(paths: Any, workspace: Path) -> dict[str, Any]:
    record = load_json(paths.record)
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("T1 run identity is missing")
    producer = str(identity.get("git_commit", ""))
    artifact_hashes = _t1_artifact_hashes(paths)
    return {
        "producer_git_commit": producer,
        "normalized_command": list(_normalized_t1_command(identity.get("command"))),
        "postprocess_git_blob": _git_blob_identity(
            workspace, producer, "postprocess.py"
        ),
        "schema": str(identity.get("schema", "")),
        "scene_id": str(identity.get("scene_id", "")),
        "condition": str(identity.get("condition", "")),
        "input_budget": str(identity.get("input_budget", "")),
        "contributor_weight": str(identity.get("contributor_weight", "")),
        "seed": int(identity.get("seed", -1)),
        **artifact_hashes,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _snapshot_paths(config: ExperimentConfig, scene_id: str) -> dict[str, Path]:
    root = config.runs_root / "snapshots" / scene_id
    return {
        "root": root,
        "arrays": root / "snapshot.npz",
        "metadata": root / "snapshot.json",
        "votes": root / "gaussian_votes_33.npz",
    }


def _snapshot_complete(
    config: ExperimentConfig,
    paths: Mapping[str, Path],
    scene_id: str,
    *,
    expected_snapshot_identity: Mapping[str, Any] | None = None,
    expected_vote_identity: Mapping[str, Any] | None = None,
) -> bool:
    try:
        from .full_instance_vote import vote_evidence_is_complete

        metadata = load_json(paths["metadata"])
        with np.load(paths["arrays"], allow_pickle=False) as arrays:
            point_count = int(metadata["point_count"])
            base_rows = metadata["base_rows"]
            rows = metadata["rows"]
            required_arrays = {
                "merged_partition",
                "post_global_knn",
                "post_filter",
                "final_internal_labels",
                "exported_prediction",
            }
            return bool(
                metadata.get("schema") == SCHEMA
                and metadata.get("kind") == "full_instance_snapshot"
                and metadata.get("scene_id") == scene_id
                and metadata.get("consumer_git_commit") == config.git_commit
                and (
                    expected_snapshot_identity is None
                    or metadata.get("snapshot_identity")
                    == dict(expected_snapshot_identity)
                )
                and metadata.get("snapshot_arrays_sha256")
                == sha256_file(paths["arrays"])
                and metadata.get("vote_evidence_sha256")
                == sha256_file(paths["votes"])
                and metadata.get("base_rows_sha256") == hash_json(base_rows)
                and metadata.get("rows_sha256") == hash_json(rows)
                and metadata.get("empty_restoration_replayed_from_post_filter")
                is True
                and metadata.get("empty_restoration_native_scores_exact") is True
                and required_arrays.issubset(arrays.files)
                and arrays["merged_partition"].shape == (point_count,)
                and arrays["post_global_knn"].shape == (point_count,)
                and arrays["post_filter"].shape == (point_count,)
                and arrays["final_internal_labels"].shape == (point_count,)
                and arrays["exported_prediction"].shape == (point_count,)
                and vote_evidence_is_complete(
                    paths["votes"], expected_identity=expected_vote_identity
                )
            )
    except (
        EOFError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return False


def _plain_candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"member_indices", "vote_counts"}
    result: dict[str, Any] = {}
    for key, value in candidate.items():
        if key in excluded:
            continue
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def materialize_scene_snapshot(
    config: ExperimentConfig,
    scene_id: str,
    scene: Mapping[str, Any],
    taxonomy: Taxonomy,
    *,
    runtime_manifest: Path | None = None,
) -> dict[str, Any]:
    """Build the immutable no-GT snapshot and pre-KNN vote evidence."""

    from .full_instance_size_prior import build_full_instance_candidates
    from .full_instance_vote import (
        aggregate_instance_votes,
        build_gaussian_vote_evidence,
    )

    destination = _snapshot_paths(config, scene_id)
    t1_paths = _load_t1_paths(
        config, scene_id, scene, runtime_manifest=runtime_manifest
    )
    arrays, trace_metadata, output = _load_trace(t1_paths)
    t1_identity = _trace_identity(t1_paths, config.workspace)
    gaussian_xyz = load_ply_xyz(_point_cloud(scene))
    if len(gaussian_xyz) != len(arrays["merged_partition"]):
        raise ValueError(f"{scene_id}: Gaussian/trace point count mismatch")
    evidence = build_gaussian_vote_evidence(
        scene_id,
        scene,
        destination["votes"],
        class_names=CLASSES_32,
    )
    vote_identity = dict(evidence.metadata.get("input_identity", {}))
    snapshot_identity = {
        "schema": SCHEMA,
        "consumer_git_commit": config.git_commit,
        "scene_id": scene_id,
        "runtime_scene_hash": hash_json(dict(scene)),
        "t1_identity": t1_identity,
        "vote_input_identity": vote_identity,
        "taxonomy_sha256": taxonomy.content_hash,
        "taxonomy_classes": list(taxonomy.canonical_classes),
    }
    if _snapshot_complete(
        config,
        destination,
        scene_id,
        expected_snapshot_identity=snapshot_identity,
        expected_vote_identity=vote_identity,
    ):
        return load_json(destination["metadata"])
    vote_histograms = aggregate_instance_votes(arrays["merged_partition"], evidence)
    branch_classes = {
        int(key): str(value)
        for key, value in trace_metadata.get("branch_instance_classes", {}).items()
    }
    candidate_snapshot = build_full_instance_candidates(
        arrays["merged_partition"],
        gaussian_xyz,
        float(scene["scene_scale_m_per_unit"]),
        vote_histograms,
        CLASSES_32,
        scene_id=scene_id,
        branch_instance_classes=branch_classes,
        saga20_classes=taxonomy.canonical_classes,
    )
    candidates = candidate_snapshot.rows(include_members=False)
    raw_instances = trace_metadata.get("raw_instances", {})
    if not isinstance(raw_instances, Mapping):
        raise TypeError("trace metadata raw_instances must be a mapping")
    raw_to_export = _raw_to_export(trace_metadata)
    final_internal = arrays["final_internal_labels"].astype(np.int64, copy=False)
    post_knn = arrays["post_global_knn"].astype(np.int64, copy=False)
    post_filter = arrays["post_filter"].astype(np.int64, copy=False)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        raw_id = int(candidate["raw_instance_id"])
        member = arrays["merged_partition"] == raw_id
        exported_id = raw_to_export.get(raw_id)
        row = _plain_candidate_row(candidate)
        row.update(
            {
                "scene_id": scene_id,
                "physical_scene_id": physical_scene_id(scene_id),
                "post_knn_point_count": int(np.count_nonzero(post_knn == raw_id)),
                "post_filter_point_count": int(np.count_nonzero(post_filter == raw_id)),
                "post_knn_original_member_count": int(
                    np.count_nonzero(member & (post_knn == raw_id))
                ),
                "post_filter_original_member_count": int(
                    np.count_nonzero(member & (post_filter == raw_id))
                ),
                "final_internal_point_count": int(
                    np.count_nonzero(final_internal == raw_id)
                ),
                "exported_instance_id": exported_id,
                "exported": exported_id is not None,
                "exported_class": (
                    str(raw_instances[str(raw_id)]["class"])
                    if str(raw_id) in raw_instances
                    else None
                ),
            }
        )
        rows.append(row)
    is_big_gaussian = np.asarray(output.get("is_big_gaussian"), dtype=bool)
    if is_big_gaussian.shape != (len(gaussian_xyz),):
        raise ValueError(
            f"{scene_id}: frozen T1-B1 lacks a point-aligned is_big_gaussian mask"
        )
    _, no_restore_native, no_restore_audit = _finalize_restored_partition(
        scene_id=scene_id,
        restored=post_filter,
        candidate_by_id={int(row["raw_instance_id"]): row for row in rows},
        evidence=evidence,
        xyz=gaussian_xyz,
        is_big_gaussian=is_big_gaussian,
        selected=set(),
    )
    if not _same_prediction_geometry_and_class(no_restore_native, output):
        raise AssertionError(
            f"{scene_id}: post_filter empty replay differs from frozen T1-B1"
        )
    no_restore_native_scores_exact = all(
        math.isclose(
            float(no_restore_native["instances"][key]["score"]),
            float(output["instances"][key]["score"]),
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        for key in output["instances"]
    )
    if not no_restore_native_scores_exact:
        raise AssertionError(
            f"{scene_id}: post_filter empty replay changed native final-vote scores"
        )
    reconstructed = normalize_prediction(final_internal, raw_instances)
    if not np.array_equal(reconstructed.point_labels, arrays["exported_prediction"]):
        raise AssertionError(f"{scene_id}: no-restoration partition parity failed")
    if reconstructed.instances != output["instances"]:
        raise AssertionError(
            f"{scene_id}: no-restoration class/bbox/score metadata parity failed"
        )
    _atomic_npz(
        destination["arrays"],
        merged_partition=arrays["merged_partition"].astype(np.int64, copy=False),
        post_global_knn=post_knn,
        post_filter=post_filter,
        final_internal_labels=final_internal,
        exported_prediction=arrays["exported_prediction"].astype(np.int64, copy=False),
    )
    metadata = {
        "schema": SCHEMA,
        "kind": "full_instance_snapshot",
        "scene_id": scene_id,
        "physical_scene_id": physical_scene_id(scene_id),
        "seed": V9_FEATURE_SEED,
        "consumer_git_commit": config.git_commit,
        "point_count": int(len(gaussian_xyz)),
        "candidate_count": len(rows),
        "global_candidate_count": sum(row["source"] == "global" for row in rows),
        "other_classes_candidate_count": sum(
            row["source"] == "other_classes" for row in rows
        ),
        "t1_identity": t1_identity,
        "vote_input_identity": vote_identity,
        "snapshot_identity": snapshot_identity,
        "snapshot_arrays_sha256": sha256_file(destination["arrays"]),
        "vote_evidence_sha256": sha256_file(destination["votes"]),
        "vote_evidence": str(destination["votes"]),
        "base_rows": rows,
        "base_rows_sha256": hash_json(rows),
        "rows": rows,
        "rows_sha256": hash_json(rows),
        "baseline_output": str(t1_paths.output),
        "baseline_trace": str(t1_paths.stage_trace),
        "no_restoration_exact": True,
        "empty_restoration_replayed_from_post_filter": True,
        "empty_restoration_native_scores_exact": no_restore_native_scores_exact,
        "empty_restoration_replay_audit": no_restore_audit,
        "candidate_construction": dict(candidate_snapshot.diagnostics),
    }
    write_json(destination["metadata"], metadata)
    return metadata


def _load_snapshot_arrays(config: ExperimentConfig, scene_id: str) -> dict[str, np.ndarray]:
    path = _snapshot_paths(config, scene_id)["arrays"]
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _candidate_rows(config: ExperimentConfig, scene_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        payload = load_json(_snapshot_paths(config, scene_id)["metadata"])
        rows.extend(dict(row) for row in payload["rows"])
    return rows


def _rehydrate_candidate_members(
    candidates: Sequence[Mapping[str, Any]], merged_partition: np.ndarray
) -> list[dict[str, Any]]:
    """Attach immutable candidate members from the sole frozen partition.

    Member vectors are intentionally not duplicated in JSON.  Reconstructing
    them from ``merged_partition`` is exact and keeps replay independent of GT.
    """

    labels = np.asarray(merged_partition, dtype=np.int64)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    foreground = np.flatnonzero(labels >= 0)
    if len(foreground):
        order = np.argsort(labels[foreground], kind="stable")
        grouped_points = foreground[order]
        grouped_labels = labels[grouped_points]
        unique_ids, starts, counts = np.unique(
            grouped_labels, return_index=True, return_counts=True
        )
        members_by_id = {
            int(raw_id): grouped_points[start : start + count].astype(
                np.int64, copy=False
            )
            for raw_id, start, count in zip(unique_ids, starts, counts)
        }
    else:
        members_by_id = {}
    for candidate in candidates:
        raw_id = int(candidate["raw_instance_id"])
        if raw_id in seen:
            raise ValueError(f"duplicate raw candidate ID {raw_id}")
        seen.add(raw_id)
        members = members_by_id.get(raw_id, np.empty(0, dtype=np.int64))
        if len(members) != int(candidate["point_count"]):
            raise ValueError(
                f"candidate {raw_id} member count differs from frozen partition"
            )
        row = dict(candidate)
        row["member_indices"] = members
        result.append(row)
    frozen_ids = {int(value) for value in np.unique(labels) if int(value) >= 0}
    if seen != frozen_ids:
        raise ValueError("candidate JSON does not cover the frozen pre-KNN partition")
    return result


_ARM_FIELDS = frozenset(
    {
        "score_mode",
        "size_lookup_class",
        "size_fallback_global",
        "prior_applied",
        "G",
        "S",
        "S_q_only",
        "G_global",
        "S_global",
        "G_class",
        "S_class",
        "class_prior_fallback",
    }
)


def _score_snapshot_rows(
    snapshot: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    priors: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Idempotently score one immutable bank and prove exact member/Q identity."""

    from .full_instance_size_prior import score_same_bank_size_priors

    source_rows = snapshot.get("base_rows")
    if not isinstance(source_rows, list):
        raise TypeError("snapshot lacks immutable base_rows")
    if snapshot.get("base_rows_sha256") != hash_json(source_rows):
        raise ValueError("snapshot base_rows content hash mismatch")
    clean = [
        {key: value for key, value in dict(row).items() if key not in _ARM_FIELDS}
        for row in source_rows
    ]
    hydrated = _rehydrate_candidate_members(clean, arrays["merged_partition"])
    scores = score_same_bank_size_priors(hydrated, priors)
    rows: list[dict[str, Any]] = []
    for q_row, global_row, class_row in zip(
        scores.q_only, scores.global_size, scores.class_size, strict=True
    ):
        shared = {
            key: value for key, value in q_row.items() if key not in _ARM_FIELDS
        }
        shared.update(
            {
                "S_q_only": float(q_row["S"]),
                "G_global": float(global_row["G"]),
                "S_global": float(global_row["S"]),
                "G_class": float(class_row["G"]),
                "S_class": float(class_row["S"]),
                "class_prior_fallback": bool(
                    class_row.get("size_fallback_global", False)
                ),
            }
        )
        rows.append(_plain_candidate_row(shared))
    return rows, asdict(scores.identity)


def _load_scene_gt(
    scene_id: str,
    scene: Mapping[str, Any],
    gt_dir: Path,
    *,
    locked_row: Mapping[str, Any] | None = None,
    locked_spec_dir: Path | None = None,
) -> tuple[np.ndarray, GroundTruthScene, np.ndarray]:
    gt_xyz, ground_truth = load_ground_truth_npz(
        _gt_path(
            scene,
            gt_dir,
            scene_id,
            locked_row=locked_row,
            locked_spec_dir=locked_spec_dir,
        ),
        scene_id,
    )
    gaussian_xyz = apply_transform(load_ply_xyz(_point_cloud(scene)), _transform(scene))
    return gt_xyz, ground_truth, gaussian_xyz


def _teacher_bbox_corners(points: np.ndarray) -> list[float]:
    """Reproduce the corrected teacher flow's XZ-oriented 3-D bbox exactly."""

    # ``postprocess.py`` receives the PLY positions as a float32 torch tensor.
    # Preserve that input dtype before trimesh computes its 2-D orientation;
    # silently promoting here can move the serialized corners in the last bits
    # and would break the registered empty-restoration parity check.
    values = np.asarray(points, dtype=np.float32)
    if len(values) == 0:
        return [0.0] * 24
    if len(values) < 3:
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        return np.asarray(
            [
                [upper[0], upper[1], upper[2]],
                [upper[0], upper[1], lower[2]],
                [upper[0], lower[1], lower[2]],
                [upper[0], lower[1], upper[2]],
                [lower[0], upper[1], upper[2]],
                [lower[0], upper[1], lower[2]],
                [lower[0], lower[1], lower[2]],
                [lower[0], lower[1], upper[2]],
            ],
            dtype=np.float64,
        ).reshape(-1).tolist()
    from trimesh.bounds import oriented_bounds_2D

    points_2d = values[:, [0, 2]]
    transform_2d, rectangle_extents_2d = oriented_bounds_2D(points_2d)
    transform_3d = np.eye(4, dtype=np.float64)
    transform_3d[0, 0] = transform_2d[0, 0]
    transform_3d[0, 2] = transform_2d[0, 1]
    transform_3d[0, 3] = transform_2d[0, 2]
    transform_3d[2, 0] = transform_2d[1, 0]
    transform_3d[2, 2] = transform_2d[1, 1]
    transform_3d[2, 3] = transform_2d[1, 2]
    homogeneous = np.hstack((values, np.ones((len(values), 1))))
    transformed = (transform_3d @ homogeneous.T).T[:, :3]
    lower_y = float(transformed[:, 1].min())
    upper_y = float(transformed[:, 1].max())
    half_x, half_z = np.asarray(rectangle_extents_2d, dtype=np.float64) / 2.0
    local = np.asarray(
        [
            [half_x, upper_y, half_z],
            [half_x, upper_y, -half_z],
            [half_x, lower_y, -half_z],
            [half_x, lower_y, half_z],
            [-half_x, upper_y, half_z],
            [-half_x, upper_y, -half_z],
            [-half_x, lower_y, -half_z],
            [-half_x, lower_y, half_z],
        ],
        dtype=np.float64,
    )
    local_homogeneous = np.hstack((local, np.ones((8, 1))))
    world = (np.linalg.inv(transform_3d) @ local_homogeneous.T).T[:, :3]
    return world.reshape(-1).tolist()


def _final_vote_class(
    counts: np.ndarray, class_names: Sequence[str]
) -> tuple[str, float]:
    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (len(class_names) + 1,):
        raise ValueError("final vote vector has the wrong length")
    denominator = float(values.sum())
    ratios = values[:-1] / denominator if denominator > 0 else np.zeros(len(class_names))
    maximum = float(ratios.max()) if len(ratios) else 0.0
    if maximum < 0.30:
        return "background", maximum
    return str(class_names[int(np.argmax(ratios))]), maximum


def _condition_output_paths(
    config: ExperimentConfig, stage: str, condition: str, scene_id: str
) -> dict[str, Path]:
    root = config.runs_root / stage / condition / scene_id / f"seed-{V9_FEATURE_SEED}"
    return {
        "root": root,
        "output": root / "output.json",
        "native_output": root / "output.native-score.json",
        "diagnostics": root / "diagnostics.json",
    }


def _condition_input_identity(
    config: ExperimentConfig,
    scene_id: str,
    scene: Mapping[str, Any],
    condition: str,
    threshold: float,
) -> dict[str, Any]:
    snapshot_paths = _snapshot_paths(config, scene_id)
    snapshot = load_json(snapshot_paths["metadata"])
    return {
        "schema": SCHEMA,
        "consumer_git_commit": config.git_commit,
        "scene_id": scene_id,
        "condition": condition,
        "threshold": float(threshold),
        "runtime_scene_hash": hash_json(dict(scene)),
        "snapshot_identity": snapshot.get("snapshot_identity"),
        "snapshot_arrays_sha256": sha256_file(snapshot_paths["arrays"]),
        "vote_evidence_sha256": sha256_file(snapshot_paths["votes"]),
        "category_priors_sha256": sha256_file(config.category_priors),
        "score_arm_identity": snapshot.get("score_arm_identity"),
        "snapshot_rows_sha256": snapshot.get("rows_sha256"),
    }


def _same_prediction_geometry_and_class(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    """Main/native views may differ only in score, never mask/class/bbox."""

    if left.get("point_labels") != right.get("point_labels"):
        return False
    left_instances = left.get("instances")
    right_instances = right.get("instances")
    if not isinstance(left_instances, Mapping) or not isinstance(
        right_instances, Mapping
    ):
        return False
    if set(left_instances) != set(right_instances):
        return False
    return all(
        left_instances[key].get("class") == right_instances[key].get("class")
        and left_instances[key].get("bbox") == right_instances[key].get("bbox")
        for key in left_instances
    )


def _finalize_restored_partition(
    *,
    scene_id: str,
    restored: np.ndarray,
    candidate_by_id: Mapping[int, Mapping[str, Any]],
    evidence: Any,
    xyz: np.ndarray,
    is_big_gaussian: np.ndarray,
    selected: set[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Run the real final vote and output contract for one restored partition."""

    from .full_instance_vote import aggregate_instance_votes

    final_votes = aggregate_instance_votes(restored, evidence)
    raw_instances: dict[int, dict[str, Any]] = {}
    native_instances: dict[int, dict[str, Any]] = {}
    class_changes = 0
    selected_not_exported = 0
    for raw_id in sorted(int(value) for value in np.unique(restored) if int(value) >= 0):
        counts = final_votes.get(raw_id, np.zeros(len(CLASSES_32) + 1, dtype=np.int64))
        class_name, native_score = _final_vote_class(counts, CLASSES_32)
        candidate = candidate_by_id.get(raw_id)
        if candidate is None:
            raise AssertionError(f"{scene_id}: post-filter raw ID lacks pre-KNN candidate")
        if class_name not in SELECTED_CLASSES_28:
            if raw_id in selected:
                selected_not_exported += 1
            continue
        early_class = candidate.get("predicted_class")
        class_changes += int(early_class is not None and class_name != early_class)
        instance_mask = (restored == raw_id) & ~is_big_gaussian
        bbox = _teacher_bbox_corners(xyz[instance_mask])
        raw_instances[raw_id] = {
            "bbox": bbox,
            "class": class_name,
            "score": float(candidate["Q"]),
        }
        native_instances[raw_id] = {
            "bbox": bbox,
            "class": class_name,
            "score": float(native_score),
        }
    contracted = normalize_prediction(restored, raw_instances)
    native = normalize_prediction(restored, native_instances)
    if not np.array_equal(contracted.point_labels, native.point_labels):
        raise AssertionError("main/native score outputs changed the partition")
    output = {
        "point_labels": contracted.point_labels.tolist(),
        "instances": contracted.instances,
        "prediction_contract": contracted.audit,
    }
    native_output = {
        "point_labels": native.point_labels.tolist(),
        "instances": native.instances,
        "prediction_contract": native.audit,
    }
    return output, native_output, {
        "selected_not_exported": selected_not_exported,
        "early_to_final_class_change_count": class_changes,
        "early_to_final_class_compared_count": len(raw_instances),
    }


def _materialize_condition_output(
    config: ExperimentConfig,
    stage: str,
    scene_id: str,
    scene: Mapping[str, Any],
    condition: str,
    threshold: float,
) -> dict[str, Any]:
    from .full_instance_size_prior import restore_selected_instances
    from .full_instance_vote import load_gaussian_vote_evidence

    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    paths = _condition_output_paths(config, stage, condition, scene_id)
    input_identity = _condition_input_identity(
        config, scene_id, scene, condition, threshold
    )
    if all(
        paths[key].is_file()
        for key in ("output", "native_output", "diagnostics")
    ):
        try:
            output = load_json(paths["output"])
            native_output = load_json(paths["native_output"])
            diagnostics = load_json(paths["diagnostics"])
            validate_prediction_contract(output["point_labels"], output["instances"])
            validate_prediction_contract(
                native_output["point_labels"], native_output["instances"]
            )
            if all(
                (
                    diagnostics.get("schema") == SCHEMA,
                    diagnostics.get("consumer_git_commit") == config.git_commit,
                    diagnostics.get("scene_id") == scene_id,
                    diagnostics.get("stage") == stage,
                    diagnostics.get("condition") == condition,
                    diagnostics.get("input_identity") == input_identity,
                    math.isclose(
                        float(diagnostics.get("threshold", math.nan)),
                        float(threshold),
                        abs_tol=0.0,
                        rel_tol=0.0,
                    ),
                    _same_prediction_geometry_and_class(output, native_output),
                    diagnostics.get("output_sha256")
                    == sha256_file(paths["output"]),
                    diagnostics.get("native_output_sha256")
                    == sha256_file(paths["native_output"]),
                )
            ):
                return diagnostics
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    snapshot = load_json(_snapshot_paths(config, scene_id)["metadata"])
    arrays = _load_snapshot_arrays(config, scene_id)
    candidates = _rehydrate_candidate_members(
        [dict(row) for row in snapshot["rows"]], arrays["merged_partition"]
    )
    candidate_by_id = {int(row["raw_instance_id"]): row for row in candidates}
    evidence = load_gaussian_vote_evidence(_snapshot_paths(config, scene_id)["votes"])
    xyz = load_ply_xyz(_point_cloud(scene))
    baseline_reference = load_json(Path(str(snapshot["baseline_output"])))
    is_big_gaussian = np.asarray(
        baseline_reference.get("is_big_gaussian"), dtype=bool
    )
    if is_big_gaussian.shape != (len(xyz),):
        raise ValueError(
            f"{scene_id}: frozen T1-B1 lacks a point-aligned is_big_gaussian mask"
        )
    if condition == "controlled-baseline":
        # Prove the *whole* registered no-restoration path, starting at
        # post_filter and including the original final vote/output contract,
        # reproduces T1-B1.  Copying the frozen JSON alone would not test the
        # recovery interface's zero-selection control.
        baseline = baseline_reference
        validate_prediction_contract(baseline["point_labels"], baseline["instances"])
        output, replayed_native, replay_audit = _finalize_restored_partition(
            scene_id=scene_id,
            restored=arrays["post_filter"].astype(np.int64, copy=False),
            candidate_by_id=candidate_by_id,
            evidence=evidence,
            xyz=xyz,
            is_big_gaussian=is_big_gaussian,
            selected=set(),
        )
        if not _same_prediction_geometry_and_class(replayed_native, baseline):
            raise AssertionError(
                f"{scene_id}: empty restoration does not reproduce frozen T1-B1 geometry/class"
            )
        native_scores_exact = all(
            math.isclose(
                float(replayed_native["instances"][key]["score"]),
                float(baseline["instances"][key]["score"]),
                abs_tol=1e-12,
                rel_tol=0.0,
            )
            for key in baseline["instances"]
        )
        if not native_scores_exact:
            raise AssertionError(
                f"{scene_id}: empty restoration changed frozen native final-vote scores"
            )
        native_output = replayed_native
        paths["root"].mkdir(parents=True, exist_ok=True)
        write_json(paths["output"], output)
        write_json(paths["native_output"], native_output)
        diagnostics = {
            "schema": SCHEMA,
            "kind": "full_instance_size_condition",
            "consumer_git_commit": config.git_commit,
            "scene_id": scene_id,
            "stage": stage,
            "condition": condition,
            "threshold": float(threshold),
            "input_identity": input_identity,
            "selected_raw_instance_ids": [],
            "selected_count": 0,
            "selected_not_exported": replay_audit["selected_not_exported"],
            "early_to_final_class_change_count": replay_audit[
                "early_to_final_class_change_count"
            ],
            "early_to_final_class_compared_count": replay_audit[
                "early_to_final_class_compared_count"
            ],
            "early_to_final_class_change_rate": (
                replay_audit["early_to_final_class_change_count"]
                / replay_audit["early_to_final_class_compared_count"]
                if replay_audit["early_to_final_class_compared_count"]
                else 0.0
            ),
            "restore": {
                "selected_candidate_count": 0,
                "changed_point_count": 0,
                "outside_selected_changed_count": 0,
            },
            "output_instance_count": len(output["instances"]),
            "changed_export_points_vs_frozen_baseline": 0,
            "score_contract": "pre-knn-Q",
            "native_score_output": str(paths["native_output"]),
            "frozen_mask_class_bbox_instance_count_exact": True,
            "empty_restoration_replayed_from_post_filter": True,
            "empty_restoration_native_scores_exact": native_scores_exact,
            "output_sha256": sha256_file(paths["output"]),
            "native_output_sha256": sha256_file(paths["native_output"]),
        }
        write_json(paths["diagnostics"], diagnostics)
        return diagnostics
    score_field = "S_global" if condition == "global-size" else "S_class"
    selected = [
        int(row["raw_instance_id"])
        for row in candidates
        if bool(row["eligible"]) and float(row[score_field]) >= float(threshold)
    ]
    restoration = restore_selected_instances(
        arrays["post_filter"], candidates, selected
    )
    restored = restoration.point_labels
    restore_audit = dict(restoration.diagnostics)
    output, native_output, final_audit = _finalize_restored_partition(
        scene_id=scene_id,
        restored=restored,
        candidate_by_id=candidate_by_id,
        evidence=evidence,
        xyz=xyz,
        is_big_gaussian=is_big_gaussian,
        selected=set(selected),
    )
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_json(paths["output"], output)
    write_json(paths["native_output"], native_output)
    baseline_labels = arrays["exported_prediction"].astype(np.int64, copy=False)
    diagnostics = {
        "schema": SCHEMA,
        "kind": "full_instance_size_condition",
        "consumer_git_commit": config.git_commit,
        "scene_id": scene_id,
        "stage": stage,
        "condition": condition,
        "threshold": float(threshold),
        "input_identity": input_identity,
        "selected_raw_instance_ids": sorted(selected),
        "selected_count": len(selected),
        "selected_not_exported": final_audit["selected_not_exported"],
        "early_to_final_class_change_count": final_audit[
            "early_to_final_class_change_count"
        ],
        "early_to_final_class_compared_count": final_audit[
            "early_to_final_class_compared_count"
        ],
        "early_to_final_class_change_rate": (
            final_audit["early_to_final_class_change_count"]
            / final_audit["early_to_final_class_compared_count"]
            if final_audit["early_to_final_class_compared_count"]
            else 0.0
        ),
        "restore": restore_audit,
        "output_instance_count": len(output["instances"]),
        "changed_export_points_vs_frozen_baseline": int(
            np.count_nonzero(
                np.asarray(output["point_labels"], dtype=np.int64) != baseline_labels
            )
        ),
        "score_contract": "pre-knn-Q",
        "native_score_output": str(paths["native_output"]),
        "output_sha256": sha256_file(paths["output"]),
        "native_output_sha256": sha256_file(paths["native_output"]),
    }
    write_json(paths["diagnostics"], diagnostics)
    return diagnostics


def _predictions_from_output(
    scene_id: str,
    output: Mapping[str, Any],
    gt_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    taxonomy: Taxonomy,
) -> tuple[list[PredictedInstance], dict[str, Any]]:
    labels = np.asarray(output["point_labels"], dtype=np.int64)
    mapped, diagnostics = map_gaussians_to_gt(gt_xyz, gaussian_xyz, labels, RADIUS_M)
    declared_count = int(np.count_nonzero(mapped >= 0))
    diagnostics["gt_point_count"] = int(len(mapped))
    diagnostics["gt_nearest_declared_count"] = declared_count
    diagnostics["gt_nearest_declared_fraction"] = (
        float(declared_count / len(mapped)) if len(mapped) else 0.0
    )
    class_to_id = {name: index for index, name in enumerate(taxonomy.canonical_classes)}
    predictions: list[PredictedInstance] = []
    for key, metadata in output["instances"].items():
        class_name = str(metadata["class"])
        if class_name not in class_to_id:
            continue
        instance_id = int(key)
        predictions.append(
            PredictedInstance(
                scene_id=scene_id,
                instance_id=instance_id,
                class_id=class_to_id[class_name],
                score=float(metadata["score"]),
                mask=mapped == instance_id,
            )
        )
    return predictions, diagnostics


def _size_bin(diagonal_m: float, size_spec: Mapping[str, Any] | None) -> str | None:
    if size_spec is None:
        return None
    limits = size_spec.get("boundaries_m", size_spec)
    tiny = float(limits["tiny_max_m"])
    small = float(limits["small_max_m"])
    medium = float(limits["medium_max_m"])
    if diagonal_m <= tiny:
        return "tiny"
    if diagonal_m <= small:
        return "small"
    if diagonal_m <= medium:
        return "medium"
    return "large"


def _endpoint_scene_metrics(
    config: ExperimentConfig,
    stage: str,
    scene_id: str,
    scene: Mapping[str, Any],
    condition: str,
    taxonomy: Taxonomy,
    size_spec: Mapping[str, Any],
    *,
    gt_dir: Path | None = None,
    locked_row: Mapping[str, Any] | None = None,
    locked_spec_dir: Path | None = None,
) -> tuple[
    dict[str, Any],
    GroundTruthScene,
    list[PredictedInstance],
    list[PredictedInstance],
]:
    from .clean_baseline.metric_reaudit import evaluate_dual_protocols
    from .full_instance_size_evaluation import matched_recall_summary

    paths = _condition_output_paths(config, stage, condition, scene_id)
    output = load_json(paths["output"])
    gt_xyz, ground_truth, gaussian_xyz = _load_scene_gt(
        scene_id,
        scene,
        gt_dir or config.gt_dir,
        locked_row=locked_row,
        locked_spec_dir=locked_spec_dir,
    )
    predictions, projection = _predictions_from_output(
        scene_id, output, gt_xyz, gaussian_xyz, taxonomy
    )
    native_output = load_json(paths["native_output"])
    validate_prediction_contract(
        native_output["point_labels"], native_output["instances"]
    )
    if native_output["point_labels"] != output["point_labels"]:
        raise AssertionError("main-Q and native-score outputs differ in point labels")
    native_predictions, _ = _predictions_from_output(
        scene_id, native_output, gt_xyz, gaussian_xyz, taxonomy
    )
    protocols = evaluate_dual_protocols(
        (ground_truth,), predictions, taxonomy.canonical_classes, min_region_size=MIN_REGION_SIZE
    )
    native_protocols = evaluate_dual_protocols(
        (ground_truth,),
        native_predictions,
        taxonomy.canonical_classes,
        min_region_size=MIN_REGION_SIZE,
    )
    precision = evaluate_gaussian_object_precision(
        gaussian_xyz,
        np.asarray(output["point_labels"], dtype=np.int64),
        output["instances"],
        gt_xyz,
        ground_truth.semantic,
        ground_truth.instance,
        RADIUS_M,
        canonical_classes=taxonomy.canonical_classes,
    )
    size_by_gt: dict[tuple[int, int], str | None] = {}
    for class_id in range(len(taxonomy.canonical_classes)):
        class_mask = ground_truth.semantic == class_id
        for instance_id in np.unique(ground_truth.instance[class_mask]):
            if int(instance_id) < 0:
                continue
            mask = class_mask & (ground_truth.instance == int(instance_id))
            if int(np.count_nonzero(mask)) < MIN_REGION_SIZE:
                continue
            extent = np.ptp(gt_xyz[mask], axis=0)
            size_by_gt[(class_id, int(instance_id))] = _size_bin(
                float(np.linalg.norm(extent)), size_spec
            )
    recall = matched_recall_summary(
        ground_truth,
        predictions,
        size_by_gt=size_by_gt,
        thresholds=(0.25, 0.50),
        min_region_size=MIN_REGION_SIZE,
    )
    official = protocols["official_9"]["aggregate"]
    historical = protocols["historical_10"]["aggregate"]
    native_official = native_protocols["official_9"]["aggregate"]
    native_historical = native_protocols["historical_10"]["aggregate"]
    row = {
        "stage": stage,
        "scene_id": scene_id,
        "physical_scene_id": physical_scene_id(scene_id),
        "condition": condition,
        "official_map_50_90": float(official["map_50_90"]),
        "ap50": float(official["map_0.50"]),
        "ap25": float(official["map_0.25"]),
        "historical_map_50_95": float(historical["map_50_95"]),
        "native_official_map_50_90": float(native_official["map_50_90"]),
        "native_ap50": float(native_official["map_0.50"]),
        "native_ap25": float(native_official["map_0.25"]),
        "native_historical_map_50_95": float(
            native_historical["map_50_95"]
        ),
        "instance_count": len(output["instances"]),
        "coverage": float(projection["gt_nearest_declared_fraction"]),
        "gt_point_count": int(projection["gt_point_count"]),
        "gt_nearest_declared_count": int(
            projection["gt_nearest_declared_count"]
        ),
        "gaussian_micro_precision": float(
            precision["aggregate"]["micro_point_precision"]
        ),
        "correct_gaussian_count": int(
            precision["aggregate"]["correct_gaussian_count"]
        ),
        "predicted_gaussian_count": int(
            precision["aggregate"]["predicted_gaussian_count"]
        ),
        "fp_tp_ratio_025": recall["fp_tp_ratio_025"],
        "true_positive_count_025": int(
            recall["thresholds"]["025"]["true_positive_count"]
        ),
        "false_positive_count_025": int(
            recall["thresholds"]["025"]["false_positive_count"]
        ),
        "tiny_small_gt_count": recall["tiny_small_gt_count"],
        "tiny_small_match_count_025": int(
            recall["thresholds"]["025"]["tiny_small_match_count"]
        ),
        "tiny_small_match_count_050": int(
            recall["thresholds"]["050"]["tiny_small_match_count"]
        ),
        "tiny_small_recall_025": recall["tiny_small_recall_025"],
        "tiny_small_recall_050": recall["tiny_small_recall_050"],
    }
    return row, ground_truth, predictions, native_predictions


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _condition_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conditions = sorted({str(row["condition"]) for row in rows})
    result: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        has_structural_counts = bool(
            selected
            and all(
                row.get("correct_gaussian_count") is not None
                and row.get("predicted_gaussian_count") is not None
                and row.get("true_positive_count_025") is not None
                and row.get("false_positive_count_025") is not None
                for row in selected
            )
        )
        summary = {
            "scene_count": len(selected),
            "aggregation": "scene-equal-ap",
            "ap_aggregation": "scene-equal",
            "structural_aggregation": (
                "pooled-counts" if has_structural_counts else "scene-equal-fallback"
            ),
            **{
                field: _mean(selected, field)
                for field in (
                    "official_map_50_90",
                    "ap50",
                    "ap25",
                    "historical_map_50_95",
                    "native_official_map_50_90",
                    "native_ap50",
                    "native_ap25",
                    "native_historical_map_50_95",
                    "instance_count",
                    "coverage",
                    "gaussian_micro_precision",
                    "fp_tp_ratio_025",
                    "tiny_small_recall_025",
                    "tiny_small_recall_050",
                )
            },
        }
        if selected and all(
            row.get("correct_gaussian_count") is not None
            and row.get("predicted_gaussian_count") is not None
            for row in selected
        ):
            correct = sum(int(row["correct_gaussian_count"]) for row in selected)
            predicted = sum(int(row["predicted_gaussian_count"]) for row in selected)
            summary.update(
                {
                    "correct_gaussian_count": correct,
                    "predicted_gaussian_count": predicted,
                    "gaussian_micro_precision": correct / predicted
                    if predicted
                    else 0.0,
                }
            )
        if selected and all(
            row.get("true_positive_count_025") is not None
            and row.get("false_positive_count_025") is not None
            for row in selected
        ):
            true_positive = sum(
                int(row["true_positive_count_025"]) for row in selected
            )
            false_positive = sum(
                int(row["false_positive_count_025"]) for row in selected
            )
            summary.update(
                {
                    "true_positive_count_025": true_positive,
                    "false_positive_count_025": false_positive,
                    "fp_tp_ratio_025": false_positive / max(true_positive, 1),
                }
            )
        if selected and all(
            row.get("gt_point_count") is not None
            and row.get("gt_nearest_declared_count") is not None
            for row in selected
        ):
            gt_count = sum(int(row["gt_point_count"]) for row in selected)
            mapped_count = sum(
                int(row["gt_nearest_declared_count"]) for row in selected
            )
            summary.update(
                {
                    "gt_point_count": gt_count,
                    "gt_nearest_declared_count": mapped_count,
                    "coverage": mapped_count / gt_count if gt_count else 0.0,
                }
            )
        if selected and all(
            row.get("tiny_small_gt_count") is not None
            and row.get("tiny_small_match_count_025") is not None
            and row.get("tiny_small_match_count_050") is not None
            for row in selected
        ):
            tiny_gt = sum(int(row["tiny_small_gt_count"]) for row in selected)
            tiny_025 = sum(
                int(row["tiny_small_match_count_025"]) for row in selected
            )
            tiny_050 = sum(
                int(row["tiny_small_match_count_050"]) for row in selected
            )
            summary.update(
                {
                    "tiny_small_gt_count": tiny_gt,
                    "tiny_small_match_count_025": tiny_025,
                    "tiny_small_match_count_050": tiny_050,
                    "tiny_small_recall_025": tiny_025 / tiny_gt
                    if tiny_gt
                    else None,
                    "tiny_small_recall_050": tiny_050 / tiny_gt
                    if tiny_gt
                    else None,
                }
            )
        result[condition] = summary
    return result


def _pooled_official_condition_summary(
    *,
    ground_truth_by_scene: Mapping[str, GroundTruthScene],
    predictions_by_condition: Mapping[str, Sequence[PredictedInstance]],
    native_predictions_by_condition: Mapping[str, Sequence[PredictedInstance]],
    taxonomy: Taxonomy,
    structural_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Evaluate official AP once over the complete stage, never by averaging AP.

    Per-scene AP remains available for paired directions and scene-level
    uncertainty.  The fields named ``official_*`` in this returned mapping are
    the pooled ScanNet evaluator result across all scans in the stage.
    """

    from .clean_baseline.metric_reaudit import evaluate_dual_protocols

    gt = tuple(ground_truth_by_scene[key] for key in sorted(ground_truth_by_scene))
    result: dict[str, dict[str, Any]] = {}
    for condition in sorted(predictions_by_condition):
        main = evaluate_dual_protocols(
            gt,
            list(predictions_by_condition[condition]),
            taxonomy.canonical_classes,
            min_region_size=MIN_REGION_SIZE,
        )
        native = evaluate_dual_protocols(
            gt,
            list(native_predictions_by_condition[condition]),
            taxonomy.canonical_classes,
            min_region_size=MIN_REGION_SIZE,
        )
        official = main["official_9"]["aggregate"]
        historical = main["historical_10"]["aggregate"]
        native_official = native["official_9"]["aggregate"]
        native_historical = native["historical_10"]["aggregate"]
        structural = dict(structural_summaries[condition])
        for field in (
            "official_map_50_90",
            "ap50",
            "ap25",
            "historical_map_50_95",
            "native_official_map_50_90",
            "native_ap50",
            "native_ap25",
            "native_historical_map_50_95",
        ):
            structural.pop(field, None)
        result[condition] = {
            **structural,
            "aggregation": "pooled-official-evaluator",
            "official_map_50_90": float(official["map_50_90"]),
            "ap50": float(official["map_0.50"]),
            "ap25": float(official["map_0.25"]),
            "historical_map_50_95": float(historical["map_50_95"]),
            "native_official_map_50_90": float(
                native_official["map_50_90"]
            ),
            "native_ap50": float(native_official["map_0.50"]),
            "native_ap25": float(native_official["map_0.25"]),
            "native_historical_map_50_95": float(
                native_historical["map_50_95"]
            ),
        }
    return result


def _pair_delta_rows(
    rows: Sequence[Mapping[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    by_key = {(str(row["scene_id"]), str(row["condition"])): row for row in rows}
    result: list[dict[str, Any]] = []
    for scene_id in sorted({str(row["scene_id"]) for row in rows}):
        a = by_key[(scene_id, left)]
        b = by_key[(scene_id, right)]
        result.append(
            {
                "scene_id": scene_id,
                "physical_scene_id": physical_scene_id(scene_id),
                **{
                    f"delta_{field}": float(b[field]) - float(a[field])
                    for field in (
                        "official_map_50_90",
                        "ap50",
                        "ap25",
                        "native_official_map_50_90",
                        "native_ap50",
                        "native_ap25",
                        "coverage",
                        "gaussian_micro_precision",
                        "tiny_small_recall_025",
                        "tiny_small_recall_050",
                    )
                    if a.get(field) is not None and b.get(field) is not None
                },
            }
        )
    return result


def _parse_scene_spec(path: Path) -> tuple[str, ...]:
    payload = load_json(path)
    raw: Any
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, Mapping):
        raw = payload.get("scene_ids", payload.get("scenes"))
    else:
        raise TypeError(f"{path}: scene spec must be a list or object")
    if not isinstance(raw, list):
        raise TypeError(f"{path}: scene spec lacks a scene list")
    result: list[str] = []
    for item in raw:
        scene_id = str(item["scene_id"] if isinstance(item, Mapping) else item)
        result.append(scene_id)
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate scene IDs")
    return tuple(result)


def _locked_scene_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, Mapping) or payload.get("kind") != "locked_evaluation_scenes":
        raise ValueError(f"{path}: expected kind=locked_evaluation_scenes")
    raw = payload.get("scenes")
    if not isinstance(raw, list):
        raise TypeError(f"{path}: locked scene rows must be a list")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        row = item if isinstance(item, Mapping) else {"scene_id": str(item)}
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id or scene_id in result:
            raise ValueError(f"{path}: locked scene IDs must be non-empty and unique")
        registered_physical = row.get("physical_scene_id")
        if registered_physical not in (None, "") and str(
            registered_physical
        ) != physical_scene_id(scene_id):
            raise ValueError(f"{scene_id}: inconsistent locked physical_scene_id")
        result[scene_id] = row
    return result


def _load_train_scene_ids(path: Path) -> tuple[str, ...]:
    """Read the exact instance-statistics table used to fit the priors."""

    if path.suffix.lower() == ".json":
        payload = load_json(path)
        raw: Any = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError("train statistics JSON must contain a row sequence")
        rows = list(raw)
    else:
        rows = read_rows(path)
    scene_ids = tuple(
        sorted(
            {
                str(row.get("scene_id", row.get("scan_id", ""))).strip()
                for row in rows
                if isinstance(row, Mapping)
            }
        )
    )
    if not scene_ids or any(not value for value in scene_ids):
        raise ValueError("train statistics must identify every row's scene/scan")
    split_values = {
        str(row.get("split", "")).strip().lower()
        for row in rows
        if isinstance(row, Mapping)
    }
    if split_values != {"train"}:
        raise ValueError("train statistics must contain exactly split=train")
    return scene_ids


def _validate_registered_inputs(
    config: ExperimentConfig,
    *,
    tune_scenes: Mapping[str, Mapping[str, Any]],
    final_scenes: Mapping[str, Mapping[str, Any]],
    tune24: Sequence[str],
    final48: Sequence[str],
    priors: Mapping[str, Any],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Prove split, prior, metric-unit, and replacement-GT identity."""

    from .priors import validate_priors

    validate_priors(priors)
    provenance = priors.get("provenance", {})
    normalization = priors.get("normalization", {})
    if provenance.get("splits") != ["train"]:
        raise ValueError("category priors must use exactly the train split")
    if normalization.get("units") != "meters":
        raise ValueError("category priors must use metric units")
    if provenance.get("taxonomy_sha256") != taxonomy.content_hash:
        raise ValueError("category priors were built with a different taxonomy")
    default_taxonomy = load_taxonomy(None)
    if taxonomy.canonical_classes != default_taxonomy.canonical_classes:
        raise ValueError(
            "formal SAGA20 evaluation requires the registered canonical class order"
        )
    categories = priors.get("categories")
    if not isinstance(categories, Mapping) or set(categories) != set(
        taxonomy.canonical_classes
    ):
        raise ValueError("category priors must contain exactly the SAGA20 classes")

    tune = tuple(map(str, tune24))
    final = tuple(map(str, final48))
    if set(tune) != set(tune_scenes) or len(tune) != len(tune_scenes):
        raise ValueError("tune scene spec and tune runtime manifest disagree")
    if set(final) != set(final_scenes) or len(final) != len(final_scenes):
        raise ValueError("locked final spec and final runtime manifest disagree")
    if not set(DEV8).union(HOLDOUT5).issubset(tune):
        raise ValueError("tune24 must contain the frozen DEV8 and HOLDOUT5")
    tune_physical = {physical_scene_id(value) for value in tune}
    expected_tune_physical = {
        physical_scene_id(value) for value in (*DEV8, *HOLDOUT5)
    }
    if tune_physical != expected_tune_physical:
        raise ValueError("tune24 physical scenes must equal DEV8 plus HOLDOUT5")

    locked_rows = _locked_scene_rows(config.locked_evaluation_scenes)
    if set(locked_rows) != set(final) or len(locked_rows) != len(final):
        raise ValueError("locked evaluation artifact and final runtime disagree")
    source_hash = str(provenance.get("source_table_sha256", ""))
    if not source_hash or sha256_file(config.train_stats) != source_hash:
        raise ValueError(
            "train statistics do not match category-prior source_table_sha256"
        )
    train = _load_train_scene_ids(config.train_stats)
    groups = {
        "train": {physical_scene_id(value) for value in train},
        "tune13": tune_physical,
        "final48": {physical_scene_id(value) for value in final},
    }
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left].intersection(groups[right])
            if overlap:
                raise ValueError(
                    f"physical-scene split overlap {left}/{right}: {sorted(overlap)}"
                )

    replacement_id = "scene0019_01"
    if replacement_id not in final_scenes or "scene0019_00" in final_scenes:
        raise ValueError("final48 must use scene0019_01, never scene0019_00")
    replacement_gt = _gt_path(
        final_scenes[replacement_id],
        config.locked_gt_dir,
        replacement_id,
        locked_row=locked_rows[replacement_id],
        locked_spec_dir=config.locked_evaluation_scenes.parent,
    )
    if not replacement_gt.is_file():
        raise FileNotFoundError(
            f"registered scene0019_01 replacement GT is missing: {replacement_gt}"
        )
    if replacement_gt.stem != replacement_id:
        raise ValueError("scene0019_01 replacement GT has an inconsistent filename")

    def gt_identities(
        scene_ids: Sequence[str],
        runtime_scenes: Mapping[str, Mapping[str, Any]],
        gt_root: Path,
        *,
        registered_rows: Mapping[str, Mapping[str, Any]] | None = None,
        registered_base: Path | None = None,
    ) -> dict[str, dict[str, str]]:
        identities: dict[str, dict[str, str]] = {}
        for scene_id in scene_ids:
            path = _gt_path(
                runtime_scenes[scene_id],
                gt_root,
                scene_id,
                locked_row=(
                    registered_rows[scene_id]
                    if registered_rows is not None
                    else None
                ),
                locked_spec_dir=registered_base,
            )
            if not path.is_file():
                raise FileNotFoundError(f"registered GT is missing: {path}")
            identities[scene_id] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        return identities

    tune_gt = gt_identities(tune, tune_scenes, config.gt_dir)
    final_gt = gt_identities(
        final,
        final_scenes,
        config.locked_gt_dir,
        registered_rows=locked_rows,
        registered_base=config.locked_evaluation_scenes.parent,
    )
    return {
        "tune_scan_count": len(tune),
        "tune_physical_scene_count": len(tune_physical),
        "final_scan_count": len(final),
        "final_physical_scene_count": len(groups["final48"]),
        "train_scan_count": len(train),
        "train_physical_scene_count": len(groups["train"]),
        "physical_split_overlap": False,
        "category_prior_splits": ["train"],
        "taxonomy_sha256": taxonomy.content_hash,
        "taxonomy_classes": list(taxonomy.canonical_classes),
        "category_priors_sha256": sha256_file(config.category_priors),
        "runtime_manifest": str(config.runtime_manifest),
        "runtime_manifest_sha256": sha256_file(config.runtime_manifest),
        "locked_runtime_manifest": str(config.locked_runtime_manifest),
        "locked_runtime_manifest_sha256": sha256_file(
            config.locked_runtime_manifest
        ),
        "locked_evaluation_scenes_sha256": sha256_file(
            config.locked_evaluation_scenes
        ),
        "size_bins": str(config.size_bins),
        "size_bins_sha256": sha256_file(config.size_bins),
        "train_stats": str(config.train_stats),
        "train_stats_sha256": source_hash,
        "gt_dir": str(config.gt_dir.resolve()),
        "locked_gt_dir": str(config.locked_gt_dir.resolve()),
        "tune_gt": tune_gt,
        "tune_gt_identity_sha256": hash_json(tune_gt),
        "final_gt": final_gt,
        "final_gt_identity_sha256": hash_json(final_gt),
        "scene0019_replacement_gt": str(replacement_gt),
        "scene0019_replacement_gt_sha256": sha256_file(replacement_gt),
    }


def _gate_stage2(analysis: Mapping[str, Any]) -> tuple[bool, str]:
    capacity = analysis["capacity"]
    if not all(
        (
            capacity["match_025"] >= 20,
            capacity["match_025_scene_count"] >= 6,
            capacity["match_025_class_count"] >= 4,
            capacity["match_050"] >= 12,
            capacity["match_050_scene_count"] >= 4,
            capacity["match_050_class_count"] >= 4,
        )
    ):
        return False, "baseline_candidate_capacity_insufficient"
    mechanical = analysis["mechanical_effect"]
    if not mechanical["passed"]:
        return False, "class_size_did_not_materially_intervene"
    ranking = analysis["ranking_gate"]
    if not ranking["passed"]:
        if not analysis["oracle_gate"]["oracle_better_than_global"]:
            return False, "class_size_has_no_oracle_discrimination_value"
        if analysis["oracle_gate"]["oracle_better_than_predicted"]:
            return False, "early_classification_is_the_bottleneck"
        return False, "class_size_clean_negative_at_candidate_ranking"
    return True, "proceed_to_end_to_end"


REGISTERED_CONCLUSIONS = {
    "baseline_candidate_capacity_insufficient": "baseline候选容量不足",
    "class_size_did_not_materially_intervene": "类别尺寸没有形成有效干预",
    "early_classification_is_the_bottleneck": "oracle类别有效但自动类别错误",
    "ranking_improved_but_recovery_failed": "类别尺寸改善排序但恢复接口无效",
    "tiny_small_exploratory_signal_only": "仅小物体出现探索性收益",
    "full_class_size_prior_stably_effective": "全类别尺寸先验在锁定内部复核中稳定有效",
    "full_class_size_prior_clean_negative": "全类别尺寸先验在当前老师自动流程上得到干净负结果",
}


def _registered_conclusion(key: str) -> dict[str, str]:
    if key not in REGISTERED_CONCLUSIONS:
        raise KeyError(f"unknown registered conclusion: {key}")
    return {"registered_conclusion": key, "registered_conclusion_zh": REGISTERED_CONCLUSIONS[key]}


def _write_status(config: ExperimentConfig, **values: Any) -> None:
    write_json(
        config.artifacts_root / "full_instance_size_status.json",
        {"schema": SCHEMA, "updated_at_unix": time.time(), **values},
    )


def _resource_checkpoint(config: ExperimentConfig, **status: Any) -> dict[str, Any]:
    """Enforce the disk floor and capture the read-only cgroup state."""

    snapshot = _resource_snapshot(config)
    _write_status(config, **status, resource_snapshot=snapshot)
    return snapshot


def run_full_instance_size_prior(config: ExperimentConfig) -> dict[str, Any]:
    """Run all registered stages, resuming exact per-scene artifacts."""

    from .clean_baseline.metric_reaudit import evaluate_gt_as_prediction_dual_protocols
    from .full_instance_size_evaluation import (
        analyze_candidate_ranking,
        choose_global_threshold,
        evaluate_candidate_scenes,
        paired_physical_scene_bootstrap,
    )
    config.validate()
    if _git_commit(config.workspace) != config.git_commit:
        raise ValueError("workspace HEAD differs from registered git_commit")
    config.runs_root.mkdir(parents=True, exist_ok=True)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    resources_start = _resource_snapshot(config)
    tune_scenes = load_scene_runtime_manifest(config.runtime_manifest)
    final_scenes = load_scene_runtime_manifest(config.locked_runtime_manifest)
    taxonomy = load_taxonomy(config.taxonomy_path)
    priors = load_json(config.category_priors)
    size_spec = load_json(config.size_bins)
    size_boundaries = size_spec.get("boundaries_m", size_spec)
    if not isinstance(size_boundaries, Mapping):
        raise TypeError("size bins must contain metric boundaries")
    size_values = [
        float(size_boundaries[key])
        for key in ("tiny_max_m", "small_max_m", "medium_max_m")
    ]
    if not all(math.isfinite(value) for value in size_values) or not (
        0.0 < size_values[0] <= size_values[1] <= size_values[2]
    ):
        raise ValueError("size-bin boundaries must be finite, positive, and ordered")
    tune24 = tuple(sorted(tune_scenes))
    final48 = _parse_scene_spec(config.locked_evaluation_scenes)
    if len(tune24) != 24 or len({physical_scene_id(value) for value in tune24}) != 13:
        raise ValueError("tune scene spec must contain 24 scans / 13 physical scenes")
    if len(final48) != 48 or len({physical_scene_id(value) for value in final48}) != 48:
        raise ValueError("locked final spec must contain 48 physical scenes")
    if "scene0019_00" in final48 or (
        any(physical_scene_id(value) == "scene0019" for value in final48)
        and "scene0019_01" not in final48
    ):
        raise ValueError("locked final scene0019 must use scene0019_01")
    split_audit = _validate_registered_inputs(
        config,
        tune_scenes=tune_scenes,
        final_scenes=final_scenes,
        tune24=tune24,
        final48=final48,
        priors=priors,
        taxonomy=taxonomy,
    )
    write_json(config.artifacts_root / "full_instance_split_audit.json", split_audit)
    locked_rows = _locked_scene_rows(config.locked_evaluation_scenes)

    _resource_checkpoint(config, stage="stage0", status="running")
    dev8_gt: list[GroundTruthScene] = []
    for scene_id in DEV8:
        _resource_checkpoint(
            config, stage="stage0_gt_parity", scene_id=scene_id, status="running"
        )
        _, gt, _ = _load_scene_gt(
            scene_id, tune_scenes[scene_id], config.gt_dir
        )
        dev8_gt.append(gt)
    gt_parity = evaluate_gt_as_prediction_dual_protocols(
        dev8_gt, taxonomy.canonical_classes, min_region_size=MIN_REGION_SIZE
    )
    for protocol_name, primary in (
        ("official_9", "map_50_90"),
        ("historical_10", "map_50_95"),
    ):
        aggregate = gt_parity[protocol_name]["aggregate"]
        for field in (primary, "map_0.50", "map_0.25"):
            if not math.isclose(float(aggregate[field]), 1.0, abs_tol=1e-12):
                raise AssertionError(
                    f"GT-as-prediction parity failed for {protocol_name}/{field}"
                )
    write_json(config.artifacts_root / "full_instance_gt_parity.json", gt_parity)

    snapshots: dict[str, dict[str, Any]] = {}
    for scene_id in DEV8:
        _resource_checkpoint(
            config, stage="snapshot_dev8", scene_id=scene_id, status="running"
        )
        snapshots[scene_id] = materialize_scene_snapshot(
            config,
            scene_id,
            tune_scenes[scene_id],
            taxonomy,
            runtime_manifest=config.runtime_manifest,
        )
    dev2_source_counts = {
        source: sum(
            row["source"] == source
            for scene_id in DEV2
            for row in snapshots[scene_id]["rows"]
        )
        for source in ("global", "other_classes")
    }
    if any(value <= 0 for value in dev2_source_counts.values()):
        raise AssertionError("DEV2 snapshots must contain global and other_classes instances")

    scored_rows: list[dict[str, Any]] = []
    for scene_id in DEV8:
        _resource_checkpoint(
            config, stage="score_snapshot_dev8", scene_id=scene_id, status="running"
        )
        arrays = _load_snapshot_arrays(config, scene_id)
        scene_rows, identity = _score_snapshot_rows(
            snapshots[scene_id], arrays, priors
        )
        scored_rows.extend(scene_rows)
        snapshots[scene_id]["rows"] = scene_rows
        snapshots[scene_id]["rows_sha256"] = hash_json(scene_rows)
        snapshots[scene_id]["score_arm_identity"] = identity
        write_json(_snapshot_paths(config, scene_id)["metadata"], snapshots[scene_id])
    write_rows(config.artifacts_root / "full_instance_snapshot_dev8.parquet", scored_rows)

    dev2_audit_rows: list[dict[str, Any]] = []
    for scene_id in DEV2:
        _resource_checkpoint(
            config, stage="dev2_mechanical_audit", scene_id=scene_id, status="running"
        )
        from .full_instance_vote import (
            aggregate_instance_votes,
            load_gaussian_vote_evidence,
        )

        arrays = _load_snapshot_arrays(config, scene_id)
        snapshot = load_json(_snapshot_paths(config, scene_id)["metadata"])
        repeated_rows, repeated_identity = _score_snapshot_rows(
            snapshot, arrays, priors
        )
        if repeated_rows != snapshot["rows"]:
            raise AssertionError(f"{scene_id}: repeated size scoring is not exact")
        if repeated_identity != snapshot["score_arm_identity"]:
            raise AssertionError(f"{scene_id}: repeated score identity changed")
        evidence = load_gaussian_vote_evidence(
            _snapshot_paths(config, scene_id)["votes"]
        )
        first_votes = aggregate_instance_votes(arrays["merged_partition"], evidence)
        second_votes = aggregate_instance_votes(arrays["merged_partition"], evidence)
        if first_votes.keys() != second_votes.keys() or any(
            not np.array_equal(first_votes[key], second_votes[key])
            for key in first_votes
        ):
            raise AssertionError(f"{scene_id}: repeated KNN-pre vote changed")
        early_final_comparable = [
            row
            for row in snapshot["rows"]
            if row.get("exported_class") is not None
            and row.get("predicted_class") is not None
        ]
        dev2_audit_rows.append(
            {
                "scene_id": scene_id,
                "global_candidate_count": sum(
                    row["source"] == "global" for row in snapshot["rows"]
                ),
                "other_classes_candidate_count": sum(
                    row["source"] == "other_classes" for row in snapshot["rows"]
                ),
                "candidate_count": len(snapshot["rows"]),
                "vote_repeat_exact": True,
                "score_repeat_exact": True,
                "same_bank_q_and_eligibility_by_construction": True,
                "class_prior_fallback_count": sum(
                    bool(row.get("class_prior_fallback", False))
                    for row in snapshot["rows"]
                ),
                "early_to_exported_class_change_count": sum(
                    str(row["predicted_class"]) != str(row["exported_class"])
                    for row in early_final_comparable
                ),
                "early_to_exported_class_compared_count": len(
                    early_final_comparable
                ),
                "vote_evidence_sha256": sha256_file(
                    _snapshot_paths(config, scene_id)["votes"]
                ),
            }
        )
    write_json(
        config.artifacts_root / "full_instance_dev2_mechanical_audit.json",
        {
            "schema": "saga-full-instance-size-dev2-mechanical-v1",
            "scene_ids": list(DEV2),
            "gt_used": False,
            "passed": True,
            "scenes": dev2_audit_rows,
        },
    )

    _resource_checkpoint(config, stage="candidate_ranking_dev8", status="running")
    candidate_eval = evaluate_candidate_scenes(
        scene_ids=DEV8,
        scenes=tune_scenes,
        gt_dir=config.gt_dir,
        snapshots=snapshots,
        taxonomy=taxonomy,
        size_spec=size_spec,
        radius_m=RADIUS_M,
        min_region_size=MIN_REGION_SIZE,
        priors=priors,
    )
    ranking_analysis = analyze_candidate_ranking(candidate_eval, scored_rows)
    proceed, decision = _gate_stage2(ranking_analysis)
    tiny_small_gate_enabled = bool(
        ranking_analysis["capacity"]["tiny_small_capacity_sufficient_for_gate"]
    )
    write_rows(
        config.artifacts_root / "full_instance_size_ranking_dev8.parquet",
        candidate_eval["rows"],
    )
    write_json(
        config.artifacts_root / "full_instance_size_ranking_analysis.json",
        {**ranking_analysis, "decision": decision, "proceed": proceed},
    )
    if not proceed:
        conclusion_key = {
            "baseline_candidate_capacity_insufficient": "baseline_candidate_capacity_insufficient",
            "class_size_did_not_materially_intervene": "class_size_did_not_materially_intervene",
            "early_classification_is_the_bottleneck": "early_classification_is_the_bottleneck",
            "class_size_has_no_oracle_discrimination_value": "full_class_size_prior_clean_negative",
            "class_size_clean_negative_at_candidate_ranking": "full_class_size_prior_clean_negative",
        }[decision]
        result = {
            "schema": SCHEMA,
            "status": "stopped",
            "stage": "stage2",
            "decision": decision,
            "category_prior_tested": decision != "baseline_candidate_capacity_insufficient",
            **_registered_conclusion(conclusion_key),
            "evidence": {"candidate_ranking": ranking_analysis},
            "resources_start": resources_start,
            "resources_end": _resource_snapshot(config),
        }
        write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
        _write_status(config, **result)
        return result

    threshold_result = choose_global_threshold(candidate_eval, DEV2, THRESHOLD_GRID)
    if int(threshold_result["retained_true_positives_025"]) <= 0:
        decision = "no_global_threshold_retains_a_true_positive"
        result = {
            "schema": SCHEMA,
            "status": "stopped",
            "stage": "stage3",
            "decision": decision,
            **_registered_conclusion("ranking_improved_but_recovery_failed"),
            "evidence": {
                "candidate_ranking": ranking_analysis,
                "threshold_selection": threshold_result,
            },
            "resources_start": resources_start,
            "resources_end": _resource_snapshot(config),
        }
        write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
        _write_status(config, **result)
        return result
    threshold = float(threshold_result["threshold"])
    write_json(config.artifacts_root / "full_instance_size_threshold.json", threshold_result)

    def run_endpoint_stage(
        stage: str,
        scene_ids: Sequence[str],
        *,
        runtime_scenes: Mapping[str, Mapping[str, Any]],
        runtime_manifest: Path,
        gt_dir: Path,
        stage_locked_rows: Mapping[str, Mapping[str, Any]] | None = None,
        locked_spec_dir: Path | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        for scene_id in scene_ids:
            _resource_checkpoint(
                config,
                stage=stage,
                scene_id=scene_id,
                phase="materialize",
                status="running",
            )
            if scene_id not in runtime_scenes:
                raise KeyError(f"{stage}: runtime manifest lacks {scene_id}")
            materialize_scene_snapshot(
                config,
                scene_id,
                runtime_scenes[scene_id],
                taxonomy,
                runtime_manifest=runtime_manifest,
            )
            snapshot = load_json(_snapshot_paths(config, scene_id)["metadata"])
            arrays = _load_snapshot_arrays(config, scene_id)
            merged_rows, identity = _score_snapshot_rows(snapshot, arrays, priors)
            snapshot["rows"] = merged_rows
            snapshot["rows_sha256"] = hash_json(merged_rows)
            snapshot["score_arm_identity"] = identity
            write_json(_snapshot_paths(config, scene_id)["metadata"], snapshot)
            for condition in CONDITIONS:
                _materialize_condition_output(
                    config,
                    stage,
                    scene_id,
                    runtime_scenes[scene_id],
                    condition,
                    threshold,
                )
            _resource_checkpoint(
                config,
                stage=stage,
                scene_id=scene_id,
                phase="materialize_complete",
                status="running",
            )
        rows: list[dict[str, Any]] = []
        ground_truth_by_scene: dict[str, GroundTruthScene] = {}
        predictions_by_condition: dict[str, list[PredictedInstance]] = {
            condition: [] for condition in CONDITIONS
        }
        native_predictions_by_condition: dict[str, list[PredictedInstance]] = {
            condition: [] for condition in CONDITIONS
        }
        for scene_id in scene_ids:
            _resource_checkpoint(
                config,
                stage=stage,
                scene_id=scene_id,
                phase="evaluate",
                status="running",
            )
            condition_labels = {
                condition: np.asarray(
                    load_json(
                        _condition_output_paths(
                            config, stage, condition, scene_id
                        )["output"]
                    )["point_labels"],
                    dtype=np.int64,
                )
                for condition in CONDITIONS
            }
            class_global_changed = int(
                np.count_nonzero(
                    condition_labels["class-size"]
                    != condition_labels["global-size"]
                )
            )
            for condition in CONDITIONS:
                row, ground_truth, predictions, native_predictions = _endpoint_scene_metrics(
                    config,
                    stage,
                    scene_id,
                    runtime_scenes[scene_id],
                    condition,
                    taxonomy,
                    size_spec,
                    gt_dir=gt_dir,
                    locked_row=(
                        stage_locked_rows.get(scene_id)
                        if stage_locked_rows is not None
                        else None
                    ),
                    locked_spec_dir=locked_spec_dir,
                )
                previous_gt = ground_truth_by_scene.setdefault(scene_id, ground_truth)
                if previous_gt is not ground_truth and (
                    not np.array_equal(previous_gt.semantic, ground_truth.semantic)
                    or not np.array_equal(previous_gt.instance, ground_truth.instance)
                ):
                    raise AssertionError(f"{scene_id}: GT changed across conditions")
                predictions_by_condition[condition].extend(predictions)
                native_predictions_by_condition[condition].extend(native_predictions)
                condition_diagnostics = load_json(
                    _condition_output_paths(
                        config, stage, condition, scene_id
                    )["diagnostics"]
                )
                row.update(
                    {
                        "selected_count": int(
                            condition_diagnostics["selected_count"]
                        ),
                        "selected_not_exported": int(
                            condition_diagnostics["selected_not_exported"]
                        ),
                        "early_to_final_class_change_count": int(
                            condition_diagnostics[
                                "early_to_final_class_change_count"
                            ]
                        ),
                        "early_to_final_class_change_rate": float(
                            condition_diagnostics[
                                "early_to_final_class_change_rate"
                            ]
                        ),
                        "changed_export_points_vs_frozen_baseline": int(
                            condition_diagnostics[
                                "changed_export_points_vs_frozen_baseline"
                            ]
                        ),
                        "changed_points_class_vs_global": class_global_changed,
                    }
                )
                rows.append(row)
            _resource_checkpoint(
                config,
                stage=stage,
                scene_id=scene_id,
                phase="evaluate_complete",
                status="running",
            )
        scene_equal_conditions = _condition_summary(rows)
        pooled_conditions = _pooled_official_condition_summary(
            ground_truth_by_scene=ground_truth_by_scene,
            predictions_by_condition=predictions_by_condition,
            native_predictions_by_condition=native_predictions_by_condition,
            taxonomy=taxonomy,
            structural_summaries=scene_equal_conditions,
        )
        analysis = {
            "stage": stage,
            "conditions": pooled_conditions,
            "pooled_official_conditions": pooled_conditions,
            "scene_equal_conditions": scene_equal_conditions,
            "global_vs_baseline": _pair_delta_rows(
                rows, "controlled-baseline", "global-size"
            ),
            "class_vs_global": _pair_delta_rows(rows, "global-size", "class-size"),
        }
        return rows, analysis

    dev8_rows, dev8_analysis = run_endpoint_stage(
        "dev8",
        DEV8,
        runtime_scenes=tune_scenes,
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
    )
    write_rows(
        config.artifacts_root / "full_instance_size_end_to_end_dev8.parquet", dev8_rows
    )
    dev8_gate = _endpoint_gate(
        dev8_rows,
        threshold_dev_scenes=set(DEV2),
        tiny_small_gate_enabled=tiny_small_gate_enabled,
        official_summaries=dev8_analysis["pooled_official_conditions"],
    )
    dev8_analysis["gate"] = dev8_gate
    write_json(config.artifacts_root / "full_instance_size_dev8_analysis.json", dev8_analysis)
    if not dev8_gate["passed"]:
        conclusion_key = (
            "tiny_small_exploratory_signal_only"
            if dev8_gate["decision"] == "tiny_small_exploratory_signal_only"
            else "ranking_improved_but_recovery_failed"
        )
        result = {
            "schema": SCHEMA,
            "status": "stopped",
            "stage": "stage3",
            "decision": dev8_gate["decision"],
            **_registered_conclusion(conclusion_key),
            "evidence": {
                "candidate_ranking": ranking_analysis,
                "threshold_selection": threshold_result,
                "dev8": dev8_analysis,
            },
            "resources_start": resources_start,
            "resources_end": _resource_snapshot(config),
        }
        write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
        _write_status(config, **result)
        return result

    holdout_rows, holdout_analysis = run_endpoint_stage(
        "holdout5",
        HOLDOUT5,
        runtime_scenes=tune_scenes,
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
    )
    write_rows(config.artifacts_root / "full_instance_size_holdout5.parquet", holdout_rows)
    holdout_gate = _holdout_gate(
        holdout_rows, tiny_small_gate_enabled=tiny_small_gate_enabled
    )
    holdout_analysis["gate"] = holdout_gate
    write_json(
        config.artifacts_root / "full_instance_size_holdout5_analysis.json",
        holdout_analysis,
    )
    if not holdout_gate["passed"]:
        result = {
            "schema": SCHEMA,
            "status": "stopped",
            "stage": "stage4",
            "decision": "holdout5_failed",
            **_registered_conclusion("full_class_size_prior_clean_negative"),
            "evidence": {
                "candidate_ranking": ranking_analysis,
                "threshold_selection": threshold_result,
                "dev8": dev8_analysis,
                "holdout5": holdout_analysis,
            },
            "resources_start": resources_start,
            "resources_end": _resource_snapshot(config),
        }
        write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
        _write_status(config, **result)
        return result

    tune_rows, tune_analysis = run_endpoint_stage(
        "tune24",
        tune24,
        runtime_scenes=tune_scenes,
        runtime_manifest=config.runtime_manifest,
        gt_dir=config.gt_dir,
    )
    write_rows(config.artifacts_root / "full_instance_size_tune24.parquet", tune_rows)
    tune_gate = _tune_gate(
        tune_rows, tiny_small_gate_enabled=tiny_small_gate_enabled
    )
    tune_analysis["gate"] = tune_gate
    tune_analysis["physical_scene_equal_conditions"] = _condition_summary(
        _physical_macro(tune_rows)
    )
    write_json(
        config.artifacts_root / "full_instance_size_tune24_analysis.json",
        tune_analysis,
    )
    if not tune_gate["passed"]:
        result = {
            "schema": SCHEMA,
            "status": "stopped",
            "stage": "stage5_tune",
            "decision": "tune24_failed",
            **_registered_conclusion("full_class_size_prior_clean_negative"),
            "evidence": {
                "candidate_ranking": ranking_analysis,
                "threshold_selection": threshold_result,
                "dev8": dev8_analysis,
                "holdout5": holdout_analysis,
                "tune24": tune_analysis,
            },
            "resources_start": resources_start,
            "resources_end": _resource_snapshot(config),
        }
        write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
        _write_status(config, **result)
        return result

    final_rows, final_analysis = run_endpoint_stage(
        "final48",
        final48,
        runtime_scenes=final_scenes,
        runtime_manifest=config.locked_runtime_manifest,
        gt_dir=config.locked_gt_dir,
        stage_locked_rows=locked_rows,
        locked_spec_dir=config.locked_evaluation_scenes.parent,
    )
    write_rows(config.artifacts_root / "full_instance_size_final48.parquet", final_rows)
    global_by_scan = {
        str(row["scene_id"]): float(row["official_map_50_90"])
        for row in final_rows
        if row["condition"] == "global-size"
    }
    class_by_scan = {
        str(row["scene_id"]): float(row["official_map_50_90"])
        for row in final_rows
        if row["condition"] == "class-size"
    }
    bootstrap = paired_physical_scene_bootstrap(
        global_by_scan,
        class_by_scan,
        physical_scene_by_scan={
            scene_id: physical_scene_id(scene_id) for scene_id in global_by_scan
        },
        samples=10_000,
        seed=20260804,
    )
    bootstrap["estimand"] = "scene-level-map-50-90-delta-after-physical-scene-mean"
    bootstrap["not_pooled_official_ap"] = True
    final_gate = _final_gate(
        final_rows,
        bootstrap,
        tiny_small_gate_enabled=tiny_small_gate_enabled,
        official_summaries=final_analysis["pooled_official_conditions"],
    )
    final_analysis.update({"bootstrap": bootstrap, "gate": final_gate})
    write_json(
        config.artifacts_root / "full_instance_size_final48_analysis.json",
        final_analysis,
    )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "stage": "stage5_final",
        "decision": final_gate["decision"],
        **_registered_conclusion(final_gate["decision"]),
        "threshold": threshold,
        "resources_start": resources_start,
        "resources_end": _resource_snapshot(config),
        "stage_summaries": {
            "candidate_ranking": ranking_analysis,
            "dev8": dev8_analysis,
            "holdout5": holdout_analysis,
            "tune24": tune_analysis,
        },
        "final": final_analysis,
    }
    write_json(config.artifacts_root / "full_instance_size_analysis.json", result)
    _write_status(config, **result)
    return result


def _endpoint_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_dev_scenes: set[str],
    tiny_small_gate_enabled: bool = True,
    official_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = (
        dict(official_summaries)
        if official_summaries is not None
        else _condition_summary(rows)
    )
    base = summaries["controlled-baseline"]
    global_size = summaries["global-size"]
    class_size = summaries["class-size"]
    deltas = _pair_delta_rows(rows, "global-size", "class-size")
    non_threshold = [row for row in deltas if row["scene_id"] not in threshold_dev_scenes]
    baseline_guard = all(
        (
            global_size["official_map_50_90"] >= base["official_map_50_90"] - 0.001,
            global_size["ap50"] >= base["ap50"] - 0.002,
            global_size["ap25"] >= base["ap25"] - 0.005,
            global_size["instance_count"] <= 1.25 * max(base["instance_count"], 1e-12),
            global_size["coverage"] >= base["coverage"] - 0.01,
            global_size["gaussian_micro_precision"] >= base["gaussian_micro_precision"] - 0.01,
        )
    )
    delta_map = class_size["official_map_50_90"] - global_size["official_map_50_90"]
    tiny_small_guard = _tiny_small_guard_from_summaries(
        global_size,
        class_size,
        enabled=tiny_small_gate_enabled,
    )
    class_guard = all(
        (
            delta_map >= 0.002,
            class_size["ap50"] >= global_size["ap50"] - 0.002,
            class_size["ap25"] >= global_size["ap25"] - 0.002,
            sum(row["delta_official_map_50_90"] > 0 for row in deltas) >= 5,
            sum(row["delta_official_map_50_90"] > 0 for row in non_threshold) >= 4,
            tiny_small_guard,
            class_size["fp_tp_ratio_025"] <= 1.2 * max(global_size["fp_tp_ratio_025"], 1e-12),
            class_size["gaussian_micro_precision"] >= global_size["gaussian_micro_precision"] - 0.01,
            class_size["instance_count"] <= 1.20 * max(global_size["instance_count"], 1e-12),
        )
    )
    changed = _has_real_class_global_change(rows)
    exploratory_small = bool(
        tiny_small_gate_enabled
        and class_size["tiny_small_recall_050"] is not None
        and global_size["tiny_small_recall_050"] is not None
        and class_size["tiny_small_recall_050"]
        >= global_size["tiny_small_recall_050"] + 0.01
        and delta_map > -0.0005
    )
    passed = bool(baseline_guard and class_guard and changed)
    decision = (
        "proceed_to_holdout5"
        if passed
        else "tiny_small_exploratory_signal_only"
        if exploratory_small
        else "end_to_end_size_prior_gate_failed"
    )
    return {
        "passed": passed,
        "decision": decision,
        "global_structure_safe": baseline_guard,
        "class_improvement_gate": class_guard,
        "real_endpoint_change": changed,
        "tiny_small_gate_enabled": tiny_small_gate_enabled,
        "tiny_small_guard_passed": tiny_small_guard,
        "tiny_small_exploratory_signal": exploratory_small,
        "primary_ap_aggregation": summaries["global-size"].get(
            "aggregation", "scene-equal"
        ),
    }


def _has_real_class_global_change(rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        int(row.get("changed_points_class_vs_global", 0)) > 0
        for row in rows
        if row["condition"] == "class-size"
    )


def _tiny_small_guard_from_summaries(
    uniform: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    enabled: bool,
) -> bool:
    if not enabled:
        return True
    return bool(
        uniform.get("tiny_small_recall_025") is not None
        and data.get("tiny_small_recall_025") is not None
        and uniform.get("tiny_small_recall_050") is not None
        and data.get("tiny_small_recall_050") is not None
        and data["tiny_small_recall_025"] >= uniform["tiny_small_recall_025"]
        and data["tiny_small_recall_050"] >= uniform["tiny_small_recall_050"]
    )


def _holdout_gate(
    rows: Sequence[Mapping[str, Any]], *, tiny_small_gate_enabled: bool = True
) -> dict[str, Any]:
    summaries = _condition_summary(rows)
    uniform = summaries["global-size"]
    data = summaries["class-size"]
    deltas = _pair_delta_rows(rows, "global-size", "class-size")
    changed = _has_real_class_global_change(rows)
    guard = _holdout_guard_from_summaries(
        uniform,
        data,
        tiny_small_gate_enabled=tiny_small_gate_enabled,
    )
    passed = all(
        (
            data["official_map_50_90"] > uniform["official_map_50_90"],
            sum(row["delta_official_map_50_90"] > 0 for row in deltas) >= 3,
            guard,
            changed,
        )
    )
    return {
        "passed": bool(passed),
        "positive_scene_count": sum(
            row["delta_official_map_50_90"] > 0 for row in deltas
        ),
        "real_endpoint_change": changed,
        "tiny_small_gate_enabled": tiny_small_gate_enabled,
        "tiny_small_guard_passed": _tiny_small_guard_from_summaries(
            uniform, data, enabled=tiny_small_gate_enabled
        ),
    }


def _physical_macro(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(physical_scene_id(str(row["scene_id"])), str(row["condition"]))].append(row)
    result: list[dict[str, Any]] = []
    for (physical, condition), group in sorted(by_key.items()):
        summary = _condition_summary(group)[condition]
        result.append(
            {
                "physical_scene_id": physical,
                "condition": condition,
                **{
                    field: summary.get(field)
                    for field in (
                        "official_map_50_90",
                        "ap50",
                        "ap25",
                        "tiny_small_recall_025",
                        "tiny_small_recall_050",
                        "fp_tp_ratio_025",
                        "gaussian_micro_precision",
                        "instance_count",
                        "correct_gaussian_count",
                        "predicted_gaussian_count",
                        "true_positive_count_025",
                        "false_positive_count_025",
                        "tiny_small_gt_count",
                        "tiny_small_match_count_025",
                        "tiny_small_match_count_050",
                    )
                    if summary.get(field) is not None
                },
            }
        )
    return result


def _tune_gate(
    rows: Sequence[Mapping[str, Any]], *, tiny_small_gate_enabled: bool = True
) -> dict[str, Any]:
    physical = _physical_macro(rows)
    summaries = _condition_summary(physical)
    uniform = summaries["global-size"]
    data = summaries["class-size"]
    deltas = _pair_delta_rows(
        [dict(row, scene_id=row["physical_scene_id"]) for row in physical],
        "global-size",
        "class-size",
    )
    guard = _holdout_guard_from_summaries(
        uniform,
        data,
        tiny_small_gate_enabled=tiny_small_gate_enabled,
    )
    changed = _has_real_class_global_change(rows)
    passed = (
        data["official_map_50_90"] - uniform["official_map_50_90"] >= 0.002
        and sum(row["delta_official_map_50_90"] > 0 for row in deltas) >= 7
        and guard
        and changed
    )
    return {
        "passed": bool(passed),
        "physical_scene_count": len(deltas),
        "positive_physical_scene_count": sum(
            row["delta_official_map_50_90"] > 0 for row in deltas
        ),
        "real_endpoint_change": changed,
        "tiny_small_gate_enabled": tiny_small_gate_enabled,
        "tiny_small_guard_passed": _tiny_small_guard_from_summaries(
            uniform, data, enabled=tiny_small_gate_enabled
        ),
    }


def _holdout_guard_from_summaries(
    uniform: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    tiny_small_gate_enabled: bool = True,
) -> bool:
    return bool(
        data["ap50"] >= uniform["ap50"] - 0.002
        and data["ap25"] >= uniform["ap25"] - 0.002
        and _tiny_small_guard_from_summaries(
            uniform,
            data,
            enabled=tiny_small_gate_enabled,
        )
        and data["fp_tp_ratio_025"] <= 1.2 * max(uniform["fp_tp_ratio_025"], 1e-12)
        and data["gaussian_micro_precision"] >= uniform["gaussian_micro_precision"] - 0.01
        and data["instance_count"] <= 1.2 * max(uniform["instance_count"], 1e-12)
    )


def _final_gate(
    rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    *,
    tiny_small_gate_enabled: bool = True,
    official_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summaries = (
        dict(official_summaries)
        if official_summaries is not None
        else _condition_summary(rows)
    )
    uniform = summaries["global-size"]
    data = summaries["class-size"]
    deltas = _pair_delta_rows(rows, "global-size", "class-size")
    changed = _has_real_class_global_change(rows)
    passed = bool(
        data["official_map_50_90"] - uniform["official_map_50_90"] >= 0.002
        and sum(row["delta_official_map_50_90"] > 0 for row in deltas)
        > sum(row["delta_official_map_50_90"] < 0 for row in deltas)
        and float(bootstrap["ci95"][0]) > 0.0
        and _holdout_guard_from_summaries(
            uniform,
            data,
            tiny_small_gate_enabled=tiny_small_gate_enabled,
        )
        and changed
    )
    return {
        "passed": passed,
        "decision": (
            "full_class_size_prior_stably_effective"
            if passed
            else "full_class_size_prior_clean_negative"
        ),
        "real_endpoint_change": changed,
        "tiny_small_gate_enabled": tiny_small_gate_enabled,
        "tiny_small_guard_passed": _tiny_small_guard_from_summaries(
            uniform, data, enabled=tiny_small_gate_enabled
        ),
        "primary_ap_aggregation": uniform.get("aggregation", "scene-equal"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path, required=True)
    parser.add_argument("--t1-root", type=Path, required=True)
    parser.add_argument("--rebuild-t1-root", type=Path)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-gt-dir", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--allow-rebuild-missing-traces", action="store_true")
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--disk-floor-gib", type=float, default=80.0)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    commit = str(args.git_commit or _git_commit(workspace))
    config = ExperimentConfig(
        workspace=workspace,
        runtime_manifest=args.runtime_manifest.resolve(),
        locked_runtime_manifest=args.locked_runtime_manifest.resolve(),
        t1_root=args.t1_root.resolve(),
        rebuild_t1_root=(
            args.rebuild_t1_root.resolve()
            if args.rebuild_t1_root is not None
            else (args.runs_root / "t1-rebuild").resolve()
        ),
        gt_dir=args.gt_dir.resolve(),
        locked_gt_dir=args.locked_gt_dir.resolve(),
        train_stats=args.train_stats.resolve(),
        category_priors=args.category_priors.resolve(),
        size_bins=args.size_bins.resolve(),
        locked_evaluation_scenes=args.locked_evaluation_scenes.resolve(),
        runs_root=args.runs_root.resolve(),
        artifacts_root=args.artifacts_root.resolve(),
        taxonomy_path=args.taxonomy.resolve() if args.taxonomy else None,
        git_commit=commit,
        allow_rebuild_missing_traces=bool(args.allow_rebuild_missing_traces),
        disk_floor_gib=float(args.disk_floor_gib),
        cgroup_root=args.cgroup_root.resolve(),
        python_bin=args.python_bin.resolve() if args.python_bin is not None else None,
    )
    result = run_full_instance_size_prior(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
