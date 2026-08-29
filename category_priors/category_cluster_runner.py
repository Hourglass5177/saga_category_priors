from __future__ import annotations

"""Sequential scene runner for the section-31 clustering repair.

The worker receives no GT path.  It creates B0 plus registered candidate banks
one scene at a time and resumes only from parseable, contract-valid outputs.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import json
import subprocess

import numpy as np

from .category_candidate_trace import compare_candidate_bank_identity
from .category_candidate_trace import load_candidate_formation_trace
from .category_cluster_bank import (
    CLUSTER_CONDITIONS,
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
    load_cluster_raw_audit,
)
from .category_denoise import load_candidate_bank, save_candidate_bank
from .category_denoise_runner import (
    _build_command,
    _normalize_scene_ids,
    _run_command,
    _valid_candidate_bank,
    _valid_prediction,
    _validate_common_inputs,
)
from .io import load_json, sha256_file, write_json


CLUSTER_BANK_IDENTITY_SCHEMA = "saga-category-cluster-bank-identity-v1"
DETERMINISM_REFERENCE_SCHEMA = "saga-category-cluster-determinism-reference-v1"


def _file_identity(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    stat = source.stat()
    return {
        "path": str(source),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(source),
    }


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = str(completed.stdout).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit.lower()):
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _cluster_bank_identity(
    *,
    repository: Path,
    runtime_manifest: Path,
    category_priors: Path,
    scene_id: str,
    seed: int,
    conditions: Sequence[str],
    verify_determinism: bool,
    determinism_reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the embedded producer/consumer identity for one scene bank."""

    commit = _git_commit(repository)
    return {
        "schema": CLUSTER_BANK_IDENTITY_SCHEMA,
        "scene_id": str(scene_id),
        "seed": int(seed),
        "conditions": list(map(str, conditions)),
        "producer": {
            "git_commit": commit,
            "postprocess": _file_identity(repository / "postprocess.py"),
            "cluster_bank_module": _file_identity(
                repository / "category_priors" / "category_cluster_bank.py"
            ),
        },
        "consumer": {
            "git_commit": commit,
            "cluster_runner_module": _file_identity(Path(__file__)),
        },
        "runtime_manifest": _file_identity(runtime_manifest),
        "category_priors": _file_identity(category_priors),
        "determinism": {
            "mode": (
                "measured_this_scene"
                if verify_determinism
                else "algorithm_contract_reference"
            ),
            "reference": (
                None
                if determinism_reference is None
                else dict(determinism_reference)
            ),
        },
    }


def _load_determinism_reference(
    path: str | Path, *, conditions: Sequence[str]
) -> dict[str, Any]:
    """Validate a DEV2 measured-repeatability artifact for later stages.

    The reference is embedded in every later bank identity.  It is not a
    standalone checksum file: it contains the source JSON path/stat/content
    digest plus the exact condition-level witness counts used by the gate.
    """

    source = Path(path).resolve()
    payload = load_json(source)
    if payload.get("schema") != "saga-category-cluster-evaluation-v1":
        raise ValueError("determinism reference must be a cluster evaluation artifact")
    if payload.get("phase") != "dev2":
        raise ValueError("determinism reference must come from DEV2")
    scene_ids = tuple(map(str, payload.get("scene_ids", ())))
    if len(scene_ids) < 2 or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("determinism reference needs at least two unique DEV2 scenes")
    aggregates = payload.get("conditions")
    if not isinstance(aggregates, Mapping):
        raise TypeError("determinism reference lacks condition aggregates")
    witnesses: dict[str, Any] = {}
    for condition in conditions:
        aggregate = aggregates.get(condition)
        if not isinstance(aggregate, Mapping):
            raise ValueError(
                f"determinism reference does not cover requested condition {condition}"
            )
        scene_count = int(aggregate.get("scene_count", -1))
        measured_count = int(
            aggregate.get("determinism_measured_this_scene_count", -1)
        )
        reference_count = int(
            aggregate.get("determinism_reference_scene_count", 0)
        )
        violation_count = int(
            aggregate.get("determinism_violation_count", -1)
        )
        if (
            scene_count != len(scene_ids)
            or measured_count != scene_count
            or reference_count != 0
            or violation_count != 0
        ):
            raise ValueError(
                f"{condition}: DEV2 determinism was not directly measured and clean"
            )
        witnesses[condition] = {
            "scene_count": scene_count,
            "measured_this_scene_count": measured_count,
            "violation_count": violation_count,
        }
    return {
        "schema": DETERMINISM_REFERENCE_SCHEMA,
        "artifact": _file_identity(source),
        "source_schema": str(payload["schema"]),
        "source_phase": "dev2",
        "scene_ids": list(scene_ids),
        "conditions": witnesses,
    }


def _bind_cluster_bank_identity(
    scene_root: Path,
    *,
    conditions: Sequence[str],
    identity: Mapping[str, Any],
    determinism_reference: Mapping[str, Any] | None,
) -> None:
    """Embed a verified identity in every NPZ and JSON bank sidecar."""

    expected = dict(identity)
    for condition in conditions:
        path = scene_root / condition
        bank = load_candidate_bank(path)
        existing = bank.diagnostics.get("cluster_bank_identity")
        if existing is not None and existing != expected:
            raise ValueError(
                f"{condition}: fresh bank carries a conflicting producer identity"
            )
        diagnostics = {**dict(bank.diagnostics), "cluster_bank_identity": expected}
        if determinism_reference is not None:
            diagnostics.update(
                {
                    "determinism_measured": False,
                    "determinism_measured_this_scene": False,
                    "determinism_contract_verified": True,
                    "determinism_check": "verified-dev2-algorithm-contract-reference",
                    "determinism_algorithm_contract_reference": dict(
                        determinism_reference
                    ),
                }
            )
            # Do not fabricate a zero observed violation count for a scene
            # that was intentionally not rebuilt.
            diagnostics.pop("determinism_violation_count", None)
        save_candidate_bank(replace(bank, diagnostics=diagnostics), path)


DEFAULT_CLUSTER_CONDITIONS = (
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)


def _reference_scene_root(root: Path, scene_id: str) -> Path:
    candidates = (
        root / scene_id / R0_LEGACY,
        root / scene_id / "C0-legacy",
        root / scene_id,
        root / "bank" / scene_id / R0_LEGACY,
        root / "bank" / scene_id / "C0-legacy",
        root / "bank" / scene_id,
    )
    for candidate in candidates:
        if (candidate / "bank_labels.npz").is_file():
            return candidate
    return candidates[-1]


def _normalize_conditions(conditions: Sequence[str] | None) -> tuple[str, ...]:
    requested = DEFAULT_CLUSTER_CONDITIONS if conditions is None else tuple(map(str, conditions))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("conditions must be a non-empty unique sequence")
    unknown = set(requested).difference(CLUSTER_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown cluster conditions: {sorted(unknown)}")
    if R0_LEGACY not in requested:
        requested = (R0_LEGACY, *requested)
    return tuple(condition for condition in CLUSTER_CONDITIONS if condition in requested)


def _scene_complete(
    root: Path,
    scene_id: str,
    *,
    seed: int,
    conditions: Sequence[str],
    reference_bank_root: Path | None,
    expected_identity: Mapping[str, Any],
    verify_determinism: bool,
    determinism_reference: Mapping[str, Any] | None,
) -> bool:
    try:
        if not _valid_prediction(root / "b0" / scene_id / "output.json"):
            return False
        load_cluster_raw_audit(root / "cluster_audit" / scene_id)
        bank_scene = root / "bank" / scene_id
        if not (bank_scene / "cluster_bank.log").is_file():
            return False
        banks = {}
        for condition in conditions:
            path = bank_scene / condition
            if not _valid_candidate_bank(path, scene_id=scene_id, seed=seed):
                return False
            banks[condition] = load_candidate_bank(path)
        r0 = banks[R0_LEGACY]
        if reference_bank_root is not None:
            reference = load_candidate_bank(
                _reference_scene_root(reference_bank_root, scene_id)
            )
            if not compare_candidate_bank_identity(reference, r0).matches:
                return False
        for condition, bank in banks.items():
            if bank.global_pre_knn.shape != r0.global_pre_knn.shape:
                return False
            if not (
                (bank.global_pre_knn == r0.global_pre_knn).all()
                and (bank.semantic_top1 == r0.semantic_top1).all()
                and (bank.semantic_top1_score == r0.semantic_top1_score).all()
            ):
                return False
            if bank.diagnostics.get("candidate_cluster_condition") != condition:
                return False
            if bank.diagnostics.get("cluster_bank_identity") != dict(
                expected_identity
            ):
                return False
            if verify_determinism:
                if bank.diagnostics.get("determinism_measured") is not True:
                    return False
                if (
                    bank.diagnostics.get("determinism_measured_this_scene")
                    is not True
                ):
                    return False
                if int(bank.diagnostics.get("determinism_violation_count", -1)) != 0:
                    return False
                if bank.diagnostics.get("determinism_contract_verified") is not True:
                    return False
            else:
                if determinism_reference is None:
                    return False
                if bank.diagnostics.get("determinism_measured_this_scene") is not False:
                    return False
                if bank.diagnostics.get("determinism_contract_verified") is not True:
                    return False
                if "determinism_violation_count" in bank.diagnostics:
                    return False
                if bank.diagnostics.get(
                    "determinism_algorithm_contract_reference"
                ) != dict(determinism_reference):
                    return False
        return True
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_category_cluster_bank(
    runtime_manifest: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    category_priors: str | Path,
    scene_ids: str | Sequence[str] | None = None,
    *,
    conditions: Sequence[str] | None = None,
    reference_bank_root: str | Path | None = None,
    verify_determinism: bool = False,
    determinism_reference: str | Path | None = None,
    seed: int = 42,
    python_bin: str | Path | None = None,
) -> dict[str, Any]:
    """Build R0 plus requested repaired banks without reading ground truth."""

    registered_conditions = _normalize_conditions(conditions)
    scenes, pipeline, priors, root = _validate_common_inputs(
        runtime_manifest, repo_root, category_priors, output_root, seed
    )
    selected = _normalize_scene_ids(scene_ids, scenes)
    reference = (
        Path(reference_bank_root).resolve()
        if reference_bank_root is not None
        else None
    )
    if reference is not None and not reference.is_dir():
        raise FileNotFoundError(reference)
    if verify_determinism and determinism_reference is not None:
        raise ValueError(
            "verify_determinism and determinism_reference are mutually exclusive"
        )
    if not verify_determinism and determinism_reference is None:
        raise ValueError(
            "unmeasured cluster banks require a verified DEV2 determinism reference"
        )
    determinism_contract = (
        None
        if verify_determinism
        else _load_determinism_reference(
            determinism_reference, conditions=registered_conditions
        )
    )

    rows: list[dict[str, Any]] = []
    for scene_id in selected:
        expected_identity = _cluster_bank_identity(
            repository=Path(repo_root).resolve(),
            runtime_manifest=Path(runtime_manifest).resolve(),
            category_priors=Path(category_priors).resolve(),
            scene_id=scene_id,
            seed=seed,
            conditions=registered_conditions,
            verify_determinism=verify_determinism,
            determinism_reference=determinism_contract,
        )
        bank_scene = root / "bank" / scene_id
        b0_scene = root / "b0" / scene_id
        bank_scene.mkdir(parents=True, exist_ok=True)
        b0_scene.mkdir(parents=True, exist_ok=True)
        if _scene_complete(
            root,
            scene_id,
            seed=seed,
            conditions=registered_conditions,
            reference_bank_root=reference,
            expected_identity=expected_identity,
            verify_determinism=verify_determinism,
            determinism_reference=determinism_contract,
        ):
            rows.append(
                {
                    "scene_id": scene_id,
                    "status": "skipped_complete",
                    "bank_root": str(bank_scene),
                }
            )
            continue
        command = _build_command(
            pipeline_path=pipeline,
            priors_path=priors,
            scene_id=scene_id,
            scene=scenes[scene_id],
            output_path=b0_scene / "output.json",
            progress_path=b0_scene / "progress.txt",
            diagnostics_path=b0_scene / "diagnostics.json",
            bank_path=bank_scene,
            action="cluster-bank",
            mode="uniform",
            seed=seed,
            python_bin=python_bin,
        )
        for condition in registered_conditions:
            command.extend(("--category-cluster-condition", condition))
        command.extend(
            (
                "--category-cluster-audit-path",
                str(root / "cluster_audit" / scene_id),
            )
        )
        if verify_determinism:
            command.append("--category-cluster-verify-determinism")
        return_code = _run_command(
            command, cwd=pipeline.parent, log_path=bank_scene / "cluster_bank.log"
        )
        if return_code != 0:
            raise RuntimeError(
                f"category cluster bank failed for {scene_id}; inspect "
                f"{bank_scene / 'cluster_bank.log'}"
            )
        _bind_cluster_bank_identity(
            bank_scene,
            conditions=registered_conditions,
            identity=expected_identity,
            determinism_reference=determinism_contract,
        )
        if not _scene_complete(
            root,
            scene_id,
            seed=seed,
            conditions=registered_conditions,
            reference_bank_root=reference,
            expected_identity=expected_identity,
            verify_determinism=verify_determinism,
            determinism_reference=determinism_contract,
        ):
            raise RuntimeError(
                f"category cluster bank failed for {scene_id}; inspect "
                f"{bank_scene / 'cluster_bank.log'}"
            )
        rows.append(
            {"scene_id": scene_id, "status": "complete", "bank_root": str(bank_scene)}
        )
    return {
        "schema": "saga-category-cluster-runner-result-v1",
        "action": "cluster-bank",
        "seed": int(seed),
        "conditions": list(registered_conditions),
        "reference_identity_required": reference is not None,
        "determinism_mode": (
            "measured_this_scene"
            if verify_determinism
            else "algorithm_contract_reference"
        ),
        "determinism_reference": determinism_contract,
        "total": len(rows),
        "complete": len(rows),
        "runs": rows,
    }


def audit_category_cluster_distance(
    *,
    run_root: str | Path,
    scene_ids: Sequence[str],
    reference_bank_root: str | Path,
    reference_trace_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Close R0 identity and registered distance/contract checks from banks."""

    root = Path(run_root).resolve()
    reference_root = Path(reference_bank_root).resolve()
    trace_root = Path(reference_trace_root).resolve()
    scenes = tuple(map(str, scene_ids))
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene_ids must be a non-empty unique sequence")
    rows = []
    for scene_id in scenes:
        reference = load_candidate_bank(_reference_scene_root(reference_root, scene_id))
        r0 = load_candidate_bank(root / "bank" / scene_id / R0_LEGACY)
        comparison = compare_candidate_bank_identity(reference, r0)
        reference_trace = load_candidate_formation_trace(
            trace_root / scene_id
            if (trace_root / scene_id).is_dir()
            else trace_root / "candidate_trace" / scene_id
        )
        raw_audit = load_cluster_raw_audit(root / "cluster_audit" / scene_id)
        raw_checks = {
            "sample_rank_exact": bool(
                (raw_audit["sample_rank"] == reference_trace.sample_rank).all()
            ),
            "hdbscan_labels_exact": bool(
                (
                    raw_audit["hdbscan_labels"]
                    == reference_trace.hdbscan_labels
                ).all()
            ),
            "hdbscan_membership_atol_1e-6": bool(
                np.allclose(
                    raw_audit["hdbscan_membership"],
                    reference_trace.hdbscan_membership,
                    rtol=0.0,
                    atol=1e-6,
                )
            ),
        }
        r0_determinism = {
            "condition": R0_LEGACY,
            "measured_this_scene": bool(
                r0.diagnostics.get("determinism_measured_this_scene", False)
            ),
            "violation_count": int(
                r0.diagnostics.get("determinism_violation_count", -1)
            ),
        }
        corrected = []
        for condition in (R1_METRIC_HDBSCAN, R2_ANCHORED_HDBSCAN):
            bank = load_candidate_bank(root / "bank" / scene_id / condition)
            if float(bank.diagnostics.get("global_typical_diag_m", 0.0)) <= 0:
                raise ValueError(f"{scene_id}/{condition} lacks a global metric scale")
            corrected.append(
                {
                    "condition": condition,
                    "candidate_count": len(bank.candidates),
                    "global_typical_diag_m": float(
                        bank.diagnostics["global_typical_diag_m"]
                    ),
                    "raw_member_count": int(
                        bank.diagnostics.get("raw_member_count", 0)
                    ),
                    "raw_member_retained_count": int(
                        bank.diagnostics.get("raw_member_retained_count", 0)
                    ),
                    "core_outside_full_count": int(
                        bank.diagnostics.get("core_outside_full_count", 0)
                    ),
                    "distance_matrix_count": int(
                        bank.diagnostics.get("distance_matrix_count", 0)
                    ),
                    "distance_all_finite": bool(
                        bank.diagnostics.get("distance_all_finite", False)
                    ),
                    "distance_symmetry_max_abs": float(
                        bank.diagnostics.get(
                            "distance_symmetry_max_abs", float("inf")
                        )
                    ),
                    "distance_diagonal_max_abs": float(
                        bank.diagnostics.get(
                            "distance_diagonal_max_abs", float("inf")
                        )
                    ),
                    "distance_min": float(
                        bank.diagnostics.get("distance_min", float("-inf"))
                    ),
                    "distance_max": float(
                        bank.diagnostics.get("distance_max", float("inf"))
                    ),
                    "corrected_distance_contract_passed": bool(
                        bank.diagnostics.get(
                            "corrected_distance_contract_passed", False
                        )
                    ),
                    "corrected_distance_contract_measured": bool(
                        bank.diagnostics.get(
                            "corrected_distance_contract_measured", False
                        )
                    ),
                    "determinism_measured_this_scene": bool(
                        bank.diagnostics.get(
                            "determinism_measured_this_scene", False
                        )
                    ),
                    "determinism_violation_count": int(
                        bank.diagnostics.get("determinism_violation_count", -1)
                    ),
                }
            )
        rows.append(
            {
                "scene_id": scene_id,
                "r0_identity_matches": comparison.matches,
                "r0_identity_mismatches": list(comparison.mismatches),
                "r0_max_abs_differences": dict(comparison.max_abs_differences),
                "r0_raw_identity_checks": raw_checks,
                "r0_determinism": r0_determinism,
                "corrected_conditions": corrected,
            }
        )
    payload = {
        "schema": "saga-category-cluster-distance-audit-v1",
        "scene_ids": list(scenes),
        "r0_identity_passed": all(
            row["r0_identity_matches"]
            and all(row["r0_raw_identity_checks"].values())
            for row in rows
        ),
        "corrected_distance_contract_measured": all(
            bool(condition["corrected_distance_contract_measured"])
            and int(condition["distance_matrix_count"]) > 0
            for row in rows
            for condition in row["corrected_conditions"]
        ),
        "corrected_distance_contract_passed": all(
            bool(condition["corrected_distance_contract_measured"])
            and int(condition["distance_matrix_count"]) > 0
            and bool(condition["corrected_distance_contract_passed"])
            for row in rows
            for condition in row["corrected_conditions"]
        ),
        "determinism_passed": all(
            bool(row["r0_determinism"]["measured_this_scene"])
            and int(row["r0_determinism"]["violation_count"]) == 0
            and all(
                bool(condition["determinism_measured_this_scene"])
                and int(condition["determinism_violation_count"]) == 0
                for condition in row["corrected_conditions"]
            )
            for row in rows
        ),
        "distance_contract": {
            "affinity": "acos(clipped-cosine)/pi",
            "spatial": "min(metric-distance/global-train-diag-q50,1)",
            "weights": {"affinity": 0.625, "spatial": 0.375},
            "semantic_confidence_in_pair_distance": False,
            "sample_max_normalization": False,
        },
        "scenes": rows,
    }
    if not payload["r0_identity_passed"]:
        raise ValueError("R0 identity audit failed; corrected arms must not be evaluated")
    if not payload["corrected_distance_contract_passed"]:
        raise ValueError(
            "corrected distance contract failed; repaired arms must not be evaluated"
        )
    if not payload["determinism_passed"]:
        raise ValueError(
            "DEV2 pointwise determinism was not directly measured or was violated"
        )
    write_json(output_path, payload)
    return payload


__all__ = [
    "DEFAULT_CLUSTER_CONDITIONS",
    "audit_category_cluster_distance",
    "run_category_cluster_bank",
]
