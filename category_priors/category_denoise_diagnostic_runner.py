from __future__ import annotations

"""Filesystem orchestration for the frozen category-denoise diagnostics.

The algorithms live in the diagnostic modules.  This file only resolves the
existing scene assets, validates identities, and writes independent artifacts.
It never mutates a candidate bank or a baseline prediction.
"""

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise import CandidateBank, load_candidate_bank
from .category_denoise_diagnostics import (
    aggregate_candidate_funnel_results,
    diagnose_candidate_funnel_scene,
)
from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, read_rows, write_json, write_rows
from .prediction_contract import validate_prediction_contract
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import Taxonomy
from .v9_metrics import _gaussian_ply, _transform


def _normalize_scene_ids(
    scene_ids: str | Sequence[str] | None,
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    if scene_ids is None:
        selected = tuple(sorted(scenes))
    elif isinstance(scene_ids, str):
        selected = (scene_ids,)
    else:
        selected = tuple(map(str, scene_ids))
    if not selected:
        raise ValueError("at least one scene is required")
    if len(selected) != len(set(selected)):
        raise ValueError("scene_ids contains duplicates")
    invalid = [scene_id for scene_id in selected if Path(scene_id).name != scene_id]
    if invalid:
        raise ValueError(f"invalid scene IDs: {invalid}")
    missing = sorted(set(selected).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest is missing scenes: {missing}")
    return selected


def _as_finite_float(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _as_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _validate_knn_oracle_plan(
    payload: Any,
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, Mapping[str, Any]]]:
    """Validate the complete E2 plan before either replay or evaluation.

    The plan contains GT-derived *selection metadata*, but the replay consumes
    only the validated candidate IDs.  Keeping the validation here makes that
    boundary explicit and prevents a partial or stale plan from silently
    selecting a different mechanics run.
    """

    from .category_denoise_knn_oracle import PLAN_SCHEMA

    if not isinstance(payload, Mapping) or payload.get("schema") != PLAN_SCHEMA:
        raise ValueError("oracle plan has an unsupported schema")
    if payload.get("evaluation_only") is not True:
        raise ValueError("oracle plan must be marked evaluation_only")
    raw_scene_ids = payload.get("scene_ids")
    if not isinstance(raw_scene_ids, list):
        raise TypeError("oracle plan scene_ids must be a list")
    selected = _normalize_scene_ids(tuple(map(str, raw_scene_ids)), scenes)
    scene_entries = payload.get("scenes")
    if not isinstance(scene_entries, list):
        raise TypeError("oracle plan scenes must be a list")
    if not all(isinstance(row, Mapping) for row in scene_entries):
        raise TypeError("every oracle plan scene must be an object")
    scene_by_id = {str(row.get("scene_id")): row for row in scene_entries}
    if len(scene_by_id) != len(scene_entries) or set(scene_by_id) != set(selected):
        raise ValueError("oracle plan scene identity is inconsistent")

    iou_threshold = _as_finite_float(payload.get("iou_threshold"), "iou_threshold")
    radius_m = _as_finite_float(payload.get("radius_m"), "radius_m")
    min_region_size = _as_nonnegative_int(
        payload.get("min_region_size"), "min_region_size"
    )
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if radius_m <= 0.0 or min_region_size <= 0:
        raise ValueError("radius_m and min_region_size must be positive")

    total_candidates = 0
    scenes_with_candidates = 0
    for scene_id in selected:
        row = scene_by_id[scene_id]
        if row.get("schema") != PLAN_SCHEMA or row.get("evaluation_only") is not True:
            raise ValueError(f"{scene_id}: inconsistent oracle scene schema")
        if _as_nonnegative_int(row.get("point_count"), "point_count") <= 0:
            raise ValueError(f"{scene_id}: point_count must be positive")
        _as_nonnegative_int(row.get("bank_seed"), "bank_seed")
        if not isinstance(row.get("bank_schema"), str) or not row["bank_schema"]:
            raise ValueError(f"{scene_id}: bank_schema is missing")
        if _as_finite_float(row.get("iou_threshold"), "iou_threshold") != iou_threshold:
            raise ValueError(f"{scene_id}: iou_threshold differs from the plan")
        if _as_finite_float(row.get("radius_m"), "radius_m") != radius_m:
            raise ValueError(f"{scene_id}: radius_m differs from the plan")
        if _as_nonnegative_int(row.get("min_region_size"), "min_region_size") != min_region_size:
            raise ValueError(f"{scene_id}: min_region_size differs from the plan")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, Mapping) for candidate in candidates
        ):
            raise TypeError(f"{scene_id}: candidates must be a list of objects")
        candidate_ids = [
            _as_nonnegative_int(candidate.get("candidate_id"), "candidate_id")
            for candidate in candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{scene_id}: candidate IDs contain duplicates")
        if candidate_ids != sorted(candidate_ids):
            raise ValueError(f"{scene_id}: candidate IDs must be sorted")
        for candidate in candidates:
            if str(candidate.get("scene_id")) != scene_id:
                raise ValueError(f"{scene_id}: candidate records another scene")
        total_candidates += len(candidates)
        scenes_with_candidates += bool(candidates)

    if _as_nonnegative_int(payload.get("candidate_count"), "candidate_count") != total_candidates:
        raise ValueError("oracle plan candidate_count is inconsistent")
    if _as_nonnegative_int(
        payload.get("scene_count_with_candidates"), "scene_count_with_candidates"
    ) != scenes_with_candidates:
        raise ValueError("oracle plan scene_count_with_candidates is inconsistent")
    return selected, scene_by_id


def _selection_matches(
    expected: Sequence[Mapping[str, Any]],
    recorded: Any,
) -> bool:
    """Return whether replay preserved the GT-derived selection verbatim."""

    if not isinstance(recorded, list) or len(recorded) != len(expected):
        return False
    for expected_row, recorded_row in zip(expected, recorded):
        if not isinstance(recorded_row, Mapping):
            return False
        if any(recorded_row.get(key) != value for key, value in expected_row.items()):
            return False
    return True


def _bank_scene_root(bank_root: str | Path, scene_id: str) -> Path:
    root = Path(bank_root)
    candidates = (root / scene_id, root / "bank" / scene_id)
    found = [path for path in candidates if (path / "bank_labels.npz").is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"{scene_id}: expected exactly one bank under {root}, found {found}"
        )
    return found[0]


def _load_bank(bank_root: str | Path, scene_id: str) -> CandidateBank:
    bank = load_candidate_bank(_bank_scene_root(bank_root, scene_id))
    recorded_scene = bank.diagnostics.get("scene_id")
    if recorded_scene not in {None, scene_id}:
        raise ValueError(
            f"{scene_id}: candidate bank records another scene: {recorded_scene}"
        )
    return bank


def _load_prediction(path: Path, *, point_count: int | None = None) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: prediction must be an object")
    labels = np.asarray(payload.get("point_labels"), dtype=np.int64)
    instances = payload.get("instances")
    if not isinstance(instances, Mapping):
        raise TypeError(f"{path}: instances must be an object")
    if point_count is not None and labels.shape != (int(point_count),):
        raise ValueError(f"{path}: prediction/Gaussian point count mismatch")
    validate_prediction_contract(labels, instances)
    return {"point_labels": labels, "instances": dict(instances)}


def _load_scene_arrays(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    gt_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    xyz_raw = load_ply_xyz(_gaussian_ply(scene))
    xyz_metric = apply_transform(xyz_raw, _transform(scene))
    gt_xyz, ground_truth = load_ground_truth_npz(
        Path(gt_dir) / f"{scene_id}.npz", scene_id
    )
    return xyz_raw, xyz_metric, gt_xyz, ground_truth


def _valid_completed_funnel(
    output_dir: Path,
    *,
    scene_ids: Sequence[str],
    radius_m: float,
    min_region_size: int,
    sources: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        analysis = load_json(output_dir / "candidate_funnel_analysis.json")
        read_rows(output_dir / "candidate_funnel_candidates.parquet")
        read_rows(output_dir / "candidate_funnel_gt.parquet")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    identity = analysis.get("input_identity", {}) if isinstance(analysis, Mapping) else {}
    if (
        analysis.get("schema") == "saga-category-denoise-funnel-v1"
        and tuple(identity.get("scene_ids", ())) == tuple(scene_ids)
        and float(identity.get("radius_m", -1.0)) == float(radius_m)
        and int(identity.get("min_region_size", -1)) == int(min_region_size)
        and identity.get("sources") == dict(sources)
    ):
        result = dict(analysis)
        result["status"] = "skipped_complete"
        return result
    return None


def diagnose_category_denoise_funnel(
    *,
    runtime_manifest: str | Path,
    gt_dir: str | Path,
    bank_root: str | Path,
    category_priors: str | Path,
    output_dir: str | Path,
    scene_ids: str | Sequence[str] | None,
    taxonomy: Taxonomy,
    size_bins: str | Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Run E1 over immutable banks and write its two tables and analysis."""

    scenes = load_scene_runtime_manifest(runtime_manifest)
    selected = _normalize_scene_ids(scene_ids, scenes)
    destination = Path(output_dir)
    sources = {
        "runtime_manifest": str(Path(runtime_manifest).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "bank_root": str(Path(bank_root).resolve()),
        "category_priors": str(Path(category_priors).resolve()),
        "size_bins": str(Path(size_bins).resolve()) if size_bins is not None else None,
        "taxonomy_classes": list(taxonomy.canonical_classes),
    }
    completed = _valid_completed_funnel(
        destination,
        scene_ids=selected,
        radius_m=radius_m,
        min_region_size=min_region_size,
        sources=sources,
    )
    if completed is not None:
        return completed

    priors = load_json(category_priors)
    size_spec = load_json(size_bins) if size_bins is not None else None
    results = []
    banks: list[dict[str, Any]] = []
    for scene_id in selected:
        scene = scenes[scene_id]
        bank = _load_bank(bank_root, scene_id)
        _, xyz_metric, gt_xyz, ground_truth = _load_scene_arrays(
            scene_id=scene_id, scene=scene, gt_dir=gt_dir
        )
        if bank.point_count != len(xyz_metric):
            raise ValueError(f"{scene_id}: bank/Gaussian point count mismatch")
        result = diagnose_candidate_funnel_scene(
            scene_id=scene_id,
            bank=bank,
            gaussian_xyz=xyz_metric,
            gt_xyz=gt_xyz,
            ground_truth=ground_truth,
            taxonomy=taxonomy,
            category_priors=priors,
            size_spec=size_spec,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        results.append(result)
        banks.append(
            {
                "scene_id": scene_id,
                "schema": bank.schema,
                "seed": bank.seed,
                "point_count": bank.point_count,
                "candidate_count": len(bank.candidates),
            }
        )

    candidate_rows = [row for result in results for row in result.candidate_rows]
    gt_rows = [row for result in results for row in result.gt_rows]
    analysis = aggregate_candidate_funnel_results(results)
    analysis = {
        **analysis,
        "schema": "saga-category-denoise-funnel-v1",
        "input_identity": {
            "scene_ids": list(selected),
            "radius_m": float(radius_m),
            "min_region_size": int(min_region_size),
            "sources": sources,
            "banks": banks,
        },
        "status": "complete",
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "candidate_funnel_candidates.parquet", candidate_rows)
    write_rows(destination / "candidate_funnel_gt.parquet", gt_rows)
    write_json(destination / "candidate_funnel_analysis.json", analysis)
    return analysis


def _valid_completed_prior_oracle(
    output_dir: Path,
    *,
    scene_ids: Sequence[str],
    radius_m: float,
    min_region_size: int,
    sources: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        analysis = load_json(output_dir / "prior_oracle_analysis.json")
        read_rows(output_dir / "prior_oracle_objects.parquet")
        read_rows(output_dir / "prior_oracle_pairs.parquet")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    identity = analysis.get("input_identity", {}) if isinstance(analysis, Mapping) else {}
    if (
        analysis.get("schema") == "saga-category-prior-oracle-v1"
        and tuple(identity.get("scene_ids", ())) == tuple(scene_ids)
        and float(identity.get("main_radius_m", -1.0)) == float(radius_m)
        and int(identity.get("min_region_size", -1)) == int(min_region_size)
        and identity.get("sources") == dict(sources)
    ):
        result = dict(analysis)
        result["status"] = "skipped_complete"
        return result
    return None


def diagnose_category_prior_oracle(
    *,
    runtime_manifest: str | Path,
    gt_dir: str | Path,
    category_priors: str | Path,
    output_dir: str | Path,
    scene_ids: str | Sequence[str] | None,
    taxonomy: Taxonomy,
    size_bins: str | Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Run E3 on deterministic Gaussian-domain GT objects.

    E3 is an evaluator-only capacity check.  It writes no prediction and the
    GT-derived objects never enter a bank or replay path.
    """

    from .category_denoise_prior_oracle import (
        build_oracle_scene,
        summarize_prior_oracle,
    )

    scenes = load_scene_runtime_manifest(runtime_manifest)
    selected = _normalize_scene_ids(scene_ids, scenes)
    destination = Path(output_dir)
    sources = {
        "runtime_manifest": str(Path(runtime_manifest).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "category_priors": str(Path(category_priors).resolve()),
        "size_bins": str(Path(size_bins).resolve()) if size_bins is not None else None,
        "taxonomy_classes": list(taxonomy.canonical_classes),
    }
    completed = _valid_completed_prior_oracle(
        destination,
        scene_ids=selected,
        radius_m=radius_m,
        min_region_size=min_region_size,
        sources=sources,
    )
    if completed is not None:
        return completed

    priors = load_json(category_priors)
    size_spec = load_json(size_bins) if size_bins is not None else None
    results: list[dict[str, Any]] = []
    point_counts: list[dict[str, Any]] = []
    for scene_id in selected:
        scene = scenes[scene_id]
        xyz_raw = load_ply_xyz(_gaussian_ply(scene))
        gt_xyz, ground_truth = load_ground_truth_npz(
            Path(gt_dir) / f"{scene_id}.npz", scene_id
        )
        results.append(
            build_oracle_scene(
                scene_id=scene_id,
                physical_scene_id=physical_scene_id(scene_id),
                gaussian_xyz=xyz_raw,
                gaussian_to_gt_transform=_transform(scene),
                gt_xyz_m=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                class_names=taxonomy.canonical_classes,
                priors=priors,
                size_bins=size_spec,
                radii_m=(0.02, float(radius_m), 0.10),
                main_radius_m=float(radius_m),
                min_region_size=min_region_size,
            )
        )
        point_counts.append(
            {
                "scene_id": scene_id,
                "gaussian_count": len(xyz_raw),
                "gt_point_count": len(gt_xyz),
            }
        )

    object_rows = [dict(row) for result in results for row in result["objects"]]
    pair_rows = [dict(row) for result in results for row in result["pairs"]]
    analysis = summarize_prior_oracle(results, main_radius_m=float(radius_m))
    analysis = {
        **analysis,
        "schema": "saga-category-prior-oracle-v1",
        "input_identity": {
            "scene_ids": list(selected),
            "main_radius_m": float(radius_m),
            "sensitivity_radii_m": [0.02, float(radius_m), 0.10],
            "min_region_size": int(min_region_size),
            "sources": sources,
            "point_counts": point_counts,
        },
        "status": "complete",
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "prior_oracle_objects.parquet", object_rows)
    write_rows(destination / "prior_oracle_pairs.parquet", pair_rows)
    write_json(destination / "prior_oracle_analysis.json", analysis)
    return analysis


def prepare_category_denoise_knn_oracle(
    *,
    runtime_manifest: str | Path,
    gt_dir: str | Path,
    bank_root: str | Path,
    output: str | Path,
    scene_ids: str | Sequence[str] | None,
    taxonomy: Taxonomy,
    iou_threshold: float = 0.50,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Prepare the GT-selected E2 plan without running KNN."""

    from .category_denoise_knn_oracle import PLAN_SCHEMA, prepare_knn_oracle_scene

    scenes = load_scene_runtime_manifest(runtime_manifest)
    selected = _normalize_scene_ids(scene_ids, scenes)
    destination = Path(output)
    try:
        existing = load_json(destination)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, Mapping):
        try:
            existing_scenes, existing_by_id = _validate_knn_oracle_plan(
                existing, scenes
            )
            identity_matches = (
                existing_scenes == selected
                and float(existing["iou_threshold"]) == float(iou_threshold)
                and float(existing["radius_m"]) == float(radius_m)
                and int(existing["min_region_size"]) == int(min_region_size)
            )
            for scene_id in selected:
                bank = _load_bank(bank_root, scene_id)
                row = existing_by_id[scene_id]
                identity_matches &= (
                    bank.point_count == int(row["point_count"])
                    and bank.schema == str(row["bank_schema"])
                    and bank.seed == int(row["bank_seed"])
                    and all(
                        int(candidate["candidate_id"]) < len(bank.candidates)
                        for candidate in row["candidates"]
                    )
                )
            if identity_matches:
                payload = dict(existing)
                payload["status"] = "skipped_complete"
                return payload
        except (TypeError, ValueError, KeyError, OSError):
            pass

    scene_plans = []
    for scene_id in selected:
        scene = scenes[scene_id]
        bank = _load_bank(bank_root, scene_id)
        _, xyz_metric, gt_xyz, ground_truth = _load_scene_arrays(
            scene_id=scene_id, scene=scene, gt_dir=gt_dir
        )
        if bank.point_count != len(xyz_metric):
            raise ValueError(f"{scene_id}: bank/Gaussian point count mismatch")
        scene_plans.append(
            prepare_knn_oracle_scene(
                scene_id=scene_id,
                bank=bank,
                gaussian_xyz_metric=xyz_metric,
                gt_xyz=gt_xyz,
                gt_semantic=ground_truth.semantic,
                gt_instance=ground_truth.instance,
                canonical_classes=taxonomy.canonical_classes,
                iou_threshold=iou_threshold,
                radius_m=radius_m,
                min_region_size=min_region_size,
            )
        )
    candidate_count = sum(len(plan.candidates) for plan in scene_plans)
    payload = {
        "schema": PLAN_SCHEMA,
        "evaluation_only": True,
        "iou_threshold": float(iou_threshold),
        "radius_m": float(radius_m),
        "min_region_size": int(min_region_size),
        "scene_ids": list(selected),
        "candidate_count": candidate_count,
        "scene_count_with_candidates": sum(bool(plan.candidates) for plan in scene_plans),
        "scenes": [plan.to_dict() for plan in scene_plans],
        "status": "complete",
    }
    write_json(destination, payload)
    return payload


def _write_npz_with_part(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(part, path)


def _oracle_scene_complete(
    root: Path,
    *,
    scene_plan: Mapping[str, Any],
    bank_path: Path,
    b0_path: Path,
) -> bool:
    point_count = int(scene_plan["point_count"])
    try:
        diagnostics = load_json(root / "diagnostics.json")
        if not isinstance(diagnostics, Mapping):
            return False
        copied_b0 = _load_prediction(
            root.parent.parent / "B0" / root.name / "output.json", point_count=point_count
        )
        source_b0 = _load_prediction(b0_path, point_count=point_count)
        if not np.array_equal(copied_b0["point_labels"], source_b0["point_labels"]):
            return False
        if copied_b0["instances"] != source_b0["instances"]:
            return False
        expected_candidates = scene_plan.get("candidates", ())
        expected_ids = [int(row["candidate_id"]) for row in expected_candidates]
        if (
            diagnostics.get("schema")
            != "saga-category-denoise-knn-oracle-replay-scene-v1"
            or diagnostics.get("status") != "complete"
            or diagnostics.get("scene_id") != root.name
            or int(diagnostics.get("point_count", -1)) != point_count
            or diagnostics.get("bank_schema") != scene_plan.get("bank_schema")
            or int(diagnostics.get("bank_seed", -1)) != int(scene_plan["bank_seed"])
            or Path(str(diagnostics.get("bank_path", ""))).resolve()
            != bank_path.resolve()
            or Path(str(diagnostics.get("b0_path", ""))).resolve() != b0_path.resolve()
            or diagnostics.get("candidate_ids") != expected_ids
            or not _selection_matches(expected_candidates, diagnostics.get("selection"))
            or diagnostics.get("gt_used_by_replay") is not False
        ):
            return False
        arrays = np.load(root / "oracle_replay_labels.npz", allow_pickle=False)
        if diagnostics.get("full_prediction_available"):
            _load_prediction(
                root.parent.parent / "O1-unprotected" / root.name / "output.json",
                point_count=point_count,
            )
            _load_prediction(
                root.parent.parent / "O2-protected" / root.name / "output.json",
                point_count=point_count,
            )
        required = {
            "baseline_after_knn",
            "baseline_after_filter",
            "o1_after_knn",
            "o1_after_filter",
            "o2_after_filter",
        }
        valid = required.issubset(arrays.files) and all(
            np.asarray(arrays[name]).shape == (point_count,) for name in required
        )
        arrays.close()
        return bool(valid)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _prediction_payload(labels: np.ndarray, instances: Mapping[str, Any]) -> dict[str, Any]:
    validate_prediction_contract(np.asarray(labels, dtype=np.int64), instances)
    return {
        "point_labels": np.asarray(labels, dtype=np.int64).tolist(),
        "instances": {str(key): dict(value) for key, value in instances.items()},
    }


def replay_category_denoise_knn_oracle(
    *,
    runtime_manifest: str | Path,
    bank_root: str | Path,
    b0_root: str | Path,
    oracle_plan: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the GT-free part of E2 from the frozen candidate IDs only."""

    from .category_denoise import legacy_knn_filter
    from .category_denoise_knn_oracle import (
        ExactB0MappingError,
        recover_exact_b0_mapping,
        replay_knn_oracle_scene,
        replay_protected_oracle,
        replay_unprotected_oracle,
    )

    plan = load_json(oracle_plan)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    scene_ids, scene_by_id = _validate_knn_oracle_plan(plan, scenes)
    destination = Path(output_root)
    results: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        scene_plan = scene_by_id[scene_id]
        point_count = int(scene_plan["point_count"])
        bank = _load_bank(bank_root, scene_id)
        if (
            bank.point_count != point_count
            or bank.schema != str(scene_plan["bank_schema"])
            or bank.seed != int(scene_plan["bank_seed"])
        ):
            raise ValueError(f"{scene_id}: oracle plan/bank identity mismatch")
        scene_root = destination / "scenes" / scene_id
        bank_path = _bank_scene_root(bank_root, scene_id)
        b0_path = _bank_scene_root(b0_root, scene_id) / "output.json"
        if _oracle_scene_complete(
            scene_root,
            scene_plan=scene_plan,
            bank_path=bank_path,
            b0_path=b0_path,
        ):
            diagnostics = load_json(scene_root / "diagnostics.json")
            results.append({**diagnostics, "run_status": "skipped_complete"})
            continue

        xyz_raw = load_ply_xyz(_gaussian_ply(scenes[scene_id]))
        if xyz_raw.shape != (point_count, 3):
            raise ValueError(f"{scene_id}: Gaussian point count mismatch")
        candidate_ids = tuple(
            int(row["candidate_id"]) for row in scene_plan.get("candidates", ())
        )
        baseline = legacy_knn_filter(xyz_raw, bank.global_pre_knn)
        b0 = _load_prediction(b0_path, point_count=point_count)
        # Store an exact, independent copy for the shared evaluator.  The
        # source B0 remains read-only.
        write_json(
            destination / "B0" / scene_id / "output.json",
            _prediction_payload(b0["point_labels"], b0["instances"]),
        )

        mapping_error: str | None = None
        full_prediction_available = False
        replay = None
        try:
            exact_mapping = recover_exact_b0_mapping(
                baseline.after_filter,
                b0["point_labels"],
                b0["instances"],
            )
            replay = replay_knn_oracle_scene(
                xyz_scene=xyz_raw,
                bank=bank,
                candidate_ids=candidate_ids,
                b0_mapping=exact_mapping,
            )
            write_json(
                destination / "O1-unprotected" / scene_id / "output.json",
                _prediction_payload(
                    replay.o1_prediction.point_labels,
                    replay.o1_prediction.instances,
                ),
            )
            write_json(
                destination / "O2-protected" / scene_id / "output.json",
                _prediction_payload(
                    replay.o2_prediction.point_labels,
                    replay.o2_prediction.instances,
                ),
            )
            full_prediction_available = True
            unprotected = replay.unprotected
            protected = replay.protected
        except ExactB0MappingError as exc:
            mapping_error = str(exc)
            # Candidate survival remains valid and is saved even when exact
            # global metadata cannot be reconstructed.
            unprotected = replay_unprotected_oracle(
                xyz_scene=xyz_raw, bank=bank, candidate_ids=candidate_ids
            )
            protected = replay_protected_oracle(
                xyz_scene=xyz_raw, bank=bank, candidate_ids=candidate_ids
            )

        _write_npz_with_part(
            scene_root / "oracle_replay_labels.npz",
            baseline_after_knn=baseline.after_knn,
            baseline_after_filter=baseline.after_filter,
            o1_source=unprotected.source_labels,
            o1_after_knn=unprotected.after_knn,
            o1_after_filter=unprotected.after_filter,
            o2_after_filter=protected.after_filter,
        )
        candidate_plan_by_id = {
            int(row["candidate_id"]): dict(row)
            for row in scene_plan.get("candidates", ())
        }
        o1_rows = {
            row.candidate_id: row.to_dict() for row in unprotected.candidates
        }
        o2_rows = {
            row.candidate_id: row.to_dict() for row in protected.candidates
        }
        candidate_rows = []
        for candidate_id in candidate_ids:
            candidate_rows.append(
                {
                    **candidate_plan_by_id[candidate_id],
                    "o1": o1_rows[candidate_id],
                    "o2": o2_rows[candidate_id],
                }
            )
        diagnostics = {
            "schema": "saga-category-denoise-knn-oracle-replay-scene-v1",
            "status": "complete",
            "scene_id": scene_id,
            "point_count": point_count,
            "bank_schema": bank.schema,
            "bank_seed": bank.seed,
            "bank_path": str(bank_path.resolve()),
            "b0_path": str(b0_path.resolve()),
            "candidate_count": len(candidate_ids),
            "candidate_ids": list(candidate_ids),
            "selection": [dict(row) for row in scene_plan.get("candidates", ())],
            "gt_used_by_replay": False,
            "full_prediction_available": full_prediction_available,
            "mapping_error": mapping_error,
            "baseline": {
                "k_effective": baseline.k_effective,
                "min_count": baseline.min_count,
                "instance_count_before_filter": baseline.instance_count_before_filter,
                "instance_count_after_filter": baseline.instance_count_after_filter,
                "removed_instance_ids": list(baseline.removed_instance_ids),
            },
            "candidate_raw_labels": {
                "O1-unprotected": {
                    str(key): int(value)
                    for key, value in unprotected.candidate_raw_labels.items()
                },
                "O2-protected": {
                    str(key): int(value)
                    for key, value in protected.candidate_raw_labels.items()
                },
            },
            "candidates": candidate_rows,
            "collateral": replay.diagnostics if replay is not None else None,
        }
        write_json(scene_root / "diagnostics.json", diagnostics)
        results.append({**diagnostics, "run_status": "complete"})

    summary = {
        "schema": "saga-category-denoise-knn-oracle-replay-v1",
        "status": "complete",
        "oracle_plan": str(Path(oracle_plan)),
        "evaluation_only_plan": True,
        "gt_used_by_replay": False,
        "scene_ids": list(scene_ids),
        "scene_count": len(scene_ids),
        "candidate_count": sum(row["candidate_count"] for row in results),
        "full_prediction_scene_count": sum(
            bool(row["full_prediction_available"]) for row in results
        ),
        "scenes": results,
    }
    write_json(destination / "knn_oracle_replay.json", summary)
    return summary
