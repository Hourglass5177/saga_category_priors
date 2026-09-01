from __future__ import annotations

"""Offline H'/P mask-contract ablation evaluation.

This module is deliberately evaluation-only.  It consumes already-frozen
hierarchy (H') and flat (P) evidence banks plus their C0 outputs.  Ground truth
is used only after both banks and predictions have been produced.  The main
entry point follows the two-step experiment manifest and writes the registered
flat parquet rows and nested JSON analysis without invoking a renderer or a
consensus worker.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluator import (
    GroundTruthScene,
    PredictedInstance,
    apply_transform,
    load_ground_truth_npz,
    load_ply_xyz,
)
from ..io import build_file_manifest, load_json, sha256_file, write_json, write_rows
from ..prediction_contract import validate_prediction_contract
from .evidence import load_evidence_bank
from .evaluation import CleanCandidate, ground_truth_objects_from_arrays
from .metric_reaudit import (
    FORMAL_RADIUS_M,
    build_bidirectional_nearest,
    evaluate_candidate_set_three_spaces,
    evaluate_dual_protocols,
    evaluate_gt_as_prediction_dual_protocols,
    formal_gt_point_mask,
)
from .models import AlphaMaskEvidenceBank
from .two_step_audit import (
    MANIFEST_SCHEMA,
    REGISTERED_DEV2_SCENE_IDS,
    _size_boundaries,
    _taxonomy,
    _transform,
)


MASK_ABLATION_SCHEMA = "saga-clean-mask-contract-ablation-dev2-v1"
MASK_ABLATION_ROW_SCHEMA = "saga-clean-mask-contract-ablation-row-v1"
MASK_ARMS = ("H-hierarchy", "P-flat")
CONDITION = "C0-no-prior"
DEFAULT_SCENE_COUNT = 2
FLAT_REPEAT_SCHEMA = "saga-clean-flat-full-repeat-v1"


def _resolve(base: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an explicit non-empty path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _json_value(base: Path, value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = load_json(_resolve(base, value, name=name))
    if not isinstance(result, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return result


def _pca_bbox_diagonal_m(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz):
        raise ValueError("GT object points must be a non-empty Nx3 array")
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    if len(xyz) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(np.ptp(centered, axis=0)))


def _official_tiny_small_ids(
    *,
    gt_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_count: int,
    small_max_m: float,
    min_region_size: int,
) -> set[int]:
    """Size complete GT instances; never prefilter by successful mapping."""

    selected: set[int] = set()
    for raw_instance_id in np.unique(instance[instance >= 0]):
        instance_id = int(raw_instance_id)
        instance_mask = instance == instance_id
        values, counts = np.unique(semantic[instance_mask], return_counts=True)
        valid = (values >= 0) & (values < int(class_count))
        if not np.any(valid):
            continue
        values, counts = values[valid], counts[valid]
        # Size strata use the complete raw GT instance and never depend on
        # mapping success.  The downstream GroundTruthObject constructor is
        # still the authority for official class/min-region eligibility.
        _ = int(values[counts == counts.max()].min())
        official_points = np.flatnonzero(instance_mask)
        if len(official_points) < int(min_region_size):
            continue
        complete_points = np.flatnonzero(instance_mask)
        if _pca_bbox_diagonal_m(gt_xyz[complete_points]) <= float(small_max_m):
            selected.add(instance_id)
    return selected


def _condition_specs(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    conditions = scene.get("mask_control_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("scene.mask_control_conditions must be an explicit mapping")
    if set(conditions) != set(MASK_ARMS):
        raise ValueError(
            "scene.mask_control_conditions must contain exactly H-hierarchy and P-flat"
        )
    return conditions


def _load_arm(
    *,
    base: Path,
    scene_id: str,
    arm: str,
    spec: Mapping[str, Any],
) -> tuple[AlphaMaskEvidenceBank, dict[str, Any], dict[str, Any], Path]:
    if not isinstance(spec, Mapping):
        raise TypeError(f"{scene_id}/{arm} specification must be a mapping")
    bank_dir = _resolve(base, spec.get("bank_dir"), name=f"{arm}.bank_dir")
    if not bank_dir.is_dir():
        raise NotADirectoryError(bank_dir)
    bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
    output = _json_value(base, spec.get("output"), name=f"{arm}.output")
    diagnostics = _json_value(
        base, spec.get("diagnostics"), name=f"{arm}.diagnostics"
    )
    return bank, output, diagnostics, bank_dir


def _source_without_mask_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the registered H'/P mask-source differences."""

    normalized = dict(source)
    normalized.pop("mask_observation_mode", None)
    normalized.pop("sam_masks", None)
    producer = normalized.get("producer_inputs")
    if isinstance(producer, Mapping):
        producer = dict(producer)
        # v2 evidence identities name this selected-tree payload
        # ``sam_everything_masks``.  ``sam_masks`` is retained only for older
        # imported banks.  H' and P must differ here by construction, while
        # every other producer input must remain byte-identical.
        producer.pop("sam_everything_masks", None)
        producer.pop("sam_masks", None)
        normalized["producer_inputs"] = producer
    return normalized


def _same_resolved_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, TypeError, ValueError):
        return False


def _request_identity_matches(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value).resolve()
    try:
        return bool(
            path.is_file()
            and int(value.get("size", -1)) == path.stat().st_size
            and str(value.get("sha256", "")) == sha256_file(path)
        )
    except (OSError, TypeError, ValueError):
        return False


def _input_binding_contract(
    *,
    scene_id: str,
    input_audit: Mapping[str, Any] | None,
    hierarchy: AlphaMaskEvidenceBank,
    flat: AlphaMaskEvidenceBank,
) -> dict[str, Any]:
    """Prove that the audit, requests, selected masks, and banks are one pair."""

    checks: dict[str, bool] = {
        "audit_scene_identity": False,
        "audit_declares_binding_pass": False,
        "input_repeat_identity_pass": False,
        "input_repeat_file_set_exact": False,
        "no_stranded_input_part_files": False,
        "both_arm_bindings_present": False,
        "bank_sources_match_registered_requests": False,
        "mask_roots_match_bank_sources": False,
        "mask_manifests_match_bank_sources": False,
        "request_files_match_registered_identity": False,
        "request_payloads_match_scene_mode_and_root": False,
    }
    details: dict[str, Any] = {}
    if not isinstance(input_audit, Mapping):
        return {"passed": False, "checks": checks, "details": details}
    checks["audit_scene_identity"] = str(input_audit.get("scene_id", "")) == str(
        scene_id
    )
    checks["audit_declares_binding_pass"] = bool(
        input_audit.get("input_binding_pass") is True
    )
    checks["input_repeat_identity_pass"] = bool(
        input_audit.get("flat_repeat_identity_pass") is True
    )
    checks["input_repeat_file_set_exact"] = bool(
        input_audit.get("repeat_input_manifest_before")
        == input_audit.get("repeat_input_manifest_after")
        and isinstance(input_audit.get("repeat_input_manifest_before"), list)
    )
    parts = input_audit.get("stranded_part_files")
    checks["no_stranded_input_part_files"] = isinstance(parts, list) and not parts
    bindings = input_audit.get("input_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(MASK_ARMS):
        return {"passed": False, "checks": checks, "details": details}
    checks["both_arm_bindings_present"] = True
    banks = {"H-hierarchy": hierarchy, "P-flat": flat}
    modes = {"H-hierarchy": "hierarchy", "P-flat": "flat-highest-quality"}
    source_ok = root_ok = manifest_ok = request_identity_ok = request_payload_ok = True
    for arm in MASK_ARMS:
        binding = bindings.get(arm)
        bank = banks[arm]
        if not isinstance(binding, Mapping):
            source_ok = root_ok = manifest_ok = request_identity_ok = request_payload_ok = False
            continue
        expected_source = binding.get("expected_bank_source")
        source_ok &= isinstance(expected_source, Mapping) and dict(expected_source) == dict(
            bank.source
        )
        root_ok &= _same_resolved_path(binding.get("mask_root"), bank.source.get("sam_masks"))
        producer_inputs = bank.source.get("producer_inputs")
        actual_manifest = (
            producer_inputs.get("sam_everything_masks")
            if isinstance(producer_inputs, Mapping)
            else None
        )
        manifest_ok &= isinstance(binding.get("mask_manifest"), Mapping) and dict(
            binding["mask_manifest"]
        ) == actual_manifest
        request_identity = binding.get("evidence_request")
        identity_matches = _request_identity_matches(request_identity)
        request_identity_ok &= identity_matches
        if identity_matches:
            request_path = Path(str(request_identity["path"])).resolve()
            try:
                request = load_json(request_path)
            except (OSError, ValueError, TypeError):
                request_payload_ok = False
            else:
                scene = request.get("scene") if isinstance(request, Mapping) else None
                request_payload_ok &= bool(
                    isinstance(request, Mapping)
                    and isinstance(scene, Mapping)
                    and str(scene.get("scene_id", "")) == str(scene_id)
                    and str(request.get("mask_observation_mode", "")) == modes[arm]
                    and _same_resolved_path(request.get("sam_masks"), binding.get("mask_root"))
                    and str(request.get("producer_commit", ""))
                    == str(bank.source.get("producer_commit", ""))
                )
        else:
            request_payload_ok = False
        details[arm] = {
            "mask_root": binding.get("mask_root"),
            "request": request_identity,
        }
    checks["bank_sources_match_registered_requests"] = bool(source_ok)
    checks["mask_roots_match_bank_sources"] = bool(root_ok)
    checks["mask_manifests_match_bank_sources"] = bool(manifest_ok)
    checks["request_files_match_registered_identity"] = bool(request_identity_ok)
    checks["request_payloads_match_scene_mode_and_root"] = bool(request_payload_ok)
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def _same_frame_metadata(
    left: AlphaMaskEvidenceBank, right: AlphaMaskEvidenceBank
) -> bool:
    def row(bank: AlphaMaskEvidenceBank) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                int(item.frame_id),
                str(item.image_name),
                int(item.valid_pixel_count),
                bool(item.geometry_abstained),
            )
            for item in bank.frames
        )

    return row(left) == row(right)


def _same_visibility(left: AlphaMaskEvidenceBank, right: AlphaMaskEvidenceBank) -> bool:
    return bool(
        np.array_equal(left.frame_visibility.indptr, right.frame_visibility.indptr)
        and np.array_equal(
            left.frame_visibility.gaussian_ids,
            right.frame_visibility.gaussian_ids,
        )
        and np.array_equal(
            left.frame_visibility.visible_mass,
            right.frame_visibility.visible_mass,
        )
    )


def _gaussian_binding_contract(
    *,
    bank: AlphaMaskEvidenceBank,
    gaussian_path: Path,
    gaussian_raw: np.ndarray,
) -> dict[str, Any]:
    source_path = bank.source.get("rgb_ply")
    producer_inputs = bank.source.get("producer_inputs")
    registered = (
        producer_inputs.get("gaussian_ply")
        if isinstance(producer_inputs, Mapping)
        else None
    )
    try:
        scale = float(bank.source.get("scene_scale_m_per_unit"))
    except (TypeError, ValueError, OverflowError):
        scale = float("nan")
    expected_xyz = (
        np.asarray(np.asarray(gaussian_raw, dtype=np.float64) * scale, dtype=np.float32)
        if np.isfinite(scale) and scale > 0
        else np.empty((0, 3), dtype=np.float32)
    )
    content_ok = bool(
        isinstance(registered, Mapping)
        and _same_resolved_path(registered.get("path"), gaussian_path)
        and int(registered.get("size", -1)) == gaussian_path.stat().st_size
        and str(registered.get("sha256", "")) == sha256_file(gaussian_path)
    )
    checks = {
        "source_rgb_ply_matches_manifest": _same_resolved_path(
            source_path, gaussian_path
        ),
        "embedded_ply_content_identity_matches": content_ok,
        "scene_scale_positive": bool(np.isfinite(scale) and scale > 0),
        "bank_xyz_matches_raw_ply_times_scale": bool(
            expected_xyz.shape == bank.xyz_m.shape
            and np.array_equal(expected_xyz, bank.xyz_m)
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scene_scale_m_per_unit": scale,
    }


def _flat_gaussian_unique(bank: AlphaMaskEvidenceBank) -> tuple[bool, int]:
    duplicates = 0
    masks_by_frame: dict[int, list[int]] = {}
    for row, metadata in enumerate(bank.masks):
        masks_by_frame.setdefault(int(metadata.frame_id), []).append(row)
    for rows in masks_by_frame.values():
        counts: dict[int, int] = {}
        for row in rows:
            ids = bank.mask_support.row(row, include_ambiguous=True)[0]
            for gaussian_id in ids.tolist():
                value = int(gaussian_id)
                counts[value] = counts.get(value, 0) + 1
        duplicates += sum(count - 1 for count in counts.values() if count > 1)
    ambiguity_count = int(len(bank.frame_ambiguity.ids)) + int(
        np.count_nonzero(bank.mask_support.ambiguous)
    )
    return duplicates == 0 and ambiguity_count == 0, duplicates + ambiguity_count


def _tree_byte_manifest(
    root: Path, *, expected_relative_paths: Sequence[str]
) -> dict[str, Any]:
    files = tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=str))
    relative = [
        str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
        for path in files
    ]
    parts = sorted(value for value in relative if Path(value).name.endswith(".part"))
    return {
        "root": str(root.resolve()),
        "expected_relative_paths": sorted(map(str, expected_relative_paths)),
        "actual_relative_paths": sorted(relative),
        "file_set_exact": sorted(relative) == sorted(map(str, expected_relative_paths)),
        "part_files": parts,
        "files": build_file_manifest(files, root=root),
    }


def _full_repeat_contract(
    *,
    scene_id: str,
    flat_spec: Mapping[str, Any],
    repeat_audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    checks = {
        "repeat_audit_available": False,
        "repeat_schema_and_scene": False,
        "primary_paths_match_manifest": False,
        "registered_file_sets_exact": False,
        "no_repeat_part_files": False,
        "bank_bytes_identical": False,
        "condition_bytes_identical": False,
        "stored_audit_matches_recomputed": False,
    }
    if not isinstance(repeat_audit, Mapping):
        return {"passed": False, "checks": checks}
    checks["repeat_audit_available"] = True
    checks["repeat_schema_and_scene"] = bool(
        repeat_audit.get("schema") == FLAT_REPEAT_SCHEMA
        and str(repeat_audit.get("scene_id", "")) == str(scene_id)
    )
    primary = repeat_audit.get("primary")
    repeat = repeat_audit.get("repeat")
    if not isinstance(primary, Mapping) or not isinstance(repeat, Mapping):
        return {"passed": False, "checks": checks}
    checks["primary_paths_match_manifest"] = bool(
        _same_resolved_path(primary.get("bank_dir"), flat_spec.get("bank_dir"))
        and _same_resolved_path(primary.get("output"), flat_spec.get("output"))
        and _same_resolved_path(
            primary.get("diagnostics"), flat_spec.get("diagnostics")
        )
    )
    try:
        primary_bank = Path(str(primary["bank_dir"])).resolve()
        repeat_bank = Path(str(repeat["bank_dir"])).resolve()
        primary_condition = Path(str(primary["output"])).resolve().parent
        repeat_condition = Path(str(repeat["output"])).resolve().parent
        repeat_paths_valid = bool(
            Path(str(primary["diagnostics"])).resolve()
            == primary_condition / "diagnostics.json"
            and Path(str(repeat["diagnostics"])).resolve()
            == repeat_condition / "diagnostics.json"
            and Path(str(primary["output"])).resolve()
            == primary_condition / "output.json"
            and Path(str(repeat["output"])).resolve()
            == repeat_condition / "output.json"
        )
        bank_left = _tree_byte_manifest(
            primary_bank,
            expected_relative_paths=("diagnostics.json", "evidence.npz", "masks.json"),
        )
        bank_right = _tree_byte_manifest(
            repeat_bank,
            expected_relative_paths=("diagnostics.json", "evidence.npz", "masks.json"),
        )
        condition_left = _tree_byte_manifest(
            primary_condition,
            expected_relative_paths=("diagnostics.json", "output.json"),
        )
        condition_right = _tree_byte_manifest(
            repeat_condition,
            expected_relative_paths=("diagnostics.json", "output.json"),
        )
    except (OSError, KeyError, TypeError, ValueError):
        return {"passed": False, "checks": checks}
    all_manifests = (bank_left, bank_right, condition_left, condition_right)
    checks["registered_file_sets_exact"] = bool(
        repeat_paths_valid and all(row["file_set_exact"] for row in all_manifests)
    )
    checks["no_repeat_part_files"] = bool(
        all(not row["part_files"] for row in all_manifests)
    )
    checks["bank_bytes_identical"] = bank_left["files"] == bank_right["files"]
    checks["condition_bytes_identical"] = (
        condition_left["files"] == condition_right["files"]
    )
    recomputed_bank = {
        "passed": bool(
            bank_left["file_set_exact"]
            and bank_right["file_set_exact"]
            and not bank_left["part_files"]
            and not bank_right["part_files"]
            and bank_left["files"] == bank_right["files"]
        ),
        "primary": bank_left,
        "repeat": bank_right,
    }
    recomputed_condition = {
        "passed": bool(
            condition_left["file_set_exact"]
            and condition_right["file_set_exact"]
            and not condition_left["part_files"]
            and not condition_right["part_files"]
            and condition_left["files"] == condition_right["files"]
        ),
        "primary": condition_left,
        "repeat": condition_right,
    }
    checks["stored_audit_matches_recomputed"] = bool(
        repeat_audit.get("bank_byte_identity") == recomputed_bank
        and repeat_audit.get("condition_byte_identity") == recomputed_condition
        and repeat_audit.get("passed")
        == bool(recomputed_bank["passed"] and recomputed_condition["passed"])
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "bank_byte_identity": recomputed_bank,
        "condition_byte_identity": recomputed_condition,
    }


def _prediction_contract_audit(
    *,
    payload: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    scene_id: str,
    gaussian_count: int,
    allowed_classes: Sequence[str],
) -> dict[str, Any]:
    labels = np.asarray(payload.get("point_labels"))
    instances = payload.get("instances")
    negative_metadata = 0
    orphan = 0
    empty = 0
    violation: str | None = None
    if isinstance(instances, Mapping):
        parsed_ids: set[int] = set()
        for raw_id in instances:
            try:
                value = int(raw_id)
            except (TypeError, ValueError):
                continue
            negative_metadata += int(value < 0)
            if value >= 0:
                parsed_ids.add(value)
                if labels.shape == (int(gaussian_count),):
                    empty += int(not np.any(labels == value))
        if labels.shape == (int(gaussian_count),) and labels.dtype.kind in "iu":
            orphan = int(
                np.count_nonzero((labels >= 0) & ~np.isin(labels, sorted(parsed_ids)))
            )
    try:
        if str(payload.get("scene_id")) != scene_id:
            raise ValueError("output scene identity mismatch")
        if str(payload.get("condition")) != CONDITION:
            raise ValueError("mask ablation must use C0-no-prior outputs")
        if labels.shape != (int(gaussian_count),):
            raise ValueError("output point_labels length differs from bank")
        if not isinstance(instances, Mapping):
            raise TypeError("output instances must be a mapping")
        validate_prediction_contract(labels, instances)
        allowed = set(map(str, allowed_classes))
        for metadata in instances.values():
            if str(metadata.get("class")) not in allowed:
                raise ValueError("output contains a class outside the registered SAGA20")
        if str(diagnostics.get("scene_id")) != scene_id:
            raise ValueError("diagnostics scene identity mismatch")
        if str(diagnostics.get("condition")) != CONDITION:
            raise ValueError("diagnostics condition identity mismatch")
    except (TypeError, ValueError, KeyError) as exc:
        violation = str(exc)
    embedded = diagnostics.get("prediction_contract")
    embedded_audit: Mapping[str, Any] = {}
    if isinstance(embedded, Mapping):
        nested = embedded.get("contract")
        embedded_audit = nested if isinstance(nested, Mapping) else embedded
    orphan += int(embedded_audit.get("orphan_gaussian_count", 0))
    negative_metadata += int(embedded_audit.get("negative_metadata_count", 0))
    duplicate = int(embedded_audit.get("duplicate_ownership_count", 0))
    passed = bool(
        violation is None
        and orphan == 0
        and negative_metadata == 0
        and empty == 0
        and duplicate == 0
    )
    return {
        "passed": passed,
        "violation": violation,
        "orphan_gaussian_count": orphan,
        "negative_metadata_count": negative_metadata,
        "empty_instance_count": empty,
        "duplicate_ownership_count": duplicate,
    }


def _candidates_and_predictions(
    *,
    payload: Mapping[str, Any],
    scene_id: str,
    gaussian_count: int,
    nearest: Any,
    class_names: Sequence[str],
) -> tuple[list[CleanCandidate], list[PredictedInstance]]:
    labels = np.asarray(payload["point_labels"], dtype=np.int64)
    instances = payload["instances"]
    class_to_id = {str(value): index for index, value in enumerate(class_names)}
    candidates: list[CleanCandidate] = []
    predictions: list[PredictedInstance] = []
    for raw_id in sorted(instances, key=lambda value: int(value)):
        instance_id = int(raw_id)
        metadata = instances[raw_id]
        gaussian_ids = np.flatnonzero(labels == instance_id).astype(np.int64)
        class_name = str(metadata["class"])
        if class_name not in class_to_id:
            raise ValueError(f"unknown prediction class: {class_name}")
        winner = float(metadata.get("winner_probability", metadata["score"]))
        view = float(metadata.get("view_consensus", 1.0))
        detection = float(metadata.get("detection_ratio", 1.0))
        candidate = CleanCandidate(
            object_id=instance_id,
            gaussian_ids=gaussian_ids,
            class_id=class_name,
            winner_probability=winner,
            view_consensus=view,
            detection_ratio=detection,
        )
        candidates.append(candidate)
        predictions.append(
            PredictedInstance(
                scene_id=scene_id,
                instance_id=instance_id,
                class_id=class_to_id[class_name],
                score=float(metadata["score"]),
                mask=formal_gt_point_mask(gaussian_ids, nearest),
            )
        )
    return candidates, predictions


def _aggregate_gaussian_diagnostics(candidate_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaussian_count = 0
    correct = 0
    unsupported = 0
    for row in candidate_rows:
        radius = row["radii"]["0.05"]
        gaussian_count += int(row["candidate_gaussian_count"])
        correct += int(radius["gaussian_correct_target_instance_count"])
        unsupported += int(radius["gaussian_unsupported_count"])
    return {
        "gaussian_count": gaussian_count,
        "correct_target_count_5cm": correct,
        "unsupported_count_5cm": unsupported,
        "target_precision_5cm": float(correct / gaussian_count)
        if gaussian_count
        else 0.0,
        "unsupported_fraction_5cm": float(unsupported / gaussian_count)
        if gaussian_count
        else 0.0,
    }


def _matching_values(result: Mapping[str, Any], *, subset: str) -> dict[str, Any]:
    node = result["subsets"][subset]
    values: dict[str, Any] = {"candidate_count": int(node["candidate_count"])}
    for match_type in ("geometry", "same_class"):
        for threshold in ("0.25", "0.50"):
            row = node["matching"][match_type][threshold]
            prefix = f"{match_type}_{threshold.replace('.', '')}"
            values[f"{prefix}_tp"] = int(row["true_positive_count"])
            values[f"{prefix}_fp"] = int(row["false_positive_count"])
            values[f"{prefix}_precision"] = float(row["precision"])
            values[f"{prefix}_recall"] = float(row["recall"])
            values[f"{prefix}_tiny_small_recall"] = float(
                row["tiny_small_recall"]
            )
            values[f"{prefix}_matched_iou"] = float(row["total_matched_iou"])
    return values


def _protocol_scalars(protocols: Mapping[str, Any]) -> dict[str, Any]:
    official = protocols["official_9"]["aggregate"]
    historical = protocols["historical_10"]["aggregate"]
    return {
        "official_map_50_90": official["map_50_90"],
        "official_ap25": official["map_0.25"],
        "official_ap50": official["map_0.50"],
        "historical_map_50_95": historical["map_50_95"],
        "historical_ap25": historical["map_0.25"],
        "historical_ap50": historical["map_0.50"],
    }


def _pair_contract(
    *,
    scene_id: str,
    hierarchy: AlphaMaskEvidenceBank,
    flat: AlphaMaskEvidenceBank,
    hierarchy_diagnostics: Mapping[str, Any],
    flat_diagnostics: Mapping[str, Any],
    input_audit: Mapping[str, Any] | None,
    repeat_identity_pass: bool | None,
    full_repeat_audit: Mapping[str, Any] | None,
    flat_spec: Mapping[str, Any],
    hierarchy_gaussian_binding: Mapping[str, Any],
    flat_gaussian_binding: Mapping[str, Any],
) -> dict[str, Any]:
    flat_unique, flat_duplicate_count = _flat_gaussian_unique(flat)
    input_binding = _input_binding_contract(
        scene_id=scene_id,
        input_audit=input_audit,
        hierarchy=hierarchy,
        flat=flat,
    )
    full_repeat = _full_repeat_contract(
        scene_id=scene_id,
        flat_spec=flat_spec,
        repeat_audit=full_repeat_audit,
    )
    source_modes = (
        hierarchy.source.get("mask_observation_mode") == "hierarchy"
        and flat.source.get("mask_observation_mode") == "flat-highest-quality"
    )
    shared_runtime = bool(
        hierarchy.scene_id == flat.scene_id
        and hierarchy.point_count == flat.point_count
        and np.array_equal(hierarchy.xyz_m, flat.xyz_m)
        and hierarchy.class_names == flat.class_names
        and hierarchy.thresholds.to_dict() == flat.thresholds.to_dict()
        and _same_frame_metadata(hierarchy, flat)
        and _same_visibility(hierarchy, flat)
        and _source_without_mask_contract(hierarchy.source)
        == _source_without_mask_contract(flat.source)
        and hierarchy_diagnostics.get("config") == flat_diagnostics.get("config")
    )
    audit_available = input_audit is not None
    union_exact = bool(
        audit_available
        and input_audit.get("union_changed_pixel_count") == 0
        and input_audit.get("mechanical_contract_pass") is True
    )
    pixel_overlap_zero = bool(
        audit_available and input_audit.get("flat_overlap_pixel_count") == 0
    )
    repeat_available = repeat_identity_pass is not None
    checks = {
        "input_audit_available": audit_available,
        "pixel_union_exact": union_exact,
        "flat_pixel_overlap_zero": pixel_overlap_zero,
        "flat_frame_gaussian_unique": flat_unique,
        "source_modes_registered": source_modes,
        "shared_runtime_identity_except_mask_mode": shared_runtime,
        "repeat_identity_check_available": repeat_available,
        "flat_repeat_identity_pass": bool(repeat_identity_pass),
        "input_audit_bound_to_actual_banks": bool(input_binding["passed"]),
        "flat_evidence_and_c0_full_repeat_pass": bool(full_repeat["passed"]),
        "hierarchy_gaussian_asset_bound": bool(
            hierarchy_gaussian_binding.get("passed")
        ),
        "flat_gaussian_asset_bound": bool(flat_gaussian_binding.get("passed")),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "flat_duplicate_frame_gaussian_count": flat_duplicate_count,
        "hierarchy_ambiguous_frame_gaussian_count": int(
            len(hierarchy.frame_ambiguity.ids)
        ),
        "flat_ambiguous_frame_gaussian_count": int(len(flat.frame_ambiguity.ids)),
        "input_binding_contract": input_binding,
        "flat_full_repeat_contract": full_repeat,
        "gaussian_asset_binding": {
            "H-hierarchy": dict(hierarchy_gaussian_binding),
            "P-flat": dict(flat_gaussian_binding),
        },
    }


def _input_audit_for_scene(
    *, base: Path, scene: Mapping[str, Any], flat_spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    value = scene.get(
        "flat_mask_input_audit",
        scene.get("mask_control_input_audit", flat_spec.get("input_audit")),
    )
    if value is None:
        return None
    return _json_value(base, value, name="scene.flat_mask_input_audit")


def _full_repeat_audit_for_scene(
    *, base: Path, scene: Mapping[str, Any]
) -> dict[str, Any] | None:
    value = scene.get("flat_full_repeat_audit")
    if value is None:
        return None
    return _json_value(base, value, name="scene.flat_full_repeat_audit")


def _repeat_identity_for_scene(
    scene: Mapping[str, Any], input_audit: Mapping[str, Any] | None
) -> bool | None:
    value = scene.get("flat_repeat_identity_pass")
    if value is None and input_audit is not None:
        value = input_audit.get(
            "flat_repeat_identity_pass",
            input_audit.get("deterministic_replay_pass"),
        )
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("flat_repeat_identity_pass must be boolean")
    return value


def _arm_row(
    *,
    scene_id: str,
    arm: str,
    metrics: Mapping[str, Any],
    protocols: Mapping[str, Any],
    contract: Mapping[str, Any],
    bank: AlphaMaskEvidenceBank,
) -> dict[str, Any]:
    all_values = _matching_values(metrics, subset="all")
    official_values = _matching_values(metrics, subset="official_evaluable")
    gaussian = _aggregate_gaussian_diagnostics(metrics["candidate_rows"])
    coverage = metrics["gt_to_gaussian_scene_coverage"]["0.05"]
    row: dict[str, Any] = {
        "schema": MASK_ABLATION_ROW_SCHEMA,
        "scope": "scene",
        "scene_id": scene_id,
        "arm": arm,
        "condition": CONDITION,
        "bank_mask_count": bank.mask_count,
        "bank_ambiguous_frame_gaussian_count": int(len(bank.frame_ambiguity.ids)),
        "official_gt_count": int(metrics["official_gt_count"]),
        "official_tiny_small_gt_count": int(metrics["official_tiny_small_gt_count"]),
        "gt_to_gaussian_mapped_fraction_5cm": float(coverage["mapped_fraction"]),
        "output_contract_pass": bool(contract["passed"]),
        "orphan_gaussian_count": int(contract["orphan_gaussian_count"]),
        "negative_metadata_count": int(contract["negative_metadata_count"]),
        "duplicate_ownership_count": int(contract["duplicate_ownership_count"]),
        **{f"all_{key}": value for key, value in all_values.items()},
        **{f"official_{key}": value for key, value in official_values.items()},
        **gaussian,
        **_protocol_scalars(protocols),
    }
    return row


def _aggregate_rows(rows: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == arm and row["scope"] == "scene"]
    if not selected:
        raise ValueError(f"no scene rows for {arm}")
    summed = {
        key: int(sum(int(row[key]) for row in selected))
        for key in (
            "all_candidate_count",
            "all_geometry_025_tp",
            "all_geometry_025_fp",
            "all_geometry_050_tp",
            "all_geometry_050_fp",
            "official_candidate_count",
            "official_geometry_025_tp",
            "official_geometry_025_fp",
            "official_geometry_050_tp",
            "official_geometry_050_fp",
            "official_same_class_025_tp",
            "official_same_class_050_tp",
            "official_gt_count",
            "official_tiny_small_gt_count",
        )
    }
    tiny_matched_025 = sum(
        float(row["official_geometry_025_tiny_small_recall"])
        * int(row["official_tiny_small_gt_count"])
        for row in selected
    )
    tiny_matched_050 = sum(
        float(row["official_geometry_050_tiny_small_recall"])
        * int(row["official_tiny_small_gt_count"])
        for row in selected
    )
    denominator = summed["official_tiny_small_gt_count"]
    candidates = summed["official_candidate_count"]
    return {
        "scope": "aggregate",
        "scene_id": None,
        "arm": arm,
        "condition": CONDITION,
        **summed,
        "official_geometry_025_precision": float(
            summed["official_geometry_025_tp"] / candidates
        )
        if candidates
        else 0.0,
        "official_geometry_050_precision": float(
            summed["official_geometry_050_tp"] / candidates
        )
        if candidates
        else 0.0,
        "official_geometry_025_recall": float(
            summed["official_geometry_025_tp"] / summed["official_gt_count"]
        )
        if summed["official_gt_count"]
        else 0.0,
        "official_geometry_050_recall": float(
            summed["official_geometry_050_tp"] / summed["official_gt_count"]
        )
        if summed["official_gt_count"]
        else 0.0,
        "official_geometry_025_tiny_small_recall": float(tiny_matched_025 / denominator)
        if denominator
        else 0.0,
        "official_geometry_050_tiny_small_recall": float(tiny_matched_050 / denominator)
        if denominator
        else 0.0,
    }


def _scientific_gate(
    *, scene_rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    hierarchy = aggregate["H-hierarchy"]
    flat = aggregate["P-flat"]
    h_count = int(hierarchy["all_candidate_count"])
    p_count = int(flat["all_candidate_count"])
    count_ratio = (
        float(p_count / h_count)
        if h_count
        else (0.0 if p_count == 0 else None)
    )
    scene_checks: list[dict[str, Any]] = []
    for scene_id in sorted({str(row["scene_id"]) for row in scene_rows}):
        by_arm = {
            str(row["arm"]): row
            for row in scene_rows
            if str(row["scene_id"]) == scene_id
        }
        hierarchy_row = by_arm["H-hierarchy"]
        flat_row = by_arm["P-flat"]
        recall_delta = float(flat_row["all_geometry_025_recall"]) - float(
            hierarchy_row["all_geometry_025_recall"]
        )
        match_delta = int(flat_row["all_geometry_050_tp"]) - int(
            hierarchy_row["all_geometry_050_tp"]
        )
        scene_checks.append(
            {
                "scene_id": scene_id,
                "geometry_iou050_match_delta": match_delta,
                "geometry_recall025_delta": recall_delta,
                "improved": bool(match_delta > 0 or recall_delta > 1e-12),
                "recall025_drop_within_0.05": recall_delta >= -0.05 - 1e-12,
            }
        )
    checks = {
        "adds_at_least_2_iou050_matches": int(flat["all_geometry_050_tp"])
        - int(hierarchy["all_geometry_050_tp"])
        >= 2,
        "flat_iou050_matches_at_least_6": int(flat["all_geometry_050_tp"]) >= 6,
        "flat_official_precision025_at_least_0.10": float(
            flat["official_geometry_025_precision"]
        )
        >= 0.10,
        "flat_tiny_small_recall025_at_least_0.20": float(
            flat["official_geometry_025_tiny_small_recall"]
        )
        >= 0.20,
        "flat_candidate_count_at_most_1.5x": (
            p_count == 0 if h_count == 0 else bool(count_ratio <= 1.5)
        ),
        "at_least_one_scene_improved": any(row["improved"] for row in scene_checks),
        "no_scene_recall025_drop_over_0.05": all(
            row["recall025_drop_within_0.05"] for row in scene_checks
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_count_ratio_flat_over_hierarchy": count_ratio,
        "scene_deltas": scene_checks,
    }


def evaluate_mask_contract_ablation_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    expected_scene_count: int = DEFAULT_SCENE_COUNT,
) -> dict[str, Any]:
    """Evaluate the registered two-scene H'/P control from frozen artifacts."""

    manifest_file = Path(manifest_path).resolve()
    manifest = load_json(manifest_file)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != int(expected_scene_count):
        raise ValueError(
            f"manifest must contain exactly {int(expected_scene_count)} scenes"
        )
    if int(expected_scene_count) <= 0:
        raise ValueError("expected_scene_count must be positive")
    scene_ids = tuple(str(scene.get("scene_id", "")) for scene in scenes)
    if (
        int(expected_scene_count) == DEFAULT_SCENE_COUNT
        and scene_ids != REGISTERED_DEV2_SCENE_IDS
    ):
        raise ValueError("mask ablation must use the exact registered DEV2 order")
    if tuple(map(str, manifest.get("dev2_scene_ids", ()))) != scene_ids:
        raise ValueError("mask ablation manifest dev2_scene_ids differs from scenes")
    base = manifest_file.parent
    class_names, allowed_classes = _taxonomy(manifest)
    _, small_max_m, _ = _size_boundaries(manifest)
    if int(manifest.get("min_region_size", -1)) != 100:
        raise ValueError("manifest min_region_size must explicitly equal 100")
    min_region_size = 100

    rows: list[dict[str, Any]] = []
    scene_analysis: dict[str, Any] = {}
    pair_contracts: list[dict[str, Any]] = []
    predictions_by_arm: dict[str, list[PredictedInstance]] = {
        arm: [] for arm in MASK_ARMS
    }
    gt_scenes: list[GroundTruthScene] = []

    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise TypeError("manifest scene rows must be mappings")
        scene_id = str(scene.get("scene_id", ""))
        if not scene_id or scene_id in scene_analysis:
            raise ValueError("scene IDs must be non-empty and unique")
        specs = _condition_specs(scene)
        gt_xyz, gt_scene = load_ground_truth_npz(
            _resolve(base, scene.get("gt_npz"), name="scene.gt_npz"), scene_id
        )
        gaussian_path = _resolve(
            base, scene.get("gaussian_ply"), name="scene.gaussian_ply"
        )
        gaussian_raw = load_ply_xyz(gaussian_path)
        gaussian_xyz = apply_transform(
            gaussian_raw, _transform(scene.get("transform"))
        )
        nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
        tiny_small = _official_tiny_small_ids(
            gt_xyz=gt_xyz,
            semantic=gt_scene.semantic,
            instance=gt_scene.instance,
            class_count=len(class_names),
            small_max_m=small_max_m,
            min_region_size=min_region_size,
        )
        gt_objects = ground_truth_objects_from_arrays(
            gt_scene.semantic,
            gt_scene.instance,
            class_names=class_names,
            min_region_size=min_region_size,
            tiny_small_instance_ids=tiny_small,
        )
        gt_scenes.append(gt_scene)

        loaded: dict[str, tuple[AlphaMaskEvidenceBank, dict[str, Any], dict[str, Any], Path]] = {}
        for arm in MASK_ARMS:
            loaded[arm] = _load_arm(
                base=base,
                scene_id=scene_id,
                arm=arm,
                spec=specs[arm],
            )
        hierarchy_bank, hierarchy_output, hierarchy_diagnostics, _ = loaded[
            "H-hierarchy"
        ]
        flat_bank, flat_output, flat_diagnostics, _ = loaded["P-flat"]
        if hierarchy_bank.point_count != len(gaussian_xyz) or flat_bank.point_count != len(
            gaussian_xyz
        ):
            raise ValueError("evidence bank and registered Gaussian PLY differ in point count")
        input_audit = _input_audit_for_scene(
            base=base, scene=scene, flat_spec=specs["P-flat"]
        )
        repeat_identity = _repeat_identity_for_scene(scene, input_audit)
        full_repeat_audit = _full_repeat_audit_for_scene(base=base, scene=scene)
        pair_contract = _pair_contract(
            scene_id=scene_id,
            hierarchy=hierarchy_bank,
            flat=flat_bank,
            hierarchy_diagnostics=hierarchy_diagnostics,
            flat_diagnostics=flat_diagnostics,
            input_audit=input_audit,
            repeat_identity_pass=repeat_identity,
            full_repeat_audit=full_repeat_audit,
            flat_spec=specs["P-flat"],
            hierarchy_gaussian_binding=_gaussian_binding_contract(
                bank=hierarchy_bank,
                gaussian_path=gaussian_path,
                gaussian_raw=gaussian_raw,
            ),
            flat_gaussian_binding=_gaussian_binding_contract(
                bank=flat_bank,
                gaussian_path=gaussian_path,
                gaussian_raw=gaussian_raw,
            ),
        )
        pair_contracts.append({"scene_id": scene_id, **pair_contract})
        scene_analysis[scene_id] = {"pair_contract": pair_contract, "arms": {}}

        for arm in MASK_ARMS:
            bank, output, diagnostics, _ = loaded[arm]
            output_contract = _prediction_contract_audit(
                payload=output,
                diagnostics=diagnostics,
                scene_id=scene_id,
                gaussian_count=bank.point_count,
                allowed_classes=allowed_classes,
            )
            candidates, predictions = _candidates_and_predictions(
                payload=output,
                scene_id=scene_id,
                gaussian_count=bank.point_count,
                nearest=nearest,
                class_names=class_names,
            )
            predictions_by_arm[arm].extend(predictions)
            metrics = evaluate_candidate_set_three_spaces(
                candidates=candidates,
                gt_objects=gt_objects,
                nearest=nearest,
                min_region_size=min_region_size,
            )
            scene_protocols = evaluate_dual_protocols(
                [gt_scene], predictions, class_names, min_region_size=min_region_size
            )
            rows.append(
                _arm_row(
                    scene_id=scene_id,
                    arm=arm,
                    metrics=metrics,
                    protocols=scene_protocols,
                    contract=output_contract,
                    bank=bank,
                )
            )
            scene_analysis[scene_id]["arms"][arm] = {
                "three_spaces": metrics,
                "dual_protocols": scene_protocols,
                "output_contract": output_contract,
            }

    scene_rows = tuple(rows)
    combined_protocols = {
        arm: evaluate_dual_protocols(
            gt_scenes,
            predictions_by_arm[arm],
            class_names,
            min_region_size=min_region_size,
        )
        for arm in MASK_ARMS
    }
    aggregate = {arm: _aggregate_rows(rows, arm=arm) for arm in MASK_ARMS}
    for arm in MASK_ARMS:
        aggregate[arm].update(_protocol_scalars(combined_protocols[arm]))
        aggregate[arm]["schema"] = MASK_ABLATION_ROW_SCHEMA
        rows.append(dict(aggregate[arm]))
    output_contract_pass = all(
        bool(scene_analysis[scene_id]["arms"][arm]["output_contract"]["passed"])
        for scene_id in scene_analysis
        for arm in MASK_ARMS
    )
    mechanical = {
        "pair_contracts": pair_contracts,
        "output_contract_pass": output_contract_pass,
    }
    mechanical["passed"] = bool(
        output_contract_pass and all(bool(row["passed"]) for row in pair_contracts)
    )
    science = _scientific_gate(scene_rows=scene_rows, aggregate=aggregate)
    science["eligible"] = bool(mechanical["passed"])
    science["passed"] = bool(science["passed"] and science["eligible"])
    gt_parity = evaluate_gt_as_prediction_dual_protocols(
        gt_scenes, class_names, min_region_size=min_region_size
    )

    if not mechanical["passed"]:
        conclusion = "mechanical-contract-failed"
    elif science["passed"]:
        conclusion = "flat-mask-contract-passed"
    else:
        ambiguity_reduced = all(
            row["flat_ambiguous_frame_gaussian_count"]
            < row["hierarchy_ambiguous_frame_gaussian_count"]
            for row in pair_contracts
        )
        conclusion = (
            "ambiguity-reduced-without-registered-geometric-gain"
            if ambiguity_reduced
            else "flat-mask-contract-did-not-pass"
        )

    analysis = {
        "schema": MASK_ABLATION_SCHEMA,
        "manifest_schema": MANIFEST_SCHEMA,
        "scene_ids": list(scene_analysis),
        "arms": list(MASK_ARMS),
        "condition": CONDITION,
        "evaluation_only": True,
        "gt_used_by_runtime": False,
        "three_metric_spaces_kept_separate": True,
        "synthetic_false_positive_sentinels": False,
        "scene_analysis": scene_analysis,
        "aggregate": aggregate,
        "combined_dual_protocols": combined_protocols,
        "gt_as_prediction": gt_parity,
        "mechanical_gate": mechanical,
        "scientific_gate": science,
        "conclusion": conclusion,
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "mask_contract_ablation_dev2.parquet", rows)
    write_json(destination / "mask_contract_ablation_dev2.json", analysis)
    return {
        "rows": rows,
        "analysis": analysis,
        "output_dir": str(destination),
    }


__all__ = [
    "CONDITION",
    "DEFAULT_SCENE_COUNT",
    "MASK_ABLATION_ROW_SCHEMA",
    "MASK_ABLATION_SCHEMA",
    "MASK_ARMS",
    "evaluate_mask_contract_ablation_manifest",
]
