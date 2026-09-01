from __future__ import annotations

"""Strictly materialize the clean-baseline cloud registration.

This module is intentionally a one-time registration tool, not another
runtime manifest layer.  It resolves the already existing tune/final assets,
checks the preregistered physical-scene split, derives the official-valid
tiny/small GT instance IDs, and writes one experiment config plus one evidence
request per scene.  It does not download data or create lock files.  The only
large-file hashes it calculates are for explicitly imported evidence banks,
whose old producer bytes must be proved unchanged before reuse.
"""

import argparse
import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from ..io import load_json, read_rows, sha256_file, write_json
from ..runner import load_scene_runtime_manifest
from ..scannet import physical_scene_id
from ..taxonomy import load_taxonomy
from .evaluation import gt_point_to_gaussian_mapping
from .experiment import (
    CONFIG_KIND,
    DEV2,
    DEV8,
    EVIDENCE_IMPORT_SCHEMA,
    is_evaluation_only_runtime_field,
)
from .evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    evidence_bank_is_complete,
    evidence_request_source,
    load_evidence_bank,
)
from .identity_control import (
    IDENTITY_CONTROL_REGISTRATION_SCHEMA,
    IDENTITY_CONTROL_SCHEMA,
    IDENTITY_TRAIN_SCENES,
    IDENTITY_VALIDATION_SCENE,
    load_affinity_feature_ply,
    load_gaussian_attributes_ply,
)
from .size_prior import SizePriorTable
from .sam_inputs import SAM_EVERYTHING_CONFIG
from .validation import HOLDOUT5, validate_final48_scene_ids
from .worker import DEFAULT_CLASSES, resolve_clean_scene_inputs


REGISTRATION_SCHEMA = "saga-clean-baseline-registration-v1"
REQUEST_SCHEMA = "saga-clean-alpha-mask-evidence-request-v1"
EVIDENCE_IMPORT_MANIFEST_SCHEMA = "saga-clean-evidence-import-manifest-v1"
IDENTITY_4X4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _resolve_from(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_evidence_imports(
    path: Path | None,
    *,
    allow_legacy_hierarchy_mode_missing: bool = False,
) -> dict[str, dict[str, Any]]:
    """Load and byte-validate explicitly imported evidence producer outputs."""

    if path is None:
        return {}
    source = Path(path).resolve()
    payload = load_json(source)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != EVIDENCE_IMPORT_MANIFEST_SCHEMA
    ):
        raise ValueError(
            "evidence imports must use the registered import-manifest schema"
        )
    rows = payload.get("scenes")
    if not isinstance(rows, Mapping):
        raise TypeError("evidence import manifest.scenes must be an object")
    base = source.parent
    result: dict[str, dict[str, Any]] = {}
    expected_files = {
        EVIDENCE_ARRAY_FILE,
        EVIDENCE_METADATA_FILE,
        EVIDENCE_DIAGNOSTICS_FILE,
    }
    for raw_scene_id, raw in rows.items():
        scene_id = str(raw_scene_id)
        if not scene_id or not isinstance(raw, Mapping):
            raise TypeError("each evidence import row must be a named object")
        bank_dir = _resolve_from(base, raw["bank_dir"])
        source_request = _resolve_from(base, raw["source_request"])
        producer = str(raw.get("producer_commit", "")).strip()
        files_raw = raw.get("files")
        if len(producer) != 40 or any(
            character not in "0123456789abcdef" for character in producer
        ):
            raise ValueError(
                f"{scene_id}: imported producer_commit must be a full lowercase commit"
            )
        if not isinstance(files_raw, Mapping) or set(map(str, files_raw)) != expected_files:
            raise ValueError(
                f"{scene_id}: import must register exactly the three evidence files"
            )
        files = {str(key): str(value) for key, value in files_raw.items()}
        for name, expected_digest in files.items():
            if len(expected_digest) != 64 or any(
                character not in "0123456789abcdef" for character in expected_digest
            ):
                raise ValueError(f"{scene_id}: invalid SHA-256 for imported {name}")
            candidate = bank_dir / name
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            if sha256_file(candidate) != expected_digest:
                raise ValueError(
                    f"{scene_id}: imported evidence byte identity changed for {name}"
                )
        request = load_json(source_request)
        if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
            raise ValueError(f"{scene_id}: imported evidence request has the wrong schema")
        if request.get("producer_commit") != producer:
            raise ValueError(f"{scene_id}: import/request producer commits differ")
        expected_source = evidence_request_source(scene_id=scene_id, request=request)
        complete = evidence_bank_is_complete(
            bank_dir,
            expected_scene_id=scene_id,
            expected_source=expected_source,
        )
        legacy_mode_proof = False
        if not complete and allow_legacy_hierarchy_mode_missing:
            bank = load_evidence_bank(bank_dir, expected_scene_id=scene_id)
            legacy_expected = dict(expected_source)
            requested_mode = legacy_expected.pop("mask_observation_mode", None)
            actual_source = dict(bank.source)
            legacy_mode_proof = bool(
                requested_mode == "hierarchy"
                and "mask_observation_mode" not in actual_source
                and actual_source == legacy_expected
            )
            complete = legacy_mode_proof
        if not complete:
            raise ValueError(f"{scene_id}: imported evidence bank is incomplete")
        result[scene_id] = {
            "bank_dir": bank_dir,
            "source_request": source_request,
            "producer_commit": producer,
            "files": files,
            "request": dict(request),
            "legacy_hierarchy_mode_proof": legacy_mode_proof,
        }
    return result


def _scene_ids_from_spec(path: Path) -> tuple[str, ...]:
    payload = load_json(path)
    rows: Any = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        result = tuple(str(value) for value in rows)
    elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        result = tuple(
            str(row.get("scene_id", row.get("scan_id")))
            if isinstance(row, Mapping)
            else str(row)
            for row in rows
        )
    else:
        raise TypeError(f"{path}: scene specification must contain a list or object")
    if any(not value or value == "None" for value in result):
        raise ValueError(f"{path}: a scene row lacks scene_id")
    return result


def _locked_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = load_json(path)
    rows: Any = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("locked evaluation scenes must contain a row sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = raw if isinstance(raw, Mapping) else {"scene_id": str(raw)}
        scene_id = str(row.get("scene_id", row.get("scan_id", ""))).strip()
        if not scene_id or scene_id in result:
            raise ValueError("locked evaluation scene IDs must be unique and non-empty")
        registered_physical = row.get("physical_scene_id")
        if registered_physical not in (None, "") and str(registered_physical) != physical_scene_id(scene_id):
            raise ValueError(f"{scene_id}: locked physical_scene_id is inconsistent")
        result[scene_id] = row
    return result


def _load_train_scene_ids(path: Path) -> tuple[str, ...]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows: Any = read_rows(path)
        else:
            payload = load_json(path)
            rows = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
        if isinstance(rows, Mapping):
            values: Sequence[Any] = tuple(rows)
        elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            values = rows
        else:
            raise TypeError("train scene list must contain a sequence or mapping")
        result = tuple(
            str(value.get("scene_id", value.get("scan_id", "")))
            if isinstance(value, Mapping)
            else str(value)
            for value in values
        )
    else:
        result = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not result or any(not value for value in result):
        raise ValueError("train scene list must contain non-empty scene IDs")
    return result


def _validate_split(
    *, tune24: Sequence[str], final48: Sequence[str], train: Sequence[str]
) -> dict[str, Any]:
    tune = tuple(map(str, tune24))
    if len(tune) != 24 or len(set(tune)) != 24:
        raise ValueError("tune runtime must contain exactly 24 unique scans")
    tune_physical = {physical_scene_id(value) for value in tune}
    if len(tune_physical) != 13:
        raise ValueError("tune runtime must contain exactly 13 physical scenes")
    if not set(DEV8 + HOLDOUT5).issubset(tune):
        raise ValueError("tune runtime does not contain the frozen DEV8 and holdout5 scans")
    final = validate_final48_scene_ids(final48)
    train_physical = {physical_scene_id(value) for value in train}
    dev_physical = {physical_scene_id(value) for value in DEV8}
    holdout_physical = {physical_scene_id(value) for value in HOLDOUT5}
    final_physical = {physical_scene_id(value) for value in final}
    groups = {
        "train": train_physical,
        "dev8": dev_physical,
        "holdout5": holdout_physical,
        "final48": final_physical,
    }
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = groups[left].intersection(groups[right])
            if overlap:
                raise ValueError(f"physical-scene split overlap {left}/{right}: {sorted(overlap)}")
    if tune_physical != dev_physical.union(holdout_physical):
        raise ValueError("tune24 physical scenes must equal DEV8 plus holdout5")
    return {
        "tune_scan_count": len(tune),
        "tune_physical_scene_count": len(tune_physical),
        "train_physical_scene_count": len(train_physical),
        "dev8_physical_scene_count": len(dev_physical),
        "holdout5_physical_scene_count": len(holdout_physical),
        "final48_physical_scene_count": len(final_physical),
        "physical_split_overlap": False,
    }


def _resolve_path(base: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _root_scene_path(root: Path | None, template: str, scene_id: str) -> Path | None:
    if root is None:
        return None
    rendered = template.format(root=str(root.resolve()), scene_id=scene_id)
    return Path(rendered).resolve()


def _runtime_with_overrides(
    scene_id: str,
    raw: Mapping[str, Any],
    *,
    sam_root: Path | None,
    sam_template: str,
    grounded_masks_root: Path | None,
    grounded_labels_root: Path | None,
    grounded_template: str,
) -> dict[str, Any]:
    scene = dict(raw)
    scene["scene_id"] = scene_id
    # The clean worker consumes ``point_cloud_path``.  Historical manifests
    # sometimes registered the same explicit asset as ``gaussian_ply``;
    # normalize that spelling in the emitted request rather than silently
    # falling back to another file under the scene directory.
    if scene.get("gaussian_ply") not in (None, "") and scene.get(
        "point_cloud_path"
    ) in (None, ""):
        scene["point_cloud_path"] = scene["gaussian_ply"]
    sam = _root_scene_path(sam_root, sam_template, scene_id)
    masks = _root_scene_path(grounded_masks_root, grounded_template, scene_id)
    labels = _root_scene_path(grounded_labels_root, grounded_template, scene_id)
    if sam is not None:
        scene["segment_everything_root"] = str(sam)
    if masks is not None:
        scene["grounded_masks_path"] = str(masks)
    if labels is not None:
        scene["grounded_labels_path"] = str(labels)
    return scene


def _formal_runtime_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove GT/evaluation-only fields before a worker request is written."""

    def sanitized(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): sanitized(child)
                for key, child in item.items()
                if not is_evaluation_only_runtime_field(key)
            }
        if isinstance(item, list):
            return [sanitized(child) for child in item]
        if isinstance(item, tuple):
            return [sanitized(child) for child in item]
        return item

    result = sanitized(value)
    if not isinstance(result, dict):  # pragma: no cover - input contract
        raise TypeError("formal runtime registration must remain an object")
    return result


def _validate_mask_roots(scene_id: str, *, sam: Path, masks: Path, labels: Path) -> None:
    sam_frames = {path.stem for path in sam.glob("*.npz") if path.is_file()}
    mask_frames = {path.name for path in masks.iterdir() if path.is_file()}
    label_frames = {path.name for path in labels.iterdir() if path.is_file()}
    if not sam_frames:
        raise ValueError(f"{scene_id}: SAM-everything root contains no packed .npz frames")
    # Both empty is a valid all-abstain scene.  One-sided evidence is a data
    # error, exactly as in the runtime worker.
    if bool(mask_frames) != bool(label_frames):
        raise ValueError(f"{scene_id}: Grounded-SAM mask/label roots are one-sided")
    if mask_frames and not mask_frames.intersection(label_frames):
        raise ValueError(f"{scene_id}: Grounded-SAM mask/label roots share no frame files")


def _transform(scene_id: str, scene: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    if "gaussian_to_gt_transform" not in scene:
        raise ValueError(f"{scene_id}: gaussian_to_gt_transform must be explicit")
    matrix = np.asarray(scene["gaussian_to_gt_transform"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{scene_id}: gaussian_to_gt_transform must be finite 4x4")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError(f"{scene_id}: gaussian_to_gt_transform is singular")
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _prove_30k_ply(scene_id: str, scene: Mapping[str, Any], ply: Path) -> None:
    registered = scene.get("gaussian_iteration", scene.get("iteration"))
    path_proves = "iteration_30000" in {part.lower() for part in ply.parts}
    field_proves = registered is not None and int(registered) == 30000
    if not path_proves and not field_proves:
        raise ValueError(f"{scene_id}: cannot prove that {ply} is the 30k Gaussian PLY")


def _identity_feature_asset(scene: Mapping[str, Any]) -> tuple[Path, str]:
    base = Path(str(scene["base_path"])).resolve()
    for key in (
        "contrastive_feature_point_cloud_path",
        "feature_point_cloud_path",
        "feature_ply_path",
    ):
        value = scene.get(key)
        if value not in (None, ""):
            path = Path(str(value))
            return (
                (path if path.is_absolute() else base / path).resolve(),
                f"runtime:{key}",
            )
    return (
        (base / "saga/contrastive_feature_point_cloud.ply").resolve(),
        "runtime:base_path/native-2k",
    )


def _identity_control_registration(
    *,
    runtime: Mapping[str, Mapping[str, Any]],
    scenes: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Register existing identity-control inputs without producing assets."""

    scene_ids = (*IDENTITY_TRAIN_SCENES, IDENTITY_VALIDATION_SCENE)
    assets: dict[str, dict[str, str]] = {}
    registration_assets: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    for scene_id in scene_ids:
        raw = runtime.get(scene_id)
        scene = scenes.get(scene_id)
        if not isinstance(raw, Mapping) or not isinstance(scene, Mapping):
            issues.append(f"{scene_id}: runtime/scene registration is missing")
            continue
        feature, source = _identity_feature_asset(raw)
        gaussian = Path(str(scene["gaussian_ply"])).resolve()
        registration_assets[scene_id] = {
            "feature_ply": str(feature),
            "gaussian_ply": str(gaussian),
            "feature_resolution": source,
        }
        iteration = next(
            (
                raw[key]
                for key in (
                    "affinity_feature_iterations",
                    "feature_iterations",
                    "feature_iteration",
                )
                if raw.get(key) not in (None, "")
            ),
            None,
        )
        if iteration is not None:
            try:
                registered_iteration = int(iteration)
            except (TypeError, ValueError, OverflowError):
                issues.append(
                    f"{scene_id}: affinity feature iteration is not an integer: {iteration}"
                )
            else:
                if registered_iteration != 2000:
                    issues.append(
                        f"{scene_id}: registered affinity feature iteration is {iteration}, not 2000"
                    )
        lowered = str(feature).lower()
        if "10k" in lowered or "10000" in lowered:
            issues.append(f"{scene_id}: feature path is not the native 2k asset: {feature}")
        if not feature.is_file():
            issues.append(f"{scene_id}: native 2k affinity PLY is missing: {feature}")
        if not gaussian.is_file():
            issues.append(f"{scene_id}: 30k Gaussian PLY is missing: {gaussian}")
        if not feature.is_file() or not gaussian.is_file():
            continue
        try:
            feature_xyz, affinity = load_affinity_feature_ply(feature)
            gaussian_xyz, scale, opacity = load_gaussian_attributes_ply(gaussian)
            if len(feature_xyz) != len(gaussian_xyz):
                raise ValueError("feature/Gaussian point counts differ")
            if not np.allclose(
                feature_xyz, gaussian_xyz, rtol=1e-5, atol=1e-5
            ):
                raise ValueError("feature/Gaussian point order or XYZ differs")
            if affinity.shape != (len(feature_xyz), 32):
                raise ValueError("affinity feature dimension is not 32")
            if scale.shape != (len(feature_xyz), 3) or opacity.shape != (
                len(feature_xyz),
            ):
                raise ValueError("Gaussian scale/opacity arrays are incomplete")
        except (ImportError, OSError, TypeError, ValueError) as exc:
            issues.append(f"{scene_id}: identity asset validation failed: {exc}")
            continue
        assets[scene_id] = {
            "feature_ply": str(feature),
            "gaussian_ply": str(gaussian),
        }
    status = "available" if not issues and len(assets) == len(scene_ids) else "unavailable"
    registration = {
        "schema": IDENTITY_CONTROL_REGISTRATION_SCHEMA,
        "status": status,
        "train_scene_ids": list(IDENTITY_TRAIN_SCENES),
        "validation_scene_id": IDENTITY_VALIDATION_SCENE,
        "assets": registration_assets,
        "issues": issues,
        "download_attempted": False,
        "training_attempted": False,
    }
    if status != "available":
        return registration, None
    control = {
        "schema": IDENTITY_CONTROL_SCHEMA,
        "train_scene_ids": list(IDENTITY_TRAIN_SCENES),
        "validation_scene_id": IDENTITY_VALIDATION_SCENE,
        "assets": assets,
        "seed": 42,
        "physical_neighbors": 24,
        "max_edge_distance_m": 0.10,
        "max_training_edges_per_class": 200_000,
        "l2_c": 1.0,
        "probability_threshold": 0.50,
        "min_component_points": 4,
    }
    return registration, control


def _gt_path(
    scene_id: str,
    *,
    root: Path,
    locked_row: Mapping[str, Any] | None,
    locked_spec_dir: Path,
) -> Path:
    if locked_row is not None:
        for key in ("gt_npz", "replacement_gt_npz", "replacement_gt", "gt_path"):
            if locked_row.get(key) not in (None, ""):
                return _resolve_path(locked_spec_dir, locked_row[key])
    return (root.resolve() / f"{scene_id}.npz").resolve()


def _bbox_diagonal_m(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if not len(xyz):
        return 0.0
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    if len(xyz) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(np.ptp(centered, axis=0)))


def _tiny_small_instance_ids(
    *,
    scene_id: str,
    gt_npz: Path,
    gaussian_ply: Path,
    transform: Sequence[Sequence[float]],
    size_bins: Mapping[str, Any],
    evaluation_class_count: int,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    gt_xyz, gt = load_ground_truth_npz(gt_npz, scene_id)
    gaussian_xyz = apply_transform(load_ply_xyz(gaussian_ply), transform)
    mapping, diagnostics = gt_point_to_gaussian_mapping(
        gt_xyz, gaussian_xyz, radius_m=radius_m
    )
    boundaries = size_bins.get("boundaries_m", size_bins)
    small_max = float(boundaries["small_max_m"])
    if not math.isfinite(small_max) or small_max <= 0:
        raise ValueError("size-bin small_max_m must be finite and positive")
    selected: list[int] = []
    official_count = 0
    for instance_id in np.unique(gt.instance[gt.instance >= 0]):
        mask = gt.instance == instance_id
        values, counts = np.unique(gt.semantic[mask], return_counts=True)
        valid = (values >= 0) & (values < int(evaluation_class_count))
        if not np.any(valid):
            continue
        values, counts = values[valid], counts[valid]
        class_id = int(values[counts == counts.max()].min())
        object_points = np.flatnonzero(mask & (gt.semantic == class_id))
        mapped_points = object_points[mapping[object_points] >= 0]
        if len(mapped_points) < int(min_region_size):
            continue
        official_count += 1
        if _bbox_diagonal_m(gt_xyz[object_points]) <= small_max:
            selected.append(int(instance_id))
    return tuple(sorted(selected)), {
        "official_mapped_instance_count": official_count,
        "tiny_small_official_mapped_instance_count": len(selected),
        "gt_point_mapping": diagnostics,
    }


def _metric_pair(node: Mapping[str, Any]) -> tuple[float, float] | None:
    map_keys = ("map_50_95", "mAP", "map")
    ap50_keys = ("map_0.50", "ap50", "AP50", "ap_0.50")
    left = next((node[key] for key in map_keys if key in node), None)
    right = next((node[key] for key in ap50_keys if key in node), None)
    if left is None or right is None:
        return None
    result = float(left), float(right)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in result):
        raise ValueError("B1-fixed metrics must be finite values in [0,1]")
    return result


def _find_metric_pairs(value: Any) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    if isinstance(value, Mapping):
        pair = _metric_pair(value)
        if pair is not None:
            result.append(pair)
        for child in value.values():
            result.extend(_find_metric_pairs(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            result.extend(_find_metric_pairs(child))
    return result


def _load_b1_metrics(path: Path, condition: str | None) -> tuple[float, float]:
    if path.suffix.lower() == ".json":
        payload: Any = load_json(path)
        if condition and isinstance(payload, Mapping):
            for key in ("conditions", "metrics", "by_condition"):
                table = payload.get(key)
                if isinstance(table, Mapping) and condition in table:
                    payload = table[condition]
                    break
        pairs = _find_metric_pairs(payload)
    else:
        rows = read_rows(path)
        if condition:
            rows = [row for row in rows if str(row.get("condition", "")) == condition]
        pairs = [pair for row in rows if (pair := _metric_pair(row)) is not None]
    unique = sorted(set(pairs))
    if len(unique) != 1:
        raise ValueError(
            f"B1-fixed metrics must resolve to exactly one metric pair, found {len(unique)}; "
            "use --b1-condition when the artifact contains several conditions"
        )
    return unique[0]


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_identity_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = load_json(path)
        if existing != dict(payload):
            raise ValueError(f"refusing to overwrite a different registered artifact: {path}")
        return
    write_json(path, payload)


def materialize_clean_baseline_config(
    *,
    tune_runtime_manifest: Path,
    final_runtime_manifest: Path,
    locked_evaluation_scenes: Path,
    tune_gt_root: Path,
    final_gt_root: Path,
    train_scene_list: Path,
    category_priors: Path,
    size_bins: Path,
    taxonomy: Path,
    b1_fixed_metrics: Path,
    repo_root: Path,
    run_root: Path,
    artifact_root: Path,
    output_dir: Path,
    code_commit: str,
    evidence_imports: Path | None = None,
    b1_condition: str | None = None,
    tune_sam_root: Path | None = None,
    final_sam_root: Path | None = None,
    sam_template: str = "{root}/{scene_id}",
    tune_grounded_masks_root: Path | None = None,
    tune_grounded_labels_root: Path | None = None,
    final_grounded_masks_root: Path | None = None,
    final_grounded_labels_root: Path | None = None,
    grounded_template: str = "{root}/{scene_id}",
    sam_checkpoint: Path | None = None,
    sam_arch: str = "vit_h",
    git_head_reader: Callable[[Path], str] = _git_head,
) -> dict[str, Any]:
    required_files = (
        tune_runtime_manifest,
        final_runtime_manifest,
        locked_evaluation_scenes,
        train_scene_list,
        category_priors,
        size_bins,
        taxonomy,
        b1_fixed_metrics,
    )
    for path in required_files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    # GT roots and all per-scene assets are deliberately stage-local.  Their
    # paths are frozen below, but final48 being offline must not prevent DEV2
    # from starting.
    for path in (repo_root,):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    actual_head = git_head_reader(Path(repo_root).resolve()).strip()
    registered_commit = code_commit.strip()
    if (
        len(registered_commit) != 40
        or any(character not in "0123456789abcdef" for character in registered_commit)
    ):
        raise ValueError("code_commit must be a full lowercase git commit")
    if actual_head != registered_commit:
        raise ValueError(f"registered code commit {code_commit!r} differs from repo HEAD {actual_head!r}")

    tune_runtime = load_scene_runtime_manifest(tune_runtime_manifest)
    final_runtime = load_scene_runtime_manifest(final_runtime_manifest)
    locked = _locked_rows(locked_evaluation_scenes)
    tune24 = tuple(tune_runtime)
    final48 = tuple(final_runtime)
    if set(final48) != set(locked) or len(final48) != len(locked):
        raise ValueError("final runtime does not exactly match locked_evaluation_scenes")
    imported_evidence = _load_evidence_imports(evidence_imports)
    unknown_imports = sorted(
        set(imported_evidence).difference(set(tune24).union(final48))
    )
    if unknown_imports:
        raise ValueError(
            f"evidence imports reference unregistered scenes: {unknown_imports}"
        )
    split_report = _validate_split(
        tune24=tune24,
        final48=final48,
        train=_load_train_scene_ids(train_scene_list),
    )

    taxonomy_value = load_taxonomy(taxonomy)
    evidence_classes = tuple(DEFAULT_CLASSES)
    evaluation_classes = tuple(taxonomy_value.canonical_classes)
    if len(evidence_classes) != 32 or len(set(evidence_classes)) != 32:
        raise ValueError("evidence class vocabulary must contain exactly 32 unique classes")
    if len(evaluation_classes) != 20 or len(set(evaluation_classes)) != 20:
        raise ValueError("evaluation/SAGA vocabulary must contain exactly 20 unique classes")
    if not set(evaluation_classes).issubset(evidence_classes):
        raise ValueError("all 20 evaluation classes must exist in the 32-class evidence vocabulary")

    prior_payload = load_json(category_priors)
    provenance = prior_payload.get("provenance", {}) if isinstance(prior_payload, Mapping) else {}
    if not isinstance(prior_payload, Mapping) or prior_payload.get("kind") != "category_priors":
        raise ValueError("category priors artifact has the wrong kind")
    if provenance.get("splits") != ["train"]:
        raise ValueError("category priors must be fitted from the train split only")
    if prior_payload.get("normalization", {}).get("units") != "meters":
        raise ValueError("category priors must use metric units")
    prior_categories = prior_payload.get("categories")
    if not isinstance(prior_categories, Mapping) or set(prior_categories) != set(evaluation_classes):
        raise ValueError("category priors must contain exactly the official SAGA20 classes")
    SizePriorTable.from_category_priors(prior_payload)
    size_spec = load_json(size_bins)
    if not isinstance(size_spec, Mapping):
        raise TypeError("size bins must be a JSON object")
    boundaries = size_spec.get("boundaries_m", size_spec)
    for key in ("tiny_max_m", "small_max_m", "medium_max_m"):
        if key not in boundaries or not math.isfinite(float(boundaries[key])):
            raise ValueError(f"size bins lack finite {key}")
    b1_map, b1_ap50 = _load_b1_metrics(b1_fixed_metrics, b1_condition)

    output = Path(output_dir).resolve()
    requests = output / "evidence_requests"
    requests.mkdir(parents=True, exist_ok=True)
    Path(run_root).resolve().mkdir(parents=True, exist_ok=True)
    Path(artifact_root).resolve().mkdir(parents=True, exist_ok=True)

    scenes: dict[str, Any] = {}
    scene_reports: list[dict[str, Any]] = []
    all_runtime = {**tune_runtime, **final_runtime}
    checkpoint = (
        Path(sam_checkpoint).resolve()
        if sam_checkpoint is not None
        else (
            Path(repo_root).resolve()
            / "third_party/segment-anything/weights/sam_vit_h_4b8939.pth"
        ).resolve()
    )
    for scene_id in tuple(tune24) + tuple(final48):
        is_final = scene_id in final_runtime
        raw = all_runtime[scene_id]
        scene = _runtime_with_overrides(
            scene_id,
            raw,
            sam_root=final_sam_root if is_final else tune_sam_root,
            sam_template=sam_template,
            grounded_masks_root=(final_grounded_masks_root if is_final else tune_grounded_masks_root),
            grounded_labels_root=(final_grounded_labels_root if is_final else tune_grounded_labels_root),
            grounded_template=grounded_template,
        )
        if not any(
            scene.get(key) not in (None, "")
            for key in (
                "segment_everything_root",
                "sam_everything_packed_path",
                "sam_everything_root",
            )
        ):
            scene["segment_everything_root"] = str(
                Path(run_root).resolve() / "sam-everything" / scene_id
            )
        scale = float(scene.get("scene_scale_m_per_unit", 0.0))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{scene_id}: scene_scale_m_per_unit must be finite and positive")
        transform = _transform(scene_id, scene)
        scene = _formal_runtime_fields(scene)
        formal_runtime_registration = _formal_runtime_fields(raw)
        inputs = resolve_clean_scene_inputs(scene, require_exists=False)
        _prove_30k_ply(scene_id, scene, inputs.rgb_ply)
        locked_row = locked.get(scene_id) if is_final else None
        gt_npz = _gt_path(
            scene_id,
            root=final_gt_root if is_final else tune_gt_root,
            locked_row=locked_row,
            locked_spec_dir=Path(locked_evaluation_scenes).resolve().parent,
        )
        request_path = requests / f"{scene_id}.json"
        imported = imported_evidence.get(scene_id)
        if imported is None:
            request = {
                "schema": REQUEST_SCHEMA,
                "producer_commit": code_commit,
                "classes": list(evidence_classes),
                "runtime_registration": formal_runtime_registration,
                "scene": scene,
                "sam_generation": {
                    "output_root": str(
                        Path(run_root).resolve() / "sam-everything" / scene_id
                    ),
                    "checkpoint": str(checkpoint),
                    "sam_arch": str(sam_arch),
                    "device": "cuda",
                    "config": dict(SAM_EVERYTHING_CONFIG),
                    "download_allowed": False,
                    "overwrite_registered_masks": False,
                },
            }
        else:
            request = dict(imported["request"])
            if tuple(map(str, request.get("classes", ()))) != evidence_classes:
                raise ValueError(
                    f"{scene_id}: imported request class vocabulary changed"
                )
            if request.get("runtime_registration") != formal_runtime_registration:
                raise ValueError(
                    f"{scene_id}: imported request runtime registration changed"
                )
            request_scene = request.get("scene")
            if not isinstance(request_scene, Mapping) or _formal_runtime_fields(
                request_scene
            ) != scene:
                raise ValueError(
                    f"{scene_id}: imported request scene registration changed"
                )
        _write_identity_json(request_path, request)
        scenes[scene_id] = {
            "evidence_request": str(request_path),
            "gt_npz": str(gt_npz),
            "gaussian_ply": str(inputs.rgb_ply),
            "gaussian_to_gt_transform": [list(row) for row in transform],
            "tiny_small_instance_ids": [],
            "tiny_small_deferred": True,
        }
        scene_reports.append(
            {
                "scene_id": scene_id,
                "physical_scene_id": physical_scene_id(scene_id),
                "split": "final48" if is_final else "tune24",
                "base_path": str(inputs.base_path),
                "scene_scale_m_per_unit": scale,
                "gaussian_ply": str(inputs.rgb_ply),
                "gt_npz": str(gt_npz),
                "sam_masks": str(inputs.sam_masks),
                "grounded_masks": str(inputs.grounded_masks),
                "grounded_labels": str(inputs.grounded_labels),
                "asset_validation": "deferred-until-stage",
                "tiny_small_derivation": "deferred-until-stage",
                "evidence_producer_commit": str(request["producer_commit"]),
                "evidence_imported": imported is not None,
            }
        )

    identity_registration, identity_control = _identity_control_registration(
        runtime=all_runtime,
        scenes=scenes,
    )
    train_physical = sorted({physical_scene_id(value) for value in _load_train_scene_ids(train_scene_list)})
    config = {
        "kind": CONFIG_KIND,
        "code_commit": code_commit,
        "repo_root": str(Path(repo_root).resolve()),
        "run_root": str(Path(run_root).resolve()),
        "artifact_root": str(Path(artifact_root).resolve()),
        "category_priors": str(Path(category_priors).resolve()),
        "size_bins": str(Path(size_bins).resolve()),
        "size_bin_boundaries_m": {
            key: float(boundaries[key])
            for key in ("tiny_max_m", "small_max_m", "medium_max_m")
        },
        "class_names": list(evidence_classes),
        "evidence_class_names": list(evidence_classes),
        "evaluation_class_names": list(evaluation_classes),
        "allowed_classes": list(evaluation_classes),
        "b1_fixed_metrics": {"map_50_95": b1_map, "map_0.50": b1_ap50},
        "dev2": list(DEV2),
        "dev8": list(DEV8),
        "holdout5": list(HOLDOUT5),
        "tune24": list(tune24),
        "final48": list(final48),
        "train_physical_scene_ids": train_physical,
        "holdout_physical_scene_ids": sorted({physical_scene_id(value) for value in HOLDOUT5}),
        "final_physical_scene_ids": sorted({physical_scene_id(value) for value in final48}),
        "min_region_size": 100,
        "radius_m": 0.05,
        "identity_control_registration": identity_registration,
        "evidence_imports": {
            scene_id: {
                "schema": EVIDENCE_IMPORT_SCHEMA,
                "bank_dir": str(row["bank_dir"]),
                "producer_commit": str(row["producer_commit"]),
                "files": dict(row["files"]),
            }
            for scene_id, row in sorted(imported_evidence.items())
        },
        "runtime_registration": {
            scene_id: _formal_runtime_fields(all_runtime[scene_id])
            for scene_id in tuple(tune24) + tuple(final48)
        },
        "scenes": scenes,
    }
    if identity_control is not None:
        config["identity_control"] = identity_control
    config_path = output / "clean_baseline_experiment.json"
    _write_identity_json(config_path, config)
    report = {
        "schema": REGISTRATION_SCHEMA,
        "status": "complete",
        "config": str(config_path),
        "code_commit": code_commit,
        "evidence_class_count": len(evidence_classes),
        "evaluation_class_count": len(evaluation_classes),
        "allowed_classes_equal_evaluation_classes": list(evaluation_classes) == config["allowed_classes"],
        "locked_runtime_matches_spec": True,
        "identity_control_status": identity_registration["status"],
        "identity_control_issues": list(identity_registration["issues"]),
        "imported_evidence_scene_ids": sorted(imported_evidence),
        **split_report,
        "scenes": scene_reports,
    }
    report_path = output / "clean_baseline_registration.json"
    _write_identity_json(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune-runtime-manifest", type=Path, required=True)
    parser.add_argument("--final-runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--tune-gt-root", type=Path, required=True)
    parser.add_argument("--final-gt-root", type=Path, required=True)
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--b1-fixed-metrics", type=Path, required=True)
    parser.add_argument("--b1-condition")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--evidence-imports",
        type=Path,
        help=(
            "optional registered producer manifest for byte-identical evidence "
            "banks that must be reused read-only"
        ),
    )
    parser.add_argument("--tune-sam-root", type=Path)
    parser.add_argument("--final-sam-root", type=Path)
    parser.add_argument("--sam-template", default="{root}/{scene_id}")
    parser.add_argument("--tune-grounded-masks-root", type=Path)
    parser.add_argument("--tune-grounded-labels-root", type=Path)
    parser.add_argument("--final-grounded-masks-root", type=Path)
    parser.add_argument("--final-grounded-labels-root", type=Path)
    parser.add_argument("--grounded-template", default="{root}/{scene_id}")
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-arch", default="vit_h")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = materialize_clean_baseline_config(
        tune_runtime_manifest=args.tune_runtime_manifest,
        final_runtime_manifest=args.final_runtime_manifest,
        locked_evaluation_scenes=args.locked_evaluation_scenes,
        tune_gt_root=args.tune_gt_root,
        final_gt_root=args.final_gt_root,
        train_scene_list=args.train_scene_list,
        category_priors=args.category_priors,
        size_bins=args.size_bins,
        taxonomy=args.taxonomy,
        b1_fixed_metrics=args.b1_fixed_metrics,
        b1_condition=args.b1_condition,
        repo_root=args.repo_root,
        run_root=args.run_root,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        code_commit=args.code_commit,
        evidence_imports=args.evidence_imports,
        tune_sam_root=args.tune_sam_root,
        final_sam_root=args.final_sam_root,
        sam_template=args.sam_template,
        tune_grounded_masks_root=args.tune_grounded_masks_root,
        tune_grounded_labels_root=args.tune_grounded_labels_root,
        final_grounded_masks_root=args.final_grounded_masks_root,
        final_grounded_labels_root=args.final_grounded_labels_root,
        grounded_template=args.grounded_template,
        sam_checkpoint=args.sam_checkpoint,
        sam_arch=args.sam_arch,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
