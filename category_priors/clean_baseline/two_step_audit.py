from __future__ import annotations

"""Read-only orchestration for step one of the clean mask-contract audit."""

import json
import math
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
from ..io import (
    build_file_manifest,
    hash_json,
    load_json,
    read_rows,
    sha256_file,
    write_json,
    write_rows,
)
from ..prediction_contract import validate_prediction_contract
from .evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    evidence_request_source,
    load_evidence_bank,
)
from .evaluation import (
    RUN_IDENTITY_SCHEMA,
    CleanCandidate,
    GroundTruthObject,
    ground_truth_objects_from_arrays,
    validate_embedded_identity,
)
from .metric_reaudit import (
    FORMAL_RADIUS_M,
    build_bidirectional_nearest,
    evaluate_candidate_set_three_spaces,
    evaluate_dual_protocols,
    evaluate_gt_as_prediction_dual_protocols,
    formal_gt_point_mask,
)
from .materialize_config import LEGACY_HIERARCHY_PRODUCER_COMMITS
from .stage_funnel import FunnelObject, audit_frozen_clean_scene


MANIFEST_SCHEMA = "saga-clean-mask-contract-manifest-v1"
METRIC_OUTPUT_SCHEMA = "saga-clean-metric-reaudit-dev8-v1"
FUNNEL_OUTPUT_SCHEMA = "saga-clean-stage-funnel-dev8-v1"
REGISTERED_CONDITIONS = ("C0-no-prior", "U-global")
DEFAULT_DEV8_SCENE_COUNT = 8
AUDIT_IDENTITY_SCHEMA = "saga-clean-metric-reaudit-identity-v1"
REGISTERED_DEV2_SCENE_IDS = ("scene0645_00", "scene0025_01")
REGISTERED_DEV8_SCENE_IDS = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)


def _resolve(base: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an explicit non-empty path")
    path = Path(value)
    result = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def _taxonomy(payload: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw = payload.get("taxonomy")
    if not isinstance(raw, Mapping):
        raise ValueError("manifest taxonomy must be an explicit mapping")
    class_names = raw.get("class_names")
    allowed_classes = raw.get("allowed_classes")
    for name, value in (
        ("taxonomy.class_names", class_names),
        ("taxonomy.allowed_classes", allowed_classes),
    ):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must be a class-name sequence")
    classes = tuple(str(value).strip() for value in class_names)
    allowed = tuple(str(value).strip() for value in allowed_classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("taxonomy.class_names must be non-empty and unique")
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError("taxonomy.allowed_classes must be non-empty and unique")
    if not set(allowed).issubset(classes):
        raise ValueError("allowed_classes must be a subset of class_names")
    return classes, allowed


def _size_boundaries(payload: Mapping[str, Any]) -> tuple[float, float, float]:
    raw = payload.get("size_bins")
    if not isinstance(raw, Mapping):
        raise ValueError("manifest size_bins must be explicit")
    boundaries = raw.get("boundaries_m", raw)
    if not isinstance(boundaries, Mapping):
        raise ValueError("size_bins.boundaries_m must be a mapping")
    try:
        values = tuple(
            float(boundaries[key])
            for key in ("tiny_max_m", "small_max_m", "medium_max_m")
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("size bins require tiny/small/medium metric boundaries") from exc
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("size-bin boundaries must be positive and finite")
    if not values[0] < values[1] < values[2]:
        raise ValueError("size-bin boundaries must be strictly increasing")
    return values


def _bbox_diagonal_m(points: np.ndarray) -> float:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz):
        raise ValueError("object GT points must have shape (N, 3) and be non-empty")
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    if len(xyz) >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        centered = centered @ axes.T
    return float(np.linalg.norm(np.ptp(centered, axis=0)))


def _tiny_small_ids(
    *,
    gt_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_count: int,
    small_max_m: float,
    min_region_size: int,
) -> set[int]:
    """Use complete original GT objects; mapping success is intentionally absent."""

    selected: set[int] = set()
    for instance_id in np.unique(instance[instance >= 0]):
        instance_mask = instance == int(instance_id)
        values, counts = np.unique(semantic[instance_mask], return_counts=True)
        valid = (values >= 0) & (values < int(class_count))
        if not np.any(valid):
            continue
        values, counts = values[valid], counts[valid]
        # Size strata use the complete raw GT instance and never depend on
        # mapping success.  The downstream GroundTruthObject constructor is
        # still the authority for official class/min-region eligibility.
        _ = int(values[counts == counts.max()].min())
        object_points = np.flatnonzero(instance_mask)
        if len(object_points) < int(min_region_size):
            continue
        if _bbox_diagonal_m(gt_xyz[object_points]) <= float(small_max_m):
            selected.add(int(instance_id))
    return selected


def _transform(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("scene transform must be an explicit finite 4x4 matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
        raise ValueError("scene transform is singular")
    return matrix


def _expected_metrics(
    payload: Mapping[str, Any], *, tolerance: float
) -> dict[str, dict[str, float]]:
    raw = payload.get("expected_metrics")
    if not isinstance(raw, Mapping):
        raise ValueError("manifest expected_metrics is required; old metrics are not inferred")
    result: dict[str, dict[str, float]] = {}
    required = {"official_ap25", "official_ap50", "historical_map_50_95"}
    for condition in REGISTERED_CONDITIONS:
        row = raw.get(condition)
        if not isinstance(row, Mapping):
            raise ValueError(f"expected_metrics omitted {condition}")
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"expected_metrics[{condition!r}] omitted {sorted(missing)}"
            )
        values = {key: float(row[key]) for key in required}
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values.values()):
            raise ValueError("expected metrics must be finite values in [0, 1]")
        result[condition] = values
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("metric_tolerance must be finite and non-negative")
    return result


def _output_candidates(
    *,
    payload: Mapping[str, Any],
    scene_id: str,
    condition: str,
    gaussian_count: int,
    nearest: Any,
    class_names: Sequence[str],
) -> tuple[list[CleanCandidate], list[PredictedInstance]]:
    if str(payload.get("scene_id")) != str(scene_id):
        raise ValueError("prediction scene identity mismatch")
    if str(payload.get("condition")) != str(condition):
        raise ValueError("prediction condition identity mismatch")
    labels = np.asarray(payload.get("point_labels"))
    instances = payload.get("instances")
    if labels.shape != (int(gaussian_count),) or labels.dtype.kind not in "iuf":
        raise ValueError("prediction point_labels have the wrong shape or type")
    if labels.dtype.kind == "f" and (
        np.any(~np.isfinite(labels)) or np.any(labels != np.floor(labels))
    ):
        raise TypeError("prediction point_labels must contain integers")
    labels = labels.astype(np.int64, copy=False)
    if not isinstance(instances, Mapping):
        raise TypeError("prediction instances must be a mapping")
    validate_prediction_contract(labels, instances)
    class_to_id = {str(name): index for index, name in enumerate(class_names)}
    candidates: list[CleanCandidate] = []
    predictions: list[PredictedInstance] = []
    for raw_id in sorted(instances, key=lambda value: int(value)):
        instance_id = int(raw_id)
        metadata = instances[raw_id]
        if not isinstance(metadata, Mapping):
            raise TypeError("prediction instance metadata must be a mapping")
        if "class" not in metadata or "score" not in metadata:
            raise ValueError("prediction metadata must explicitly contain class and score")
        class_name = str(metadata["class"])
        if class_name not in class_to_id:
            raise ValueError(f"prediction uses unknown class {class_name!r}")
        score = float(metadata["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("prediction score must be finite and in [0, 1]")
        gaussian_ids = np.flatnonzero(labels == instance_id).astype(np.int64)
        candidates.append(
            CleanCandidate(
                object_id=instance_id,
                gaussian_ids=gaussian_ids,
                class_id=class_name,
                winner_probability=score,
                view_consensus=1.0,
                detection_ratio=1.0,
            )
        )
        predictions.append(
            PredictedInstance(
                scene_id=str(scene_id),
                instance_id=instance_id,
                class_id=class_to_id[class_name],
                score=score,
                mask=formal_gt_point_mask(gaussian_ids, nearest),
            )
        )
    return candidates, predictions


def _compact_candidate_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for subset_name in ("all", "official_evaluable"):
        subset = result["subsets"][subset_name]
        output[f"{subset_name}_candidate_count"] = int(subset["candidate_count"])
        for match_name in ("geometry", "same_class"):
            for threshold in ("0.25", "0.50"):
                row = subset["matching"][match_name][threshold]
                prefix = f"{subset_name}_{match_name}_{threshold.replace('.', '')}"
                output[f"{prefix}_match_count"] = int(row["true_positive_count"])
                output[f"{prefix}_precision"] = float(row["precision"])
                output[f"{prefix}_recall"] = float(row["recall"])
                output[f"{prefix}_tiny_small_recall"] = float(
                    row["tiny_small_recall"]
                )
    return output


def _funnel_metric_callback(
    *,
    gt_objects: Sequence[GroundTruthObject],
    nearest: Any,
    min_region_size: int,
) -> Any:
    def callback(stage: str, objects: tuple[FunnelObject, ...]) -> dict[str, Any]:
        del stage
        candidates = [
            {
                "object_id": item.stable_id,
                "gaussian_ids": item.gaussian_ids,
                "class_id": item.class_name,
                "winner_probability": 1.0,
                "view_consensus": 1.0,
                "detection_ratio": 1.0,
            }
            for item in objects
        ]
        result = evaluate_candidate_set_three_spaces(
            candidates=candidates,
            gt_objects=gt_objects,
            nearest=nearest,
            radii_m=(FORMAL_RADIUS_M,),
            min_region_size=min_region_size,
        )
        return _compact_candidate_metrics(result)

    return callback


def _flatten_candidate_rows(
    *, scene_id: str, condition: str, result: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in result["candidate_rows"]:
        shared = {key: value for key, value in candidate.items() if key != "radii"}
        for radius, values in candidate["radii"].items():
            rows.append(
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    **shared,
                    "radius_key": radius,
                    **dict(values),
                }
            )
    return rows


def _protected_paths(
    *, manifest_path: Path, base: Path, scenes: Sequence[Mapping[str, Any]]
) -> list[Path]:
    paths: set[Path] = {manifest_path.resolve()}
    for scene in scenes:
        bank_dir = _resolve(base, scene.get("bank_dir"), name="scene.bank_dir")
        if not bank_dir.is_dir():
            raise NotADirectoryError(bank_dir)
        paths.update(path.resolve() for path in bank_dir.rglob("*") if path.is_file())
        paths.add(_resolve(base, scene.get("gt_npz"), name="scene.gt_npz"))
        paths.add(_resolve(base, scene.get("gaussian_ply"), name="scene.gaussian_ply"))
        conditions = scene.get("conditions")
        if not isinstance(conditions, Mapping):
            raise ValueError("scene.conditions must be an explicit mapping")
        for condition in REGISTERED_CONDITIONS:
            row = conditions.get(condition)
            if not isinstance(row, Mapping):
                raise ValueError(f"scene conditions omitted {condition}")
            paths.add(_resolve(base, row.get("output"), name="condition.output"))
            paths.add(
                _resolve(base, row.get("diagnostics"), name="condition.diagnostics")
            )
    return sorted(paths, key=lambda value: str(value))


def _assert_separate_output_directory(
    *,
    destination: Path,
    protected: Sequence[Path],
    base: Path,
    scenes: Sequence[Mapping[str, Any]],
) -> None:
    """Keep audit products outside every immutable production artifact tree."""

    immutable_directories: set[Path] = set()
    for scene in scenes:
        immutable_directories.add(
            _resolve(base, scene.get("gt_npz"), name="scene.gt_npz").parent
        )
        immutable_directories.add(
            _resolve(
                base, scene.get("gaussian_ply"), name="scene.gaussian_ply"
            ).parent
        )
        immutable_directories.add(
            _resolve(base, scene.get("bank_dir"), name="scene.bank_dir")
        )
        conditions = scene.get("conditions")
        if not isinstance(conditions, Mapping):
            raise ValueError("scene.conditions must be an explicit mapping")
        for condition in REGISTERED_CONDITIONS:
            row = conditions.get(condition)
            if not isinstance(row, Mapping):
                raise ValueError(f"scene conditions omitted {condition}")
            immutable_directories.add(
                _resolve(base, row.get("output"), name="condition.output").parent
            )
            immutable_directories.add(
                _resolve(
                    base,
                    row.get("diagnostics"),
                    name="condition.diagnostics",
                ).parent
            )
    for path in protected:
        if path == destination:
            raise ValueError("audit output_dir cannot be a frozen input file")
    for directory in immutable_directories:
        if (
            destination == directory
            or destination.is_relative_to(directory)
            or directory.is_relative_to(destination)
        ):
            raise ValueError(
                "audit output_dir must be separate from frozen artifact trees"
            )


def _same_file_manifests(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> bool:
    return [dict(row) for row in before] == [dict(row) for row in after]


def _audit_identity(manifest_file: Path, frozen_inputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    implementation = []
    for relative in (
        "category_priors/evaluator.py",
        "category_priors/clean_baseline/evaluation.py",
        "category_priors/clean_baseline/metric_reaudit.py",
        "category_priors/clean_baseline/stage_funnel.py",
        "category_priors/clean_baseline/two_step_audit.py",
    ):
        path = repository / relative
        implementation.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    identity: dict[str, Any] = {
        "schema": AUDIT_IDENTITY_SCHEMA,
        "manifest_sha256": sha256_file(manifest_file),
        "frozen_inputs": [dict(row) for row in frozen_inputs],
        "implementation": implementation,
    }
    identity["content_sha256"] = hash_json(identity)
    return identity


def _load_complete_audit(
    destination: Path, *, expected_identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    metric_json = destination / "clean_metric_reaudit_dev8.json"
    funnel_json = destination / "clean_stage_funnel_dev8.json"
    metric_parquet = destination / "clean_metric_reaudit_dev8.parquet"
    funnel_parquet = destination / "clean_stage_funnel_dev8.parquet"
    try:
        metric = load_json(metric_json)
        funnel = load_json(funnel_json)
        if (
            not isinstance(metric, Mapping)
            or not isinstance(funnel, Mapping)
            or metric.get("schema") != METRIC_OUTPUT_SCHEMA
            or funnel.get("schema") != FUNNEL_OUTPUT_SCHEMA
            or metric.get("audit_identity") != dict(expected_identity)
            or funnel.get("audit_identity") != dict(expected_identity)
        ):
            return None
        # Parsing both tables is part of completeness; a truncated Parquet is
        # never accepted merely because its JSON sibling is intact.
        read_rows(metric_parquet)
        read_rows(funnel_parquet)
        gates = metric.get("technical_gates")
        if not isinstance(gates, Mapping):
            return None
        return {
            "metric": dict(metric),
            "funnel": dict(funnel),
            "technical_gates": dict(gates),
            "output_dir": str(destination),
            "runner_status": "skipped-complete",
        }
    except (OSError, TypeError, ValueError, KeyError):
        return None


def preflight_audit_output_directory(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    expected_scene_ids: Sequence[str] = REGISTERED_DEV8_SCENE_IDS,
) -> None:
    """Validate scene scope and path isolation before creating any output."""

    manifest_file = Path(manifest_path).resolve()
    payload = load_json(manifest_file)
    if not isinstance(payload, Mapping) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("manifest scenes must be an explicit list")
    received = tuple(str(row.get("scene_id", "")) for row in scenes)
    expected = tuple(map(str, expected_scene_ids))
    if received != expected:
        raise ValueError("manifest scenes must exactly match registered DEV8 order")
    base = manifest_file.parent
    protected = _protected_paths(
        manifest_path=manifest_file, base=base, scenes=scenes
    )
    _assert_separate_output_directory(
        destination=Path(output_dir).resolve(),
        protected=protected,
        base=base,
        scenes=scenes,
    )


def _audit_registered_evidence(
    *,
    scene: Mapping[str, Any],
    scene_id: str,
    bank_dir: Path,
    bank: Any,
    gaussian_path: Path,
    gaussian_raw_xyz: np.ndarray,
) -> dict[str, Any]:
    registration = scene.get("evidence_import_identity")
    checks: dict[str, bool] = {}
    reason: str | None = None
    try:
        if not isinstance(registration, Mapping):
            raise ValueError("scene omitted evidence_import_identity")
        files = registration.get("files")
        expected_names = {
            EVIDENCE_ARRAY_FILE,
            EVIDENCE_METADATA_FILE,
            EVIDENCE_DIAGNOSTICS_FILE,
        }
        if not isinstance(files, Mapping) or set(map(str, files)) != expected_names:
            raise ValueError("evidence import did not register exactly three files")
        actual = {
            name: sha256_file(bank_dir / name) for name in sorted(expected_names)
        }
        scale = float(bank.source.get("scene_scale_m_per_unit", math.nan))
        expected_xyz_m = np.asarray(gaussian_raw_xyz, dtype=np.float64) * scale
        checks = {
            "scene_id_exact": str(bank.scene_id) == scene_id,
            "bank_dir_exact": Path(str(registration.get("bank_dir", ""))).resolve()
            == bank_dir.resolve(),
            "producer_commit_exact": str(registration.get("producer_commit", ""))
            == str(bank.source.get("producer_commit", "")),
            "evidence_file_hashes_exact": actual
            == {str(key): str(value) for key, value in files.items()},
            "registered_gaussian_path_exact": Path(
                str(bank.source.get("rgb_ply", ""))
            ).resolve()
            == gaussian_path.resolve(),
            "registered_gaussian_xyz_order_exact": bool(
                math.isfinite(scale)
                and scale > 0
                and expected_xyz_m.shape == bank.xyz_m.shape
                and np.allclose(
                    expected_xyz_m,
                    np.asarray(bank.xyz_m, dtype=np.float64),
                    rtol=0.0,
                    atol=1e-6,
                )
            ),
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        reason = str(exc)
    return {
        "scene_id": scene_id,
        "passed": reason is None and bool(checks) and all(checks.values()),
        "checks": checks,
        "violation": reason,
    }


def audit_condition_run_identity(
    *,
    scene_id: str,
    condition: str,
    spec: Mapping[str, Any],
    output_path: Path,
    diagnostics_path: Path,
    output: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    bank_dir: Path,
    bank: Any,
) -> dict[str, Any]:
    """Bind output, diagnostics and their embedded identity to one bank."""

    checks: dict[str, bool] = {}
    reason: str | None = None
    try:
        output_identity = validate_embedded_identity(
            output.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        diagnostic_identity = validate_embedded_identity(
            diagnostics.get("run_identity"), expected_schema=RUN_IDENTITY_SCHEMA
        )
        evidence = output_identity.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("run identity omitted evidence identity")
        files = evidence.get("files")
        expected_names = {
            EVIDENCE_ARRAY_FILE,
            EVIDENCE_METADATA_FILE,
            EVIDENCE_DIAGNOSTICS_FILE,
        }
        if not isinstance(files, Mapping) or set(map(str, files)) != expected_names:
            raise ValueError("run identity omitted the exact evidence file set")
        actual_evidence = {
            name: sha256_file(bank_dir / name) for name in sorted(expected_names)
        }
        registered_output_sha = str(spec.get("output_sha256", ""))
        registered_diagnostics_sha = str(spec.get("diagnostics_sha256", ""))
        registered_identity_sha = str(spec.get("run_identity_sha256", ""))
        registered_consumer = str(spec.get("consumer_commit", ""))
        checks = {
            "output_and_diagnostics_identity_exact": output_identity
            == diagnostic_identity,
            "scene_exact": str(output_identity.get("scene_id")) == scene_id,
            "condition_exact": str(output_identity.get("condition")) == condition,
            "consumer_commit_registered": str(
                output_identity.get("consumer_commit", "")
            )
            == registered_consumer,
            "run_identity_registered": str(output_identity.get("content_sha256", ""))
            == registered_identity_sha,
            "output_bytes_registered": sha256_file(output_path)
            == registered_output_sha,
            "diagnostics_bytes_registered": sha256_file(diagnostics_path)
            == registered_diagnostics_sha,
            "evidence_files_current": actual_evidence
            == {str(key): str(value) for key, value in files.items()},
            "evidence_scene_exact": str(evidence.get("scene_id")) == scene_id,
            "evidence_schema_exact": str(evidence.get("schema")) == str(bank.schema),
            "evidence_point_count_exact": int(evidence.get("point_count", -1))
            == int(bank.point_count),
            "evidence_source_exact": evidence.get("source") == dict(bank.source),
            "evidence_classes_exact": tuple(map(str, evidence.get("class_names", ())))
            == tuple(map(str, bank.class_names)),
            "evidence_thresholds_exact": evidence.get("thresholds")
            == bank.thresholds.to_dict(),
            "diagnostics_config_exact": diagnostics.get("config")
            == output_identity.get("consensus_config"),
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        reason = str(exc)
    return {
        "scene_id": scene_id,
        "condition": condition,
        "passed": reason is None and bool(checks) and all(checks.values()),
        "checks": checks,
        "violation": reason,
    }


def audit_clean_baseline_manifest(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    expected_scene_count: int = DEFAULT_DEV8_SCENE_COUNT,
) -> dict[str, Any]:
    """Run the complete read-only DEV8 metric and stage-funnel audit."""

    manifest_file = Path(manifest_path).resolve()
    payload = load_json(manifest_file)
    if not isinstance(payload, Mapping) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    base = manifest_file.parent
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != int(expected_scene_count):
        raise ValueError(
            f"manifest must contain exactly {int(expected_scene_count)} scenes"
        )
    if int(expected_scene_count) <= 0:
        raise ValueError("expected_scene_count must be positive")
    scene_ids = [str(scene.get("scene_id", "")) for scene in scenes]
    if any(not value for value in scene_ids) or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("scene IDs must be non-empty and unique")
    if int(expected_scene_count) == DEFAULT_DEV8_SCENE_COUNT and tuple(
        scene_ids
    ) != REGISTERED_DEV8_SCENE_IDS:
        raise ValueError("DEV8 manifest must exactly match the registered scene order")
    class_names, allowed_classes = _taxonomy(payload)
    _, small_max_m, _ = _size_boundaries(payload)
    if int(payload.get("min_region_size", -1)) != 100:
        raise ValueError("manifest min_region_size must explicitly equal 100")
    min_region_size = 100
    tolerance = float(payload.get("metric_tolerance", math.nan))
    expected = _expected_metrics(payload, tolerance=tolerance)
    protected = _protected_paths(
        manifest_path=manifest_file, base=base, scenes=scenes
    )
    before = build_file_manifest(protected)
    destination = Path(output_dir).resolve()
    _assert_separate_output_directory(
        destination=destination,
        protected=protected,
        base=base,
        scenes=scenes,
    )
    audit_identity = _audit_identity(manifest_file, before)
    complete = _load_complete_audit(
        destination, expected_identity=audit_identity
    )
    if complete is not None:
        return complete

    ground_truth_scenes: list[GroundTruthScene] = []
    predictions_by_condition: dict[str, list[PredictedInstance]] = {
        condition: [] for condition in REGISTERED_CONDITIONS
    }
    metric_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    scene_metric_summaries: dict[str, Any] = {}
    funnel_summaries: dict[str, Any] = {}
    mapping_gate_rows: list[dict[str, Any]] = []
    funnel_gate_rows: list[dict[str, Any]] = []
    evidence_registration_rows: list[dict[str, Any]] = []
    condition_identity_rows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_id = str(scene["scene_id"])
        gt_path = _resolve(base, scene.get("gt_npz"), name="scene.gt_npz")
        gaussian_path = _resolve(
            base, scene.get("gaussian_ply"), name="scene.gaussian_ply"
        )
        gt_xyz, gt_scene = load_ground_truth_npz(gt_path, scene_id)
        gaussian_raw_xyz = load_ply_xyz(gaussian_path)
        gaussian_xyz = apply_transform(
            gaussian_raw_xyz, _transform(scene.get("transform"))
        )
        nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
        mapped_fraction = float(
            np.mean(nearest.gt_to_gaussian_distance_m <= FORMAL_RADIUS_M)
        )
        mapping_gate_rows.append(
            {
                "scene_id": scene_id,
                "gt_point_count": nearest.gt_count,
                "gaussian_count": nearest.gaussian_count,
                "mapped_fraction_5cm": mapped_fraction,
                "passed": mapped_fraction >= 0.90,
            }
        )
        tiny_small = _tiny_small_ids(
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
        ground_truth_scenes.append(gt_scene)

        source_request = scene.get("source_evidence_request")
        if not isinstance(source_request, Mapping):
            raise ValueError("scene.source_evidence_request must be explicit")
        expected_source = evidence_request_source(
            scene_id=scene_id, request=source_request
        )
        registered_source = dict(expected_source)
        registration = scene.get("evidence_import_identity")
        proof = (
            registration.get("legacy_hierarchy_mode_proof")
            if isinstance(registration, Mapping)
            else None
        )
        if proof is not None:
            if not isinstance(proof, Mapping):
                raise ValueError("legacy hierarchy proof must be an object")
            expected_proof = {
                "producer_commit": str(registration.get("producer_commit", "")),
                "assumed_mode": "hierarchy",
                "missing_fields": ["mask_observation_mode"],
            }
            if (
                dict(proof) != expected_proof
                or expected_proof["producer_commit"]
                not in LEGACY_HIERARCHY_PRODUCER_COMMITS
            ):
                raise ValueError("legacy hierarchy proof is not allowlisted")
            if registered_source.pop("mask_observation_mode", None) != "hierarchy":
                raise ValueError(
                    "legacy hierarchy proof is incompatible with the source request"
                )
        bank_dir = _resolve(base, scene.get("bank_dir"), name="scene.bank_dir")
        bank = load_evidence_bank(
            bank_dir,
            expected_scene_id=scene_id,
            expected_point_count=len(gaussian_xyz),
            expected_source=registered_source,
        )
        evidence_registration_rows.append(
            _audit_registered_evidence(
                scene=scene,
                scene_id=scene_id,
                bank_dir=bank_dir,
                bank=bank,
                gaussian_path=gaussian_path,
                gaussian_raw_xyz=gaussian_raw_xyz,
            )
        )
        scene_metric_summaries[scene_id] = {}
        funnel_summaries[scene_id] = {}
        for condition in REGISTERED_CONDITIONS:
            spec = scene["conditions"][condition]
            output_path = _resolve(
                base, spec.get("output"), name="condition.output"
            )
            diagnostics_path = _resolve(
                base, spec.get("diagnostics"), name="condition.diagnostics"
            )
            output = load_json(output_path)
            diagnostics = load_json(diagnostics_path)
            if not isinstance(output, Mapping) or not isinstance(diagnostics, Mapping):
                raise TypeError("output and diagnostics must contain JSON objects")
            if str(diagnostics.get("scene_id")) != scene_id or str(
                diagnostics.get("condition")
            ) != condition:
                raise ValueError("condition diagnostics identity mismatch")
            condition_identity_rows.append(
                audit_condition_run_identity(
                    scene_id=scene_id,
                    condition=condition,
                    spec=spec,
                    output_path=output_path,
                    diagnostics_path=diagnostics_path,
                    output=output,
                    diagnostics=diagnostics,
                    bank_dir=bank_dir,
                    bank=bank,
                )
            )
            candidates, predictions = _output_candidates(
                payload=output,
                scene_id=scene_id,
                condition=condition,
                gaussian_count=bank.point_count,
                nearest=nearest,
                class_names=class_names,
            )
            predictions_by_condition[condition].extend(predictions)
            candidate_metrics = evaluate_candidate_set_three_spaces(
                candidates=candidates,
                gt_objects=gt_objects,
                nearest=nearest,
                min_region_size=min_region_size,
            )
            metric_rows.extend(
                _flatten_candidate_rows(
                    scene_id=scene_id,
                    condition=condition,
                    result=candidate_metrics,
                )
            )
            scene_metric_summaries[scene_id][condition] = {
                "subsets": candidate_metrics["subsets"],
                "gt_to_gaussian_scene_coverage": candidate_metrics[
                    "gt_to_gaussian_scene_coverage"
                ],
                "official_gt_count": candidate_metrics["official_gt_count"],
                "official_tiny_small_gt_count": candidate_metrics[
                    "official_tiny_small_gt_count"
                ],
            }

            funnel = audit_frozen_clean_scene(
                bank_dir=bank_dir,
                diagnostics_path=diagnostics_path,
                output_path=output_path,
                allowed_classes=allowed_classes,
                metric_callback=_funnel_metric_callback(
                    gt_objects=gt_objects,
                    nearest=nearest,
                    min_region_size=min_region_size,
                ),
            )
            funnel_summaries[scene_id][condition] = funnel.to_summary()
            previous: Mapping[str, Any] | None = None
            for stage in funnel.stages:
                summary = stage.summary
                metrics = dict(stage.metrics)
                row: dict[str, Any] = {
                    "scene_id": scene_id,
                    "condition": condition,
                    "stage": stage.name,
                    **summary,
                    **metrics,
                    # Parquet schemas cannot reliably unify arbitrary nested
                    # dictionaries emitted by different stages.  Preserve the
                    # complete value as canonical, queryable JSON instead.
                    "details_json": json.dumps(
                        dict(stage.details),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
                if stage.available and previous is not None:
                    for key in (
                        "object_count",
                        "assignment_count",
                        "all_geometry_025_match_count",
                        "all_geometry_050_match_count",
                    ):
                        if key in row and key in previous:
                            row[f"loss_from_previous_{key}"] = int(previous[key]) - int(
                                row[key]
                            )
                funnel_rows.append(row)
                previous = row if stage.available else None
            equivalence = funnel.final_equivalence
            exact = bool(
                equivalence is not None
                and equivalence.equivalent
                and equivalence.changed_points == 0
                and equivalence.class_exact
                and equivalence.reconstructed_instance_count
                == equivalence.frozen_instance_count
            )
            funnel_gate_rows.append(
                {
                    "scene_id": scene_id,
                    "condition": condition,
                    "passed": exact,
                    "equivalence": (
                        None if equivalence is None else equivalence.to_dict()
                    ),
                }
            )

    protocols = {
        condition: evaluate_dual_protocols(
            ground_truth_scenes,
            predictions_by_condition[condition],
            class_names,
            min_region_size=min_region_size,
        )
        for condition in REGISTERED_CONDITIONS
    }
    gt_parity = evaluate_gt_as_prediction_dual_protocols(
        ground_truth_scenes, class_names, min_region_size=min_region_size
    )
    parity_rows: list[dict[str, Any]] = []
    for condition in REGISTERED_CONDITIONS:
        observed = {
            "official_ap25": protocols[condition]["official_9"]["aggregate"][
                "map_0.25"
            ],
            "official_ap50": protocols[condition]["official_9"]["aggregate"][
                "map_0.50"
            ],
            "historical_map_50_95": protocols[condition]["historical_10"][
                "aggregate"
            ]["map_50_95"],
        }
        for metric_name, expected_value in expected[condition].items():
            value = observed[metric_name]
            if value is None:
                passed = False
                difference = None
            else:
                difference = float(value) - float(expected_value)
                passed = math.isclose(
                    float(value),
                    float(expected_value),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
            parity_rows.append(
                {
                    "condition": condition,
                    "metric": metric_name,
                    "expected": float(expected_value),
                    "observed": value,
                    "difference": difference,
                    "passed": passed,
                }
            )

    after = build_file_manifest(protected)
    frozen_unchanged = _same_file_manifests(before, after)
    gates = {
        "gt_as_prediction_unit": bool(gt_parity["gt_as_prediction_parity"]),
        "formal_and_historical_metric_parity": all(
            bool(row["passed"]) for row in parity_rows
        ),
        "gt_to_gaussian_5cm_at_least_0.90": all(
            bool(row["passed"]) for row in mapping_gate_rows
        ),
        "stage_final_partition_exact": all(
            bool(row["passed"]) for row in funnel_gate_rows
        ),
        "registered_evidence_identity_exact": all(
            bool(row["passed"]) for row in evidence_registration_rows
        ),
        "condition_run_identity_exact": all(
            bool(row["passed"]) for row in condition_identity_rows
        ),
        "frozen_inputs_unchanged": frozen_unchanged,
    }
    gates["passed"] = all(gates.values())
    metric_output = {
        "schema": METRIC_OUTPUT_SCHEMA,
        "audit_identity": audit_identity,
        "scene_ids": scene_ids,
        "conditions": list(REGISTERED_CONDITIONS),
        "protocols": protocols,
        "gt_as_prediction": gt_parity,
        "scene_candidate_metrics": scene_metric_summaries,
        "mapping_gate_rows": mapping_gate_rows,
        "evidence_registration_rows": evidence_registration_rows,
        "condition_identity_rows": condition_identity_rows,
        "metric_parity_rows": parity_rows,
        "frozen_input_manifest_before": before,
        "frozen_input_manifest_after": after,
        "frozen_inputs_unchanged": frozen_unchanged,
        "technical_gates": gates,
    }
    funnel_output = {
        "schema": FUNNEL_OUTPUT_SCHEMA,
        "audit_identity": audit_identity,
        "scene_ids": scene_ids,
        "conditions": list(REGISTERED_CONDITIONS),
        "funnels": funnel_summaries,
        "final_partition_gate_rows": funnel_gate_rows,
        "technical_gate_passed": gates["stage_final_partition_exact"],
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / "clean_metric_reaudit_dev8.parquet", metric_rows)
    write_json(destination / "clean_metric_reaudit_dev8.json", metric_output)
    write_rows(destination / "clean_stage_funnel_dev8.parquet", funnel_rows)
    write_json(destination / "clean_stage_funnel_dev8.json", funnel_output)
    return {
        "metric": metric_output,
        "funnel": funnel_output,
        "technical_gates": gates,
        "output_dir": str(destination),
    }


__all__ = [
    "AUDIT_IDENTITY_SCHEMA",
    "DEFAULT_DEV8_SCENE_COUNT",
    "FUNNEL_OUTPUT_SCHEMA",
    "MANIFEST_SCHEMA",
    "METRIC_OUTPUT_SCHEMA",
    "REGISTERED_CONDITIONS",
    "REGISTERED_DEV2_SCENE_IDS",
    "REGISTERED_DEV8_SCENE_IDS",
    "audit_condition_run_identity",
    "audit_clean_baseline_manifest",
    "preflight_audit_output_directory",
]
