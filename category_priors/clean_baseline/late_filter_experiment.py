from __future__ import annotations

"""Offline orchestration and evaluation for the registered late-filter audit.

This module is the only layer in the late-filter experiment that may read
ground truth.  The replay implementation is imported lazily and is called
with only the frozen evidence bank, production diagnostics, the registered
taxonomy, and (for the historical arm) its frozen output.  Keeping that
boundary explicit prevents a diagnostic IoU from leaking into membership,
pruning, ownership, or classification.

The input is the mask-contract manifest emitted by ``run-clean-baseline-two-
step``.  H' (``H-hierarchy``) is the preregistered primary analysis; P
(``P-flat``) is retained as a sensitivity analysis because one P lifting had
already failed the byte-repeat contract.  Candidate geometry is evaluated in
the three disjoint spaces registered in :mod:`metric_reaudit`.  B0 arms can
overlap and/or abstain semantically, so they never receive official AP.
"""

import hashlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
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
    write_json,
    write_rows,
)
from ..prediction_contract import validate_prediction_contract
from ..scannet import physical_scene_id
from .evidence import (
    EVIDENCE_ARRAY_FILE,
    EVIDENCE_DIAGNOSTICS_FILE,
    EVIDENCE_METADATA_FILE,
    load_evidence_bank,
)
from .evaluation import CleanCandidate, ground_truth_objects_from_arrays
from .metric_reaudit import (
    FORMAL_RADIUS_M,
    build_bidirectional_nearest,
    evaluate_candidate_set_three_spaces,
    evaluate_dual_protocols,
    formal_gt_point_mask,
)
from .stage_funnel import FunnelObject, audit_frozen_clean_scene
from .two_step_audit import (
    MANIFEST_SCHEMA,
    REGISTERED_DEV2_SCENE_IDS,
    _size_boundaries,
    _taxonomy,
    _tiny_small_ids,
    _transform,
)


LATE_FILTER_FACTORIAL_SCHEMA = "saga-clean-late-filter-factorial-dev2-v1"
LATE_FILTER_ROW_SCHEMA = "saga-clean-late-filter-factorial-row-v1"
LATE_FILTER_ANALYSIS_SCHEMA = "saga-clean-late-filter-analysis-dev2-v1"

MASK_MODES = ("H-hierarchy", "P-flat")
PRIMARY_MASK_MODE = "H-hierarchy"
SENSITIVITY_MASK_MODE = "P-flat"
CONDITION = "C0-no-prior"

ARM_CODES = ("A1B1", "A0B1", "A1B0", "A0B0")
ARM_NAMES = {
    "A1B1": "current",
    "A0B1": "no-detection-hard-filter",
    "A1B0": "pre-late-pruning",
    "A0B0": "both-relaxed",
}
RELAXED_ARM_CODES = ("A0B1", "A1B0", "A0B0")

OUTPUT_TABLE = "clean_late_filter_factorial_dev2.parquet"
OUTPUT_JSON = "clean_late_filter_factorial_dev2.json"
OUTPUT_ANALYSIS = "clean_late_filter_analysis_dev2.json"


ReplayCallable = Callable[..., Any]


def _default_replay_callable() -> ReplayCallable:
    # Kept behind one lazy adapter so the replay module remains GT/prior-free
    # and can be developed/tested independently from this evaluator.
    from .late_filter_audit import replay_late_filter_factorial

    return replay_late_filter_factorial


def _resolve(base: Path, value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be an explicit non-empty path")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(path).resolve()
    payload = load_json(manifest_path)
    if not isinstance(payload, dict) or payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 2:
        raise ValueError("late-filter manifest must contain exactly DEV2")
    scene_ids = tuple(str(row.get("scene_id", "")) for row in scenes)
    if scene_ids != REGISTERED_DEV2_SCENE_IDS:
        raise ValueError("late-filter scenes must exactly match registered DEV2 order")
    if tuple(map(str, payload.get("dev2_scene_ids", ()))) != REGISTERED_DEV2_SCENE_IDS:
        raise ValueError("manifest dev2_scene_ids differ from registered DEV2")
    if int(payload.get("min_region_size", -1)) != 100:
        raise ValueError("late-filter manifest min_region_size must equal 100")
    _taxonomy(payload)
    _size_boundaries(payload)
    for scene in scenes:
        specs = scene.get("mask_control_conditions")
        if not isinstance(specs, Mapping) or set(map(str, specs)) != set(MASK_MODES):
            raise ValueError(
                "each DEV2 scene must register exactly H-hierarchy and P-flat"
            )
    return manifest_path, payload


def _mask_specs(scene: Mapping[str, Any]) -> Mapping[str, Any]:
    value = scene.get("mask_control_conditions")
    if not isinstance(value, Mapping):  # guarded by _load_manifest
        raise TypeError("mask_control_conditions must be a mapping")
    return value


def _flat_full_repeat_pass(
    *, base: Path, scene: Mapping[str, Any], scene_id: str
) -> bool:
    audit_path = _resolve(
        base,
        scene.get("flat_full_repeat_audit"),
        name="scene.flat_full_repeat_audit",
    )
    payload = load_json(audit_path)
    if not isinstance(payload, Mapping):
        raise TypeError("flat_full_repeat_audit must contain a JSON object")
    if payload.get("schema") != "saga-clean-flat-full-repeat-v1":
        raise ValueError("flat_full_repeat_audit has an incompatible schema")
    if str(payload.get("scene_id", "")) != str(scene_id):
        raise ValueError("flat_full_repeat_audit scene identity mismatch")
    primary = payload.get("primary")
    flat_spec = _mask_specs(scene)[SENSITIVITY_MASK_MODE]
    if not isinstance(primary, Mapping) or not isinstance(flat_spec, Mapping):
        raise TypeError("flat repeat primary and P-flat spec must be mappings")
    for key in ("bank_dir", "output", "diagnostics"):
        if _resolve(base, primary.get(key), name=f"repeat.primary.{key}") != _resolve(
            base, flat_spec.get(key), name=f"P-flat.{key}"
        ):
            raise ValueError("flat_full_repeat_audit is not bound to the P-flat input")
    return payload.get("passed") is True


def _protected_files(
    manifest_path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, ...]:
    base = manifest_path.parent
    files: set[Path] = {manifest_path}
    for scene in payload["scenes"]:
        files.add(_resolve(base, scene.get("gt_npz"), name="scene.gt_npz"))
        files.add(
            _resolve(base, scene.get("gaussian_ply"), name="scene.gaussian_ply")
        )
        for mask_mode, spec in _mask_specs(scene).items():
            if not isinstance(spec, Mapping):
                raise TypeError(f"{mask_mode} specification must be a mapping")
            bank_dir = _resolve(
                base, spec.get("bank_dir"), name=f"{mask_mode}.bank_dir"
            )
            if not bank_dir.is_dir():
                raise NotADirectoryError(bank_dir)
            for name in (
                EVIDENCE_ARRAY_FILE,
                EVIDENCE_METADATA_FILE,
                EVIDENCE_DIAGNOSTICS_FILE,
            ):
                source = bank_dir / name
                if not source.is_file():
                    raise FileNotFoundError(source)
                files.add(source.resolve())
            files.add(_resolve(base, spec.get("output"), name=f"{mask_mode}.output"))
            files.add(
                _resolve(
                    base,
                    spec.get("diagnostics"),
                    name=f"{mask_mode}.diagnostics",
                )
            )
        files.add(
            _resolve(
                base,
                scene.get("flat_full_repeat_audit"),
                name="scene.flat_full_repeat_audit",
            )
        )
    return tuple(sorted(files, key=str))


def _preflight_output(output_dir: Path, protected: Sequence[Path]) -> None:
    destination = output_dir.resolve()
    registered_outputs = {
        (destination / OUTPUT_TABLE).resolve(),
        (destination / OUTPUT_JSON).resolve(),
        (destination / OUTPUT_ANALYSIS).resolve(),
    }
    if any(path in set(map(Path.resolve, protected)) for path in registered_outputs):
        raise ValueError("late-filter outputs overlap frozen inputs")
    # A result directory inside an evidence bank or frozen condition directory
    # would make the before/after identity depend on our own writes.
    source_directories = {
        path.parent.resolve()
        for path in protected
        if path.name
        in {
            EVIDENCE_ARRAY_FILE,
            EVIDENCE_METADATA_FILE,
            EVIDENCE_DIAGNOSTICS_FILE,
            "output.json",
            "diagnostics.json",
        }
    }
    for source in source_directories:
        if destination == source or destination.is_relative_to(source):
            raise ValueError("output directory must be separate from frozen artifacts")


def _implementation_manifest() -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parent
    package_root = root.parent
    return build_file_manifest(
        (
            Path(__file__),
            root / "late_filter_audit.py",
            root / "stage_funnel.py",
            root / "metric_reaudit.py",
            root / "consensus.py",
            root / "pipeline.py",
            root / "models.py",
            root / "evidence.py",
            root / "evaluation.py",
            root / "two_step_audit.py",
            package_root / "evaluator.py",
            package_root / "prediction_contract.py",
            package_root / "io.py",
            package_root / "scannet.py",
        ),
        root=package_root,
    )


def _table_rows_equivalent(
    parquet_rows: Sequence[Mapping[str, Any]],
    json_rows: Sequence[Mapping[str, Any]],
) -> bool:
    keys = sorted(
        {
            str(key)
            for rows in (parquet_rows, json_rows)
            for row in rows
            for key in row
        }
    )

    def normalize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in rows:
            normalized: dict[str, Any] = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and math.isnan(value):
                    value = None
                normalized[key] = value
            result.append(normalized)
        return result
    return normalize(parquet_rows) == normalize(json_rows)


def _load_complete_result(
    destination: Path, *, input_identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Reuse only a complete result bound to the same data and code."""

    try:
        factorial = load_json(destination / OUTPUT_JSON)
        analysis = load_json(destination / OUTPUT_ANALYSIS)
        rows = read_rows(destination / OUTPUT_TABLE)
        if (
            not isinstance(factorial, Mapping)
            or factorial.get("schema") != LATE_FILTER_FACTORIAL_SCHEMA
            or factorial.get("input_identity") != dict(input_identity)
            or not isinstance(analysis, Mapping)
            or analysis.get("schema") != LATE_FILTER_ANALYSIS_SCHEMA
            or analysis.get("input_identity_sha256")
            != input_identity.get("content_sha256")
            or len(rows) != 16
            or len(factorial.get("rows", ())) != 16
            or not _table_rows_equivalent(rows, factorial["rows"])
        ):
            return None
        result = dict(analysis)
        result["runner_status"] = "skipped-complete"
        return result
    except (OSError, TypeError, ValueError, KeyError):
        return None


def _callable_boundary_pass(replay_callable: ReplayCallable) -> tuple[bool, list[str]]:
    """Check that the replay API itself exposes no GT/prior parameter."""

    try:
        parameters = inspect.signature(replay_callable).parameters
    except (TypeError, ValueError):
        return False, ["uninspectable-replay-callable"]
    forbidden = []
    for name in parameters:
        normalized = str(name).lower()
        if normalized in {"gt", "ground_truth", "prior", "priors"} or normalized.startswith(
            ("gt_", "ground_truth_", "prior_")
        ):
            forbidden.append(str(name))
    return not forbidden, sorted(forbidden)


def _metadata_scalar(metadata: Mapping[str, Any], names: Sequence[str], default: float) -> float:
    for name in names:
        if name in metadata:
            value = float(metadata[name])
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                return value
            raise ValueError(f"object metadata {name} must be finite and in [0, 1]")
    return float(default)


def _candidate_from_object(value: FunnelObject) -> CleanCandidate:
    metadata = value.metadata
    winner = _metadata_scalar(metadata, ("winner_probability",), 0.0)
    view = _metadata_scalar(
        metadata,
        ("mean_view_consensus", "view_consensus"),
        0.0,
    )
    detection = _metadata_scalar(
        metadata,
        ("mean_detection_ratio", "detection_ratio"),
        0.0,
    )
    return CleanCandidate(
        object_id=value.stable_id,
        gaussian_ids=value.gaussian_ids,
        class_id=value.class_name,
        winner_probability=winner,
        view_consensus=view,
        detection_ratio=detection,
    )


def _prediction_from_candidate(
    candidate: CleanCandidate,
    *,
    scene_id: str,
    instance_id: int,
    class_to_id: Mapping[str, int],
    nearest: Any,
    score: float | None = None,
) -> PredictedInstance:
    class_name = None if candidate.class_id is None else str(candidate.class_id)
    if class_name not in class_to_id:
        raise ValueError("formal prediction has no registered SAGA20 class")
    return PredictedInstance(
        scene_id=str(scene_id),
        instance_id=int(instance_id),
        class_id=int(class_to_id[class_name]),
        score=float(candidate.score if score is None else score),
        mask=formal_gt_point_mask(candidate.gaussian_ids, nearest),
    )


def _frozen_output_objects(
    payload: Mapping[str, Any], *, point_count: int
) -> tuple[FunnelObject, ...]:
    labels = np.asarray(payload.get("point_labels"))
    instances = payload.get("instances")
    if labels.shape != (int(point_count),) or labels.dtype.kind not in "iu":
        raise ValueError("frozen point_labels have the wrong shape or dtype")
    if not isinstance(instances, Mapping):
        raise TypeError("frozen instances must be a mapping")
    validate_prediction_contract(labels, instances)
    rows: list[FunnelObject] = []
    for raw_id in sorted(instances, key=lambda value: int(value)):
        instance_id = int(raw_id)
        metadata = instances[raw_id]
        if not isinstance(metadata, Mapping):
            raise TypeError("frozen instance metadata must be a mapping")
        rows.append(
            FunnelObject(
                stable_id=f"frozen:{instance_id}",
                gaussian_ids=np.flatnonzero(labels == instance_id),
                class_name=str(metadata["class"]),
                metadata={
                    "score": float(metadata["score"]),
                    "winner_probability": float(
                        metadata.get("winner_probability", metadata["score"])
                    ),
                },
            )
        )
    return tuple(rows)


def _object_digest(
    objects: Sequence[FunnelObject], *, include_class: bool
) -> tuple[tuple[str, int, str | None], ...]:
    rows: list[tuple[str, int, str | None]] = []
    for item in objects:
        ids = np.asarray(item.gaussian_ids, dtype="<i8")
        digest = hashlib.sha256(ids.tobytes(order="C")).hexdigest()
        rows.append(
            (
                digest,
                int(len(ids)),
                str(item.class_name) if include_class and item.class_name is not None else None,
            )
        )
    return tuple(sorted(rows))


def _maximum_overlap_score_equivalence(
    reconstructed: Sequence[FunnelObject],
    frozen: Sequence[FunnelObject],
) -> dict[str, Any]:
    """Compare formal scores after deterministic maximum-overlap matching.

    Formal AP is always computed from the frozen output score.  The replayed
    score is a reconstruction check, and the historical producer can differ
    by one binary64 rounding unit when otherwise equivalent reductions are
    evaluated on a different NumPy/libm build.  Preserve bit equality as a
    diagnostic, but use a strict one-ULP contract rather than a broad decimal
    tolerance.
    """

    left_rows = tuple(reconstructed)
    right_rows = tuple(frozen)
    if len(left_rows) != len(right_rows):
        return {
            "passed": False,
            "bit_exact": False,
            "matched_instance_count": 0,
            "max_abs_error": None,
            "max_ulp_distance": None,
            "nonexact_score_count": None,
            "maximum_allowed_ulp_distance": 1,
            "missing_or_nonfinite_score": True,
            "out_of_range_score": False,
        }
    if not left_rows:
        return {
            "passed": True,
            "bit_exact": True,
            "matched_instance_count": 0,
            "max_abs_error": 0.0,
            "max_ulp_distance": 0,
            "nonexact_score_count": 0,
            "maximum_allowed_ulp_distance": 1,
            "missing_or_nonfinite_score": False,
            "out_of_range_score": False,
        }
    overlap = np.zeros((len(left_rows), len(right_rows)), dtype=np.int64)
    for left_index, left in enumerate(left_rows):
        for right_index, right in enumerate(right_rows):
            overlap[left_index, right_index] = int(
                np.intersect1d(
                    left.gaussian_ids,
                    right.gaussian_ids,
                    assume_unique=True,
                ).size
            )
    from scipy.optimize import linear_sum_assignment

    left_indices, right_indices = linear_sum_assignment(-overlap)
    within_one_ulp = len(left_indices) == len(left_rows)
    bit_exact = within_one_ulp
    missing = False
    out_of_range = False
    deltas: list[float] = []
    ulp_distances: list[int] = []
    nonexact_count = 0
    for left_index, right_index in zip(left_indices, right_indices):
        left = left_rows[int(left_index)]
        right = right_rows[int(right_index)]
        if "score" not in left.metadata or "score" not in right.metadata:
            missing = True
            within_one_ulp = False
            bit_exact = False
            continue
        left_score = float(left.metadata["score"])
        right_score = float(right.metadata["score"])
        if not math.isfinite(left_score) or not math.isfinite(right_score):
            missing = True
            within_one_ulp = False
            bit_exact = False
            continue
        if not (0.0 <= left_score <= 1.0 and 0.0 <= right_score <= 1.0):
            out_of_range = True
            within_one_ulp = False
            bit_exact = False
            continue
        delta = abs(left_score - right_score)
        deltas.append(delta)
        if left_score == right_score:
            ulp_distance = 0
        else:
            left_bits = int(np.float64(left_score).view(np.uint64))
            right_bits = int(np.float64(right_score).view(np.uint64))
            ulp_distance = abs(left_bits - right_bits)
            nonexact_count += 1
            bit_exact = False
        ulp_distances.append(ulp_distance)
        if ulp_distance > 1:
            within_one_ulp = False
    return {
        "passed": bool(within_one_ulp and not missing and not out_of_range),
        "bit_exact": bool(bit_exact and not missing and not out_of_range),
        "matched_instance_count": len(left_indices),
        "max_abs_error": max(deltas, default=0.0),
        "max_ulp_distance": max(ulp_distances, default=0),
        "nonexact_score_count": int(nonexact_count),
        "maximum_allowed_ulp_distance": 1,
        "missing_or_nonfinite_score": missing,
        "out_of_range_score": out_of_range,
    }


def _replay_fingerprint(result: Any) -> str:
    arms = getattr(result, "arms", None)
    if not isinstance(arms, Mapping):
        raise TypeError("late-filter replay result must expose an arms mapping")
    payload: dict[str, Any] = {
        "scene_id": str(getattr(result, "scene_id", "")),
        "condition": str(getattr(result, "condition", "")),
        "point_count": int(getattr(result, "point_count", -1)),
        "accepted_components": _object_digest(
            tuple(getattr(result, "accepted_components", ())), include_class=False
        ),
        "shared_identity": dict(getattr(result, "shared_identity", {})),
        "issues": list(getattr(result, "issues", ())),
        "arms": {},
    }
    for code in ARM_CODES:
        arm = arms.get(code)
        if arm is None:
            raise ValueError(f"late-filter replay omitted {code}")
        payload["arms"][code] = {
            "detection": _object_digest(
                tuple(getattr(arm, "detection_objects", ())), include_class=False
            ),
            "physical": _object_digest(
                tuple(getattr(arm, "physical_objects", ())), include_class=True
            ),
            "ownership": _object_digest(
                tuple(getattr(arm, "ownership_objects", ())), include_class=True
            ),
            "formal": (
                None
                if getattr(arm, "formal_output", None) is None
                else _object_digest(tuple(arm.formal_output), include_class=True)
            ),
            "drop_reasons": dict(getattr(arm, "drop_reasons", {})),
        }
    return hash_json(_json_safe(payload))


def _json_safe(value: Any) -> Any:
    """Normalize replay summaries without weakening their exact semantics."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _partition_equivalence_pass(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        row = value
    elif hasattr(value, "to_dict"):
        row = value.to_dict()
    else:
        row = {
            key: getattr(value, key, None)
            for key in (
                "equivalent",
                "changed_points",
                "class_exact",
                "point_count_exact",
                "reconstructed_instance_count",
                "frozen_instance_count",
            )
        }
    return bool(
        row.get("equivalent")
        and int(row.get("changed_points", -1)) == 0
        and row.get("class_exact") is True
        and row.get("point_count_exact", True) is True
        and int(row.get("reconstructed_instance_count", -1))
        == int(row.get("frozen_instance_count", -2))
    )


def _detection_identity_contract(replay: Any) -> dict[str, bool]:
    """Verify that A only changes the registered membership filter.

    This deliberately compares the replay objects to the immutable per-component
    evidence rather than comparing A0 and A1 to each other: their memberships
    are expected to differ, while their component identity and aligned ratio
    vector must remain shared.
    """

    evidence = tuple(getattr(replay, "detection_evidence", ()))
    arms = getattr(replay, "arms", {})
    stable_ids_unique = len({row.stable_id for row in evidence}) == len(evidence)
    membership_exact = True
    ratio_mean_exact = True
    for code in ARM_CODES:
        arm = arms.get(code)
        if arm is None:
            membership_exact = False
            ratio_mean_exact = False
            continue
        actual = {
            item.stable_id: item
            for item in tuple(getattr(arm, "detection_objects", ()))
        }
        hard = code.startswith("A1")
        expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for row in evidence:
            ids, ratios = row.selected(detection_hard_filter=hard)
            if ids.size:
                expected[row.stable_id] = (ids, ratios)
        if set(actual) != set(expected):
            membership_exact = False
            ratio_mean_exact = False
            continue
        for stable_id, (ids, ratios) in expected.items():
            item = actual[stable_id]
            if not np.array_equal(np.asarray(item.gaussian_ids), ids):
                membership_exact = False
            recorded = float(item.metadata.get("mean_detection_ratio", math.nan))
            if not math.isclose(
                recorded,
                float(np.mean(ratios)),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                ratio_mean_exact = False
    shared = getattr(replay, "shared_identity", {})
    shared_identity_complete = bool(
        isinstance(shared, Mapping)
        and {
            "active_mask_ids",
            "accepted_edge_count",
            "component_mask_ids",
            "component_count",
        }.issubset(shared)
    )
    return {
        "component_evidence_stable_ids_unique": stable_ids_unique,
        "a_membership_matches_shared_detection_evidence": membership_exact,
        "detection_ratio_alignment_and_mean_exact": ratio_mean_exact,
        "shared_identity_registry_complete": shared_identity_complete,
    }


def _matching_scalars(result: Mapping[str, Any], *, subset: str) -> dict[str, Any]:
    node = result["subsets"][subset]
    values: dict[str, Any] = {"candidate_count": int(node["candidate_count"])}
    for kind in ("geometry", "same_class"):
        for threshold in ("0.25", "0.50"):
            row = node["matching"][kind][threshold]
            prefix = f"{kind}_{threshold.replace('.', '')}"
            values[f"{prefix}_tp"] = int(row["true_positive_count"])
            values[f"{prefix}_fp"] = int(row["false_positive_count"])
            values[f"{prefix}_fn"] = int(row["false_negative_count"])
            values[f"{prefix}_precision"] = float(row["precision"])
            values[f"{prefix}_recall"] = float(row["recall"])
            values[f"{prefix}_tiny_small_recall"] = float(
                row["tiny_small_recall"]
            )
            values[f"{prefix}_matched_iou"] = float(row["total_matched_iou"])
    return values


def _gaussian_scalars(result: Mapping[str, Any]) -> dict[str, Any]:
    assignments = geometry_correct = mapped_other = unsupported = 0
    for candidate in result["candidate_rows"]:
        radius = candidate["radii"][f"{FORMAL_RADIUS_M:.2f}"]
        assignments += int(candidate["candidate_gaussian_count"])
        geometry_correct += int(radius["gaussian_geometry_target_instance_count"])
        mapped_other += int(radius["gaussian_mapped_other_instance_count"])
        unsupported += int(radius["gaussian_unsupported_count"])
    return {
        "gaussian_assignment_count_5cm": assignments,
        "gaussian_geometry_target_count_5cm": geometry_correct,
        "gaussian_mapped_other_count_5cm": mapped_other,
        "gaussian_unsupported_count_5cm": unsupported,
        "gaussian_geometry_precision_5cm": (
            float(geometry_correct / assignments) if assignments else 0.0
        ),
        "gaussian_pollution_fraction_5cm": (
            float((mapped_other + unsupported) / assignments)
            if assignments
            else 0.0
        ),
    }


def _candidate_metrics(
    objects: Sequence[FunnelObject],
    *,
    gt_objects: Sequence[Any],
    nearest: Any,
    min_region_size: int,
) -> dict[str, Any]:
    candidates = [_candidate_from_object(item) for item in objects]
    return evaluate_candidate_set_three_spaces(
        candidates=candidates,
        gt_objects=gt_objects,
        nearest=nearest,
        radii_m=(FORMAL_RADIUS_M,),
        min_region_size=int(min_region_size),
    )


def _nonempty_geometric_candidates(
    objects: Sequence[FunnelObject],
) -> tuple[tuple[FunnelObject, ...], dict[str, int]]:
    """Separate structural empty components from evaluable geometry.

    An accepted-edge component may legitimately have an empty full-support
    union.  It remains part of the replay lineage and its ``empty_full`` drop
    reason remains visible, but an empty set is not an object hypothesis and
    must never be constructed as :class:`CleanCandidate`.
    """

    rows = tuple(objects)
    candidates = tuple(item for item in rows if len(item.gaussian_ids) > 0)
    inventory = {
        "total_component_count": len(rows),
        "empty_full_component_count": len(rows) - len(candidates),
        "geometric_candidate_count": len(candidates),
    }
    if (
        inventory["total_component_count"]
        != inventory["empty_full_component_count"]
        + inventory["geometric_candidate_count"]
    ):
        raise AssertionError("accepted-component inventory is not conserved")
    return candidates, inventory


def _formal_predictions_from_frozen(
    payload: Mapping[str, Any],
    *,
    scene_id: str,
    point_count: int,
    nearest: Any,
    class_to_id: Mapping[str, int],
) -> list[PredictedInstance]:
    labels = np.asarray(payload["point_labels"], dtype=np.int64)
    if labels.shape != (int(point_count),):
        raise ValueError("frozen output point count differs from evidence bank")
    instances = payload["instances"]
    predictions: list[PredictedInstance] = []
    for raw_id in sorted(instances, key=lambda value: int(value)):
        instance_id = int(raw_id)
        metadata = instances[raw_id]
        class_name = str(metadata["class"])
        if class_name not in class_to_id:
            raise ValueError("frozen output contains a class outside SAGA20")
        predictions.append(
            PredictedInstance(
                scene_id=scene_id,
                instance_id=instance_id,
                class_id=int(class_to_id[class_name]),
                score=float(metadata["score"]),
                mask=formal_gt_point_mask(
                    np.flatnonzero(labels == instance_id), nearest
                ),
            )
        )
    return predictions


def _scene_row(
    *,
    scene_id: str,
    mask_mode: str,
    arm_code: str,
    arm: Any,
    accepted_metrics: Mapping[str, Any],
    selected_metrics: Mapping[str, Any],
    technical_contract: Mapping[str, Any],
    historical_lifting_repeat_pass: bool | None,
    official_protocols: Mapping[str, Any] | None,
) -> dict[str, Any]:
    accepted_all = _matching_scalars(accepted_metrics, subset="all")
    accepted_official = _matching_scalars(
        accepted_metrics, subset="official_evaluable"
    )
    selected_all = _matching_scalars(selected_metrics, subset="all")
    selected_official = _matching_scalars(
        selected_metrics, subset="official_evaluable"
    )
    row: dict[str, Any] = {
        "schema": LATE_FILTER_ROW_SCHEMA,
        "scene_id": scene_id,
        "physical_scene_id": physical_scene_id(scene_id),
        "mask_mode": mask_mode,
        "primary_mask_mode": mask_mode == PRIMARY_MASK_MODE,
        "arm_code": arm_code,
        "arm_name": ARM_NAMES[arm_code],
        "detection_hard_filter": bool(
            getattr(arm, "detection_hard_filter", arm_code.startswith("A1"))
        ),
        "strict_late_export": bool(
            getattr(arm, "strict_late_export", arm_code.endswith("B1"))
        ),
        "formal_output_allowed": bool(getattr(arm, "formal_output_allowed", False)),
        "official_ap_reported": official_protocols is not None,
        "historical_lifting_repeat_pass": historical_lifting_repeat_pass,
        "technical_contract_pass": bool(technical_contract.get("passed")),
        "drop_reasons_json": json.dumps(
            dict(getattr(arm, "drop_reasons", {})),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        **{f"accepted_all_{key}": value for key, value in accepted_all.items()},
        **{
            f"accepted_official_{key}": value
            for key, value in accepted_official.items()
        },
        **{f"all_{key}": value for key, value in selected_all.items()},
        **{f"official_{key}": value for key, value in selected_official.items()},
        **_gaussian_scalars(selected_metrics),
    }
    if official_protocols is not None:
        official = official_protocols["official_9"]["aggregate"]
        historical = official_protocols["historical_10"]["aggregate"]
        row.update(
            {
                "official_map_50_90": official["map_50_90"],
                "official_ap25": official["map_0.25"],
                "official_ap50": official["map_0.50"],
                "historical_map_50_95": historical["map_50_95"],
            }
        )
    return row


def _aggregate_arm_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise ValueError("DEV2 arm aggregation requires exactly two scene rows")
    candidate_count = sum(int(row["official_candidate_count"]) for row in rows)
    tp25 = sum(int(row["official_geometry_025_tp"]) for row in rows)
    fp25 = sum(int(row["official_geometry_025_fp"]) for row in rows)
    # Section 36 deliberately mixes scopes: the object-capacity count and
    # per-scene recall use all candidates, while precision is restricted to
    # official-evaluable predictions (>=100 projected GT points).
    tp50 = sum(int(row["all_geometry_050_tp"]) for row in rows)
    tiny_total = sum(int(row["official_tiny_small_gt_count"]) for row in rows)
    tiny_matched25 = sum(
        round(
            float(row["official_geometry_025_tiny_small_recall"])
            * int(row["official_tiny_small_gt_count"])
        )
        for row in rows
    )
    accepted_component_count = sum(
        int(row["accepted_component_count_total"]) for row in rows
    )
    accepted_empty_full_count = sum(
        int(row["accepted_empty_component_count"]) for row in rows
    )
    accepted_candidate_count = sum(int(row["accepted_all_candidate_count"]) for row in rows)
    accepted_tp25 = sum(
        int(row["accepted_all_geometry_025_tp"]) for row in rows
    )
    accepted_tp50 = sum(
        int(row["accepted_all_geometry_050_tp"]) for row in rows
    )
    accepted_tiny_matched25 = sum(
        round(
            float(row["accepted_official_geometry_025_tiny_small_recall"])
            * int(row["official_tiny_small_gt_count"])
        )
        for row in rows
    )
    return {
        "scene_count": len(rows),
        "accepted_component_count_total": accepted_component_count,
        "accepted_empty_component_count": accepted_empty_full_count,
        "accepted_component_geometric_candidate_count": accepted_candidate_count,
        "accepted_candidate_count": accepted_candidate_count,
        "accepted_geometry_025_tp": accepted_tp25,
        "accepted_geometry_025_precision": (
            float(accepted_tp25 / accepted_candidate_count)
            if accepted_candidate_count
            else 0.0
        ),
        "accepted_geometry_050_tp": accepted_tp50,
        "accepted_geometry_025_tiny_small_recall": (
            float(accepted_tiny_matched25 / tiny_total) if tiny_total else 0.0
        ),
        "candidate_count": candidate_count,
        "geometry_025_tp": tp25,
        "geometry_025_fp": fp25,
        "geometry_025_precision": float(tp25 / candidate_count)
        if candidate_count
        else 0.0,
        "geometry_050_tp": tp50,
        "official_tiny_small_gt_count": tiny_total,
        "geometry_025_tiny_small_matched": tiny_matched25,
        "geometry_025_tiny_small_recall": float(tiny_matched25 / tiny_total)
        if tiny_total
        else 0.0,
        "scene_geometry_025_recall": {
            str(row["scene_id"]): float(row["all_geometry_025_recall"])
            for row in rows
        },
        "all_technical_contracts_pass": all(
            bool(row["technical_contract_pass"]) for row in rows
        ),
    }


def _arm_scientific_gate(
    *, aggregate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    baseline_scene = baseline["scene_geometry_025_recall"]
    deltas = {
        scene_id: float(value) - float(baseline_scene[scene_id])
        for scene_id, value in aggregate["scene_geometry_025_recall"].items()
    }
    checks = {
        "at_least_6_geometry_iou_050": int(aggregate["geometry_050_tp"]) >= 6,
        "official_candidate_precision_025_at_least_0.10": float(
            aggregate["geometry_025_precision"]
        )
        >= 0.10,
        "tiny_small_recall_025_at_least_0.20": float(
            aggregate["geometry_025_tiny_small_recall"]
        )
        >= 0.20,
        "at_least_2_new_geometry_iou_050": int(aggregate["geometry_050_tp"])
        - int(baseline["geometry_050_tp"])
        >= 2,
        "at_least_one_scene_recall_025_improves": any(value > 0.0 for value in deltas.values()),
        "no_scene_recall_025_loses_more_than_0.05": all(
            value >= -0.05 for value in deltas.values()
        ),
        "technical_contracts_pass": bool(
            aggregate["all_technical_contracts_pass"]
        ),
    }
    return {
        "checks": checks,
        "scene_geometry_recall_025_deltas": deltas,
        "passed": all(checks.values()),
    }


def _attribution(passed: Mapping[str, bool]) -> str:
    detection = bool(passed.get("A0B1"))
    late = bool(passed.get("A1B0"))
    interaction = bool(passed.get("A0B0"))
    if detection and late:
        return "detection-and-late-pruning-independently-sufficient"
    if detection:
        return "detection-hard-filter-primary"
    if late:
        return "late-pruning-block-primary"
    if interaction:
        return "detection-and-late-pruning-interaction-required"
    return "accepted-components-or-earlier-insufficient"


def audit_clean_late_filters(
    manifest_path: str | Path,
    output_dir: str | Path,
    replay_callable: ReplayCallable | None = None,
) -> dict[str, Any]:
    """Run and evaluate the frozen DEV2 late-filter 2x2.

    The callable boundary is intentionally small and GT-free.  A test may
    inject a pure replay implementation; production lazily imports
    ``replay_late_filter_factorial``.
    """

    manifest_file, payload = _load_manifest(manifest_path)
    base = manifest_file.parent
    destination = Path(output_dir).resolve()
    protected = _protected_files(manifest_file, payload)
    _preflight_output(destination, protected)
    before = build_file_manifest(protected)
    input_identity = {
        "schema": "saga-clean-late-filter-input-identity-v1",
        "manifest": str(manifest_file),
        "files": before,
        "implementation": _implementation_manifest(),
    }
    input_identity["content_sha256"] = hash_json(input_identity)
    complete = _load_complete_result(
        destination,
        input_identity=input_identity,
    )
    if complete is not None:
        return complete
    runner = _default_replay_callable() if replay_callable is None else replay_callable
    boundary_pass, forbidden_parameters = _callable_boundary_pass(runner)
    class_names, allowed_classes = _taxonomy(payload)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    _, small_max_m, _ = _size_boundaries(payload)
    min_region_size = int(payload["min_region_size"])

    table_rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    ground_truth_scenes: list[GroundTruthScene] = []
    frozen_predictions_by_mode: dict[str, list[PredictedInstance]] = {
        mask_mode: [] for mask_mode in MASK_MODES
    }

    for scene in payload["scenes"]:
        scene_id = str(scene["scene_id"])
        gt_path = _resolve(base, scene.get("gt_npz"), name="scene.gt_npz")
        gaussian_path = _resolve(
            base, scene.get("gaussian_ply"), name="scene.gaussian_ply"
        )
        gt_xyz, gt_scene = load_ground_truth_npz(gt_path, scene_id)
        gaussian_xyz = apply_transform(
            load_ply_xyz(gaussian_path), _transform(scene.get("transform"))
        )
        nearest = build_bidirectional_nearest(gt_xyz, gaussian_xyz)
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
        nested[scene_id] = {}
        flat_full_repeat_pass = _flat_full_repeat_pass(
            base=base, scene=scene, scene_id=scene_id
        )

        for mask_mode in MASK_MODES:
            spec = _mask_specs(scene)[mask_mode]
            if not isinstance(spec, Mapping):
                raise TypeError(f"{scene_id}/{mask_mode} spec must be a mapping")
            bank_dir = _resolve(
                base, spec.get("bank_dir"), name=f"{mask_mode}.bank_dir"
            )
            output_path = _resolve(
                base, spec.get("output"), name=f"{mask_mode}.output"
            )
            diagnostics_path = _resolve(
                base,
                spec.get("diagnostics"),
                name=f"{mask_mode}.diagnostics",
            )
            bank = load_evidence_bank(
                bank_dir,
                expected_scene_id=scene_id,
                expected_point_count=len(gaussian_xyz),
            )
            output = load_json(output_path)
            diagnostics = load_json(diagnostics_path)
            if not isinstance(output, Mapping) or not isinstance(diagnostics, Mapping):
                raise TypeError("frozen output/diagnostics must be JSON objects")
            if str(output.get("scene_id")) != scene_id or str(
                diagnostics.get("scene_id")
            ) != scene_id:
                raise ValueError("frozen scene identity mismatch")
            if str(output.get("condition")) != CONDITION or str(
                diagnostics.get("condition")
            ) != CONDITION:
                raise ValueError("late-filter replay requires frozen C0-no-prior")

            replay = runner(
                bank=bank,
                diagnostics=diagnostics,
                allowed_classes=allowed_classes,
                frozen_output=output,
            )
            repeated = runner(
                bank=bank,
                diagnostics=diagnostics,
                allowed_classes=allowed_classes,
                frozen_output=output,
            )
            replay_digest = _replay_fingerprint(replay)
            repeat_digest = _replay_fingerprint(repeated)
            detection_identity_checks = _detection_identity_contract(replay)
            arms = getattr(replay, "arms", None)
            if not isinstance(arms, Mapping) or set(map(str, arms)) != set(ARM_CODES):
                raise ValueError("late-filter replay must return exactly four arms")
            accepted_components = tuple(
                getattr(replay, "accepted_components", ())
            )
            accepted_objects, accepted_inventory = _nonempty_geometric_candidates(
                accepted_components
            )
            accepted_metrics = _candidate_metrics(
                accepted_objects,
                gt_objects=gt_objects,
                nearest=nearest,
                min_region_size=min_region_size,
            )

            historical_funnel = audit_frozen_clean_scene(
                bank_dir=bank_dir,
                diagnostics_path=diagnostics_path,
                output_path=output_path,
                allowed_classes=allowed_classes,
            )
            historical_physical = historical_funnel.stage(
                "physical_split_and_deduplicated"
            ).objects
            frozen_objects = _frozen_output_objects(
                output, point_count=bank.point_count
            )
            score_equivalence = _maximum_overlap_score_equivalence(
                tuple(getattr(arms["A1B1"], "formal_output", ()) or ()),
                frozen_objects,
            )
            historical_lifting_repeat_pass = flat_full_repeat_pass
            if mask_mode == PRIMARY_MASK_MODE:
                # H' completed a full repeat; the manifest's flat flag belongs
                # only to P and must not downgrade the primary analysis.
                historical_lifting_repeat_pass = True

            scene_mode_rows: dict[str, Any] = {}
            for arm_code in ARM_CODES:
                arm = arms[arm_code]
                formal = getattr(arm, "formal_output", None)
                selected_objects = (
                    tuple(getattr(arm, "physical_objects", ()))
                    if formal is None
                    else tuple(formal)
                )
                selected_metrics = _candidate_metrics(
                    selected_objects,
                    gt_objects=gt_objects,
                    nearest=nearest,
                    min_region_size=min_region_size,
                )
                historical_equivalence = _partition_equivalence_pass(
                    getattr(arms["A1B1"], "final_equivalence", None)
                )
                b0_matches_physical = bool(
                    _object_digest(
                        tuple(getattr(arm, "physical_objects", ())),
                        include_class=False,
                    )
                    == _object_digest(
                        tuple(
                            getattr(
                                arms[
                                    "A1B1"
                                    if arm_code.startswith("A1")
                                    else "A0B1"
                                ],
                                "physical_objects",
                                (),
                            )
                        ),
                        include_class=False,
                    )
                ) if arm_code.endswith("B0") else True
                historical_physical_exact = bool(
                    _object_digest(
                        tuple(getattr(arms["A1B0"], "physical_objects", ())),
                        include_class=False,
                    )
                    == _object_digest(historical_physical, include_class=False)
                )
                a1b1_partition_exact = bool(
                    historical_equivalence
                    and _object_digest(
                        tuple(getattr(arms["A1B1"], "formal_output", ()) or ()),
                        include_class=True,
                    )
                    == _object_digest(frozen_objects, include_class=True)
                )
                issues = tuple(getattr(replay, "issues", ()))
                technical_checks = {
                    "replay_callable_has_no_gt_or_prior_parameter": boundary_pass,
                    "scene_identity_exact": str(getattr(replay, "scene_id", ""))
                    == scene_id,
                    "condition_identity_exact": str(
                        getattr(replay, "condition", "")
                    )
                    == CONDITION,
                    "point_count_exact": int(getattr(replay, "point_count", -1))
                    == int(bank.point_count),
                    "no_replay_issues": not issues,
                    "repeat_replay_byte_semantics_exact": replay_digest
                    == repeat_digest,
                    "a1b1_frozen_partition_exact": a1b1_partition_exact,
                    "a1b1_frozen_score_within_1ulp": bool(
                        score_equivalence["passed"]
                    ),
                    "a1b0_historical_physical_stage_exact": historical_physical_exact,
                    "b0_equals_corresponding_physical_candidates": b0_matches_physical,
                    "b0_has_no_formal_output": bool(
                        not arm_code.endswith("B0")
                        or (
                            getattr(arm, "formal_output", None) is None
                            and not bool(getattr(arm, "formal_output_allowed", False))
                        )
                    ),
                    "empty_accepted_components_are_diagnostic_only": bool(
                        int(
                            accepted_metrics["subsets"]["all"][
                                "candidate_count"
                            ]
                        )
                        == accepted_inventory["geometric_candidate_count"]
                        and accepted_inventory["total_component_count"]
                        == accepted_inventory["empty_full_component_count"]
                        + accepted_inventory["geometric_candidate_count"]
                    ),
                    **detection_identity_checks,
                }
                technical_contract = {
                    "checks": technical_checks,
                    "passed": all(technical_checks.values()),
                    "replay_digest": replay_digest,
                    "repeat_digest": repeat_digest,
                    "issues": list(issues),
                    "a1b1_score_equivalence": score_equivalence,
                }

                official_protocols = None
                # A1B1 is the only arm with a registered frozen ranking score.
                # Formal AP always reads that frozen value; replay parity is a
                # separate one-ULP technical check.  A0B1 remains candidate-
                # level unless its complete formal score is registered later.
                if arm_code == "A1B1":
                    scene_predictions = _formal_predictions_from_frozen(
                        output,
                        scene_id=scene_id,
                        point_count=bank.point_count,
                        nearest=nearest,
                        class_to_id=class_to_id,
                    )
                    frozen_predictions_by_mode[mask_mode].extend(scene_predictions)
                    official_protocols = evaluate_dual_protocols(
                        [gt_scene],
                        scene_predictions,
                        class_names,
                        min_region_size=min_region_size,
                    )

                row = _scene_row(
                    scene_id=scene_id,
                    mask_mode=mask_mode,
                    arm_code=arm_code,
                    arm=arm,
                    accepted_metrics=accepted_metrics,
                    selected_metrics=selected_metrics,
                    technical_contract=technical_contract,
                    historical_lifting_repeat_pass=(
                        None
                        if historical_lifting_repeat_pass is None
                        else bool(historical_lifting_repeat_pass)
                    ),
                    official_protocols=official_protocols,
                )
                row["official_gt_count"] = int(selected_metrics["official_gt_count"])
                row["official_tiny_small_gt_count"] = int(
                    selected_metrics["official_tiny_small_gt_count"]
                )
                row["accepted_component_count_total"] = int(
                    accepted_inventory["total_component_count"]
                )
                row["accepted_empty_component_count"] = int(
                    accepted_inventory["empty_full_component_count"]
                )
                row["accepted_component_geometric_candidate_count"] = int(
                    accepted_inventory["geometric_candidate_count"]
                )
                table_rows.append(row)
                scene_mode_rows[arm_code] = {
                    "row": row,
                    "technical_contract": technical_contract,
                    "candidate_metrics": selected_metrics,
                    "drop_reasons": dict(getattr(arm, "drop_reasons", {})),
                }

            nested[scene_id][mask_mode] = {
                "accepted_component_metrics": accepted_metrics,
                "accepted_component_inventory": accepted_inventory,
                "shared_identity": _json_safe(
                    dict(getattr(replay, "shared_identity", {}))
                ),
                "detection_identity_contract": detection_identity_checks,
                "replay_issues": list(getattr(replay, "issues", ())),
                "arms": scene_mode_rows,
            }

    after = build_file_manifest(protected)
    frozen_inputs_unchanged = before == after
    rows_by_mode_arm: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in table_rows:
        rows_by_mode_arm.setdefault((row["mask_mode"], row["arm_code"]), []).append(
            row
        )
    aggregates: dict[str, dict[str, Any]] = {}
    scientific_gates: dict[str, dict[str, Any]] = {}
    for mask_mode in MASK_MODES:
        aggregates[mask_mode] = {
            arm_code: _aggregate_arm_rows(rows_by_mode_arm[(mask_mode, arm_code)])
            for arm_code in ARM_CODES
        }
        baseline = aggregates[mask_mode]["A1B1"]
        scientific_gates[mask_mode] = {
            arm_code: _arm_scientific_gate(
                aggregate=aggregates[mask_mode][arm_code], baseline=baseline
            )
            for arm_code in RELAXED_ARM_CODES
        }

    primary_passed = {
        arm: bool(scientific_gates[PRIMARY_MASK_MODE][arm]["passed"])
        for arm in RELAXED_ARM_CODES
    }
    sensitivity_passed = {
        arm: bool(scientific_gates[SENSITIVITY_MASK_MODE][arm]["passed"])
        for arm in RELAXED_ARM_CODES
    }
    exporter_authorized = any(primary_passed.values())
    p_only = not exporter_authorized and any(sensitivity_passed.values())
    technical_gates = {
        "replay_callable_has_no_gt_or_prior_parameter": boundary_pass,
        "frozen_inputs_unchanged": frozen_inputs_unchanged,
        "all_hierarchy_replay_contracts_pass": all(
            bool(row["technical_contract_pass"])
            for row in table_rows
            if row["mask_mode"] == PRIMARY_MASK_MODE
        ),
        "all_flat_sensitivity_replay_contracts_pass": all(
            bool(row["technical_contract_pass"])
            for row in table_rows
            if row["mask_mode"] == SENSITIVITY_MASK_MODE
        ),
        "b0_official_ap_never_reported": all(
            not bool(row["official_ap_reported"])
            for row in table_rows
            if not bool(row["strict_late_export"])
        ),
        "only_a1b1_has_registered_official_ap": all(
            bool(row["official_ap_reported"]) == (row["arm_code"] == "A1B1")
            for row in table_rows
        ),
    }
    # P is a sensitivity arm with a known lifting repeat failure.  Its replay
    # contracts remain visible, but cannot veto a technically sound H' primary
    # analysis and can never authorize the exporter by itself.
    technical_gates["primary_passed"] = all(
        bool(technical_gates[key])
        for key in (
            "replay_callable_has_no_gt_or_prior_parameter",
            "frozen_inputs_unchanged",
            "all_hierarchy_replay_contracts_pass",
            "b0_official_ap_never_reported",
            "only_a1b1_has_registered_official_ap",
        )
    )
    technical_gates["passed"] = technical_gates["primary_passed"]

    aggregate_current_protocols = {
        mask_mode: evaluate_dual_protocols(
            ground_truth_scenes,
            frozen_predictions_by_mode[mask_mode],
            class_names,
            min_region_size=min_region_size,
        )
        for mask_mode in MASK_MODES
    }
    attribution = _attribution(primary_passed)
    if not technical_gates["passed"]:
        decision = "stop-technical-contract-failed"
        exporter_authorized = False
    elif exporter_authorized:
        decision = "proceed-to-conditional-clean-exporter"
    elif p_only:
        decision = "stop-p-only-requires-deterministic-lifting"
    else:
        decision = "stop-accepted-components-or-earlier-insufficient"

    factorial = {
        "schema": LATE_FILTER_FACTORIAL_SCHEMA,
        "manifest": str(manifest_file),
        "scene_ids": list(REGISTERED_DEV2_SCENE_IDS),
        "physical_scene_is_independent_unit": True,
        "candidate_and_stage_rows_are_nested_diagnostics": True,
        "mask_modes": list(MASK_MODES),
        "primary_mask_mode": PRIMARY_MASK_MODE,
        "sensitivity_mask_mode": SENSITIVITY_MASK_MODE,
        "arms": [{"code": code, "name": ARM_NAMES[code]} for code in ARM_CODES],
        "category_prior_tested": False,
        "gt_used_only_by_this_evaluator": True,
        "replay_callable_forbidden_parameters": forbidden_parameters,
        "input_identity": input_identity,
        "technical_gates": technical_gates,
        "aggregates": aggregates,
        "scientific_gates": scientific_gates,
        "registered_current_official_protocols": aggregate_current_protocols,
        "rows": table_rows,
        "scenes": nested,
    }
    analysis = {
        "schema": LATE_FILTER_ANALYSIS_SCHEMA,
        "manifest": str(manifest_file),
        "input_identity_sha256": input_identity["content_sha256"],
        "technical_gates": technical_gates,
        "primary_mask_mode": PRIMARY_MASK_MODE,
        "primary_arm_passed": primary_passed,
        "sensitivity_arm_passed": sensitivity_passed,
        "accepted_component_capacity_iou_050": int(
            aggregates[PRIMARY_MASK_MODE]["A1B1"]["accepted_geometry_050_tp"]
        ),
        "accepted_component_capacity_at_least_6": int(
            aggregates[PRIMARY_MASK_MODE]["A1B1"]["accepted_geometry_050_tp"]
        )
        >= 6,
        "attribution": attribution,
        "exporter_authorized": bool(exporter_authorized),
        "p_only_sensitivity_cannot_authorize": bool(p_only),
        "decision": decision,
        "category_prior_tested": False,
        "official_ap_reported_for_b0": False,
        "next_stage": (
            "repair-late-filter-audit-contract"
            if decision == "stop-technical-contract-failed"
            else (
                "conditional-clean-exporter-dev2"
                if exporter_authorized
                else (
                    "preregister-deterministic-flat-lifting"
                    if p_only
                    else "object-level-mask-or-geometric-superpoint-input"
                )
            )
        ),
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination / OUTPUT_TABLE, table_rows)
    write_json(destination / OUTPUT_JSON, factorial)
    write_json(destination / OUTPUT_ANALYSIS, analysis)
    return analysis


# Descriptive alias used by programmatic callers that think of this as an
# evaluator rather than the registered audit CLI.
evaluate_clean_late_filter_factorial = audit_clean_late_filters


__all__ = [
    "ARM_CODES",
    "ARM_NAMES",
    "LATE_FILTER_ANALYSIS_SCHEMA",
    "LATE_FILTER_FACTORIAL_SCHEMA",
    "LATE_FILTER_ROW_SCHEMA",
    "MASK_MODES",
    "OUTPUT_ANALYSIS",
    "OUTPUT_JSON",
    "OUTPUT_TABLE",
    "PRIMARY_MASK_MODE",
    "SENSITIVITY_MASK_MODE",
    "audit_clean_late_filters",
    "evaluate_clean_late_filter_factorial",
]
