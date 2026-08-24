from __future__ import annotations

"""Contributor repair and structural ablations for the teacher closeout.

This stage deliberately consumes the already trained ``full950/adaptive``
feature assets and never changes them.  GT is read only after both L0 outputs
exist, solely to run the registered official-evaluator parity gate.  The exact
patched-950 implementation is run first; the current L0 harness is allowed to
drive L1--L3 only after mechanical and official metric parity are established.
"""

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .baseline_closure import (
    CLOSURE_SCENES,
    TAXONOMIES,
    RuntimeScene,
    asset_paths,
    feature_is_complete,
    feature_paths,
    load_runtime_scenes,
    output_is_complete,
    output_paths,
    validate_scene_inputs,
)
from .baseline_closure_runner import (
    EXPECTED_CGROUP_MAX_BYTES,
    StageInvocation,
    _ensure_stage_parents,
    execute_stage,
    read_cgroup_snapshot,
)
from .baseline_closure_analysis import (
    _gaussian_ply,
    _runtime_rows,
    _transform,
    bfc18_saga20_intersection,
)
from .baseline_closure_evaluation import evaluate_baseline_closure
from .evaluator import load_ground_truth_npz, saga_scene_predictions
from .instance_projection import (
    DeclaredInstanceProjection,
    project_declared_instances,
)
from .taxonomy import load_taxonomy

STRUCTURAL_LEVELS = ("L0", "L1", "L2", "L3")
FIXED_VARIANT = "full950-contributor-fixed"
HARNESS_VARIANT = "current-causal-harness"
DISABLED_OTHER_CLASS = "__disabled__"

Executor = Callable[[StageInvocation], int]


def _official_metric_surface(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return only quantities that are official AP results.

    ``evaluate_instances`` also emits diagnostic support counts such as
    ``pred_instances``.  Those counts can differ when a sub-region prediction
    is removed by ``min_region_size`` even though every reported AP value is
    identical.  Such a diagnostic difference is useful evidence, but it is
    not an official-metric difference and must not block the L1--L3 causal
    ablation gate.
    """

    surface: dict[str, Any] = {}
    for view in ("full_saga20", "predictable_intersection"):
        result = protocol[view]
        surface[view] = {
            "aggregate": dict(result["aggregate"]),
            "per_class": {
                str(class_name): {
                    str(key): value
                    for key, value in class_result.items()
                    if str(key).startswith("ap_")
                }
                for class_name, class_result in result.get("per_class", {}).items()
            },
        }
    return surface


def evaluate_official_parity(
    *,
    scene_id: str,
    reference_output: Path,
    candidate_output: Path,
    gt_dir: Path,
    runtime_scene: Mapping[str, Any],
    min_region_size: int = 100,
    radius_m: float = 0.05,
) -> dict[str, Any]:
    """Run the actual registered official protocol on one output pair."""

    taxonomy = load_taxonomy()
    class_names = taxonomy.canonical_classes
    predictable = bfc18_saga20_intersection(class_names)
    gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
    common = {
        "scene_id": scene_id,
        "gt_coords": gt_xyz,
        "gaussian_ply": _gaussian_ply(runtime_scene),
        "taxonomy": taxonomy,
        "metadata_json": None,
        "transform": _transform(runtime_scene),
        "radius_m": float(radius_m),
        "require_scores": False,
    }
    reference_predictions, reference_alignment = saga_scene_predictions(
        output_json=reference_output, **common
    )
    candidate_predictions, candidate_alignment = saga_scene_predictions(
        output_json=candidate_output, **common
    )
    reference = evaluate_baseline_closure(
        [gt], reference_predictions, class_names,
        predictable_classes=predictable,
        score_mode="unit",
        min_region_size=int(min_region_size),
    )
    candidate = evaluate_baseline_closure(
        [gt], candidate_predictions, class_names,
        predictable_classes=predictable,
        score_mode="unit",
        min_region_size=int(min_region_size),
    )
    reference_protocol = reference["protocols"]["scannet_official_9"]
    candidate_protocol = candidate["protocols"]["scannet_official_9"]
    reference_metrics = _official_metric_surface(reference_protocol)
    candidate_metrics = _official_metric_surface(candidate_protocol)
    metric_surface_equal = reference_metrics == candidate_metrics
    diagnostic_protocol_equal = reference_protocol == candidate_protocol

    def aggregates(protocol: Mapping[str, Any]) -> dict[str, Any]:
        return {
            view: protocol[view]["aggregate"]
            for view in ("full_saga20", "predictable_intersection")
        }

    return {
        "evaluated": True,
        # ``equal`` is intentionally the official AP surface, not auxiliary
        # prediction/support counts contained in the evaluator payload.
        "equal": bool(metric_surface_equal),
        "metric_surface_equal": bool(metric_surface_equal),
        "diagnostic_protocol_equal": bool(diagnostic_protocol_equal),
        "protocol": "scannet-official-instance-9-v1",
        "min_region_size": int(min_region_size),
        "radius_m": float(radius_m),
        "reference_output": str(reference_output),
        "candidate_output": str(candidate_output),
        "reference_aggregate": aggregates(reference_protocol),
        "candidate_aggregate": aggregates(candidate_protocol),
        "reference_alignment": reference_alignment,
        "candidate_alignment": candidate_alignment,
    }


def _default_executor(invocation: StageInvocation) -> int:
    invocation.log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    local_module_roots = (
        invocation.cwd / "submodules/diff-gaussian-rasterization-max-contributor",
        invocation.cwd,
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in local_module_roots]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    with invocation.log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            list(invocation.command),
            cwd=invocation.cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    return int(result.returncode)


def _common_args(
    scene: RuntimeScene, closure_root: Path, *, feature_budget: str
) -> list[str]:
    assets = asset_paths(closure_root, scene.scene_id, "bfc18")
    feature_variant = (
        "full950-iterations-cli"
        if feature_budget in {"adaptive-iterations-cli", "10000"}
        else "full950"
    )
    source_budget = (
        "adaptive" if feature_budget == "adaptive-iterations-cli" else feature_budget
    )
    features = feature_paths(
        closure_root, scene.scene_id, feature_variant, source_budget, "bfc18"
    )
    return [
        "--sh_degree",
        "0",
        "--feature_dim",
        "32",
        "--images_path",
        str(scene.images_path),
        "--sparse_path",
        str(scene.sparse_path),
        "--masks_path",
        str(assets.masks),
        "--labels_path",
        str(assets.labels),
        "--label_features_path",
        str(assets.label_features),
        "--mask_scales_path",
        str(assets.mask_scales),
        "--point_cloud_path",
        str(scene.point_cloud_path),
        "--contrastive_feature_point_cloud_path",
        str(features.point_cloud),
        "--scale_gate_path",
        str(features.scale_gate),
    ]


def build_fixed_invocation(
    scene: RuntimeScene,
    closure_root: Path,
    fixed_workspace: Path,
    *,
    condition: str,
    feature_budget: str = "adaptive",
) -> StageInvocation:
    taxonomy = TAXONOMIES["bfc18"]
    target = output_paths(
        closure_root,
        scene.scene_id,
        FIXED_VARIANT,
        feature_budget,
        condition,
        "bfc18",
    )
    if condition == "B0-global":
        other = (DISABLED_OTHER_CLASS,)
    elif condition == "B1-original":
        other = taxonomy.other_classes
    else:
        raise ValueError(f"unknown fixed condition: {condition}")
    command = (
        str(scene.python_bin),
        str(fixed_workspace / "postprocess.py"),
        "--progress_path",
        str(target.progress),
        *_common_args(scene, closure_root, feature_budget=feature_budget),
        "--json_path",
        str(target.output_json),
        "--classes",
        *taxonomy.classes,
        "--selected_classes",
        *taxonomy.selected_classes,
        "--other_classes",
        *other,
    )
    return StageInvocation(
        stage="contributor-fixed",
        scene_id=scene.scene_id,
        variant_id=FIXED_VARIANT,
        budget=feature_budget,
        condition=condition,
        command=command,
        cwd=fixed_workspace,
        log_path=target.root / "postprocess.log",
    )


def build_harness_invocation(
    scene: RuntimeScene,
    closure_root: Path,
    current_workspace: Path,
    *,
    level: str,
    condition: str,
    scene_scale_m_per_unit: float,
) -> StageInvocation:
    if level not in STRUCTURAL_LEVELS:
        raise ValueError(f"unknown structural level: {level}")
    if scene_scale_m_per_unit <= 0:
        raise ValueError("scene_scale_m_per_unit must be positive")
    taxonomy = TAXONOMIES["bfc18"]
    output_condition = f"{level}-{condition}"
    target = output_paths(
        closure_root,
        scene.scene_id,
        HARNESS_VARIANT,
        "adaptive",
        output_condition,
        "bfc18",
    )
    command = [
        str(scene.python_bin),
        str(current_workspace / "postprocess.py"),
        "--progress_path",
        str(target.progress),
        *_common_args(scene, closure_root, feature_budget="adaptive"),
        "--json_path",
        str(target.output_json),
        "--stage_trace_path",
        str(target.root / "stage_trace.npz"),
        "--classes",
        *taxonomy.classes,
        "--selected_classes",
        *taxonomy.selected_classes,
        "--other_classes",
        *taxonomy.other_classes,
        "--teacher-prior-mode",
        "original",
        "--v7-causal-ablation",
        level,
        "--scene_scale_m_per_unit",
        str(float(scene_scale_m_per_unit)),
        "--seed",
        "42",
    ]
    if condition == "B0-global":
        command.append("--disable_other_classes")
    elif condition != "B1-original":
        raise ValueError(f"unknown harness condition: {condition}")
    return StageInvocation(
        stage="causal-ablation",
        scene_id=scene.scene_id,
        variant_id=HARNESS_VARIANT,
        budget="adaptive",
        condition=output_condition,
        command=tuple(command),
        cwd=current_workspace,
        log_path=target.root / "postprocess.log",
    )


def _load_output(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"prediction must be an object: {path}")
    return payload


def stage_trace_is_complete(trace_path: Path, output_json: Path) -> bool:
    """Validate the compact V9 forensic sidecar for one completed output."""

    metadata_path = trace_path.with_suffix(".json")
    if not trace_path.is_file() or not metadata_path.is_file() or not output_json.is_file():
        return False
    try:
        payload = _load_output(output_json)
        labels = payload.get("point_labels")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(labels, list):
            return False
        if metadata.get("schema") != "saga-v9-legacy-stage-trace-v1":
            return False
        if int(metadata.get("point_count", -1)) != len(labels):
            return False
        required = {
            "global_sample_core",
            "global_full_assignment",
            "other_class_candidates",
            "branch_class_before_merge",
            "merged_partition",
            "post_global_knn",
            "post_filter",
            "post_attach",
            "final_internal_labels",
        }
        with np.load(trace_path, allow_pickle=False) as arrays:
            return required.issubset(arrays.files) and all(
                np.asarray(arrays[key]).shape == (len(labels),) for key in required
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def canonical_point_partition(
    payload: Mapping[str, Any],
) -> tuple[tuple[str, int] | None, ...]:
    """Canonicalize IDs with a membership-stable first-point anchor."""

    partition, _projection = _canonical_point_partition(payload)
    return partition


def _canonical_point_partition(
    payload: Mapping[str, Any],
) -> tuple[
    tuple[tuple[str, int] | None, ...],
    DeclaredInstanceProjection,
]:
    """Build a partition plus the explicit declared-instance projection."""

    raw_labels = payload.get("point_labels")
    raw_instances = payload.get("instances")
    if not isinstance(raw_labels, list) or not isinstance(raw_instances, Mapping):
        raise TypeError("prediction is missing point_labels/instances")
    projection = project_declared_instances(raw_labels, raw_instances)
    members: dict[int, list[int]] = {}
    for index, raw in enumerate(projection.point_labels):
        value = int(raw)
        if value >= 0:
            members.setdefault(value, []).append(index)
    records: list[tuple[str, tuple[int, ...], int]] = []
    for instance_id, indices in members.items():
        metadata = raw_instances.get(str(instance_id), raw_instances.get(instance_id))
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("class"), str
        ):
            raise TypeError(f"instance {instance_id} has no class metadata")
        records.append((str(metadata["class"]), tuple(indices), instance_id))
    canonical: list[tuple[str, int] | None] = [None] * len(raw_labels)
    for class_name, indices, _raw_id in records:
        membership_anchor = indices[0]
        for index in indices:
            canonical[index] = (class_name, membership_anchor)
    return tuple(canonical), projection


def _metadata_for_instance(
    instances: Mapping[str | int, Any], instance_id: int
) -> Mapping[str, Any] | None:
    """Return metadata for either JSON string IDs or in-memory integer IDs."""

    metadata = instances.get(str(instance_id), instances.get(instance_id))
    return metadata if isinstance(metadata, Mapping) else None


def _partition_records(
    labels: np.ndarray,
    instances: Mapping[str | int, Any],
) -> tuple[dict[int, int], dict[int, str | None]]:
    """Return instance sizes and optional classes for a point partition."""

    foreground = labels[labels >= 0]
    if foreground.size == 0:
        return {}, {}
    instance_ids, counts = np.unique(foreground, return_counts=True)
    sizes = {
        int(instance_id): int(count)
        for instance_id, count in zip(instance_ids, counts)
    }
    classes: dict[int, str | None] = {}
    for instance_id in sizes:
        metadata = _metadata_for_instance(instances, instance_id)
        class_name = metadata.get("class") if metadata is not None else None
        classes[instance_id] = class_name if isinstance(class_name, str) else None
    return sizes, classes


def _overlap_matrix(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_ids: Sequence[int],
    right_ids: Sequence[int],
) -> np.ndarray:
    """Build the foreground contingency table without assuming aligned IDs."""

    overlaps = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
    if overlaps.size == 0:
        return overlaps
    jointly_foreground = (left_labels >= 0) & (right_labels >= 0)
    if not np.any(jointly_foreground):
        return overlaps
    pairs, counts = np.unique(
        np.column_stack(
            (left_labels[jointly_foreground], right_labels[jointly_foreground])
        ),
        axis=0,
        return_counts=True,
    )
    left_index = {instance_id: index for index, instance_id in enumerate(left_ids)}
    right_index = {
        instance_id: index for index, instance_id in enumerate(right_ids)
    }
    for (left_id, right_id), count in zip(pairs, counts):
        overlaps[left_index[int(left_id)], right_index[int(right_id)]] = int(count)
    return overlaps


def _maximum_overlap_pairs(
    overlaps: np.ndarray,
    left_ids: Sequence[int],
    right_ids: Sequence[int],
    *,
    compatible: np.ndarray | None = None,
) -> list[tuple[int, int, int]]:
    """Return an overlap-optimal one-to-one instance matching.

    Instance IDs are sorted before this helper is called, so SciPy's stable row
    and column ordering also gives deterministic tie handling.  Zero-overlap
    assignments are deliberately omitted: they are unmatched instances, not
    evidence that two arbitrary clusters correspond.
    """

    if overlaps.size == 0:
        return []
    scores = overlaps.copy()
    if compatible is not None:
        if compatible.shape != scores.shape:
            raise ValueError("compatibility matrix shape differs from overlaps")
        scores[~compatible] = 0
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - closure environments use scipy
        raise RuntimeError("partition comparison requires scipy") from exc
    rows, columns = linear_sum_assignment(-scores)
    result: list[tuple[int, int, int]] = []
    for row, column in zip(rows, columns):
        overlap = int(scores[row, column])
        if overlap > 0:
            result.append((int(left_ids[row]), int(right_ids[column]), overlap))
    return result


def _metadata_field_differences(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> list[str]:
    left_fields = left or {}
    right_fields = right or {}
    keys = sorted(set(left_fields) | set(right_fields))
    return [key for key in keys if left_fields.get(key) != right_fields.get(key)]


def _compare_partition_arrays(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_instances: Mapping[str | int, Any],
    right_instances: Mapping[str | int, Any],
    *,
    include_semantics: bool,
) -> dict[str, Any]:
    """Compare point partitions using maximum-overlap instance matching."""

    if left_labels.shape != right_labels.shape:
        raise ValueError("prediction point counts differ")
    left_sizes, left_classes = _partition_records(left_labels, left_instances)
    right_sizes, right_classes = _partition_records(right_labels, right_instances)
    left_ids = sorted(left_sizes)
    right_ids = sorted(right_sizes)
    overlaps = _overlap_matrix(left_labels, right_labels, left_ids, right_ids)
    geometry_pairs = _maximum_overlap_pairs(overlaps, left_ids, right_ids)
    both_background = int(np.count_nonzero((left_labels < 0) & (right_labels < 0)))
    geometry_unchanged = both_background + sum(pair[2] for pair in geometry_pairs)
    point_count = int(left_labels.size)
    geometry_changed = point_count - geometry_unchanged

    semantic_pairs: list[tuple[int, int, int]] = []
    semantic_changed = geometry_changed
    if include_semantics:
        compatible = np.zeros(overlaps.shape, dtype=bool)
        for left_index, left_id in enumerate(left_ids):
            left_class = left_classes[left_id]
            if left_class is None:
                continue
            for right_index, right_id in enumerate(right_ids):
                compatible[left_index, right_index] = (
                    left_class == right_classes[right_id]
                )
        semantic_pairs = _maximum_overlap_pairs(
            overlaps, left_ids, right_ids, compatible=compatible
        )
        semantic_unchanged = both_background + sum(
            pair[2] for pair in semantic_pairs
        )
        semantic_changed = point_count - semantic_unchanged

    matched_left = {left_id for left_id, _right_id, _overlap in geometry_pairs}
    matched_right = {right_id for _left_id, right_id, _overlap in geometry_pairs}
    match_rows: list[dict[str, Any]] = []
    class_mismatch_overlap = 0
    metadata_difference_count = 0
    for left_id, right_id, overlap in geometry_pairs:
        left_metadata = _metadata_for_instance(left_instances, left_id)
        right_metadata = _metadata_for_instance(right_instances, right_id)
        different_fields = _metadata_field_differences(
            left_metadata, right_metadata
        )
        class_equal = (
            left_classes[left_id] is not None
            and left_classes[left_id] == right_classes[right_id]
        )
        if include_semantics and not class_equal:
            class_mismatch_overlap += overlap
        if different_fields:
            metadata_difference_count += 1
        match_rows.append(
            {
                "left_instance_id": left_id,
                "right_instance_id": right_id,
                "overlap": overlap,
                "left_size": left_sizes[left_id],
                "right_size": right_sizes[right_id],
                "symmetric_difference_points": (
                    left_sizes[left_id] + right_sizes[right_id] - 2 * overlap
                ),
                "left_class": left_classes[left_id],
                "right_class": right_classes[right_id],
                "class_equal": class_equal,
                "metadata_different_fields": different_fields,
            }
        )

    unmatched_left = [
        {
            "instance_id": instance_id,
            "size": left_sizes[instance_id],
            "class": left_classes[instance_id],
        }
        for instance_id in left_ids
        if instance_id not in matched_left
    ]
    unmatched_right = [
        {
            "instance_id": instance_id,
            "size": right_sizes[instance_id],
            "class": right_classes[instance_id],
        }
        for instance_id in right_ids
        if instance_id not in matched_right
    ]
    changed_points = semantic_changed if include_semantics else geometry_changed
    result = {
        "point_count": point_count,
        "left_instance_count": len(left_ids),
        "right_instance_count": len(right_ids),
        "matched_instance_count": len(geometry_pairs),
        "both_background_points": both_background,
        "geometry_changed_points": geometry_changed,
        "geometry_changed_fraction": geometry_changed / max(point_count, 1),
        "class_changed_points": semantic_changed - geometry_changed,
        "class_changed_fraction": (
            semantic_changed - geometry_changed
        )
        / max(point_count, 1),
        "class_mismatch_overlap_points": class_mismatch_overlap,
        "changed_points": changed_points,
        "changed_fraction": changed_points / max(point_count, 1),
        "geometry_equivalent": geometry_changed == 0,
        "equivalent": changed_points == 0,
        "matches": match_rows,
        "unmatched_left_instances": unmatched_left,
        "unmatched_right_instances": unmatched_right,
        "metadata_difference_count": metadata_difference_count,
    }
    if include_semantics:
        result["semantic_matched_instance_count"] = len(semantic_pairs)
    return result


def _internal_labels_from_trace(
    trace_path: Path | None, fallback: Sequence[int]
) -> np.ndarray:
    if trace_path is None:
        return np.asarray(fallback, dtype=np.int64)
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    with np.load(trace_path, allow_pickle=False) as arrays:
        if "final_internal_labels" not in arrays.files:
            raise KeyError(f"{trace_path}: missing final_internal_labels")
        labels = np.asarray(arrays["final_internal_labels"], dtype=np.int64)
    if labels.ndim != 1:
        raise ValueError(f"{trace_path}: final_internal_labels must be 1D")
    return labels


def compare_partitions(
    left_path: Path,
    right_path: Path,
    *,
    left_internal_trace: Path | None = None,
    right_internal_trace: Path | None = None,
) -> dict[str, Any]:
    """Compare raw/internal and declared/exported partitions independently.

    The historical comparator assigned a global rank after sorting instances.
    Adding one tiny instance therefore shifted every later rank and falsely
    marked unrelated Gaussians as changed.  This comparator instead performs a
    maximum-overlap bipartite matching.  Raw labels diagnose internal geometry;
    the declared projection separately diagnoses what evaluators/viewers see.
    """

    left_payload = _load_output(left_path)
    right_payload = _load_output(right_path)
    left_raw = left_payload.get("point_labels")
    right_raw = right_payload.get("point_labels")
    left_instances = left_payload.get("instances")
    right_instances = right_payload.get("instances")
    if not isinstance(left_raw, list) or not isinstance(left_instances, Mapping):
        raise TypeError("left prediction is missing point_labels/instances")
    if not isinstance(right_raw, list) or not isinstance(
        right_instances, Mapping
    ):
        raise TypeError("right prediction is missing point_labels/instances")
    left_projection = project_declared_instances(left_raw, left_instances)
    right_projection = project_declared_instances(right_raw, right_instances)
    left_output_labels = np.asarray(left_raw, dtype=np.int64)
    right_output_labels = np.asarray(right_raw, dtype=np.int64)
    if not np.array_equal(np.asarray(left_raw), left_output_labels):
        raise TypeError("left point_labels must contain integers")
    if not np.array_equal(np.asarray(right_raw), right_output_labels):
        raise TypeError("right point_labels must contain integers")
    left_raw_labels = _internal_labels_from_trace(
        left_internal_trace, left_raw
    )
    right_raw_labels = _internal_labels_from_trace(
        right_internal_trace, right_raw
    )
    if left_raw_labels.shape != left_output_labels.shape:
        raise ValueError("left internal trace point count differs from output")
    if right_raw_labels.shape != right_output_labels.shape:
        raise ValueError("right internal trace point count differs from output")
    raw_internal = _compare_partition_arrays(
        left_raw_labels,
        right_raw_labels,
        left_instances,
        right_instances,
        include_semantics=False,
    )
    declared_exported = _compare_partition_arrays(
        left_projection.point_labels,
        right_projection.point_labels,
        left_instances,
        right_instances,
        include_semantics=True,
    )
    matched_left_ids = {
        int(row["left_instance_id"]) for row in declared_exported["matches"]
    }
    matched_right_ids = {
        int(row["right_instance_id"]) for row in declared_exported["matches"]
    }
    left_only_declared = sorted(
        set(left_projection.declared_instance_ids) - matched_left_ids
    )
    right_only_declared = sorted(
        set(right_projection.declared_instance_ids) - matched_right_ids
    )
    class_difference_count = sum(
        not bool(row["class_equal"]) for row in declared_exported["matches"]
    )
    metadata = {
        "matched_metadata_difference_count": declared_exported[
            "metadata_difference_count"
        ],
        "matched_class_difference_count": class_difference_count,
        "left_only_declared_instance_ids": left_only_declared,
        "right_only_declared_instance_ids": right_only_declared,
        "left_ignored_negative_metadata_ids": list(
            left_projection.ignored_negative_metadata_ids
        ),
        "right_ignored_negative_metadata_ids": list(
            right_projection.ignored_negative_metadata_ids
        ),
    }
    metadata["equivalent"] = (
        metadata["matched_metadata_difference_count"] == 0
        and not left_only_declared
        and not right_only_declared
        and metadata["left_ignored_negative_metadata_ids"]
        == metadata["right_ignored_negative_metadata_ids"]
    )
    partition_equivalent = (
        raw_internal["equivalent"]
        and declared_exported["equivalent"]
    )
    return {
        "left": str(left_path),
        "right": str(right_path),
        "left_internal_trace": (
            str(left_internal_trace) if left_internal_trace is not None else None
        ),
        "right_internal_trace": (
            str(right_internal_trace) if right_internal_trace is not None else None
        ),
        "point_count": raw_internal["point_count"],
        # Compatibility aliases retain the old exported/class-aware meaning.
        "changed_points": declared_exported["changed_points"],
        "changed_fraction": declared_exported["changed_fraction"],
        "equivalent": partition_equivalent,
        "metadata_equivalent": metadata["equivalent"],
        "fully_equivalent": partition_equivalent and metadata["equivalent"],
        "raw_geometry_changed_points": raw_internal["geometry_changed_points"],
        "exported_geometry_changed_points": declared_exported[
            "geometry_changed_points"
        ],
        "exported_class_changed_points": declared_exported[
            "class_changed_points"
        ],
        "raw_internal": raw_internal,
        "declared_exported": declared_exported,
        "metadata": metadata,
        "left_projection": left_projection.stats(),
        "right_projection": right_projection.stats(),
    }


def _scene_scales(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: Any = (
        payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    )
    if isinstance(rows, Mapping):
        rows = [dict(value, scene_id=key) for key, value in rows.items()]
    if not isinstance(rows, list):
        raise TypeError("runtime manifest must contain scenes")
    return {
        str(row["scene_id"]): float(row.get("scene_scale_m_per_unit", 1.0))
        for row in rows
    }


def _resource_snapshot(closure_root: Path, cgroup_root: Path) -> dict[str, Any]:
    free_gib = shutil.disk_usage(closure_root).free / 1024**3
    if free_gib < 80.0:
        raise RuntimeError(f"less than 80 GiB free: {free_gib:.1f}")
    cgroup = read_cgroup_snapshot(cgroup_root)
    if cgroup["max"] != EXPECTED_CGROUP_MAX_BYTES:
        raise RuntimeError(f"expected 90 GiB cgroup; found {cgroup['max']}")
    if cgroup["current"] >= cgroup["max"]:
        raise RuntimeError("memory.current reached memory.max")
    return {"disk_free_gib": free_gib, "cgroup": cgroup}


def parity_allows_structural_ablation(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the V9 mechanical-equivalence gate to one parity comparison.

    Raw instance geometry must be byte-for-byte equivalent after overlap
    matching.  A tiny exported-only metadata boundary is recorded instead of
    being amplified into a whole-partition failure.  ``min_region_size`` is
    measured on mapped evaluation points, not on Gaussian count, so a
    sub-100-Gaussian declaration is never treated as proof of AP neutrality.
    The gate only opens when an actual official-evaluator parity result is
    attached to the comparison.
    """

    raw_exact = int(comparison.get("raw_geometry_changed_points", -1)) == 0
    exported_changed = int(
        comparison.get("exported_geometry_changed_points", 10**18)
    )
    exported_boundary_small = 0 <= exported_changed < 100
    matched_class_exact = (
        int(comparison.get("exported_class_changed_points", -1)) == 0
        and int(
            comparison.get("metadata", {}).get(
                "matched_class_difference_count", -1
            )
        )
        == 0
    )
    unmatched = tuple(
        comparison.get("declared_exported", {}).get(
            "unmatched_left_instances", ()
        )
    ) + tuple(
        comparison.get("declared_exported", {}).get(
            "unmatched_right_instances", ()
        )
    )
    boundary_is_only_subregion_declarations = (
        exported_changed == sum(int(row.get("size", -1)) for row in unmatched)
        and all(0 <= int(row.get("size", -1)) < 100 for row in unmatched)
    )
    official_evidence = comparison.get("official_evaluator_parity")
    official_evaluated = (
        isinstance(official_evidence, Mapping)
        and official_evidence.get("evaluated") is True
    )
    official_equal = (
        official_evaluated and official_evidence.get("equal") is True
    )
    allowed = (
        raw_exact
        and exported_boundary_small
        and matched_class_exact
        and boundary_is_only_subregion_declarations
        and official_equal
    )
    return {
        "allowed": allowed,
        "raw_geometry_exact": raw_exact,
        "exported_boundary_points": exported_changed,
        "exported_boundary_below_100": exported_boundary_small,
        "matched_instance_classes_exact": matched_class_exact,
        "boundary_is_only_subregion_declarations": (
            boundary_is_only_subregion_declarations
        ),
        "official_evaluator_parity_evaluated": official_evaluated,
        "official_evaluator_parity_equal": official_equal,
        "blocked_without_official_evaluator_parity": not official_evaluated,
    }


def run_ablation_closeout(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    closure_root: Path,
    fixed_workspace: Path,
    current_workspace: Path,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    executor: Executor = _default_executor,
) -> dict[str, Any]:
    scenes = load_runtime_scenes(runtime_manifest)
    runtime_rows = _runtime_rows(runtime_manifest)
    scales = _scene_scales(runtime_manifest)
    taxonomy = TAXONOMIES["bfc18"]
    for required in CLOSURE_SCENES:
        if required not in scenes or required not in scales:
            raise KeyError(f"runtime manifest is missing {required}")
    if not (fixed_workspace / "postprocess.py").is_file():
        raise FileNotFoundError(fixed_workspace / "postprocess.py")
    if not (current_workspace / "postprocess.py").is_file():
        raise FileNotFoundError(current_workspace / "postprocess.py")
    closure_root.mkdir(parents=True, exist_ok=True)
    initial = _resource_snapshot(closure_root, cgroup_root)
    records: list[dict[str, Any]] = []

    for scene_id in CLOSURE_SCENES:
        scene = scenes[scene_id]
        validate_scene_inputs(scene)
        features = feature_paths(closure_root, scene_id, "full950", "adaptive", "bfc18")
        if not feature_is_complete(scene, features):
            raise RuntimeError(f"full950/adaptive feature is incomplete: {scene_id}")
        fixed_budgets = (
            ("adaptive", "adaptive-iterations-cli", "10000")
            if scene_id == "scene0064_01"
            else ("adaptive",)
        )
        for fixed_budget in fixed_budgets:
            feature_variant = (
                "full950-iterations-cli"
                if fixed_budget in {"adaptive-iterations-cli", "10000"}
                else "full950"
            )
            source_budget = (
                "adaptive"
                if fixed_budget == "adaptive-iterations-cli"
                else fixed_budget
            )
            fixed_features = feature_paths(
                closure_root,
                scene_id,
                feature_variant,
                source_budget,
                "bfc18",
            )
            if not feature_is_complete(scene, fixed_features):
                raise RuntimeError(
                    f"{feature_variant}/{source_budget} feature is incomplete: {scene_id}"
                )
            for condition in ("B0-global", "B1-original"):
                invocation = build_fixed_invocation(
                    scene,
                    closure_root,
                    fixed_workspace,
                    condition=condition,
                    feature_budget=fixed_budget,
                )
                _ensure_stage_parents(invocation)
                target = output_paths(
                    closure_root,
                    scene_id,
                    FIXED_VARIANT,
                    fixed_budget,
                    condition,
                    "bfc18",
                )
                result = execute_stage(
                    invocation,
                    is_complete=lambda s=scene, p=target: output_is_complete(
                        s, p, taxonomy
                    ),
                    executor=executor,
                )
                records.append(
                    {
                        **result,
                        "scene_id": scene_id,
                        "budget": fixed_budget,
                        "condition": condition,
                    }
                )

        for condition in ("B0-global", "B1-original"):
            invocation = build_harness_invocation(
                scene,
                closure_root,
                current_workspace,
                level="L0",
                condition=condition,
                scene_scale_m_per_unit=scales[scene_id],
            )
            _ensure_stage_parents(invocation)
            target = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"L0-{condition}",
                "bfc18",
            )
            result = execute_stage(
                invocation,
                is_complete=lambda s=scene, p=target: (
                    output_is_complete(s, p, taxonomy)
                    and stage_trace_is_complete(
                        p.root / "stage_trace.npz", p.output_json
                    )
                ),
                executor=executor,
            )
            records.append(
                {**result, "scene_id": scene_id, "condition": f"L0-{condition}"}
            )

    parity: list[dict[str, Any]] = []
    for scene_id in CLOSURE_SCENES:
        for condition in ("B0-global", "B1-original"):
            fixed = output_paths(
                closure_root, scene_id, FIXED_VARIANT, "adaptive", condition, "bfc18"
            )
            harness = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"L0-{condition}",
                "bfc18",
            )
            comparison = compare_partitions(
                fixed.output_json,
                harness.output_json,
                right_internal_trace=harness.root / "stage_trace.npz",
            )
            comparison["official_evaluator_parity"] = evaluate_official_parity(
                scene_id=scene_id,
                reference_output=fixed.output_json,
                candidate_output=harness.output_json,
                gt_dir=gt_dir,
                runtime_scene=runtime_rows[scene_id],
            )
            parity.append({**comparison, "scene_id": scene_id, "condition": condition})
    parity_gate = [parity_allows_structural_ablation(row) for row in parity]
    for row, gate in zip(parity, parity_gate):
        row["v9_structural_gate"] = gate
    if not all(gate["allowed"] for gate in parity_gate):
        summary = {
            "schema": "saga-teacher-baseline-structural-v1",
            "status": "stopped-current-harness-parity-failed",
            "initial_resources": initial,
            "final_resources": _resource_snapshot(closure_root, cgroup_root),
            "parity": parity,
            "parity_gate": parity_gate,
            "records": records,
        }
        (closure_root / "structural_run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    for scene_id in CLOSURE_SCENES:
        scene = scenes[scene_id]
        for level in ("L1", "L2", "L3"):
            invocation = build_harness_invocation(
                scene,
                closure_root,
                current_workspace,
                level=level,
                condition="B1-original",
                scene_scale_m_per_unit=scales[scene_id],
            )
            _ensure_stage_parents(invocation)
            target = output_paths(
                closure_root,
                scene_id,
                HARNESS_VARIANT,
                "adaptive",
                f"{level}-B1-original",
                "bfc18",
            )
            result = execute_stage(
                invocation,
                is_complete=lambda s=scene, p=target: (
                    output_is_complete(s, p, taxonomy)
                    and stage_trace_is_complete(
                        p.root / "stage_trace.npz", p.output_json
                    )
                ),
                executor=executor,
            )
            records.append({**result, "scene_id": scene_id, "condition": level})

    summary = {
        "schema": "saga-teacher-baseline-structural-v1",
        "status": "completed",
        "initial_resources": initial,
        "final_resources": _resource_snapshot(closure_root, cgroup_root),
        "parity": parity,
        "parity_gate": parity_gate,
        "records": records,
    }
    (closure_root / "structural_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--closure-root", type=Path, required=True)
    parser.add_argument("--fixed-workspace", type=Path, required=True)
    parser.add_argument("--current-workspace", type=Path, required=True)
    parser.add_argument("--cgroup-root", type=Path, default=Path("/sys/fs/cgroup"))
    args = parser.parse_args(argv)
    result = run_ablation_closeout(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        closure_root=args.closure_root,
        fixed_workspace=args.fixed_workspace,
        current_workspace=args.current_workspace,
        cgroup_root=args.cgroup_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
