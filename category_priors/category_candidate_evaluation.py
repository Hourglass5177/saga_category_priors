from __future__ import annotations

"""GT-only diagnostics for candidate formation and C0/C1/C2 quality.

Nothing in this module is imported by the postprocess worker.  The split is a
hard guard against validation GT leaking into candidate construction.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_runner import CANDIDATE_REPAIR_CONDITIONS
from .category_candidate_legacy_replay import (
    LegacyReplayCandidate,
    replay_candidates_through_legacy,
)
from .category_candidate_trace import load_candidate_formation_trace
from .category_denoise import CandidateBank, load_candidate_bank
from .category_denoise_diagnostics import (
    _object_index_by_gt_point,
    _project_gaussian_labels,
    _reverse_arrays,
    _stage_iou,
    build_bidirectional_mapping,
    build_official_gt_objects,
)
from .evaluator import apply_transform, load_ground_truth_npz, load_ply_xyz
from .io import load_json, sha256_file, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy
from .v9_metrics import _gaussian_ply, _transform


DEV2_SCENE_IDS = ("scene0645_00", "scene0025_01")
DEV8_SCENE_IDS = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
FROZEN_REPAIR_ARM_SCHEMA = "saga-category-candidate-frozen-repair-arm-v1"
REPAIR_ANALYSIS_SCHEMA = "saga-category-candidate-repair-analysis-v1"


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _registered_scene_ids(
    scene_ids: Sequence[str], *, phase: str | None = None
) -> tuple[str, ...]:
    if isinstance(scene_ids, (str, bytes)):
        raise TypeError("scene_ids must be a sequence of scene identifiers")
    received = tuple(map(str, scene_ids))
    duplicates = sorted(
        scene_id for scene_id, count in Counter(received).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "scene_ids contains duplicate registered scenes: "
            + ", ".join(duplicates)
        )
    if phase == "dev2":
        expected = DEV2_SCENE_IDS
    elif phase == "dev8":
        expected = DEV8_SCENE_IDS
    elif phase is None:
        received_set = set(received)
        if received_set == set(DEV2_SCENE_IDS):
            expected = DEV2_SCENE_IDS
        elif received_set == set(DEV8_SCENE_IDS):
            expected = DEV8_SCENE_IDS
        else:
            raise ValueError(
                "diagnosis scene_ids must be exactly the registered DEV2 or DEV8 set"
            )
    else:  # pragma: no cover - guarded by the public evaluator
        raise ValueError(f"unknown evaluation phase: {phase!r}")
    if set(received) != set(expected):
        missing = sorted(set(expected) - set(received))
        unexpected = sorted(set(received) - set(expected))
        raise ValueError(
            f"{phase} scene_ids must be exactly the registered scene set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    # Canonicalize ordering so output identity is independent of caller order.
    return expected


def _load_scene_banks(
    root: Path, scene_id: str, conditions: Sequence[str]
) -> dict[str, CandidateBank]:
    return {
        condition: load_candidate_bank(root / "bank" / scene_id / condition)
        for condition in conditions
    }


def _resolve_frozen_repair_arm(
    artifact: Path,
    *,
    run_root: Path,
) -> tuple[str, dict[str, Any]]:
    path = Path(artifact)
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError("frozen repair-arm artifact must be a JSON object")
    if payload.get("schema") != FROZEN_REPAIR_ARM_SCHEMA:
        raise ValueError("frozen repair-arm artifact has the wrong schema")
    _registered_scene_ids(
        payload.get("selected_on_scene_ids", ()), phase="dev2"
    )
    condition = str(payload.get("condition", ""))
    if condition not in CANDIDATE_REPAIR_CONDITIONS[1:]:
        raise ValueError("frozen repair-arm artifact names an unregistered arm")
    artifact_root = Path(str(payload.get("run_root", "")))
    if artifact_root.resolve() != Path(run_root).resolve():
        raise ValueError("frozen repair-arm artifact belongs to a different run_root")

    analysis_path = Path(str(payload.get("selection_analysis", "")))
    expected_sha = payload.get("selection_analysis_sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise ValueError("frozen repair-arm artifact lacks the DEV2 analysis hash")
    if not analysis_path.is_file() or sha256_file(analysis_path) != expected_sha:
        raise ValueError("DEV2 repair-selection analysis changed after freezing")
    analysis = load_json(analysis_path)
    if not isinstance(analysis, Mapping):
        raise TypeError("DEV2 repair-selection analysis must be a JSON object")
    if analysis.get("schema") != REPAIR_ANALYSIS_SCHEMA:
        raise ValueError("DEV2 repair-selection analysis has the wrong schema")
    if analysis.get("phase") != "dev2":
        raise ValueError("repair arm was not selected during DEV2")
    _registered_scene_ids(analysis.get("scene_ids", ()), phase="dev2")
    if analysis.get("selected_condition") != condition:
        raise ValueError("frozen repair arm and DEV2 selection analysis disagree")
    gates = analysis.get("dev2_arm_gates")
    if not isinstance(gates, Mapping):
        raise TypeError("DEV2 selection analysis lacks repair-arm gates")
    gate = gates.get(condition)
    if not isinstance(gate, Mapping) or gate.get("passed") is not True:
        raise ValueError("frozen repair arm did not pass its registered DEV2 gate")
    return condition, dict(payload)


def _load_scene_postprocess_survival(
    root: Path | None, scene_id: str
) -> dict[int, dict[str, Any]]:
    if root is None:
        return {}
    path = Path(root) / scene_id / "diagnostics.json"
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: replay diagnostics must be a JSON object")
    category = payload.get("category_denoise", payload)
    if not isinstance(category, Mapping):
        raise TypeError(f"{path}: category_denoise must be a JSON object")
    raw_rows = category.get("candidate_survival")
    if not isinstance(raw_rows, list):
        raise TypeError(f"{path}: candidate_survival must be a list")
    result: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"{path}: candidate_survival[{index}] must be an object")
        candidate_id = int(raw["candidate_id"])
        if candidate_id in result:
            raise ValueError(
                f"{path}: duplicate candidate_survival id {candidate_id}"
            )
        accepted = raw.get("accepted")
        survived_knn = raw.get("survived_post_knn")
        survived_filter = raw.get("survived_post_filter")
        if not isinstance(accepted, bool):
            raise TypeError(f"{path}: candidate {candidate_id} accepted must be boolean")
        if not isinstance(survived_knn, bool) or not isinstance(
            survived_filter, bool
        ):
            raise TypeError(
                f"{path}: candidate {candidate_id} survival flags must be boolean"
            )
        if survived_filter and not survived_knn:
            raise ValueError(
                f"{path}: candidate {candidate_id} cannot survive the filter "
                "after being lost by KNN"
            )
        result[candidate_id] = {
            "accepted": accepted,
            "survived_post_knn": survived_knn,
            "survived_post_filter": survived_filter,
        }
    return result


def _postprocess_loss_stage(
    survival: Mapping[str, Any] | None,
) -> str | None:
    if survival is None or survival.get("accepted") is not True:
        return None
    if survival.get("survived_post_knn") is False:
        return "post_knn"
    if survival.get("survived_post_filter") is False:
        return "post_filter"
    return None


def _all_c0_postprocess_survival(
    bank: CandidateBank, gaussian_xyz: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Run the common legacy KNN/filter on every C0 candidate for diagnosis."""

    full_labels = np.asarray(bank.branch_full_labels, dtype=np.int64)
    core_labels = np.asarray(bank.branch_core_labels, dtype=np.int64)

    # C0 is the historical control whose formation bug we are diagnosing: a
    # raw-cluster member may have been reassigned outside its emitted full
    # candidate.  The legacy replay contract is intentionally stricter for
    # repaired candidates, so feed it only the exported part of C0's core.
    # The original core/full disagreement remains untouched in the bank and is
    # reported separately by ``_candidate_scene_metrics``.
    candidates = tuple(
        LegacyReplayCandidate(
            candidate_id=int(row["candidate_id"]),
            branch_class=str(row["branch_class"]),
            q_score=float(row["base_score"]),
            full_point_indices=np.flatnonzero(
                full_labels == int(row["candidate_id"])
            ),
            trusted_core_indices=np.flatnonzero(
                (core_labels == int(row["candidate_id"]))
                & (full_labels == int(row["candidate_id"]))
            ),
        )
        for row in bank.candidates
    )
    result = replay_candidates_through_legacy(
        xyz_scene=np.asarray(gaussian_xyz, dtype=np.float64),
        global_pre_knn=np.asarray(bank.global_pre_knn, dtype=np.int64),
        candidates=candidates,
        accepted_candidate_ids=tuple(row.candidate_id for row in candidates),
    )
    return {
        int(row.candidate_id): {
            "accepted": bool(row.accepted),
            "survived_post_knn": bool(row.survived_post_knn),
            "survived_post_filter": bool(row.survived_post_filter),
        }
        for row in result.candidates
    }


def _candidate_failure_status(
    *,
    sampled_count: int,
    best_raw_f1: float,
    retained_candidate_id: int | None,
    best_raw_iou: float,
    full_iou: float,
    best_raw_precision: float,
    full_precision: float,
    postprocess_loss_stage: str | None,
) -> str:
    if sampled_count < 3:
        return "sample_starved"
    if best_raw_f1 < 0.50:
        return "raw_clustering_failed"
    if retained_candidate_id is None or full_iou <= best_raw_iou - 0.10:
        return "full_assignment_loss"
    if full_precision <= best_raw_precision - 0.20:
        return "full_assignment_pollution"
    if postprocess_loss_stage is not None:
        return "postprocess_loss"
    return "candidate_formation_healthy"


def _scene_context(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    gt_dir: Path,
    taxonomy: Taxonomy,
    size_spec: Mapping[str, Any] | None,
    radius_m: float,
    min_region_size: int,
) -> dict[str, Any]:
    gt_xyz, ground_truth = load_ground_truth_npz(
        gt_dir / f"{scene_id}.npz", scene_id
    )
    gaussian_xyz = apply_transform(
        load_ply_xyz(_gaussian_ply(scene)), _transform(scene)
    )
    objects = build_official_gt_objects(
        scene_id,
        gt_xyz,
        ground_truth,
        taxonomy,
        min_region_size=min_region_size,
        size_spec=size_spec,
    )
    mapping = build_bidirectional_mapping(gt_xyz, gaussian_xyz, radius_m)
    object_index = _object_index_by_gt_point(len(gt_xyz), objects)
    object_counts = np.asarray([item.point_count for item in objects], dtype=np.int64)
    reverse_semantic, reverse_instance, reverse_evaluable = _reverse_arrays(
        mapping.gaussian_to_gt, ground_truth, len(taxonomy.canonical_classes)
    )
    return {
        "gt_xyz": gt_xyz,
        "ground_truth": ground_truth,
        "gaussian_xyz": gaussian_xyz,
        "objects": objects,
        "mapping": mapping,
        "object_index": object_index,
        "object_counts": object_counts,
        "reverse_semantic": reverse_semantic,
        "reverse_instance": reverse_instance,
        "reverse_evaluable": reverse_evaluable,
    }


def _candidate_scene_metrics(
    scene_id: str,
    bank: CandidateBank,
    context: Mapping[str, Any],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    objects = context["objects"]
    mapping = context["mapping"]
    projected = _project_gaussian_labels(
        np.asarray(bank.branch_full_labels, dtype=np.int64),
        mapping.gt_to_gaussian,
    )
    iou, intersections, predicted_counts = _stage_iou(
        projected,
        len(bank.candidates),
        context["object_index"],
        context["object_counts"],
    )
    class_to_id = {
        str(name): index for index, name in enumerate(taxonomy.canonical_classes)
    }
    candidate_class = np.asarray(
        [class_to_id[str(row["branch_class"])] for row in bank.candidates],
        dtype=np.int64,
    )
    reverse_semantic = np.asarray(context["reverse_semantic"], dtype=np.int64)
    reverse_instance = np.asarray(context["reverse_instance"], dtype=np.int64)
    reverse_evaluable = np.asarray(context["reverse_evaluable"], dtype=bool)
    full_labels = np.asarray(bank.branch_full_labels, dtype=np.int64)
    core_labels = np.asarray(bank.branch_core_labels, dtype=np.int64)

    best_by_object = np.zeros(len(objects), dtype=np.float64)
    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(bank.candidates):
        same_objects = np.asarray(
            [
                index
                for index, item in enumerate(objects)
                if item.class_id == candidate_class[candidate_index]
            ],
            dtype=np.int64,
        )
        if len(same_objects):
            local = iou[candidate_index, same_objects]
            chosen_local = int(np.argmax(local))
            best_object = int(same_objects[chosen_local])
            best_iou = float(local[chosen_local])
            best_by_object[same_objects] = np.maximum(
                best_by_object[same_objects], local
            )
        else:
            best_object = None
            best_iou = 0.0
        candidate_id = int(candidate["candidate_id"])
        gaussian_mask = full_labels == candidate_id
        unsupported = int(np.count_nonzero(gaussian_mask & ~reverse_evaluable))
        full_count = int(np.count_nonzero(gaussian_mask))
        target = objects[best_object] if best_object is not None else None
        correct = (
            int(
                np.count_nonzero(
                    gaussian_mask
                    & reverse_evaluable
                    & (reverse_semantic == target.class_id)
                    & (reverse_instance == target.instance_id)
                )
            )
            if target is not None
            else 0
        )
        candidate_rows.append(
            {
                "scene_id": scene_id,
                "candidate_id": candidate_id,
                "branch_class": str(candidate["branch_class"]),
                "full_point_count": full_count,
                "trusted_core_point_count": int(
                    np.count_nonzero(core_labels == candidate_id)
                ),
                "best_same_class_object_index": best_object,
                "best_same_class_instance_id": (
                    int(target.instance_id) if target is not None else None
                ),
                "best_same_class_size_bin": (
                    str(target.size_bin) if target is not None else None
                ),
                "best_same_class_iou": best_iou,
                "gaussian_precision": _fraction(correct, full_count),
                "unsupported_fraction": _fraction(unsupported, full_count),
                "core_subset_full": bool(
                    np.all(~(core_labels == candidate_id) | gaussian_mask)
                ),
            }
        )
    positive_025 = sum(row["best_same_class_iou"] >= 0.25 for row in candidate_rows)
    positive_050 = sum(row["best_same_class_iou"] >= 0.50 for row in candidate_rows)
    tiny_indices = [
        index
        for index, item in enumerate(objects)
        if item.size_bin in {"tiny", "small"}
    ]
    tiny_025 = sum(best_by_object[index] >= 0.25 for index in tiny_indices)
    tiny_050 = sum(best_by_object[index] >= 0.50 for index in tiny_indices)
    total_points = sum(row["full_point_count"] for row in candidate_rows)
    unsupported_weighted = sum(
        row["unsupported_fraction"] * row["full_point_count"]
        for row in candidate_rows
    )
    return {
        "scene_id": scene_id,
        "candidate_count": len(candidate_rows),
        "same_class_iou_025_count": int(positive_025),
        "same_class_iou_050_count": int(positive_050),
        "candidate_precision_025": _fraction(positive_025, len(candidate_rows)),
        "candidate_precision_050": _fraction(positive_050, len(candidate_rows)),
        "unsupported_fraction": _fraction(unsupported_weighted, total_points),
        "tiny_small_gt_count": len(tiny_indices),
        "tiny_small_recall_025": _fraction(tiny_025, len(tiny_indices)),
        "tiny_small_recall_050": _fraction(tiny_050, len(tiny_indices)),
        "core_subset_full_violation_count": int(
            sum(not row["core_subset_full"] for row in candidate_rows)
        ),
        "best_iou_by_gt": best_by_object.tolist(),
        "candidate_rows": candidate_rows,
    }


def diagnose_category_candidates(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    run_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    trace_output: Path,
    analysis_output: Path,
    postprocess_survival_root: Path | None = None,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Classify reachable GT failures using the exact raw-formation trace."""

    registered_scenes = _registered_scene_ids(scene_ids)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins is not None else None
    rows: list[dict[str, Any]] = []
    for scene_id in registered_scenes:
        survival_by_candidate = _load_scene_postprocess_survival(
            postprocess_survival_root, scene_id
        )
        trace = load_candidate_formation_trace(
            run_root / "candidate_trace" / scene_id
        )
        c0 = load_candidate_bank(
            run_root / "bank" / scene_id / "C0-legacy"
        )
        context = _scene_context(
            scene_id=scene_id,
            scene=scenes[scene_id],
            gt_dir=gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        if postprocess_survival_root is None:
            survival_by_candidate = _all_c0_postprocess_survival(
                c0, np.asarray(context["gaussian_xyz"], dtype=np.float64)
            )
        objects = context["objects"]
        mapping = context["mapping"]
        selected_projected = _project_gaussian_labels(
            np.asarray(trace.semantic_selected_class_index, dtype=np.int64),
            mapping.gt_to_gaussian,
        )
        raw_projected = _project_gaussian_labels(
            np.asarray(trace.raw_cluster_membership, dtype=np.int64),
            mapping.gt_to_gaussian,
        )
        raw_iou, raw_intersections, raw_predicted = _stage_iou(
            raw_projected,
            len(trace.raw_cluster_rows),
            context["object_index"],
            context["object_counts"],
        )
        taxonomy_class_to_id = {
            str(name): index
            for index, name in enumerate(taxonomy.canonical_classes)
        }
        raw_class = np.asarray(
            [
                taxonomy_class_to_id[str(row["branch_class"])]
                for row in trace.raw_cluster_rows
            ],
            dtype=np.int64,
        )
        c0_projected = _project_gaussian_labels(
            np.asarray(c0.branch_full_labels, dtype=np.int64),
            mapping.gt_to_gaussian,
        )
        full_iou, full_intersections, full_predicted = _stage_iou(
            c0_projected,
            len(c0.candidates),
            context["object_index"],
            context["object_counts"],
        )
        raw_to_candidate = {
            int(row["raw_cluster_id"]): row.get("retained_candidate_id")
            for row in trace.raw_cluster_rows
        }
        reverse_semantic = np.asarray(context["reverse_semantic"], dtype=np.int64)
        reverse_instance = np.asarray(context["reverse_instance"], dtype=np.int64)
        sample_rank = np.asarray(trace.sample_rank, dtype=np.int64)
        selected_class = np.asarray(
            trace.semantic_selected_class_index, dtype=np.int64
        )
        for object_index, item in enumerate(objects):
            selected_coverage = _fraction(
                np.count_nonzero(
                    selected_projected[item.point_indices] == item.class_id
                ),
                item.point_count,
            )
            if selected_coverage < 0.25:
                continue
            sampled_object = (
                (sample_rank >= 0)
                & (selected_class == item.class_id)
                & (reverse_semantic == item.class_id)
                & (reverse_instance == item.instance_id)
            )
            sampled_count = int(np.count_nonzero(sampled_object))
            eligible_raw = np.flatnonzero(raw_class == item.class_id)
            if len(eligible_raw):
                raw_f1 = np.divide(
                    2.0 * raw_intersections[eligible_raw, object_index],
                    raw_predicted[eligible_raw]
                    + context["object_counts"][object_index],
                    out=np.zeros(len(eligible_raw), dtype=np.float64),
                    where=(
                        raw_predicted[eligible_raw]
                        + context["object_counts"][object_index]
                    )
                    > 0,
                )
                best_local = int(np.argmax(raw_f1))
                best_raw = int(eligible_raw[best_local])
                best_raw_f1 = float(raw_f1[best_local])
                best_raw_iou = float(raw_iou[best_raw, object_index])
                raw_precision = _fraction(
                    raw_intersections[best_raw, object_index],
                    raw_predicted[best_raw],
                )
            else:
                best_raw = None
                best_raw_f1 = 0.0
                best_raw_iou = 0.0
                raw_precision = 0.0
            retained = raw_to_candidate.get(best_raw) if best_raw is not None else None
            if retained is None:
                full_value = 0.0
                full_precision = 0.0
            else:
                retained = int(retained)
                full_value = float(full_iou[retained, object_index])
                full_precision = _fraction(
                    full_intersections[retained, object_index],
                    full_predicted[retained],
                )
            survival = (
                survival_by_candidate.get(retained)
                if retained is not None
                else None
            )
            if (
                postprocess_survival_root is not None
                and retained is not None
                and survival is None
            ):
                raise ValueError(
                    f"{scene_id}: postprocess survival lacks candidate {retained}"
                )
            postprocess_loss_stage = _postprocess_loss_stage(survival)

            status = _candidate_failure_status(
                sampled_count=sampled_count,
                best_raw_f1=best_raw_f1,
                retained_candidate_id=retained,
                best_raw_iou=best_raw_iou,
                full_iou=full_value,
                best_raw_precision=raw_precision,
                full_precision=full_precision,
                postprocess_loss_stage=postprocess_loss_stage,
            )
            rows.append(
                {
                    "scene_id": scene_id,
                    "gt_class": item.class_name,
                    "gt_instance_id": item.instance_id,
                    "size_bin": item.size_bin,
                    "semantic_coverage": selected_coverage,
                    "sampled_same_object_count": sampled_count,
                    "best_raw_cluster_id": best_raw,
                    "best_raw_f1": best_raw_f1,
                    "best_raw_iou": best_raw_iou,
                    "best_raw_precision": raw_precision,
                    "retained_candidate_id": retained,
                    "full_iou": full_value,
                    "full_precision": full_precision,
                    "postprocess_survival_observed": survival is not None,
                    "postprocess_candidate_accepted": (
                        survival["accepted"] if survival is not None else None
                    ),
                    "survived_post_knn": (
                        survival["survived_post_knn"]
                        if survival is not None
                        else None
                    ),
                    "survived_post_filter": (
                        survival["survived_post_filter"]
                        if survival is not None
                        else None
                    ),
                    "postprocess_loss_stage": postprocess_loss_stage,
                    "failure_status": status,
                }
            )

    counts = Counter(str(row["failure_status"]) for row in rows)
    failed = [row for row in rows if row["failure_status"] != "candidate_formation_healthy"]
    sample_majority = bool(
        failed
        and counts["sample_starved"] > len(failed) / 2.0
    )
    sufficiently_sampled = [
        row for row in failed if row["failure_status"] != "sample_starved"
    ]
    raw_majority = bool(
        sufficiently_sampled
        and counts["raw_clustering_failed"] > len(sufficiently_sampled) / 2.0
    )
    if len(rows) < 8:
        next_action = "extend-trace-only-to-dev8"
    elif sample_majority:
        next_action = "compare-nested-sample-cap-5000-10000"
    elif raw_majority:
        next_action = "evaluate-affinity-auroc-and-oracle-seed"
    else:
        next_action = "evaluate-C1-C2-full-assignment-repair"
    analysis = {
        "schema": "saga-candidate-formation-root-cause-v1",
        "scene_count": len(set(row["scene_id"] for row in rows)),
        "diagnosable_object_count": len(rows),
        "failure_status_counts": dict(sorted(counts.items())),
        "sample_starved_is_majority_of_failures": sample_majority,
        "raw_clustering_is_majority_of_sufficiently_sampled_failures": raw_majority,
        "postprocess_loss_count": int(counts["postprocess_loss"]),
        "postprocess_survival_audited": True,
        "postprocess_survival_source": (
            "provided_replay_diagnostics"
            if postprocess_survival_root is not None
            else "offline_all_C0_common_legacy_replay"
        ),
        "next_action": next_action,
        "gt_boundary": "offline_diagnosis_only",
    }
    write_rows(trace_output, rows)
    write_json(analysis_output, analysis)
    return analysis


def _aggregate_condition(
    condition: str, scene_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidate_count = sum(int(row["candidate_count"]) for row in scene_rows)
    count_025 = sum(int(row["same_class_iou_025_count"]) for row in scene_rows)
    count_050 = sum(int(row["same_class_iou_050_count"]) for row in scene_rows)
    point_weights = sum(
        sum(int(item["full_point_count"]) for item in row["candidate_rows"])
        for row in scene_rows
    )
    unsupported = sum(
        sum(
            float(item["unsupported_fraction"]) * int(item["full_point_count"])
            for item in row["candidate_rows"]
        )
        for row in scene_rows
    )
    tiny_count = sum(int(row["tiny_small_gt_count"]) for row in scene_rows)
    tiny_025 = sum(
        float(row["tiny_small_recall_025"]) * int(row["tiny_small_gt_count"])
        for row in scene_rows
    )
    return {
        "condition": condition,
        "scene_count": len(scene_rows),
        "candidate_count": candidate_count,
        "same_class_iou_025_count": count_025,
        "same_class_iou_050_count": count_050,
        "same_class_iou_050_scene_count": sum(
            int(row["same_class_iou_050_count"]) > 0 for row in scene_rows
        ),
        "candidate_precision_025": _fraction(count_025, candidate_count),
        "unsupported_fraction": _fraction(unsupported, point_weights),
        "tiny_small_gt_count": tiny_count,
        "tiny_small_recall_025": _fraction(tiny_025, tiny_count),
        "core_subset_full_violation_count": sum(
            int(row["core_subset_full_violation_count"]) for row in scene_rows
        ),
        "per_scene": [dict(row) for row in scene_rows],
    }


def evaluate_category_candidates(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    run_root: Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    phase: str,
    selected_condition: str | None = None,
    frozen_repair_artifact: Path | None = None,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate and gate C0/C1/C2 with physical scenes as the unit."""

    if phase not in {"dev2", "dev8"}:
        raise ValueError("phase must be 'dev2' or 'dev8'")
    frozen_identity = None
    if phase == "dev2":
        if selected_condition is not None:
            raise ValueError("selected_condition must be omitted during dev2 selection")
        if frozen_repair_artifact is not None:
            raise ValueError("frozen_repair_artifact must be omitted during dev2 selection")
        selected = None
        conditions = tuple(CANDIDATE_REPAIR_CONDITIONS)
    else:
        if frozen_repair_artifact is None:
            raise ValueError("dev8 requires the DEV2 frozen repair-arm artifact")
        selected, _ = _resolve_frozen_repair_arm(
            frozen_repair_artifact,
            run_root=run_root,
        )
        if selected_condition is not None and selected_condition != selected:
            raise ValueError(
                "selected_condition disagrees with the DEV2 frozen repair-arm artifact"
            )
        conditions = ("C0-legacy", selected)
        frozen_path = Path(frozen_repair_artifact)
        frozen_identity = {
            "path": str(frozen_path.resolve()),
            "sha256": sha256_file(frozen_path),
        }
    registered_scenes = _registered_scene_ids(scene_ids, phase=phase)
    scenes = load_scene_runtime_manifest(runtime_manifest)
    size_spec = load_json(size_bins) if size_bins is not None else None
    by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    for scene_id in registered_scenes:
        context = _scene_context(
            scene_id=scene_id,
            scene=scenes[scene_id],
            gt_dir=gt_dir,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=radius_m,
            min_region_size=min_region_size,
        )
        for condition, bank in _load_scene_banks(
            run_root, scene_id, conditions
        ).items():
            by_condition[condition].append(
                _candidate_scene_metrics(scene_id, bank, context, taxonomy)
            )
    aggregates = {
        condition: _aggregate_condition(condition, rows)
        for condition, rows in by_condition.items()
    }
    c0 = aggregates["C0-legacy"]
    arm_gates: dict[str, Any] = {}
    for condition in conditions[1:] if phase == "dev2" else ():
        arm = aggregates[condition]
        c0_scenes = {row["scene_id"]: row for row in c0["per_scene"]}
        arm_scenes = {row["scene_id"]: row for row in arm["per_scene"]}
        best_drop = []
        improved_scenes = 0
        for scene_id in c0_scenes:
            left = np.asarray(c0_scenes[scene_id]["best_iou_by_gt"], dtype=np.float64)
            right = np.asarray(arm_scenes[scene_id]["best_iou_by_gt"], dtype=np.float64)
            best_drop.append(float(np.max(left - right)) if len(left) else 0.0)
            improved_scenes += int(
                (
                    int(arm_scenes[scene_id]["same_class_iou_050_count"]),
                    int(arm_scenes[scene_id]["same_class_iou_025_count"]),
                    float(arm_scenes[scene_id]["candidate_precision_025"]),
                    -float(arm_scenes[scene_id]["unsupported_fraction"]),
                )
                > (
                    int(c0_scenes[scene_id]["same_class_iou_050_count"]),
                    int(c0_scenes[scene_id]["same_class_iou_025_count"]),
                    float(c0_scenes[scene_id]["candidate_precision_025"]),
                    -float(c0_scenes[scene_id]["unsupported_fraction"]),
                )
            )
        precision_relative = _fraction(
            arm["candidate_precision_025"] - c0["candidate_precision_025"],
            c0["candidate_precision_025"],
        ) if c0["candidate_precision_025"] else (
            1.0 if arm["candidate_precision_025"] > 0 else 0.0
        )
        unsupported_drop = (
            float(c0["unsupported_fraction"])
            - float(arm["unsupported_fraction"])
        )
        checks = {
            "iou025_not_lower": arm["same_class_iou_025_count"]
            >= c0["same_class_iou_025_count"],
            "iou050_not_lower": arm["same_class_iou_050_count"]
            >= c0["same_class_iou_050_count"],
            "precision_or_unsupported_improved": precision_relative >= 0.25
            or unsupported_drop >= 0.10,
            "tiny_small_recall_not_lower": arm["tiny_small_recall_025"]
            >= c0["tiny_small_recall_025"],
            "candidate_count_within_1.25x": arm["candidate_count"]
            <= 1.25 * max(c0["candidate_count"], 1),
            "core_subset_full": arm["core_subset_full_violation_count"] == 0,
            # Both safeguards are explicitly preregistered for the two-scene
            # repair choice.  They are gates, not descriptive diagnostics:
            # one scene must genuinely improve and no GT object in the other
            # scene may lose more than 0.05 best IoU.
            "at_least_one_scene_improved": improved_scenes >= 1,
            "per_gt_drop_at_most_0.05": max(best_drop, default=0.0) <= 0.05,
        }
        arm_gates[condition] = {
            "passed": all(checks.values()),
            "checks": checks,
            "candidate_precision_relative_change": precision_relative,
            "unsupported_fraction_drop": unsupported_drop,
            "improved_scene_count": improved_scenes,
            "maximum_gt_best_iou_drop": max(best_drop, default=0.0),
        }

    if phase == "dev2":
        passed = [
            condition
            for condition, gate in arm_gates.items()
            if gate["passed"]
        ]
        if passed:
            selected = max(
                passed,
                key=lambda condition: (
                    aggregates[condition]["same_class_iou_050_count"],
                    aggregates[condition]["same_class_iou_025_count"],
                    aggregates[condition]["candidate_precision_025"],
                    -aggregates[condition]["unsupported_fraction"],
                    -aggregates[condition]["candidate_count"],
                    condition == "C1-consistent-envelope",
                ),
            )
        else:
            selected = None
    dev8_gate = None
    if phase == "dev8" and selected is not None:
        arm = aggregates[selected]
        c0_scenes = {row["scene_id"]: row for row in c0["per_scene"]}
        arm_scenes = {row["scene_id"]: row for row in arm["per_scene"]}
        positive = 0
        negative = 0
        for scene_id in c0_scenes:
            arm_key = (
                int(arm_scenes[scene_id]["same_class_iou_050_count"]),
                int(arm_scenes[scene_id]["same_class_iou_025_count"]),
                float(arm_scenes[scene_id]["candidate_precision_025"]),
                -float(arm_scenes[scene_id]["unsupported_fraction"]),
            )
            c0_key = (
                int(c0_scenes[scene_id]["same_class_iou_050_count"]),
                int(c0_scenes[scene_id]["same_class_iou_025_count"]),
                float(c0_scenes[scene_id]["candidate_precision_025"]),
                -float(c0_scenes[scene_id]["unsupported_fraction"]),
            )
            positive += int(arm_key > c0_key)
            negative += int(arm_key < c0_key)
        checks = {
            "iou050_at_least_12": arm["same_class_iou_050_count"] >= 12,
            "iou050_at_least_4_scenes": arm["same_class_iou_050_scene_count"] >= 4,
            "candidate_precision025_at_least_5pct": arm["candidate_precision_025"] >= 0.05,
            "tiny_small_recall025_at_least_0.20": arm["tiny_small_recall_025"] >= 0.20,
            "positive_scenes_more_than_negative": positive > negative,
            "candidate_count_within_1.25x": arm["candidate_count"]
            <= 1.25 * max(c0["candidate_count"], 1),
            "core_subset_full": arm["core_subset_full_violation_count"] == 0,
        }
        dev8_gate = {
            "passed": all(checks.values()),
            "checks": checks,
            "positive_scene_count": positive,
            "negative_scene_count": negative,
        }

    metric_rows = []
    for condition in conditions:
        row = {
            key: value
            for key, value in aggregates[condition].items()
            if key != "per_scene"
        }
        row["phase"] = phase
        metric_rows.append(row)
    analysis = {
        "schema": "saga-category-candidate-repair-analysis-v1",
        "phase": phase,
        "scene_ids": list(registered_scenes),
        "conditions": aggregates,
        "dev2_arm_gates": arm_gates,
        "selected_condition": selected,
        "frozen_repair_artifact": frozen_identity,
        "dev8_health_gate": dev8_gate,
        "category_prior_tested": False,
        "gt_boundary": "evaluation_only",
    }
    write_rows(metrics_output, metric_rows)
    write_json(analysis_output, analysis)
    return analysis


__all__ = [
    "DEV2_SCENE_IDS",
    "DEV8_SCENE_IDS",
    "diagnose_category_candidates",
    "evaluate_category_candidates",
]
