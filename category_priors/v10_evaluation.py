from __future__ import annotations

"""Offline V10 association and official prediction evaluation adapters."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_json, write_json, write_rows
from .taxonomy import Taxonomy
from .v10_metrics import (
    adapt_v10_persisted_bank,
    analyse_v10_rows,
    build_gaussian_gt_index,
    evaluate_v10_audit,
    ground_truth_objects_from_arrays,
    load_v10_scene_geometry,
)
from .v9_metrics import evaluate_v9_predictions


def _size_bin_mapping(
    gt_xyz: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    class_names: Sequence[str],
    size_spec: Mapping[str, Any] | None,
) -> dict[tuple[str, int], str]:
    if size_spec is None:
        return {}
    bounds = size_spec["boundaries_m"]
    tiny = float(bounds["tiny_max_m"])
    small = float(bounds["small_max_m"])
    valid = (semantic >= 0) & (instance >= 0)
    result: dict[tuple[str, int], str] = {}
    for class_id, instance_id in sorted(
        set(zip(semantic[valid].tolist(), instance[valid].tolist()))
    ):
        if not 0 <= int(class_id) < len(class_names):
            continue
        mask = valid & (semantic == class_id) & (instance == instance_id)
        extent = np.ptp(np.asarray(gt_xyz)[mask], axis=0)
        diagonal = float(np.linalg.norm(extent))
        if diagonal <= tiny:
            result[(str(class_names[int(class_id)]), int(instance_id))] = "tiny"
        elif diagonal <= small:
            result[(str(class_names[int(class_id)]), int(instance_id))] = "small"
    return result


def audit_v10_associations(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    bank_root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    classifiers: Sequence[str],
    taxonomy: Taxonomy,
    rows_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Evaluate persisted real edges and the complete eight-stage funnel."""

    size_spec = load_json(size_bins) if size_bins is not None else None
    combined_rows: list[dict[str, Any]] = []
    analyses: dict[str, Any] = {}
    for condition in map(str, conditions):
        for classifier in map(str, classifiers):
            condition_rows: list[dict[str, Any]] = []
            for scene_id in map(str, scene_ids):
                persisted = adapt_v10_persisted_bank(
                    bank_root / condition / scene_id, classifier=classifier
                )
                if persisted.scene_id != scene_id:
                    raise ValueError(
                        f"V10 bank scene mismatch for {condition}/{scene_id}"
                    )
                gt_xyz, semantic, instance, gaussian_xyz = load_v10_scene_geometry(
                    runtime_manifest=runtime_manifest,
                    gt_dir=gt_dir,
                    scene_id=scene_id,
                )
                sizes = _size_bin_mapping(
                    gt_xyz,
                    semantic,
                    instance,
                    taxonomy.canonical_classes,
                    size_spec,
                )
                ground_truth = ground_truth_objects_from_arrays(
                    scene_id,
                    semantic,
                    instance,
                    taxonomy.canonical_classes,
                    size_bins=sizes,
                    min_region_size=min_region_size,
                )
                index = build_gaussian_gt_index(
                    gt_xyz, gaussian_xyz, radius_m=radius_m
                )
                result = evaluate_v10_audit(
                    ground_truth=ground_truth,
                    gaussian_gt_index=index,
                    fragments=persisted.fragments,
                    accepted_edges=persisted.accepted_edges,
                    stage_candidates=persisted.stage_candidates,
                )
                for row in result["rows"]:
                    condition_rows.append(
                        {
                            "condition": condition,
                            "classifier": classifier,
                            **dict(row),
                        }
                    )
            # Keep each condition/classifier separate so duplicated GT rows
            # across alternatives never mix in one denominator.
            analysis = analyse_v10_rows(condition_rows)
            final = analysis["stages"]["final_candidate"]
            association = analysis["accepted_fragment_pairs"]
            analyses[f"{condition}/{classifier}"] = {
                **analysis,
                "gate_metrics": {
                    "candidate_count": int(final["candidate_count"]),
                    "geometric_match_050_count": int(
                        final["candidate_match_050_count"]
                    ),
                    "geometric_match_050_scene_count": int(
                        final.get("candidate_match_050_scene_count", 0)
                    ),
                    "geometric_candidate_precision_025": float(
                        final["candidate_precision_025"]
                    ),
                    "geometric_tiny_small_recall_025": float(
                        final["geometric_tiny_small_recall_025"]
                    ),
                    "geometric_tiny_small_recall_050": float(
                        final["geometric_tiny_small_recall_050"]
                    ),
                    "same_class_match_025_count": int(
                        final["same_class_candidate_match_025_count"]
                    ),
                    "same_class_match_050_count": int(
                        final["same_class_candidate_match_050_count"]
                    ),
                    "same_class_match_050_scene_count": int(
                        final["same_class_candidate_match_050_scene_count"]
                    ),
                    "same_class_candidate_precision_025": float(
                        final["same_class_candidate_precision_025"]
                    ),
                    "same_class_tiny_small_recall_025": float(
                        final["same_class_tiny_small_recall_025"]
                    ),
                    "same_class_tiny_small_recall_050": float(
                        final["same_class_tiny_small_recall_050"]
                    ),
                    "score_iou_spearman": float(final["score_iou_spearman"]),
                    "identifiable_association_precision": float(
                        association["identifiable_precision"]
                    ),
                    "all_edge_precision": float(association["all_edge_precision"]),
                    "unknown_edge_rate": float(association["unknown_rate"]),
                },
            }
            combined_rows.extend(condition_rows)
    payload = {
        "schema": "saga-v10-association-audit-v1",
        "scene_ids": list(map(str, scene_ids)),
        "conditions": analyses,
    }
    write_rows(rows_output, combined_rows)
    write_json(analysis_output, payload)
    return payload


def evaluate_v10_replays(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    replay_root: Path,
    structure_condition: str,
    classifier: str,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    taxonomy: Taxonomy,
    metrics_output: Path,
    analysis_output: Path,
    size_bins: Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
    viewer_output: Path | None = None,
) -> dict[str, Any]:
    """Run the established official protocol on strict V10 replay outputs."""

    payload = evaluate_v9_predictions(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        prediction_root=Path(replay_root) / str(structure_condition) / str(classifier),
        scene_ids=scene_ids,
        conditions=conditions,
        taxonomy=taxonomy,
        metrics_output=metrics_output,
        analysis_output=analysis_output,
        size_bins=size_bins,
        radius_m=radius_m,
        min_region_size=min_region_size,
        viewer_output=viewer_output,
    )
    result = {
        **payload,
        "schema": "saga-v10-object-system-analysis-v1",
        "structure_condition": str(structure_condition),
        "classifier": str(classifier),
    }
    write_json(analysis_output, result)
    return result
