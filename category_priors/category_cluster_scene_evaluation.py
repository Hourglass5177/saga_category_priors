from __future__ import annotations

"""Scene-I/O adapter for the pure section-31 cluster evaluator.

Ground truth is loaded only here.  The bank worker and clustering modules have
no import path to this adapter.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_candidate_evaluation import _scene_context
from .category_cluster_bank import (
    G1_MUTUAL_LOCAL_GRAPH,
    R0_LEGACY,
    R1_METRIC_HDBSCAN,
    R2_ANCHORED_HDBSCAN,
)
from .category_cluster_evaluation import (
    ClusterEvaluationScene,
    evaluate_cluster_candidate_banks,
)
from .category_denoise import load_candidate_bank
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


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


def _evaluation_scene(
    *,
    scene_id: str,
    scene: dict[str, Any],
    gt_dir: Path,
    taxonomy: Taxonomy,
    size_spec: dict[str, Any] | None,
    radius_m: float,
    min_region_size: int,
) -> ClusterEvaluationScene:
    context = _scene_context(
        scene_id=scene_id,
        scene=scene,
        gt_dir=gt_dir,
        taxonomy=taxonomy,
        size_spec=size_spec,
        radius_m=radius_m,
        min_region_size=min_region_size,
    )
    objects = context["objects"]
    mapping = context["mapping"]
    gt_point_object = np.asarray(context["object_index"], dtype=np.int64)
    gaussian_to_gt_point = np.asarray(
        mapping.gaussian_to_gt.indices, dtype=np.int64
    )
    gaussian_to_object = np.full(len(gaussian_to_gt_point), -1, dtype=np.int64)
    valid = gaussian_to_gt_point >= 0
    gaussian_to_object[valid] = gt_point_object[gaussian_to_gt_point[valid]]
    return ClusterEvaluationScene(
        scene_id=scene_id,
        gt_to_gaussian_indices=np.asarray(
            mapping.gt_to_gaussian.indices, dtype=np.int64
        ),
        gt_point_object_indices=gt_point_object,
        gt_object_class_ids=np.asarray(
            [item.class_id for item in objects], dtype=np.int64
        ),
        gt_object_size_bins=tuple(str(item.size_bin) for item in objects),
        gt_object_instance_ids=np.asarray(
            [item.instance_id for item in objects], dtype=np.int64
        ),
        gaussian_to_gt_object_indices=gaussian_to_object,
        class_name_to_id={
            str(name): index
            for index, name in enumerate(taxonomy.canonical_classes)
        },
    )


def evaluate_category_cluster_run(
    *,
    runtime_manifest: str | Path,
    gt_dir: str | Path,
    run_root: str | Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    phase: str,
    metrics_output: str | Path,
    analysis_output: str | Path,
    selected_condition: str | None = None,
    primary_analysis: str | Path | None = None,
    frozen_selection_artifact: str | Path | None = None,
    size_bins: str | Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Load registered banks and apply DEV2/DEV8 gates by physical scene."""

    if phase not in {"dev2", "dev8"}:
        raise ValueError("phase must be 'dev2' or 'dev8'")
    requested_scenes = tuple(map(str, scene_ids))
    expected_scenes = DEV2_SCENE_IDS if phase == "dev2" else DEV8_SCENE_IDS
    if len(requested_scenes) != len(set(requested_scenes)) or set(
        requested_scenes
    ) != set(expected_scenes):
        raise ValueError(
            f"{phase} requires the exact registered scene set {expected_scenes}"
        )
    registered_scenes = expected_scenes
    if phase == "dev8" and selected_condition not in {
        R1_METRIC_HDBSCAN,
        R2_ANCHORED_HDBSCAN,
        G1_MUTUAL_LOCAL_GRAPH,
    }:
        raise ValueError("dev8 requires one DEV2-selected repair condition")
    if phase == "dev8":
        if frozen_selection_artifact is None:
            raise ValueError("dev8 requires the frozen DEV2 selection artifact")
        frozen = load_json(frozen_selection_artifact)
        if (
            frozen.get("phase") != "dev2"
            or frozen.get("selected_condition") != selected_condition
            or not bool((frozen.get("selected_gate") or {}).get("passed"))
        ):
            raise ValueError("DEV2 selection artifact does not authorize DEV8")
    scenes = load_scene_runtime_manifest(runtime_manifest)
    missing = sorted(set(registered_scenes).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest lacks scenes: {missing}")
    size_spec = load_json(size_bins) if size_bins is not None else None
    if size_spec is not None and not isinstance(size_spec, dict):
        raise TypeError("size bins must be a JSON object")
    root = Path(run_root).resolve()

    evaluation_scenes: dict[str, ClusterEvaluationScene] = {}
    banks_by_condition: dict[str, dict[str, Any]] = {}
    for scene_id in registered_scenes:
        evaluation_scenes[scene_id] = _evaluation_scene(
            scene_id=scene_id,
            scene=scenes[scene_id],
            gt_dir=Path(gt_dir).resolve(),
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=float(radius_m),
            min_region_size=int(min_region_size),
        )
        conditions = (
            (R0_LEGACY, selected_condition)
            if phase == "dev8"
            else (
                R0_LEGACY,
                R1_METRIC_HDBSCAN,
                R2_ANCHORED_HDBSCAN,
                *((G1_MUTUAL_LOCAL_GRAPH,) if primary_analysis is not None else ()),
            )
        )
        for condition in conditions:
            if condition is None:
                continue
            bank_path = root / "bank" / scene_id / condition
            if not (bank_path / "bank_labels.npz").is_file():
                if phase == "dev2" and condition == G1_MUTUAL_LOCAL_GRAPH:
                    continue
                raise FileNotFoundError(bank_path / "bank_labels.npz")
            banks_by_condition.setdefault(condition, {})[scene_id] = (
                load_candidate_bank(bank_path)
            )
    # G1 is all-or-none across DEV2; a partial fallback run is invalid.
    if G1_MUTUAL_LOCAL_GRAPH in banks_by_condition and len(
        banks_by_condition[G1_MUTUAL_LOCAL_GRAPH]
    ) != len(registered_scenes):
        raise ValueError("G1 fallback banks are incomplete across DEV2")
    if phase == "dev2" and G1_MUTUAL_LOCAL_GRAPH in banks_by_condition:
        if primary_analysis is None:
            raise ValueError("G1 evaluation requires the prior R1/R2 DEV2 analysis")
        primary = load_json(primary_analysis)
        primary_gates = primary.get("gates", {})
        if (
            primary.get("phase") != "dev2"
            or primary.get("selected_condition") is not None
            or bool((primary_gates.get(R1_METRIC_HDBSCAN) or {}).get("passed"))
            or bool((primary_gates.get(R2_ANCHORED_HDBSCAN) or {}).get("passed"))
        ):
            raise ValueError("G1 is allowed only after both primary DEV2 arms fail")
    result = evaluate_cluster_candidate_banks(
        evaluation_scenes,
        banks_by_condition,
        phase=phase,
        selected_condition=selected_condition,
    )
    metric_rows = []
    for condition, aggregate in result["conditions"].items():
        for scene_row in aggregate["per_scene"]:
            metric_rows.append(
                {
                    "phase": phase,
                    "condition": condition,
                    **{
                        key: value
                        for key, value in scene_row.items()
                        if key != "candidate_rows"
                    },
                }
            )
    write_rows(metrics_output, metric_rows)
    write_json(analysis_output, result)
    return result


__all__ = ["evaluate_category_cluster_run"]
