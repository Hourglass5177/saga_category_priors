from __future__ import annotations

"""Scene-I/O adapter for the GT-aware section-33 offline evaluator.

Runtime fragment construction and merging live in
``category_fragment_merge_runner`` and cannot import this module.  This adapter
opens GT only after both paired replay artifacts are complete and immutable.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .category_cluster_evaluation import evaluate_cluster_scene
from .category_cluster_scene_evaluation import _evaluation_scene
from .category_fragment_merge_evaluation import (
    DEV2_SCENE_IDS,
    DEV8_SCENE_IDS,
    evaluate_category_fragment_merge,
)
from .category_fragment_merge_runner import load_category_fragment_scene
from .io import load_json, write_json, write_rows
from .runner import load_scene_runtime_manifest
from .taxonomy import Taxonomy


def _validate_registered_scenes(
    scene_ids: Sequence[str], phase: str
) -> tuple[str, ...]:
    if phase not in {"dev2", "dev8"}:
        raise ValueError("phase must be 'dev2' or 'dev8'")
    requested = tuple(map(str, scene_ids))
    expected = DEV2_SCENE_IDS if phase == "dev2" else DEV8_SCENE_IDS
    if len(requested) != len(set(requested)) or set(requested) != set(expected):
        raise ValueError(f"{phase} requires the exact registered scene set {expected}")
    # Registered order, not caller order, controls evaluation and output.
    return expected


def _require_dev2_authorization(analysis_output: str | Path) -> dict[str, Any]:
    authorization = (
        Path(analysis_output).resolve().parent
        / "category_fragment_merge_dev2_analysis.json"
    )
    if not authorization.is_file():
        raise FileNotFoundError(
            "DEV8 requires the sibling category_fragment_merge_dev2_analysis.json"
        )
    payload = load_json(authorization)
    if (
        payload.get("schema") != "saga-category-fragment-merge-evaluation-v1"
        or payload.get("phase") != "dev2"
        or payload.get("scene_ids") != list(DEV2_SCENE_IDS)
        or payload.get("passed") is not True
        or payload.get("conclusion") != "dev2-passed-proceed-to-dev8"
    ):
        raise ValueError("DEV2 analysis artifact does not authorize DEV8")
    return payload


def _metric_rows(result: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, condition in (
        ("graph_oracle", "graph-oracle"),
        ("uniform", "U-global"),
        ("class", "D-class"),
    ):
        for scene in result[source]["per_scene"]:
            rows.append(
                {
                    "phase": phase,
                    "condition": condition,
                    **{
                        key: value
                        for key, value in scene.items()
                        if key not in {"candidate_rows", "best_iou_by_gt"}
                    },
                }
            )
    mechanical = result.get("mechanical_effect", {})
    for scene in mechanical.get("per_scene", []):
        rows.append(
            {
                "phase": phase,
                "condition": "U-vs-D-mechanical-effect",
                **dict(scene),
            }
        )
    for scene in result.get("raw_fragment_identity", {}).get("per_scene", []):
        rows.append(
            {
                "phase": phase,
                "condition": "raw-fragment-identity",
                **{
                    key: value
                    for key, value in scene.items()
                    if key not in {"candidate_rows", "best_iou_by_gt"}
                },
            }
        )
    return rows


def evaluate_category_fragment_merge_run(
    *,
    runtime_manifest: str | Path,
    gt_dir: str | Path,
    run_root: str | Path,
    scene_ids: Sequence[str],
    taxonomy: Taxonomy,
    phase: str,
    metrics_output: str | Path,
    analysis_output: str | Path,
    size_bins: str | Path | None = None,
    radius_m: float = 0.05,
    min_region_size: int = 100,
) -> dict[str, Any]:
    """Load complete paired scenes, evaluate them, and persist parquet/JSON."""

    registered = _validate_registered_scenes(scene_ids, phase)
    if not math.isfinite(float(radius_m)) or float(radius_m) <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if isinstance(min_region_size, bool) or int(min_region_size) != min_region_size:
        raise TypeError("min_region_size must be an integer")
    if int(min_region_size) <= 0:
        raise ValueError("min_region_size must be positive")
    authorization = (
        _require_dev2_authorization(analysis_output) if phase == "dev8" else None
    )
    scenes = load_scene_runtime_manifest(runtime_manifest)
    missing = sorted(set(registered).difference(scenes))
    if missing:
        raise ValueError(f"runtime manifest lacks scenes: {missing}")
    size_spec = load_json(size_bins) if size_bins is not None else None
    if size_spec is not None and not isinstance(size_spec, dict):
        raise TypeError("size bins must be a JSON object")
    root = Path(run_root).resolve()
    gt_root = Path(gt_dir).resolve()
    evaluation_scenes = {}
    graphs = {}
    uniform_results = {}
    class_results = {}
    raw_fragment_metrics = []
    for scene_id in registered:
        artifacts = load_category_fragment_scene(root / scene_id)
        evaluation_scene = _evaluation_scene(
            scene_id=scene_id,
            scene=scenes[scene_id],
            gt_dir=gt_root,
            taxonomy=taxonomy,
            size_spec=size_spec,
            radius_m=float(radius_m),
            min_region_size=int(min_region_size),
        )
        evaluation_scenes[scene_id] = evaluation_scene
        raw_fragment_metrics.append(
            evaluate_cluster_scene(evaluation_scene, artifacts.raw_bank)
        )
        graphs[scene_id] = artifacts.graph
        uniform_results[scene_id] = artifacts.uniform
        class_results[scene_id] = artifacts.class_shrunk
    result = evaluate_category_fragment_merge(
        scenes=evaluation_scenes,
        graphs=graphs,
        uniform_results=uniform_results,
        class_results=class_results,
        phase=phase,
    )
    raw_identity_observed = {
        "candidate_count": sum(row.candidate_count for row in raw_fragment_metrics),
        "same_class_iou_025_count": sum(
            row.same_class_iou_025_count for row in raw_fragment_metrics
        ),
        "same_class_iou_050_count": sum(
            row.same_class_iou_050_count for row in raw_fragment_metrics
        ),
    }
    raw_identity_expected = (
        {
            "candidate_count": 5033,
            "same_class_iou_025_count": 0,
            "same_class_iou_050_count": 0,
        }
        if phase == "dev2"
        else None
    )
    raw_identity_passed = (
        raw_identity_expected is None or raw_identity_observed == raw_identity_expected
    )
    result["raw_fragment_identity"] = {
        "source": "section-32.4-native-2k-grounded-predicted-32-top1",
        "gate_applies": phase == "dev2",
        "expected": raw_identity_expected,
        "observed": raw_identity_observed,
        "passed": raw_identity_passed,
        "per_scene": [row.as_dict() for row in raw_fragment_metrics],
    }
    if phase == "dev2" and not raw_identity_passed:
        result["passed"] = False
        result["conclusion"] = "raw-fragment-identity-mismatch-fix-wiring"
    result["evaluation_io"] = {
        "runtime_manifest": str(Path(runtime_manifest).resolve()),
        "gt_dir": str(gt_root),
        "run_root": str(root),
        "radius_m": float(radius_m),
        "min_region_size": int(min_region_size),
        "dev2_authorization": (
            None
            if authorization is None
            else {
                "phase": authorization["phase"],
                "passed": authorization["passed"],
                "conclusion": authorization["conclusion"],
            }
        ),
        "gt_loaded_only_in_scene_evaluation_adapter": True,
    }
    artifacts_root = Path(analysis_output).resolve().parent
    write_rows(
        artifacts_root / f"category_fragment_graph_{phase}.parquet",
        [
            {
                "phase": phase,
                "condition": "graph-oracle",
                **{
                    key: value
                    for key, value in scene.items()
                    if key not in {"candidate_rows", "best_iou_by_gt"}
                },
            }
            for scene in result["graph_oracle"]["per_scene"]
        ],
    )
    write_json(
        artifacts_root / f"category_fragment_graph_oracle_{phase}.json",
        result["graph_oracle"],
    )
    write_rows(metrics_output, _metric_rows(result, phase))
    write_json(analysis_output, result)
    return result


__all__ = ["evaluate_category_fragment_merge_run"]
