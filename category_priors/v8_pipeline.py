from __future__ import annotations

"""Preregistered, sequential controller for the SAGA V8 experiment."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluator import (
    apply_transform,
    load_ground_truth_npz,
    load_ply_xyz,
    map_gaussians_to_gt,
)
from .io import load_json, read_rows, write_json, write_rows
from .gaussian_object_audit import (
    GT_COLOR,
    _export_viewer_case,
    _write_colored_ply,
    evaluate_gaussian_object_precision,
)
from .taxonomy import Taxonomy, load_taxonomy
from .v7_runner import load_runtime_scenes
from .v8_analysis import (
    evaluate_v8_lifting_factorial,
    evaluate_v8_object_banks,
    paired_scannet_scene_bootstrap_from_replays,
    stage2_bank_health_gate,
)
from .v8_bank import build_v8_object_bank, replay_v8_priors
from .v8_evaluation import evaluate_v8_replays
from .v8_replay import CONDITIONS
from .v8_runner import run_v8_lifting_banks, run_v8_lifting_factorial


DEV8 = (
    "scene0645_00", "scene0025_01", "scene0046_00", "scene0474_01",
    "scene0591_02", "scene0329_02", "scene0164_03", "scene0064_01",
)
CAUSAL2 = DEV8[:2]
HOLDOUT5 = (
    "scene0231_00", "scene0608_00", "scene0356_00", "scene0011_00",
    "scene0593_00",
)


def _physical_scene_id(scene_id: str) -> str:
    return str(scene_id).rsplit("_", 1)[0]


def _scene_ids_from_spec(path: Path) -> tuple[str, ...]:
    payload = load_json(path)
    rows: Any = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        return tuple(map(str, rows))
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"scene specification must contain a list or mapping: {path}")
    ids: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            scene_id = row.get("scene_id", row.get("scan_id"))
            if scene_id is None:
                raise ValueError(f"scene row lacks scene_id: {path}")
            ids.append(str(scene_id))
        else:
            ids.append(str(row))
    return tuple(ids)


def _validate_tune_manifest(scene_ids: Sequence[str]) -> None:
    normalized = tuple(map(str, scene_ids))
    required = set(DEV8) | set(HOLDOUT5)
    physical = {_physical_scene_id(scene_id) for scene_id in normalized}
    if (
        len(normalized) != 24
        or len(set(normalized)) != 24
        or len(physical) != 13
        or not required.issubset(normalized)
    ):
        raise ValueError(
            "tune runtime must contain exactly 24 scans from 13 physical "
            "scenes and include the frozen DEV8/HOLDOUT5 scans"
        )


def _validate_locked_manifest(
    scene_ids: Sequence[str], expected_scene_ids: Sequence[str]
) -> None:
    actual = tuple(map(str, scene_ids))
    expected = tuple(map(str, expected_scene_ids))
    if len(expected) != 48 or len(set(expected)) != 48:
        raise ValueError("locked_evaluation_scenes must contain 48 unique scans")
    if set(actual) != set(expected) or len(actual) != 48:
        raise ValueError("locked runtime does not exactly match locked_evaluation_scenes")
    if len({_physical_scene_id(scene_id) for scene_id in actual}) != 48:
        raise ValueError("final48 must contain 48 distinct physical scenes")
    if "scene0019_01" not in actual or "scene0019_00" in actual:
        raise ValueError("final48 must use the scene0019_01 replacement")


def _record(status_path: Path, status: dict[str, Any], stage: str, value: Any) -> None:
    status[stage] = value
    status["updated_at_unix"] = time.time()
    write_json(status_path, status)


def _disk_guard(path: Path, minimum_gib: float = 80.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(path).free / 1024**3
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"V8 requires at least {minimum_gib:.0f} GiB free; found {free_gib:.1f} GiB"
        )


def _metrics(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): dict(row) for row in read_rows(path)}


def _mean_scene_delta(
    analysis: Mapping[str, Any], treatment: str, reference: str
) -> tuple[float, int, int]:
    ref = {
        str(row["scene_id"]): float(row["map_50_95"])
        for row in analysis["conditions"][reference]["per_scene"]
    }
    trt = {
        str(row["scene_id"]): float(row["map_50_95"])
        for row in analysis["conditions"][treatment]["per_scene"]
    }
    deltas = [trt[key] - ref[key] for key in sorted(ref)]
    return (
        float(np.mean(deltas)) if deltas else 0.0,
        sum(value > 0 for value in deltas),
        sum(value < 0 for value in deltas),
    )


def _build_banks(
    *,
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    lifting_root: Path,
    bank_root: Path,
    repo_root: Path,
    mask_source: str,
    lifting_source: str,
    sam_masks_root: Path,
    sam_checkpoint: Path,
    label_features: Path,
    feature_ply_by_scene: Mapping[str, Path] | None = None,
) -> None:
    _disk_guard(bank_root)
    run_v8_lifting_banks(
        runtime_manifest,
        scene_ids,
        lifting_root,
        repo_root,
        mask_source=mask_source,
        lifting_source=lifting_source,
        sam_masks_root=sam_masks_root,
        sam_checkpoint=sam_checkpoint,
        label_features=label_features,
        feature_ply_by_scene=feature_ply_by_scene,
    )
    for scene_id in scene_ids:
        build_v8_object_bank(lifting_root / scene_id, bank_root / scene_id)


def _evaluate_replays(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    replay_root: Path,
    scene_ids: Sequence[str],
    conditions: Sequence[str],
    taxonomy: Taxonomy,
    metrics_path: Path,
    analysis_path: Path,
    size_bins: Path,
    viewer: Path | None = None,
) -> dict[str, Any]:
    return evaluate_v8_replays(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        replay_root=replay_root,
        scene_ids=scene_ids,
        conditions=conditions,
        taxonomy=taxonomy,
        metrics_output=metrics_path,
        analysis_output=analysis_path,
        size_bins=size_bins,
        viewer_output=viewer,
    )


def _resolve_b1_output(root: Path, scene_id: str) -> Path:
    candidates = (
        root / scene_id / "seed-42" / "output.json",
        root / scene_id / "output.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def _b1_run_is_complete(
    directory: Path, expected_point_count: int | None = None
) -> bool:
    """Validate both B1 payloads before a resumable V8 skip."""
    try:
        output = load_json(directory / "output.json")
        diagnostics = load_json(directory / "diagnostics.json")
        labels = output.get("point_labels")
        instances = output.get("instances")
        diagnostic_instances = diagnostics.get("instances")
        if (
            not isinstance(labels, list)
            or not labels
            or (
                expected_point_count is not None
                and len(labels) != int(expected_point_count)
            )
            or not isinstance(instances, Mapping)
            or not isinstance(diagnostic_instances, Mapping)
        ):
            return False
        if any(
            not isinstance(values, Mapping)
            or not isinstance(values.get("class"), str)
            or not str(values["class"]).strip()
            for values in instances.values()
        ):
            return False
        if any(
            not isinstance(values, Mapping)
            or not isinstance(values.get("class"), str)
            or not str(values["class"]).strip()
            for values in diagnostic_instances.values()
        ):
            return False
        output_instance_ids = {int(value) for value in instances}
        diagnostic_instance_ids = {int(value) for value in diagnostic_instances}
        assigned = {int(value) for value in labels if int(value) >= 0}
        for raw_id, values in instances.items():
            diagnostic = diagnostic_instances.get(
                str(raw_id), diagnostic_instances.get(int(raw_id))
            )
            if not isinstance(diagnostic, Mapping):
                return False
            if str(diagnostic.get("class", "")).strip() != str(
                values["class"]
            ).strip():
                return False
            score = diagnostic.get("score")
            if (
                not isinstance(score, (int, float))
                or not np.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                return False
        # ``postprocess.py`` intentionally retains labels for every clustered
        # instance, while ``output.instances`` contains only selected ScanNet
        # classes.  The unfiltered diagnostics table is the authoritative
        # completeness record for all assigned labels.
        return (
            output_instance_ids.issubset(assigned)
            and diagnostic_instance_ids == assigned
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _run_b1_fixed(
    *, runtime_manifest: Path, pipeline: Path, output_root: Path,
    scene_ids: Sequence[str], level: str = "L0",
) -> None:
    """Run the fixed-contributor a800 baseline with strict resume checks."""
    scenes = load_runtime_scenes(runtime_manifest)
    for scene_id in map(str, scene_ids):
        scene = scenes[scene_id]
        target = output_root / level / scene_id
        point_count = len(load_ply_xyz(_gaussian_ply(scene.base_path)))
        if _b1_run_is_complete(target, point_count):
            print(f"[{level}/{scene_id}] reused", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "bash", str(pipeline), "--stage", "postprocess",
            "--base-path", str(scene.base_path), "--python", str(scene.python_bin),
            "--json-path", str(target / "output.json"),
            "--prior-metadata-path", str(target / "diagnostics.json"),
            "--progress-path", str(target / "progress.txt"),
            "--scene-scale-m-per-unit", str(scene.scene_scale_m_per_unit),
            "--teacher-prior-mode", "original", "--minimal-metadata",
            "--v7-causal-ablation", level,
        ]
        with (target / "postprocess.log").open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=pipeline.parent, stdout=log, stderr=subprocess.STDOUT
            )
        if completed.returncode or not _b1_run_is_complete(target, point_count):
            raise RuntimeError(
                f"{level}/{scene_id} failed or produced incomplete output; "
                f"see {target / 'postprocess.log'}"
            )


def _compare_b1_outputs(
    historical_root: Path | None,
    fixed_root: Path | None,
    scene_ids: Sequence[str],
) -> dict[str, Any]:
    if historical_root is None or fixed_root is None:
        return {"status": "not-configured", "all_equal": None, "scenes": []}
    rows: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        historical_path = _resolve_b1_output(historical_root, scene_id)
        fixed_path = _resolve_b1_output(fixed_root, scene_id)
        if not historical_path.is_file() or not fixed_path.is_file():
            raise FileNotFoundError(
                f"missing B1 comparison output for {scene_id}: "
                f"{historical_path}, {fixed_path}"
            )
        historical = load_json(historical_path)
        fixed = load_json(fixed_path)
        labels_equal = historical.get("point_labels") == fixed.get("point_labels")
        instances_equal = historical.get("instances") == fixed.get("instances")
        rows.append({
            "scene_id": scene_id,
            "point_labels_equal": labels_equal,
            "instances_equal": instances_equal,
            "equal": labels_equal and instances_equal,
        })
    return {
        "status": "compared",
        "all_equal": all(row["equal"] for row in rows),
        "scenes": rows,
    }


def _select_final_viewer_scenes(
    analysis: Mapping[str, Any], treatment: str
) -> dict[str, str]:
    """Select best/median/worst physical scenes without changing the method."""
    reference = {
        str(row["scene_id"]): float(row["map_50_95"])
        for row in analysis["conditions"]["U00"]["per_scene"]
    }
    treated = {
        str(row["scene_id"]): float(row["map_50_95"])
        for row in analysis["conditions"][treatment]["per_scene"]
    }
    deltas = sorted(
        ((scene_id, treated[scene_id] - value) for scene_id, value in reference.items()),
        key=lambda item: (item[1], item[0]),
    )
    if not deltas:
        raise ValueError("final viewer selection requires at least one evaluated scene")
    median_delta = float(np.median([value for _, value in deltas]))
    worst_scene = deltas[0][0]
    best_scene = deltas[-1][0]
    remaining = [
        item for item in deltas if item[0] not in {worst_scene, best_scene}
    ]
    median_scene = min(
        remaining or deltas,
        key=lambda item: (abs(item[1] - median_delta), item[0]),
    )[0]
    return {
        "worst": worst_scene,
        "median": median_scene,
        "best": best_scene,
    }


def _copy_replay_case(source_output: Path, target: Path) -> None:
    """Materialize a tiny viewer-only replay without mutating source results."""
    if not source_output.is_file():
        raise FileNotFoundError(f"viewer source output missing: {source_output}")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_output, target / "output.json")
    source_diagnostics = source_output.parent / "diagnostics.json"
    if source_diagnostics.is_file():
        shutil.copy2(source_diagnostics, target / "diagnostics.json")
    else:
        output = load_json(source_output)
        write_json(target / "diagnostics.json", {
            "schema": "saga-v8-viewer-b1-diagnostics-v1",
            "instances": output.get("instances", {}),
        })


def _export_missing_prediction_case(
    *, viewer_root: Path, role: str, condition: str, scene_id: str,
    class_id: int, class_name: str, gt_instance: int,
    gt_xyz: np.ndarray, gt_semantic: np.ndarray, gt_instances: np.ndarray,
) -> dict[str, Any]:
    target = viewer_root / "same-object" / role / condition
    empty_xyz = np.empty((0, 3), dtype=np.float32)
    empty_rgb = np.empty((0, 3), dtype=np.uint8)
    gt_mask = (gt_semantic == int(class_id)) & (gt_instances == int(gt_instance))
    gt_points = np.asarray(gt_xyz[gt_mask], dtype=np.float32)
    gt_colors = np.tile(GT_COLOR, (len(gt_points), 1))
    _write_colored_ply(target / "predicted_gaussians.ply", empty_xyz, empty_rgb)
    _write_colored_ply(target / "matched_gt_points.ply", gt_points, gt_colors)
    _write_colored_ply(target / "overlay.ply", gt_points, gt_colors)
    metrics = {
        "role": role,
        "condition": condition,
        "scene_id": scene_id,
        "class_id": int(class_id),
        "class_name": class_name,
        "dominant_gt_instance": int(gt_instance),
        "missing_prediction": True,
        "point_precision": 0.0,
        "gt_to_gaussian_recall": 0.0,
        "official_iou": 0.0,
    }
    write_json(target / "metrics.json", metrics)
    return {**metrics, "directory": str(target)}


def _export_same_object_comparisons(
    *, runtime_manifest: Path, gt_dir: Path, comparison_root: Path,
    scene_roles: Mapping[str, str], conditions: Sequence[str],
    taxonomy: Taxonomy, viewer_root: Path,
) -> list[dict[str, Any]]:
    """Compare every method against the same GT object in each scene role."""
    scenes = load_runtime_scenes(runtime_manifest)
    runtime_payload = load_json(runtime_manifest)
    runtime_rows = runtime_payload.get("scenes", runtime_payload)
    if isinstance(runtime_rows, Mapping):
        rows_by_scene = {
            str(scene_id): dict(row) for scene_id, row in runtime_rows.items()
        }
    else:
        rows_by_scene = {str(row["scene_id"]): dict(row) for row in runtime_rows}
    output: list[dict[str, Any]] = []
    for role, scene_id in scene_roles.items():
        scene = scenes[scene_id]
        gaussian_xyz = apply_transform(
            load_ply_xyz(_gaussian_ply(scene.base_path)),
            rows_by_scene[scene_id].get(
                "gaussian_to_gt_transform", np.eye(4).tolist()
            ),
        )
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        audits: dict[str, dict[str, Any]] = {}
        labels: dict[str, np.ndarray] = {}
        for condition in conditions:
            prediction = load_json(
                comparison_root / condition / scene_id / "output.json"
            )
            labels[condition] = np.asarray(prediction["point_labels"], dtype=np.int64)
            audits[condition] = evaluate_gaussian_object_precision(
                gaussian_xyz,
                labels[condition],
                prediction.get("instances", {}),
                gt_xyz,
                gt.semantic,
                gt.instance,
                0.05,
                canonical_classes=taxonomy.canonical_classes,
            )

        anchor_rows = [
            row for row in audits["V8-D"]["instances"]
            if row["dominant_gt_instance"] is not None
        ]
        if anchor_rows:
            if role == "best":
                anchor = max(anchor_rows, key=lambda row: (row["point_precision"], row["official_iou"]))
            elif role == "worst":
                anchor = min(anchor_rows, key=lambda row: (row["point_precision"], row["official_iou"]))
            else:
                median = float(np.median([row["point_precision"] for row in anchor_rows]))
                anchor = min(
                    anchor_rows,
                    key=lambda row: (abs(row["point_precision"] - median), -row["official_iou"]),
                )
            class_id = int(anchor["class_id"])
            gt_instance = int(anchor["dominant_gt_instance"])
        else:
            valid = (gt.semantic >= 0) & (gt.instance >= 0)
            pairs = sorted(set(zip(gt.semantic[valid].tolist(), gt.instance[valid].tolist())))
            if not pairs:
                continue
            class_id, gt_instance = map(int, pairs[0])
        class_name = str(taxonomy.canonical_classes[class_id])
        for condition in conditions:
            matches = [
                row for row in audits[condition]["instances"]
                if int(row["class_id"]) == class_id
                and row["dominant_gt_instance"] == gt_instance
            ]
            if not matches:
                output.append(_export_missing_prediction_case(
                    viewer_root=viewer_root,
                    role=role,
                    condition=condition,
                    scene_id=scene_id,
                    class_id=class_id,
                    class_name=class_name,
                    gt_instance=gt_instance,
                    gt_xyz=gt_xyz,
                    gt_semantic=gt.semantic,
                    gt_instances=gt.instance,
                ))
                continue
            current = max(
                matches,
                key=lambda row: (row["official_iou"], row["point_precision"], -row["instance_id"]),
            )
            case = {
                "role": f"same_object_{role}",
                "scene_id": scene_id,
                "condition": condition,
                **current,
            }
            exported = _export_viewer_case(
                case,
                audits[condition],
                gt_xyz,
                gt.semantic,
                gt.instance,
                gaussian_xyz,
                labels[condition],
                viewer_root / "same-object",
            )
            output.append({**exported, "missing_prediction": False})
    return output


def _build_final_method_viewer(
    *, analysis: Mapping[str, Any], treatment: str,
    runtime_manifest: Path, gt_dir: Path, historical_b1_root: Path,
    pipeline: Path, final_replay_root: Path, runs_root: Path,
    artifacts_root: Path, taxonomy: Taxonomy, size_bins: Path,
) -> dict[str, Any]:
    """Export matched B1-historical/B1-fixed/V8-U/V8-D 3D views."""
    roles = _select_final_viewer_scenes(analysis, treatment)
    scene_ids = tuple(dict.fromkeys(roles.values()))
    fixed_root = runs_root / "b1-fixed-final-viewer"
    _run_b1_fixed(
        runtime_manifest=runtime_manifest,
        pipeline=pipeline,
        output_root=fixed_root, scene_ids=scene_ids, level="L0",
    )
    comparison_root = runs_root / "viewer-final-method-comparison"
    sources = {
        "B1-historical": lambda scene_id: _resolve_b1_output(
            historical_b1_root, scene_id
        ),
        "B1-fixed": lambda scene_id: fixed_root / "L0" / scene_id / "output.json",
        "V8-U": lambda scene_id: final_replay_root / "U00" / scene_id / "output.json",
        "V8-D": lambda scene_id: final_replay_root / treatment / scene_id / "output.json",
    }
    for condition, resolver in sources.items():
        for scene_id in scene_ids:
            _copy_replay_case(resolver(scene_id), comparison_root / condition / scene_id)

    viewer_root = artifacts_root / "viewer" / "method-comparison"
    evaluations: dict[str, Any] = {}
    for condition in sources:
        evaluations[condition] = _evaluate_replays(
            runtime_manifest=runtime_manifest,
            gt_dir=gt_dir,
            replay_root=comparison_root,
            scene_ids=scene_ids,
            conditions=(condition,),
            taxonomy=taxonomy,
            metrics_path=viewer_root / condition / "metrics.parquet",
            analysis_path=viewer_root / condition / "analysis.json",
            size_bins=size_bins,
            viewer=viewer_root / condition / "objects",
        )
    same_object = _export_same_object_comparisons(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        comparison_root=comparison_root,
        scene_roles=roles,
        conditions=tuple(sources),
        taxonomy=taxonomy,
        viewer_root=viewer_root,
    )
    index = {
        "schema": "saga-v8-final-viewer-comparison-v1",
        "scene_roles": roles,
        "scene_ids": list(scene_ids),
        "conditions": list(sources),
        "qualitative_only": True,
        "selection_biased": True,
        "not_for_parameter_selection": True,
        "directories": {
            condition: str(viewer_root / condition) for condition in sources
        },
        "evaluations": evaluations,
        "same_object_comparisons": same_object,
    }
    write_json(viewer_root / "index.json", index)
    return index


def _contributor_closeout(
    *, factorial_root: Path, scene_ids: Sequence[str], v7_status: Path | None,
    historical_b1_root: Path | None, fixed_b1_root: Path | None,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        bank = load_json(factorial_root / "G-M1" / scene_id / "lifting_bank.json")
        comparisons.extend(bank.get("contributor_comparisons", ()))
    compared = [row for row in comparisons if row.get("status") == "compared"]
    pixel_count = sum(int(row.get("pixel_count", 0)) for row in compared)
    changed = sum(int(row.get("changed_pixel_count", 0)) for row in compared)
    if v7_status is None or not v7_status.is_file():
        raise FileNotFoundError("Stage 0 requires the actual V7 status artifact")
    previous = load_json(v7_status)
    b1_comparison = _compare_b1_outputs(
        historical_b1_root, fixed_b1_root, scene_ids
    )
    if not compared:
        raise RuntimeError("Stage 0 contributor audit produced no comparable frames")
    if b1_comparison["status"] != "compared":
        raise RuntimeError("Stage 0 requires configured historical and fixed B1 roots")
    return {
        "schema": "saga-v8-provenance-and-v7-closeout-v1",
        "v7_status": previous,
        "v7_actual_stop": "candidate/oracle gate failed before category-prior replay",
        "contributor_frame_count": len(compared),
        "contributor_pixel_count": pixel_count,
        "contributor_changed_pixel_count": changed,
        "contributor_changed_pixel_fraction": changed / pixel_count if pixel_count else 0.0,
        "b1_comparison": b1_comparison,
        "b1_historical_equals_b1_fixed": b1_comparison["all_equal"],
    }


def _gaussian_ply(base_path: Path) -> Path:
    root = base_path / "output_models/point_cloud/iteration_30000"
    scene_path = root / "scene_point_cloud.ply"
    return scene_path if scene_path.is_file() else root / "point_cloud.ply"


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from scipy.stats import rankdata

    y = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(np.count_nonzero(y))
    negatives = int(len(y) - positives)
    if not positives or not negatives:
        return 0.5
    ranks = rankdata(values, method="average")
    return float(
        (ranks[y].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _selected_mask_affinity_aucs(
    *, runtime_manifest: Path, gt_dir: Path, lifting_root: Path,
    scene_ids: Sequence[str], radius_m: float = 0.05,
) -> dict[str, float]:
    from scipy.spatial import cKDTree
    from .v8_analysis import load_lifting_bank, unpack_ragged

    scenes = load_runtime_scenes(runtime_manifest)
    output: dict[str, float] = {}
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        metadata, arrays = load_lifting_bank(lifting_root / scene_id)
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        gaussian_xyz = load_ply_xyz(_gaussian_ply(scene.base_path))
        # Runtime assets use the same metric scale as the bank.  A nontrivial
        # registered transform is still honored when present.
        runtime_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        runtime_rows = runtime_payload.get("scenes", runtime_payload)
        if isinstance(runtime_rows, Mapping):
            runtime_row = dict(runtime_rows[scene_id])
        else:
            runtime_row = next(row for row in runtime_rows if row["scene_id"] == scene_id)
        transform = runtime_row.get(
            "gaussian_to_gt_transform", np.eye(4).tolist()
        )
        gaussian_xyz = apply_transform(gaussian_xyz, transform)
        distance, nearest = cKDTree(gt_xyz).query(
            gaussian_xyz, k=1, distance_upper_bound=radius_m
        )
        valid = np.isfinite(distance) & (nearest < len(gt_xyz))
        instance_key = np.full(len(gaussian_xyz), -1, dtype=np.int64)
        gt_valid = (gt.semantic >= 0) & (gt.instance >= 0)
        mapped = valid & gt_valid[np.minimum(nearest, len(gt_xyz) - 1)]
        instance_key[mapped] = (
            gt.semantic[nearest[mapped]] * 1_000_000 + gt.instance[nearest[mapped]]
        )
        fragments = unpack_ragged(
            arrays["fragment_full_indptr"], arrays["fragment_full_ids"]
        )
        support = np.unique(np.concatenate(fragments)) if fragments else np.empty(0, dtype=np.int64)
        support = support[instance_key[support] >= 0]
        if len(support) < 3:
            output[scene_id] = 0.5
            continue
        k = min(24, len(support) - 1)
        _, neighbor_position = cKDTree(gaussian_xyz[support]).query(
            gaussian_xyz[support], k=k + 1
        )
        source_ids = np.repeat(support, k)
        target_ids = support[np.asarray(neighbor_position)[:, 1:].reshape(-1)]
        affinity = np.asarray(arrays["affinity"], dtype=np.float64)
        edge_scores = np.sum(affinity[source_ids] * affinity[target_ids], axis=1)
        edge_labels = instance_key[source_ids] == instance_key[target_ids]
        output[scene_id] = _binary_auc(edge_labels, edge_scores)
    return output


def _final_prediction_fp_tp(
    *, runtime_manifest: Path, gt_dir: Path, replay_root: Path,
    scene_ids: Sequence[str], condition: str, taxonomy: Taxonomy,
    radius_m: float = 0.05, min_region_size: int = 100,
) -> dict[str, Any]:
    scenes = load_runtime_scenes(runtime_manifest)
    runtime_payload = load_json(runtime_manifest)
    runtime_rows = runtime_payload.get("scenes", runtime_payload)
    if isinstance(runtime_rows, Mapping):
        runtime_rows_by_id = {
            str(scene_id): dict(row) for scene_id, row in runtime_rows.items()
        }
    else:
        runtime_rows_by_id = {
            str(row["scene_id"]): dict(row) for row in runtime_rows
        }
    class_to_id = {
        name: index for index, name in enumerate(taxonomy.canonical_classes)
    }
    tp = 0
    fp = 0
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        output = load_json(replay_root / condition / scene_id / "output.json")
        diagnostics = load_json(replay_root / condition / scene_id / "diagnostics.json")
        gaussian_labels = np.asarray(output["point_labels"], dtype=np.int64)
        gaussian_xyz = apply_transform(
            load_ply_xyz(_gaussian_ply(scene.base_path)),
            runtime_rows_by_id[scene_id].get(
                "gaussian_to_gt_transform", np.eye(4).tolist()
            ),
        )
        gt_xyz, gt = load_ground_truth_npz(gt_dir / f"{scene_id}.npz", scene_id)
        mapped_labels, _ = map_gaussians_to_gt(
            gt_xyz, gaussian_xyz, gaussian_labels, radius_m
        )
        gt_rows: dict[int, list[np.ndarray]] = {}
        valid = (gt.semantic >= 0) & (gt.instance >= 0)
        for class_id, instance_id in sorted(set(zip(
            gt.semantic[valid].tolist(), gt.instance[valid].tolist()
        ))):
            mask = valid & (gt.semantic == class_id) & (gt.instance == instance_id)
            if int(np.count_nonzero(mask)) >= int(min_region_size):
                gt_rows.setdefault(int(class_id), []).append(mask)
        instance_meta = diagnostics.get("instances", {})
        predictions: list[tuple[float, int, int, np.ndarray]] = []
        for raw_id, values in output.get("instances", {}).items():
            class_id = class_to_id.get(str(values.get("class", "")))
            if class_id is None:
                continue
            instance_id = int(raw_id)
            prediction_mask = mapped_labels == instance_id
            if int(np.count_nonzero(prediction_mask)) < int(min_region_size):
                continue
            score = float(instance_meta.get(str(instance_id), {}).get("score", 1.0))
            predictions.append((score, instance_id, class_id, prediction_mask))
        matched: dict[int, set[int]] = {}
        for _score, instance_id, class_id, prediction in sorted(
            predictions, key=lambda row: (-row[0], row[1])
        ):
            best_iou = 0.0
            best_index: int | None = None
            for index, target in enumerate(gt_rows.get(class_id, ())):
                if index in matched.setdefault(class_id, set()):
                    continue
                intersection = int(np.count_nonzero(prediction & target))
                union = int(np.count_nonzero(prediction | target))
                iou = intersection / union if union else 0.0
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index is not None and best_iou >= 0.25:
                tp += 1
                matched[class_id].add(best_index)
            else:
                fp += 1
    ratio = float(fp / max(tp, 1))
    return {"tp": tp, "fp": fp, "fp_tp_ratio": ratio}


def _mechanical_prior_effect(
    replay_root: Path, scene_ids: Sequence[str], condition: str
) -> dict[str, Any]:
    changed_scores = 0
    candidate_count = 0
    accepted_or_owner_changed = False
    for scene_id in scene_ids:
        uniform = load_json(replay_root / "U00" / scene_id / "diagnostics.json")
        data = load_json(replay_root / condition / scene_id / "diagnostics.json")
        u_scores = {
            int(row["candidate_id"]): float(row["score"])
            for row in uniform["candidate_scores"]
        }
        d_scores = {
            int(row["candidate_id"]): float(row["score"])
            for row in data["candidate_scores"]
        }
        candidate_count += len(u_scores)
        changed_scores += sum(
            abs(u_scores[candidate_id] - d_scores[candidate_id]) >= 0.01
            for candidate_id in u_scores
        )
        accepted_or_owner_changed |= (
            uniform["accepted_candidate_ids"] != data["accepted_candidate_ids"]
            or load_json(replay_root / "U00" / scene_id / "output.json")["point_labels"]
            != load_json(replay_root / condition / scene_id / "output.json")["point_labels"]
        )
    fraction = changed_scores / candidate_count if candidate_count else 0.0
    return {
        "candidate_count": candidate_count,
        "score_difference_at_least_001_count": changed_scores,
        "score_difference_at_least_001_fraction": fraction,
        "accepted_or_owner_changed": accepted_or_owner_changed,
        "passed": fraction >= 0.10 or accepted_or_owner_changed,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.runs.mkdir(parents=True, exist_ok=True)
    _disk_guard(args.runs)
    status_path = args.artifacts / "v8_status.json"
    status = load_json(status_path) if status_path.is_file() else {
        "schema": "saga-v8-stage-status-v1", "state": "running"
    }
    taxonomy = load_taxonomy(args.taxonomy)
    size_spec = load_json(args.size_bins)

    factorial_root = args.runs / "lifting-factorial2"
    run_v8_lifting_factorial(
        args.runtime_manifest,
        CAUSAL2,
        factorial_root,
        args.repo_root,
        sam_masks_root=args.sam_masks_root,
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
        contributor_audit=True,
    )
    lifting = evaluate_v8_lifting_factorial(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        arm_roots={arm: factorial_root / arm for arm in ("G-M1", "G-AM", "S-M1", "S-AM")},
        scene_ids=CAUSAL2,
        canonical_classes=taxonomy.canonical_classes,
        size_spec=size_spec,
    )
    write_rows(args.artifacts / "v8_lifting_factorial2.parquet", lifting["arm_rows"])
    write_json(args.artifacts / "v8_lifting_analysis2.json", lifting)
    closeout = _contributor_closeout(
        factorial_root=factorial_root,
        scene_ids=CAUSAL2,
        v7_status=args.v7_status,
        historical_b1_root=args.historical_b1_root,
        fixed_b1_root=args.fixed_b1_root,
    )
    write_json(args.artifacts / "v8_provenance_and_v7_closeout.json", closeout)
    _record(status_path, status, "stage0_closeout", closeout)
    _record(status_path, status, "stage1_lifting", lifting)
    if not lifting["selection"]["passed"]:
        status["state"] = "stopped"
        status["stop_reason"] = "all four mask/lifting combinations failed the geometric support gate"
        write_json(status_path, status)
        return status

    selected = str(lifting["selection"]["selected_combination"])
    mask_source, lifting_source = selected.split("-")
    selected_lifting_root = factorial_root / selected
    bank_root = args.runs / "bank-tune"
    _build_banks(
        runtime_manifest=args.runtime_manifest,
        scene_ids=DEV8,
        lifting_root=selected_lifting_root,
        bank_root=bank_root,
        repo_root=args.repo_root,
        mask_source=mask_source,
        lifting_source=lifting_source,
        sam_masks_root=args.sam_masks_root,
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
    )
    bank_analysis = evaluate_v8_object_banks(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        bank_root=bank_root,
        scene_ids=DEV8,
        canonical_classes=taxonomy.canonical_classes,
        size_spec=size_spec,
    )
    classifier = str(bank_analysis["selected_classifier"])
    replay_root = args.runs / "replay-tune"
    replay_v8_priors(
        bank_root=bank_root,
        output_root=replay_root,
        scene_ids=DEV8,
        classifier=classifier,
        conditions=("U00",),
        category_priors=args.category_priors,
    )
    u_analysis_path = args.artifacts / "v8_u00_8_analysis.json"
    u_metrics_path = args.artifacts / "v8_u00_8_metrics.parquet"
    _evaluate_replays(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=replay_root,
        scene_ids=DEV8,
        conditions=("U00",),
        taxonomy=taxonomy,
        metrics_path=u_metrics_path,
        analysis_path=u_analysis_path,
        size_bins=args.size_bins,
    )
    b1_root = args.runs / "b1-fixed8"
    _run_b1_fixed(
        runtime_manifest=args.runtime_manifest,
        pipeline=args.pipeline,
        output_root=b1_root,
        scene_ids=DEV8,
        level="L0",
    )
    b1_metrics_path = args.artifacts / "v8_b1_fixed8_metrics.parquet"
    _evaluate_replays(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=b1_root,
        scene_ids=DEV8,
        conditions=("L0",),
        taxonomy=taxonomy,
        metrics_path=b1_metrics_path,
        analysis_path=args.artifacts / "v8_b1_fixed8_analysis.json",
        size_bins=args.size_bins,
    )
    uniform = _metrics(u_metrics_path)["U00"]
    b1 = _metrics(b1_metrics_path)["L0"]
    health_row = {
        **bank_analysis,
        "gaussian_micro_precision": uniform["gaussian_micro_precision"],
        "unsupported_instance_fraction": uniform["unsupported_instance_fraction"],
        "gt_recall": uniform["official_gt_recall_025"],
        "map_50_95": uniform["map_50_95"],
        "ap50": uniform["map_0.50"],
        "predicted_instance_count": uniform["predicted_instance_count"],
    }
    b1_row = {
        "gaussian_micro_precision": b1["gaussian_micro_precision"],
        "unsupported_instance_fraction": b1["unsupported_instance_fraction"],
        "gt_recall": b1["official_gt_recall_025"],
        "map_50_95": b1["map_50_95"],
        "ap50": b1["map_0.50"],
        "predicted_instance_count": b1["predicted_instance_count"],
    }
    bank_gate = stage2_bank_health_gate(health_row, b1_row)
    bank_analysis["uniform_metrics"] = uniform
    bank_analysis["b1_fixed_metrics"] = b1
    bank_analysis["health_gate"] = bank_gate
    write_rows(args.artifacts / "v8_bank8.parquet", bank_analysis["per_scene"])
    write_json(args.artifacts / "v8_bank8_analysis.json", bank_analysis)
    _record(status_path, status, "stage2_bank", bank_analysis)
    if not bank_gate["passed"]:
        affinity_by_scene = _selected_mask_affinity_aucs(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            lifting_root=selected_lifting_root,
            scene_ids=DEV8,
        )
        affinity_auc = float(np.mean(list(affinity_by_scene.values())))
        _record(status_path, status, "stage2_followup", {
            "selected_mask_affinity_edge_auroc": affinity_auc,
            "per_scene": affinity_by_scene,
            "decision": "stop_bank_unhealthy",
            "reason": (
                "V8 tracking is mask-overlap-only and does not consume affinity; "
                "a 10k affinity retrain cannot causally change this bank"
            ),
        })
        status["state"] = "stopped"
        status["stop_reason"] = (
            "Stage-1 oracle passed but the mask-overlap V8 automatic bank was unhealthy"
        )
        write_json(status_path, status)
        return status

    replay_v8_priors(
        bank_root=bank_root,
        output_root=replay_root,
        scene_ids=DEV8,
        classifier=classifier,
        conditions=CONDITIONS,
        category_priors=args.category_priors,
    )
    prior_metrics_path = args.artifacts / "v8_prior_replay8.parquet"
    prior_analysis = _evaluate_replays(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=replay_root,
        scene_ids=DEV8,
        conditions=CONDITIONS,
        taxonomy=taxonomy,
        metrics_path=prior_metrics_path,
        analysis_path=args.artifacts / "v8_prior_replay8_analysis.json",
        size_bins=args.size_bins,
    )
    prior_metrics = _metrics(prior_metrics_path)
    uniform = prior_metrics["U00"]
    uniform_fp_tp = _final_prediction_fp_tp(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=replay_root,
        scene_ids=DEV8,
        condition="U00",
        taxonomy=taxonomy,
    )
    candidates: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:]:
        mechanical = _mechanical_prior_effect(replay_root, DEV8, condition)
        _scene_mean_delta, positive, negative = _mean_scene_delta(
            prior_analysis, condition, "U00"
        )
        delta = float(
            prior_metrics[condition]["map_50_95"] - uniform["map_50_95"]
        )
        tiny_delta = (
            prior_metrics[condition]["tiny_small_recall_050"]
            - uniform["tiny_small_recall_050"]
        )
        fp_tp = _final_prediction_fp_tp(
            runtime_manifest=args.runtime_manifest,
            gt_dir=args.gt_dir,
            replay_root=replay_root,
            scene_ids=DEV8,
            condition=condition,
            taxonomy=taxonomy,
        )
        candidates.append(
            {
                "condition": condition,
                "delta_map": delta,
                "mean_scene_delta_map": _scene_mean_delta,
                "tiny_small_recall_050_delta": tiny_delta,
                "ap50": prior_metrics[condition]["map_0.50"],
                "positive_scenes": positive,
                "negative_scenes": negative,
                "fp_tp": fp_tp,
                "mechanical": mechanical,
                "passed": (
                    mechanical["passed"]
                    and (delta >= 0.002 or (tiny_delta >= 0.01 and delta >= -0.0005))
                    and positive > negative
                    and fp_tp["fp_tp_ratio"]
                    <= 1.20 * max(uniform_fp_tp["fp_tp_ratio"], 1e-12)
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["delta_map"],
            -row["tiny_small_recall_050_delta"],
            -row["ap50"],
            0 if row["condition"] in {"D10", "D01"} else 1,
            row["condition"],
        )
    )
    best_row = next((row for row in candidates if row["passed"]), None)
    stage3 = {
        "conditions": candidates,
        "uniform_fp_tp": uniform_fp_tp,
        "best_condition": best_row["condition"] if best_row else None,
        "passed": best_row is not None,
    }
    _record(status_path, status, "stage3_prior", stage3)
    if best_row is None:
        status["state"] = "stopped"
        status["stop_reason"] = (
            "prior mapping did not materially intervene"
            if not any(row["mechanical"]["passed"] for row in candidates)
            else "V8-U was healthy but train-derived size/support priors added no stable value"
        )
        write_json(status_path, status)
        return status
    best = str(best_row["condition"])

    _build_banks(
        runtime_manifest=args.runtime_manifest,
        scene_ids=HOLDOUT5,
        lifting_root=selected_lifting_root,
        bank_root=bank_root,
        repo_root=args.repo_root,
        mask_source=mask_source,
        lifting_source=lifting_source,
        sam_masks_root=args.sam_masks_root,
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
    )
    replay_v8_priors(
        bank_root=bank_root, output_root=replay_root,
        scene_ids=HOLDOUT5, classifier=classifier,
        conditions=("U00", best), category_priors=args.category_priors,
    )
    holdout_analysis = _evaluate_replays(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=replay_root,
        scene_ids=HOLDOUT5,
        conditions=("U00", best),
        taxonomy=taxonomy,
        metrics_path=args.artifacts / "v8_holdout5_metrics.parquet",
        analysis_path=args.artifacts / "v8_holdout5_analysis.json",
        size_bins=args.size_bins,
    )
    holdout_metrics = _metrics(args.artifacts / "v8_holdout5_metrics.parquet")
    delta, positive, negative = _mean_scene_delta(holdout_analysis, best, "U00")
    tiny_delta = (
        holdout_metrics[best]["tiny_small_recall_050"]
        - holdout_metrics["U00"]["tiny_small_recall_050"]
    )
    holdout_pass = delta > 0 and positive >= 3 and tiny_delta > 0
    _record(status_path, status, "stage4_holdout5", {
        "delta_map": delta, "positive_scenes": positive,
        "negative_scenes": negative, "tiny_small_recall_050_delta": tiny_delta,
        "passed": holdout_pass,
    })
    if not holdout_pass:
        status["state"] = "stopped"
        status["stop_reason"] = "data-driven prior failed the independent five-scene tune holdout"
        write_json(status_path, status)
        return status

    tune_scenes = tuple(load_runtime_scenes(args.runtime_manifest))
    _validate_tune_manifest(tune_scenes)
    remaining = tuple(
        scene for scene in tune_scenes if scene not in set(DEV8) | set(HOLDOUT5)
    )
    _build_banks(
        runtime_manifest=args.runtime_manifest,
        scene_ids=remaining,
        lifting_root=selected_lifting_root,
        bank_root=bank_root,
        repo_root=args.repo_root,
        mask_source=mask_source,
        lifting_source=lifting_source,
        sam_masks_root=args.sam_masks_root,
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
    )
    replay_v8_priors(
        bank_root=bank_root, output_root=replay_root,
        scene_ids=remaining, classifier=classifier,
        conditions=("U00", best), category_priors=args.category_priors,
    )
    tune_analysis = _evaluate_replays(
        runtime_manifest=args.runtime_manifest,
        gt_dir=args.gt_dir,
        replay_root=replay_root,
        scene_ids=tune_scenes,
        conditions=("U00", best),
        taxonomy=taxonomy,
        metrics_path=args.artifacts / "v8_tune24_metrics.parquet",
        analysis_path=args.artifacts / "v8_tune24_analysis.json",
        size_bins=args.size_bins,
    )
    reference = {
        row["scene_id"]: float(row["map_50_95"])
        for row in tune_analysis["conditions"]["U00"]["per_scene"]
    }
    treatment = {
        row["scene_id"]: float(row["map_50_95"])
        for row in tune_analysis["conditions"][best]["per_scene"]
    }
    physical: dict[str, list[float]] = {}
    for scene_id in tune_scenes:
        physical.setdefault(scene_id.rsplit("_", 1)[0], []).append(
            treatment[scene_id] - reference[scene_id]
        )
    physical_delta = {key: float(np.mean(values)) for key, values in physical.items()}
    macro_delta = float(np.mean(list(physical_delta.values())))
    tune_pass = macro_delta >= 0.002
    _record(status_path, status, "stage4_tune24", {
        "physical_scene_count": len(physical_delta),
        "physical_scene_deltas": physical_delta,
        "macro_delta_map": macro_delta,
        "passed": tune_pass,
    })
    if not tune_pass:
        status["state"] = "stopped"
        status["stop_reason"] = "data-driven prior failed physical-scene-weighted tune24"
        write_json(status_path, status)
        return status

    if args.locked_runtime_manifest is None or args.locked_gt_dir is None:
        raise ValueError("final48 paths are required after tune24 passes")
    locked_scenes = tuple(load_runtime_scenes(args.locked_runtime_manifest))
    _validate_locked_manifest(
        locked_scenes, _scene_ids_from_spec(args.locked_evaluation_scenes)
    )
    final_lifting = args.runs / "lifting-final48"
    final_bank = args.runs / "bank-final48"
    _build_banks(
        runtime_manifest=args.locked_runtime_manifest,
        scene_ids=locked_scenes,
        lifting_root=final_lifting,
        bank_root=final_bank,
        repo_root=args.repo_root,
        mask_source=mask_source,
        lifting_source=lifting_source,
        sam_masks_root=args.sam_masks_root / "final48",
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
    )
    final_replay = args.runs / "replay-final48"
    replay_v8_priors(
        bank_root=final_bank, output_root=final_replay,
        scene_ids=locked_scenes, classifier=classifier,
        conditions=("U00", best), category_priors=args.category_priors,
    )
    final_detail = _evaluate_replays(
        runtime_manifest=args.locked_runtime_manifest,
        gt_dir=args.locked_gt_dir,
        replay_root=final_replay,
        scene_ids=locked_scenes,
        conditions=("U00", best),
        taxonomy=taxonomy,
        metrics_path=args.artifacts / "v8_final_metrics.parquet",
        analysis_path=args.artifacts / "v8_final_detail.json",
        size_bins=args.size_bins,
        viewer=args.artifacts / "viewer",
    )
    final_metrics = _metrics(args.artifacts / "v8_final_metrics.parquet")
    bootstrap = paired_scannet_scene_bootstrap_from_replays(
        runtime_manifest=args.locked_runtime_manifest,
        gt_dir=args.locked_gt_dir,
        replay_root=final_replay,
        scene_ids=locked_scenes,
        reference_condition="U00",
        treatment_condition=best,
        class_names=taxonomy.canonical_classes,
        samples=10_000,
        seed=20260804,
    )
    final_delta = float(bootstrap["delta_map_50_95"])
    ci = list(map(float, bootstrap["paired_bootstrap_ci95"]))
    supports = final_delta >= 0.002 and ci[0] > 0
    viewer_comparison = _build_final_method_viewer(
        analysis=final_detail,
        treatment=best,
        runtime_manifest=args.locked_runtime_manifest,
        gt_dir=args.locked_gt_dir,
        historical_b1_root=args.historical_b1_final_root,
        pipeline=args.pipeline,
        final_replay_root=final_replay,
        runs_root=args.runs,
        artifacts_root=args.artifacts,
        taxonomy=taxonomy,
        size_bins=args.size_bins,
    )
    summary = {
        "schema": "saga-v8-final-analysis-v1",
        "selected_lifting": selected,
        "selected_classifier": classifier,
        "best_condition": best,
        "delta_map_50_95": final_delta,
        "paired_bootstrap_samples": 10000,
        "paired_bootstrap_ci95": ci,
        "paired_bootstrap": bootstrap,
        "supports_stable_category_prior": supports,
        "uniform": final_metrics["U00"],
        "data": final_metrics[best],
        "viewer_comparison": {
            "directory": str(args.artifacts / "viewer" / "method-comparison"),
            "scene_roles": viewer_comparison["scene_roles"],
            "conditions": viewer_comparison["conditions"],
        },
    }
    write_json(args.artifacts / "v8_analysis.json", summary)
    _record(status_path, status, "stage5_final48", summary)
    status["state"] = "complete"
    status["best_condition"] = best
    status["stop_reason"] = (
        None if supports else "V8 proposal-level category prior showed no stable improvement"
    )
    write_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path)
    parser.add_argument("--locked-gt-dir", type=Path)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--label-features", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-masks-root", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--v7-status", type=Path, required=True)
    parser.add_argument("--historical-b1-root", type=Path, required=True)
    parser.add_argument("--historical-b1-final-root", type=Path, required=True)
    parser.add_argument("--fixed-b1-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
