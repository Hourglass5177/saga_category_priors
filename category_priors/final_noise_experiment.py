"""Final fixed-candidate audit for category-aware small-cluster retention.

This module intentionally runs after every candidate-forming operation.  It
reads the frozen partition immediately before the historical ``filter_num(10)``
call and replays only the final keep/delete decision.  It never clusters,
trains, changes KNN labels, or uses ground truth to construct predictions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluator import load_ground_truth_npz, load_ply_xyz, map_gaussians_to_gt
from .final_noise_prior import (
    class_threshold,
    load_category_priors,
    replay_filter,
    replay_u10,
)
from .taxonomy import load_taxonomy


DEFAULT_SCENES = (
    "scene0025_01",
    "scene0046_00",
    "scene0064_01",
    "scene0164_03",
    "scene0329_02",
    "scene0474_01",
    "scene0591_02",
    "scene0645_00",
)
DEFAULT_DEV_SCENES = ("scene0591_02", "scene0645_00")


@dataclass(frozen=True)
class SceneTrace:
    scene_id: str
    post_global_knn: np.ndarray
    post_filter: np.ndarray
    branch_before_merge: np.ndarray
    branch_classes: dict[int, str]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_scene_trace(trace_root: str | Path, scene_id: str) -> SceneTrace:
    root = Path(trace_root) / scene_id / "seed-42"
    metadata = _read_json(root / "stage_trace.json")
    with np.load(root / "stage_trace.npz", allow_pickle=False) as arrays:
        required = {"post_global_knn", "post_filter", "branch_class_before_merge"}
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"{root}: missing trace arrays {sorted(missing)}")
        post_knn = np.asarray(arrays["post_global_knn"], dtype=np.int64).copy()
        post_filter = np.asarray(arrays["post_filter"], dtype=np.int64).copy()
        branch_before = np.asarray(
            arrays["branch_class_before_merge"], dtype=np.int64
        ).copy()
    if not (post_knn.shape == post_filter.shape == branch_before.shape):
        raise ValueError(f"{root}: inconsistent trace shapes")
    classes = {
        int(instance_id): str(class_name).strip().lower()
        for instance_id, class_name in metadata["branch_instance_classes"].items()
    }
    return SceneTrace(
        scene_id=scene_id,
        post_global_knn=post_knn,
        post_filter=post_filter,
        branch_before_merge=branch_before,
        branch_classes=classes,
    )


def _count_by_id(labels: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {
        int(instance_id): int(count)
        for instance_id, count in zip(ids.tolist(), counts.tolist())
    }


def candidate_rows(trace: SceneTrace, priors: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe every frozen teacher-branch instance without using GT."""

    before = _count_by_id(trace.branch_before_merge)
    post_knn = _count_by_id(trace.post_global_knn)
    rows: list[dict[str, Any]] = []
    for instance_id, class_name in sorted(trace.branch_classes.items()):
        count = post_knn.get(instance_id, 0)
        threshold = class_threshold(priors, class_name)
        rows.append(
            {
                "scene_id": trace.scene_id,
                "instance_id": instance_id,
                "class_name": class_name,
                "branch_points_before_merge": before.get(instance_id, 0),
                "post_knn_points": count,
                "u10_retained": count >= 10,
                "s5_retained": count >= 5,
                "s3_retained": count >= 3,
                "class_threshold": threshold,
                "d_class_retained": count >= threshold,
                "s3_changes_u10": 3 <= count < 10,
                "s5_changes_u10": 5 <= count < 10,
                "d_changes_u10": threshold <= count < 10,
                "same_class_best_iou": None,
                "mapped_prediction_points": None,
                "official_valid_same_class_gt": None,
            }
        )
    return rows


def add_candidate_oracle(
    rows: list[dict[str, Any]],
    trace: SceneTrace,
    *,
    gt_root: str | Path,
    data_root: str | Path,
    radius_m: float = 0.05,
) -> None:
    """Add score-free GT diagnostics for only the potentially recoverable rows.

    Ground truth is used strictly after candidate construction.  Rows that no
    threshold in {3, 5, 10} can change are intentionally left unevaluated.
    """

    eligible = [row for row in rows if bool(row["s3_changes_u10"])]
    if not eligible:
        return
    taxonomy = load_taxonomy()
    class_to_id = {
        class_name: class_id
        for class_id, class_name in enumerate(taxonomy.canonical_classes)
    }
    gt_coords, gt_scene = load_ground_truth_npz(
        Path(gt_root) / f"{trace.scene_id}.npz", trace.scene_id
    )
    gaussian_path = (
        Path(data_root)
        / trace.scene_id
        / "output_models"
        / "point_cloud"
        / "iteration_30000"
        / "scene_point_cloud.ply"
    )
    gaussian_coords = load_ply_xyz(gaussian_path)
    if len(gaussian_coords) != len(trace.post_global_knn):
        raise ValueError(f"{trace.scene_id}: Gaussian/trace point count mismatch")

    for row in eligible:
        instance_id = int(row["instance_id"])
        candidate_labels = np.where(
            trace.post_global_knn == instance_id, instance_id, -1
        )
        mapped, _ = map_gaussians_to_gt(
            gt_coords, gaussian_coords, candidate_labels, radius_m=radius_m
        )
        predicted = mapped == instance_id
        row["mapped_prediction_points"] = int(np.count_nonzero(predicted))
        class_id = class_to_id.get(str(row["class_name"]))
        if class_id is None:
            row["same_class_best_iou"] = 0.0
            row["official_valid_same_class_gt"] = 0
            continue
        class_mask = gt_scene.semantic == class_id
        gt_ids = np.unique(gt_scene.instance[class_mask & (gt_scene.instance >= 0)])
        best_iou = 0.0
        valid_count = 0
        for gt_id in gt_ids:
            truth = class_mask & (gt_scene.instance == gt_id)
            if int(np.count_nonzero(truth)) >= 100:
                valid_count += 1
            intersection = int(np.count_nonzero(predicted & truth))
            union = int(np.count_nonzero(predicted | truth))
            if union:
                best_iou = max(best_iou, intersection / union)
        row["same_class_best_iou"] = float(best_iou)
        row["official_valid_same_class_gt"] = valid_count


def _condition_changed_ids(
    trace: SceneTrace, *, threshold: int, branch_only: bool = True
) -> list[int]:
    thresholds = (
        {instance_id: threshold for instance_id in trace.branch_classes}
        if branch_only
        else None
    )
    replayed = replay_filter(
        trace.post_global_knn,
        thresholds,
        default=10 if branch_only else threshold,
    )
    u10 = replay_u10(trace.post_global_knn)
    changed = trace.post_global_knn[(replayed != u10) & (trace.post_global_knn >= 0)]
    return sorted(int(value) for value in np.unique(changed))


def run_final_noise_audit(
    *,
    trace_root: str | Path,
    prior_json: str | Path,
    output_root: str | Path,
    scenes: Sequence[str] = DEFAULT_SCENES,
    dev_scenes: Sequence[str] = DEFAULT_DEV_SCENES,
    gt_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the pre-registered mechanical gate and stop when it fails."""

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    priors = load_category_priors(prior_json)
    traces = {scene: load_scene_trace(trace_root, scene) for scene in scenes}
    rows: list[dict[str, Any]] = []
    parity: dict[str, bool] = {}
    changes: dict[str, dict[str, list[int]]] = {}
    for scene, trace in traces.items():
        replayed = replay_u10(trace.post_global_knn)
        parity[scene] = bool(np.array_equal(replayed, trace.post_filter))
        scene_rows = candidate_rows(trace, priors)
        if gt_root is not None and data_root is not None:
            add_candidate_oracle(
                scene_rows,
                trace,
                gt_root=gt_root,
                data_root=data_root,
            )
        rows.extend(scene_rows)
        d_thresholds = {
            instance_id: class_threshold(priors, class_name)
            for instance_id, class_name in trace.branch_classes.items()
        }
        d_replayed = replay_filter(trace.post_global_knn, d_thresholds, default=10)
        u10 = replay_u10(trace.post_global_knn)
        d_changed = trace.post_global_knn[
            (d_replayed != u10) & (trace.post_global_knn >= 0)
        ]
        changes[scene] = {
            "S3": _condition_changed_ids(trace, threshold=3),
            "S5": _condition_changed_ids(trace, threshold=5),
            "D-class": sorted(int(value) for value in np.unique(d_changed)),
        }

    if not all(parity.values()):
        failed = [scene for scene, exact in parity.items() if not exact]
        raise RuntimeError(f"U10 replay parity failed for: {failed}")

    dev_s3_counts = {
        scene: len(changes[scene]["S3"])
        for scene in dev_scenes
    }
    gate2 = all(count >= 1 for count in dev_s3_counts.values()) and sum(
        dev_s3_counts.values()
    ) >= 3
    dev_eligible = [
        row
        for row in rows
        if row["scene_id"] in set(dev_scenes) and row["s3_changes_u10"]
    ]
    gate3 = (
        sum(float(row["same_class_best_iou"] or 0.0) >= 0.25 for row in dev_eligible)
        >= 2
        and len(
            {
                str(row["class_name"])
                for row in dev_eligible
                if float(row["same_class_best_iou"] or 0.0) >= 0.25
            }
        )
        >= 2
    )
    passed = gate2 and gate3
    stop_reason = None
    if not gate2:
        stop_reason = (
            "The shared threshold-3 arm changed no branch candidate in at least "
            "one development scene, or fewer than three candidates in total."
        )
    elif not gate3:
        stop_reason = (
            "Fewer than two recoverable candidates reached same-class IoU 0.25 "
            "across two classes."
        )

    plan = {
        "schema_version": "1.0",
        "question": "Does class-aware final small-cluster retention beat one shared threshold?",
        "frozen_stage": "post_global_knn",
        "historical_reference": "U10",
        "uniform_controls": ["S3", "S5", "U10"],
        "data_condition": "D-class",
        "scenes": list(scenes),
        "development_scenes": list(dev_scenes),
        "confirmation_scenes": [scene for scene in scenes if scene not in set(dev_scenes)],
    }
    dev = {
        "schema_version": "1.0",
        "u10_exact_by_scene": parity,
        "s3_changed_branch_candidates_by_dev_scene": dev_s3_counts,
        "s3_changed_branch_candidates_total": sum(dev_s3_counts.values()),
        "gate_1_u10_exact": True,
        "gate_2_minimum_intervention": gate2,
        "gate_3_recoverable_true_positives": gate3 if gate2 else None,
        "gate_4_gt_oracle_capacity": None,
        "passed": passed,
        "stopped_at_gate": 2 if not gate2 else (3 if not gate3 else None),
        "stop_reason": stop_reason,
        "confirmation_run": False,
    }
    eligible_all = [row for row in rows if row["s3_changes_u10"]]
    analysis = {
        "schema_version": "1.0",
        "decision": (
            "stop-no-recoverable-final-filter-intervention"
            if not passed
            else "proceed-to-confirmation"
        ),
        "main_conclusion": (
            "The frozen candidate pool leaves no usable development-set space "
            "for a category-aware final small-cluster threshold."
            if not passed
            else "The development capacity gates passed."
        ),
        "u10_parity_scenes": sum(parity.values()),
        "scene_count": len(scenes),
        "branch_candidate_count": len(rows),
        "s3_eligible_candidate_count_all_scenes": len(eligible_all),
        "s3_eligible_scene_count_all_scenes": len(
            {str(row["scene_id"]) for row in eligible_all}
        ),
        "d_class_changed_candidate_count_all_scenes": sum(
            len(item["D-class"]) for item in changes.values()
        ),
        "d_class_changed_scene_count_all_scenes": sum(
            bool(item["D-class"]) for item in changes.values()
        ),
        "eligible_same_class_iou_at_least_0_25_all_scenes": sum(
            float(row["same_class_best_iou"] or 0.0) >= 0.25
            for row in eligible_all
        ),
        "changes_by_scene": changes,
        "dev": dev,
        "confirmation": {
            "status": "not_run" if not passed else "pending",
            "reason": stop_reason,
        },
        "conclusion_boundary": (
            "This rejects the tested train-derived final cluster-size mapping on "
            "the current frozen SAGA candidates; it does not reject all possible "
            "uses of category knowledge."
        ),
    }

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Writing the audit table requires pandas") from exc
    pd.DataFrame(rows).to_parquet(output / "noise_threshold_candidates.parquet", index=False)
    confirmation_rows = [
        {
            "scene_id": scene,
            "status": "not_run" if not passed else "pending",
            "reason": stop_reason,
        }
        for scene in plan["confirmation_scenes"]
    ]
    pd.DataFrame(confirmation_rows).to_parquet(
        output / "noise_threshold_confirm6.parquet", index=False
    )
    _write_json(output / "noise_threshold_plan.json", plan)
    _write_json(output / "noise_threshold_dev2.json", dev)
    _write_json(output / "noise_threshold_analysis.json", analysis)
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--prior-json", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gt-root")
    parser.add_argument("--data-root")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--dev-scenes", nargs="+", default=list(DEFAULT_DEV_SCENES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_final_noise_audit(
        trace_root=args.trace_root,
        prior_json=args.prior_json,
        output_root=args.output_root,
        scenes=args.scenes,
        dev_scenes=args.dev_scenes,
        gt_root=args.gt_root,
        data_root=args.data_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
