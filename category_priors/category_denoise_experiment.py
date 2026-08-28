from __future__ import annotations

"""Preregistered controller for full-class category-aware denoising.

The controller contains no clustering or prior formulas.  It only runs the
three public denoising operations over fixed scene groups, evaluates complete
outputs, applies the registered gates, and leaves one resumable status file.
"""

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .category_denoise_evaluation import evaluate_category_denoise
from .category_denoise_runner import (
    replay_category_denoise,
    run_category_denoise_b0_control,
    run_category_denoise_bank,
)
from .io import load_json, write_json, write_rows
from .prediction_contract import validate_prediction_contract
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .taxonomy import load_taxonomy
from .v9_metrics import paired_scannet_bootstrap_from_predictions

DEV2 = ("scene0645_00", "scene0025_01")
DEV8 = (
    "scene0645_00",
    "scene0025_01",
    "scene0046_00",
    "scene0474_01",
    "scene0591_02",
    "scene0329_02",
    "scene0164_03",
    "scene0064_01",
)
HOLDOUT5 = (
    "scene0231_00",
    "scene0608_00",
    "scene0356_00",
    "scene0011_00",
    "scene0593_00",
)


def _status(path: Path, **values: Any) -> dict[str, Any]:
    payload = {"schema": "saga-category-denoise-status-v1", **values}
    write_json(path, payload)
    return payload


def _assert_resources(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(root).free / 1024**3
    if free_gib < 80.0:
        raise RuntimeError(
            f"category denoising requires at least 80 GiB free; found {free_gib:.1f}"
        )
    cgroup = Path("/sys/fs/cgroup")
    if not (cgroup / "memory.current").is_file():
        return {"disk_free_gib": free_gib, "cgroup": "unavailable"}
    current = int((cgroup / "memory.current").read_text().strip())
    maximum_text = (cgroup / "memory.max").read_text().strip()
    maximum = int(maximum_text) if maximum_text != "max" else None
    if maximum is not None and current >= maximum:
        raise RuntimeError("cgroup memory.current has reached memory.max")
    return {
        "disk_free_gib": free_gib,
        "memory_current_bytes": current,
        "memory_max": maximum_text,
        "memory_events": (cgroup / "memory.events").read_text().strip(),
    }


def _prediction(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    labels = np.asarray(payload.get("point_labels"), dtype=np.int64)
    instances = payload.get("instances")
    if labels.ndim != 1 or not isinstance(instances, Mapping):
        raise ValueError(f"invalid prediction: {path}")
    validate_prediction_contract(labels, instances)
    return payload


def _dev2_parity(runs_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene_id in DEV2:
        disabled = _prediction(runs_root / "b0-off" / scene_id / "output.json")
        bank_b0 = _prediction(runs_root / "bank" / scene_id / "output.json")
        disabled_labels = np.asarray(disabled["point_labels"], dtype=np.int64)
        bank_labels = np.asarray(bank_b0["point_labels"], dtype=np.int64)
        changed = int(np.count_nonzero(disabled_labels != bank_labels))
        instances_exact = disabled["instances"] == bank_b0["instances"]
        rows.append(
            {
                "scene_id": scene_id,
                "point_count": len(disabled_labels),
                "changed_point_count": changed,
                "point_labels_exact": changed == 0,
                "instances_exact": bool(instances_exact),
                "passed": changed == 0 and bool(instances_exact),
            }
        )
    return {"passed": all(row["passed"] for row in rows), "scenes": rows}


def _replay_diagnostics(runs_root: Path, scene_id: str, mode: str) -> Mapping[str, Any]:
    payload = load_json(runs_root / mode / scene_id / "diagnostics.json")
    value = payload.get("category_denoise", payload)
    if not isinstance(value, Mapping):
        raise TypeError(f"invalid denoising diagnostics: {mode}/{scene_id}")
    return value


def _mechanical_effect(runs_root: Path, scene_ids: Sequence[str]) -> dict[str, Any]:
    changed_scores = 0
    candidate_count = 0
    changed_acceptance: list[dict[str, Any]] = []
    for scene_id in map(str, scene_ids):
        uniform = {
            int(row["candidate_id"]): row
            for row in _replay_diagnostics(runs_root, scene_id, "uniform")["decisions"]
        }
        data = {
            int(row["candidate_id"]): row
            for row in _replay_diagnostics(runs_root, scene_id, "class")["decisions"]
        }
        if set(uniform) != set(data):
            raise ValueError(f"U/D candidate IDs differ for {scene_id}")
        for candidate_id in sorted(uniform):
            left = uniform[candidate_id]
            right = data[candidate_id]
            candidate_count += 1
            changed_scores += int(abs(float(right["score"]) - float(left["score"])) >= 0.01)
            if bool(left["accepted"]) != bool(right["accepted"]):
                changed_acceptance.append(
                    {
                        "scene_id": scene_id,
                        "candidate_id": candidate_id,
                        "class": str(right["branch_class"]),
                        "uniform_accepted": bool(left["accepted"]),
                        "class_accepted": bool(right["accepted"]),
                    }
                )
    changed_classes = {row["class"] for row in changed_acceptance}
    changed_scenes = {row["scene_id"] for row in changed_acceptance}
    score_fraction = changed_scores / candidate_count if candidate_count else 0.0
    passed = score_fraction >= 0.10 or (
        len(changed_acceptance) >= 5
        and len(changed_classes) >= 2
        and len(changed_scenes) >= 2
    )
    return {
        "passed": passed,
        "candidate_count": candidate_count,
        "score_delta_ge_0.01_count": changed_scores,
        "score_delta_ge_0.01_fraction": score_fraction,
        "acceptance_change_count": len(changed_acceptance),
        "acceptance_change_class_count": len(changed_classes),
        "acceptance_change_scene_count": len(changed_scenes),
        "acceptance_changes": changed_acceptance,
    }


def _condition(analysis: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return analysis["conditions"][name]


def _metric(analysis: Mapping[str, Any], condition: str, key: str) -> float:
    value = _condition(analysis, condition)["metrics"].get(key)
    if value is None or not np.isfinite(float(value)):
        raise ValueError(f"missing finite metric {condition}.{key}")
    return float(value)


def _per_scene(analysis: Mapping[str, Any], condition: str) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["scene_id"]): row
        for row in _condition(analysis, condition)["per_scene"]
    }


def _fp_tp_ratio(metrics: Mapping[str, Any]) -> float:
    return float(metrics["false_positive_count"]) / max(
        float(metrics["true_positive_count"]), 1.0
    )


def _evaluate(
    *,
    runtime_manifest: Path,
    gt_dir: Path,
    runs_root: Path,
    artifacts_root: Path,
    size_bins: Path,
    scene_ids: Sequence[str],
    metrics_name: str,
    analysis_name: str,
    viewer: bool = False,
) -> dict[str, Any]:
    return evaluate_category_denoise(
        runtime_manifest=runtime_manifest,
        gt_dir=gt_dir,
        bank_root=runs_root / "bank",
        prediction_root=runs_root,
        scene_ids=tuple(map(str, scene_ids)),
        conditions=("bank", "uniform", "class"),
        taxonomy=load_taxonomy(),
        metrics_output=artifacts_root / metrics_name,
        analysis_output=artifacts_root / analysis_name,
        size_bins=size_bins,
        viewer_output=artifacts_root / "viewer" if viewer else None,
    )


def _bank_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in analysis["candidate_bank"]["per_scene"]:
        result.append({key: value for key, value in row.items() if key != "candidates"})
    return result


def _dev8_gate(analysis: Mapping[str, Any], mechanical: Mapping[str, Any]) -> dict[str, Any]:
    bank = analysis["candidate_bank"]
    b0 = _condition(analysis, "bank")["metrics"]
    uniform = _condition(analysis, "uniform")["metrics"]
    data = _condition(analysis, "class")["metrics"]
    uniform_safety = {
        "map": float(uniform["map_50_95"]) - float(b0["map_50_95"]) >= -0.001,
        "ap50": float(uniform["ap50"]) - float(b0["ap50"]) >= -0.002,
        "instances": float(uniform["predicted_instance_count"])
        <= 1.25 * max(float(b0["predicted_instance_count"]), 1.0),
        "coverage": float(uniform["prediction_coverage"])
        >= float(b0["prediction_coverage"]) - 0.01,
    }
    uniform_rows = _per_scene(analysis, "uniform")
    data_rows = _per_scene(analysis, "class")
    deltas = {
        scene_id: float(data_rows[scene_id]["map_50_95"])
        - float(uniform_rows[scene_id]["map_50_95"])
        for scene_id in uniform_rows
    }
    positive = sum(value > 0 for value in deltas.values())
    negative = sum(value < 0 for value in deltas.values())
    delta_map = float(data["map_50_95"]) - float(uniform["map_50_95"])
    delta_tiny = float(data["tiny_small_recall_050"]) - float(
        uniform["tiny_small_recall_050"]
    )
    ratio_ok = _fp_tp_ratio(data) <= 1.20 * _fp_tp_ratio(uniform) + 1e-12
    benefit = delta_map >= 0.002 or (delta_tiny >= 0.01 and delta_map >= -0.0005)
    checks = {
        "candidate_space": int(bank["same_class_iou_050_count"]) >= 12
        and int(bank["same_class_iou_050_scene_count"]) >= 4,
        "uniform_structure_safe": all(uniform_safety.values()),
        "data_prior_mechanical": bool(mechanical["passed"]),
        "data_prior_benefit": benefit,
        "positive_scenes_more_than_negative": positive > negative,
        "fp_tp_degradation_within_20_percent": ratio_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "uniform_safety": uniform_safety,
        "delta_map_50_95": delta_map,
        "delta_tiny_small_recall_050": delta_tiny,
        "positive_scene_count": positive,
        "negative_scene_count": negative,
        "per_scene_delta": deltas,
        "uniform_fp_tp_ratio": _fp_tp_ratio(uniform),
        "class_fp_tp_ratio": _fp_tp_ratio(data),
    }


def _holdout_gate(analysis: Mapping[str, Any]) -> dict[str, Any]:
    uniform = _per_scene(analysis, "uniform")
    data = _per_scene(analysis, "class")
    deltas = {
        scene_id: float(data[scene_id]["map_50_95"])
        - float(uniform[scene_id]["map_50_95"])
        for scene_id in HOLDOUT5
    }
    tiny_delta = _metric(analysis, "class", "tiny_small_recall_050") - _metric(
        analysis, "uniform", "tiny_small_recall_050"
    )
    checks = {
        "mean_delta_positive": float(np.mean(list(deltas.values()))) > 0,
        "at_least_three_of_five_positive": sum(value > 0 for value in deltas.values())
        >= 3,
        "tiny_small_positive": tiny_delta > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mean_delta_map_50_95": float(np.mean(list(deltas.values()))),
        "tiny_small_recall_050_delta": tiny_delta,
        "per_scene_delta": deltas,
    }


def _physical_macro_gate(analysis: Mapping[str, Any]) -> dict[str, Any]:
    uniform = _per_scene(analysis, "uniform")
    data = _per_scene(analysis, "class")
    grouped: dict[str, list[float]] = defaultdict(list)
    for scene_id in uniform:
        grouped[physical_scene_id(scene_id)].append(
            float(data[scene_id]["map_50_95"])
            - float(uniform[scene_id]["map_50_95"])
        )
    physical_deltas = {
        group: float(np.mean(values)) for group, values in sorted(grouped.items())
    }
    macro = float(np.mean(list(physical_deltas.values())))
    return {
        "passed": len(physical_deltas) == 13 and macro >= 0.002,
        "physical_scene_count": len(physical_deltas),
        "macro_delta_map_50_95": macro,
        "per_physical_scene_delta": physical_deltas,
    }


def _final_gate(analysis: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    uniform = _condition(analysis, "uniform")["metrics"]
    data = _condition(analysis, "class")["metrics"]
    uniform_rows = _per_scene(analysis, "uniform")
    data_rows = _per_scene(analysis, "class")
    scene_delta = {
        scene_id: float(data_rows[scene_id]["map_50_95"])
        - float(uniform_rows[scene_id]["map_50_95"])
        for scene_id in uniform_rows
    }
    positive = sum(value > 0 for value in scene_delta.values())
    negative = sum(value < 0 for value in scene_delta.values())
    delta_map = float(bootstrap["delta_map_50_95"])
    ci = list(map(float, bootstrap["paired_bootstrap_ci95"]))
    checks = {
        "delta_at_least_0.002": delta_map >= 0.002,
        "ci_lower_above_zero": ci[0] > 0,
        "positive_scenes_more_than_negative": positive > negative,
        "ap50_not_regressed": float(data["ap50"]) - float(uniform["ap50"]) >= -0.002,
        "gaussian_precision_not_regressed": float(data["gaussian_micro_precision"])
        - float(uniform["gaussian_micro_precision"])
        >= -0.05,
        "fp_tp_degradation_within_20_percent": _fp_tp_ratio(data)
        <= 1.20 * _fp_tp_ratio(uniform) + 1e-12,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "delta_map_50_95": delta_map,
        "paired_bootstrap_ci95": ci,
        "positive_scene_count": positive,
        "negative_scene_count": negative,
    }


def _registered_final_scenes(runtime_manifest: Path, registered_path: Path) -> tuple[str, ...]:
    runtime = tuple(load_scene_runtime_manifest(runtime_manifest))
    payload = load_json(registered_path)
    rows = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("locked evaluation scenes must be a sequence")
    registered = tuple(
        str(row["scene_id"] if isinstance(row, Mapping) else row) for row in rows
    )
    if len(runtime) != 48 or set(runtime) != set(registered):
        raise ValueError("locked runtime must exactly match the registered 48 scenes")
    if len({physical_scene_id(scene_id) for scene_id in runtime}) != 48:
        raise ValueError("final48 must contain 48 distinct physical scenes")
    if "scene0019_01" not in runtime or "scene0019_00" in runtime:
        raise ValueError("final48 must use scene0019_01")
    return runtime


def _run_batch(
    *,
    runtime_manifest: Path,
    scene_ids: Sequence[str],
    runs_root: Path,
    repo_root: Path,
    priors: Path,
) -> None:
    _assert_resources(runs_root)
    run_category_denoise_bank(
        runtime_manifest, runs_root, repo_root, priors, scene_ids
    )
    replay_category_denoise(
        runtime_manifest,
        runs_root / "bank",
        runs_root,
        repo_root,
        priors,
        scene_ids,
        mode=("uniform", "class"),
    )


def run_category_denoise_experiment(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    runs_root = Path(args.runs_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    tune_manifest = Path(args.runtime_manifest).resolve()
    locked_manifest = Path(args.locked_runtime_manifest).resolve()
    priors = Path(args.category_priors).resolve()
    size_bins = Path(args.size_bins).resolve()
    tune_gt = Path(args.gt_dir).resolve()
    locked_gt = Path(args.locked_gt_dir).resolve()
    for path in (runs_root, artifacts_root):
        path.mkdir(parents=True, exist_ok=True)
    status_path = artifacts_root / "category_denoise_status.json"
    history: list[dict[str, Any]] = []
    tune_scenes = tuple(load_scene_runtime_manifest(tune_manifest))
    if len(tune_scenes) != 24 or not set(DEV8).union(HOLDOUT5).issubset(tune_scenes):
        raise ValueError("tune runtime does not contain the registered 24 scans")
    final_scenes = _registered_final_scenes(
        locked_manifest, Path(args.locked_evaluation_scenes).resolve()
    )
    try:
        _status(
            status_path,
            state="running",
            checkpoint="stage-a-dev2-mechanical",
            history=history,
            category_prior_tested=False,
            resources=_assert_resources(runs_root),
        )
        run_category_denoise_b0_control(
            tune_manifest, runs_root, repo_root, priors, DEV2
        )
        _run_batch(
            runtime_manifest=tune_manifest,
            scene_ids=DEV2,
            runs_root=runs_root,
            repo_root=repo_root,
            priors=priors,
        )
        shutil.copyfile(
            runs_root / "category_denoise_params.json",
            artifacts_root / "category_denoise_params.json",
        )
        parity = _dev2_parity(runs_root)
        mechanical2 = _mechanical_effect(runs_root, DEV2)
        history.append(
            {"stage": "stage-a-dev2-mechanical", "parity": parity, "mechanical": mechanical2}
        )
        if not parity["passed"]:
            return _status(
                status_path,
                state="stopped",
                checkpoint="stage-a-b0-parity-failed",
                history=history,
                category_prior_tested=False,
                stop_reason="bank side path changed B0 output before denoising",
            )

        _status(
            status_path,
            state="running",
            checkpoint="stage-b-dev8",
            history=history,
            category_prior_tested=False,
        )
        _run_batch(
            runtime_manifest=tune_manifest,
            scene_ids=DEV8,
            runs_root=runs_root,
            repo_root=repo_root,
            priors=priors,
        )
        analysis8 = _evaluate(
            runtime_manifest=tune_manifest,
            gt_dir=tune_gt,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
            size_bins=size_bins,
            scene_ids=DEV8,
            metrics_name="category_denoise_metrics8.parquet",
            analysis_name="category_denoise_analysis8.json",
            viewer=True,
        )
        write_rows(
            artifacts_root / "category_denoise_bank8.parquet", _bank_rows(analysis8)
        )
        mechanical8 = _mechanical_effect(runs_root, DEV8)
        gate8 = _dev8_gate(analysis8, mechanical8)
        history.append(
            {"stage": "stage-b-dev8", "gate": gate8, "mechanical": mechanical8}
        )
        if not gate8["passed"]:
            result = _status(
                status_path,
                state="stopped",
                checkpoint="stage-b-dev8-gate-failed",
                history=history,
                category_prior_tested=bool(mechanical8["passed"]),
                stop_reason="DEV8 full-class denoising gate did not pass",
                gate=gate8,
            )
            write_json(artifacts_root / "category_denoise_analysis.json", result)
            return result

        _status(
            status_path,
            state="running",
            checkpoint="stage-c-holdout5",
            history=history,
            category_prior_tested=True,
        )
        _run_batch(
            runtime_manifest=tune_manifest,
            scene_ids=HOLDOUT5,
            runs_root=runs_root,
            repo_root=repo_root,
            priors=priors,
        )
        holdout_analysis = _evaluate(
            runtime_manifest=tune_manifest,
            gt_dir=tune_gt,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
            size_bins=size_bins,
            scene_ids=HOLDOUT5,
            metrics_name="category_denoise_holdout5.parquet",
            analysis_name="category_denoise_holdout5.json",
        )
        holdout_gate = _holdout_gate(holdout_analysis)
        history.append({"stage": "stage-c-holdout5", "gate": holdout_gate})
        if not holdout_gate["passed"]:
            result = _status(
                status_path,
                state="stopped",
                checkpoint="stage-c-holdout5-gate-failed",
                history=history,
                category_prior_tested=True,
                stop_reason="DEV8 prior benefit did not replicate on holdout5",
                gate=holdout_gate,
            )
            write_json(artifacts_root / "category_denoise_analysis.json", result)
            return result

        _status(
            status_path,
            state="running",
            checkpoint="stage-c-tune24",
            history=history,
            category_prior_tested=True,
        )
        _run_batch(
            runtime_manifest=tune_manifest,
            scene_ids=tune_scenes,
            runs_root=runs_root,
            repo_root=repo_root,
            priors=priors,
        )
        tune_analysis = _evaluate(
            runtime_manifest=tune_manifest,
            gt_dir=tune_gt,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
            size_bins=size_bins,
            scene_ids=tune_scenes,
            metrics_name="category_denoise_tune24.parquet",
            analysis_name="category_denoise_tune24.json",
        )
        tune_gate = _physical_macro_gate(tune_analysis)
        history.append({"stage": "stage-c-tune24", "gate": tune_gate})
        if not tune_gate["passed"]:
            result = _status(
                status_path,
                state="stopped",
                checkpoint="stage-c-tune24-gate-failed",
                history=history,
                category_prior_tested=True,
                stop_reason="class denoising did not reach +0.002 physical-scene macro mAP",
                gate=tune_gate,
            )
            write_json(artifacts_root / "category_denoise_analysis.json", result)
            return result

        _status(
            status_path,
            state="running",
            checkpoint="stage-d-final48",
            history=history,
            category_prior_tested=True,
        )
        final_root = runs_root / "final48"
        _run_batch(
            runtime_manifest=locked_manifest,
            scene_ids=final_scenes,
            runs_root=final_root,
            repo_root=repo_root,
            priors=priors,
        )
        final_analysis = _evaluate(
            runtime_manifest=locked_manifest,
            gt_dir=locked_gt,
            runs_root=final_root,
            artifacts_root=artifacts_root,
            size_bins=size_bins,
            scene_ids=final_scenes,
            metrics_name="category_denoise_final48.parquet",
            analysis_name="category_denoise_final48.json",
            viewer=True,
        )
        bootstrap = paired_scannet_bootstrap_from_predictions(
            runtime_manifest=locked_manifest,
            gt_dir=locked_gt,
            prediction_root=final_root,
            scene_ids=final_scenes,
            reference_condition="uniform",
            treatment_condition="class",
            taxonomy=load_taxonomy(),
            samples=10_000,
        )
        write_json(artifacts_root / "category_denoise_final48_bootstrap.json", bootstrap)
        final_gate = _final_gate(final_analysis, bootstrap)
        history.append({"stage": "stage-d-final48", "gate": final_gate})
        result = _status(
            status_path,
            state="complete" if final_gate["passed"] else "stopped",
            checkpoint=(
                "stage-d-category-prior-supported"
                if final_gate["passed"]
                else "stage-d-no-stable-improvement"
            ),
            history=history,
            category_prior_tested=True,
            category_prior_supported=bool(final_gate["passed"]),
            gate=final_gate,
        )
        write_json(artifacts_root / "category_denoise_analysis.json", result)
        return result
    except BaseException as error:
        failed = _status(
            status_path,
            state="failed",
            checkpoint="experiment-exception",
            history=history,
            category_prior_tested=False,
            error_type=type(error).__name__,
            error=str(error),
        )
        write_json(artifacts_root / "category_denoise_analysis.json", failed)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--locked-runtime-manifest", required=True)
    parser.add_argument("--locked-evaluation-scenes", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--locked-gt-dir", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--artifacts-root", required=True)
    parser.add_argument("--category-priors", required=True)
    parser.add_argument("--size-bins", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run_category_denoise_experiment(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
