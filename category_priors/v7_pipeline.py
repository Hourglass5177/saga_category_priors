from __future__ import annotations

"""Registered staged V7 experiment controller.

The controller is deliberately sequential.  It records each completed gate in
one readable status JSON and stops immediately when a preregistered condition
fails.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .io import load_json, read_rows, write_json
from .taxonomy import load_taxonomy
from .v7_evaluation import evaluate_v7_bank, evaluate_v7_replays
from .v7_replay import CONDITIONS, replay_v7_priors
from .v7_runner import load_runtime_scenes, run_v7_banks


DEV8 = (
    "scene0645_00", "scene0025_01", "scene0046_00", "scene0474_01",
    "scene0591_02", "scene0329_02", "scene0164_03", "scene0064_01",
)
CAUSAL2 = DEV8[:2]
HOLDOUT5 = (
    "scene0231_00", "scene0608_00", "scene0356_00", "scene0011_00",
    "scene0593_00",
)


def _valid_output(path: Path) -> bool:
    try:
        payload = load_json(path)
        return isinstance(payload.get("point_labels"), list) and isinstance(payload.get("instances"), Mapping)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _run_b1_ablation(
    *, runtime_manifest: Path, pipeline: Path, output_root: Path,
    scene_ids: Sequence[str], level: str,
) -> None:
    scenes = load_runtime_scenes(runtime_manifest)
    for scene_id in scene_ids:
        scene = scenes[scene_id]
        target = output_root / level / scene_id
        output = target / "output.json"
        if _valid_output(output):
            print(f"[{level}/{scene_id}] reused", flush=True)
            continue
        target.mkdir(parents=True, exist_ok=True)
        command = [
            "bash", str(pipeline), "--stage", "postprocess",
            "--base-path", str(scene.base_path), "--python", str(scene.python_bin),
            "--json-path", str(output),
            "--prior-metadata-path", str(target / "diagnostics.json"),
            "--progress-path", str(target / "progress.txt"),
            "--scene-scale-m-per-unit", str(scene.scene_scale_m_per_unit),
            "--teacher-prior-mode", "original", "--minimal-metadata",
            "--v7-causal-ablation", level,
        ]
        with (target / "postprocess.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=pipeline.parent, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not _valid_output(output):
            raise RuntimeError(f"{level}/{scene_id} failed; see {target / 'postprocess.log'}")


def _link_historical_b1(source_root: Path, output_root: Path, scene_ids: Sequence[str]) -> None:
    """Expose immutable historical B1 files in the V7 evaluation layout."""
    for scene_id in scene_ids:
        source = source_root / scene_id / "seed-42"
        target = output_root / "P0" / scene_id
        target.mkdir(parents=True, exist_ok=True)
        for name in ("output.json", "diagnostics.json"):
            destination = target / name
            if destination.exists() or destination.is_symlink():
                continue
            origin = source / name
            if not origin.is_file():
                raise FileNotFoundError(f"historical B1 file missing: {origin}")
            os.symlink(origin, destination)


def _metrics_by_condition(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["condition"]): dict(row) for row in read_rows(path)}


def _mean_scene_delta(analysis: Mapping[str, Any], treatment: str, reference: str) -> tuple[float, int, int]:
    ref = {row["scene_id"]: row for row in analysis["conditions"][reference]["per_scene"]}
    trt = {row["scene_id"]: row for row in analysis["conditions"][treatment]["per_scene"]}
    deltas = [float(trt[key]["map_50_95"]) - float(ref[key]["map_50_95"]) for key in sorted(ref)]
    return float(np.mean(deltas)), sum(value > 0 for value in deltas), sum(value < 0 for value in deltas)


def _record(status_path: Path, status: dict[str, Any], stage: str, payload: Any) -> None:
    status[stage] = payload
    status["updated_at_unix"] = time.time()
    write_json(status_path, status)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    artifacts = args.artifacts
    runs = args.runs
    artifacts.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    status_path = artifacts / "v7_status.json"
    status = load_json(status_path) if status_path.is_file() else {
        "schema": "saga-v7-stage-status-v1", "state": "running"
    }
    taxonomy = load_taxonomy(args.taxonomy)

    causal_root = runs / "causal2"
    if args.historical_b1_root is not None:
        _link_historical_b1(args.historical_b1_root, causal_root, CAUSAL2)
    for level in ("L0", "L1", "L2", "L3"):
        _run_b1_ablation(
            runtime_manifest=args.runtime_manifest, pipeline=args.pipeline,
            output_root=causal_root, scene_ids=CAUSAL2, level=level,
        )
    causal_metrics = artifacts / "v7_causal_ablation2.parquet"
    causal_analysis_path = artifacts / "v7_contributor_audit2.json"
    causal_analysis = evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=causal_root, scene_ids=CAUSAL2,
        conditions=(("P0",) if args.historical_b1_root is not None else ()) + ("L0", "L1", "L2", "L3"), taxonomy=taxonomy,
        metrics_output=causal_metrics, analysis_output=causal_analysis_path,
        size_bins=args.size_bins,
    )
    causal = _metrics_by_condition(causal_metrics)
    knn_pollution = (
        causal["L1"]["gaussian_micro_precision"] - causal["L0"]["gaussian_micro_precision"] >= 0.05
        or causal["L0"]["unsupported_instance_fraction"] - causal["L1"]["unsupported_instance_fraction"] >= 0.05
    ) and causal["L1"]["mean_matched_gt_recall"] >= causal["L0"]["mean_matched_gt_recall"] - 0.10
    absorption_pollution = (
        causal["L2"]["gaussian_micro_precision"] - causal["L1"]["gaussian_micro_precision"] >= 0.05
        or causal["L1"]["unsupported_instance_fraction"] - causal["L2"]["unsupported_instance_fraction"] >= 0.05
    ) and causal["L2"]["mean_matched_gt_recall"] >= causal["L1"]["mean_matched_gt_recall"] - 0.10
    recall_gap = max(0.0, causal["L0"]["mean_matched_gt_recall"] - causal["L2"]["mean_matched_gt_recall"])
    halo_recovers = (
        causal["L3"]["mean_matched_gt_recall"] - causal["L2"]["mean_matched_gt_recall"] >= 0.5 * recall_gap
        and causal["L3"]["gaussian_micro_precision"] >= causal["L2"]["gaussian_micro_precision"]
        - 0.2 * max(causal["L2"]["gaussian_micro_precision"] - causal["L0"]["gaussian_micro_precision"], 0.0)
    )
    halo = bool(halo_recovers)
    _record(status_path, status, "stage0_causal", {
        "knn_pollution_confirmed": bool(knn_pollution),
        "center_absorption_pollution_confirmed": bool(absorption_pollution),
        "halo_enabled": halo,
        "historical_p0_minus_fixed_l0_changed": (
            causal.get("P0", {}).get("map_50_95") != causal["L0"]["map_50_95"]
            if "P0" in causal else None
        ),
        "metrics": causal,
    })

    bank2 = runs / ("bank-halo" if halo else "bank-core")
    run_v7_banks(args.runtime_manifest, CAUSAL2, bank2, args.repo_root, halo=halo)
    oracle_path = artifacts / "v7_oracle2.json"
    oracle = evaluate_v7_bank(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        bank_root=bank2, scene_ids=CAUSAL2, taxonomy=taxonomy,
        rows_output=artifacts / "v7_bank2.parquet", analysis_output=oracle_path,
        size_bins=args.size_bins,
    )
    oracle_pass = (
        int(oracle["oracles"]["association_match_050_count"]) >= 6
        and float(oracle["oracles"]["association_tiny_small_recall_025"]) >= 0.20
    )
    _record(status_path, status, "stage0_oracle", {**oracle, "passed": oracle_pass})
    if not oracle_pass:
        status["state"] = "stopped"
        status["stop_reason"] = "mask/Gaussian support oracle below V7 feasibility threshold"
        write_json(status_path, status)
        return status

    bank8 = runs / ("bank-tune24-halo" if halo else "bank-tune24-core")
    run_v7_banks(args.runtime_manifest, DEV8, bank8, args.repo_root, halo=halo)
    bank8_analysis = evaluate_v7_bank(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        bank_root=bank8, scene_ids=DEV8, taxonomy=taxonomy,
        rows_output=artifacts / "v7_bank8.parquet",
        analysis_output=artifacts / "v7_bank8_analysis.json", size_bins=args.size_bins,
    )
    bank_gate = (
        bank8_analysis["match_050_count"] >= 12
        and bank8_analysis["positive_050_scene_count"] >= 4
        and bank8_analysis["candidate_precision_025"] >= 0.10
        and bank8_analysis["tiny_small_recall_025"] >= 0.20
        and bank8_analysis["score_iou_spearman"] >= 0.20
    )
    _record(status_path, status, "stage1_bank", {**bank8_analysis, "candidate_gate_passed": bank_gate})
    if not bank_gate:
        status["state"] = "stopped"
        status["stop_reason"] = "V7 cross-view bank failed the preregistered candidate-health gate"
        write_json(status_path, status)
        return status

    _run_b1_ablation(
        runtime_manifest=args.runtime_manifest, pipeline=args.pipeline,
        output_root=runs / "b1-fixed8", scene_ids=DEV8, level="L0",
    )
    replay8 = runs / "replay-tune24"
    replay_v7_priors(
        bank_root=bank8, output_root=replay8, scene_ids=DEV8,
        conditions=CONDITIONS, category_priors=args.category_priors,
    )
    replay8_analysis = evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=replay8, scene_ids=DEV8, conditions=CONDITIONS,
        taxonomy=taxonomy, metrics_output=artifacts / "v7_prior_replay8.parquet",
        analysis_output=artifacts / "v7_prior_replay8_analysis.json",
        size_bins=args.size_bins,
    )
    replay_metrics = _metrics_by_condition(artifacts / "v7_prior_replay8.parquet")
    uniform = replay_metrics["U00-uniform"]
    b1_analysis = evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=runs / "b1-fixed8", scene_ids=DEV8, conditions=("L0",),
        taxonomy=taxonomy, metrics_output=artifacts / "v7_b1_fixed8_metrics.parquet",
        analysis_output=artifacts / "v7_b1_fixed8_analysis.json",
        size_bins=args.size_bins,
    )
    b1 = _metrics_by_condition(artifacts / "v7_b1_fixed8_metrics.parquet")["L0"]
    precision_gain = uniform["gaussian_micro_precision"] - b1["gaussian_micro_precision"]
    unsupported_drop = b1["unsupported_instance_fraction"] - uniform["unsupported_instance_fraction"]
    structure_gate = (
        (precision_gain >= 0.05 or unsupported_drop >= 0.10)
        and uniform["mean_matched_gt_recall"] >= b1["mean_matched_gt_recall"] - 0.05
        and uniform["map_50_95"] >= b1["map_50_95"] - 0.001
        and uniform["map_0.50"] >= b1["map_0.50"] - 0.002
        and uniform["predicted_instance_count"] <= 1.25 * max(b1["predicted_instance_count"], 1)
    )
    _record(status_path, status, "stage1_structure", {
        "passed": structure_gate, "b1_fixed": b1, "v7_uniform": uniform,
        "gaussian_precision_gain": precision_gain,
        "unsupported_instance_fraction_drop": unsupported_drop,
    })
    if not structure_gate:
        status["state"] = "stopped"
        status["stop_reason"] = "V7 uniform bank failed precision/AP safety relative to B1-fixed"
        write_json(status_path, status)
        return status
    data_candidates = []
    for condition in CONDITIONS[1:]:
        delta, positive, negative = _mean_scene_delta(replay8_analysis, condition, "U00-uniform")
        tiny_delta = (
            replay_metrics[condition]["tiny_small_recall_050"]
            - uniform["tiny_small_recall_050"]
        )
        fp_ratio = replay_metrics[condition]["predicted_instance_count"] / max(uniform["predicted_instance_count"], 1)
        data_candidates.append((condition, delta, tiny_delta, positive, negative, fp_ratio))
    data_candidates.sort(key=lambda item: (-item[1], -item[2], item[0]))
    best, delta, tiny_delta, positive, negative, fp_ratio = data_candidates[0]
    score_changed = False
    for scene_id in DEV8:
        u = load_json(replay8 / "U00-uniform" / scene_id / "diagnostics.json")
        d = load_json(replay8 / best / scene_id / "diagnostics.json")
        u_scores = {row["candidate_id"]: row["score_parts"]["score"] for row in u["accepted"] + u["rejected"]}
        d_scores = {row["candidate_id"]: row["score_parts"]["score"] for row in d["accepted"] + d["rejected"]}
        score_changed |= any(abs(u_scores[key] - d_scores[key]) > 1e-9 for key in u_scores)
    prior_gate = (
        score_changed
        and (delta >= 0.002 or (tiny_delta >= 0.01 and delta >= -0.0005))
        and positive > negative and fp_ratio <= 1.20
    )
    _record(status_path, status, "stage2_prior", {
        "mechanically_effective": score_changed, "best_condition": best,
        "mean_scene_delta_map": delta, "positive_scenes": positive,
        "negative_scenes": negative, "tiny_small_recall_050_delta": tiny_delta,
        "fp_ratio": fp_ratio, "passed": prior_gate,
        "metrics": replay_metrics,
    })
    if not prior_gate:
        status["state"] = "stopped"
        status["stop_reason"] = (
            "prior mapping did not affect scores" if not score_changed
            else "data-driven prior failed the preregistered 8-scene gate"
        )
        write_json(status_path, status)
        return status

    # The independent holdout is reached only after both the bank and prior gates pass.
    bank5 = bank8
    run_v7_banks(args.runtime_manifest, HOLDOUT5, bank5, args.repo_root, halo=halo)
    replay5 = replay8
    replay_v7_priors(
        bank_root=bank5, output_root=replay5, scene_ids=HOLDOUT5,
        conditions=("U00-uniform", best), category_priors=args.category_priors,
    )
    holdout_analysis = evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=replay5, scene_ids=HOLDOUT5,
        conditions=("U00-uniform", best), taxonomy=taxonomy,
        metrics_output=artifacts / "v7_holdout5_metrics.parquet",
        analysis_output=artifacts / "v7_holdout5_analysis.json",
        size_bins=args.size_bins,
    )
    holdout_delta, holdout_positive, holdout_negative = _mean_scene_delta(
        holdout_analysis, best, "U00-uniform"
    )
    holdout_metrics = _metrics_by_condition(artifacts / "v7_holdout5_metrics.parquet")
    holdout_tiny_delta = (
        holdout_metrics[best]["tiny_small_recall_050"]
        - holdout_metrics["U00-uniform"]["tiny_small_recall_050"]
    )
    holdout_pass = holdout_delta > 0 and holdout_positive >= 3 and holdout_tiny_delta > 0
    _record(status_path, status, "stage3_holdout5", {
        "best_condition": best, "mean_scene_delta_map": holdout_delta,
        "positive_scenes": holdout_positive, "negative_scenes": holdout_negative,
        "tiny_small_recall_050_delta": holdout_tiny_delta,
        "passed": holdout_pass,
    })
    if not holdout_pass:
        status["state"] = "stopped"
        status["stop_reason"] = "data-driven prior failed independent 5-scene holdout"
        write_json(status_path, status)
        return status

    tune_scenes = tuple(load_runtime_scenes(args.runtime_manifest))
    remaining = tuple(scene for scene in tune_scenes if scene not in set(DEV8) | set(HOLDOUT5))
    run_v7_banks(args.runtime_manifest, remaining, bank8, args.repo_root, halo=halo)
    replay_v7_priors(
        bank_root=bank8, output_root=replay8, scene_ids=remaining,
        conditions=("U00-uniform", best), category_priors=args.category_priors,
    )
    tune_analysis = evaluate_v7_replays(
        runtime_manifest=args.runtime_manifest, gt_dir=args.gt_dir,
        replay_root=replay8, scene_ids=tune_scenes,
        conditions=("U00-uniform", best), taxonomy=taxonomy,
        metrics_output=artifacts / "v7_tune24_metrics.parquet",
        analysis_output=artifacts / "v7_tune24_analysis.json",
        size_bins=args.size_bins,
    )
    reference_rows = {
        row["scene_id"]: row for row in tune_analysis["conditions"]["U00-uniform"]["per_scene"]
    }
    treatment_rows = {
        row["scene_id"]: row for row in tune_analysis["conditions"][best]["per_scene"]
    }
    physical_deltas: dict[str, list[float]] = {}
    for scene_id in tune_scenes:
        physical = scene_id.rsplit("_", 1)[0]
        physical_deltas.setdefault(physical, []).append(
            float(treatment_rows[scene_id]["map_50_95"])
            - float(reference_rows[scene_id]["map_50_95"])
        )
    physical_means = {key: float(np.mean(value)) for key, value in physical_deltas.items()}
    tune_macro_delta = float(np.mean(list(physical_means.values())))
    tune_pass = tune_macro_delta >= 0.002
    _record(status_path, status, "stage3_tune24", {
        "best_condition": best, "physical_scene_count": len(physical_means),
        "physical_scene_deltas": physical_means,
        "macro_delta_map": tune_macro_delta, "passed": tune_pass,
    })
    if not tune_pass:
        status["state"] = "stopped"
        status["stop_reason"] = "data-driven prior failed physical-scene-weighted tune24 gate"
        write_json(status_path, status)
        return status

    if args.locked_runtime_manifest is None or args.locked_gt_dir is None:
        status["state"] = "stopped"
        status["stop_reason"] = "locked runtime/GT paths were not supplied after tune24 passed"
        write_json(status_path, status)
        return status
    locked_scenes = tuple(load_runtime_scenes(args.locked_runtime_manifest))
    if len(locked_scenes) != 48 or "scene0019_01" not in locked_scenes or "scene0019_00" in locked_scenes:
        raise ValueError("locked runtime must contain 48 scenes and the scene0019_01 replacement")
    final_bank = runs / "bank-final48"
    run_v7_banks(args.locked_runtime_manifest, locked_scenes, final_bank, args.repo_root, halo=halo)
    final_replay = runs / "replay-final48"
    replay_v7_priors(
        bank_root=final_bank, output_root=final_replay, scene_ids=locked_scenes,
        conditions=("U00-uniform", best), category_priors=args.category_priors,
    )
    final_analysis = evaluate_v7_replays(
        runtime_manifest=args.locked_runtime_manifest, gt_dir=args.locked_gt_dir,
        replay_root=final_replay, scene_ids=locked_scenes,
        conditions=("U00-uniform", best), taxonomy=taxonomy,
        metrics_output=artifacts / "v7_final_metrics.parquet",
        analysis_output=artifacts / "v7_final_detail.json",
        viewer_output=artifacts / "viewer",
        size_bins=args.size_bins,
    )
    final_metrics = _metrics_by_condition(artifacts / "v7_final_metrics.parquet")
    final_delta = float(final_metrics[best]["map_50_95"] - final_metrics["U00-uniform"]["map_50_95"])
    _, _, _ = _mean_scene_delta(final_analysis, best, "U00-uniform")
    final_ref = {
        row["scene_id"]: float(row["map_50_95"])
        for row in final_analysis["conditions"]["U00-uniform"]["per_scene"]
    }
    final_trt = {
        row["scene_id"]: float(row["map_50_95"])
        for row in final_analysis["conditions"][best]["per_scene"]
    }
    deltas = np.asarray([final_trt[key] - final_ref[key] for key in sorted(final_ref)])
    rng = np.random.default_rng(20260804)
    indices = rng.integers(0, len(deltas), size=(10000, len(deltas)))
    bootstrap = deltas[indices].mean(axis=1)
    ci = [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))]
    final_pass = final_delta >= 0.002 and ci[0] > 0
    final_summary = {
        "schema": "saga-v7-final-analysis-v1", "best_condition": best,
        "delta_map_50_95": final_delta, "paired_bootstrap_samples": 10000,
        "paired_bootstrap_ci95": ci, "supports_stable_category_prior": final_pass,
        "uniform": final_metrics["U00-uniform"], "data": final_metrics[best],
    }
    write_json(artifacts / "v7_analysis.json", final_summary)
    _record(status_path, status, "stage4_final48", final_summary)
    status["state"] = "complete"
    status["best_condition"] = best
    status["stop_reason"] = None if final_pass else "V7 proposal-level category prior showed no stable improvement"
    write_json(status_path, status)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path)
    parser.add_argument("--locked-gt-dir", type=Path)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--historical-b1-root", type=Path)
    args = parser.parse_args(argv)
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
