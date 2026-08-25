from __future__ import annotations

"""Production entry point for the preregistered V10 experiment."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import load_json, write_json
from .runner import load_scene_runtime_manifest
from .scannet import physical_scene_id
from .v10_closeout import write_v10_v9_closeout
from .v10_lifting_worker import (
    compatible_lifting_bank_is_complete,
)
from .v10_orchestrator import run_v10_orchestrator
from .v10_pipeline import DEV2, DEV8, HOLDOUT5
from .v10_runtime import FilesystemV10Config, FilesystemV10Hooks
from .v9_lifting_runner import ensure_v9_segment_everything


def _registered_final_scenes(
    locked_runtime_manifest: Path,
    locked_evaluation_scenes: Path,
) -> tuple[str, ...]:
    runtime = tuple(load_scene_runtime_manifest(locked_runtime_manifest))
    payload = load_json(locked_evaluation_scenes)
    rows = payload.get("scenes", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("locked evaluation scenes must be a sequence")
    registered = tuple(
        str(row["scene_id"] if isinstance(row, Mapping) else row) for row in rows
    )
    if len(runtime) != 48 or len(registered) != 48 or set(runtime) != set(registered):
        raise ValueError("locked runtime must exactly match the registered 48 scenes")
    if len({physical_scene_id(scene_id) for scene_id in runtime}) != 48:
        raise ValueError("final48 must contain 48 distinct physical scenes")
    if "scene0019_01" not in runtime or "scene0019_00" in runtime:
        raise ValueError("final48 must use the registered scene0019_01 replacement")
    return runtime


def register_readonly_dev2_lifting(
    *,
    v9_lifting_root: Path,
    v10_lifting_root: Path,
) -> None:
    """Expose immutable V9 DEV2 banks through the V10 run root by symlink."""

    source_root = Path(v9_lifting_root).resolve()
    target_root = Path(v10_lifting_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    for scene_id in DEV2:
        source = source_root / scene_id
        if not compatible_lifting_bank_is_complete(source, expected_scene_id=scene_id):
            raise RuntimeError(f"registered V9 DEV2 lifting is missing: {source}")
        target = target_root / scene_id
        if target.exists() or target.is_symlink():
            if (
                target.resolve() != source.resolve()
                or not compatible_lifting_bank_is_complete(
                    target, expected_scene_id=scene_id
                )
            ):
                raise RuntimeError(f"V10 lifting target is occupied or incomplete: {target}")
            continue
        target.symlink_to(source, target_is_directory=True)


def _assert_cloud_resources(path: Path) -> dict[str, Any]:
    free_gib = shutil.disk_usage(path).free / 1024**3
    if free_gib < 80.0:
        raise RuntimeError(f"V10 requires at least 80 GiB free; found {free_gib:.1f}")
    cgroup = Path("/sys/fs/cgroup")
    if not (cgroup / "memory.current").is_file():
        return {"disk_free_gib": free_gib, "cgroup": "unavailable"}
    current = int((cgroup / "memory.current").read_text().strip())
    maximum_text = (cgroup / "memory.max").read_text().strip()
    maximum = int(maximum_text) if maximum_text != "max" else None
    if maximum != 90 * 1024**3:
        raise RuntimeError(f"expected cgroup memory.max=90GiB, found {maximum_text}")
    if current >= maximum:
        raise RuntimeError("cgroup memory.current has reached memory.max")
    events = (cgroup / "memory.events").read_text().strip()
    return {
        "disk_free_gib": free_gib,
        "memory_current_bytes": current,
        "memory_max_bytes": maximum,
        "memory_events": events,
    }


def _resource_snapshot(path: Path) -> dict[str, Any]:
    try:
        return _assert_cloud_resources(path)
    except BaseException as error:
        return {"error_type": type(error).__name__, "error": str(error)}


def _metric_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = load_json(path)
    conditions = payload.get("conditions") if isinstance(payload, Mapping) else None
    if not isinstance(conditions, Mapping):
        return None
    keys = (
        "map_50_95",
        "ap50",
        "ap25",
        "tiny_small_recall_050",
        "predicted_instance_count",
        "gaussian_micro_precision",
        "unsupported_instance_fraction",
        "gt_recall",
    )
    result: dict[str, Any] = {}
    for condition, raw in conditions.items():
        metrics = raw.get("metrics") if isinstance(raw, Mapping) else None
        if isinstance(metrics, Mapping):
            result[str(condition)] = {key: metrics[key] for key in keys if key in metrics}
    return result


def _analysis_payload(
    *,
    result: Mapping[str, Any],
    runs_root: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    names = (
        "v10_v9_closeout.json",
        "v10_association_funnel2.parquet",
        "v10_pair_reconstruction_factorial2.parquet",
        "v10_view_consensus2.json",
        "v10_bank8.parquet",
        "v10_uniform_health8.json",
        "v10_prior_factorial8.parquet",
        "v10_tune24_metrics.parquet",
        "v10_final_metrics.parquet",
        "v10_final_bootstrap.json",
        "V10B_IDENTITY_TRAINING_PROPOSAL.md",
    )
    artifacts = {
        name: str((artifacts_root / name).resolve())
        for name in names
        if (artifacts_root / name).is_file()
    }
    viewer = artifacts_root / "viewer"
    if viewer.is_dir():
        artifacts["viewer"] = str(viewer.resolve())
    bootstrap_path = artifacts_root / "v10_final_bootstrap.json"
    return {
        **dict(result),
        "resource_snapshot": _resource_snapshot(runs_root),
        "runs_root": str(runs_root),
        "artifacts": artifacts,
        "tune24_metrics_summary": _metric_summary(
            artifacts_root / "v10_tune24_metrics.json"
        ),
        "final48_metrics_summary": _metric_summary(
            artifacts_root / "v10_final_metrics.json"
        ),
        "final_bootstrap": (
            load_json(bootstrap_path) if bootstrap_path.is_file() else None
        ),
    }


def _run_lifting_subprocess(
    *,
    scene_id: str,
    scene: Mapping[str, Any],
    runtime_manifest: Path,
    output_root: Path,
    sam_scene: Path,
    label_features: Path,
    workspace: Path,
    git_commit: str,
) -> None:
    python_bin = Path(str(scene.get("python_bin", ""))).resolve()
    if not python_bin.is_file():
        raise FileNotFoundError(f"scene Python does not exist: {python_bin}")
    log_path = output_root / "_logs" / f"{scene_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = (
        str(python_bin),
        "-m",
        "category_priors.v10_lifting_worker",
        "--runtime-manifest",
        str(runtime_manifest),
        "--scene",
        scene_id,
        "--output-root",
        str(output_root),
        "--git-commit",
        str(git_commit),
        "--segment-everything-root",
        str(Path(sam_scene).parent),
        "--label-features",
        str(label_features),
    )
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=workspace,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"V10 lifting worker exited {completed.returncode}; see {log_path}"
        )
    if not compatible_lifting_bank_is_complete(
        output_root / scene_id, expected_scene_id=scene_id
    ):
        raise RuntimeError(f"V10 lifting worker left an incomplete bank: {scene_id}")


def _run_v10_experiment_impl(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    runs_root = Path(args.runs_root).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    lifting_root = runs_root / "lifting"
    bank_root = runs_root / "banks"
    replay_root = runs_root / "replay"
    generated_sam_root = runs_root / "sam-everything"
    for path in (runs_root, artifacts_root, lifting_root, bank_root, replay_root):
        path.mkdir(parents=True, exist_ok=True)
    _assert_cloud_resources(runs_root)

    runtime_manifest = Path(args.runtime_manifest).resolve()
    locked_runtime_manifest = Path(args.locked_runtime_manifest).resolve()
    tune_scenes = tuple(load_scene_runtime_manifest(runtime_manifest))
    if len(tune_scenes) != 24 or len(set(tune_scenes)) != 24:
        raise ValueError("V10 tune runtime must contain exactly 24 unique scans")
    final_scenes = _registered_final_scenes(
        locked_runtime_manifest,
        Path(args.locked_evaluation_scenes).resolve(),
    )
    tune_rows = load_scene_runtime_manifest(runtime_manifest)
    final_rows = load_scene_runtime_manifest(locked_runtime_manifest)
    registered_development = set(DEV8).union(HOLDOUT5)
    if not registered_development.issubset(tune_rows):
        missing = sorted(registered_development.difference(tune_rows))
        raise ValueError(f"tune runtime lacks registered V10 development scans: {missing}")
    tune_physical = {physical_scene_id(scene_id) for scene_id in tune_scenes}
    final_physical = {physical_scene_id(scene_id) for scene_id in final_scenes}
    if len(tune_physical) != 13:
        raise ValueError("V10 tune24 must represent exactly 13 physical scenes")
    if tune_physical.intersection(final_physical):
        raise ValueError("V10 tune and final manifests must be physically disjoint")
    if set(tune_rows).intersection(final_rows):
        raise ValueError("V10 tune and final manifests contain ambiguous duplicate scan IDs")

    register_readonly_dev2_lifting(
        v9_lifting_root=Path(args.v9_lifting_root),
        v10_lifting_root=lifting_root,
    )
    closeout_path = artifacts_root / "v10_v9_closeout.json"
    write_v10_v9_closeout(
        v9_artifacts_root=Path(args.v9_artifacts_root),
        output_path=closeout_path,
        git_commit=args.git_commit,
    )

    def ensure_lifting(scene_id: str) -> None:
        _assert_cloud_resources(runs_root)
        if scene_id in tune_rows:
            manifest = runtime_manifest
            scene = tune_rows[scene_id]
        elif scene_id in final_rows:
            manifest = locked_runtime_manifest
            scene = final_rows[scene_id]
        else:
            raise KeyError(f"scene is absent from tune and final manifests: {scene_id}")
        sam_scene = ensure_v9_segment_everything(
            scene_id=scene_id,
            scene=scene,
            repo_root=workspace,
            output_root=generated_sam_root,
            sam_checkpoint=Path(args.sam_checkpoint).resolve(),
            reusable_root=(
                Path(args.sam_reusable_root).resolve()
                if args.sam_reusable_root is not None
                else None
            ),
        )
        _run_lifting_subprocess(
            scene_id=scene_id,
            scene=scene,
            runtime_manifest=manifest,
            output_root=lifting_root,
            sam_scene=sam_scene,
            label_features=Path(args.label_features).resolve(),
            workspace=workspace,
            git_commit=args.git_commit,
        )

    hooks = FilesystemV10Hooks(
        FilesystemV10Config(
            runtime_manifest=runtime_manifest,
            gt_dir=Path(args.gt_dir).resolve(),
            lifting_root=lifting_root,
            bank_root=bank_root,
            replay_root=replay_root,
            artifacts_root=artifacts_root,
            category_priors=Path(args.category_priors).resolve(),
            size_bins=Path(args.size_bins).resolve(),
            b1_fixed_prediction_root=Path(args.b1_fixed_prediction_root).resolve(),
            b1_fixed_condition=args.b1_fixed_condition,
            v9_closeout=closeout_path,
            git_commit=args.git_commit,
            taxonomy_path=(Path(args.taxonomy).resolve() if args.taxonomy else None),
            locked_runtime_manifest=locked_runtime_manifest,
            locked_gt_dir=Path(args.locked_gt_dir).resolve(),
            ensure_lifting=ensure_lifting,
        )
    )
    try:
        result = run_v10_orchestrator(
            hooks=hooks,
            artifacts_root=artifacts_root,
            git_commit=args.git_commit,
            tune24_scene_ids=tune_scenes,
            final48_scene_ids=final_scenes,
        )
    except BaseException:
        status_path = artifacts_root / "v10_orchestrator_status.json"
        if status_path.is_file():
            failed = load_json(status_path)
            write_json(
                artifacts_root / "v10_analysis.json",
                _analysis_payload(
                    result=failed,
                    runs_root=runs_root,
                    artifacts_root=artifacts_root,
                ),
            )
        raise
    write_json(
        artifacts_root / "v10_analysis.json",
        _analysis_payload(
            result=result,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
        ),
    )
    return result


def run_v10_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run V10 and always leave a self-contained terminal analysis record."""

    artifacts_root = Path(args.artifacts_root).resolve()
    runs_root = Path(args.runs_root).resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    try:
        return _run_v10_experiment_impl(args)
    except BaseException as error:
        status_path = artifacts_root / "v10_orchestrator_status.json"
        if status_path.is_file():
            result = load_json(status_path)
        else:
            result = {
                "schema": "saga-v10-orchestrator-status-v1",
                "state": "failed",
                "checkpoint": "experiment-boundary-exception",
                "git_commit": str(args.git_commit),
                "category_prior_tested": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        write_json(
            artifacts_root / "v10_analysis.json",
            _analysis_payload(
                result=result,
                runs_root=runs_root,
                artifacts_root=artifacts_root,
            ),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--v9-artifacts-root", type=Path, required=True)
    parser.add_argument("--v9-lifting-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-gt-dir", type=Path, required=True)
    parser.add_argument("--sam-reusable-root", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--label-features", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--b1-fixed-prediction-root", type=Path, required=True)
    parser.add_argument("--b1-fixed-condition", default="T1-B1")
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--git-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_v10_experiment(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "main",
    "register_readonly_dev2_lifting",
    "run_v10_experiment",
]
