from __future__ import annotations

"""One recoverable entrypoint for the complete preregistered V9 experiment.

The historical baseline closeout is executed by the outer shell because it
uses several isolated source trees.  This controller starts at the corrected
T1 dev8 reference and then owns every V9 stage.  Subcontrollers remain the
authority for their own stopping gates; this module never changes a threshold
or continues past a registered stop.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .io import write_json
from .v9_pipeline import V9Stage2Config, run_v9_stage2
from .v9_stage3_controller import V9ContinuationConfig, run_v9_stage3_to_6
from .v9_t1_runner import V9_T1_DEV8, execute_v9_t1_runs


def _status(
    path: Path,
    *,
    state: str,
    checkpoint: str,
    git_commit: str,
    **payload: Any,
) -> dict[str, Any]:
    result = {
        "schema": "saga-v9-orchestrator-status-v1",
        "state": str(state),
        "checkpoint": str(checkpoint),
        "git_commit": str(git_commit),
        "updated_at_unix": time.time(),
        **payload,
    }
    write_json(path, result)
    return result


def run_v9_orchestrator(
    *,
    runtime_manifest: Path,
    locked_runtime_manifest: Path,
    locked_evaluation_scenes: Path,
    workspace: Path,
    runs_root: Path,
    artifacts_root: Path,
    gt_dir: Path,
    locked_gt_dir: Path,
    sam_reusable_root: Path | None,
    sam_checkpoint: Path,
    label_features: Path,
    size_bins: Path,
    category_priors: Path,
    git_commit: str,
    taxonomy_path: Path | None = None,
) -> dict[str, Any]:
    """Run T1, Stage 2, then Stages 3--6 without crossing a failed gate."""

    workspace = workspace.resolve()
    runs_root = runs_root.resolve()
    artifacts_root = artifacts_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    status_path = artifacts_root / "v9_orchestrator_status.json"
    commit = str(git_commit).strip()
    if not commit:
        raise ValueError("git_commit must be non-empty")

    try:
        _status(
            status_path,
            state="running",
            checkpoint="t1-corrected-dev8",
            git_commit=commit,
        )
        t1 = execute_v9_t1_runs(
            scene_manifest=runtime_manifest,
            output_root=runs_root / "t1-legacy",
            workspace=workspace,
            git_commit=commit,
            scene_ids=V9_T1_DEV8,
            resume=True,
            continue_on_error=False,
        )
        if any(row.get("status") == "failed" for row in t1.get("runs", ())):
            raise RuntimeError("corrected T1 dev8 contains a failed run")

        _status(
            status_path,
            state="running",
            checkpoint="stage2-two-scene-objectbank",
            git_commit=commit,
            t1_complete=True,
        )
        stage2 = run_v9_stage2(
            V9Stage2Config(
                runtime_manifest=runtime_manifest,
                workspace=workspace,
                runs_root=runs_root,
                artifacts_root=artifacts_root,
                gt_dir=gt_dir,
                sam_packed_root=(
                    sam_reusable_root
                    if sam_reusable_root is not None
                    else runs_root / "sam-everything"
                ),
                sam_checkpoint=sam_checkpoint,
                label_features=label_features,
                size_bins=size_bins,
                git_commit=commit,
                taxonomy_path=taxonomy_path,
            )
        )
        if stage2.get("state") != "complete":
            return _status(
                status_path,
                state="stopped",
                checkpoint=str(stage2.get("checkpoint", "stage2-stopped")),
                git_commit=commit,
                t1_complete=True,
                stage2=stage2,
                stop_reason=stage2.get("stop_reason"),
            )

        _status(
            status_path,
            state="running",
            checkpoint="stage3-to-6",
            git_commit=commit,
            t1_complete=True,
            stage2_checkpoint=stage2.get("checkpoint"),
        )
        continuation = run_v9_stage3_to_6(
            V9ContinuationConfig(
                stage2_status=artifacts_root / "v9_status.json",
                runtime_manifest=runtime_manifest,
                locked_runtime_manifest=locked_runtime_manifest,
                locked_evaluation_scenes=locked_evaluation_scenes,
                workspace=workspace,
                runs_root=runs_root,
                artifacts_root=artifacts_root,
                gt_dir=gt_dir,
                locked_gt_dir=locked_gt_dir,
                sam_packed_root=runs_root / "sam-everything",
                sam_reusable_root=sam_reusable_root,
                sam_checkpoint=sam_checkpoint,
                label_features=label_features,
                size_bins=size_bins,
                category_priors=category_priors,
                t1_b1_root=runs_root / "t1-legacy",
                t1_b1_condition="T1-B1",
                git_commit=commit,
                taxonomy_path=taxonomy_path,
            )
        )
        final_state = str(continuation.get("state", "failed"))
        if final_state not in {"complete", "stopped"}:
            raise RuntimeError(
                "V9 continuation returned a non-terminal state: " + final_state
            )
        return _status(
            status_path,
            state=final_state,
            checkpoint=str(continuation.get("checkpoint", "stage3-to-6-terminal")),
            git_commit=commit,
            t1_complete=True,
            stage2_checkpoint=stage2.get("checkpoint"),
            continuation=continuation,
            stop_reason=continuation.get("stop_reason"),
        )
    except BaseException as error:
        _status(
            status_path,
            state="failed",
            checkpoint="orchestrator-exception",
            git_commit=commit,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete SAGA V9 experiment")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-runtime-manifest", type=Path, required=True)
    parser.add_argument("--locked-evaluation-scenes", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--locked-gt-dir", type=Path, required=True)
    parser.add_argument("--sam-reusable-root", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--label-features", type=Path, required=True)
    parser.add_argument("--size-bins", type=Path, required=True)
    parser.add_argument("--category-priors", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--git-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_v9_orchestrator(
        runtime_manifest=args.runtime_manifest,
        locked_runtime_manifest=args.locked_runtime_manifest,
        locked_evaluation_scenes=args.locked_evaluation_scenes,
        workspace=args.workspace,
        runs_root=args.runs_root,
        artifacts_root=args.artifacts_root,
        gt_dir=args.gt_dir,
        locked_gt_dir=args.locked_gt_dir,
        sam_reusable_root=args.sam_reusable_root,
        sam_checkpoint=args.sam_checkpoint,
        label_features=args.label_features,
        size_bins=args.size_bins,
        category_priors=args.category_priors,
        taxonomy_path=args.taxonomy,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
