#!/usr/bin/env bash
set -euo pipefail

# Single-process V9 supervisor.  The historical baseline closure remains an
# isolated prerequisite; the Python orchestrator then owns T1 through final48
# and enforces every preregistered stopping gate.

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${SAGA_V9_RUNS_ROOT:-/root/autodl-tmp/saga/runs/v9-objectbank}"
ARTIFACTS_ROOT="${SAGA_V9_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/v9-objectbank}"
CONTROLLER_PY="${SAGA_CONTROLLER_PYTHON:-/root/autodl-tmp/saga/venvs/category-priors/bin/python}"
RUNTIME_MANIFEST="${SAGA_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
LOCKED_RUNTIME_MANIFEST="${SAGA_LOCKED_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_scene_runtime.json}"
LOCKED_SCENES="${SAGA_LOCKED_SCENES:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_evaluation_scenes.json}"
GT_DIR="${SAGA_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
LOCKED_GT_DIR="${SAGA_LOCKED_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_locked}"
CATEGORY_PRIORS="${SAGA_CATEGORY_PRIORS:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/category_priors.json}"
SIZE_BINS="${SAGA_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
SAM_REUSABLE_ROOT="${SAGA_SAM_REUSABLE_ROOT:-/root/autodl-tmp/saga/runs/v8-mask-alpha/sam-everything}"
SAM_CHECKPOINT="${SAGA_SAM_CHECKPOINT:-/root/autodl-tmp/saga/workspace/saga/weights/sam_vit_h_4b8939.pth}"
LABEL_FEATURES="${SAGA_LABEL_FEATURES:-/root/autodl-tmp/saga/runs/baseline_20260731_rtx4090/outputs/labels/label_features.pt}"
TOP_STATUS="$ARTIFACTS_ROOT/v9_orchestrator_status.json"
CURRENT_STAGE="initializing"
COMMIT="$(git -C "$WORKSPACE" rev-parse HEAD)"

mkdir -p "$RUNS_ROOT" "$ARTIFACTS_ROOT"
export PYTHONPATH="$WORKSPACE/submodules/diff-gaussian-rasterization-max-contributor:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42

write_shell_status() {
    local state="$1"
    local message="${2:-}"
    "$CONTROLLER_PY" - "$TOP_STATUS" "$CURRENT_STAGE" "$state" "$COMMIT" "$message" <<'PY'
import json
import sys
import time
from pathlib import Path

path, checkpoint, state, commit, message = sys.argv[1:]
payload = {
    "schema": "saga-v9-orchestrator-status-v1",
    "state": state,
    "checkpoint": checkpoint,
    "git_commit": commit,
    "updated_at_unix": time.time(),
}
if message:
    payload["error"] = message
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

assert_resources() {
    "$CONTROLLER_PY" - "$RUNS_ROOT" <<'PY'
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 80.0:
    raise SystemExit(f"V9 requires >=80 GiB free, found {free_gib:.1f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum_text = (cgroup / "memory.max").read_text().strip()
maximum = int(maximum_text) if maximum_text != "max" else None
if maximum != 90 * 1024**3:
    raise SystemExit(f"expected 90 GiB memory.max, found {maximum_text}")
if current >= maximum:
    raise SystemExit("memory.current has reached memory.max")
PY
}

on_exit() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        write_shell_status "failed" "runner exited with code $code"
    fi
}
trap on_exit EXIT

CURRENT_STAGE="baseline-forensic-closure"
write_shell_status "running"
assert_resources
SAGA_BASELINE_WORK_ROOT="/root/autodl-tmp/saga/workspace/teacher-baseline-closure" \
SAGA_BASELINE_RUN_ROOT="/root/autodl-tmp/saga/runs/teacher-baseline-closure" \
SAGA_BASELINE_ARTIFACT_ROOT="/root/autodl-tmp/saga/artifacts/teacher-baseline-closure" \
SAGA_RUNTIME_MANIFEST="$RUNTIME_MANIFEST" \
SAGA_GT_DIR="$GT_DIR" \
bash "$WORKSPACE/continue_teacher_baseline_closure.sh"

CURRENT_STAGE="v9-t1-through-final"
write_shell_status "running"
assert_resources
"$CONTROLLER_PY" -m category_priors.v9_orchestrator \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --locked-runtime-manifest "$LOCKED_RUNTIME_MANIFEST" \
    --locked-evaluation-scenes "$LOCKED_SCENES" \
    --workspace "$WORKSPACE" \
    --runs-root "$RUNS_ROOT" \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --gt-dir "$GT_DIR" \
    --locked-gt-dir "$LOCKED_GT_DIR" \
    --sam-reusable-root "$SAM_REUSABLE_ROOT" \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --label-features "$LABEL_FEATURES" \
    --size-bins "$SIZE_BINS" \
    --category-priors "$CATEGORY_PRIORS" \
    --git-commit "$COMMIT"

trap - EXIT
