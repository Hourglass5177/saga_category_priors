#!/usr/bin/env bash
set -euo pipefail

# Recoverable, single-process V10 supervisor.  The Python orchestrator owns all
# preregistered stage gates; this shell only fixes paths, resources and logging.

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${SAGA_V10_RUNS_ROOT:-/root/autodl-tmp/saga/runs/v10-view-consensus}"
ARTIFACTS_ROOT="${SAGA_V10_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/v10-view-consensus}"
V9_RUNS_ROOT="${SAGA_V9_RUNS_ROOT:-/root/autodl-tmp/saga/runs/v9-objectbank}"
V9_ARTIFACTS_ROOT="${SAGA_V9_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/v9-objectbank}"
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
B1_ROOT="${SAGA_B1_FIXED_ROOT:-$V9_RUNS_ROOT/t1-legacy}"
COMMIT="$(git -C "$WORKSPACE" rev-parse HEAD)"
LOG="$ARTIFACTS_ROOT/continuation.log"

mkdir -p "$RUNS_ROOT" "$ARTIFACTS_ROOT"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$WORKSPACE/submodules/diff-gaussian-rasterization-max-contributor:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42

"$CONTROLLER_PY" - "$RUNS_ROOT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 80.0:
    raise SystemExit(f"V10 requires >=80 GiB free, found {free_gib:.1f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum_text = (cgroup / "memory.max").read_text().strip()
maximum = int(maximum_text) if maximum_text != "max" else None
if maximum != 90 * 1024**3:
    raise SystemExit(f"expected memory.max=90GiB, found {maximum_text}")
if current >= maximum:
    raise SystemExit("memory.current reached memory.max")
events = {}
for line in (cgroup / "memory.events").read_text().splitlines():
    key, value = line.split()
    events[key] = int(value)
print(json.dumps({
    "disk_free_gib": free_gib,
    "memory_current_bytes": current,
    "memory_max_bytes": maximum,
    "memory_events": events,
}, sort_keys=True))
PY

"$CONTROLLER_PY" -m category_priors.v10_experiment \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --locked-runtime-manifest "$LOCKED_RUNTIME_MANIFEST" \
    --locked-evaluation-scenes "$LOCKED_SCENES" \
    --workspace "$WORKSPACE" \
    --runs-root "$RUNS_ROOT" \
    --artifacts-root "$ARTIFACTS_ROOT" \
    --v9-artifacts-root "$V9_ARTIFACTS_ROOT" \
    --v9-lifting-root "$V9_RUNS_ROOT/lifting/S-AM" \
    --gt-dir "$GT_DIR" \
    --locked-gt-dir "$LOCKED_GT_DIR" \
    --sam-reusable-root "$SAM_REUSABLE_ROOT" \
    --sam-checkpoint "$SAM_CHECKPOINT" \
    --label-features "$LABEL_FEATURES" \
    --size-bins "$SIZE_BINS" \
    --category-priors "$CATEGORY_PRIORS" \
    --b1-fixed-prediction-root "$B1_ROOT" \
    --b1-fixed-condition T1-B1 \
    --git-commit "$COMMIT"
