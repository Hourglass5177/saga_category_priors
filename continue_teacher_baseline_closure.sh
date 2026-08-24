#!/usr/bin/env bash
set -euo pipefail

# Original teacher-handoff closeout.  This script is intentionally sequential:
# one GPU process at a time, isolated 18-class assets, no downloads and no GT
# until the final read-only evaluation stages.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${SAGA_BASELINE_WORK_ROOT:-/root/autodl-tmp/saga/workspace/teacher-baseline-closure}"
CLOSURE_ROOT="${SAGA_BASELINE_RUN_ROOT:-/root/autodl-tmp/saga/runs/teacher-baseline-closure}"
ARTIFACT_ROOT="${SAGA_BASELINE_ARTIFACT_ROOT:-/root/autodl-tmp/saga/artifacts/teacher-baseline-closure}"
SOURCE_ROOT="$WORK_ROOT/sources"
PY="${SAGA_PYTHON:-/root/autodl-tmp/saga/conda/envs/saga/bin/python}"
CUDA_HOME="${SAGA_CUDA_HOME:-/root/autodl-tmp/saga/conda/envs/saga}"
RUNTIME_MANIFEST="${SAGA_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
GT_DIR="${SAGA_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
WEIGHT_ROOT="${SAGA_WEIGHT_ROOT:-/root/autodl-tmp/saga/workspace/saga/weights}"
HF_HOME="${SAGA_HF_HOME:-/root/autodl-tmp/saga/cache/huggingface}"
STATUS_PATH="$ARTIFACT_ROOT/baseline_status.json"
CURRENT_STAGE="initializing"

mkdir -p "$WORK_ROOT" "$CLOSURE_ROOT" "$ARTIFACT_ROOT" "$SOURCE_ROOT"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

write_status() {
    local state="$1"
    "$PY" - "$STATUS_PATH" "$CURRENT_STAGE" "$state" "$(git -C "$REPO_DIR" rev-parse HEAD)" <<'PY'
import json
import sys
import time
from pathlib import Path

path, stage, state, commit = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "schema": "saga-teacher-baseline-status-v1",
            "stage": stage,
            "state": state,
            "commit": commit,
            "updated_at_unix": time.time(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

on_exit() {
    local code=$?
    if [[ $code -ne 0 ]]; then
        write_status "failed"
    fi
}
trap on_exit EXIT

assert_resources() {
    "$PY" - "$CLOSURE_ROOT" <<'PY'
import shutil
from pathlib import Path

root = Path(__import__("sys").argv[1])
free = shutil.disk_usage(root).free / 1024**3
if free < 80:
    raise SystemExit(f"need >=80 GiB free, found {free:.1f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum = int((cgroup / "memory.max").read_text().strip())
if maximum != 90 * 1024**3:
    raise SystemExit(f"expected 90 GiB memory.max, found {maximum}")
if current >= maximum:
    raise SystemExit("memory.current has reached memory.max")
PY
}

export_commit() {
    local commit="$1"
    local destination="$2"
    mkdir -p "$destination"
    git -C "$REPO_DIR" archive "$commit" | tar -xf - -C "$destination"
}

build_contributor_extension() {
    local source_tree="$1"
    local extension="$source_tree/submodules/diff-gaussian-rasterization-max-contributor"
    test -f "$extension/setup.py"
    (
        cd "$extension"
        "$PY" setup.py build_ext --inplace
    )
    find "$extension/diff_gaussian_rasterization_max_contributor" \
        -maxdepth 1 -type f -name '_C*.so' -print -quit | grep -q .
}

CURRENT_STAGE="prepare-isolated-sources"
write_status "running"
assert_resources

LITERAL_ROOT="$SOURCE_ROOT/literal-bfc"
FULL950_ROOT="$SOURCE_ROOT/full950"
ARGS_ONLY_ROOT="$SOURCE_ROOT/args-only"
ARGS_NORM_ROOT="$SOURCE_ROOT/args-norm"
FIXED_ROOT="$SOURCE_ROOT/full950-contributor-fixed"

export_commit bfc21922384cc991a71b5e51429354b5d6b06375 "$LITERAL_ROOT"
export_commit 95073c640a77984c6af24abb276147e4315abcd1 "$FULL950_ROOT"
build_contributor_extension "$LITERAL_ROOT"

if [[ ! -f "$ARGS_ONLY_ROOT/train_contrastive_feature.py" && ! -e "$ARGS_NORM_ROOT" ]]; then
    "$PY" -m category_priors.baseline_closure_variants \
        --bfc-root "$LITERAL_ROOT" \
        --output-root "$SOURCE_ROOT"
fi
test -f "$ARGS_ONLY_ROOT/train_contrastive_feature.py"
test -f "$ARGS_NORM_ROOT/train_contrastive_feature.py"

# The partial variants are byte copies of the built literal tree except for
# their registered training repairs, so each has its own isolated extension.
find "$ARGS_ONLY_ROOT/submodules/diff-gaussian-rasterization-max-contributor" \
    -name '_C*.so' -print -quit | grep -q .
find "$ARGS_NORM_ROOT/submodules/diff-gaussian-rasterization-max-contributor" \
    -name '_C*.so' -print -quit | grep -q .
build_contributor_extension "$FULL950_ROOT"

if [[ ! -d "$FIXED_ROOT" ]]; then
    "$PY" -m category_priors.baseline_closure_contributor \
        --full950-root "$FULL950_ROOT" \
        --fixed-rasterizer-root \
            "$REPO_DIR/submodules/diff-gaussian-rasterization-max-contributor" \
        --output-root "$FIXED_ROOT"
fi

cat > "$ARTIFACT_ROOT/workspace_manifest.json" <<EOF
{
  "workspaces": {
    "literal-bfc": "$LITERAL_ROOT",
    "args-only": "$ARGS_ONLY_ROOT",
    "args-norm": "$ARGS_NORM_ROOT",
    "full950": "$FULL950_ROOT"
  }
}
EOF

CURRENT_STAGE="historical-handoff-runs"
write_status "running"
assert_resources
"$PY" -m category_priors.baseline_closure_runner \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --workspace-manifest "$ARTIFACT_ROOT/workspace_manifest.json" \
    --output-root "$CLOSURE_ROOT" \
    --sam-checkpoint "$WEIGHT_ROOT/sam_vit_h_4b8939.pth" \
    --groundingdino-checkpoint "$WEIGHT_ROOT/groundingdino_swint_ogc.pth" \
    --groundingdino-config "$WEIGHT_ROOT/GroundingDINO_SwinT_OGC.py"

CURRENT_STAGE="fixed-contributor-and-structural-ablation"
write_status "running"
assert_resources
build_contributor_extension "$FIXED_ROOT"
build_contributor_extension "$REPO_DIR"
"$PY" -m category_priors.baseline_closure_ablation \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --closure-root "$CLOSURE_ROOT" \
    --fixed-workspace "$FIXED_ROOT" \
    --current-workspace "$REPO_DIR"

CURRENT_STAGE="read-only-evaluation-and-viewer"
write_status "running"
assert_resources
"$PY" -m category_priors.baseline_closure_analysis \
    --closure-root "$CLOSURE_ROOT" \
    --gt-dir "$GT_DIR" \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --output-dir "$ARTIFACT_ROOT"
"$PY" -m category_priors.baseline_closure_precision \
    --closure-root "$CLOSURE_ROOT" \
    --gt-dir "$GT_DIR" \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --output-dir "$ARTIFACT_ROOT"

CURRENT_STAGE="complete"
write_status "complete"
trap - EXIT
