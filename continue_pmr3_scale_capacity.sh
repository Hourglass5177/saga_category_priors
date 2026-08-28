#!/usr/bin/env bash
set -Eeuo pipefail

# PMR-3: one GPU, one process at a time, and one fixed sequence.  Training and
# segmentation never receive a GT path; GT is supplied only to the two offline
# evaluation calls.  Existing complete scene/checkpoint outputs are validated
# and reused by category_priors.pmr3_scale_capacity.

usage() {
    cat <<'EOF'
Usage: bash continue_pmr3_scale_capacity.sh [options]

Options (the matching SAGA_PMR3_* environment variable may be used instead):
  --workspace PATH          SAGA_PMR3_WORKSPACE
  --python PATH             SAGA_PMR3_PYTHON
  --runtime-manifest PATH   SAGA_PMR3_RUNTIME_MANIFEST
  --prompts-root PATH       SAGA_PMR3_PROMPTS_ROOT
  --parameters PATH         SAGA_PMR3_PARAMETERS
  --gt-dir PATH             SAGA_PMR3_GT_DIR
  --size-bins PATH          SAGA_PMR3_SIZE_BINS
  --historical-analysis PATH SAGA_PMR3_HISTORICAL_ANALYSIS
  --training-root PATH      SAGA_PMR3_TRAINING_ROOT
  --runs-root PATH          SAGA_PMR3_RUNS_ROOT
  --artifacts-root PATH     SAGA_PMR3_ARTIFACTS_ROOT
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${SAGA_PMR3_WORKSPACE:-$SCRIPT_DIR}"
PYTHON_BIN="${SAGA_PMR3_PYTHON:-/root/autodl-tmp/saga/venvs/category-priors/bin/python}"
RUNTIME_MANIFEST="${SAGA_PMR3_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
PROMPTS_ROOT="${SAGA_PMR3_PROMPTS_ROOT:-/root/autodl-tmp/saga/runs/prompt-prior-minimal-da22c5f/prepared}"
PARAMETERS="${SAGA_PMR3_PARAMETERS:-$PROMPTS_ROOT/prompt_prior_params.json}"
GT_DIR="${SAGA_PMR3_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
SIZE_BINS="${SAGA_PMR3_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
HISTORICAL_ANALYSIS="${SAGA_PMR3_HISTORICAL_ANALYSIS:-/root/autodl-tmp/saga/artifacts/prompt-prior-diagnostics-1f23085/prompt_prior_scale_capacity.json}"
RUNS_ROOT="${SAGA_PMR3_RUNS_ROOT:-/root/autodl-tmp/saga/runs/pmr3-scale-capacity-10k}"
TRAINING_ROOT="${SAGA_PMR3_TRAINING_ROOT:-$RUNS_ROOT/pmr3}"
ARTIFACTS_ROOT="${SAGA_PMR3_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/pmr3-scale-capacity-10k}"

while (($#)); do
    case "$1" in
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --runtime-manifest) RUNTIME_MANIFEST="$2"; shift 2 ;;
        --prompts-root) PROMPTS_ROOT="$2"; shift 2 ;;
        --parameters) PARAMETERS="$2"; shift 2 ;;
        --gt-dir) GT_DIR="$2"; shift 2 ;;
        --size-bins) SIZE_BINS="$2"; shift 2 ;;
        --historical-analysis) HISTORICAL_ANALYSIS="$2"; shift 2 ;;
        --training-root) TRAINING_ROOT="$2"; shift 2 ;;
        --runs-root) RUNS_ROOT="$2"; shift 2 ;;
        --artifacts-root) ARTIFACTS_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

WORKSPACE="$(cd "$WORKSPACE" && pwd)"
mkdir -p "$TRAINING_ROOT" "$RUNS_ROOT/masks" "$ARTIFACTS_ROOT"

STATUS="$ARTIFACTS_ROOT/pmr3_status.json"
LOG="$ARTIFACTS_ROOT/continuation.log"
NATIVE_TABLE="$ARTIFACTS_ROOT/pmr3_scale_capacity_native.parquet"
NATIVE_ANALYSIS="$ARTIFACTS_ROOT/pmr3_scale_capacity_native.json"
TENK_TABLE="$ARTIFACTS_ROOT/pmr3_scale_capacity_10k.parquet"
TENK_ANALYSIS="$ARTIFACTS_ROOT/pmr3_scale_capacity.json"
COMBINED_TABLE="$ARTIFACTS_ROOT/pmr3_scale_capacity.parquet"
PAIR_ANALYSIS="$ARTIFACTS_ROOT/pmr3_analysis.json"
SCENE_ARGS=(--scene scene0591_02 --scene scene0645_00)

# Start the append-only human-readable log before preflight so a bad path or
# interpreter is visible without having to reproduce the launch.
exec > >(tee -a "$LOG") 2>&1

write_bootstrap_failure() {
    local detail="$1"
    local escaped="$detail"
    local temporary="$STATUS.bootstrap.$$"
    escaped="${escaped//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    escaped="${escaped//$'\n'/\\n}"
    printf '%s\n' \
        '{' \
        '  "kind": "pmr3_scale_capacity_status",' \
        '  "status": "failed",' \
        '  "stage": "preflight",' \
        "  \"detail\": \"$escaped\"" \
        '}' >"$temporary"
    mv -f "$temporary" "$STATUS"
}

preflight_die() {
    local detail="$1"
    echo "PMR-3 preflight failed: $detail" >&2
    write_bootstrap_failure "$detail"
    exit 1
}

if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || preflight_die "python is not executable: $PYTHON_BIN"
else
    command -v "$PYTHON_BIN" >/dev/null || preflight_die "python command not found: $PYTHON_BIN"
fi

for required_file in \
    "$WORKSPACE/run_pipeline.sh" \
    "$WORKSPACE/train_contrastive_feature.py" \
    "$WORKSPACE/category_priors/pmr3_scale_capacity.py" \
    "$RUNTIME_MANIFEST" \
    "$PARAMETERS" \
    "$SIZE_BINS" \
    "$HISTORICAL_ANALYSIS" \
    "$PROMPTS_ROOT/prompts/scene0591_02/runtime_prompts.json" \
    "$PROMPTS_ROOT/prompts/scene0591_02/evaluation_prompts.json" \
    "$PROMPTS_ROOT/prompts/scene0645_00/runtime_prompts.json" \
    "$PROMPTS_ROOT/prompts/scene0645_00/evaluation_prompts.json"; do
    [[ -f "$required_file" ]] || preflight_die "required file missing: $required_file"
done
for required_dir in "$GT_DIR"; do
    [[ -d "$required_dir" ]] || preflight_die "required directory missing: $required_dir"
done

COMMIT="$(git -C "$WORKSPACE" rev-parse HEAD)" || \
    preflight_die "cannot resolve git commit in workspace: $WORKSPACE"
export SAGA_EXPERIMENT_COMMIT="$COMMIT"

write_status() {
    local state="$1"
    local stage="$2"
    local detail="$3"
    "$PYTHON_BIN" - "$STATUS" "$state" "$stage" "$detail" "$COMMIT" \
        "$TRAINING_ROOT" "$RUNS_ROOT" "$ARTIFACTS_ROOT" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "kind": "pmr3_scale_capacity_status",
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "detail": sys.argv[4],
    "git_commit": sys.argv[5],
    "training_root": sys.argv[6],
    "runs_root": sys.argv[7],
    "artifacts_root": sys.argv[8],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    "w", encoding="utf-8", dir=target.parent, delete=False
) as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, target)
PY
}

on_error() {
    local code="$?"
    local line="${1:-unknown}"
    trap - ERR INT TERM
    write_status failed failed "runner exited with code $code near line $line; existing .part files were retained" || true
    exit "$code"
}

on_signal() {
    local signal="$1"
    local code="$2"
    trap - ERR INT TERM
    write_status interrupted interrupted "runner received $signal; existing .part files were retained" || true
    exit "$code"
}

trap 'on_error "$LINENO"' ERR
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONHASHSEED=0
export PYTHONPATH="$WORKSPACE/submodules/diff-gaussian-rasterization-max-contributor:$WORKSPACE/submodules/diff-gaussian-rasterization-depth:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"

resource_audit() {
    "$PYTHON_BIN" - "$TRAINING_ROOT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 80.0:
    raise SystemExit(f"PMR-3 requires >=80 GiB free, found {free_gib:.2f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum_text = (cgroup / "memory.max").read_text().strip()
events = dict(
    line.split() for line in (cgroup / "memory.events").read_text().splitlines()
)
print(json.dumps({
    "disk_free_gib": free_gib,
    "memory_current_bytes": current,
    "memory_max": maximum_text,
    "memory_events": events,
}, sort_keys=True))
PY
}

echo "PMR-3 workspace: $WORKSPACE"
echo "PMR-3 commit: $COMMIT"
echo "PMR-3 training root: $TRAINING_ROOT"
echo "PMR-3 runs root: $RUNS_ROOT"
echo "PMR-3 artifacts root: $ARTIFACTS_ROOT"
resource_audit

write_status running train "train or validate the two continuous native-to-10k trajectories"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity train \
    --workspace "$WORKSPACE" \
    --python "$PYTHON_BIN" \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --training-root "$TRAINING_ROOT" \
    "${SCENE_ARGS[@]}"

resource_audit
write_status running segment-native "render U plus the frozen nine-point grid from each native-budget snapshot"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity segment \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --prompts-root "$PROMPTS_ROOT" \
    --parameters "$PARAMETERS" \
    --training-root "$TRAINING_ROOT" \
    --output-root "$RUNS_ROOT/masks" \
    --checkpoint native \
    "${SCENE_ARGS[@]}"

write_status running evaluate-native "evaluate the native-budget checkpoint; GT first enters here"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity evaluate \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --prompts-root "$PROMPTS_ROOT" \
    --gt-dir "$GT_DIR" \
    --training-root "$TRAINING_ROOT" \
    --masks-root "$RUNS_ROOT/masks" \
    --checkpoint native \
    --table-output "$NATIVE_TABLE" \
    --analysis-output "$NATIVE_ANALYSIS" \
    --size-bins "$SIZE_BINS" \
    "${SCENE_ARGS[@]}"

resource_audit
write_status running segment-10k "render U plus the identical frozen grid from each 10k snapshot"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity segment \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --prompts-root "$PROMPTS_ROOT" \
    --parameters "$PARAMETERS" \
    --training-root "$TRAINING_ROOT" \
    --output-root "$RUNS_ROOT/masks" \
    --checkpoint 10k \
    "${SCENE_ARGS[@]}"

write_status running evaluate-10k "evaluate the 10k checkpoint with the same registered objects and grid"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity evaluate \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --prompts-root "$PROMPTS_ROOT" \
    --gt-dir "$GT_DIR" \
    --training-root "$TRAINING_ROOT" \
    --masks-root "$RUNS_ROOT/masks" \
    --checkpoint 10k \
    --table-output "$TENK_TABLE" \
    --analysis-output "$TENK_ANALYSIS" \
    --size-bins "$SIZE_BINS" \
    "${SCENE_ARGS[@]}"

write_status running analyze "merge checkpoint rows and apply the five preregistered PMR-3 gates"
"$PYTHON_BIN" -m category_priors.pmr3_scale_capacity analyze \
    --native-analysis "$NATIVE_ANALYSIS" \
    --tenk-analysis "$TENK_ANALYSIS" \
    --native-table "$NATIVE_TABLE" \
    --tenk-table "$TENK_TABLE" \
    --combined-table "$COMBINED_TABLE" \
    --historical-analysis "$HISTORICAL_ANALYSIS" \
    --output "$PAIR_ANALYSIS"

DECISION="$($PYTHON_BIN - "$PAIR_ANALYSIS" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["decision"])
PY
)"
write_status complete complete "PMR-3 completed: $DECISION"
echo "PMR-3 completed: $DECISION"
