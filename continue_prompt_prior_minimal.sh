#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal prompt-conditioned category-size experiment.  This runner does no
# training, downloading, clustering, tracking, or automatic instance proposal.

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="$(git -C "$WORKSPACE" rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:7}"
PYTHON="${SAGA_PROMPT_PYTHON:-/root/autodl-tmp/saga/venvs/category-priors/bin/python}"
RUNTIME_MANIFEST="${SAGA_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
GT_DIR="${SAGA_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
CATEGORY_PRIORS="${SAGA_CATEGORY_PRIORS:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/category_priors.json}"
SIZE_BINS="${SAGA_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
V10_BANK_ROOT="${SAGA_V10_BANK_ROOT:-/root/autodl-tmp/saga/runs/v10-view-consensus/banks}"
RUNS_ROOT="${SAGA_PROMPT_RUNS_ROOT:-/root/autodl-tmp/saga/runs/prompt-prior-minimal-$SHORT_COMMIT}"
ARTIFACTS_ROOT="${SAGA_PROMPT_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/prompt-prior-minimal-$SHORT_COMMIT}"
PROMPTS_ROOT="$RUNS_ROOT/prepared"
MASKS_ROOT="$RUNS_ROOT/masks"
STATUS="$ARTIFACTS_ROOT/prompt_prior_status.json"
LOG="$ARTIFACTS_ROOT/continuation.log"

DEV2=(scene0645_00 scene0025_01)
DEV8=(
  scene0645_00 scene0025_01 scene0046_00 scene0474_01
  scene0591_02 scene0329_02 scene0164_03 scene0064_01
)

mkdir -p "$RUNS_ROOT" "$ARTIFACTS_ROOT"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$WORKSPACE/submodules/diff-gaussian-rasterization-max-contributor:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42

write_status() {
  "$PYTHON" - "$STATUS" "$1" "$2" "$COMMIT" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

target = Path(sys.argv[1])
payload = {
    "kind": "prompt_prior_status",
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "git_commit": sys.argv[4],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
target.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, target)
PY
}

on_error() {
  local code="$?"
  local line="${1:-unknown}"
  trap - ERR
  write_status failed "runner exited with code $code near line $line"
  exit "$code"
}
trap 'on_error "$LINENO"' ERR

scene_args() {
  local scene
  for scene in "$@"; do
    printf '%s\n' --scene "$scene"
  done
}

resource_audit() {
  "$PYTHON" - "$RUNS_ROOT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 80.0:
    raise SystemExit(f"prompt experiment requires >=80 GiB free, found {free_gib:.1f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum_text = (cgroup / "memory.max").read_text().strip()
maximum = int(maximum_text) if maximum_text != "max" else None
if maximum != 90 * 1024**3:
    raise SystemExit(f"expected memory.max=90GiB, found {maximum_text}")
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
}

contributor_preflight() {
  "$PYTHON" - "$WORKSPACE" <<'PY'
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1]).resolve()
source = workspace / "submodules/diff-gaussian-rasterization-max-contributor/cuda_rasterizer/forward.cu"
text = source.read_text(encoding="utf-8")
required = (
    "float max_weight = 0.0f;",
    "int max_id = -1;",
    "const float weight = alpha * T;",
)
missing = [token for token in required if token not in text]
if missing:
    raise SystemExit(f"corrected contributor source check failed: {missing}")

from diff_gaussian_rasterization_max_contributor import _C

extension = Path(_C.__file__).resolve()
package_root = (
    workspace
    / "submodules/diff-gaussian-rasterization-max-contributor"
).resolve()
if package_root not in extension.parents:
    raise SystemExit(
        f"loaded contributor extension is not from this workspace: {extension}"
    )
print(json.dumps({
    "corrected_contributor_source": str(source),
    "loaded_extension": str(extension),
}, sort_keys=True))
PY
}

resource_audit
contributor_preflight
write_status running "Stage A: V10 metric closeout"

if [[ -d "$V10_BANK_ROOT/VC1/scene0645_00" && -d "$V10_BANK_ROOT/VC1/scene0025_01" ]]; then
  "$PYTHON" -m category_priors.v10_bank_reassessment \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --gt-dir "$GT_DIR" \
    --bank-root "$V10_BANK_ROOT" \
    --scene scene0645_00 \
    --scene scene0025_01 \
    --condition VC1 \
    --classifier mv-label \
    --legacy-analysis /root/autodl-tmp/saga/artifacts/v10-view-consensus/v10_view_consensus2.json \
    --rows-output "$ARTIFACTS_ROOT/v10_metric_closeout.parquet" \
    --analysis-output "$ARTIFACTS_ROOT/v10_metric_closeout.json"
else
  write_status failed "required read-only V10 VC1 banks are missing"
  exit 1
fi

write_status running "Stage B: prepare two-scene frozen prompts"
mapfile -t DEV2_ARGS < <(scene_args "${DEV2[@]}")
"$PYTHON" -m category_priors.prompt_prior_experiment prepare \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --category-priors "$CATEGORY_PRIORS" \
  --output-root "$PROMPTS_ROOT" \
  "${DEV2_ARGS[@]}"

write_status running "Stage B: run two-scene native SAGA prompt pairs"
"$PYTHON" -m category_priors.prompt_prior_experiment segment \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --prompts-root "$PROMPTS_ROOT" \
  --parameters "$PROMPTS_ROOT/prompt_prior_params.json" \
  --output-root "$MASKS_ROOT" \
  --mechanical-only \
  "${DEV2_ARGS[@]}"

"$PYTHON" -m category_priors.prompt_prior_experiment evaluate \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --prompts-root "$PROMPTS_ROOT" \
  --masks-root "$MASKS_ROOT" \
  --table-output "$ARTIFACTS_ROOT/prompt_prior_mechanical2.parquet" \
  --analysis-output "$ARTIFACTS_ROOT/prompt_prior_mechanical2.json" \
  --size-bins "$SIZE_BINS" \
  --viewer-root "$ARTIFACTS_ROOT/viewer" \
  --mechanical-only \
  "${DEV2_ARGS[@]}"

MECHANICAL_PASSED="$($PYTHON - "$ARTIFACTS_ROOT/prompt_prior_mechanical2.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = bool(json.load(handle).get("mechanical_passed"))
print("true" if value else "false")
PY
)"
if [[ "$MECHANICAL_PASSED" != "true" ]]; then
  write_status stopped "Stage B mechanical intervention gate did not pass"
  exit 0
fi

write_status running "Stage C: prepare eight-scene paired prompt experiment"
mapfile -t DEV8_ARGS < <(scene_args "${DEV8[@]}")
"$PYTHON" -m category_priors.prompt_prior_experiment prepare \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --category-priors "$CATEGORY_PRIORS" \
  --output-root "$PROMPTS_ROOT" \
  "${DEV8_ARGS[@]}"

"$PYTHON" -m category_priors.prompt_prior_experiment segment \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --prompts-root "$PROMPTS_ROOT" \
  --parameters "$PROMPTS_ROOT/prompt_prior_params.json" \
  --output-root "$MASKS_ROOT" \
  "${DEV8_ARGS[@]}"

"$PYTHON" -m category_priors.prompt_prior_experiment evaluate \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --prompts-root "$PROMPTS_ROOT" \
  --masks-root "$MASKS_ROOT" \
  --table-output "$ARTIFACTS_ROOT/prompt_prior_paired8.parquet" \
  --analysis-output "$ARTIFACTS_ROOT/prompt_prior_analysis.json" \
  --size-bins "$SIZE_BINS" \
  --viewer-root "$ARTIFACTS_ROOT/viewer" \
  "${DEV8_ARGS[@]}"

MECHANISM_PASSED="$($PYTHON - "$ARTIFACTS_ROOT/prompt_prior_analysis.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = bool(json.load(handle).get("mechanism_passed"))
print("true" if value else "false")
PY
)"
if [[ "$MECHANISM_PASSED" == "true" ]]; then
  write_status complete "native SAGA class-size mechanism passed the registered eight-scene gate"
else
  write_status complete "native SAGA class-size mechanism did not pass the registered eight-scene gate"
fi
