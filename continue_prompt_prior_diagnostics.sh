#!/usr/bin/env bash
set -Eeuo pipefail

# PMR-2: no training, clustering, tracking, or automatic proposals.  This
# runner only audits the completed U/D masks and scans the frozen native SAGA
# scale gate on the exact same prompted objects.

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT="$(git -C "$WORKSPACE" rev-parse HEAD)"
SHORT_COMMIT="${COMMIT:0:7}"
PYTHON="${SAGA_PROMPT_PYTHON:-/root/autodl-tmp/saga/venvs/category-priors/bin/python}"
RUNTIME_MANIFEST="${SAGA_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
GT_DIR="${SAGA_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
SIZE_BINS="${SAGA_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
OLD_RUNS="${SAGA_PROMPT_OLD_RUNS:-/root/autodl-tmp/saga/runs/prompt-prior-minimal-da22c5f}"
OLD_ARTIFACTS="${SAGA_PROMPT_OLD_ARTIFACTS:-/root/autodl-tmp/saga/artifacts/prompt-prior-minimal-da22c5f}"
RUNS_ROOT="${SAGA_PROMPT_DIAGNOSTIC_RUNS_ROOT:-/root/autodl-tmp/saga/runs/prompt-prior-diagnostics-$SHORT_COMMIT}"
ARTIFACTS_ROOT="${SAGA_PROMPT_DIAGNOSTIC_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/prompt-prior-diagnostics-$SHORT_COMMIT}"
PROMPTS_ROOT="$OLD_RUNS/prepared"
OLD_MASKS_ROOT="$OLD_RUNS/masks"
PARAMETERS="$PROMPTS_ROOT/prompt_prior_params.json"
CAPACITY_MASKS="$RUNS_ROOT/scale-capacity"
PLAN="$ARTIFACTS_ROOT/prompt_prior_scale_capacity_plan.json"
STATUS="$ARTIFACTS_ROOT/prompt_prior_diagnostic_status.json"
LOG="$ARTIFACTS_ROOT/continuation.log"

DEV8=(
  scene0645_00 scene0025_01 scene0046_00 scene0474_01
  scene0591_02 scene0329_02 scene0164_03 scene0064_01
)

mkdir -p "$RUNS_ROOT" "$ARTIFACTS_ROOT"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH="$WORKSPACE/submodules/diff-gaussian-rasterization-max-contributor:$WORKSPACE/submodules/diff-gaussian-rasterization-depth:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42
export SAGA_EXPERIMENT_COMMIT="$COMMIT"

write_status() {
  "$PYTHON" - "$STATUS" "$1" "$2" "$COMMIT" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
target = Path(sys.argv[1])
payload = {
    "kind": "prompt_prior_diagnostic_status",
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
  trap - ERR
  write_status failed "runner exited with code $code near line ${1:-unknown}"
  exit "$code"
}
trap 'on_error "$LINENO"' ERR

scene_args() {
  local scene
  for scene in "$@"; do printf '%s\n' --scene "$scene"; done
}

resource_audit() {
  "$PYTHON" - "$RUNS_ROOT" <<'PY'
import json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
free_gib = shutil.disk_usage(root).free / 1024**3
if free_gib < 80.0:
    raise SystemExit(f"PMR-2 requires >=80 GiB free, found {free_gib:.1f}")
cgroup = Path("/sys/fs/cgroup")
current = int((cgroup / "memory.current").read_text().strip())
maximum_text = (cgroup / "memory.max").read_text().strip()
maximum = int(maximum_text) if maximum_text != "max" else None
if maximum != 90 * 1024**3:
    raise SystemExit(f"expected memory.max=90GiB, found {maximum_text}")
events = dict(line.split() for line in (cgroup / "memory.events").read_text().splitlines())
print(json.dumps({"disk_free_gib": free_gib, "memory_current_bytes": current,
                  "memory_max_bytes": maximum, "memory_events": events}, sort_keys=True))
PY
}

contributor_preflight() {
  "$PYTHON" - "$WORKSPACE" <<'PY'
import json, sys
from pathlib import Path
workspace = Path(sys.argv[1]).resolve()
source = workspace / "submodules/diff-gaussian-rasterization-max-contributor/cuda_rasterizer/forward.cu"
text = source.read_text(encoding="utf-8")
required = ("float max_weight = 0.0f;", "int max_id = -1;", "const float weight = alpha * T;")
missing = [token for token in required if token not in text]
if missing:
    raise SystemExit(f"corrected contributor source check failed: {missing}")
from diff_gaussian_rasterization_max_contributor import _C
extension = Path(_C.__file__).resolve()
package = (workspace / "submodules/diff-gaussian-rasterization-max-contributor").resolve()
if package not in extension.parents:
    raise SystemExit(f"loaded extension outside workspace: {extension}")
from diff_gaussian_rasterization_depth import _C as depth_C
depth_extension = Path(depth_C.__file__).resolve()
depth_package = (workspace / "submodules/diff-gaussian-rasterization-depth").resolve()
if depth_package not in depth_extension.parents:
    raise SystemExit(f"loaded depth extension outside workspace: {depth_extension}")
print(json.dumps({"source": str(source), "extension": str(extension),
                  "depth_extension": str(depth_extension)}, sort_keys=True))
PY
}

for required in "$PROMPTS_ROOT/preparation.json" "$PARAMETERS" \
  "$OLD_ARTIFACTS/prompt_prior_paired8.parquet"; do
  [[ -f "$required" ]] || { write_status failed "missing read-only PMR-1 input: $required"; exit 1; }
done

mapfile -t DEV8_ARGS < <(scene_args "${DEV8[@]}")
resource_audit
contributor_preflight

write_status running "PMR-2A: audit whether existing class-scale changes move boundaries helpfully"
"$PYTHON" -m category_priors.prompt_prior_diagnostics audit-directions \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --prompts-root "$PROMPTS_ROOT" \
  --masks-root "$OLD_MASKS_ROOT" \
  --parameters "$PARAMETERS" \
  --table-output "$ARTIFACTS_ROOT/prompt_prior_direction_audit.parquet" \
  --analysis-output "$ARTIFACTS_ROOT/prompt_prior_direction_audit.json" \
  "${DEV8_ARGS[@]}"

write_status running "PMR-2B: materialize historical native visible-mask O-instance oracle scales"
"$PYTHON" -m category_priors.prompt_prior_diagnostics prepare-capacity \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --prompts-root "$PROMPTS_ROOT" \
  --parameters "$PARAMETERS" \
  --output "$PLAN" \
  "${DEV8_ARGS[@]}"

write_status running "PMR-2B: run registered five-point gate-capacity control"
"$PYTHON" -m category_priors.prompt_prior_diagnostics segment-capacity \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --plan "$PLAN" \
  --output-root "$CAPACITY_MASKS" \
  --grid 0 --grid .25 --grid .50 --grid .75 --grid 1 \
  "${DEV8_ARGS[@]}"

"$PYTHON" -m category_priors.prompt_prior_diagnostics evaluate-capacity \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --gt-dir "$GT_DIR" \
  --prompts-root "$PROMPTS_ROOT" \
  --old-masks-root "$OLD_MASKS_ROOT" \
  --capacity-masks-root "$CAPACITY_MASKS" \
  --plan "$PLAN" \
  --grid 0 --grid .25 --grid .50 --grid .75 --grid 1 \
  --table-output "$ARTIFACTS_ROOT/prompt_prior_scale_capacity5.parquet" \
  --analysis-output "$ARTIFACTS_ROOT/prompt_prior_scale_capacity5.json" \
  --size-bins "$SIZE_BINS" \
  "${DEV8_ARGS[@]}"

DECISION="$($PYTHON - "$ARTIFACTS_ROOT/prompt_prior_scale_capacity5.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"

if [[ "$DECISION" == "run-registered-nine-point-supplement" ]]; then
  write_status running "PMR-2B: five-point control lacked capacity; run preregistered four intermediate scales"
  "$PYTHON" -m category_priors.prompt_prior_diagnostics segment-capacity \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --plan "$PLAN" \
    --output-root "$CAPACITY_MASKS" \
    --grid .125 --grid .375 --grid .625 --grid .875 \
    "${DEV8_ARGS[@]}"
  "$PYTHON" -m category_priors.prompt_prior_diagnostics evaluate-capacity \
    --runtime-manifest "$RUNTIME_MANIFEST" \
    --gt-dir "$GT_DIR" \
    --prompts-root "$PROMPTS_ROOT" \
    --old-masks-root "$OLD_MASKS_ROOT" \
    --capacity-masks-root "$CAPACITY_MASKS" \
    --plan "$PLAN" \
    --grid 0 --grid .125 --grid .25 --grid .375 --grid .50 \
    --grid .625 --grid .75 --grid .875 --grid 1 \
    --table-output "$ARTIFACTS_ROOT/prompt_prior_scale_capacity.parquet" \
    --analysis-output "$ARTIFACTS_ROOT/prompt_prior_scale_capacity.json" \
    --size-bins "$SIZE_BINS" \
    "${DEV8_ARGS[@]}"
else
  cp "$ARTIFACTS_ROOT/prompt_prior_scale_capacity5.parquet" \
     "$ARTIFACTS_ROOT/prompt_prior_scale_capacity.parquet"
  cp "$ARTIFACTS_ROOT/prompt_prior_scale_capacity5.json" \
     "$ARTIFACTS_ROOT/prompt_prior_scale_capacity.json"
fi

FINAL_DECISION="$($PYTHON - "$ARTIFACTS_ROOT/prompt_prior_scale_capacity.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"
write_status complete "PMR-2 completed: $FINAL_DECISION"
