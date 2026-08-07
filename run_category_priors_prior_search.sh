#!/usr/bin/env bash

set -euo pipefail

repo_path=""
pipeline_path=""
artifacts_dir=""
output_root=""
cpu_python=""
max_attempts=3

err() { echo "Error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-path) repo_path="$2"; shift 2 ;;
        --pipeline) pipeline_path="$2"; shift 2 ;;
        --artifacts-dir) artifacts_dir="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --cpu-python) cpu_python="$2"; shift 2 ;;
        --max-attempts) max_attempts="$2"; shift 2 ;;
        *) err "unknown argument: $1" ;;
    esac
done

[[ -d "$repo_path" ]] || err "repository not found: $repo_path"
[[ -s "$pipeline_path" ]] || err "pipeline not found: $pipeline_path"
[[ -d "$artifacts_dir" ]] || err "artifacts directory not found: $artifacts_dir"
[[ -x "$cpu_python" ]] || err "Python executable not found: $cpu_python"
[[ "$max_attempts" =~ ^[1-9][0-9]*$ ]] || err "--max-attempts must be positive"

schedule="$artifacts_dir/prior_tune_schedule.json"
scene_manifest="$artifacts_dir/scene_runtime_manifest.json"
priors="$artifacts_dir/category_priors.json"
execution="$output_root/execution.json"
metrics="$artifacts_dir/prior_tune_metrics.parquet"
best="$artifacts_dir/prior_best.json"
evaluation_dir="$artifacts_dir/prior_evaluation"
status_path="$artifacts_dir/prior_search_supervisor.status"

for required in "$schedule" "$scene_manifest" "$priors"; do
    [[ -s "$required" ]] || err "required input missing: $required"
done
mkdir -p "$output_root" "$evaluation_dir"
export PYTHONPATH="$repo_path${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_path"

attempt=0
while true; do
    attempt=$((attempt + 1))
    printf 'running stage=prior-search attempt=%s\n' "$attempt" > "$status_path"
    "$cpu_python" -m category_priors run-experiment \
        --schedule "$schedule" \
        --scene-manifest "$scene_manifest" \
        --output-root "$output_root" \
        --output "$execution" \
        --pipeline "$pipeline_path" \
        --priors "$priors" \
        --continue-on-error

    if "$cpu_python" - "$execution" <<'PY'
import json
import sys
from pathlib import Path

runs = json.load(open(sys.argv[1], encoding="utf-8")).get("runs", [])
complete = len(runs) == 768
for run in runs:
    complete &= run.get("status") in {"complete", "skipped_complete"}
    complete &= Path(run["output_json"]).is_file()
    complete &= Path(run["metadata_json"]).is_file()
raise SystemExit(0 if complete else 1)
PY
    then
        break
    fi
    if (( attempt >= max_attempts )); then
        printf 'blocked stage=prior-search attempts=%s\n' "$attempt" > "$status_path"
        exit 2
    fi
done

printf 'running stage=prior-evaluation attempts=%s\n' "$attempt" > "$status_path"
"$cpu_python" -m category_priors evaluate-search \
    --schedule "$schedule" \
    --execution "$execution" \
    --scene-manifest "$scene_manifest" \
    --gt-manifest "$artifacts_dir/gt_val_tune/manifest.json" \
    --output-dir "$evaluation_dir" \
    --output "$metrics"
"$cpu_python" -m category_priors select-config \
    --design "$artifacts_dir/prior_search_design.json" \
    --metrics "$metrics" \
    --output "$best"
"$cpu_python" -m category_priors build-mapping \
    --global-best "$artifacts_dir/global_best.json" \
    --prior-best "$best" \
    --priors "$priors" \
    --taxonomy "$repo_path/category_priors/default_taxonomy.json" \
    --scene-selection "$artifacts_dir/scene_selection.json" \
    --output "$artifacts_dir/prior_mapping_config.json"
"$cpu_python" -m category_priors validate \
    --priors "$priors" \
    --mapping "$artifacts_dir/prior_mapping_config.json"

[[ -s "$metrics" && -s "$best" && -s "$artifacts_dir/prior_mapping_config.json" ]] \
    || err "prior search final artifacts are incomplete"
printf 'complete stage=prior-selection attempts=%s\n' "$attempt" > "$status_path"
