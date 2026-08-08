#!/usr/bin/env bash

set -euo pipefail

scene_list=""
download_manifest=""
dataset_root=""
output_root=""
artifacts_dir=""
gt_dir=""
clean_workspace=""
gpu_repo=""
cpu_python=""
gpu_python=""
hf_home=""
wave_name="wave"
limit=8
iterations=30000
frame_stride=20
max_frames=200
max_initial_points=200000
min_free_gb=80
download_wait_hours=12
delete_sens_after_success=0

usage() {
    cat <<'EOF'
Usage:
  bash run_scannet_tune_wave.sh [required options] [options]

Required:
  --scene-list PATH
  --dataset-root PATH
  --output-root PATH
  --artifacts-dir PATH
  --gt-dir PATH
  --clean-workspace PATH
  --gpu-repo PATH
  --cpu-python PATH
  --gpu-python PATH
  --hf-home PATH

Options:
  --download-manifest PATH     Optional downloader status used only for waiting
  --wave-name NAME             Default: wave
  --limit INT                  Default: 8
  --iterations INT             Default: 30000
  --frame-stride INT           Default: 20
  --max-frames INT             Default: 200
  --max-initial-points INT     Default: 200000
  --min-free-gb INT            Default: 80
  --download-wait-hours INT    Default: 12
  --delete-sens-after-success  Remove this wave's raw .sens only after every scene succeeds

The script accepts official nonempty, readable .sens files, then prepares and
trains each scene sequentially. Existing nonempty stage outputs are retained and
consumed by the per-scene resume gates.
EOF
}

err() {
    echo "Error: $*" >&2
    return 1
}

require_file() {
    [[ -s "$1" ]] || err "$2 is missing or empty: $1"
}

require_dir() {
    [[ -d "$1" ]] || err "$2 directory is missing: $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene-list) scene_list="$2"; shift 2 ;;
        --download-manifest) download_manifest="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --artifacts-dir) artifacts_dir="$2"; shift 2 ;;
        --gt-dir) gt_dir="$2"; shift 2 ;;
        --clean-workspace) clean_workspace="$2"; shift 2 ;;
        --gpu-repo) gpu_repo="$2"; shift 2 ;;
        --cpu-python) cpu_python="$2"; shift 2 ;;
        --gpu-python) gpu_python="$2"; shift 2 ;;
        --hf-home) hf_home="$2"; shift 2 ;;
        --wave-name) wave_name="$2"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --frame-stride) frame_stride="$2"; shift 2 ;;
        --max-frames) max_frames="$2"; shift 2 ;;
        --max-initial-points) max_initial_points="$2"; shift 2 ;;
        --min-free-gb) min_free_gb="$2"; shift 2 ;;
        --download-wait-hours) download_wait_hours="$2"; shift 2 ;;
        --delete-sens-after-success) delete_sens_after_success=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1" ;;
    esac
done

for value in scene_list dataset_root output_root artifacts_dir gt_dir \
    clean_workspace gpu_repo cpu_python gpu_python hf_home; do
    [[ -n "${!value}" ]] || err "--${value//_/-} is required"
done
for value in limit iterations frame_stride max_frames max_initial_points \
    min_free_gb download_wait_hours; do
    [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || err "--${value//_/-} must be positive"
done
[[ "$wave_name" =~ ^[A-Za-z0-9._-]+$ ]] || err "--wave-name contains unsafe characters"

require_file "$scene_list" "Scene list"
require_file "$cpu_python" "CPU Python"
require_file "$gpu_python" "GPU Python"
require_file "$gpu_repo/run_scannet_scene_pipeline.sh" "Per-scene pipeline"
require_dir "$dataset_root" "ScanNet scans"
require_dir "$gt_dir" "Canonical ground truth"
require_dir "$clean_workspace" "Clean workspace"
require_dir "$hf_home" "Hugging Face cache"
mkdir -p "$output_root" "$artifacts_dir/alignment"

status_path="$artifacts_dir/scannet_${wave_name}_supervisor.status"
log_path="$artifacts_dir/scannet_${wave_name}_supervisor.log"
stage="waiting_for_download"
scene_id="none"
completed=0

exec >"$log_path" 2>&1

write_status() {
    printf 'running stage=%s scene=%s completed=%s/%s updated_at=%s\n' \
        "$stage" "$scene_id" "$completed" "$limit" "$(date -Is)" > "$status_path"
}

on_error() {
    local rc=$?
    printf 'failed stage=%s scene=%s completed=%s/%s return_code=%s failed_at=%s\n' \
        "$stage" "$scene_id" "$completed" "$limit" "$rc" "$(date -Is)" \
        > "$status_path"
    exit "$rc"
}
trap on_error ERR

mapfile -t scenes < <(sed '/^[[:space:]]*$/d' "$scene_list" | head -n "$limit")
[[ "${#scenes[@]}" -eq "$limit" ]] \
    || err "scene list has fewer than $limit nonempty rows"
for scene_id in "${scenes[@]}"; do
    [[ "$scene_id" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] \
        || err "invalid ScanNet scene id: $scene_id"
done
scene_id="none"

write_status
if [[ -n "$download_manifest" ]]; then
    wait_deadline=$(( $(date +%s) + download_wait_hours * 3600 ))
    while [[ ! -s "$download_manifest" ]]; do
        (( $(date +%s) < wait_deadline )) || err "timed out waiting for download manifest"
        sleep 30
    done
fi

PYTHONPATH="$clean_workspace" "$cpu_python" - \
    "$download_manifest" "$dataset_root" "${scenes[@]}" <<'PY'
import sys
from pathlib import Path

from category_priors.io import load_json
from category_priors.scannet_saga import read_sens_header

manifest_arg = sys.argv[1]
dataset_root = Path(sys.argv[2])
scene_ids = sys.argv[3:]
records = None
if manifest_arg:
    manifest = load_json(manifest_arg)
    records = {item["scene_id"]: item for item in manifest.get("files", [])}
for scene_id in scene_ids:
    if records is not None:
        record = records.get(scene_id)
        if not record or record.get("status") not in {"existing", "downloaded"}:
            raise SystemExit(f"download manifest is incomplete for {scene_id}")
    sens = dataset_root / scene_id / f"{scene_id}.sens"
    if not sens.is_file() or sens.stat().st_size <= 0:
        raise SystemExit(f"downloaded .sens is missing or empty: {sens}")
    with sens.open("rb") as handle:
        header = read_sens_header(handle)
    if header.num_frames <= 0:
        raise SystemExit(f"downloaded .sens has no frames: {sens}")
PY

free_space_gate() {
    local available_kb required_kb
    available_kb="$(df -Pk "$output_root" | awk 'NR==2 {print $4}')"
    required_kb=$(( min_free_gb * 1024 * 1024 ))
    (( available_kb >= required_kb )) \
        || err "free-space gate failed: ${available_kb}KB available"
}

export HF_HOME="$hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

completed=0
for scene_id in "${scenes[@]}"; do
    base="$output_root/$scene_id"
    sens="$dataset_root/$scene_id/$scene_id.sens"
    gt="$gt_dir/$scene_id.npz"
    preparation_manifest="$base/scene_preparation_manifest.json"
    trained_audit="$artifacts_dir/alignment/$scene_id-trained-alignment.json"

    free_space_gate
    require_file "$sens" "ScanNet .sens"
    require_file "$gt" "Validation ground truth"

    stage="scene_preparation"
    write_status
    if [[ ! -s "$preparation_manifest" \
          || ! -s "$base/fastRecon/dense/sparse/0/cameras.txt" \
          || ! -s "$base/fastRecon/dense/sparse/0/images.txt" \
          || ! -s "$base/fastRecon/dense/sparse/0/points3D.ply" ]]; then
        cd "$clean_workspace"
        PYTHONPATH=. "$cpu_python" -m category_priors prepare-saga-scene \
            --dataset-root "$dataset_root" \
            --scene-id "$scene_id" \
            --sens "$sens" \
            --output-root "$output_root" \
            --frame-stride "$frame_stride" \
            --max-frames "$max_frames" \
            --max-initial-points "$max_initial_points"
    fi

    stage="scene_pipeline"
    write_status
    bash "$gpu_repo/run_scannet_scene_pipeline.sh" \
        --repo-path "$gpu_repo" \
        --base-path "$base" \
        --python "$gpu_python" \
        --hf-home "$hf_home" \
        --stage all \
        --iterations "$iterations" \
        --sh-degree 0

    stage="trained_alignment_audit"
    write_status
    gaussian="$base/output_models/point_cloud/iteration_${iterations}/point_cloud.ply"
    if [[ ! -s "$gaussian" ]]; then
        gaussian="$base/output_models/point_cloud/iteration_${iterations}/scene_point_cloud.ply"
    fi
    require_file "$gaussian" "Trained Gaussian point cloud"
    cd "$clean_workspace"
    PYTHONPATH=. "$cpu_python" -m category_priors audit-saga-alignment \
        --preparation-manifest "$preparation_manifest" \
        --gt-npz "$gt" \
        --gaussian-ply "$gaussian" \
        --output "$trained_audit" \
        --minimal

    completed=$((completed + 1))
done

if (( delete_sens_after_success )); then
    stage="delete_wave_sens"
    scene_id="none"
    write_status
    for completed_scene_id in "${scenes[@]}"; do
        rm -- "$dataset_root/$completed_scene_id/$completed_scene_id.sens"
    done
fi

stage="complete"
scene_id="none"
printf 'complete stage=%s completed=%s/%s completed_at=%s\n' \
    "$stage" "$completed" "$limit" "$(date -Is)" > "$status_path"
echo "ScanNet tune wave complete: $completed/$limit scenes"
