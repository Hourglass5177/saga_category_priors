#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_path="$SCRIPT_DIR"
python_bin=""
base_path=""
stage="all"
iterations=30000
sh_degree=0
feature_dim=32
downsample=1
num_sampled_rays=1000
hf_home="${HF_HOME:-}"

usage() {
    cat <<'EOF'
Usage:
  bash run_scannet_scene_pipeline.sh --base-path PATH --python PYTHON [options]

Stages:
  train       Train the metric ScanNet 3DGS model.
  masks       Run Grounded-SAM mask/label extraction.
  scale       Estimate per-mask 3D scales.
  feature     Train contrastive features and scale gate.
  all         Run train -> masks -> scale -> feature with resume gates.

Options:
  --repo-path PATH
  --iterations INT          Default: 30000
  --sh-degree INT           Default: 0
  --feature-dim INT         Default: 32
  --downsample INT          Default: 1
  --num-sampled-rays INT    Default: 1000
  --hf-home PATH            Use an existing Hugging Face cache in offline mode.

The script never passes --clean and never deletes prior artifacts. Completed
stages are skipped only after their registered nonempty-output gate passes.
EOF
}

err() {
    echo "Error: $*" >&2
    exit 1
}

require_file() {
    [[ -s "$1" ]] || err "$2 is missing or empty: $1"
}

require_dir() {
    [[ -d "$1" ]] || err "$2 directory is missing: $1"
}

archive_log() {
    local log_path="$1"
    if [[ -s "$log_path" ]]; then
        mv "$log_path" "${log_path}.attempt-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
}

run_logged() {
    local stage_name="$1"
    shift
    local log_path="${log_dir}/${stage_name}.log"
    local status_path="${log_dir}/${stage_name}.status"
    archive_log "$log_path"
    echo "running" > "$status_path"
    local started
    started="$(date +%s)"
    if "$@" > "$log_path" 2>&1; then
        local elapsed=$(( $(date +%s) - started ))
        printf 'complete elapsed_seconds=%s\n' "$elapsed" > "$status_path"
    else
        local rc=$?
        local elapsed=$(( $(date +%s) - started ))
        printf 'failed return_code=%s elapsed_seconds=%s\n' "$rc" "$elapsed" > "$status_path"
        return "$rc"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-path) base_path="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --repo-path) repo_path="$2"; shift 2 ;;
        --stage) stage="$2"; shift 2 ;;
        --iterations) iterations="$2"; shift 2 ;;
        --sh-degree) sh_degree="$2"; shift 2 ;;
        --feature-dim) feature_dim="$2"; shift 2 ;;
        --downsample) downsample="$2"; shift 2 ;;
        --num-sampled-rays) num_sampled_rays="$2"; shift 2 ;;
        --hf-home) hf_home="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1" ;;
    esac
done

[[ -n "$base_path" ]] || err "--base-path is required"
[[ -n "$python_bin" ]] || err "--python is required"
require_file "$python_bin" "Python executable"
require_file "${repo_path}/train_scene.py" "3DGS training script"
require_file "${repo_path}/run_pipeline.sh" "SAGA pipeline script"
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || err "--iterations must be positive"

base_path="$(cd "$(dirname "$base_path")" && pwd)/$(basename "$base_path")"
repo_path="$(cd "$repo_path" && pwd)"
sparse_path="${base_path}/fastRecon/dense/sparse/0"
images_path="${sparse_path}/images"
model_path="${base_path}/output_models"
saga_path="${base_path}/saga"
log_dir="${base_path}/logs"
mkdir -p "$model_path" "$saga_path" "$log_dir"
require_dir "$sparse_path" "COLMAP sparse model"
require_dir "$images_path" "RGB images"
require_file "${sparse_path}/cameras.txt" "COLMAP cameras"
require_file "${sparse_path}/images.txt" "COLMAP images"
require_file "${sparse_path}/points3D.ply" "Initial point cloud"

if [[ -n "$hf_home" ]]; then
    require_dir "$hf_home" "Hugging Face cache"
    export HF_HOME="$hf_home"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi

export PATH="$(dirname "$python_bin"):${PATH}"

trained_point_cloud() {
    local preferred="${model_path}/point_cloud/iteration_${iterations}/point_cloud.ply"
    local fallback="${model_path}/point_cloud/iteration_${iterations}/scene_point_cloud.ply"
    if [[ -s "$preferred" ]]; then
        printf '%s\n' "$preferred"
    elif [[ -s "$fallback" ]]; then
        printf '%s\n' "$fallback"
    else
        return 1
    fi
}

run_train() {
    if trained_point_cloud >/dev/null; then
        echo "skip train: registered point cloud already complete"
        return
    fi
    run_logged train "$python_bin" "${repo_path}/train_scene.py" \
        --source_path "$sparse_path" \
        --model_path "$model_path" \
        --sparse_path "$sparse_path" \
        --images_path "$images_path" \
        --iterations "$iterations" \
        --save_iterations "$iterations" \
        --test_iterations "$iterations" \
        --sh_degree "$sh_degree"
    trained_point_cloud >/dev/null || err "3DGS training output gate failed"
}

run_saga_stage() {
    local log_name="$1"
    local pipeline_stage="$2"
    shift 2
    run_logged "$log_name" bash "${repo_path}/run_pipeline.sh" \
        --python "$python_bin" \
        --base-path "$base_path" \
        --stage "$pipeline_stage" \
        --sh-degree "$sh_degree" \
        --feature-dim "$feature_dim" \
        --downsample "$downsample" \
        --num-sampled-rays "$num_sampled_rays" \
        "$@"
}

run_masks() {
    if [[ -s "${saga_path}/labels/label_features.pt" ]] \
        && find "${saga_path}/masks" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .; then
        echo "skip masks: registered outputs already complete"
        return
    fi
    run_saga_stage masks masks
    require_file "${saga_path}/labels/label_features.pt" "Label features"
    find "${saga_path}/masks" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q . \
        || err "Mask output gate failed"
}

run_scale() {
    if find "${saga_path}/mask_scales" -maxdepth 1 -type f -name '*.pt' -print -quit 2>/dev/null | grep -q .; then
        echo "skip scale: registered outputs already complete"
        return
    fi
    run_saga_stage scale scale
    find "${saga_path}/mask_scales" -maxdepth 1 -type f -name '*.pt' -print -quit | grep -q . \
        || err "Mask-scale output gate failed"
}

run_feature() {
    if [[ -s "${saga_path}/contrastive_feature_point_cloud.ply" ]] \
        && [[ -s "${saga_path}/scale_gate.pt" ]]; then
        echo "skip feature: registered outputs already complete"
        return
    fi
    run_saga_stage feature train
    require_file "${saga_path}/contrastive_feature_point_cloud.ply" "Contrastive feature point cloud"
    require_file "${saga_path}/scale_gate.pt" "Scale gate"
}

case "$stage" in
    train) run_train ;;
    masks) run_masks ;;
    scale) run_scale ;;
    feature) run_feature ;;
    all)
        run_train
        run_masks
        run_scale
        run_feature
        ;;
    *) err "unsupported stage: $stage" ;;
esac

echo "scene pipeline stage complete: $stage"
