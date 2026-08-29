#!/usr/bin/env bash

set -euo pipefail

# This entrypoint is deployed directly to Linux; keep it LF-normalized.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin=""

stage="all"
base_path=""

images_path=""
sparse_path=""
point_cloud_path=""

masks_path=""
labels_path=""
label_features_path=""
mask_scales_path=""

contrastive_feature_point_cloud_path=""
scale_gate_path=""
json_path=""
progress_path=""
render_path=""

prior_config=""
prior_mapping_config=""
prior_mode="off"
prior_gate="on"
prior_shrink="on"
prior_metadata_path=""
max_contributor_cache_path=""
scene_scale_m_per_unit="0"
seed=42
disable_other_classes=0
minimal_metadata=0
clustering_mode="legacy"
class_prior_mode="uniform"
category_priors=""
class_first_config=""
legacy_prior_config=""
legacy_prior_mode="uniform"
legacy_prior_score="unit"
legacy_prior_semantic_source="gaussian"
teacher_prior_mode="original"
teacher_category_params=""
teacher_evidence_protection="off"
v3_shadow_mode="off"
v3_shadow_output=""
v3_branch_labels_output=""
v3_shadow_git_commit=""
v3_shadow_scene_id=""
v4_candidate_mode="off"
v4_candidate_output=""
v4_candidate_labels_output=""
v4_git_commit=""
v4_scene_id=""
v5_candidate_source="off"
v5_candidate_output=""
v5_candidate_labels_output=""
v5_git_commit=""
v5_scene_id=""
v6_candidate_mode="off"
v6_candidate_output=""
v6_candidate_labels_output=""
v6_git_commit=""
v6_scene_id=""
v7_causal_ablation="L0"
category_denoise_action="off"
category_denoise_bank_path=""
category_candidate_trace_path=""
category_candidate_sample_cap=5000
category_candidate_score_threshold="0.20"
category_cluster_audit_path=""
category_cluster_conditions=()
category_cluster_verify_determinism=0
category_denoise_mode="uniform"
category_denoise_scene_id=""

sam_checkpoint_path="${SCRIPT_DIR}/weights/sam_vit_h_4b8939.pth"
groundingdino_checkpoint_path="${SCRIPT_DIR}/weights/groundingdino_swint_ogc.pth"
groundingdino_config_path="${SCRIPT_DIR}/weights/GroundingDINO_SwinT_OGC.py"

sh_degree=0
feature_dim=32
downsample=1
num_sampled_rays=1000
feature_iterations=0
feature_snapshot_root=""
feature_seed=0

usage() {
    cat <<EOF
Usage:
  bash run_pipeline.sh --base-path PATH [options]

Stages:
  all          Run masks -> scale -> train -> postprocess (default)
  masks        Run grounded_SAM_masks.py only
  scale        Run get_scale.py only
  train        Run train_contrastive_feature.py only
  postprocess  Run postprocess.py only
  render       Run render_instance.py only
  gui          Run saga_gui.py only

Required:
  --base-path PATH

Core options:
  --python PATH
  --stage STAGE
  --images-path PATH
  --sparse-path PATH
  --point-cloud-path PATH
  --masks-path PATH
  --labels-path PATH
  --label-features-path PATH
  --mask-scales-path PATH
  --contrastive-feature-point-cloud-path PATH
  --scale-gate-path PATH
  --json-path PATH
  --progress-path PATH
  --render-path PATH

Category-prior postprocess options:
  --prior-config PATH
  --prior-mapping-config PATH
  --prior-mode MODE              off|global|size|smooth|small|size-smooth|size-small|smooth-small|combined
  --prior-gate MODE              on|off (default: on)
  --prior-shrink MODE            on|off (default: on)
  --prior-metadata-path PATH     Default: BASE/saga/output.json.metadata.json
  --max-contributor-cache-path PATH  Shared cache for config-invariant renders
  --scene-scale-m-per-unit FLOAT Required and positive when priors are enabled
  --seed INT                     Default: 42
  --disable-other-classes        Registered B0; default is B1-compatible enabled
  --minimal-metadata             Omit per-artifact hashes in locked-run metadata

Class-first postprocess options:
  --clustering-mode MODE         legacy|class-first|legacy-prior
  --class-prior-mode MODE        uniform|size|smooth|small|combined
  --category-priors PATH
  --class-first-config PATH
  --legacy-prior-config PATH
  --legacy-prior-mode MODE       uniform|size|smooth|small|combined
  --legacy-prior-score MODE      unit|vote|assignment
  --legacy-prior-semantic-source MODE  gaussian|vote

Teacher-prior postprocess options:
  --teacher-prior-mode MODE      off|original|all-uniform|size|smooth|small|combined
  --teacher-category-params PATH Shared train-only category statistics/parameters
  --teacher-evidence-protection MODE  off|multi-anchor

V3 shadow audit options:
  --v3-shadow-mode MODE          off|exact|exclusive|both
  --v3-shadow-output PATH
  --v3-branch-labels-output PATH
  --v3-shadow-git-commit COMMIT
  --v3-shadow-scene-id SCENE

V4 candidate shadow options:
  --v4-candidate-mode MODE       off|uniform|class-scale|class-core|combined
  --v4-candidate-output PATH
  --v4-candidate-labels-output PATH
  --v4-git-commit COMMIT
  --v4-scene-id SCENE

V5 proposal-bank options:
  --v5-candidate-source SOURCE  off|codebook|multiview
  --v5-candidate-output PATH
  --v5-candidate-labels-output PATH
  --v5-git-commit COMMIT
  --v5-scene-id SCENE

V6 affinity-first proposal-bank options:
  --v6-candidate-mode MODE      off|affinity-first
  --v6-candidate-output PATH
  --v6-candidate-labels-output PATH
  --v6-git-commit COMMIT
  --v6-scene-id SCENE

All-category denoising options:
  --category-denoise-action MODE  off|bank|candidate-repair|cluster-bank|replay|candidate-replay
  --category-denoise-bank-path PATH
  --category-candidate-trace-path PATH
  --category-candidate-sample-cap INT  Default: 5000; only 10000 in the registered nested-sampling control
  --category-candidate-score-threshold FLOAT  Frozen DEV2 U threshold for candidate-replay
  --category-cluster-audit-path PATH  Required raw-label identity sidecar for cluster-bank
  --category-cluster-condition MODE  Repeatable: R0-legacy|
                                     R1-corrected-distance-legacy-expand|
                                     R2-corrected-distance-anchored-expand|
                                     G1-mutual-local-graph
  --category-cluster-verify-determinism  Rebuild and compare the full family (DEV2 only)
  --category-denoise-mode MODE    uniform|class
  --category-denoise-scene-id SCENE

Model options:
  --sam-checkpoint-path PATH
  --groundingdino-checkpoint-path PATH
  --groundingdino-config-path PATH

Tunables:
  --sh-degree INT          Default: 0
  --feature-dim INT        Default: 32
  --downsample INT         Default: 1
  --num-sampled-rays INT   Default: 1000
  --feature-iterations INT Default: 0 (adaptive: min(10 * cameras, 10000))
  --feature-snapshot-root PATH Save the native-budget checkpoint during a longer feature run
  --feature-seed INT       Default: 0 (the historical trainer default)
  -h, --help               Show this help message

Examples:
  bash run_pipeline.sh --base-path data/temp/suzongbangongshi
  bash run_pipeline.sh --base-path data/temp/suzongbangongshi --stage train
  bash run_pipeline.sh --base-path data/temp/suzongbangongshi --stage render
  bash run_pipeline.sh --base-path data/temp/suzongbangongshi --stage gui
EOF
}

err() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || err "Required command not found: $1"
}

find_python() {
    if command -v python >/dev/null 2>&1; then
        python_bin="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin="$(command -v python3)"
    else
        err "Neither 'python' nor 'python3' is available"
    fi
}

require_file() {
    [[ -f "$1" ]] || err "$2 not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || err "$2 not found: $1"
}

ensure_dir() {
    mkdir -p "$1"
}

ensure_parent_dir() {
    mkdir -p "$(dirname "$1")"
}

resolve_defaults() {
    [[ -n "$base_path" ]] || err "--base-path is required"

    : "${images_path:=${base_path}/fastRecon/dense/sparse/0/images/}"
    : "${sparse_path:=${base_path}/fastRecon/dense/sparse/0/}"
    if [[ -z "$point_cloud_path" ]]; then
        point_cloud_path="${base_path}/output_models/point_cloud/iteration_30000/point_cloud.ply"
        local scene_point_cloud_path="${base_path}/output_models/point_cloud/iteration_30000/scene_point_cloud.ply"
        if [[ ! -f "$point_cloud_path" && -f "$scene_point_cloud_path" ]]; then
            point_cloud_path="$scene_point_cloud_path"
        fi
    fi

    : "${masks_path:=${base_path}/saga/masks}"
    : "${labels_path:=${base_path}/saga/labels}"
    : "${label_features_path:=${base_path}/saga/labels/label_features.pt}"
    : "${mask_scales_path:=${base_path}/saga/mask_scales}"

    : "${contrastive_feature_point_cloud_path:=${base_path}/saga/contrastive_feature_point_cloud.ply}"
    : "${scale_gate_path:=${base_path}/saga/scale_gate.pt}"

    : "${json_path:=${base_path}/saga/output.json}"
    : "${progress_path:=${base_path}/saga/progress}"
    : "${render_path:=${base_path}/saga/render}"
    : "${prior_metadata_path:=${json_path}.metadata.json}"
}

print_config() {
    local cluster_conditions_display="<default:R0/R1/R2>"
    if (( ${#category_cluster_conditions[@]} > 0 )); then
        cluster_conditions_display="${category_cluster_conditions[*]}"
    fi
    cat <<EOF
Resolved configuration:
  python_bin: $python_bin
  stage: $stage
  base_path: $base_path
  images_path: $images_path
  sparse_path: $sparse_path
  point_cloud_path: $point_cloud_path
  masks_path: $masks_path
  labels_path: $labels_path
  label_features_path: $label_features_path
  mask_scales_path: $mask_scales_path
  contrastive_feature_point_cloud_path: $contrastive_feature_point_cloud_path
  scale_gate_path: $scale_gate_path
  json_path: $json_path
  progress_path: $progress_path
  render_path: $render_path
  prior_config: $prior_config
  prior_mapping_config: $prior_mapping_config
  prior_mode: $prior_mode
  prior_gate: $prior_gate
  prior_shrink: $prior_shrink
  prior_metadata_path: $prior_metadata_path
  max_contributor_cache_path: $max_contributor_cache_path
  scene_scale_m_per_unit: $scene_scale_m_per_unit
  seed: $seed
  disable_other_classes: $disable_other_classes
  clustering_mode: $clustering_mode
  class_prior_mode: $class_prior_mode
  category_priors: $category_priors
  class_first_config: $class_first_config
  teacher_prior_mode: $teacher_prior_mode
  teacher_category_params: $teacher_category_params
  category_denoise_action: $category_denoise_action
  category_denoise_bank_path: $category_denoise_bank_path
  category_denoise_scene_id: $category_denoise_scene_id
  category_cluster_audit_path: $category_cluster_audit_path
  category_cluster_conditions: $cluster_conditions_display
  category_cluster_verify_determinism: $category_cluster_verify_determinism
  sam_checkpoint_path: $sam_checkpoint_path
  groundingdino_checkpoint_path: $groundingdino_checkpoint_path
  groundingdino_config_path: $groundingdino_config_path
  sh_degree: $sh_degree
  feature_dim: $feature_dim
  downsample: $downsample
  num_sampled_rays: $num_sampled_rays
  feature_iterations: $feature_iterations
  feature_snapshot_root: $feature_snapshot_root
  feature_seed: $feature_seed
EOF
}

check_python_scripts() {
    case "$stage" in
        all)
            require_file "${SCRIPT_DIR}/grounded_SAM_masks.py" "Python entry script"
            require_file "${SCRIPT_DIR}/get_scale.py" "Python entry script"
            require_file "${SCRIPT_DIR}/train_contrastive_feature.py" "Python entry script"
            require_file "${SCRIPT_DIR}/postprocess.py" "Python entry script"
            ;;
        masks)
            require_file "${SCRIPT_DIR}/grounded_SAM_masks.py" "Python entry script"
            ;;
        scale)
            require_file "${SCRIPT_DIR}/get_scale.py" "Python entry script"
            ;;
        train)
            require_file "${SCRIPT_DIR}/train_contrastive_feature.py" "Python entry script"
            ;;
        postprocess)
            require_file "${SCRIPT_DIR}/postprocess.py" "Python entry script"
            ;;
        render)
            require_file "${SCRIPT_DIR}/render_instance.py" "Python entry script"
            ;;
        gui)
            require_file "${SCRIPT_DIR}/saga_gui.py" "Python entry script"
            ;;
        *)
            err "Unsupported stage: $stage"
            ;;
    esac
}

preflight_stage() {
    case "$1" in
        masks)
            require_dir "$images_path" "Images directory"
            require_file "$sam_checkpoint_path" "SAM checkpoint"
            require_file "$groundingdino_checkpoint_path" "GroundingDINO checkpoint"
            require_file "$groundingdino_config_path" "GroundingDINO config"
            ensure_dir "$masks_path"
            ensure_dir "$labels_path"
            ensure_parent_dir "$label_features_path"
            ensure_parent_dir "$progress_path"
            ;;
        scale)
            require_dir "$images_path" "Images directory"
            require_dir "$sparse_path" "Sparse directory"
            require_file "$point_cloud_path" "Point cloud"
            require_dir "$masks_path" "Masks directory"
            ensure_dir "$mask_scales_path"
            ensure_parent_dir "$progress_path"
            ;;
        train)
            require_dir "$images_path" "Images directory"
            require_dir "$sparse_path" "Sparse directory"
            require_file "$point_cloud_path" "Point cloud"
            require_dir "$masks_path" "Masks directory"
            require_dir "$mask_scales_path" "Mask scales directory"
            require_dir "$labels_path" "Labels directory"
            require_file "$label_features_path" "Label features file"
            ensure_parent_dir "$contrastive_feature_point_cloud_path"
            ensure_parent_dir "$scale_gate_path"
            ensure_parent_dir "$progress_path"
            ;;
        postprocess)
            require_file "$contrastive_feature_point_cloud_path" "Contrastive feature point cloud"
            require_file "$scale_gate_path" "Scale gate weights"
            ensure_parent_dir "$json_path"
            ensure_parent_dir "$progress_path"
            ensure_parent_dir "$prior_metadata_path"
            if [[ "$clustering_mode" == "class-first" ]]; then
                require_file "$label_features_path" "Label features file"
                require_file "$category_priors" "Category priors"
                require_file "$class_first_config" "Class-first config"
            else
                require_dir "$images_path" "Images directory"
                require_dir "$sparse_path" "Sparse directory"
                require_file "$point_cloud_path" "Point cloud"
                require_dir "$masks_path" "Masks directory"
                require_dir "$labels_path" "Labels directory"
                require_dir "$mask_scales_path" "Mask scales directory"
                if [[ "$disable_other_classes" -eq 0 || "$prior_mode" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                else
                    ensure_parent_dir "$label_features_path"
                fi
                if [[ "$prior_mode" != "off" ]]; then
                    require_file "$prior_config" "Category priors"
                    require_file "$prior_mapping_config" "Prior mapping config"
                fi
                if [[ "$clustering_mode" == "legacy-prior" ]]; then
                    require_file "$category_priors" "Category priors"
                    require_file "$legacy_prior_config" "Legacy-prior config"
                fi
                case "$teacher_prior_mode" in
                    all-uniform|size|smooth|small|combined)
                        require_file "$teacher_category_params" "Teacher category parameters"
                        ;;
                esac
                if [[ "$v3_shadow_mode" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                    ensure_parent_dir "$v3_shadow_output"
                    ensure_parent_dir "$v3_branch_labels_output"
                fi
                if [[ "$v4_candidate_mode" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                    require_file "$category_priors" "Category priors"
                    ensure_parent_dir "$v4_candidate_output"
                    ensure_parent_dir "$v4_candidate_labels_output"
                fi
                if [[ "$v5_candidate_source" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                    ensure_parent_dir "$v5_candidate_output"
                    ensure_parent_dir "$v5_candidate_labels_output"
                fi
                if [[ "$v6_candidate_mode" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                    ensure_parent_dir "$v6_candidate_output"
                    ensure_parent_dir "$v6_candidate_labels_output"
                fi
                if [[ "$category_denoise_action" != "off" ]]; then
                    require_file "$label_features_path" "Label features file"
                    require_file "$category_priors" "Category priors"
                    [[ -n "$category_denoise_bank_path" ]] || err "--category-denoise-bank-path is required"
                    ensure_parent_dir "$category_denoise_bank_path"
                    if [[ "$category_denoise_action" == "candidate-repair" ]]; then
                        [[ -n "$category_candidate_trace_path" ]] || err "--category-candidate-trace-path is required"
                        ensure_parent_dir "$category_candidate_trace_path"
                    fi
                    if [[ "$category_denoise_action" == "cluster-bank" ]]; then
                        [[ -n "$category_cluster_audit_path" ]] || err "--category-cluster-audit-path is required"
                        ensure_parent_dir "$category_cluster_audit_path"
                    fi
                fi
            fi
            ;;
        render)
            require_dir "$images_path" "Images directory"
            require_dir "$sparse_path" "Sparse directory"
            require_file "$point_cloud_path" "Point cloud"
            require_file "$json_path" "JSON output"
            ensure_dir "$render_path"
            ;;
        gui)
            require_file "$scale_gate_path" "Scale gate weights"
            require_file "$contrastive_feature_point_cloud_path" "Contrastive feature point cloud"
            require_file "$point_cloud_path" "Point cloud"
            require_file "$json_path" "JSON output"
            ;;
        *)
            err "Unsupported stage: $1"
            ;;
    esac
}

run_masks() {
    echo "Running stage: masks"
    "$python_bin" "${SCRIPT_DIR}/grounded_SAM_masks.py" \
        --progress_path "$progress_path" \
        --images_path "$images_path" \
        --masks_path "$masks_path" \
        --labels_path "$labels_path" \
        --label_features_path "$label_features_path" \
        --sam_checkpoint_path "$sam_checkpoint_path" \
        --groundingdino_checkpoint_path "$groundingdino_checkpoint_path" \
        --groundingdino_config_path "$groundingdino_config_path" \
        --downsample "$downsample"
}

run_scale() {
    echo "Running stage: scale"
    "$python_bin" "${SCRIPT_DIR}/get_scale.py" \
        --progress_path "$progress_path" \
        --sh_degree "$sh_degree" \
        --masks_path "$masks_path" \
        --point_cloud_path "$point_cloud_path" \
        --sparse_path "$sparse_path" \
        --images_path "$images_path" \
        --mask_scales_path "$mask_scales_path"
}

run_train() {
    echo "Running stage: train"
    local snapshot_args=()
    if [[ -n "$feature_snapshot_root" ]]; then
        snapshot_args+=(--feature_snapshot_root "$feature_snapshot_root")
    fi
    "$python_bin" "${SCRIPT_DIR}/train_contrastive_feature.py" \
        --progress_path "$progress_path" \
        --sh_degree "$sh_degree" \
        --feature_dim "$feature_dim" \
        --images_path "$images_path" \
        --sparse_path "$sparse_path" \
        --masks_path "$masks_path" \
        --mask_scales_path "$mask_scales_path" \
        --point_cloud_path "$point_cloud_path" \
        --labels_path "$labels_path" \
        --label_features_path "$label_features_path" \
        --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
        --scale_gate_path "$scale_gate_path" \
        --num_sampled_rays "$num_sampled_rays" \
        --iterations "$feature_iterations" \
        --seed "$feature_seed" \
        "${snapshot_args[@]}"
}

run_postprocess() {
    echo "Running stage: postprocess"
    local prior_args=(
        --prior_mode "$prior_mode"
        --prior_gate "$prior_gate"
        --prior_shrink "$prior_shrink"
        --scene_scale_m_per_unit "$scene_scale_m_per_unit"
        --seed "$seed"
        --prior_metadata_path "$prior_metadata_path"
        --teacher-prior-mode "$teacher_prior_mode"
        --teacher-evidence-protection "$teacher_evidence_protection"
        --v7-causal-ablation "$v7_causal_ablation"
        --category-denoise-action "$category_denoise_action"
        --category-denoise-mode "$category_denoise_mode"
        --category-candidate-score-threshold "$category_candidate_score_threshold"
    )
    if [[ "$category_denoise_action" != "off" ]]; then
        prior_args+=(
            --category-denoise-bank-path "$category_denoise_bank_path"
            --category-denoise-scene-id "$category_denoise_scene_id"
            --category-priors "$category_priors"
        )
        if [[ "$category_denoise_action" == "candidate-repair" ]]; then
            prior_args+=(
                --category-candidate-trace-path "$category_candidate_trace_path"
                --category-candidate-sample-cap "$category_candidate_sample_cap"
            )
        fi
        if [[ "$category_denoise_action" == "cluster-bank" ]]; then
            prior_args+=(--category-cluster-audit-path "$category_cluster_audit_path")
            if [[ "$category_cluster_verify_determinism" -eq 1 ]]; then
                prior_args+=(--category-cluster-verify-determinism)
            fi
            local cluster_condition
            for cluster_condition in "${category_cluster_conditions[@]}"; do
                prior_args+=(--category-cluster-condition "$cluster_condition")
            done
        fi
    fi
    if [[ "$prior_mode" != "off" ]]; then
        prior_args+=(
            --prior_config "$prior_config"
            --prior_mapping_config "$prior_mapping_config"
        )
    fi
    if [[ "$disable_other_classes" -eq 1 ]]; then
        prior_args+=(--disable_other_classes)
    fi
    if [[ -n "$max_contributor_cache_path" ]]; then
        prior_args+=(--max_contributor_cache_path "$max_contributor_cache_path")
    fi
    if [[ "$minimal_metadata" -eq 1 ]]; then
        prior_args+=(--minimal_metadata)
    fi
    case "$teacher_prior_mode" in
        all-uniform|size|smooth|small|combined)
            prior_args+=(--teacher-category-params "$teacher_category_params")
            ;;
    esac
    if [[ "$v3_shadow_mode" != "off" ]]; then
        prior_args+=(
            --v3-shadow-mode "$v3_shadow_mode"
            --v3-shadow-output "$v3_shadow_output"
            --v3-branch-labels-output "$v3_branch_labels_output"
            --v3-shadow-git-commit "$v3_shadow_git_commit"
            --v3-shadow-scene-id "$v3_shadow_scene_id"
        )
    fi
    if [[ "$v4_candidate_mode" != "off" ]]; then
        prior_args+=(
            --v4-candidate-mode "$v4_candidate_mode"
            --category-priors "$category_priors"
            --v4-candidate-output "$v4_candidate_output"
            --v4-candidate-labels-output "$v4_candidate_labels_output"
            --v4-git-commit "$v4_git_commit"
            --v4-scene-id "$v4_scene_id"
        )
    fi
    if [[ "$v5_candidate_source" != "off" ]]; then
        prior_args+=(
            --v5-candidate-source "$v5_candidate_source"
            --v5-candidate-output "$v5_candidate_output"
            --v5-candidate-labels-output "$v5_candidate_labels_output"
            --v5-git-commit "$v5_git_commit"
            --v5-scene-id "$v5_scene_id"
        )
    fi
    if [[ "$v6_candidate_mode" != "off" ]]; then
        prior_args+=(
            --v6-candidate-mode "$v6_candidate_mode"
            --v6-candidate-output "$v6_candidate_output"
            --v6-candidate-labels-output "$v6_candidate_labels_output"
            --v6-git-commit "$v6_git_commit"
            --v6-scene-id "$v6_scene_id"
        )
    fi
    if [[ "$clustering_mode" == "class-first" ]]; then
        prior_args+=(
            --clustering-mode "$clustering_mode"
            --class-prior-mode "$class_prior_mode"
            --category-priors "$category_priors"
            --class-first-config "$class_first_config"
        )
    elif [[ "$clustering_mode" == "legacy-prior" ]]; then
        prior_args+=(
            --clustering-mode "$clustering_mode"
            --category-priors "$category_priors"
            --legacy-prior-config "$legacy_prior_config"
            --legacy-prior-mode "$legacy_prior_mode"
            --legacy-prior-score "$legacy_prior_score"
            --legacy-prior-semantic-source "$legacy_prior_semantic_source"
        )
    fi
    "$python_bin" "${SCRIPT_DIR}/postprocess.py" \
        --progress_path "$progress_path" \
        --sh_degree "$sh_degree" \
        --feature_dim "$feature_dim" \
        --images_path "$images_path" \
        --sparse_path "$sparse_path" \
        --masks_path "$masks_path" \
        --labels_path "$labels_path" \
        --label_features_path "$label_features_path" \
        --mask_scales_path "$mask_scales_path" \
        --point_cloud_path "$point_cloud_path" \
        --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
        --scale_gate_path "$scale_gate_path" \
        --json_path "$json_path" \
        "${prior_args[@]}"
}

run_render() {
    echo "Running stage: render"
    "$python_bin" "${SCRIPT_DIR}/render_instance.py" \
        --sh_degree "$sh_degree" \
        --feature_dim "$feature_dim" \
        --images_path "$images_path" \
        --sparse_path "$sparse_path" \
        --masks_path "$masks_path" \
        --labels_path "$labels_path" \
        --mask_scales_path "$mask_scales_path" \
        --point_cloud_path "$point_cloud_path" \
        --contrastive_feature_point_cloud_path "$contrastive_feature_point_cloud_path" \
        --scale_gate_path "$scale_gate_path" \
        --json_path "$json_path" \
        --render_path "$render_path"
}

run_gui() {
    echo "Running stage: gui"
    "$python_bin" "${SCRIPT_DIR}/saga_gui.py" \
        --sh_degree "$sh_degree" \
        --scale_gate_path "$scale_gate_path" \
        --feature_pcd_path "$contrastive_feature_point_cloud_path" \
        --scene_pcd_path "$point_cloud_path" \
        --json_path "$json_path"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-path)
            base_path="$2"
            shift 2
            ;;
        --python)
            python_bin="$2"
            shift 2
            ;;
        --stage)
            stage="$2"
            shift 2
            ;;
        --images-path)
            images_path="$2"
            shift 2
            ;;
        --sparse-path)
            sparse_path="$2"
            shift 2
            ;;
        --point-cloud-path)
            point_cloud_path="$2"
            shift 2
            ;;
        --masks-path)
            masks_path="$2"
            shift 2
            ;;
        --labels-path)
            labels_path="$2"
            shift 2
            ;;
        --label-features-path)
            label_features_path="$2"
            shift 2
            ;;
        --mask-scales-path)
            mask_scales_path="$2"
            shift 2
            ;;
        --contrastive-feature-point-cloud-path)
            contrastive_feature_point_cloud_path="$2"
            shift 2
            ;;
        --scale-gate-path)
            scale_gate_path="$2"
            shift 2
            ;;
        --json-path)
            json_path="$2"
            shift 2
            ;;
        --progress-path)
            progress_path="$2"
            shift 2
            ;;
        --render-path)
            render_path="$2"
            shift 2
            ;;
        --prior-config)
            prior_config="$2"
            shift 2
            ;;
        --prior-mapping-config)
            prior_mapping_config="$2"
            shift 2
            ;;
        --prior-mode)
            prior_mode="$2"
            shift 2
            ;;
        --prior-gate)
            prior_gate="$2"
            shift 2
            ;;
        --prior-shrink)
            prior_shrink="$2"
            shift 2
            ;;
        --prior-metadata-path)
            prior_metadata_path="$2"
            shift 2
            ;;
        --max-contributor-cache-path)
            max_contributor_cache_path="$2"
            shift 2
            ;;
        --scene-scale-m-per-unit)
            scene_scale_m_per_unit="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --disable-other-classes)
            disable_other_classes=1
            shift
            ;;
        --minimal-metadata)
            minimal_metadata=1
            shift
            ;;
        --clustering-mode)
            clustering_mode="$2"
            shift 2
            ;;
        --class-prior-mode)
            class_prior_mode="$2"
            shift 2
            ;;
        --category-priors)
            category_priors="$2"
            shift 2
            ;;
        --class-first-config)
            class_first_config="$2"
            shift 2
            ;;
        --legacy-prior-config)
            legacy_prior_config="$2"
            shift 2
            ;;
        --legacy-prior-mode)
            legacy_prior_mode="$2"
            shift 2
            ;;
        --legacy-prior-score)
            legacy_prior_score="$2"
            shift 2
            ;;
        --legacy-prior-semantic-source)
            legacy_prior_semantic_source="$2"
            shift 2
            ;;
        --teacher-prior-mode)
            teacher_prior_mode="$2"
            shift 2
            ;;
        --teacher-category-params)
            teacher_category_params="$2"
            shift 2
            ;;
        --teacher-evidence-protection)
            teacher_evidence_protection="$2"
            shift 2
            ;;
        --v7-causal-ablation)
            v7_causal_ablation="$2"
            shift 2
            ;;
        --category-denoise-action)
            category_denoise_action="$2"
            shift 2
            ;;
        --category-denoise-bank-path)
            category_denoise_bank_path="$2"
            shift 2
            ;;
        --category-candidate-trace-path)
            category_candidate_trace_path="$2"
            shift 2
            ;;
        --category-candidate-sample-cap)
            category_candidate_sample_cap="$2"
            shift 2
            ;;
        --category-candidate-score-threshold)
            category_candidate_score_threshold="$2"
            shift 2
            ;;
        --category-cluster-audit-path)
            category_cluster_audit_path="$2"
            shift 2
            ;;
        --category-cluster-condition)
            category_cluster_conditions+=("$2")
            shift 2
            ;;
        --category-cluster-verify-determinism)
            category_cluster_verify_determinism=1
            shift
            ;;
        --category-denoise-mode)
            category_denoise_mode="$2"
            shift 2
            ;;
        --category-denoise-scene-id)
            category_denoise_scene_id="$2"
            shift 2
            ;;
        --v3-shadow-mode)
            v3_shadow_mode="$2"
            shift 2
            ;;
        --v3-shadow-output)
            v3_shadow_output="$2"
            shift 2
            ;;
        --v3-branch-labels-output)
            v3_branch_labels_output="$2"
            shift 2
            ;;
        --v3-shadow-git-commit)
            v3_shadow_git_commit="$2"
            shift 2
            ;;
        --v3-shadow-scene-id)
            v3_shadow_scene_id="$2"
            shift 2
            ;;
        --v4-candidate-mode)
            v4_candidate_mode="$2"
            shift 2
            ;;
        --v4-candidate-output)
            v4_candidate_output="$2"
            shift 2
            ;;
        --v4-candidate-labels-output)
            v4_candidate_labels_output="$2"
            shift 2
            ;;
        --v4-git-commit)
            v4_git_commit="$2"
            shift 2
            ;;
        --v4-scene-id)
            v4_scene_id="$2"
            shift 2
            ;;
        --v5-candidate-source)
            v5_candidate_source="$2"
            shift 2
            ;;
        --v5-candidate-output)
            v5_candidate_output="$2"
            shift 2
            ;;
        --v5-candidate-labels-output)
            v5_candidate_labels_output="$2"
            shift 2
            ;;
        --v5-git-commit)
            v5_git_commit="$2"
            shift 2
            ;;
        --v5-scene-id)
            v5_scene_id="$2"
            shift 2
            ;;
        --v6-candidate-mode)
            v6_candidate_mode="$2"
            shift 2
            ;;
        --v6-candidate-output)
            v6_candidate_output="$2"
            shift 2
            ;;
        --v6-candidate-labels-output)
            v6_candidate_labels_output="$2"
            shift 2
            ;;
        --v6-git-commit)
            v6_git_commit="$2"
            shift 2
            ;;
        --v6-scene-id)
            v6_scene_id="$2"
            shift 2
            ;;
        --sam-checkpoint-path)
            sam_checkpoint_path="$2"
            shift 2
            ;;
        --groundingdino-checkpoint-path)
            groundingdino_checkpoint_path="$2"
            shift 2
            ;;
        --groundingdino-config-path)
            groundingdino_config_path="$2"
            shift 2
            ;;
        --sh-degree)
            sh_degree="$2"
            shift 2
            ;;
        --feature-dim)
            feature_dim="$2"
            shift 2
            ;;
        --downsample)
            downsample="$2"
            shift 2
            ;;
        --num-sampled-rays)
            num_sampled_rays="$2"
            shift 2
            ;;
        --feature-iterations)
            feature_iterations="$2"
            shift 2
            ;;
        --feature-snapshot-root)
            feature_snapshot_root="$2"
            shift 2
            ;;
        --feature-seed)
            feature_seed="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "Unknown argument: $1"
            ;;
    esac
done

resolve_defaults
[[ "$feature_iterations" =~ ^[0-9]+$ ]] || err "--feature-iterations must be nonnegative"
[[ "$feature_seed" =~ ^[0-9]+$ ]] || err "--feature-seed must be nonnegative"
[[ "$clustering_mode" == "legacy" || "$clustering_mode" == "class-first" \
    || "$clustering_mode" == "legacy-prior" ]] \
    || err "--clustering-mode must be legacy, class-first, or legacy-prior"
if [[ "$clustering_mode" == "class-first" ]]; then
    case "$class_prior_mode" in
        uniform|size|smooth|small|combined) ;;
        *) err "unsupported --class-prior-mode: $class_prior_mode" ;;
    esac
fi
case "$teacher_prior_mode" in
    off|original|all-uniform|size|smooth|small|combined) ;;
    *) err "unsupported --teacher-prior-mode: $teacher_prior_mode" ;;
esac
case "$teacher_evidence_protection" in
    off|multi-anchor) ;;
    *) err "unsupported --teacher-evidence-protection: $teacher_evidence_protection" ;;
esac
case "$v3_shadow_mode" in
    off|exact|exclusive|both) ;;
    *) err "unsupported --v3-shadow-mode: $v3_shadow_mode" ;;
esac
case "$v4_candidate_mode" in
    off|uniform|class-scale|class-core|combined) ;;
    *) err "unsupported --v4-candidate-mode: $v4_candidate_mode" ;;
esac
case "$v5_candidate_source" in
    off|codebook|multiview) ;;
    *) err "unsupported --v5-candidate-source: $v5_candidate_source" ;;
esac
case "$v6_candidate_mode" in
    off|affinity-first) ;;
    *) err "unsupported --v6-candidate-mode: $v6_candidate_mode" ;;
esac
case "$category_denoise_action" in
    off|bank|candidate-repair|cluster-bank|replay|candidate-replay) ;;
    *) err "unsupported --category-denoise-action: $category_denoise_action" ;;
esac
if [[ "$category_denoise_action" == "cluster-bank" ]]; then
    for cluster_condition in "${category_cluster_conditions[@]}"; do
        case "$cluster_condition" in
            R0-legacy|R1-corrected-distance-legacy-expand|R2-corrected-distance-anchored-expand|G1-mutual-local-graph) ;;
            *) err "unsupported --category-cluster-condition: $cluster_condition" ;;
        esac
    done
elif [[ -n "$category_cluster_audit_path" || ${#category_cluster_conditions[@]} -gt 0 || "$category_cluster_verify_determinism" -eq 1 ]]; then
    err "category cluster arguments require --category-denoise-action cluster-bank"
fi
if [[ -z "$python_bin" ]]; then
    find_python
else
    require_file "$python_bin" "Python executable"
fi
check_python_scripts
print_config

case "$stage" in
    all)
        preflight_stage masks
        run_masks
        preflight_stage scale
        run_scale
        preflight_stage train
        run_train
        preflight_stage postprocess
        run_postprocess
        ;;
    masks)
        preflight_stage masks
        run_masks
        ;;
    scale)
        preflight_stage scale
        run_scale
        ;;
    train)
        preflight_stage train
        run_train
        ;;
    postprocess)
        preflight_stage postprocess
        run_postprocess
        ;;
    render)
        preflight_stage render
        run_render
        ;;
    gui)
        preflight_stage gui
        run_gui
        ;;
    *)
        err "Unsupported stage: $stage"
        ;;
esac
