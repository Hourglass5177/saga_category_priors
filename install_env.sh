#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

ENV_NAME="saga"
CUDA_HOME_VALUE="${CUDA_HOME:-/usr/local/cuda}"
SKIP_WEIGHTS=0
FORCE_GROUNDINGDINO_REINSTALL=0

PYTHON_VERSION="3.10.19"
TORCH_VERSION="2.4.1+cu118"
TORCHVISION_VERSION="0.19.1+cu118"
TORCHAUDIO_VERSION="2.4.1+cu118"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
GROUNDINGDINO_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

CONDA_BIN=""
DOWNLOAD_TOOL=""

usage() {
    cat <<EOF
Usage:
  bash install_env.sh [options]

Options:
  --env-name NAME                   Conda environment name (default: saga)
  --cuda-home PATH                  CUDA_HOME to use (default: current CUDA_HOME or /usr/local/cuda)
  --skip-weights                    Skip downloading SAM and GroundingDINO weights
  --force-groundingdino-reinstall   Reinstall GroundingDINO even if already installed
  -h, --help                        Show this help message

Examples:
  bash install_env.sh
  bash install_env.sh --env-name saga
  bash install_env.sh --skip-weights
  bash install_env.sh --cuda-home /usr/local/cuda-11.8
EOF
}

log() {
    echo "[install_env] $*"
}

err() {
    echo "Error: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || err "Required command not found: $1"
}

require_dir() {
    [[ -d "$1" ]] || err "$2 not found: $1"
}

require_file() {
    [[ -f "$1" ]] || err "$2 not found: $1"
}

find_conda() {
    require_command conda
    CONDA_BIN="$(command -v conda)"
}

find_download_tool() {
    if command -v wget >/dev/null 2>&1; then
        DOWNLOAD_TOOL="wget"
    elif command -v curl >/dev/null 2>&1; then
        DOWNLOAD_TOOL="curl"
    else
        err "Required command not found: wget or curl"
    fi
}

download_file() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [[ -f "$dest" ]]; then
        log "Found existing ${name}: $dest"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    log "Downloading ${name} -> $dest"
    if [[ "$DOWNLOAD_TOOL" == "wget" ]]; then
        wget -q -O "$dest" "$url"
    else
        curl -L --fail -o "$dest" "$url"
    fi
}

init_conda_shell() {
    eval "$($CONDA_BIN shell.bash hook)"
}

ensure_env() {
    if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
        log "Using existing conda environment: $ENV_NAME"
        conda install -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
    else
        log "Creating conda environment: $ENV_NAME"
        conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}" pip
    fi
}

ensure_submodules() {
    require_command git
    log "Initializing git submodules"
    git -C "$REPO_ROOT" submodule update --init --recursive

    require_dir "$REPO_ROOT/third_party/GroundingDINO" "GroundingDINO submodule"
    require_dir "$REPO_ROOT/third_party/segment-anything" "segment-anything submodule"
    require_dir "$REPO_ROOT/submodules/diff-gaussian-rasterization" "diff-gaussian-rasterization submodule"
    require_dir "$REPO_ROOT/submodules/diff-gaussian-rasterization_contrastive_f" "diff-gaussian-rasterization_contrastive_f submodule"
    require_dir "$REPO_ROOT/submodules/diff-gaussian-rasterization-depth" "diff-gaussian-rasterization-depth submodule"
    require_dir "$REPO_ROOT/submodules/simple-knn" "simple-knn submodule"
}

configure_cuda_home() {
    export CUDA_HOME="$CUDA_HOME_VALUE"
    require_dir "$CUDA_HOME" "CUDA_HOME directory"
    log "Using CUDA_HOME=$CUDA_HOME"
}

install_python_packages() {
    log "Activating conda environment: $ENV_NAME"
    conda activate "$ENV_NAME"

    log "Installing PyTorch CUDA 11.8 stack"
    pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --extra-index-url "$TORCH_INDEX_URL"

    log "Installing requirements.txt"
    pip install --no-build-isolation -r "$REPO_ROOT/requirements.txt"

    log "Installing GroundingDINO"
    if [[ "$FORCE_GROUNDINGDINO_REINSTALL" -eq 1 ]]; then
        pip install --no-build-isolation --force-reinstall "$REPO_ROOT/third_party/GroundingDINO"
    else
        pip install --no-build-isolation "$REPO_ROOT/third_party/GroundingDINO"
    fi
}

install_weights() {
    local weights_dir="$REPO_ROOT/weights"
    local sam_dest="$weights_dir/sam_vit_h_4b8939.pth"
    local groundingdino_dest="$weights_dir/groundingdino_swint_ogc.pth"
    local groundingdino_config_src="$REPO_ROOT/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    local groundingdino_config_dest="$weights_dir/GroundingDINO_SwinT_OGC.py"

    mkdir -p "$weights_dir"
    require_file "$groundingdino_config_src" "GroundingDINO config source"
    cp -f "$groundingdino_config_src" "$groundingdino_config_dest"
    log "Copied GroundingDINO config -> $groundingdino_config_dest"

    if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
        log "Skipping weight downloads"
        return 0
    fi

    find_download_tool
    download_file "$SAM_URL" "$sam_dest" "SAM ViT-H weights"
    download_file "$GROUNDINGDINO_URL" "$groundingdino_dest" "GroundingDINO SwinT OGC weights"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --cuda-home)
            CUDA_HOME_VALUE="$2"
            shift 2
            ;;
        --skip-weights)
            SKIP_WEIGHTS=1
            shift
            ;;
        --force-groundingdino-reinstall)
            FORCE_GROUNDINGDINO_REINSTALL=1
            shift
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

find_conda
init_conda_shell
ensure_submodules
configure_cuda_home
ensure_env
install_python_packages
install_weights

log "Installation complete"
log "Activate with: conda activate $ENV_NAME"
