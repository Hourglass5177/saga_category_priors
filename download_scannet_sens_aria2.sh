#!/usr/bin/env bash

set -euo pipefail

scene_list=""
out_dir=""
official_downloader=""
manifest=""
python_bin=""
workspace=""
status_path=""
limit=8
proxy="http://127.0.0.1:17890"
concurrent_downloads=4
connections_per_download=4
min_free_gb=80

usage() {
    cat <<'EOF'
Usage:
  bash download_scannet_sens_aria2.sh [required options] [options]

Required:
  --scene-list PATH
  --out-dir PATH
  --official-downloader PATH
  --manifest PATH
  --python PATH
  --workspace PATH

Options:
  --status PATH                    Default: MANIFEST.status
  --limit INT                      Default: 8
  --proxy URL                      Default: http://127.0.0.1:17890
  --concurrent-downloads INT       Default: 4
  --connections-per-download INT   Default: 4
  --min-free-gb INT                Default: 80

Downloads ScanNet v1 .sens streams into resumable .part files with aria2, then
atomically renames successful files and uses the audited Python downloader to
hash all final files and materialize the registered manifest.
EOF
}

err() {
    echo "Error: $*" >&2
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scene-list) scene_list="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --official-downloader) official_downloader="$2"; shift 2 ;;
        --manifest) manifest="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        --workspace) workspace="$2"; shift 2 ;;
        --status) status_path="$2"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --proxy) proxy="$2"; shift 2 ;;
        --concurrent-downloads) concurrent_downloads="$2"; shift 2 ;;
        --connections-per-download) connections_per_download="$2"; shift 2 ;;
        --min-free-gb) min_free_gb="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1" ;;
    esac
done

for value in scene_list out_dir official_downloader manifest python_bin workspace; do
    [[ -n "${!value}" ]] || err "--${value//_/-} is required"
done
for value in limit concurrent_downloads connections_per_download min_free_gb; do
    [[ "${!value}" =~ ^[1-9][0-9]*$ ]] || err "--${value//_/-} must be positive"
done
[[ -n "$status_path" ]] || status_path="${manifest}.status"
[[ -s "$scene_list" ]] || err "scene list is missing or empty: $scene_list"
[[ -s "$official_downloader" ]] \
    || err "official downloader is missing or empty: $official_downloader"
[[ -s "$python_bin" ]] || err "Python executable is missing or empty: $python_bin"
[[ -d "$workspace" ]] || err "workspace directory is missing: $workspace"
command -v aria2c >/dev/null 2>&1 || err "aria2c is not installed"
mkdir -p "$out_dir"

stage="initializing"
input_path="$(mktemp)"
cleanup() {
    rm -f -- "$input_path"
}
on_error() {
    local rc=$?
    printf 'failed stage=%s return_code=%s failed_at=%s\n' \
        "$stage" "$rc" "$(date -Is)" > "$status_path"
    cleanup
    exit "$rc"
}
trap on_error ERR
trap cleanup EXIT

mapfile -t scenes < <(sed '/^[[:space:]]*$/d' "$scene_list" | head -n "$limit")
[[ "${#scenes[@]}" -eq "$limit" ]] \
    || err "scene list has fewer than $limit nonempty rows"

stage="building_input"
for scene_id in "${scenes[@]}"; do
    [[ "$scene_id" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] \
        || err "invalid ScanNet scene id: $scene_id"
    scene_dir="$out_dir/scans/$scene_id"
    final="$scene_dir/$scene_id.sens"
    partial="$final.part"
    mkdir -p "$scene_dir"
    if [[ -s "$final" ]]; then
        continue
    fi
    printf 'http://kaldir.vc.in.tum.de/scannet/v1/scans/%s/%s.sens\n' \
        "$scene_id" "$scene_id" >> "$input_path"
    printf '  dir=%s\n  out=%s.sens.part\n' "$scene_dir" "$scene_id" \
        >> "$input_path"
done

if [[ -s "$input_path" ]]; then
    stage="aria2_download"
    printf 'running stage=%s started_at=%s\n' "$stage" "$(date -Is)" \
        > "$status_path"
    aria2c \
        --input-file="$input_path" \
        --continue=true \
        --max-concurrent-downloads="$concurrent_downloads" \
        --max-connection-per-server="$connections_per_download" \
        --split="$connections_per_download" \
        --min-split-size=1M \
        --file-allocation=none \
        --allow-overwrite=true \
        --auto-file-renaming=false \
        --max-tries=8 \
        --retry-wait=5 \
        --connect-timeout=60 \
        --timeout=60 \
        --lowest-speed-limit=1K \
        --summary-interval=10 \
        --console-log-level=notice \
        --download-result=full \
        --all-proxy="$proxy"
fi

stage="atomic_finalize"
for scene_id in "${scenes[@]}"; do
    scene_dir="$out_dir/scans/$scene_id"
    final="$scene_dir/$scene_id.sens"
    partial="$final.part"
    if [[ ! -s "$final" ]]; then
        [[ -s "$partial" ]] || err "aria2 output is missing or empty: $partial"
        [[ ! -e "$partial.aria2" ]] \
            || err "aria2 control file remains after successful exit: $partial.aria2"
        mv "$partial" "$final"
    fi
done

stage="manifest_audit"
printf 'running stage=%s started_at=%s\n' "$stage" "$(date -Is)" \
    > "$status_path"
cd "$workspace"
PYTHONPATH=. "$python_bin" -m category_priors download-scannet-saga \
    --official-downloader "$official_downloader" \
    --scene-list "$scene_list" \
    --out-dir "$out_dir" \
    --manifest "$manifest" \
    --workers 1 \
    --retries 1 \
    --timeout-s 60 \
    --min-free-gb "$min_free_gb" \
    --limit "$limit" \
    --accept-tos

stage="complete"
printf 'complete stage=%s scenes=%s completed_at=%s\n' \
    "$stage" "$limit" "$(date -Is)" > "$status_path"
echo "ScanNet .sens aria2 wave complete: $limit scenes"
