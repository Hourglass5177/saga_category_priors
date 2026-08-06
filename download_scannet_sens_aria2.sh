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
max_tries=0
transfer_timeout=120
verified_url_map=""

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
  --max-tries INT                  Default: 0 (retry indefinitely)
  --transfer-timeout INT           Default: 120
  --verified-url-map PATH          Optional TSV: scene_id, URL, bytes, SHA-256

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
        --max-tries) max_tries="$2"; shift 2 ;;
        --transfer-timeout) transfer_timeout="$2"; shift 2 ;;
        --verified-url-map) verified_url_map="$2"; shift 2 ;;
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
[[ "$max_tries" =~ ^[0-9]+$ ]] || err "--max-tries must be nonnegative"
[[ "$transfer_timeout" =~ ^[1-9][0-9]*$ ]] \
    || err "--transfer-timeout must be positive"
[[ -n "$status_path" ]] || status_path="${manifest}.status"
[[ -s "$scene_list" ]] || err "scene list is missing or empty: $scene_list"
[[ -s "$official_downloader" ]] \
    || err "official downloader is missing or empty: $official_downloader"
[[ -s "$python_bin" ]] || err "Python executable is missing or empty: $python_bin"
[[ -d "$workspace" ]] || err "workspace directory is missing: $workspace"
if [[ -n "$verified_url_map" ]]; then
    [[ -s "$verified_url_map" ]] \
        || err "verified URL map is missing or empty: $verified_url_map"
fi
command -v aria2c >/dev/null 2>&1 || err "aria2c is not installed"
mkdir -p "$out_dir"

declare -A verified_urls=()
declare -A verified_bytes=()
declare -A verified_hashes=()
if [[ -n "$verified_url_map" ]]; then
    while IFS=$'\t' read -r scene_id url expected_bytes expected_hash extra; do
        [[ -z "$scene_id" || "$scene_id" == \#* ]] && continue
        [[ -z "$extra" ]] || err "too many columns in verified URL map: $scene_id"
        [[ "$scene_id" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] \
            || err "invalid scene id in verified URL map: $scene_id"
        [[ "$url" =~ ^https?:// ]] \
            || err "invalid URL in verified URL map: $scene_id"
        [[ "$expected_bytes" =~ ^[1-9][0-9]*$ ]] \
            || err "invalid byte count in verified URL map: $scene_id"
        [[ "$expected_hash" =~ ^[0-9a-f]{64}$ ]] \
            || err "invalid SHA-256 in verified URL map: $scene_id"
        [[ -z "${verified_urls[$scene_id]+x}" ]] \
            || err "duplicate scene in verified URL map: $scene_id"
        verified_urls[$scene_id]="$url"
        verified_bytes[$scene_id]="$expected_bytes"
        verified_hashes[$scene_id]="$expected_hash"
    done < "$verified_url_map"
fi

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
    if [[ -n "${verified_urls[$scene_id]+x}" ]]; then
        printf '%s\n' "${verified_urls[$scene_id]}" >> "$input_path"
    else
        printf 'http://kaldir.vc.in.tum.de/scannet/v1/scans/%s/%s.sens\n' \
            "$scene_id" "$scene_id" >> "$input_path"
    fi
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
        --max-tries="$max_tries" \
        --retry-wait=5 \
        --connect-timeout=60 \
        --timeout="$transfer_timeout" \
        --lowest-speed-limit=0 \
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
    if [[ -n "${verified_urls[$scene_id]+x}" ]]; then
        candidate="$partial"
        [[ ! -s "$final" ]] || candidate="$final"
        [[ -s "$candidate" ]] \
            || err "verified mirror output is missing or empty: $candidate"
        actual_bytes="$(stat -c '%s' "$candidate")"
        [[ "$actual_bytes" == "${verified_bytes[$scene_id]}" ]] \
            || err "verified mirror size mismatch for $scene_id: expected ${verified_bytes[$scene_id]}, got $actual_bytes"
        actual_hash="$(sha256sum "$candidate" | awk '{print $1}')"
        [[ "$actual_hash" == "${verified_hashes[$scene_id]}" ]] \
            || err "verified mirror SHA-256 mismatch for $scene_id"
    fi
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
