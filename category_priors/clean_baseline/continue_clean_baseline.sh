#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 CONFIG_JSON" >&2
    exit 64
fi

CONFIG_PATH=$1
PYTHON_BIN=${PYTHON_BIN:-python}
MIN_AVAILABLE_KIB=$((80 * 1024 * 1024))
CGROUP_ROOT=${CLEAN_BASELINE_CGROUP_ROOT:-/sys/fs/cgroup}

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "registered config does not exist: $CONFIG_PATH" >&2
    exit 66
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python executable is unavailable: $PYTHON_BIN" >&2
    exit 69
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing to start without a GPU audit" >&2
    exit 69
fi

# Read only the three deployment paths needed by this launcher.  The actual
# experiment module performs the authoritative config validation.
mapfile -t CONFIG_VALUES < <(
    "$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(source.read_text(encoding="utf-8"))
base = source.parent

def resolved(value):
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()

print(source)
print(resolved(payload["repo_root"]))
print(resolved(payload["run_root"]))
print(resolved(payload["artifact_root"]))
PY
)
if [[ ${#CONFIG_VALUES[@]} -ne 4 ]]; then
    echo "could not resolve registered deployment paths" >&2
    exit 65
fi
CONFIG_PATH=${CONFIG_VALUES[0]}
REPO_ROOT=${CONFIG_VALUES[1]}
RUN_ROOT=${CONFIG_VALUES[2]}
ARTIFACT_ROOT=${CONFIG_VALUES[3]}

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "registered repository does not exist: $REPO_ROOT" >&2
    exit 66
fi
mkdir -p "$ARTIFACT_ROOT"

PID_PATH="$ARTIFACT_ROOT/clean_baseline_launcher.pid"
STATUS_PATH="$ARTIFACT_ROOT/clean_baseline_launcher_status.json"
LOG_PATH="$ARTIFACT_ROOT/continuation.log"
RESOURCE_LOG="$ARTIFACT_ROOT/resource_audit.log"

pid_matches_launcher() {
    local candidate=$1
    [[ "$candidate" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$candidate" 2>/dev/null || return 1
    [[ -r "/proc/$candidate/cmdline" ]] || return 1
    local command_line
    command_line=$(tr '\0' ' ' < "/proc/$candidate/cmdline")
    [[ "$command_line" == *"continue_clean_baseline.sh"* || \
       "$command_line" == *"category_priors.clean_baseline.experiment"* ]]
}

if [[ -f "$PID_PATH" ]]; then
    EXISTING_PID=$(tr -d '[:space:]' < "$PID_PATH")
    if pid_matches_launcher "$EXISTING_PID"; then
        echo "clean baseline is already active with PID $EXISTING_PID"
        exit 0
    fi
    rm -f -- "$PID_PATH"
fi

if ! (set -o noclobber; printf '%s\n' "$$" > "$PID_PATH") 2>/dev/null; then
    EXISTING_PID=$(tr -d '[:space:]' < "$PID_PATH" 2>/dev/null || true)
    if pid_matches_launcher "$EXISTING_PID"; then
        echo "clean baseline is already active with PID $EXISTING_PID"
        exit 0
    fi
    echo "could not claim the launcher PID file safely" >&2
    exit 75
fi

cleanup_pid() {
    if [[ -f "$PID_PATH" ]] && [[ "$(tr -d '[:space:]' < "$PID_PATH")" == "$$" ]]; then
        rm -f -- "$PID_PATH"
    fi
}
trap cleanup_pid EXIT

for required in memory.current memory.max memory.events; do
    if [[ ! -r "$CGROUP_ROOT/$required" ]]; then
        echo "missing cgroup audit input: $CGROUP_ROOT/$required" >&2
        exit 69
    fi
done

DISK_PROBE=$RUN_ROOT
while [[ ! -e "$DISK_PROBE" && "$DISK_PROBE" != "/" ]]; do
    DISK_PROBE=$(dirname "$DISK_PROBE")
done
AVAILABLE_KIB=$(df -Pk "$DISK_PROBE" | awk 'NR == 2 {print $4}')
if [[ ! "$AVAILABLE_KIB" =~ ^[0-9]+$ ]]; then
    echo "could not read available disk space for $DISK_PROBE" >&2
    exit 69
fi
if (( AVAILABLE_KIB < MIN_AVAILABLE_KIB )); then
    echo "data disk has less than 80 GiB available; refusing to start" >&2
    exit 75
fi

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'timestamp=%s\n' "$TIMESTAMP"
    printf 'config=%s\n' "$CONFIG_PATH"
    printf 'gpu='
    nvidia-smi \
        --query-gpu=name,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits
    printf 'disk='
    df -Pk "$DISK_PROBE" | awk 'NR == 2 {print $1, $2, $3, $4, $6}'
    printf 'memory.current=%s\n' "$(<"$CGROUP_ROOT/memory.current")"
    printf 'memory.max=%s\n' "$(<"$CGROUP_ROOT/memory.max")"
    printf 'memory.events='
    tr '\n' ' ' < "$CGROUP_ROOT/memory.events"
    printf '\n'
} >> "$RESOURCE_LOG"

write_status() {
    local phase=$1
    local exit_code=$2
    CLEAN_STATUS_PATH=$STATUS_PATH \
    CLEAN_STATUS_PHASE=$phase \
    CLEAN_STATUS_EXIT_CODE=$exit_code \
    CLEAN_STATUS_CONFIG=$CONFIG_PATH \
    CLEAN_STATUS_PID=$$ \
    CLEAN_STATUS_LOG=$LOG_PATH \
    "$PYTHON_BIN" - <<'PY'
import datetime
import json
import os
import pathlib

target = pathlib.Path(os.environ["CLEAN_STATUS_PATH"])
payload = {
    "schema": "saga-clean-alpha-mask-launcher-status-v1",
    "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "phase": os.environ["CLEAN_STATUS_PHASE"],
    "exit_code": int(os.environ["CLEAN_STATUS_EXIT_CODE"]),
    "config": os.environ["CLEAN_STATUS_CONFIG"],
    "launcher_pid": int(os.environ["CLEAN_STATUS_PID"]),
    "log": os.environ["CLEAN_STATUS_LOG"],
}
temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, target)
PY
}

write_status running 0
printf '[%s] starting clean baseline config=%s\n' "$TIMESTAMP" "$CONFIG_PATH" >> "$LOG_PATH"

set +e
(
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -m category_priors.clean_baseline.experiment --config "$CONFIG_PATH"
) >> "$LOG_PATH" 2>&1
EXPERIMENT_EXIT=$?
set -e

FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if (( EXPERIMENT_EXIT == 0 )); then
    EXPERIMENT_STATE=$(
        "$PYTHON_BIN" - "$ARTIFACT_ROOT/clean_baseline_status.json" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
try:
    payload = json.loads(source.read_text(encoding="utf-8"))
except (OSError, ValueError):
    print("invalid")
else:
    print(payload.get("status", "invalid"))
PY
    )
    if [[ "$EXPERIMENT_STATE" == "complete" ]]; then
        printf '[%s] clean baseline completed\n' "$FINISHED_AT" >> "$LOG_PATH"
        write_status completed 0
    elif [[ "$EXPERIMENT_STATE" == "stopped" ]]; then
        printf '[%s] clean baseline reached a registered stop gate\n' "$FINISHED_AT" >> "$LOG_PATH"
        write_status stopped 0
    else
        EXPERIMENT_EXIT=70
        printf '[%s] clean baseline returned success without a terminal state (%s)\n' \
            "$FINISHED_AT" "$EXPERIMENT_STATE" >> "$LOG_PATH"
        write_status failed "$EXPERIMENT_EXIT"
    fi
else
    printf '[%s] clean baseline failed exit=%s\n' "$FINISHED_AT" "$EXPERIMENT_EXIT" >> "$LOG_PATH"
    write_status failed "$EXPERIMENT_EXIT"
fi
exit "$EXPERIMENT_EXIT"
