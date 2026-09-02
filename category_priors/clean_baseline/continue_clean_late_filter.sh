#!/usr/bin/env bash
set -euo pipefail

: "${SAGA_LATE_FILTER_MANIFEST:?set SAGA_LATE_FILTER_MANIFEST}"
: "${SAGA_LATE_FILTER_OUTPUT:?set SAGA_LATE_FILTER_OUTPUT}"
: "${SAGA_LATE_FILTER_COMMIT:?set SAGA_LATE_FILTER_COMMIT}"
: "${SAGA_LATE_FILTER_PYTHON:?set SAGA_LATE_FILTER_PYTHON}"

if [[ ! "${SAGA_LATE_FILTER_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SAGA_LATE_FILTER_COMMIT must be a full lowercase commit" >&2
  exit 2
fi
if [[ "${SAGA_LATE_FILTER_PYTHON}" != /* ]]; then
  echo "SAGA_LATE_FILTER_PYTHON must be an absolute executable path" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
manifest_path="$(readlink -f -- "${SAGA_LATE_FILTER_MANIFEST}")"
output_root="$(readlink -m -- "${SAGA_LATE_FILTER_OUTPUT}")"

if [[ ! -f "${manifest_path}" || ! -x "${SAGA_LATE_FILTER_PYTHON}" ]]; then
  echo "registered manifest or Python executable is unavailable" >&2
  exit 2
fi

actual_commit="$(git -C "${repo_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${SAGA_LATE_FILTER_COMMIT}" ]]; then
  echo "workspace commit mismatch: ${actual_commit} != ${SAGA_LATE_FILTER_COMMIT}" >&2
  exit 2
fi
if ! git -C "${repo_root}" diff --quiet -- || \
   ! git -C "${repo_root}" diff --cached --quiet --; then
  echo "workspace has tracked or staged modifications" >&2
  exit 2
fi

data_root="${SAGA_LATE_FILTER_DATA_ROOT:-/root/autodl-tmp}"
available_kib="$(df -Pk "${data_root}" | awk 'NR==2 {print $4}')"
if (( available_kib < 80 * 1024 * 1024 )); then
  echo "data disk has less than 80 GiB available" >&2
  exit 2
fi

for name in memory.current memory.max memory.events; do
  if [[ ! -r "/sys/fs/cgroup/${name}" ]]; then
    echo "missing cgroup audit input: /sys/fs/cgroup/${name}" >&2
    exit 2
  fi
done
echo "memory.current=$(</sys/fs/cgroup/memory.current)"
echo "memory.max=$(</sys/fs/cgroup/memory.max)"
sed 's/^/memory.events /' /sys/fs/cgroup/memory.events
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
fi

if pgrep -af "[p]ython.*category_priors\.cli audit-clean-late-filters" >/dev/null; then
  echo "another clean late-filter audit is already active" >&2
  exit 2
fi

mkdir -p -- "${output_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${repo_root}"
exec "${SAGA_LATE_FILTER_PYTHON}" -m category_priors.cli \
  audit-clean-late-filters \
  --manifest "${manifest_path}" \
  --output-root "${output_root}"
