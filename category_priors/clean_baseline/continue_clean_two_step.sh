#!/usr/bin/env bash
set -euo pipefail

: "${SAGA_TWO_STEP_MANIFEST:?set SAGA_TWO_STEP_MANIFEST}"
: "${SAGA_TWO_STEP_ARTIFACTS:?set SAGA_TWO_STEP_ARTIFACTS}"
: "${SAGA_TWO_STEP_RUNS:?set SAGA_TWO_STEP_RUNS}"
: "${SAGA_TWO_STEP_COMMIT:?set SAGA_TWO_STEP_COMMIT}"
: "${SAGA_TWO_STEP_PYTHON:?set SAGA_TWO_STEP_PYTHON to the registered environment}"

if [[ ! "${SAGA_TWO_STEP_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SAGA_TWO_STEP_COMMIT must be a full lowercase commit" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
manifest_path="$(readlink -f -- "${SAGA_TWO_STEP_MANIFEST}")"
artifacts_root="$(readlink -m -- "${SAGA_TWO_STEP_ARTIFACTS}")"
runs_root="$(readlink -m -- "${SAGA_TWO_STEP_RUNS}")"
python_bin="$(readlink -f -- "${SAGA_TWO_STEP_PYTHON}")"

if [[ ! -f "${manifest_path}" || ! -x "${python_bin}" ]]; then
  echo "registered manifest or Python executable is unavailable" >&2
  exit 2
fi

actual_commit="$(git -C "${repo_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${SAGA_TWO_STEP_COMMIT}" ]]; then
  echo "workspace commit mismatch: ${actual_commit} != ${SAGA_TWO_STEP_COMMIT}" >&2
  exit 2
fi
if ! git -C "${repo_root}" diff --quiet -- || \
   ! git -C "${repo_root}" diff --cached --quiet --; then
  echo "workspace has tracked or staged modifications" >&2
  exit 2
fi

data_root="${SAGA_TWO_STEP_DATA_ROOT:-/root/autodl-tmp}"
available_kib="$(df -Pk "${data_root}" | awk 'NR==2 {print $4}')"
if (( available_kib < 80 * 1024 * 1024 )); then
  echo "data disk has less than 80 GiB available" >&2
  exit 2
fi

if [[ -r /sys/fs/cgroup/memory.current ]]; then
  echo "memory.current=$(</sys/fs/cgroup/memory.current)"
  echo "memory.max=$(</sys/fs/cgroup/memory.max)"
  sed 's/^/memory.events /' /sys/fs/cgroup/memory.events
fi
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

if pgrep -af "[p]ython.*category_priors\.cli run-clean-baseline-two-step" >/dev/null; then
  echo "another clean two-step runner is already active" >&2
  exit 2
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd -- "${repo_root}"
"${python_bin}" - <<'PY'
import torch
import segment_anything  # noqa: F401
import gaussian_renderer  # noqa: F401

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the registered Python environment")
properties = torch.cuda.get_device_properties(0)
if (properties.major, properties.minor) < (12, 0):
    raise SystemExit(
        f"expected an RTX 5090 class sm_120 GPU, got sm_{properties.major}{properties.minor}"
    )
print(f"runtime-preflight gpu={properties.name} sm={properties.major}{properties.minor}")
PY

exec "${python_bin}" -m category_priors.cli run-clean-baseline-two-step \
  --manifest "${manifest_path}" \
  --output-root "${artifacts_root}" \
  --run-root "${runs_root}" \
  --producer-commit "${SAGA_TWO_STEP_COMMIT}"
