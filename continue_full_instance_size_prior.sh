#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace"

if ! git diff --quiet -- || ! git diff --cached --quiet --; then
    echo "tracked worktree changes are not allowed for a registered run" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

python_bin="${PYTHON_BIN:-/root/autodl-tmp/saga/venvs/pmr3-cu128/bin/python}"
data_root="${SAGA_DATA_ROOT:-/root/autodl-tmp}"
runtime_manifest="${SAGA_TUNE_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
locked_runtime_manifest="${SAGA_LOCKED_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_scene_runtime.json}"
t1_root="${SAGA_T1_ROOT:-/root/autodl-tmp/saga/runs/v9-objectbank/t1-legacy}"
gt_dir="${SAGA_TUNE_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
locked_gt_dir="${SAGA_LOCKED_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_locked}"
train_stats="${SAGA_TRAIN_STATS:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/train_instance_stats.parquet}"
category_priors="${SAGA_CATEGORY_PRIORS:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/category_priors.json}"
size_bins="${SAGA_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
locked_scenes="${SAGA_LOCKED_SCENES:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_evaluation_scenes.json}"
git_commit="$(git rev-parse HEAD)"
short_commit="$(git rev-parse --short=7 HEAD)"
runs_root="${SAGA_RUNS_ROOT:-/root/autodl-tmp/saga/runs/full-instance-size-${short_commit}}"
artifacts_root="${SAGA_ARTIFACTS_ROOT:-/root/autodl-tmp/saga/artifacts/full-instance-size-${short_commit}}"

for required in \
    "$python_bin" \
    "$runtime_manifest" \
    "$locked_runtime_manifest" \
    "$gt_dir" \
    "$locked_gt_dir" \
    "$train_stats" \
    "$category_priors" \
    "$size_bins" \
    "$locked_scenes"; do
    if [[ ! -e "$required" ]]; then
        echo "missing required registered input: $required" >&2
        exit 2
    fi
done

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
if [[ -z "$available_kib" || "$available_kib" -lt 83886080 ]]; then
    echo "data disk has less than 80 GiB available" >&2
    exit 3
fi

if [[ -r /sys/fs/cgroup/memory.current ]]; then
    echo "memory.current=$(</sys/fs/cgroup/memory.current)"
fi
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    echo "memory.max=$(</sys/fs/cgroup/memory.max)"
fi
if [[ -r /sys/fs/cgroup/memory.events ]]; then
    echo "memory.events:"
    sed 's/^/  /' /sys/fs/cgroup/memory.events
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi

mkdir -p "$runs_root" "$artifacts_root"
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=42
export PYTHONPATH="$workspace/submodules/diff-gaussian-rasterization:$workspace/submodules/diff-gaussian-rasterization-max-contributor:$workspace${PYTHONPATH:+:$PYTHONPATH}"

# Fail before a long run if the selected environment cannot reproduce the
# corrected contributor contract.  Importing the extension is not enough:
# the tiny render below is deliberately chosen so alpha*T_prev and the old
# alpha*T_new implementation select different Gaussians at the centre pixel.
"$python_bin" - "$workspace" <<'PY'
import math
import sys
from pathlib import Path

import pandas  # noqa: F401 - deployment dependency check
import pyarrow  # noqa: F401 - parquet engine dependency check
import torch

import diff_gaussian_rasterization
import diff_gaussian_rasterization_max_contributor
from diff_gaussian_rasterization import _C as _base_extension  # noqa: F401
from diff_gaussian_rasterization_max_contributor import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diff_gaussian_rasterization_max_contributor import (
    _C as _contributor_extension,  # noqa: F401
)
from gaussian_renderer import (  # noqa: F401 - exercise the production import path
    render_with_max_contributor as _render_with_max_contributor,
)

workspace = Path(sys.argv[1]).resolve()
expected_packages = {
    "diff_gaussian_rasterization": (
        workspace / "submodules/diff-gaussian-rasterization"
    ).resolve(),
    "diff_gaussian_rasterization_max_contributor": (
        workspace / "submodules/diff-gaussian-rasterization-max-contributor"
    ).resolve(),
}
for name, module in (
    ("diff_gaussian_rasterization", diff_gaussian_rasterization),
    (
        "diff_gaussian_rasterization_max_contributor",
        diff_gaussian_rasterization_max_contributor,
    ),
):
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(expected_packages[name])
    except ValueError as exc:
        raise RuntimeError(
            f"{name} resolved outside the registered worktree: {module_path}"
        ) from exc

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")
device = torch.device("cuda:0")
identity = torch.eye(4, dtype=torch.float32, device=device)
settings = GaussianRasterizationSettings(
    image_height=9,
    image_width=9,
    tanfovx=1.0,
    tanfovy=1.0,
    bg=torch.zeros(3, dtype=torch.float32, device=device),
    scale_modifier=1.0,
    viewmatrix=identity,
    projmatrix=identity,
    sh_degree=0,
    campos=torch.zeros(3, dtype=torch.float32, device=device),
    prefiltered=False,
    debug=False,
)
rasterizer = GaussianRasterizer(raster_settings=settings)
means3d = torch.tensor(
    [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
    dtype=torch.float32,
    device=device,
)
(
    _rendered,
    fixed_ids,
    fixed_weights,
    historical_ids,
    historical_weights,
    radii,
) = rasterizer(
    means3D=means3d,
    means2D=torch.zeros_like(means3d),
    opacities=torch.tensor([[0.30], [0.50]], device=device),
    colors_precomp=torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    ),
    scales=torch.full((2, 3), 0.05, device=device),
    rotations=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], device=device
    ),
    cov3D_precomp=None,
)
torch.cuda.synchronize(device)
centre = (4, 4)
if not bool(torch.all(radii > 0)):
    raise RuntimeError("contributor smoke Gaussians were not rasterized")
if int(fixed_ids[centre].item()) != 1:
    raise RuntimeError(
        "corrected contributor smoke failed: alpha*T_prev did not select Gaussian 1"
    )
if int(historical_ids[centre].item()) != 0:
    raise RuntimeError(
        "contributor smoke is non-discriminating: historical arm did not select Gaussian 0"
    )
if not math.isclose(float(fixed_weights[centre].item()), 0.35, abs_tol=1e-5):
    raise RuntimeError("corrected contributor centre weight is not alpha*T_prev")
if not math.isclose(float(historical_weights[centre].item()), 0.21, abs_tol=1e-5):
    raise RuntimeError("historical contributor centre weight is not alpha*T_new")
if int(fixed_ids[0, 0].item()) != -1 or float(fixed_weights[0, 0].item()) != 0.0:
    raise RuntimeError("empty contributor pixels must return id=-1 and weight=0")
print(
    "deployment preflight passed: pandas, pyarrow, CUDA, worktree rasterizers, "
    "and corrected contributor semantics"
)
PY

exec "$python_bin" -m category_priors run-full-instance-size-prior \
    --workspace "$workspace" \
    --runtime-manifest "$runtime_manifest" \
    --locked-runtime-manifest "$locked_runtime_manifest" \
    --t1-root "$t1_root" \
    --rebuild-t1-root "$runs_root/t1-rebuild" \
    --gt-dir "$gt_dir" \
    --locked-gt-dir "$locked_gt_dir" \
    --train-stats "$train_stats" \
    --category-priors "$category_priors" \
    --size-bins "$size_bins" \
    --locked-evaluation-scenes "$locked_scenes" \
    --runs-root "$runs_root" \
    --artifacts-root "$artifacts_root" \
    --git-commit "$git_commit" \
    --python-bin "$python_bin" \
    --allow-rebuild-missing-traces \
    --disk-floor-gib 80 \
    "$@"
