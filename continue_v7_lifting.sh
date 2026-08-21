#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${V7_PYTHON:-/root/autodl-tmp/saga/venvs/category-priors/bin/python}"
artifact_root="${V7_ARTIFACT_ROOT:-/root/autodl-tmp/saga/artifacts/v7-object-tracks}"
run_root="${V7_RUN_ROOT:-/root/autodl-tmp/saga/runs/v7-object-tracks}"
runtime_manifest="${V7_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json}"
gt_dir="${V7_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune}"
locked_runtime_manifest="${V7_LOCKED_RUNTIME_MANIFEST:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_scene_runtime.json}"
locked_gt_dir="${V7_LOCKED_GT_DIR:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_locked}"
category_priors="${V7_CATEGORY_PRIORS:-/root/autodl-tmp/saga/artifacts/category-priors-20260804/category_priors.json}"
size_bins="${V7_SIZE_BINS:-/root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json}"
historical_b1_root="${V7_HISTORICAL_B1_ROOT:-/root/autodl-tmp/saga/runs/teacher-prior-v3-7bd8be8/stage2-b1-gate/original}"

mkdir -p "$artifact_root" "$run_root"
export PYTHONPATH="$repo_root/submodules/diff-gaussian-rasterization-max-contributor:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m category_priors.v7_pipeline \
  --runtime-manifest "$runtime_manifest" \
  --gt-dir "$gt_dir" \
  --locked-runtime-manifest "$locked_runtime_manifest" \
  --locked-gt-dir "$locked_gt_dir" \
  --category-priors "$category_priors" \
  --size-bins "$size_bins" \
  --historical-b1-root "$historical_b1_root" \
  --pipeline "$repo_root/run_pipeline.sh" \
  --repo-root "$repo_root" \
  --artifacts "$artifact_root" \
  --runs "$run_root"
