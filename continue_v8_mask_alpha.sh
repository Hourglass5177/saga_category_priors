#!/usr/bin/env bash

set -euo pipefail

workspace="/root/autodl-tmp/saga/workspace/v8-mask-alpha"
artifacts="/root/autodl-tmp/saga/artifacts/v8-mask-alpha"
runs="/root/autodl-tmp/saga/runs/v8-mask-alpha"
controller_python="/root/autodl-tmp/saga/venvs/category-priors/bin/python"

mkdir -p "$artifacts" "$runs"
export PYTHONPATH="${workspace}${PYTHONPATH:+:${PYTHONPATH}}"

echo "v8_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
df -BG /root/autodl-tmp
echo -n "cgroup_memory_current="
cat /sys/fs/cgroup/memory.current
echo -n "cgroup_memory_max="
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

exec "$controller_python" -m category_priors.v8_pipeline \
  --runtime-manifest /root/autodl-tmp/saga/artifacts/category-priors-20260804/scene_runtime_manifest.json \
  --gt-dir /root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune \
  --locked-runtime-manifest /root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_scene_runtime.json \
  --locked-gt-dir /root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_locked \
  --locked-evaluation-scenes /root/autodl-tmp/saga/artifacts/category-priors-20260804/locked_evaluation_scenes.json \
  --category-priors /root/autodl-tmp/saga/artifacts/category-priors-20260804/category_priors.json \
  --size-bins /root/autodl-tmp/saga/artifacts/teacher-prior-v3-83be512/v3_gt_size_bins.json \
  --label-features /root/autodl-tmp/saga/runs/baseline_20260731_rtx4090/outputs/labels/label_features.pt \
  --sam-checkpoint /root/autodl-tmp/saga/workspace/saga/weights/sam_vit_h_4b8939.pth \
  --sam-masks-root "$runs/sam-everything" \
  --pipeline "$workspace/run_pipeline.sh" \
  --repo-root "$workspace" \
  --artifacts "$artifacts" \
  --runs "$runs" \
  --v7-status /root/autodl-tmp/saga/artifacts/v7-object-tracks/v7_status.json \
  --historical-b1-root /root/autodl-tmp/saga/runs/teacher-prior-v3-7bd8be8/stage2-b1-gate/original \
  --historical-b1-final-root /root/autodl-tmp/saga/runs/locked-confirmatory/B1-other-classes \
  --fixed-b1-root /root/autodl-tmp/saga/runs/v7-object-tracks/causal2/L0
