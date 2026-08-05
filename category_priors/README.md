# SAGA category-prior experiments

This package implements the first-stage, postprocess-only study of category
priors for SAGA. It keeps the three information boundaries executable:

- ScanNet200 **train** scenes fit `category_priors.json`.
- The 24-scene **val-tune** set selects global and prior mapping coefficients.
- The 48-scene **val-locked** set is evaluated only after an experiment lock is
  written. ScanNet++ is a separately reported cross-dataset check.

The independent experimental unit is the physical scene, not a rendered view,
Gaussian, point, or repeated scan. The confirmatory analysis therefore resamples
physical-scene groups.

## Environment

Keep the CPU statistics environment separate from the CUDA SAGA environment:

```bash
python -m venv .venv-priors
source .venv-priors/bin/activate
pip install -r requirements-priors.txt
python -m category_priors --help
```

The SAGA CUDA environment additionally needs the repository's existing
dependencies and `hdbscan`. Do not pass `--clean`; masks, labels, features, JSON,
metadata, logs, and render artifacts are experimental evidence.

The licensed official ScanNet downloader can be wrapped without exposing its
restricted URL in logs. The command below accepts only the registered four-file
statistics subset, uses atomic resumable downloads, records hashes, and stops if
free space falls below 80GB:

```bash
python -m category_priors download-scannet \
  --official-downloader /secure/tools/download_scannetv2.py \
  --scene-list splits/scannetv2_train.txt --out-dir /data/scannet \
  --manifest artifacts/scannet_train_download.json --workers 4 --accept-tos
```

### 350GB cloud-disk policy

The current cloud data disk is 350GB (about 327GB free after the existing SAGA
environment and baseline). This is sufficient for the staged protocol, but not for
an indiscriminate full ScanNet release download:

- For train/validation statistics, download only `.aggregation.json`, `.txt`,
  `_vh_clean_2.ply`, and `_vh_clean_2.0.010000.segs.json`, plus the task split and
  label-map files. The extractor does not need RGB-D streams.
- Download `.sens` only for scenes already chosen for SAGA tune/locked execution;
  extract only the color frames, poses, and intrinsics needed by the reconstruction
  path.
- Execute SAGA scenes in waves (recommended: 8 scenes), checking free space before
  every wave. Keep at least 80GB free for temporary reconstructions, CUDA caches,
  and failure recovery.
- Retain run outputs, logs, manifests, hashes, masks, labels, features, and rendered
  evidence. Never delete raw inputs automatically; if input eviction becomes
  necessary, stop and make it an explicit, checksum-backed decision.

## Registered data flow

1. Extract train and validation tables separately. Never concatenate them before
   `fit`.

```bash
python -m category_priors extract \
  --dataset-root /data/scannet/scans \
  --scene-list splits/scannetv2_train.txt --split train \
  --label-map /data/scannet/scannetv2-labels.combined.tsv \
  --output artifacts/train_instance_stats.parquet \
  --audit-output artifacts/train_stats_audit.json --workers 8

python -m category_priors extract \
  --dataset-root /data/scannet/scans \
  --scene-list splits/scannetv2_val.txt --split val \
  --label-map /data/scannet/scannetv2-labels.combined.tsv \
  --output artifacts/val_instance_stats.parquet \
  --audit-output artifacts/val_stats_audit.json --workers 8
```

The TSV is hashed for provenance. Category inclusion is controlled only by
`default_taxonomy.json`: exact ScanNet200 labels or the explicitly registered safe
synonyms map into the 20-class protocol; all other labels remain void.

2. Fit train-only hierarchical priors, then make the leakage-safe 24/48 split.

```bash
python -m category_priors fit \
  --stats artifacts/train_instance_stats.parquet \
  --output artifacts/category_priors.json

python -m category_priors select-scenes \
  --stats artifacts/val_instance_stats.parquet \
  --output artifacts/scene_selection.json
```

3. Generate seeded search designs. Every metrics row passed to `select-config`
must carry `split=val-tune`; the command rejects any other split.

```bash
python -m category_priors search-design --kind global --samples 32 \
  --output artifacts/global_search_design.json
python -m category_priors search-design --kind prior --samples 32 \
  --output artifacts/prior_search_design.json

# Turn every global LHS row into a hashed executable mapping, then randomize the
# 32 configurations inside each scene/seed block (one tuning seed by default).
python -m category_priors materialize-search \
  --design artifacts/global_search_design.json \
  --priors artifacts/category_priors.json \
  --taxonomy category_priors/default_taxonomy.json \
  --scene-selection artifacts/scene_selection.json \
  --output-dir artifacts/global_mappings \
  --output artifacts/global_mapping_manifest.json
python -m category_priors search-schedule \
  --scene-selection artifacts/scene_selection.json \
  --mapping-manifest artifacts/global_mapping_manifest.json \
  --output artifacts/global_tune_schedule.json
python -m category_priors run-experiment \
  --schedule artifacts/global_tune_schedule.json \
  --scene-manifest artifacts/scene_runtime_manifest.json \
  --output-root runs/global-search --output runs/global-search/execution.json \
  --priors artifacts/category_priors.json
python -m category_priors evaluate-search \
  --schedule artifacts/global_tune_schedule.json \
  --execution runs/global-search/execution.json \
  --scene-manifest artifacts/scene_runtime_manifest.json \
  --gt-manifest artifacts/gt_val_tune/manifest.json \
  --output-dir artifacts/global_evaluation \
  --output artifacts/global_tune_metrics.parquet
python -m category_priors select-config --design artifacts/global_search_design.json \
  --metrics artifacts/global_tune_metrics.parquet --output artifacts/global_best.json

# Freeze the selected global values while retaining registered default prior
# coefficients, then repeat the same materialize/schedule/run/evaluate flow for
# the prior design with --base-mapping artifacts/global_mapping_config.json.
python -m category_priors build-mapping \
  --global-best artifacts/global_best.json \
  --priors artifacts/category_priors.json \
  --taxonomy category_priors/default_taxonomy.json \
  --scene-selection artifacts/scene_selection.json \
  --output artifacts/global_mapping_config.json
python -m category_priors materialize-search \
  --design artifacts/prior_search_design.json \
  --base-mapping artifacts/global_mapping_config.json \
  --priors artifacts/category_priors.json \
  --taxonomy category_priors/default_taxonomy.json \
  --scene-selection artifacts/scene_selection.json \
  --output-dir artifacts/prior_mappings \
  --output artifacts/prior_mapping_manifest.json
python -m category_priors select-config --design artifacts/prior_search_design.json \
  --metrics artifacts/prior_tune_metrics.parquet --output artifacts/prior_best.json
python -m category_priors build-mapping \
  --global-best artifacts/global_best.json --prior-best artifacts/prior_best.json \
  --priors artifacts/category_priors.json \
  --taxonomy category_priors/default_taxonomy.json \
  --scene-selection artifacts/scene_selection.json \
  --output artifacts/prior_mapping_config.json
python -m category_priors validate --priors artifacts/category_priors.json \
  --mapping artifacts/prior_mapping_config.json
python -m category_priors schedule --scene-selection artifacts/scene_selection.json \
  --split locked --output artifacts/locked_run_schedule.json
python -m category_priors run-experiment \
  --schedule artifacts/locked_run_schedule.json \
  --scene-manifest artifacts/scene_runtime_manifest.json \
  --output-root runs/category-priors --output runs/category-priors/execution.json \
  --priors artifacts/category_priors.json \
  --mapping artifacts/prior_mapping_config.json --dry-run
```

`select-config` requires every registered configuration and equal replicate
counts. Its default runtime tie-break tolerance is `0.002` on the `[0,1]` AP
scale, i.e. 0.2 AP points. Missing or non-finite metrics are rejected rather than
silently selecting from a partial search.

Before either search can run, prepare each already-selected ScanNet scene. The
registered exporter streams the `.sens` file, keeps only subsampled compressed
color frames, applies `axisAlignment` to both poses and mesh vertices, and emits
a metric COLMAP text model plus a hashed preparation manifest:

```bash
python -m category_priors download-scannet-saga \
  --official-downloader /secure/tools/download_scannetv2.py \
  --scene-list artifacts/tune_scenes.txt --out-dir /data/scannet \
  --manifest artifacts/tune_sens_download.json --workers 1 --accept-tos
python -m category_priors prepare-saga-scene \
  --dataset-root /data/scannet/scans --scene-id scene0231_00 \
  --sens /data/scannet/scans/scene0231_00/scene0231_00.sens \
  --output-root /data/saga_scannet --frame-stride 20 --max-frames 200
python -m category_priors audit-saga-alignment \
  --preparation-manifest /data/saga_scannet/scene0231_00/scene_preparation_manifest.json \
  --gt-npz artifacts/gt/scene0231_00.npz \
  --output artifacts/scene0231_00-initial-alignment.json
bash run_scannet_scene_pipeline.sh \
  --base-path /data/saga_scannet/scene0231_00 \
  --python /path/to/saga-env/bin/python \
  --hf-home /data/cache/huggingface --stage all
python -m category_priors audit-saga-alignment \
  --preparation-manifest /data/saga_scannet/scene0231_00/scene_preparation_manifest.json \
  --gt-npz artifacts/gt/scene0231_00.npz \
  --gaussian-ply /data/saga_scannet/scene0231_00/output_models/point_cloud/iteration_30000/scene_point_cloud.ply \
  --output artifacts/scene0231_00-trained-alignment.json
```

The `.sens` downloader uses a `.part` file, HTTP range resumption, a nonempty
final-file check, a sanitized failure manifest, and the same 80GB free-space
gate. The exporter records `scene_scale_m_per_unit=1.0` and an identity
Gaussian-to-GT transform; these are accepted only after the one-scene mapping
audit passes. The audit writes diagnostics even when it fails and gates the
registered 5 cm GT coverage, identity transform, metric scale, and padded camera
trajectory. Run it once on the prepared initial points and again on the trained
Gaussian point cloud before postprocessing.

`run_scannet_scene_pipeline.sh` resumes at four nonempty-output gates: metric
3DGS, masks/labels, mask scales, and contrastive features/scale gate. It archives
an earlier stage log before retrying, records elapsed time/status per stage, and
never invokes `--clean`. When `--hf-home` is supplied, the GroundingDINO BERT
dependency is resolved from that cache with Hugging Face and Transformers
offline modes enabled by default.

After the one-scene gate passes, `run_scannet_tune_wave.sh` expands tune assets
in bounded waves. It waits for a hashed `.sens` manifest, validates every target
file, then sequentially prepares, audits, trains, and re-audits each scene while
retaining completed stage outputs. Use `--limit 8` for the registered first wave.
When Python `urllib` cannot obtain ScanNet response headers through the cloud
proxy, `download_scannet_sens_aria2.sh` provides bounded concurrent, atomic
`.part` resume and finishes by generating the same hashed download manifest.

4. Run SAGA postprocessing. `scene_scale_m_per_unit` must be established by a
coordinate-alignment audit; it must not be tuned for AP.

Each runtime-manifest scene should also record the absolute `python_bin` for the
validated SAGA environment. The experiment runner passes it explicitly to
`run_pipeline.sh`, avoiding dependence on the supervisor shell's `PATH`.

```bash
bash run_pipeline.sh --stage postprocess --base-path /data/scene \
  --prior-config artifacts/category_priors.json \
  --prior-mapping-config artifacts/prior_mapping_config.json \
  --prior-mode combined --prior-gate on --prior-shrink on \
  --scene-scale-m-per-unit 1.0 --seed 42
```

Conditions are `B0-legacy`, `B1-other-classes`, `P000-B2`, the full `P000` through
`P111` size/smooth/small factorial, `P111-no-gate`, and `P111-no-shrink`. Run order
within a scene/seed block should be seeded and shuffled; every condition uses the
same input artifacts.

The executable mapping is:

| Condition | `--prior-mode` | Extra switches |
| --- | --- | --- |
| B0-legacy | `off` | `--disable-other-classes` |
| B1-other-classes | `off` | none |
| P000-B2 | `global` | prior files + metric scale |
| P001-small | `small` | prior files + metric scale |
| P010-smooth | `smooth` | prior files + metric scale |
| P011-smooth-small | `smooth-small` | prior files + metric scale |
| P100-size | `size` | prior files + metric scale |
| P101-size-small | `size-small` | prior files + metric scale |
| P110-size-smooth | `size-smooth` | prior files + metric scale |
| P111-combined | `combined` | prior files + metric scale |
| P111-no-gate | `combined` | `--prior-gate off` |
| P111-no-shrink | `combined` | `--prior-shrink off` |

5. Build canonical GT, evaluate, lock, and analyze. Start from the example
manifests in `configs/category_priors/`.

```bash
python -m category_priors prepare-gt --dataset-root /data/scannet/scans \
  --scene-list artifacts/locked_scenes.txt --output-dir artifacts/gt
python -m category_priors evaluate --manifest artifacts/evaluation_manifest.json \
  --output artifacts/P111_metrics.json
python -m category_priors lock --output artifacts/experiment_lock.json \
  --artifact priors=artifacts/category_priors.json \
  --artifact mapping=artifacts/prior_mapping_config.json \
  --artifact selection=artifacts/scene_selection.json \
  --artifact schedule=artifacts/locked_run_schedule.json \
  --artifact analysis=artifacts/analysis_manifest.json
python -m category_priors analyze --manifest artifacts/analysis_manifest.json \
  --output artifacts/confirmatory_analysis.json --bootstrap-samples 10000
```

Primary endpoint: ScanNet-style `mAP@0.50:0.95`; secondary endpoints: `AP50`,
`AP25`, per-class AP, runtime, and failure rate. Hypotheses are supported only if
the locked comparison improves the primary endpoint with a physical-scene
bootstrap interval that excludes zero; exploratory tune-set gains remain
exploratory. The factorial report includes all three main effects, all three
two-factor interactions, and the three-factor interaction, with one Holm family
across those seven registered contrasts.
