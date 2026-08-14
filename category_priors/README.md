# SAGA category-prior experiments

> **Active plan:** Before changing code or resuming any cloud experiment, read
> [TEACHER_PRIOR_V3_EXPERIMENT_PLAN.md](TEACHER_PRIOR_V3_EXPERIMENT_PLAN.md) in full.
> It is the single source of truth for the current category-prior and small-object
> protection study; the workflows below are retained as historical implementation
> notes unless that plan explicitly reactivates them.

This package implements the first-stage, postprocess-only study of category
priors for SAGA. It keeps the three information boundaries executable:

- ScanNet200 **train** scenes fit `category_priors.json`.
- The 24-scene **val-tune** set selects global and prior mapping coefficients.
- The 48-scene **val-locked** set is evaluated only after an experiment lock is
  written. ScanNet++ is a separately reported cross-dataset check.

The independent experimental unit is the physical scene, not a rendered view,
Gaussian, point, or repeated scan. The confirmatory analysis therefore resamples
physical-scene groups.

## Current teacher-style experiment

The active lightweight experiment keeps the trained SAGA assets and changes only
postprocessing. `original` means the `source/a800` function path on the current
32-class label-feature asset: its compatible B1 route is the eight classes
`switch, socket, book, remote, key, cup, vase, phone`. The five registered
teacher conditions (`U0-all-uniform`, `D-size`, `D-smooth`, `D-small`, and
`D-combined`) instead run the same class branch over the fixed SAGA20 evaluation
taxonomy. Extra entries in the 32-class codebook do not become teacher branches.

Stage 1 first materializes the table without branch preservation to measure the
unaltered a800 branch's order sensitivity, post-filter survival, and vote
agreement. If any preregistered threshold fires, rematerialize the one formal
table with `branch_preservation=True`; that defers every U0/D class branch until
after the unchanged legacy global KNN and 10-point filter. All subsequent U0/D
runs must use the same selected table.

```bash
python -m category_priors build-teacher-category-params \
  --category-priors artifacts/category_priors.json \
  --output artifacts/teacher_category_params_unprotected.json

# Run the Stage-1 diagnostics, then use this only if a preservation threshold fires.
python -m category_priors build-teacher-category-params \
  --category-priors artifacts/category_priors.json \
  --output artifacts/teacher_category_params.json \
  --branch-preservation

python -m category_priors run-teacher-prior \
  --scene-manifest artifacts/scene_runtime_smoke2.json \
  --output-root runs/teacher-prior-smoke2 \
  --teacher-category-params artifacts/teacher_category_params.json \
  --condition U0-all-uniform --condition D-combined --seed 42
```

U0 and every D condition must use that same materialized file. They therefore
share semantic selection, sampling, feature weights, explicit
`min_samples=3`, and branch preservation; a D condition changes only its named
train-statistics-derived factor. No scene download or model training belongs to
this workflow.

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
gate. Official downloads are accepted after the header and frame count are
readable; SHA-256 is reserved for explicitly approved third-party mirrors. The
exporter records `scene_scale_m_per_unit=1.0` and an identity
Gaussian-to-GT transform; these are accepted only after the one-scene mapping
audit passes. The audit writes diagnostics even when it fails and gates the
registered 5 cm GT coverage, identity transform, metric scale, and padded camera
trajectory. The wave runner performs one audit on the trained Gaussian point
cloud before postprocessing.

`run_scannet_scene_pipeline.sh` resumes at four nonempty-output gates: metric
3DGS, masks/labels, mask scales, and contrastive features/scale gate. It archives
an earlier stage log before retrying, records elapsed time/status per stage, and
never invokes `--clean`. When `--hf-home` is supplied, the GroundingDINO BERT
dependency is resolved from that cache with Hugging Face and Transformers
offline modes enabled by default.

After the one-scene gate passes, `run_scannet_tune_wave.sh` expands assets in
bounded waves. It optionally waits for a downloader status file, validates each
official `.sens` header, then sequentially prepares, trains, and audits each
scene while retaining completed stage outputs. `--gt-dir` selects tune or locked
GT, and `--delete-sens-after-success` removes a raw stream only after its trained
alignment passes. Use `--limit 8` for one bounded wave.
When Python `urllib` cannot obtain ScanNet response headers through the cloud
proxy, `download_scannet_sens_aria2.sh` provides bounded concurrent, atomic
`.part` resume and finishes by generating a downloader status file.
`--verified-url-map` may route explicitly listed scenes through a transport
mirror. The TSV must pin scene ID, URL, byte length, and SHA-256; both size and
hash are checked before atomic rename, while scenes absent from the map retain
the official ScanNet URL. Resized, recompressed, or resampled frame archives are
not valid substitutes for raw `.sens` input.

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

## Lightweight locked evaluation

The confirmatory path deliberately avoids the search workflow's hash-heavy
manifests. Existing search artifacts remain readable, while new locked runs use
one human-readable `locked_plan.json`, field-based resume, and one
`locked_progress.json`.

Before the seed audit, re-evaluate **all existing global-search outputs** with
the `scannet-official-instance-v1` evaluator and rerun `select-config`. If the
selected global configuration changes, materialize and rerun prior search under
that new global baseline. Otherwise re-evaluate all existing prior-search
outputs and rerun its selection directly. Build the final mapping from these
official-protocol selections. Seed 42 and both added seeds must then be evaluated
with that same protocol; old approximate metrics must never be mixed into the
six-row seed decision.

1. Select 48 independent physical scenes from the previously partitioned locked
   pool. The selector uses validation class counts for coverage only; it never
   reads model predictions or AP values.

   ```bash
   python -m category_priors select-locked-scenes \
     --stats artifacts/val_instance_stats.parquet \
     --scene-selection artifacts/scene_selection.json \
     --output artifacts/locked_evaluation_scenes.json
   ```

2. On `val-tune`, evaluate `P000-B2` and `P111-combined` at seeds 42, 3407,
   and 20260804. Combine the metric tables and apply the preregistered rule:

   ```bash
   python -m category_priors schedule \
     --scene-selection artifacts/scene_selection.json --split tune \
     --condition P000-B2 --condition P111-combined \
     --run-seed 3407 --run-seed 20260804 \
     --output artifacts/seed_audit_schedule.json

   python -m category_priors run-experiment \
     --schedule artifacts/seed_audit_schedule.json \
     --scene-manifest artifacts/scene_runtime_manifest.json \
     --output-root runs/prior-search \
     --output artifacts/seed_audit_execution.json \
     --pipeline run_pipeline.sh \
     --priors artifacts/category_priors.json \
     --mapping artifacts/final_mapping_config_official_v1.json

   python -m category_priors evaluate-seed-audit \
     --schedule artifacts/seed_audit_schedule.json \
     --execution artifacts/seed_audit_execution.json \
     --scene-manifest artifacts/scene_runtime_manifest.json \
     --gt-dir artifacts/gt_val_tune \
     --output artifacts/seed3407-20260804.parquet

   python -m category_priors assess-seeds \
     --metrics artifacts/seed42_p000_official_v1.parquet \
     --metrics artifacts/seed42_p111_official_v1.parquet \
     --metrics artifacts/seed3407-20260804.parquet \
     --output artifacts/seed_sensitivity_decision.json
   ```

   A single locked seed is used only when both conditions have a cross-seed
   `mAP@[.50:.95]` range at most 0.002 and `P111-P000` has the same non-zero sign
   for all three seeds. Otherwise all three seeds are retained as technical
   replicates.

3. Freeze and dry-run the plan before preparing locked model assets:

   ```bash
   python -m category_priors build-locked-plan \
     --locked-scenes artifacts/locked_evaluation_scenes.json \
     --seed-decision artifacts/seed_sensitivity_decision.json \
     --priors artifacts/category_priors.json \
     --mapping artifacts/prior_mapping_config.json \
     --output artifacts/locked_plan.json

   python -m category_priors run-locked \
     --plan artifacts/locked_plan.json \
     --scene-manifest artifacts/locked_scene_runtime.json \
     --output-root runs/locked \
     --progress artifacts/locked_progress.json \
     --pipeline run_pipeline.sh --dry-run
   ```

   A normal Git checkout is verified with `git rev-parse HEAD`. When the same
   committed files are deployed into a non-Git runtime directory, deployment
   writes that commit to `.category_priors_commit` beside `run_pipeline.sh`
   using a temporary file followed by rename. This readable marker is the only
   code identity used by the lightweight runner.

4. After every run is complete, run official-protocol metrics and the registered
   physical-scene analysis together:

   ```bash
   python -m category_priors evaluate-locked \
     --plan artifacts/locked_plan.json \
     --scene-manifest artifacts/locked_scene_runtime.json \
     --gt-dir artifacts/gt_val_locked \
     --output-root runs/locked \
     --metrics-output artifacts/locked_metrics.parquet \
     --analysis-output artifacts/confirmatory_analysis.json
   ```

The locked evaluator follows ScanNet's GT-first matching, strict IoU comparison,
void/small-region ignore rules, and AP integration. The experimental unit is the
physical scene. Seeds are technical replicates averaged inside each resample;
they are never counted as independent scenes.

## Proposal-first category priors (current workflow)

The current experiment keeps the validated a800 legacy pipeline as its
backbone: global instance proposals, filtering, and 2D semantic/background
voting remain unchanged. Category branches add proposals only to points left
unassigned by the global path. They never overwrite a global proposal and
never smooth across semantic classes. The five conditions share this exact
structure; only the train-derived category parameters differ:

- `L1-uniform`: fixed category parameters;
- `D-size`: category size controls physical spatial scaling and the scale gate;
- `D-smooth`: boundary statistics select a physical within-class vote radius;
- `D-small`: expected scene support controls sampled-domain cluster size and a
  conservative multi-anchor noise rescue;
- `D-combined`: enables all three mechanisms.

All runs are postprocess-only and reuse the existing 3DGS, SAGA features,
masks, cameras, and GT. The runner creates no schedule, lock, cache, or hash.
First write the readable parameter table:

```bash
python -m category_priors build-prior-v2-params \
  --category-priors artifacts/category_priors.json \
  --legacy-prior-config category_priors/default_legacy_prior_config.json \
  --output artifacts/legacy_prior_params.json
```

Run the two-scene smoke test with the uniform and combined conditions:

```bash
python -m category_priors run-prior-v2 \
  --scene-manifest artifacts/scene_runtime_manifest.json \
  --output-root runs/prior-v2-smoke \
  --category-priors artifacts/category_priors.json \
  --legacy-prior-config category_priors/default_legacy_prior_config.json \
  --scene scene0231_00 --scene scene0011_00 \
  --condition L1-uniform --condition D-combined --seed 42

python -m category_priors evaluate-prior-v2 \
  --scene-manifest artifacts/scene_runtime_manifest.json \
  --gt-dir artifacts/gt_val_tune \
  --output-root runs/prior-v2-smoke \
  --scene scene0231_00 --scene scene0011_00 \
  --condition L1-uniform --condition D-combined --seed 42 \
  --reference L1-uniform --treatment D-combined \
  --metrics-output artifacts/prior_v2_smoke_metrics.parquet \
  --analysis-output artifacts/prior_v2_smoke_analysis.json \
  --bootstrap-samples 1000 --split val-tune-smoke
```

Before tuning priors, use `diagnose-backbone` on existing legacy and failed
class-first outputs. It reports semantic coverage, GT/prediction instance
ratio, proposal recall, split/merge counts, and score/IoU correlation. The
eight-scene structural target is to recover near the existing B0/B1 AP and
coverage range; a tiny improvement over the failed class-first score is not a
success criterion.

Then run the five conditions on the existing 24 tune scenes at seed 42. Only
when a data-driven condition beats `L1-uniform` is it paired with the uniform
condition at seeds 3407 and 20260804. A positive three-seed mean with at least
two positive seed differences advances to the existing 48-scene assets. No
scene is downloaded or retrained.

## Deprecated class-first diagnostic workflow

This workflow is retained only to reproduce and diagnose the failed
`source/refactor` experiment. It is **not** the teacher's validated a800
baseline: its hard 32-way semantic split, destructive SOR, and class-local
propagation produced severe under-coverage and over-segmentation. Do not use it
as the performance backbone for category-prior experiments. It still reuses
existing assets and remains useful for failure analysis.

First materialize the readable 20-class parameter table. The default config is
`category_priors/default_class_first_config.json`; the output contains the
resolved `d_c`, `A_c`, `b_c`, `m_c`, `K_c`, and rescue radius without a hash.

```bash
python -m category_priors build-class-first-params \
  --category-priors artifacts/category_priors.json \
  --class-first-config category_priors/default_class_first_config.json \
  --output artifacts/class_first_params.json --mode combined
```

Use manifests containing exactly the intended scenes. Start with two scenes and
only the uniform teacher baseline plus the combined treatment:

```bash
python -m category_priors run-class-first \
  --scene-manifest artifacts/scene_runtime_smoke2.json \
  --output-root runs/class-first-smoke \
  --category-priors artifacts/category_priors.json \
  --class-first-config category_priors/default_class_first_config.json \
  --condition U0-uniform --condition D-combined --seed 42
```

For the fixed eight-scene calibration subset, first create three configs that
only set `min_samples` to `3`, `5`, or `10`, and run `U0-uniform` for each.
Freeze the value with the best official mAP (AP50, then the smaller value, break
an exact tie). Next create four configs using that frozen `min_samples` and one
pair from `rescue_radius_ratio in {0.10,0.20}` x
`small_area_exponent in {0.25,0.50}`. Run `D-small` for those four configs and
freeze the best pair with the same tie rule. Each config may contain only
`{"kind":"class_first_config"}` plus the fields being calibrated.

```bash
for cfg in artifacts/class-first-calibration/min-samples-*.json; do
  tag="$(basename "$cfg" .json)"
  python -m category_priors run-class-first \
    --scene-manifest artifacts/scene_runtime_calibration8.json \
    --output-root "runs/class-first-calibration/$tag" \
    --category-priors artifacts/category_priors.json \
    --class-first-config "$cfg" --condition U0-uniform --seed 42
  python -m category_priors evaluate-class-first \
    --scene-manifest artifacts/scene_runtime_calibration8.json \
    --gt-dir artifacts/gt_val_tune \
    --output-root "runs/class-first-calibration/$tag" \
    --condition U0-uniform --seed 42 \
    --metrics-output "artifacts/$tag-metrics.parquet" \
    --analysis-output "artifacts/$tag-analysis.json"
done

for cfg in artifacts/class-first-calibration/small-*.json; do
  tag="$(basename "$cfg" .json)"
  python -m category_priors run-class-first \
    --scene-manifest artifacts/scene_runtime_calibration8.json \
    --output-root "runs/class-first-calibration/$tag" \
    --category-priors artifacts/category_priors.json \
    --class-first-config "$cfg" --condition D-small --seed 42
  python -m category_priors evaluate-class-first \
    --scene-manifest artifacts/scene_runtime_calibration8.json \
    --gt-dir artifacts/gt_val_tune \
    --output-root "runs/class-first-calibration/$tag" \
    --condition D-small --seed 42 \
    --metrics-output "artifacts/$tag-metrics.parquet" \
    --analysis-output "artifacts/$tag-analysis.json"
done
```

Finally run all five registered conditions on the 24-scene tune manifest with
the frozen config, then evaluate once. If a data-driven condition beats
`U0-uniform`, rerun only that best condition and `U0-uniform` at seeds 3407 and
20260804. Enter the 48-scene evaluation only when the three-seed mean delta is
positive and at least two of three seed deltas are positive.

```bash
python -m category_priors run-class-first \
  --scene-manifest artifacts/scene_runtime_tune24.json \
  --output-root runs/class-first-tune24 \
  --category-priors artifacts/category_priors.json \
  --class-first-config artifacts/class_first_config_frozen.json \
  --seed 42

python -m category_priors evaluate-class-first \
  --scene-manifest artifacts/scene_runtime_tune24.json \
  --gt-dir artifacts/gt_val_tune \
  --output-root runs/class-first-tune24 \
  --metrics-output artifacts/class_first_tune24_metrics.parquet \
  --analysis-output artifacts/class_first_tune24_analysis.json \
  --seed 42
```

Pass the tune-selected condition as `--treatment` (with
`--reference U0-uniform`) only after selection; it need not be `D-combined`.
