# First-stage next-phase checkpoint

Checkpoint time: 2026-08-05 (Asia/Hong_Kong)

## Completed before this checkpoint

- The cloud ScanNet minimal subset is complete: train `1201/1201`, validation
  `312/312`, four required nonempty files per scene, zero failed manifest rows,
  and zero `.part` files.
- Train/validation statistics, the train-only `category_priors.json`, the 24/48
  leakage-safe scene selection, and both 32-point search designs passed their
  content/hash audits under:
  `/root/autodl-tmp/saga/artifacts/category-priors-20260804`.
- Canonical, axis-aligned SAGA20 ground truth was generated for all 24 val-tune
  scenes at:
  `/root/autodl-tmp/saga/artifacts/category-priors-20260804/gt_val_tune`.
  Its manifest records 24 scenes and occupies about 95 MiB.
- The two-hour `scannet` heartbeat automation is paused because the registered
  download/statistics phase completed.

## Audit result for the next phase

- The cloud has a working RTX 4090 environment, about 317 GiB free on the 350 GiB
  data disk, and the Mihomo proxy remains active.
- Only the existing `lab` baseline scene has complete 3DGS/SAGA runtime assets.
  None of the 24 selected ScanNet val-tune scenes yet has RGB frames, camera
  models, a trained 3DGS point cloud, masks, mask scales, contrastive features,
  or a scale gate.
- Therefore the 32-configuration searches cannot start yet. The immediate gate
  is a one-scene ScanNet-to-SAGA end-to-end smoke test.
- The cloud working copy at `/root/autodl-tmp/saga/workspace/saga` is heavily
  dirty from the pre-existing deployment/build tree. Do not reset, clean, or
  overwrite it. Continue deploying research utilities through a separate clean
  archive/work directory.

## Local work in progress (not committed yet)

- Added `category_priors/scannet_saga.py` with a streaming `.sens` reader and a
  deterministic exporter that:
  - subsamples compressed color frames without decoding depth;
  - applies ScanNet `axisAlignment` to camera poses and mesh vertices;
  - writes a PINHOLE COLMAP text model plus an aligned initial point cloud;
  - records metric scale `1.0`, identity Gaussian-to-GT mapping, hashes, selected
    frames, intrinsics, and point counts in a scene preparation manifest.
- Added the CLI command `prepare-saga-scene` in `category_priors/cli.py`.
- Added `tests/category_priors/test_scannet_saga.py`.
- Installed local `plyfile` for testing. The new focused test passes:
  `1 passed`.

## Explicitly not started

- No selected-scene `.sens` download is running.
- No ScanNet RGB extraction, 3DGS training, Grounded-SAM mask generation, SAGA
  feature training, postprocess run, or parameter search is running.
- No new local changes from this checkpoint have been committed or pushed.

## Resume sequence

1. Review the uncommitted diff and run the full category-prior test suite.
2. Finish the missing search-execution glue: materialize each LHS configuration
   as a hashed mapping, randomize configurations within scene/seed blocks, and
   emit evaluator-ready manifests/metric rows.
3. Document the new scene-preparation and search commands.
4. Commit and push the tested implementation, then deploy it to a separate clean
   cloud work directory.
5. Download only `scene0231_00.sens` through the existing cloud proxy using an
   atomic resumable `.part` path; keep at least 80 GiB free.
6. Prepare `scene0231_00`, train its 3DGS, run SAGA prerequisite stages once, and
   audit coordinate/GT mapping before any 24-scene expansion.
7. Run paired baseline/global smoke postprocessing and evaluation. Expand in
   eight-scene waves only after the one-scene gate passes.

Experimental unit remains the physical scene. Tuning uses val-tune only; the
48-scene val-locked set remains untouched until the mapping and experiment lock
are frozen.
