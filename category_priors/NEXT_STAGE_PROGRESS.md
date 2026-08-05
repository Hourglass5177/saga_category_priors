# First-stage next-phase checkpoint

Checkpoint time: 2026-08-06 00:40 (Asia/Hong_Kong)

## Completed before this checkpoint

- The cloud ScanNet minimal subset is complete: train `1201/1201`, validation
  `312/312`, four required nonempty files per scene, zero failed manifest rows,
  and zero `.part` files.
- Train/validation statistics, the train-only `category_priors.json`, the 24/48
  leakage-safe scene selection, and both 32-point search designs passed their
  content/hash audits under
  `/root/autodl-tmp/saga/artifacts/category-priors-20260804`.
- Canonical, axis-aligned SAGA20 ground truth was generated for all 24 val-tune
  scenes under `gt_val_tune`; its manifest records 24 scenes and occupies about
  95 MiB.
- The ScanNet-to-SAGA exporter, alignment audits, resumable wave runner, search
  execution glue, and adaptive 2,000-iteration feature training are committed
  and pushed on branch `a800`.
- The one-scene `scene0231_00` gate passed preparation, initial and trained
  alignment audits, 30k 3DGS training, masks/labels, mask scales, contrastive
  features, and scale-gate checks. Its postprocess results are exploratory only:
  B0 mAP50-95 `0.14722`, B1 `0.15389`, and P000-B2 `0.05972`.

## Current cloud execution

- Mihomo is active and an actual ScanNet range-transfer comparison selected
  Taiwan IEPL 03. It was the only tested node to return a complete 256 KiB
  sample in the observation window; latency alone is not used for selection.
- Commit `7015562` removed aria2's low-speed abort, retained resumable `.part`
  files, and made retries unlimited by default. The patched downloader is
  deployed in both clean and GPU workspaces.
- Wave 1 contains the first eight val-tune scenes. Four `.sens` streams are
  complete (`scene0231_00`, `scene0329_02`, `scene0474_01`, and
  `scene0025_01`). At the checkpoint, allocated-block progress for the remaining
  four was approximately 55% (`scene0025_02`), 14% (`scene0046_02`), 35%
  (`scene0645_00`), and 6% (`scene0645_02`). The logical length of an aria2
  sparse file must not be interpreted as downloaded bytes.
- The downloader and wave supervisor are healthy. The supervisor wait window
  was safely restarted at 48 hours and will automatically enter scene
  preparation and training when the hashed eight-scene manifest passes.
- The 350 GiB data disk has about 309 GiB free. GPU work is idle only while the
  wave waits for its official-resolution RGB streams.

## Mirror audit and experimental boundary

- `insomnia7/SIU3R` provides fast per-scene archives, but its RGB is resized to
  `256x256`; it is not admissible for the formal SAGA baseline.
- `WHB139426/Scannet` exposes a range-readable 99.1 GB frames ZIP with
  `1296x968` RGB, but it keeps only about 300 frames per scene and recompresses
  JPEGs. It is also not admissible as a silent replacement for official `.sens`
  input.
- `General-Level/General-Bench-Closeset` contains original `.sens` LFS objects.
  The mirrored `scene0231_00` has the same byte length and SHA-256
  (`615f0a816579bf77076ef0d7f4c5f2c9c84377b02a829b7e41e60a84b0c04574`)
  as the official local file. Five of the 24 selected scenes are present; four
  not already complete (`scene0025_00`, `scene0046_00`, `scene0356_00`, and
  `scene0608_00`) may be transported from this mirror in later waves only with
  fixed size/SHA-256 verification. All other scenes remain on the official
  downloader. The source route must never change experiment content.

## Resume sequence

1. Keep the current wave-1 aria2 process running and preserve every `.part` and
   `.aria2` checkpoint. Do not use `--clean`.
2. Require eight nonempty `.sens` files, no aria2 control files, and a complete
   hashed download manifest before accepting the wave.
3. Let the existing supervisor prepare and gate the remaining seven scenes
   sequentially. Every scene must pass preparation, initial/trained alignment,
   30k 3DGS, masks/labels, mask scales, 2,000-step feature, and scale-gate checks.
4. Expand the remaining 16 val-tune scenes in two bounded waves. A verified raw
   mirror may accelerate only the four byte-preserving scenes listed above.
5. After all 24 assets pass, run randomized resumable global search, select the
   global configuration on val-tune, run prior search, and freeze the mapping
   and experiment lock before touching val-locked.
6. Continue with registered baselines, factor ablations, locked evaluation, and
   executable cross-dataset validation. A scene or intermediate wave remains
   exploratory and cannot support an effectiveness claim.

Experimental unit remains the physical scene. Priors are fitted from train only,
search coefficients are selected on val-tune only, and the 48-scene val-locked
set remains untouched until the experiment lock is frozen.
