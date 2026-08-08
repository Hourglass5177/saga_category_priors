# First-stage locked-evaluation checkpoint

Checkpoint time: 2026-08-09 (Asia/Hong_Kong)

## Completed evidence

- The official ScanNet minimal subset is complete: train `1201/1201`, validation
  `312/312`; priors were fitted from train only.
- All 24 val-tune SAGA assets are complete.
- The 32-point global and prior searches are complete. The registered selections
  are `global-003` and `prior-011`; the tune-only improvement is approximately
  `+0.00377 mAP@[.50:.95]` and remains exploratory.
- The cloud instance has a 350GB data disk with about 177GB free. Its actual
  cgroup memory limit is 90GiB; host-level `free` output is not a valid limit.
- No val-locked `.sens`, GT, 3DGS, feature, or postprocess output has been
  generated yet.

## Frozen design corrections

- The old 48-scan locked list contains only 32 independent physical scenes. The
  confirmatory list will instead choose 48 unique physical scenes solely from
  the old `locked + locked_replacements` pool. The original selection remains
  unchanged because the fitted mapping references it.
- The new selector uses only pre-model class counts and a deterministic coverage
  objective. It never reads predictions or AP values.
- ScanNet matching/AP, positive-weight physical-scene bootstrap, paired
  permutation, the full `2^3` factorial, Holm correction, and multi-seed
  technical-replicate handling are implemented and covered by tests.
- The locked workflow uses `locked_plan.json` plus `locked_progress.json` rather
  than a hash-heavy experiment-lock stack. Git commit and the mapping format's
  existing internal content hash are the only routine identities retained.

## Execution sequence

1. Re-evaluate all existing global-search outputs with the corrected official
   evaluator and rerun global selection. If the selected global configuration
   changes, rerun prior search under it; otherwise re-evaluate all existing
   prior-search outputs and rerun prior selection. Build the final mapping from
   those official-protocol selections.
2. Re-evaluate the selected seed-42 tune outputs with that same evaluator, then
   run only `P000-B2` and `P111-combined` for seeds 3407 and 20260804 (`96` new
   postprocess runs). Never mix old approximate metrics into this decision.
3. Apply the preregistered rule: use locked seed 42 only when both conditions'
   cross-seed ranges are at most `0.002` and all three `P111-P000` deltas have
   the same non-zero sign; otherwise retain all three seeds as technical
   replicates.
4. Freeze `locked_evaluation_scenes.json` and `locked_plan.json` before producing
   any locked model output.
5. Remove only the two completed search contributor caches (about 50GB). Preserve
   every search output, metric, selected config, model, and log.
6. Download the 48 official `.sens` streams in six waves of eight. In no-GPU mode
   downloading stops at complete readable files and never starts training.
7. In GPU mode, prepare each wave sequentially: 30k 3DGS, masks/labels, scales,
   2,000-step feature training, and one trained-Gaussian 5cm alignment. Delete a
   wave's raw `.sens` only after every scene in that wave passes.
8. Run all 12 registered conditions for every scene. Conditions share one
   contributor cache inside a scene; the cache is removed only after that
   scene's full block completes. Do not inspect comparative locked AP early.
9. Produce `locked_metrics.parquet` and `confirmatory_analysis.json`; only then
   select best/median/worst qualitative viewer examples.

The experimental unit is the physical scene. Seeds, scans, views, points,
Gaussians, and instances are not counted as independent experimental units.
Failure of the registered primary comparison is reported as failure or
insufficient evidence and is not followed by retuning on val-locked.
