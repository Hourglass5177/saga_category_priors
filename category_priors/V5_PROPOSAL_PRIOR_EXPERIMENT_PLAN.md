# SAGA V5: proposal-quality and proposal-level category-prior revalidation

## Fixed method boundary

V5 leaves the existing a800-compatible B1 result untouched while generating
sidecar proposal banks.  It does not invoke B2, class-first, prior-v2, global
KNN, `filter_num`, a second vote, halo recovery, carve-out, or contributor
caching for the new candidates.

Each candidate source competes over all 32 SAGA codebook labels.  Only a
SAGA20 winner can enter a branch.  Branch HDBSCAN is frozen to threshold 0.7,
`min_cluster_size=min_samples=5`, cap 5000, 0.5/0.3/0.2 feature/spatial/
semantic weights, epsilon .01, and assignment threshold .3.  Deterministic
per-scene/seed/class sampling is shared by both sources.

Sources are `codebook` and `multiview`.  The latter streams existing labelled
SAM masks over max-contributor Gaussians, requires three labelled views,
winner ratio .60 and margin .10, and retains no contributor cache.

## Candidate bank and replay

The bank stores `output.json` (frozen B1), candidate statistics, and compact
full/core labels.  Core requires branch label, assignment confidence .70, and
the source’s branch-class support.  The bank is created before any V5 scoring.

CPU replay evaluates exactly four score arms: `U00-uniform`, `D10-size`,
`D01-core`, and `D11-combined`.  Their candidate IDs, core labels, votes,
HDBSCAN outputs and B1 labels are identical.  Only (G) (one-sided sorted
extent compatibility) and (C) (density-calibrated core compatibility) vary.
All arms use the same evidence gates, NMS and score threshold .20.

A proposal can complete background points of an overlapping same-class B1
instance, or form a new core-only instance when it has no B1 overlap above
.25.  It never overwrites B1 foreground and rejects an overlap with an
other-class B1 instance above .25.

## Registered funnel

1. Two scenes, both sources: B1 must remain pointwise identical and both banks
   must parse.
2. Development eight scenes: select a source only if its registered relative
   and absolute candidate-recall/precision gates pass.  Failure stops V5.
3. CPU replay on the eight: `U00` must be structurally safe versus B1; select
   a data-driven arm only at the registered mAP/tiny-small threshold.
4. If the source has enough positives, U replay is safe, and all D arms fail,
   fit only the fixed L2 balanced logistic fallback on these eight scenes;
   validate on the disjoint remaining sixteen.
5. Run the remaining sixteen, then technical seeds, then the existing 48
   internal-validation scenes only after each registered gate passes.

The 48-scene result is internal validation, not an untouched external test;
it is never used to alter V5 thresholds or parameters.
