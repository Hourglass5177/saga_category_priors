# Repository agent instructions

For work on the active SAGA category-prior study:

1. Read `category_priors/ITERATIVE_REFINEMENT_EXPERIMENT_STANDARD.md` before editing code,
   starting a run, or interpreting results.
2. Treat that document as the source of truth for the baseline identity, controlled
   variables, evaluation protocol, data-leakage boundary, and prohibited workflows.
3. The active question is the two-round, local 2D--3D refinement of the frozen
   all-SAGA20 branch candidates. Do not resume the retired V3--V10, ObjectBank,
   clean-baseline, prompt-scale, HDBSCAN-repair, full-instance-size, or Boolean-only
   recheck workflows.
4. Git history is the archive for retired experimental code. Do not add compatibility
   adapters or restore old state machines to the active runtime.
5. Ground truth may enter evaluation and post-hoc viewer selection only. It must not
   enter candidate construction, projection, crop generation, model inference, or
   replay decisions.

This file is only a discovery pointer; do not duplicate the baseline standard here.
