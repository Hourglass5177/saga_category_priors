# Repository instructions

- Preserve the teacher's original files from `source/a800` (`8c5e167`), including
  `README.md`, `CLAUDE.md`, `command.txt`, third-party documentation and licenses.
- Current research entry points and data locations are in `category_priors/README.md`.
  Keep documentation short. Do not recreate audit reports, review ladders or
  abandoned experiment controllers. Git history contains earlier tracked work.
- Preserve original annotations, frozen B0/C0 inputs, evaluation denominators,
  negative results and the cumulative GPU ledger. Ground truth and human answers
  belong in evaluation, never in automatic model decisions.
- Use focused tests for the changed behavior. GPU experiments require the existing
  budget accounting; the authorized DEV2 budget is cumulative, not reset per run.
  DEV8 execution remains outside the authorized experiment scope.
- Do not commit datasets, model weights, generated reports, deployment bundles,
  source checkpoints or runtime artifacts.
