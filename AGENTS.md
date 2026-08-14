# Repository agent instructions

For any task involving SAGA category priors, teacher-prior experiments, small-object
protection, ScanNet tune/final evaluation, or continuation of the current research:

1. Read `category_priors/TEACHER_PRIOR_V3_EXPERIMENT_PLAN.md` **in full** before
   editing code, starting a run, or interpreting results.
2. Treat that document as the single source of truth for condition definitions,
   stage gates, experimental units, stopping rules, and the current checkpoint.
3. After completing a stage, update its "当前执行检查点" and "阶段记录" before
   proceeding.
4. Do not resume B2, class-first, prior-v2, branch-preservation, or selective-restore
   workflows unless the active plan explicitly reactivates them.

This file is only a discovery pointer. Do not duplicate the experiment plan here.
