# Precision-first 3D Gaussian object audit

This audit answers the teacher's concrete question: are the Gaussians assigned
to a predicted object actually accurate?  It keeps the official ScanNet AP as
the primary benchmark but adds the missing prediction-side measurement.

## What is measured

The existing official path maps every GT vertex to its nearest Gaussian within
5 cm and evaluates the resulting prediction mask on GT support.  That remains
unchanged.  The new audit additionally maps every predicted foreground
Gaussian to its nearest valid GT point within the same 5 cm radius.  Unmapped,
void, wrong-class and wrong-instance Gaussians remain false positives in the
precision denominator.

For each predicted instance the audit reports Gaussian point precision,
semantic precision, unsupported fraction, dominant same-class GT instance,
GT-to-Gaussian recall, GT-support IoU, duplicate predictions and likely merges.
It also exports color-coded 3D PLY files.  No 2D rendering metric is calculated.

## Architecture comparison boundary

- Public SAGA is a promptable, scale-aware Gaussian segmentation method, not an
  automatic closed-set ScanNet instance AP baseline.  Its published numbers
  must not be compared directly with this project's B0/B1 AP.
- ObjectGS assigns object IDs in 2D and lifts object-aware information into 3D;
  its one-hot/object-aware design is relevant to auditing alpha-blending
  ambiguity, but its reconstruction and segmentation metrics are a different
  protocol.
- OpenGaussian is useful as a point-level semantic-feature and ScanNet
  evaluation reference, but its staged training and task are not a drop-in
  baseline for the current assets.
- The comparable engineering anchors remain the current a800-compatible B0 and
  B1 outputs.  The audit therefore compares those exact outputs first.

References: [SAGA](https://github.com/Jumpat/SegAnyGAussians),
[ObjectGS](https://github.com/RuijieZhu94/ObjectGS), and
[OpenGaussian](https://github.com/yanmin-wu/OpenGaussian).

## Interpretation

- Higher B1 recall with lower precision means the category branch restores
  support but introduces contamination or false-positive instances.
- Stable precision with improved small-object recall is evidence that the
  teacher's early category branch has value.
- High precision with low recall points to missing SAM/affinity support.
- Low precision with reasonable recall points to clustering, expansion, global
  KNN/filtering, or final class assignment.
- Low precision and recall means the candidate backbone is not healthy enough
  to judge data-driven category priors.
- Visually correct objects with low AP require a score, duplicate-prediction,
  metadata, or evaluator-mapping audit.

## Command

```bash
python -m category_priors audit-gaussian-objects \
  --scene-manifest /path/to/locked_scene_runtime.json \
  --gt-dir /path/to/gt \
  --runs-root /path/to/locked-runs \
  --scene scene0011_00 --scene scene0231_00 \
  --seed 42 \
  --table-output gaussian_to_gt_precision.parquet \
  --audit-output gaussian_object_audit.json \
  --comparison-output b0_b1_precision_comparison.json \
  --viewer-output viewer
```
