# ASTRO Final Pipeline Validation

## Decision

The existing PRIVATE submission is **not replaced**. TRAIN/DEV evidence does
not demonstrate a sufficiently reliable precision improvement from a new
candidate-vetting classifier. The preserved production submission remains the
reference output.

One detector correction is retained: the duration grid now includes 36 hours.
This fixes a demonstrated representational problem without changing the BLS
algorithm or adding an arbitrary astrophysical cutoff.

## Root cause of over-detection

The detector produces a candidate for nearly every TRAIN and DEV star. The
frozen decision stage then accepts most candidates because the statistical
confidence threshold is low (`0.19`) and the candidate features have weak
positive/negative separability.

Rank-1 candidates show:

- frequent periods in the 300--400 day range, including exact 400-day
  boundary solutions;
- substantial overlap in SDE, local SNR, recurrence, transit count, depth,
  odd/even consistency, and event consistency;
- a strong old concentration at the 24-hour duration-grid boundary;
- large depths that are more common among negatives but are not separable
  enough for a validated hard cutoff;
- false positives with weak recurrence, gap involvement, or
  single-quarter/event dominance.

These results show that a BLS peak alone is not sufficient evidence of a
planet, but they do not support a single robust rejection rule on the supplied
labels.

The injection truth tables were not mixed into classification labels or used
for PRIVATE decisions.

## Exact changes

- Added 36 hours to `configs/final_config.json`.
- Added `period_boundary_proximity` and `duration_grid_boundary` candidate
  features.
- Preserved the existing preprocessing, optimized BLS, event measurements,
  score-space thresholding, and separate confidence calibration.
- Regenerated:
  `feature_diagnostics.csv`, `bls_diagnostics_summary.json`,
  `candidate_separability.csv`, `threshold_sweep.csv`,
  `model_comparison.csv`, and `dev_error_analysis.csv`.
- Kept experiments in `outputs_experimental/` and
  `outputs_experimental_repeat/`.
- Did not modify `outputs_private/` or `outputs_private_repeat/`.

## Algorithms evaluated

1. Existing statistical SDE confidence baseline.
2. Existing dependency-free logistic model with star-level TRAIN OOF scores.
3. Expanded deterministic logistic vetting variants using event consistency,
   log-depth/local-SNR transforms, boundary indicators, recurrence,
   single-quarter dominance, and duration-boundary indicators.

No XGBoost implementation was available. No PRIVATE labels or truth were
used. Calibration remains separate from the raw score used for thresholding.

## Metrics

### Existing frozen baseline

| Split | TP | FP | FN | Precision | Recall | F1 | Positive predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 66 | 189 | 1 | 0.259 | 0.985 | 0.410 | 255/269 |
| DEV | 20 | 63 | 1 | 0.241 | 0.952 | 0.385 | 83/89 |

### Expanded-feature TRAIN OOF comparison

| Model | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Statistical baseline | 54 | 140 | 13 | 0.278 | 0.806 | 0.414 | 0.259 |
| Dependency-free ML | 57 | 158 | 10 | 0.265 | 0.851 | 0.404 | 0.236 |

The dependency-free ML model was not selected because its OOF AP and F1 were
lower. Expanded nonlinear feature variants improved TRAIN ranking in some
settings but did not generalize to a meaningful DEV precision improvement.

### New experimental DEV run

Using the existing frozen statistical threshold (`0.19`) with the corrected
duration grid:

| TP | FP | FN | Precision | Recall | F1 | Positive predictions |
|---:|---:|---:|---:|---:|---:|---:|
| 21 | 61 | 0 | 0.256 | 1.000 | 0.408 | 82/89 |

This is only a modest improvement over the old DEV result and does not satisfy
the requested “substantially stronger” acceptance gate.

## Long-period and depth findings

Long-period candidates are not removed wholesale. Boundary proximity is
reported as a feature, allowing stronger evidence to be required in a future
validated model. Current TRAIN/DEV distributions do not justify a fixed
period cutoff.

Negative stars have higher median rank-1 depth than positives in TRAIN
(13,823 ppm versus 12,191 ppm), but the distributions overlap broadly.
Therefore no arbitrary depth threshold was added.

## Duration validation

Synthetic injections at 2h, 4h, 8h, 12h, 24h, and 36h recovered the matching
duration grid values. The previous 36h injection could only be reported as
24h; after the correction it is reported as 36h. This is a representational
fix, not evidence that all 36h candidates are planetary.

## PRIVATE and submission status

The existing frozen PRIVATE output remains:

- old PRIVATE positives: **79/87**
- new PRIVATE positives: **not run**
- PRIVATE labels/truth used: **no**
- preserved files: `outputs_private/submission.csv` and
  `outputs_private_repeat/submission.csv`

The experimental DEV outputs are not production submissions. The production
submission remains validated and was previously reproduced byte-for-byte.

## Final answer

**FINAL METHOD:** Existing statistical SDE decision baseline, preserved
preprocessing and optimized BLS, with 36-hour duration support retained in the
experimental/frozen configuration.

**TRAIN:** Existing baseline 0.259 precision, 0.985 recall, 0.410 F1.
Expanded TRAIN OOF alternatives did not provide a reliable generalization
advantage.

**DEV:** Existing baseline 0.241 precision, 0.952 recall, 0.385 F1. Corrected
duration-grid experiment: 0.256 precision, 1.000 recall, 0.408 F1.

**PRIVATE:** Preserved existing result, 79 positives; no new PRIVATE run.

**OLD PRIVATE POSITIVES:** 79/87.

**NEW PRIVATE POSITIVES:** Not generated because the new method was not
validated as a decisive improvement.

**SUBMISSION VALID:** Yes for the preserved production submission.

**DETERMINISTIC:** Yes; repeated experimental DEV predictions and submissions
were byte-for-byte identical.

**REPLACED OLD SUBMISSION:** No.

**WHY NEW SUBMISSION IS STRONGER:** It is not scientifically defensible to
claim that it is stronger on current evidence. The validated improvement is
limited to correct 36-hour duration representation; the high false-positive
rate and weak candidate separability remain unresolved, so the old submission
is correctly preserved.
