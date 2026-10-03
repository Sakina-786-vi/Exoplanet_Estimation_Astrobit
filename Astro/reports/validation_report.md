# Validation Status

The corrected full run used TRAIN labels to select the score family and
decision threshold, then applied the frozen choice to DEV.

| Split | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 66 | 189 | 1 | 0.259 | 0.985 | 0.410 |
| DEV | 20 | 63 | 1 | 0.241 | 0.952 | 0.385 |

## Fix applied

Previously, threshold selection operated on raw scores while inference
compared calibrated confidence. That mismatch produced the earlier all-zero
recall run and then an all-positive tendency after the first threshold
adjustment. The runner now:

1. selects the score family and threshold in raw-score space using TRAIN OOF
   predictions;
2. makes the final prediction using that same raw-score space;
3. reports calibrated confidence separately.

The selected TRAIN configuration was:

- score source: statistical candidate confidence
- threshold: 0.19
- TRAIN OOF F1: 0.410

## Remaining issue

False positives remain too high on both splits. This is not a threshold-space
bug anymore; the current candidate features do not separate the supplied
classification labels well enough. The BLS/injection diagnostics also show
that the provided `train_truth.csv` and `dev_truth.csv` are a separate
injection experiment: their KIC IDs have zero overlap with the positive
classification labels. They must not be mixed for supervised evaluation.

The pipeline is therefore improved from zero recall and reproducibly
validated, but it is not yet a high-precision detector or submission-ready.
