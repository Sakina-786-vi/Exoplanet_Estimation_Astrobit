"""
preprocessing/config.py
=======================
Central configuration for all preprocessing parameters.

All tuneable thresholds live here.  Every science module imports from
this file so that a single edit propagates everywhere.

LEAKAGE RULE
------------
Parameters that are learned from data (e.g. per-star medians) are
learned independently per star and never shared across stars.
Parameters selected by grid search were tuned on TRAIN only.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots (adjust if you move data directories)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_DIR     = PROJECT_ROOT / "train_pack" / "train"
DEV_DIR       = PROJECT_ROOT / "dev_pack"   / "dev"
TRAIN_LABELS  = PROJECT_ROOT / "train_pack" / "train_labels.csv"
TRAIN_TRUTH   = PROJECT_ROOT / "train_pack" / "train_truth.csv"
DEV_LABELS    = PROJECT_ROOT / "dev_pack"   / "dev_labels.csv"
DEV_TRUTH     = PROJECT_ROOT / "dev_pack"   / "dev_truth.csv"

# Output directories
OUTPUT_ROOT    = PROJECT_ROOT / "data" / "processed"
QC_DIR         = OUTPUT_ROOT / "qc"
NORM_DIR       = OUTPUT_ROOT / "normalized"
DETREND_DIR    = OUTPUT_ROOT / "detrended"
DIAG_DIR       = PROJECT_ROOT / "data" / "diagnostics"
PLOT_DIR       = PROJECT_ROOT / "data" / "plots"

# ---------------------------------------------------------------------------
# Kepler cadence
# ---------------------------------------------------------------------------
KEPLER_CADENCE_DAYS  = 29.4 / (60 * 24)   # ~0.0204 days
KEPLER_CADENCE_MIN   = 29.4                 # minutes

# ---------------------------------------------------------------------------
# Quality filtering
# ---------------------------------------------------------------------------
# Only retain cadences where quality == 0.
# This is the official challenge definition of "good" measurements.
QUALITY_GOOD_VALUE = 0

# ---------------------------------------------------------------------------
# Minimum viable data per star / per quarter
# ---------------------------------------------------------------------------
MIN_POINTS_STAR    = 1000   # Flag star if fewer clean points remain
MIN_POINTS_QUARTER = 50     # Flag small quarters; do not discard by default
DROP_SMALL_QUARTERS = False # Preserve valid observations from short quarters

# ---------------------------------------------------------------------------
# Time axis quality control
# ---------------------------------------------------------------------------
# A gap is defined relative to the expected cadence so internal gaps and
# quarter boundaries are treated consistently without hard-coding one mission gap.
EXPECTED_CADENCE_DAYS = KEPLER_CADENCE_DAYS
GAP_FACTOR = 3.0
LARGE_GAP_THRESHOLD_DAYS = EXPECTED_CADENCE_DAYS * GAP_FACTOR

# ---------------------------------------------------------------------------
# Outlier policy (Task 7)
# ---------------------------------------------------------------------------
# We use a TWO-STAGE conservative approach:
#   Stage 1 – flag isolated POSITIVE (bright) outliers only
#              (instrumental cosmic rays / saturation spikes)
#   Stage 2 – flag isolated NEGATIVE outliers ONLY if they are
#              well above local scatter AND completely isolated
#              (a single bad cadence, not a coherent dip)
#
# NEVER remove a group of consecutive downward points as an outlier.
# Transit signals are coherent downward dips.
#
# Thresholds are intentionally very conservative (8-sigma).
OUTLIER_SIGMA_POSITIVE   = 8.0   # upper spike rejection
OUTLIER_SIGMA_NEGATIVE   = 8.0   # lower isolated-point diagnostic flag
OUTLIER_ISOLATION_WINDOW = 5     # must be isolated: check ±N neighbors
OUTLIER_MIN_COHERENT_LEN = 3     # >=3 consecutive low points → protect (transit candidate)

# ---------------------------------------------------------------------------
# Detrending windows to evaluate (days)
# ---------------------------------------------------------------------------
# Kepler transit durations for Earth-like planets around Sun-like stars
# range from ~3 h to ~13 h (0.125 – 0.54 days).
# A detrending window must be MUCH longer than the transit duration to avoid
# absorbing the transit into the baseline.
#
# Safe minimum: 3× the longest expected transit duration ≈ 3 × 0.55 ≈ 1.65 d
# We test a range from 1 day to 5 days to find the best trade-off.
DETREND_MIN_WINDOW_DAYS = 1.65
DETREND_DEFAULT_WINDOW_DAYS = 2.0
DETREND_MAX_WINDOW_DAYS = 5.0
DETREND_WINDOWS_DAYS = [1.65, 2.0, 3.0, 5.0]

# Default / selected window (updated after comparison run)
DETREND_WINDOW_SELECTED_DAYS = DETREND_DEFAULT_WINDOW_DAYS

# Savitzky-Golay polynomial order
SAVGOL_POLYORDER = 3

# LOWESS / Biweight fraction of data used per local fit
# (tuned per quarter length; overridden dynamically)
LOWESS_FRAC = 0.05   # 5% of quarter points per local fit

# ---------------------------------------------------------------------------
# Detrending methods to evaluate
# ---------------------------------------------------------------------------
DETREND_METHODS = [
    "robust_rolling_median",    # baseline: rolling median per quarter
    "savgol",                   # Savitzky-Golay polynomial smoothing
    "iterative_clip_median",    # iterative rolling median with transit masking
    "adaptive_window",          # window scaled to expected transit duration
]

ENABLE_EXPENSIVE_DETREND = False  # O(N^2) LOWESS is research-only by default

# Keep boundary cadences visible to later quality review rather than deleting them.
EDGE_FLAG_ENABLED = True

# Selected method after evaluation (updated after comparison run)
DETREND_METHOD_SELECTED = "robust_rolling_median"

# ---------------------------------------------------------------------------
# Transit injection testing (Task 26)
# ---------------------------------------------------------------------------
INJECTION_DEPTHS_PPM   = [100, 200, 400, 800]   # ppm
INJECTION_PERIODS_DAYS = [10.0, 50.0, 150.0, 300.0]
INJECTION_DURATIONS_HRS = [2.0, 4.0, 8.0, 12.0, 24.0, 36.0]
INJECTION_DURATION_HRS = 8.0    # Backward-compatible default duration
INJECTION_PHASES = ["middle", "segment_start", "segment_end", "before_gap", "after_gap", "quarter_boundary"]
# These labels are reporting gates, not training targets. They make attenuation
# categories explicit without silently declaring a signal scientifically lost.
INJECTION_PRESERVED_MIN_RATIO = 0.80
INJECTION_PARTIAL_MIN_RATIO = 0.50
INJECTION_DISTORTION_JUMP_PPM = 500.0
N_INJECTION_STARS      = 20     # number of TRAIN stars to test on

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
# Robust statistic for per-quarter baseline.
# "median" is the standard Kepler approach.
NORM_STATISTIC = "median"   # options: "median", "biweight_location"

# ---------------------------------------------------------------------------
# Diagnostics output
# ---------------------------------------------------------------------------
SAVE_PARQUET   = True    # save cleaned/normalized/detrended light curves
SAVE_PLOTS     = True    # save diagnostic plots
PLOT_DPI       = 120
PLOT_FORMAT    = "png"

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
N_JOBS = 1   # number of parallel workers (set >1 if multiprocessing desired)
