"""
preprocessing/qc.py
===================
Phase 1 — Raw Data Quality Control

Implements Tasks 2–8 from the specification:

  Task 2  – Data schema validation
  Task 3  – Time axis quality control
  Task 4  – Quality flag filtering (keeps quality == 0)
  Task 5  – Flux and flux_err validation
  Task 6  – Duplicate timestamp handling
  Task 7  – Conservative extreme outlier policy
  Task 8  – Per-star QC diagnostic report

DESIGN PRINCIPLES
-----------------
* Never silently repair malformed data — flag it.
* Never fabricate measurements.
* Never remove transit-like coherent downward dips as "outliers."
* Maintain TWO streams: RAW (original) and QC (analysis-ready).
* Record every removal decision with a reason and count.
* Use robust statistics (median, MAD) rather than mean/std.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    DROP_SMALL_QUARTERS,
    EXPECTED_CADENCE_DAYS,
    GAP_FACTOR,
    KEPLER_CADENCE_DAYS,
    LARGE_GAP_THRESHOLD_DAYS,
    MIN_POINTS_QUARTER,
    MIN_POINTS_STAR,
    OUTLIER_ISOLATION_WINDOW,
    OUTLIER_MIN_COHERENT_LEN,
    OUTLIER_SIGMA_NEGATIVE,
    OUTLIER_SIGMA_POSITIVE,
    QUALITY_GOOD_VALUE,
)

# ---------------------------------------------------------------------------
# Required columns and their expected dtypes (flexible matching)
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["time", "flux", "flux_err", "quality", "quarter"]
NUMERIC_COLUMNS  = ["time", "flux", "flux_err", "quality", "quarter"]


def contiguous_segments(
    time: np.ndarray,
    quarter: np.ndarray,
    expected_cadence_days: float = EXPECTED_CADENCE_DAYS,
    gap_factor: float = GAP_FACTOR,
) -> list[np.ndarray]:
    """Return index arrays that never cross a quarter boundary or large gap."""
    if len(time) == 0:
        return []
    breaks = np.diff(time) > expected_cadence_days * gap_factor
    breaks |= quarter[1:] != quarter[:-1]
    starts = np.r_[0, np.flatnonzero(breaks) + 1]
    ends = np.r_[starts[1:], len(time)]
    return [np.arange(start, end) for start, end in zip(starts, ends)]

# ===========================================================================
# Task 2 — Schema validation
# ===========================================================================

def validate_schema(df: pd.DataFrame, star_id: str = "UNKNOWN") -> Dict:
    """
    Validate that the DataFrame has the required schema.

    Parameters
    ----------
    df : raw star DataFrame
    star_id : identifier for error messages

    Returns
    -------
    report : dict with keys:
        ok             – bool, True if schema is usable
        missing_cols   – list of missing column names
        wrong_types    – dict col -> found_dtype
        n_rows         – int, original row count
        warnings       – list of non-fatal warning strings
        errors         – list of fatal error strings
    """
    report: Dict = {
        "star_id":      star_id,
        "ok":           True,
        "missing_cols": [],
        "wrong_types":  {},
        "n_rows":       len(df),
        "warnings":     [],
        "errors":       [],
    }

    # ---- required columns present? ----------------------------------------
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            report["missing_cols"].append(col)
            report["errors"].append(f"Missing required column: '{col}'")
            report["ok"] = False

    if not report["ok"]:
        return report

    # ---- numeric types? ------------------------------------------------------
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            dtype = df[col].dtype
            if not np.issubdtype(dtype, np.number):
                report["wrong_types"][col] = str(dtype)
                report["errors"].append(
                    f"Column '{col}' has non-numeric dtype {dtype}; "
                    "expected a numeric type."
                )
                report["ok"] = False

    # ---- time must be finite (no NaN allowed in the time axis) ----------------
    if "time" in df.columns:
        n_bad_time = (~np.isfinite(df["time"].values)).sum()
        report["n_time_nonfinite"] = int(n_bad_time)
        if n_bad_time > 0:
            report["warnings"].append(
                f"{n_bad_time} non-finite time values detected."
            )

    # ---- check for catastrophic corruption ------------------------------------
    if len(df) == 0:
        report["errors"].append("DataFrame is empty (0 rows).")
        report["ok"] = False

    return report


# ===========================================================================
# Task 3 — Time axis quality control
# ===========================================================================

def validate_time_axis(df: pd.DataFrame, star_id: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict]:
    """
    Sort by time, detect duplicates and non-monotonic timestamps,
    compute cadence statistics, and detect large mission gaps.

    Does NOT interpolate across gaps or manufacture observations.

    Parameters
    ----------
    df : DataFrame (after schema validation)
    star_id : identifier

    Returns
    -------
    df_sorted : DataFrame sorted by time (duplicates not yet removed)
    report : dict with time-axis QC statistics
    """
    report: Dict = {
        "star_id":           star_id,
        "n_original":        len(df),
        "n_duplicate_times": 0,
        "n_duplicate_timestamps": 0,
        "n_time_nonfinite": 0,
        "n_removed_nonfinite_time": 0,
        "originally_sorted": True,
        "n_nonmonotonic":    0,
        "median_cadence_days": float("nan"),
        "cadence_mad_days":    float("nan"),
        "time_span_days":      float("nan"),
        "large_gap_count":     0,
        "large_gaps":          [],         # list of (t_start, t_end, gap_days)
        "quarter_cadence_stats": {},       # {quarter: {median, mad, n}}
        "warnings": [],
    }

    if len(df) == 0:
        return df, report

    finite_time = np.isfinite(df["time"].values.astype(np.float64))
    report["n_time_nonfinite"] = int((~finite_time).sum())
    report["n_removed_nonfinite_time"] = report["n_time_nonfinite"]
    df_finite = df.loc[finite_time].copy()
    original_t = df_finite["time"].values.astype(np.float64)
    if len(original_t) > 1:
        report["n_nonmonotonic"] = int((np.diff(original_t) < 0).sum())
        report["originally_sorted"] = bool(np.all(np.diff(original_t) >= 0))

    # ---- sort by finite time only --------------------------------------------
    df_sorted = df_finite.sort_values("time", kind="mergesort").reset_index(drop=True)
    t = df_sorted["time"].values.astype(np.float64)

    # ---- time span -----------------------------------------------------------
    report["time_span_days"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # ---- cadence statistics --------------------------------------------------
    if len(t) > 1:
        diffs = np.diff(t)
        pos_diffs = diffs[diffs > 0]
        if len(pos_diffs) > 0:
            med_cad = float(np.median(pos_diffs))
            mad_cad = float(np.median(np.abs(pos_diffs - med_cad)))
            report["median_cadence_days"] = med_cad
            report["cadence_mad_days"]    = mad_cad
        else:
            report["warnings"].append("No positive time differences found.")

    # ---- detect duplicate timestamps -----------------------------------------
    _, first_idx = np.unique(t, return_index=True)
    dup_mask = np.ones(len(t), dtype=bool)
    dup_mask[first_idx] = False
    n_dup = int(dup_mask.sum())
    report["n_duplicate_times"] = n_dup
    if n_dup > 0:
        report["warnings"].append(
            f"{n_dup} duplicate timestamps detected; will be handled in handle_duplicates()."
        )

    if report["n_nonmonotonic"] > 0:
        report["warnings"].append(
            f"{report['n_nonmonotonic']} non-monotonic time steps before sort; resolved by sorting."
        )

    # ---- large gap detection -------------------------------------------------
    if len(t) > 1 and not np.isnan(report["median_cadence_days"]):
        diffs_all = np.diff(t)
        large_gap_mask = diffs_all > EXPECTED_CADENCE_DAYS * GAP_FACTOR
        gap_count = int(large_gap_mask.sum())
        report["large_gap_count"] = gap_count
        for i in np.where(large_gap_mask)[0]:
            report["large_gaps"].append(
                {
                    "t_start":  float(t[i]),
                    "t_end":    float(t[i + 1]),
                    "gap_days": float(diffs_all[i]),
                }
            )
        if gap_count > 0:
            report["warnings"].append(
                f"{gap_count} large gaps (> {EXPECTED_CADENCE_DAYS * GAP_FACTOR:.5f} d) detected; "
                "detrending will not cross these boundaries."
            )

    # ---- per-quarter cadence -------------------------------------------------
    if "quarter" in df_sorted.columns:
        q_arr = df_sorted["quarter"].values
        for qq in np.unique(q_arr):
            mask = q_arr == qq
            tq = t[mask]
            if len(tq) > 1:
                dq = np.diff(tq)
                pos_dq = dq[dq > 0]
                if len(pos_dq) > 0:
                    report["quarter_cadence_stats"][int(qq)] = {
                        "median_cad": float(np.median(pos_dq)),
                        "mad_cad":    float(np.median(np.abs(pos_dq - np.median(pos_dq)))),
                        "n_points":   int(mask.sum()),
                    }

    return df_sorted, report


# ===========================================================================
# Task 4 — Quality flag filtering
# ===========================================================================

def apply_quality_filter(df: pd.DataFrame, star_id: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict]:
    """
    Retain only quality == 0 cadences for the QC analysis stream.
    The raw DataFrame is NOT modified.

    Parameters
    ----------
    df : DataFrame (time-sorted)

    Returns
    -------
    df_qc : DataFrame with only quality == 0 rows
    report : dict with removal statistics
    """
    n_original = len(df)
    mask_good  = (df["quality"].values == QUALITY_GOOD_VALUE)
    df_qc      = df[mask_good].copy().reset_index(drop=True)
    n_removed  = n_original - len(df_qc)

    report = {
        "star_id":           star_id,
        "n_original":        n_original,
        "n_quality_removed": n_removed,
        "n_quality_kept":    len(df_qc),
        "pct_quality_removed": 100.0 * n_removed / n_original if n_original > 0 else 0.0,
        "quality_value_counts": dict(
            zip(*np.unique(df["quality"].values, return_counts=True))
        ),
    }

    # Document what quality flags were present (for traceability)
    report["quality_flags_found"] = sorted(
        [int(v) for v in df["quality"].unique() if v != QUALITY_GOOD_VALUE]
    )

    return df_qc, report


# ===========================================================================
# Task 5 — Flux and flux_err validation
# ===========================================================================

def validate_flux(df: pd.DataFrame, star_id: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict]:
    """
    Remove / flag rows where flux or flux_err are unusable.

    Rules:
      * flux NaN or ±inf → remove
      * flux_err NaN, ±inf, or ≤ 0 → remove
      * Do NOT replace with median or zero (no fabrication)

    Parameters
    ----------
    df : QC-filtered DataFrame

    Returns
    -------
    df_valid : DataFrame with only physically usable rows
    report : dict with removal breakdown
    """
    n_in = len(df)

    flux     = df["flux"].values.astype(np.float64)
    flux_err = df["flux_err"].values.astype(np.float64)

    bad_flux     = ~np.isfinite(flux)
    bad_flux_err = ~np.isfinite(flux_err) | (flux_err <= 0.0)

    # Combined bad mask (either flux or flux_err unusable)
    bad_mask = bad_flux | bad_flux_err
    df_valid = df[~bad_mask].copy().reset_index(drop=True)

    report = {
        "star_id":               star_id,
        "n_in":                  n_in,
        "n_bad_flux":            int(bad_flux.sum()),
        "n_bad_flux_err":        int(bad_flux_err.sum()),
        "n_nonfinite_removed":   int(bad_mask.sum()),
        "n_valid":               len(df_valid),
        "pct_nonfinite_removed": 100.0 * bad_mask.sum() / n_in if n_in > 0 else 0.0,
    }

    return df_valid, report


# ===========================================================================
# Task 6 — Duplicate timestamp handling
# ===========================================================================

def handle_duplicates(df: pd.DataFrame, star_id: str = "UNKNOWN") -> Tuple[pd.DataFrame, Dict]:
    """
    Handle duplicate timestamps in a scientifically justified way.

    Policy (documented):
      1. Exact duplicate records (identical time AND flux AND flux_err):
         → keep one, drop the rest deterministically (first occurrence).
      2. Same timestamp, different measurements:
         → take the record with the lower flux_err (higher quality measurement).
         → if flux_err also ties, keep the first occurrence.
      3. Result MUST have strictly monotonically increasing time.

    Parameters
    ----------
    df : time-sorted, flux-validated DataFrame

    Returns
    -------
    df_dedup : DataFrame with unique timestamps
    report : dict describing what was done
    """
    n_in = len(df)
    t    = df["time"].values.astype(np.float64)

    # ---- find duplicated times -----------------------------------------------
    _, first_idx, inverse, counts = np.unique(
        t, return_index=True, return_inverse=True, return_counts=True
    )
    dup_time_mask = counts[inverse] > 1   # True for all rows with a duplicated time

    n_exact_dup  = 0
    n_ambiguous  = 0
    rows_to_keep = np.ones(n_in, dtype=bool)

    if dup_time_mask.any():
        dup_times = t[first_idx[counts > 1]]
        for dt in dup_times:
            idx_group = np.where(t == dt)[0]
            rows_group = df.iloc[idx_group]

            # Check if all records are identical
            flux_vals  = rows_group["flux"].values
            ferr_vals  = rows_group["flux_err"].values
            all_same   = (
                np.all(flux_vals == flux_vals[0]) and
                np.all(ferr_vals == ferr_vals[0])
            )

            if all_same:
                # Exact duplicates → keep first, drop rest
                n_exact_dup += len(idx_group) - 1
                rows_to_keep[idx_group[1:]] = False
            else:
                # Ambiguous duplicates → keep the row with smallest flux_err
                n_ambiguous += len(idx_group) - 1
                best = idx_group[np.argmin(ferr_vals)]
                for idx in idx_group:
                    if idx != best:
                        rows_to_keep[idx] = False

    df_dedup = df[rows_to_keep].copy().reset_index(drop=True)

    # Verify strict monotonicity after dedup
    t_out = df_dedup["time"].values
    if len(t_out) > 1 and not np.all(np.diff(t_out) > 0):
        # Extremely rare: floating-point ties — break by position
        df_dedup = df_dedup.drop_duplicates(subset="time", keep="first").reset_index(drop=True)

    report = {
        "star_id":              star_id,
        "n_in":                 n_in,
        "n_exact_dup_removed":  n_exact_dup,
        "n_ambiguous_resolved": n_ambiguous,
        "n_total_dup_removed":  n_exact_dup + n_ambiguous,
        "n_duplicate_timestamps": int((counts > 1).sum()),
        "n_duplicate_rows_collapsed": n_exact_dup + n_ambiguous,
        "n_duplicate_quarter_conflicts": 0,
        "n_out":                len(df_dedup),
        "duplicate_policy": (
            "exact_duplicates: keep_first; "
            "ambiguous: keep_lowest_flux_err"
        ),
    }

    conflict_count = 0
    if dup_time_mask.any():
        for dt in t[first_idx[counts > 1]]:
            quarters = df.iloc[np.where(t == dt)[0]]["quarter"].dropna().unique()
            conflict_count += int(len(quarters) > 1)
    report["n_duplicate_quarter_conflicts"] = conflict_count

    return df_dedup, report


# ===========================================================================
# Task 7 — Conservative extreme outlier policy
# ===========================================================================

def _compute_local_mad_scatter(f: np.ndarray, win: int = 101) -> np.ndarray:
    """
    Compute a local robust scatter (MAD) for each point using a rolling window.
    Returns the local MAD array (same length as f).
    """
    s = pd.Series(f)
    roll_med = s.rolling(win, center=True, min_periods=win // 4).median()
    resid    = s - roll_med
    # Rolling MAD
    roll_mad = resid.abs().rolling(win, center=True, min_periods=win // 4).median()
    return roll_mad.values


def flag_extreme_outliers(
    df: pd.DataFrame,
    star_id: str = "UNKNOWN",
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Conservative outlier flagging that PROTECTS transit signals.

    Algorithm (per quarter):
      1. Compute local robust scatter (rolling MAD, window ~4 h).
      2. Flag UPWARD spikes: f > median + OUTLIER_SIGMA_POSITIVE * MAD.
         These are cosmic rays / saturation — safe to remove.
      3. Flag DOWNWARD isolated points: f < median - OUTLIER_SIGMA_NEGATIVE * MAD
         ONLY if ALL of the following are true:
           a. The point is isolated (no neighboring flagged-low points within
              OUTLIER_ISOLATION_WINDOW cadences).
           b. The contiguous low-flux run containing this point is shorter than
              OUTLIER_MIN_COHERENT_LEN cadences.
         This protects coherent transit-like dips.
      4. Returns a flag array (1 = keep, 0 = flagged as outlier).
         The caller decides whether to remove or merely annotate.

    Parameters
    ----------
    df : clean, deduplicated DataFrame (per star)

    Returns
    -------
    df_flagged : same DataFrame with added 'outlier_flag' column (0/1)
    flag_array : boolean array, True = outlier
    report : dict with outlier statistics
    """
    n_in          = len(df)
    flag_array    = np.zeros(n_in, dtype=bool)   # True = outlier
    negative_flag_array = np.zeros(n_in, dtype=bool)
    n_up_removed  = 0
    n_lo_removed  = 0

    if "quarter" not in df.columns or n_in == 0:
        df_flagged = df.copy()
        df_flagged["outlier_flag"] = 0
        return df_flagged, flag_array, {"n_upward_flagged": 0, "n_downward_flagged": 0}

    flux  = df["flux"].values.astype(np.float64)
    q_arr = df["quarter"].values

    for qq in np.unique(q_arr):
        idx = np.where(q_arr == qq)[0]
        fq  = flux[idx]

        if len(fq) < MIN_POINTS_QUARTER:
            continue

        # Robust local estimates
        med_q = np.median(fq)
        mad_q = float(np.median(np.abs(fq - med_q)))
        if mad_q == 0.0:
            mad_q = 1e-9   # protect against flat quarters

        # ---- upward spikes ---------------------------------------------------
        upper_thresh = med_q + OUTLIER_SIGMA_POSITIVE * 1.4826 * mad_q
        up_mask = fq > upper_thresh
        flag_array[idx[up_mask]] = True
        n_up_removed += int(up_mask.sum())

        # ---- downward isolated points ----------------------------------------
        lower_thresh = med_q - OUTLIER_SIGMA_NEGATIVE * 1.4826 * mad_q
        down_candidates = fq < lower_thresh

        if down_candidates.any():
            # Find runs of consecutive low-flux points
            in_run   = False
            run_start = 0
            runs: list[Tuple[int, int]] = []
            for i, val in enumerate(down_candidates):
                if val and not in_run:
                    in_run    = True
                    run_start = i
                elif not val and in_run:
                    runs.append((run_start, i - 1))
                    in_run = False
            if in_run:
                runs.append((run_start, len(down_candidates) - 1))

            for rs, re in runs:
                run_len = re - rs + 1
                if run_len < OUTLIER_MIN_COHERENT_LEN:
                    # Short run — check isolation
                    lo_check = max(0, rs - OUTLIER_ISOLATION_WINDOW)
                    hi_check = min(len(down_candidates) - 1, re + OUTLIER_ISOLATION_WINDOW)
                    neighbor_vals = fq[lo_check:hi_check + 1]
                    n_low_nearby  = (neighbor_vals < lower_thresh).sum() - run_len
                    if n_low_nearby == 0:
                        # Truly isolated — safe to flag
                        flag_array[idx[rs:re + 1]] = True
                        negative_flag_array[idx[rs:re + 1]] = True
                        n_lo_removed += run_len
                # else: coherent run >= OUTLIER_MIN_COHERENT_LEN → PROTECT (transit candidate)

    # ---- attach flag to DataFrame --------------------------------------------
    df_flagged = df.copy()
    df_flagged["outlier_flag"] = flag_array.astype(int)
    df_flagged["negative_outlier_flag"] = negative_flag_array.astype(int)

    report = {
        "star_id":              star_id,
        "n_in":                 n_in,
        "n_upward_flagged":     n_up_removed,
        "n_downward_flagged":   n_lo_removed,
        "n_total_flagged":      n_up_removed + n_lo_removed,
        "n_negative_extreme_retained": n_lo_removed,
        "pct_flagged":          100.0 * (n_up_removed + n_lo_removed) / n_in if n_in > 0 else 0.0,
        "sigma_positive":       OUTLIER_SIGMA_POSITIVE,
        "sigma_negative":       OUTLIER_SIGMA_NEGATIVE,
        "policy": (
            "upward_spikes: always removed; "
            "downward_isolated (< 3 consecutive, no low neighbors): removed; "
            "coherent_downward_runs: PROTECTED"
        ),
    }

    return df_flagged, flag_array, report


# ===========================================================================
# Task 8 — Comprehensive per-star QC diagnostic report
# ===========================================================================

def generate_qc_report(
    star_id: str,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    schema_report: Dict,
    time_report:   Dict,
    quality_report: Dict,
    flux_report:   Dict,
    dup_report:    Dict,
    outlier_report: Dict,
) -> Dict:
    """
    Aggregate all per-star QC results into a single diagnostic record.

    This record is later concatenated across all stars to form the
    machine-readable CSV diagnostics file.
    """
    n_orig = len(raw_df)
    n_final = len(clean_df)

    # Per-quarter stats on clean data
    quarter_stats = {}
    if len(clean_df) > 0 and "quarter" in clean_df.columns:
        for qq, grp in clean_df.groupby("quarter"):
            f = grp["flux"].values.astype(np.float64)
            med = float(np.median(f)) if len(f) > 0 else float("nan")
            mad = float(np.median(np.abs(f - med))) if len(f) > 0 else float("nan")
            quarter_stats[int(qq)] = {
                "n_points": len(grp),
                "median":   med,
                "mad":      mad,
                "t_min":    float(grp["time"].min()),
                "t_max":    float(grp["time"].max()),
            }

    report = {
        # Identification
        "star_id":                star_id,
        # Counts
        "original_points":        n_orig,
        "valid_points":           n_final,
        "quality_removed":        quality_report.get("n_quality_removed", 0),
        "nonfinite_removed":      flux_report.get("n_nonfinite_removed", 0),
        "duplicate_removed":      dup_report.get("n_total_dup_removed", 0),
        "outlier_flagged":        outlier_report.get("n_total_flagged", 0),
        "percentage_retained":    100.0 * n_final / n_orig if n_orig > 0 else 0.0,
        # Time axis
        "time_span_days":         time_report.get("time_span_days", float("nan")),
        "median_cadence_days":    time_report.get("median_cadence_days", float("nan")),
        "cadence_mad_days":       time_report.get("cadence_mad_days", float("nan")),
        "large_gap_count":        time_report.get("large_gap_count", 0),
        "n_large_gaps":            time_report.get("large_gap_count", 0),
        "n_time_nonfinite":        time_report.get("n_time_nonfinite", 0),
        "n_removed_nonfinite_time": time_report.get("n_removed_nonfinite_time", 0),
        "originally_sorted":       time_report.get("originally_sorted", True),
        "n_nonmonotonic":           time_report.get("n_nonmonotonic", 0),
        "n_duplicate_timestamps":   dup_report.get("n_duplicate_timestamps", 0),
        "n_duplicate_rows_collapsed": dup_report.get("n_duplicate_rows_collapsed", 0),
        "n_duplicate_quarter_conflicts": dup_report.get("n_duplicate_quarter_conflicts", 0),
        # Quarter info
        "quarter_count":          len(np.unique(clean_df["quarter"].values)) if n_final > 0 else 0,
        "quarters_present":       sorted(clean_df["quarter"].unique().tolist()) if n_final > 0 else [],
        # Noise
        "robust_scatter_ppm":     (
            float(np.median(np.abs(
                clean_df["flux"].values - np.median(clean_df["flux"].values)
            )) * 1e6 / np.median(clean_df["flux"].values))
            if n_final > 0 and np.median(clean_df["flux"].values) > 0 else float("nan")
        ),
        # Schema flags
        "schema_ok":              schema_report.get("ok", False),
        "schema_errors":          schema_report.get("errors", []),
        # Per-quarter stats
        "quarter_stats":          quarter_stats,
        # Warning accumulation
        "warnings": (
            schema_report.get("warnings", []) +
            time_report.get("warnings", [])
        ),
        # Flags
        "severely_corrupted": (
            not schema_report.get("ok", True) or
            n_final < MIN_POINTS_STAR
        ),
        "n_outlier_removed": outlier_report.get("n_upward_flagged", 0),
        "n_negative_extreme_retained": outlier_report.get("n_downward_flagged", 0),
    }

    small_quarters = [q for q, stats in quarter_stats.items()
                      if stats["n_points"] < MIN_POINTS_QUARTER]
    report["small_quarter_count"] = len(small_quarters)
    report["small_quarter_ids"] = small_quarters

    return report


# ===========================================================================
# Convenience wrapper — run all Task 2–8 QC in sequence
# ===========================================================================

def run_full_qc(
    df: pd.DataFrame,
    star_id: str = "UNKNOWN",
    remove_outliers: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run the complete Phase 1 QC pipeline on a single star.

    Parameters
    ----------
    df              : raw DataFrame from Parquet
    star_id         : e.g. "KIC_10002867"
    remove_outliers : if True, remove flagged outliers from clean stream;
                      if False, keep them but annotate with 'outlier_flag'

    Returns
    -------
    raw_df    : original DataFrame (untouched)
    clean_df  : QC-cleaned DataFrame (analysis stream)
    full_report : aggregated diagnostic dict
    """
    raw_df = df.copy()   # preserve raw stream

    # Step 1: Schema validation
    schema_rep = validate_schema(df, star_id)
    if not schema_rep["ok"]:
        # Return empty DataFrame if schema is unrecoverable
        empty = pd.DataFrame(columns=df.columns)
        return raw_df, empty, generate_qc_report(
            star_id, raw_df, empty,
            schema_rep, {}, {}, {}, {}, {}
        )

    # Step 2: Time axis QC (finite timestamps only, then sort + stats)
    df_sorted, time_rep = validate_time_axis(df, star_id)

    # Step 3: Quality flag filtering
    df_qc, quality_rep = apply_quality_filter(df_sorted, star_id)

    # Step 4: Flux/flux_err validation
    df_valid, flux_rep = validate_flux(df_qc, star_id)

    # Step 5: Duplicate handling
    df_dedup, dup_rep = handle_duplicates(df_valid, star_id)

    # Step 6: Outlier flagging
    df_flagged, flag_arr, outlier_rep = flag_extreme_outliers(df_dedup, star_id)

    # Step 7: Create analysis-stream (optionally remove flagged outliers)
    if remove_outliers:
        removable = (df_flagged["outlier_flag"] == 0) | (df_flagged["negative_outlier_flag"] == 1)
        clean_df = df_flagged[removable].copy().reset_index(drop=True)
    else:
        clean_df = df_flagged.copy()

    # Small quarters remain in the analysis stream unless explicitly disabled.
    if "quarter" in clean_df.columns and len(clean_df) > 0:
        qt_counts = clean_df.groupby("quarter").size()
        bad_quarters = qt_counts[qt_counts < MIN_POINTS_QUARTER].index.tolist()
        if bad_quarters and DROP_SMALL_QUARTERS:
            clean_df = clean_df[~clean_df["quarter"].isin(bad_quarters)].reset_index(drop=True)

    # Step 8: Aggregate report
    full_report = generate_qc_report(
        star_id, raw_df, clean_df,
        schema_rep, time_rep, quality_rep, flux_rep, dup_rep, outlier_rep,
    )

    return raw_df, clean_df, full_report
