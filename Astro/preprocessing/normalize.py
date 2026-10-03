"""
preprocessing/normalize.py
==========================
Phase 2 — Quarter-wise Normalization (Tasks 9–11)

PURPOSE
-------
Remove inter-quarter flux jumps caused by detector resets, temperature
changes, and spacecraft re-pointing between quarters.

DESIGN PRINCIPLES
-----------------
* Process each quarter independently — never treat the full 4-year
  baseline as one stationary time series.
* Use the robust median (not mean) as the normalization baseline.
  This is insensitive to a small number of transit-like dips.
* Never smooth across quarter boundaries.
* Record every normalization parameter for reproducibility and
  downstream uncertainty propagation.
* Validate that transit depth is not systematically changed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MIN_POINTS_QUARTER, NORM_STATISTIC

# ---------------------------------------------------------------------------
# Robust location estimators
# ---------------------------------------------------------------------------

def _biweight_location(x: np.ndarray, c: float = 6.0, max_iter: int = 10) -> float:
    """
    Tukey's biweight location estimator — more robust than median for
    distributions with occasional outliers.
    """
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    m = float(np.median(x))
    for _ in range(max_iter):
        u  = (x - m) / (c * 1.4826 * np.median(np.abs(x - m)) + 1e-30)
        w  = np.where(np.abs(u) < 1.0, (1 - u**2) ** 2, 0.0)
        sw = w.sum()
        if sw == 0:
            break
        m = float(np.sum(w * x) / sw)
    return m


def _robust_location(x: np.ndarray, method: str = "median") -> float:
    """Return a robust central location estimate for flux array x."""
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if method == "biweight_location":
        return _biweight_location(x)
    return float(np.median(x))   # default: median


# ===========================================================================
# Task 9/10 — Quarter-wise normalization
# ===========================================================================

def normalize_quarters(
    time:      np.ndarray,
    flux:      np.ndarray,
    flux_err:  np.ndarray,
    quarter:   np.ndarray,
    method:    str = NORM_STATISTIC,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Normalize each Kepler quarter to its own robust flux baseline.

    For each quarter q:
        baseline_q  = robust_location(flux[quarter == q])
        norm_flux_q = flux[quarter == q] / baseline_q
        norm_err_q  = flux_err[quarter == q] / baseline_q   (propagated)

    Parameters
    ----------
    time, flux, flux_err, quarter : 1-D arrays (aligned, same length)
    method : "median" or "biweight_location"

    Returns
    -------
    norm_flux    : normalized flux array (same shape as flux)
    norm_err     : normalized uncertainty array (same shape as flux_err)
    norm_info    : dict with per-quarter normalization statistics
                   {quarter_id: {baseline, mad_scatter, n_points, t_min, t_max}}
    """
    norm_flux  = np.ones_like(flux,     dtype=np.float64)
    norm_err   = np.ones_like(flux_err, dtype=np.float64)
    norm_info: Dict = {}

    unique_quarters = np.unique(quarter)

    for qq in unique_quarters:
        mask = quarter == qq
        fq   = flux[mask].astype(np.float64)
        eq   = flux_err[mask].astype(np.float64)
        tq   = time[mask]

        if mask.sum() < MIN_POINTS_QUARTER:
            # Too few points — normalise to 1 (no reliable baseline)
            norm_flux[mask] = 1.0
            norm_err[mask]  = eq / (np.median(np.abs(fq)) + 1e-30)
            norm_info[int(qq)] = {
                "baseline":    float("nan"),
                "mad_scatter": float("nan"),
                "n_points":    int(mask.sum()),
                "t_min":       float(tq.min()),
                "t_max":       float(tq.max()),
                "skipped":     True,
            }
            continue

        baseline = _robust_location(fq, method)

        if not np.isfinite(baseline) or baseline <= 0:
            # Degenerate quarter — skip normalization, leave as-is
            norm_flux[mask] = fq / (np.nanmedian(np.abs(fq)) or 1.0)
            norm_err[mask]  = eq / (np.nanmedian(np.abs(fq)) or 1.0)
            norm_info[int(qq)] = {
                "baseline":    float("nan"),
                "mad_scatter": float("nan"),
                "n_points":    int(mask.sum()),
                "t_min":       float(tq.min()),
                "t_max":       float(tq.max()),
                "degenerate":  True,
            }
            continue

        norm_flux[mask] = fq / baseline
        norm_err[mask]  = eq / baseline

        residual = fq / baseline - 1.0
        mad = float(np.median(np.abs(residual)))

        norm_info[int(qq)] = {
            "baseline":        baseline,
            "mad_scatter":     mad,
            "mad_ppm":         mad * 1e6,
            "n_points":        int(mask.sum()),
            "t_min":           float(tq.min()),
            "t_max":           float(tq.max()),
            "norm_method":     method,
        }

    return norm_flux, norm_err, norm_info


# ===========================================================================
# Task 11 — Normalization validation
# ===========================================================================

def validate_normalization(
    time:          np.ndarray,
    norm_flux:     np.ndarray,
    quarter:       np.ndarray,
    norm_info:     Dict,
    truth_row:     Optional[pd.Series] = None,
) -> Dict:
    """
    Validate that quarter normalization did not introduce artifacts.

    Checks:
      1. Inter-quarter median consistency (should all be ~1.0 after norm).
      2. No artificial variance inflation.
      3. If truth_row provided: transit depth preservation.

    Parameters
    ----------
    time, norm_flux, quarter : 1-D arrays
    norm_info : output of normalize_quarters()
    truth_row : optional row from train_truth.csv with
                [period_days, epoch_t0, depth_ppm, duration_hours]

    Returns
    -------
    report : dict with validation results
    """
    report: Dict = {
        "inter_quarter_median_consistency": {},
        "quarter_medians_post_norm": {},
        "transit_depth_preserved": None,
        "warnings": [],
    }

    # Per-quarter median after normalization
    medians_post = {}
    for qq in np.unique(quarter):
        mask = quarter == qq
        fq = norm_flux[mask]
        if len(fq) > 0:
            medians_post[int(qq)] = float(np.median(fq))

    report["quarter_medians_post_norm"] = medians_post

    if len(medians_post) > 1:
        med_vals = np.array(list(medians_post.values()))
        spread   = float(np.max(med_vals) - np.min(med_vals))
        report["inter_quarter_spread"] = spread
        if spread > 0.01:
            report["warnings"].append(
                f"Inter-quarter median spread = {spread:.4f} > 0.01 "
                "(possible normalization artifact)"
            )

    # Transit depth preservation test (if truth available)
    if truth_row is not None and not truth_row.empty:
        try:
            period    = float(truth_row["period_days"])
            t0        = float(truth_row["epoch_t0"])
            depth_ppm = float(truth_row["depth_ppm"])
            dur_hrs   = float(truth_row["duration_hours"])
            dur_days  = dur_hrs / 24.0

            # Find transit windows and measure median flux therein
            phase = ((time - t0) % period)
            # Wrap to [-P/2, P/2]
            phase = np.where(phase > period / 2, phase - period, phase)
            in_transit = np.abs(phase) < (dur_days / 2)

            if in_transit.sum() >= 3:
                depth_measured_ppm = (1.0 - float(np.median(norm_flux[in_transit]))) * 1e6
                preservation_ratio = depth_measured_ppm / depth_ppm if depth_ppm > 0 else float("nan")
                report["transit_depth_preserved"] = {
                    "true_depth_ppm":     depth_ppm,
                    "measured_depth_ppm": depth_measured_ppm,
                    "preservation_ratio": preservation_ratio,
                    "n_in_transit":       int(in_transit.sum()),
                }
                if preservation_ratio < 0.7:
                    report["warnings"].append(
                        f"Transit depth preservation ratio = {preservation_ratio:.3f} < 0.70 "
                        "(normalization may be distorting transit signal)"
                    )
        except Exception as e:
            report["warnings"].append(f"Could not compute transit depth preservation: {e}")

    return report


# ===========================================================================
# Main entry point for normalization
# ===========================================================================

def normalize_star(
    clean_df:  pd.DataFrame,
    star_id:   str = "UNKNOWN",
    method:    str = NORM_STATISTIC,
    truth_row: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply quarter-wise normalization to a QC-cleaned star DataFrame.

    Parameters
    ----------
    clean_df  : QC-cleaned DataFrame (time, flux, flux_err, quality, quarter, [outlier_flag])
    star_id   : identifier
    method    : normalization statistic ("median" or "biweight_location")
    truth_row : optional truth row for validation (TRAIN only; never PRIVATE)

    Returns
    -------
    norm_df   : DataFrame with added columns:
                  'norm_flux'  – quarter-normalized flux
                  'norm_err'   – propagated uncertainty
    norm_report : dict with normalization statistics and validation
    """
    if len(clean_df) == 0:
        return clean_df.copy(), {"star_id": star_id, "n_points": 0}

    t   = clean_df["time"].values.astype(np.float64)
    f   = clean_df["flux"].values.astype(np.float64)
    e   = clean_df["flux_err"].values.astype(np.float64)
    q   = clean_df["quarter"].values

    norm_flux, norm_err, norm_info = normalize_quarters(t, f, e, q, method)

    # Validate
    val_report = validate_normalization(t, norm_flux, q, norm_info, truth_row)

    norm_df = clean_df.copy()
    norm_df["norm_flux"] = norm_flux
    norm_df["norm_err"]  = norm_err

    norm_report = {
        "star_id":         star_id,
        "n_points":        len(norm_df),
        "n_quarters":      len(norm_info),
        "norm_method":     method,
        "quarter_info":    norm_info,
        "validation":      val_report,
    }

    return norm_df, norm_report
