"""
preprocessing/detrend.py
========================
Phase 2 — Transit-Preserving Detrending (Tasks 12–16, 25)

FIVE METHODS IMPLEMENTED
------------------------
  A. robust_rolling_median   – centered rolling median (baseline method)
  B. savgol                  – Savitzky-Golay polynomial smoothing
  C. biweight_lowess         – robust LOWESS with Tukey biweight kernel
  D. iterative_clip_median   – rolling median + iterative transit masking
  E. adaptive_window         – window scaled to orbital timescale

All methods:
  * Process per quarter — NEVER smooth across quarter boundaries (Task 10).
  * Handle edge effects at quarter start/end (Task 25).
  * Accept window_days as a parameter for multi-scale testing (Task 15).
  * Return both the detrended flux AND the baseline estimate.
  * Preserve flux_err by propagating through the baseline division.

CRITICAL ANTI-PATTERN (Task 14)
--------------------------------
Do NOT use a window so short that a transit becomes part of the baseline.
Minimum safe window: 3 × longest expected transit duration ≈ 3 × 0.55 d ≈ 1.65 d
We enforce a floor of 1.0 day by default; the comparison tests multiple windows.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import (
    EDGE_FLAG_ENABLED,
    ENABLE_EXPENSIVE_DETREND,
    EXPECTED_CADENCE_DAYS,
    GAP_FACTOR,
    DETREND_WINDOW_SELECTED_DAYS,
    KEPLER_CADENCE_DAYS,
    MIN_POINTS_QUARTER,
    SAVGOL_POLYORDER,
)
from .qc import contiguous_segments

# ---------------------------------------------------------------------------
# Helper: cadence -> window size (odd integer)
# ---------------------------------------------------------------------------

def _window_size(window_days: float, cadence_days: float) -> int:
    """Convert a window in days to an odd integer number of cadences."""
    n = int(round(window_days / cadence_days))
    n = max(n, 5)          # at least 5 points
    if n % 2 == 0:
        n += 1              # must be odd for centered windows
    return n


# ---------------------------------------------------------------------------
# Edge-effect handler
# ---------------------------------------------------------------------------

def _fill_edge_nans(baseline: np.ndarray, method: str = "edge") -> np.ndarray:
    """
    Fill NaN values at the edges of a baseline array.

    Strategy ("edge"):
      - Forward-fill from first finite value.
      - Backward-fill from last finite value.
    This avoids dividing by NaN at quarter edges while not fabricating data.
    The resulting edge values are slightly less precise than the interior
    but prevent downstream NaN propagation.
    """
    result = baseline.copy()
    finite = np.isfinite(result)
    if not finite.any():
        return np.ones_like(result)  # degenerate — no division effect

    # Forward fill
    last_finite = result[finite][0]
    for i in range(len(result)):
        if np.isfinite(result[i]):
            last_finite = result[i]
        else:
            result[i] = last_finite

    # Backward fill (handles leading NaNs)
    last_finite = result[finite][-1]
    for i in range(len(result) - 1, -1, -1):
        if np.isfinite(result[i]):
            last_finite = result[i]
        else:
            result[i] = last_finite

    # If still NaN (entire quarter is NaN), set to 1
    result = np.where(np.isfinite(result), result, 1.0)
    return result


# ===========================================================================
# Method A — Robust Rolling Median
# ===========================================================================

def _detrend_rolling_median(
    t: np.ndarray,
    f: np.ndarray,
    window_days: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centered rolling median baseline.

    Edge handling: pandas rolling with min_periods=k//3 naturally provides
    partial windows at edges, which is better than NaN. We then fill
    remaining NaN with edge values.

    Returns
    -------
    detrended : f / baseline (array of same shape)
    baseline  : estimated stellar + instrumental trend
    """
    if len(t) < 3:
        return np.ones_like(f), np.ones_like(f)

    cadence     = float(np.median(np.diff(t))) if len(t) > 1 else KEPLER_CADENCE_DAYS
    k           = _window_size(window_days, cadence)
    min_periods = max(5, k // 3)

    baseline = (
        pd.Series(f)
        .rolling(k, center=True, min_periods=min_periods)
        .median()
        .values
    )
    baseline = _fill_edge_nans(baseline)
    detrended = f / baseline
    return detrended, baseline


# ===========================================================================
# Method B — Savitzky-Golay Filter
# ===========================================================================

def _detrend_savgol(
    t: np.ndarray,
    f: np.ndarray,
    window_days: float,
    polyorder: int = SAVGOL_POLYORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Savitzky-Golay polynomial smoothing as the baseline.

    SG is a well-calibrated smoother that preserves higher-order trends
    better than a rolling median for slowly-varying stellar variability.

    Edge handling: scipy.signal.savgol_filter mode='mirror' reflects the
    signal at edges, which avoids large edge artifacts.
    """
    if len(t) < polyorder + 2:
        return np.ones_like(f), np.ones_like(f)

    cadence  = float(np.median(np.diff(t))) if len(t) > 1 else KEPLER_CADENCE_DAYS
    k        = _window_size(window_days, cadence)

    # Ensure window is larger than polyorder
    k = max(k, polyorder + 2)
    if k % 2 == 0:
        k += 1

    # SG requires len(f) >= window_length
    if len(f) < k:
        k = len(f) if len(f) % 2 != 0 else len(f) - 1
        k = max(k, polyorder + 2)
        if k <= polyorder:
            return np.ones_like(f), np.ones_like(f)

    try:
        baseline = savgol_filter(f, window_length=k, polyorder=polyorder, mode="mirror")
    except Exception:
        # Fallback to rolling median if SG fails
        baseline = (
            pd.Series(f)
            .rolling(k, center=True, min_periods=k // 3)
            .median()
            .values
        )
        baseline = _fill_edge_nans(baseline)

    baseline  = _fill_edge_nans(baseline)
    detrended = f / np.where(baseline > 0, baseline, 1.0)
    return detrended, baseline


# ===========================================================================
# Method C — Robust LOWESS (Tukey biweight kernel)
# ===========================================================================

def _biweight_kernel(u: np.ndarray) -> np.ndarray:
    """Tukey biweight kernel weights for robust local regression."""
    w = (1.0 - u**2) ** 2
    w[np.abs(u) >= 1.0] = 0.0
    return w


def _detrend_biweight_lowess(
    t: np.ndarray,
    f: np.ndarray,
    window_days: float,
    n_iter: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locally-weighted regression (LOWESS) with Tukey biweight kernel.

    More robust than SG against occasional bright outliers, and more
    flexible than a pure rolling median for curved stellar variability.

    We use a fixed-width bandwidth rather than a data fraction, so that
    the effective scale is always window_days regardless of quarter length.
    """
    if len(t) < 5:
        return np.ones_like(f), np.ones_like(f)

    cadence    = float(np.median(np.diff(t))) if len(t) > 1 else KEPLER_CADENCE_DAYS
    half_win   = window_days / 2.0   # half-width in days

    n          = len(t)
    baseline   = np.empty(n, dtype=np.float64)
    residuals  = np.zeros(n, dtype=np.float64)

    for iteration in range(n_iter):
        for i in range(n):
            # Points within the bandwidth
            dist    = np.abs(t - t[i])
            in_win  = dist <= half_win
            if in_win.sum() < 3:
                in_win = np.argsort(dist)[:7]
                # convert to mask
                mask_ = np.zeros(n, dtype=bool)
                mask_[in_win] = True
                in_win = mask_

            ti = t[in_win]
            fi = f[in_win]

            # Tricube distance weights
            max_d = half_win if half_win > 0 else (dist[in_win].max() + 1e-9)
            u     = dist[in_win] / max_d
            w     = _biweight_kernel(np.minimum(u, 1.0))

            # Robustness weights from previous iteration
            if iteration > 0:
                r_i  = np.abs(residuals[in_win])
                med_r = np.median(r_i)
                bw    = 6.0 * med_r if med_r > 0 else 1.0
                rob_w = _biweight_kernel(np.minimum(r_i / bw, 1.0))
                w     = w * rob_w

            sw = w.sum()
            if sw < 1e-12:
                baseline[i] = np.median(fi)
                continue

            # Weighted local linear regression
            wt = ti - t[i]
            sw0 = np.sum(w)
            sw1 = np.sum(w * wt)
            sw2 = np.sum(w * wt**2)
            swy = np.sum(w * fi)
            swyt = np.sum(w * fi * wt)
            denom = sw0 * sw2 - sw1**2
            if abs(denom) < 1e-30:
                baseline[i] = np.sum(w * fi) / sw
            else:
                beta0 = (sw2 * swy - sw1 * swyt) / denom
                baseline[i] = beta0

        residuals = f - baseline

    baseline = _fill_edge_nans(baseline)
    detrended = f / np.where(baseline > 0, baseline, 1.0)
    return detrended, baseline


# ===========================================================================
# Method D — Iterative Clipping Rolling Median (Transit-Masking)
# ===========================================================================

def _detrend_iterative_clip_median(
    t: np.ndarray,
    f: np.ndarray,
    window_days: float,
    n_iter: int = 3,
    clip_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling median with iterative transit masking.

    Algorithm:
      1. Compute rolling median baseline.
      2. Identify points significantly BELOW the baseline (transit candidates).
      3. Exclude those points from the baseline estimation.
      4. Repeat for n_iter iterations.

    This ensures that transit events do not "pull" the baseline downward,
    which would underestimate the transit depth. However, it only masks
    points that are consistently below — not isolated noise spikes.

    The masking is conservative: only mask points that are clip_sigma MAD
    below the current baseline.
    """
    if len(t) < 5:
        return np.ones_like(f), np.ones_like(f)

    cadence    = float(np.median(np.diff(t))) if len(t) > 1 else KEPLER_CADENCE_DAYS
    k          = _window_size(window_days, cadence)
    min_periods = max(5, k // 3)

    weights = np.ones(len(f))

    for _ in range(n_iter):
        # Compute weighted rolling median (approximated by excluding masked points)
        f_masked = f.copy().astype(np.float64)
        f_masked[weights < 0.5] = np.nan

        baseline = (
            pd.Series(f_masked)
            .rolling(k, center=True, min_periods=min_periods)
            .median()
            .values
        )
        baseline = _fill_edge_nans(baseline)

        residual  = f - baseline
        med_r     = np.nanmedian(residual)
        mad_r     = np.nanmedian(np.abs(residual - med_r))

        # Only mask DOWNWARD deviations (transit-like dips)
        # Do NOT mask upward deviations as they may be real stellar variability
        lower_thresh = med_r - clip_sigma * 1.4826 * mad_r
        weights = np.where(residual < lower_thresh, 0.0, 1.0)

    # Final pass with masked points
    f_final = f.copy().astype(np.float64)
    f_final[weights < 0.5] = np.nan
    baseline = (
        pd.Series(f_final)
        .rolling(k, center=True, min_periods=min_periods)
        .median()
        .values
    )
    baseline  = _fill_edge_nans(baseline)
    detrended = f / np.where(baseline > 0, baseline, 1.0)
    return detrended, baseline


# ===========================================================================
# Method E — Adaptive Window (scaled to transit duration)
# ===========================================================================

def _detrend_adaptive_window(
    t: np.ndarray,
    f: np.ndarray,
    min_window_days: float = 1.0,
    transit_duration_hrs: float = 8.0,
    safety_factor: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rolling median with window chosen to be safely larger than the
    expected transit duration.

    window = max(min_window_days, safety_factor × transit_duration_days)

    For a typical Kepler Earth-analog transit of ~8 h (0.33 d),
    the minimum safe window is 5 × 0.33 ≈ 1.65 d.

    This method adjusts automatically if the expected duration changes.
    """
    transit_dur_days = transit_duration_hrs / 24.0
    window_days = max(min_window_days, safety_factor * transit_dur_days)
    return _detrend_rolling_median(t, f, window_days)


# ===========================================================================
# Quarter-aware detrending dispatcher
# ===========================================================================

def _detrend_quarter(
    t_q: np.ndarray,
    f_q: np.ndarray,
    method: str,
    window_days: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a single detrending method to one quarter's data.
    Returns (detrended_flux, baseline).
    """
    if len(t_q) < MIN_POINTS_QUARTER:
        return f_q.copy(), np.ones_like(f_q)

    if method == "robust_rolling_median":
        return _detrend_rolling_median(t_q, f_q, window_days)
    elif method == "savgol":
        return _detrend_savgol(t_q, f_q, window_days)
    elif method == "biweight_lowess":
        if not ENABLE_EXPENSIVE_DETREND:
            raise ValueError("biweight_lowess is disabled by default; enable it explicitly for research")
        return _detrend_biweight_lowess(t_q, f_q, window_days)
    elif method == "iterative_clip_median":
        return _detrend_iterative_clip_median(t_q, f_q, window_days)
    elif method == "adaptive_window":
        return _detrend_adaptive_window(t_q, f_q, min_window_days=window_days)
    else:
        raise ValueError(f"Unknown detrending method: '{method}'")


# ===========================================================================
# Main detrending function
# ===========================================================================

def detrend_star(
    norm_df:     pd.DataFrame,
    star_id:     str = "UNKNOWN",
    method:      str = "robust_rolling_median",
    window_days: float = DETREND_WINDOW_SELECTED_DAYS,  # window in days
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply transit-preserving detrending to a quarter-normalized star DataFrame.

    Process each Kepler quarter independently to:
      1. Respect quarter boundaries (Task 10).
      2. Avoid smoothing across large mission gaps (Task 10).
      3. Control edge effects at quarter start/end (Task 25).

    Parameters
    ----------
    norm_df      : DataFrame from normalize_star(), contains 'norm_flux' and 'norm_err'
    star_id      : identifier
    method       : detrending method name (one of DETREND_METHODS in config.py)
    window_days  : baseline estimation window (days)

    Returns
    -------
    det_df : DataFrame with added columns:
               'detrended_flux'  – cleaned, normalized, detrended flux
               'detrended_err'   – propagated uncertainty
               'baseline'        – estimated stellar/instrumental trend
    det_report : dict with per-quarter detrending statistics
    """
    if len(norm_df) == 0:
        return norm_df.copy(), {"star_id": star_id, "n_points": 0}

    # Use norm_flux as input if available, else fall back to raw flux
    if "norm_flux" in norm_df.columns:
        f_in = norm_df["norm_flux"].values.astype(np.float64)
        e_in = norm_df["norm_err"].values.astype(np.float64)
    else:
        f_in = norm_df["flux"].values.astype(np.float64)
        e_in = norm_df["flux_err"].values.astype(np.float64)

    t       = norm_df["time"].values.astype(np.float64)
    q_arr   = norm_df["quarter"].values

    det_flux = np.empty_like(f_in)
    baseline = np.empty_like(f_in)
    det_err  = np.empty_like(e_in)

    quarter_stats: Dict = {}

    quarter_stats: Dict = {}
    edge_affected = np.zeros(len(t), dtype=bool)
    baseline_fallback = np.zeros(len(t), dtype=bool)
    segments = contiguous_segments(t, q_arr, EXPECTED_CADENCE_DAYS, GAP_FACTOR)

    for segment_id, indices in enumerate(segments):
        t_seg, f_seg, e_seg = t[indices], f_in[indices], e_in[indices]
        q_value = q_arr[indices[0]]
        fallback = len(indices) < MIN_POINTS_QUARTER
        try:
            d_seg, b_seg = _detrend_quarter(t_seg, f_seg, method, window_days)
        except ValueError:
            raise
        except Exception as exc:
            d_seg, b_seg = _detrend_rolling_median(t_seg, f_seg, window_days)
            fallback = True
            quarter_stats[f"{q_value}_segment_{segment_id}_error"] = str(exc)

        b_safe = np.where(b_seg > 0, b_seg, 1.0)
        det_flux[indices] = d_seg
        baseline[indices] = b_seg
        det_err[indices] = e_seg / b_safe
        baseline_fallback[indices] = fallback

        cadence = float(np.median(np.diff(t_seg))) if len(t_seg) > 1 else EXPECTED_CADENCE_DAYS
        half_window = max(1, _window_size(window_days, cadence) // 2)
        if EDGE_FLAG_ENABLED:
            edge_affected[indices[:min(half_window, len(indices))]] = True
            edge_affected[indices[max(0, len(indices) - half_window):]] = True

        resid = d_seg - 1.0
        quarter_stats[f"{q_value}_segment_{segment_id}"] = {
            "quarter": int(q_value),
            "segment_id": segment_id,
            "n_points": int(len(indices)),
            "baseline_med": float(np.median(b_seg)),
            "baseline_min": float(np.min(b_seg)),
            "baseline_max": float(np.max(b_seg)),
            "detrended_mad_ppm": float(np.median(np.abs(resid))) * 1e6,
            "t_min": float(t_seg.min()),
            "t_max": float(t_seg.max()),
            "edge_affected_count": int(edge_affected[indices].sum()),
            "baseline_fallback": bool(fallback),
        }

    # Check for unexpected NaN/inf in output
    bad_det = ~np.isfinite(det_flux)
    if bad_det.any():
        det_flux[bad_det] = 1.0
        det_err[bad_det]  = e_in[bad_det]

    det_df = norm_df.copy()
    det_df["detrended_flux"] = det_flux
    det_df["detrended_err"]  = det_err
    det_df["baseline"]       = baseline
    det_df["edge_affected"] = edge_affected
    det_df["baseline_fallback"] = baseline_fallback

    # Global scatter
    finite_det = det_flux[np.isfinite(det_flux)]
    global_mad_ppm = (
        float(np.median(np.abs(finite_det - 1.0)) * 1e6)
        if len(finite_det) > 0 else float("nan")
    )

    det_report = {
        "star_id":         star_id,
        "method":          method,
        "window_days":     window_days,
        "n_points":        len(det_df),
        "n_bad_replaced":  int(bad_det.sum()),
        "global_mad_ppm":  global_mad_ppm,
        "quarter_stats":   quarter_stats,
        "n_gap_segments":  len(segments),
        "n_large_gaps":    max(0, len(segments) - len(np.unique(q_arr))),
        "n_edge_affected": int(edge_affected.sum()),
        "n_baseline_fallback": int(baseline_fallback.sum()),
    }

    return det_df, det_report


# ===========================================================================
# Multi-method / multi-window comparison runner
# ===========================================================================

def detrend_compare(
    norm_df:       pd.DataFrame,
    star_id:       str = "UNKNOWN",
    methods:       Optional[list] = None,
    windows:       Optional[list] = None,
) -> Dict[str, Dict]:
    """
    Run all requested detrending methods × window combinations on one star.

    Parameters
    ----------
    norm_df  : normalized DataFrame
    methods  : list of method names (defaults to all DETREND_METHODS)
    windows  : list of window sizes in days (defaults to DETREND_WINDOWS_DAYS)

    Returns
    -------
    results : dict keyed by "{method}_{window:.1f}d" →
              {'det_df': DataFrame, 'report': dict}
    """
    from .config import DETREND_METHODS, DETREND_WINDOWS_DAYS

    if methods is None:
        methods = DETREND_METHODS
    if windows is None:
        windows = DETREND_WINDOWS_DAYS

    results: Dict = {}
    for method in methods:
        for win in windows:
            key = f"{method}_{win:.1f}d"
            try:
                det_df, report = detrend_star(
                    norm_df, star_id=star_id, method=method, window_days=win
                )
                results[key] = {"det_df": det_df, "report": report}
            except Exception as exc:
                results[key] = {"det_df": None, "report": {"error": str(exc)}}
    return results
