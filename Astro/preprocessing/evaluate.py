"""
preprocessing/evaluate.py
=========================
Transit preservation evaluation, injection-recovery tests, and method
selection scoring (Tasks 13, 22–26, 28).

LEAKAGE RULE (Task 18)
-----------------------
Ground-truth period / epoch / depth / duration may ONLY be used here —
for measuring transit preservation and selecting the best method.

They must NEVER be passed into the production preprocessor.
The production function preprocess_star() is entirely blind to truth.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import periodogram

from .config import (
    INJECTION_DEPTHS_PPM,
    INJECTION_DURATIONS_HRS,
    INJECTION_DURATION_HRS,
    INJECTION_PERIODS_DAYS,
    N_INJECTION_STARS,
    EXPECTED_CADENCE_DAYS,
    GAP_FACTOR,
    INJECTION_DISTORTION_JUMP_PPM,
    INJECTION_PARTIAL_MIN_RATIO,
    INJECTION_PRESERVED_MIN_RATIO,
)
from .qc import contiguous_segments


def evaluate_on_train(results, truth_df, metric_fn) -> pd.DataFrame:
    """Evaluate TRAIN choices; this report is explicitly in-sample."""
    return _evaluate_dataset(results, truth_df, metric_fn, "TRAIN DEVELOPMENT / IN-SAMPLE")


def evaluate_on_dev(results, truth_df, metric_fn) -> pd.DataFrame:
    """Evaluate a frozen strategy on DEV with no TRAIN fallback."""
    return _evaluate_dataset(results, truth_df, metric_fn, "DEV INDEPENDENT VALIDATION")


def _evaluate_dataset(results, truth_df, metric_fn, dataset) -> pd.DataFrame:
    if results is None or len(results) == 0:
        raise RuntimeError(f"{dataset.split()[0]} data not available")
    rows = []
    for star_id, result in results.items():
        if result.get("ok", False):
            row = metric_fn(star_id, result, truth_df)
            row["dataset"] = dataset
            rows.append(row)
    return pd.DataFrame(rows)

# ===========================================================================
# Transit window helpers
# ===========================================================================

def _transit_mask(
    time:         np.ndarray,
    period:       float,
    t0:           float,
    duration_hrs: float,
) -> np.ndarray:
    """
    Return a boolean mask: True = in-transit cadence.

    Uses a simple box model: all cadences within ±(duration/2) of
    any transit centre are flagged.
    """
    dur_days = duration_hrs / 24.0
    half_dur = dur_days / 2.0

    # Phase-fold to nearest transit centre
    phase = ((time - t0) % period)
    # Wrap to [-P/2, P/2]
    phase = np.where(phase > period / 2, phase - period, phase)
    return np.abs(phase) <= half_dur


# ===========================================================================
# Task 13 — Transit preservation measurement
# ===========================================================================

def measure_transit_preservation(
    time:         np.ndarray,
    flux:         np.ndarray,
    period:       float,
    t0:           float,
    depth_ppm:    float,
    duration_hrs: float,
    label:        str = "",
) -> Dict:
    """
    Measure how well a transit signal is preserved in a (normalized/detrended)
    light curve.

    Metrics computed:
      * measured_depth_ppm : median in-transit vs. out-of-transit flux × 1e6
      * preservation_ratio : measured / true depth
      * local_snr          : depth / local scatter (using out-of-transit MAD)
      * n_in_transit       : number of cadences inside transit windows
      * n_transits_visible : number of distinct transit windows with ≥1 point

    Parameters
    ----------
    time, flux : 1-D arrays (detrended, normalized to ~1.0)
    period, t0 : orbital parameters (from truth; evaluation only)
    depth_ppm  : true transit depth in ppm
    duration_hrs : true transit duration in hours
    label      : descriptive string for logging

    Returns
    -------
    report : dict with all metrics
    """
    report: Dict = {
        "label":              label,
        "true_depth_ppm":     depth_ppm,
        "measured_depth_ppm": float("nan"),
        "preservation_ratio": float("nan"),
        "local_snr":          float("nan"),
        "n_in_transit":       0,
        "n_transits_visible": 0,
        "warning":            "",
    }

    if len(time) < 10 or period <= 0 or duration_hrs <= 0:
        report["warning"] = "Insufficient data or invalid parameters"
        return report

    in_transit = _transit_mask(time, period, t0, duration_hrs)
    n_in       = int(in_transit.sum())

    if n_in < 3:
        report["warning"] = "Fewer than 3 in-transit cadences; cannot measure depth"
        return report

    report["n_in_transit"] = n_in

    # Count distinct transit windows
    t_starts = []
    dur_days  = duration_hrs / 24.0
    n_transits_obs = int(np.floor((time.max() - t0) / period)) - \
                     int(np.floor((time.min() - t0) / period))
    for k in range(max(0, int(np.floor((time.min() - t0) / period))),
                   int(np.ceil((time.max() - t0) / period)) + 1):
        tc = t0 + k * period
        if time.min() <= tc <= time.max():
            # Count cadences in this window
            win_mask = np.abs(time - tc) <= dur_days / 2
            if win_mask.sum() >= 1:
                t_starts.append(tc)
    report["n_transits_visible"] = len(t_starts)

    # Median in-transit vs. out-of-transit flux
    f_in  = flux[in_transit]
    f_out = flux[~in_transit]

    if len(f_out) < 10:
        report["warning"] = "Insufficient out-of-transit cadences"
        return report

    med_in  = float(np.median(f_in))
    med_out = float(np.median(f_out))

    if med_out <= 0:
        report["warning"] = "Out-of-transit median <= 0"
        return report

    measured_depth_ppm = (med_out - med_in) / med_out * 1e6
    report["measured_depth_ppm"] = measured_depth_ppm
    report["preservation_ratio"] = measured_depth_ppm / depth_ppm if depth_ppm > 0 else float("nan")

    # Local SNR = depth / out-of-transit MAD
    mad_out = float(np.median(np.abs(f_out - med_out)))
    if mad_out > 0:
        report["local_snr"] = (measured_depth_ppm / 1e6) / (1.4826 * mad_out)

    return report


# ===========================================================================
# Task 26 — Synthetic transit injection
# ===========================================================================

def inject_transit(
    time:         np.ndarray,
    flux:         np.ndarray,
    period:       float,
    t0:           float,
    depth_ppm:    float,
    duration_hrs: float,
    shape:        str = "box",
) -> np.ndarray:
    """
    Inject a synthetic transit signal into a real light curve.

    The injection uses a simple box model (no limb darkening) to
    create a known signal of controlled depth and duration.

    This is used ONLY for preprocessing validation — it does NOT
    contaminate the actual competition dataset.

    Parameters
    ----------
    time, flux : original light curve
    period     : orbital period (days)
    t0         : reference transit centre (BKJD days)
    depth_ppm  : transit depth (parts per million)
    duration_hrs : full transit duration (hours)
    shape      : "box" (only implemented option)

    Returns
    -------
    injected_flux : flux with transit signal added
    """
    injected = flux.copy().astype(np.float64)
    depth_frac = depth_ppm / 1e6

    in_transit = _transit_mask(time, period, t0, duration_hrs)
    injected[in_transit] *= (1.0 - depth_frac)

    return injected


def _location_epochs(time: np.ndarray, quarter: np.ndarray) -> Dict[str, float]:
    """Choose deterministic observed timestamps for each injection location."""
    segments = contiguous_segments(time, quarter, EXPECTED_CADENCE_DAYS, GAP_FACTOR)
    if not segments:
        return {}
    epochs = {"middle": float(time[segments[0][len(segments[0]) // 2]])}
    first = segments[0]
    last = segments[-1]
    epochs["segment_start"] = float(time[first[0]])
    epochs["segment_end"] = float(time[last[-1]])
    if len(segments) > 1:
        epochs["before_gap"] = float(time[segments[0][-1]])
        epochs["after_gap"] = float(time[segments[1][0]])
    else:
        epochs["before_gap"] = epochs["segment_end"]
        epochs["after_gap"] = epochs["segment_start"]
    changes = np.flatnonzero(quarter[1:] != quarter[:-1])
    if len(changes):
        epochs["quarter_boundary"] = float(time[changes[0] + 1])
    else:
        epochs["quarter_boundary"] = epochs["middle"]
    return epochs


def _injection_category(depth_recovery_ratio: float, edge_affected: bool,
                        gap_affected: bool, signal_jump_ppm: float) -> str:
    if edge_affected:
        return "EDGE_AFFECTED"
    if gap_affected:
        return "GAP_AFFECTED"
    if np.isfinite(signal_jump_ppm) and signal_jump_ppm > INJECTION_DISTORTION_JUMP_PPM:
        return "DISTORTED"
    if not np.isfinite(depth_recovery_ratio) or depth_recovery_ratio < INJECTION_PARTIAL_MIN_RATIO:
        return "SUPPRESSED"
    if depth_recovery_ratio < INJECTION_PRESERVED_MIN_RATIO:
        return "PARTIALLY_SUPPRESSED"
    return "PRESERVED"


def injection_recovery_test(
    time:          np.ndarray,
    flux:          np.ndarray,
    flux_err:      np.ndarray,
    quarter:       np.ndarray,
    detrend_fn,    # callable(norm_df) -> det_df
    star_id:       str = "UNKNOWN",
    depths_ppm:    Optional[List[float]] = None,
    periods_days:  Optional[List[float]] = None,
    duration_hrs:  float = INJECTION_DURATION_HRS,
    durations_hrs: Optional[List[float]] = None,
    method: str = "unknown",
    window_days: float = float("nan"),
    location_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Inject synthetic transits at multiple (period, depth) combinations and
    measure how well the preprocessing preserves them.

    Parameters
    ----------
    time, flux, flux_err, quarter : clean (but NOT detrended) arrays
    detrend_fn   : callable that takes a DataFrame and returns a detrended DataFrame
    depths_ppm   : list of depths to test (ppm)
    periods_days : list of orbital periods to test (days)
    duration_hrs : transit duration (hours) — fixed for all injections

    Returns
    -------
    results_df : DataFrame with one row per (star, period, depth) combination:
                 columns include measured_depth_ppm, preservation_ratio, local_snr
    """
    if depths_ppm is None:
        depths_ppm = INJECTION_DEPTHS_PPM
    if periods_days is None:
        periods_days = INJECTION_PERIODS_DAYS
    if durations_hrs is None:
        durations_hrs = INJECTION_DURATIONS_HRS if duration_hrs == INJECTION_DURATION_HRS else [duration_hrs]
    if location_types is None:
        from .config import INJECTION_PHASES
        location_types = INJECTION_PHASES

    location_epochs = _location_epochs(time, quarter)

    rows = []
    for period in periods_days:
        for depth in depths_ppm:
            for duration in durations_hrs:
                for location_type in location_types:
                    t0_ref = location_epochs.get(location_type, float(time[len(time) // 2]))
                    f_inj = inject_transit(time, flux, period, t0_ref, depth, duration)
                    df_inj = pd.DataFrame({
                        "time": time, "norm_flux": f_inj, "norm_err": flux_err,
                        "flux": f_inj, "flux_err": flux_err, "quarter": quarter,
                    })
                    try:
                        det_df = detrend_fn(df_inj)
                        det_flux = det_df["detrended_flux"].values if "detrended_flux" in det_df else f_inj
                    except Exception as e:
                        rows.append({"star_id": star_id, "method": method,
                                     "window_days": window_days, "duration_hours": duration,
                                     "depth_ppm": depth, "location_type": location_type,
                                     "error": str(e)})
                        continue

                    before = measure_transit_preservation(time, f_inj, period, t0_ref, depth, duration)
                    rep = measure_transit_preservation(time, det_flux, period, t0_ref, depth, duration)
                    in_mask = _transit_mask(time, period, t0_ref, duration)
                    edge_values = det_df.get("edge_affected", pd.Series(False, index=range(len(time))))
                    fallback_values = det_df.get("baseline_fallback", pd.Series(False, index=range(len(time))))
                    edge = bool(np.asarray(edge_values)[in_mask].any())
                    fallback = bool(np.asarray(fallback_values)[in_mask].any())
                    gap_mask = np.zeros(len(time), dtype=bool)
                    segments = contiguous_segments(time, quarter, EXPECTED_CADENCE_DAYS, GAP_FACTOR)
                    for segment in segments:
                        gap_mask[segment[:1]] = True
                        gap_mask[segment[-1:]] = True
                    gap = bool((in_mask & gap_mask).any())
                    signal_jump = (float(np.nanmax(det_flux[in_mask]) - np.nanmin(det_flux[in_mask])) * 1e6
                                   if in_mask.any() else float("nan"))
                    ratio = rep["preservation_ratio"]
                    rows.append({
                        "star_id": star_id, "method": method, "window_days": window_days,
                        "period_days": period, "duration_hours": duration, "depth_ppm": depth,
                        "location_type": location_type, "injected_depth": depth,
                        "recovered_depth": rep["measured_depth_ppm"], "depth_recovery_ratio": ratio,
                        "signal_before": before["measured_depth_ppm"], "signal_after": rep["measured_depth_ppm"],
                        "signal_recovery_ratio": (rep["measured_depth_ppm"] / before["measured_depth_ppm"]
                                                   if np.isfinite(before["measured_depth_ppm"]) and before["measured_depth_ppm"] else float("nan")),
                        "edge_affected": edge, "gap_affected": gap, "baseline_fallback": fallback,
                        "failure_category": _injection_category(ratio, edge, gap, signal_jump),
                        "local_snr": rep["local_snr"], "warning": rep["warning"],
                    })

def artifact_diagnostics(time: np.ndarray, flux: np.ndarray, quarter: np.ndarray) -> Dict:
    """Report residual correlation, periodic concentration, and boundary jumps."""
    residual = np.asarray(flux, dtype=float) - 1.0
    finite = np.isfinite(residual)
    residual = residual[finite]
    autocorr_lag1 = float("nan")
    if len(residual) > 2 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
        autocorr_lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    frequencies, power = periodogram(residual) if len(residual) > 4 else (np.array([]), np.array([]))
    if len(power) > 2:
        peak = int(np.argmax(power[1:]) + 1)
        median_power = float(np.median(power[1:]))
        peak_ratio = float(power[peak] / median_power) if median_power > 0 else float("inf")
        dominant_period_days = float(1.0 / frequencies[peak] * EXPECTED_CADENCE_DAYS) if frequencies[peak] > 0 else float("nan")
    else:
        peak_ratio = float("nan")
        dominant_period_days = float("nan")

    jumps = np.abs(np.diff(flux)) * 1e6
    gap_mask = np.diff(time) > EXPECTED_CADENCE_DAYS * GAP_FACTOR
    quarter_mask = quarter[1:] != quarter[:-1]
    return {
        "residual_autocorrelation_lag1": autocorr_lag1,
        "dominant_period_days": dominant_period_days,
        "periodogram_peak_to_median": peak_ratio,
        "median_edge_jump": float(np.median(jumps)) if len(jumps) else float("nan"),
        "median_gap_edge_jump": float(np.median(jumps[gap_mask])) if gap_mask.any() else float("nan"),
        "median_quarter_boundary_jump": float(np.median(jumps[quarter_mask])) if quarter_mask.any() else float("nan"),
        "suspicious_autocorrelation": bool(np.isfinite(autocorr_lag1) and abs(autocorr_lag1) > 0.5),
        "suspicious_periodicity": bool(np.isfinite(peak_ratio) and peak_ratio > 10.0),
    }


# ===========================================================================
# Task 24 — Detrending method scoring framework
# ===========================================================================

def score_detrending_method(
    train_results: List[Dict],
    dev_results:   List[Dict],
    method_key:    str,
) -> Dict:
    """
    Score a detrending method using a multi-criteria framework.

    Rewards:
      1. Transit depth preservation ratio (closer to 1.0 is better)
      2. Shallow signal preservation specifically
      3. Reduced noise (lower global_mad_ppm)
      4. Stable behavior across stars
      5. TRAIN/DEV consistency (penalizes overfitting)

    Penalizes:
      * Transit suppression (preservation_ratio < 0.8)
      * Very high noise
      * Large TRAIN/DEV gap

    Parameters
    ----------
    train_results : list of per-star result dicts (from TRAIN)
    dev_results   : list of per-star result dicts (from DEV)
    method_key    : string like "robust_rolling_median_1.5d"

    Returns
    -------
    score_dict : {
        "method": method_key,
        "train_preservation_mean": float,
        "dev_preservation_mean": float,
        "train_noise_mad_ppm": float,
        "dev_noise_mad_ppm": float,
        "train_dev_gap": float,
        "shallow_preservation": float,
        "composite_score": float,   (higher = better)
    }
    """

    def _extract(results: List[Dict], key: str, default=float("nan")) -> np.ndarray:
        vals = []
        for r in results:
            v = r.get(key, default)
            if v is not None and np.isfinite(float(v)):
                vals.append(float(v))
        return np.array(vals) if vals else np.array([default])

    # ---- TRAIN metrics -------------------------------------------------------
    tr_pres  = _extract(train_results, "preservation_ratio")
    tr_noise = _extract(train_results, "global_mad_ppm")
    tr_sh    = _extract(train_results, "shallow_preservation_ratio")   # shallow signals only

    # ---- DEV metrics ---------------------------------------------------------
    dv_pres  = _extract(dev_results, "preservation_ratio")
    dv_noise = _extract(dev_results, "global_mad_ppm")

    tr_pres_mean  = float(np.nanmedian(tr_pres))
    dv_pres_mean  = float(np.nanmedian(dv_pres)) if dev_results else float("nan")
    tr_noise_mean = float(np.nanmedian(tr_noise))
    dv_noise_mean = float(np.nanmedian(dv_noise)) if dev_results else float("nan")
    shallow_pres  = float(np.nanmedian(tr_sh)) if len(tr_sh) > 0 else float("nan")

    # ---- TRAIN/DEV consistency gap ------------------------------------------
    pres_gap  = abs(tr_pres_mean - dv_pres_mean) if np.isfinite(dv_pres_mean) else 0.0
    noise_gap = abs(tr_noise_mean - dv_noise_mean) if np.isfinite(dv_noise_mean) else 0.0

    # ---- Composite score -----------------------------------------------------
    # Weights reflect the primary science objective:
    #   1. Preserve transits (especially shallow) — weight 0.40
    #   2. Reduce noise — weight 0.20
    #   3. TRAIN/DEV consistency — weight 0.25
    #   4. Shallow signal specifically — weight 0.15

    # Normalize: 1.0 = perfect; 0 = bad
    pres_score    = min(tr_pres_mean, 1.0) if tr_pres_mean >= 0 else 0.0
    shallow_score = min(shallow_pres, 1.0) if np.isfinite(shallow_pres) else pres_score
    # Noise score: lower ppm is better; penalize >500 ppm, reward <200 ppm
    noise_score   = max(0.0, 1.0 - tr_noise_mean / 500.0)
    # Consistency: penalize large gap
    consist_score = max(0.0, 1.0 - pres_gap)

    composite = (
        0.40 * pres_score
        + 0.15 * shallow_score
        + 0.20 * noise_score
        + 0.25 * consist_score
    )

    return {
        "method":                  method_key,
        "train_preservation_mean": tr_pres_mean,
        "dev_preservation_mean":   dv_pres_mean,
        "train_noise_mad_ppm":     tr_noise_mean,
        "dev_noise_mad_ppm":       dv_noise_mean,
        "shallow_preservation":    shallow_pres,
        "train_dev_pres_gap":      pres_gap,
        "composite_score":         composite,
    }


# ===========================================================================
# Task 23 — Quantitative preprocessing metrics
# ===========================================================================

def compute_preprocessing_metrics(
    star_id:       str,
    time:          np.ndarray,
    raw_flux:      np.ndarray,
    norm_flux:     np.ndarray,
    det_flux:      np.ndarray,
    quarter:       np.ndarray,
    truth_row:     Optional[pd.Series] = None,
) -> Dict:
    """
    Compute comprehensive preprocessing quality metrics for one star.

    DATA QUALITY
      * retention_rate
    NOISE
      * raw, normalized, detrended robust scatter (MAD in ppm)
    TRANSIT PRESERVATION (if truth available)
      * depth before/after normalization and detrending
      * depth preservation ratios
      * local SNR improvement

    Returns a flat dict suitable for CSV export.
    """

    def _mad_ppm(f: np.ndarray) -> float:
        f = f[np.isfinite(f)]
        if len(f) < 2:
            return float("nan")
        med = np.median(f)
        if med <= 0:
            return float("nan")
        return float(np.median(np.abs(f - med)) / med * 1e6)

    row: Dict = {
        "star_id":                   star_id,
        "n_raw":                     len(raw_flux),
        "n_clean":                   len(det_flux),
        "retention_rate":            len(det_flux) / len(raw_flux) if len(raw_flux) > 0 else float("nan"),
        "raw_mad_ppm":               _mad_ppm(raw_flux),
        "norm_mad_ppm":              _mad_ppm(norm_flux),
        "det_mad_ppm":               _mad_ppm(det_flux),
        "noise_reduction_raw_to_det": float("nan"),
        # Transit metrics
        "true_depth_ppm":              float("nan"),
        "det_depth_ppm":               float("nan"),
        "preservation_ratio":          float("nan"),
        "local_snr_before":            float("nan"),
        "local_snr_after":             float("nan"),
        "n_transits_visible":          0,
        "has_truth":                   False,
    }

    raw_mad = row["raw_mad_ppm"]
    det_mad = row["det_mad_ppm"]
    if np.isfinite(raw_mad) and np.isfinite(det_mad) and raw_mad > 0:
        row["noise_reduction_raw_to_det"] = raw_mad / det_mad

    # Transit preservation
    if truth_row is not None and not truth_row.empty:
        try:
            period    = float(truth_row["period_days"])
            t0        = float(truth_row["epoch_t0"])
            depth_ppm = float(truth_row["depth_ppm"])
            dur_hrs   = float(truth_row["duration_hours"])

            row["true_depth_ppm"] = depth_ppm
            row["has_truth"]      = True

            # Before detrending (normalized flux)
            rep_norm = measure_transit_preservation(
                time, norm_flux, period, t0, depth_ppm, dur_hrs, "norm"
            )
            row["local_snr_before"] = rep_norm["local_snr"]

            # After detrending
            rep_det = measure_transit_preservation(
                time, det_flux, period, t0, depth_ppm, dur_hrs, "det"
            )
            row["det_depth_ppm"]     = rep_det["measured_depth_ppm"]
            row["preservation_ratio"] = rep_det["preservation_ratio"]
            row["local_snr_after"]    = rep_det["local_snr"]
            row["n_transits_visible"] = rep_det["n_transits_visible"]

        except Exception as e:
            row["truth_error"] = str(e)

    return row
