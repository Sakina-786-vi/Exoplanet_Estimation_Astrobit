"""
preprocessing/pipeline.py
=========================
Task 19 — Blind Production Preprocessor

The single public entry-point for the preprocessing pipeline.

    preprocess_star(raw_df)  →  result_dict

DESIGN CONTRACT (Task 27)
--------------------------
INPUT (required):
  * time, flux, flux_err, quality, quarter   ← the only columns needed

INPUT (must NOT be required):
  * true period, epoch, depth, duration, label

OUTPUT:
  * time            – cleaned, monotonic time axis
  * flux            – raw flux (cleaned)
  * norm_flux       – quarter-normalized flux
  * detrended_flux  – detrended flux (final analysis product)
  * flux_err        – propagated uncertainty (never discarded)
  * norm_err        – normalized uncertainty
  * detrended_err   – detrended uncertainty
  * quarter         – quarter assignment
  * baseline        – estimated stellar/instrumental baseline
  * qc              – QC diagnostic dict
  * norm_info       – normalization statistics dict
  * det_info        – detrending statistics dict

LEAKAGE RULE (Task 18)
-----------------------
This function works identically on TRAIN, DEV, and PRIVATE.
Ground truth is NEVER passed in or used inside this function.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config import DETREND_METHOD_SELECTED, DETREND_WINDOW_SELECTED_DAYS
from .detrend import detrend_star
from .normalize import normalize_star
from .qc import run_full_qc


def preprocess_star(
    raw_df:       pd.DataFrame,
    star_id:      str = "UNKNOWN",
    method:       str = DETREND_METHOD_SELECTED,
    window_days:  float = DETREND_WINDOW_SELECTED_DAYS,
    remove_outliers: bool = False,
    config:       Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Blind production preprocessor for a single Kepler star.

    Parameters
    ----------
    raw_df          : raw DataFrame loaded from Parquet
                      Required columns: time, flux, flux_err, quality, quarter
    star_id         : KIC identifier string (for diagnostics only)
    method          : detrending method name (from config.DETREND_METHODS)
    window_days     : detrending window in days
    remove_outliers : if True, remove flagged outliers from analysis stream
    config          : optional dict to override specific parameters

    Returns
    -------
    result : dict with all output arrays and diagnostic information
    """
    # --- Apply config overrides if provided ----------------------------------
    if config:
        method      = config.get("method",      method)
        window_days = config.get("window_days", window_days)
        remove_outliers = config.get("remove_outliers", remove_outliers)

    # =========================================================================
    # Phase 1: Quality Control (Tasks 2–8)
    # =========================================================================
    raw_copy, clean_df, qc_report = run_full_qc(
        raw_df,
        star_id=star_id,
        remove_outliers=remove_outliers,
    )

    # If QC left nothing usable, return an empty result
    if len(clean_df) == 0:
        return _empty_result(star_id, qc_report)

    # =========================================================================
    # Phase 2a: Quarter-wise Normalization (Tasks 9–11)
    # =========================================================================
    norm_df, norm_report = normalize_star(
        clean_df,
        star_id=star_id,
        # truth_row intentionally NOT passed — this is production code
    )

    # =========================================================================
    # Phase 2b: Detrending (Tasks 12–16)
    # =========================================================================
    det_df, det_report = detrend_star(
        norm_df,
        star_id=star_id,
        method=method,
        window_days=window_days,
    )

    # =========================================================================
    # Build output dict
    # =========================================================================
    result: Dict[str, Any] = {
        # --- Identification ---
        "star_id":       star_id,
        "ok":            True,
        "n_points":      len(det_df),

        # --- Time axis ---
        "time":          det_df["time"].values.astype(np.float64),

        # --- Flux at each stage ---
        "flux":          clean_df["flux"].values.astype(np.float64),
        "norm_flux":     det_df["norm_flux"].values.astype(np.float64)
                         if "norm_flux" in det_df.columns
                         else det_df["flux"].values.astype(np.float64),
        "detrended_flux": det_df["detrended_flux"].values.astype(np.float64),

        # --- Uncertainties (never discarded) ---
        "flux_err":      clean_df["flux_err"].values.astype(np.float64),
        "norm_err":      det_df["norm_err"].values.astype(np.float64)
                         if "norm_err" in det_df.columns
                         else clean_df["flux_err"].values.astype(np.float64),
        "detrended_err": det_df["detrended_err"].values.astype(np.float64),

        # --- Baseline ---
        "baseline":      det_df["baseline"].values.astype(np.float64),

        # --- Quarter assignment ---
        "quarter":       det_df["quarter"].values,

        # --- Outlier flags (preserved for reference) ---
        "outlier_flag":  det_df["outlier_flag"].values
                         if "outlier_flag" in det_df.columns
                         else np.zeros(len(det_df), dtype=int),
        "negative_outlier_flag": clean_df["negative_outlier_flag"].values
                 if "negative_outlier_flag" in clean_df.columns
                 else np.zeros(len(clean_df), dtype=int),
        "edge_affected": det_df["edge_affected"].values
                 if "edge_affected" in det_df.columns
                 else np.zeros(len(det_df), dtype=bool),

        # --- Diagnostic dicts ---
        "qc":            qc_report,
        "norm_info":     norm_report,
        "det_info":      det_report,

        # --- Method metadata ---
        "method":        method,
        "window_days":   window_days,
    }

    # Integrity check
    _check_output_integrity(result, star_id)

    return result


def _empty_result(star_id: str, qc_report: Dict) -> Dict[str, Any]:
    """Return a minimal result dict for a star that failed QC."""
    return {
        "star_id":        star_id,
        "ok":             False,
        "n_points":       0,
        "time":           np.array([]),
        "flux":           np.array([]),
        "norm_flux":      np.array([]),
        "detrended_flux": np.array([]),
        "flux_err":       np.array([]),
        "norm_err":       np.array([]),
        "detrended_err":  np.array([]),
        "baseline":       np.array([]),
        "quarter":        np.array([]),
        "outlier_flag":   np.array([]),
        "negative_outlier_flag": np.array([]),
        "edge_affected":  np.array([], dtype=bool),
        "qc":             qc_report,
        "norm_info":      {},
        "det_info":       {},
        "method":         "none",
        "window_days":    0.0,
    }


def _check_output_integrity(result: Dict, star_id: str) -> None:
    """
    Verify output arrays do not contain unexpected NaN/inf.
    Log warnings but do NOT raise — downstream code must be robust.
    (Task 29 failure conditions)
    """
    import warnings

    arrays_to_check = ["time", "detrended_flux", "detrended_err", "baseline"]
    for key in arrays_to_check:
        arr = result.get(key, np.array([]))
        if len(arr) == 0:
            continue
        n_bad = (~np.isfinite(arr)).sum()
        if n_bad > 0:
            warnings.warn(
                f"[{star_id}] Output '{key}' contains {n_bad} non-finite values. "
                "Investigate detrending edge cases.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Check time ordering
    t = result.get("time", np.array([]))
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        warnings.warn(
            f"[{star_id}] Output time array is not strictly monotonically increasing!",
            RuntimeWarning,
            stacklevel=3,
        )


def preprocess_batch(
    file_paths:     list,
    method:         str = DETREND_METHOD_SELECTED,
    window_days:    float = DETREND_WINDOW_SELECTED_DAYS,
    remove_outliers: bool = True,
    verbose:        bool = True,
) -> Dict[str, Dict]:
    """
    Run preprocess_star() on a batch of Parquet file paths.

    Parameters
    ----------
    file_paths : list of pathlib.Path or str objects
    method, window_days, remove_outliers : preprocessing parameters
    verbose    : print progress

    Returns
    -------
    results : {star_id: result_dict}
    """
    import os
    import time as time_module

    results: Dict[str, Dict] = {}
    t_start = time_module.perf_counter()

    for i, fpath in enumerate(file_paths, 1):
        star_id = os.path.basename(str(fpath)).replace(".parquet", "")
        t0 = time_module.perf_counter()

        try:
            raw_df = pd.read_parquet(fpath)
            res    = preprocess_star(
                raw_df,
                star_id=star_id,
                method=method,
                window_days=window_days,
                remove_outliers=remove_outliers,
            )
        except Exception as e:
            import traceback
            res = {
                "star_id": star_id,
                "ok":      False,
                "error":   str(e),
                "traceback": traceback.format_exc(),
            }

        results[star_id] = res
        elapsed = time_module.perf_counter() - t0

        if verbose:
            status = "OK" if res.get("ok", False) else "FAIL"
            n_pts  = res.get("n_points", 0)
            print(
                f"  [{i:3d}/{len(file_paths)}] {star_id}  "
                f"{status}  n={n_pts:5d}  {elapsed:.2f}s",
                flush=True,
            )

    total_elapsed = time_module.perf_counter() - t_start
    if verbose:
        n_ok   = sum(1 for r in results.values() if r.get("ok", False))
        n_fail = len(results) - n_ok
        print(
            f"\nBatch complete: {n_ok} OK, {n_fail} failed, "
            f"total {total_elapsed:.1f}s "
            f"({total_elapsed / len(file_paths):.2f}s/star)",
            flush=True,
        )

    return results
