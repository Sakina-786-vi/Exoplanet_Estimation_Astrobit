"""
run_preprocessing.py
====================
Phase 1 + Phase 2 runner script.

Performs the following steps:
  1. Run full preprocessing on all TRAIN stars (269)
  2. Run full preprocessing on all DEV stars (89)
  3. Evaluate transit preservation on TRAIN + DEV using ground truth
  4. Compare all 5 detrending methods × 5 window sizes on a sample
  5. Select the best method based on the scoring framework
  6. Run injection-recovery tests on a TRAIN sample
  7. Save all processed data, diagnostics, and plots
  8. Print the final report

USAGE
-----
    py run_preprocessing.py

    # Faster development run on a small sample:
    py run_preprocessing.py --sample 10 --no-compare

    # Skip plot generation:
    py run_preprocessing.py --no-plots

OUTPUT
------
  data/processed/qc/         – QC-filtered per-star Parquet files
  data/processed/normalized/ – Normalized per-star Parquet files
  data/processed/detrended/  – Detrended per-star Parquet files
  data/diagnostics/          – CSV diagnostics + comparison table
  data/plots/                – PNG diagnostic plots
  data/config.json           – Final selected parameters
"""

import argparse
import json
import os
import sys
import time as time_module
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas as pd

# Make sure preprocessing package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import (
    preprocess_star,
    preprocess_batch,
    detrend_compare,
    normalize_star,
    measure_transit_preservation,
    injection_recovery_test,
    score_detrending_method,
    compute_preprocessing_metrics,
    run_full_qc,
    artifact_diagnostics,
)
from preprocessing.config import (
    TRAIN_DIR, DEV_DIR,
    TRAIN_LABELS, TRAIN_TRUTH,
    DEV_LABELS, DEV_TRUTH,
    OUTPUT_ROOT, QC_DIR, NORM_DIR, DETREND_DIR,
    DIAG_DIR, PLOT_DIR,
    DETREND_METHODS, DETREND_WINDOWS_DAYS,
    DETREND_METHOD_SELECTED, DETREND_WINDOW_SELECTED_DAYS,
    N_INJECTION_STARS,
    SAVE_PARQUET, SAVE_PLOTS,
)
from preprocessing.visualize import (
    plot_star_preprocessing,
    plot_method_comparison,
    plot_qc_summary,
    plot_transit_preservation,
)
from preprocessing.detrend import detrend_star

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===========================================================================
# Directory setup
# ===========================================================================

def setup_directories():
    for d in [QC_DIR, NORM_DIR, DETREND_DIR, DIAG_DIR,
              PLOT_DIR / "representative",
              PLOT_DIR / "method_comparison",
              PLOT_DIR / "summary"]:
        Path(d).mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Load truth tables
# ===========================================================================

def load_truth_tables():
    train_labels = pd.read_csv(TRAIN_LABELS)
    train_truth  = pd.read_csv(TRAIN_TRUTH)
    dev_labels   = pd.read_csv(DEV_LABELS)
    dev_truth    = pd.read_csv(DEV_TRUTH)

    # Merge for convenience
    train_truth = train_truth.merge(
        train_labels[["kepid", "label"]], on="kepid", how="left"
    )
    dev_truth = dev_truth.merge(
        dev_labels[["kepid", "label"]], on="kepid", how="left"
    )

    return train_labels, train_truth, dev_labels, dev_truth


# ===========================================================================
# Get truth row for a star
# ===========================================================================

def get_truth_row(star_id: str, truth_df: pd.DataFrame) -> pd.Series:
    """Return the truth row for star_id, or empty Series if not found."""
    kepid_str = star_id.replace("KIC_", "")
    try:
        kepid = int(kepid_str)
    except ValueError:
        return pd.Series(dtype=float)
    rows = truth_df[truth_df["kepid"] == kepid]
    if len(rows) == 0:
        return pd.Series(dtype=float)
    return rows.iloc[0]


# ===========================================================================
# Step 1 & 2: Run preprocessing on TRAIN and DEV
# ===========================================================================

def run_split(
    split_name: str,
    data_dir:   Path,
    truth_df:   pd.DataFrame,
    method:     str,
    window_days: float,
    sample:     int = 0,
    save_parquet: bool = True,
    save_plots:   bool = True,
    plot_stars:   list = None,
) -> tuple:
    """
    Run preprocessing on all stars in a split.

    Returns (results_dict, qc_records, metrics_records)
    """
    files = sorted(Path(data_dir).glob("*.parquet"))
    if sample > 0:
        files = files[:sample]

    print(f"\n{'='*60}")
    print(f"  Processing {split_name}: {len(files)} stars")
    print(f"  Method: {method}  Window: {window_days:.1f}d")
    print(f"{'='*60}")

    qc_records      = []
    metrics_records = []
    results_dict    = {}

    t_split_start = time_module.perf_counter()

    for i, fpath in enumerate(files, 1):
        star_id = fpath.stem   # e.g. "KIC_10002867"
        t0 = time_module.perf_counter()

        try:
            raw_df = pd.read_parquet(fpath)
            result = preprocess_star(
                raw_df,
                star_id=star_id,
                method=method,
                window_days=window_days,
            )
        except Exception as e:
            import traceback
            print(f"  [{i:3d}] {star_id}  ERROR: {e}")
            results_dict[star_id] = {"ok": False, "error": str(e)}
            continue

        elapsed = time_module.perf_counter() - t0
        results_dict[star_id] = result

        # ---- QC record -------------------------------------------------------
        qc = result.get("qc", {})
        qc_rec = {
            "star_id":             star_id,
            "split":               split_name,
            "original_points":     qc.get("original_points", 0),
            "valid_points":        qc.get("valid_points", 0),
            "quality_removed":     qc.get("quality_removed", 0),
            "nonfinite_removed":   qc.get("nonfinite_removed", 0),
            "n_time_nonfinite":    qc.get("n_time_nonfinite", 0),
            "n_removed_nonfinite_time": qc.get("n_removed_nonfinite_time", 0),
            "duplicate_removed":   qc.get("duplicate_removed", 0),
            "outlier_flagged":     qc.get("outlier_flagged", 0),
            "outlier_removed":     qc.get("n_outlier_removed", 0),
            "negative_extreme_retained": qc.get("n_negative_extreme_retained", 0),
            "percentage_retained": qc.get("percentage_retained", float("nan")),
            "quarter_count":       qc.get("quarter_count", 0),
            "large_gap_count":     qc.get("large_gap_count", 0),
            "small_quarter_count": qc.get("small_quarter_count", 0),
            "small_quarter_ids":   qc.get("small_quarter_ids", []),
            "time_span_days":      qc.get("time_span_days", float("nan")),
            "median_cadence_days": qc.get("median_cadence_days", float("nan")),
            "robust_scatter_ppm":  qc.get("robust_scatter_ppm", float("nan")),
            "severely_corrupted":  qc.get("severely_corrupted", False),
        }
        qc_records.append(qc_rec)

        # ---- Transit preservation metrics ------------------------------------
        truth_row = get_truth_row(star_id, truth_df)
        has_truth = (not truth_row.empty and
                     truth_row.get("injected", 0) == 1 and
                     np.isfinite(truth_row.get("period_days", float("nan"))))

        if result.get("ok", False) and result["n_points"] > 0:
            # Compute metrics
            t_arr  = result["time"]
            f_raw  = result["flux"]
            f_norm = result["norm_flux"]
            f_det  = result["detrended_flux"]
            q_arr  = result["quarter"]

            metrics = compute_preprocessing_metrics(
                star_id=star_id,
                time=t_arr,
                raw_flux=f_raw,
                norm_flux=f_norm,
                det_flux=f_det,
                quarter=q_arr,
                truth_row=truth_row if has_truth else None,
            )
            metrics["split"] = split_name
            metrics["method"] = method
            metrics["window_days"] = window_days
            metrics_records.append(metrics)

        # ---- Save Parquet outputs --------------------------------------------
        if save_parquet and result.get("ok", False):
            _save_processed(result, star_id, split_name)

        # ---- Diagnostic plot -------------------------------------------------
        do_plot = save_plots and (
            plot_stars is None or star_id in plot_stars
        )
        if do_plot and result.get("ok", False):
            try:
                plot_path = PLOT_DIR / "representative" / f"{split_name}_{star_id}.png"
                plot_star_preprocessing(
                    star_id=star_id,
                    raw_df=raw_df,
                    result=result,
                    truth_row=truth_row if has_truth else None,
                    save_path=plot_path,
                )
            except Exception as pe:
                print(f"    Warning: plot failed for {star_id}: {pe}")

        # Progress
        status = "OK" if result.get("ok", False) else "FAIL"
        n_pts  = result.get("n_points", 0)
        pct    = qc.get("percentage_retained", float("nan"))
        det_mad = float("nan")
        if result.get("ok", False) and "detrended_flux" in result:
            d = result["detrended_flux"]
            if len(d) > 0:
                det_mad = float(np.median(np.abs(d - 1.0)) * 1e6)

        transit_info = ""
        if has_truth:
            d_ppm = truth_row.get("depth_ppm", "?")
            transit_info = f"  [transit {d_ppm:.0f}ppm]"

        print(
            f"  [{i:3d}/{len(files)}] {star_id}  {status}  "
            f"n={n_pts:5d}  ret={pct:.1f}%  "
            f"det_mad={det_mad:5.0f}ppm  {elapsed:.2f}s{transit_info}",
            flush=True,
        )

    t_total = time_module.perf_counter() - t_split_start
    n_ok    = sum(1 for r in results_dict.values() if r.get("ok", False))
    print(
        f"\n  {split_name} summary: {n_ok}/{len(files)} OK, "
        f"total {t_total:.0f}s ({t_total/len(files):.2f}s/star)"
    )

    return results_dict, qc_records, metrics_records


def _save_processed(result: dict, star_id: str, split_name: str) -> None:
    """Save QC, normalized, and detrended data as Parquet files."""
    t    = result["time"]
    q    = result["quarter"]
    ofl  = result["outlier_flag"]

    # QC stream (cleaned, not yet normalized)
    qc_df = pd.DataFrame({
        "time":         t,
        "flux":         result["flux"],
        "flux_err":     result["flux_err"],
        "quarter":      q,
        "outlier_flag": ofl,
    })
    qc_path = QC_DIR / f"{split_name}_{star_id}.parquet"
    qc_df.to_parquet(qc_path, index=False)

    # Normalized stream
    norm_df = pd.DataFrame({
        "time":      t,
        "norm_flux": result["norm_flux"],
        "norm_err":  result["norm_err"],
        "quarter":   q,
    })
    norm_path = NORM_DIR / f"{split_name}_{star_id}.parquet"
    norm_df.to_parquet(norm_path, index=False)

    # Detrended stream (final analysis product)
    det_df = pd.DataFrame({
        "time":           t,
        "detrended_flux": result["detrended_flux"],
        "detrended_err":  result["detrended_err"],
        "baseline":       result["baseline"],
        "quarter":        q,
    })
    det_path = DETREND_DIR / f"{split_name}_{star_id}.parquet"
    det_df.to_parquet(det_path, index=False)


# ===========================================================================
# Step 3: Detrending method comparison on TRAIN sample
# ===========================================================================

def run_method_comparison(
    train_results:  dict,
    train_truth_df: pd.DataFrame,
    dev_results:    dict,
    dev_truth_df:   pd.DataFrame,
    sample_size:    int = 30,
    save_plots:     bool = True,
    include_dev:    bool = True,
) -> pd.DataFrame:
    """
    Compare all methods × windows on a sample of TRAIN stars,
    then validate on a sample of DEV stars.

    Returns a DataFrame with one row per (method, window) combination.
    """
    print("\n" + "="*60)
    print("  Method Comparison")
    print("="*60)

    # Pick representative TRAIN sample stars:
    # - earth_analog (hardest)
    # - shallow
    # - mid
    # - quiet (no transit)
    def get_sample_keys(truth_df: pd.DataFrame, results: dict, n: int) -> list:
        # Include all earth_analog and shallow, rest from no-transit
        has_transit = set(
            f"KIC_{k}" for k in truth_df[truth_df.get("injected", 0) == 1]["kepid"].astype(str)
        )
        transit_keys = [k for k in results if k in has_transit][:n // 2]
        quiet_keys   = [k for k in results if k not in has_transit][:n // 2]
        return transit_keys + quiet_keys

    sample_keys = get_sample_keys(train_truth_df, train_results, sample_size)
    print(f"  Comparing {len(DETREND_METHODS)} methods × {len(DETREND_WINDOWS_DAYS)} windows "
          f"on {len(sample_keys)} TRAIN stars")

    comparison_rows = []

    for star_id in sample_keys:
        tr_res = train_results.get(star_id)
        if not tr_res or not tr_res.get("ok", False):
            continue

        truth_row = get_truth_row(star_id, train_truth_df)
        has_truth = (
            not truth_row.empty and
            truth_row.get("injected", 0) == 1 and
            np.isfinite(truth_row.get("period_days", float("nan")))
        )

        # Build normalized DataFrame for this star
        t    = tr_res["time"]
        norm = tr_res["norm_flux"]
        nerr = tr_res["norm_err"]
        q    = tr_res["quarter"]

        norm_df = pd.DataFrame({
            "time":      t,
            "norm_flux": norm,
            "norm_err":  nerr,
            "flux":      norm,
            "flux_err":  nerr,
            "quarter":   q,
        })

        method_dets = {}  # for comparison plot

        for method in DETREND_METHODS:
            for win in DETREND_WINDOWS_DAYS:
                method_key = f"{method}_{win:.1f}d"
                try:
                    det_df, det_rep = detrend_star(
                        norm_df, star_id=star_id, method=method, window_days=win
                    )
                    d_flux = det_df["detrended_flux"].values

                    # Global scatter and residual RMS are both retained; neither
                    # is sufficient on its own to select a science strategy.
                    mad_ppm = float(np.median(np.abs(d_flux - 1.0)) * 1e6)
                    rms_ppm = float(np.sqrt(np.mean((d_flux - 1.0) ** 2)) * 1e6)
                    artifact = artifact_diagnostics(t, d_flux, q)

                    # Transit preservation
                    pres_ratio = float("nan")
                    snr_after  = float("nan")
                    shallow_ok = float("nan")

                    if has_truth:
                        pres = measure_transit_preservation(
                            t, d_flux,
                            float(truth_row["period_days"]),
                            float(truth_row["epoch_t0"]),
                            float(truth_row["depth_ppm"]),
                            float(truth_row["duration_hours"]),
                        )
                        pres_ratio = pres["preservation_ratio"]
                        snr_after  = pres["local_snr"]
                        if truth_row.get("bin", "") in ("earth_analog", "shallow"):
                            shallow_ok = pres_ratio

                    comparison_rows.append({
                        "star_id":              star_id,
                        "split":                "TRAIN",
                        "method":               method,
                        "window_days":          win,
                        "method_key":           method_key,
                        "global_mad_ppm":       mad_ppm,
                        "residual_rms_ppm":      rms_ppm,
                        "runtime_seconds":       det_rep.get("runtime_seconds", float("nan")),
                        "artifact_score":        float(artifact["suspicious_autocorrelation"])
                                                  + float(artifact["suspicious_periodicity"]),
                        "median_gap_edge_jump":  artifact["median_gap_edge_jump"],
                        "median_quarter_boundary_jump": artifact["median_quarter_boundary_jump"],
                        "preservation_ratio":   pres_ratio,
                        "local_snr_after":      snr_after,
                        "shallow_preservation_ratio": shallow_ok,
                        "has_truth":            has_truth,
                        "bin":                  truth_row.get("bin", "none") if has_truth else "none",
                    })

                    if method_key not in method_dets:
                        method_dets[method_key] = d_flux

                except Exception as e:
                    comparison_rows.append({
                        "star_id": star_id, "method": method, "window_days": win,
                        "method_key": f"{method}_{win:.1f}d", "error": str(e),
                    })

        # Method comparison plot for this star
        if save_plots and method_dets:
            try:
                plot_method_comparison(
                    star_id=star_id,
                    time=t,
                    quarter=q,
                    norm_flux=norm,
                    method_results=method_dets,
                    truth_row=truth_row if has_truth else None,
                    save_path=PLOT_DIR / "method_comparison" / f"{star_id}.png",
                )
            except Exception as pe:
                print(f"    Warning: method comparison plot failed for {star_id}: {pe}")

        print(f"    {star_id} done", flush=True)

    comp_df = pd.DataFrame(comparison_rows)

    if include_dev and not dev_results:
        raise RuntimeError("DEV data not available for independent method comparison")
    if include_dev:
        dev_comp_df = run_method_comparison(
            dev_results, dev_truth_df, {}, dev_truth_df,
            sample_size=sample_size, save_plots=False, include_dev=False,
        )
        dev_comp_df["split"] = "DEV"
        comp_df = pd.concat([comp_df, dev_comp_df], ignore_index=True)

    # Save comparison CSV
    comp_path = DIAG_DIR / "detrending_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Comparison saved → {comp_path}")

    return comp_df


# ===========================================================================
# Step 4: Select the best method
# ===========================================================================

def select_best_method(comp_df: pd.DataFrame) -> tuple:
    """
    Score each (method, window) combination and select the best.

    Returns (best_method, best_window, score_table_df)
    """
    print("\n" + "="*60)
    print("  Scoring Methods")
    print("="*60)

    if comp_df.empty or "method_key" not in comp_df.columns:
        print("  No comparison data — using defaults.")
        return DETREND_METHOD_SELECTED, DETREND_WINDOW_SELECTED_DAYS, pd.DataFrame()

    score_rows = []

    for method_key in comp_df["method_key"].dropna().unique():
        sub = comp_df[comp_df["method_key"] == method_key]

        train_recs = sub[sub.get("split", "TRAIN") == "TRAIN"].to_dict("records")
        dev_recs   = sub[sub.get("split", "DEV") == "DEV"].to_dict("records")
        scores = score_detrending_method(train_recs, dev_recs, method_key)
        score_rows.append(scores)

    score_df = pd.DataFrame(score_rows).sort_values("composite_score", ascending=False)
    score_df.to_csv(DIAG_DIR / "method_scores.csv", index=False)

    # Print table
    print("\n  Method Comparison Table:")
    print(f"  {'Method':<40} {'TRAIN Pres':>10} {'DEV Pres':>9} "
          f"{'TRAIN MAD':>10} {'Shallow':>8} {'Score':>7}  SELECTED")
    print("  " + "-"*95)
    for _, row in score_df.iterrows():
        marker = "  ← SELECTED" if row.name == score_df.index[0] else ""
        print(
            f"  {row['method']:<40} "
            f"{row['train_preservation_mean']:>10.3f} "
            f"{row['dev_preservation_mean']:>9.3f} "
            f"{row['train_noise_mad_ppm']:>10.1f} "
            f"{row['shallow_preservation']:>8.3f} "
            f"{row['composite_score']:>7.4f}{marker}"
        )

    best_row    = score_df.iloc[0]
    best_key    = best_row["method"]
    # Parse method and window from key like "robust_rolling_median_1.5d"
    parts = best_key.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("d"):
        best_method = parts[0]
        try:
            best_window = float(parts[1][:-1])
        except ValueError:
            best_method = best_key
            best_window = DETREND_WINDOW_SELECTED_DAYS
    else:
        best_method = best_key
        best_window = DETREND_WINDOW_SELECTED_DAYS

    print(f"\n  → Selected: {best_method}  window={best_window:.1f}d  "
          f"score={best_row['composite_score']:.4f}")

    return best_method, best_window, score_df


# ===========================================================================
# Step 5: Injection-recovery tests
# ===========================================================================

def run_injection_tests(
    train_results: dict,
    train_truth_df: pd.DataFrame,
    method: str,
    window_days: float,
    n_stars: int = N_INJECTION_STARS,
) -> pd.DataFrame:
    """
    Inject synthetic transits into TRAIN stars and measure recovery.
    """
    print("\n" + "="*60)
    print(f"  Injection-Recovery Tests  (n={n_stars} stars)")
    print("="*60)

    # Pick quiet (no-transit) TRAIN stars for injection
    transit_kepids = set(
        f"KIC_{k}" for k in
        train_truth_df[train_truth_df.get("injected", 0) == 1]["kepid"].astype(str)
    )
    quiet_stars = [k for k in train_results if k not in transit_kepids
                   and train_results[k].get("ok", False)][:n_stars]

    all_rows = []

    for star_id in quiet_stars:
        res = train_results[star_id]
        t    = res["time"]
        norm = res["norm_flux"]
        nerr = res["norm_err"]
        q    = res["quarter"]

        # Detrend function for this star
        def _detrend_fn(df: pd.DataFrame) -> pd.DataFrame:
            det_df, _ = detrend_star(
                df, star_id=star_id, method=method, window_days=window_days
            )
            return det_df

        rows = injection_recovery_test(
            time=t, flux=norm, flux_err=nerr, quarter=q,
            detrend_fn=_detrend_fn,
            star_id=star_id,
        )
        all_rows.append(rows)
        print(f"    {star_id}  {len(rows)} injections done", flush=True)

    if all_rows:
        inj_df = pd.concat(all_rows, ignore_index=True)
    else:
        inj_df = pd.DataFrame()

    inj_path = DIAG_DIR / "injection_recovery.csv"
    inj_df.to_csv(inj_path, index=False)
    print(f"\n  Injection-recovery saved → {inj_path}")

    if len(inj_df) > 0:
        print("\n  Recovery statistics by depth:")
        for depth in sorted(inj_df["depth_ppm"].unique()):
            sub = inj_df[inj_df["depth_ppm"] == depth]
            med_pres = sub["preservation_ratio"].median()
            print(f"    {depth:4.0f} ppm:  median preservation = {med_pres:.3f}  "
                  f"(n={len(sub)})")

    return inj_df


# ===========================================================================
# Step 6: Save summary diagnostics and plots
# ===========================================================================

def save_diagnostics(
    train_qc:      list,
    dev_qc:        list,
    train_metrics: list,
    dev_metrics:   list,
    save_plots:    bool = True,
) -> tuple:
    """Save QC CSV, metrics CSV, and summary plots."""

    qc_train_df  = pd.DataFrame(train_qc)
    qc_dev_df    = pd.DataFrame(dev_qc)
    met_train_df = pd.DataFrame(train_metrics)
    met_dev_df   = pd.DataFrame(dev_metrics)

    qc_all = pd.concat([qc_train_df, qc_dev_df], ignore_index=True)

    qc_train_df.to_csv(DIAG_DIR / "qc_report_train.csv", index=False)
    qc_dev_df.to_csv(DIAG_DIR / "qc_report_dev.csv", index=False)
    met_train_df.to_csv(DIAG_DIR / "metrics_train.csv", index=False)
    met_dev_df.to_csv(DIAG_DIR / "metrics_dev.csv", index=False)
    qc_all.to_csv(DIAG_DIR / "qc_report_all.csv", index=False)

    print(f"\n  QC reports saved → {DIAG_DIR}")

    if save_plots:
        # QC summary plot (TRAIN)
        try:
            plot_qc_summary(
                qc_train_df,
                save_path=PLOT_DIR / "summary" / "qc_summary_train.png",
                title="QC Summary — TRAIN (269 stars)",
            )
        except Exception as e:
            print(f"  Warning: QC summary plot failed: {e}")

        # Transit preservation plot (TRAIN)
        if len(met_train_df) > 0 and "has_truth" in met_train_df.columns:
            try:
                plot_transit_preservation(
                    met_train_df,
                    save_path=PLOT_DIR / "summary" / "transit_preservation_train.png",
                    title="Transit Preservation — TRAIN",
                )
            except Exception as e:
                print(f"  Warning: transit preservation plot failed: {e}")

    return qc_train_df, qc_dev_df, met_train_df, met_dev_df


# ===========================================================================
# Step 7: Final report
# ===========================================================================

def print_final_report(
    train_qc_df:  pd.DataFrame,
    dev_qc_df:    pd.DataFrame,
    train_met_df: pd.DataFrame,
    dev_met_df:   pd.DataFrame,
    inj_df:       pd.DataFrame,
    best_method:  str,
    best_window:  float,
    total_time:   float,
) -> None:
    """Print a concise final report to stdout."""
    sep = "="*65

    print(f"\n{sep}")
    print("  PHASE 1 + 2 PREPROCESSING — FINAL REPORT")
    print(sep)

    # Data quality
    print("\n  DATA QUALITY")
    for name, df in [("TRAIN", train_qc_df), ("DEV", dev_qc_df)]:
        if len(df) == 0:
            continue
        n       = len(df)
        n_corr  = df["severely_corrupted"].sum() if "severely_corrupted" in df.columns else 0
        ret     = df["percentage_retained"].median()
        q_rem   = (df["quality_removed"] / df["original_points"] * 100).median()
        out_rem = (df["outlier_flagged"] / df["original_points"] * 100).median()
        print(f"  {name} ({n} stars):")
        print(f"    Severely corrupted: {n_corr}")
        print(f"    Median retention: {ret:.1f}%")
        print(f"    Median quality-removed: {q_rem:.1f}%")
        print(f"    Median outlier-flagged: {out_rem:.2f}%")

    # Noise reduction
    print("\n  NOISE REDUCTION")
    for name, df in [("TRAIN", train_met_df), ("DEV", dev_met_df)]:
        if len(df) == 0:
            continue
        raw_mad = df["raw_mad_ppm"].median()
        det_mad = df["det_mad_ppm"].median()
        red     = df["noise_reduction_raw_to_det"].median()
        print(f"  {name}: raw={raw_mad:.0f}ppm → detrended={det_mad:.0f}ppm  "
              f"(×{red:.1f} reduction)")

    # Transit preservation
    print("\n  TRANSIT PRESERVATION")
    for name, df in [("TRAIN", train_met_df), ("DEV", dev_met_df)]:
        if len(df) == 0 or "has_truth" not in df.columns:
            continue
        sub = df[df["has_truth"]]
        if len(sub) == 0:
            continue
        pres = sub["preservation_ratio"].median()
        snr  = sub["local_snr_after"].median()
        print(f"  {name} ({len(sub)} transit stars):")
        print(f"    Median depth preservation ratio: {pres:.3f}")
        print(f"    Median local transit SNR: {snr:.2f}")

        # By bin
        if "bin" in sub.columns:
            for b in ["earth_analog", "shallow", "mid", "deep"]:
                bsub = sub[sub.get("bin", "") == b]
                if len(bsub) > 0:
                    bp = bsub["preservation_ratio"].median()
                    print(f"      {b:<14}: {bp:.3f}  (n={len(bsub)})")

    # Injection-recovery
    if len(inj_df) > 0:
        print("\n  INJECTION-RECOVERY")
        for depth in sorted(inj_df["depth_ppm"].unique()):
            sub = inj_df[inj_df["depth_ppm"] == depth]
            med = sub["preservation_ratio"].median()
            print(f"    {depth:4.0f} ppm: median preservation = {med:.3f}")

    # Method selection
    print("\n  SELECTED PREPROCESSING STRATEGY")
    print(f"    Method     : {best_method}")
    print(f"    Window     : {best_window:.1f} days")

    # Runtime
    n_total = len(train_qc_df) + len(dev_qc_df)
    t_per   = total_time / n_total if n_total > 0 else float("nan")
    print(f"\n  RUNTIME")
    print(f"    Total: {total_time:.0f}s  |  Per star: {t_per:.2f}s")

    print(f"\n  OUTPUT FILES")
    print(f"    QC data       : {QC_DIR}")
    print(f"    Normalized    : {NORM_DIR}")
    print(f"    Detrended     : {DETREND_DIR}")
    print(f"    Diagnostics   : {DIAG_DIR}")
    print(f"    Plots         : {PLOT_DIR}")

    print(f"\n  REPRODUCTION COMMAND")
    print(f"    py run_preprocessing.py")

    print(f"\n{sep}\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Kepler Phase 1+2 Preprocessing")
    parser.add_argument("--sample",    type=int, default=0,
                        help="Process only first N stars per split (0=all)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip the multi-method comparison run")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip diagnostic plot generation")
    parser.add_argument("--no-inject",  action="store_true",
                        help="Skip injection-recovery tests")
    parser.add_argument("--method",    default=DETREND_METHOD_SELECTED,
                        help="Detrending method for main run")
    parser.add_argument("--window",    type=float, default=DETREND_WINDOW_SELECTED_DAYS,
                        help="Detrending window (days) for main run")
    args = parser.parse_args()

    t_global_start = time_module.perf_counter()

    setup_directories()
    train_labels, train_truth, dev_labels, dev_truth = load_truth_tables()

    method       = args.method
    window_days  = args.window
    save_plots   = not args.no_plots
    do_compare   = not args.no_compare
    do_inject    = not args.no_inject

    # Identify representative stars for plots
    # (at most 15 stars across categories)
    transit_kepids = set(
        f"KIC_{k}" for k in train_truth[train_truth.get("injected", 0) == 1]["kepid"].astype(str)
    )
    plot_stars = list(transit_kepids)[:10]

    # ---- TRAIN ---------------------------------------------------------------
    train_results, train_qc, train_metrics = run_split(
        split_name="train",
        data_dir=TRAIN_DIR,
        truth_df=train_truth,
        method=method,
        window_days=window_days,
        sample=args.sample,
        save_parquet=SAVE_PARQUET,
        save_plots=save_plots,
        plot_stars=plot_stars if save_plots else [],
    )

    # ---- DEV -----------------------------------------------------------------
    dev_results, dev_qc, dev_metrics = run_split(
        split_name="dev",
        data_dir=DEV_DIR,
        truth_df=dev_truth,
        method=method,
        window_days=window_days,
        sample=args.sample,
        save_parquet=SAVE_PARQUET,
        save_plots=save_plots,
        plot_stars=[],  # fewer plots for dev
    )

    # ---- Method comparison ---------------------------------------------------
    best_method  = method
    best_window  = window_days
    score_df     = pd.DataFrame()
    comp_df      = pd.DataFrame()

    if do_compare:
        comp_df = run_method_comparison(
            train_results, train_truth,
            dev_results, dev_truth,
            sample_size=min(30, max(5, args.sample or 30)),
            save_plots=save_plots,
        )
        best_method, best_window, score_df = select_best_method(comp_df)

        # Re-run with best method if it differs from initial
        if best_method != method or abs(best_window - window_days) > 0.01:
            print(f"\n  Re-running with selected method: {best_method} {best_window:.1f}d")
            train_results, train_qc, train_metrics = run_split(
                "train", TRAIN_DIR, train_truth,
                best_method, best_window,
                sample=args.sample,
                save_parquet=SAVE_PARQUET,
                save_plots=False,  # already done
                plot_stars=[],
            )
            dev_results, dev_qc, dev_metrics = run_split(
                "dev", DEV_DIR, dev_truth,
                best_method, best_window,
                sample=args.sample,
                save_parquet=SAVE_PARQUET,
                save_plots=False,
                plot_stars=[],
            )

    # ---- Injection tests -----------------------------------------------------
    inj_df = pd.DataFrame()
    if do_inject:
        inj_df = run_injection_tests(
            train_results, train_truth,
            best_method, best_window,
            n_stars=min(N_INJECTION_STARS, max(3, args.sample or N_INJECTION_STARS)),
        )

    # ---- Save diagnostics and plots -----------------------------------------
    qc_tr, qc_dev, met_tr, met_dev = save_diagnostics(
        train_qc, dev_qc, train_metrics, dev_metrics, save_plots
    )

    # ---- Save config ---------------------------------------------------------
    config_path = Path("data") / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_out = {
        "method":       best_method,
        "window_days":  best_window,
        "selection_score": float(score_df["composite_score"].iloc[0]) if len(score_df) > 0 else None,
        "dataset_separation": {
            "TRAIN": "development / in-sample",
            "DEV": "independent validation",
            "PRIVATE": "untouched",
        },
        "configuration_status": "development-selected; DEV validation follows",
        "qc_policy": "quality == 0; finite flux/time; deterministic duplicate resolution",
        "gap_factor": 3.0,
        "duplicate_policy": "keep lowest valid flux_err; stable first on ties",
        "outlier_policy": "flag; retain negative transit-like events; remove only explicit positive spikes",
        "small_quarter_policy": "retain and flag",
        "edge_policy": "flag edge-affected cadences; do not delete",
    }
    config_path.write_text(json.dumps(config_out, indent=2))
    print(f"\n  Config saved → {config_path}")

    # ---- Final report --------------------------------------------------------
    total_time = time_module.perf_counter() - t_global_start
    print_final_report(
        qc_tr, qc_dev, met_tr, met_dev,
        inj_df, best_method, best_window, total_time,
    )


if __name__ == "__main__":
    main()
