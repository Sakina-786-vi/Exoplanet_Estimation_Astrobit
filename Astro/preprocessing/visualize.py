"""
preprocessing/visualize.py
==========================
Diagnostic visualization (Task 22)

Produces publication-quality diagnostic plots for preprocessing validation.

WHAT IS PLOTTED (per star, 5 panels):
  1. Raw SAP flux (all data)
  2. Quality-filtered flux (quality == 0 only)
  3. Quarter-normalized flux (inter-quarter jumps removed)
  4. Stellar/instrumental baseline estimate
  5. Detrended flux (final analysis product)
  + Optional: phase-folded transit (only when truth is provided for validation)

LEAKAGE RULE
------------
Ground truth is ONLY used in validation plots.
Phase-folded plots are labelled clearly as "validation only".
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for batch processing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from .config import PLOT_DPI, PLOT_FORMAT, PLOT_DIR

# ---------------------------------------------------------------------------
# Colour palette (quarter-distinguishable)
# ---------------------------------------------------------------------------
QUARTER_CMAP = plt.cm.tab20
_ALPHA_POINTS = 0.35
_MS = 0.4


def _quarter_colors(quarters: np.ndarray) -> np.ndarray:
    """Map quarter numbers to colours from the tab20 colormap."""
    unique_q = np.unique(quarters)
    color_map = {q: QUARTER_CMAP(i % 20 / 20) for i, q in enumerate(unique_q)}
    return np.array([color_map[q] for q in quarters])


# ===========================================================================
# Core diagnostic plot (5 panels)
# ===========================================================================

def plot_star_preprocessing(
    star_id:      str,
    raw_df:       pd.DataFrame,
    result:       Dict,
    truth_row:    Optional[pd.Series] = None,
    save_path:    Optional[Path] = None,
    show:         bool = False,
) -> Optional[plt.Figure]:
    """
    Full 5-panel preprocessing diagnostic plot for one star.

    Parameters
    ----------
    star_id   : e.g. "KIC_10002867"
    raw_df    : original Parquet DataFrame
    result    : output of preprocess_star()
    truth_row : optional row from train_truth.csv (validation only)
    save_path : if provided, save figure to this path
    show      : if True, call plt.show()

    Returns
    -------
    fig : matplotlib Figure
    """
    if not result.get("ok", False) or result["n_points"] == 0:
        return None

    t_clean  = result["time"]
    q_clean  = result["quarter"]
    f_raw_q  = result["flux"]          # quality-filtered flux (SAP, not normalized)
    f_norm   = result["norm_flux"]
    f_det    = result["detrended_flux"]
    baseline = result["baseline"]

    # Raw flux (all cadences, quality 0 only shown for panel 2)
    t_raw  = raw_df["time"].values.astype(np.float64)
    f_raw  = raw_df["flux"].values.astype(np.float64)
    q_raw  = raw_df["quarter"].values

    has_transit = (truth_row is not None and not truth_row.empty
                   and "period_days" in truth_row.index)
    n_panels = 6 if has_transit else 5

    fig = plt.figure(figsize=(17, 2.8 * n_panels), constrained_layout=True)
    fig.suptitle(
        f"{star_id}  —  Preprocessing Diagnostic",
        fontsize=13, fontweight="bold",
    )
    axes = fig.subplots(n_panels, 1)

    # ---- Panel 1: Raw SAP flux -----------------------------------------------
    ax = axes[0]
    colors = _quarter_colors(q_raw)
    ax.scatter(t_raw, f_raw, c=colors, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
    ax.set_ylabel("Raw SAP flux\n(e⁻/s)", fontsize=8)
    _format_lc_ax(ax, t_raw, f_raw, title="Raw SAP flux (all quality flags)")
    _mark_quarter_boundaries(ax, t_raw, q_raw)

    # ---- Panel 2: Quality-filtered flux --------------------------------------
    ax = axes[1]
    colors_c = _quarter_colors(q_clean)
    ax.scatter(t_clean, f_raw_q, c=colors_c, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
    ax.set_ylabel("Filtered flux\n(e⁻/s)", fontsize=8)
    _format_lc_ax(ax, t_clean, f_raw_q, title="Quality-filtered (quality == 0 only)")
    _mark_quarter_boundaries(ax, t_clean, q_clean)

    # ---- Panel 3: Quarter-normalized flux ------------------------------------
    ax = axes[2]
    ax.scatter(t_clean, f_norm, c=colors_c, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Norm. flux", fontsize=8)
    _format_lc_ax(ax, t_clean, f_norm, title="Quarter-normalized flux")
    _mark_quarter_boundaries(ax, t_clean, q_clean)

    # ---- Panel 4: Baseline estimate ------------------------------------------
    ax = axes[3]
    ax.scatter(t_clean, f_norm, c=colors_c, s=_MS**2, alpha=0.15, rasterized=True)
    ax.plot(t_clean, baseline, "r-", lw=0.8, alpha=0.85, label="Baseline")
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Baseline", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")
    _format_lc_ax(ax, t_clean, baseline, title=f"Baseline estimate ({result.get('method','?')})")
    _mark_quarter_boundaries(ax, t_clean, q_clean)

    # ---- Panel 5: Detrended flux ---------------------------------------------
    ax = axes[4]
    ax.scatter(t_clean, f_det, c=colors_c, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Detrended flux", fontsize=8)
    ax.set_xlabel("BKJD (days)", fontsize=8)
    det_mad_ppm = (
        float(np.median(np.abs(f_det - 1.0)) * 1e6)
        if len(f_det) > 0 else float("nan")
    )
    _format_lc_ax(
        ax, t_clean, f_det,
        title=f"Detrended flux  (scatter = {det_mad_ppm:.0f} ppm robust MAD)"
    )
    _mark_quarter_boundaries(ax, t_clean, q_clean)

    if has_transit:
        # ---- Panel 6: Phase-folded transit (VALIDATION ONLY) -----------------
        try:
            period    = float(truth_row["period_days"])
            t0        = float(truth_row["epoch_t0"])
            depth_ppm = float(truth_row["depth_ppm"])
            dur_hrs   = float(truth_row["duration_hours"])

            ax = axes[5]
            _plot_folded_transit(ax, t_clean, f_det, period, t0, depth_ppm, dur_hrs)
            ax.set_title(
                f"Phase-folded transit  [VALIDATION ONLY]  "
                f"P={period:.3f} d  depth={depth_ppm:.0f} ppm  "
                f"dur={dur_hrs:.1f} h",
                fontsize=8
            )
            ax.set_xlabel("Phase (days from transit centre)", fontsize=8)
            ax.set_ylabel("Detrended flux", fontsize=8)
        except Exception as e:
            axes[5].text(0.5, 0.5, f"Phase-fold failed: {e}",
                         ha="center", va="center", transform=axes[5].transAxes)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ===========================================================================
# Phase-folded transit plot helper
# ===========================================================================

def _plot_folded_transit(
    ax:           plt.Axes,
    time:         np.ndarray,
    flux:         np.ndarray,
    period:       float,
    t0:           float,
    depth_ppm:    float,
    duration_hrs: float,
    bins:         int = 100,
) -> None:
    """Plot a phase-folded light curve with binned median and true depth marker."""
    dur_days = duration_hrs / 24.0
    phase    = ((time - t0) % period)
    phase    = np.where(phase > period / 2, phase - period, phase)

    # Only show ±4× transit duration for clarity
    show_win  = 4.0 * dur_days
    in_window = np.abs(phase) < show_win
    ph = phase[in_window]
    ff = flux[in_window]

    # Scatter plot
    ax.scatter(ph, ff, s=0.5, c="steelblue", alpha=0.3, rasterized=True)

    # Binned median
    edges  = np.linspace(-show_win, show_win, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    binned = np.array([
        np.median(ff[(ph >= edges[i]) & (ph < edges[i + 1])])
        if ((ph >= edges[i]) & (ph < edges[i + 1])).sum() > 0
        else np.nan
        for i in range(bins)
    ])
    ax.plot(centres, binned, "o-", color="crimson", ms=2, lw=1.2, label="Binned median")

    # True depth line
    ax.axhline(1.0 - depth_ppm / 1e6, ls="--", lw=1.0, color="darkorange",
               label=f"True depth {depth_ppm:.0f} ppm")
    ax.axhline(1.0, ls=":", lw=0.5, color="k")

    # Transit duration markers
    ax.axvline(-dur_days / 2, ls="--", lw=0.6, color="gray", alpha=0.6)
    ax.axvline( dur_days / 2, ls="--", lw=0.6, color="gray", alpha=0.6)

    ax.legend(fontsize=7, loc="lower right")

    # y-limits: zoom to 3× the true depth below 1.0
    y_lo = 1.0 - 3.0 * depth_ppm / 1e6
    y_hi = 1.0 + 2.0 * depth_ppm / 1e6
    ax.set_ylim(
        min(y_lo, np.nanpercentile(binned, 0.5) - 1e-4),
        max(y_hi, np.nanpercentile(binned, 99.5) + 1e-4),
    )


# ===========================================================================
# Format helpers
# ===========================================================================

def _format_lc_ax(
    ax:    plt.Axes,
    t:     np.ndarray,
    f:     np.ndarray,
    title: str = "",
) -> None:
    ax.set_title(title, fontsize=8, pad=2)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(t.min() - 5, t.max() + 5)

    fin = f[np.isfinite(f)]
    if len(fin) > 0:
        p01 = np.percentile(fin, 0.5)
        p99 = np.percentile(fin, 99.5)
        margin = 0.1 * (p99 - p01)
        ax.set_ylim(p01 - margin, p99 + margin)


def _mark_quarter_boundaries(
    ax:       plt.Axes,
    time:     np.ndarray,
    quarter:  np.ndarray,
) -> None:
    """Draw vertical lines at quarter boundaries."""
    for qq in np.unique(quarter):
        mask = quarter == qq
        t_start = time[mask].min()
        ax.axvline(t_start, color="gray", lw=0.4, ls=":", alpha=0.6, zorder=0)


# ===========================================================================
# Method comparison plot (Task 28)
# ===========================================================================

def plot_method_comparison(
    star_id:      str,
    time:         np.ndarray,
    quarter:      np.ndarray,
    norm_flux:    np.ndarray,
    method_results: Dict[str, np.ndarray],
    truth_row:    Optional[pd.Series] = None,
    save_path:    Optional[Path] = None,
    show:         bool = False,
) -> Optional[plt.Figure]:
    """
    Side-by-side comparison of detrended flux from all tested methods
    for one star, zoomed into a representative transit window if truth available.

    Parameters
    ----------
    method_results : {method_key: detrended_flux_array}
    """
    n_methods = len(method_results)
    if n_methods == 0:
        return None

    fig, axes = plt.subplots(
        n_methods + 1, 1,
        figsize=(16, 2.5 * (n_methods + 1)),
        constrained_layout=True,
        sharex=True,
    )
    fig.suptitle(f"{star_id}  —  Detrending Method Comparison", fontsize=12, fontweight="bold")

    colors = _quarter_colors(quarter)

    # Top panel: normalized input
    ax = axes[0]
    ax.scatter(time, norm_flux, c=colors, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
    ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_title("Quarter-normalized (input to all methods)", fontsize=8)
    _mark_quarter_boundaries(ax, time, quarter)
    _format_lc_ax(ax, time, norm_flux)

    # One panel per method
    for i, (method_key, det_flux) in enumerate(method_results.items()):
        ax = axes[i + 1]
        ax.scatter(time, det_flux, c=colors, s=_MS**2, alpha=_ALPHA_POINTS, rasterized=True)
        ax.axhline(1.0, color="k", lw=0.5, ls="--", alpha=0.5)

        # Mark true transit depth if available
        if truth_row is not None and not truth_row.empty:
            try:
                depth_ppm = float(truth_row["depth_ppm"])
                ax.axhline(1.0 - depth_ppm / 1e6, color="darkorange",
                            lw=0.8, ls="--", alpha=0.7, label=f"True depth {depth_ppm:.0f} ppm")
                ax.legend(fontsize=6, loc="upper right")
            except Exception:
                pass

        mad_ppm = float(np.median(np.abs(det_flux - 1.0)) * 1e6) if len(det_flux) > 0 else float("nan")
        ax.set_title(f"{method_key}  (scatter = {mad_ppm:.0f} ppm)", fontsize=8)
        _mark_quarter_boundaries(ax, time, quarter)
        _format_lc_ax(ax, time, det_flux)

    axes[-1].set_xlabel("BKJD (days)", fontsize=8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# ===========================================================================
# Summary plots across all stars
# ===========================================================================

def plot_qc_summary(
    qc_df:     pd.DataFrame,
    save_path: Optional[Path] = None,
    title:     str = "QC Summary",
) -> plt.Figure:
    """
    Produce a 2×3 summary figure of QC statistics across all stars.

    Parameters
    ----------
    qc_df : DataFrame with one row per star, columns from generate_qc_report()
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. Retention rate distribution
    ax = axes[0, 0]
    ax.hist(qc_df["percentage_retained"].dropna(), bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("% points retained after QC")
    ax.set_ylabel("Number of stars")
    ax.set_title("Data Retention Rate")

    # 2. Quality-removed fraction
    ax = axes[0, 1]
    pct = qc_df["quality_removed"] / qc_df["original_points"] * 100
    ax.hist(pct.dropna(), bins=30, color="tomato", edgecolor="white")
    ax.set_xlabel("% removed by quality flag")
    ax.set_title("Quality Flag Removal")

    # 3. Robust scatter before detrending
    ax = axes[0, 2]
    ax.hist(qc_df["robust_scatter_ppm"].dropna(), bins=40, color="mediumseagreen", edgecolor="white")
    ax.set_xlabel("Robust MAD scatter (ppm)")
    ax.set_title("Pre-detrending scatter (QC flux)")

    # 4. Large gap count
    ax = axes[1, 0]
    ax.hist(qc_df["large_gap_count"].dropna(), bins=20, color="orchid", edgecolor="white")
    ax.set_xlabel("Number of large gaps")
    ax.set_title("Large Mission Gaps per Star")

    # 5. Quarter count
    ax = axes[1, 1]
    ax.hist(qc_df["quarter_count"].dropna(), bins=range(0, 20), color="gold", edgecolor="white")
    ax.set_xlabel("Number of quarters")
    ax.set_title("Quarters per Star")

    # 6. Outlier flagged fraction
    ax = axes[1, 2]
    pct_out = qc_df["outlier_flagged"] / qc_df["original_points"] * 100
    ax.hist(pct_out.dropna(), bins=30, color="cornflowerblue", edgecolor="white")
    ax.set_xlabel("% points flagged as outliers")
    ax.set_title("Outlier Flag Rate")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_transit_preservation(
    metrics_df: pd.DataFrame,
    save_path:  Optional[Path] = None,
    title:      str = "Transit Preservation",
) -> plt.Figure:
    """
    Plot transit preservation ratio vs. depth, period, and method.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    df = metrics_df[metrics_df["has_truth"]].copy()

    if len(df) == 0:
        return fig

    # 1. Preservation ratio vs. true depth
    ax = axes[0]
    ax.scatter(df["true_depth_ppm"], df["preservation_ratio"],
               c="steelblue", s=25, alpha=0.7)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.axhline(0.8, color="tomato", lw=0.8, ls=":", label="80% threshold")
    ax.set_xlabel("True transit depth (ppm)")
    ax.set_ylabel("Preservation ratio")
    ax.set_title("Depth Preservation vs. Transit Depth")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.5)

    # 2. SNR improvement
    ax = axes[1]
    if "local_snr_before" in df.columns and "local_snr_after" in df.columns:
        has_both = df[["local_snr_before", "local_snr_after"]].notna().all(axis=1)
        ax.scatter(df.loc[has_both, "local_snr_before"],
                   df.loc[has_both, "local_snr_after"],
                   c="mediumseagreen", s=25, alpha=0.7)
        lim = max(df["local_snr_before"].max(), df["local_snr_after"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="No improvement")
        ax.set_xlabel("Local SNR (before detrending)")
        ax.set_ylabel("Local SNR (after detrending)")
        ax.set_title("Transit SNR Improvement")
        ax.legend(fontsize=8)

    # 3. Noise reduction histogram
    ax = axes[2]
    if "noise_reduction_raw_to_det" in df.columns:
        ax.hist(df["noise_reduction_raw_to_det"].dropna(), bins=20,
                color="orchid", edgecolor="white")
        ax.axvline(1.0, color="k", lw=0.8, ls="--", label="No reduction")
        ax.set_xlabel("Noise reduction factor (raw/detrended MAD)")
        ax.set_title("Noise Reduction Distribution")
        ax.legend(fontsize=8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig
