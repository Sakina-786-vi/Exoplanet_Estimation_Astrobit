"""
preprocessing/__init__.py
=========================
Public API for the Kepler preprocessing pipeline.

Usage:
    from preprocessing import preprocess_star, preprocess_batch

    result = preprocess_star(raw_df, star_id="KIC_10002867")
    detrended_flux = result["detrended_flux"]
    qc_report      = result["qc"]
"""

from .pipeline import preprocess_star, preprocess_batch
from .qc       import run_full_qc
from .normalize import normalize_star
from .detrend   import detrend_star, detrend_compare
from .evaluate  import (
    measure_transit_preservation,
    inject_transit,
    injection_recovery_test,
    score_detrending_method,
    compute_preprocessing_metrics,
    artifact_diagnostics,
)

__all__ = [
    "preprocess_star",
    "preprocess_batch",
    "run_full_qc",
    "normalize_star",
    "detrend_star",
    "detrend_compare",
    "measure_transit_preservation",
    "inject_transit",
    "injection_recovery_test",
    "score_detrending_method",
    "compute_preprocessing_metrics",
    "artifact_diagnostics",
]

__version__ = "1.0.0"
