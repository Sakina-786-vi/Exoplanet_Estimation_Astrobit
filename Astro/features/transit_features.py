from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "sde", "depth_ppm", "n_transit_points", "transit_count",
    "in_transit_fraction", "local_snr", "odd_even_difference",
    "quarter_recurrence", "edge_fraction", "gap_fraction",
    "depth_scatter", "depth_consistency", "event_snr_scatter",
    "recovered_event_fraction", "single_quarter_fraction", "phase_shape_score",
    "period_boundary_proximity", "duration_grid_boundary",
)

# Secondary/eclipse-shape evidence derived from the same candidate/event metrics.
VETTING_FEATURE_NAMES = FEATURE_NAMES + (
    "secondary_eclipse_ratio",
    "phase_shape_consistency",
    "event_recurrence_strength",
)


def candidate_features(candidate) -> np.ndarray:
    return np.asarray([getattr(candidate, name) for name in FEATURE_NAMES], dtype=float)


def candidate_vetting_features(candidate) -> np.ndarray:
    base = candidate_features(candidate)
    depth = float(getattr(candidate, "depth_ppm", 0.0))
    odd_even = float(getattr(candidate, "odd_even_difference", 0.0))
    recurrence = float(getattr(candidate, "quarter_recurrence", 0.0))
    phase_shape = float(getattr(candidate, "phase_shape_score", 0.0))
    recovered = float(getattr(candidate, "recovered_event_fraction", 0.0))
    single_quarter = float(getattr(candidate, "single_quarter_fraction", 1.0))
    secondary_eclipse_ratio = np.clip(1.0 - odd_even / max(1.0, depth), 0.0, 1.0)
    phase_shape_consistency = np.clip(phase_shape / max(1.0, abs(depth) + 1e-9), 0.0, 1.0)
    event_recurrence_strength = np.clip(recurrence * recovered * (1.0 - single_quarter), 0.0, 1.0)
    return np.concatenate([base, np.asarray([secondary_eclipse_ratio, phase_shape_consistency, event_recurrence_strength], dtype=float)])


def feature_matrix(candidates) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return np.vstack([candidate_features(candidate) for candidate in candidates])


def vetting_feature_matrix(candidates) -> np.ndarray:
    if not candidates:
        return np.empty((0, len(VETTING_FEATURE_NAMES)), dtype=float)
    return np.vstack([candidate_vetting_features(candidate) for candidate in candidates])
