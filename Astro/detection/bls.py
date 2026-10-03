from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class Candidate:
    star_id: str
    period_days: float
    epoch: float
    duration_hours: float
    depth_ppm: float
    power: float
    sde: float
    n_transit_points: int
    transit_count: int
    in_transit_fraction: float
    local_snr: float
    odd_even_difference: float
    quarter_recurrence: float
    edge_fraction: float
    gap_fraction: float
    score: float = 0.0
    confidence: float = 0.0
    depth_scatter: float = 0.0
    depth_consistency: float = 0.0
    event_snr_scatter: float = 0.0
    recovered_event_fraction: float = 0.0
    single_quarter_fraction: float = 0.0
    phase_shape_score: float = 0.0
    period_boundary_proximity: float = 0.0
    duration_grid_boundary: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 1e-9
    mad = 1.4826 * np.median(np.abs(values - np.median(values)))
    return max(float(mad), float(np.std(values)), 1e-9)


def _box_stat(time: np.ndarray, flux: np.ndarray, period: float, duration_days: float):
    # Aggregate into phase bins once per period. Searching contiguous bins
    # with prefix sums avoids constructing thousands of full-length masks.
    n_bins = 1024
    phase = ((time - time[0]) % period) / period
    indices = np.minimum((phase * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(indices, minlength=n_bins).astype(float)
    sums = np.bincount(indices, weights=flux, minlength=n_bins)
    scale = robust_scale(flux)
    median = float(np.median(flux))
    width = max(1, int(round(duration_days / period * n_bins)))
    if width >= n_bins:
        return (-np.inf, 0.0, 0, 0.0)
    doubled_counts = np.concatenate((counts, counts[:width]))
    doubled_sums = np.concatenate((sums, sums[:width]))
    cumulative_counts = np.concatenate(([0.0], np.cumsum(doubled_counts)))
    cumulative_sums = np.concatenate(([0.0], np.cumsum(doubled_sums)))
    window_counts = cumulative_counts[width:width + n_bins] - cumulative_counts[:-width][:n_bins]
    window_sums = cumulative_sums[width:width + n_bins] - cumulative_sums[:-width][:n_bins]
    valid = (window_counts >= 3) & (window_counts <= 0.25 * len(flux))
    if not valid.any():
        return (-np.inf, 0.0, 0, 0.0)
    depths = median - window_sums / np.maximum(window_counts, 1.0)
    depths[~valid] = -np.inf
    start = int(np.argmax(depths))
    power = float(depths[start] * math.sqrt(window_counts[start]) / scale)
    centre = ((start + width / 2.0) % n_bins) / n_bins
    return power, centre, int(window_counts[start]), float(depths[start])


def generate_candidates(star_id: str, time: np.ndarray, flux: np.ndarray,
                        quarter: np.ndarray, gap: np.ndarray, config: dict) -> list[Candidate]:
    # Search on a bounded representative cadence set; retain the full arrays
    # for final candidate measurements. This keeps long Kepler quarters tractable.
    if len(time) > 8000:
        stride = int(np.ceil(len(time) / 8000))
        time = time[::stride]
        flux = flux[::stride]
        quarter = quarter[::stride]
        gap = gap[::stride]
    pmin = float(config["period_min_days"])
    pmax = min(float(config["period_max_days"]), max(pmin + 1e-6, float(np.ptp(time))))
    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax),
                                 max(32, min(int(config["coarse_trials"]), 512))))
    durations_hours = np.asarray(config["durations_hours"], dtype=float)
    durations = durations_hours / 24.0
    rows = []
    powers = []
    for period in periods:
        result = max((_box_stat(time, flux, period, duration) + (duration,)
                      for duration in durations), key=lambda row: row[0])
        powers.append(result[0])
        rows.append((period, result))
    powers = np.asarray(powers)
    scale = robust_scale(powers)
    order = np.argsort(powers)[::-1]
    chosen: list[tuple[float, tuple]] = []
    for index in order:
        period = float(rows[index][0])
        if any(abs(period / previous - 1.0) < 0.015 or
               min(abs(period / previous - harmonic)
                   for harmonic in (0.5, 2.0, 1 / 3, 3.0)) < 0.015
               for previous, _ in chosen):
            continue
        chosen.append((period, rows[index][1]))
        if len(chosen) >= int(config["max_candidates"]):
            break

    output = []
    for period, (power, phase, n_points, depth, duration) in chosen:
        phase_values = ((time - time[0]) % period) / period
        mask = np.abs(((phase_values - phase + 0.5) % 1.0) - 0.5) <= duration / (2 * period)
        transit_numbers = np.floor((time - (time[0] + phase * period)) / period + 0.5).astype(int)
        transit_ids = np.unique(transit_numbers[mask])
        per_depth = []
        per_snr = []
        event_quarters = []
        for transit_id in transit_ids:
            transit_mask = mask & (transit_numbers == transit_id)
            if transit_mask.sum() >= 2:
                event_depth = np.median(flux[~transit_mask]) - np.median(flux[transit_mask])
                per_depth.append(event_depth)
                local_scale = robust_scale(flux[~transit_mask])
                per_snr.append(float(event_depth / max(local_scale, 1e-9)))
                event_quarters.append(quarter[transit_mask][0])
        odd = np.median(per_depth[::2]) if per_depth[::2] else 0.0
        even = np.median(per_depth[1::2]) if per_depth[1::2] else 0.0
        edge = np.mean((time - time.min() < duration) | (time.max() - time < duration))
        gap_fraction = np.mean(gap[mask]) if mask.any() else 0.0
        recurrence = len(set(quarter[mask])) / max(1, len(set(quarter)))
        depth_median = float(np.median(per_depth)) if per_depth else 0.0
        depth_scatter = float(np.std(per_depth)) if len(per_depth) > 1 else 0.0
        depth_consistency = float(
            depth_median / max(depth_median + depth_scatter, 1e-9)
        ) if per_depth else 0.0
        snr_scatter = float(np.std(per_snr)) if len(per_snr) > 1 else 0.0
        expected_events = max(1, int(np.floor(np.ptp(time) / period)) + 1)
        recovered_fraction = min(1.0, len(per_depth) / expected_events)
        if event_quarters:
            counts = np.unique(event_quarters, return_counts=True)[1]
            single_quarter_fraction = float(np.max(counts) / len(event_quarters))
        else:
            single_quarter_fraction = 1.0
        # A box-like event has a larger central deficit than its immediate
        # shoulders; this is a soft morphology feature, not a rejection rule.
        inner = np.abs(((phase_values - phase + 0.5) % 1.0) - 0.5) <= duration / (4 * period)
        shoulder = mask & ~inner
        phase_shape = float(
            np.mean(flux[~inner]) - np.mean(flux[inner])
        ) if inner.any() else 0.0
        candidate = Candidate(
            star_id, period, time[0] + phase * period, duration * 24,
            max(0.0, depth * 1e6), float(power), float((power - np.median(powers)) / scale),
            int(n_points), len(transit_ids), float(mask.mean()), float(power),
            float(abs(odd - even)), float(recurrence), float(edge), float(gap_fraction),
            0.0, 0.0, depth_scatter, depth_consistency, snr_scatter,
            recovered_fraction, single_quarter_fraction, phase_shape,
            max(0.0, (period - 0.9 * pmax) / max(0.1 * pmax, 1e-9)),
            float(duration * 24 == durations_hours.max()),
        )
        candidate.score = candidate.power * (1 - candidate.edge_fraction) * (1 - candidate.gap_fraction)
        output.append(candidate)
    return output


def detect_star(star_id: str, time: np.ndarray, flux: np.ndarray,
                quarter: np.ndarray, gap: np.ndarray, config: dict) -> list[Candidate]:
    valid = np.isfinite(time) & np.isfinite(flux)
    time, flux, quarter, gap = (np.asarray(v)[valid] for v in (time, flux, quarter, gap))
    if len(time) < 20 or np.ptp(time) <= 0:
        return []
    order = np.argsort(time)
    time, flux, quarter, gap = (v[order] for v in (time, flux, quarter, gap))
    candidates = generate_candidates(star_id, time, flux, quarter, gap, config)
    for candidate in candidates:
        candidate.confidence = float(1 / (1 + np.exp(-0.8 * (candidate.sde - 5))))
        candidate.confidence *= (1 - 0.5 * candidate.edge_fraction) * (1 - 0.5 * candidate.gap_fraction)
    return candidates
