import csv
from pathlib import Path

import numpy as np

from detection.bls import detect_star
from submission.validate_submission import validate


def test_period_recovery():
    time = np.arange(0, 120, 0.05)
    flux = np.ones_like(time)
    phase = (time - 2) % 10
    flux[(phase < 0.15) | (phase > 9.85)] -= 400e-6
    candidates = detect_star(
        "S", time, flux, np.zeros_like(time, dtype=int),
        np.zeros_like(time, dtype=bool),
        {"period_min_days": 3, "period_max_days": 30, "coarse_trials": 120,
         "durations_hours": [2, 4, 8, 12], "max_candidates": 5},
    )
    assert candidates
    assert any(abs(candidate.period_days / 10 - 1) < 0.05 for candidate in candidates)


def test_submission_validation(tmp_path):
    path = tmp_path / "submission.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"])
        writer.writerow(["S", "0", "0.2", "", "", ""])
    assert validate(path, ["S"])
