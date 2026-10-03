from __future__ import annotations

import numpy as np


def fit_oof_calibrator(scores, labels):
    """Fit a leakage-safe monotonic calibration using OOF scores when supplied."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) < 4 or len(np.unique(labels)) < 2:
        return {"method": "identity"}
    order = np.argsort(scores)
    x, y = scores[order], labels[order]
    # Isotonic pool-adjacent-violators, implemented without sklearn.
    blocks = [[float(xi), float(xi), float(yi), 1.0] for xi, yi in zip(x, y)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            left, right = blocks[i], blocks[i + 1]
            total = left[3] + right[3]
            blocks[i:i + 2] = [[left[0], right[1],
                                (left[2] * left[3] + right[2] * right[3]) / total, total]]
            i = max(0, i - 1)
        else:
            i += 1
    return {"method": "isotonic", "x": [b[0] for b in blocks],
            "y": [b[2] for b in blocks]}


def calibrate(calibrator, scores):
    scores = np.asarray(scores, dtype=float)
    if calibrator.get("method") != "isotonic":
        return np.clip(scores, 0, 1)
    return np.interp(scores, calibrator["x"], calibrator["y"],
                     left=calibrator["y"][0], right=calibrator["y"][-1])
