from __future__ import annotations

import numpy as np


def binary_metrics(labels, predictions, scores):
    labels, predictions, scores = map(np.asarray, (labels, predictions, scores))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"n": len(labels), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
            "mean_positive_score": float(np.mean(scores[labels == 1])) if np.any(labels == 1) else 0.0}
