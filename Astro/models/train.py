from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features.transit_features import (
    FEATURE_NAMES,
    VETTING_FEATURE_NAMES,
    candidate_features,
    candidate_vetting_features,
)


def _standardize(matrix):
    mean = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    return np.nan_to_num((matrix - mean) / np.where(scale > 0, scale, 1.0)), mean, scale


def fit_model(candidates, labels, seed=20260914):
    """Fit a small dependency-free logistic model; returns None for one-class data."""
    x = np.vstack([candidate_features(c) for c in candidates])
    y = np.asarray(labels, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    x, mean, scale = _standardize(x)
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(800):
        logits = np.clip(x @ weights + bias, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        gradient = probabilities - y
        weights -= 0.05 * (x.T @ gradient / len(y) + 1e-3 * weights)
        bias -= 0.05 * float(np.mean(gradient))
    return {"weights": weights.tolist(), "bias": bias, "mean": mean.tolist(),
            "scale": scale.tolist(), "features": list(FEATURE_NAMES)}


def fit_vetting_model(candidates, labels, seed=20260914):
    """Fit a stricter second-stage vetting model using candidate-level and shape evidence."""
    x = np.vstack([candidate_vetting_features(c) for c in candidates])
    y = np.asarray(labels, dtype=float)
    if len(np.unique(y)) < 2:
        return None
    x, mean, scale = _standardize(x)
    rng = np.random.default_rng(seed)
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(1000):
        logits = np.clip(x @ weights + bias, -30, 30)
        probabilities = 1 / (1 + np.exp(-logits))
        gradient = probabilities - y
        penalty = 1e-3 * weights
        weights -= 0.04 * (x.T @ gradient / len(y) + penalty)
        bias -= 0.04 * float(np.mean(gradient))
    return {"weights": weights.tolist(), "bias": bias, "mean": mean.tolist(),
            "scale": scale.tolist(), "features": list(VETTING_FEATURE_NAMES)}


def predict_probability(model, candidates):
    if model is None or not candidates:
        return np.full(len(candidates), np.nan)
    x = np.vstack([candidate_features(c) for c in candidates])
    x = np.nan_to_num((x - np.asarray(model["mean"])) /
                      np.where(np.asarray(model["scale"]) > 0, model["scale"], 1.0))
    logits = np.clip(x @ np.asarray(model["weights"]) + model["bias"], -30, 30)
    return 1 / (1 + np.exp(-logits))


def predict_vetting_probability(model, candidates):
    if model is None or not candidates:
        return np.full(len(candidates), np.nan)
    x = np.vstack([candidate_vetting_features(c) for c in candidates])
    x = np.nan_to_num((x - np.asarray(model["mean"])) /
                      np.where(np.asarray(model["scale"]) > 0, model["scale"], 1.0))
    logits = np.clip(x @ np.asarray(model["weights"]) + model["bias"], -30, 30)
    return 1 / (1 + np.exp(-logits))


def save_model(model, path):
    if model is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(model, indent=2))
