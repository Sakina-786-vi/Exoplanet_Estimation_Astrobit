from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from features.transit_features import FEATURE_NAMES
from models.train import fit_model, predict_probability
from detection.bls import Candidate


ROOT = Path(__file__).parents[1]
FEATURE_PATH = ROOT / "reports" / "feature_diagnostics.csv"
CONFIG_PATH = ROOT / "configs" / "final_config.json"


def candidates(frame):
    output = []
    for row in frame.itertuples():
        values = {
            name: getattr(row, name, 0.0)
            for name in FEATURE_NAMES
            if name not in {"depth_ppm"}
        }
        output.append(Candidate(
            star_id=row.star_id, period_days=row.period_days, epoch=0.0,
            duration_hours=row.duration_hours, depth_ppm=row.depth_ppm,
            power=0.0, **values,
        ))
    return output


def average_precision(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = max(1, int(sorted_labels.sum()))
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / positives)


def metrics(labels, scores, threshold):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores) >= threshold
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "average_precision": average_precision(labels, scores),
        "positive_predictions": int(predictions.sum()), "tp": tp, "fp": fp, "fn": fn,
    }


def fit_oof(frame, folds=5):
    scores = np.full(len(frame), np.nan)
    for fold in range(folds):
        valid = np.arange(len(frame)) % folds == fold
        model = fit_model(
            candidates(frame.loc[~valid]), frame.loc[~valid, "label"].astype(int).tolist(),
            20260914 + fold,
        )
        scores[valid] = predict_probability(model, candidates(frame.loc[valid]))
    return scores


def main():
    all_features = pd.read_csv(FEATURE_PATH)
    rank1 = all_features[all_features["rank"] == 1].copy().reset_index(drop=True)
    train = rank1[rank1["split"] == "TRAIN"].copy().reset_index(drop=True)
    dev = rank1[rank1["split"] == "DEV"].copy().reset_index(drop=True)
    train["statistical_score"] = 1 / (1 + np.exp(-0.8 * (train.sde - 5)))
    dev["statistical_score"] = 1 / (1 + np.exp(-0.8 * (dev.sde - 5)))
    train["ml_oof_score"] = fit_oof(train)
    model = fit_model(candidates(train), train.label.astype(int).tolist(), 20260914)
    dev["ml_score"] = predict_probability(model, candidates(dev))

    predictions = {}
    for split, frame in (("TRAIN", train), ("DEV", dev)):
        path = ROOT / ("outputs_validation2/train_predictions.csv" if split == "TRAIN"
                       else "outputs_validation2/dev_predictions.csv")
        output = pd.read_csv(path)[["star_id", "prediction", "confidence"]]
        frame = frame.merge(output, on="star_id", how="left")
        frame["selected_score"] = frame.statistical_score
        frame["is_false_positive"] = (frame.prediction == 1) & (frame.label == 0)
        frame["is_false_negative"] = (frame.prediction == 0) & (frame.label == 1)
        columns = ["star_id", "selected_score", "confidence", "prediction", "label",
                   "period_days", "depth_ppm", "duration_hours", "sde",
                   "n_transit_points", "transit_count", "local_snr",
                   "odd_even_difference", "quarter_recurrence", "edge_fraction",
                   "gap_fraction"]
        frame.sort_values(["selected_score", "confidence"], ascending=False)[
            columns
        ].loc[frame.is_false_positive].to_csv(
            ROOT / f"reports/{split.lower()}_false_positive_analysis.csv", index=False
        )
        predictions[split] = frame

    separability = []
    feature_columns = ["sde", "depth_ppm", "duration_hours", "n_transit_points",
                       "transit_count", "local_snr", "odd_even_difference",
                       "quarter_recurrence", "edge_fraction", "gap_fraction",
                       "period_days", "selected_score"]
    train_frame = predictions["TRAIN"]
    for feature in feature_columns:
        positive = train_frame.loc[train_frame.label == 1, feature].astype(float)
        negative = train_frame.loc[train_frame.label == 0, feature].astype(float)
        separability.append({
            "feature": feature,
            "positive_median": positive.median(),
            "negative_median": negative.median(),
            "positive_mean": positive.mean(),
            "negative_mean": negative.mean(),
            "median_difference": positive.median() - negative.median(),
            "positive_iqr": positive.quantile(.75) - positive.quantile(.25),
            "negative_iqr": negative.quantile(.75) - negative.quantile(.25),
        })
    pd.DataFrame(separability).to_csv(ROOT / "reports/candidate_separability.csv", index=False)

    rows = []
    for name, scores in (
        ("statistical_baseline", train.statistical_score.to_numpy()),
        ("dependency_free_ml_oof", train.ml_oof_score.to_numpy()),
    ):
        best = max(
            (metrics(train.label, scores, threshold) | {"threshold": threshold}
             for threshold in np.unique(np.r_[scores, np.linspace(0, 1, 101)])),
            key=lambda row: row["f1"],
        )
        rows.append({"model": name, **best, "fold_std": np.nan})
    pd.DataFrame(rows).to_csv(ROOT / "reports/model_comparison.csv", index=False)

    sweep = []
    scores = train.statistical_score.to_numpy()
    for threshold in np.unique(np.r_[scores, np.linspace(0, 1, 101)]):
        sweep.append({"threshold": threshold, **metrics(train.label, scores, threshold)})
    pd.DataFrame(sweep).to_csv(ROOT / "reports/threshold_sweep.csv", index=False)

    errors = predictions["DEV"][
        predictions["DEV"].is_false_positive | predictions["DEV"].is_false_negative
    ].copy()
    errors["error_type"] = np.where(errors.is_false_positive, "false_positive", "false_negative")
    errors["failure_category"] = np.select(
        [
            errors.transit_count <= 1,
            errors.quarter_recurrence < 0.15,
            errors.gap_fraction > 0.25,
            errors.odd_even_difference > 0.05,
        ],
        ["single_event_or_weak_recurrence", "single_quarter_dominance",
         "gap_dominated", "odd_even_inconsistency"],
        default="candidate_not_separable",
    )
    errors.to_csv(ROOT / "reports/dev_error_analysis.csv", index=False)

    config = json.loads(CONFIG_PATH.read_text())
    print("classifier_training_score=feature_matrix(rank1 TRAIN candidates)")
    print("classifier_oof_score=dependency_free_ml_oof")
    print("selected_decision_score=statistical_baseline")
    print("reported_confidence=OOF isotonic calibration of selected score")
    print(json.dumps({
        "train_false_positives": int(predictions["TRAIN"].is_false_positive.sum()),
        "dev_false_positives": int(predictions["DEV"].is_false_positive.sum()),
        "model_comparison": rows,
        "frozen_score_source": config.get("selected_score_source"),
        "frozen_threshold": config.get("selected_threshold"),
    }, indent=2))


if __name__ == "__main__":
    main()
