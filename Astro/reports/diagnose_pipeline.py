from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from detection.bls import detect_star
from features.transit_features import FEATURE_NAMES, candidate_features
from preprocessing import preprocess_star


ROOT = Path(__file__).parents[1]
CONFIG = json.loads((ROOT / "configs" / "final_config.json").read_text())


def truth_map(path):
    frame = pd.read_csv(path)
    return frame.set_index("kepid").to_dict("index")


def run_split(data_dir, labels_path, truth_path, split):
    labels = pd.read_csv(labels_path).set_index("kepid")["label"].to_dict()
    truth = truth_map(truth_path)
    records = []
    feature_rows = []
    for path in sorted(Path(data_dir).glob("*.parquet")):
        kepid = int(path.stem.split("_")[-1])
        result = preprocess_star(pd.read_parquet(path), star_id=path.stem)
        if not result["ok"]:
            continue
        gap = np.diff(result["time"], prepend=result["time"][0]) > 3 * 29.4 / (60 * 24)
        candidates = detect_star(path.stem, result["time"], result["detrended_flux"],
                                 result["quarter"], gap, CONFIG)
        row_truth = truth.get(kepid, {})
        true_period = float(row_truth.get("period_days", np.nan))
        positive = int(labels.get(kepid, 0)) == 1

        def match(candidate):
            if not positive or not np.isfinite(true_period):
                return False
            ratio = candidate.period_days / true_period
            aliases = (1, 0.5, 2, 1 / 3, 3, 0.25, 4)
            return min(abs(ratio / alias - 1) for alias in aliases) <= 0.02

        matches = [match(candidate) for candidate in candidates]
        best = candidates[0] if candidates else None
        records.append({
            "split": split, "star_id": path.stem, "label": int(positive),
            "candidate_count": len(candidates),
            "best_period_days": best.period_days if best else np.nan,
            "best_score": best.score if best else np.nan,
            "best_sde": best.sde if best else np.nan,
            "best_depth_ppm": best.depth_ppm if best else np.nan,
            "best_duration_hours": best.duration_hours if best else np.nan,
            "best_epoch": best.epoch if best else np.nan,
            "best_n_transit_points": best.n_transit_points if best else 0,
            "best_confidence": best.confidence if best else 0.0,
            "truth_period_days": true_period,
            "top1_match": bool(matches[0]) if matches else False,
            "top3_match": any(matches[:3]),
            "top5_match": any(matches[:5]),
            "top10_match": any(matches[:10]),
        })
        for rank, candidate in enumerate(candidates, 1):
            values = candidate_features(candidate)
            feature_rows.append({
                "split": split, "star_id": path.stem, "label": int(positive),
                "rank": rank, "period_days": candidate.period_days,
                "depth_ppm": candidate.depth_ppm, "duration_hours": candidate.duration_hours,
                **{name: value for name, value in zip(FEATURE_NAMES, values)},
                "matches_truth": bool(match(candidate)),
            })
    return pd.DataFrame(records), pd.DataFrame(feature_rows)


def main():
    train, train_features = run_split(
        ROOT / "train_pack" / "train", ROOT / "train_pack" / "train_labels.csv",
        ROOT / "train_pack" / "train_truth.csv", "TRAIN")
    dev, dev_features = run_split(
        ROOT / "dev_pack" / "dev", ROOT / "dev_pack" / "dev_labels.csv",
        ROOT / "dev_pack" / "dev_truth.csv", "DEV")
    records = pd.concat([train, dev], ignore_index=True)
    features = pd.concat([train_features, dev_features], ignore_index=True)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    positive = records[records.label == 1]
    diagnostics = positive.copy()
    diagnostics["period_error_percent"] = (
        (diagnostics.best_period_days - diagnostics.truth_period_days)
        .abs() / diagnostics.truth_period_days * 100
    )
    diagnostics.to_csv(reports / "bls_positive_diagnostics.csv", index=False)
    features.to_csv(reports / "feature_diagnostics.csv", index=False)
    summary = {}
    for split, frame in (("TRAIN", train), ("DEV", dev)):
        pos = frame[frame.label == 1]
        summary[split] = {
            "stars": int(len(frame)), "positive_stars": int(len(pos)),
            "valid_candidate_rate": float(np.mean(pos.candidate_count > 0)),
            "top1_recovery": float(pos.top1_match.mean()),
            "top3_recovery": float(pos.top3_match.mean()),
            "top5_recovery": float(pos.top5_match.mean()),
            "top10_recovery": float(pos.top10_match.mean()),
            "median_period_error_percent": float(pos.period_error_percent.median())
            if "period_error_percent" in pos else None,
            "candidate_count_mean": float(frame.candidate_count.mean()),
        }
    (reports / "bls_diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("feature_names:", list(FEATURE_NAMES))


if __name__ == "__main__":
    main()
