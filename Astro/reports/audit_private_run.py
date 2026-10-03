from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from detection.bls import detect_star
from preprocessing import preprocess_star

ROOT = Path(__file__).parents[1]
PRIVATE = ROOT / "private_pack" / "private"
CONFIG = json.loads((ROOT / "outputs_private" / "config_used.json").read_text())


def main():
    preprocessing = []
    candidates = []
    for path in sorted(PRIVATE.glob("*.parquet")):
        result = preprocess_star(pd.read_parquet(path), star_id=path.stem)
        gap = np.diff(result["time"], prepend=result["time"][0]) > 3 * 29.4 / (60 * 24)
        found = detect_star(path.stem, result["time"], result["detrended_flux"],
                            result["quarter"], gap, CONFIG)
        preprocessing.append({
            "star_id": path.stem, "ok": bool(result["ok"]),
            "n_points": int(result["n_points"]),
            "large_gap_count": int(result["qc"].get("large_gap_count", 0)),
            "edge_affected_points": int(np.sum(result["edge_affected"])),
            "finite_time": bool(np.isfinite(result["time"]).all()),
            "finite_flux": bool(np.isfinite(result["detrended_flux"]).all()),
        })
        best = max(found, key=lambda c: c.score, default=None)
        candidates.append({
            "star_id": path.stem, "candidate_count": len(found),
            "period": best.period_days if best else None,
            "epoch": best.epoch if best else None,
            "depth_ppm": best.depth_ppm if best else None,
            "duration_hours": best.duration_hours if best else None,
            "score": best.confidence if best else 0.0,
            "sde": best.sde if best else None,
            "transit_count": best.transit_count if best else 0,
            "quarter_recurrence": best.quarter_recurrence if best else 0.0,
            "edge_fraction": best.edge_fraction if best else 0.0,
            "gap_fraction": best.gap_fraction if best else 0.0,
        })
    (ROOT / "reports" / "private_preprocessing_audit.json").write_text(
        json.dumps({"stars": preprocessing, "count": len(preprocessing)}, indent=2)
    )
    (ROOT / "reports" / "private_candidate_audit.csv").write_text(
        pd.DataFrame(candidates).to_csv(index=False)
    )

    submission = pd.read_csv(ROOT / "outputs_private" / "submission.csv", keep_default_na=False)
    expected = {f"STAR_{i:04d}" for i in range(87)}
    positive = submission[submission.prediction == 1]
    audit = {
        "star_count": len(submission),
        "positive_predictions": int((submission.prediction == 1).sum()),
        "negative_predictions": int((submission.prediction == 0).sum()),
        "missing_ids": sorted(expected - set(submission.star_id)),
        "unexpected_ids": sorted(set(submission.star_id) - expected),
        "duplicate_ids": int(submission.star_id.duplicated().sum()),
        "characterization_complete_for_positive": bool(
            positive[["period", "depth_ppm", "duration_hours"]].ne("").all().all()
        ),
        "negative_characterization_blank": bool(
            submission.loc[submission.prediction == 0,
                           ["period", "depth_ppm", "duration_hours"]].eq("").all().all()
        ),
        "confidence_min": float(submission.confidence.astype(float).min()),
        "confidence_max": float(submission.confidence.astype(float).max()),
        "period_min_positive": float(positive.period.astype(float).min()) if len(positive) else None,
        "period_max_positive": float(positive.period.astype(float).max()) if len(positive) else None,
        "depth_min_positive": float(positive.depth_ppm.astype(float).min()) if len(positive) else None,
        "duration_min_positive": float(positive.duration_hours.astype(float).min()) if len(positive) else None,
    }
    (ROOT / "reports" / "private_predictions_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
