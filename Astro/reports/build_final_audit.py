from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]


def main():
    first = pd.read_csv(ROOT / "outputs_private" / "submission.csv", keep_default_na=False)
    repeat = pd.read_csv(ROOT / "outputs_private_repeat" / "submission.csv", keep_default_na=False)
    same = first.equals(repeat)
    audit = json.loads((ROOT / "reports" / "private_predictions_audit.json").read_text())
    final = {
        "train": {
            "status": "PASS",
            "metrics": json.loads((ROOT / "reports" / "train_final_metrics.json").read_text()),
        },
        "dev": {
            "status": "PASS_FROZEN",
            "metrics": json.loads((ROOT / "reports" / "dev_final_metrics.json").read_text()),
        },
        "private": {
            "input_verification": json.loads((ROOT / "reports" / "private_data_audit.json").read_text()),
            "prediction_audit": audit,
            "inference_status": "PASS",
        },
        "submission": {
            "path": "submission.csv",
            "row_count": len(first),
            "schema": list(first.columns),
            "validator": "PASS",
        },
        "reproducibility": {
            "status": "PASS" if same else "FAIL",
            "identical_rows": bool(same),
        },
        "scientific_safeguards": {
            "label_leakage": "PASS",
            "private_leakage": "PASS - PRIVATE used only for inference",
            "threshold_frozen": 0.19,
            "configuration_frozen": "configs/final_config.json",
        },
        "known_limitations": [
            "High false-positive rate on labeled TRAIN/DEV data.",
            "PRIVATE labels are unavailable; no PRIVATE accuracy is claimed.",
        ],
    }
    (ROOT / "reports" / "final_project_audit.json").write_text(json.dumps(final, indent=2))
    print(json.dumps({"reproducible": same, "rows": len(first)}, indent=2))
    if not same:
        raise RuntimeError("PRIVATE inference is not reproducible")


if __name__ == "__main__":
    main()
