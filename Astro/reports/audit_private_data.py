from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
PRIVATE = ROOT / "private_pack" / "private"
EXPECTED = [f"STAR_{i:04d}" for i in range(87)]
REQUIRED = ["time", "flux", "flux_err", "quality", "quarter"]


def main():
    files = sorted(PRIVATE.glob("*.parquet"))
    ids = [path.stem for path in files]
    rows = []
    missing_columns = {}
    for path in files:
        frame = pd.read_parquet(path)
        missing = [column for column in REQUIRED if column not in frame.columns]
        if missing:
            missing_columns[path.stem] = missing
        numeric = frame.select_dtypes(include="number")
        nonfinite = int((~np.isfinite(numeric.to_numpy())).sum())
        time = frame["time"].to_numpy(dtype=float) if "time" in frame else np.array([])
        quarter = frame["quarter"].dropna().unique().tolist() if "quarter" in frame else []
        rows.append({
            "star_id": path.stem,
            "rows": len(frame),
            "missing_columns": missing,
            "nonfinite_numeric_values": nonfinite,
            "time_min": float(np.nanmin(time)) if len(time) else None,
            "time_max": float(np.nanmax(time)) if len(time) else None,
            "quarter_count": len(quarter),
            "quarters": quarter,
        })
    audit = {
        "directory": str(PRIVATE),
        "file_count": len(files),
        "unique_star_ids": len(set(ids)),
        "missing_expected_ids": sorted(set(EXPECTED) - set(ids)),
        "unexpected_ids": sorted(set(ids) - set(EXPECTED)),
        "required_columns": REQUIRED,
        "missing_columns": missing_columns,
        "stars": rows,
    }
    (ROOT / "reports" / "private_data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        "file_count": audit["file_count"],
        "missing_expected_ids": audit["missing_expected_ids"],
        "unexpected_ids": audit["unexpected_ids"],
        "missing_columns": audit["missing_columns"],
    }, indent=2))
    if audit["file_count"] != 87 or audit["missing_expected_ids"] or audit["unexpected_ids"] or missing_columns:
        raise RuntimeError("PRIVATE data audit failed")


if __name__ == "__main__":
    main()
