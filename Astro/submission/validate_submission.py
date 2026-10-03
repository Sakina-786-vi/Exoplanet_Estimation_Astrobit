from __future__ import annotations

import csv
import math
from pathlib import Path

HEADER = ["star_id", "prediction", "confidence", "period", "depth_ppm", "duration_hours"]


def validate(path, expected_ids):
    path = Path(path)
    if not path.exists():
        raise AssertionError(f"missing submission: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != HEADER:
        raise AssertionError(f"exact header mismatch: {reader.fieldnames}")
    if len(rows) != len(expected_ids):
        raise AssertionError(f"row count {len(rows)} != {len(expected_ids)}")
    ids = [row["star_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise AssertionError("submission IDs are duplicated or do not match input IDs")
    for row in rows:
        if row["prediction"] not in {"0", "1"}:
            raise AssertionError("prediction must be 0 or 1")
        confidence = float(row["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise AssertionError("confidence must be finite and in [0, 1]")
        characterization = [row["period"], row["depth_ppm"], row["duration_hours"]]
        if row["prediction"] == "0" and any(characterization):
            raise AssertionError("negative rows must not contain characterization")
        if row["prediction"] == "1" and not all(float(value) > 0 for value in characterization):
            raise AssertionError("positive rows need positive characterization")
    return True
