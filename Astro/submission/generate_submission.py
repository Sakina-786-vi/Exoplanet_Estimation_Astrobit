from __future__ import annotations

import csv
from pathlib import Path

from .validate_submission import HEADER, validate


def write_submission(rows, path, expected_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["star_id"]))
    validate(path, expected_ids)
    return path
