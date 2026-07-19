from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from core_types import ProfileMeasurement
from validation import validate_profile_measurements, validate_profile_table_header


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_measurements(path: Path) -> list[ProfileMeasurement]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_profile_table_header(reader.fieldnames or [], path)
        measurements = [ProfileMeasurement.from_row(row) for row in reader]
    validate_profile_measurements(measurements, path)
    return measurements


def json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value
