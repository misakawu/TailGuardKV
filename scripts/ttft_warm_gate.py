#!/usr/bin/env python3
"""Fail closed on the three-session TTFT warmup replay gate."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


POLICIES = ("full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic")
AUDIT_FIELDS = {"worker_startup_ms", "worker_model_load_ms"}


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _number(row: dict[str, object], field: str, *, default: float | None = None) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def validate_gate_records(
    rows: list[dict[str, object]],
    *,
    calibration_p99_ms: dict[str, float],
    expected_request_ids: set[str] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    expected_count = len(POLICIES) * 15
    if len(rows) != expected_count:
        errors.append(f"expected 75 records, got {len(rows)}")

    by_policy: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        policy = str(row.get("policy") or "")
        if policy not in POLICIES:
            errors.append(f"unexpected policy: {policy or '<empty>'}")
            continue
        by_policy[policy].append(row)

    for policy in POLICIES:
        policy_rows = by_policy.get(policy, [])
        if len(policy_rows) != 15:
            errors.append(f"{policy}: expected 15 records, got {len(policy_rows)}")
            continue
        request_ids = {str(row.get("request_id") or "") for row in policy_rows}
        if len(request_ids) != 15 or "" in request_ids:
            errors.append(f"{policy}: missing or duplicate request records")
        if expected_request_ids is not None and request_ids != expected_request_ids:
            errors.append(f"{policy}: request coverage differs from the three-session fixture")
        worker_ids = {
            (str(row.get("extra_worker_pid") or ""), str(row.get("extra_worker_generation") or ""))
            for row in policy_rows
        }
        if "" in {part for worker in worker_ids for part in worker} or len(worker_ids) != 1:
            errors.append(f"{policy}: worker restart detected")

    for index, row in enumerate(rows):
        for field in AUDIT_FIELDS:
            if field in row:
                errors.append(f"row {index}: {field} appears in a service column")
        if _truthy(row.get("worker_state_lost")) or _truthy(row.get("extra_worker_state_lost")):
            errors.append(f"row {index}: worker state loss")
        if not _truthy(row.get("ok")):
            errors.append(f"row {index}: execution failure")
        profile = str(row.get("action_profile") or row.get("profile") or "")
        p99 = calibration_p99_ms.get(profile)
        if p99 is None or not math.isfinite(p99) or p99 <= 0:
            errors.append(f"row {index}: missing calibration p99 for {profile or '<empty>'}")
            continue
        limit = p99 * 5.0
        for field, label in (("ttft_ms", "TTFT"), ("recompute_ms", "recompute")):
            value = _number(row, field, default=0.0)
            if value is None or value < 0 or value >= limit:
                errors.append(f"row {index}: {label} threshold exceeded ({value}, limit {limit})")

    return {
        "diagnostic_only": True,
        "record_count": len(rows),
        "expected_record_count": expected_count,
        "policies": list(POLICIES),
        "calibration_p99_ms": calibration_p99_ms,
        "passed": not errors,
        "errors": errors,
    }


def calibration_p99(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    by_profile: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row, "ttft_ms")
        profile = str(row.get("profile") or "")
        if profile and value is not None and value > 0:
            by_profile[profile].append(value)
    result: dict[str, float] = {}
    for profile, values in by_profile.items():
        values.sort()
        result[profile] = values[min(len(values) - 1, math.ceil(len(values) * 0.99) - 1)]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in args.records:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    with args.calibration.open(encoding="utf-8", newline="") as handle:
        p99 = calibration_p99(csv.DictReader(handle))
    expected_ids = {str(row.get("request_id") or "") for row in rows[:15]}
    report = validate_gate_records(rows, calibration_p99_ms=p99, expected_request_ids=expected_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
