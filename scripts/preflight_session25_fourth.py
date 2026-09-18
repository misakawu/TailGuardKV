#!/usr/bin/env python3
"""Fail closed on the 25-session diagnostic input before any GPU jobs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def audit_fixture(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    sessions: dict[str, list[int]] = {}
    request_ids: set[str] = set()
    arrivals: list[int] = []
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id") or "")
        session_id = str(row.get("session_id") or "")
        if not request_id or request_id in request_ids or not session_id:
            raise ValueError(f"invalid/duplicate request or session ID at row {index}")
        request_ids.add(request_id)
        sessions.setdefault(session_id, []).append(int(row["turn_index"]))
        arrivals.append(int(row["arrival_index"]))
    if len(rows) != 125 or len(sessions) != 25:
        raise ValueError(f"expected 25 sessions / 125 requests, got {len(sessions)} / {len(rows)}")
    if any(arrival < 0 for arrival in arrivals) or arrivals != sorted(set(arrivals)):
        raise ValueError("arrival_index must be nonnegative, unique, and strictly ordered")
    for session_id, turns in sessions.items():
        if turns != [0, 1, 2, 3, 4]:
            raise ValueError(f"session {session_id} has non-continuous turns: {turns}")
    return {
        "diagnostic_only": True,
        "session_count": len(sessions),
        "request_count": len(rows),
        "arrival_gap_count": arrivals[-1] - arrivals[0] + 1 - len(arrivals),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "fixture": str(path.resolve()),
    }


def audit_tight_budget_rows(
    rows: list[dict[str, str]], *, budgets: list[float], policies: list[str]
) -> dict[str, object]:
    """Fail closed on missing records or budget overruns; retain no-pressure controls."""
    if len(budgets) != 2 or len(set(budgets)) != 2:
        raise ValueError("tight budget audit requires exactly two distinct budgets")
    lossy_policies = [policy for policy in policies if policy not in {"full_lru", "static_safe"}]
    cells: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        policy = str(row.get("policy") or "")
        if policy not in policies:
            continue
        budget = float(row.get("global_budget_mib") or 0.0)
        cells.setdefault((policy, budget), []).append(row)
    audit_cells: list[dict[str, object]] = []
    failures: list[str] = []
    for policy in lossy_policies:
        for budget in budgets:
            cell = cells.get((policy, budget), [])
            filtered = sum(str(row.get("policy_budget_filtered") or "").lower() == "true" for row in cell)
            backend_hits = sum(str(row.get("backend_budget_hit") or "").lower() == "true" for row in cell)
            residents = [float(row.get("global_resident_kv_mib") or 0.0) for row in cell]
            maximum = max(residents, default=0.0)
            has_pressure = filtered > 0 or backend_hits > 0
            passed = bool(cell) and maximum <= budget + 1e-9
            audit_cells.append({"policy": policy, "budget_mib": budget, "policy_budget_filtered_count": filtered,
                                "backend_budget_hit_count": backend_hits, "max_global_resident_mib": maximum,
                                "pressure_status": "observed" if has_pressure else "expected_no_pressure_control",
                                "passed": passed})
            if not passed:
                failures.append(f"{policy}@{budget:g}")
    return {"diagnostic_only": True, "budgets_mib": budgets, "cells": audit_cells, "passed": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-csv", type=Path, action="append", default=[])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    if config.get("diagnostic_only") is not True or config.get("experiment", {}).get("type") != "baseline_session":
        raise ValueError("config must be diagnostic_only baseline_session")
    if config.get("backend", {}).get("name") != "online_qwen" or config.get("policies", {}).get("backend") != "online_qwen":
        raise ValueError("config must use online_qwen backend")
    if int(config.get("data", {}).get("max_requests", 0)) != 125:
        raise ValueError("config must retain all 125 requests")
    if args.fixture.resolve() != (args.config.resolve().parent.parent / config["data"]["requests"]).resolve():
        raise ValueError("config and input fixture paths differ")
    report = audit_fixture(args.fixture)
    if args.policy_csv:
        rows: list[dict[str, str]] = []
        for path in args.policy_csv:
            with path.open(encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        budgets = [float(value) for value in config["pilot"]["memory_budgets_mib"]]
        report["tight_budget_audit"] = audit_tight_budget_rows(rows, budgets=budgets, policies=config["policies"]["names"])
        if not report["tight_budget_audit"]["passed"]:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
