from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiment_common import json_ready


SUMMARY_KEY_COLUMNS = [
    "section",
    "name",
    "ok",
    "error",
    "diagnostic_output",
    "failures",
    "config",
    "profile_rows",
    "policy_rows",
    "epsilon",
    "delta",
    "memory_budget_mib",
    "count",
    "ok_count",
    "measured_count",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_peak_memory_mib",
    "p95_peak_memory_mib",
    "mean_kv_cache_memory_mib",
    "p95_kv_cache_memory_mib",
    "mean_quality_loss",
    "p95_quality_loss",
    "p99_quality_loss",
    "cvar_quality_loss",
    "violation_rate",
    "delta_slack",
    "worst_group_violation",
    "safe_ratio",
    "fallback_ratio",
    "exact_fallback_ratio",
    "exact_action_ratio",
    "lossy_action_ratio",
    "unique_action_count",
    "identical_to_full_lru",
    "unsafe_action_count",
    "candidate_safe_count",
    "action_distribution",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return json_ready(value)


def _summary_error(payload: dict[str, Any]) -> Any:
    if payload.get("error"):
        return payload.get("error")
    policy_runs = payload.get("policy_runs")
    sections = ("profile",) if isinstance(policy_runs, list) else ("profile", "policy")
    for section in sections:
        nested = payload.get(section)
        if isinstance(nested, dict) and nested.get("error"):
            return nested.get("error")
    if isinstance(policy_runs, list):
        for policy_run in policy_runs:
            if not isinstance(policy_run, dict):
                continue
            run_payload = policy_run.get("payload")
            if isinstance(run_payload, dict) and run_payload.get("error"):
                return run_payload.get("error")
    return ""


def summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "section": "experiment",
            "name": "pilot-smoke-measured",
            "ok": payload.get("ok"),
            "error": _summary_error(payload),
            "diagnostic_output": payload.get("diagnostic_output", ""),
            "failures": payload.get("failures", ""),
            "config": payload.get("config"),
            "profile_rows": (payload.get("rows") or {}).get("profiles") if isinstance(payload.get("rows"), dict) else "",
            "policy_rows": (payload.get("rows") or {}).get("policy") if isinstance(payload.get("rows"), dict) else "",
            "epsilon": payload.get("epsilon"),
            "delta": payload.get("delta"),
            "memory_budget_mib": payload.get("memory_budget_mib"),
        }
    ]
    policy_runs = payload.get("policy_runs")
    sections = ("profile",) if isinstance(policy_runs, list) else ("profile", "policy")
    for section in sections:
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            continue
        summary = section_payload.get("summary")
        if not isinstance(summary, dict):
            continue
        for name, metrics in summary.items():
            row = {
                "section": section,
                "name": name,
                "ok": payload.get("ok"),
                "config": payload.get("config"),
                "epsilon": payload.get("epsilon"),
                "delta": payload.get("delta"),
                "memory_budget_mib": payload.get("memory_budget_mib"),
            }
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
    if isinstance(policy_runs, list):
        for policy_run in policy_runs:
            if not isinstance(policy_run, dict):
                continue
            run_payload = policy_run.get("payload")
            if not isinstance(run_payload, dict):
                continue
            summary = run_payload.get("summary")
            if not isinstance(summary, dict):
                continue
            for name, metrics in summary.items():
                row = {
                    "section": "policy",
                    "name": name,
                    "ok": policy_run.get("ok"),
                    "config": payload.get("config"),
                    "epsilon": policy_run.get("epsilon"),
                    "delta": policy_run.get("delta"),
                    "memory_budget_mib": policy_run.get("memory_budget_mib"),
                }
                if isinstance(metrics, dict):
                    row.update(metrics)
                rows.append(row)
    return rows


def write_summary(payload: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(payload)
    fieldnames = SUMMARY_KEY_COLUMNS + sorted({key for row in rows for key in row}.difference(SUMMARY_KEY_COLUMNS))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
