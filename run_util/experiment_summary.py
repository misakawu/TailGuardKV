from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from run_util.experiment_common import json_ready


SUMMARY_KEY_COLUMNS = [
    "section",
    "name",
    "ok",
    "error",
    "diagnostic_output",
    "failures",
    "config",
    "run_dir",
    "visual_outputs",
    "visual_error",
    "profile_rows",
    "policy_rows",
    "epsilon",
    "delta",
    "memory_budget_mib",
    "count",
    "ok_count",
    "measured_count",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_peak_memory_mib",
    "p95_peak_memory_mib",
    "mean_kv_cache_memory_mib",
    "p95_kv_cache_memory_mib",
    "mean_quality_loss",
    "p50_quality_loss",
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
    "has_h0_tail_metrics",
    "has_h1_coverage_metrics",
    "has_h2_lite_benefit_metrics",
    "deployable_baseline_names",
]

TOTAL_POLICY_SUMMARY_COLUMNS = [
    "config",
    "run_dir",
    "policy",
    "memory_budget_mib",
    "epsilon",
    "delta",
    "mean_quality_loss",
    "p50_quality_loss",
    "p95_quality_loss",
    "p99_quality_loss",
    "cvar_quality_loss",
    "violation_rate",
    "worst_group_violation",
    "delta_slack",
    "safe_ratio",
    "risk_upper_mean",
    "pred_loss_mean",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_peak_memory_mib",
    "p95_peak_memory_mib",
    "mean_kv_cache_memory_mib",
    "p95_kv_cache_memory_mib",
    "fallback_ratio",
    "exact_fallback_ratio",
    "exact_action_ratio",
    "lossy_action_ratio",
    "candidate_safe_count",
    "controller_overhead_ms",
    "controller_qrp_ms",
    "controller_cg_ms",
    "controller_stc_ms",
    "oracle_cost_ms",
    "optimality_gap",
    "audit_rate",
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
    evidence = _evidence_fields(payload)
    rows = [
        {
            "section": "experiment",
            "name": str(payload.get("experiment_name") or "pilot-smoke-measured"),
            "ok": payload.get("ok"),
            "error": _summary_error(payload),
            "diagnostic_output": payload.get("diagnostic_output", ""),
            "failures": payload.get("failures", ""),
            "config": payload.get("config"),
            "run_dir": payload.get("run_dir", ""),
            "visual_outputs": payload.get("visual_outputs", ""),
            "visual_error": payload.get("visual_error", ""),
            "profile_rows": (payload.get("rows") or {}).get("profiles") if isinstance(payload.get("rows"), dict) else "",
            "policy_rows": (payload.get("rows") or {}).get("policy") if isinstance(payload.get("rows"), dict) else "",
            "epsilon": payload.get("epsilon"),
            "delta": payload.get("delta"),
            "memory_budget_mib": payload.get("memory_budget_mib"),
            **evidence,
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


def total_policy_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten backend/policy outcome summaries for downstream tables and plots."""
    rows: list[dict[str, Any]] = []
    policy_runs = payload.get("policy_runs")
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
            for policy, metrics in summary.items():
                row = {column: "" for column in TOTAL_POLICY_SUMMARY_COLUMNS}
                row.update(
                    {
                        "config": payload.get("config"),
                        "run_dir": payload.get("run_dir", ""),
                        "policy": policy,
                        "memory_budget_mib": policy_run.get("memory_budget_mib"),
                        "epsilon": policy_run.get("epsilon"),
                        "delta": policy_run.get("delta"),
                    }
                )
                if isinstance(metrics, dict):
                    for column in TOTAL_POLICY_SUMMARY_COLUMNS:
                        if column in metrics:
                            row[column] = metrics[column]
                rows.append(row)
    else:
        policy_payload = payload.get("policy")
        if isinstance(policy_payload, dict):
            summary = policy_payload.get("summary")
            if isinstance(summary, dict):
                for policy, metrics in summary.items():
                    row = {column: "" for column in TOTAL_POLICY_SUMMARY_COLUMNS}
                    row.update(
                        {
                            "config": payload.get("config"),
                            "run_dir": payload.get("run_dir", ""),
                            "policy": policy,
                            "memory_budget_mib": payload.get("memory_budget_mib"),
                            "epsilon": payload.get("epsilon"),
                            "delta": payload.get("delta"),
                        }
                    )
                    if isinstance(metrics, dict):
                        for column in TOTAL_POLICY_SUMMARY_COLUMNS:
                            if column in metrics:
                                row[column] = metrics[column]
                    rows.append(row)
    return rows


def _evidence_fields(payload: dict[str, Any]) -> dict[str, Any]:
    policy_metrics = []
    policy_runs = payload.get("policy_runs")
    if isinstance(policy_runs, list):
        for policy_run in policy_runs:
            if isinstance(policy_run, dict):
                policy_metrics.extend(_all_summary_metrics(policy_run.get("payload")))
    else:
        policy_metrics.extend(_all_summary_metrics(payload.get("policy")))
    evidence_metrics = policy_metrics
    if not evidence_metrics and payload.get("policy") is None and not isinstance(policy_runs, list):
        evidence_metrics = _all_summary_metrics(payload.get("profile"))
    deployable = [
        str(name)
        for name in payload.get("policies", [])
        if str(name) != "quality_oracle"
    ]
    return {
        "has_h0_tail_metrics": _has_any_metric(evidence_metrics, {"p95_quality_loss", "p99_quality_loss", "cvar_quality_loss", "worst_group_violation"}),
        "has_h1_coverage_metrics": _has_any_metric(evidence_metrics, {"safe_ratio", "fallback_ratio", "exact_fallback_ratio", "candidate_safe_count"}),
        "has_h2_lite_benefit_metrics": bool(
            deployable
            and _has_any_metric(evidence_metrics, {"p95_ttft_ms", "mean_ttft_ms"})
            and _has_any_metric(evidence_metrics, {"mean_kv_cache_memory_mib", "p95_kv_cache_memory_mib", "mean_peak_memory_mib"})
        ),
        "deployable_baseline_names": deployable,
    }


def _all_summary_metrics(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return []
    return [metrics for metrics in summary.values() if isinstance(metrics, dict)]


def _has_any_metric(metrics: list[dict[str, Any]], names: set[str]) -> bool:
    return any(any(name in metric for name in names) for metric in metrics)


def write_summary(payload: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(payload)
    fieldnames = SUMMARY_KEY_COLUMNS + sorted({key for row in rows for key in row}.difference(SUMMARY_KEY_COLUMNS))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def write_total_policy_summary(payload: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = total_policy_summary_rows(payload)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOTAL_POLICY_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
