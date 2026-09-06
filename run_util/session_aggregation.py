"""Aggregate per-turn policy CSVs into per-cell baseline smoke summaries.

Each policy CSV produced by a session27 sweep contains one turn-record per row.
Records for the same ``(memory_budget_mib, epsilon, delta)`` cell may be spread
across multiple directories/batches; this module merges the raw per-turn records
first and only then computes metrics, so the overall P95 is never an average of
batch P95 values.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from math import isnan
from pathlib import Path
from typing import Any, Iterable

from metrics import session_block_bootstrap_ci
from metrics.collector import MetricCollector
from run_util.core_types import PolicyRunRecord
from run_util.experiment_common import json_ready
from run_util.experiment_summary import TOTAL_POLICY_SUMMARY_COLUMNS, add_shadow_audit_metrics


SWEEP_FILENAME_RE = re.compile(
    r"policy_eps(?P<epsilon>[0-9mp]+)_delta(?P<delta>[0-9mp]+)_mem(?P<memory>[0-9mp]+)\.csv$"
)
SESSION_BLOCK_SEED = 20260906
SESSION_POINT_COLUMNS = [
    "policy",
    "memory_budget_mib",
    "epsilon",
    "delta",
    "session_id",
    "count",
    "p95_ttft_ms",
    "mean_ttft_ms",
    "p95_quality_loss",
    "mean_quality_loss",
    "mean_kv_cache_memory_mib",
    "violation_rate",
]
EVENTS_CSV_COLUMNS = [
    "policy",
    "memory_budget_mib",
    "epsilon",
    "delta",
    "count",
    "ok_count",
    "session_count",
    "budget_hit_count",
    "budget_hit_rate",
    "restore_count",
    "restore_time_ms_total",
    "recompute_count",
    "recompute_time_ms_total",
    "queue_event_count",
    "queue_delay_ms_mean",
    "evict_event_count",
    "session_reuse_evidence",
    "backend_event_evidence",
]
BOOTSTRAP_METRICS = (
    "p95_ttft_ms",
    "mean_ttft_ms",
    "mean_quality_loss",
    "p95_quality_loss",
    "violation_rate",
)


def _parse_number(token: str) -> float:
    return float(token.replace("p", ".").replace("m", "-"))


def parse_sweep_filename(filename: str) -> dict[str, float]:
    """Extract (memory_budget_mib, epsilon, delta) from a sweep policy CSV name."""
    match = SWEEP_FILENAME_RE.search(Path(filename).name)
    if not match:
        raise ValueError(f"无法从文件名解析 sweep 参数: {filename}")
    return {
        "epsilon": _parse_number(match.group("epsilon")),
        "delta": _parse_number(match.group("delta")),
        "memory_budget_mib": _parse_number(match.group("memory")),
    }


def bootstrap_ci_columns(metric: str) -> tuple[str, str]:
    """Return the low/high bootstrap CI column names for a summary metric."""
    if metric.endswith("_ms"):
        stem = metric[: -len("_ms")]
        return f"{stem}_ci_low_ms", f"{stem}_ci_high_ms"
    return f"{metric}_ci_low", f"{metric}_ci_high"


BOOTSTRAP_CI_COLUMNS: list[str] = []
for _metric in BOOTSTRAP_METRICS:
    BOOTSTRAP_CI_COLUMNS.extend(bootstrap_ci_columns(_metric))


def total_summary_columns() -> list[str]:
    return [*TOTAL_POLICY_SUMMARY_COLUMNS, *BOOTSTRAP_CI_COLUMNS]


@dataclass
class PolicyCell:
    memory_budget_mib: float
    epsilon: float
    delta: float
    records: list[PolicyRunRecord] = field(default_factory=list)
    config: str = ""
    run_dir: str = ""
    diagnostic_only: bool = True
    quality_status: str = "risk_evidence_insufficient"
    violation_status: str = "risk_evidence_insufficient"
    source_paths: tuple[Path, ...] = ()

    @property
    def key(self) -> tuple[float, float, float]:
        return (self.memory_budget_mib, self.epsilon, self.delta)


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else 0


def _record_from_row(row: dict[str, str]) -> PolicyRunRecord:
    return PolicyRunRecord(
        policy=row["policy"],
        request_id=row["request_id"],
        action_profile=row["action_profile"],
        ok=_parse_bool(row.get("ok")),
        measured=_parse_bool(row.get("measured")),
        backend_name=row.get("backend_name") or "",
        session_id=row.get("session_id") or None,
        turn_index=_parse_int(row.get("turn_index")),
        task=row.get("task") or "unknown",
        length_bucket=row.get("length_bucket") or "unknown",
        placeholder=_parse_bool(row.get("placeholder")),
        reason=row.get("reason") or "",
        error=row.get("error") or None,
        latency_ms=_parse_float(row.get("latency_ms")),
        ttft_ms=_parse_float(row.get("ttft_ms")),
        peak_memory_mib=_parse_float(row.get("peak_memory_mib")),
        kv_cache_memory_mib=_parse_float(row.get("kv_cache_memory_mib")),
        resident_memory_mib=_parse_float(row.get("resident_memory_mib")),
        kv_cumulative_mib=_parse_float(row.get("kv_cumulative_mib")),
        kv_incremental_mib=_parse_float(row.get("kv_incremental_mib")),
        resident_kv_mib_before=_parse_float(row.get("resident_kv_mib_before")),
        resident_kv_mib_after=_parse_float(row.get("resident_kv_mib_after")),
        restore_ms=_parse_float(row.get("restore_ms")),
        recompute_ms=_parse_float(row.get("recompute_ms")),
        queue_delay_ms=_parse_float(row.get("queue_delay_ms")),
        evicted_kv_mib=_parse_float(row.get("evicted_kv_mib")),
        budget_hit=_parse_bool(row.get("budget_hit")),
        policy_budget_filtered=_parse_bool(row.get("policy_budget_filtered")),
        backend_budget_hit=_parse_bool(row.get("backend_budget_hit")),
        global_resident_kv_mib=_parse_float(row.get("global_resident_kv_mib")),
        global_budget_mib=_parse_float(row.get("global_budget_mib")),
        quality_loss=_parse_float(row.get("quality_loss")),
        audit_selected=_parse_bool(row.get("audit_selected")),
        predicted_quality_loss=_parse_float(row.get("predicted_quality_loss")),
        observed_quality_loss=_parse_float(row.get("observed_quality_loss")),
        quality_estimate=_parse_float(row.get("quality_estimate")),
        primary_profile=row.get("primary_profile") or "",
        exact=_parse_bool(row.get("exact")),
        oracle=_parse_bool(row.get("oracle")),
        pred_loss=_parse_float(row.get("pred_loss")),
        risk_upper=_parse_float(row.get("risk_upper")),
        safe=None if row.get("safe", "").strip() == "" else _parse_bool(row.get("safe")),
        epsilon=_parse_float(row.get("epsilon")),
        delta=_parse_float(row.get("delta")),
        fallback_reason=row.get("fallback_reason") or "",
        safety_reason=row.get("safety_reason") or "",
        rejected_profile=row.get("rejected_profile") or "",
        rejected_pred_loss=_parse_float(row.get("rejected_pred_loss")),
        rejected_risk_upper=_parse_float(row.get("rejected_risk_upper")),
        candidate_safe_count=_parse_float(row.get("candidate_safe_count")),
        controller_overhead_ms=_parse_float(row.get("controller_overhead_ms")),
        controller_qrp_ms=_parse_float(row.get("controller_qrp_ms")),
        controller_cg_ms=_parse_float(row.get("controller_cg_ms")),
        controller_stc_ms=_parse_float(row.get("controller_stc_ms")),
        oracle_cost_ms=_parse_float(row.get("oracle_cost_ms")),
        optimality_gap=_parse_float(row.get("optimality_gap")),
        audit_rate=_parse_float(row.get("audit_rate")),
        drift_state=row.get("drift_state") or "",
        active_session_count=_parse_float(row.get("active_session_count")),
    )


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"policy", "request_id", "action_profile", "ok", "measured"}
            fieldnames = set(reader.fieldnames or ())
            if not required.issubset(fieldnames):
                missing = sorted(required - fieldnames)
                raise ValueError(f"policy CSV 缺少列 {missing}: {path}")
            return list(reader)
    except (OSError, csv.Error) as exc:
        raise ValueError(f"无法读取 policy CSV {path}: {exc}") from exc


def _consistent_text(rows: list[dict[str, str]], key: str, *, default: str = "") -> str:
    values = {str(row.get(key) or "").strip() for row in rows}
    values.discard("")
    if len(values) > 1:
        raise ValueError(f"policy CSV 的 {key} 不一致")
    return next(iter(values), default)


def _file_provenance(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy CSV 没有数据行")
    diagnostic_columns = ["diagnostic_only", "quality_status", "violation_status"]
    has_provenance = [any(row.get(column, "").strip() for column in diagnostic_columns) for row in rows]
    if any(has_provenance) and not all(has_provenance):
        raise ValueError("policy CSV 的 diagnostic provenance 必须出现在每一行")
    diagnostic_only = all(_parse_bool(row.get("diagnostic_only")) for row in rows)
    quality_status = _consistent_text(rows, "quality_status")
    violation_status = _consistent_text(rows, "violation_status")
    config = _consistent_text(rows, "config")
    run_dir = _consistent_text(rows, "run_dir")
    if not diagnostic_only:
        raise ValueError("session27 policy CSV 必须带 diagnostic_only=true")
    if {quality_status, violation_status} != {"risk_evidence_insufficient"}:
        raise ValueError("session27 policy CSV 的 quality/violation 状态必须一致且为 risk_evidence_insufficient")
    if not config or not run_dir:
        raise ValueError("session27 policy CSV 必须携带 config 与 run_dir provenance")
    return {
        "config": config,
        "run_dir": run_dir,
        "diagnostic_only": True,
        "quality_status": quality_status,
        "violation_status": violation_status,
    }


def aggregate_policy_csvs(paths: Iterable[Path]) -> list[PolicyCell]:
    """Merge per-turn records across every policy CSV sharing a constraint cell."""
    cells_by_key: dict[tuple[float, float, float], PolicyCell] = {}
    source_paths: dict[tuple[float, float, float], list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"policy CSV 不存在: {path}")
        sweep = parse_sweep_filename(path.name)
        key = (sweep["memory_budget_mib"], sweep["epsilon"], sweep["delta"])
        rows = _csv_rows(path)
        if not rows:
            raise ValueError(f"policy CSV 为空: {path}")
        records = [_record_from_row(row) for row in rows]
        provenance = _file_provenance(rows)
        source_paths.setdefault(key, []).append(path)
        cell = cells_by_key.get(key)
        if cell is None:
            cells_by_key[key] = PolicyCell(
                memory_budget_mib=sweep["memory_budget_mib"],
                epsilon=sweep["epsilon"],
                delta=sweep["delta"],
                records=list(records),
                **provenance,
            )
            continue
        for field_name in ("config", "run_dir", "quality_status", "violation_status"):
            if getattr(cell, field_name) != provenance[field_name]:
                raise ValueError(
                    f"同 cell {key} 的 {field_name} provenance 不一致: "
                    f"{getattr(cell, field_name)!r} vs {provenance[field_name]!r}"
                )
        cell.records.extend(records)
    cells = list(cells_by_key.values())
    for cell in cells:
        cell.source_paths = tuple(source_paths[cell.key])
    return cells


def _records_for_policy(cell: PolicyCell, policy: str) -> list[PolicyRunRecord]:
    return [record for record in cell.records if record.policy == policy]


def _cell_row(
    cell: PolicyCell,
    policy: str,
    records: list[PolicyRunRecord],
    summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {column: "" for column in total_summary_columns()}
    row.update(
        {
            "config": cell.config,
            "run_dir": cell.run_dir,
            "diagnostic_only": True,
            "quality_status": cell.quality_status,
            "violation_status": cell.violation_status,
            "policy": policy,
            "memory_budget_mib": cell.memory_budget_mib,
            "epsilon": cell.epsilon,
            "delta": cell.delta,
        }
    )
    metrics = summary.get(policy, {})
    for column in total_summary_columns():
        if column in metrics:
            row[column] = metrics[column]
    for column, value in metrics.items():
        if column not in row:
            row[column] = value
    for metric in BOOTSTRAP_METRICS:
        low_column, high_column = bootstrap_ci_columns(metric)
        low, high = session_block_bootstrap_ci(
            records,
            metric,
            samples=1000,
            seed=SESSION_BLOCK_SEED,
            epsilon=cell.epsilon,
        )
        row[low_column] = low
        row[high_column] = high
    row["_records"] = records
    return row


def summarize_cells(cells: list[PolicyCell]) -> list[dict[str, Any]]:
    """Summarize merged per-turn records into one row per cell x policy."""
    rows: list[dict[str, Any]] = []
    ordered_cells = sorted(cells, key=lambda cell: (cell.key, cell.config, cell.run_dir))
    for cell in ordered_cells:
        exact_profiles = {record.action_profile for record in cell.records if record.exact}
        summary = add_shadow_audit_metrics(
            MetricCollector().summarize_policy_runs(
                cell.records,
                epsilon=cell.epsilon,
                delta=cell.delta,
                exact_profiles=exact_profiles,
                experiment_type="baseline_session",
            ),
            cell.records,
        )
        for policy in sorted(summary):
            records = _records_for_policy(cell, policy)
            if not records:
                continue
            rows.append(_cell_row(cell, policy, records, summary))
    return rows


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and isnan(value):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return value
    return json_ready(value)


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = total_summary_columns()
    fieldname_set = set(fieldnames)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _csv_value(value) for key, value in row.items() if key in fieldname_set}
            )
    return output


def write_events_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write one backend-event row per cell x policy, derived from turn records."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENTS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            event_row = {
                "policy": row["policy"],
                "memory_budget_mib": _csv_value(row["memory_budget_mib"]),
                "epsilon": _csv_value(row["epsilon"]),
                "delta": _csv_value(row["delta"]),
                "count": _csv_value(row.get("count")),
                "ok_count": _csv_value(row.get("ok_count")),
                "session_count": _csv_value(row.get("session_count")),
                "budget_hit_count": _csv_value(row.get("budget_hit_count")),
                "budget_hit_rate": _csv_value(row.get("budget_hit_rate")),
                "restore_count": _csv_value(row.get("restore_count")),
                "restore_time_ms_total": _csv_value(row.get("restore_time_ms_total")),
                "recompute_count": _csv_value(row.get("recompute_count")),
                "recompute_time_ms_total": _csv_value(row.get("recompute_time_ms_total")),
                "queue_event_count": _csv_value(row.get("queue_event_count")),
                "queue_delay_ms_mean": _csv_value(row.get("queue_delay_ms")),
                "evict_event_count": _csv_value(row.get("evict_event_count")),
                "session_reuse_evidence": _csv_value(row.get("session_reuse_evidence")),
                "backend_event_evidence": _csv_value(row.get("backend_event_evidence")),
            }
            writer.writerow(event_row)
    return output


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if value is not None and not isnan(float(value))]


def _mean(values: list[float]) -> float:
    finite_values = _finite(values)
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(_finite(values))
    if not finite_values:
        return float("nan")
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]


def _session_points(cell: PolicyCell, policy: str, records: list[PolicyRunRecord]) -> list[dict[str, Any]]:
    by_session: dict[str, list[PolicyRunRecord]] = {}
    for record in records:
        session_id = record.session_id or record.request_id
        by_session.setdefault(session_id, []).append(record)
    points: list[dict[str, Any]] = []
    for session_id in sorted(by_session):
        session_records = by_session[session_id]
        ttfts = [record.ttft_ms for record in session_records]
        losses = [record.quality_loss for record in session_records]
        kv_memories = [record.kv_cache_memory_mib for record in session_records]
        finite_losses = _finite(losses)
        points.append(
            {
                "policy": policy,
                "memory_budget_mib": cell.memory_budget_mib,
                "epsilon": cell.epsilon,
                "delta": cell.delta,
                "session_id": session_id,
                "count": float(len(session_records)),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "mean_ttft_ms": _mean(ttfts),
                "p95_quality_loss": _percentile(losses, 0.95),
                "mean_quality_loss": _mean(losses),
                "mean_kv_cache_memory_mib": _mean(kv_memories),
                "violation_rate": (
                    sum(1 for loss in finite_losses if loss > cell.epsilon) / len(finite_losses)
                    if finite_losses
                    else float("nan")
                ),
            }
        )
    return points


def write_session_points_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write per-session aggregate points for scatter overlays on summary charts."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SESSION_POINT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            records: list[PolicyRunRecord] = row["_records"]
            cell = PolicyCell(
                memory_budget_mib=float(row["memory_budget_mib"]),
                epsilon=float(row["epsilon"]),
                delta=float(row["delta"]),
            )
            for point in _session_points(cell, str(row["policy"]), records):
                writer.writerow({key: _csv_value(value) for key, value in point.items()})
    return output


def _display_number(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if isnan(number):
        return ""
    return f"{number:g}"


def _markdown_row(row: dict[str, Any], event: dict[str, Any] | None) -> dict[str, str]:
    event = event if event is not None else row
    return {
        "memory_budget_mib": _display_number(row.get("memory_budget_mib")),
        "policy": str(row.get("policy") or ""),
        "p95_ttft_ms": _display_number(row.get("p95_ttft_ms")),
        "mean_ttft_ms": _display_number(row.get("mean_ttft_ms")),
        "mean_kv_cache_memory_mib": _display_number(row.get("mean_kv_cache_memory_mib")),
        "budget_hit_rate": _display_number(event.get("budget_hit_rate")),
        "restore_count": _display_number(event.get("restore_count")),
        "recompute_count": _display_number(event.get("recompute_count")),
        "mean_quality_loss": _display_number(row.get("mean_quality_loss")),
        "quality_status": str(row.get("quality_status") or ""),
    }


MARKDOWN_COLUMNS = [
    "memory_budget_mib",
    "policy",
    "p95_ttft_ms",
    "mean_ttft_ms",
    "mean_kv_cache_memory_mib",
    "budget_hit_rate",
    "restore_count",
    "recompute_count",
    "mean_quality_loss",
    "quality_status",
]


def _key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row.get("memory_budget_mib") or 0.0),
        float(row.get("epsilon") or 0.0),
        float(row.get("delta") or 0.0),
        str(row.get("policy") or ""),
    )


def write_baseline_smoke_markdown(
    rows: list[dict[str, Any]],
    events: Iterable[dict[str, Any]] | None,
    path: str | Path,
) -> Path:
    """Write the baseline smoke markdown table for the session27 run."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    event_by_key: dict[tuple[float, float, float, str], dict[str, Any]] = {
        _key(event): event for event in (events or rows)
    }
    ordered_rows = sorted(rows, key=_key)
    lines = [
        "# Session27 online baseline smoke（diagnostic_only）",
        "",
        "| memory_budget_mib | policy | p95_ttft_ms | mean_ttft_ms | mean_kv_cache_memory_mib | budget_hit_rate | restore_count | recompute_count | mean_quality_loss | quality_status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ordered_rows:
        rendered = _markdown_row(row, event_by_key.get(_key(row)))
        lines.append(
            "| " + " | ".join(str(rendered[column]) for column in MARKDOWN_COLUMNS) + " |"
        )
    lines.extend(
        [
            "",
            "> 全部输出为 `diagnostic_only=true`；quality/violation 均为 "
            "`risk_evidence_insufficient`，不作为 tail SLO 结论。",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
