from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics import MetricCollector
from run_util.core_types import PolicyRunRecord
from run_util.experiment_common import json_ready
from run_util.experiment_summary import TOTAL_POLICY_SUMMARY_COLUMNS
from visual.plot_summary import plot_summary


FILENAME_RE = re.compile(
    r"pilot_smoke_measured_policy_eps(?P<epsilon>[0-9mp]+)_delta(?P<delta>[0-9mp]+)_mem(?P<memory>[0-9mp]+)\.csv$"
)


def _parse_number(token: str) -> float:
    return float(token.replace("p", ".").replace("m", "-"))


def parse_sweep_filename(filename: str) -> dict[str, float]:
    match = FILENAME_RE.search(Path(filename).name)
    if not match:
        raise ValueError(f"无法从文件名解析 sweep 参数: {filename}")
    return {
        "epsilon": _parse_number(match.group("epsilon")),
        "delta": _parse_number(match.group("delta")),
        "memory_budget_mib": _parse_number(match.group("memory")),
    }


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def _record_from_row(row: dict[str, str]) -> PolicyRunRecord:
    return PolicyRunRecord(
        policy=row["policy"],
        request_id=row["request_id"],
        action_profile=row["action_profile"],
        ok=_parse_bool(row.get("ok")),
        measured=_parse_bool(row.get("measured")),
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
        quality_loss=_parse_float(row.get("quality_loss")),
        exact=_parse_bool(row.get("exact")),
        oracle=_parse_bool(row.get("oracle")),
        pred_loss=_parse_float(row.get("pred_loss")),
        risk_upper=_parse_float(row.get("risk_upper")),
        safe=None if row.get("safe", "").strip() == "" else _parse_bool(row.get("safe")),
        epsilon=_parse_float(row.get("epsilon")),
        delta=_parse_float(row.get("delta")),
        fallback_reason=row.get("fallback_reason") or "",
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
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return json_ready(value)


def _policy_csvs(policy_dir: Path) -> list[Path]:
    return sorted(path for path in policy_dir.glob("*.csv") if FILENAME_RE.search(path.name))


def aggregate_directory(policy_dir: str | Path, output_csv: str | Path) -> tuple[Path, list[Path]]:
    source_dir = Path(policy_dir)
    csv_paths = _policy_csvs(source_dir)
    if not csv_paths:
        raise FileNotFoundError(f"未找到可聚合的 policy CSV: {source_dir}")

    rows: list[dict[str, Any]] = []
    collector = MetricCollector()
    for csv_path in csv_paths:
        sweep = parse_sweep_filename(csv_path.name)
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            records = [_record_from_row(row) for row in csv.DictReader(handle)]
        exact_profiles = {record.action_profile for record in records if record.exact}
        summary = collector.summarize_policy_runs(
            records,
            epsilon=sweep["epsilon"],
            delta=sweep["delta"],
            exact_profiles=exact_profiles,
        )
        for policy, metrics in summary.items():
            row = {column: "" for column in TOTAL_POLICY_SUMMARY_COLUMNS}
            row.update(
                {
                    "config": "configs/baseline_wide_sweep.yaml",
                    "run_dir": str(source_dir.parent),
                    "policy": policy,
                    "memory_budget_mib": sweep["memory_budget_mib"],
                    "epsilon": sweep["epsilon"],
                    "delta": sweep["delta"],
                }
            )
            for column in TOTAL_POLICY_SUMMARY_COLUMNS:
                if column in metrics:
                    row[column] = metrics[column]
            rows.append(row)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOTAL_POLICY_SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)
    plots = plot_summary(output_path, output_path.parent)
    return output_path, plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="聚合 baseline wide sweep policy CSV 并生成图表。")
    parser.add_argument("--input-dir", default="out/baseline_wide_sweep/policy_tables")
    parser.add_argument(
        "--output",
        default="out/baseline_wide_sweep/policy_tables/baseline_wide_sweep_total_summary.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary_path, plots = aggregate_directory(args.input_dir, args.output)
    payload = {
        "summary_csv": str(summary_path),
        "plot_outputs": [str(path) for path in plots],
        "plot_count": len(plots),
    }
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
