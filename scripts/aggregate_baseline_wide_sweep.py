from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metrics import MetricCollector
from run_util.experiment_common import json_ready
from run_util.experiment_summary import TOTAL_POLICY_SUMMARY_COLUMNS, add_shadow_audit_metrics
from run_util.session_aggregation import _record_from_row, parse_sweep_filename
from visual.plot_summary import plot_summary


def _parse_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _source_provenance(rows: list[dict[str, str]], source_dir: Path) -> dict[str, Any]:
    if not rows:
        raise ValueError("policy CSV has no rows")
    diagnostic_columns = ["diagnostic_only", "quality_status", "violation_status"]
    has_provenance = [any(row.get(column, "").strip() for column in diagnostic_columns) for row in rows]
    if any(has_provenance) and not all(has_provenance):
        raise ValueError("policy CSV diagnostic provenance must be present on every row")
    diagnostic_only = _consistent_bool(rows, "diagnostic_only", default=False)
    quality_status = _consistent_text(rows, "quality_status", default="")
    violation_status = _consistent_text(rows, "violation_status", default="")
    if diagnostic_only and {quality_status, violation_status} != {"risk_evidence_insufficient"}:
        raise ValueError("diagnostic policy CSV must retain risk_evidence_insufficient quality and violation statuses")
    config = _consistent_text(rows, "config", default="")
    run_dir = _consistent_text(rows, "run_dir", default="")
    if diagnostic_only and (not config or not run_dir):
        raise ValueError("diagnostic policy CSV must carry non-empty config and run_dir provenance")
    return {
        "diagnostic_only": diagnostic_only,
        "quality_status": quality_status,
        "violation_status": violation_status,
        "config": config or "configs/baseline_wide_sweep.yaml",
        "run_dir": run_dir or str(source_dir.parent),
    }


def _consistent_text(rows: list[dict[str, str]], key: str, *, default: str) -> str:
    values = {str(row.get(key) or "").strip() for row in rows}
    values.discard("")
    if len(values) > 1:
        raise ValueError(f"policy CSV has mixed {key} values")
    return next(iter(values), default)


def _consistent_bool(rows: list[dict[str, str]], key: str, *, default: bool) -> bool:
    values = {_parse_bool(row.get(key)) for row in rows if str(row.get(key) or "").strip()}
    if len(values) > 1:
        raise ValueError(f"policy CSV has mixed {key} values")
    return next(iter(values), default)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return json_ready(value)


def _is_sweep_csv(name: str) -> bool:
    try:
        parse_sweep_filename(name)
    except ValueError:
        return False
    return True


def _policy_csvs(policy_dir: Path) -> list[Path]:
    return sorted(path for path in policy_dir.glob("*.csv") if _is_sweep_csv(path.name))


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
            source_rows = list(csv.DictReader(handle))
        records = [_record_from_row(row) for row in source_rows]
        provenance = _source_provenance(source_rows, source_dir)
        exact_profiles = {record.action_profile for record in records if record.exact}
        summary = add_shadow_audit_metrics(collector.summarize_policy_runs(
            records,
            epsilon=sweep["epsilon"],
            delta=sweep["delta"],
            exact_profiles=exact_profiles,
        ), records)
        for policy, metrics in summary.items():
            row = {column: "" for column in TOTAL_POLICY_SUMMARY_COLUMNS}
            row.update(
                {
                    **provenance,
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

    provenance_keys = {(row["config"], row["run_dir"]) for row in rows}
    if len(provenance_keys) > 1:
        raise ValueError(f"policy CSV directory mixes provenance across runs: {sorted(provenance_keys)}")

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
