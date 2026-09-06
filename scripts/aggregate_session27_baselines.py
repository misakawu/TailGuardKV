#!/usr/bin/env python3
"""Aggregate session27 online baseline policy CSVs into total summary + plots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_util.experiment_common import json_ready
from run_util.session_aggregation import (
    aggregate_policy_csvs,
    summarize_cells,
    write_baseline_smoke_markdown,
    write_events_csv,
    write_session_points_csv,
    write_summary_csv,
)
from visual.plot_summary import plot_summary


TOTAL_SUMMARY_NAME = "session27_total_summary.csv"
EVENTS_NAME = "session27_events.csv"
SESSION_POINTS_NAME = "session27_session_points.csv"
SMOKE_MARKDOWN_NAME = "baseline_smoke.md"


def aggregate_session27(input_dirs: list[str], output_root: str) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for raw_dir in input_dirs:
        directory = Path(raw_dir)
        if not directory.exists():
            raise FileNotFoundError(f"policy CSV 目录不存在: {directory}")
        paths.extend(sorted(directory.glob("*_policy_*.csv")))
    if not paths:
        raise FileNotFoundError(f"没有找到 *_policy_*.csv: {input_dirs}")

    cells = aggregate_policy_csvs(paths)
    rows = summarize_cells(cells)
    if not rows:
        raise ValueError("汇总后没有 policy 行")

    summary_path = write_summary_csv(rows, root / TOTAL_SUMMARY_NAME)
    events_path = write_events_csv(rows, root / EVENTS_NAME)
    points_path = write_session_points_csv(rows, root / SESSION_POINTS_NAME)
    plots = plot_summary(summary_path, root, session_points_csv=points_path)
    markdown_path = write_baseline_smoke_markdown(rows, rows, root / SMOKE_MARKDOWN_NAME)
    return {
        "ok": True,
        "diagnostic_only": True,
        "quality_status": "risk_evidence_insufficient",
        "violation_status": "risk_evidence_insufficient",
        "input_dirs": [str(directory) for directory in input_dirs],
        "cells": len(cells),
        "policy_rows": len(rows),
        "summary_csv": str(summary_path),
        "events_csv": str(events_path),
        "session_points_csv": str(points_path),
        "baseline_smoke_markdown": str(markdown_path),
        "plot_outputs": [str(path) for path in plots],
        "plot_count": len(plots),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="聚合 session27 online baseline policy CSV 并生成汇总与图表。")
    parser.add_argument("--input-dir", required=True, nargs="+")
    parser.add_argument("--output-root", default="out/session27_online/policy_tables")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = aggregate_session27(args.input_dir, args.output_root)
    except Exception as exc:
        print(
            json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "diagnostic_only": True},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
