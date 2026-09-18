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


def audit_diagnostic_batches(
    input_root: str | Path,
    *,
    expected_batches: int,
    expected_sessions: int,
    expected_requests: int,
    require_diagnostic_only: bool = True,
) -> dict[str, Any]:
    """Validate a diagnostic batch manifest and its local fixtures."""
    root = Path(input_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing batch manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise ValueError("batch manifest must contain a batches list")

    expected_ids = [f"batch{index:03d}" for index in range(expected_batches)]
    actual_ids = [batch.get("batch_id") for batch in batches]
    if actual_ids != expected_ids:
        raise ValueError(
            f"batch ids mismatch: expected {expected_ids}, found {actual_ids}"
        )

    diagnostic_only = manifest.get("diagnostic_only") is True
    if require_diagnostic_only and not diagnostic_only:
        raise ValueError("diagnostic_only provenance is required")

    session_ids: set[str] = set()
    request_ids: set[str] = set()
    batch_audits: list[dict[str, Any]] = []
    for batch in batches:
        batch_id = str(batch["batch_id"])
        fixture_path = root / "fixtures" / f"{batch_id}.jsonl"
        run_dir = root / "batch_outputs" / batch_id
        if not fixture_path.is_file():
            raise ValueError(f"missing batch fixture: {fixture_path}")
        if not run_dir.is_dir():
            raise ValueError(f"missing batch output directory: {run_dir}")

        rows = [
            json.loads(line)
            for line in fixture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        batch_sessions = {str(row.get("session_id", "")) for row in rows}
        batch_requests = [str(row.get("request_id", "")) for row in rows]
        if "" in batch_sessions or "" in batch_requests:
            raise ValueError(f"batch {batch_id} fixture has missing identifiers")
        if len(set(batch_requests)) != len(batch_requests):
            raise ValueError(f"duplicate request id within batch {batch_id}")
        duplicate_requests = request_ids.intersection(batch_requests)
        if duplicate_requests:
            raise ValueError(
                f"duplicate request id across batches: {sorted(duplicate_requests)}"
            )
        duplicate_sessions = session_ids.intersection(batch_sessions)
        if duplicate_sessions:
            raise ValueError(
                f"duplicate session id across batches: {sorted(duplicate_sessions)}"
            )

        manifest_sessions = int(batch.get("sessions", -1))
        manifest_requests = int(batch.get("requests", -1))
        if manifest_sessions != len(batch_sessions):
            raise ValueError(
                f"batch {batch_id} session count mismatch: "
                f"manifest={manifest_sessions}, fixture={len(batch_sessions)}"
            )
        if manifest_requests != len(rows):
            raise ValueError(
                f"batch {batch_id} request count mismatch: "
                f"manifest={manifest_requests}, fixture={len(rows)}"
            )

        session_ids.update(batch_sessions)
        request_ids.update(batch_requests)
        batch_audits.append(
            {
                "batch_id": batch_id,
                "fixture": str(fixture_path),
                "run_dir": str(run_dir),
                "sessions": len(batch_sessions),
                "requests": len(rows),
            }
        )

    if len(session_ids) != expected_sessions:
        raise ValueError(
            f"session count mismatch: expected {expected_sessions}, found {len(session_ids)}"
        )
    if len(request_ids) != expected_requests:
        raise ValueError(
            f"request count mismatch: expected {expected_requests}, found {len(request_ids)}"
        )

    return {
        "status": "passed",
        "diagnostic_only": diagnostic_only,
        "batch_count": len(batches),
        "session_count": len(session_ids),
        "request_count": len(request_ids),
        "batches": batch_audits,
    }


def aggregate_session27(
    input_dirs: list[str],
    output_root: str,
    *,
    name: str = "session27",
    input_paths: list[Path] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if input_paths is not None:
        paths = [Path(path) for path in input_paths]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"policy CSV 不存在: {path}")
    else:
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

    summary_path = write_summary_csv(rows, root / f"{name}_total_summary.csv")
    events_path = write_events_csv(rows, root / f"{name}_events.csv")
    points_path = write_session_points_csv(rows, root / f"{name}_session_points.csv")
    plots = plot_summary(summary_path, root, session_points_csv=points_path, always_emit_all=True)
    markdown_path = write_baseline_smoke_markdown(
        rows, rows, root / SMOKE_MARKDOWN_NAME,
        title=f"{name.replace('_', ' ').title()} online baseline smoke",
    )
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
