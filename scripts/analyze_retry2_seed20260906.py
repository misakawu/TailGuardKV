#!/usr/bin/env python3
"""Create fail-closed diagnostic analysis artifacts for one retry2 policy seed."""
from __future__ import annotations

import csv
import json
import argparse
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt

from run_util.session_aggregation import (
    aggregate_policy_csvs,
    summarize_cells,
    write_events_csv,
    write_session_points_csv,
    write_summary_csv,
)
from visual.plot_summary import plot_summary


EXPECTED_POLICIES = {
    "full_lru",
    "static_best",
    "static_safe",
    "utility_dynamic",
    "uncalibrated_dynamic",
}
EXPECTED_CELLS = 16
EXPECTED_REQUESTS_PER_POLICY = 65


def _is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_csv(path: Path) -> dict[str, Any]:
    rows = _read_rows(path)
    if not rows:
        raise ValueError(f"empty policy CSV: {path}")
    policies = {row.get("policy", "") for row in rows}
    if policies != EXPECTED_POLICIES:
        raise ValueError(f"unexpected policy set in {path}: {sorted(policies)}")
    for policy in sorted(EXPECTED_POLICIES):
        policy_rows = [row for row in rows if row.get("policy") == policy]
        request_ids = [row.get("request_id", "") for row in policy_rows]
        if len(request_ids) != EXPECTED_REQUESTS_PER_POLICY or len(set(request_ids)) != EXPECTED_REQUESTS_PER_POLICY:
            raise ValueError(f"{path}: {policy} must contain 65 unique evaluation requests")
    for row in rows:
        if not _is_true(row.get("diagnostic_only")):
            raise ValueError(f"{path}: diagnostic_only=true is required")
        if row.get("backend_name") != "online_qwen":
            raise ValueError(f"{path}: backend_name=online_qwen is required")
        if not _is_true(row.get("ok")):
            raise ValueError(f"{path}: ok=true is required")
    return {"path": str(path), "records": len(rows), "policies": sorted(policies)}


def audit_seed_attempt(attempt_root: Path, seed: str) -> tuple[list[Path], dict[str, Any]]:
    """Return audited CSV paths and a reproducibility manifest for one complete seed."""
    status_path = attempt_root / "fourth_grid_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not _is_true(status.get("diagnostic_only")):
        raise ValueError("attempt status must set diagnostic_only=true")
    cells = [cell for cell in status.get("cells", []) if str(cell.get("seed")) == seed]
    if len(cells) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} cells for seed {seed}, found {len(cells)}")
    if any(cell.get("state") != "success" for cell in cells):
        raise ValueError(f"all cells for seed {seed} must be success")

    paths: list[Path] = []
    audits: list[dict[str, Any]] = []
    for cell in cells:
        if not _is_true(cell.get("diagnostic_only")):
            raise ValueError("cell must set diagnostic_only=true")
        if int(cell.get("expected_requests_per_policy", -1)) != EXPECTED_REQUESTS_PER_POLICY:
            raise ValueError("cell must expect 65 evaluation requests per policy")
        path = Path(str(cell.get("output", "")))
        if not path.is_file():
            raise FileNotFoundError(f"missing policy CSV: {path}")
        paths.append(path)
        audits.append(_validate_csv(path))

    return paths, {
        "ok": True,
        "diagnostic_only": True,
        "seed": seed,
        "cell_count": len(cells),
        "raw_record_count": sum(audit["records"] for audit in audits),
        "requests_per_policy": EXPECTED_REQUESTS_PER_POLICY,
        "cells": audits,
    }


def _write_extra_charts(rows: list[dict[str, Any]], output_root: Path) -> list[Path]:
    action_counts: dict[str, dict[str, int]] = {}
    event_rates: dict[str, list[float]] = {}
    for row in rows:
        policy = str(row["policy"])
        actions = row.get("action_distribution", {})
        action_counts.setdefault(policy, {})
        for action, count in actions.items():
            action_counts[policy][str(action)] = action_counts[policy].get(str(action), 0) + int(count)
        event_rates.setdefault(policy, [0.0, 0.0, 0.0, 0.0])
        event_rates[policy][0] += float(row.get("budget_hit_rate", 0.0))
        event_rates[policy][1] += float(row.get("policy_budget_filter_rate", 0.0))
        event_rates[policy][2] += float(row.get("restore_rate", 0.0))
        event_rates[policy][3] += float(row.get("recompute_rate", 0.0))

    action_path = output_root / "action_distribution.png"
    policies = sorted(action_counts)
    actions = sorted({action for counts in action_counts.values() for action in counts})
    figure, axis = plt.subplots(figsize=(9, 4.8))
    bottoms = [0] * len(policies)
    for action in actions:
        values = [action_counts[policy].get(action, 0) for policy in policies]
        axis.bar(policies, values, bottom=bottoms, label=action)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    axis.set_title("Action distribution (diagnostic_only=true)")
    axis.set_ylabel("Request count across 16 cells")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(action_path, dpi=160)
    plt.close(figure)

    event_path = output_root / "event_rates.png"
    labels = ["budget hit", "policy filter", "restore", "recompute"]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / max(1, len(policies))
    for index, policy in enumerate(policies):
        values = [value / 16.0 for value in event_rates[policy]]
        positions = [position - 0.4 + width / 2 + index * width for position in range(len(labels))]
        axis.bar(positions, values, width=width, label=policy)
    axis.set_title("Mean policy event rates (diagnostic_only=true)")
    axis.set_ylabel("Rate")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_ylim(0.0, 1.0)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(event_path, dpi=160)
    plt.close(figure)
    return [action_path, event_path]


def _write_markdown(rows: list[dict[str, Any]], audit: dict[str, Any], path: Path) -> None:
    policies = sorted({str(row["policy"]) for row in rows})
    lines = [
        "# retry2 seed20260906 policy analysis",
        "",
        "`diagnostic_only=true`。本报告仅覆盖 retry2 的单个 seed 和 16 个成功 cell，不能用作正式 baseline、TailGuard 或论文结论。",
        "",
        "## Coverage",
        "",
        f"- Seed: `{audit['seed']}`",
        f"- Successful cells: `{audit['cell_count']}`",
        f"- Raw records: `{audit['raw_record_count']}`",
        f"- Summary rows: `{len(rows)}`",
        f"- Policies: `{', '.join(policies)}`",
        "",
        "## Reading the outputs",
        "",
        "- `seed20260906_total_summary.csv` is the recomputable per-cell policy table; it retains epsilon, delta, and memory budget provenance.",
        "- `seed20260906_events.csv` and `seed20260906_session_points.csv` retain event and session-level evidence.",
        "- PNG curves are separated by epsilon/delta panels. They only contain accepted cells; no failed cell is converted to zero or connected into a line.",
        "- Quality and violation fields remain `risk_evidence_insufficient`; interpret TTFT, KV, budget, events, and actions as runtime diagnostics only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_analysis(attempt_root: Path, output_root: Path, seed: str) -> dict[str, Any]:
    """Audit one seed then emit reproducible summaries, plots, and a constrained report."""
    paths, audit = audit_seed_attempt(attempt_root, seed)
    output_root.mkdir(parents=True, exist_ok=True)
    cells = aggregate_policy_csvs(paths)
    rows = summarize_cells(cells)
    summary_path = write_summary_csv(rows, output_root / f"seed{seed}_total_summary.csv")
    events_path = write_events_csv(rows, output_root / f"seed{seed}_events.csv")
    points_path = write_session_points_csv(rows, output_root / f"seed{seed}_session_points.csv")
    standard_charts = plot_summary(summary_path, output_root, session_points_csv=points_path, always_emit_all=True)
    supplementary_charts = _write_extra_charts(rows, output_root)
    audit["policy_rows"] = len(rows)
    audit["summary_csv"] = str(summary_path)
    audit["events_csv"] = str(events_path)
    audit["session_points_csv"] = str(points_path)
    audit["plot_outputs"] = [str(path) for path in [*standard_charts, *supplementary_charts]]
    (output_root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_root / "baseline_smoke.md"
    _write_markdown(rows, audit, markdown_path)
    return {**audit, "baseline_smoke_markdown": str(markdown_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one retry2 policy seed with fail-closed provenance checks.")
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", default="20260906")
    args = parser.parse_args()
    try:
        print(json.dumps(generate_analysis(args.attempt_root, args.output_root, args.seed), ensure_ascii=False, indent=2))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "diagnostic_only": True, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
