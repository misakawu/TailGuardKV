from __future__ import annotations

import csv
from pathlib import Path

from metrics import MetricCollector
from run_util.core_types import PolicyRunRecord
from visual.plot_summary import plot_summary


def test_metric_collector_reports_session_lifecycle_metrics() -> None:
    records = [
        PolicyRunRecord(
            policy="tailguard",
            request_id="s1_t0",
            session_id="s1",
            turn_index=0,
            action_profile="kivi_4bit_residual32",
            ok=True,
            measured=True,
            restore_ms=0.0,
            recompute_ms=0.0,
            resident_kv_mib_after=20.0,
            kv_cumulative_mib=20.0,
            budget_hit=False,
        ),
        PolicyRunRecord(
            policy="tailguard",
            request_id="s1_t1",
            session_id="s1",
            turn_index=1,
            action_profile="full_cpu",
            ok=True,
            measured=True,
            restore_ms=3.0,
            recompute_ms=7.0,
            resident_kv_mib_before=20.0,
            resident_kv_mib_after=40.0,
            kv_cumulative_mib=40.0,
            budget_hit=True,
        ),
    ]

    summary = MetricCollector().summarize_policy_runs(records, epsilon=0.05, delta=0.05, exact_profiles={"full_cpu"})

    assert "switch_count" in summary["tailguard"]
    assert "restore_time_ms" in summary["tailguard"]
    assert "profile_residence_share" in summary["tailguard"]


def test_plot_summary_creates_subplot_grid_by_constraint(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "name", "epsilon", "delta", "memory_budget_mib", "mean_ttft_ms", "mean_peak_memory_mib", "mean_quality_loss", "violation_rate", "fallback_ratio"],
        )
        writer.writeheader()
        writer.writerow({"section": "policy", "name": "tailguard", "epsilon": "0.05", "delta": "0.05", "memory_budget_mib": "64", "mean_ttft_ms": "10", "mean_peak_memory_mib": "20", "mean_quality_loss": "0.01", "violation_rate": "0.0", "fallback_ratio": "0.2"})
        writer.writerow({"section": "policy", "name": "tailguard", "epsilon": "0.10", "delta": "0.05", "memory_budget_mib": "64", "mean_ttft_ms": "9", "mean_peak_memory_mib": "18", "mean_quality_loss": "0.02", "violation_rate": "0.0", "fallback_ratio": "0.1"})

    outputs = plot_summary(summary_csv)

    assert outputs
