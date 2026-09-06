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


def test_metric_collector_reports_server_side_replay_metrics() -> None:
    records = [
        PolicyRunRecord(
            policy="tailguard",
            request_id="s1_t0",
            session_id="s1",
            turn_index=0,
            action_profile="kivi_4bit_residual32",
            ok=True,
            measured=True,
            ttft_ms=10.0,
            kv_cache_memory_mib=20.0,
            restore_ms=0.0,
            recompute_ms=0.0,
            resident_kv_mib_after=20.0,
            budget_hit=False,
            backend_budget_hit=False,
            policy_budget_filtered=True,
            backend_name="measured_replay",
            reason="safe",
        ),
        PolicyRunRecord(
            policy="tailguard",
            request_id="s2_t0",
            session_id="s2",
            turn_index=0,
            action_profile="kivi_4bit_residual32",
            ok=True,
            measured=True,
            ttft_ms=18.0,
            kv_cache_memory_mib=20.0,
            restore_ms=6.0,
            recompute_ms=0.0,
            evicted_kv_mib=12.0,
            resident_kv_mib_after=18.0,
            budget_hit=True,
            backend_budget_hit=True,
            policy_budget_filtered=False,
            backend_name="measured_replay",
            reason="safe",
        ),
        PolicyRunRecord(
            policy="tailguard",
            request_id="s1_t1",
            session_id="s1",
            turn_index=1,
            action_profile="full_cpu",
            ok=True,
            measured=True,
            ttft_ms=30.0,
            kv_cache_memory_mib=45.0,
            restore_ms=0.0,
            recompute_ms=9.0,
            evicted_kv_mib=0.0,
            resident_kv_mib_after=45.0,
            budget_hit=True,
            backend_budget_hit=True,
            policy_budget_filtered=False,
            backend_name="measured_replay",
            reason="fallback",
        ),
    ]

    summary = MetricCollector().summarize_policy_runs(records, epsilon=0.05, delta=0.05, exact_profiles={"full_cpu"})

    assert "mean_global_resident_kv_mib" in summary["tailguard"]
    assert "p95_global_resident_kv_mib" in summary["tailguard"]
    assert "queue_delay_ms" in summary["tailguard"]
    assert summary["tailguard"]["budget_hit_rate"] > 0.0
    assert summary["tailguard"]["policy_budget_filter_rate"] > 0.0


def test_plot_summary_creates_subplot_grid_by_constraint(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "name", "epsilon", "delta", "memory_budget_mib", "p95_ttft_ms", "mean_kv_cache_memory_mib", "p95_quality_loss", "violation_rate", "budget_hit_rate"],
        )
        writer.writeheader()
        writer.writerow({"section": "policy", "name": "tailguard", "epsilon": "0.05", "delta": "0.05", "memory_budget_mib": "64", "p95_ttft_ms": "10", "mean_kv_cache_memory_mib": "20", "p95_quality_loss": "0.01", "violation_rate": "0.0", "budget_hit_rate": "0.2"})
        writer.writerow({"section": "policy", "name": "tailguard", "epsilon": "0.10", "delta": "0.05", "memory_budget_mib": "64", "p95_ttft_ms": "9", "mean_kv_cache_memory_mib": "18", "p95_quality_loss": "0.02", "violation_rate": "0.0", "budget_hit_rate": "0.1"})

    outputs = plot_summary(summary_csv)

    assert outputs


def _ci_row(
    *,
    policy: str,
    memory: str,
    ttft: str,
    ci_low: str,
    ci_high: str,
    kv_mib: str,
    quality: str,
    violation: str,
    quality_status: str = "risk_evidence_insufficient",
) -> dict[str, str]:
    return {
        "section": "policy",
        "name": policy,
        "policy": policy,
        "epsilon": "0.05",
        "delta": "0.05",
        "memory_budget_mib": memory,
        "quality_status": quality_status,
        "violation_status": "risk_evidence_insufficient",
        "p95_ttft_ms": ttft,
        "p95_ttft_ci_low_ms": ci_low,
        "p95_ttft_ci_high_ms": ci_high,
        "mean_kv_cache_memory_mib": kv_mib,
        "p95_quality_loss": quality,
        "p95_quality_loss_ci_low": "0.0",
        "p95_quality_loss_ci_high": "0.05",
        "violation_rate": violation,
        "violation_rate_ci_low": "0.0",
        "violation_rate_ci_high": "0.5",
    }


def test_plot_summary_with_ci_and_session_scatter_keeps_chart_names(tmp_path: Path) -> None:
    from visual.plot_summary import POLICY_CHARTS, _policy_session_values, _read_rows, plot_summary

    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_ci_row(policy="full_lru", memory="1", ttft="1", ci_low="0.9", ci_high="1.1", kv_mib="2", quality="0.01", violation="0.0").keys()))
        writer.writeheader()
        writer.writerow(_ci_row(policy="full_lru", memory="1", ttft="1", ci_low="0.9", ci_high="1.1", kv_mib="2", quality="0.01", violation="0.0"))
        writer.writerow(_ci_row(policy="full_lru", memory="2", ttft="2", ci_low="1.8", ci_high="2.2", kv_mib="3", quality="0.02", violation="0.0"))
        writer.writerow(_ci_row(policy="static_best", memory="1", ttft="0.8", ci_low="0.7", ci_high="0.9", kv_mib="1", quality="0.03", violation="0.2"))
        writer.writerow(_ci_row(policy="static_best", memory="2", ttft="1.2", ci_low="1.0", ci_high="1.4", kv_mib="1.5", quality="0.04", violation="0.2"))

    session_csv = tmp_path / "session_points.csv"
    with session_csv.open("w", encoding="utf-8", newline="") as handle:
        session_fields = ["policy", "memory_budget_mib", "epsilon", "delta", "session_id", "p95_ttft_ms", "mean_kv_cache_memory_mib", "p95_quality_loss", "mean_quality_loss", "violation_rate"]
        writer = csv.DictWriter(handle, fieldnames=session_fields)
        writer.writeheader()
        writer.writerow({"policy": "full_lru", "memory_budget_mib": "1", "epsilon": "0.05", "delta": "0.05", "session_id": "s1", "p95_ttft_ms": "1.1", "mean_kv_cache_memory_mib": "2", "p95_quality_loss": "0.01", "mean_quality_loss": "0.01", "violation_rate": "0.0"})
        writer.writerow({"policy": "static_best", "memory_budget_mib": "1", "epsilon": "0.05", "delta": "0.05", "session_id": "s2", "p95_ttft_ms": "0.75", "mean_kv_cache_memory_mib": "1", "p95_quality_loss": "0.03", "mean_quality_loss": "0.03", "violation_rate": "0.0"})

    outputs = plot_summary(summary_csv, tmp_path, session_points_csv=session_csv)

    expected_names = {filename for _, filename, _, _ in POLICY_CHARTS}
    assert {path.name for path in outputs} == expected_names
    assert all(path.exists() for path in outputs)

    session_rows = _read_rows(session_csv)
    assert _policy_session_values(session_rows, "p95_ttft_ms", ("0.05", "0.05"))[("full_lru", "B=1\ne=0.05\nd=0.05")] == [1.1]


def test_plot_summary_without_ci_or_scatter_still_works(tmp_path: Path) -> None:
    summary_csv = tmp_path / "legacy_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "policy", "epsilon", "delta", "memory_budget_mib", "p95_ttft_ms", "mean_kv_cache_memory_mib", "p95_quality_loss", "violation_rate"])
        writer.writeheader()
        writer.writerow({"section": "policy", "name": "tailguard", "policy": "tailguard", "epsilon": "0.05", "delta": "0.05", "memory_budget_mib": "64", "p95_ttft_ms": "10", "mean_kv_cache_memory_mib": "20", "p95_quality_loss": "0.01", "violation_rate": "0.0"})
        writer.writerow({"section": "policy", "name": "tailguard", "policy": "tailguard", "epsilon": "0.10", "delta": "0.05", "memory_budget_mib": "64", "p95_ttft_ms": "9", "mean_kv_cache_memory_mib": "18", "p95_quality_loss": "0.02", "violation_rate": "0.0"})

    outputs = plot_summary(summary_csv, tmp_path)

    from visual.plot_summary import POLICY_CHARTS
    assert {path.name for path in outputs} == {filename for _, filename, _, _ in POLICY_CHARTS}


def test_risk_label_detection_for_quality_rows() -> None:
    from visual.plot_summary import _is_diagnostic_risk

    assert _is_diagnostic_risk({"quality_status": "risk_evidence_insufficient"}) is True
    assert _is_diagnostic_risk({"quality_status": "validated"}) is False
    assert _is_diagnostic_risk({"quality_status": ""}) is False
