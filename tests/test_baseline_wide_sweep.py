from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_util.core_types import PolicyRunRecord
from run_util.experiment_common import load_config, write_csv
from run_util.run_policies import _policy_rows_with_provenance

from scripts.aggregate_baseline_wide_sweep import aggregate_directory, parse_sweep_filename
from scripts.baseline_wide_sweep_grid import load_sweep_grid


def _record(
    policy: str,
    request_id: str,
    profile: str,
    *,
    ttft_ms: float,
    peak_memory_mib: float,
    kv_cache_memory_mib: float,
    quality_loss: float,
    fallback_reason: str = "",
    safe: bool | None = None,
) -> PolicyRunRecord:
    return PolicyRunRecord(
        policy=policy,
        request_id=request_id,
        action_profile=profile,
        ok=True,
        measured=True,
        task="summary",
        length_bucket="short",
        ttft_ms=ttft_ms,
        latency_ms=ttft_ms + 1.0,
        peak_memory_mib=peak_memory_mib,
        kv_cache_memory_mib=kv_cache_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=quality_loss,
        exact=profile == "full_gpu",
        fallback_reason=fallback_reason,
        safe=safe,
    )


class BaselineWideSweepTest(unittest.TestCase):
    def test_wide_sweep_config_and_runtime_grid_match_requested_range(self) -> None:
        config = load_config(Path("configs/baseline_wide_sweep.yaml"))
        grid = load_sweep_grid("configs/baseline_wide_sweep.yaml")

        self.assertEqual(config["pilot"]["memory_budgets_mib"], [18, 22, 26, 30, 35, 40, 50, 60, 75])
        self.assertEqual(config["pilot"]["epsilons"], [0.02, 0.05, 0.10])
        self.assertEqual(config["pilot"]["deltas"], [0.01, 0.05, 0.10])
        self.assertEqual(grid["memory_budgets_mib"], [18.0, 22.0, 26.0, 30.0, 35.0, 40.0, 50.0, 60.0, 75.0])
        self.assertEqual(grid["epsilons"], [0.02, 0.05, 0.1])
        self.assertEqual(grid["deltas"], [0.01, 0.05, 0.1])

    def test_parse_sweep_filename_extracts_constraint_cell(self) -> None:
        parsed = parse_sweep_filename("pilot_smoke_measured_policy_eps0p05_delta0p1_mem5000.csv")

        self.assertEqual(parsed["epsilon"], 0.05)
        self.assertEqual(parsed["delta"], 0.1)
        self.assertEqual(parsed["memory_budget_mib"], 5000.0)

    def test_aggregate_directory_writes_summary_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            policy_dir = root / "policy_tables"
            policy_dir.mkdir(parents=True)

            first_rows = [
                _record("full_lru", "r1", "full_gpu", ttft_ms=10.0, peak_memory_mib=100.0, kv_cache_memory_mib=70.0, quality_loss=0.0),
                _record("full_lru", "r2", "full_gpu", ttft_ms=12.0, peak_memory_mib=110.0, kv_cache_memory_mib=75.0, quality_loss=0.0),
                _record("static_best", "r1", "kivi_4bit_residual32", ttft_ms=8.0, peak_memory_mib=90.0, kv_cache_memory_mib=40.0, quality_loss=0.03, safe=False),
                _record("static_best", "r2", "kivi_4bit_residual32", ttft_ms=9.0, peak_memory_mib=95.0, kv_cache_memory_mib=42.0, quality_loss=0.07, safe=False),
                _record("static_safe", "r1", "full_gpu", ttft_ms=10.0, peak_memory_mib=100.0, kv_cache_memory_mib=70.0, quality_loss=0.0, fallback_reason="safe_fallback", safe=True),
                _record("static_safe", "r2", "full_gpu", ttft_ms=12.0, peak_memory_mib=110.0, kv_cache_memory_mib=75.0, quality_loss=0.0, fallback_reason="safe_fallback", safe=True),
                _record("utility_dynamic", "r1", "kivi_4bit_residual32", ttft_ms=7.5, peak_memory_mib=88.0, kv_cache_memory_mib=38.0, quality_loss=0.01, safe=True),
                _record("utility_dynamic", "r2", "kivi_4bit_residual32", ttft_ms=8.5, peak_memory_mib=90.0, kv_cache_memory_mib=39.0, quality_loss=0.04, safe=True),
                _record("uncalibrated_dynamic", "r1", "kivi_2bit_residual32", ttft_ms=7.0, peak_memory_mib=85.0, kv_cache_memory_mib=30.0, quality_loss=0.02, safe=False),
                _record("uncalibrated_dynamic", "r2", "kivi_2bit_residual32", ttft_ms=8.0, peak_memory_mib=87.0, kv_cache_memory_mib=31.0, quality_loss=0.08, safe=False),
            ]
            second_rows = [
                _record("full_lru", "r1", "full_gpu", ttft_ms=11.0, peak_memory_mib=120.0, kv_cache_memory_mib=72.0, quality_loss=0.0),
                _record("full_lru", "r2", "full_gpu", ttft_ms=13.0, peak_memory_mib=122.0, kv_cache_memory_mib=76.0, quality_loss=0.0),
                _record("static_best", "r1", "h2o_heavy15_recent15", ttft_ms=8.0, peak_memory_mib=91.0, kv_cache_memory_mib=33.0, quality_loss=0.02, safe=False),
                _record("static_best", "r2", "h2o_heavy15_recent15", ttft_ms=9.0, peak_memory_mib=94.0, kv_cache_memory_mib=35.0, quality_loss=0.04, safe=False),
                _record("static_safe", "r1", "full_gpu", ttft_ms=11.0, peak_memory_mib=120.0, kv_cache_memory_mib=72.0, quality_loss=0.0, fallback_reason="safe_fallback", safe=True),
                _record("static_safe", "r2", "full_gpu", ttft_ms=13.0, peak_memory_mib=122.0, kv_cache_memory_mib=76.0, quality_loss=0.0, fallback_reason="safe_fallback", safe=True),
                _record("utility_dynamic", "r1", "h2o_heavy20_recent20", ttft_ms=7.0, peak_memory_mib=86.0, kv_cache_memory_mib=28.0, quality_loss=0.03, safe=True),
                _record("utility_dynamic", "r2", "h2o_heavy20_recent20", ttft_ms=8.0, peak_memory_mib=89.0, kv_cache_memory_mib=29.0, quality_loss=0.05, safe=True),
                _record("uncalibrated_dynamic", "r1", "kivi_2bit_residual64", ttft_ms=6.5, peak_memory_mib=82.0, kv_cache_memory_mib=24.0, quality_loss=0.01, safe=False),
                _record("uncalibrated_dynamic", "r2", "kivi_2bit_residual64", ttft_ms=7.5, peak_memory_mib=83.0, kv_cache_memory_mib=26.0, quality_loss=0.06, safe=False),
            ]
            write_csv(
                policy_dir / "pilot_smoke_measured_policy_eps0p05_delta0p05_mem1000.csv",
                [row.to_row() for row in first_rows],
            )
            write_csv(
                policy_dir / "pilot_smoke_measured_policy_eps0p1_delta0p1_mem1500.csv",
                [row.to_row() for row in second_rows],
            )

            summary_path, plots = aggregate_directory(policy_dir, policy_dir / "baseline_wide_sweep_total_summary.csv")

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 10)
            self.assertEqual({row["policy"] for row in rows}, {"full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"})
            self.assertEqual({float(row["memory_budget_mib"]) for row in rows}, {1000.0, 1500.0})
            self.assertEqual({float(row["epsilon"]) for row in rows}, {0.05, 0.1})
            self.assertEqual({float(row["delta"]) for row in rows}, {0.05, 0.1})

            static_safe_1000 = next(row for row in rows if row["policy"] == "static_safe" and float(row["memory_budget_mib"]) == 1000.0)
            self.assertEqual(float(static_safe_1000["exact_fallback_ratio"]), 1.0)
            self.assertEqual(json.loads(static_safe_1000["action_distribution"]), {"full_gpu": 2})

            utility_1500 = next(row for row in rows if row["policy"] == "utility_dynamic" and float(row["memory_budget_mib"]) == 1500.0)
            self.assertAlmostEqual(float(utility_1500["mean_quality_loss"]), 0.04)
            self.assertEqual(json.loads(utility_1500["action_distribution"]), {"h2o_heavy20_recent20": 2})

            self.assertEqual(len(plots), 4)
            for plot in plots:
                self.assertTrue(plot.exists(), plot)

    def test_aggregate_directory_preserves_shadow_audit_quality_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir) / "policy_tables"
            policy_dir.mkdir(parents=True)
            records = [
                PolicyRunRecord(
                    policy="utility_dynamic",
                    request_id="s1_t0",
                    session_id="s1",
                    turn_index=0,
                    action_profile="lossy",
                    primary_profile="lossy",
                    backend_name="online_qwen",
                    ok=True,
                    measured=True,
                    ttft_ms=10.0,
                    latency_ms=12.0,
                    peak_memory_mib=10.0,
                    kv_cache_memory_mib=10.0,
                    quality_loss=None,
                    audit_selected=False,
                    predicted_quality_loss=0.1,
                    observed_quality_loss=None,
                    quality_estimate=0.1,
                ),
                PolicyRunRecord(
                    policy="utility_dynamic",
                    request_id="s2_t0",
                    session_id="s2",
                    turn_index=0,
                    action_profile="lossy",
                    primary_profile="lossy",
                    backend_name="online_qwen",
                    ok=True,
                    measured=True,
                    ttft_ms=11.0,
                    latency_ms=13.0,
                    peak_memory_mib=11.0,
                    kv_cache_memory_mib=11.0,
                    quality_loss=None,
                    audit_selected=True,
                    predicted_quality_loss=0.1,
                    observed_quality_loss=0.3,
                    quality_estimate=2.1,
                ),
            ]
            write_csv(
                policy_dir / "pilot_smoke_measured_policy_eps0p05_delta0p05_mem1000.csv",
                [record.to_row() for record in records],
            )

            summary_path, _ = aggregate_directory(policy_dir, policy_dir / "baseline_wide_sweep_total_summary.csv")

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                [summary] = list(csv.DictReader(handle))
            self.assertAlmostEqual(float(summary["mean_quality_estimate"]), 1.1)
            self.assertEqual(float(summary["audit_sample_count"]), 1.0)

    def test_aggregate_directory_preserves_diagnostic_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir) / "policy_tables"
            policy_dir.mkdir(parents=True)
            record = PolicyRunRecord(
                policy="full_lru",
                request_id="s1_t0",
                session_id="s1",
                turn_index=0,
                action_profile="full_gpu",
                backend_name="online_qwen",
                ok=True,
                measured=True,
                ttft_ms=10.0,
                latency_ms=12.0,
                peak_memory_mib=10.0,
                kv_cache_memory_mib=10.0,
                quality_loss=0.0,
                audit_selected=True,
                quality_estimate=0.0,
            )
            write_csv(
                policy_dir / "pilot_smoke_measured_policy_eps0p05_delta0p05_mem1000.csv",
                _policy_rows_with_provenance(
                    [record],
                    {"data": {"diagnostic_only": True}},
                    source_config="configs/pilot_diagnostic_session27.yaml",
                    run_dir="out/diagnostic_session27",
                ),
            )

            summary_path, _ = aggregate_directory(policy_dir, policy_dir / "baseline_wide_sweep_total_summary.csv")

            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                [summary] = list(csv.DictReader(handle))
            self.assertEqual(summary["diagnostic_only"], "True")
            self.assertEqual(summary["quality_status"], "risk_evidence_insufficient")
            self.assertEqual(summary["violation_status"], "risk_evidence_insufficient")
            self.assertEqual(summary["config"], "configs/pilot_diagnostic_session27.yaml")
            self.assertEqual(summary["run_dir"], "out/diagnostic_session27")

    def test_aggregate_directory_rejects_diagnostic_csv_without_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir) / "policy_tables"
            policy_dir.mkdir(parents=True)
            record = PolicyRunRecord(
                policy="full_lru",
                request_id="s1_t0",
                session_id="s1",
                turn_index=0,
                action_profile="full_gpu",
                backend_name="online_qwen",
                ok=True,
                measured=True,
                ttft_ms=10.0,
                latency_ms=12.0,
                peak_memory_mib=10.0,
                kv_cache_memory_mib=10.0,
                quality_loss=0.0,
            )
            write_csv(
                policy_dir / "pilot_smoke_measured_policy_eps0p05_delta0p05_mem1000.csv",
                [
                    {
                        **record.to_row(),
                        "diagnostic_only": True,
                        "quality_status": "risk_evidence_insufficient",
                        "violation_status": "risk_evidence_insufficient",
                    }
                ],
            )

            with self.assertRaises(ValueError):
                aggregate_directory(policy_dir, policy_dir / "baseline_wide_sweep_total_summary.csv")

    def test_aggregate_directory_rejects_mixed_provenance_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            policy_dir = Path(tmpdir) / "policy_tables"
            policy_dir.mkdir(parents=True)
            record = PolicyRunRecord(
                policy="full_lru",
                request_id="s1_t0",
                session_id="s1",
                turn_index=0,
                action_profile="full_gpu",
                backend_name="online_qwen",
                ok=True,
                measured=True,
                ttft_ms=10.0,
                latency_ms=12.0,
                peak_memory_mib=10.0,
                kv_cache_memory_mib=10.0,
                quality_loss=0.0,
            )
            for filename, config, run_dir in (
                ("pilot_smoke_measured_policy_eps0p05_delta0p05_mem1000.csv", "configs/run_a.yaml", "out/run_a"),
                ("pilot_smoke_measured_policy_eps0p05_delta0p05_mem1500.csv", "configs/run_b.yaml", "out/run_b"),
            ):
                write_csv(
                    policy_dir / filename,
                    _policy_rows_with_provenance(
                        [record],
                        {"data": {"diagnostic_only": True}},
                        source_config=config,
                        run_dir=run_dir,
                    ),
                )

            with self.assertRaises(ValueError):
                aggregate_directory(policy_dir, policy_dir / "baseline_wide_sweep_total_summary.csv")


if __name__ == "__main__":
    unittest.main()
