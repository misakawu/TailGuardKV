from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from run_util.core_types import PolicyRunRecord
from run_util.experiment_common import load_config, write_csv

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

        self.assertEqual(config["pilot"]["memory_budgets_mib"], [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
        self.assertEqual(config["pilot"]["epsilons"], [0.02, 0.05, 0.10])
        self.assertEqual(config["pilot"]["deltas"], [0.01, 0.05, 0.10])
        self.assertEqual(grid["memory_budgets_mib"], [1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0])
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


if __name__ == "__main__":
    unittest.main()
