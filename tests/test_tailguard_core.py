from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tempfile
import time
from types import SimpleNamespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import unittest
from unittest.mock import patch

from calibration.conformal import ConformalGuard
from aal import AALState, AuditSample, WilsonDriftDetector
from backends.measured_replay import MeasuredReplayBackend
from core_types import Action
from core_types import PolicyRunRecord, ProfileMeasurement, ProfileSpec, Request
from experiment_common import config_adapters, config_policies, config_profiles, config_runtime, exact_profiles, limit_requests_by_split, load_config, validate_profile_measurements, with_quality, write_csv
from metrics.quality import compute_quality_loss, normalized_exact_match_loss, rouge_l_loss, token_f1_loss
from metrics import MetricCollector
from policies.base import Policy
from policies.registry import build_policies
from profiles import qwen2_kv_runtime
from profiles.registry import build_profile_adapters
from vllm_lru_policy import create_vllm_policy
from env_asset_prepare.prepare_pilot_assets import format_longbench_prompt
import run_build_profile_table as profile_table_module
from run_build_profile_table import build_profile_table
from run_cli_common import run_command
from run_experiment import _policy_output_for_sweep, _policy_sweep_points, _summary_rows, build_parser as build_experiment_parser, pilot_smoke_measured
from run_run_policies import run_policies


def _measurement(
    request_id: str,
    profile: str,
    quality_loss: float | None,
    ttft_ms: float = 10.0,
    peak_memory_mib: float = 100.0,
    measured: bool = True,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        profile=profile,
        adapter="full",
        ok=True,
        measured=measured,
        output_text=f"{request_id}:{profile}",
        ttft_ms=ttft_ms,
        peak_memory_mib=peak_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=quality_loss,
        extra={"task": "qa", "length_bucket": "short", "split": "calibration"},
    )


FORMAL_PROFILES = [
    "full_gpu",
    "full_cpu",
    "recompute",
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
]


def _write_pilot_test_config(path: Path, profile_output: Path, policy_output: Path, summary_output: Path) -> None:
    profile_names = "\n".join(f"    - {profile}" for profile in FORMAL_PROFILES)
    specs = "\n".join(f"    {profile}: {{exact: {str(profile in {'full_gpu', 'full_cpu', 'recompute'}).lower()}}}" for profile in FORMAL_PROFILES)
    path.write_text(
        "\n".join(
            [
                "profiles:",
                "  adapters: [full, kivi, h2o]",
                "  names:",
                profile_names,
                "  specs:",
                specs,
                "policies:",
                "  record_rejected_unsafe: true",
                "  names: [full_lru, static_best, static_safe, utility_dynamic, uncalibrated_dynamic]",
                "pilot:",
                "  epsilons: [0.05]",
                "  deltas: [0.05]",
                "  memory_budgets_mib: [6144]",
                "outputs:",
                f"  smoke_profiles: {profile_output}",
                f"  smoke_policy: {policy_output}",
                f"  smoke_summary: {summary_output}",
            ]
        ),
        encoding="utf-8",
    )


class TailGuardCoreTest(unittest.TestCase):
    def test_format_longbench_prompt_includes_context_and_question(self) -> None:
        row = {
            "context": "Paper context sentence. " * 40,
            "input": "What is the result?",
        }

        prompt = format_longbench_prompt(row)

        self.assertIn("Context:\nPaper context sentence.", prompt)
        self.assertIn("\n\nQuestion:\nWhat is the result?", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))
        self.assertGreater(len(prompt), 900)

    def test_format_longbench_prompt_caps_pilot_context_without_dropping_question(self) -> None:
        row = {
            "context": "Long context sentence. " * 1000,
            "input": "What is the result?",
        }

        prompt = format_longbench_prompt(row, max_chars=1200)

        self.assertLessEqual(len(prompt), 1200)
        self.assertTrue(prompt.startswith("Context:\nLong context sentence."))
        self.assertIn("\n\nQuestion:\nWhat is the result?", prompt)
        self.assertTrue(prompt.endswith("\nAnswer:"))

    def test_qwen2_runtime_dispatches_parameterized_h2o_profiles(self) -> None:
        calls = []

        def fake_h2o(payload):
            calls.append(payload["profile"])
            return {"ok": True, "measured": True, "backend": "qwen2_h2o"}

        with patch.object(qwen2_kv_runtime, "_run_h2o_profile", side_effect=fake_h2o):
            result = qwen2_kv_runtime.run_profile({"profile": "h2o_heavy10_recent10"})

        self.assertTrue(result["ok"])
        self.assertEqual(calls, ["h2o_heavy10_recent10"])

    def test_kivi_failure_message_reports_prompt_tokens(self) -> None:
        error = (
            "KIVI proof missing: no quantized cache block and/or quant GEMV kernel call was observed. "
            "prompt_tokens=9 max_new_tokens=16 residual_length=32 "
            "quantized_layers=0 kernel_calls=0"
        )

        self.assertIn("prompt_tokens=9", error)
        self.assertIn("residual_length=32", error)

    def test_kivi_proof_error_includes_token_context(self) -> None:
        message = qwen2_kv_runtime._kivi_proof_error(
            prompt_tokens=9,
            max_new_tokens=16,
            residual_length=32,
            quantized_layers=0,
            kernel_calls=0,
        )

        self.assertIn("prompt_tokens=9", message)
        self.assertIn("max_new_tokens=16", message)
        self.assertIn("residual_length=32", message)
        self.assertIn("quantized_layers=0", message)
        self.assertIn("kernel_calls=0", message)

    def test_pytest_ini_limits_collection_to_project_tests(self) -> None:
        pytest_ini = Path("pytest.ini").read_text(encoding="utf-8")
        self.assertIn("testpaths = tests", pytest_ini)
        self.assertIn("third_party", pytest_ini)

    def test_conformal_guard_uses_bonferroni_delta_and_residual_quantile(self) -> None:
        guard = ConformalGuard(
            epsilon=0.5,
            delta=0.2,
            calibration_rows=[
                _measurement("r1", "kivi_4bit", 0.1),
                _measurement("r2", "kivi_4bit", 0.2),
                _measurement("r3", "kivi_4bit", 0.3),
                _measurement("r4", "kivi_2bit", 0.4),
                _measurement("r5", "kivi_2bit", 0.5),
            ],
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertAlmostEqual(guard.delta_a, 0.1)
        self.assertAlmostEqual(guard.risk_upper(request, "kivi_4bit", 0.1), 0.25)
        self.assertTrue(guard.is_safe(request, "kivi_4bit", 0.1))

    def test_conformal_guard_counts_configured_lossy_profiles_without_samples(self) -> None:
        guard = ConformalGuard(
            epsilon=0.5,
            delta=0.2,
            exact_profiles={"full_gpu"},
            profiles=["full_gpu", "kivi_4bit", "kivi_2bit", "h2o_heavy_hitter"],
            calibration_rows=[
                _measurement("r1", "kivi_4bit", 0.1),
                _measurement("r2", "kivi_4bit", 0.2),
            ],
        )
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit", "kivi_2bit", "h2o_heavy_hitter"})
        self.assertAlmostEqual(guard.delta_a, 0.2 / 3)

    def test_conformal_guard_sparse_groups_are_unsafe_and_exact_safe(self) -> None:
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            calibration_rows=[_measurement("r1", "kivi_4bit", 0.1)],
            exact_profiles={"full_gpu"},
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(guard.risk_upper(request, "full_gpu", 0.9), 0.0)
        self.assertFalse(guard.is_safe(request, "kivi_4bit", 0.0))

    def test_conformal_guard_empty_calibration_set_makes_lossy_profiles_unsafe(self) -> None:
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"full_gpu"},
            profiles=["full_gpu", "kivi_4bit"],
            calibration_rows=[],
        )
        request = Request("eval1", "qa", "short prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit"})
        self.assertEqual(guard.risk_upper(request, "full_gpu", 0.9), 0.0)
        self.assertFalse(guard.is_safe(request, "kivi_4bit", 0.0))

    def test_conformal_guard_builds_large_calibration_table_without_quadratic_regression(self) -> None:
        rows = [
            _measurement(f"r{i}", "kivi_4bit" if i % 2 == 0 else "h2o_heavy_hitter", (i % 10) / 100.0)
            for i in range(6000)
        ]
        started = time.perf_counter()
        guard = ConformalGuard(
            epsilon=0.2,
            delta=0.05,
            profiles=["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            exact_profiles={"full_gpu"},
            calibration_rows=rows,
        )
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 2.0)
        self.assertEqual(guard.lossy_profiles, {"kivi_4bit", "h2o_heavy_hitter"})

    def test_registry_builds_only_required_august_baselines(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        policies = build_policies(
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(
            [policy.name for policy in policies],
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
        )

    def test_registry_rejects_quality_oracle_after_policy_cleanup(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "未知 policy: quality_oracle"):
            build_policies(
                ["quality_oracle"],
                rows,
                rows,
                ["full_gpu"],
                0.2,
                0.05,
                {"full_gpu"},
            )

    def test_registry_keeps_august_baseline_order_stable(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=20.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=5.0),
        ]
        policies = build_policies(
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        self.assertEqual(
            [policy.name for policy in policies],
            ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
        )

    def test_static_baselines_fallback_to_exact_when_no_lossy_profile_is_selectable(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=20.0),
            _measurement("c1", "kivi_4bit", 0.8, ttft_ms=5.0),
        ]
        policies = build_policies(
            ["static_best", "static_safe"],
            rows,
            rows,
            ["kivi_4bit", "full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual([policy.decide(request, None, None).profile for policy in policies], ["full_gpu", "full_gpu"])

    def test_full_lru_fixed_to_full_gpu(self) -> None:
        [policy] = build_policies(
            ["full_lru"],
            [_measurement("c1", "full_gpu", 0.0)],
            [_measurement("e1", "full_gpu", 0.0)],
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt"), None, None)
        self.assertEqual(action.profile, "full_gpu")
        self.assertEqual(action.reason, "full precision exact profile")

    def test_static_best_selects_lowest_ttft_under_mean_quality_constraint(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=8.0),
            _measurement("c2", "kivi_4bit", 0.1, ttft_ms=9.0),
            _measurement("c1", "kivi_2bit", 0.3, ttft_ms=3.0),
            _measurement("c2", "kivi_2bit", 0.3, ttft_ms=4.0),
        ]
        [policy] = build_policies(
            ["static_best"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "kivi_2bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "kivi_4bit")

    def test_static_safe_rejects_high_violation_rate(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=8.0),
            _measurement("c2", "kivi_4bit", 0.3, ttft_ms=9.0),
            _measurement("c1", "h2o_heavy_hitter", 0.1, ttft_ms=12.0),
            _measurement("c2", "h2o_heavy_hitter", 0.1, ttft_ms=13.0),
        ]
        [policy] = build_policies(
            ["static_safe"],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "h2o_heavy_hitter")

    def test_utility_dynamic_uses_configured_utility_weights(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0, peak_memory_mib=10.0),
            _measurement("c1", "kivi_4bit", 0.1, ttft_ms=5.0, peak_memory_mib=100.0),
            _measurement("c2", "kivi_4bit", 0.1, ttft_ms=6.0, peak_memory_mib=100.0),
            _measurement("c1", "h2o_heavy_hitter", 0.12, ttft_ms=8.0, peak_memory_mib=10.0),
            _measurement("c2", "h2o_heavy_hitter", 0.12, ttft_ms=9.0, peak_memory_mib=10.0),
        ]
        [policy] = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 1.0, "loss_weight": 1.0}],
            rows,
            rows,
            ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(policy.decide(request, None, None).profile, "h2o_heavy_hitter")

    def test_uncalibrated_dynamic_uses_point_prediction_threshold_only(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c1", "kivi_4bit", 0.3, ttft_ms=5.0),
            _measurement("c2", "kivi_4bit", 0.3, ttft_ms=6.0),
            _measurement("c1", "kivi_2bit", 0.1, ttft_ms=3.0),
            _measurement("c2", "kivi_2bit", 0.1, ttft_ms=4.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "kivi_2bit", "kivi_4bit"],
            0.2,
            0.05,
            {"full_gpu"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        action = policy.decide(request, None, None)
        self.assertEqual(action.profile, "kivi_2bit")
        self.assertEqual(action.fallback_reason, "点预测阈值通过")

    def test_utility_dynamic_falls_back_from_calibrated_unsafe_and_records_rejected(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "full_cpu", 0.0, ttft_ms=9.0),
            _measurement("c2", "full_cpu", 0.0, ttft_ms=10.0),
            _measurement("c1", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
            _measurement("c2", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
        ]
        [policy] = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 0.0, "loss_weight": 0.0}],
            rows,
            rows,
            ["full_gpu", "full_cpu", "kivi_4bit_residual32"],
            0.2,
            0.05,
            {"full_gpu", "full_cpu"},
            record_rejected_unsafe=True,
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "full_cpu")
        self.assertEqual(action.rejected_profile, "kivi_4bit_residual32")
        self.assertIsNotNone(action.rejected_pred_loss)
        self.assertIsNotNone(action.rejected_risk_upper)

    def test_uncalibrated_dynamic_falls_back_from_calibrated_unsafe_and_records_rejected(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "recompute", 0.0, ttft_ms=8.0),
            _measurement("c2", "recompute", 0.0, ttft_ms=9.0),
            _measurement("c1", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
            _measurement("c2", "kivi_4bit_residual32", 0.5, ttft_ms=1.0),
        ]
        [policy] = build_policies(
            ["uncalibrated_dynamic"],
            rows,
            rows,
            ["full_gpu", "recompute", "kivi_4bit_residual32"],
            0.2,
            0.05,
            {"full_gpu", "recompute"},
            record_rejected_unsafe=True,
        )
        action = policy.decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "recompute")
        self.assertEqual(action.rejected_profile, "kivi_4bit_residual32")

    def test_vllm_lru_rank_tuple_orders_by_recency_then_block_id(self) -> None:
        policy = create_vllm_policy("full_lru")
        pool = SimpleNamespace()
        policy.on_admit(pool, 2)
        policy.on_admit(pool, 1)
        self.assertLess(policy.rank_tuple(pool, 2), policy.rank_tuple(pool, 1))
        self.assertLess(policy.rank_tuple(pool, 3), policy.rank_tuple(pool, 4))

    def test_profile_registry_uses_full_kivi_h2o_adapters(self) -> None:
        adapters = build_profile_adapters(
            ["full", "kivi", "h2o"],
            {"profile_smoke_model": "/tmp/model"},
        )
        self.assertEqual([adapter.name for adapter in adapters], ["full", "kivi", "h2o"])

    def test_profile_registry_exposes_parameterized_profile_grid(self) -> None:
        adapters = build_profile_adapters(
            ["full", "kivi", "h2o"],
            {"profile_smoke_model": "/tmp/model"},
        )
        specs = [spec for adapter in adapters for spec in adapter.profiles()]
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "full_gpu",
                "full_cpu",
                "recompute",
                "kivi_4bit_residual32",
                "kivi_4bit_residual64",
                "kivi_2bit_residual32",
                "kivi_2bit_residual64",
                "h2o_heavy10_recent10",
                "h2o_heavy15_recent15",
                "h2o_heavy20_recent20",
            ],
        )
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["bits"], 4)
        self.assertEqual(by_name["kivi_4bit_residual32"].metadata["kivi_residual_length"], 32)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["bits"], 2)
        self.assertEqual(by_name["kivi_2bit_residual64"].metadata["kivi_residual_length"], 64)
        self.assertEqual(by_name["h2o_heavy15_recent15"].metadata["h2o_heavy_ratio"], 0.15)
        self.assertEqual(by_name["h2o_heavy15_recent15"].metadata["h2o_recent_ratio"], 0.15)

    def test_recompute_profile_disables_transformers_kv_cache(self) -> None:
        captured = {}

        def fake_run(command, **kwargs):
            captured["code"] = command[-1]
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "ok": True,
                        "output_text": "ok",
                        "latency_ms": 1.0,
                        "ttft_ms": 1.0,
                        "peak_memory_mib": 1.0,
                        "resident_memory_mib": 1.0,
                    }
                ),
                stderr="",
            )

        [adapter] = build_profile_adapters(["full"], {"profile_smoke_model": "/tmp/model"})
        with patch("profiles.base.subprocess.run", side_effect=fake_run):
            row = adapter.profile(Request("r1", "qa", "prompt"), "recompute", dry_run=False)

        self.assertTrue(row.ok)
        self.assertIn("'use_cache': False", captured["code"])

    def test_pilot_configs_use_formal_profile_grid_and_two_request_scales(self) -> None:
        pilot = load_config(Path("configs/pilot.yaml"))
        pilot_50 = load_config(Path("configs/pilot_50.yaml"))
        expected_profiles = [
            "full_gpu",
            "full_cpu",
            "recompute",
            "kivi_4bit_residual32",
            "kivi_4bit_residual64",
            "kivi_2bit_residual32",
            "kivi_2bit_residual64",
            "h2o_heavy10_recent10",
            "h2o_heavy15_recent15",
            "h2o_heavy20_recent20",
        ]
        self.assertEqual(config_adapters(pilot), ["full", "kivi", "h2o"])
        self.assertEqual(config_profiles(pilot), expected_profiles)
        self.assertEqual(config_profiles(pilot_50), expected_profiles)
        self.assertEqual(pilot["data"]["max_requests"], 200)
        self.assertEqual(pilot_50["data"]["max_requests"], 50)
        self.assertTrue(pilot["policies"]["record_rejected_unsafe"])
        self.assertTrue(pilot_50["policies"]["record_rejected_unsafe"])
        self.assertEqual(config_policies(pilot), ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"])
        self.assertEqual(
            exact_profiles(expected_profiles, pilot),
            {"full_gpu", "full_cpu", "recompute"},
        )

    def test_registry_rejects_structured_static_policy_config_after_cleanup(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "未知 policy: static"):
            build_policies(
                [{"type": "static", "profile": "full_gpu", "name": "static_profile:full_gpu"}],
                rows,
                rows,
                ["full_gpu"],
                0.2,
                0.05,
                {"full_gpu"},
            )

    def test_measured_replay_pandas_mode_preserves_lookup_semantics(self) -> None:
        row = _measurement("r1", "full_gpu", 0.0)
        backend = MeasuredReplayBackend([row], use_pandas=True)
        self.assertEqual(backend.run([Request("r1", "qa", "p")], ["full_gpu"])[0], row)

    def test_worst_group_uses_task_length_bucket_and_profile(self) -> None:
        records = [
            PolicyRunRecord(
                policy="p",
                request_id="r1",
                task="qa",
                length_bucket="short",
                action_profile="kivi_4bit",
                ok=True,
                measured=True,
                reason="same",
                quality_loss=0.3,
            ),
            PolicyRunRecord(
                policy="p",
                request_id="r2",
                task="summary",
                length_bucket="long",
                action_profile="kivi_4bit",
                ok=True,
                measured=True,
                reason="same",
                quality_loss=0.0,
            ),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["worst_group_violation"], 1.0)

    def test_metrics_delta_slack_uses_delta_not_epsilon(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "kivi_4bit", True, True, quality_loss=0.3),
            PolicyRunRecord("p", "r2", "kivi_4bit", True, True, quality_loss=0.0),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["target_delta"], 0.05)
        self.assertEqual(summary["p"]["violation_rate"], 0.5)
        self.assertEqual(summary["p"]["delta_slack"], -0.45)

    def test_metrics_reports_controller_oracle_and_aal_fields(self) -> None:
        records = [
            PolicyRunRecord(
                "p",
                "r1",
                "full_gpu",
                True,
                True,
                controller_overhead_ms=1.0,
                controller_qrp_ms=0.2,
                controller_cg_ms=0.3,
                controller_stc_ms=0.5,
                oracle_cost_ms=9.0,
                optimality_gap=0.1,
                audit_rate=0.05,
                drift_state="stable",
            )
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["controller_qrp_ms"], 0.2)
        self.assertEqual(summary["p"]["controller_cg_ms"], 0.3)
        self.assertEqual(summary["p"]["controller_stc_ms"], 0.5)
        self.assertEqual(summary["p"]["oracle_cost_ms"], 9.0)
        self.assertEqual(summary["p"]["optimality_gap"], 0.1)
        self.assertEqual(summary["p"]["audit_rate"], 0.05)
        self.assertEqual(summary["p"]["drift_state"], "stable")

    def test_policy_summary_reports_profile_optimization_diagnostics(self) -> None:
        records = [
            PolicyRunRecord("p", "r1", "full_gpu", True, True, rejected_profile="", candidate_safe_count=2.0),
            PolicyRunRecord("p", "r2", "full_gpu", True, True, rejected_profile="kivi_4bit_residual32", candidate_safe_count=1.0),
            PolicyRunRecord("q", "r1", "kivi_4bit_residual32", True, True, safe=True, candidate_safe_count=3.0),
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})
        self.assertEqual(summary["p"]["unique_action_count"], 1.0)
        self.assertTrue(summary["p"]["identical_to_full_lru"])
        self.assertEqual(summary["p"]["unsafe_action_count"], 1.0)
        self.assertEqual(summary["p"]["candidate_safe_count"], 1.5)
        self.assertFalse(summary["q"]["identical_to_full_lru"])

    def test_metrics_smoke_summary_fields_are_complete_and_nan_on_empty_values(self) -> None:
        records = [
            PolicyRunRecord(
                "p",
                "r1",
                "full_gpu",
                True,
                True,
                quality_loss=None,
                safe=None,
                controller_overhead_ms=None,
            )
        ]
        summary = MetricCollector().summarize_policy_runs(records, epsilon=0.2, delta=0.05, exact_profiles={"full_gpu"})["p"]
        for field in (
            "p95_ttft_ms",
            "p95_peak_memory_mib",
            "violation_rate",
            "safe_ratio",
            "fallback_ratio",
            "controller_overhead_ms",
        ):
            self.assertIn(field, summary)
        self.assertTrue(math.isnan(summary["p95_ttft_ms"]))
        self.assertTrue(math.isnan(summary["p95_peak_memory_mib"]))
        self.assertTrue(math.isnan(summary["violation_rate"]))
        self.assertTrue(math.isnan(summary["controller_overhead_ms"]))

    def test_validate_profile_measurements_requires_all_configured_profiles(self) -> None:
        rows = [_measurement("r1", "full_gpu", 0.0)]
        with self.assertRaisesRegex(ValueError, "缺少必需 profile"):
            validate_profile_measurements(rows, required_profiles=["full_gpu", "kivi_4bit"])

    def test_validate_profile_measurements_requires_ok_for_measured_replay(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="full_gpu",
            adapter="full",
            ok=False,
            measured=True,
            output_text="x",
            ttft_ms=1.0,
            peak_memory_mib=1.0,
            quality_loss=0.0,
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )
        with self.assertRaisesRegex(ValueError, "ok=True"):
            validate_profile_measurements([row], require_measured=True)

    def test_validate_profile_measurements_rejects_empty_table(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile 表为空"):
            validate_profile_measurements([])

    def test_validate_profile_measurements_accepts_empty_measured_output_text(self) -> None:
        row = ProfileMeasurement(
            request_id="r1",
            profile="engine_full_lru",
            adapter="vllm_lru",
            ok=True,
            measured=True,
            output_text="",
            ttft_ms=1.0,
            peak_memory_mib=1.0,
            quality_loss=1.0,
            extra={"task": "qa", "length_bucket": "short", "split": "eval"},
        )
        validate_profile_measurements([row], require_measured=True, required_profiles=["engine_full_lru"])

    def test_validate_profile_measurements_rejects_lossy_without_full_baseline_quality(self) -> None:
        rows = [
            _measurement("r1", "kivi_4bit", None),
        ]
        with self.assertRaisesRegex(ValueError, "quality_loss"):
            validate_profile_measurements(rows, require_measured=True, required_profiles=["kivi_4bit"])

    def test_action_and_policy_record_expose_rejected_unsafe_fields(self) -> None:
        action = Action(
            profile="full_gpu",
            reason="fallback",
            rejected_profile="kivi_4bit_residual32",
            rejected_pred_loss=0.1,
            rejected_risk_upper=0.3,
            candidate_safe_count=1.0,
        )
        record = PolicyRunRecord(
            "utility_dynamic",
            "r1",
            action.profile,
            True,
            True,
            rejected_profile=action.rejected_profile,
            rejected_pred_loss=action.rejected_pred_loss,
            rejected_risk_upper=action.rejected_risk_upper,
            candidate_safe_count=action.candidate_safe_count,
        )
        row = record.to_row()
        self.assertEqual(row["rejected_profile"], "kivi_4bit_residual32")
        self.assertEqual(row["rejected_pred_loss"], 0.1)
        self.assertEqual(row["rejected_risk_upper"], 0.3)
        self.assertEqual(row["candidate_safe_count"], 1.0)

    def test_stats_policy_fastest_exact_uses_calibration_p95_ttft(self) -> None:
        rows = [
            _measurement("c1", "full_gpu", 0.0, ttft_ms=30.0),
            _measurement("c2", "full_gpu", 0.0, ttft_ms=31.0),
            _measurement("c1", "full_cpu", 0.0, ttft_ms=10.0),
            _measurement("c2", "full_cpu", 0.0, ttft_ms=11.0),
            _measurement("c1", "recompute", 0.0, ttft_ms=20.0),
        ]
        [policy] = build_policies(
            ["utility_dynamic"],
            rows,
            rows,
            ["full_gpu", "full_cpu", "recompute"],
            0.2,
            0.05,
            {"full_gpu", "full_cpu", "recompute"},
        )
        self.assertEqual(policy._fastest_exact_profile(), "full_cpu")

    def test_build_policies_accepts_record_rejected_unsafe_flag(self) -> None:
        rows = [_measurement("c1", "full_gpu", 0.0)]
        [policy] = build_policies(
            ["utility_dynamic"],
            rows,
            rows,
            ["full_gpu"],
            0.2,
            0.05,
            {"full_gpu"},
            record_rejected_unsafe=True,
        )
        self.assertTrue(policy.record_rejected_unsafe)

    def test_e0_config_uses_three_reproducible_profiles(self) -> None:
        config = load_config(Path("configs/e0_reproduce.yaml"))
        self.assertEqual(config_profiles(config), ["full_gpu", "kivi_4bit", "h2o_heavy_hitter"])
        self.assertEqual(config["model"]["pilot_model"], config["model"]["profile_smoke_model"])
        self.assertEqual(config["data"]["requests"], "data/fixtures/e0_reproduce_requests.jsonl")

    def test_config_lists_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "profiles.adapters"):
            config_adapters({"profiles": {"names": ["full_gpu"]}, "policies": {"names": ["full_lru"]}})
        with self.assertRaisesRegex(ValueError, "profiles.names"):
            config_profiles({"profiles": {"adapters": ["full"]}, "policies": {"names": ["full_lru"]}})
        with self.assertRaisesRegex(ValueError, "policies.names"):
            config_policies({"profiles": {"adapters": ["full"], "names": ["full_gpu"]}, "policies": {}})

    def test_pilot_config_limits_profile_requests_to_200(self) -> None:
        config = load_config(Path("configs/pilot.yaml"))
        self.assertEqual(config_runtime(config)["max_requests"], 200)

    def test_limit_requests_by_split_balances_calibration_and_eval(self) -> None:
        requests = [
            Request(f"c{index}", "qa", "prompt", metadata={"split": "calibration"})
            for index in range(250)
        ] + [
            Request(f"e{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(250)
        ]

        limited = limit_requests_by_split(requests, 50)

        self.assertEqual(len(limited), 50)
        self.assertEqual(sum(request.metadata.get("split") == "calibration" for request in limited), 25)
        self.assertEqual(sum(request.metadata.get("split") == "eval" for request in limited), 25)

    def test_limit_requests_by_split_preserves_single_split_prefix(self) -> None:
        requests = [
            Request(f"e{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(23)
        ]

        limited = limit_requests_by_split(requests, 20)

        self.assertEqual([request.request_id for request in limited], [f"e{index}" for index in range(20)])

    def test_limit_requests_by_split_zero_keeps_all_requests(self) -> None:
        requests = [
            Request(f"r{index}", "qa", "prompt", metadata={"split": "eval"})
            for index in range(3)
        ]

        self.assertIs(limit_requests_by_split(requests, 0), requests)

    def test_run_experiment_parser_only_exposes_pilot_smoke_measured(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured"])
        self.assertIs(args.func, pilot_smoke_measured)
        with self.assertRaises(SystemExit):
            parser.parse_args(["run-policies"])

    def test_build_profile_table_uses_profile_many(self) -> None:
        class StubAdapter:
            name = "stub"

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                self.requests = list(requests)
                self.profile_name = profile_name
                self.dry_run = dry_run
                return [_measurement(request.request_id, profile_name, 0.0) for request in requests]

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            output_path = Path(tmpdir) / "profiles.csv"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters:",
                        "    - stub",
                        "  names:",
                        "    - engine_full_lru",
                        "data:",
                        "  requests: data/fixtures/e0_reproduce_requests.jsonl",
                    ]
                ),
                encoding="utf-8",
            )
            stub = StubAdapter()
            with (
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}, "data": {"requests": "data/fixtures/e0_reproduce_requests.jsonl"}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch(
                    "run_build_profile_table.load_requests",
                    return_value=(
                        [
                            Request("r1", "qa", "p", metadata={"split": "eval"}),
                            Request("r2", "qa", "q", metadata={"split": "eval"}),
                        ],
                        False,
                    ),
                ),
            ):
                code = build_profile_table(
                    argparse.Namespace(
                        config=str(config_path),
                        adapters=None,
                        output=str(output_path),
                        import_measurements="",
                        dry_run=False,
                    )
                )
            self.assertEqual(code, 0)
            self.assertFalse(stub.dry_run)
            self.assertEqual(stub.profile_name, "full_gpu")
            self.assertEqual([request.request_id for request in stub.requests], ["r1", "r2"])

    def test_build_profile_table_respects_max_requests_before_chunking(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = []

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                chunk = list(requests)
                self.calls.append([request.request_id for request in chunk])
                return [_measurement(request.request_id, profile_name, 0.0) for request in chunk]

        requests = [
            Request(f"c{index}", "qa", f"prompt {index}", metadata={"split": "calibration"})
            for index in range(25)
        ] + [
            Request(f"e{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(25)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            with (
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1, "max_requests": 20}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
            ):
                code = build_profile_table(
                    argparse.Namespace(
                        config="config.yaml",
                        adapters=None,
                        output=str(output_path),
                        import_measurements="",
                        dry_run=False,
                    )
                )

            self.assertEqual(code, 0)
            self.assertEqual([len(call) for call in stub.calls], [10, 10])
            self.assertEqual(stub.calls[0], [f"c{index}" for index in range(10)])
            self.assertEqual(stub.calls[1], [f"e{index}" for index in range(10)])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 20)
            self.assertEqual(sum(row["split"] == "calibration" for row in rows), 10)
            self.assertEqual(sum(row["split"] == "eval" for row in rows), 10)

    def test_build_profile_table_dry_run_covers_required_pilot_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pilot_profiles.csv"
            code = build_profile_table(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    adapters=None,
                    output=str(output_path),
                    import_measurements="",
                    dry_run=True,
                )
            )
            self.assertEqual(code, 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            profiles = {row["profile"] for row in rows}
            self.assertEqual(
                profiles,
                {
                    "full_gpu",
                    "full_cpu",
                    "recompute",
                    "kivi_4bit_residual32",
                    "kivi_4bit_residual64",
                    "kivi_2bit_residual32",
                    "kivi_2bit_residual64",
                    "h2o_heavy10_recent10",
                    "h2o_heavy15_recent15",
                    "h2o_heavy20_recent20",
                },
            )
            self.assertTrue(all(row["measured"] == "False" for row in rows))

    def test_run_policies_runs_five_baselines_on_pilot_dry_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"

            build_code = build_profile_table(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    adapters=None,
                    output=str(profile_path),
                    import_measurements="",
                    dry_run=True,
                )
            )
            self.assertEqual(build_code, 0)

            policy_code = run_policies(
                argparse.Namespace(
                    config="configs/pilot.yaml",
                    measurements=str(profile_path),
                    output=str(policy_path),
                    profiles=None,
                    policies=None,
                    policy_config=None,
                    epsilon=None,
                    delta=None,
                    memory_budget_mib=None,
                    use_pandas_replay=False,
                    allow_dry_run_replay=True,
                )
            )
            self.assertEqual(policy_code, 0)

            with policy_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["policy"] for row in rows},
                {"full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"},
            )
            self.assertIn("rejected_profile", rows[0])
            self.assertIn("candidate_safe_count", rows[0])

    def test_build_profile_table_chunks_profile_many_and_writes_incrementally(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = []

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                chunk = list(requests)
                self.calls.append([request.request_id for request in chunk])
                return [_measurement(request.request_id, profile_name, 0.0) for request in chunk]

        requests = [
            Request(f"r{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(23)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            append_counts = []
            append_impl = profile_table_module._append_profile_rows

            def record_append(path, rows):
                append_counts.append(len(rows))
                append_impl(path, rows)

            with (
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
                patch("run_build_profile_table._append_profile_rows", side_effect=record_append),
            ):
                stream = io.StringIO()
                with redirect_stderr(stream):
                    code = build_profile_table(
                        argparse.Namespace(
                            config="config.yaml",
                            adapters=None,
                            output=str(output_path),
                            import_measurements="",
                            dry_run=False,
                        )
                    )

            self.assertEqual(code, 0)
            self.assertEqual([len(call) for call in stub.calls], [10, 10, 3])
            self.assertEqual(append_counts, [10, 10, 3])
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 23)
            progress = [
                json.loads(line)
                for line in stream.getvalue().splitlines()
                if '"event": "profile_chunk_complete"' in line
            ]
            self.assertEqual([row["chunk_index"] for row in progress], [1, 2, 3])
            self.assertEqual([row["completed_requests"] for row in progress], [10, 20, 23])
            self.assertTrue(all(row["output"] == str(output_path) for row in progress))

    def test_build_profile_table_chunk_failure_keeps_completed_output(self) -> None:
        class StubAdapter:
            name = "stub"

            def __init__(self) -> None:
                self.calls = 0

            def profiles(self):
                return (ProfileSpec("full_gpu", "stub", "env", lossy=False, exact=True),)

            def profile_many(self, requests, profile_name, dry_run=True):
                self.calls += 1
                if self.calls == 2:
                    return [
                        ProfileMeasurement(
                            request_id=request.request_id,
                            profile=profile_name,
                            adapter="stub",
                            ok=False,
                            measured=False,
                            error="chunk failed",
                        )
                        for request in requests
                    ]
                return [_measurement(request.request_id, profile_name, 0.0) for request in requests]

        requests = [
            Request(f"r{index}", "qa", f"prompt {index}", metadata={"split": "eval"})
            for index in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "profiles.csv"
            stub = StubAdapter()
            with (
                patch("run_build_profile_table.load_config", return_value={"profiles": {"adapters": ["stub"], "names": ["full_gpu"]}}),
                patch("run_build_profile_table.config_adapters", return_value=["stub"]),
                patch("run_build_profile_table.config_profiles", return_value=["full_gpu"]),
                patch("run_build_profile_table.config_runtime", return_value={"repeat": 1}),
                patch("run_build_profile_table.build_profile_adapters", return_value=[stub]),
                patch("run_build_profile_table.load_requests", return_value=(requests, False)),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = build_profile_table(
                        argparse.Namespace(
                            config="config.yaml",
                            adapters=None,
                            output=str(output_path),
                            import_measurements="",
                            dry_run=False,
                        )
                    )

            self.assertEqual(code, 2)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["request_id"] for row in rows], [f"r{index}" for index in range(10)])
            diagnostic_path = Path(tmpdir) / "profiles_failed_chunks.csv"
            self.assertTrue(diagnostic_path.exists())
            with diagnostic_path.open("r", encoding="utf-8", newline="") as handle:
                failed_rows = list(csv.DictReader(handle))
            self.assertEqual([row["request_id"] for row in failed_rows], ["r10", "r11"])
            payload = json.loads(stream.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["diagnostic_output"], str(diagnostic_path))
            self.assertEqual(payload["failures"][0]["error"], "chunk failed")

    def test_pilot_smoke_measured_builds_profiles_without_dry_run_and_replays_same_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            profiles = FORMAL_PROFILES
            calls = {}

            def fake_build(args: argparse.Namespace) -> int:
                calls["profile_args"] = args
                write_csv(Path(args.output), [_measurement("e1", profile, 0.0).to_row() for profile in profiles])
                print(json.dumps({"output": args.output, "rows": len(profiles), "summary": {"profiles": len(profiles)}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                calls["policy_args"] = args
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 5,
                            "epsilon": 0.05,
                            "delta": 0.05,
                            "memory_budget_mib": 6144.0,
                            "summary": {"full_lru": {"p95_ttft_ms": 1.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertEqual(stream.getvalue(), "")
            self.assertFalse(calls["profile_args"].dry_run)
            self.assertEqual(calls["profile_args"].output, str(profile_path))
            self.assertEqual(calls["policy_args"].measurements, str(profile_path))
            self.assertEqual(calls["policy_args"].output, str(policy_path))
            self.assertFalse(calls["policy_args"].allow_dry_run_replay)
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["section"], "experiment")
            self.assertEqual(summary_rows[0]["ok"], "True")
            self.assertEqual(summary_rows[0]["profile_rows"], "10")
            self.assertEqual(summary_rows[0]["policy_rows"], "5")
            full_lru = next(row for row in summary_rows if row["section"] == "policy" and row["name"] == "full_lru")
            self.assertEqual(full_lru["p95_ttft_ms"], "1.0")
            self.assertLess(
                list(summary_rows[0]).index("mean_ttft_ms"),
                list(summary_rows[0]).index("action_distribution"),
            )
            for deleted in ("return_code", "step", "profile_output", "policy_output", "summary_output"):
                self.assertNotIn(deleted, summary_rows[0])

    def test_policy_sweep_points_expand_cartesian_product(self) -> None:
        config = {
            "pilot": {
                "epsilons": [0.05, 0.10],
                "deltas": [0.05],
                "memory_budgets_mib": [6144, 8192],
            }
        }

        self.assertEqual(
            _policy_sweep_points(config),
            [
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 6144.0},
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 8192.0},
                {"epsilon": 0.10, "delta": 0.05, "memory_budget_mib": 6144.0},
                {"epsilon": 0.10, "delta": 0.05, "memory_budget_mib": 8192.0},
            ],
        )

    def test_policy_output_for_sweep_keeps_single_run_path(self) -> None:
        self.assertEqual(
            _policy_output_for_sweep(
                "out/policy_tables/pilot_policy.csv",
                {"epsilon": 0.05, "delta": 0.05, "memory_budget_mib": 6144.0},
                1,
            ),
            "out/policy_tables/pilot_policy.csv",
        )

    def test_policy_output_for_sweep_namespaces_multi_run_path(self) -> None:
        self.assertEqual(
            _policy_output_for_sweep(
                "out/policy_tables/pilot_policy.csv",
                {"epsilon": 0.05, "delta": 0.10, "memory_budget_mib": 8192.0},
                4,
            ),
            "out/policy_tables/pilot_policy_eps0p05_delta0p1_mem8192.csv",
        )

    def test_pilot_smoke_measured_runs_all_policy_sweep_outputs_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.05, 0.10]",
                        "  deltas: [0.05]",
                        "  memory_budgets_mib: [6144, 8192]",
                        "outputs:",
                        f"  smoke_profiles: {profile_path}",
                        f"  smoke_policy: {policy_path}",
                        f"  smoke_summary: {summary_path}",
                    ]
                ),
                encoding="utf-8",
            )
            policy_calls = []

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                policy_calls.append((args.epsilon, args.delta, args.memory_budget_mib, args.output))
                print(
                    json.dumps(
                        {
                            "output": args.output,
                            "rows": 2,
                            "epsilon": float(args.epsilon),
                            "delta": float(args.delta),
                            "memory_budget_mib": float(args.memory_budget_mib),
                            "summary": {"full_lru": {"count": 2.0}},
                        }
                    )
                )
                return 0

            with (
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertEqual(len(policy_calls), 4)
            self.assertEqual(policy_calls[0][0:3], (0.05, 0.05, 6144.0))
            self.assertEqual(policy_calls[-1][0:3], (0.10, 0.05, 8192.0))
            self.assertEqual(len({call[3] for call in policy_calls}), 4)
            self.assertTrue(all(call[3] != str(policy_path) for call in policy_calls))

    def test_pilot_summary_contains_each_policy_sweep_row(self) -> None:
        rows = _summary_rows(
            {
                "ok": True,
                "config": "pilot.yaml",
                "rows": {"profiles": 10, "policy": 4},
                "profile": {"summary": {}},
                "policy_runs": [
                    {
                        "ok": True,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 6144.0,
                        "payload": {"summary": {"full_lru": {"count": 2.0, "p95_ttft_ms": 1.0}}},
                    },
                    {
                        "ok": True,
                        "epsilon": 0.1,
                        "delta": 0.05,
                        "memory_budget_mib": 6144.0,
                        "payload": {"summary": {"full_lru": {"count": 2.0, "p95_ttft_ms": 2.0}}},
                    },
                ],
            }
        )

        policy_rows = [row for row in rows if row["section"] == "policy" and row["name"] == "full_lru"]
        self.assertEqual(len(policy_rows), 2)
        self.assertEqual([row["epsilon"] for row in policy_rows], [0.05, 0.1])
        self.assertEqual([row["p95_ttft_ms"] for row in policy_rows], [1.0, 2.0])

    def test_pilot_summary_experiment_error_uses_failed_policy_sweep_payload(self) -> None:
        rows = _summary_rows(
            {
                "ok": False,
                "config": "pilot.yaml",
                "rows": {"profiles": 10, "policy": 2},
                "policy_runs": [
                    {
                        "ok": False,
                        "epsilon": 0.05,
                        "delta": 0.05,
                        "memory_budget_mib": 6144.0,
                        "payload": {"error": "policy failed"},
                    },
                ],
            }
        )

        self.assertEqual(rows[0]["error"], "policy failed")

    def test_run_experiment_pilot_accepts_config_argument(self) -> None:
        parser = build_experiment_parser()
        args = parser.parse_args(["pilot-smoke-measured", "--config", "configs/pilot_50.yaml"])
        self.assertEqual(args.config, "configs/pilot_50.yaml")

    def test_pilot_smoke_measured_writes_configured_summary_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "configured_profiles.csv"
            policy_path = Path(tmpdir) / "configured_policy.csv"
            summary_path = Path(tmpdir) / "configured_summary.csv"
            config_path = Path(tmpdir) / "pilot_custom.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  adapters: [full]",
                        "  names: [full_gpu]",
                        "  specs:",
                        "    full_gpu: {exact: true}",
                        "policies:",
                        "  names: [full_lru]",
                        "pilot:",
                        "  epsilons: [0.2]",
                        "  deltas: [0.05]",
                        "outputs:",
                        f"  smoke_profiles: {profile_path}",
                        f"  smoke_policy: {policy_path}",
                        f"  smoke_summary: {summary_path}",
                    ]
                ),
                encoding="utf-8",
            )

            def fake_build(args: argparse.Namespace) -> int:
                write_csv(Path(args.output), [_measurement("e1", "full_gpu", 0.0).to_row()])
                print(json.dumps({"output": args.output, "rows": 1, "summary": {"full_gpu": {"count": 1.0}}}))
                return 0

            def fake_policies(args: argparse.Namespace) -> int:
                print(json.dumps({"output": args.output, "rows": 1, "epsilon": 0.2, "delta": 0.05, "summary": {"full_lru": {"count": 1.0}}}))
                return 0

            with (
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies", side_effect=fake_policies),
            ):
                code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 0)
            self.assertTrue(summary_path.exists())

    def test_pilot_smoke_measured_stops_when_profile_stage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)

            def fake_build(args: argparse.Namespace) -> int:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "output": args.output,
                            "diagnostic_output": str(Path(tmpdir) / "profiles_failed_chunks.csv"),
                            "error": "profile failed",
                            "failures": [{"request_id": "r10", "error": "chunk failed"}],
                        }
                    )
                )
                return 2

            with (
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies") as policy_mock,
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["ok"], "False")
            self.assertEqual(summary_rows[0]["error"], "profile failed")
            self.assertEqual(summary_rows[0]["diagnostic_output"], str(Path(tmpdir) / "profiles_failed_chunks.csv"))
            self.assertEqual(json.loads(summary_rows[0]["failures"])[0]["error"], "chunk failed")
            self.assertNotIn("step", summary_rows[0])
            policy_mock.assert_not_called()

    def test_pilot_smoke_measured_rejects_unmeasured_profile_before_policy_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profiles.csv"
            policy_path = Path(tmpdir) / "policy.csv"
            summary_path = Path(tmpdir) / "summary.csv"
            config_path = Path(tmpdir) / "pilot.yaml"
            _write_pilot_test_config(config_path, profile_path, policy_path, summary_path)
            profiles = FORMAL_PROFILES

            def fake_build(args: argparse.Namespace) -> int:
                rows = [_measurement("e1", profile, 0.0, measured=(profile != "kivi_4bit_residual32")).to_row() for profile in profiles]
                write_csv(Path(args.output), rows)
                print(json.dumps({"output": args.output, "rows": len(rows)}))
                return 0

            with (
                patch("run_experiment.PILOT_PROFILE_OUTPUT", str(profile_path)),
                patch("run_experiment.PILOT_POLICY_OUTPUT", str(policy_path)),
                patch("run_experiment.PILOT_SUMMARY_OUTPUT", str(summary_path)),
                patch("run_experiment.build_profile_table", side_effect=fake_build),
                patch("run_experiment.run_policies") as policy_mock,
            ):
                stream = io.StringIO()
                with redirect_stdout(stream):
                    code = pilot_smoke_measured(argparse.Namespace(config=str(config_path)))

            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(summary_rows[0]["ok"], "False")
            self.assertIn("measured=True", summary_rows[0]["error"])
            self.assertNotIn("step", summary_rows[0])
            policy_mock.assert_not_called()

    def test_run_policies_rejects_dry_run_replay_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dry.csv"
            write_csv(
                path,
                [
                    {
                        "request_id": "r1",
                        "profile": "engine_full_lru",
                        "adapter": "vllm_lru",
                        "ok": True,
                        "measured": False,
                        "output_text": "x",
                        "error": "",
                        "latency_ms": 1.0,
                        "ttft_ms": 1.0,
                        "peak_memory_mib": 10.0,
                        "resident_memory_mib": 10.0,
                        "quality_score": 1.0,
                        "quality_loss": 0.0,
                        "task": "qa",
                        "length_bucket": "short",
                        "split": "eval",
                    }
                ],
            )
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(path),
                output=str(Path(tmpdir) / "policy.csv"),
                profiles=None,
                policies=["full_lru"],
                epsilon=None,
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            self.assertEqual(run_policies(args), 2)

    def test_build_profile_table_import_bad_table_returns_2_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "bad.csv"
            output_path = Path(tmpdir) / "profiles.csv"
            bad_path.write_text("request_id,profile\nr1,full_gpu\n", encoding="utf-8")
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                adapters=None,
                output=str(output_path),
                import_measurements=str(bad_path),
                dry_run=True,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = build_profile_table(args)
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 2)
            self.assertFalse(payload["ok"])
            self.assertNotIn("Traceback", stream.getvalue())

    def test_run_policies_uses_defaults_when_pilot_thresholds_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            config_path.write_text(
                "\n".join(
                    [
                        "profiles:",
                        "  names:",
                        "    - full_gpu",
                        "policies:",
                        "  names:",
                        "    - full_lru",
                    ]
                ),
                encoding="utf-8",
            )
            write_csv(measurements_path, [_measurement("e1", "full_gpu", 0.0).to_row()])
            args = argparse.Namespace(
                config=str(config_path),
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=None,
                policies=None,
                epsilon=None,
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = run_policies(args)
            payload = json.loads(stream.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["epsilon"], 0.2)
            self.assertEqual(payload["delta"], 0.05)

    def test_run_policies_rejects_invalid_numeric_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            write_csv(measurements_path, [_measurement("e1", "full_gpu", 0.0).to_row()])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["full_lru"],
                epsilon="bad",
                delta=None,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with redirect_stdout(stream):
                code = run_policies(args)
            self.assertEqual(code, 2)
            self.assertIn("epsilon", stream.getvalue())

    def test_run_command_does_not_catch_keyerror_or_indexerror(self) -> None:
        with self.assertRaises(KeyError):
            run_command(lambda args: (_ for _ in ()).throw(KeyError("bug")), argparse.Namespace())
        with self.assertRaises(IndexError):
            run_command(lambda args: (_ for _ in ()).throw(IndexError("bug")), argparse.Namespace())

    def test_policy_decide_exception_records_failure_and_continues(self) -> None:
        class BrokenPolicy(Policy):
            name = "broken"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                raise RuntimeError(f"cannot decide {request.request_id}")

        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            rows = [
                _measurement("c1", "full_gpu", 0.0),
                _measurement("e1", "full_gpu", 0.0),
                _measurement("e2", "full_gpu", 0.0),
            ]
            rows = [
                row if row.request_id == "c1" else ProfileMeasurement(
                    request_id=row.request_id,
                    profile=row.profile,
                    adapter=row.adapter,
                    ok=row.ok,
                    measured=row.measured,
                    output_text=row.output_text,
                    error=row.error,
                    latency_ms=row.latency_ms,
                    ttft_ms=row.ttft_ms,
                    peak_memory_mib=row.peak_memory_mib,
                    resident_memory_mib=row.resident_memory_mib,
                    quality_score=row.quality_score,
                    quality_loss=row.quality_loss,
                    extra={**row.extra, "split": "eval"},
                )
                for row in rows
            ]
            write_csv(measurements_path, [row.to_row() for row in rows])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["broken"],
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=None,
                allow_dry_run_replay=False,
            )
            with patch("run_run_policies.build_policies", return_value=[BrokenPolicy()]):
                code = run_policies(args)
            self.assertEqual(code, 1)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["ok"] == "False" for record in records))
            self.assertIn("cannot decide e1", records[0]["error"])

    def test_policy_replay_missing_profile_records_failure_and_continues_other_policy(self) -> None:
        class MissingProfilePolicy(Policy):
            name = "missing_profile_policy"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                return Action(profile="kivi_4bit", reason="missing profile test")

        class ExactPolicy(Policy):
            name = "exact_policy"

            def decide(self, request: Request, cache_state, device_state) -> Action:
                return Action(profile="full_gpu", reason="exact test")

        with tempfile.TemporaryDirectory() as tmpdir:
            measurements_path = Path(tmpdir) / "measured.csv"
            output_path = Path(tmpdir) / "policy.csv"
            rows = [
                ProfileMeasurement(
                    request_id="e1",
                    profile="full_gpu",
                    adapter="full",
                    ok=True,
                    measured=True,
                    output_text="x",
                    ttft_ms=1.0,
                    peak_memory_mib=1.0,
                    quality_loss=0.0,
                    extra={"task": "qa", "length_bucket": "short", "split": "eval"},
                )
            ]
            write_csv(measurements_path, [row.to_row() for row in rows])
            args = argparse.Namespace(
                config="configs/pilot.yaml",
                measurements=str(measurements_path),
                output=str(output_path),
                profiles=["full_gpu"],
                policies=["missing_profile_policy", "exact_policy"],
                epsilon=0.2,
                delta=0.05,
                memory_budget_mib=None,
                use_pandas_replay=False,
                allow_dry_run_replay=False,
            )
            stream = io.StringIO()
            with patch("run_run_policies.build_policies", return_value=[MissingProfilePolicy(), ExactPolicy()]), redirect_stdout(stream):
                code = run_policies(args)
            self.assertEqual(code, 1)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([record["policy"] for record in records], ["missing_profile_policy", "exact_policy"])
            self.assertEqual(records[0]["ok"], "False")
            self.assertIn("缺少回放数据", records[0]["error"])
            self.assertEqual(records[1]["ok"], "True")

    def test_with_quality_uses_full_gpu_baseline_only(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_gpu", "full", True, True, output_text="alpha beta", extra={"task": "qa"}),
            ProfileMeasurement("r1", "full_cpu", "full", True, True, output_text="different exact", extra={"task": "qa"}),
            ProfileMeasurement("r1", "kivi_4bit_residual32", "kivi", True, True, output_text="alpha", extra={"task": "qa"}),
        ]
        updated = with_quality(rows, {"full_gpu", "full_cpu", "recompute"})
        by_profile = {row.profile: row for row in updated}
        self.assertEqual(by_profile["full_gpu"].quality_loss, 0.0)
        self.assertEqual(by_profile["full_cpu"].quality_loss, 0.0)
        self.assertLess(by_profile["kivi_4bit_residual32"].quality_loss, 1.0)

    def test_with_quality_requires_full_gpu_for_lossy_quality(self) -> None:
        rows = [
            ProfileMeasurement("r1", "full_cpu", "full", True, True, output_text="alpha beta", extra={"task": "qa"}),
            ProfileMeasurement("r1", "kivi_4bit_residual32", "kivi", True, True, output_text="alpha", extra={"task": "qa"}),
        ]
        updated = with_quality(rows, {"full_gpu", "full_cpu", "recompute"})
        by_profile = {row.profile: row for row in updated}
        self.assertEqual(by_profile["full_cpu"].quality_loss, 0.0)
        self.assertIsNone(by_profile["kivi_4bit_residual32"].quality_loss)

    def test_quality_metrics_tolerate_none_but_compute_missing_as_full_loss(self) -> None:
        self.assertEqual(normalized_exact_match_loss(None, ""), 0.0)
        self.assertEqual(token_f1_loss(None, ""), 0.0)
        self.assertEqual(rouge_l_loss(None, ""), 0.0)
        loss, metrics = compute_quality_loss("qa", None, "answer")
        self.assertEqual(loss, 1.0)
        self.assertEqual(metrics, {"em": 1.0, "f1": 1.0, "rouge_l": 1.0})
        loss, metrics = compute_quality_loss("summary", "candidate", None)
        self.assertEqual(loss, 1.0)
        self.assertEqual(metrics["rouge_l"], 1.0)
        loss, metrics = compute_quality_loss("unknown", "same", "same")
        self.assertEqual(loss, 0.0)
        self.assertEqual(metrics["em"], 0.0)

    def test_with_quality_marks_missing_candidate_as_full_loss_and_requires_full_gpu_baseline(self) -> None:
        rows = [
            _measurement("r1", "full_gpu", 0.0),
            ProfileMeasurement(
                request_id="r1",
                profile="kivi_4bit",
                adapter="kivi",
                ok=True,
                measured=True,
                output_text="",
                ttft_ms=1.0,
                peak_memory_mib=1.0,
                quality_loss=None,
                extra={"task": "qa", "length_bucket": "short", "split": "eval"},
            ),
            ProfileMeasurement(
                request_id="r2",
                profile="kivi_4bit",
                adapter="kivi",
                ok=True,
                measured=True,
                output_text="",
                ttft_ms=1.0,
                peak_memory_mib=1.0,
                quality_loss=None,
                extra={"task": "qa", "length_bucket": "short", "split": "eval", "reference": "answer"},
            ),
        ]
        updated = {row.request_id: row for row in with_quality(rows, {"full_gpu"})}
        self.assertEqual(updated["r1"].quality_loss, 1.0)
        self.assertEqual(updated["r1"].quality_score, 0.0)
        self.assertIsNone(updated["r2"].quality_loss)
        self.assertIsNone(updated["r2"].quality_score)

    def test_aal_wilson_marks_and_releases_unsafe_profile(self) -> None:
        state = AALState(epsilon=0.2, delta=0.05, window_size=5)
        for index in range(5):
            state.record(AuditSample(f"bad{index}", "kivi_4bit", 0.1, 0.9, 0.2))
        self.assertIn("kivi_4bit", state.unsafe_profiles)
        state.calibration_update("kivi_4bit")
        self.assertNotIn("kivi_4bit", state.unsafe_profiles)

    def test_wilson_empty_window_is_stable_zero_upper_bound(self) -> None:
        detector = WilsonDriftDetector(epsilon=0.2, delta=0.05)
        self.assertEqual(detector.wilson_upper(), 0.0)


if __name__ == "__main__":
    unittest.main()
