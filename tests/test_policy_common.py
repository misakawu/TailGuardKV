from __future__ import annotations

import math
import unittest

from core_types import ProfileMeasurement, Request
from policies.common import PolicyContext, profile_stats, percentile


def row(request_id: str, profile: str, loss: float | None, ttft: float, memory: float) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        profile=profile,
        adapter="test",
        ok=True,
        measured=True,
        output_text="x",
        ttft_ms=ttft,
        peak_memory_mib=memory,
        resident_memory_mib=memory,
        quality_loss=loss,
        extra={"task": "qa", "length_bucket": "short", "split": "calibration"},
    )


class PolicyCommonTest(unittest.TestCase):
    def test_profile_stats_compute_mean_violation_ttft_and_memory(self) -> None:
        stats = profile_stats(
            [
                row("c1", "engine_full_lru", None, 30.0, 900.0),
                row("c1", "compress_light", 0.1, 8.0, 200.0),
                row("c2", "compress_light", 0.3, 9.0, 250.0),
            ],
            ["engine_full_lru", "compress_light", "missing"],
            epsilon=0.2,
            exact_profiles={"engine_full_lru"},
        )
        self.assertEqual(stats["engine_full_lru"].known_loss_count, 1)
        self.assertEqual(stats["engine_full_lru"].mean_loss, 0.0)
        self.assertAlmostEqual(stats["compress_light"].mean_loss, 0.2)
        self.assertEqual(stats["compress_light"].violation_rate, 0.5)
        self.assertEqual(stats["compress_light"].p95_ttft_ms, 9.0)
        self.assertEqual(stats["compress_light"].p95_peak_memory_mib, 250.0)
        self.assertIsNone(stats["missing"].mean_loss)

    def test_profile_stats_does_not_treat_empty_exact_output_as_zero_loss(self) -> None:
        stats = profile_stats(
            [
                ProfileMeasurement(
                    request_id="c1",
                    profile="recompute_default",
                    adapter="test",
                    ok=True,
                    measured=True,
                    output_text="",
                    ttft_ms=12.0,
                    peak_memory_mib=300.0,
                    resident_memory_mib=300.0,
                    quality_loss=None,
                    extra={"task": "qa", "length_bucket": "short", "split": "calibration"},
                )
            ],
            ["recompute_default"],
            epsilon=0.2,
            exact_profiles={"recompute_default"},
        )
        self.assertEqual(stats["recompute_default"].known_loss_count, 0)
        self.assertIsNone(stats["recompute_default"].mean_loss)

    def test_policy_context_fallback_prefers_engine_full_lru_then_exact(self) -> None:
        context = PolicyContext(
            calibration_rows=[],
            profiles=["compress_light", "engine_full_lru"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        self.assertEqual(context.fallback_profile(), "engine_full_lru")

        context = PolicyContext(
            calibration_rows=[],
            profiles=["compress_light", "full_gpu"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"full_gpu"},
        )
        self.assertEqual(context.fallback_profile(), "full_gpu")

    def test_policy_context_filters_memory_budget_and_keeps_fallback(self) -> None:
        rows = [
            row("c1", "engine_full_lru", 0.0, 30.0, 900.0),
            row("c1", "compress_light", 0.1, 8.0, 200.0),
            row("c1", "compress_heavy", 0.2, 5.0, 100.0),
        ]
        context = PolicyContext(
            calibration_rows=rows,
            profiles=["engine_full_lru", "compress_light", "compress_heavy"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
            memory_budget_mib=150.0,
        )
        self.assertEqual(context.candidate_profiles(include_exact=False), ["compress_heavy"])

        tight_context = PolicyContext(
            calibration_rows=rows,
            profiles=["engine_full_lru", "compress_light"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
            memory_budget_mib=50.0,
        )
        self.assertEqual(tight_context.candidate_profiles(include_exact=False), ["engine_full_lru"])

    def test_predict_and_guard_reports_exact_fallback_as_safe(self) -> None:
        context = PolicyContext(
            calibration_rows=[row("c1", "engine_full_lru", 0.0, 30.0, 900.0)],
            profiles=["engine_full_lru"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        pred_loss, risk_upper, safe, reason = context.predict_and_guard(request, "engine_full_lru")
        self.assertEqual(pred_loss, 0.0)
        self.assertEqual(risk_upper, 0.0)
        self.assertTrue(safe)
        self.assertEqual(reason, "exact fallback")

    def test_predict_and_guard_short_circuits_non_engine_full_lru_exact_profile(self) -> None:
        context = PolicyContext(
            calibration_rows=[
                row("c1", "recompute_default", 0.4, 13.0, 300.0),
            ],
            profiles=["recompute_default"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"recompute_default"},
        )
        request = Request("e2", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        pred_loss, risk_upper, safe, reason = context.predict_and_guard(request, "recompute_default")
        self.assertEqual(pred_loss, 0.0)
        self.assertEqual(risk_upper, 0.0)
        self.assertTrue(safe)
        self.assertEqual(reason, "exact fallback")

    def test_percentile_empty_returns_inf_for_policy_costs(self) -> None:
        self.assertTrue(math.isinf(percentile([], 0.95)))
        self.assertEqual(percentile([10.0, 20.0], 0.95), 20.0)
