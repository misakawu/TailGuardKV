from __future__ import annotations

import unittest

import policies.base as policy_base
from core_types import ProfileMeasurement, Request
from policies.base import Policy
from policies.full_lru import FullLRUPolicy
from policies.registry import build_policies


def measurement(
    request_id: str,
    profile: str,
    loss: float | None,
    ttft: float = 10.0,
    memory: float = 100.0,
) -> ProfileMeasurement:
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


class PolicyRefactorTest(unittest.TestCase):
    def test_stats_policy_superclass_is_removed_from_base_module(self) -> None:
        self.assertFalse(hasattr(policy_base, "StatsPolicy"))

    def test_full_lru_inherits_policy_directly_and_selects_engine_full_lru(self) -> None:
        self.assertEqual(FullLRUPolicy.__mro__[1], Policy)
        policy = FullLRUPolicy()
        action = policy.decide(Request("e1", "qa", "prompt"), None, None)
        self.assertEqual(policy.name, "full_lru")
        self.assertEqual(action.profile, "engine_full_lru")
        self.assertEqual(action.reason, "full precision vLLM LRU")
        self.assertFalse(policy.placeholder)
        self.assertFalse(policy.oracle)

    def test_registry_builds_only_configured_policy_names(self) -> None:
        rows = [measurement("c1", "engine_full_lru", 0.0)]
        policies = build_policies(
            ["full_lru"],
            calibration_measurements=rows,
            oracle_measurements=rows,
            profiles=["engine_full_lru", "compress_light"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        self.assertEqual([policy.name for policy in policies], ["full_lru"])

    def test_static_policy_classes_inherit_policy_directly(self) -> None:
        from policies.static_best import StaticBestPolicy
        from policies.static_safe import StaticSafePolicy

        self.assertEqual(StaticBestPolicy.__mro__[1], Policy)
        self.assertEqual(StaticSafePolicy.__mro__[1], Policy)

    def test_static_best_selects_fastest_lossy_mean_safe_profile(self) -> None:
        policies = build_policies(
            ["static_best"],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("c1", "compress_light", 0.1, ttft=8.0),
                measurement("c2", "compress_light", 0.1, ttft=9.0),
                measurement("c1", "compress_heavy", 0.3, ttft=3.0),
                measurement("c2", "compress_heavy", 0.3, ttft=4.0),
            ],
            oracle_measurements=[],
            profiles=["engine_full_lru", "compress_light", "compress_heavy"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        action = policies[0].decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "compress_light")
        self.assertEqual(action.reason, "static_best")
        self.assertEqual(action.epsilon, 0.2)
        self.assertEqual(action.delta, 0.05)

    def test_static_safe_rejects_high_tail_violation_profile(self) -> None:
        policies = build_policies(
            ["static_safe"],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("c1", "compress_light", 0.1, ttft=8.0),
                measurement("c2", "compress_light", 0.3, ttft=9.0),
                measurement("c1", "offload_default", 0.1, ttft=12.0),
                measurement("c2", "offload_default", 0.1, ttft=13.0),
            ],
            oracle_measurements=[],
            profiles=["engine_full_lru", "compress_light", "offload_default"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        action = policies[0].decide(Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"}), None, None)
        self.assertEqual(action.profile, "offload_default")
        self.assertEqual(action.reason, "static_safe")

    def test_static_policies_fallback_to_exact_when_no_lossy_profile_qualifies(self) -> None:
        policies = build_policies(
            ["static_best", "static_safe"],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("c1", "compress_light", 0.8, ttft=5.0),
            ],
            oracle_measurements=[],
            profiles=["compress_light", "engine_full_lru"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual([policy.decide(request, None, None).profile for policy in policies], ["engine_full_lru", "engine_full_lru"])

    def test_dynamic_policy_classes_inherit_policy_directly(self) -> None:
        from policies.uncalibrated_dynamic import UncalibratedDynamicPolicy
        from policies.utility_dynamic import UtilityDynamicPolicy

        self.assertEqual(UtilityDynamicPolicy.__mro__[1], Policy)
        self.assertEqual(UncalibratedDynamicPolicy.__mro__[1], Policy)

    def test_utility_dynamic_uses_configured_utility_weights_without_tail_filter(self) -> None:
        policies = build_policies(
            [{"type": "utility_dynamic", "memory_weight": 1.0, "loss_weight": 1.0}],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0, memory=10.0),
                measurement("c1", "compress_light", 0.1, ttft=5.0, memory=100.0),
                measurement("c1", "offload_default", 0.12, ttft=8.0, memory=10.0),
            ],
            oracle_measurements=[],
            profiles=["engine_full_lru", "compress_light", "offload_default"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        action = policies[0].decide(request, None, None)
        self.assertEqual(action.profile, "offload_default")
        self.assertEqual(action.reason, "utility_dynamic")

    def test_uncalibrated_dynamic_uses_point_prediction_threshold_only(self) -> None:
        policies = build_policies(
            ["uncalibrated_dynamic"],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("c1", "compress_light", 0.1, ttft=5.0),
                measurement("c1", "compress_heavy", 0.3, ttft=3.0),
            ],
            oracle_measurements=[],
            profiles=["engine_full_lru", "compress_heavy", "compress_light"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        action = policies[0].decide(request, None, None)
        self.assertEqual(action.profile, "compress_light")
        self.assertEqual(action.fallback_reason, "point prediction accepted")

    def test_uncalibrated_dynamic_falls_back_to_exact_when_no_point_safe_lossy_candidate_exists(self) -> None:
        policies = build_policies(
            ["uncalibrated_dynamic"],
            calibration_measurements=[
                measurement("c1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("c1", "compress_heavy", 0.4, ttft=3.0),
            ],
            oracle_measurements=[],
            profiles=["engine_full_lru", "compress_heavy"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        request = Request("e1", "qa", "prompt", metadata={"task": "qa", "length_bucket": "short"})
        self.assertEqual(policies[0].decide(request, None, None).profile, "engine_full_lru")

    def test_quality_oracle_inherits_policy_directly_and_marks_oracle(self) -> None:
        from policies.quality_oracle import QualityOraclePolicy

        self.assertEqual(QualityOraclePolicy.__mro__[1], Policy)
        policy = QualityOraclePolicy(
            measurements=[measurement("e1", "engine_full_lru", 0.0)],
            profiles=["engine_full_lru"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        self.assertTrue(policy.oracle)
        self.assertFalse(policy.placeholder)

    def test_quality_oracle_uses_request_truth_and_ttft(self) -> None:
        policies = build_policies(
            ["quality_oracle"],
            calibration_measurements=[],
            oracle_measurements=[
                measurement("e1", "engine_full_lru", 0.0, ttft=30.0),
                measurement("e1", "compress_light", 0.1, ttft=9.0),
                measurement("e1", "compress_heavy", 0.3, ttft=2.0),
                measurement("e2", "engine_full_lru", 0.0, ttft=30.0),
                measurement("e2", "compress_light", 0.1, ttft=12.0),
                measurement("e2", "compress_heavy", 0.1, ttft=3.0),
            ],
            profiles=["engine_full_lru", "compress_light", "compress_heavy"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        self.assertEqual(policies[0].decide(Request("e1", "qa", "prompt"), None, None).profile, "compress_light")
        self.assertEqual(policies[0].decide(Request("e2", "qa", "prompt"), None, None).profile, "compress_heavy")

    def test_quality_oracle_falls_back_when_no_lossy_profile_is_true_safe(self) -> None:
        policies = build_policies(
            ["quality_oracle"],
            calibration_measurements=[],
            oracle_measurements=[
                measurement("e1", "engine_full_lru", None, ttft=30.0),
                measurement("e1", "compress_light", 0.4, ttft=9.0),
            ],
            profiles=["engine_full_lru", "compress_light"],
            epsilon=0.2,
            delta=0.05,
            exact_profiles={"engine_full_lru"},
        )
        self.assertEqual(policies[0].decide(Request("e1", "qa", "prompt"), None, None).profile, "engine_full_lru")
