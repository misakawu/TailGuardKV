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
