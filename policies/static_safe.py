from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy
from policies.common import PolicyContext


class StaticSafePolicy(Policy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        self.name = "static_safe"
        self.placeholder = False
        self.oracle = False
        self.context = PolicyContext(list(calibration_measurements), profiles, epsilon, delta, exact_profiles, memory_budget_mib)
        self.profile = self._select_profile()

    def _select_profile(self) -> str:
        best_profile = ""
        best_ttft = inf
        for profile in self.context.candidate_profiles(include_exact=False):
            stat = self.context.stats.get(profile)
            if stat is None or stat.known_loss_count == 0:
                continue
            if self.context.loss_or_inf(profile) > self.context.epsilon:
                continue
            violation_rate = stat.violation_rate if stat.violation_rate is not None else inf
            if violation_rate > self.context.delta:
                continue
            ttft = self.context.ttft_or_inf(profile)
            if ttft < best_ttft:
                best_profile = profile
                best_ttft = ttft
        if best_profile:
            return best_profile
        return self.context.fallback_profile()

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        pred_loss, risk_upper, safe, reason = self.context.predict_and_guard(request, self.profile)
        return Action(
            profile=self.profile,
            reason="static_safe",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.context.epsilon,
            delta=self.context.delta,
            fallback_reason=reason,
        )
