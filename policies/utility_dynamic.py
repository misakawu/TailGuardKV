from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy
from policies.common import PolicyContext


class UtilityDynamicPolicy(Policy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
        memory_weight: float = 0.05,
        loss_weight: float = 1000.0,
    ) -> None:
        self.name = "utility_dynamic"
        self.placeholder = False
        self.oracle = False
        self.context = PolicyContext(list(calibration_measurements), profiles, epsilon, delta, exact_profiles, memory_budget_mib)
        self.memory_weight = memory_weight
        self.loss_weight = loss_weight

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        best_profile = self.context.fallback_profile()
        best_score = inf
        for profile in self.context.candidate_profiles(include_exact=True):
            pred_loss = 0.0 if profile in self.context.exact_profiles else self.context.predictor.predict_loss(request, profile)
            score = (
                self.context.ttft_or_inf(profile)
                + self.memory_weight * self.context.memory_or_inf(profile)
                + self.loss_weight * pred_loss
            )
            if score < best_score:
                best_profile = profile
                best_score = score
        pred_loss, risk_upper, safe, reason = self.context.predict_and_guard(request, best_profile)
        return Action(
            profile=best_profile,
            reason="utility_dynamic",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.context.epsilon,
            delta=self.context.delta,
            fallback_reason=reason,
        )
