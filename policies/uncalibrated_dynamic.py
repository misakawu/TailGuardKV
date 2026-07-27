from __future__ import annotations

from collections.abc import Iterable

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy
from policies.common import PolicyContext


class UncalibratedDynamicPolicy(Policy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        self.name = "uncalibrated_dynamic"
        self.placeholder = False
        self.oracle = False
        self.context = PolicyContext(list(calibration_measurements), profiles, epsilon, delta, exact_profiles, memory_budget_mib)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        for profile in sorted(self.context.candidate_profiles(include_exact=False), key=self.context.ttft_or_inf):
            pred_loss = self.context.predictor.predict_loss(request, profile)
            if pred_loss <= self.context.epsilon:
                risk_upper = self.context.guard.risk_upper(request, profile, pred_loss)
                return Action(
                    profile=profile,
                    reason="uncalibrated_dynamic",
                    pred_loss=pred_loss,
                    risk_upper=risk_upper,
                    safe=risk_upper <= self.context.epsilon or profile in self.context.exact_profiles,
                    epsilon=self.context.epsilon,
                    delta=self.context.delta,
                    fallback_reason="point prediction accepted",
                )
        fallback = self.context.fallback_profile()
        pred_loss, risk_upper, safe, reason = self.context.predict_and_guard(request, fallback)
        return Action(
            profile=fallback,
            reason="uncalibrated_dynamic",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.context.epsilon,
            delta=self.context.delta,
            fallback_reason=reason,
        )
