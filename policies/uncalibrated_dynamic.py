from __future__ import annotations

from collections.abc import Iterable

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class UncalibratedDynamicPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        super().__init__("uncalibrated_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        for profile in sorted(self.profiles, key=self._ttft_or_inf):
            pred_loss = self.predictor.predict_loss(request, profile)
            if pred_loss <= self.epsilon:
                risk_upper = self.guard.risk_upper(request, profile, pred_loss)
                return Action(
                    profile=profile,
                    reason="uncalibrated_dynamic",
                    pred_loss=pred_loss,
                    risk_upper=risk_upper,
                    safe=risk_upper <= self.epsilon or profile in self.exact_profiles,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    fallback_reason="点预测阈值通过",
                )
        fallback = self._fallback_profile()
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, fallback)
        return Action(
            profile=fallback,
            reason="uncalibrated_dynamic",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=reason,
        )
