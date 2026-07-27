from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.common import StatsPolicy


class UtilityDynamicPolicy(StatsPolicy):
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
        super().__init__("utility_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        self.memory_weight = memory_weight
        self.loss_weight = loss_weight

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        best_profile = self._fallback_profile()
        best_score = inf
        for profile in self._candidate_profiles():
            pred_loss = self.predictor.predict_loss(request, profile)
            score = self._ttft_or_inf(profile) + self.memory_weight * self._memory_or_inf(profile) + self.loss_weight * pred_loss
            if score < best_score:
                best_profile = profile
                best_score = score
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, best_profile)
        return Action(
            profile=best_profile,
            reason="utility_dynamic",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=reason,
        )
