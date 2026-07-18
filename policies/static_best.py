from __future__ import annotations

from collections.abc import Iterable

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class StaticBestPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        super().__init__("static_best", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        self.profile = self._best_profile(use_tail_constraint=False)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, self.profile)
        return Action(
            profile=self.profile,
            reason="static_best",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=reason,
        )
