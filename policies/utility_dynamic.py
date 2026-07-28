from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


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
        record_rejected_unsafe: bool = False,
    ) -> None:
        super().__init__("utility_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        self.memory_weight = memory_weight
        self.loss_weight = loss_weight
        self.record_rejected_unsafe = record_rejected_unsafe

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
        candidate_safe_count = float(self._candidate_safe_count(request))
        rejected_profile = ""
        rejected_pred_loss = None
        rejected_risk_upper = None
        final_profile = best_profile
        final_pred_loss = pred_loss
        final_risk_upper = risk_upper
        final_safe = safe
        final_reason = reason
        if not safe and best_profile not in self.exact_profiles:
            final_profile = self._fastest_exact_profile()
            final_pred_loss, final_risk_upper, final_safe, final_reason = self._predict_and_guard(request, final_profile)
            final_reason = f"unsafe lossy fallback: {reason}"
            if self.record_rejected_unsafe:
                rejected_profile = best_profile
                rejected_pred_loss = pred_loss
                rejected_risk_upper = risk_upper
        return Action(
            profile=final_profile,
            reason="utility_dynamic",
            pred_loss=final_pred_loss,
            risk_upper=final_risk_upper,
            safe=final_safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=final_reason,
            rejected_profile=rejected_profile,
            rejected_pred_loss=rejected_pred_loss,
            rejected_risk_upper=rejected_risk_upper,
            candidate_safe_count=candidate_safe_count,
        )
