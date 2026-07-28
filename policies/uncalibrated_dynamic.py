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
        record_rejected_unsafe: bool = False,
    ) -> None:
        super().__init__("uncalibrated_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        self.record_rejected_unsafe = record_rejected_unsafe

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        candidate_safe_count = float(self._candidate_safe_count(request))
        preferred = sorted(self._candidate_profiles(), key=self._ttft_or_inf)[0]
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, preferred)
        if not safe and preferred not in self.exact_profiles:
            fallback = self._fastest_exact_profile()
            final_pred_loss, final_risk_upper, final_safe, final_reason = self._predict_and_guard(request, fallback)
            return Action(
                profile=fallback,
                reason="uncalibrated_dynamic",
                pred_loss=final_pred_loss,
                risk_upper=final_risk_upper,
                safe=final_safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=f"unsafe lossy fallback: {reason}",
                rejected_profile=preferred if self.record_rejected_unsafe else "",
                rejected_pred_loss=pred_loss if self.record_rejected_unsafe else None,
                rejected_risk_upper=risk_upper if self.record_rejected_unsafe else None,
                candidate_safe_count=candidate_safe_count,
            )

        for profile in sorted(self._candidate_profiles(), key=self._ttft_or_inf):
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
                    candidate_safe_count=candidate_safe_count,
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
            candidate_safe_count=candidate_safe_count,
        )
