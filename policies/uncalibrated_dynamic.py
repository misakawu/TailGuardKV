from __future__ import annotations

from collections.abc import Iterable

from run_util.core_types import Action, ActionDecision, CacheState, DeviceState, ProfileMeasurement, Request
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
        candidate_safe_count = float(self._candidate_safe_count(request, cache_state))
        budget_filtered = False
        for profile in [profile for profile in self.profiles if profile not in self.exact_profiles]:
            if not self._within_memory_budget(profile, request, cache_state):
                budget_filtered = True
                continue
            pred_loss = self.predictor.predict_loss(request, profile)
            if pred_loss <= self.epsilon:
                candidate = self._candidate_action(request, profile, cache_state)
                return self._finalize_action(
                    ActionDecision(
                        profile=profile,
                        reason="uncalibrated_dynamic",
                        mode="lossy",
                        projected_memory_mib=candidate.projected_memory_mib,
                        pred_loss=pred_loss,
                        risk_upper=candidate.risk_upper,
                        safe=candidate.safe,
                        epsilon=self.epsilon,
                        delta=self.delta,
                        fallback_reason="点预测阈值通过",
                        candidate_safe_count=candidate_safe_count,
                        budget_hit=budget_filtered,
                    )
                )
        fallback = self._best_exact_candidate(request, cache_state)
        return self._finalize_action(
            ActionDecision(
                profile=fallback.profile,
                reason="uncalibrated_dynamic",
                mode="exact",
                projected_memory_mib=fallback.projected_memory_mib,
                pred_loss=fallback.pred_loss,
                risk_upper=fallback.risk_upper,
                safe=fallback.safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=fallback.reason,
                candidate_safe_count=candidate_safe_count,
                budget_hit=budget_filtered,
            )
        )
