from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class TailGuardPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        super().__init__("tailguard", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)

    def _safe_candidates(self, request: Request) -> list[str]:
        candidates: list[str] = []
        for profile in self.profiles:
            pred_loss = self.predictor.predict_loss(request, profile)
            risk_upper = self.guard.risk_upper(request, profile, pred_loss)
            if profile in self.exact_profiles or risk_upper <= self.epsilon:
                candidates.append(profile)
        return candidates

    def _exact_fallback_profile(self) -> str | None:
        for profile in ("full_gpu", "full_cpu", "recompute"):
            if profile in self.profiles and profile in self.exact_profiles:
                return profile
        for profile in self.profiles:
            if profile in self.exact_profiles:
                return profile
        return None

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        candidates = self._safe_candidates(request)
        if not candidates:
            fallback = self._exact_fallback_profile()
            if fallback is None:
                raise RuntimeError("tailguard requires at least one exact profile when no safe lossy candidate exists")
            pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, fallback)
            return Action(
                profile=fallback,
                reason="tailguard",
                pred_loss=pred_loss,
                risk_upper=risk_upper,
                safe=safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=f"no safe candidate; {reason}",
            )
        feasible_lossy = [
            profile
            for profile in candidates
            if profile not in self.exact_profiles
            and self._memory_or_inf(profile) <= self.memory_budget_mib
        ]
        if not feasible_lossy:
            fallback = self._exact_fallback_profile()
            if fallback is None:
                raise RuntimeError("tailguard requires at least one exact profile when no feasible candidate exists")
            pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, fallback)
            return Action(
                profile=fallback,
                reason="tailguard",
                pred_loss=pred_loss,
                risk_upper=risk_upper,
                safe=safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=f"no feasible lossy candidate within memory budget; {reason}",
            )
        best_profile = min(feasible_lossy, key=self._ttft_or_inf)
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, best_profile)
        return Action(
            profile=best_profile,
            reason="tailguard",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=reason,
        )
