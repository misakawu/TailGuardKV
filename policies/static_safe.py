from __future__ import annotations

from collections.abc import Iterable

from run_util.core_types import Action, ActionDecision, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class StaticSafePolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        super().__init__("static_safe", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        self.profile = self._lowest_empirical_loss_lossy_profile()

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        candidate = self._candidate_action(request, self.profile, cache_state)
        chosen = candidate
        fallback_reason = ""
        if not candidate.exact and not candidate.safe:
            chosen = self._best_exact_candidate(request, cache_state)
            fallback_reason = candidate.reason
        return self._finalize_action(
            ActionDecision(
                profile=chosen.profile,
                reason="static_safe",
                mode="exact" if chosen.exact else "lossy",
                projected_memory_mib=chosen.projected_memory_mib,
                pred_loss=chosen.pred_loss,
                risk_upper=chosen.risk_upper,
                safe=chosen.safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=fallback_reason,
                safety_reason=candidate.reason,
                rejected_profile=(candidate.profile if fallback_reason else ""),
                rejected_pred_loss=(candidate.pred_loss if fallback_reason else None),
                rejected_risk_upper=(candidate.risk_upper if fallback_reason else None),
            )
        )
