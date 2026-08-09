from __future__ import annotations

from collections.abc import Iterable

from run_util.core_types import Action, ActionDecision, CacheState, DeviceState, ProfileMeasurement, Request
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
        candidate = self._candidate_action(request, self.profile, cache_state)
        return self._finalize_action(
            ActionDecision(
                profile=self.profile,
                reason="static_best",
                mode="static",
                projected_memory_mib=candidate.projected_memory_mib,
                pred_loss=candidate.pred_loss,
                risk_upper=candidate.risk_upper,
                safe=candidate.safe,
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=candidate.reason,
            )
        )
