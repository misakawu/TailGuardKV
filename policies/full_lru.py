from __future__ import annotations

from math import isfinite

from run_util.core_types import Action, ActionDecision, CacheState, Request
from policies.base import StaticProfilePolicy


class FullLRUPolicy(StaticProfilePolicy):
    def __init__(self, profile: str = "full_gpu") -> None:
        super().__init__(profile, name="full_lru")

    def decide(self, request: Request, cache_state: CacheState | None, device_state) -> Action:
        if cache_state is None:
            return self._finalize_action(
                ActionDecision(
                    profile=self.profile,
                    reason="full precision exact profile",
                    mode="exact",
                    projected_memory_mib=0.0,
                    budget_hit=False,
                )
            )
        projected_memory = self._projected_memory(request, cache_state)
        budget_hit = False
        if isfinite(cache_state.global_budget_mib):
            budget_hit = projected_memory > cache_state.global_budget_mib
        return self._finalize_action(
            ActionDecision(
                profile=self.profile,
                reason="full precision exact profile",
                mode="exact",
                projected_memory_mib=projected_memory,
                budget_hit=budget_hit,
            )
        )

    def _projected_memory(self, request: Request, cache_state: CacheState) -> float:
        if not request.session_id:
            return cache_state.global_resident_kv_mib
        current_profile = cache_state.get_current_profile(request.session_id) or self.profile
        current_resident = cache_state.get_resident_kv(request.session_id, current_profile)
        incremental = current_resident
        if incremental <= 0.0:
            cumulative = cache_state.get_cumulative_kv(request.session_id, current_profile)
            incremental = cumulative if cumulative > 0.0 else 0.0
        if incremental <= 0.0:
            return cache_state.global_resident_kv_mib
        projected_resident = current_resident + incremental
        global_resident = cache_state.global_resident_kv_mib
        if global_resident <= 0.0:
            global_resident = sum(
                cache_state.get_resident_kv(session_id, profile)
                for session_id, profile in cache_state.session_current_profile.items()
            )
        return max(0.0, global_resident - current_resident + projected_resident)
