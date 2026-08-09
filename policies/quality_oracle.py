from __future__ import annotations

import time
from collections.abc import Iterable
from math import inf

from run_util.core_types import Action, ActionDecision, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy


class QualityOraclePolicy(Policy):
    name = "quality_oracle"
    placeholder = False
    oracle = True

    def __init__(
        self,
        measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        self.measurements = {(row.session_id or "", row.turn_index, row.request_id, row.profile): row for row in measurements}
        self.profiles = profiles
        self.epsilon = epsilon
        self.delta = delta
        self.exact_profiles = exact_profiles

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        started = time.perf_counter()
        feasible = []
        for profile in self.profiles:
            row = self.measurements.get((request.session_id or "", request.turn_index, request.request_id, profile))
            if row is None or row.quality_loss is None:
                continue
            if row.quality_loss <= self.epsilon:
                feasible.append((self._latency(row), profile, row.quality_loss))
        fallback_reason = ""
        if not feasible:
            profile = self._fastest_exact_profile(request.request_id)
            row = self.measurements.get((request.session_id or "", request.turn_index, request.request_id, profile))
            loss = row.quality_loss if row is not None else None
            fallback_reason = "oracle exact fallback"
        else:
            _, profile, loss = min(feasible, key=lambda item: item[0])
        projected_memory = self._projected_memory(request, profile, cache_state)
        return self._finalize_action(
            ActionDecision(
                profile=profile,
                reason="quality oracle",
                mode="exact" if profile in self.exact_profiles else "lossy",
                projected_memory_mib=projected_memory,
                pred_loss=loss,
                risk_upper=loss,
                safe=(loss is None or loss <= self.epsilon),
                epsilon=self.epsilon,
                delta=self.delta,
                fallback_reason=fallback_reason,
                oracle_cost_ms=(time.perf_counter() - started) * 1000,
            )
        )

    @staticmethod
    def _latency(row: ProfileMeasurement) -> float:
        if row.ttft_ms is not None:
            return row.ttft_ms
        if row.latency_ms is not None:
            return row.latency_ms
        return float("inf")

    def _fastest_exact_profile(self, request_id: str) -> str:
        best_profile = ""
        best_latency = float("inf")
        for profile in self.profiles:
            if profile not in self.exact_profiles:
                continue
            matching = [row for key, row in self.measurements.items() if key[2] == request_id and key[3] == profile]
            row = matching[0] if matching else None
            latency = self._latency(row) if row is not None else float("inf")
            if latency < best_latency:
                best_profile = profile
                best_latency = latency
        if best_profile:
            return best_profile
        return self.profiles[0]

    def _projected_memory(self, request: Request, profile: str, cache_state: CacheState) -> float:
        if not request.session_id:
            return inf
        current_profile = cache_state.get_current_profile(request.session_id) or profile
        current = cache_state.get_cumulative_kv(request.session_id, current_profile)
        if current <= 0:
            row = self.measurements.get((request.session_id or "", request.turn_index, request.request_id, profile))
            return float(row.kv_cumulative_mib or row.kv_cache_memory_mib or 0.0) if row is not None else inf
        row = self.measurements.get((request.session_id or "", request.turn_index, request.request_id, profile))
        incremental = float(row.kv_incremental_mib or 0.0) if row is not None else 0.0
        return current + incremental
