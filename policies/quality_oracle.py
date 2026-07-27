from __future__ import annotations

from collections.abc import Iterable

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy
from policies.common import percentile


class QualityOraclePolicy(Policy):
    def __init__(
        self,
        measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        self.name = "quality_oracle"
        self.profiles = profiles
        self.epsilon = epsilon
        self.delta = delta
        self.exact_profiles = exact_profiles
        self.placeholder = False
        self.oracle = True
        self.measurements = {
            (measurement.request_id, measurement.profile): measurement for measurement in measurements
        }

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        feasible: list[str] = []
        for profile in self.profiles:
            measurement = self.measurements.get((request.request_id, profile))
            if measurement is None or measurement.quality_loss is None:
                continue
            if measurement.quality_loss <= self.epsilon:
                feasible.append(profile)
        if not feasible:
            feasible = [self._fallback_profile()]
        chosen = min(feasible, key=lambda profile: self._request_ttft_or_inf(request.request_id, profile))
        measurement = self.measurements.get((request.request_id, chosen))
        return Action(
            profile=chosen,
            reason="oracle 上界：允许查评估真值",
            pred_loss=None if measurement is None else measurement.quality_loss,
            risk_upper=None if measurement is None else measurement.quality_loss,
            safe=True,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason="oracle",
        )

    def _fallback_profile(self) -> str:
        for profile in ("engine_full_lru",):
            if profile in self.profiles:
                return profile
        return self.profiles[0]

    def _ttft_or_inf(self, profile: str) -> float:
        values = [
            measurement.ttft_ms
            for (request_id, row_profile), measurement in self.measurements.items()
            if row_profile == profile and measurement.ttft_ms is not None
        ]
        return percentile(values, 0.95)

    def _request_ttft_or_inf(self, request_id: str, profile: str) -> float:
        measurement = self.measurements.get((request_id, profile))
        if measurement is None or measurement.ttft_ms is None:
            return float("inf")
        return measurement.ttft_ms
