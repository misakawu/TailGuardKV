from __future__ import annotations

from collections.abc import Iterable

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy, _percentile


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
        chosen = min(feasible, key=self._ttft_or_inf)
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
        for profile in ("full_gpu", "full_cpu", "recompute"):
            if profile in self.profiles:
                return profile
        return self.profiles[0]

    def _ttft_or_inf(self, profile: str) -> float:
        values = [
            measurement.ttft_ms
            for (request_id, row_profile), measurement in self.measurements.items()
            if row_profile == profile and measurement.ttft_ms is not None
        ]
        return _percentile(values, 0.95)
