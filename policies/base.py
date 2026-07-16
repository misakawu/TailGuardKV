from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request


@dataclass(frozen=True)
class ProfileStats:
    profile: str
    count: int
    known_loss_count: int
    mean_loss: float | None
    violation_rate: float | None
    p95_ttft_ms: float | None
    mean_peak_memory_mib: float | None


class Policy(ABC):
    """所有 baseline 和 TailGuardKV 策略必须实现同一个决策接口。"""

    name: str
    placeholder: bool = False
    oracle: bool = False

    @abstractmethod
    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        ...


class StaticProfilePolicy(Policy):
    """固定 profile baseline。"""

    def __init__(self, profile: str, name: str | None = None, placeholder: bool = False) -> None:
        self.profile = profile
        self.name = name or f"static_{profile}"
        self.placeholder = placeholder

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="静态 profile baseline")


class FullLRUPolicy(StaticProfilePolicy):
    def __init__(self) -> None:
        super().__init__("full_gpu", name="full_lru")


class StatsPolicy(Policy):
    def __init__(
        self,
        name: str,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        placeholder: bool = True,
    ) -> None:
        self.name = name
        self.profiles = profiles
        self.epsilon = epsilon
        self.delta = delta
        self.exact_profiles = exact_profiles
        self.placeholder = placeholder
        self.stats = _profile_stats(calibration_measurements, profiles, epsilon, exact_profiles)

    def _fallback_profile(self) -> str:
        for profile in ("full_gpu", "full_cpu", "recompute"):
            if profile in self.profiles:
                return profile
        return self.profiles[0]

    def _loss_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if profile in self.exact_profiles:
            return 0.0 if stat and stat.known_loss_count > 0 else inf
        if stat is None or stat.mean_loss is None:
            return inf
        return stat.mean_loss

    def _ttft_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if stat is None or stat.p95_ttft_ms is None:
            return inf
        return stat.p95_ttft_ms

    def _memory_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if stat is None or stat.mean_peak_memory_mib is None:
            return inf
        return stat.mean_peak_memory_mib

    def _best_profile(self, use_tail_constraint: bool) -> str:
        best_profile = self._fallback_profile()
        best_score = inf
        for profile in self.profiles:
            stat = self.stats.get(profile)
            if stat is None or stat.known_loss_count == 0:
                continue
            mean_loss = self._loss_or_inf(profile)
            violation = stat.violation_rate if stat.violation_rate is not None else inf
            if mean_loss > self.epsilon:
                continue
            if use_tail_constraint and violation > self.delta:
                continue
            score = self._ttft_or_inf(profile)
            if score < best_score:
                best_profile = profile
                best_score = score
        return best_profile


class StaticBestPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        super().__init__("static_best", calibration_measurements, profiles, epsilon, delta, exact_profiles)
        self.profile = self._best_profile(use_tail_constraint=False)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="校准集聚合质量约束下的固定 profile")


class StaticSafePolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        super().__init__("static_safe", calibration_measurements, profiles, epsilon, delta, exact_profiles)
        self.profile = self._best_profile(use_tail_constraint=True)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="校准集 tail SLO 约束下的固定 profile")


class UtilityDynamicPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        super().__init__("utility_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        best_profile = self._fallback_profile()
        best_score = inf
        for profile in self.profiles:
            loss = self._loss_or_inf(profile)
            score = self._ttft_or_inf(profile) + 0.05 * self._memory_or_inf(profile) + 1000.0 * loss
            if score < best_score:
                best_profile = profile
                best_score = score
        return Action(profile=best_profile, reason="占位 predictor：仅使用校准集聚合统计")


class UncalibratedDynamicPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        super().__init__("uncalibrated_dynamic", calibration_measurements, profiles, epsilon, delta, exact_profiles)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        for profile in sorted(self.profiles, key=self._ttft_or_inf):
            if self._loss_or_inf(profile) <= self.epsilon:
                return Action(profile=profile, reason="占位点预测：校准统计满足阈值")
        return Action(profile=self._fallback_profile(), reason="校准统计无安全近似动作，回退 exact")


class TailGuardPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> None:
        super().__init__("tailguard", calibration_measurements, profiles, epsilon, delta, exact_profiles)

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        safe_profiles = [
            profile
            for profile in self.profiles
            if profile in self.exact_profiles or self._loss_or_inf(profile) <= self.epsilon
        ]
        if not safe_profiles:
            return Action(profile=self._fallback_profile(), reason="占位 guard 安全集为空，回退 exact")
        return Action(
            profile=min(safe_profiles, key=self._ttft_or_inf),
            reason="占位 guard：仅使用校准集聚合统计，未查评估真值",
        )


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
        self.placeholder = True
        self.oracle = True
        self.measurements = {
            (measurement.request_id, measurement.profile): measurement for measurement in measurements
        }

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        feasible = []
        for profile in self.profiles:
            measurement = self.measurements.get((request.request_id, profile))
            if measurement is None or measurement.quality_loss is None:
                continue
            if measurement.quality_loss <= self.epsilon:
                feasible.append(profile)
        if not feasible:
            feasible = [self._fallback_profile()]
        return Action(profile=min(feasible, key=self._ttft_or_inf), reason="oracle 上界：允许查评估真值")

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


def _profile_stats(
    measurements: Iterable[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    exact_profiles: set[str],
) -> dict[str, ProfileStats]:
    grouped: dict[str, list[ProfileMeasurement]] = {profile: [] for profile in profiles}
    for measurement in measurements:
        if measurement.profile in grouped:
            grouped[measurement.profile].append(measurement)

    stats: dict[str, ProfileStats] = {}
    for profile, rows in grouped.items():
        losses = [row.quality_loss for row in rows if row.quality_loss is not None]
        if profile in exact_profiles and not losses:
            losses = [0.0 for row in rows if row.ok and row.measured and row.output_text]
        ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
        memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
        stats[profile] = ProfileStats(
            profile=profile,
            count=len(rows),
            known_loss_count=len(losses),
            mean_loss=(sum(losses) / len(losses) if losses else None),
            violation_rate=(sum(1 for loss in losses if loss > epsilon) / len(losses) if losses else None),
            p95_ttft_ms=(_percentile(ttfts, 0.95) if ttfts else None),
            mean_peak_memory_mib=(sum(memories) / len(memories) if memories else None),
        )
    return stats


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value != inf)
    if not finite_values:
        return inf
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]
