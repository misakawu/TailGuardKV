from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import inf

from calibration.conformal import ConformalGuard
from calibration.predictor import MetadataOnlyRiskPredictor
from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request


@dataclass(frozen=True)
class ProfileStats:
    profile: str
    count: int
    known_loss_count: int
    mean_loss: float | None
    violation_rate: float | None
    p95_ttft_ms: float | None
    p95_peak_memory_mib: float | None


class Policy(ABC):
    name: str
    placeholder: bool = False
    oracle: bool = False

    @abstractmethod
    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        ...


class StaticProfilePolicy(Policy):
    def __init__(self, profile: str, name: str | None = None, placeholder: bool = False) -> None:
        self.profile = profile
        self.name = name or f"static_{profile}"
        self.placeholder = placeholder

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="static profile baseline")


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
        memory_budget_mib: float = float("inf"),
    ) -> None:
        self.name = name
        self.profiles = profiles
        self.epsilon = epsilon
        self.delta = delta
        self.exact_profiles = exact_profiles
        self.placeholder = placeholder
        self.memory_budget_mib = memory_budget_mib
        self.stats = _profile_stats(calibration_measurements, profiles, epsilon, exact_profiles)
        self.predictor = MetadataOnlyRiskPredictor(list(calibration_measurements))
        self.guard = ConformalGuard(
            epsilon=epsilon,
            delta=delta,
            exact_profiles=exact_profiles,
            calibration_rows=list(calibration_measurements),
        )

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
        if stat is None or stat.p95_peak_memory_mib is None:
            return inf
        return stat.p95_peak_memory_mib

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

    def _predict_and_guard(self, request: Request, profile: str) -> tuple[float, float, bool, str]:
        pred_loss = self.predictor.predict_loss(request, profile)
        risk_upper = self.guard.risk_upper(request, profile, pred_loss)
        safe = risk_upper <= self.epsilon or profile in self.exact_profiles
        reason = "exact fallback" if profile in self.exact_profiles else ("calibrated safe" if safe else "calibrated unsafe")
        return pred_loss, risk_upper, safe, reason


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
            p95_peak_memory_mib=(_percentile(memories, 0.95) if memories else None),
        )
    return stats


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value != inf)
    if not finite_values:
        return inf
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]
