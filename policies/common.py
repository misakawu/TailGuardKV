from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import inf, isfinite

from calibration.conformal import ConformalGuard
from calibration.predictor import MetadataOnlyRiskPredictor
from core_types import ProfileMeasurement, Request
from policies.base import Policy


@dataclass(frozen=True)
class ProfileStats:
    profile: str
    count: int
    known_loss_count: int
    mean_loss: float | None
    violation_rate: float | None
    p95_ttft_ms: float | None
    p95_peak_memory_mib: float | None


def percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value != inf)
    if not finite_values:
        return inf
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]


def profile_stats(
    measurements: Iterable[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    exact_profiles: set[str],
) -> dict[str, ProfileStats]:
    grouped: defaultdict[str, list[ProfileMeasurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.profile in profiles:
            grouped[measurement.profile].append(measurement)

    stats: dict[str, ProfileStats] = {}
    for profile in profiles:
        rows = grouped[profile]
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
            p95_ttft_ms=(percentile(ttfts, 0.95) if ttfts else None),
            p95_peak_memory_mib=(percentile(memories, 0.95) if memories else None),
        )
    return stats


@dataclass
class PolicyContext:
    calibration_rows: list[ProfileMeasurement]
    profiles: list[str]
    epsilon: float
    delta: float
    exact_profiles: set[str]
    memory_budget_mib: float = float("inf")
    stats: dict[str, ProfileStats] = field(init=False)
    predictor: MetadataOnlyRiskPredictor = field(init=False)
    guard: ConformalGuard = field(init=False)

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError("policy 至少需要一个 profile")
        self.memory_budget_mib = self.memory_budget_mib if self.memory_budget_mib > 0 else inf
        self.stats = profile_stats(self.calibration_rows, self.profiles, self.epsilon, self.exact_profiles)
        self.predictor = MetadataOnlyRiskPredictor(self.calibration_rows)
        self.guard = ConformalGuard(
            epsilon=self.epsilon,
            delta=self.delta,
            exact_profiles=self.exact_profiles,
            profiles=self.profiles,
            calibration_rows=self.calibration_rows,
        )

    def fallback_profile(self) -> str:
        if "engine_full_lru" in self.profiles:
            return "engine_full_lru"
        for profile in self.profiles:
            if profile in self.exact_profiles:
                return profile
        return self.profiles[0]

    def candidate_profiles(self, include_exact: bool = True) -> list[str]:
        candidates = [
            profile
            for profile in self.profiles
            if (include_exact or profile not in self.exact_profiles) and self.within_memory_budget(profile)
        ]
        return candidates if candidates else [self.fallback_profile()]

    def within_memory_budget(self, profile: str) -> bool:
        if not isfinite(self.memory_budget_mib):
            return True
        memory = self.memory_or_inf(profile)
        return memory <= self.memory_budget_mib

    def loss_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if profile in self.exact_profiles:
            return 0.0 if stat and stat.known_loss_count > 0 else inf
        if stat is None or stat.mean_loss is None:
            return inf
        return stat.mean_loss

    def ttft_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if stat is None or stat.p95_ttft_ms is None:
            return inf
        return stat.p95_ttft_ms

    def memory_or_inf(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if stat is None or stat.p95_peak_memory_mib is None:
            return inf
        return stat.p95_peak_memory_mib

    def predict_and_guard(self, request: Request, profile: str) -> tuple[float, float, bool, str]:
        if profile in self.exact_profiles:
            return 0.0, 0.0, True, "exact fallback"
        pred_loss = self.predictor.predict_loss(request, profile)
        risk_upper = self.guard.risk_upper(request, profile, pred_loss)
        safe = risk_upper <= self.epsilon
        reason = "calibrated safe" if safe else "calibrated unsafe"
        return pred_loss, risk_upper, safe, reason


class StatsPolicy(PolicyContext, Policy):
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
        self.placeholder = placeholder
        self.oracle = False
        super().__init__(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)

    def _fallback_profile(self) -> str:
        return self.fallback_profile()

    def _within_memory_budget(self, profile: str) -> bool:
        return self.within_memory_budget(profile)

    def _candidate_profiles(self, *, include_exact: bool = True) -> list[str]:
        return self.candidate_profiles(include_exact=include_exact)

    def _loss_or_inf(self, profile: str) -> float:
        return self.loss_or_inf(profile)

    def _ttft_or_inf(self, profile: str) -> float:
        return self.ttft_or_inf(profile)

    def _memory_or_inf(self, profile: str) -> float:
        return self.memory_or_inf(profile)

    def _best_profile(self, use_tail_constraint: bool) -> str:
        best_profile = ""
        best_score = inf
        for profile in self._candidate_profiles(include_exact=False):
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
        return best_profile or self._fallback_profile()

    def _predict_and_guard(self, request: Request, profile: str) -> tuple[float, float, bool, str]:
        return self.predict_and_guard(request, profile)
