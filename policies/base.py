from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import inf, isfinite

from calibration.conformal import ConformalGuard
from calibration.predictor import MetadataOnlyRiskPredictor
from run_util.core_types import Action, ActionDecision, CacheState, CandidateAction, DeviceState, ProfileMeasurement, Request


@dataclass(frozen=True)
class ProfileStats:
    profile: str
    count: int
    known_loss_count: int
    mean_loss: float | None
    violation_rate: float | None
    p95_ttft_ms: float | None
    p95_peak_memory_mib: float | None
    p95_kv_cache_memory_mib: float | None
    p95_kv_incremental_mib: float | None
    p95_kv_cumulative_mib: float | None


class Policy(ABC):
    name: str
    # placeholder=True 表示当前策略是 smoke/replay 阶段的可替换控制器近似，
    # oracle=True 表示策略读取了评估集真值或离线最优信息，只作为上界对照。
    placeholder: bool = False
    oracle: bool = False

    @abstractmethod
    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        ...

    @staticmethod
    def _finalize_action(decision: ActionDecision) -> Action:
        return decision.to_action()


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
        if not profiles:
            raise ValueError(f"{name} 至少需要一个 profile")
        self.memory_budget_mib = memory_budget_mib if memory_budget_mib > 0 else inf
        calibration_rows = list(calibration_measurements)
        self.stats = _profile_stats(calibration_rows, profiles, epsilon, exact_profiles)
        self.predictor = MetadataOnlyRiskPredictor(calibration_rows)
        self.guard = ConformalGuard(
            epsilon=epsilon,
            delta=delta,
            exact_profiles=exact_profiles,
            profiles=profiles,
            calibration_rows=calibration_rows,
        )

    def _fallback_profile(self) -> str:
        for profile in ("engine_full_lru",):
            if profile in self.profiles:
                return profile
        for profile in self.profiles:
            if profile in self.exact_profiles:
                return profile
        return self.profiles[0]

    def _fastest_exact_profile(self) -> str:
        best_profile = ""
        best_ttft = inf
        for profile in self.profiles:
            if profile not in self.exact_profiles:
                continue
            ttft = self._ttft_or_inf(profile)
            if ttft < best_ttft:
                best_profile = profile
                best_ttft = ttft
        if best_profile:
            return best_profile
        return self._fallback_profile()

    def _within_memory_budget(self, profile: str, request: Request | None = None, cache_state: CacheState | None = None) -> bool:
        if not isfinite(self.memory_budget_mib):
            return True
        memory = self._projected_memory(profile, request, cache_state)
        return memory <= self.memory_budget_mib

    def _candidate_profiles(
        self,
        request: Request | None = None,
        cache_state: CacheState | None = None,
        *,
        include_exact: bool = True,
    ) -> list[str]:
        candidates: list[str] = []
        for profile in self.profiles:
            if not include_exact and profile in self.exact_profiles:
                continue
            if profile in self.exact_profiles:
                candidates.append(profile)
                continue
            if self._within_memory_budget(profile, request, cache_state):
                candidates.append(profile)
        if candidates:
            return candidates
        return []

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
        if stat is None:
            return inf
        if stat.p95_kv_cache_memory_mib is not None:
            return stat.p95_kv_cache_memory_mib
        if stat.p95_peak_memory_mib is not None:
            return stat.p95_peak_memory_mib
        return inf

    def _incremental_memory_or_zero(self, profile: str) -> float:
        stat = self.stats.get(profile)
        if stat is None:
            return 0.0
        if stat.p95_kv_incremental_mib is not None:
            return stat.p95_kv_incremental_mib
        if stat.p95_kv_cumulative_mib is not None:
            return stat.p95_kv_cumulative_mib
        memory = self._memory_or_inf(profile)
        return 0.0 if memory is inf else memory

    def _projected_memory(self, profile: str, request: Request | None, cache_state: CacheState | None) -> float:
        if request is None or cache_state is None or not request.session_id:
            return self._memory_or_inf(profile)
        current_profile = cache_state.get_current_profile(request.session_id)
        current_resident = cache_state.get_resident_kv(request.session_id, current_profile or profile)
        projected_session = self._projected_session_resident(profile, request, cache_state)
        global_resident = self._global_resident(cache_state)
        return max(0.0, global_resident - current_resident + projected_session)

    def _projected_session_resident(self, profile: str, request: Request | None, cache_state: CacheState | None) -> float:
        if request is None or cache_state is None or not request.session_id:
            memory = self._memory_or_inf(profile)
            return 0.0 if memory is inf else memory
        current_profile = cache_state.get_current_profile(request.session_id)
        current_resident = cache_state.get_resident_kv(request.session_id, current_profile or profile)
        session_history = cache_state.get_cumulative_kv(request.session_id, current_profile or profile)
        if current_profile == profile:
            return current_resident + self._incremental_memory_or_zero(profile)
        if session_history <= 0.0:
            measured_cumulative = cache_state.get_cumulative_kv(request.session_id, profile)
            if measured_cumulative > 0.0:
                session_history = measured_cumulative
            else:
                stat = self.stats.get(profile)
                if stat and stat.p95_kv_cumulative_mib is not None:
                    session_history = float(stat.p95_kv_cumulative_mib)
        return max(0.0, session_history) + self._incremental_memory_or_zero(profile)

    @staticmethod
    def _global_resident(cache_state: CacheState) -> float:
        derived = 0.0
        for session_id, current_profile in cache_state.session_current_profile.items():
            derived += cache_state.get_resident_kv(session_id, current_profile)
        return max(cache_state.global_resident_kv_mib, derived)

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

    def _best_static_best_profile(self) -> str:
        fastest_lossy = ""
        fastest_lossy_ttft = inf
        for profile in self._candidate_profiles(include_exact=False):
            stat = self.stats.get(profile)
            if stat is None or stat.known_loss_count == 0 or self._loss_or_inf(profile) > self.epsilon:
                continue
            ttft = self._ttft_or_inf(profile)
            if ttft < fastest_lossy_ttft:
                fastest_lossy = profile
                fastest_lossy_ttft = ttft

        fastest_exact = self._fastest_exact_profile()
        exact_ttft = self._ttft_or_inf(fastest_exact)
        if (
            fastest_lossy
            and isfinite(fastest_lossy_ttft)
            and isfinite(exact_ttft)
            and fastest_lossy_ttft <= 0.95 * exact_ttft
        ):
            return fastest_lossy
        return fastest_exact

    def _lowest_empirical_loss_lossy_profile(self) -> str:
        best_profile = ""
        best_loss = inf
        for profile in self._candidate_profiles(include_exact=False):
            stat = self.stats.get(profile)
            if stat is None or stat.known_loss_count == 0:
                continue
            loss = self._loss_or_inf(profile)
            if loss < best_loss:
                best_profile = profile
                best_loss = loss
        return best_profile or self._fastest_exact_profile()

    def _ttft_normalizer(self) -> float:
        values = [self._ttft_or_inf(profile) for profile in self.profiles]
        finite_values = [value for value in values if isfinite(value) and value > 0.0]
        return max(finite_values, default=1.0)

    def _kv_normalizer(self) -> float:
        values = [self._memory_or_inf(profile) for profile in self.profiles]
        finite_values = [value for value in values if isfinite(value) and value > 0.0]
        return max(finite_values, default=1.0)

    def _predict_and_guard(self, request: Request, profile: str) -> tuple[float, float, bool, str]:
        pred_loss = self.predictor.predict_loss(request, profile)
        risk_upper = self.guard.risk_upper(request, profile, pred_loss)
        safe = risk_upper <= self.epsilon or profile in self.exact_profiles
        reason = "exact fallback" if profile in self.exact_profiles else ("calibrated safe" if safe else "calibrated unsafe")
        return pred_loss, risk_upper, safe, reason

    def _candidate_safe_count(self, request: Request, cache_state: CacheState | None = None) -> int:
        count = 0
        for profile in self._candidate_profiles(request, cache_state):
            pred_loss = self.predictor.predict_loss(request, profile)
            risk_upper = self.guard.risk_upper(request, profile, pred_loss)
            if risk_upper <= self.epsilon or profile in self.exact_profiles:
                count += 1
        return count

    def _candidate_action(self, request: Request, profile: str, cache_state: CacheState | None = None) -> CandidateAction:
        pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, profile)
        projected_memory = self._projected_memory(profile, request, cache_state)
        within_memory_budget = profile in self.exact_profiles or self._within_memory_budget(profile, request, cache_state)
        return CandidateAction(
            profile=profile,
            predicted_ttft_ms=self._ttft_or_inf(profile),
            projected_memory_mib=projected_memory,
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            exact=profile in self.exact_profiles,
            within_memory_budget=within_memory_budget,
            reason=reason,
        )

    def _best_exact_candidate(self, request: Request, cache_state: CacheState | None = None) -> CandidateAction:
        exact_profiles = [profile for profile in self.profiles if profile in self.exact_profiles]
        if not exact_profiles:
            exact_profiles = [self._fallback_profile()]
        candidates = [self._candidate_action(request, profile, cache_state) for profile in exact_profiles]
        return min(candidates, key=lambda item: (item.predicted_ttft_ms, item.profile))


def _profile_stats(
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
        kv_memories = [row.kv_cache_memory_mib for row in rows if row.kv_cache_memory_mib is not None]
        incremental_kv = [row.kv_incremental_mib for row in rows if row.kv_incremental_mib is not None]
        cumulative_kv = [row.kv_cumulative_mib for row in rows if row.kv_cumulative_mib is not None]
        stats[profile] = ProfileStats(
            profile=profile,
            count=len(rows),
            known_loss_count=len(losses),
            mean_loss=(sum(losses) / len(losses) if losses else None),
            violation_rate=(sum(1 for loss in losses if loss > epsilon) / len(losses) if losses else None),
            p95_ttft_ms=(_percentile(ttfts, 0.95) if ttfts else None),
            p95_peak_memory_mib=(_percentile(memories, 0.95) if memories else None),
            p95_kv_cache_memory_mib=(_percentile(kv_memories, 0.95) if kv_memories else None),
            p95_kv_incremental_mib=(_percentile(incremental_kv, 0.95) if incremental_kv else None),
            p95_kv_cumulative_mib=(_percentile(cumulative_kv, 0.95) if cumulative_kv else None),
        )
    return stats


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value != inf)
    if not finite_values:
        return inf
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]
