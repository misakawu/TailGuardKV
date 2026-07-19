from __future__ import annotations

from collections.abc import Iterable
from math import inf
import time

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class TailGuardPolicy(StatsPolicy):
    DEFAULT_STC_CONFIG = {
        "cost_error_p95_ms": 0.0,
        "switch_cost_ms": 0.0,
        "min_residency_requests": 1,
        "hysteresis_enabled": True,
    }

    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
        stc_config: dict | None = None,
    ) -> None:
        super().__init__("tailguard", calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib=memory_budget_mib)
        config = {**self.DEFAULT_STC_CONFIG, **(stc_config or {})}
        self.cost_error_p95_ms = float(config["cost_error_p95_ms"])
        self.switch_cost_ms = float(config["switch_cost_ms"])
        self.min_residency_requests = max(1, int(config["min_residency_requests"]))
        self.hysteresis_enabled = bool(config["hysteresis_enabled"])
        self._current_profile: str | None = None
        self._current_residency = 0

    def _safe_candidates(self, request: Request) -> dict[str, tuple[float, float, bool, str]]:
        candidates: dict[str, tuple[float, float, bool, str]] = {}
        for profile in self.profiles:
            pred_loss, risk_upper, safe, reason = self._predict_and_guard(request, profile)
            if safe:
                candidates[profile] = (pred_loss, risk_upper, safe, reason)
        return candidates

    def _exact_fallback_profile(self) -> str | None:
        for profile in ("full_gpu", "full_cpu", "recompute"):
            if profile in self.profiles and profile in self.exact_profiles:
                return profile
        for profile in self.profiles:
            if profile in self.exact_profiles:
                return profile
        return None

    def _record_profile(self, profile: str) -> None:
        if profile == self._current_profile:
            self._current_residency += 1
        else:
            self._current_profile = profile
            self._current_residency = 1

    def _action(
        self,
        profile: str,
        guarded: tuple[float, float, bool, str],
        fallback_reason: str = "",
        controller_qrp_ms: float | None = None,
        controller_cg_ms: float | None = None,
        controller_stc_ms: float | None = None,
    ) -> Action:
        pred_loss, risk_upper, safe, reason = guarded
        self._record_profile(profile)
        return Action(
            profile=profile,
            reason="tailguard",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=fallback_reason or reason,
            controller_overhead_ms=sum(
                value for value in (controller_qrp_ms, controller_cg_ms, controller_stc_ms) if value is not None
            ),
            controller_qrp_ms=controller_qrp_ms,
            controller_cg_ms=controller_cg_ms,
            controller_stc_ms=controller_stc_ms,
        )

    def _stc_scores(self, candidates: dict[str, tuple[float, float, bool, str]], exact_profile: str) -> dict[str, tuple[float, float]]:
        exact_cost = self._ttft_or_inf(exact_profile)
        exact_memory = self._memory_or_inf(exact_profile)
        scores: dict[str, tuple[float, float]] = {}
        for profile in candidates:
            if profile in self.exact_profiles:
                continue
            memory = self._memory_or_inf(profile)
            delta_memory = exact_memory - memory
            if delta_memory <= 0.0 or memory > self.memory_budget_mib:
                continue
            delta_cost = self._ttft_or_inf(profile) - exact_cost
            total_cost = max(delta_cost, 0.0) + self.cost_error_p95_ms
            scores[profile] = (total_cost / delta_memory, total_cost)
        return scores

    def _select_with_hysteresis(self, scores: dict[str, tuple[float, float]]) -> str | None:
        if not scores:
            return None
        best_profile = min(scores, key=lambda profile: (scores[profile][0], self._ttft_or_inf(profile), profile))
        if not self.hysteresis_enabled or self._current_profile is None or self._current_profile == best_profile:
            return best_profile
        current_score = scores.get(self._current_profile)
        if current_score is None:
            return best_profile
        if self._current_residency < self.min_residency_requests:
            return self._current_profile
        current_cost = current_score[1]
        best_cost = scores[best_profile][1]
        if current_cost - best_cost <= self.switch_cost_ms:
            return self._current_profile
        return best_profile

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        started = time.perf_counter()
        candidates = self._safe_candidates(request)
        guard_done = time.perf_counter()
        fallback = self._exact_fallback_profile()
        if fallback is None:
            raise RuntimeError("tailguard requires at least one exact profile")
        fallback_guarded = candidates.get(fallback) or self._predict_and_guard(request, fallback)
        if self._memory_or_inf(fallback) <= self.memory_budget_mib:
            return self._action(
                fallback,
                fallback_guarded,
                controller_qrp_ms=0.0,
                controller_cg_ms=(guard_done - started) * 1000,
                controller_stc_ms=0.0,
            )

        scores = self._stc_scores(candidates, fallback)
        selected = self._select_with_hysteresis(scores)
        stc_done = time.perf_counter()
        if selected is None:
            return self._action(
                fallback,
                fallback_guarded,
                "no safe feasible lossy profile can satisfy memory budget; exact fallback",
                controller_qrp_ms=0.0,
                controller_cg_ms=(guard_done - started) * 1000,
                controller_stc_ms=(stc_done - guard_done) * 1000,
            )
        fallback_reason = "stc greedy memory release"
        if selected == self._current_profile and selected != min(scores, key=lambda profile: (scores[profile][0], self._ttft_or_inf(profile), profile)):
            fallback_reason = "hysteresis kept current profile"
        return self._action(
            selected,
            candidates[selected],
            fallback_reason,
            controller_qrp_ms=0.0,
            controller_cg_ms=(guard_done - started) * 1000,
            controller_stc_ms=(stc_done - guard_done) * 1000,
        )
