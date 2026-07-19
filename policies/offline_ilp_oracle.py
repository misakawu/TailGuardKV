from __future__ import annotations

from collections.abc import Iterable
from math import inf

from core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import Policy


class OfflineILPOraclePolicy(Policy):
    """Small-trace oracle equivalent to a multiple-choice knapsack solver.

    The online policy API decides per request, so this class precomputes a global
    oracle plan over the supplied replay table and then serves the planned action.
    It uses exhaustive dynamic programming over scaled MiB budgets for small Pilot
    traces and falls back to per-request greedy when the budget is infinite.
    """

    def __init__(
        self,
        measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
    ) -> None:
        self.name = "offline_ilp_oracle"
        self.profiles = profiles
        self.epsilon = epsilon
        self.delta = delta
        self.exact_profiles = exact_profiles
        self.placeholder = False
        self.oracle = True
        self.memory_budget_mib = memory_budget_mib
        self.measurements = {
            (measurement.request_id, measurement.profile): measurement for measurement in measurements
        }
        self.plan, self.oracle_cost_ms = solve_offline_oracle(
            list(measurements),
            profiles,
            epsilon,
            memory_budget_mib,
        )

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        profile = self.plan.get(request.request_id) or self._fallback_profile()
        measurement = self.measurements.get((request.request_id, profile))
        greedy_profile = _best_feasible_profile(request.request_id, self.measurements, self.profiles, self.epsilon)
        greedy = self.measurements.get((request.request_id, greedy_profile)) if greedy_profile else None
        oracle_cost = None if measurement is None else measurement.ttft_ms
        greedy_cost = None if greedy is None else greedy.ttft_ms
        gap = None
        if greedy_cost is not None and oracle_cost is not None and oracle_cost > 0:
            gap = max(0.0, (greedy_cost - oracle_cost) / oracle_cost)
        return Action(
            profile=profile,
            reason="offline ILP oracle",
            pred_loss=None if measurement is None else measurement.quality_loss,
            risk_upper=None if measurement is None else measurement.quality_loss,
            safe=True,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason="offline_ilp_oracle",
            oracle_cost_ms=oracle_cost,
            optimality_gap=gap,
        )

    def _fallback_profile(self) -> str:
        for profile in ("full_gpu", "full_cpu", "recompute"):
            if profile in self.profiles:
                return profile
        return self.profiles[0]


def solve_offline_oracle(
    measurements: list[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    memory_budget_mib: float,
    memory_scale_mib: float = 1.0,
) -> tuple[dict[str, str], float]:
    by_request: dict[str, list[ProfileMeasurement]] = {}
    for row in measurements:
        if row.profile not in profiles or row.quality_loss is None or row.ttft_ms is None:
            continue
        if row.quality_loss <= epsilon:
            by_request.setdefault(row.request_id, []).append(row)
    if not by_request:
        return {}, inf
    if memory_budget_mib == inf:
        plan = {
            request_id: min(rows, key=lambda row: (row.ttft_ms if row.ttft_ms is not None else inf, row.profile)).profile
            for request_id, rows in by_request.items()
        }
        return plan, _plan_cost(plan, measurements)

    budget = max(0, int(round(memory_budget_mib / memory_scale_mib)))
    states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
    for request_id, rows in sorted(by_request.items()):
        next_states: dict[int, tuple[float, dict[str, str]]] = {}
        for used, (cost, plan) in states.items():
            for row in rows:
                if row.peak_memory_mib is None:
                    continue
                memory = int(round(row.peak_memory_mib / memory_scale_mib))
                if used + memory > budget:
                    continue
                next_cost = cost + float(row.ttft_ms or inf)
                current = next_states.get(used + memory)
                if current is None or next_cost < current[0]:
                    next_states[used + memory] = (next_cost, {**plan, request_id: row.profile})
        if not next_states:
            return {}, inf
        states = next_states
    best_cost, best_plan = min(states.values(), key=lambda item: item[0])
    return best_plan, best_cost


def _best_feasible_profile(
    request_id: str,
    measurements: dict[tuple[str, str], ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
) -> str | None:
    rows = [
        measurements[(request_id, profile)]
        for profile in profiles
        if (request_id, profile) in measurements
        and measurements[(request_id, profile)].quality_loss is not None
        and measurements[(request_id, profile)].quality_loss <= epsilon
    ]
    if not rows:
        return None
    return min(rows, key=lambda row: (row.ttft_ms if row.ttft_ms is not None else inf, row.profile)).profile


def _plan_cost(plan: dict[str, str], measurements: list[ProfileMeasurement]) -> float:
    by_key = {(row.request_id, row.profile): row for row in measurements}
    return sum(float(by_key[(request_id, profile)].ttft_ms or inf) for request_id, profile in plan.items())
