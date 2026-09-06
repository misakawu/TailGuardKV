from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from math import inf, isfinite
from threading import Lock
from typing import Any

from backends.base import Backend
from profiles.base import ProfileAdapter, create_persistent_profile_worker
from run_util.core_types import Action, BackendResult, CacheEvent, CacheState, ProfileMeasurement, ProfileSpec, Request


WorkerFactory = Callable[..., Any]


class OnlineQwenSessionBackend(Backend):
    """Execute fixture turns on persistent Qwen workers with a global KV LRU."""

    name = "online_qwen"

    def __init__(
        self,
        *,
        adapters: Sequence[ProfileAdapter],
        global_budget_mib: float = inf,
        worker_factory: WorkerFactory = create_persistent_profile_worker,
    ) -> None:
        self._adapters: dict[str, ProfileAdapter] = {}
        self._specs: dict[str, ProfileSpec] = {}
        for adapter in adapters:
            self._adapters[adapter.name] = adapter
            for spec in adapter.profiles():
                self._specs[spec.name] = spec
        self._worker_factory = worker_factory
        self._workers: dict[str, Any] = {}
        self._loaded_profile: str | None = None
        self._global_budget_mib = float(global_budget_mib) if global_budget_mib > 0 else inf
        self._lock = Lock()
        self.cache_state = CacheState(global_budget_mib=self._global_budget_mib)

    def reset(self) -> None:
        self.close()
        self.cache_state = CacheState(global_budget_mib=self._global_budget_mib)

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()
        self._loaded_profile = None

    def __enter__(self) -> "OnlineQwenSessionBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        del exc_info
        self.close()

    def run(self, requests: list[Request], profiles: list[str]) -> list[BackendResult]:
        if len(profiles) not in {1, len(requests)}:
            raise ValueError("profiles length must be 1 or match requests")
        results = []
        for index, request in enumerate(requests):
            profile = profiles[0] if len(profiles) == 1 else profiles[index]
            results.append(self.execute(request, Action(profile=profile), self.cache_state))
        return results

    def execute(self, request: Request, action: Action, cache_state: CacheState) -> BackendResult:
        queued_at = time.perf_counter()
        with self._lock:
            queue_delay_ms = (time.perf_counter() - queued_at) * 1000.0
            if queue_delay_ms < 0.01:
                queue_delay_ms = 0.0
            return self._execute_locked(request, action, cache_state, queue_delay_ms)

    def _execute_locked(
        self,
        request: Request,
        action: Action,
        cache_state: CacheState,
        queue_delay_ms: float,
    ) -> BackendResult:
        profile = action.profile
        if profile not in self._specs:
            raise KeyError(f"online Qwen backend has no profile: {profile}")
        cache_state, runtime_victims = self._prepare_profile_runtime(cache_state, profile)
        session_id = request.session_id or request.request_id
        previous_profile = cache_state.get_current_profile(session_id)
        previous_resident = cache_state.get_resident_kv(session_id, previous_profile or profile)
        switched = bool(previous_profile and previous_profile != profile)
        was_dropped = cache_state.get_dropped_kv(session_id) > 0

        if switched and previous_profile:
            self._evict_runtime_session(previous_profile, session_id)

        measurement = self._measure(request, profile, cache_state)
        baseline = BackendResult.from_profile_measurement(
            measurement,
            backend_name=self.name,
            replay_source="",
        )
        runtime_evicted_mib = sum(item[2] for item in runtime_victims)
        runtime_cache_events = _runtime_release_events(runtime_victims, cache_state)
        if not baseline.ok:
            self.cache_state = cache_state
            return replace(
                baseline,
                resident_kv_mib_before=0.0 if switched or was_dropped else previous_resident,
                resident_kv_mib_after=cache_state.get_resident_kv(session_id, profile),
                restore_ms=0.0,
                recompute_ms=0.0,
                queue_delay_ms=queue_delay_ms,
                evicted_kv_mib=runtime_evicted_mib,
                budget_hit=False,
                global_resident_kv_mib=cache_state.global_resident_kv_mib,
                global_budget_mib=self._global_budget_mib,
                extra={
                    **baseline.extra,
                    "active_sessions": cache_state.active_sessions,
                    "event_kind": "evict" if runtime_victims else "failure",
                    "event_reason": "profile_runtime_released" if runtime_victims else "online_execution_failed",
                    "victim_session": ",".join(item[0] for item in runtime_victims),
                    "budget_resolution": "runtime_switch" if runtime_victims else "none",
                    "cache_events": runtime_cache_events,
                    "online_source": "persistent_qwen_worker",
                },
            )

        measured_resident = float(
            measurement.resident_kv_mib_after
            if measurement.resident_kv_mib_after is not None
            else measurement.kv_cache_memory_mib
            if measurement.kv_cache_memory_mib is not None
            else measurement.resident_memory_mib
            if measurement.resident_memory_mib is not None
            else 0.0
        )
        resident_before = 0.0 if switched or was_dropped else previous_resident
        cumulative_after = float(
            measurement.kv_cumulative_mib
            if measurement.kv_cumulative_mib is not None
            else measured_resident
        )
        recompute_ms = float(measurement.recompute_ms or 0.0)
        if switched or was_dropped:
            recompute_ms = max(recompute_ms, float(measurement.extra.get("stage_prefill_ms") or measurement.ttft_ms or 0.0))
        restore_ms = float(measurement.restore_ms or 0.0)

        state = self._record_current(
            cache_state,
            session_id=session_id,
            previous_profile=previous_profile,
            profile=profile,
            turn_index=request.turn_index,
            arrival_index=request.arrival_index,
            cumulative_after=cumulative_after,
            resident_after=measured_resident,
        )
        state, budget_victims, budget_evicted_mib = self._enforce_budget(state, current_session_id=session_id)
        budget_hit = bool(budget_victims)
        for victim_session, victim_profile, _ in budget_victims:
            self._evict_runtime_session(victim_profile, victim_session)

        resident_after = state.get_resident_kv(session_id, profile)
        event_kind = "recompute" if (switched or was_dropped) else "evict" if budget_victims else "resident"
        event_reason = (
            "incompatible_profile_switch"
            if switched
            else "restore_from_dropped_state"
            if was_dropped
            else "global_lru_budget"
            if budget_victims
            else "online_measurement"
        )
        all_victims = [*runtime_victims, *budget_victims]
        victim_names = ",".join(dict.fromkeys(item[0] for item in all_victims))
        cache_event = CacheEvent(
            event_kind=event_kind,
            event_reason=event_reason,
            session_id=session_id,
            profile=profile,
            turn_index=request.turn_index,
            kv_mib=resident_after,
            victim_session=victim_names,
            budget_resolution="drop_lru" if budget_victims else "none",
        )
        cache_events = list(runtime_cache_events)
        cache_events.append(asdict(cache_event))
        self.cache_state = state
        extra = {
            **baseline.extra,
            "active_sessions": state.active_sessions,
            "event_kind": event_kind,
            "event_reason": event_reason,
            "victim_session": victim_names,
            "budget_resolution": cache_event.budget_resolution,
            "cache_events": cache_events,
            "online_source": "persistent_qwen_worker",
        }
        return replace(
            baseline,
            kv_cache_memory_mib=resident_after,
            resident_memory_mib=resident_after,
            kv_incremental_mib=max(0.0, measured_resident - resident_before),
            kv_cumulative_mib=cumulative_after,
            resident_kv_mib_before=resident_before,
            resident_kv_mib_after=resident_after,
            restore_ms=restore_ms,
            recompute_ms=recompute_ms,
            queue_delay_ms=queue_delay_ms,
            evicted_kv_mib=budget_evicted_mib + runtime_evicted_mib,
            budget_hit=budget_hit,
            global_resident_kv_mib=state.global_resident_kv_mib,
            global_budget_mib=self._global_budget_mib,
            extra=extra,
        )

    def _measure(self, request: Request, profile: str, cache_state: CacheState) -> ProfileMeasurement:
        spec = self._specs[profile]
        adapter = self._adapters[spec.family]
        worker = self._worker(adapter)
        runtime_state = {"state": _runtime_state_payload(cache_state)}
        return adapter.profile_many(
            [request],
            profile,
            dry_run=False,
            session_runtime=runtime_state,
            memory_budget_mib=None,
            persistent_worker=worker,
        )[0]

    def _worker(self, adapter: ProfileAdapter):
        worker = self._workers.get(adapter.name)
        if worker is None:
            worker = self._worker_factory(
                adapter=adapter.name,
                env_name=adapter.env,
                runtime_module="profiles.qwen2_kv_runtime",
                runtime_config=adapter.runtime_config,
                pythonpath=getattr(adapter, "pythonpath", ()),
            )
            self._workers[adapter.name] = worker
        return worker

    def _prepare_profile_runtime(
        self,
        state: CacheState,
        profile: str,
    ) -> tuple[CacheState, list[tuple[str, str, float]]]:
        if self._loaded_profile in {None, profile}:
            self._loaded_profile = profile
            return state, []
        self.close()
        self._loaded_profile = profile
        next_state = state
        invalidated: list[tuple[str, str, float]] = []
        for session_id in state.active_sessions:
            resident_profile = next_state.get_current_profile(session_id)
            if not resident_profile:
                continue
            resident_mib = next_state.get_resident_kv(session_id, resident_profile)
            if resident_mib > 0:
                invalidated.append((session_id, resident_profile, resident_mib))
                next_state = _drop_session(next_state, session_id, resident_profile, resident_mib)
        return next_state, invalidated

    def _record_current(
        self,
        state: CacheState,
        *,
        session_id: str,
        previous_profile: str | None,
        profile: str,
        turn_index: int,
        arrival_index: int,
        cumulative_after: float,
        resident_after: float,
    ) -> CacheState:
        previous_resident = state.get_resident_kv(session_id, previous_profile or profile)
        global_resident = max(0.0, state.global_resident_kv_mib - previous_resident + resident_after)
        next_state = state.with_session_turn(
            session_id,
            profile=profile,
            turn_index=turn_index,
            cumulative_kv_mib=cumulative_after,
            resident_kv_mib=resident_after,
            offloaded_kv_mib=0.0,
            dropped_kv_mib=0.0,
            access_index=arrival_index,
            global_resident_kv_mib=global_resident,
            global_budget_mib=self._global_budget_mib,
        )
        if previous_profile and previous_profile != profile:
            residents = {key: dict(value) for key, value in next_state.session_resident_kv_mib.items()}
            residents.setdefault(session_id, {})[previous_profile] = 0.0
            next_state = replace(next_state, session_resident_kv_mib=residents)
        return next_state

    def _enforce_budget(
        self,
        state: CacheState,
        *,
        current_session_id: str,
    ) -> tuple[CacheState, list[tuple[str, str, float]], float]:
        if not isfinite(self._global_budget_mib) or state.global_resident_kv_mib <= self._global_budget_mib:
            return state, [], 0.0
        ordered = sorted(
            state.active_sessions,
            key=lambda session_id: (state.session_last_access_index.get(session_id, 0), session_id),
        )
        other_sessions = [session_id for session_id in ordered if session_id != current_session_id]
        victims: list[tuple[str, str, float]] = []
        next_state = state
        for victim_session in [*other_sessions, current_session_id]:
            if next_state.global_resident_kv_mib <= self._global_budget_mib:
                break
            victim_profile = next_state.get_current_profile(victim_session)
            if not victim_profile:
                continue
            victim_mib = next_state.get_resident_kv(victim_session, victim_profile)
            if victim_mib <= 0:
                continue
            victims.append((victim_session, victim_profile, victim_mib))
            next_state = _drop_session(next_state, victim_session, victim_profile, victim_mib)
        return next_state, victims, sum(item[2] for item in victims)

    def _evict_runtime_session(self, profile: str, session_id: str) -> None:
        spec = self._specs.get(profile)
        if spec is None:
            return
        worker = self._workers.get(spec.family)
        if worker is not None:
            self._send_cache_control(worker, [session_id])

    def _send_cache_control(self, worker: Any, sessions: list[str]) -> None:
        worker.request(
            {
                "op": "run_batch",
                "requests": [],
                "evict_sessions": sessions,
                "session_runtime_state": _runtime_state_payload(self.cache_state),
            },
            timeout_s=30,
        )


def _drop_session(state: CacheState, session_id: str, profile: str, kv_mib: float) -> CacheState:
    return state.with_session_turn(
        session_id,
        profile=profile,
        turn_index=state.session_last_turn_index.get(session_id, 0),
        cumulative_kv_mib=state.get_cumulative_kv(session_id, profile),
        resident_kv_mib=0.0,
        offloaded_kv_mib=0.0,
        dropped_kv_mib=state.get_dropped_kv(session_id) + kv_mib,
        access_index=state.session_last_access_index.get(session_id, 0),
        global_resident_kv_mib=max(0.0, state.global_resident_kv_mib - kv_mib),
        global_budget_mib=state.global_budget_mib,
    )


def _runtime_release_events(
    victims: list[tuple[str, str, float]],
    state: CacheState,
) -> list[dict[str, object]]:
    return [
        asdict(
            CacheEvent(
                event_kind="evict",
                event_reason="profile_runtime_released",
                session_id=session_id,
                profile=profile,
                turn_index=state.session_last_turn_index.get(session_id, 0),
                kv_mib=kv_mib,
                victim_session=session_id,
                budget_resolution="runtime_switch",
            )
        )
        for session_id, profile, kv_mib in victims
    ]


def _runtime_state_payload(state: CacheState) -> dict[str, object]:
    sessions: dict[str, dict[str, object]] = {}
    for session_id in state.active_sessions:
        profile = state.get_current_profile(session_id) or ""
        sessions[session_id] = {
            "current_profile": profile,
            "last_turn_index": state.session_last_turn_index.get(session_id, -1),
            "resident_gpu_mib": state.get_resident_kv(session_id, profile),
            "offloaded_cpu_mib": state.get_offloaded_kv(session_id),
            "dropped_mib": state.get_dropped_kv(session_id),
            "rebuild_required": state.get_dropped_kv(session_id) > 0,
        }
    return {"sessions": sessions}
