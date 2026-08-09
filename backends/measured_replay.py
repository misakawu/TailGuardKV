from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from dataclasses import replace
from math import inf, isfinite

from backends.base import Backend
from run_util.core_types import BackendResult, CacheEvent, CacheState, ProfileMeasurement, ReplayState, Request, SessionState


RESTORE_MS_PER_MIB = 0.35
RECOMPUTE_MS_PER_MIB = 0.6
QUEUE_MS_PER_MIB = 0.15


class MeasuredReplayBackend(Backend):
    """用实测 profile 表驱动 server-side replay 仿真。"""

    name = "measured_replay"

    def __init__(
        self,
        measurements: Iterable[ProfileMeasurement],
        allow_dry_run: bool = False,
        use_pandas: bool = False,
        global_budget_mib: float = float("inf"),
    ) -> None:
        rows = list(measurements)
        dry_rows = [measurement for measurement in rows if not measurement.measured]
        if dry_rows and not allow_dry_run:
            sample = dry_rows[0]
            raise ValueError(
                "MeasuredReplayBackend 默认只接受 measured=True 的实测 profile 表；"
                f"发现 dry-run 行: request={sample.request_id} profile={sample.profile}。"
                "如仅做 smoke，请显式传入 allow_dry_run=True。"
            )
        self._use_pandas = False
        self._frame = None
        self._global_budget_mib = global_budget_mib if global_budget_mib > 0 else inf
        if use_pandas:
            try:
                import pandas as pd
            except ModuleNotFoundError:
                self.measurements = {self._measurement_key(measurement): measurement for measurement in rows}
            else:
                self._use_pandas = True
                self._frame = pd.DataFrame(
                    {
                        "request_id": measurement.request_id,
                        "session_id": measurement.session_id or "",
                        "turn_index": measurement.turn_index,
                        "profile": measurement.profile,
                        "measurement": measurement,
                    }
                    for measurement in rows
                ).set_index(["session_id", "turn_index", "request_id", "profile"])
                self.measurements = {}
        else:
            self.measurements = {self._measurement_key(measurement): measurement for measurement in rows}
        self.reset()

    def reset(self) -> None:
        self.cache_state = CacheState(global_budget_mib=self._global_budget_mib)

    def snapshot(self) -> ReplayState:
        sessions: list[SessionState] = []
        for session_id, profile in sorted(self.cache_state.session_current_profile.items()):
            sessions.append(
                SessionState(
                    session_id=session_id,
                    profile=profile,
                    turn_index=self.cache_state.session_last_turn_index.get(session_id, 0),
                    cumulative_kv_mib=self.cache_state.get_cumulative_kv(session_id, profile),
                    resident_kv_mib=self.cache_state.get_resident_kv(session_id, profile),
                    offloaded_kv_mib=self.cache_state.get_offloaded_kv(session_id),
                    dropped_kv_mib=self.cache_state.get_dropped_kv(session_id),
                    last_access_index=self.cache_state.session_last_access_index.get(session_id, 0),
                )
            )
        return ReplayState(
            global_resident_kv_mib=self.cache_state.global_resident_kv_mib,
            global_budget_mib=self.cache_state.global_budget_mib,
            sessions=tuple(sessions),
        )

    def run(self, requests: list[Request], profiles: list[str]) -> list[BackendResult]:
        if len(profiles) not in {1, len(requests)}:
            raise ValueError("profiles 长度必须为 1 或与 requests 一致")
        rows: list[BackendResult] = []
        for index, request in enumerate(requests):
            profile = profiles[0] if len(profiles) == 1 else profiles[index]
            measurement = self._lookup(request, profile)
            rows.append(self._simulate_result(request, profile, measurement))
        return rows

    def _lookup(self, request: Request, profile: str) -> ProfileMeasurement:
        key = self._request_key(request, profile)
        if self._use_pandas:
            try:
                return self._frame.loc[key, "measurement"]
            except KeyError as exc:
                raise KeyError(f"缺少回放数据: session={key[0] or '-'} turn={key[1]} request={key[2]} profile={key[3]}") from exc
        if key not in self.measurements:
            raise KeyError(f"缺少回放数据: session={key[0] or '-'} turn={key[1]} request={key[2]} profile={key[3]}")
        return self.measurements[key]

    def _simulate_result(self, request: Request, profile: str, measurement: ProfileMeasurement) -> BackendResult:
        baseline = BackendResult.from_profile_measurement(
            measurement,
            backend_name=self.name,
            replay_source="measured_profile_table",
            extra={"source_profile": measurement.profile},
        )
        if not request.session_id:
            return baseline
        session_id = request.session_id or request.request_id
        current_profile = self.cache_state.get_current_profile(session_id) or profile
        previous_resident = self.cache_state.get_resident_kv(session_id, current_profile)
        previous_offloaded = self.cache_state.get_offloaded_kv(session_id)
        previous_dropped = self.cache_state.get_dropped_kv(session_id)
        previous_global = self.cache_state.global_resident_kv_mib

        resident_after = baseline.resident_kv_mib_after
        if resident_after is None:
            resident_after = baseline.resident_memory_mib or baseline.kv_cumulative_mib or baseline.kv_cache_memory_mib or 0.0
        cumulative_after = baseline.kv_cumulative_mib if baseline.kv_cumulative_mib is not None else resident_after

        restore_ms = float(baseline.restore_ms or 0.0)
        recompute_ms = float(baseline.recompute_ms or 0.0)
        queue_delay_ms = 0.0
        budget_hit = bool(baseline.budget_hit)
        evicted_kv_mib = float(baseline.evicted_kv_mib or 0.0)
        event_kind = "resident"
        event_reason = "measured_transition"
        victim_session = ""
        budget_resolution = "none"

        if previous_dropped > 0:
            recompute_ms += previous_dropped * RECOMPUTE_MS_PER_MIB
            event_kind = "recompute"
            event_reason = "restore_from_dropped_state"
        elif previous_offloaded > 0:
            restore_ms += previous_offloaded * RESTORE_MS_PER_MIB
            event_kind = "restore"
            event_reason = "restore_from_offloaded_state"

        projected_global = max(0.0, previous_global - previous_resident + resident_after)
        if isfinite(self._global_budget_mib) and projected_global > self._global_budget_mib:
            budget_hit = True
            projected_global, evicted_from_others, victim_session = self._evict_other_sessions(
                current_session_id=session_id,
                projected_global=projected_global,
            )
            evicted_kv_mib += evicted_from_others
            if projected_global > self._global_budget_mib:
                overflow = projected_global - self._global_budget_mib
                queue_delay_ms += overflow * QUEUE_MS_PER_MIB
                event_kind = "queue"
                event_reason = "global_budget_pressure"
                budget_resolution = "offload_current_session"
                self.cache_state = self.cache_state.with_session_turn(
                    session_id,
                    profile=current_profile,
                    turn_index=self.cache_state.session_last_turn_index.get(session_id, request.turn_index),
                    cumulative_kv_mib=self.cache_state.get_cumulative_kv(session_id, current_profile),
                    resident_kv_mib=0.0,
                    offloaded_kv_mib=previous_offloaded + previous_resident,
                    dropped_kv_mib=0.0,
                    access_index=self.cache_state.session_last_access_index.get(session_id, request.arrival_index),
                    global_resident_kv_mib=max(0.0, previous_global - previous_resident),
                    global_budget_mib=self._global_budget_mib,
                )
                previous_resident = 0.0
                projected_global = self.cache_state.global_resident_kv_mib + resident_after
                if projected_global > self._global_budget_mib:
                    current_overflow = projected_global - self._global_budget_mib
                    offloaded_after = previous_offloaded + current_overflow
                    resident_after = max(0.0, resident_after - current_overflow)
                    evicted_kv_mib += current_overflow
                    projected_global = self._global_budget_mib
                else:
                    offloaded_after = previous_offloaded
            else:
                offloaded_after = previous_offloaded
                event_kind = "evict"
                event_reason = "global_budget_pressure"
                budget_resolution = "evict_other_sessions"
        else:
            offloaded_after = 0.0

        global_resident = max(0.0, projected_global)
        cache_event = CacheEvent(
            event_kind=event_kind,
            event_reason=event_reason,
            session_id=session_id,
            profile=profile,
            turn_index=request.turn_index,
            kv_mib=resident_after,
            victim_session=victim_session,
            budget_resolution=budget_resolution,
        )
        active_sessions = tuple(sorted({*self.cache_state.active_sessions, session_id}))
        self.cache_state = self.cache_state.with_session_turn(
            session_id,
            profile=profile,
            turn_index=request.turn_index,
            cumulative_kv_mib=cumulative_after,
            resident_kv_mib=resident_after,
            offloaded_kv_mib=offloaded_after,
            dropped_kv_mib=0.0,
            access_index=request.arrival_index,
            global_resident_kv_mib=global_resident,
            global_budget_mib=self._global_budget_mib,
        )
        result_extra = {
            **baseline.extra,
            "active_sessions": active_sessions,
            "global_resident_kv_mib": global_resident,
            "global_budget_mib": self._global_budget_mib,
            "event_kind": event_kind,
            "event_reason": event_reason,
            "victim_session": victim_session,
            "budget_resolution": budget_resolution,
            "cache_events": [asdict(cache_event)],
            "replay_state": asdict(self.snapshot()),
        }
        total_overhead = queue_delay_ms + restore_ms + recompute_ms - float(baseline.restore_ms or 0.0) - float(baseline.recompute_ms or 0.0)
        return replace(
            baseline,
            latency_ms=(baseline.latency_ms + total_overhead) if baseline.latency_ms is not None else None,
            ttft_ms=(baseline.ttft_ms + total_overhead) if baseline.ttft_ms is not None else None,
            kv_cache_memory_mib=resident_after,
            resident_memory_mib=resident_after,
            resident_kv_mib_before=previous_resident,
            resident_kv_mib_after=resident_after,
            restore_ms=restore_ms,
            recompute_ms=recompute_ms,
            queue_delay_ms=queue_delay_ms,
            evicted_kv_mib=evicted_kv_mib,
            budget_hit=budget_hit,
            global_resident_kv_mib=global_resident,
            global_budget_mib=self._global_budget_mib,
            extra=result_extra,
        )

    def _evict_other_sessions(self, *, current_session_id: str, projected_global: float) -> tuple[float, float, str]:
        evicted_total = 0.0
        victims_seen: list[str] = []
        victims = sorted(
            (
                (session_id, access_index)
                for session_id, access_index in self.cache_state.session_last_access_index.items()
                if session_id != current_session_id
            ),
            key=lambda item: (item[1], item[0]),
        )
        for victim_session, _ in victims:
            if projected_global <= self._global_budget_mib:
                break
            victim_profile = self.cache_state.get_current_profile(victim_session)
            if not victim_profile:
                continue
            victim_resident = self.cache_state.get_resident_kv(victim_session, victim_profile)
            if victim_resident <= 0:
                continue
            projected_global = max(0.0, projected_global - victim_resident)
            evicted_total += victim_resident
            victims_seen.append(victim_session)
            self.cache_state = self.cache_state.with_session_turn(
                victim_session,
                profile=victim_profile,
                turn_index=self.cache_state.session_last_turn_index.get(victim_session, 0),
                cumulative_kv_mib=self.cache_state.get_cumulative_kv(victim_session, victim_profile),
                resident_kv_mib=0.0,
                offloaded_kv_mib=self.cache_state.get_offloaded_kv(victim_session) + victim_resident,
                dropped_kv_mib=0.0,
                access_index=self.cache_state.session_last_access_index.get(victim_session, 0),
                global_resident_kv_mib=projected_global,
                global_budget_mib=self._global_budget_mib,
            )
        return projected_global, evicted_total, ",".join(victims_seen)

    @staticmethod
    def _measurement_key(measurement: ProfileMeasurement) -> tuple[str, int, str, str]:
        return (measurement.session_id or "", measurement.turn_index, measurement.request_id, measurement.profile)

    @staticmethod
    def _request_key(request: Request, profile: str) -> tuple[str, int, str, str]:
        return (request.session_id or "", request.turn_index, request.request_id, profile)
