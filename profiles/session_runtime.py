from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class SessionObjectState:
    current_profile: str = ""
    last_turn_index: int = -1
    resident_gpu_mib: float = 0.0
    offloaded_cpu_mib: float = 0.0
    dropped_mib: float = 0.0
    rebuild_required: bool = False

    def to_payload(self) -> dict[str, object]:
        return {
            "current_profile": self.current_profile,
            "last_turn_index": self.last_turn_index,
            "resident_gpu_mib": self.resident_gpu_mib,
            "offloaded_cpu_mib": self.offloaded_cpu_mib,
            "dropped_mib": self.dropped_mib,
            "rebuild_required": self.rebuild_required,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SessionObjectState":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            current_profile=str(payload.get("current_profile") or ""),
            last_turn_index=int(payload.get("last_turn_index") or -1),
            resident_gpu_mib=float(payload.get("resident_gpu_mib") or 0.0),
            offloaded_cpu_mib=float(payload.get("offloaded_cpu_mib") or 0.0),
            dropped_mib=float(payload.get("dropped_mib") or 0.0),
            rebuild_required=bool(payload.get("rebuild_required", False)),
        )


@dataclass(frozen=True)
class SessionRuntimeState:
    sessions: dict[str, SessionObjectState] = field(default_factory=dict)

    def _session(self, session_id: str) -> SessionObjectState:
        return self.sessions.get(session_id, SessionObjectState())

    def _replace_session(self, session_id: str, session: SessionObjectState) -> "SessionRuntimeState":
        sessions = dict(self.sessions)
        sessions[session_id] = session
        return replace(self, sessions=sessions)

    def reset_session(self, session_id: str, profile: str = "") -> "SessionRuntimeState":
        return self._replace_session(
            session_id,
            SessionObjectState(current_profile=profile),
        )

    def record_resident(self, session_id: str, profile: str, *, turn_index: int, kv_mib: float) -> "SessionRuntimeState":
        session = self._session(session_id)
        return self._replace_session(
            session_id,
            replace(
                session,
                current_profile=profile,
                last_turn_index=turn_index,
                resident_gpu_mib=session.resident_gpu_mib + max(0.0, kv_mib),
                rebuild_required=False,
            ),
        )

    def record_offload(self, session_id: str, profile: str, *, kv_mib: float) -> "SessionRuntimeState":
        session = self._session(session_id)
        moved = min(session.resident_gpu_mib, max(0.0, kv_mib))
        return self._replace_session(
            session_id,
            replace(
                session,
                current_profile=profile or session.current_profile,
                resident_gpu_mib=session.resident_gpu_mib - moved,
                offloaded_cpu_mib=session.offloaded_cpu_mib + moved,
            ),
        )

    def record_restore(self, session_id: str, profile: str, *, kv_mib: float) -> "SessionRuntimeState":
        session = self._session(session_id)
        moved = min(session.offloaded_cpu_mib, max(0.0, kv_mib))
        return self._replace_session(
            session_id,
            replace(
                session,
                current_profile=profile or session.current_profile,
                resident_gpu_mib=session.resident_gpu_mib + moved,
                offloaded_cpu_mib=session.offloaded_cpu_mib - moved,
            ),
        )

    def record_drop(self, session_id: str, profile: str, *, kv_mib: float) -> "SessionRuntimeState":
        session = self._session(session_id)
        dropped = min(session.resident_gpu_mib, max(0.0, kv_mib))
        return self._replace_session(
            session_id,
            replace(
                session,
                current_profile=profile or session.current_profile,
                resident_gpu_mib=session.resident_gpu_mib - dropped,
                dropped_mib=session.dropped_mib + dropped,
                rebuild_required=True if dropped > 0 else session.rebuild_required,
            ),
        )

    def switch_profile(self, session_id: str, *, from_profile: str, to_profile: str, rebuild_required: bool) -> "SessionRuntimeState":
        session = self._session(session_id)
        if from_profile and session.current_profile and session.current_profile != from_profile:
            from_profile = session.current_profile
        return self._replace_session(
            session_id,
            replace(session, current_profile=to_profile, rebuild_required=rebuild_required),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "sessions": {
                session_id: session.to_payload()
                for session_id, session in self.sessions.items()
            }
        }

    @classmethod
    def from_payload(cls, payload: object) -> "SessionRuntimeState":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls()
        raw_sessions = payload.get("sessions")
        if not isinstance(raw_sessions, dict):
            return cls()
        return cls(
            sessions={
                str(session_id): SessionObjectState.from_payload(session_payload)
                for session_id, session_payload in raw_sessions.items()
            }
        )


def project_session_memory(state: SessionRuntimeState, profile: str) -> float:
    total = 0.0
    for session in state.sessions.values():
        if session.current_profile == profile:
            total += session.resident_gpu_mib + session.offloaded_cpu_mib
    return total


def apply_budget_policy(
    state: SessionRuntimeState,
    *,
    session_id: str,
    profile: str,
    turn_index: int,
    kv_incremental_mib: float,
    memory_budget_mib: float | None,
    prefer_restore: bool = False,
) -> tuple[SessionRuntimeState, list[dict[str, object]], bool, float]:
    events: list[dict[str, object]] = []
    resident_before = state.sessions.get(session_id, SessionObjectState()).resident_gpu_mib
    next_state = state.record_resident(session_id, profile, turn_index=turn_index, kv_mib=kv_incremental_mib)
    session = next_state.sessions[session_id]
    budget_hit = False
    restore_ms = 0.0
    events.append({"event": "resident", "session_id": session_id, "profile": profile, "turn_index": turn_index})
    budget = float(memory_budget_mib) if memory_budget_mib is not None else None
    if budget is not None and budget >= 0 and session.resident_gpu_mib > budget:
        budget_hit = True
        overflow = session.resident_gpu_mib - budget
        next_state = next_state.record_offload(session_id, profile, kv_mib=overflow)
        events.append({"event": "evict", "session_id": session_id, "profile": profile, "turn_index": turn_index, "kv_mib": overflow})
        if prefer_restore:
            restore_amount = min(overflow, next_state.sessions[session_id].offloaded_cpu_mib)
            next_state = next_state.record_restore(session_id, profile, kv_mib=restore_amount)
            restore_ms = max(1.0, restore_amount)
            events.append({"event": "restore", "session_id": session_id, "profile": profile, "turn_index": turn_index, "kv_mib": restore_amount})
    if next_state.sessions[session_id].dropped_mib > 0:
        events.append({"event": "recompute", "session_id": session_id, "profile": profile, "turn_index": turn_index})
    _ = resident_before
    return next_state, events, budget_hit, restore_ms
