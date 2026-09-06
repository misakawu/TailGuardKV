from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class Request:
    request_id: str
    task: str
    prompt: str
    reference: str | None = None
    session_id: str | None = None
    turn_index: int = 0
    arrival_index: int = 0
    history_turns: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    _prompt_chars: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_prompt_chars", len(self.effective_prompt))

    @property
    def prompt_chars(self) -> int:
        return self._prompt_chars

    @property
    def effective_prompt(self) -> str:
        if not self.history_turns:
            return self.prompt
        return "\n".join([*self.history_turns, self.prompt])

    @property
    def is_session_start(self) -> bool:
        return self.turn_index == 0

    @property
    def is_session_end(self) -> bool:
        return str(self.metadata.get("last_turn", "")).lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    family: str
    env: str
    lossy: bool
    exact: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileActionCost:
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    kv_cache_memory_mib: float | None = None
    kv_incremental_mib: float | None = None
    quality_loss: float | None = None


@dataclass(frozen=True)
class ProfileStateTransition:
    resident_memory_mib: float | None = None
    kv_cumulative_mib: float | None = None
    resident_kv_mib_before: float | None = None
    resident_kv_mib_after: float | None = None
    restore_ms: float | None = None
    recompute_ms: float | None = None
    evicted_kv_mib: float | None = None
    budget_hit: bool = False


@dataclass(frozen=True)
class ProfileMeasurement:
    request_id: str
    profile: str
    adapter: str
    ok: bool
    measured: bool
    session_id: str | None = None
    turn_index: int = 0
    output_text: str = ""
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    kv_cache_memory_mib: float | None = None
    resident_memory_mib: float | None = None
    kv_incremental_mib: float | None = None
    kv_cumulative_mib: float | None = None
    resident_kv_mib_before: float | None = None
    resident_kv_mib_after: float | None = None
    restore_ms: float | None = None
    recompute_ms: float | None = None
    evicted_kv_mib: float | None = None
    budget_hit: bool = False
    quality_score: float | None = None
    quality_loss: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def action_cost(self) -> ProfileActionCost:
        return ProfileActionCost(
            latency_ms=self.latency_ms,
            ttft_ms=self.ttft_ms,
            peak_memory_mib=self.peak_memory_mib,
            kv_cache_memory_mib=self.kv_cache_memory_mib,
            kv_incremental_mib=self.kv_incremental_mib,
            quality_loss=self.quality_loss,
        )

    @property
    def state_transition(self) -> ProfileStateTransition:
        return ProfileStateTransition(
            resident_memory_mib=self.resident_memory_mib,
            kv_cumulative_mib=self.kv_cumulative_mib,
            resident_kv_mib_before=self.resident_kv_mib_before,
            resident_kv_mib_after=self.resident_kv_mib_after,
            restore_ms=self.restore_ms,
            recompute_ms=self.recompute_ms,
            evicted_kv_mib=self.evicted_kv_mib,
            budget_hit=self.budget_hit,
        )

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        for key, value in extra.items():
            if key in {"task", "length_bucket", "split"}:
                row[key] = value
            else:
                row[f"extra_{key}"] = value
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ProfileMeasurement":
        extra: dict[str, Any] = {}
        payload = dict(row)
        for key in list(payload):
            if key.startswith("extra_"):
                extra[key.removeprefix("extra_")] = payload.pop(key)
            elif key in {"task", "length_bucket", "split"}:
                extra[key] = payload.pop(key)

        return cls(
            request_id=str(payload.get("request_id", "")),
            session_id=_parse_optional_str(payload.get("session_id")),
            turn_index=_parse_int(payload.get("turn_index")),
            profile=str(payload.get("profile", "")),
            adapter=str(payload.get("adapter", "")),
            ok=_parse_bool(payload.get("ok")),
            measured=_parse_bool(payload.get("measured")),
            output_text=str(payload.get("output_text") or ""),
            error=_parse_optional_str(payload.get("error")),
            latency_ms=_parse_optional_float(payload.get("latency_ms")),
            ttft_ms=_parse_optional_float(payload.get("ttft_ms")),
            peak_memory_mib=_parse_optional_float(payload.get("peak_memory_mib")),
            kv_cache_memory_mib=_parse_optional_float(payload.get("kv_cache_memory_mib")),
            resident_memory_mib=_parse_optional_float(payload.get("resident_memory_mib")),
            kv_incremental_mib=_parse_optional_float(payload.get("kv_incremental_mib")),
            kv_cumulative_mib=_parse_optional_float(payload.get("kv_cumulative_mib")),
            resident_kv_mib_before=_parse_optional_float(payload.get("resident_kv_mib_before")),
            resident_kv_mib_after=_parse_optional_float(payload.get("resident_kv_mib_after")),
            restore_ms=_parse_optional_float(payload.get("restore_ms")),
            recompute_ms=_parse_optional_float(payload.get("recompute_ms")),
            evicted_kv_mib=_parse_optional_float(payload.get("evicted_kv_mib")),
            budget_hit=_parse_bool(payload.get("budget_hit")),
            quality_score=_parse_optional_float(payload.get("quality_score")),
            quality_loss=_parse_optional_float(payload.get("quality_loss")),
            extra=extra,
        )


@dataclass(frozen=True)
class BackendResult:
    request_id: str
    profile: str
    ok: bool
    measured: bool
    session_id: str | None = None
    turn_index: int = 0
    output_text: str = ""
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    kv_cache_memory_mib: float | None = None
    resident_memory_mib: float | None = None
    kv_incremental_mib: float | None = None
    kv_cumulative_mib: float | None = None
    resident_kv_mib_before: float | None = None
    resident_kv_mib_after: float | None = None
    restore_ms: float | None = None
    recompute_ms: float | None = None
    queue_delay_ms: float | None = None
    evicted_kv_mib: float | None = None
    budget_hit: bool = False
    global_resident_kv_mib: float | None = None
    global_budget_mib: float | None = None
    quality_loss: float | None = None
    backend_name: str = ""
    replay_source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_profile_measurement(
        cls,
        measurement: ProfileMeasurement,
        *,
        backend_name: str = "",
        replay_source: str = "measured_profile_table",
        extra: dict[str, Any] | None = None,
    ) -> "BackendResult":
        payload = dict(measurement.extra)
        if extra:
            payload.update(extra)
        return cls(
            request_id=measurement.request_id,
            session_id=measurement.session_id,
            turn_index=measurement.turn_index,
            profile=measurement.profile,
            ok=measurement.ok,
            measured=measurement.measured,
            output_text=measurement.output_text,
            error=measurement.error,
            latency_ms=measurement.latency_ms,
            ttft_ms=measurement.ttft_ms,
            peak_memory_mib=measurement.peak_memory_mib,
            kv_cache_memory_mib=measurement.kv_cache_memory_mib,
            resident_memory_mib=measurement.resident_memory_mib,
            kv_incremental_mib=measurement.kv_incremental_mib,
            kv_cumulative_mib=measurement.kv_cumulative_mib,
            resident_kv_mib_before=measurement.resident_kv_mib_before,
            resident_kv_mib_after=measurement.resident_kv_mib_after,
            restore_ms=measurement.restore_ms,
            recompute_ms=measurement.recompute_ms,
            queue_delay_ms=None,
            evicted_kv_mib=measurement.evicted_kv_mib,
            budget_hit=measurement.budget_hit,
            global_resident_kv_mib=None,
            global_budget_mib=None,
            quality_loss=measurement.quality_loss,
            backend_name=backend_name,
            replay_source=replay_source,
            extra=payload,
        )


@dataclass(frozen=True)
class SmokeResult:
    adapter: str
    env: str
    ok: bool
    profiles: tuple[str, ...]
    detail: str = ""
    error: str | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["profiles"] = ",".join(self.profiles)
        versions = row.pop("versions")
        for key, value in versions.items():
            row[f"version_{key}"] = value
        return row


@dataclass(frozen=True)
class Action:
    profile: str
    reason: str = ""
    mode: str = ""
    projected_memory_mib: float | None = None
    pred_loss: float | None = None
    risk_upper: float | None = None
    safe: bool | None = None
    epsilon: float | None = None
    delta: float | None = None
    fallback_reason: str = ""
    safety_reason: str = ""
    rejected_profile: str = ""
    rejected_pred_loss: float | None = None
    rejected_risk_upper: float | None = None
    candidate_safe_count: float | None = None
    controller_overhead_ms: float | None = None
    controller_qrp_ms: float | None = None
    controller_cg_ms: float | None = None
    controller_stc_ms: float | None = None
    oracle_cost_ms: float | None = None
    optimality_gap: float | None = None
    audit_rate: float | None = None
    drift_state: str = ""
    budget_hit: bool = False
    policy_budget_filtered: bool = False


@dataclass(frozen=True)
class CandidateAction:
    profile: str
    predicted_ttft_ms: float
    projected_memory_mib: float
    pred_loss: float
    risk_upper: float
    safe: bool
    exact: bool = False
    within_memory_budget: bool = True
    reason: str = ""


@dataclass(frozen=True)
class SessionState:
    session_id: str
    profile: str
    turn_index: int
    cumulative_kv_mib: float
    resident_kv_mib: float
    offloaded_kv_mib: float = 0.0
    dropped_kv_mib: float = 0.0
    last_access_index: int = 0


@dataclass(frozen=True)
class CacheEvent:
    event_kind: str
    event_reason: str
    session_id: str
    profile: str
    turn_index: int
    kv_mib: float = 0.0
    victim_session: str = ""
    budget_resolution: str = ""


@dataclass(frozen=True)
class ReplayState:
    global_resident_kv_mib: float
    global_budget_mib: float
    sessions: tuple[SessionState, ...] = ()


@dataclass(frozen=True)
class ActionDecision:
    profile: str
    reason: str
    mode: str
    projected_memory_mib: float | None = None
    pred_loss: float | None = None
    risk_upper: float | None = None
    safe: bool | None = None
    epsilon: float | None = None
    delta: float | None = None
    fallback_reason: str = ""
    safety_reason: str = ""
    rejected_profile: str = ""
    rejected_pred_loss: float | None = None
    rejected_risk_upper: float | None = None
    candidate_safe_count: float | None = None
    controller_overhead_ms: float | None = None
    controller_qrp_ms: float | None = None
    controller_cg_ms: float | None = None
    controller_stc_ms: float | None = None
    oracle_cost_ms: float | None = None
    optimality_gap: float | None = None
    audit_rate: float | None = None
    drift_state: str = ""
    budget_hit: bool = False
    policy_budget_filtered: bool = False

    def to_action(self) -> Action:
        return Action(
            profile=self.profile,
            reason=self.reason,
            mode=self.mode,
            projected_memory_mib=self.projected_memory_mib,
            pred_loss=self.pred_loss,
            risk_upper=self.risk_upper,
            safe=self.safe,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason=self.fallback_reason,
            safety_reason=self.safety_reason,
            rejected_profile=self.rejected_profile,
            rejected_pred_loss=self.rejected_pred_loss,
            rejected_risk_upper=self.rejected_risk_upper,
            candidate_safe_count=self.candidate_safe_count,
            controller_overhead_ms=self.controller_overhead_ms,
            controller_qrp_ms=self.controller_qrp_ms,
            controller_cg_ms=self.controller_cg_ms,
            controller_stc_ms=self.controller_stc_ms,
            oracle_cost_ms=self.oracle_cost_ms,
            optimality_gap=self.optimality_gap,
            audit_rate=self.audit_rate,
            drift_state=self.drift_state,
            budget_hit=self.budget_hit,
            policy_budget_filtered=self.policy_budget_filtered or self.budget_hit,
        )


@dataclass(frozen=True)
class CacheState:
    resident_memory_mib: float = 0.0
    objects: tuple[str, ...] = ()
    global_resident_kv_mib: float = 0.0
    global_budget_mib: float = float("inf")
    active_sessions: tuple[str, ...] = ()
    session_cumulative_kv_mib: dict[str, dict[str, float]] = field(default_factory=dict)
    session_resident_kv_mib: dict[str, dict[str, float]] = field(default_factory=dict)
    session_offloaded_kv_mib: dict[str, float] = field(default_factory=dict)
    session_dropped_kv_mib: dict[str, float] = field(default_factory=dict)
    session_current_profile: dict[str, str] = field(default_factory=dict)
    session_last_turn_index: dict[str, int] = field(default_factory=dict)
    session_last_access_index: dict[str, int] = field(default_factory=dict)

    def get_cumulative_kv(self, session_id: str | None, profile: str) -> float:
        if not session_id:
            return 0.0
        return float(self.session_cumulative_kv_mib.get(session_id, {}).get(profile, 0.0))

    def get_resident_kv(self, session_id: str | None, profile: str) -> float:
        if not session_id:
            return 0.0
        return float(self.session_resident_kv_mib.get(session_id, {}).get(profile, 0.0))

    def get_current_profile(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        return self.session_current_profile.get(session_id)

    def get_offloaded_kv(self, session_id: str | None) -> float:
        if not session_id:
            return 0.0
        return float(self.session_offloaded_kv_mib.get(session_id, 0.0))

    def get_dropped_kv(self, session_id: str | None) -> float:
        if not session_id:
            return 0.0
        return float(self.session_dropped_kv_mib.get(session_id, 0.0))

    def with_session_turn(
        self,
        session_id: str | None,
        *,
        profile: str,
        turn_index: int,
        cumulative_kv_mib: float,
        resident_kv_mib: float | None = None,
        offloaded_kv_mib: float | None = None,
        dropped_kv_mib: float | None = None,
        access_index: int | None = None,
        global_resident_kv_mib: float | None = None,
        global_budget_mib: float | None = None,
    ) -> "CacheState":
        if not session_id:
            return self
        cumulative = {key: dict(value) for key, value in self.session_cumulative_kv_mib.items()}
        resident = {key: dict(value) for key, value in self.session_resident_kv_mib.items()}
        offloaded = dict(self.session_offloaded_kv_mib)
        dropped = dict(self.session_dropped_kv_mib)
        current_profile = dict(self.session_current_profile)
        last_turn = dict(self.session_last_turn_index)
        last_access = dict(self.session_last_access_index)
        cumulative.setdefault(session_id, {})[profile] = cumulative_kv_mib
        resident.setdefault(session_id, {})[profile] = resident_kv_mib if resident_kv_mib is not None else cumulative_kv_mib
        if offloaded_kv_mib is not None:
            offloaded[session_id] = offloaded_kv_mib
        if dropped_kv_mib is not None:
            dropped[session_id] = dropped_kv_mib
        current_profile[session_id] = profile
        last_turn[session_id] = turn_index
        if access_index is not None:
            last_access[session_id] = access_index
        active_sessions = tuple(sorted(current_profile))
        return replace(
            self,
            global_resident_kv_mib=self.global_resident_kv_mib if global_resident_kv_mib is None else global_resident_kv_mib,
            global_budget_mib=self.global_budget_mib if global_budget_mib is None else global_budget_mib,
            active_sessions=active_sessions,
            session_cumulative_kv_mib=cumulative,
            session_resident_kv_mib=resident,
            session_offloaded_kv_mib=offloaded,
            session_dropped_kv_mib=dropped,
            session_current_profile=current_profile,
            session_last_turn_index=last_turn,
            session_last_access_index=last_access,
        )


@dataclass(frozen=True)
class DeviceState:
    gpu_free_mib: float | None = None
    gpu_total_mib: float | None = None
    concurrency: int = 1


@dataclass(frozen=True)
class PolicyRunRecord:
    policy: str
    request_id: str
    action_profile: str
    ok: bool
    measured: bool
    backend_name: str = ""
    session_id: str | None = None
    turn_index: int = 0
    task: str = "unknown"
    length_bucket: str = "unknown"
    placeholder: bool = False
    reason: str = ""
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    kv_cache_memory_mib: float | None = None
    resident_memory_mib: float | None = None
    kv_cumulative_mib: float | None = None
    kv_incremental_mib: float | None = None
    resident_kv_mib_before: float | None = None
    resident_kv_mib_after: float | None = None
    restore_ms: float | None = None
    recompute_ms: float | None = None
    queue_delay_ms: float | None = None
    evicted_kv_mib: float | None = None
    budget_hit: bool = False
    policy_budget_filtered: bool = False
    backend_budget_hit: bool = False
    global_resident_kv_mib: float | None = None
    global_budget_mib: float | None = None
    quality_loss: float | None = None
    audit_selected: bool = False
    predicted_quality_loss: float | None = None
    observed_quality_loss: float | None = None
    quality_estimate: float | None = None
    primary_profile: str = ""
    exact: bool = False
    oracle: bool = False
    pred_loss: float | None = None
    risk_upper: float | None = None
    safe: bool | None = None
    epsilon: float | None = None
    delta: float | None = None
    fallback_reason: str = ""
    safety_reason: str = ""
    rejected_profile: str = ""
    rejected_pred_loss: float | None = None
    rejected_risk_upper: float | None = None
    candidate_safe_count: float | None = None
    controller_overhead_ms: float | None = None
    controller_qrp_ms: float | None = None
    controller_cg_ms: float | None = None
    controller_stc_ms: float | None = None
    oracle_cost_ms: float | None = None
    optimality_gap: float | None = None
    audit_rate: float | None = None
    drift_state: str = ""
    active_session_count: float | None = None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_action_and_backend_result(
        cls,
        *,
        policy_name: str,
        request: Request,
        action_profile: str,
        action_reason: str,
        placeholder: bool,
        exact_profiles: set[str],
        oracle: bool,
        backend_result: BackendResult,
        error: str | None = None,
        pred_loss: float | None = None,
        risk_upper: float | None = None,
        safe: bool | None = None,
        epsilon: float | None = None,
        delta: float | None = None,
        fallback_reason: str = "",
        safety_reason: str = "",
        rejected_profile: str = "",
        rejected_pred_loss: float | None = None,
        rejected_risk_upper: float | None = None,
        candidate_safe_count: float | None = None,
        controller_overhead_ms: float | None = None,
        controller_qrp_ms: float | None = None,
        controller_cg_ms: float | None = None,
        controller_stc_ms: float | None = None,
        oracle_cost_ms: float | None = None,
        optimality_gap: float | None = None,
        audit_rate: float | None = None,
        drift_state: str = "",
        active_session_count: float | None = None,
        budget_hit: bool = False,
        policy_budget_filtered: bool | None = None,
        audit_selected: bool = False,
        predicted_quality_loss: float | None = None,
        observed_quality_loss: float | None = None,
        quality_estimate: float | None = None,
        primary_profile: str = "",
    ) -> "PolicyRunRecord":
        policy_filtered = budget_hit if policy_budget_filtered is None else policy_budget_filtered
        backend_budget_hit = bool(backend_result.budget_hit)
        return cls(
            policy=policy_name,
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            task=str(request.task or request.metadata.get("task") or "unknown"),
            length_bucket=str(request.metadata.get("length_bucket") or backend_result.extra.get("length_bucket") or "unknown"),
            action_profile=action_profile,
            ok=backend_result.ok,
            measured=backend_result.measured,
            backend_name=backend_result.backend_name,
            placeholder=placeholder,
            reason=action_reason,
            error=error if error is not None else backend_result.error,
            latency_ms=backend_result.latency_ms,
            ttft_ms=backend_result.ttft_ms,
            peak_memory_mib=backend_result.peak_memory_mib,
            kv_cache_memory_mib=backend_result.kv_cache_memory_mib,
            resident_memory_mib=backend_result.resident_memory_mib,
            kv_cumulative_mib=backend_result.kv_cumulative_mib,
            kv_incremental_mib=backend_result.kv_incremental_mib,
            resident_kv_mib_before=backend_result.resident_kv_mib_before,
            resident_kv_mib_after=backend_result.resident_kv_mib_after,
            restore_ms=backend_result.restore_ms,
            recompute_ms=backend_result.recompute_ms,
            queue_delay_ms=backend_result.queue_delay_ms,
            evicted_kv_mib=backend_result.evicted_kv_mib,
            budget_hit=backend_budget_hit,
            policy_budget_filtered=policy_filtered,
            backend_budget_hit=backend_budget_hit,
            global_resident_kv_mib=backend_result.global_resident_kv_mib,
            global_budget_mib=backend_result.global_budget_mib,
            quality_loss=backend_result.quality_loss,
            audit_selected=audit_selected,
            predicted_quality_loss=predicted_quality_loss,
            observed_quality_loss=observed_quality_loss,
            quality_estimate=quality_estimate,
            primary_profile=primary_profile,
            exact=action_profile in exact_profiles,
            oracle=oracle,
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            epsilon=epsilon,
            delta=delta,
            fallback_reason=fallback_reason,
            safety_reason=safety_reason,
            rejected_profile=rejected_profile,
            rejected_pred_loss=rejected_pred_loss,
            rejected_risk_upper=rejected_risk_upper,
            candidate_safe_count=candidate_safe_count,
            controller_overhead_ms=controller_overhead_ms,
            controller_qrp_ms=controller_qrp_ms,
            controller_cg_ms=controller_cg_ms,
            controller_stc_ms=controller_stc_ms,
            oracle_cost_ms=oracle_cost_ms,
            optimality_gap=optimality_gap,
            audit_rate=audit_rate,
            drift_state=drift_state,
            active_session_count=active_session_count,
        )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _parse_int(value: Any) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))
