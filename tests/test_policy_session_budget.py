from __future__ import annotations

import json

from backends.measured_replay import MeasuredReplayBackend
from policies.full_lru import FullLRUPolicy
from policies.quality_oracle import QualityOraclePolicy
from policies.static_best import StaticBestPolicy
from policies.static_safe import StaticSafePolicy
from policies.tailguard import TailGuardPolicy
from policies.uncalibrated_dynamic import UncalibratedDynamicPolicy
from policies.utility_dynamic import UtilityDynamicPolicy
from run_util.core_types import BackendResult, CacheState, DeviceState, PolicyRunRecord, ProfileMeasurement, Request
from run_util.data_utils import requests_from_measurements


def _measurement(
    request_id: str,
    profile: str,
    *,
    session_id: str = "s1",
    turn_index: int = 0,
    quality_loss: float = 0.0,
    kv_incremental_mib: float = 20.0,
    kv_cumulative_mib: float = 20.0,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        session_id=session_id,
        turn_index=turn_index,
        profile=profile,
        adapter="test",
        ok=True,
        measured=True,
        output_text=request_id,
        ttft_ms=10.0,
        peak_memory_mib=kv_cumulative_mib,
        kv_cache_memory_mib=kv_cumulative_mib,
        resident_memory_mib=kv_cumulative_mib,
        kv_incremental_mib=kv_incremental_mib,
        kv_cumulative_mib=kv_cumulative_mib,
        resident_kv_mib_before=max(0.0, kv_cumulative_mib - kv_incremental_mib),
        resident_kv_mib_after=kv_cumulative_mib,
        quality_loss=quality_loss,
        extra={"task": "chat", "length_bucket": "short", "split": "calibration"},
    )


def test_tailguard_can_fallback_to_full_cpu_and_recompute() -> None:
    calibration = [
        _measurement("s1_t0", "kivi_4bit_residual32", quality_loss=0.01, kv_incremental_mib=20.0, kv_cumulative_mib=20.0),
        _measurement("s1_t0", "full_cpu", quality_loss=0.0, kv_incremental_mib=40.0, kv_cumulative_mib=40.0),
        _measurement("s1_t0", "recompute", quality_loss=0.0, kv_incremental_mib=45.0, kv_cumulative_mib=45.0),
    ]
    policy = TailGuardPolicy(
        calibration_measurements=calibration,
        profiles=["kivi_4bit_residual32", "full_cpu", "recompute"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_cpu", "recompute"},
        memory_budget_mib=30.0,
    )
    request = Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1)
    cache_state = CacheState().with_session_turn(
        "s1",
        profile="kivi_4bit_residual32",
        turn_index=0,
        cumulative_kv_mib=20.0,
        resident_kv_mib=20.0,
    )

    action = policy.decide(request, cache_state, DeviceState())

    assert action.profile in {"full_cpu", "recompute"}


def test_measured_replay_rebuilds_session_after_profile_switch() -> None:
    backend = MeasuredReplayBackend(
        [
            _measurement("s1_t0", "kivi_4bit_residual32", turn_index=0, kv_incremental_mib=20.0, kv_cumulative_mib=20.0),
            _measurement("s1_t1", "full_cpu", turn_index=1, kv_incremental_mib=35.0, kv_cumulative_mib=55.0),
        ]
    )
    backend.run(
        [
            Request("s1_t0", "chat", "turn0", session_id="s1", turn_index=0),
            Request("s1_t1", "chat", "turn1", session_id="s1", turn_index=1),
        ],
        ["kivi_4bit_residual32", "full_cpu"],
    )

    assert backend.cache_state.get_current_profile("s1") == "full_cpu"


def test_quality_oracle_uses_session_key() -> None:
    policy = QualityOraclePolicy(
        measurements=[_measurement("s1_t1", "full_cpu", turn_index=1, kv_cumulative_mib=55.0)],
        profiles=["full_cpu"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_cpu"},
    )

    action = policy.decide(Request("s1_t1", "chat", "turn1", session_id="s1", turn_index=1), CacheState(), DeviceState())

    assert action.profile == "full_cpu"


def test_requests_from_measurements_restores_original_request_flow() -> None:
    measurements = [
        ProfileMeasurement(
            request_id="session_b_turn_001",
            session_id="session_b",
            turn_index=1,
            profile="full_gpu",
            adapter="test",
            ok=True,
            measured=True,
            output_text="assistant output should not become prompt",
            ttft_ms=12.0,
            latency_ms=20.0,
            peak_memory_mib=32.0,
            kv_cache_memory_mib=32.0,
            quality_loss=0.0,
            extra={
                "task": "chat",
                "length_bucket": "short",
                "split": "eval",
                "arrival_index": 3,
                "prompt_text": "How are you?",
                "history_turns": json.dumps(["User: hello", "Assistant: hi"]),
                "effective_prompt_chars": 30,
            },
        ),
        ProfileMeasurement(
            request_id="session_a_turn_000",
            session_id="session_a",
            turn_index=0,
            profile="full_gpu",
            adapter="test",
            ok=True,
            measured=True,
            output_text="ignored output",
            ttft_ms=10.0,
            latency_ms=18.0,
            peak_memory_mib=20.0,
            kv_cache_memory_mib=20.0,
            quality_loss=0.0,
            extra={
                "task": "chat",
                "length_bucket": "short",
                "split": "eval",
                "arrival_index": 1,
                "prompt_text": "First question",
                "history_turns": json.dumps([]),
                "effective_prompt_chars": 14,
            },
        ),
    ]

    requests = requests_from_measurements(measurements)

    assert [request.request_id for request in requests] == ["session_a_turn_000", "session_b_turn_001"]
    assert requests[1].prompt == "How are you?"
    assert requests[1].history_turns == ("User: hello", "Assistant: hi")
    assert requests[1].arrival_index == 3
    assert requests[1].effective_prompt == "User: hello\nAssistant: hi\nHow are you?"


def test_measured_replay_applies_cross_session_budget_pressure() -> None:
    backend = MeasuredReplayBackend(
        [
            _measurement("s1_t0", "kivi_4bit_residual32", session_id="s1", turn_index=0, kv_incremental_mib=30.0, kv_cumulative_mib=30.0),
            _measurement("s2_t0", "kivi_4bit_residual32", session_id="s2", turn_index=0, kv_incremental_mib=30.0, kv_cumulative_mib=30.0),
            _measurement("s1_t1", "kivi_4bit_residual32", session_id="s1", turn_index=1, kv_incremental_mib=15.0, kv_cumulative_mib=45.0),
        ],
        global_budget_mib=50.0,
    )

    results = backend.run(
        [
            Request("s1_t0", "chat", "turn0", session_id="s1", turn_index=0, arrival_index=0),
            Request("s2_t0", "chat", "turn0", session_id="s2", turn_index=0, arrival_index=1),
            Request("s1_t1", "chat", "turn1", session_id="s1", turn_index=1, arrival_index=2),
        ],
        ["kivi_4bit_residual32"],
    )

    assert results[0].budget_hit is False
    assert results[1].budget_hit is True
    assert (results[1].extra.get("global_resident_kv_mib") or 0.0) <= 50.0
    assert results[2].restore_ms is not None and results[2].restore_ms > 0
    assert results[2].ttft_ms is not None and results[2].ttft_ms > 10.0


def test_uncalibrated_dynamic_uses_cache_state_for_memory_filtering() -> None:
    calibration = [
        _measurement("s1_t0", "kivi_4bit_residual32", quality_loss=0.01, kv_incremental_mib=20.0, kv_cumulative_mib=20.0),
        _measurement("s1_t0", "kivi_2bit_residual32", quality_loss=0.02, kv_incremental_mib=10.0, kv_cumulative_mib=10.0),
        _measurement("s1_t0", "full_cpu", quality_loss=0.0, kv_incremental_mib=40.0, kv_cumulative_mib=40.0),
    ]
    policy = UncalibratedDynamicPolicy(
        calibration_measurements=calibration,
        profiles=["kivi_4bit_residual32", "kivi_2bit_residual32", "full_cpu"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_cpu"},
        memory_budget_mib=25.0,
    )
    request = Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1)
    cache_state = CacheState().with_session_turn(
        "s1",
        profile="kivi_4bit_residual32",
        turn_index=0,
        cumulative_kv_mib=20.0,
        resident_kv_mib=20.0,
    )

    action = policy.decide(request, cache_state, DeviceState())

    assert action.profile == "full_cpu"
    assert action.budget_hit is True
    assert action.candidate_safe_count == 1.0


def test_utility_dynamic_uses_cache_state_for_memory_filtering() -> None:
    calibration = [
        _measurement("s1_t0", "kivi_4bit_residual32", quality_loss=0.01, kv_incremental_mib=20.0, kv_cumulative_mib=20.0),
        _measurement("s1_t0", "kivi_2bit_residual32", quality_loss=0.02, kv_incremental_mib=10.0, kv_cumulative_mib=10.0),
        _measurement("s1_t0", "full_cpu", quality_loss=0.0, kv_incremental_mib=40.0, kv_cumulative_mib=40.0),
    ]
    policy = UtilityDynamicPolicy(
        calibration_measurements=calibration,
        profiles=["kivi_4bit_residual32", "kivi_2bit_residual32", "full_cpu"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_cpu"},
        memory_budget_mib=25.0,
        memory_weight=0.0,
        loss_weight=0.0,
    )
    request = Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1)
    cache_state = CacheState().with_session_turn(
        "s1",
        profile="kivi_4bit_residual32",
        turn_index=0,
        cumulative_kv_mib=20.0,
        resident_kv_mib=20.0,
    )

    action = policy.decide(request, cache_state, DeviceState())

    assert action.profile == "full_cpu"
    assert action.budget_hit is True
    assert action.candidate_safe_count == 1.0


def test_full_lru_reports_state_aware_budget_projection() -> None:
    policy = FullLRUPolicy("full_gpu")
    request = Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1)
    cache_state = CacheState(global_budget_mib=25.0).with_session_turn(
        "s1",
        profile="full_gpu",
        turn_index=0,
        cumulative_kv_mib=20.0,
        resident_kv_mib=20.0,
        global_resident_kv_mib=20.0,
        global_budget_mib=25.0,
    )

    action = policy.decide(request, cache_state, DeviceState())

    assert action.profile == "full_gpu"
    assert action.budget_hit is True
    assert action.reason == "full precision exact profile"


def test_static_best_prefers_lossy_profile_when_lossy_is_deployable() -> None:
    calibration = [
        _measurement("s1_t0", "full_gpu", quality_loss=0.0, kv_incremental_mib=40.0, kv_cumulative_mib=40.0),
        _measurement("s1_t0", "kivi_4bit_residual32", quality_loss=0.01, kv_incremental_mib=18.0, kv_cumulative_mib=18.0),
        _measurement("s1_t0", "kivi_2bit_residual32", quality_loss=0.03, kv_incremental_mib=12.0, kv_cumulative_mib=12.0),
    ]
    policy = StaticBestPolicy(
        calibration_measurements=calibration,
        profiles=["full_gpu", "kivi_4bit_residual32", "kivi_2bit_residual32"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_gpu"},
        memory_budget_mib=32.0,
    )

    action = policy.decide(Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1), CacheState(), DeviceState())

    assert action.profile == "kivi_4bit_residual32"


def test_static_safe_explicitly_marks_exact_fallback_when_no_safe_lossy_exists() -> None:
    calibration = [
        _measurement("s1_t0", "full_gpu", quality_loss=0.0, kv_incremental_mib=40.0, kv_cumulative_mib=40.0),
        _measurement("s1_t0", "kivi_4bit_residual32", quality_loss=0.20, kv_incremental_mib=18.0, kv_cumulative_mib=18.0),
        _measurement("s2_t0", "kivi_4bit_residual32", quality_loss=0.20, kv_incremental_mib=18.0, kv_cumulative_mib=18.0),
    ]
    policy = StaticSafePolicy(
        calibration_measurements=calibration,
        profiles=["full_gpu", "kivi_4bit_residual32"],
        epsilon=0.05,
        delta=0.05,
        exact_profiles={"full_gpu"},
        memory_budget_mib=32.0,
    )

    action = policy.decide(Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1), CacheState(), DeviceState())

    assert action.profile == "full_gpu"
    assert action.fallback_reason


def test_policy_run_record_keeps_policy_and_backend_budget_signals_separate() -> None:
    request = Request("s1_t1", "chat", "next turn", session_id="s1", turn_index=1)
    backend_result = BackendResult(
        request_id="s1_t1",
        session_id="s1",
        turn_index=1,
        profile="full_gpu",
        ok=True,
        measured=True,
        budget_hit=False,
        backend_name="measured_replay",
    )

    record = PolicyRunRecord.from_action_and_backend_result(
        policy_name="uncalibrated_dynamic",
        request=request,
        action_profile="full_gpu",
        action_reason="fallback",
        placeholder=False,
        exact_profiles={"full_gpu"},
        oracle=False,
        backend_result=backend_result,
        budget_hit=True,
    )

    assert record.policy_budget_filtered is True
    assert record.backend_budget_hit is False
    assert record.budget_hit is False
