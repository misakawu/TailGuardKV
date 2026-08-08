from __future__ import annotations

from backends.measured_replay import MeasuredReplayBackend
from policies.quality_oracle import QualityOraclePolicy
from policies.tailguard import TailGuardPolicy
from run_util.core_types import CacheState, DeviceState, ProfileMeasurement, Request


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
