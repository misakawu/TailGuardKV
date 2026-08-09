from __future__ import annotations

from profiles.base import transformers_profile_many_measurements
from profiles.session_runtime import SessionRuntimeState
from profiles.transformers_runtime import _rebuild_session_from_turn0, _run_one_request_with_session
from run_util.core_types import ProfileSpec, Request


def test_transformers_run_profile_batch_round_trips_session_state() -> None:
    runtime = {"kv_estimator": lambda payload: float(len(str(payload["prompt"]))), "clock_ms": 2.0}
    initial = SessionRuntimeState()
    payload_turn0 = {"session_id": "s1", "turn_index": 0, "profile": "full_gpu", "prompt": "hello", "history_turns": []}
    payload_turn1 = {
        "session_id": "s1",
        "turn_index": 1,
        "profile": "full_gpu",
        "prompt": "hello world",
        "history_turns": ["hello"],
    }

    result1, state = _run_one_request_with_session(runtime, payload_turn0, initial)
    assert state.sessions["s1"].resident_gpu_mib == result1["resident_kv_mib_after"]

    result2, next_state = _run_one_request_with_session(runtime, payload_turn1, state)

    assert result2["resident_kv_mib_before"] == result1["resident_kv_mib_after"]
    assert next_state.sessions["s1"].resident_gpu_mib == result2["resident_kv_mib_after"]


def test_transformers_session_runtime_appends_on_same_profile() -> None:
    runtime = {"kv_estimator": lambda payload: float(len(str(payload["prompt"]))), "clock_ms": 2.0}
    state = SessionRuntimeState()
    payload_turn0 = {"session_id": "s1", "turn_index": 0, "profile": "full_gpu", "prompt": "hello", "history_turns": []}
    payload_turn1 = {
        "session_id": "s1",
        "turn_index": 1,
        "profile": "full_gpu",
        "prompt": "hello world",
        "history_turns": ["hello"],
    }

    result1, state = _run_one_request_with_session(runtime, payload_turn0, state)
    result2, state = _run_one_request_with_session(runtime, payload_turn1, state)

    assert result1["kv_incremental_mib"] == result1["kv_cumulative_mib"]
    assert result2["kv_incremental_mib"] < result2["kv_cumulative_mib"]
    assert result2["resident_kv_mib_before"] == result1["resident_kv_mib_after"]


def test_transformers_session_runtime_rebuilds_after_profile_switch() -> None:
    runtime = {"kv_estimator": lambda payload: float(len(str(payload["prompt"]))), "clock_ms": 3.0}
    history_payloads = [
        {"session_id": "s1", "turn_index": 0, "profile": "full_gpu", "prompt": "hello", "history_turns": []},
        {"session_id": "s1", "turn_index": 1, "profile": "full_cpu", "prompt": "hello world", "history_turns": ["hello"]},
    ]

    switched = _rebuild_session_from_turn0(runtime, history_payloads, target_profile="full_cpu")

    assert switched["recompute_ms"] > 0
    assert switched["budget_hit"] in {True, False}
    assert any(event["event"] == "recompute" for event in switched["event_trace"])
