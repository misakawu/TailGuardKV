from __future__ import annotations

from profiles.qwen2_kv_runtime import run_profile
from profiles.session_runtime import SessionRuntimeState


def test_qwen2_runtime_emits_evict_and_restore_events_under_budget_pressure() -> None:
    state = SessionRuntimeState().record_resident("s1", "kivi_4bit_residual32", turn_index=0, kv_mib=40.0)
    turn1 = run_profile(
        {
            "profile": "kivi_4bit_residual32",
            "session_id": "s1",
            "turn_index": 1,
            "prompt": "hello world",
            "history_turns": ["hello"],
            "memory_budget_mib": 50.0,
            "session_runtime_state": state,
            "dry_session": True,
            "restore_from_cpu": True,
        }
    )

    assert any(event["event"] == "evict" for event in turn1["event_trace"])
    assert any(event["event"] == "restore" for event in turn1["event_trace"])
    assert turn1["budget_hit"] is True


def test_qwen2_runtime_uses_recompute_when_cache_was_dropped() -> None:
    state = SessionRuntimeState().record_resident("s1", "kivi_4bit_residual32", turn_index=0, kv_mib=28.0)
    state = state.record_drop("s1", "kivi_4bit_residual32", kv_mib=28.0)
    dropped = run_profile(
        {
            "profile": "kivi_4bit_residual32",
            "session_id": "s1",
            "turn_index": 2,
            "prompt": "hello world again",
            "history_turns": ["hello", "hello world"],
            "memory_budget_mib": 64.0,
            "session_runtime_state": state,
            "dry_session": True,
        }
    )

    assert any(event["event"] == "recompute" for event in dropped["event_trace"])
    assert dropped["recompute_ms"] > 0
