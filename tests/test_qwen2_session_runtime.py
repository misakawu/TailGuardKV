from __future__ import annotations

import pytest

from profiles import qwen2_kv_runtime
from profiles.qwen2_kv_runtime import run_profile
from run_util.canonical_history import canonical_history_hash
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


def test_kivi_canonical_history_reuses_prefix_and_rejects_forged_hash() -> None:
    class FakeIds:
        def tolist(self):
            return [[1, 2, 3]]

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            del prompt, return_tensors
            return {"input_ids": FakeIds()}

    history = ["User: first", "Assistant: generated"]
    payload = {
        "profile": "kivi_4bit_residual32",
        "session_id": "s1",
        "prompt": "User: first\nAssistant: generated\nsecond",
        "history_turns": history,
        "canonical_history": history,
        "canonical_history_mode": "full_gpu_generated_v1",
        "canonical_history_source_profile": "full_gpu",
        "canonical_history_hash": canonical_history_hash(history),
    }
    runtime = {
        "tokenizer": FakeTokenizer(),
        "session_reuse": {
            "s1": {
                "profile": "kivi_4bit_residual32",
                "cache": object(),
                "prompt_token_ids": [1, 2],
            }
        },
    }

    attached = qwen2_kv_runtime._attach_kivi_session_cache(runtime, payload)

    assert attached["_runtime_cache_rebuild_reason"] == ""
    assert attached["_runtime_cached_prompt_token_ids"] == [1, 2]

    payload["canonical_history_hash"] = "forged"
    with pytest.raises(ValueError, match="canonical_history_mismatch"):
        qwen2_kv_runtime._attach_kivi_session_cache(runtime, payload)


def test_all_qwen_profiles_fail_closed_on_forged_canonical_history() -> None:
    history = ["User: first", "Assistant: generated"]
    for profile in ("full_gpu", "kivi_4bit_residual32", "h2o_heavy10_recent10"):
        result = run_profile(
            {
                "profile": profile,
                "session_id": "s1",
                "prompt": "second",
                "history_turns": history,
                "canonical_history": history,
                "canonical_history_mode": "full_gpu_generated_v1",
                "canonical_history_source_profile": "full_gpu",
                "canonical_history_hash": "forged",
            }
        )

        assert result["ok"] is False
        assert "canonical_history_mismatch" in result["error"]


@pytest.mark.parametrize("profile", ("full_gpu", "h2o_heavy10_recent10"))
def test_full_and_h2o_canonical_cache_require_a_strict_prompt_prefix(profile: str) -> None:
    class FakeIds:
        def tolist(self):
            return [[1, 2, 3]]

    history = ["User: first", "Assistant: generated"]
    payload = {
        "profile": profile, "session_id": "s1", "turn_index": 1, "prompt": "second",
        "history_turns": history, "canonical_history": history,
        "canonical_history_mode": "full_gpu_generated_v1", "canonical_history_source_profile": "full_gpu",
        "canonical_history_hash": canonical_history_hash(history),
    }
    runtime = {"tokenizer": lambda prompt, return_tensors="pt": {"input_ids": FakeIds()}, "session_reuse": {
        "s1": {"profile": profile, "cache": object(), "prompt_token_ids": [1, 2], "canonical_history_hash": canonical_history_hash(history), "last_turn": 0}
    }}
    attached = qwen2_kv_runtime._attach_session_cache(runtime, payload)
    assert attached["_runtime_cached_prompt_token_ids"] == [1, 2]
    runtime["session_reuse"]["s1"]["prompt_token_ids"] = [9]
    with pytest.raises(ValueError, match="canonical_history_mismatch"):
        qwen2_kv_runtime._attach_session_cache(runtime, payload)


def test_full_session_cache_rebuilds_noncanonical_pressure_replay() -> None:
    class FakeIds:
        def tolist(self):
            return [[9, 8, 7]]

    runtime = {
        "tokenizer": lambda prompt, return_tensors="pt": {"input_ids": FakeIds()},
        "session_reuse": {"s1": {"profile": "full_gpu", "cache": object(), "prompt_token_ids": [1, 2], "last_turn": 0}},
    }

    attached = qwen2_kv_runtime._attach_session_cache(
        runtime, {"profile": "full_gpu", "session_id": "s1", "turn_index": 1, "prompt": "pressure replay"}
    )

    assert attached["_runtime_cache_rebuild_reason"] == "noncanonical_history"
    assert "s1" not in runtime["session_reuse"]
