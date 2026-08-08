from __future__ import annotations

from profiles.session_runtime import SessionRuntimeState, project_session_memory
from run_util.core_types import CacheState


def test_session_runtime_tracks_gpu_cpu_drop_and_profile_switch() -> None:
    state = SessionRuntimeState()
    state = state.record_resident("s1", "kivi", turn_index=0, kv_mib=32.0)
    state = state.record_offload("s1", "kivi", kv_mib=12.0)
    state = state.record_drop("s1", "kivi", kv_mib=4.0)
    state = state.switch_profile("s1", from_profile="kivi", to_profile="full_cpu", rebuild_required=True)

    assert state.sessions["s1"].current_profile == "full_cpu"
    assert state.sessions["s1"].rebuild_required is True
    assert state.sessions["s1"].resident_gpu_mib == 16.0
    assert state.sessions["s1"].offloaded_cpu_mib == 12.0
    assert state.sessions["s1"].dropped_mib == 4.0


def test_project_session_memory_uses_resident_and_offloaded_memory() -> None:
    state = SessionRuntimeState()
    state = state.record_resident("s1", "kivi", turn_index=1, kv_mib=20.0)
    state = state.record_offload("s1", "kivi", kv_mib=6.0)

    assert project_session_memory(state, "kivi") == 20.0
    assert project_session_memory(state, "full_cpu") == 0.0


def test_cache_state_tracks_session_cumulative_and_resident_kv() -> None:
    cache_state = CacheState().with_session_turn(
        "s1",
        profile="kivi",
        turn_index=1,
        cumulative_kv_mib=48.0,
        resident_kv_mib=32.0,
    )

    assert cache_state.get_cumulative_kv("s1", "kivi") == 48.0
    assert cache_state.get_resident_kv("s1", "kivi") == 32.0
    assert cache_state.get_current_profile("s1") == "kivi"
