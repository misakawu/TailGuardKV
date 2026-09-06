from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from backends.qwen_session import OnlineQwenSessionBackend
from profiles.full import FullKVAdapter
from profiles.kivi import KIVIAdapter
from profiles import qwen2_kv_runtime
from run_util.core_types import Action, CacheState, Request
from run_util.run_policies import _configured_backend_name, _load_online_evaluation_requests


class FakeOnlineWorker:
    def __init__(self, kv_by_request: dict[str, float | None]) -> None:
        self.kv_by_request = kv_by_request
        self.messages: list[dict[str, object]] = []
        self.closed = False

    def request(self, message: dict[str, object], *, timeout_s: int) -> dict[str, object]:
        del timeout_s
        self.messages.append(message)
        requests = list(message.get("requests") or [])
        if not requests:
            return {
                "ok": True,
                "results": [],
                "worker": {"mode": "persistent"},
                "session_runtime_state": message.get("session_runtime_state") or {"sessions": {}},
            }
        request = requests[0]
        request_id = str(request["request_id"])
        kv_mib = self.kv_by_request[request_id]
        if kv_mib is None:
            return {
                "ok": False,
                "worker": {"mode": "persistent"},
                "results": [{"ok": False, "measured": False, "error": "injected online failure"}],
                "session_runtime_state": message.get("session_runtime_state") or {"sessions": {}},
            }
        return {
            "ok": True,
            "worker": {"mode": "persistent"},
            "results": [
                {
                    "ok": True,
                    "measured": True,
                    "output_text": f"generated:{request_id}",
                    "latency_ms": 8.0,
                    "ttft_ms": 3.0,
                    "stage_prefill_ms": 2.5,
                    "peak_memory_mib": kv_mib + 1.0,
                    "kv_cache_memory_mib": kv_mib,
                    "resident_memory_mib": kv_mib,
                }
            ],
            "session_runtime_state": message.get("session_runtime_state") or {"sessions": {}},
        }

    def close(self) -> None:
        self.closed = True


def _backend(
    kv_by_request: dict[str, float | None],
    *,
    budget_mib: float = 100.0,
) -> tuple[OnlineQwenSessionBackend, dict[str, FakeOnlineWorker]]:
    workers: dict[str, FakeOnlineWorker] = {}

    def factory(**kwargs) -> FakeOnlineWorker:
        worker = FakeOnlineWorker(kv_by_request)
        workers[str(kwargs["adapter"])] = worker
        return worker

    backend = OnlineQwenSessionBackend(
        adapters=[
            FullKVAdapter({"pilot_model": "/fake/qwen", "max_new_tokens": 4}),
            KIVIAdapter({"pilot_model": "/fake/qwen", "max_new_tokens": 4}),
        ],
        global_budget_mib=budget_mib,
        worker_factory=factory,
    )
    return backend, workers


def test_same_profile_consecutive_turn_reuses_resident_kv_and_fixture_history() -> None:
    backend, workers = _backend({"s1_t0": 10.0, "s1_t1": 14.0})
    first_request = Request("s1_t0", "chat", "hello", session_id="s1", turn_index=0, arrival_index=0)
    second_request = Request(
        "s1_t1",
        "chat",
        "follow up",
        session_id="s1",
        turn_index=1,
        arrival_index=1,
        history_turns=("User: hello", "Assistant: fixture answer"),
    )

    first = backend.execute(first_request, Action("full_gpu"), CacheState(global_budget_mib=100.0))
    second = backend.execute(second_request, Action("full_gpu"), backend.cache_state)

    assert first.resident_kv_mib_before == 0.0
    assert second.resident_kv_mib_before == 10.0
    assert second.resident_kv_mib_after == 14.0
    assert second.kv_incremental_mib == 4.0
    assert second.recompute_ms == 0.0
    assert second.extra["event_kind"] == "resident"
    run_messages = [message for message in workers["full"].messages if message.get("requests")]
    assert len(run_messages) == 2
    assert run_messages[1]["requests"][0]["prompt"] == second_request.effective_prompt


def test_incompatible_profile_switch_records_online_recompute() -> None:
    backend, _ = _backend({"s1_t0": 10.0, "s1_t1": 6.0})
    turn0 = Request("s1_t0", "chat", "hello", session_id="s1", turn_index=0, arrival_index=0)
    turn1 = Request("s1_t1", "chat", "next", session_id="s1", turn_index=1, arrival_index=1)

    backend.execute(turn0, Action("full_gpu"), CacheState(global_budget_mib=100.0))
    switched = backend.execute(turn1, Action("kivi_4bit_residual32"), backend.cache_state)

    assert switched.backend_name == "online_qwen"
    assert switched.replay_source == ""
    assert switched.resident_kv_mib_before == 0.0
    assert switched.resident_kv_mib_after == 6.0
    assert switched.recompute_ms == 2.5
    assert switched.extra["event_kind"] == "recompute"
    assert switched.extra["event_reason"] == "incompatible_profile_switch"


def test_global_lru_evicts_oldest_session_when_online_measurement_exceeds_budget() -> None:
    backend, workers = _backend({"s1_t0": 10.0, "s2_t0": 10.0}, budget_mib=15.0)
    state = CacheState(global_budget_mib=15.0)
    backend.execute(Request("s1_t0", "chat", "one", session_id="s1", arrival_index=0), Action("full_gpu"), state)
    pressured = backend.execute(
        Request("s2_t0", "chat", "two", session_id="s2", arrival_index=1),
        Action("full_gpu"),
        backend.cache_state,
    )

    assert pressured.budget_hit is True
    assert pressured.evicted_kv_mib == 10.0
    assert pressured.global_resident_kv_mib == 10.0
    assert pressured.queue_delay_ms == 0.0
    assert pressured.restore_ms == 0.0
    assert pressured.extra["victim_session"] == "s1"
    assert backend.cache_state.get_resident_kv("s1", "full_gpu") == 0.0
    assert backend.cache_state.get_dropped_kv("s1") == 10.0
    controls = [message for message in workers["full"].messages if message.get("evict_sessions")]
    assert controls[-1]["evict_sessions"] == ["s1"]


def test_profile_runtime_switch_invalidates_other_sessions_from_the_released_runtime() -> None:
    backend, _ = _backend({"s1_t0": 10.0, "s2_t0": 10.0, "s2_t1": 6.0})
    state = CacheState(global_budget_mib=100.0)
    backend.execute(Request("s1_t0", "chat", "one", session_id="s1", arrival_index=0), Action("full_gpu"), state)
    backend.execute(
        Request("s2_t0", "chat", "two", session_id="s2", arrival_index=1),
        Action("full_gpu"),
        backend.cache_state,
    )

    switched = backend.execute(
        Request("s2_t1", "chat", "next", session_id="s2", turn_index=1, arrival_index=2),
        Action("kivi_4bit_residual32"),
        backend.cache_state,
    )

    assert switched.recompute_ms > 0
    assert switched.evicted_kv_mib == 20.0
    assert backend.cache_state.get_resident_kv("s1", "full_gpu") == 0.0
    assert backend.cache_state.get_dropped_kv("s1") == 10.0
    assert backend.cache_state.global_resident_kv_mib == 6.0
    assert {
        event["session_id"]
        for event in switched.extra["cache_events"]
        if event["event_reason"] == "profile_runtime_released"
    } == {"s1", "s2"}


def test_failed_profile_switch_does_not_resurrect_released_runtime_state() -> None:
    backend, _ = _backend({"s1_t0": 10.0, "s1_t1": None})
    backend.execute(
        Request("s1_t0", "chat", "one", session_id="s1", arrival_index=0),
        Action("full_gpu"),
        CacheState(global_budget_mib=100.0),
    )

    failed = backend.execute(
        Request("s1_t1", "chat", "next", session_id="s1", turn_index=1, arrival_index=1),
        Action("kivi_4bit_residual32"),
        backend.cache_state,
    )

    assert failed.ok is False
    assert failed.evicted_kv_mib == 10.0
    assert backend.cache_state.get_resident_kv("s1", "full_gpu") == 0.0
    assert backend.cache_state.get_dropped_kv("s1") == 10.0


def test_session27_selects_online_backend_but_legacy_configs_remain_explicit_replay() -> None:
    assert _configured_backend_name({"policies": {"backend": "online_qwen"}}) == "online_qwen"
    assert _configured_backend_name({"policies": {"backend": "measured_replay"}}) == "measured_replay"
    assert _configured_backend_name({"policies": {"names": ["full_lru"]}}) == "measured_replay"


def test_online_runner_loads_evaluation_turns_from_fixture_instead_of_profile_outputs(tmp_path: Path) -> None:
    fixture = tmp_path / "requests.jsonl"
    rows = [
        {
            "request_id": "s1_t0",
            "session_id": "s1",
            "turn_index": 0,
            "arrival_index": 0,
            "task": "chat",
            "prompt": "fixture prompt",
            "reference": "fixture answer",
        },
        {
            "request_id": "s1_t1",
            "session_id": "s1",
            "turn_index": 1,
            "arrival_index": 1,
            "task": "chat",
            "prompt": "fixture followup",
            "reference": "fixture answer 2",
        },
        {
            "request_id": "cal_t0",
            "session_id": "cal",
            "turn_index": 0,
            "arrival_index": 2,
            "task": "chat",
            "prompt": "calibration only",
        },
    ]
    fixture.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    config = {"data": {"source": "fixture", "requests": str(fixture), "calibration_fraction": 0.5}}
    evaluation_keys = {("s1", 0, "s1_t0"), ("s1", 1, "s1_t1")}

    requests = _load_online_evaluation_requests(config, evaluation_keys)

    assert [request.request_id for request in requests] == ["s1_t0", "s1_t1"]
    assert requests[0].prompt == "fixture prompt"
    assert requests[1].effective_prompt == "User: fixture prompt\nAssistant: fixture answer\nfixture followup"


def test_qwen_worker_cache_control_removes_resident_session_entry() -> None:
    entry = {"cache": SimpleNamespace()}
    worker_state = {
        "runtime": {"session_reuse": {"s1": entry}},
        "runtime_profile": "full_gpu",
    }

    result = qwen2_kv_runtime.worker_run_batch(
        {
            "requests": [],
            "evict_sessions": ["s1"],
            "session_runtime_state": {"sessions": {}},
        },
        worker_state,
    )

    assert result["evicted_sessions"] == ["s1"]
    assert worker_state["runtime"]["session_reuse"] == {}
    assert entry == {}
