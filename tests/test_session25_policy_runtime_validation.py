from __future__ import annotations

from dataclasses import replace

import pytest

from run_util.core_types import BackendResult, ProfileMeasurement, Request
from run_util.data_utils import split_measurements
from scripts.validate_session25_policy_runtime import (
    run_scenarios,
    select_validation_sessions,
    validate_scenario,
)


def _requests(session_count: int = 6) -> list[Request]:
    return [
        Request(
            request_id=f"r{session_index}-{turn_index}",
            session_id=f"s{session_index}",
            turn_index=turn_index,
            arrival_index=session_index * 3 + turn_index,
            task="qa",
            prompt=f"prompt {session_index} {turn_index}",
        )
        for session_index in range(session_count)
        for turn_index in range(3)
    ]


def _measurements(requests: list[Request]) -> list[ProfileMeasurement]:
    return [
        ProfileMeasurement(
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            profile="full_gpu",
            adapter="full",
            ok=True,
            measured=True,
            extra={"task": request.task, "length_bucket": "short"},
        )
        for request in requests
    ]


def test_select_validation_sessions_uses_stable_complete_evaluation_sessions() -> None:
    requests = _requests()
    measurements = _measurements(requests)
    config = {
        "data": {
            "split_seed": 20260906,
            "calibration_fraction": 0.5,
            "stratify_session": True,
        }
    }

    selected = select_validation_sessions(config, measurements, requests)
    _, evaluation = split_measurements(
        measurements,
        split_seed=20260906,
        calibration_fraction=0.5,
        stratify_session=True,
    )
    evaluation_ids = {row.request_id for row in evaluation}

    assert len(selected) == 3
    assert [[request.turn_index for request in session] for session in selected] == [[0, 1, 2]] * 3
    assert all(request.request_id in evaluation_ids for session in selected for request in session)
    assert selected == select_validation_sessions(config, measurements, list(reversed(requests)))


def test_select_validation_sessions_rejects_insufficient_complete_evaluation_sessions() -> None:
    requests = _requests(session_count=4)
    with pytest.raises(ValueError, match="three complete evaluation sessions"):
        select_validation_sessions(
            {"data": {"split_seed": 20260906, "calibration_fraction": 0.5, "stratify_session": True}},
            _measurements(requests),
            requests,
        )


@pytest.mark.parametrize(
    "rows,requirements,error",
    [
        ([{"ok": False, "ttft_ms": 1.0, "error": "boom"}], {}, "runtime failure"),
        ([{"ok": True, "ttft_ms": None}], {}, "invalid TTFT"),
        ([{"ok": True, "ttft_ms": 1.0}, {"ok": True, "ttft_ms": 1.0, "cache_reused": False}], {"require_reuse_after_first": True}, "cache reuse"),
        ([{"ok": True, "ttft_ms": 1.0}, {"ok": True, "ttft_ms": 1.0, "runtime_transition_ms": 0.0, "recompute_ms": 0.0}], {"require_transition_last": True}, "transition/recompute"),
    ],
)
def test_validate_scenario_fails_closed(rows, requirements, error: str) -> None:
    result = validate_scenario("case", rows, requirements)
    assert result["ok"] is False
    assert any(error in message for message in result["errors"])


class FakeBackend:
    def __init__(self, *, fail_profile: str = "") -> None:
        self.fail_profile = fail_profile
        self.calls: list[tuple[str, str, int]] = []
        self.closed = False
        self.last_profile = ""
        self.seen: set[tuple[str, str]] = set()

    def execute(self, request: Request, action, cache_state) -> BackendResult:
        del cache_state
        profile = action.profile
        session_id = request.session_id or request.request_id
        reused = (session_id, profile) in self.seen
        transitioned = bool(self.last_profile and self.last_profile != profile)
        self.calls.append((session_id, profile, request.turn_index))
        self.seen.add((session_id, profile))
        self.last_profile = profile
        return BackendResult(
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            profile=profile,
            ok=profile != self.fail_profile,
            measured=profile != self.fail_profile,
            error="forced failure" if profile == self.fail_profile else "",
            output_text="ok",
            latency_ms=2.0,
            ttft_ms=1.0,
            peak_memory_mib=1.0,
            kv_cache_memory_mib=1.0,
            resident_memory_mib=1.0,
            kv_incremental_mib=1.0,
            kv_cumulative_mib=1.0,
            recompute_ms=1.0 if transitioned else 0.0,
            backend_name="online_qwen",
            extra={"cache_reused": reused, "runtime_transition_ms": 1.0 if transitioned else 0.0},
        )

    def close(self) -> None:
        self.closed = True


def test_run_scenarios_uses_expected_orders_and_closes_backends() -> None:
    sessions = [
        [replace(request, session_id=f"selected-{index}") for request in session]
        for index, session in enumerate(select_validation_sessions(
            {"data": {"split_seed": 20260906, "calibration_fraction": 0.5, "stratify_session": True}},
            _measurements(_requests()),
            _requests(),
        ))
    ]
    backends: list[FakeBackend] = []

    def factory() -> FakeBackend:
        backend = FakeBackend()
        backends.append(backend)
        return backend

    report = run_scenarios(sessions, factory)

    assert report["ok"] is True
    assert backends[0].calls == [("selected-0", "full_gpu", 0), ("selected-0", "full_gpu", 1)]
    assert backends[1].calls == [
        ("selected-1", "kivi_4bit_residual64", 0),
        ("selected-1", "kivi_4bit_residual64", 1),
        ("selected-1", "kivi_4bit_residual64", 2),
    ]
    assert backends[2].calls == [
        ("selected-2", "full_gpu", 0),
        ("selected-1", "kivi_4bit_residual64", 0),
        ("selected-2", "full_gpu", 1),
    ]
    assert all(backend.closed for backend in backends)


def test_run_scenarios_closes_backend_when_runtime_fails() -> None:
    sessions = [[request for request in _requests()[index * 3 : index * 3 + 3]] for index in range(3)]
    backend = FakeBackend(fail_profile="full_gpu")

    report = run_scenarios(sessions, lambda: backend)

    assert report["ok"] is False
    assert backend.closed is True
