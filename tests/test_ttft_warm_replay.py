from __future__ import annotations

from run_util.core_types import Request
from scripts.run_ttft_warm_replay import select_gate_requests


def test_select_gate_requests_keeps_three_named_sessions_in_arrival_order() -> None:
    requests = [
        Request(f"other-{turn}", "qa", "x", session_id="other", turn_index=turn, arrival_index=turn)
        for turn in range(5)
    ]
    requests.extend(
        Request(
            f"hybrid-session-{session:03d}-turn-{turn}",
            "qa",
            "x",
            session_id=f"hybrid-session-{session:03d}",
            turn_index=turn,
            arrival_index=10 + (turn * 3) + (session - 2),
        )
        for session in (2, 3, 4)
        for turn in range(5)
    )

    selected = select_gate_requests(list(reversed(requests)))

    assert len(selected) == 15
    assert [request.arrival_index for request in selected] == sorted(request.arrival_index for request in selected)
    for session in (2, 3, 4):
        assert [request.turn_index for request in selected if request.session_id == f"hybrid-session-{session:03d}"] == [0, 1, 2, 3, 4]
