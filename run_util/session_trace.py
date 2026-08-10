from __future__ import annotations

import json
from dataclasses import replace
from math import inf

from run_util.core_types import Request


def _ordered_seed(requests: list[Request]) -> list[Request]:
    return sorted(
        requests,
        key=lambda request: (
            request.arrival_index,
            request.session_id or request.request_id,
            request.turn_index,
            request.request_id,
        ),
    )


def synthesize_pressure_trace(
    requests: list[Request],
    *,
    copies: int = 2,
    repeat_rounds: int = 1,
    memory_budget_mib: float = inf,
) -> list[Request]:
    """Expand session requests into an interleaved pressure request trace."""
    if copies < 1 or repeat_rounds < 1:
        raise ValueError("pressure trace copies/repeat_rounds 必须为正整数")
    if memory_budget_mib <= 0:
        raise ValueError("pressure trace memory_budget_mib 必须为正数")
    if not requests:
        raise ValueError("pressure trace 输入不能为空")

    seed = _ordered_seed(list(requests))
    session_ids = {request.session_id for request in seed if request.session_id}
    if len(session_ids) < 2:
        raise ValueError("pressure trace 要求至少两个 session")
    if not any(request.turn_index > 0 for request in seed):
        raise ValueError("pressure trace 要求至少一个多 turn session")

    trace: list[Request] = []
    arrival_index = 0
    for round_index in range(repeat_rounds):
        for request in seed:
            for copy_index in range(copies):
                source_session = request.session_id or request.request_id
                session_suffix = f"__pressure_r{round_index + 1}_c{copy_index + 1}"
                trace.append(
                    replace(
                        request,
                        request_id=f"{request.request_id}{session_suffix}",
                        session_id=f"{source_session}{session_suffix}",
                        arrival_index=arrival_index,
                        metadata={
                            **request.metadata,
                            "original_request_id": request.request_id,
                            "original_session_id": source_session,
                            "pressure_round": round_index + 1,
                            "pressure_copy": copy_index + 1,
                            "pressure_budget_mib": memory_budget_mib,
                            "history_turns_json": json.dumps(list(request.history_turns), ensure_ascii=False),
                        },
                    )
                )
                arrival_index += 1
    return trace
