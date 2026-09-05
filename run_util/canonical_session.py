"""Sequential full-GPU bootstrap for canonical multi-turn histories."""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

from run_util.canonical_history import (
    CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS,
    CanonicalBootstrapManifest,
    CanonicalHistoryError,
    build_canonical_fixture,
)
from run_util.core_types import Request


def bootstrap_canonical_session(
    requests: Iterable[Request],
    run_full_gpu: Callable[[Request], str],
    *,
    bootstrap_table_hash: str,
    output_token_count: Callable[[str], int],
) -> tuple[list[dict[str, object]], dict[str, str], CanonicalBootstrapManifest]:
    """Run arrival-ordered turns and return replay rows using generated history.

    ``reference`` remains in the fixture for quality scoring but never advances
    session history.
    """
    ordered = sorted(
        list(requests),
        key=lambda request: (
            request.arrival_index,
            request.session_id or request.request_id,
            request.turn_index,
            request.request_id,
        ),
    )
    histories: dict[str, list[str]] = {}
    expected_turns: dict[str, int] = {}
    outputs: dict[str, str] = {}
    for request in ordered:
        if not request.session_id:
            raise CanonicalHistoryError("canonical_history_mismatch: bootstrap request missing session_id")
        expected_turn = expected_turns.get(request.session_id, 0)
        if request.turn_index != expected_turn:
            raise CanonicalHistoryError(
                "canonical_history_mismatch: bootstrap turn sequence "
                f"session={request.session_id} expected={expected_turn} actual={request.turn_index}"
            )
        history = tuple(histories.setdefault(request.session_id, []))
        generated = str(run_full_gpu(replace(request, history_turns=history)))
        if int(output_token_count(generated)) > CANONICAL_HISTORY_GENERATION_MAX_NEW_TOKENS:
            raise CanonicalHistoryError(
                "canonical_history_mismatch: full_gpu bootstrap output exceeds 16 tokens "
                f"request_id={request.request_id}"
            )
        outputs[request.request_id] = generated
        histories[request.session_id].extend((f"User: {request.prompt}", f"Assistant: {generated}"))
        expected_turns[request.session_id] = expected_turn + 1

    rows = [
        {
            "request_id": request.request_id,
            "task": request.task,
            "prompt": request.prompt,
            "reference": request.reference,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "arrival_index": request.arrival_index,
            "metadata": dict(request.metadata),
        }
        for request in ordered
    ]
    fixture = build_canonical_fixture(
        rows,
        outputs,
        bootstrap_table_hash=bootstrap_table_hash,
        output_token_count=output_token_count,
    )
    return fixture, outputs, CanonicalBootstrapManifest.from_fixture(
        fixture, bootstrap_table_hash=bootstrap_table_hash
    )
