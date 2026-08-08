from __future__ import annotations

from profiles.base import _qwen2_payload, _transformers_payload
from run_util.core_types import ProfileSpec, Request


def test_qwen2_payload_uses_effective_prompt_and_session_metadata() -> None:
    request = Request(
        request_id="s1_t1",
        task="chat",
        prompt="User: next question",
        session_id="s1",
        turn_index=1,
        history_turns=("User: hi", "Assistant: hello"),
    )
    spec = ProfileSpec("kivi_4bit_residual32", "kivi", "edgekv-kivi", lossy=True)

    payload = _qwen2_payload(request, spec, {"max_new_tokens": 8, "memory_budget_mib": 64.0}, "/tmp/model")

    assert payload["prompt"] == request.effective_prompt
    assert payload["session_id"] == "s1"
    assert payload["turn_index"] == 1
    assert payload["history_turns"] == ["User: hi", "Assistant: hello"]
    assert payload["memory_budget_mib"] == 64.0
    assert payload["execution_mode"] == "append"


def test_transformers_payload_uses_effective_prompt_and_session_metadata() -> None:
    request = Request(
        request_id="s2_t0",
        task="chat",
        prompt="User: hello",
        session_id="s2",
        turn_index=0,
        history_turns=(),
    )
    spec = ProfileSpec("full_gpu", "full", "tailguardkv-base", lossy=False, exact=True)

    payload = _transformers_payload(request, spec, {"max_new_tokens": 4}, "/tmp/model")

    assert payload["prompt"] == request.effective_prompt
    assert payload["session_id"] == "s2"
    assert payload["turn_index"] == 0
    assert payload["history_turns"] == []
    assert payload["execution_mode"] == "append"
