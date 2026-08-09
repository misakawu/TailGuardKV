from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from typing import Any

from profiles.cache_common import legacy_kv_cache_memory_mib
from profiles.generation_timing import generate_with_first_token_timing
from profiles.session_runtime import SessionRuntimeState, apply_budget_policy


def run_profile_batch(payload: dict[str, Any]) -> dict[str, Any]:
    requests = list(payload.get("requests") or [])
    if not requests:
        return {
            "ok": True,
            "results": [],
            "worker": {"mode": "batch"},
            "session_runtime_state": SessionRuntimeState().to_payload(),
        }

    worker_start = time.perf_counter()
    try:
        runtime = _load_runtime(requests[0], worker_start)
    except Exception as exc:
        startup_ms = (time.perf_counter() - worker_start) * 1000
        failure = _failure_result(
            error=f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}",
            failure_stage="model_load",
            stage_startup_ms=startup_ms,
            stage_model_load_ms=0.0,
            worker_mode="batch",
        )
        return {
            "ok": False,
            "results": [dict(failure) for _ in requests],
            "worker": {"mode": "batch"},
            "session_runtime_state": SessionRuntimeState.from_payload(payload.get("session_runtime_state")).to_payload(),
        }

    results = []
    state = SessionRuntimeState.from_payload(payload.get("session_runtime_state"))
    for index, request in enumerate(requests):
        result, state = _run_measured_request_with_session(runtime, request, state)
        results.append(result)
        if _is_cuda_oom(result.get("error")):
            for _ in requests[index + 1 :]:
                results.append(dict(result))
            break
    return {
        "ok": all(item.get("ok") for item in results),
        "results": results,
        "worker": {"mode": "batch"},
        "session_runtime_state": state.to_payload(),
    }


def _estimate_kv_mib(runtime: dict[str, Any], payload: dict[str, Any]) -> float:
    estimator = runtime.get("kv_estimator")
    if callable(estimator):
        return float(estimator(payload))
    prompt = str(payload.get("prompt") or "")
    return float(max(1, len(prompt)))


def _append_with_existing_kv(runtime: dict[str, Any], payload: dict[str, Any], state: SessionRuntimeState) -> tuple[dict[str, Any], SessionRuntimeState]:
    session_id = str(payload.get("session_id") or "")
    profile = str(payload.get("profile") or "")
    turn_index = int(payload.get("turn_index") or 0)
    previous = state.sessions.get(session_id)
    resident_before = previous.resident_gpu_mib if previous else 0.0
    current_total = _estimate_kv_mib(runtime, payload)
    incremental = max(1.0, current_total - resident_before) if turn_index > 0 else current_total
    next_state = state.record_resident(session_id, profile, turn_index=turn_index, kv_mib=incremental)
    resident_after = next_state.sessions[session_id].resident_gpu_mib
    result = {
        "ok": True,
        "measured": True,
        "kv_incremental_mib": incremental,
        "kv_cumulative_mib": resident_after,
        "resident_kv_mib_before": resident_before,
        "resident_kv_mib_after": resident_after,
        "restore_ms": 0.0,
        "recompute_ms": 0.0,
        "evicted_kv_mib": 0.0,
        "budget_hit": False,
        "event_trace": [{"event": "resident", "session_id": session_id, "profile": profile, "turn_index": turn_index}],
    }
    return result, next_state


def _rebuild_session_from_turn0(runtime: dict[str, Any], history_payloads: list[dict[str, Any]], *, target_profile: str) -> dict[str, Any]:
    state = SessionRuntimeState()
    last_result: dict[str, Any] = {}
    recompute_ms = 0.0
    for payload in history_payloads:
        rebuilt = dict(payload)
        rebuilt["profile"] = target_profile
        result, state = _append_with_existing_kv(runtime, rebuilt, state)
        recompute_ms += float(runtime.get("clock_ms", 1.0))
        last_result = result
    last_result["recompute_ms"] = recompute_ms
    last_result["event_trace"] = [*last_result.get("event_trace", []), {"event": "recompute", "profile": target_profile}]
    return last_result


def _run_one_request_with_session(runtime: dict[str, Any], payload: dict[str, Any], state: SessionRuntimeState) -> tuple[dict[str, Any], SessionRuntimeState]:
    session_id = str(payload.get("session_id") or "")
    profile = str(payload.get("profile") or "")
    session = state.sessions.get(session_id)
    if session and session.current_profile and session.current_profile != profile:
        history_payloads = []
        for turn in range(int(payload.get("turn_index") or 0) + 1):
            prompt = "\n".join(list(payload.get("history_turns") or [])[:turn] + [str(payload.get("prompt") or "")])
            history_payloads.append({**payload, "turn_index": turn, "prompt": prompt, "history_turns": list(payload.get("history_turns") or [])[:turn]})
        result = _rebuild_session_from_turn0(runtime, history_payloads, target_profile=profile)
        rebuilt_state = state.switch_profile(session_id, from_profile=session.current_profile, to_profile=profile, rebuild_required=True)
        rebuilt_state = rebuilt_state.record_resident(
            session_id,
            profile,
            turn_index=int(payload.get("turn_index") or 0),
            kv_mib=float(result["kv_cumulative_mib"]) - rebuilt_state.sessions[session_id].resident_gpu_mib,
        )
        return result, rebuilt_state
    return _append_with_existing_kv(runtime, payload, state)


def _run_measured_request_with_session(runtime: dict[str, Any], payload: dict[str, Any], state: SessionRuntimeState) -> tuple[dict[str, Any], SessionRuntimeState]:
    result = _run_one_request(runtime, payload)
    if not bool(result.get("ok")):
        return result, state

    session_id = str(payload.get("session_id") or "")
    if not session_id:
        return result, state

    profile = str(payload.get("profile") or "")
    turn_index = int(payload.get("turn_index") or 0)
    previous = state.sessions.get(session_id)
    next_state = state
    events: list[dict[str, object]] = []
    resident_before = previous.resident_gpu_mib if previous else 0.0
    restore_ms = 0.0
    recompute_ms = 0.0

    if previous and previous.current_profile and previous.current_profile != profile:
        next_state = next_state.reset_session(session_id, profile)
        resident_before = 0.0
        recompute_ms = max(1.0, float(result.get("stage_prefill_ms") or result.get("latency_ms") or 1.0))
        events.append({"event": "recompute", "session_id": session_id, "profile": profile, "turn_index": turn_index})
    elif previous and previous.offloaded_cpu_mib > 0:
        restore_amount = previous.offloaded_cpu_mib
        next_state = next_state.record_restore(session_id, profile, kv_mib=restore_amount)
        resident_before = next_state.sessions[session_id].resident_gpu_mib
        restore_ms = max(1.0, restore_amount)
        events.append({"event": "restore", "session_id": session_id, "profile": profile, "turn_index": turn_index, "kv_mib": restore_amount})
    elif previous and previous.dropped_mib > 0:
        next_state = next_state.reset_session(session_id, profile)
        resident_before = 0.0
        recompute_ms = max(1.0, float(result.get("stage_prefill_ms") or result.get("latency_ms") or 1.0))
        events.append({"event": "recompute", "session_id": session_id, "profile": profile, "turn_index": turn_index})

    measured_total = float(result.get("kv_cache_memory_mib") or result.get("peak_memory_mib") or 0.0)
    kv_incremental_mib = measured_total if turn_index <= 0 else max(0.0, measured_total - resident_before)
    next_state, budget_events, budget_hit, budget_restore_ms = apply_budget_policy(
        next_state,
        session_id=session_id,
        profile=profile,
        turn_index=turn_index,
        kv_incremental_mib=kv_incremental_mib,
        memory_budget_mib=float(payload.get("memory_budget_mib")) if payload.get("memory_budget_mib") is not None else None,
        prefer_restore=False,
    )
    session = next_state.sessions[session_id]
    result.update(
        {
            "kv_incremental_mib": kv_incremental_mib,
            "kv_cumulative_mib": session.resident_gpu_mib + session.offloaded_cpu_mib,
            "resident_kv_mib_before": resident_before,
            "resident_kv_mib_after": session.resident_gpu_mib,
            "restore_ms": restore_ms + budget_restore_ms,
            "recompute_ms": recompute_ms,
            "evicted_kv_mib": session.offloaded_cpu_mib,
            "budget_hit": budget_hit,
            "event_trace": [*events, *budget_events],
        }
    )
    return result, next_state


def main() -> int:
    payload_text = sys.stdin.read().strip() or os.environ.get("TRANSFORMERS_PROFILE_PAYLOAD", "")
    payload = json.loads(payload_text)
    result = run_profile_batch(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


def _load_runtime(request: dict[str, Any], worker_start: float) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    startup_ms = (time.perf_counter() - worker_start) * 1000
    model_name = str(request.get("model_name") or "")
    if not model_name:
        raise ValueError("missing model_name")
    cache_dir = request.get("cache_dir") or None
    local_files_only = bool(request.get("local_files_only", True))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    device_mode = str(request.get("device_mode") or "auto")
    has_cuda = bool(torch.cuda.is_available())
    kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    if device_mode == "cpu" or not has_cuda:
        kwargs["device_map"] = "cpu"
    else:
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = torch.float16
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": model_device,
        "has_cuda": has_cuda and device_mode != "cpu",
        "device_mode": device_mode,
        "startup_ms": startup_ms,
        "model_load_ms": (time.perf_counter() - load_start) * 1000,
    }


def _run_one_request(runtime: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]
    device = runtime["device"]
    has_cuda = bool(runtime["has_cuda"])
    stage_tokenize_ms = 0.0
    stage_transfer_ms = 0.0
    stage_generate_ms = 0.0
    stage_decode_ms = 0.0
    request_start = time.perf_counter()

    try:
        tokenize_start = time.perf_counter()
        inputs = tokenizer(str(request.get("prompt") or ""), return_tensors="pt")
        stage_tokenize_ms = (time.perf_counter() - tokenize_start) * 1000
    except Exception as exc:
        return _failure_result(
            error=f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="tokenize",
            stage_startup_ms=float(runtime["startup_ms"]),
            stage_model_load_ms=float(runtime["model_load_ms"]),
            stage_tokenize_ms=stage_tokenize_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode="batch",
        )

    try:
        transfer_start = time.perf_counter()
        if has_cuda:
            inputs = {key: value.to(device) for key, value in inputs.items()}
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        stage_transfer_ms = (time.perf_counter() - transfer_start) * 1000
    except Exception as exc:
        return _failure_result(
            error=f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="transfer",
            stage_startup_ms=float(runtime["startup_ms"]),
            stage_model_load_ms=float(runtime["model_load_ms"]),
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode="batch",
        )

    try:
        generate_start = time.perf_counter()
        with torch.inference_mode():
            generated = generate_with_first_token_timing(
                model,
                tokenizer,
                torch,
                inputs,
                request_start=request_start,
                device=device,
                max_new_tokens=int(request.get("max_new_tokens") or 16),
                has_cuda=has_cuda,
                use_cache=bool(request.get("use_cache", True)),
                pad_token_id=tokenizer.eos_token_id,
            )
        stage_generate_ms = float(generated["stage_generate_ms"])
    except Exception as exc:
        return _failure_result(
            error=f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="generate",
            stage_startup_ms=float(runtime["startup_ms"]),
            stage_model_load_ms=float(runtime["model_load_ms"]),
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_generate_ms=stage_generate_ms or (time.perf_counter() - generate_start) * 1000,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode="batch",
        )

    peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if has_cuda:
        peak_memory_mib = runtime["torch"].cuda.max_memory_allocated(device) / 1024 / 1024
    total_ms = (time.perf_counter() - request_start) * 1000
    return {
        "ok": True,
        "measured": True,
        "output_text": generated["output_text"],
        "latency_ms": total_ms,
        "ttft_ms": generated["ttft_ms"],
        "peak_memory_mib": peak_memory_mib,
        "kv_cache_memory_mib": legacy_kv_cache_memory_mib(generated.get("past_key_values")),
        "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stage_startup_ms": float(runtime["startup_ms"]),
        "stage_model_load_ms": float(runtime["model_load_ms"]),
        "stage_tokenize_ms": stage_tokenize_ms,
        "stage_transfer_ms": stage_transfer_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_decode_ms": stage_decode_ms,
        "stage_prefill_ms": generated["stage_prefill_ms"],
        "stage_first_token_ms": generated["stage_first_token_ms"],
        "stage_total_ms": total_ms,
        "ttft_semantics": "first_token",
        "worker_mode": "batch",
    }


def _failure_result(
    *,
    error: str,
    failure_stage: str,
    stage_startup_ms: float = 0.0,
    stage_model_load_ms: float = 0.0,
    stage_tokenize_ms: float = 0.0,
    stage_transfer_ms: float = 0.0,
    stage_generate_ms: float = 0.0,
    stage_decode_ms: float = 0.0,
    stage_total_ms: float = 0.0,
    worker_mode: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "measured": False,
        "error": error,
        "failure_stage": failure_stage,
        "stage_startup_ms": stage_startup_ms,
        "stage_model_load_ms": stage_model_load_ms,
        "stage_tokenize_ms": stage_tokenize_ms,
        "stage_transfer_ms": stage_transfer_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_decode_ms": stage_decode_ms,
        "stage_total_ms": stage_total_ms,
        "worker_mode": worker_mode,
    }


def _is_cuda_oom(error: object) -> bool:
    if error is None:
        return False
    text = str(error).lower()
    return "out of memory" in text


if __name__ == "__main__":
    raise SystemExit(main())
