from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

from run_util.canonical_history import (
    CANONICAL_HISTORY_MODE,
    CANONICAL_HISTORY_SOURCE_PROFILE,
    canonical_history_hash,
)
from profiles.h2o_cache import H2OCache, H2OLayerState, build_h2o_cache
from profiles.kivi_cache import KIVICache, KIVILayerState, build_kivi_cache
from profiles.qwen2_h2o_runtime import Qwen2H2OAttention, h2o_sizes as _h2o_sizes_impl, prepare_h2o_runtime as _prepare_h2o_runtime_impl, reset_h2o_attention as _reset_h2o_attention_impl, run_h2o_request as _run_h2o_request_impl
from profiles.qwen2_kivi_runtime import Qwen2KIVIAttention, kivi_proof_error as _kivi_proof_error_impl, prepare_kivi_runtime as _prepare_kivi_runtime_impl, run_kivi_request as _run_kivi_request_impl, split_prefill_kivi_states as _split_prefill_kivi_states_impl
from profiles.qwen2_runtime_common import binding_diagnostics as _binding_diagnostics_impl, clone_failure_result as _clone_failure_result_impl, failure as _failure_impl, generate_decode as _generate_decode_impl, import_runtime_modules as _import_runtime_modules_impl, invoke_generate_decode as _invoke_generate_decode_impl, is_fatal_cuda_error as _is_fatal_cuda_error_impl, is_oom_result as _is_oom_result_impl, load_qwen2_model as _load_qwen2_model_impl, release_runtime_cuda_resources as _release_runtime_cuda_resources_impl, require_cuda as _require_cuda_impl
from profiles.qwen2_runtime_layout import apply_rope as _apply_rope_impl, install_qwen2_attention as _install_qwen2_attention_impl, mask_softmax as _mask_softmax_impl, qwen2_layout_error as _kivi_compatibility_error_impl
from profiles.session_runtime import SessionRuntimeState, apply_budget_policy


_import_runtime_modules = _import_runtime_modules_impl
_require_cuda = _require_cuda_impl
_load_qwen2_model = _load_qwen2_model_impl
_install_qwen2_attention = _install_qwen2_attention_impl
_generate_decode = _generate_decode_impl
_clone_failure_result = _clone_failure_result_impl
_is_oom_result = _is_oom_result_impl
_is_fatal_cuda_error = _is_fatal_cuda_error_impl
_failure = _failure_impl
_apply_rope = _apply_rope_impl
_mask_softmax = _mask_softmax_impl
_split_prefill_kivi_states = _split_prefill_kivi_states_impl
_release_runtime_cuda_resources = _release_runtime_cuda_resources_impl
_binding_diagnostics = _binding_diagnostics_impl


def _invoke_generate_decode(model: Any, tokenizer: Any, device: Any, payload: dict[str, Any], torch: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _generate_decode(model, tokenizer, device, payload, torch, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        fallback_kwargs = {}
        if "past_key_values" in kwargs:
            fallback_kwargs["past_key_values"] = kwargs["past_key_values"]
        return _generate_decode(model, tokenizer, device, payload, torch, **fallback_kwargs)


def _greedy_decode(*args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("manual decode path has been removed")


def run_profile(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("dry_session"):
        return _run_dry_session_profile(payload)
    profile = str(payload.get("profile") or "")
    canonical_error = _canonical_history_payload_error(payload)
    if canonical_error:
        return _failure(payload, canonical_error, failure_stage="canonical_history")
    try:
        if profile == "full_gpu":
            return _sanitize_public_result(_run_full_profile(payload))
        if profile.startswith("kivi_"):
            return _sanitize_public_result(_run_kivi_profile(payload))
        if profile.startswith("h2o_heavy"):
            return _sanitize_public_result(_run_h2o_profile(payload))
        return _failure(payload, f"unsupported Qwen2 KV profile: {profile}")
    except Exception as exc:
        return _failure(payload, f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}")


def _run_dry_session_profile(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("session_runtime_state")
    if not isinstance(state, SessionRuntimeState):
        state = SessionRuntimeState()
    profile = str(payload.get("profile") or "")
    session_id = str(payload.get("session_id") or "")
    turn_index = int(payload.get("turn_index") or 0)
    kv_incremental_mib = float(max(1, len(str(payload.get("prompt") or ""))))
    if state.sessions.get(session_id, SessionRuntimeState().sessions.get(session_id, None)) and state.sessions.get(session_id) and state.sessions[session_id].dropped_mib > 0:
        recompute_ms = max(1.0, kv_incremental_mib)
        event_trace = [{"event": "recompute", "session_id": session_id, "profile": profile, "turn_index": turn_index}]
        next_state = state.record_resident(session_id, profile, turn_index=turn_index, kv_mib=kv_incremental_mib)
        return {
            "ok": True,
            "measured": True,
            "output_text": str(payload.get("prompt") or ""),
            "latency_ms": kv_incremental_mib,
            "ttft_ms": 1.0,
            "peak_memory_mib": kv_incremental_mib,
            "kv_cache_memory_mib": kv_incremental_mib,
            "resident_memory_mib": next_state.sessions[session_id].resident_gpu_mib,
            "kv_incremental_mib": kv_incremental_mib,
            "kv_cumulative_mib": next_state.sessions[session_id].resident_gpu_mib,
            "resident_kv_mib_before": state.sessions[session_id].resident_gpu_mib if session_id in state.sessions else 0.0,
            "resident_kv_mib_after": next_state.sessions[session_id].resident_gpu_mib,
            "restore_ms": 0.0,
            "recompute_ms": recompute_ms,
            "evicted_kv_mib": 0.0,
            "budget_hit": False,
            "event_trace": event_trace,
            "backend": "qwen2_kv_runtime",
        }
    next_state, events, budget_hit, restore_ms = apply_budget_policy(
        state,
        session_id=session_id,
        profile=profile,
        turn_index=turn_index,
        kv_incremental_mib=kv_incremental_mib,
        memory_budget_mib=float(payload.get("memory_budget_mib")) if payload.get("memory_budget_mib") is not None else None,
        prefer_restore=bool(payload.get("restore_from_cpu")),
    )
    session = next_state.sessions[session_id]
    return {
        "ok": True,
        "measured": True,
        "output_text": str(payload.get("prompt") or ""),
        "latency_ms": kv_incremental_mib,
        "ttft_ms": 1.0,
        "peak_memory_mib": session.resident_gpu_mib,
        "kv_cache_memory_mib": session.resident_gpu_mib,
        "resident_memory_mib": session.resident_gpu_mib,
        "kv_incremental_mib": kv_incremental_mib,
        "kv_cumulative_mib": session.resident_gpu_mib + session.offloaded_cpu_mib,
        "resident_kv_mib_before": state.sessions[session_id].resident_gpu_mib if session_id in state.sessions else 0.0,
        "resident_kv_mib_after": session.resident_gpu_mib,
        "restore_ms": restore_ms,
        "recompute_ms": 0.0,
        "evicted_kv_mib": max(0.0, session.offloaded_cpu_mib),
        "budget_hit": budget_hit,
        "event_trace": events,
        "backend": "qwen2_kv_runtime",
    }


def run_profile_batch(payload: dict[str, Any]) -> dict[str, Any]:
    requests = list(payload.get("requests") or [])
    if not requests:
        return {"ok": True, "results": [], "worker": {"mode": "batch"}, "session_runtime_state": SessionRuntimeState().to_payload()}

    profiles = {str(request.get("profile") or "") for request in requests}
    if len(requests) == 1 or len(profiles) != 1:
        results = [run_profile(request) for request in requests]
        return {
            "ok": all(item.get("ok") for item in results),
            "results": results,
            "worker": {"mode": "batch"},
            "session_runtime_state": _session_state_from_payload(payload.get("session_runtime_state")).to_payload(),
        }

    worker_start = time.perf_counter()
    profile = next(iter(profiles))
    state = _session_state_from_payload(payload.get("session_runtime_state"))
    try:
        if profile == "full_gpu":
            results, state = _run_full_profile_batch(requests, worker_start, state)
        elif profile.startswith("kivi_"):
            results, state = _run_kivi_profile_batch(requests, worker_start, state)
        elif profile.startswith("h2o_heavy"):
            results, state = _run_h2o_profile_batch(requests, worker_start, state)
        else:
            results = [_failure(request, f"unsupported Qwen2 KV profile: {profile}") for request in requests]
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}"
        results = [_failure(request, error) for request in requests]
    return {
        "ok": all(item.get("ok") for item in results),
        "results": [_sanitize_public_result(item) for item in results],
        "worker": {"mode": "batch"},
        "session_runtime_state": state.to_payload(),
    }


def main() -> int:
    payload_text = sys.stdin.read().strip() or os.environ.get("QWEN2_KV_PAYLOAD", "")
    payload = json.loads(payload_text)
    if isinstance(payload, dict) and "requests" in payload:
        result = run_profile_batch(payload)
    else:
        result = run_profile(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


def worker_init(payload: dict[str, Any], worker_state: dict[str, Any]) -> dict[str, Any]:
    worker_state.clear()
    worker_state["adapter"] = str(payload.get("adapter") or "")
    worker_state["runtime_config"] = dict(payload.get("runtime_config") or {})
    worker_state["binding_diagnostics"] = _worker_binding_diagnostics(worker_state["runtime_config"])
    return {
        "ok": True,
        "worker": _worker_metadata(worker_state),
    }


def worker_run_batch(payload: dict[str, Any], worker_state: dict[str, Any]) -> dict[str, Any]:
    requests = list(payload.get("requests") or [])
    evicted_sessions = _evict_worker_sessions(worker_state, payload.get("evict_sessions"))
    if not requests:
        return {
            "ok": True,
            "results": [],
            "worker": _worker_metadata(worker_state),
            "session_runtime_state": _session_state_from_payload(payload.get("session_runtime_state")).to_payload(),
            "evicted_sessions": evicted_sessions,
        }
    state = _session_state_from_payload(payload.get("session_runtime_state"))
    profile = str(payload.get("profile") or requests[0].get("profile") or "")
    try:
        runtime = _ensure_worker_runtime(worker_state, profile, requests[0])
        if profile == "full_gpu":
            results, state = _run_full_profile_batch_with_runtime(runtime, requests, state)
        elif profile.startswith("kivi_"):
            results, state = _run_kivi_profile_batch_with_runtime(runtime, requests, state)
        elif profile.startswith("h2o_heavy"):
            results, state = _run_h2o_profile_batch_with_runtime(runtime, requests, state)
        else:
            results = [_failure(request, f"unsupported Qwen2 KV profile: {profile}") for request in requests]
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}"
        _clear_worker_runtime(worker_state)
        results = [_failure(request, detail) for request in requests]
        return {
            "ok": False,
            "results": _annotate_results_with_binding(results, worker_state=worker_state),
            "worker": _worker_metadata(worker_state),
            "session_runtime_state": state.to_payload(),
            "fatal_error": detail,
        }

    fatal_error = _persistent_worker_fatal_error(results)
    if fatal_error:
        _clear_worker_runtime(worker_state)
    return {
        "ok": all(item.get("ok") for item in results),
        "results": _annotate_results_with_binding(results, runtime=runtime, worker_state=worker_state),
        "worker": _worker_metadata(worker_state),
        "session_runtime_state": state.to_payload(),
        **({"fatal_error": fatal_error} if fatal_error else {}),
    }


def worker_shutdown(payload: dict[str, Any], worker_state: dict[str, Any]) -> dict[str, Any]:
    del payload
    _clear_worker_runtime(worker_state)
    return {"ok": True, "worker": _worker_metadata(worker_state)}


def _kivi_proof_error(
    *,
    prompt_tokens: int,
    max_new_tokens: int,
    residual_length: int,
    quantized_layers: int,
    kernel_calls: int,
) -> str:
    return _kivi_proof_error_impl(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        residual_length=residual_length,
        quantized_layers=quantized_layers,
        kernel_calls=kernel_calls,
    )


def _run_kivi_profile(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = _prepare_kivi_runtime(payload, worker_start=time.perf_counter())
    return _sanitize_public_result(_run_kivi_request(runtime, payload, worker_mode="single"))


def _run_full_profile(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = _prepare_full_runtime(payload, worker_start=time.perf_counter())
    try:
        return _sanitize_public_result(_run_full_request(runtime, payload, worker_mode="single"))
    finally:
        _release_runtime_resources(runtime)


def _run_h2o_profile(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = _prepare_h2o_runtime(payload, worker_start=time.perf_counter())
    return _sanitize_public_result(_run_h2o_request(runtime, payload, worker_mode="single"))


def _run_kivi_profile_batch(
    requests: list[dict[str, Any]],
    worker_start: float,
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    runtime = _prepare_kivi_runtime(requests[0], worker_start=worker_start)
    runtime["session_reuse"] = {}
    results = []
    try:
        for index, request in enumerate(requests):
            session_request = _attach_kivi_session_cache(runtime, request)
            result, state = _run_request_with_session(state, session_request, lambda item: _run_kivi_request(runtime, item, worker_mode="batch"))
            _update_kivi_session_cache(runtime, session_request, result)
            results.append(result)
            if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
                results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
                break
        return results, state
    finally:
        _release_runtime_resources(runtime)


def _run_full_profile_batch(
    requests: list[dict[str, Any]],
    worker_start: float,
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    runtime = _prepare_full_runtime(requests[0], worker_start=worker_start)
    results = []
    try:
        for index, request in enumerate(requests):
            result, state = _run_request_with_session(state, request, lambda item: _run_full_request(runtime, item, worker_mode="batch"))
            results.append(result)
            if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
                results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
                break
        return results, state
    finally:
        _release_runtime_resources(runtime)


def _run_h2o_profile_batch(
    requests: list[dict[str, Any]],
    worker_start: float,
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    runtime = _prepare_h2o_runtime(requests[0], worker_start=worker_start)
    results = []
    for index, request in enumerate(requests):
        result, state = _run_request_with_session(state, request, lambda item: _run_h2o_request(runtime, item, worker_mode="batch"))
        results.append(result)
        if _is_oom_result(result):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results, state


def _prepare_kivi_runtime(payload: dict[str, Any], *, worker_start: float) -> dict[str, Any]:
    return _prepare_kivi_runtime_impl(
        payload,
        worker_start=worker_start,
        import_runtime_modules=_import_runtime_modules,
        require_cuda=_require_cuda,
        load_qwen2_model=_load_qwen2_model,
        install_qwen2_attention=_install_qwen2_attention,
        attention_cls=Qwen2KIVIAttention,
    )


def _prepare_full_runtime(payload: dict[str, Any], *, worker_start: float) -> dict[str, Any]:
    modules = _import_runtime_modules(use_kivi=False)
    torch = modules["torch"]
    _require_cuda(torch)
    startup_ms = (time.perf_counter() - worker_start) * 1000
    load_start = time.perf_counter()
    model, tokenizer, device = _load_qwen2_model(payload, torch, modules["AutoModelForCausalLM"], modules["AutoTokenizer"])
    return {
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "startup_ms": startup_ms,
        "model_load_ms": (time.perf_counter() - load_start) * 1000,
        "binding_diagnostics": _binding_diagnostics(payload, torch),
    }


def _prepare_h2o_runtime(payload: dict[str, Any], *, worker_start: float) -> dict[str, Any]:
    return _prepare_h2o_runtime_impl(
        payload,
        worker_start=worker_start,
        import_runtime_modules=_import_runtime_modules,
        require_cuda=_require_cuda,
        load_qwen2_model=_load_qwen2_model,
        install_qwen2_attention=_install_qwen2_attention,
        attention_cls=Qwen2H2OAttention,
    )


def _run_full_request(runtime: dict[str, Any], payload: dict[str, Any], *, worker_mode: str) -> dict[str, Any]:
    cached_ids = payload.get("_runtime_cached_prompt_token_ids")
    tokenized = runtime["tokenizer"](str(payload.get("prompt") or ""), return_tensors="pt")
    if isinstance(cached_ids, list):
        tokenized = _trim_tokenized_inputs(runtime["torch"], tokenized, prefix_len=len(cached_ids))
    result = _invoke_generate_decode(
        runtime["model"],
        runtime["tokenizer"],
        runtime["device"],
        payload,
        runtime["torch"],
        past_key_values=payload.get("_runtime_reusable_cache"),
        tokenized_inputs=tokenized,
        stage_startup_ms=float(runtime.get("startup_ms") or 0.0),
        stage_model_load_ms=float(runtime.get("model_load_ms") or 0.0),
        worker_mode=worker_mode,
    )
    result.update({"runtime_cache": result.get("past_key_values"), "runtime_prompt_token_ids": _request_prompt_token_ids(runtime, payload)})
    return result


def _run_kivi_request(runtime: dict[str, Any], payload: dict[str, Any], *, worker_mode: str) -> dict[str, Any]:
    return _run_kivi_request_impl(
        runtime,
        payload,
        worker_mode=worker_mode,
        build_kivi_cache=build_kivi_cache,
        invoke_generate_decode=_invoke_generate_decode,
    )


def _run_h2o_request(runtime: dict[str, Any], payload: dict[str, Any], *, worker_mode: str) -> dict[str, Any]:
    return _run_h2o_request_impl(
        runtime,
        payload,
        worker_mode=worker_mode,
        build_h2o_cache=build_h2o_cache,
        invoke_generate_decode=_invoke_generate_decode,
    )


def _h2o_sizes(tokenizer: Any, payload: dict[str, Any]) -> dict[str, int]:
    return _h2o_sizes_impl(tokenizer, payload)


def _reset_h2o_attention(model: Any, tracker: dict[str, int], heavy_size: int, recent_size: int) -> None:
    _reset_h2o_attention_impl(model, tracker, heavy_size, recent_size)


def _kivi_compatibility_error(model: Any, payload: dict[str, Any]) -> str | None:
    del payload
    return _kivi_compatibility_error_impl(model)


def _session_state_from_payload(payload: object) -> SessionRuntimeState:
    return SessionRuntimeState.from_payload(payload)


def _sanitize_public_result(result: dict[str, Any]) -> dict[str, Any]:
    public = dict(result)
    public.pop("runtime_cache", None)
    public.pop("runtime_prompt_token_ids", None)
    public.pop("past_key_values", None)
    return public


def _release_runtime_resources(runtime: dict[str, Any]) -> None:
    session_reuse = runtime.pop("session_reuse", None)
    if isinstance(session_reuse, dict):
        for cached in session_reuse.values():
            cache = cached.get("cache") if isinstance(cached, dict) else None
            if isinstance(cache, KIVICache):
                cache.clear()
        session_reuse.clear()
    torch = runtime.get("torch")
    # Drop strong references before empty_cache so CUDA memory can actually be reclaimed.
    for key in ("model", "tokenizer", "device", "modules", "tracker"):
        runtime.pop(key, None)
    if torch is not None:
        _release_runtime_cuda_resources(torch, *(session_reuse.values() if isinstance(session_reuse, dict) else ()))
    runtime.clear()


def _ensure_worker_runtime(worker_state: dict[str, Any], profile: str, request: dict[str, Any]) -> dict[str, Any]:
    runtime = worker_state.get("runtime")
    if isinstance(runtime, dict) and str(worker_state.get("runtime_profile") or "") == profile:
        return runtime
    _clear_worker_runtime(worker_state)
    worker_start = time.perf_counter()
    if profile == "full_gpu":
        runtime = _prepare_full_runtime(request, worker_start=worker_start)
    elif profile.startswith("kivi_"):
        runtime = _prepare_kivi_runtime(request, worker_start=worker_start)
        runtime["session_reuse"] = {}
    elif profile.startswith("h2o_heavy"):
        runtime = _prepare_h2o_runtime(request, worker_start=worker_start)
    else:
        raise ValueError(f"unsupported Qwen2 KV profile: {profile}")
    worker_state["runtime"] = runtime
    runtime.setdefault("session_reuse", {})
    worker_state["runtime_profile"] = profile
    return runtime


def _worker_binding_diagnostics(runtime_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "worker_cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "worker_device_strategy": str(runtime_config.get("device_strategy") or ""),
    }


def _worker_metadata(worker_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "persistent",
        "adapter": str(worker_state.get("adapter") or ""),
        **dict(worker_state.get("binding_diagnostics") or {}),
    }


def _annotate_results_with_binding(
    results: list[dict[str, Any]],
    *,
    runtime: dict[str, Any] | None = None,
    worker_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    runtime_binding = dict(runtime.get("binding_diagnostics") or {}) if isinstance(runtime, dict) else {}
    worker_binding = dict(worker_state.get("binding_diagnostics") or {}) if isinstance(worker_state, dict) else {}
    annotated: list[dict[str, Any]] = []
    for item in results:
        public = _sanitize_public_result(item)
        for key, value in runtime_binding.items():
            public.setdefault(key, value)
        for key, value in worker_binding.items():
            public.setdefault(key, value)
        annotated.append(public)
    return annotated


def _clear_worker_runtime(worker_state: dict[str, Any]) -> None:
    runtime = worker_state.pop("runtime", None)
    worker_state.pop("runtime_profile", None)
    if isinstance(runtime, dict):
        _release_runtime_resources(runtime)


def _evict_worker_sessions(worker_state: dict[str, Any], raw_sessions: object) -> list[str]:
    if not isinstance(raw_sessions, list):
        return []
    runtime = worker_state.get("runtime")
    session_reuse = runtime.get("session_reuse") if isinstance(runtime, dict) else None
    if not isinstance(session_reuse, dict):
        return []
    evicted: list[str] = []
    for raw_session in raw_sessions:
        session_id = str(raw_session)
        entry = session_reuse.pop(session_id, None)
        if not isinstance(entry, dict):
            continue
        _clear_runtime_cache_entry(entry)
        entry.clear()
        evicted.append(session_id)
    if evicted:
        torch = runtime.get("torch")
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
    return evicted


def _persistent_worker_fatal_error(results: list[dict[str, Any]]) -> str:
    for item in results:
        error = str(item.get("error") or "")
        if _is_oom_result(item) or _is_fatal_cuda_error(error):
            return error or "fatal runtime error"
    return ""


def _run_kivi_profile_batch_with_runtime(
    runtime: dict[str, Any],
    requests: list[dict[str, Any]],
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    results = []
    for index, request in enumerate(requests):
        session_request = _attach_kivi_session_cache(runtime, request)
        result, state = _run_request_with_session(state, session_request, lambda item: _run_kivi_request(runtime, item, worker_mode="persistent"))
        _update_kivi_session_cache(runtime, session_request, result)
        results.append(result)
        if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results, state


def _run_full_profile_batch_with_runtime(
    runtime: dict[str, Any],
    requests: list[dict[str, Any]],
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    results = []
    for index, request in enumerate(requests):
        session_request = _attach_session_cache(runtime, request)
        result, state = _run_request_with_session(state, session_request, lambda item: _run_full_request(runtime, item, worker_mode="persistent"))
        _update_session_cache(runtime, session_request, result)
        results.append(result)
        if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results, state


def _run_h2o_profile_batch_with_runtime(
    runtime: dict[str, Any],
    requests: list[dict[str, Any]],
    state: SessionRuntimeState,
) -> tuple[list[dict[str, Any]], SessionRuntimeState]:
    results = []
    for index, request in enumerate(requests):
        session_request = _attach_session_cache(runtime, request)
        result, state = _run_request_with_session(state, session_request, lambda item: _run_h2o_request(runtime, item, worker_mode="persistent"))
        _update_session_cache(runtime, session_request, result)
        results.append(result)
        if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results, state


def _attach_kivi_session_cache(runtime: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    request = dict(payload)
    session_id = str(payload.get("session_id") or "")
    profile = str(payload.get("profile") or "")
    if not session_id:
        request["_runtime_cache_rebuild_reason"] = "no_session"
        return request
    session_reuse = runtime.setdefault("session_reuse", {})
    entry = session_reuse.get(session_id)
    if not isinstance(entry, dict):
        request["_runtime_cache_rebuild_reason"] = "new_session"
        return request
    canonical_error = _canonical_history_payload_error(payload)
    if canonical_error:
        raise ValueError(canonical_error)
    is_canonical = str(payload.get("canonical_history_mode") or "") == CANONICAL_HISTORY_MODE
    if payload.get("history_turns") and not is_canonical:
        _clear_runtime_cache_entry(entry)
        session_reuse.pop(session_id, None)
        request["_runtime_cache_rebuild_reason"] = "history_turns_present"
        return request
    if str(entry.get("profile") or "") != profile:
        _clear_runtime_cache_entry(entry)
        request["_runtime_cache_rebuild_reason"] = "profile_changed"
        return request
    cached_prompt_token_ids = entry.get("prompt_token_ids")
    prompt_token_ids = _request_prompt_token_ids(runtime, payload)
    if not isinstance(cached_prompt_token_ids, list) or prompt_token_ids[: len(cached_prompt_token_ids)] != cached_prompt_token_ids:
        _clear_runtime_cache_entry(entry)
        request["_runtime_cache_rebuild_reason"] = "prompt_mismatch"
        return request
    request["_runtime_reusable_kivi_cache"] = entry.get("cache")
    request["_runtime_cached_prompt_token_ids"] = list(cached_prompt_token_ids)
    request["_runtime_cache_rebuild_reason"] = ""
    return request


def _attach_session_cache(runtime: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Attach a profile-local cache only for the exact canonical continuation."""
    request = dict(payload)
    session_id, profile = str(payload.get("session_id") or ""), str(payload.get("profile") or "")
    if not session_id:
        request["_runtime_cache_rebuild_reason"] = "no_session"
        return request
    canonical_error = _canonical_history_payload_error(payload)
    if canonical_error:
        raise ValueError(canonical_error)
    is_canonical = str(payload.get("canonical_history_mode") or "") == CANONICAL_HISTORY_MODE
    entry = runtime.setdefault("session_reuse", {}).get(session_id)
    if not isinstance(entry, dict):
        request["_runtime_cache_rebuild_reason"] = "new_session"
        return request
    if not is_canonical:
        _clear_runtime_cache_entry(entry)
        runtime["session_reuse"].pop(session_id, None)
        request["_runtime_cache_rebuild_reason"] = "noncanonical_history"
        return request
    if str(entry.get("profile") or "") != profile:
        raise ValueError("canonical_history_mismatch: cached profile differs from replay profile")
    if int(entry.get("last_turn", -1)) != int(payload.get("turn_index") or 0) - 1:
        raise ValueError("canonical_history_mismatch: cached turn is not the prior canonical turn")
    if entry.get("canonical_history_hash") != payload.get("canonical_history_hash"):
        raise ValueError("canonical_history_mismatch: cached canonical history hash differs")
    cached_ids, prompt_ids = entry.get("prompt_token_ids"), _request_prompt_token_ids(runtime, payload)
    if not isinstance(cached_ids, list) or prompt_ids[:len(cached_ids)] != cached_ids:
        raise ValueError("canonical_history_mismatch: canonical prompt is not a strict cached prefix")
    request.update({"_runtime_reusable_cache": entry.get("cache"), "_runtime_cached_prompt_token_ids": list(cached_ids), "_runtime_cache_rebuild_reason": ""})
    return request


def _update_session_cache(runtime: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> None:
    if not bool(result.get("ok")) or not str(payload.get("session_id") or ""):
        return
    runtime.setdefault("session_reuse", {})[str(payload["session_id"])] = {
        "profile": str(payload.get("profile") or ""), "cache": result.get("runtime_cache") or result.get("past_key_values"),
        "prompt_token_ids": list(result.get("runtime_prompt_token_ids") or _request_prompt_token_ids(runtime, payload)),
        "canonical_history_hash": payload.get("canonical_history_hash"), "last_turn": int(payload.get("turn_index") or 0),
    }


def _canonical_history_payload_error(payload: dict[str, Any]) -> str:
    mode = str(payload.get("canonical_history_mode") or "")
    if not mode:
        return ""
    if mode != CANONICAL_HISTORY_MODE:
        return "canonical_history_mismatch: unsupported canonical history mode"
    if str(payload.get("canonical_history_source_profile") or "") != CANONICAL_HISTORY_SOURCE_PROFILE:
        return "canonical_history_mismatch: canonical source profile must be full_gpu"
    canonical_history = payload.get("canonical_history")
    history_turns = payload.get("history_turns")
    if not isinstance(canonical_history, list) or canonical_history != history_turns:
        return "canonical_history_mismatch: rendered history differs from canonical fixture"
    if not all(isinstance(item, str) for item in canonical_history):
        return "canonical_history_mismatch: canonical history must contain strings"
    if str(payload.get("canonical_history_hash") or "") != canonical_history_hash(canonical_history):
        return "canonical_history_mismatch: canonical history hash mismatch"
    return ""


def _update_kivi_session_cache(runtime: dict[str, Any], payload: dict[str, Any], result: dict[str, Any]) -> None:
    session_id = str(payload.get("session_id") or "")
    if not session_id:
        return
    session_reuse = runtime.setdefault("session_reuse", {})
    prior = session_reuse.get(session_id)
    if not bool(result.get("ok")):
        if isinstance(prior, dict):
            _clear_runtime_cache_entry(prior)
            session_reuse.pop(session_id, None)
        return
    session_reuse[session_id] = {
        "profile": str(payload.get("profile") or ""),
        "cache": result.get("runtime_cache"),
        "prompt_token_ids": list(result.get("runtime_prompt_token_ids") or []),
    }


def _clear_runtime_cache_entry(entry: dict[str, Any]) -> None:
    cache = entry.get("cache")
    if isinstance(cache, KIVICache):
        cache.clear()


def _request_prompt_token_ids(runtime: dict[str, Any], payload: dict[str, Any]) -> list[int]:
    tokenizer = runtime.get("tokenizer")
    if tokenizer is None:
        return []
    tokenized = tokenizer(str(payload.get("prompt") or ""), return_tensors="pt")
    input_ids = tokenized.get("input_ids")
    if input_ids is None:
        return []
    values = input_ids
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    while isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    return [int(token) for token in values]


def _trim_tokenized_inputs(torch: Any, tokenized: dict[str, Any], *, prefix_len: int) -> dict[str, Any]:
    del torch
    trimmed = dict(tokenized)
    for key, value in tuple(trimmed.items()):
        if not hasattr(value, "__getitem__"):
            continue
        try:
            suffix = value[:, prefix_len:]
            trimmed[key] = suffix if int(suffix.shape[-1]) > 0 else value[:, -1:]
        except (IndexError, TypeError, AttributeError):
            continue
    return trimmed


def _run_request_with_session(
    state: SessionRuntimeState,
    payload: dict[str, Any],
    execute: Any,
) -> tuple[dict[str, Any], SessionRuntimeState]:
    result = execute(payload)
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


if __name__ == "__main__":
    raise SystemExit(main())
