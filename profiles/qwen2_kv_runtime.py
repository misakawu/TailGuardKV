from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

from profiles.kivi_cache import KIVICache, KIVILayerState, build_kivi_cache
from profiles.qwen2_h2o_runtime import Qwen2H2OAttention, h2o_sizes as _h2o_sizes_impl, prepare_h2o_runtime as _prepare_h2o_runtime_impl, reset_h2o_attention as _reset_h2o_attention_impl, run_h2o_request as _run_h2o_request_impl
from profiles.qwen2_kivi_runtime import Qwen2KIVIAttention, kivi_proof_error as _kivi_proof_error_impl, prepare_kivi_runtime as _prepare_kivi_runtime_impl, run_kivi_request as _run_kivi_request_impl, split_prefill_kivi_states as _split_prefill_kivi_states_impl
from profiles.qwen2_runtime_common import clone_failure_result as _clone_failure_result_impl, failure as _failure_impl, generate_decode as _generate_decode_impl, import_runtime_modules as _import_runtime_modules_impl, invoke_generate_decode as _invoke_generate_decode_impl, is_fatal_cuda_error as _is_fatal_cuda_error_impl, is_oom_result as _is_oom_result_impl, load_qwen2_model as _load_qwen2_model_impl, require_cuda as _require_cuda_impl
from profiles.qwen2_runtime_layout import apply_rope as _apply_rope_impl, install_qwen2_attention as _install_qwen2_attention_impl, mask_softmax as _mask_softmax_impl, qwen2_layout_error as _kivi_compatibility_error_impl


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
    profile = str(payload.get("profile") or "")
    try:
        if profile.startswith("kivi_"):
            return _run_kivi_profile(payload)
        if profile.startswith("h2o_heavy"):
            return _run_h2o_profile(payload)
        return _failure(payload, f"unsupported Qwen2 KV profile: {profile}")
    except Exception as exc:
        return _failure(payload, f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}")


def run_profile_batch(payload: dict[str, Any]) -> dict[str, Any]:
    requests = list(payload.get("requests") or [])
    if not requests:
        return {"ok": True, "results": [], "worker": {"mode": "batch"}}

    profiles = {str(request.get("profile") or "") for request in requests}
    if len(requests) == 1 or len(profiles) != 1:
        results = [run_profile(request) for request in requests]
        return {"ok": all(item.get("ok") for item in results), "results": results, "worker": {"mode": "batch"}}

    worker_start = time.perf_counter()
    profile = next(iter(profiles))
    try:
        if profile.startswith("kivi_"):
            results = _run_kivi_profile_batch(requests, worker_start)
        elif profile.startswith("h2o_heavy"):
            results = _run_h2o_profile_batch(requests, worker_start)
        else:
            results = [_failure(request, f"unsupported Qwen2 KV profile: {profile}") for request in requests]
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:1200]}\n{traceback.format_exc()[-3000:]}"
        results = [_failure(request, error) for request in requests]
    return {"ok": all(item.get("ok") for item in results), "results": results, "worker": {"mode": "batch"}}


def main() -> int:
    payload_text = sys.stdin.read().strip() or os.environ.get("QWEN2_KV_PAYLOAD", "")
    payload = json.loads(payload_text)
    if isinstance(payload, dict) and "requests" in payload:
        result = run_profile_batch(payload)
    else:
        result = run_profile(payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("ok") else 1


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
    return _run_kivi_request(runtime, payload, worker_mode="single")


def _run_h2o_profile(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = _prepare_h2o_runtime(payload, worker_start=time.perf_counter())
    return _run_h2o_request(runtime, payload, worker_mode="single")


def _run_kivi_profile_batch(requests: list[dict[str, Any]], worker_start: float) -> list[dict[str, Any]]:
    runtime = _prepare_kivi_runtime(requests[0], worker_start=worker_start)
    results = []
    for index, request in enumerate(requests):
        result = _run_kivi_request(runtime, request, worker_mode="batch")
        results.append(result)
        if _is_oom_result(result) or _is_fatal_cuda_error(str(result.get("error") or "")):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results


def _run_h2o_profile_batch(requests: list[dict[str, Any]], worker_start: float) -> list[dict[str, Any]]:
    runtime = _prepare_h2o_runtime(requests[0], worker_start=worker_start)
    results = []
    for index, request in enumerate(requests):
        result = _run_h2o_request(runtime, request, worker_mode="batch")
        results.append(result)
        if _is_oom_result(result):
            results.extend(_clone_failure_result(result) for _ in requests[index + 1 :])
            break
    return results


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
        invoke_generate_decode=_invoke_generate_decode,
    )


def _h2o_sizes(tokenizer: Any, payload: dict[str, Any]) -> dict[str, int]:
    return _h2o_sizes_impl(tokenizer, payload)


def _reset_h2o_attention(model: Any, tracker: dict[str, int], heavy_size: int, recent_size: int) -> None:
    _reset_h2o_attention_impl(model, tracker, heavy_size, recent_size)


def _kivi_compatibility_error(model: Any, payload: dict[str, Any]) -> str | None:
    del payload
    return _kivi_compatibility_error_impl(model)


if __name__ == "__main__":
    raise SystemExit(main())
