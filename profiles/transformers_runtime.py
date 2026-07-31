from __future__ import annotations

import json
import os
import resource
import sys
import time
import traceback
from typing import Any


def run_profile_batch(payload: dict[str, Any]) -> dict[str, Any]:
    requests = list(payload.get("requests") or [])
    if not requests:
        return {"ok": True, "results": [], "worker": {"mode": "batch"}}

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
        }

    results = []
    for index, request in enumerate(requests):
        result = _run_one_request(runtime, request)
        results.append(result)
        if _is_cuda_oom(result.get("error")):
            for _ in requests[index + 1 :]:
                results.append(dict(result))
            break
    return {
        "ok": all(item.get("ok") for item in results),
        "results": results,
        "worker": {"mode": "batch"},
    }


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
        prompt_tokens = int(inputs["input_ids"].shape[-1])
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(request.get("max_new_tokens") or 16),
                do_sample=False,
                use_cache=bool(request.get("use_cache", True)),
                pad_token_id=tokenizer.eos_token_id,
            )
        if has_cuda:
            torch.cuda.synchronize(device)
        stage_generate_ms = (time.perf_counter() - generate_start) * 1000
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

    try:
        decode_start = time.perf_counter()
        generated_ids = output_ids[0][prompt_tokens:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if not output_text:
            output_text = " ".join(str(int(token)) for token in generated_ids)
        stage_decode_ms = (time.perf_counter() - decode_start) * 1000
    except Exception as exc:
        return _failure_result(
            error=f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="decode",
            stage_startup_ms=float(runtime["startup_ms"]),
            stage_model_load_ms=float(runtime["model_load_ms"]),
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_generate_ms=stage_generate_ms,
            stage_decode_ms=stage_decode_ms,
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
        "output_text": output_text,
        "latency_ms": total_ms,
        "ttft_ms": total_ms,
        "peak_memory_mib": peak_memory_mib,
        "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stage_startup_ms": float(runtime["startup_ms"]),
        "stage_model_load_ms": float(runtime["model_load_ms"]),
        "stage_tokenize_ms": stage_tokenize_ms,
        "stage_transfer_ms": stage_transfer_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_decode_ms": stage_decode_ms,
        "stage_total_ms": total_ms,
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
