from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Any

from profiles.cache_common import legacy_kv_cache_memory_mib
from profiles.generation_timing import generate_with_first_token_timing


def import_runtime_modules(*, use_kivi: bool) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    modules = {
        "torch": torch,
        "F": F,
        "nn": nn,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "apply_rotary_pos_emb": apply_rotary_pos_emb,
        "repeat_kv": repeat_kv,
    }
    if use_kivi:
        repo_root = Path(__file__).resolve().parents[1]
        kivi_root = repo_root / "third_party" / "KIVI"
        if str(kivi_root) not in sys.path:
            sys.path.insert(0, str(kivi_root))
        from quant.matmul import cuda_bmm_fA_qB_outer
        from quant.new_pack import triton_quantize_and_pack_along_last_dim

        modules["cuda_bmm_fA_qB_outer"] = cuda_bmm_fA_qB_outer
        modules["triton_quantize_and_pack_along_last_dim"] = triton_quantize_and_pack_along_last_dim
    return modules


def require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for true Qwen2 KIVI/H2O profile execution")


def load_qwen2_model(payload: dict[str, Any], torch: Any, auto_model: Any, auto_tokenizer: Any) -> tuple[Any, Any, Any]:
    model_name = str(payload.get("model_name") or "")
    if not model_name:
        raise ValueError("missing model_name")
    cache_dir = payload.get("cache_dir") or None
    local_files_only = bool(payload.get("local_files_only", True))
    tokenizer = auto_tokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = auto_model.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer, model.model.embed_tokens.weight.device


def invoke_generate_decode(model: Any, tokenizer: Any, device: Any, payload: dict[str, Any], torch: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return generate_decode(model, tokenizer, device, payload, torch, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        fallback_kwargs = {}
        if "past_key_values" in kwargs:
            fallback_kwargs["past_key_values"] = kwargs["past_key_values"]
        return generate_decode(model, tokenizer, device, payload, torch, **fallback_kwargs)


def generate_decode(
    model: Any,
    tokenizer: Any,
    device: Any,
    payload: dict[str, Any],
    torch: Any,
    past_key_values: Any = None,
    tokenized_inputs: Any = None,
    stage_startup_ms: float = 0.0,
    stage_model_load_ms: float = 0.0,
    worker_mode: str = "single",
) -> dict[str, Any]:
    stage_tokenize_ms = 0.0
    stage_transfer_ms = 0.0
    stage_generate_ms = 0.0
    stage_decode_ms = 0.0
    request_start = time.perf_counter()

    try:
        tokenize_start = time.perf_counter()
        inputs = tokenized_inputs or tokenizer(str(payload.get("prompt") or ""), return_tensors="pt")
        if tokenized_inputs is None:
            stage_tokenize_ms = (time.perf_counter() - tokenize_start) * 1000
        max_new_tokens = int(payload.get("max_new_tokens") or 16)
    except Exception as exc:
        return failure(
            payload,
            f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="tokenize",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
        )

    try:
        transfer_start = time.perf_counter()
        inputs = {key: value.to(device) for key, value in inputs.items()}
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        stage_transfer_ms = (time.perf_counter() - transfer_start) * 1000
    except Exception as exc:
        return failure(
            payload,
            f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="transfer",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
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
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
                has_cuda=True,
                use_cache=True,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        stage_generate_ms = float(generated["stage_generate_ms"])
    except Exception as exc:
        return failure(
            payload,
            f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="generate",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_generate_ms=(time.perf_counter() - generate_start) * 1000,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
        )

    total_ms = (time.perf_counter() - request_start) * 1000
    kv_cache_memory_mib = _cache_memory_mib(generated.get("past_key_values"))
    return {
        "ok": True,
        "measured": True,
        "output_text": generated["output_text"],
        "latency_ms": total_ms,
        "ttft_ms": generated["ttft_ms"],
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        "kv_cache_memory_mib": kv_cache_memory_mib,
        "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stage_startup_ms": stage_startup_ms,
        "stage_model_load_ms": stage_model_load_ms,
        "stage_tokenize_ms": stage_tokenize_ms,
        "stage_transfer_ms": stage_transfer_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_decode_ms": stage_decode_ms,
        "stage_prefill_ms": generated["stage_prefill_ms"],
        "stage_first_token_ms": generated["stage_first_token_ms"],
        "stage_total_ms": total_ms,
        "ttft_semantics": "first_token",
        "worker_mode": worker_mode,
    }


def _cache_memory_mib(cache: Any) -> float:
    if cache is None:
        return 0.0
    if hasattr(cache, "kv_cache_memory_mib"):
        return float(cache.kv_cache_memory_mib())
    return legacy_kv_cache_memory_mib(cache)


def sequence_suffix(sequences: Any, prompt_len: int) -> list[int]:
    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    tokens = token_list(first)
    return tokens[prompt_len:]


def token_list(values: Any) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    return [int(token) for token in values]


def clone_failure_result(result: dict[str, Any]) -> dict[str, Any]:
    return dict(result)


def is_oom_result(result: dict[str, Any]) -> bool:
    return not bool(result.get("ok")) and "out of memory" in str(result.get("error") or "").lower()


def is_fatal_cuda_error(error: str) -> bool:
    text = str(error or "").lower()
    return (
        "device-side assert triggered" in text
        or ("cuda error" in text and "device-side assert" in text)
        or ("cublas" in text and "execution failed" in text)
    )


def failure(
    payload: dict[str, Any],
    error: str,
    *,
    failure_stage: str = "worker",
    stage_startup_ms: float = 0.0,
    stage_model_load_ms: float = 0.0,
    stage_tokenize_ms: float = 0.0,
    stage_transfer_ms: float = 0.0,
    stage_generate_ms: float = 0.0,
    stage_decode_ms: float = 0.0,
    stage_total_ms: float = 0.0,
    worker_mode: str = "single",
) -> dict[str, Any]:
    return {
        "ok": False,
        "measured": False,
        "error": error,
        "backend": "qwen2_kv_runtime",
        "profile": payload.get("profile", ""),
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
