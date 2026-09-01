from __future__ import annotations

import gc
import os
import resource
import sys
import time
from math import ceil
from pathlib import Path
from typing import Any

from profiles.cache_common import legacy_kv_cache_memory_mib
from profiles.generation_timing import generate_with_first_token_timing

QWEN2_RUNTIME_GPU_MEMORY_LIMIT_MIB = 10_240
QWEN2_RUNTIME_GPU_MEMORY_RESERVE_MIB = {
    0: 2_560,
    1: 2_048,
}


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


def _managed_gpu_indices(torch: Any, *, require_two: bool = False) -> tuple[int, ...]:
    device_count_fn = getattr(torch.cuda, "device_count", None)
    device_count = int(device_count_fn()) if callable(device_count_fn) else 2
    if device_count <= 0:
        raise RuntimeError("Qwen2 measured runtime requires at least one CUDA device")
    if require_two and device_count < 2:
        raise RuntimeError(
            "Qwen2 measured runtime requires at least two CUDA devices for balanced_two_gpu, "
            f"but only detected {device_count} CUDA device(s)"
        )
    if device_count == 1:
        return (0,)
    return (0, 1)


def _gpu_memory_reserve_mib(gpu_index: int) -> int:
    reserve = QWEN2_RUNTIME_GPU_MEMORY_RESERVE_MIB.get(gpu_index)
    if reserve is not None:
        return int(reserve)
    return int(min(QWEN2_RUNTIME_GPU_MEMORY_RESERVE_MIB.values()))


def _safe_max_memory_mib(torch: Any, gpu_index: int) -> int:
    total_mib = int(torch.cuda.get_device_properties(gpu_index).total_memory / 1024 / 1024)
    allowed_mib = min(total_mib - _gpu_memory_reserve_mib(gpu_index), QWEN2_RUNTIME_GPU_MEMORY_LIMIT_MIB)
    if allowed_mib <= 0:
        raise RuntimeError(f"GPU {gpu_index} total memory too small for Qwen2 runtime: {total_mib} MiB")
    return allowed_mib


def _build_visible_gpu_max_memory(torch: Any, gpu_indices: tuple[int, ...]) -> dict[int, str]:
    return {gpu_index: f"{_safe_max_memory_mib(torch, gpu_index)}MiB" for gpu_index in gpu_indices}


def _build_qwen2_device_map(num_hidden_layers: int, gpu_indices: tuple[int, ...] = (0, 1)) -> dict[str, int]:
    if num_hidden_layers <= 0:
        raise ValueError(f"invalid Qwen2 hidden layer count: {num_hidden_layers}")
    if len(gpu_indices) == 1:
        only_gpu = gpu_indices[0]
        device_map = {"model.embed_tokens": only_gpu}
        for layer_idx in range(num_hidden_layers):
            device_map[f"model.layers.{layer_idx}"] = only_gpu
        device_map["model.norm"] = only_gpu
        device_map["lm_head"] = only_gpu
        return device_map
    split_index = int(ceil(num_hidden_layers / 2))
    left_gpu, right_gpu = gpu_indices[0], gpu_indices[1]
    device_map = {"model.embed_tokens": left_gpu}
    for layer_idx in range(num_hidden_layers):
        device_map[f"model.layers.{layer_idx}"] = left_gpu if layer_idx < split_index else right_gpu
    device_map["model.norm"] = right_gpu
    device_map["lm_head"] = right_gpu
    return device_map


def _peak_memory_by_gpu_mib(torch: Any) -> dict[int, float]:
    peaks: dict[int, float] = {}
    for gpu_index in _managed_gpu_indices(torch):
        peaks[gpu_index] = float(torch.cuda.max_memory_allocated(gpu_index) / 1024 / 1024)
    return peaks


def _gpu_memory_snapshot(torch: Any) -> dict[str, float]:
    snapshot: dict[str, float] = {}
    for gpu_index in _managed_gpu_indices(torch):
        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_index)
            free_mib = float(free_bytes / 1024 / 1024)
            total_mib = float(total_bytes / 1024 / 1024)
            used_mib = total_mib - free_mib
        else:
            total_mib = float(torch.cuda.get_device_properties(gpu_index).total_memory / 1024 / 1024)
            used_mib = float(torch.cuda.memory_allocated(gpu_index) / 1024 / 1024)
            free_mib = total_mib - used_mib
        snapshot[f"gpu{gpu_index}_used_mib"] = used_mib
        snapshot[f"gpu{gpu_index}_free_mib"] = free_mib
        snapshot[f"gpu{gpu_index}_total_mib"] = total_mib
    return snapshot


def _reset_peak_memory_stats(torch: Any) -> None:
    for gpu_index in _managed_gpu_indices(torch):
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)


def _peak_memory_fields(torch: Any) -> dict[str, float]:
    peaks = _peak_memory_by_gpu_mib(torch)
    fields: dict[str, float] = {"peak_memory_mib": sum(peaks.values())}
    for gpu_index, peak in peaks.items():
        fields[f"gpu{gpu_index}_peak_memory_mib"] = peak
    return fields


def _oom_extra_fields(torch: Any, error: str) -> dict[str, float]:
    if "out of memory" not in str(error or "").lower():
        return {}
    return _gpu_memory_snapshot(torch)


def release_runtime_cuda_resources(torch: Any, *resources: Any) -> None:
    for resource_obj in resources:
        if hasattr(resource_obj, "clear"):
            try:
                resource_obj.clear()
            except Exception:
                pass
    gc.collect()
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
        for gpu_index in _managed_gpu_indices(torch):
            torch.cuda.synchronize(gpu_index)


def _load_qwen2_num_hidden_layers(model_name: str, cache_dir: Any, local_files_only: bool) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    if num_hidden_layers <= 0:
        raise ValueError(f"unsupported Qwen2 config: num_hidden_layers={num_hidden_layers}")
    return num_hidden_layers


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
    device_strategy = str(payload.get("device_strategy") or "balanced_two_gpu")
    num_hidden_layers = int(payload.get("num_hidden_layers") or 0) or _load_qwen2_num_hidden_layers(
        model_name,
        cache_dir,
        local_files_only,
    )
    gpu_indices = _managed_gpu_indices(torch, require_two=device_strategy == "balanced_two_gpu")
    device_map = _build_qwen2_device_map(num_hidden_layers, gpu_indices)
    model = auto_model.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        device_map=device_map,
        max_memory=_build_visible_gpu_max_memory(torch, gpu_indices),
        low_cpu_mem_usage=True,
    )
    model.hf_device_map = device_map
    model.eval()
    return model, tokenizer, model.model.embed_tokens.weight.device


def binding_diagnostics(payload: dict[str, Any], torch: Any | None = None) -> dict[str, Any]:
    visible_count: int | None = None
    if torch is not None:
        device_count_fn = getattr(getattr(torch, "cuda", None), "device_count", None)
        if callable(device_count_fn):
            try:
                visible_count = int(device_count_fn())
            except Exception:
                visible_count = None
    diagnostics: dict[str, Any] = {
        "runtime_device_strategy": str(payload.get("device_strategy") or ""),
        "runtime_cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
        "worker_cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES") or ""),
    }
    if visible_count is not None:
        diagnostics["runtime_visible_device_count"] = visible_count
    return diagnostics


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
        _reset_peak_memory_stats(torch)
        stage_transfer_ms = (time.perf_counter() - transfer_start) * 1000
    except Exception as exc:
        error = f"{type(exc).__name__}: {str(exc)[:1200]}"
        return failure(
            payload,
            error,
            failure_stage="transfer",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
            extra=_oom_extra_fields(torch, error),
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
        error = f"{type(exc).__name__}: {str(exc)[:1200]}"
        return failure(
            payload,
            error,
            failure_stage="generate",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_generate_ms=(time.perf_counter() - generate_start) * 1000,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
            extra=_oom_extra_fields(torch, error),
        )

    total_ms = (time.perf_counter() - request_start) * 1000
    kv_cache_memory_mib = _cache_memory_mib(generated.get("past_key_values"))
    return {
        "ok": True,
        "measured": True,
        "output_text": generated["output_text"],
        "latency_ms": total_ms,
        "ttft_ms": generated["ttft_ms"],
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
        "past_key_values": generated.get("past_key_values"),
        **binding_diagnostics(payload, torch),
        **_peak_memory_fields(torch),
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
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
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
        **binding_diagnostics(payload),
    }
    if extra:
        result.update(extra)
    return result
