from __future__ import annotations

import json
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from profiles.kivi_cache import KIVICache, KIVILayerState, build_kivi_cache


def run_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a Qwen2 KV-cache profile in a subprocess with heavyweight imports delayed."""

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
    return (
        "KIVI proof missing: no quantized cache block and/or quant GEMV kernel call was observed. "
        f"prompt_tokens={prompt_tokens} max_new_tokens={max_new_tokens} residual_length={residual_length} "
        f"quantized_layers={quantized_layers} kernel_calls={kernel_calls}"
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
        if _is_oom_result(result):
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
    modules = _import_runtime_modules(use_kivi=True)
    torch = modules["torch"]
    _require_cuda(torch)

    startup_ms = (time.perf_counter() - worker_start) * 1000
    load_start = time.perf_counter()
    model, tokenizer, device = _load_qwen2_model(payload, torch, modules["AutoModelForCausalLM"], modules["AutoTokenizer"])
    model_load_ms = (time.perf_counter() - load_start) * 1000
    bits = int(payload.get("bits") or (2 if str(payload.get("profile")) == "kivi_2bit" else 4))
    residual_length = int(payload.get("kivi_residual_length") or 32)
    tracker = {
        "kivi_kernel_calls": 0,
        "kivi_quantize_calls": 0,
        "kivi_quantized_layers": 0,
        "kivi_quantized_tokens": 0,
    }
    _install_qwen2_attention(model, Qwen2KIVIAttention, tracker, bits=bits, payload=payload, modules=modules)
    return {
        "modules": modules,
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "startup_ms": startup_ms,
        "model_load_ms": model_load_ms,
        "bits": bits,
        "residual_length": residual_length,
        "tracker": tracker,
    }


def _prepare_h2o_runtime(payload: dict[str, Any], *, worker_start: float) -> dict[str, Any]:
    modules = _import_runtime_modules(use_kivi=False)
    torch = modules["torch"]
    _require_cuda(torch)

    startup_ms = (time.perf_counter() - worker_start) * 1000
    load_start = time.perf_counter()
    model, tokenizer, device = _load_qwen2_model(payload, torch, modules["AutoModelForCausalLM"], modules["AutoTokenizer"])
    model_load_ms = (time.perf_counter() - load_start) * 1000
    tracker = {
        "h2o_prune_events": 0,
        "h2o_mask_events": 0,
        "h2o_cache_budget": 0,
        "h2o_kept_tokens": 0,
        "h2o_prompt_tokens": 0,
    }
    initial_sizes = _h2o_sizes(tokenizer, payload)
    _install_qwen2_attention(
        model,
        Qwen2H2OAttention,
        tracker,
        bits=0,
        payload={**payload, "h2o_heavy_size": initial_sizes["heavy_size"], "h2o_recent_size": initial_sizes["recent_size"]},
        modules=modules,
    )
    return {
        "modules": modules,
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "startup_ms": startup_ms,
        "model_load_ms": model_load_ms,
        "tracker": tracker,
    }


def _run_kivi_request(runtime: dict[str, Any], payload: dict[str, Any], *, worker_mode: str) -> dict[str, Any]:
    tracker = runtime["tracker"]
    if worker_mode == "batch":
        tracker.update({
            "kivi_kernel_calls": 0,
            "kivi_quantize_calls": 0,
            "kivi_quantized_layers": 0,
            "kivi_quantized_tokens": 0,
        })
    tokenized = runtime["tokenizer"](str(payload.get("prompt") or ""), return_tensors="pt")
    prompt_tokens = int(tokenized["input_ids"].shape[-1])
    max_new_tokens = int(payload.get("max_new_tokens") or 16)
    residual_length = int(runtime["residual_length"])
    bits = int(runtime["bits"])
    cache = build_kivi_cache(
        runtime["model"].config,
        residual_length=residual_length,
        group_size=int(payload.get("kivi_group_size") or 32),
        k_bits=bits,
        v_bits=bits,
    )
    result = _invoke_generate_decode(
        runtime["model"],
        runtime["tokenizer"],
        runtime["device"],
        payload,
        runtime["torch"],
        past_key_values=cache,
        tokenized_inputs=tokenized,
        stage_startup_ms=float(runtime["startup_ms"]),
        stage_model_load_ms=float(runtime["model_load_ms"]),
        worker_mode=worker_mode,
    )
    result.update(tracker)
    result.update(
        {
            "backend": "qwen2_kivi",
            "bits": bits,
            "kivi_group_size": int(payload.get("kivi_group_size") or 32),
            "kivi_residual_length": residual_length,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
        }
    )
    if result.get("ok") and (tracker["kivi_quantized_layers"] <= 0 or tracker["kivi_kernel_calls"] <= 0):
        result["ok"] = False
        result["measured"] = False
        result["error"] = _kivi_proof_error(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            residual_length=residual_length,
            quantized_layers=tracker["kivi_quantized_layers"],
            kernel_calls=tracker["kivi_kernel_calls"],
        )
        result["failure_stage"] = "generate"
    return result


def _run_h2o_request(runtime: dict[str, Any], payload: dict[str, Any], *, worker_mode: str) -> dict[str, Any]:
    tracker = runtime["tracker"]
    sizes = _h2o_sizes(runtime["tokenizer"], payload)
    if worker_mode == "batch":
        tracker.update(
            {
                "h2o_prune_events": 0,
                "h2o_mask_events": 0,
                "h2o_cache_budget": sizes["heavy_size"] + sizes["recent_size"],
                "h2o_kept_tokens": 0,
                "h2o_prompt_tokens": sizes["prompt_tokens"],
            }
        )
    _reset_h2o_attention(runtime["model"], tracker, sizes["heavy_size"], sizes["recent_size"])
    result = _invoke_generate_decode(
        runtime["model"],
        runtime["tokenizer"],
        runtime["device"],
        payload,
        runtime["torch"],
        stage_startup_ms=float(runtime["startup_ms"]),
        stage_model_load_ms=float(runtime["model_load_ms"]),
        worker_mode=worker_mode,
    )
    result.update(tracker)
    result.update(
        {
            "backend": "qwen2_h2o",
            "h2o_heavy_ratio": float(payload.get("h2o_heavy_ratio") or 0.1),
            "h2o_recent_ratio": float(payload.get("h2o_recent_ratio") or 0.1),
            "h2o_heavy_size": sizes["heavy_size"],
            "h2o_recent_size": sizes["recent_size"],
        }
    )
    if result.get("ok") and (sizes["prompt_tokens"] <= tracker["h2o_cache_budget"] or tracker["h2o_prune_events"] <= 0):
        result["ok"] = False
        result["measured"] = False
        result["error"] = (
            "H2O proof missing: request did not exceed heavy-hitter cache budget or no prune event ran. "
            f"prompt_tokens={sizes['prompt_tokens']} budget={tracker['h2o_cache_budget']} "
            f"prune_events={tracker['h2o_prune_events']}"
        )
        result["failure_stage"] = "generate"
    return result


def _h2o_sizes(tokenizer: Any, payload: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(tokenizer(str(payload.get("prompt") or ""), return_tensors="pt")["input_ids"].shape[-1])
    heavy_ratio = float(payload.get("h2o_heavy_ratio") or 0.1)
    recent_ratio = float(payload.get("h2o_recent_ratio") or 0.1)
    heavy_size = max(1, int(prompt_tokens * heavy_ratio))
    recent_size = max(1, int(prompt_tokens * recent_ratio))
    return {"prompt_tokens": prompt_tokens, "heavy_size": heavy_size, "recent_size": recent_size}


def _reset_h2o_attention(model: Any, tracker: dict[str, int], heavy_size: int, recent_size: int) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for layer in layers:
        attention = layer.self_attn
        attention.tracker = tracker
        attention.hh_size = heavy_size
        attention.recent_size = recent_size
        attention.cache_budget = heavy_size + recent_size
        attention.hh_score = None


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


def _clone_failure_result(result: dict[str, Any]) -> dict[str, Any]:
    return dict(result)


def _is_oom_result(result: dict[str, Any]) -> bool:
    return not bool(result.get("ok")) and "out of memory" in str(result.get("error") or "").lower()


def _import_runtime_modules(use_kivi: bool) -> dict[str, Any]:
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


def _load_qwen2_model(payload: dict[str, Any], torch: Any, auto_model: Any, auto_tokenizer: Any) -> tuple[Any, Any, Any]:
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


def _install_qwen2_attention(model: Any, wrapper_cls: type, tracker: dict[str, int], bits: int, payload: dict[str, Any], modules: dict[str, Any]) -> None:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("loaded model does not expose model.layers; expected Qwen2ForCausalLM")
    for layer_idx, layer in enumerate(layers):
        layer.self_attn = wrapper_cls(layer.self_attn, model.config, layer_idx, tracker, bits, payload, modules)
    model.config.use_cache = True


def _greedy_decode(model: Any, tokenizer: Any, device: Any, payload: dict[str, Any], torch: Any) -> dict[str, Any]:
    inputs = tokenizer(str(payload.get("prompt") or ""), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    max_new_tokens = int(payload.get("max_new_tokens") or 16)
    generated: list[Any] = []
    past_key_values = None
    next_input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    seen_tokens = 0

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    ttft_ms = None
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits, past_key_values = _manual_qwen2_forward(
                model,
                next_input_ids,
                past_key_values,
                attention_mask if past_key_values is None else None,
                seen_tokens,
                torch,
            )
            seen_tokens += int(next_input_ids.shape[-1])
            logits = logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.detach().cpu())
            if ttft_ms is None:
                torch.cuda.synchronize(device)
                ttft_ms = (time.perf_counter() - start) * 1000
            next_input_ids = next_token.to(device)
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - start) * 1000
    output_ids = torch.cat(generated, dim=-1) if generated else inputs["input_ids"][:, :0]
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if not output_text:
        output_text = " ".join(str(int(token)) for token in output_ids[0])
    return {
        "ok": True,
        "measured": True,
        "output_text": output_text,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms if ttft_ms is not None else latency_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def _generate_decode(
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
        prompt_len = int(inputs["input_ids"].shape[-1])
        if tokenized_inputs is None:
            stage_tokenize_ms = (time.perf_counter() - tokenize_start) * 1000
        max_new_tokens = int(payload.get("max_new_tokens") or 16)
    except Exception as exc:
        return _failure(
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
        return _failure(
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
            generate_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "return_dict_in_generate": True,
            }
            if past_key_values is not None:
                generate_kwargs["past_key_values"] = past_key_values
            generated = model.generate(**generate_kwargs)
        torch.cuda.synchronize(device)
        stage_generate_ms = (time.perf_counter() - generate_start) * 1000
    except Exception as exc:
        return _failure(
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

    try:
        decode_start = time.perf_counter()
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        output_ids = _sequence_suffix(sequences, prompt_len)
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        if not output_text:
            output_text = " ".join(str(token) for token in output_ids)
        stage_decode_ms = (time.perf_counter() - decode_start) * 1000
    except Exception as exc:
        return _failure(
            payload,
            f"{type(exc).__name__}: {str(exc)[:1200]}",
            failure_stage="decode",
            stage_startup_ms=stage_startup_ms,
            stage_model_load_ms=stage_model_load_ms,
            stage_tokenize_ms=stage_tokenize_ms,
            stage_transfer_ms=stage_transfer_ms,
            stage_generate_ms=stage_generate_ms,
            stage_decode_ms=stage_decode_ms,
            stage_total_ms=(time.perf_counter() - request_start) * 1000,
            worker_mode=worker_mode,
        )

    total_ms = (time.perf_counter() - request_start) * 1000
    return {
        "ok": True,
        "measured": True,
        "output_text": output_text,
        "latency_ms": total_ms,
        "ttft_ms": total_ms,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
        "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "stage_startup_ms": stage_startup_ms,
        "stage_model_load_ms": stage_model_load_ms,
        "stage_tokenize_ms": stage_tokenize_ms,
        "stage_transfer_ms": stage_transfer_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_decode_ms": stage_decode_ms,
        "stage_total_ms": total_ms,
        "worker_mode": worker_mode,
    }


def _sequence_suffix(sequences: Any, prompt_len: int) -> list[int]:
    first = sequences[0] if hasattr(sequences, "__getitem__") else sequences
    tokens = _token_list(first)
    return tokens[prompt_len:]


def _token_list(values: Any) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    return [int(token) for token in values]


def _manual_qwen2_forward(
    model: Any,
    input_ids: Any,
    past_key_values: tuple[Any, ...] | None,
    attention_mask: Any,
    position_start: int,
    torch: Any,
) -> tuple[Any, tuple[Any, ...]]:
    core = model.model
    embed_device = core.embed_tokens.weight.device
    input_ids = input_ids.to(embed_device)
    hidden_states = core.embed_tokens(input_ids)
    q_len = int(hidden_states.shape[1])
    position_ids = torch.arange(position_start, position_start + q_len, device=embed_device).unsqueeze(0)
    causal_mask = _manual_causal_mask(hidden_states, attention_mask, _past_length(past_key_values), position_ids, torch)
    next_cache = []
    for layer_idx, decoder_layer in enumerate(core.layers):
        layer_past = past_key_values[layer_idx] if past_key_values is not None else None
        layer_device = next(decoder_layer.parameters()).device
        hidden_states = hidden_states.to(layer_device)
        layer_position_ids = position_ids.to(layer_device)
        layer_causal_mask = causal_mask.to(layer_device) if causal_mask is not None else None
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=layer_causal_mask,
            position_ids=layer_position_ids,
            past_key_value=layer_past,
            output_attentions=False,
            use_cache=True,
            cache_position=layer_position_ids[0],
        )
        hidden_states = layer_outputs[0]
        next_cache.append(layer_outputs[1])
    norm_device = next(core.norm.parameters()).device
    hidden_states = core.norm(hidden_states.to(norm_device))
    lm_head_device = next(model.lm_head.parameters()).device
    logits = model.lm_head(hidden_states.to(lm_head_device))
    return logits, tuple(next_cache)


def _manual_causal_mask(hidden_states: Any, attention_mask: Any, past_len: int, position_ids: Any, torch: Any) -> Any:
    q_len = int(hidden_states.shape[1])
    if q_len == 1 and attention_mask is None:
        return None
    dtype = hidden_states.dtype
    device = hidden_states.device
    kv_len = past_len + q_len
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((q_len, kv_len), fill_value=min_dtype, dtype=dtype, device=device)
    causal_mask = torch.triu(causal_mask, diagonal=1 + past_len)
    causal_mask = causal_mask[None, None, :, :].expand(hidden_states.shape[0], 1, -1, -1)
    if attention_mask is not None:
        mask_len = min(int(attention_mask.shape[-1]), kv_len)
        padding_mask = causal_mask[:, :, :, :mask_len] + attention_mask[:, None, None, :mask_len]
        causal_mask = causal_mask.clone()
        causal_mask[:, :, :, :mask_len] = causal_mask[:, :, :, :mask_len].masked_fill(padding_mask == 0, min_dtype)
    return causal_mask


def _past_length(past_key_values: Any) -> int:
    if not past_key_values:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    first = past_key_values[0]
    if first is None:
        return 0
    if isinstance(first, tuple) and len(first) >= 9:
        return int(first[-1])
    if isinstance(first, tuple) and first and hasattr(first[0], "shape"):
        return int(first[0].shape[2])
    return 0


class Qwen2KIVIAttention:
    def __new__(cls, source: Any, config: Any, layer_idx: int, tracker: dict[str, int], bits: int, payload: dict[str, Any], modules: dict[str, Any]) -> Any:
        nn = modules["nn"]

        class _Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source = source
                self.config = config
                self.layer_idx = layer_idx
                self.tracker = tracker
                self.q_proj = source.q_proj
                self.k_proj = source.k_proj
                self.v_proj = source.v_proj
                self.o_proj = source.o_proj
                self.rotary_emb = getattr(source, "rotary_emb", None)
                self.hidden_size = config.hidden_size
                self.num_heads = config.num_attention_heads
                self.head_dim = getattr(source, "head_dim", self.hidden_size // self.num_heads)
                self.num_key_value_heads = config.num_key_value_heads
                self.num_key_value_groups = self.num_heads // self.num_key_value_heads
                self.attention_dropout = getattr(source, "attention_dropout", 0.0)
                self.is_causal = True
                self.k_bits = bits
                self.v_bits = bits
                self.group_size = int(payload.get("kivi_group_size") or 32)
                self.residual_length = int(payload.get("kivi_residual_length") or 32)

            def forward(self, hidden_states: Any, attention_mask: Any = None, position_ids: Any = None, past_key_value: Any = None, output_attentions: bool = False, use_cache: bool = False, cache_position: Any = None, position_embeddings: Any = None, **kwargs: Any) -> tuple[Any, Any, Any]:
                torch = modules["torch"]
                F = modules["F"]
                repeat_kv = modules["repeat_kv"]
                cuda_bmm = modules["cuda_bmm_fA_qB_outer"]
                quant_pack = modules["triton_quantize_and_pack_along_last_dim"]
                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                layer_state = None
                if isinstance(past_key_value, KIVICache):
                    layer_state = past_key_value[self.layer_idx]
                elif past_key_value is not None:
                    legacy = _cache_to_legacy(past_key_value, self.layer_idx)
                    if legacy is not None:
                        layer_state = KIVILayerState(*legacy)
                kv_seq_len = key_states.shape[-2] + (int(layer_state.kv_seq_len) if layer_state is not None else 0)
                query_states, key_states = _apply_rope(self, query_states, key_states, value_states, position_ids, position_embeddings, modules)

                if layer_state is not None:
                    key_q = layer_state.key_q
                    key_full = layer_state.key_full
                    key_scale = layer_state.key_scale
                    key_mn = layer_state.key_mn
                    value_q = layer_state.value_q
                    value_full = layer_state.value_full
                    value_scale = layer_state.value_scale
                    value_mn = layer_state.value_mn
                    if key_q is not None:
                        with torch.cuda.device(query_states.device):
                            attn_q = cuda_bmm(self.group_size, query_states, key_q, key_scale, key_mn, self.k_bits)
                        self.tracker["kivi_kernel_calls"] += 1
                    else:
                        attn_q = None
                    key_full = torch.cat([key_full, key_states], dim=2) if key_full is not None else key_states
                    attn_full = torch.matmul(query_states, repeat_kv(key_full, self.num_key_value_groups).transpose(2, 3))
                    attn_weights = torch.cat([attn_q, attn_full], dim=-1) if attn_q is not None else attn_full
                    attn_weights = attn_weights / math.sqrt(self.head_dim)
                    if key_full.shape[-2] == self.residual_length:
                        key_new, scale_new, mn_new = quant_pack(key_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                        self.tracker["kivi_quantized_layers"] += 1
                        self.tracker["kivi_quantized_tokens"] += int(key_full.shape[-2])
                        key_full = None
                        key_q = torch.cat([key_q, key_new], dim=3) if key_q is not None else key_new
                        key_scale = torch.cat([key_scale, scale_new], dim=3) if key_scale is not None else scale_new
                        key_mn = torch.cat([key_mn, mn_new], dim=3) if key_mn is not None else mn_new

                    attn_weights = _mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
                    value_full = torch.cat([value_full, value_states], dim=2) if value_full is not None else value_states
                    value_full_len = value_full.shape[-2]
                    if value_q is None:
                        attn_output = torch.matmul(attn_weights, repeat_kv(value_full, self.num_key_value_groups))
                    else:
                        with torch.cuda.device(query_states.device):
                            attn_output = cuda_bmm(self.group_size, attn_weights[:, :, :, :-value_full_len], value_q, value_scale, value_mn, self.v_bits)
                        self.tracker["kivi_kernel_calls"] += 1
                        attn_output = attn_output + torch.matmul(attn_weights[:, :, :, -value_full_len:], repeat_kv(value_full, self.num_key_value_groups))
                    if value_full_len > self.residual_length:
                        value_new, scale_new, mn_new = quant_pack(value_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                        value_full = value_full[:, :, 1:, :].contiguous()
                        value_q = torch.cat([value_q, value_new], dim=2) if value_q is not None else value_new
                        value_scale = torch.cat([value_scale, scale_new], dim=2) if value_scale is not None else scale_new
                        value_mn = torch.cat([value_mn, mn_new], dim=2) if value_mn is not None else mn_new
                else:
                    attn_weights = torch.matmul(query_states, repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    if key_states.shape[-2] < self.residual_length:
                        key_q = None
                        key_full = key_states
                    else:
                        quant_len = key_states.shape[-2] - (key_states.shape[-2] % self.residual_length)
                        key_q_src = key_states[:, :, :quant_len, :].contiguous()
                        key_full = key_states[:, :, quant_len:, :].contiguous() if quant_len < key_states.shape[-2] else None
                        key_q, key_scale, key_mn = quant_pack(key_q_src.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                        self.tracker["kivi_quantized_layers"] += 1
                        self.tracker["kivi_quantized_tokens"] += int(quant_len)
                    if key_states.shape[-2] < self.residual_length:
                        key_scale = None
                        key_mn = None
                    if value_states.shape[-2] <= self.residual_length:
                        value_q = None
                        value_full = value_states
                        value_scale = None
                        value_mn = None
                    else:
                        value_q_src = value_states[:, :, :-self.residual_length, :].contiguous()
                        value_full = value_states[:, :, -self.residual_length:, :].contiguous()
                        value_q, value_scale, value_mn = quant_pack(value_q_src, self.group_size, self.v_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                    attn_weights = _mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
                    attn_output = torch.matmul(attn_weights, repeat_kv(value_states, self.num_key_value_groups))

                past = None
                if use_cache:
                    state = KIVILayerState(
                        key_q=key_q,
                        key_full=key_full,
                        key_scale=key_scale,
                        key_mn=key_mn,
                        value_q=value_q,
                        value_full=value_full,
                        value_scale=value_scale,
                        value_mn=value_mn,
                        kv_seq_len=int(kv_seq_len),
                    )
                    if isinstance(past_key_value, KIVICache):
                        past = past_key_value.update(self.layer_idx, state)
                    else:
                        cache = KIVICache(
                            self.config.num_hidden_layers,
                            residual_length=self.residual_length,
                            group_size=self.group_size,
                            k_bits=self.k_bits,
                            v_bits=self.v_bits,
                        )
                        past = cache.update(self.layer_idx, state)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)
                return attn_output, (attn_weights if output_attentions else None), past

        return _Attention()


class Qwen2H2OAttention:
    def __new__(cls, source: Any, config: Any, layer_idx: int, tracker: dict[str, int], bits: int, payload: dict[str, Any], modules: dict[str, Any]) -> Any:
        nn = modules["nn"]

        class _Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.source = source
                self.config = config
                self.layer_idx = layer_idx
                self.tracker = tracker
                self.q_proj = source.q_proj
                self.k_proj = source.k_proj
                self.v_proj = source.v_proj
                self.o_proj = source.o_proj
                self.rotary_emb = getattr(source, "rotary_emb", None)
                self.hidden_size = config.hidden_size
                self.num_heads = config.num_attention_heads
                self.head_dim = getattr(source, "head_dim", self.hidden_size // self.num_heads)
                self.num_key_value_heads = config.num_key_value_heads
                self.num_key_value_groups = self.num_heads // self.num_key_value_heads
                self.hh_size = int(payload["h2o_heavy_size"])
                self.recent_size = int(payload["h2o_recent_size"])
                self.cache_budget = self.hh_size + self.recent_size
                self.hh_score = None

            def forward(self, hidden_states: Any, attention_mask: Any = None, position_ids: Any = None, past_key_value: Any = None, output_attentions: bool = False, use_cache: bool = False, cache_position: Any = None, position_embeddings: Any = None, **kwargs: Any) -> tuple[Any, Any, Any]:
                torch = modules["torch"]
                repeat_kv = modules["repeat_kv"]
                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                past_key_value = _cache_to_legacy(past_key_value, self.layer_idx)
                query_states, key_states = _apply_rope(self, query_states, key_states, value_states, position_ids, position_embeddings, modules)
                if past_key_value is not None:
                    key_states = torch.cat([past_key_value[0], key_states], dim=2)
                    value_states = torch.cat([past_key_value[1], value_states], dim=2)
                kv_seq_len = key_states.shape[-2]
                key_for_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = _mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
                attn_output = torch.matmul(attn_weights, value_for_attn)
                past = (key_states, value_states) if use_cache else None
                if past is not None:
                    past = self._prune(past, attn_weights.detach())
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                return self.o_proj(attn_output), (attn_weights if output_attentions else None), past

            def _prune(self, past: tuple[Any, Any], attn_weights: Any) -> tuple[Any, Any]:
                torch = modules["torch"]
                num_new_tokens = int(attn_weights.shape[2])
                scores = attn_weights.sum(dim=2)
                if self.num_key_value_groups > 1:
                    scores = scores.view(scores.shape[0], self.num_key_value_heads, self.num_key_value_groups, scores.shape[-1]).sum(dim=2)
                if self.hh_score is None:
                    self.hh_score = scores.sum(dim=0)
                else:
                    updated = scores.sum(dim=0)
                    old_len = min(self.hh_score.shape[-1], updated.shape[-1] - num_new_tokens)
                    if old_len > 0:
                        updated[:, :old_len] += self.hh_score[:, :old_len]
                    self.hh_score = updated
                seq_len = past[0].shape[2]
                if seq_len <= self.cache_budget:
                    self.tracker["h2o_kept_tokens"] = max(self.tracker["h2o_kept_tokens"], int(seq_len))
                    return past
                select_len = max(1, seq_len - self.recent_size)
                hh_size = min(self.hh_size, select_len)
                _, keep_topk = torch.topk(self.hh_score[:, :select_len], hh_size, dim=-1)
                keep_topk = keep_topk.sort().values
                recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(self.num_key_value_heads, 1)
                keep_idx = torch.cat([keep_topk, recent], dim=-1)
                pruned_k = []
                pruned_v = []
                for batch_idx in range(past[0].shape[0]):
                    head_k = []
                    head_v = []
                    for head_idx in range(self.num_key_value_heads):
                        idx = keep_idx[head_idx]
                        head_k.append(past[0][batch_idx, head_idx].index_select(0, idx))
                        head_v.append(past[1][batch_idx, head_idx].index_select(0, idx))
                    pruned_k.append(torch.stack(head_k, dim=0))
                    pruned_v.append(torch.stack(head_v, dim=0))
                mask = torch.zeros_like(self.hh_score, dtype=torch.bool)
                mask.scatter_(1, keep_idx, True)
                self.hh_score = self.hh_score[mask].view(self.num_key_value_heads, -1)
                self.tracker["h2o_prune_events"] += 1
                self.tracker["h2o_mask_events"] += 1
                self.tracker["h2o_kept_tokens"] = int(keep_idx.shape[-1])
                return torch.stack(pruned_k, dim=0).contiguous(), torch.stack(pruned_v, dim=0).contiguous()

        return _Attention()


def _apply_rope(attn: Any, query_states: Any, key_states: Any, value_states: Any, position_ids: Any, position_embeddings: Any, modules: dict[str, Any]) -> tuple[Any, Any]:
    apply_rotary_pos_emb = modules["apply_rotary_pos_emb"]
    if position_embeddings is None:
        if attn.rotary_emb is None:
            return query_states, key_states
        try:
            seq_len = int(position_ids.max().item()) + 1 if position_ids is not None else int(key_states.shape[-2])
            seq_len = max(seq_len, int(key_states.shape[-2]))
            cos, sin = attn.rotary_emb(value_states, seq_len=seq_len)
        except TypeError:
            cos, sin = attn.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    try:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    except TypeError:
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _cache_to_legacy(past_key_value: Any, layer_idx: int) -> Any:
    if past_key_value is None or isinstance(past_key_value, tuple):
        return past_key_value
    try:
        if hasattr(past_key_value, "key_cache") and len(past_key_value.key_cache) > layer_idx:
            key = past_key_value.key_cache[layer_idx]
            value = past_key_value.value_cache[layer_idx]
            if key is not None and value is not None and key.numel() > 0:
                return (key, value)
    except Exception:
        return None
    return None


def _mask_softmax(attn_weights: Any, attention_mask: Any, bsz: int, num_heads: int, q_len: int, kv_seq_len: int, torch: Any, nn: Any) -> Any:
    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(f"attention weights size mismatch: got {tuple(attn_weights.size())}, expected {(bsz, num_heads, q_len, kv_seq_len)}")
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
    return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)


def _require_cuda(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for true Qwen2 KIVI/H2O profile execution")


def _failure(
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


if __name__ == "__main__":
    raise SystemExit(main())
