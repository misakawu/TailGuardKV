from __future__ import annotations

import math
import time
from typing import Any, Callable

from profiles.kivi_cache import KIVICache, KIVILayerState
from profiles.qwen2_runtime_layout import apply_rope, mask_softmax, qwen2_layout_error


def kivi_proof_error(
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


def prepare_kivi_runtime(
    payload: dict[str, Any],
    *,
    worker_start: float,
    import_runtime_modules: Callable[..., dict[str, Any]],
    require_cuda: Callable[[Any], None],
    load_qwen2_model: Callable[[dict[str, Any], Any, Any, Any], tuple[Any, Any, Any]],
    install_qwen2_attention: Callable[[Any, type, dict[str, int], int, dict[str, Any], dict[str, Any]], None],
    attention_cls: type,
) -> dict[str, Any]:
    modules = import_runtime_modules(use_kivi=True)
    torch = modules["torch"]
    require_cuda(torch)

    startup_ms = (time.perf_counter() - worker_start) * 1000
    load_start = time.perf_counter()
    model, tokenizer, device = load_qwen2_model(payload, torch, modules["AutoModelForCausalLM"], modules["AutoTokenizer"])
    model_load_ms = (time.perf_counter() - load_start) * 1000
    compat_error = qwen2_layout_error(model)
    if compat_error is not None:
        raise ValueError(f"unsupported KIVI runtime model: {compat_error}")
    bits = int(payload.get("bits") or (2 if str(payload.get("profile")) == "kivi_2bit" else 4))
    residual_length = int(payload.get("kivi_residual_length") or 32)
    tracker = {
        "kivi_kernel_calls": 0,
        "kivi_quantize_calls": 0,
        "kivi_quantized_layers": 0,
        "kivi_quantized_tokens": 0,
    }
    install_qwen2_attention(model, attention_cls, tracker, bits=bits, payload=payload, modules=modules)
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


def run_kivi_request(
    runtime: dict[str, Any],
    payload: dict[str, Any],
    *,
    worker_mode: str,
    build_kivi_cache: Callable[..., KIVICache],
    invoke_generate_decode: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    tracker = runtime["tracker"]
    if worker_mode == "batch":
        tracker.update(
            {
                "kivi_kernel_calls": 0,
                "kivi_quantize_calls": 0,
                "kivi_quantized_layers": 0,
                "kivi_quantized_tokens": 0,
            }
        )
    tokenized = runtime["tokenizer"](str(payload.get("prompt") or ""), return_tensors="pt")
    reusable_cache = payload.get("_runtime_reusable_kivi_cache")
    cached_prompt_token_ids = payload.get("_runtime_cached_prompt_token_ids")
    rebuild_reason = str(payload.get("_runtime_cache_rebuild_reason") or "")
    prompt_tokens = int(tokenized["input_ids"].shape[-1])
    max_new_tokens = int(payload.get("max_new_tokens") or 16)
    residual_length = int(runtime["residual_length"])
    bits = int(runtime["bits"])
    cache = reusable_cache if isinstance(reusable_cache, KIVICache) else None
    effective_tokenized = tokenized
    if cache is None:
        cache = build_kivi_cache(
            runtime["model"].config,
            residual_length=residual_length,
            group_size=int(payload.get("kivi_group_size") or 32),
            k_bits=bits,
            v_bits=bits,
        )
    elif isinstance(cached_prompt_token_ids, list):
        effective_tokenized = _trim_tokenized_inputs(runtime["torch"], tokenized, prefix_len=len(cached_prompt_token_ids))
    result = invoke_generate_decode(
        runtime["model"],
        runtime["tokenizer"],
        runtime["device"],
        payload,
        runtime["torch"],
        past_key_values=cache,
        tokenized_inputs=effective_tokenized,
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
            "cache_reused": bool(isinstance(reusable_cache, KIVICache)),
            "cache_rebuild_reason": rebuild_reason or ("new_request" if not isinstance(reusable_cache, KIVICache) else ""),
            "runtime_cache": result.get("past_key_values", cache),
            "runtime_prompt_token_ids": _token_ids_list(tokenized["input_ids"]),
        }
    )
    quantization_triggered = tracker["kivi_quantized_layers"] > 0 and tracker["kivi_kernel_calls"] > 0
    result.update(
        {
            "kivi_quantization_triggered": quantization_triggered,
            "kivi_effective_mode": "quantized" if quantization_triggered else "unquantized_short_request",
        }
    )
    return result


def _trim_tokenized_inputs(torch: Any, tokenized: dict[str, Any], *, prefix_len: int) -> dict[str, Any]:
    trimmed = dict(tokenized)
    for key in ("input_ids", "attention_mask", "token_type_ids"):
        value = trimmed.get(key)
        if value is None or not hasattr(value, "__getitem__"):
            continue
        sliced = value[:, prefix_len:]
        if hasattr(sliced, "shape") and int(sliced.shape[-1]) == 0:
            sliced = value[:, -1:]
        trimmed[key] = sliced
    return trimmed


def _token_ids_list(input_ids: Any) -> list[int]:
    values = input_ids
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    while isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, (list, tuple)):
        return []
    return [int(token) for token in values]


def split_prefill_kivi_states(
    key_states: Any,
    value_states: Any,
    residual_length: int,
    group_size: int,
    k_bits: int,
    v_bits: int,
    quant_pack: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    key_scale = None
    key_mn = None
    if key_states.shape[-2] % residual_length != 0:
        if key_states.shape[-2] < residual_length:
            key_q = None
            key_full = key_states
        else:
            quant_len = key_states.shape[-2] - (key_states.shape[-2] % residual_length)
            key_q_src = key_states[:, :, :quant_len, :].contiguous()
            key_full = key_states[:, :, quant_len:, :].contiguous()
            key_q, key_scale, key_mn = quant_pack(key_q_src.transpose(2, 3).contiguous(), group_size, k_bits)
    else:
        key_q_src = key_states
        key_full = None
        key_q, key_scale, key_mn = quant_pack(key_q_src.transpose(2, 3).contiguous(), group_size, k_bits)

    if key_states.shape[-2] < residual_length:
        key_scale = None
        key_mn = None

    if value_states.shape[-2] <= residual_length:
        value_q = None
        value_full = value_states
        value_scale = None
        value_mn = None
    else:
        value_q_src = value_states[:, :, :-residual_length, :].contiguous()
        value_full = value_states[:, :, -residual_length:, :].contiguous()
        value_q, value_scale, value_mn = quant_pack(value_q_src, group_size, v_bits)

    return key_q, key_full, key_scale, key_mn, value_q, value_full, value_scale, value_mn


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
                del cache_position, kwargs
                torch = modules["torch"]
                repeat_kv = modules["repeat_kv"]
                cuda_bmm = modules["cuda_bmm_fA_qB_outer"]
                quant_pack = modules["triton_quantize_and_pack_along_last_dim"]
                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                if past_key_value is not None and not isinstance(past_key_value, KIVICache):
                    raise TypeError("Qwen2 KIVI attention requires KIVICache past_key_value")
                layer_state = past_key_value[self.layer_idx] if isinstance(past_key_value, KIVICache) else None
                kv_seq_len = key_states.shape[-2] + (int(layer_state.kv_seq_len) if layer_state is not None else 0)
                query_states, key_states = apply_rope(self, query_states, key_states, value_states, position_ids, position_embeddings, modules)

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
                        assert self.residual_length % self.group_size == 0
                        key_new, scale_new, mn_new = quant_pack(key_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                        self.tracker["kivi_quantized_layers"] += 1
                        self.tracker["kivi_quantized_tokens"] += int(key_full.shape[-2])
                        key_full = None
                        key_q = torch.cat([key_q, key_new], dim=3) if key_q is not None else key_new
                        key_scale = torch.cat([key_scale, scale_new], dim=3) if key_scale is not None else scale_new
                        key_mn = torch.cat([key_mn, mn_new], dim=3) if key_mn is not None else mn_new

                    attn_weights = mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
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
                        assert value_full_len == self.residual_length + 1
                        value_new, scale_new, mn_new = quant_pack(value_full[:, :, :1, :].contiguous(), self.group_size, self.v_bits)
                        self.tracker["kivi_quantize_calls"] += 1
                        value_full = value_full[:, :, 1:, :].contiguous()
                        value_q = torch.cat([value_q, value_new], dim=2) if value_q is not None else value_new
                        value_scale = torch.cat([value_scale, scale_new], dim=2) if value_scale is not None else scale_new
                        value_mn = torch.cat([value_mn, mn_new], dim=2) if value_mn is not None else mn_new
                else:
                    attn_weights = torch.matmul(query_states, repeat_kv(key_states, self.num_key_value_groups).transpose(2, 3)) / math.sqrt(self.head_dim)
                    (
                        key_q,
                        key_full,
                        key_scale,
                        key_mn,
                        value_q,
                        value_full,
                        value_scale,
                        value_mn,
                    ) = split_prefill_kivi_states(
                        key_states,
                        value_states,
                        self.residual_length,
                        self.group_size,
                        self.k_bits,
                        self.v_bits,
                        quant_pack,
                    )
                    if key_q is not None:
                        self.tracker["kivi_quantize_calls"] += 1
                        self.tracker["kivi_quantized_layers"] += 1
                        quant_tokens = key_states.shape[-2] if key_full is None else key_states.shape[-2] - key_full.shape[-2]
                        self.tracker["kivi_quantized_tokens"] += int(quant_tokens)
                    if value_q is not None:
                        self.tracker["kivi_quantize_calls"] += 1
                    attn_weights = mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
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
                    cache = past_key_value
                    if cache is None:
                        cache = KIVICache(
                            self.config.num_hidden_layers,
                            residual_length=self.residual_length,
                            group_size=self.group_size,
                            k_bits=self.k_bits,
                            v_bits=self.v_bits,
                        )
                    past = cache.update_quantized(self.layer_idx, state)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)
                return attn_output, (attn_weights if output_attentions else None), past

        return _Attention()
