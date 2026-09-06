from __future__ import annotations

import time
from typing import Any


def generate_with_first_token_timing(
    model: Any,
    tokenizer: Any,
    torch: Any,
    inputs: dict[str, Any],
    *,
    request_start: float,
    device: Any,
    max_new_tokens: int,
    past_key_values: Any = None,
    has_cuda: bool = True,
    use_cache: bool = True,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    generate_start = time.perf_counter()
    stage_prefill_ms = 0.0
    ttft_ms: float | None = None
    generated_tokens: list[int] = []
    current_cache = past_key_values

    if max_new_tokens <= 0:
        return {
            "output_text": "",
            "ttft_ms": None,
            "stage_generate_ms": 0.0,
            "stage_prefill_ms": 0.0,
            "stage_first_token_ms": None,
            "past_key_values": current_cache,
        }

    model_inputs = dict(inputs)
    model_inputs["use_cache"] = use_cache
    model_inputs["return_dict"] = True
    if current_cache is not None:
        model_inputs["past_key_values"] = current_cache

    prefill_start = time.perf_counter()
    outputs = model(**model_inputs)
    if has_cuda:
        torch.cuda.synchronize(device)
    stage_prefill_ms = (time.perf_counter() - prefill_start) * 1000

    next_token = _select_next_token(torch, outputs)
    current_cache = getattr(outputs, "past_key_values", current_cache)
    if has_cuda:
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - request_start) * 1000
    generated_tokens.append(_token_to_int(next_token))

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = pad_token_id

    attention_mask = inputs.get("attention_mask")
    prompt_len = (
        int(attention_mask.shape[-1])
        if attention_mask is not None and hasattr(attention_mask, "shape")
        else int(inputs["input_ids"].shape[-1])
    )
    attention_mask = _extend_attention_mask(torch, attention_mask, next_token)
    decode_input_ids = next_token

    while len(generated_tokens) < max_new_tokens and generated_tokens[-1] != eos_token_id:
        cache_position = _cache_position(torch, device, prompt_len + len(generated_tokens) - 1)
        decode_kwargs = _decode_kwargs(
            model,
            decode_input_ids,
            past_key_values=current_cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=use_cache,
        )
        outputs = model(**decode_kwargs)
        next_token = _select_next_token(torch, outputs)
        current_cache = getattr(outputs, "past_key_values", current_cache)
        generated_tokens.append(_token_to_int(next_token))
        attention_mask = _extend_attention_mask(torch, attention_mask, next_token)
        decode_input_ids = next_token

    if has_cuda:
        torch.cuda.synchronize(device)
    stage_generate_ms = (time.perf_counter() - generate_start) * 1000
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    if not output_text:
        output_text = " ".join(str(token) for token in generated_tokens)
    return {
        "output_text": output_text,
        "ttft_ms": ttft_ms,
        "stage_generate_ms": stage_generate_ms,
        "stage_prefill_ms": stage_prefill_ms,
        "stage_first_token_ms": ttft_ms,
        "past_key_values": current_cache,
    }


def _select_next_token(torch: Any, outputs: Any) -> Any:
    logits = outputs.logits[:, -1, :]
    if hasattr(torch, "argmax"):
        try:
            return torch.argmax(logits, dim=-1, keepdim=True)
        except TypeError:
            return torch.argmax(logits, dim=-1)
    return logits.argmax(dim=-1, keepdim=True)


def _decode_kwargs(
    model: Any,
    input_ids: Any,
    *,
    past_key_values: Any,
    attention_mask: Any,
    cache_position: Any,
    use_cache: bool,
) -> dict[str, Any]:
    if hasattr(model, "prepare_inputs_for_generation"):
        kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        prepared = model.prepare_inputs_for_generation(input_ids, **kwargs)
        prepared["return_dict"] = True
        return prepared
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "return_dict": True,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    return kwargs


def _cache_position(torch: Any, device: Any, position: int) -> Any:
    if not hasattr(torch, "arange"):
        return None
    try:
        return torch.arange(position, position + 1, device=device)
    except Exception:
        return None


def _extend_attention_mask(torch: Any, attention_mask: Any, next_token: Any) -> Any:
    if attention_mask is None or not hasattr(torch, "cat") or not hasattr(torch, "ones_like"):
        return attention_mask
    try:
        return torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    except Exception:
        return attention_mask


def _token_to_int(token: Any) -> int:
    if hasattr(token, "detach"):
        token = token.detach()
    if hasattr(token, "cpu"):
        token = token.cpu()
    if hasattr(token, "item"):
        return int(token.item())
    if hasattr(token, "tolist"):
        token = token.tolist()
    while isinstance(token, (list, tuple)):
        token = token[0]
    return int(token)
