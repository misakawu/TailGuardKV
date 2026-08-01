from __future__ import annotations

from typing import Any


def qwen2_layout_error(model: Any) -> str | None:
    config = getattr(model, "config", None)
    if config is None:
        return "missing model.config"
    model_type = str(getattr(config, "model_type", "") or "")
    if model_type != "qwen2":
        return f"model_type={model_type or '<empty>'}"
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return "missing model.model.layers"
    sample_attn = getattr(layers[0], "self_attn", None)
    required = ("q_proj", "k_proj", "v_proj", "o_proj")
    missing = [name for name in required if getattr(sample_attn, name, None) is None]
    if missing:
        return f"missing attention fields {missing}"
    num_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    num_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    if num_heads <= 0 or num_kv_heads <= 0:
        return f"invalid heads num_attention_heads={num_heads} num_key_value_heads={num_kv_heads}"
    if num_heads % num_kv_heads != 0:
        return f"unsupported head grouping num_attention_heads={num_heads} num_key_value_heads={num_kv_heads}"
    return None


def iterate_qwen2_layers(model: Any) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("loaded model does not expose model.layers; expected Qwen2ForCausalLM")
    return layers


def install_qwen2_attention(model: Any, wrapper_cls: type, tracker: dict[str, int], bits: int, payload: dict[str, Any], modules: dict[str, Any]) -> None:
    # TODO: Extract a generic model adapter so non-Qwen2 attention installers can share this runtime boundary.
    for layer_idx, layer in enumerate(iterate_qwen2_layers(model)):
        layer.self_attn = wrapper_cls(layer.self_attn, model.config, layer_idx, tracker, bits, payload, modules)
    model.config.use_cache = True


def apply_rope(attn: Any, query_states: Any, key_states: Any, value_states: Any, position_ids: Any, position_embeddings: Any, modules: dict[str, Any]) -> tuple[Any, Any]:
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


def mask_softmax(attn_weights: Any, attention_mask: Any, bsz: int, num_heads: int, q_len: int, kv_seq_len: int, torch: Any, nn: Any) -> Any:
    if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
        raise ValueError(f"attention weights size mismatch: got {tuple(attn_weights.size())}, expected {(bsz, num_heads, q_len, kv_seq_len)}")
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
    return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attn_weights.dtype)
