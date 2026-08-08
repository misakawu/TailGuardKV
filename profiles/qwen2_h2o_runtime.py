from __future__ import annotations

import math
import time
from typing import Any, Callable

from profiles.h2o_cache import H2OCache, H2OLayerState


def h2o_sizes(tokenizer: Any, payload: dict[str, Any]) -> dict[str, int]:
    prompt_tokens = int(tokenizer(str(payload.get("prompt") or ""), return_tensors="pt")["input_ids"].shape[-1])
    heavy_ratio = float(payload.get("h2o_heavy_ratio") or 0.1)
    recent_ratio = float(payload.get("h2o_recent_ratio") or 0.1)
    heavy_size = max(1, int(prompt_tokens * heavy_ratio))
    recent_size = max(1, int(prompt_tokens * recent_ratio))
    return {"prompt_tokens": prompt_tokens, "heavy_size": heavy_size, "recent_size": recent_size}


def reset_h2o_attention(model: Any, tracker: dict[str, int], heavy_size: int, recent_size: int) -> None:
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


def prepare_h2o_runtime(
    payload: dict[str, Any],
    *,
    worker_start: float,
    import_runtime_modules: Callable[..., dict[str, Any]],
    require_cuda: Callable[[Any], None],
    load_qwen2_model: Callable[[dict[str, Any], Any, Any, Any], tuple[Any, Any, Any]],
    install_qwen2_attention: Callable[[Any, type, dict[str, int], int, dict[str, Any], dict[str, Any]], None],
    attention_cls: type,
) -> dict[str, Any]:
    modules = import_runtime_modules(use_kivi=False)
    torch = modules["torch"]
    require_cuda(torch)

    startup_ms = (time.perf_counter() - worker_start) * 1000
    load_start = time.perf_counter()
    model, tokenizer, device = load_qwen2_model(payload, torch, modules["AutoModelForCausalLM"], modules["AutoTokenizer"])
    model_load_ms = (time.perf_counter() - load_start) * 1000
    tracker = {
        "h2o_prune_events": 0,
        "h2o_mask_events": 0,
        "h2o_cache_budget": 0,
        "h2o_kept_tokens": 0,
        "h2o_prompt_tokens": 0,
    }
    initial_sizes = h2o_sizes(tokenizer, payload)
    install_qwen2_attention(
        model,
        attention_cls,
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


def run_h2o_request(
    runtime: dict[str, Any],
    payload: dict[str, Any],
    *,
    worker_mode: str,
    build_h2o_cache: Callable[..., H2OCache],
    invoke_generate_decode: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    tracker = runtime["tracker"]
    sizes = h2o_sizes(runtime["tokenizer"], payload)
    tracker.update(
        {
            "h2o_prune_events": 0,
            "h2o_mask_events": 0,
            "h2o_cache_budget": sizes["heavy_size"] + sizes["recent_size"],
            "h2o_kept_tokens": 0,
            "h2o_prompt_tokens": sizes["prompt_tokens"],
        }
    )
    reset_h2o_attention(runtime["model"], tracker, sizes["heavy_size"], sizes["recent_size"])
    cache = build_h2o_cache(
        runtime["model"].config,
        heavy_size=sizes["heavy_size"],
        recent_size=sizes["recent_size"],
    )
    result = invoke_generate_decode(
        runtime["model"],
        runtime["tokenizer"],
        runtime["device"],
        payload,
        runtime["torch"],
        past_key_values=cache,
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
    if result.get("ok") and tracker["h2o_prune_events"] <= 0:
        result["ok"] = False
        result["measured"] = False
        result["error"] = (
            "H2O proof missing: no prune event ran. "
            f"prompt_tokens={sizes['prompt_tokens']} budget={tracker['h2o_cache_budget']} "
            f"prune_events={tracker['h2o_prune_events']}"
        )
        result["failure_stage"] = "generate"
    return result


class Qwen2H2OAttention:
    def __new__(cls, source: Any, config: Any, layer_idx: int, tracker: dict[str, int], bits: int, payload: dict[str, Any], modules: dict[str, Any]) -> Any:
        del bits
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
                del cache_position, kwargs
                torch = modules["torch"]
                repeat_kv = modules["repeat_kv"]
                from profiles.qwen2_runtime_layout import apply_rope, mask_softmax

                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                if past_key_value is not None and not isinstance(past_key_value, H2OCache):
                    raise TypeError("Qwen2 H2O attention requires H2OCache past_key_value")
                layer_state = past_key_value[self.layer_idx] if isinstance(past_key_value, H2OCache) else None
                logical_seq_len = key_states.shape[-2] + (int(layer_state.logical_seq_len) if layer_state is not None else 0)
                query_states, key_states = apply_rope(self, query_states, key_states, value_states, position_ids, position_embeddings, modules)
                if layer_state is not None:
                    key_states = torch.cat([layer_state.key_states, key_states], dim=2)
                    value_states = torch.cat([layer_state.value_states, value_states], dim=2)
                    self.hh_score = layer_state.hh_score
                kv_seq_len = key_states.shape[-2]
                key_for_attn = repeat_kv(key_states, self.num_key_value_groups)
                value_for_attn = repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_for_attn.transpose(2, 3)) / math.sqrt(self.head_dim)
                attn_weights = mask_softmax(attn_weights, attention_mask, bsz, self.num_heads, q_len, kv_seq_len, torch, nn)
                attn_output = torch.matmul(attn_weights, value_for_attn)
                past = None
                if use_cache:
                    cache = past_key_value
                    if cache is None:
                        cache = H2OCache(
                            self.config.num_hidden_layers,
                            heavy_size=self.hh_size,
                            recent_size=self.recent_size,
                        )
                    key_states, value_states, hh_score = self._prune(key_states, value_states, attn_weights.detach())
                    past = cache.update_pruned(
                        self.layer_idx,
                        H2OLayerState(
                            key_states=key_states,
                            value_states=value_states,
                            hh_score=hh_score,
                            logical_seq_len=int(logical_seq_len),
                        ),
                    )
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                return self.o_proj(attn_output), (attn_weights if output_attentions else None), past

            def _prune(self, key_states: Any, value_states: Any, attn_weights: Any) -> tuple[Any, Any, Any]:
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
                seq_len = key_states.shape[2]
                if seq_len <= self.cache_budget:
                    self.tracker["h2o_kept_tokens"] = max(self.tracker["h2o_kept_tokens"], int(seq_len))
                    return key_states, value_states, self.hh_score
                select_len = max(1, seq_len - self.recent_size)
                hh_size = min(self.hh_size, select_len)
                _, keep_topk = torch.topk(self.hh_score[:, :select_len], hh_size, dim=-1)
                keep_topk = keep_topk.sort().values
                recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(self.num_key_value_heads, 1)
                keep_idx = torch.cat([keep_topk, recent], dim=-1)
                pruned_k = []
                pruned_v = []
                for batch_idx in range(key_states.shape[0]):
                    head_k = []
                    head_v = []
                    for head_idx in range(self.num_key_value_heads):
                        idx = keep_idx[head_idx]
                        head_k.append(key_states[batch_idx, head_idx].index_select(0, idx))
                        head_v.append(value_states[batch_idx, head_idx].index_select(0, idx))
                    pruned_k.append(torch.stack(head_k, dim=0))
                    pruned_v.append(torch.stack(head_v, dim=0))
                mask = torch.zeros_like(self.hh_score, dtype=torch.bool)
                mask.scatter_(1, keep_idx, True)
                self.hh_score = self.hh_score[mask].view(self.num_key_value_heads, -1)
                self.tracker["h2o_prune_events"] += 1
                self.tracker["h2o_mask_events"] += 1
                self.tracker["h2o_kept_tokens"] = int(keep_idx.shape[-1])
                return torch.stack(pruned_k, dim=0).contiguous(), torch.stack(pruned_v, dim=0).contiguous(), self.hh_score

        return _Attention()
