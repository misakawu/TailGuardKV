from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiles.cache_common import Cache, init_cache_list, reorder_tensor_like


@dataclass(slots=True)
class KIVILayerState:
    key_q: Any
    key_full: Any
    key_scale: Any
    key_mn: Any
    value_q: Any
    value_full: Any
    value_scale: Any
    value_mn: Any
    kv_seq_len: int


class KIVICache(Cache):
    def __init__(self, num_hidden_layers: int, *, residual_length: int, group_size: int, k_bits: int, v_bits: int) -> None:
        self.residual_length = int(residual_length)
        self.group_size = int(group_size)
        self.k_bits = int(k_bits)
        self.v_bits = int(v_bits)
        layer_count = int(num_hidden_layers)
        self.layers: list[KIVILayerState | None] = init_cache_list(layer_count)
        self.key_cache: list[Any | None] = init_cache_list(layer_count)
        self.value_cache: list[Any | None] = init_cache_list(layer_count)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> KIVILayerState | None:
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.layers[layer_idx]
        return int(state.kv_seq_len) if state is not None else 0

    def get_max_length(self) -> None:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def update(self, key_states: Any, value_states: Any, layer_idx: int, cache_kwargs: dict[str, Any] | None = None) -> tuple[Any, Any]:
        del cache_kwargs
        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        return key_states, value_states

    def update_quantized(self, layer_idx: int, state: KIVILayerState) -> KIVICache:
        self.layers[layer_idx] = state
        self.key_cache[layer_idx] = state.key_full
        self.value_cache[layer_idx] = state.value_full
        return self

    def reorder_cache(self, beam_idx: Any) -> KIVICache:
        for layer_idx, state in enumerate(self.layers):
            self.key_cache[layer_idx] = reorder_tensor_like(self.key_cache[layer_idx], beam_idx)
            self.value_cache[layer_idx] = reorder_tensor_like(self.value_cache[layer_idx], beam_idx)
            if state is None:
                continue
            self.layers[layer_idx] = KIVILayerState(
                key_q=reorder_tensor_like(state.key_q, beam_idx),
                key_full=reorder_tensor_like(state.key_full, beam_idx),
                key_scale=reorder_tensor_like(state.key_scale, beam_idx),
                key_mn=reorder_tensor_like(state.key_mn, beam_idx),
                value_q=reorder_tensor_like(state.value_q, beam_idx),
                value_full=reorder_tensor_like(state.value_full, beam_idx),
                value_scale=reorder_tensor_like(state.value_scale, beam_idx),
                value_mn=reorder_tensor_like(state.value_mn, beam_idx),
                kv_seq_len=state.kv_seq_len,
            )
        return self

    def to_legacy_cache(self) -> tuple[tuple[object, ...], ...]:
        legacy = []
        for state in self.layers:
            if state is None:
                legacy.append((None, None, None, None, None, None, None, None, 0))
                continue
            legacy.append(
                (
                    state.key_q,
                    state.key_full,
                    state.key_scale,
                    state.key_mn,
                    state.value_q,
                    state.value_full,
                    state.value_scale,
                    state.value_mn,
                    state.kv_seq_len,
                )
            )
        return tuple(legacy)


def build_kivi_cache(model_config: Any, *, residual_length: int, group_size: int, k_bits: int, v_bits: int) -> KIVICache:
    num_hidden_layers = int(getattr(model_config, "num_hidden_layers"))
    return KIVICache(
        num_hidden_layers,
        residual_length=residual_length,
        group_size=group_size,
        k_bits=k_bits,
        v_bits=v_bits,
    )
