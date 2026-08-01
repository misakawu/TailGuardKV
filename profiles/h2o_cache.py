from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from profiles.cache_common import Cache, init_cache_list, reorder_tensor_like


@dataclass(slots=True)
class H2OLayerState:
    key_states: Any
    value_states: Any
    hh_score: Any
    logical_seq_len: int


class H2OCache(Cache):
    def __init__(self, num_hidden_layers: int, *, heavy_size: int, recent_size: int) -> None:
        self.heavy_size = int(heavy_size)
        self.recent_size = int(recent_size)
        self.cache_budget = self.heavy_size + self.recent_size
        layer_count = int(num_hidden_layers)
        self.layers: list[H2OLayerState | None] = init_cache_list(layer_count)
        self.key_cache: list[Any | None] = init_cache_list(layer_count)
        self.value_cache: list[Any | None] = init_cache_list(layer_count)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> H2OLayerState | None:
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.layers[layer_idx]
        return int(state.logical_seq_len) if state is not None else 0

    def get_physical_seq_length(self, layer_idx: int = 0) -> int:
        state = self.layers[layer_idx]
        if state is None:
            return 0
        return int(state.key_states.shape[-2])

    def h2o_kept_tokens(self, layer_idx: int = 0) -> int:
        return self.get_physical_seq_length(layer_idx)

    def get_max_length(self) -> None:
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def update(self, key_states: Any, value_states: Any, layer_idx: int, cache_kwargs: dict[str, Any] | None = None) -> tuple[Any, Any]:
        del cache_kwargs
        logical_seq_len = int(key_states.shape[-2])
        self.update_pruned(
            layer_idx,
            H2OLayerState(
                key_states=key_states,
                value_states=value_states,
                hh_score=None,
                logical_seq_len=logical_seq_len,
            ),
        )
        return key_states, value_states

    def update_pruned(self, layer_idx: int, state: H2OLayerState) -> H2OCache:
        self.layers[layer_idx] = state
        self.key_cache[layer_idx] = state.key_states
        self.value_cache[layer_idx] = state.value_states
        return self

    def reorder_cache(self, beam_idx: Any) -> H2OCache:
        for layer_idx, state in enumerate(self.layers):
            self.key_cache[layer_idx] = reorder_tensor_like(self.key_cache[layer_idx], beam_idx)
            self.value_cache[layer_idx] = reorder_tensor_like(self.value_cache[layer_idx], beam_idx)
            if state is None:
                continue
            self.layers[layer_idx] = H2OLayerState(
                key_states=reorder_tensor_like(state.key_states, beam_idx),
                value_states=reorder_tensor_like(state.value_states, beam_idx),
                hh_score=reorder_tensor_like(state.hh_score, beam_idx),
                logical_seq_len=state.logical_seq_len,
            )
        return self

    def to_legacy_cache(self) -> tuple[tuple[object, ...], ...]:
        legacy = []
        for state in self.layers:
            if state is None:
                legacy.append((None, None, None, 0))
                continue
            legacy.append((state.key_states, state.value_states, state.hh_score, state.logical_seq_len))
        return tuple(legacy)


def build_h2o_cache(model_config: Any, *, heavy_size: int, recent_size: int) -> H2OCache:
    num_hidden_layers = int(getattr(model_config, "num_hidden_layers"))
    return H2OCache(num_hidden_layers, heavy_size=heavy_size, recent_size=recent_size)
