from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from transformers.cache_utils import Cache
except ModuleNotFoundError:
    class Cache:
        pass


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
        self.layers: list[KIVILayerState | None] = [None] * int(num_hidden_layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> KIVILayerState | None:
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        state = self.layers[layer_idx]
        return int(state.kv_seq_len) if state is not None else 0

    def update(self, layer_idx: int, state: KIVILayerState, *args: Any, **kwargs: Any) -> KIVICache:
        self.layers[layer_idx] = state
        return self

    def reorder_cache(self, beam_idx: Any) -> KIVICache:
        for layer_idx, state in enumerate(self.layers):
            if state is None:
                continue
            self.layers[layer_idx] = KIVILayerState(
                key_q=_reorder_tensor_like(state.key_q, beam_idx),
                key_full=_reorder_tensor_like(state.key_full, beam_idx),
                key_scale=_reorder_tensor_like(state.key_scale, beam_idx),
                key_mn=_reorder_tensor_like(state.key_mn, beam_idx),
                value_q=_reorder_tensor_like(state.value_q, beam_idx),
                value_full=_reorder_tensor_like(state.value_full, beam_idx),
                value_scale=_reorder_tensor_like(state.value_scale, beam_idx),
                value_mn=_reorder_tensor_like(state.value_mn, beam_idx),
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


def _reorder_tensor_like(value: Any, beam_idx: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "index_select"):
        return value.index_select(0, beam_idx)
    return value
