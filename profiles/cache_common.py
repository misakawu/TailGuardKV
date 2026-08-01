from __future__ import annotations

from typing import Any

try:
    from transformers.cache_utils import Cache
except ModuleNotFoundError:
    class Cache:
        pass


def init_cache_list(layer_count: int) -> list[Any | None]:
    return [None] * int(layer_count)


def reorder_tensor_like(value: Any, beam_idx: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if hasattr(value, "index_select"):
        return value.index_select(0, beam_idx)
    return value
