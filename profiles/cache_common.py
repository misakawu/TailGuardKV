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


DTYPE_SIZES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}


def tensor_nbytes(value: Any) -> int:
    if value is None:
        return 0
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    shape = getattr(value, "shape", None)
    if shape is None:
        return 0
    numel = 1
    for dim in shape:
        numel *= int(dim)
    dtype = str(getattr(value, "dtype", "float32")).replace("torch.", "")
    return numel * DTYPE_SIZES.get(dtype, 4)


def bytes_to_mib(value: int | float) -> float:
    return float(value) / 1024.0 / 1024.0


def tensors_memory_mib(*values: Any) -> float:
    return bytes_to_mib(sum(tensor_nbytes(value) for value in values))


def legacy_kv_cache_memory_mib(past_key_values: Any) -> float:
    total = 0
    if past_key_values is None:
        return 0.0
    for layer in past_key_values:
        if layer is None:
            continue
        if isinstance(layer, (list, tuple)):
            for tensor in layer[:2]:
                total += tensor_nbytes(tensor)
    return bytes_to_mib(total)
