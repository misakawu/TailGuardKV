from __future__ import annotations

import atexit
import json
import os
import time
from pathlib import Path
from typing import Any

from vllm_lru_policy import AccessInfo, AdmitInfo, EvictInfo, create_vllm_policy


_POLICY_NAME = os.environ.get("TAILGUARDKV_VLLM_POLICY", "").strip()
_ENABLED = _POLICY_NAME == "full_lru"
_POLICY = create_vllm_policy(_POLICY_NAME) if _ENABLED else None
_PATCHED = False
_STATS: dict[str, float] = {
    "cache_hits": 0,
    "cache_misses": 0,
    "evictions": 0,
    "cached_blocks": 0,
    "policy_time_ms": 0.0,
    "eviction_decision_time_ms": 0.0,
}


try:
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
except Exception:
    pass


def reset_tailguardkv_vllm_stats() -> None:
    for key in list(_STATS):
        _STATS[key] = 0.0


def get_tailguardkv_vllm_stats() -> dict[str, float]:
    return dict(_STATS)


def _write_stats_at_exit() -> None:
    stats_dir = os.environ.get("TAILGUARDKV_VLLM_STATS_DIR", "").strip()
    if not stats_dir:
        return
    path = Path(stats_dir) / "tailguardkv_vllm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(get_tailguardkv_vllm_stats(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _block_id(block: Any) -> int:
    for attr in ("block_id", "block_id_or_none"):
        value = getattr(block, attr, None)
        if value is not None:
            return int(value)
    return int(block)


def _patch_block_pool() -> bool:
    global _PATCHED
    if not _ENABLED or _POLICY is None or _PATCHED:
        return _PATCHED
    try:
        from vllm.v1.core.block_pool import BlockPool
    except Exception:
        return False

    original_get_cached_block = getattr(BlockPool, "get_cached_block", None)
    original_cache_full_blocks = getattr(BlockPool, "cache_full_blocks", None)
    original_evict_cached_block = getattr(BlockPool, "evict_cached_block", None)

    if original_get_cached_block is not None and not getattr(original_get_cached_block, "_tailguardkv_patch", False):

        def get_cached_block(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            block = original_get_cached_block(self, *args, **kwargs)
            if block is None:
                _STATS["cache_misses"] += 1
            else:
                _STATS["cache_hits"] += 1
                _POLICY.on_access(self, _block_id(block), AccessInfo())
            _STATS["policy_time_ms"] += (time.perf_counter() - started) * 1000
            return block

        get_cached_block._tailguardkv_patch = True
        BlockPool.get_cached_block = get_cached_block

    if original_cache_full_blocks is not None and not getattr(original_cache_full_blocks, "_tailguardkv_patch", False):

        def cache_full_blocks(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            before = _cached_block_count(self)
            result = original_cache_full_blocks(self, *args, **kwargs)
            after = _cached_block_count(self)
            _STATS["cached_blocks"] = after
            for block_id in _new_cached_block_ids(self, before, after):
                _POLICY.on_admit(self, block_id, AdmitInfo())
            _STATS["policy_time_ms"] += (time.perf_counter() - started) * 1000
            return result

        cache_full_blocks._tailguardkv_patch = True
        BlockPool.cache_full_blocks = cache_full_blocks

    if original_evict_cached_block is not None and not getattr(original_evict_cached_block, "_tailguardkv_patch", False):

        def evict_cached_block(self: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            block_id = _choose_lru_block_id(self)
            _STATS["eviction_decision_time_ms"] += (time.perf_counter() - started) * 1000
            result = original_evict_cached_block(self, *args, **kwargs)
            if block_id is not None:
                _POLICY.on_evict(self, block_id, EvictInfo())
            _STATS["evictions"] += 1
            _STATS["cached_blocks"] = _cached_block_count(self)
            return result

        evict_cached_block._tailguardkv_patch = True
        BlockPool.evict_cached_block = evict_cached_block

    _PATCHED = True
    return True


def _cached_block_count(pool: Any) -> int:
    for attr in ("cached_block_hash_to_block", "block_hash_to_block"):
        value = getattr(pool, attr, None)
        if hasattr(value, "__len__"):
            return len(value)
    return int(_STATS.get("cached_blocks", 0.0))


def _new_cached_block_ids(pool: Any, before: int, after: int) -> list[int]:
    if after <= before:
        return []
    blocks = getattr(pool, "cached_block_hash_to_block", None) or getattr(pool, "block_hash_to_block", None)
    if not isinstance(blocks, dict):
        return []
    return [_block_id(block) for block in list(blocks.values())[-(after - before) :]]


def _choose_lru_block_id(pool: Any) -> int | None:
    blocks = getattr(pool, "cached_block_hash_to_block", None) or getattr(pool, "block_hash_to_block", None)
    if not isinstance(blocks, dict) or not blocks:
        return None
    block_ids = [_block_id(block) for block in blocks.values()]
    return min(block_ids, key=lambda block_id: _POLICY.rank_tuple(pool, block_id))


patch_tailguardkv_vllm = _patch_block_pool
_patch_block_pool()
atexit.register(_write_stats_at_exit)
