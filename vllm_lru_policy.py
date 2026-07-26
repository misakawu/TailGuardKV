from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdmitInfo:
    score: float = 0.0
    p_reuse: float = 0.5
    pinned: bool = False
    profile: dict[str, Any] | None = None


@dataclass(frozen=True)
class AccessInfo:
    profile: dict[str, Any] | None = None
    refreshed: bool = False


@dataclass(frozen=True)
class EvictInfo:
    profile: dict[str, Any] | None = None
    score: float = 0.0
    p_reuse: float = 0.0


class VLLMLRUBlockPolicy:
    name = "full_lru"
    needs_rank_state = True

    def __init__(self) -> None:
        self._clock = 0
        self._recency: dict[int, int] = {}

    def rank_tuple(self, pool: Any, block_id: int) -> tuple[int, int]:
        recency = self._pool_recency(pool).get(int(block_id), self._recency.get(int(block_id), 0))
        return (int(recency), int(block_id))

    def on_admit(self, pool: Any, block_id: int, info: AdmitInfo | None = None) -> None:
        self._touch(pool, block_id)

    def on_access(self, pool: Any, block_id: int, info: AccessInfo | None = None) -> None:
        self._touch(pool, block_id)

    def refresh_block_score(self, pool: Any, block_id: int, profile: dict[str, Any]) -> None:
        self._touch(pool, block_id)

    def on_evict(self, pool: Any, block_id: int, info: EvictInfo | None = None) -> None:
        block_id = int(block_id)
        self._recency.pop(block_id, None)
        self._pool_recency(pool).pop(block_id, None)

    def _touch(self, pool: Any, block_id: int) -> None:
        block_id = int(block_id)
        self._clock += 1
        self._recency[block_id] = self._clock
        self._pool_recency(pool)[block_id] = self._clock

    def _pool_recency(self, pool: Any) -> dict[int, int]:
        recency = getattr(pool, "_tailguardkv_lru_recency", None)
        if recency is None:
            recency = {}
            try:
                setattr(pool, "_tailguardkv_lru_recency", recency)
            except Exception:
                return self._recency
        return recency


def create_vllm_policy(name: str | None = None) -> VLLMLRUBlockPolicy:
    policy_name = (name or "full_lru").strip()
    if policy_name != "full_lru":
        raise ValueError(f"unknown TailGuardKV vLLM policy: {policy_name}")
    return VLLMLRUBlockPolicy()
