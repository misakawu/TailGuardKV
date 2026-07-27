from __future__ import annotations

from core_types import Action, CacheState, DeviceState, Request
from policies.base import Policy


class FullLRUPolicy(Policy):
    def __init__(self) -> None:
        self.name = "full_lru"
        self.placeholder = False
        self.oracle = False

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile="engine_full_lru", reason="full precision vLLM LRU")
