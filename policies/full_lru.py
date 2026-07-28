from __future__ import annotations

from core_types import Action
from policies.base import StaticProfilePolicy


class FullLRUPolicy(StaticProfilePolicy):
    def __init__(self) -> None:
        super().__init__("full_gpu", name="full_lru")

    def decide(self, request, cache_state, device_state) -> Action:
        return Action(profile=self.profile, reason="full precision exact profile")
