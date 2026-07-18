from __future__ import annotations

from policies.base import StaticProfilePolicy


class FullLRUPolicy(StaticProfilePolicy):
    def __init__(self) -> None:
        super().__init__("full_gpu", name="full_lru")
