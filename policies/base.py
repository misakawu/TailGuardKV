from __future__ import annotations

from abc import ABC, abstractmethod

from core_types import Action, CacheState, DeviceState, Request


class Policy(ABC):
    name: str
    placeholder: bool = False
    oracle: bool = False

    @abstractmethod
    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        pass


class StaticProfilePolicy(Policy):
    def __init__(self, profile: str, name: str | None = None, placeholder: bool = False) -> None:
        self.profile = profile
        self.name = name or f"static_{profile}"
        self.placeholder = placeholder
        self.oracle = False

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="static profile baseline")
