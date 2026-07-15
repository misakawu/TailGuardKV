from __future__ import annotations

from abc import ABC, abstractmethod

from core_types import Action, CacheState, DeviceState, Request


class Policy(ABC):
    """所有 baseline 和 TailGuardKV 策略必须实现同一个决策接口。"""

    name: str

    @abstractmethod
    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        ...


class StaticProfilePolicy(Policy):
    """先保留最小静态 profile baseline，后续扩展 static_best/static_safe。"""

    def __init__(self, profile: str, name: str | None = None) -> None:
        self.profile = profile
        self.name = name or f"static_{profile}"

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        return Action(profile=self.profile, reason="静态 profile baseline")
