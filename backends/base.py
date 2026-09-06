from __future__ import annotations

from abc import ABC, abstractmethod

from run_util.core_types import Action, BackendResult, CacheState, Request


class Backend(ABC):
    """统一 backend 接口，真实 vLLM/LMCache 和 measured-replay 都走这里。"""

    name: str

    def execute(self, request: Request, action: Action, cache_state: CacheState) -> BackendResult:
        """Execute one policy action; legacy batch backends may use ``run``."""

        del cache_state
        return self.run([request], [action.profile])[0]

    @abstractmethod
    def run(self, requests: list[Request], profiles: list[str]) -> list[BackendResult]:
        ...
