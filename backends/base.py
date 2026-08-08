from __future__ import annotations

from abc import ABC, abstractmethod

from run_util.core_types import BackendResult, Request


class Backend(ABC):
    """统一 backend 接口，真实 vLLM/LMCache 和 measured-replay 都走这里。"""

    name: str

    @abstractmethod
    def run(self, requests: list[Request], profiles: list[str]) -> list[BackendResult]:
        ...
