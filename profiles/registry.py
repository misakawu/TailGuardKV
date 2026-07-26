from __future__ import annotations

from profiles.base import ProfileAdapter
from profiles.vllm_lru import VLLMLRUAdapter


def build_profile_adapters(
    names: list[str] | None = None,
    runtime_config: dict[str, object] | None = None,
) -> list[ProfileAdapter]:
    registry: dict[str, type[ProfileAdapter]] = {
        "vllm_lru": VLLMLRUAdapter,
    }
    selected = names or list(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"未知 profile adapter: {', '.join(unknown)}")
    return [registry[name](runtime_config) for name in selected]
