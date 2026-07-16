from __future__ import annotations

from profiles.base import ProfileAdapter
from profiles.full import FullKVAdapter
from profiles.h2o import H2OAdapter
from profiles.kivi import KIVIAdapter


def build_profile_adapters(
    names: list[str] | None = None,
    runtime_config: dict[str, object] | None = None,
) -> list[ProfileAdapter]:
    registry: dict[str, type[ProfileAdapter]] = {
        "full": FullKVAdapter,
        "kivi": KIVIAdapter,
        "h2o": H2OAdapter,
    }
    selected = names or list(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"未知 profile adapter: {', '.join(unknown)}")
    return [registry[name](runtime_config) for name in selected]
