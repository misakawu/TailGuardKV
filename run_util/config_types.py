from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class KIVIPayload(TypedDict, total=False):
    profile: str
    model_name: str
    prompt: str
    max_new_tokens: int
    cache_dir: str | None
    local_files_only: bool
    bits: int
    kivi_group_size: int
    kivi_residual_length: int


class H2OPayload(TypedDict, total=False):
    profile: str
    model_name: str
    prompt: str
    max_new_tokens: int
    cache_dir: str | None
    local_files_only: bool
    h2o_heavy_ratio: float
    h2o_recent_ratio: float
    h2o_heavy_size: int
    h2o_recent_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    profiles: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    pilot: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    tailguard: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "ExperimentConfig":
        def section(name: str) -> dict[str, Any]:
            value = config.get(name, {})
            return value if isinstance(value, dict) else {}

        return cls(
            profiles=section("profiles"),
            policies=section("policies"),
            pilot=section("pilot"),
            model=section("model"),
            data=section("data"),
            outputs=section("outputs"),
            tailguard=section("tailguard"),
        )
