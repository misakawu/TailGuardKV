from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Request:
    request_id: str
    task: str
    prompt: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_chars(self) -> int:
        return len(self.prompt)


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    family: str
    env: str
    lossy: bool
    exact: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileMeasurement:
    request_id: str
    profile: str
    adapter: str
    ok: bool
    measured: bool
    output_text: str = ""
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    resident_memory_mib: float | None = None
    quality_score: float | None = None
    quality_loss: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        extra = row.pop("extra")
        for key, value in extra.items():
            row[f"extra_{key}"] = value
        return row


@dataclass(frozen=True)
class SmokeResult:
    adapter: str
    env: str
    ok: bool
    profiles: tuple[str, ...]
    detail: str = ""
    error: str | None = None
    versions: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["profiles"] = ",".join(self.profiles)
        versions = row.pop("versions")
        for key, value in versions.items():
            row[f"version_{key}"] = value
        return row


@dataclass(frozen=True)
class Action:
    profile: str
    reason: str = ""


@dataclass(frozen=True)
class CacheState:
    resident_memory_mib: float = 0.0
    objects: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeviceState:
    gpu_free_mib: float | None = None
    gpu_total_mib: float | None = None
    concurrency: int = 1
