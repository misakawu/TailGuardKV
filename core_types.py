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

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ProfileMeasurement":
        extra: dict[str, Any] = {}
        payload = dict(row)
        for key in list(payload):
            if key.startswith("extra_"):
                extra[key.removeprefix("extra_")] = payload.pop(key)

        return cls(
            request_id=str(payload.get("request_id", "")),
            profile=str(payload.get("profile", "")),
            adapter=str(payload.get("adapter", "")),
            ok=_parse_bool(payload.get("ok")),
            measured=_parse_bool(payload.get("measured")),
            output_text=str(payload.get("output_text") or ""),
            error=_parse_optional_str(payload.get("error")),
            latency_ms=_parse_optional_float(payload.get("latency_ms")),
            ttft_ms=_parse_optional_float(payload.get("ttft_ms")),
            peak_memory_mib=_parse_optional_float(payload.get("peak_memory_mib")),
            resident_memory_mib=_parse_optional_float(payload.get("resident_memory_mib")),
            quality_score=_parse_optional_float(payload.get("quality_score")),
            quality_loss=_parse_optional_float(payload.get("quality_loss")),
            extra=extra,
        )


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


@dataclass(frozen=True)
class PolicyRunRecord:
    policy: str
    request_id: str
    action_profile: str
    ok: bool
    measured: bool
    placeholder: bool = False
    reason: str = ""
    error: str | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    peak_memory_mib: float | None = None
    resident_memory_mib: float | None = None
    quality_loss: float | None = None
    exact: bool = False
    oracle: bool = False

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
