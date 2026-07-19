from __future__ import annotations

from pathlib import Path

from core_types import ProfileMeasurement


REQUIRED_PROFILE_FIELDS = {
    "request_id",
    "profile",
    "adapter",
    "ok",
    "measured",
    "output_text",
    "quality_loss",
    "ttft_ms",
    "peak_memory_mib",
    "task",
    "length_bucket",
    "split",
}


def validate_profile_table_header(fieldnames: list[str], path: Path | str) -> None:
    missing = sorted(REQUIRED_PROFILE_FIELDS.difference(fieldnames))
    if missing:
        raise ValueError(f"profile 表缺少正式字段 {missing}: {path}")


def validate_profile_measurements(
    measurements: list[ProfileMeasurement],
    path: Path | str = "<memory>",
    required_profiles: list[str] | None = None,
    require_measured: bool = False,
) -> None:
    if not measurements:
        raise ValueError(f"profile 表为空: {path}")
    for index, measurement in enumerate(measurements, start=1):
        missing: list[str] = []
        if not measurement.request_id:
            missing.append("request_id")
        if not measurement.profile:
            missing.append("profile")
        if not measurement.adapter:
            missing.append("adapter")
        if measurement.ok and measurement.measured:
            if not measurement.output_text:
                missing.append("output_text")
            if measurement.quality_loss is None:
                missing.append("quality_loss")
            if measurement.ttft_ms is None:
                missing.append("ttft_ms")
            if measurement.peak_memory_mib is None:
                missing.append("peak_memory_mib")
            if not measurement.extra.get("task"):
                missing.append("task")
            if not measurement.extra.get("length_bucket"):
                missing.append("length_bucket")
            if not measurement.extra.get("split"):
                missing.append("split")
        if require_measured and not measurement.measured:
            missing.append("measured=True")
        if require_measured and not measurement.ok:
            missing.append("ok=True")
        if missing:
            raise ValueError(
                f"profile 表第 {index} 行字段不完整，缺少 {missing}: "
                f"request={measurement.request_id} profile={measurement.profile} path={path}"
            )
    if required_profiles:
        expected = set(required_profiles)
        by_request: dict[str, set[str]] = {}
        for measurement in measurements:
            by_request.setdefault(measurement.request_id, set()).add(measurement.profile)
        for request_id, seen_profiles in sorted(by_request.items()):
            missing_profiles = sorted(expected.difference(seen_profiles))
            if missing_profiles:
                raise ValueError(
                    f"profile 表 request={request_id} 缺少必需 profile {missing_profiles}: path={path}"
                )


def failed_measurement_summary(measurements: list[ProfileMeasurement]) -> list[dict[str, object]]:
    failures: list[dict[str, object]] = []
    for measurement in measurements:
        if measurement.ok and measurement.measured:
            continue
        failures.append(
            {
                "request_id": measurement.request_id,
                "profile": measurement.profile,
                "adapter": measurement.adapter,
                "env": measurement.extra.get("env") or measurement.extra.get("backend") or "",
                "ok": measurement.ok,
                "measured": measurement.measured,
                "error": measurement.error or measurement.extra.get("unsupported") or "",
            }
        )
    return failures
