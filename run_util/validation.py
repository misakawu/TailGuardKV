from __future__ import annotations

from pathlib import Path

from run_util.core_types import BackendResult, ProfileMeasurement


REQUIRED_PROFILE_FIELDS = {
    "request_id",
    "profile",
    "adapter",
    "ok",
    "measured",
    "output_text",
    "quality_loss",
    "peak_memory_mib",
    "kv_cache_memory_mib",
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
    require_ttft: bool = False,
    require_quality_loss: bool = True,
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
            if require_quality_loss and measurement.quality_loss is None:
                missing.append("quality_loss")
            if measurement.latency_ms is None:
                missing.append("latency_ms")
            if (
                measurement.extra.get("ttft_semantics") == "first_token"
                and measurement.ttft_ms is None
            ):
                missing.append("ttft_ms")
            if require_ttft:
                if measurement.ttft_ms is None:
                    missing.append("ttft_ms")
                if measurement.extra.get("ttft_semantics") != "first_token":
                    missing.append("ttft_semantics")
            if measurement.peak_memory_mib is None:
                missing.append("peak_memory_mib")
            if measurement.kv_cache_memory_mib is None:
                missing.append("kv_cache_memory_mib")
            if not measurement.extra.get("task"):
                missing.append("task")
            if not measurement.extra.get("length_bucket"):
                missing.append("length_bucket")
            if not measurement.extra.get("split"):
                missing.append("split")
            requires_request_context = bool(
                measurement.session_id
                or any(
                    key in measurement.extra
                    for key in ("arrival_index", "prompt_text", "history_turns", "effective_prompt_chars")
                )
            )
            if requires_request_context:
                if measurement.extra.get("arrival_index") in {None, ""}:
                    missing.append("arrival_index")
                if measurement.extra.get("prompt_text") in {None, ""}:
                    missing.append("prompt_text")
                if measurement.extra.get("history_turns") is None:
                    missing.append("history_turns")
                if measurement.extra.get("effective_prompt_chars") in {None, ""}:
                    missing.append("effective_prompt_chars")
        if missing:
            raise ValueError(
                f"profile 表第 {index} 行字段不完整，缺少 {missing}: "
                f"request={measurement.request_id} profile={measurement.profile} path={path}"
            )
        if require_measured and (not measurement.measured or not measurement.ok):
            raise ValueError(
                "profile 运行失败: "
                f"request={measurement.request_id} profile={measurement.profile} "
                f"measured={measurement.measured} ok={measurement.ok} "
                f"error={measurement.error or ''} path={path}"
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


def validate_backend_results(
    results: list[BackendResult],
    path: Path | str = "<memory>",
    require_memory: bool = True,
    require_ttft: bool = False,
) -> None:
    if not results:
        raise ValueError(f"backend 结果为空: {path}")
    for index, result in enumerate(results, start=1):
        missing: list[str] = []
        if not result.request_id:
            missing.append("request_id")
        if not result.profile:
            missing.append("profile")
        if result.ok and result.measured:
            if result.latency_ms is None:
                missing.append("latency_ms")
            if require_ttft and result.ttft_ms is None:
                missing.append("ttft_ms")
            if require_memory:
                if result.peak_memory_mib is None:
                    missing.append("peak_memory_mib")
                if result.kv_cache_memory_mib is None:
                    missing.append("kv_cache_memory_mib")
        if missing:
            raise ValueError(
                f"backend 结果第 {index} 行字段不完整，缺少 {missing}: "
                f"request={result.request_id} profile={result.profile} path={path}"
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
