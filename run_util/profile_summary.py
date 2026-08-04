from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_util.config_loader import exact_profiles
from run_util.core_types import ProfileMeasurement
from run_util.experiment_common import json_ready


PROFILE_SUMMARY_COLUMNS = [
    "section",
    "name",
    "ok",
    "error",
    "diagnostic_output",
    "failures",
    "config",
    "profile_rows",
    "dry_run",
    "epsilon_values",
    "delta_values",
    "family",
    "exact",
    "count",
    "ok_count",
    "measured_count",
    "task_count",
    "length_bucket_count",
    "split_count",
    "mean_quality_loss",
    "p50_quality_loss",
    "p95_quality_loss",
    "p99_quality_loss",
    "cvar_quality_loss",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_peak_memory_mib",
    "p95_peak_memory_mib",
    "p99_peak_memory_mib",
    "mean_resident_memory_mib",
    "p95_resident_memory_mib",
    "mean_kv_cache_memory_mib",
    "p95_kv_cache_memory_mib",
    "p99_kv_cache_memory_mib",
    "backend_distribution",
    "ttft_semantics_distribution",
    "worker_mode_distribution",
    "primary_metric_distribution",
]


def profile_summary_rows(
    payload: dict[str, Any],
    measurements: list[ProfileMeasurement] | None,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    config = config if isinstance(config, dict) else {}
    measurements = measurements or []
    epsilons = _number_list(_pilot(config).get("epsilons"), default=0.2)
    deltas = _number_list(_pilot(config).get("deltas"), default=0.05)
    rows = [_experiment_row(payload, measurements, epsilons, deltas)]

    configured_profiles = _configured_profiles(config)
    exact = exact_profiles(configured_profiles, config) if configured_profiles else set()
    grouped: dict[str, list[ProfileMeasurement]] = defaultdict(list)
    for measurement in measurements:
        grouped[measurement.profile].append(measurement)

    for profile in sorted(grouped):
        rows.append(_profile_row(profile, grouped[profile], payload, config, exact, epsilons, deltas))
    return rows


def write_profile_summary(
    payload: dict[str, Any],
    path: str,
    measurements: list[ProfileMeasurement] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = profile_summary_rows(payload, measurements, config)
    dynamic_columns = sorted({key for row in rows for key in row}.difference(PROFILE_SUMMARY_COLUMNS))
    fieldnames = PROFILE_SUMMARY_COLUMNS + dynamic_columns
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _experiment_row(
    payload: dict[str, Any],
    measurements: list[ProfileMeasurement],
    epsilons: list[float],
    deltas: list[float],
) -> dict[str, Any]:
    payload_rows = payload.get("rows")
    profile_rows = payload_rows.get("profiles") if isinstance(payload_rows, dict) else None
    return {
        "section": "experiment",
        "name": "pilot-smoke-measured",
        "ok": payload.get("ok"),
        "error": _summary_error(payload),
        "diagnostic_output": payload.get("diagnostic_output", ""),
        "failures": payload.get("failures", ""),
        "config": payload.get("config"),
        "profile_rows": profile_rows if profile_rows is not None else (len(measurements) if measurements else ""),
        "dry_run": payload.get("dry_run", ""),
        "epsilon_values": epsilons,
        "delta_values": deltas,
    }


def _profile_row(
    profile: str,
    rows: list[ProfileMeasurement],
    payload: dict[str, Any],
    config: dict[str, Any],
    exact: set[str],
    epsilons: list[float],
    deltas: list[float],
) -> dict[str, Any]:
    losses = [row.quality_loss for row in rows if row.quality_loss is not None]
    latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
    ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
    peak_memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
    resident_memories = [row.resident_memory_mib for row in rows if row.resident_memory_mib is not None]
    kv_memories = [row.kv_cache_memory_mib for row in rows if row.kv_cache_memory_mib is not None]

    profile_specs = config.get("profiles", {}).get("specs", {}) if isinstance(config.get("profiles"), dict) else {}
    spec = profile_specs.get(profile, {}) if isinstance(profile_specs, dict) else {}
    spec = spec if isinstance(spec, dict) else {}
    family = spec.get("family") or _first_extra(rows, "family") or rows[0].adapter

    row: dict[str, Any] = {
        "section": "profile",
        "name": profile,
        "ok": payload.get("ok"),
        "config": payload.get("config"),
        "dry_run": payload.get("dry_run", ""),
        "epsilon_values": epsilons,
        "delta_values": deltas,
        "family": family,
        "exact": profile in exact or bool(spec.get("exact")),
        "count": float(len(rows)),
        "ok_count": float(sum(1 for item in rows if item.ok)),
        "measured_count": float(sum(1 for item in rows if item.measured)),
        "task_count": float(len({item.extra.get("task") for item in rows if item.extra.get("task")})),
        "length_bucket_count": float(len({item.extra.get("length_bucket") for item in rows if item.extra.get("length_bucket")})),
        "split_count": float(len({item.extra.get("split") for item in rows if item.extra.get("split")})),
        "mean_quality_loss": _mean(losses),
        "p50_quality_loss": _percentile(losses, 0.50),
        "p95_quality_loss": _percentile(losses, 0.95),
        "p99_quality_loss": _percentile(losses, 0.99),
        "cvar_quality_loss": _cvar(losses, 0.95),
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "p99_latency_ms": _percentile(latencies, 0.99),
        "mean_ttft_ms": _mean(ttfts),
        "p50_ttft_ms": _percentile(ttfts, 0.50),
        "p95_ttft_ms": _percentile(ttfts, 0.95),
        "p99_ttft_ms": _percentile(ttfts, 0.99),
        "mean_peak_memory_mib": _mean(peak_memories),
        "p95_peak_memory_mib": _percentile(peak_memories, 0.95),
        "p99_peak_memory_mib": _percentile(peak_memories, 0.99),
        "mean_resident_memory_mib": _mean(resident_memories),
        "p95_resident_memory_mib": _percentile(resident_memories, 0.95),
        "mean_kv_cache_memory_mib": _mean(kv_memories),
        "p95_kv_cache_memory_mib": _percentile(kv_memories, 0.95),
        "p99_kv_cache_memory_mib": _percentile(kv_memories, 0.99),
        "backend_distribution": _distribution(rows, "backend"),
        "ttft_semantics_distribution": _distribution(rows, "ttft_semantics"),
        "worker_mode_distribution": _distribution(rows, "worker_mode"),
        "primary_metric_distribution": _distribution(rows, "primary_metric"),
    }
    for epsilon in epsilons:
        row[f"violation_rate_eps{_slug_number(epsilon)}"] = _violation_rate(losses, epsilon)
    return row


def _pilot(config: dict[str, Any]) -> dict[str, Any]:
    pilot = config.get("pilot", {})
    return pilot if isinstance(pilot, dict) else {}


def _configured_profiles(config: dict[str, Any]) -> list[str]:
    profiles = config.get("profiles", {})
    names = profiles.get("names") if isinstance(profiles, dict) else None
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def _summary_error(payload: dict[str, Any]) -> Any:
    if payload.get("error"):
        return payload.get("error")
    profile = payload.get("profile")
    if isinstance(profile, dict) and profile.get("error"):
        return profile.get("error")
    return ""


def _number_list(value: Any, *, default: float) -> list[float]:
    if value is None:
        return [default]
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            items = [value]
    if not items:
        return [default]
    return [float(item) for item in items]


def _slug_number(value: float) -> str:
    if math.isinf(value):
        return "inf"
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _first_extra(rows: list[ProfileMeasurement], key: str) -> Any:
    for row in rows:
        if row.extra.get(key):
            return row.extra.get(key)
    return ""


def _distribution(rows: list[ProfileMeasurement], key: str) -> dict[str, int]:
    counts = Counter(str(row.extra.get(key)) for row in rows if row.extra.get(key) not in {None, ""})
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float:
    finite_values = _finite(values)
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(_finite(values))
    if not finite_values:
        return float("nan")
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]


def _cvar(values: list[float], quantile: float) -> float:
    finite_values = sorted(_finite(values))
    if not finite_values:
        return float("nan")
    threshold = _percentile(finite_values, quantile)
    tail = [value for value in finite_values if value >= threshold]
    return _mean(tail)


def _violation_rate(values: list[float], epsilon: float) -> float:
    finite_values = _finite(values)
    return sum(1 for value in finite_values if value > epsilon) / len(finite_values) if finite_values else float("nan")


def _finite(values: list[float]) -> list[float]:
    return [float(value) for value in values if value is not None and not math.isnan(float(value))]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return json_ready(value)
