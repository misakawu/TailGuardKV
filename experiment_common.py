from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from core_types import ProfileMeasurement, Request
from metrics.quality import compute_quality_loss

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


def default_requests() -> list[Request]:
    """没有正式数据集前，先用固定 smoke 请求验证表结构和执行链路。"""

    return [
        Request(
            request_id="smoke_qa_001",
            task="qa",
            prompt="Question: What is KV cache used for in autoregressive decoding?\nAnswer:",
            reference="KV cache stores past key/value tensors to avoid recomputing attention history.",
            metadata={"source": "builtin_smoke", "split": "calibration"},
        ),
        Request(
            request_id="smoke_sum_001",
            task="summary",
            prompt="Summarize: KV cache compression can reduce memory but may hurt tail quality.",
            reference="KV compression saves memory while risking large quality loss on some requests.",
            metadata={"source": "builtin_smoke", "split": "eval"},
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_measurements(path: Path) -> list[ProfileMeasurement]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        validate_profile_table_header(reader.fieldnames or [], path)
        measurements = [ProfileMeasurement.from_row(row) for row in reader]
    validate_profile_measurements(measurements, path)
    return measurements


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


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_yaml_fallback(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return payload


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """本地环境缺 PyYAML 时的最小 fallback；生产/实验环境应安装 PyYAML。"""

    config: dict[str, Any] = {}
    section: str | None = None
    list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            section = line[:-1]
            config[section] = {}
            list_key = None
            continue
        if section is None:
            raise ValueError(f"无法解析配置行: {raw_line}")
        if indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                config[section][key] = _parse_scalar(value)
                list_key = None
            else:
                config[section][key] = []
                list_key = key
            continue
        if indent == 4 and line.startswith("- ") and list_key is not None:
            config[section][list_key].append(_parse_scalar(line[2:].strip()))
            continue
        raise ValueError(f"无法解析配置行: {raw_line}")
    return config


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def config_adapters(config: dict[str, Any]) -> list[str]:
    return list(config.get("profiles", {}).get("adapters", ["full", "kivi", "h2o"]))


def config_profiles(config: dict[str, Any]) -> list[str]:
    return list(
        config.get("profiles", {}).get(
            "names",
            ["full_gpu", "kivi_4bit", "kivi_2bit", "h2o_heavy_hitter", "full_cpu", "recompute"],
        )
    )


def config_policies(config: dict[str, Any]) -> list[str]:
    return list(
        config.get("policies", {}).get(
            "names",
            [
                "full_lru",
                "static_best",
                "static_safe",
                "utility_dynamic",
                "uncalibrated_dynamic",
                "tailguard",
                "quality_oracle",
            ],
        )
    )


def config_runtime(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    pilot = config.get("pilot", {})
    profile = config.get("profile_smoke", {})
    return {
        "pilot_model": model.get("pilot_model") or model.get("path") or model.get("name"),
        "profile_smoke_model": model.get("profile_smoke_model") or model.get("path") or model.get("name"),
        "model_cache_dir": model.get("cache_dir"),
        "max_new_tokens": int(profile.get("max_new_tokens", pilot.get("max_new_tokens", 16))),
        "timeout_s": int(profile.get("timeout_s", profile.get("timeout", 180))),
        "repeat": int(profile.get("repeat", pilot.get("repeats", 1))),
        "local_files_only": bool(profile.get("local_files_only", True)),
        "kivi_group_size": int(profile.get("kivi_group_size", 32)),
        "kivi_residual_length": int(profile.get("kivi_residual_length", 32)),
        "h2o_heavy_ratio": float(profile.get("h2o_heavy_ratio", 0.1)),
        "h2o_recent_ratio": float(profile.get("h2o_recent_ratio", 0.1)),
    }


def exact_profiles(profiles: list[str]) -> set[str]:
    exact_names = {"full_gpu", "full_cpu", "recompute", "exact_offload"}
    return {profile for profile in profiles if profile in exact_names or profile.startswith("full_")}


def load_requests(config: dict[str, Any]) -> tuple[list[Request], bool]:
    data_config = config.get("data", {})
    request_path = data_config.get("requests") or data_config.get("request_path")
    if not request_path:
        return default_requests(), True
    path = Path(str(request_path))
    if not path.exists():
        raise FileNotFoundError(f"请求输入文件不存在: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"请求输入仅支持 JSONL/CSV: {path}")
    if not rows:
        raise ValueError(f"请求输入文件为空: {path}")
    requests: list[Request] = []
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id") or row.get("id") or f"request_{index:06d}")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"request_id", "id", "task", "prompt", "reference"}
        }
        metadata.setdefault("source", str(path))
        requests.append(
            Request(
                request_id=request_id,
                task=str(row.get("task") or "unknown"),
                prompt=str(row.get("prompt") or ""),
                reference=(None if row.get("reference") in {None, ""} else str(row.get("reference"))),
                metadata=metadata,
            )
        )
    return _ensure_request_splits(requests, float(data_config.get("calibration_fraction", 0.5))), False


def length_bucket(prompt_chars: int) -> str:
    if prompt_chars < 512:
        return "short"
    if prompt_chars < 2048:
        return "medium"
    if prompt_chars < 8192:
        return "long"
    return "xl"


def _ensure_request_splits(requests: list[Request], calibration_fraction: float) -> list[Request]:
    if not requests:
        return []
    if any(request.metadata.get("split") for request in requests):
        return requests
    cutoff = max(1, min(len(requests) - 1, int(round(len(requests) * calibration_fraction)))) if len(requests) > 1 else 1
    return [
        replace(
            request,
            metadata={**request.metadata, "split": "calibration" if index < cutoff else "eval"},
        )
        for index, request in enumerate(requests)
    ]


def requests_from_measurements(measurements: list[ProfileMeasurement]) -> list[Request]:
    seen: dict[str, Request] = {}
    for measurement in measurements:
        if measurement.request_id in seen:
            continue
        task = str(measurement.extra.get("task") or (measurement.request_id.split("_", 1)[0] if "_" in measurement.request_id else "unknown"))
        seen[measurement.request_id] = Request(
            request_id=measurement.request_id,
            task=task,
            prompt=measurement.output_text,
            metadata={
                "source": "profile_measurement",
                "split": measurement.extra.get("split", ""),
                "task": task,
                "length_bucket": measurement.extra.get("length_bucket", "unknown"),
            },
        )
    return list(seen.values())


def split_measurements(measurements: list[ProfileMeasurement]) -> tuple[list[ProfileMeasurement], list[ProfileMeasurement]]:
    calibration = [row for row in measurements if row.extra.get("split") == "calibration"]
    evaluation = [row for row in measurements if row.extra.get("split") != "calibration"]
    if calibration and evaluation:
        return calibration, evaluation
    request_ids = sorted({row.request_id for row in measurements})
    if len(request_ids) <= 1:
        return measurements, measurements
    cutoff = max(1, len(request_ids) // 2)
    calibration_ids = set(request_ids[:cutoff])
    return (
        [row for row in measurements if row.request_id in calibration_ids],
        [row for row in measurements if row.request_id not in calibration_ids],
    )


def annotate_measurement(measurement: ProfileMeasurement, request: Request, fallback_requests: bool) -> ProfileMeasurement:
    return replace(
        measurement,
        extra={
            **measurement.extra,
            "request_source": request.metadata.get("source", "unknown"),
            "split": request.metadata.get("split", ""),
            "task": request.task,
            "length_bucket": request.metadata.get("length_bucket", length_bucket(request.prompt_chars)),
            "builtin_request_fallback": str(fallback_requests).lower(),
        },
    )


def expand_repeated_requests(requests: list[Request], repeat: int) -> list[Request]:
    if repeat <= 1:
        return requests
    repeated: list[Request] = []
    for repeat_index in range(repeat):
        for request in requests:
            repeated.append(
                replace(
                    request,
                    request_id=f"{request.request_id}__r{repeat_index + 1}",
                    metadata={
                        **request.metadata,
                        "original_request_id": request.request_id,
                        "repeat_index": str(repeat_index + 1),
                    },
                )
            )
    return repeated


def with_quality(measurements: list[ProfileMeasurement], exact: set[str]) -> list[ProfileMeasurement]:
    baseline_by_request = {
        row.request_id: row
        for row in measurements
        if row.profile in exact and row.ok and row.measured and row.output_text
    }
    updated: list[ProfileMeasurement] = []
    for row in measurements:
        baseline = baseline_by_request.get(row.request_id)
        if row.profile in exact and row.ok and row.measured and row.output_text:
            updated.append(
                replace(
                    row,
                    quality_loss=0.0,
                    quality_score=1.0,
                    extra={**row.extra, "metric_em": 0.0, "metric_f1": 0.0, "metric_rouge_l": 0.0},
                )
            )
            continue
        if not row.ok or not row.measured or not row.output_text or baseline is None:
            updated.append(row)
            continue
        task = str(row.extra.get("task") or baseline.extra.get("task") or "unknown")
        loss, metrics = compute_quality_loss(task, row.output_text, baseline.output_text)
        updated.append(
            replace(
                row,
                quality_loss=loss,
                quality_score=1.0 - loss,
                extra={**row.extra, **{f"metric_{key}": value for key, value in metrics.items()}},
            )
        )
    return updated
