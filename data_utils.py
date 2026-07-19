from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from core_types import ProfileMeasurement, Request
from metrics.quality import compute_quality_loss


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
        if row.profile in exact and row.ok and row.measured
    }
    updated: list[ProfileMeasurement] = []
    for row in measurements:
        baseline = baseline_by_request.get(row.request_id)
        if row.profile in exact and row.ok and row.measured:
            updated.append(
                replace(
                    row,
                    quality_loss=0.0,
                    quality_score=1.0,
                    extra={**row.extra, "metric_em": 0.0, "metric_f1": 0.0, "metric_rouge_l": 0.0},
                )
            )
            continue
        if not row.ok or not row.measured:
            updated.append(row)
            continue
        reference = baseline.output_text if baseline is not None else row.extra.get("reference")
        task = str(row.extra.get("task") or (baseline.extra.get("task") if baseline is not None else "") or "unknown")
        loss, metrics = compute_quality_loss(task, row.output_text, None if reference is None else str(reference))
        updated.append(
            replace(
                row,
                quality_loss=loss,
                quality_score=1.0 - loss,
                extra={**row.extra, **{f"metric_{key}": value for key, value in metrics.items()}},
            )
        )
    return updated
