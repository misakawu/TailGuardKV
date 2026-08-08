from __future__ import annotations

import csv
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from run_util.core_types import ProfileMeasurement, Request
from metrics.quality import compute_quality_loss, select_primary_loss


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
    if str(data_config.get("source") or "").lower() == "sharegpt":
        conversations = load_sharegpt_conversations(path)
        requests = requests_from_sharegpt_conversations(conversations)
        requests = _filter_requests_by_token_limit(requests, config)
        return _ensure_session_splits(requests, float(data_config.get("calibration_fraction", 0.5))), False
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


def load_sharegpt_conversations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"请求输入文件为空: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
    if not isinstance(rows, list):
        raise ValueError(f"ShareGPT 输入必须是数组: {path}")
    return [row for row in rows if isinstance(row, dict)]


def requests_from_sharegpt_conversations(rows: list[dict[str, Any]]) -> list[Request]:
    requests: list[Request] = []
    arrival_index = 0
    for conversation_index, row in enumerate(rows):
        session_id = str(row.get("id") or row.get("session_id") or f"session_{conversation_index:06d}")
        messages = row.get("conversations")
        if not isinstance(messages, list):
            continue
        history: list[str] = []
        user_turn_index = 0
        pending_user: str | None = None
        for message in messages:
            if not isinstance(message, dict):
                continue
            speaker = str(message.get("from") or message.get("role") or "").lower()
            value = str(message.get("value") or message.get("content") or "").strip()
            if not value:
                continue
            if speaker in {"human", "user"}:
                prompt = f"User: {value}" if history else value
                pending_user = value
                requests.append(
                    Request(
                        request_id=f"{session_id}_turn_{user_turn_index:03d}",
                        task="chat",
                        prompt=prompt,
                        session_id=session_id,
                        turn_index=user_turn_index,
                        arrival_index=arrival_index,
                        history_turns=tuple(history),
                        metadata={"source": session_id},
                    )
                )
                user_turn_index += 1
                arrival_index += 1
            elif speaker in {"gpt", "assistant"}:
                if pending_user is not None:
                    history.append(f"User: {pending_user}")
                    pending_user = None
                history.append(f"Assistant: {value}")
        if requests and requests[-1].session_id == session_id:
            requests[-1] = replace(requests[-1], metadata={**requests[-1].metadata, "last_turn": True})
    return requests


def _filter_requests_by_token_limit(requests: list[Request], config: dict[str, Any]) -> list[Request]:
    limit = _configured_prompt_token_limit(config)
    if limit <= 0 or not requests:
        return requests
    tokenizer = _load_prompt_tokenizer(config)
    if tokenizer is None:
        return requests
    filtered = [request for request in requests if _request_prompt_token_count(request, tokenizer) <= limit]
    if filtered:
        return filtered
    raise ValueError(f"所有 ShareGPT 请求都超过 token 上限 {limit}")


def _configured_prompt_token_limit(config: dict[str, Any]) -> int:
    profile_config = config.get("profile_smoke", {})
    raw_limit = profile_config.get("max_prompt_tokens", profile_config.get("vllm_max_model_len", 0))
    try:
        return max(0, int(raw_limit or 0))
    except (TypeError, ValueError):
        return 0


def _load_prompt_tokenizer(config: dict[str, Any]) -> Any | None:
    model_config = config.get("model", {})
    profile_config = config.get("profile_smoke", {})
    model_name = model_config.get("pilot_model") or model_config.get("path") or model_config.get("name")
    if not model_name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_name),
        cache_dir=model_config.get("cache_dir") or None,
        local_files_only=bool(profile_config.get("local_files_only", True)),
        trust_remote_code=True,
    )


def _request_prompt_token_count(request: Request, tokenizer: Any) -> int:
    tokenized = tokenizer(request.effective_prompt, return_tensors="pt")
    input_ids = tokenized.get("input_ids")
    if input_ids is None:
        return 0
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    values = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    while isinstance(values, list) and values and isinstance(values[0], list):
        values = values[0]
    return len(values) if isinstance(values, list) else 0


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


def _ensure_session_splits(requests: list[Request], calibration_fraction: float) -> list[Request]:
    if not requests:
        return []
    if any(request.metadata.get("split") for request in requests):
        return requests
    ordered_sessions: list[str] = []
    for request in requests:
        session_id = request.session_id or request.request_id
        if session_id not in ordered_sessions:
            ordered_sessions.append(session_id)
    cutoff = max(1, min(len(ordered_sessions) - 1, int(round(len(ordered_sessions) * calibration_fraction)))) if len(ordered_sessions) > 1 else 1
    calibration_sessions = set(ordered_sessions[:cutoff])
    return [
        replace(
            request,
            metadata={
                **request.metadata,
                "split": "calibration" if (request.session_id or request.request_id) in calibration_sessions else "eval",
            },
        )
        for request in requests
    ]


def limit_requests_by_split(requests: list[Request], max_requests: int) -> list[Request]:
    if max_requests <= 0 or len(requests) <= max_requests:
        return requests
    calibration = [request for request in requests if request.metadata.get("split") == "calibration"]
    evaluation = [request for request in requests if request.metadata.get("split") == "eval"]
    if not calibration or not evaluation or max_requests < 2:
        return requests[:max_requests]
    calibration_limit = max_requests // 2
    evaluation_limit = max_requests - calibration_limit
    return _limit_requests_for_split(calibration, calibration_limit) + _limit_requests_for_split(evaluation, evaluation_limit)


def _limit_requests_for_split(requests: list[Request], limit: int) -> list[Request]:
    if limit <= 0 or len(requests) <= limit:
        return requests[:limit] if limit > 0 else []
    groups: dict[tuple[str, str], list[Request]] = {}
    for request in requests:
        key = (request.task, str(request.metadata.get("length_bucket", length_bucket(request.prompt_chars))))
        groups.setdefault(key, []).append(request)
    if len(groups) < 2:
        return requests[:limit]
    queues = [deque(group) for group in groups.values()]
    selected: list[Request] = []
    while len(selected) < limit and any(queue for queue in queues):
        progressed = False
        for queue in queues:
            if not queue:
                continue
            selected.append(queue.popleft())
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if len(selected) < limit:
        return requests[:limit]
    return selected


def requests_from_measurements(measurements: list[ProfileMeasurement]) -> list[Request]:
    seen: dict[tuple[str, int, str], Request] = {}
    for measurement in sorted(measurements, key=lambda item: ((item.session_id or item.request_id), item.turn_index, item.request_id)):
        key = (measurement.session_id or "", measurement.turn_index, measurement.request_id)
        if key in seen:
            continue
        task = str(measurement.extra.get("task") or (measurement.request_id.split("_", 1)[0] if "_" in measurement.request_id else "unknown"))
        seen[key] = Request(
            request_id=measurement.request_id,
            task=task,
            prompt=measurement.output_text,
            session_id=measurement.session_id,
            turn_index=measurement.turn_index,
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
    session_ids = sorted({row.session_id or row.request_id for row in measurements})
    if len(session_ids) <= 1:
        return measurements, measurements
    cutoff = max(1, len(session_ids) // 2)
    calibration_ids = set(session_ids[:cutoff])
    return (
        [row for row in measurements if (row.session_id or row.request_id) in calibration_ids],
        [row for row in measurements if (row.session_id or row.request_id) not in calibration_ids],
    )


def annotate_measurement(measurement: ProfileMeasurement, request: Request, fallback_requests: bool) -> ProfileMeasurement:
    primary_metric = select_primary_loss(request.task)
    return replace(
        measurement,
        extra={
            **measurement.extra,
            "request_source": request.metadata.get("source", "unknown"),
            "split": request.metadata.get("split", ""),
            "task": request.task,
            "length_bucket": request.metadata.get("length_bucket", length_bucket(request.prompt_chars)),
            "builtin_request_fallback": str(fallback_requests).lower(),
            "reference": request.reference or "",
            "primary_metric": primary_metric,
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
    full_gpu_by_request = {
        row.request_id: row
        for row in measurements
        if row.profile == "full_gpu" and row.ok and row.measured
    }
    updated: list[ProfileMeasurement] = []
    for row in measurements:
        task = str(row.extra.get("task") or "unknown")
        primary_metric = select_primary_loss(task)
        if row.profile in exact and row.ok and row.measured:
            updated.append(
                replace(
                    row,
                    quality_loss=0.0,
                    quality_score=1.0,
                    extra={
                        **row.extra,
                        "primary_metric": primary_metric,
                        "metric_em": 0.0,
                        "metric_f1": 0.0,
                        "metric_rouge_l": 0.0,
                    },
                )
            )
            continue
        if not row.ok or not row.measured:
            updated.append(replace(row, extra={**row.extra, "primary_metric": primary_metric}))
            continue
        baseline = full_gpu_by_request.get(row.request_id)
        if baseline is None:
            updated.append(replace(row, quality_loss=None, quality_score=None, extra={**row.extra, "primary_metric": primary_metric}))
            continue
        task = str(row.extra.get("task") or baseline.extra.get("task") or "unknown")
        primary_metric = select_primary_loss(task)
        reference = str(row.extra.get("reference") or baseline.extra.get("reference") or "").strip()
        if reference:
            baseline_loss, _ = compute_quality_loss(task, baseline.output_text, reference)
            candidate_loss, metrics = compute_quality_loss(task, row.output_text, reference)
            loss = max(0.0, min(1.0, candidate_loss - baseline_loss))
        else:
            loss, metrics = compute_quality_loss(task, row.output_text, baseline.output_text)
        updated.append(
            replace(
                row,
                quality_loss=loss,
                quality_score=1.0 - loss,
                extra={
                    **row.extra,
                    "primary_metric": primary_metric,
                    **{f"metric_{key}": value for key, value in metrics.items()},
                },
            )
        )
    return updated
