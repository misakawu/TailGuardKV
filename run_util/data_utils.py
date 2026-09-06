from __future__ import annotations

import csv
import json
import random
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from run_util.core_types import ProfileMeasurement, Request
from metrics.quality import compute_quality_loss, select_primary_loss
from run_util.canonical_history import canonical_history_for_row, validate_canonical_history_rows

QUALITY_MODE_BASELINE = "baseline"
QUALITY_MODE_SESSION_DIAGNOSTIC = "session_diagnostic"
QUALITY_MODE_COMPAT = "compat"


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
        metadata = _request_metadata(row, path, index)
        task = _normalized_request_task(row.get("task") or metadata.get("task") or "unknown")
        arrival_index = _request_arrival_index(row, metadata, index)
        requests.append(
            Request(
                request_id=request_id,
                task=task,
                prompt=str(row.get("prompt") or ""),
                reference=(None if row.get("reference") in {None, ""} else str(row.get("reference"))),
                session_id=_optional_text(row.get("session_id")),
                turn_index=_optional_int(row.get("turn_index")),
                arrival_index=arrival_index,
                metadata=metadata,
            )
        )
    requests = _with_session_histories(requests)
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
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None

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


def _request_metadata(row: dict[str, Any], path: Path, index: int) -> dict[str, Any]:
    nested = row.get("metadata")
    metadata = dict(nested) if isinstance(nested, dict) else {}
    for key, value in row.items():
        if key in {"request_id", "id", "task", "prompt", "reference", "metadata", "session_id", "turn_index", "arrival_index"}:
            continue
        metadata.setdefault(key, value)
    metadata.setdefault("source", str(path))
    metadata.setdefault("source_dataset", metadata.get("dataset_config") or row.get("source") or str(path))
    if "task" in metadata:
        metadata["task"] = _normalized_request_task(metadata["task"])
    return metadata


def _request_arrival_index(row: dict[str, Any], metadata: dict[str, Any], default: int) -> int:
    raw = row.get("arrival_index", metadata.get("arrival_index", default))
    return _optional_int(raw, default=default)


def _normalized_request_task(value: Any) -> str:
    task = str(value or "unknown").strip()
    if task == "qa_long_context":
        return "qa"
    return task


def _optional_text(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)


def _optional_int(value: Any, *, default: int = 0) -> int:
    if value in {None, ""}:
        return default
    return int(float(value))


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


def _with_session_histories(requests: list[Request]) -> list[Request]:
    if not any(request.session_id for request in requests):
        return requests
    grouped: dict[str, list[Request]] = {}
    for request in requests:
        if request.session_id:
            grouped.setdefault(request.session_id, []).append(request)

    canonical_rows = [
        {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "arrival_index": request.arrival_index,
            "prompt": request.prompt,
            "metadata": request.metadata,
        }
        for request in requests
        if request.session_id
    ]
    validate_canonical_history_rows(canonical_rows)

    updated: dict[tuple[str, int, str], Request] = {}
    for session_id, session_requests in grouped.items():
        ordered = sorted(
            session_requests,
            key=lambda request: (request.turn_index, request.arrival_index, request.request_id),
        )
        history: list[str] = []
        expected_turn = 0
        previous_arrival = -1
        for request in ordered:
            if request.turn_index != expected_turn:
                raise ValueError(
                    "session 请求 turn_index 必须从 0 连续递增；"
                    f"session={session_id} expected={expected_turn} actual={request.turn_index}"
                )
            if request.arrival_index < previous_arrival:
                raise ValueError(
                    "session 请求 arrival_index 必须随 turn_index 非递减；"
                    f"session={session_id} turn_index={request.turn_index}"
                )
            explicit_history = canonical_history_for_row({"metadata": request.metadata})
            updated[(session_id, request.turn_index, request.request_id)] = replace(
                request,
                history_turns=explicit_history if explicit_history is not None else tuple(history),
            )
            if explicit_history is None:
                history.append(f"User: {request.prompt}")
                if request.reference:
                    history.append(f"Assistant: {request.reference}")
            previous_arrival = request.arrival_index
            expected_turn += 1

    normalized: list[Request] = []
    for request in requests:
        if not request.session_id:
            normalized.append(request)
            continue
        normalized.append(updated[(request.session_id, request.turn_index, request.request_id)])
    return sorted(
        normalized,
        key=lambda request: (
            request.arrival_index,
            request.session_id or request.request_id,
            request.turn_index,
            request.request_id,
        ),
    )


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
    ordered = sorted(
        measurements,
        key=lambda item: (
            _measurement_arrival_index(item),
            item.session_id or item.request_id,
            item.turn_index,
            item.request_id,
        ),
    )
    for measurement in ordered:
        key = (measurement.session_id or "", measurement.turn_index, measurement.request_id)
        if key in seen:
            continue
        task = str(measurement.extra.get("task") or (measurement.request_id.split("_", 1)[0] if "_" in measurement.request_id else "unknown"))
        prompt_text = str(measurement.extra.get("prompt_text") or measurement.output_text)
        history_turns = _history_turns_from_measurement(measurement)
        seen[key] = Request(
            request_id=measurement.request_id,
            task=task,
            prompt=prompt_text,
            session_id=measurement.session_id,
            turn_index=measurement.turn_index,
            arrival_index=_measurement_arrival_index(measurement),
            history_turns=history_turns,
            metadata={
                "source": "profile_measurement",
                "split": measurement.extra.get("split", ""),
                "task": task,
                "length_bucket": measurement.extra.get("length_bucket", "unknown"),
                "effective_prompt_chars": measurement.extra.get("effective_prompt_chars", ""),
            },
        )
    return list(seen.values())


def split_measurements(
    measurements: list[ProfileMeasurement],
    *,
    split_seed: int = 20260906,
    stratify_session: bool = True,
) -> tuple[list[ProfileMeasurement], list[ProfileMeasurement]]:
    if not stratify_session:
        calibration = [row for row in measurements if row.extra.get("split") == "calibration"]
        evaluation = [row for row in measurements if row.extra.get("split") != "calibration"]
        if calibration and evaluation:
            return calibration, evaluation

    sessions: dict[str, list[ProfileMeasurement]] = {}
    for row in measurements:
        sessions.setdefault(row.session_id or row.request_id, []).append(row)
    if len(sessions) <= 1:
        return measurements, []

    if not stratify_session:
        calibration_ids = set(sorted(sessions)[: max(1, len(sessions) // 2)])
    else:
        strata: dict[tuple[str, str, int], list[str]] = {}
        for session_id, rows in sessions.items():
            first = min(rows, key=lambda row: (row.turn_index, row.request_id, row.profile))
            stratum = (
                str(first.extra.get("task") or "unknown"),
                str(first.extra.get("length_bucket") or "unknown"),
                max(row.turn_index for row in rows),
            )
            strata.setdefault(stratum, []).append(session_id)
        calibration_ids: set[str] = set()
        for stratum, session_ids in strata.items():
            ordered = sorted(session_ids)
            random.Random(f"{split_seed}:{stratum}").shuffle(ordered)
            calibration_ids.update(ordered[: max(1, len(ordered) // 2)])
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
            "arrival_index": request.arrival_index,
            "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
            "prompt_text": request.prompt,
            "effective_prompt_chars": request.prompt_chars,
        },
    )


def _measurement_arrival_index(measurement: ProfileMeasurement) -> int:
    raw = measurement.extra.get("arrival_index")
    if raw in {None, ""}:
        return measurement.turn_index
    return int(float(raw))


def _history_turns_from_measurement(measurement: ProfileMeasurement) -> tuple[str, ...]:
    raw = measurement.extra.get("history_turns")
    if raw in {None, ""}:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw)
    text = str(raw).strip()
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return tuple(part for part in text.split("\n") if part)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return ()


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


def validate_requests_for_quality_mode(requests: list[Request], quality_mode: str, exact_profiles: set[str]) -> None:
    if quality_mode != QUALITY_MODE_BASELINE or not exact_profiles:
        return
    for request in requests:
        task = str(request.task or "unknown")
        select_primary_loss(task, strict=True)
        reference = (request.reference or "").strip()
        if reference:
            continue
        raise ValueError(
            "baseline quality smoke 不接受缺少 reference 的请求；"
            f"request={request.request_id} task={task} 只适合 session/cache 诊断，不适合 baseline quality smoke。"
        )


def validate_requests_for_experiment_type(
    requests: list[Request],
    experiment_type: str,
) -> None:
    if experiment_type != "baseline_session":
        return
    if not requests:
        raise ValueError("baseline_session 要求非空 session-aware 请求输入")
    ordered = sorted(
        requests,
        key=lambda request: (
            request.arrival_index,
            request.session_id or request.request_id,
            request.turn_index,
            request.request_id,
        ),
    )
    if ordered != requests:
        raise ValueError("baseline_session 要求请求按 arrival_index 排序")
    missing_session = [request.request_id for request in requests if not request.session_id]
    if missing_session:
        raise ValueError(
            "baseline_session 要求每条请求有非空 session_id；"
            f"缺少: {missing_session[:3]}"
        )
    session_order: dict[str, list[Request]] = {}
    for request in requests:
        session_order.setdefault(request.session_id or "", []).append(request)
    for session_id, session_requests in session_order.items():
        previous_arrival = -1
        seen_turns: set[int] = set()
        for expected_turn, request in enumerate(
            sorted(session_requests, key=lambda item: (item.turn_index, item.arrival_index, item.request_id))
        ):
            if request.turn_index in seen_turns:
                raise ValueError(
                    "baseline_session 不允许重复 turn_index；"
                    f"session_id={session_id} turn_index={request.turn_index}"
                )
            if request.turn_index != expected_turn:
                raise ValueError(
                    "baseline_session 要求每个 session 的 turn_index 从 0 连续递增；"
                    f"session_id={session_id} expected={expected_turn} actual={request.turn_index}"
                )
            if request.arrival_index < previous_arrival:
                raise ValueError(
                    "baseline_session 要求同一 session 的 arrival_index 随 turn_index 递增；"
                    f"session_id={session_id} turn_index={request.turn_index}"
                )
            seen_turns.add(request.turn_index)
            previous_arrival = request.arrival_index
    if not any(request.turn_index > 0 for request in requests):
        raise ValueError("baseline_session 要求至少一个 session 的 turn_index > 0")
    session_counts: dict[str, int] = {}
    for request in requests:
        session_counts[request.session_id or ""] = session_counts.get(request.session_id or "", 0) + 1
    if len(session_counts) < 2:
        raise ValueError("baseline_session 要求至少两个交错 session")
    if not any(count > 1 for count in session_counts.values()):
        raise ValueError("baseline_session 要求至少一个 session 包含多 turn")


def with_quality(
    measurements: list[ProfileMeasurement],
    exact: set[str],
    *,
    quality_mode: str = QUALITY_MODE_COMPAT,
) -> list[ProfileMeasurement]:
    full_gpu_by_request = {
        row.request_id: row
        for row in measurements
        if row.profile == "full_gpu" and row.ok and row.measured
    }
    updated: list[ProfileMeasurement] = []
    for row in measurements:
        task = str(row.extra.get("task") or "unknown")
        primary_metric = select_primary_loss(task, strict=quality_mode == QUALITY_MODE_BASELINE)
        if row.profile in exact and row.ok and row.measured:
            updated.append(
                replace(
                    row,
                    quality_loss=0.0,
                    quality_score=1.0,
                    extra={
                        **row.extra,
                        "primary_metric": primary_metric,
                        "metric_loss_em": 0.0,
                        "metric_loss_f1": 0.0,
                        "metric_loss_rouge_l": 0.0,
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
        primary_metric = select_primary_loss(task, strict=quality_mode == QUALITY_MODE_BASELINE)
        reference = str(row.extra.get("reference") or baseline.extra.get("reference") or "").strip()
        if reference:
            baseline_loss, _ = compute_quality_loss(task, baseline.output_text, reference)
            candidate_loss, metrics = compute_quality_loss(task, row.output_text, reference)
            loss = max(0.0, min(1.0, candidate_loss - baseline_loss))
        else:
            if quality_mode == QUALITY_MODE_BASELINE:
                raise ValueError(
                    "baseline quality smoke 不接受 chat/无 reference 或其他无参考答案请求；"
                    f"request={row.request_id} task={task} 只适合 session/cache 诊断，不适合 baseline quality smoke。"
                )
            if quality_mode == QUALITY_MODE_COMPAT:
                loss, metrics = compute_quality_loss(task, row.output_text, baseline.output_text)
                updated.append(
                    replace(
                        row,
                        quality_loss=loss,
                        quality_score=1.0 - loss,
                        extra={
                            **row.extra,
                            "primary_metric": primary_metric,
                            **{f"metric_loss_{key}": value for key, value in metrics.items()},
                        },
                    )
                )
                continue
            updated.append(replace(row, quality_loss=None, quality_score=None, extra={**row.extra, "primary_metric": primary_metric}))
            continue
        updated.append(
            replace(
                row,
                quality_loss=loss,
                quality_score=1.0 - loss,
                extra={
                    **row.extra,
                    "primary_metric": primary_metric,
                    **{f"metric_loss_{key}": value for key, value in metrics.items()},
                },
            )
        )
    return updated
