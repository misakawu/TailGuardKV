from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

SUPPORTED_PRIMARY_LOSS = {
    "qa": "f1",
    "qa_long_context": "f1",
    "summary": "rouge_l",
    "code": "f1",
}


def normalized_exact_match_loss(candidate: str | None, reference: str | None) -> float:
    return 0.0 if _normalize_text(candidate) == _normalize_text(reference) else 1.0


def token_f1_loss(candidate: str | None, reference: str | None) -> float:
    return _token_f1_loss(_tokenize(candidate), _tokenize(reference))


def _token_f1_loss(candidate_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    if not candidate_tokens and not reference_tokens:
        return 0.0
    if not candidate_tokens or not reference_tokens:
        return 1.0
    overlap = _overlap_count(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0.0:
        return 1.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return 1.0 - f1


def rouge_l_loss(candidate: str | None, reference: str | None) -> float:
    return _rouge_l_loss(_tokenize(candidate), _tokenize(reference))


def _rouge_l_loss(candidate_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    if not candidate_tokens and not reference_tokens:
        return 0.0
    if not candidate_tokens or not reference_tokens:
        return 1.0
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0.0:
        return 1.0
    score = 2.0 * precision * recall / (precision + recall)
    return 1.0 - score


def select_primary_loss(task: str, *, strict: bool = False) -> str:
    metric = SUPPORTED_PRIMARY_LOSS.get(task)
    if metric is not None:
        return metric
    if strict:
        raise ValueError(
            f"baseline quality smoke 不支持 task={task!r}；"
            "请改用带 reference 的 qa/summary/code 请求，或将该输入放到 session/cache 诊断链路。"
        )
    return "em"


def compute_quality_loss(task: str, candidate: str | None, reference: str | None) -> tuple[float, dict[str, float]]:
    if not _has_text(candidate) or not _has_text(reference):
        metrics = {"em": 1.0, "f1": 1.0, "rouge_l": 1.0}
        return metrics[select_primary_loss(task)], metrics
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
    metrics = {
        "em": normalized_exact_match_loss(candidate, reference),
        "f1": _token_f1_loss(candidate_tokens, reference_tokens),
        "rouge_l": _rouge_l_loss(candidate_tokens, reference_tokens),
    }
    primary = select_primary_loss(task)
    return metrics[primary], metrics


def _has_text(text: str | None) -> bool:
    return bool(_normalize_text(text))


def _normalize_text(text: str | None) -> str:
    return " ".join(_tokenize(text))


def _tokenize(text: str | None) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.findall(r"\w+", normalized, flags=re.UNICODE)


def _overlap_count(candidate_tokens: Iterable[str], reference_tokens: Iterable[str]) -> int:
    counts: dict[str, int] = {}
    for token in reference_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in candidate_tokens:
        remaining = counts.get(token, 0)
        if remaining > 0:
            overlap += 1
            counts[token] = remaining - 1
    return overlap


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
