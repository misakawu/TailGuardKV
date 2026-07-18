from __future__ import annotations

from collections.abc import Iterable


def normalized_exact_match_loss(candidate: str, reference: str) -> float:
    return 0.0 if candidate.strip() == reference.strip() else 1.0


def token_f1_loss(candidate: str, reference: str) -> float:
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
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


def rouge_l_loss(candidate: str, reference: str) -> float:
    candidate_tokens = _tokenize(candidate)
    reference_tokens = _tokenize(reference)
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


def select_primary_loss(task: str) -> str:
    if task == "qa":
        return "f1"
    if task == "summary":
        return "rouge_l"
    return "exact_match"


def compute_quality_loss(task: str, candidate: str, reference: str) -> tuple[float, dict[str, float]]:
    metrics = {
        "em": normalized_exact_match_loss(candidate, reference),
        "f1": token_f1_loss(candidate, reference),
        "rouge_l": rouge_l_loss(candidate, reference),
    }
    primary = select_primary_loss(task)
    return metrics[primary], metrics


def _tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


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

