from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from math import isnan
from statistics import mean


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    *,
    confidence: float = 0.95,
    samples: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if samples <= 0:
        raise ValueError("samples must be positive")
    stat = statistic or mean
    rng = random.Random(seed)
    estimates = sorted(stat([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples))
    return _confidence_interval(estimates, confidence=confidence)


def _finite(values: Sequence[float]) -> list[float]:
    return [value for value in values if value is not None and not isnan(float(value))]


def _mean(values: Sequence[float]) -> float:
    finite_values = _finite(values)
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile(values: Sequence[float], quantile: float) -> float:
    finite_values = sorted(_finite(values))
    if not finite_values:
        return float("nan")
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]


def _confidence_interval(estimates: list[float], *, confidence: float) -> tuple[float, float]:
    finite_estimates = sorted(value for value in estimates if not isnan(float(value)))
    if not finite_estimates:
        return float("nan"), float("nan")
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low = finite_estimates[int((alpha / 2) * (len(finite_estimates) - 1))]
    high = finite_estimates[int((1 - alpha / 2) * (len(finite_estimates) - 1))]
    return low, high


def _block_id(record: object) -> str:
    session_id = getattr(record, "session_id", None)
    if session_id:
        return str(session_id)
    return str(getattr(record, "request_id", id(record)))


def _metric_value(record: object, metric: str) -> float | None:
    if metric in {"p95_ttft_ms", "mean_ttft_ms"}:
        return getattr(record, "ttft_ms", None)
    if metric in {"p95_quality_loss", "mean_quality_loss", "violation_rate"}:
        return getattr(record, "quality_loss", None)
    raise ValueError(f"unsupported session bootstrap metric: {metric}")


def _pooled_statistic(records: list[object], metric: str, epsilon: float | None) -> float:
    if metric == "violation_rate":
        if epsilon is None:
            raise ValueError("violation_rate bootstrap requires epsilon")
        losses = _finite([_metric_value(record, metric) for record in records])
        return sum(1 for loss in losses if loss > epsilon) / len(losses) if losses else float("nan")
    if metric == "mean_ttft_ms" or metric == "mean_quality_loss":
        return _mean([_metric_value(record, metric) for record in records])
    return _percentile([_metric_value(record, metric) for record in records], 0.95)


def session_block_bootstrap_ci(
    records: Sequence[object],
    metric: str,
    *,
    samples: int = 1000,
    seed: int = 20260906,
    epsilon: float | None = None,
) -> tuple[float, float]:
    """Bootstrap a 95% CI by resampling whole sessions (blocks) with replacement.

    The statistic for each resample is computed over per-turn values pooled
    inside the sampled blocks, so it never averages per-session point estimates.
    ``metric`` must be one of p95_ttft_ms/mean_ttft_ms/mean_quality_loss/
    p95_quality_loss/violation_rate. Fewer than two blocks yield (nan, nan).
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    supported = {
        "p95_ttft_ms",
        "mean_ttft_ms",
        "mean_quality_loss",
        "p95_quality_loss",
        "violation_rate",
    }
    if metric not in supported:
        raise ValueError(f"unsupported session bootstrap metric: {metric}")
    blocks: dict[str, list[object]] = {}
    for record in records:
        blocks.setdefault(_block_id(record), []).append(record)
    block_ids = sorted(blocks)
    if len(block_ids) < 2:
        return float("nan"), float("nan")

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_records: list[object] = []
        for _ in block_ids:
            sampled_records.extend(blocks[block_ids[rng.randrange(len(block_ids))]])
        estimates.append(_pooled_statistic(sampled_records, metric, epsilon))
    return _confidence_interval(estimates, confidence=0.95)
