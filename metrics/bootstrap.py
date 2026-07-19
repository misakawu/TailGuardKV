from __future__ import annotations

import random
from collections.abc import Callable, Sequence
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
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low = estimates[int((alpha / 2) * (samples - 1))]
    high = estimates[int((1 - alpha / 2) * (samples - 1))]
    return low, high
