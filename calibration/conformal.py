from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil

from core_types import ProfileMeasurement, Request


def _conformal_quantile(values: list[float], delta_a: float) -> float:
    if not values:
        return float("inf")
    ordered = sorted(values)
    rank = ceil((len(ordered) + 1) * (1.0 - delta_a))
    index = min(len(ordered) - 1, max(0, rank - 1))
    return ordered[index]


def _length_bucket(prompt_chars: int) -> str:
    if prompt_chars < 512:
        return "short"
    if prompt_chars < 2048:
        return "medium"
    if prompt_chars < 8192:
        return "long"
    return "xl"


@dataclass
class ConformalGuard:
    """按 task/length/profile 分组的 residual 分位数校准。"""

    epsilon: float
    delta: float
    exact_profiles: set[str] | None = None
    calibration_rows: list[ProfileMeasurement] = field(default_factory=list)
    min_group_samples: int = 2

    def __post_init__(self) -> None:
        self.exact_profiles = self.exact_profiles or {"full_gpu", "full_cpu", "recompute"}
        self.lossy_profiles = {
            row.profile
            for row in self.calibration_rows
            if row.profile not in self.exact_profiles and row.quality_loss is not None
        }
        k = max(1, len(self.lossy_profiles))
        self.delta_a = self.delta / k
        self._group_residuals = self._build_group_residuals(self.calibration_rows)

    def _predicted_loss_for_row(self, row: ProfileMeasurement, rows: list[ProfileMeasurement]) -> float:
        task = str(row.extra.get("task") or "unknown")
        bucket = str(row.extra.get("length_bucket") or "unknown")
        matched = [
            other.quality_loss or 0.0
            for other in rows
            if other is not row
            and other.profile == row.profile
            and str(other.extra.get("task") or "unknown") == task
            and str(other.extra.get("length_bucket") or "unknown") == bucket
            and other.quality_loss is not None
        ]
        if not matched:
            matched = [
                other.quality_loss or 0.0
                for other in rows
                if other is not row and other.profile == row.profile and other.quality_loss is not None
            ]
        return sum(matched) / len(matched) if matched else 1.0

    def _build_group_residuals(self, rows: list[ProfileMeasurement]) -> dict[tuple[str, str, str], list[float]]:
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in rows:
            if row.profile in self.exact_profiles or row.quality_loss is None:
                continue
            task = str(row.extra.get("task") or "unknown")
            bucket = str(row.extra.get("length_bucket") or "unknown")
            predicted_loss = self._predicted_loss_for_row(row, rows)
            residual = row.quality_loss - predicted_loss
            grouped[(task, bucket, row.profile)].append(residual)
            grouped[(task, "*", row.profile)].append(residual)
            grouped[("*", "*", row.profile)].append(residual)
        return grouped

    def residual_slack(self, request: Request, profile: str) -> float:
        if profile in self.exact_profiles:
            return 0.0
        task = str(request.task or request.metadata.get("task") or "unknown")
        bucket = str(request.metadata.get("length_bucket") or _length_bucket(request.prompt_chars))
        for key in ((task, bucket, profile), (task, "*", profile), ("*", "*", profile)):
            residuals = self._group_residuals.get(key, [])
            if len(residuals) >= self.min_group_samples:
                return _conformal_quantile(residuals, self.delta_a)
        return float("inf")

    def risk_upper(self, request: Request, profile: str, predicted_loss: float) -> float:
        if profile in self.exact_profiles:
            return 0.0
        return predicted_loss + self.residual_slack(request, profile)

    def is_safe(self, request: Request, profile: str, predicted_loss: float) -> bool:
        return self.risk_upper(request, profile, predicted_loss) <= self.epsilon

