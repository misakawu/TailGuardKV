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
    profiles: list[str] | None = None
    calibration_rows: list[ProfileMeasurement] = field(default_factory=list)
    min_group_samples: int = 2

    def __post_init__(self) -> None:
        self.exact_profiles = self.exact_profiles or {"full_gpu", "full_cpu", "recompute"}
        configured_lossy_profiles = {
            profile
            for profile in (self.profiles or [])
            if profile not in self.exact_profiles
        }
        observed_lossy_profiles = {
            row.profile
            for row in self.calibration_rows
            if row.profile not in self.exact_profiles and row.quality_loss is not None
        }
        self.lossy_profiles = configured_lossy_profiles | observed_lossy_profiles
        k = max(1, len(self.lossy_profiles))
        self.delta_a = self.delta / k
        self._group_residuals = {} if not self.calibration_rows else self._build_group_residuals(self.calibration_rows)

    def _predicted_loss_for_row(
        self,
        row: ProfileMeasurement,
        group_sums: dict[tuple[str, str, str], float],
        group_counts: dict[tuple[str, str, str], int],
        profile_sums: dict[str, float],
        profile_counts: dict[str, int],
    ) -> float:
        task = str(row.extra.get("task") or "unknown")
        bucket = str(row.extra.get("length_bucket") or "unknown")
        group_key = (task, bucket, row.profile)
        group_count = group_counts.get(group_key, 0) - 1
        if group_count > 0:
            return (group_sums[group_key] - (row.quality_loss or 0.0)) / group_count
        profile_count = profile_counts.get(row.profile, 0) - 1
        if profile_count > 0:
            return (profile_sums[row.profile] - (row.quality_loss or 0.0)) / profile_count
        return 1.0

    def _build_group_residuals(self, rows: list[ProfileMeasurement]) -> dict[tuple[str, str, str], list[float]]:
        grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        group_sums: dict[tuple[str, str, str], float] = defaultdict(float)
        group_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        profile_sums: dict[str, float] = defaultdict(float)
        profile_counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if row.profile in self.exact_profiles or row.quality_loss is None:
                continue
            task = str(row.extra.get("task") or "unknown")
            bucket = str(row.extra.get("length_bucket") or "unknown")
            group_key = (task, bucket, row.profile)
            loss = row.quality_loss or 0.0
            group_sums[group_key] += loss
            group_counts[group_key] += 1
            profile_sums[row.profile] += loss
            profile_counts[row.profile] += 1

        for row in rows:
            if row.profile in self.exact_profiles or row.quality_loss is None:
                continue
            task = str(row.extra.get("task") or "unknown")
            bucket = str(row.extra.get("length_bucket") or "unknown")
            predicted_loss = self._predicted_loss_for_row(row, group_sums, group_counts, profile_sums, profile_counts)
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
