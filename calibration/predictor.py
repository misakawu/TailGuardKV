from __future__ import annotations

from dataclasses import dataclass, field

from core_types import ProfileMeasurement, Request


def _length_bucket(prompt_chars: int) -> str:
    if prompt_chars < 512:
        return "short"
    if prompt_chars < 2048:
        return "medium"
    if prompt_chars < 8192:
        return "long"
    return "xl"


@dataclass
class MetadataOnlyRiskPredictor:
    """QRP：只用元数据和校准集统计，不读评估真值。"""

    calibration_rows: list[ProfileMeasurement] = field(default_factory=list)
    default_loss: float = 1.0

    def _lookup(self, task: str, bucket: str, profile: str) -> list[ProfileMeasurement]:
        return [
            row
            for row in self.calibration_rows
            if row.extra.get("task") == task
            and row.extra.get("length_bucket") == bucket
            and row.profile == profile
            and row.quality_loss is not None
        ]

    def predict_loss(self, request: Request, profile: str) -> float:
        if profile.startswith("full") or profile == "recompute" or profile in {"full_gpu", "full_cpu"}:
            return 0.0
        bucket = str(request.metadata.get("length_bucket") or _length_bucket(request.prompt_chars))
        task = str(request.task or request.metadata.get("task") or "unknown")
        matched = self._lookup(task, bucket, profile)
        if matched:
            return sum(row.quality_loss or 0.0 for row in matched) / len(matched)
        matched = [
            row
            for row in self.calibration_rows
            if row.profile == profile and row.quality_loss is not None
        ]
        if matched:
            return sum(row.quality_loss or 0.0 for row in matched) / len(matched)
        return self.default_loss

