from __future__ import annotations

from collections.abc import Iterable

from backends.base import Backend
from core_types import ProfileMeasurement, Request


class MeasuredReplayBackend(Backend):
    """用实测 profile 表回放，避免 Pilot 策略层直接依赖重型引擎。"""

    name = "measured_replay"

    def __init__(self, measurements: Iterable[ProfileMeasurement]) -> None:
        self.measurements = {
            (measurement.request_id, measurement.profile): measurement for measurement in measurements
        }

    def run(self, requests: list[Request], profiles: list[str]) -> list[ProfileMeasurement]:
        rows: list[ProfileMeasurement] = []
        for request in requests:
            for profile in profiles:
                key = (request.request_id, profile)
                if key not in self.measurements:
                    raise KeyError(f"缺少回放数据: request={key[0]} profile={key[1]}")
                rows.append(self.measurements[key])
        return rows
