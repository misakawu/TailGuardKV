from __future__ import annotations

from collections.abc import Iterable

from backends.base import Backend
from core_types import ProfileMeasurement, Request


class MeasuredReplayBackend(Backend):
    """用实测 profile 表回放，避免 Pilot 策略层直接依赖重型引擎。"""

    name = "measured_replay"

    def __init__(
        self,
        measurements: Iterable[ProfileMeasurement],
        allow_dry_run: bool = False,
        use_pandas: bool = False,
    ) -> None:
        rows = list(measurements)
        dry_rows = [measurement for measurement in rows if not measurement.measured]
        if dry_rows and not allow_dry_run:
            sample = dry_rows[0]
            raise ValueError(
                "MeasuredReplayBackend 默认只接受 measured=True 的实测 profile 表；"
                f"发现 dry-run 行: request={sample.request_id} profile={sample.profile}。"
                "如仅做 smoke，请显式传入 allow_dry_run=True。"
            )
        self._use_pandas = False
        self._frame = None
        if use_pandas:
            try:
                import pandas as pd
            except ModuleNotFoundError:
                self.measurements = {
                    (measurement.request_id, measurement.profile): measurement for measurement in rows
                }
            else:
                self._use_pandas = True
                self._frame = pd.DataFrame(
                    {
                        "request_id": measurement.request_id,
                        "profile": measurement.profile,
                        "measurement": measurement,
                    }
                    for measurement in rows
                ).set_index(["request_id", "profile"])
                self.measurements = {}
        else:
            self.measurements = {
                (measurement.request_id, measurement.profile): measurement for measurement in rows
            }

    def run(self, requests: list[Request], profiles: list[str]) -> list[ProfileMeasurement]:
        rows: list[ProfileMeasurement] = []
        for request in requests:
            for profile in profiles:
                key = (request.request_id, profile)
                if self._use_pandas:
                    try:
                        rows.append(self._frame.loc[key, "measurement"])
                    except KeyError as exc:
                        raise KeyError(f"缺少回放数据: request={key[0]} profile={key[1]}") from exc
                    continue
                if key not in self.measurements:
                    raise KeyError(f"缺少回放数据: request={key[0]} profile={key[1]}")
                rows.append(self.measurements[key])
        return rows
