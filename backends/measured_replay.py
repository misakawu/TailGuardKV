from __future__ import annotations

from collections.abc import Iterable

from backends.base import Backend
from run_util.core_types import BackendResult, CacheState, ProfileMeasurement, Request


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
        self.cache_state = CacheState()
        if use_pandas:
            try:
                import pandas as pd
            except ModuleNotFoundError:
                self.measurements = {self._measurement_key(measurement): measurement for measurement in rows}
            else:
                self._use_pandas = True
                self._frame = pd.DataFrame(
                    {
                        "request_id": measurement.request_id,
                        "session_id": measurement.session_id or "",
                        "turn_index": measurement.turn_index,
                        "profile": measurement.profile,
                        "measurement": measurement,
                    }
                    for measurement in rows
                ).set_index(["session_id", "turn_index", "request_id", "profile"])
                self.measurements = {}
        else:
            self.measurements = {self._measurement_key(measurement): measurement for measurement in rows}

    def run(self, requests: list[Request], profiles: list[str]) -> list[BackendResult]:
        if len(profiles) not in {1, len(requests)}:
            raise ValueError("profiles 长度必须为 1 或与 requests 一致")
        rows: list[BackendResult] = []
        for index, request in enumerate(requests):
            profile = profiles[0] if len(profiles) == 1 else profiles[index]
            key = self._request_key(request, profile)
            if self._use_pandas:
                try:
                    measurement = self._frame.loc[key, "measurement"]
                except KeyError as exc:
                    raise KeyError(f"缺少回放数据: session={key[0] or '-'} turn={key[1]} request={key[2]} profile={key[3]}") from exc
            else:
                if key not in self.measurements:
                    raise KeyError(f"缺少回放数据: session={key[0] or '-'} turn={key[1]} request={key[2]} profile={key[3]}")
                measurement = self.measurements[key]
            result = BackendResult.from_profile_measurement(
                measurement,
                backend_name=self.name,
                replay_source="measured_profile_table",
                extra={"source_profile": measurement.profile},
            )
            self.cache_state = self.cache_state.with_session_turn(
                request.session_id,
                profile=profile,
                turn_index=request.turn_index,
                cumulative_kv_mib=result.kv_cumulative_mib or result.kv_cache_memory_mib or 0.0,
                resident_kv_mib=result.resident_kv_mib_after or result.resident_memory_mib or result.kv_cumulative_mib or 0.0,
            )
            rows.append(result)
        return rows

    @staticmethod
    def _measurement_key(measurement: ProfileMeasurement) -> tuple[str, int, str, str]:
        return (measurement.session_id or "", measurement.turn_index, measurement.request_id, measurement.profile)

    @staticmethod
    def _request_key(request: Request, profile: str) -> tuple[str, int, str, str]:
        return (request.session_id or "", request.turn_index, request.request_id, profile)
