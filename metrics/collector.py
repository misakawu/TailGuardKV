from __future__ import annotations

from core_types import ProfileMeasurement


class MetricCollector:
    """先提供 profile 表汇总，后续扩展 coverage、TTFT、memory、overhead。"""

    def summarize_profiles(self, measurements: list[ProfileMeasurement]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[ProfileMeasurement]] = {}
        for measurement in measurements:
            grouped.setdefault(measurement.profile, []).append(measurement)

        summary: dict[str, dict[str, float]] = {}
        for profile, rows in grouped.items():
            ok_count = sum(1 for row in rows if row.ok)
            latencies = [row.latency_ms for row in rows if row.latency_ms is not None]
            memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
            summary[profile] = {
                "count": float(len(rows)),
                "ok_count": float(ok_count),
                "mean_latency_ms": sum(latencies) / len(latencies) if latencies else float("nan"),
                "mean_peak_memory_mib": sum(memories) / len(memories) if memories else float("nan"),
            }
        return summary
