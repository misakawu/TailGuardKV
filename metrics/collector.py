from __future__ import annotations

from collections import Counter
from math import isnan
from typing import Any

from core_types import PolicyRunRecord, ProfileMeasurement


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
                "p95_ttft_ms": _percentile(
                    [row.ttft_ms for row in rows if row.ttft_ms is not None],
                    0.95,
                ),
                "mean_peak_memory_mib": sum(memories) / len(memories) if memories else float("nan"),
            }
        return summary

    def summarize_policy_runs(
        self,
        records: list[PolicyRunRecord],
        epsilon: float,
        exact_profiles: set[str],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[PolicyRunRecord]] = {}
        for record in records:
            grouped.setdefault(record.policy, []).append(record)

        summary: dict[str, dict[str, Any]] = {}
        for policy, rows in grouped.items():
            ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
            memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
            known_rows = [row for row in rows if row.quality_loss is not None]
            losses = [row.quality_loss for row in known_rows if row.quality_loss is not None]
            lossy_rows = [row for row in rows if row.action_profile not in exact_profiles]
            violations = [
                row for row in rows if row.quality_loss is not None and row.quality_loss > epsilon
            ]
            actions = Counter(row.action_profile for row in rows)
            quality_coverage_ratio = len(known_rows) / len(rows) if rows else float("nan")
            summary[policy] = {
                "count": float(len(rows)),
                "ok_count": float(sum(1 for row in rows if row.ok)),
                "placeholder": any(row.placeholder for row in rows),
                "oracle": any(row.oracle for row in rows),
                "mean_ttft_ms": _mean(ttfts),
                "p50_ttft_ms": _percentile(ttfts, 0.50),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p99_ttft_ms": _percentile(ttfts, 0.99),
                "mean_peak_memory_mib": _mean(memories),
                "max_peak_memory_mib": max(memories) if memories else float("nan"),
                "mean_quality_loss": _mean(losses),
                "unknown_quality_count": float(len(rows) - len(known_rows)),
                "quality_coverage_ratio": quality_coverage_ratio,
                "valid_for_slo": quality_coverage_ratio >= 0.95 if rows else False,
                "violation_rate": len(violations) / len(known_rows) if known_rows else float("nan"),
                "exact_fallback_ratio": (
                    sum(1 for row in rows if row.action_profile in exact_profiles) / len(rows)
                    if rows
                    else float("nan")
                ),
                "lossy_action_ratio": len(lossy_rows) / len(rows) if rows else float("nan"),
                "action_distribution": dict(actions),
                "coverage_target_delta": float("nan"),
                "controller_overhead_ms": float("nan"),
            }
        return summary


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if not isnan(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if not isnan(value))
    if not finite_values:
        return float("nan")
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]
