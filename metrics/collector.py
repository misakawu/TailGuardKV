from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan
from typing import Any

from core_types import PolicyRunRecord, ProfileMeasurement


class MetricCollector:
    """汇总 profile / policy 结果，覆盖 H0/H1/H2-lite 指标。"""

    def summarize_profiles(self, measurements: list[ProfileMeasurement]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[ProfileMeasurement]] = defaultdict(list)
        for measurement in measurements:
            grouped[measurement.profile].append(measurement)

        summary: dict[str, dict[str, float]] = {}
        for profile, rows in grouped.items():
            losses = [row.quality_loss for row in rows if row.quality_loss is not None]
            ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
            memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
            summary[profile] = {
                "count": float(len(rows)),
                "ok_count": float(sum(1 for row in rows if row.ok)),
                "measured_count": float(sum(1 for row in rows if row.measured)),
                "mean_quality_loss": _mean(losses),
                "p95_quality_loss": _percentile(losses, 0.95),
                "p99_quality_loss": _percentile(losses, 0.99),
                "cvar_quality_loss": _cvar(losses, 0.95),
                "violation_rate": _violation_rate(losses, 0.5),
                "mean_ttft_ms": _mean(ttfts),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p99_ttft_ms": _percentile(ttfts, 0.99),
                "mean_peak_memory_mib": _mean(memories),
                "p95_peak_memory_mib": _percentile(memories, 0.95),
            }
        return summary

    def summarize_policy_runs(
        self,
        records: list[PolicyRunRecord],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[PolicyRunRecord]] = defaultdict(list)
        for record in records:
            grouped[record.policy].append(record)

        summary: dict[str, dict[str, Any]] = {}
        for policy, rows in grouped.items():
            ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
            memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
            losses = [row.quality_loss for row in rows if row.quality_loss is not None]
            pred_losses = [row.pred_loss for row in rows if row.pred_loss is not None]
            risk_uppers = [row.risk_upper for row in rows if row.risk_upper is not None]
            safe_rows = [row for row in rows if row.safe is True]
            fallback_rows = [row for row in rows if row.fallback_reason]
            violations = [row for row in rows if row.quality_loss is not None and row.quality_loss > epsilon]
            known_rows = [row for row in rows if row.quality_loss is not None]
            exact_count = sum(1 for row in rows if row.action_profile in exact_profiles)
            actions = Counter(row.action_profile for row in rows)
            grouped_request = defaultdict(list)
            for row in rows:
                group_key = (
                    row.task or "unknown",
                    row.length_bucket or "unknown",
                    row.action_profile or "unknown",
                )
                grouped_request[group_key].append(row)
            worst_group_violation = max(
                (
                    _violation_rate([row.quality_loss for row in group if row.quality_loss is not None], epsilon)
                    for group in grouped_request.values()
                ),
                default=float("nan"),
            )
            summary[policy] = {
                "count": float(len(rows)),
                "ok_count": float(sum(1 for row in rows if row.ok)),
                "mean_ttft_ms": _mean(ttfts),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p99_ttft_ms": _percentile(ttfts, 0.99),
                "mean_peak_memory_mib": _mean(memories),
                "p95_peak_memory_mib": _percentile(memories, 0.95),
                "mean_quality_loss": _mean(losses),
                "p95_quality_loss": _percentile(losses, 0.95),
                "p99_quality_loss": _percentile(losses, 0.99),
                "cvar_quality_loss": _cvar(losses, 0.95),
                "pred_loss_mean": _mean(pred_losses),
                "risk_upper_mean": _mean(risk_uppers),
                "safe_ratio": len(safe_rows) / len(rows) if rows else float("nan"),
                "fallback_ratio": len(fallback_rows) / len(rows) if rows else float("nan"),
                "exact_fallback_ratio": exact_count / len(rows) if rows else float("nan"),
                "lossy_action_ratio": (len(rows) - exact_count) / len(rows) if rows else float("nan"),
                "target_delta": delta,
                "violation_rate": len(violations) / len(known_rows) if known_rows else float("nan"),
                "delta_slack": _delta_slack(rows, epsilon, delta),
                "worst_group_violation": worst_group_violation,
                "action_distribution": dict(actions),
                "controller_overhead_ms": _mean([row.controller_overhead_ms for row in rows if row.controller_overhead_ms is not None]),
                "oracle": any(row.oracle for row in rows),
                "placeholder": any(row.placeholder for row in rows),
            }
        return summary


def _mean(values: list[float]) -> float:
    finite_values = [value for value in values if value is not None and not isnan(value)]
    return sum(finite_values) / len(finite_values) if finite_values else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value is not None and not isnan(value))
    if not finite_values:
        return float("nan")
    index = min(len(finite_values) - 1, max(0, int(round((len(finite_values) - 1) * quantile))))
    return finite_values[index]


def _cvar(values: list[float], quantile: float) -> float:
    finite_values = sorted(value for value in values if value is not None and not isnan(value))
    if not finite_values:
        return float("nan")
    threshold = _percentile(finite_values, quantile)
    tail = [value for value in finite_values if value >= threshold]
    return _mean(tail)


def _violation_rate(losses: list[float], epsilon: float) -> float:
    finite_values = [value for value in losses if value is not None and not isnan(value)]
    return sum(1 for value in finite_values if value > epsilon) / len(finite_values) if finite_values else float("nan")


def _delta_slack(rows: list[PolicyRunRecord], epsilon: float, delta: float) -> float:
    known_rows = [row for row in rows if row.quality_loss is not None]
    if not known_rows:
        return float("nan")
    violation_rate = sum(1 for row in known_rows if row.quality_loss is not None and row.quality_loss > epsilon) / len(known_rows)
    return delta - violation_rate
