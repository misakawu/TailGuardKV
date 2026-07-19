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
            ttfts: list[float] = []
            memories: list[float] = []
            losses: list[float] = []
            pred_losses: list[float] = []
            risk_uppers: list[float] = []
            controller_overheads: list[float] = []
            qrp_overheads: list[float] = []
            cg_overheads: list[float] = []
            stc_overheads: list[float] = []
            oracle_costs: list[float] = []
            optimality_gaps: list[float] = []
            audit_rates: list[float] = []
            drift_states: Counter[str] = Counter()
            safe_count = 0
            fallback_count = 0
            violation_count = 0
            known_count = 0
            exact_count = 0
            ok_count = 0
            actions: Counter[str] = Counter()
            grouped_request = defaultdict(list)
            for row in rows:
                if row.ok:
                    ok_count += 1
                if row.ttft_ms is not None:
                    ttfts.append(row.ttft_ms)
                if row.peak_memory_mib is not None:
                    memories.append(row.peak_memory_mib)
                if row.quality_loss is not None:
                    losses.append(row.quality_loss)
                    known_count += 1
                    if row.quality_loss > epsilon:
                        violation_count += 1
                if row.pred_loss is not None:
                    pred_losses.append(row.pred_loss)
                if row.risk_upper is not None:
                    risk_uppers.append(row.risk_upper)
                if row.controller_overhead_ms is not None:
                    controller_overheads.append(row.controller_overhead_ms)
                if row.controller_qrp_ms is not None:
                    qrp_overheads.append(row.controller_qrp_ms)
                if row.controller_cg_ms is not None:
                    cg_overheads.append(row.controller_cg_ms)
                if row.controller_stc_ms is not None:
                    stc_overheads.append(row.controller_stc_ms)
                if row.oracle_cost_ms is not None:
                    oracle_costs.append(row.oracle_cost_ms)
                if row.optimality_gap is not None:
                    optimality_gaps.append(row.optimality_gap)
                if row.audit_rate is not None:
                    audit_rates.append(row.audit_rate)
                if row.drift_state:
                    drift_states[row.drift_state] += 1
                if row.safe is True:
                    safe_count += 1
                if row.fallback_reason:
                    fallback_count += 1
                if row.action_profile in exact_profiles:
                    exact_count += 1
                actions[row.action_profile] += 1
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
                "ok_count": float(ok_count),
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
                "safe_ratio": safe_count / len(rows) if rows else float("nan"),
                "fallback_ratio": fallback_count / len(rows) if rows else float("nan"),
                "exact_fallback_ratio": exact_count / len(rows) if rows else float("nan"),
                "lossy_action_ratio": (len(rows) - exact_count) / len(rows) if rows else float("nan"),
                "target_delta": delta,
                "violation_rate": violation_count / known_count if known_count else float("nan"),
                "delta_slack": _delta_slack(rows, epsilon, delta),
                "worst_group_violation": worst_group_violation,
                "action_distribution": dict(actions),
                "controller_overhead_ms": _mean(controller_overheads),
                "controller_qrp_ms": _mean(qrp_overheads),
                "controller_cg_ms": _mean(cg_overheads),
                "controller_stc_ms": _mean(stc_overheads),
                "oracle_cost_ms": _mean(oracle_costs),
                "optimality_gap": _mean(optimality_gaps),
                "audit_rate": _mean(audit_rates),
                "drift_state": drift_states.most_common(1)[0][0] if drift_states else "",
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
