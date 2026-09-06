from __future__ import annotations

from collections import Counter, defaultdict
from math import isnan
from typing import Any

from run_util.core_types import PolicyRunRecord, ProfileMeasurement


class MetricCollector:
    """汇总离线 profile 证据和 backend 驱动的 policy 运行结果。"""

    def summarize_profiles(
        self,
        measurements: list[ProfileMeasurement],
        epsilons: list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, dict[str, Any]]:
        epsilons = list(epsilons or [0.05, 0.10])
        grouped: dict[str, list[ProfileMeasurement]] = defaultdict(list)
        for measurement in measurements:
            grouped[measurement.profile].append(measurement)

        summary: dict[str, dict[str, Any]] = {}
        for profile, rows in grouped.items():
            losses = [row.quality_loss for row in rows if row.quality_loss is not None]
            ttfts = [row.ttft_ms for row in rows if row.ttft_ms is not None]
            memories = [row.peak_memory_mib for row in rows if row.peak_memory_mib is not None]
            kv_memories = [row.kv_cache_memory_mib for row in rows if row.kv_cache_memory_mib is not None]
            metrics: dict[str, Any] = {
                "count": float(len(rows)),
                "ok_count": float(sum(1 for row in rows if row.ok)),
                "measured_count": float(sum(1 for row in rows if row.measured)),
                "mean_quality_loss": _mean(losses),
                "p95_quality_loss": _percentile(losses, 0.95),
                "p99_quality_loss": _percentile(losses, 0.99),
                "cvar_quality_loss": _cvar(losses, 0.95),
                "mean_ttft_ms": _mean(ttfts),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p99_ttft_ms": _percentile(ttfts, 0.99),
                "mean_peak_memory_mib": _mean(memories),
                "p95_peak_memory_mib": _percentile(memories, 0.95),
                "mean_kv_cache_memory_mib": _mean(kv_memories),
                "p95_kv_cache_memory_mib": _percentile(kv_memories, 0.95),
                "backend_distribution": _extra_distribution(rows, "backend"),
                "primary_metric_distribution": _extra_distribution(rows, "primary_metric"),
            }
            for epsilon in epsilons:
                metrics[f"violation_rate_eps{_slug_number(epsilon)}"] = _violation_rate(losses, epsilon)
            summary[profile] = metrics
        return summary

    def summarize_policy_runs(
        self,
        records: list[PolicyRunRecord],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        experiment_type: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[PolicyRunRecord]] = defaultdict(list)
        for record in records:
            grouped[record.policy].append(record)

        summary: dict[str, dict[str, Any]] = {}
        for policy, rows in grouped.items():
            ttfts: list[float] = []
            memories: list[float] = []
            kv_memories: list[float] = []
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
            restore_times: list[float] = []
            recompute_times: list[float] = []
            queue_delays: list[float] = []
            resident_kv_after: list[float] = []
            global_resident_kv: list[float] = []
            cumulative_kv: list[float] = []
            drift_states: Counter[str] = Counter()
            task_counts: Counter[str] = Counter()
            length_bucket_counts: Counter[str] = Counter()
            task_length_counts: Counter[str] = Counter()
            session_turn_counts: Counter[str] = Counter()
            safe_count = 0
            fallback_count = 0
            exact_fallback_count = 0
            unsafe_action_count = 0
            violation_count = 0
            known_count = 0
            exact_count = 0
            ok_count = 0
            actions: Counter[str] = Counter()
            profile_resident_totals: Counter[str] = Counter()
            candidate_safe_counts: list[float] = []
            budget_hit_count = 0
            policy_budget_filter_count = 0
            switch_count = 0
            restore_count = 0
            recompute_count = 0
            previous_profile_by_session: dict[str, str] = {}
            grouped_request = defaultdict(list)
            active_session_counts: list[float] = []
            evict_event_count = 0
            queue_event_count = 0
            for row in rows:
                if row.ok:
                    ok_count += 1
                if row.ttft_ms is not None:
                    ttfts.append(row.ttft_ms)
                if row.peak_memory_mib is not None:
                    memories.append(row.peak_memory_mib)
                if row.kv_cache_memory_mib is not None:
                    kv_memories.append(row.kv_cache_memory_mib)
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
                if row.restore_ms is not None and row.restore_ms > 0:
                    restore_times.append(row.restore_ms)
                    restore_count += 1
                if row.recompute_ms is not None and row.recompute_ms > 0:
                    recompute_times.append(row.recompute_ms)
                    recompute_count += 1
                if row.queue_delay_ms is not None:
                    queue_delays.append(row.queue_delay_ms)
                    if row.queue_delay_ms > 0:
                        queue_event_count += 1
                if row.resident_kv_mib_after is not None:
                    resident_kv_after.append(row.resident_kv_mib_after)
                if row.global_resident_kv_mib is not None:
                    global_resident_kv.append(row.global_resident_kv_mib)
                if row.kv_cumulative_mib is not None:
                    cumulative_kv.append(row.kv_cumulative_mib)
                if row.drift_state:
                    drift_states[row.drift_state] += 1
                if row.safe is True:
                    safe_count += 1
                if row.safe is False or row.rejected_profile:
                    unsafe_action_count += 1
                if row.candidate_safe_count is not None:
                    candidate_safe_counts.append(row.candidate_safe_count)
                if row.fallback_reason:
                    fallback_count += 1
                if row.backend_budget_hit or row.budget_hit:
                    budget_hit_count += 1
                if row.policy_budget_filtered:
                    policy_budget_filter_count += 1
                if row.action_profile in exact_profiles:
                    exact_count += 1
                    if row.fallback_reason:
                        exact_fallback_count += 1
                actions[row.action_profile] += 1
                task = row.task or "unknown"
                bucket = row.length_bucket or "unknown"
                task_counts[task] += 1
                length_bucket_counts[bucket] += 1
                task_length_counts[f"{task}/{bucket}"] += 1
                if row.session_id:
                    session_turn_counts[row.session_id] += 1
                    previous = previous_profile_by_session.get(row.session_id)
                    if previous is not None and row.resident_kv_mib_before is not None:
                        profile_resident_totals[previous] += row.resident_kv_mib_before
                    if previous is not None and previous != row.action_profile:
                        switch_count += 1
                    previous_profile_by_session[row.session_id] = row.action_profile
                if row.resident_kv_mib_after is not None:
                    profile_resident_totals[row.action_profile] += row.resident_kv_mib_after
                elif row.resident_memory_mib is not None:
                    profile_resident_totals[row.action_profile] += row.resident_memory_mib
                if row.evicted_kv_mib is not None and row.evicted_kv_mib > 0:
                    evict_event_count += 1
                if row.active_session_count is not None:
                    active_session_counts.append(row.active_session_count)
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
            is_quality = experiment_type == "baseline_quality"
            summary[policy] = {
                "count": float(len(rows)),
                "ok_count": float(ok_count),
                "mean_ttft_ms": _mean(ttfts),
                "p50_ttft_ms": _percentile(ttfts, 0.50),
                "p95_ttft_ms": _percentile(ttfts, 0.95),
                "p99_ttft_ms": _percentile(ttfts, 0.99),
                "mean_peak_memory_mib": _mean(memories),
                "p95_peak_memory_mib": _percentile(memories, 0.95),
                "mean_kv_cache_memory_mib": _mean(kv_memories),
                "p95_kv_cache_memory_mib": _percentile(kv_memories, 0.95),
                "mean_quality_loss": _mean(losses),
                "p50_quality_loss": _percentile(losses, 0.50),
                "p95_quality_loss": _percentile(losses, 0.95),
                "p99_quality_loss": _percentile(losses, 0.99),
                "cvar_quality_loss": _cvar(losses, 0.95),
                "pred_loss_mean": _mean(pred_losses),
                "risk_upper_mean": _mean(risk_uppers),
                "safe_ratio": safe_count / len(rows) if rows else float("nan"),
                "fallback_ratio": fallback_count / len(rows) if rows else float("nan"),
                "exact_fallback_ratio": exact_fallback_count / len(rows) if rows else float("nan"),
                "exact_action_ratio": exact_count / len(rows) if rows else float("nan"),
                "lossy_action_ratio": (len(rows) - exact_count) / len(rows) if rows else float("nan"),
                "unique_action_count": float(len(actions)),
                "identical_to_full_lru": bool(rows and all(row.action_profile == "full_gpu" for row in rows)),
                "unsafe_action_count": float(unsafe_action_count),
                "candidate_safe_count": _mean(candidate_safe_counts),
                "target_delta": delta,
                "violation_rate": violation_count / known_count if known_count else float("nan"),
                "delta_slack": _delta_slack(rows, epsilon, delta),
                "worst_group_violation": worst_group_violation,
                "action_distribution": dict(actions),
                "task_distribution": dict(sorted(task_counts.items())),
                "length_bucket_distribution": dict(sorted(length_bucket_counts.items())),
                "task_length_distribution": dict(sorted(task_length_counts.items())),
                "session_count": float(len(session_turn_counts)),
                "multi_turn_session_count": float(sum(1 for count in session_turn_counts.values() if count > 1)),
                "active_session_peak": max(active_session_counts) if active_session_counts else float("nan"),
                "controller_overhead_ms": _mean(controller_overheads),
                "controller_qrp_ms": _mean(qrp_overheads),
                "controller_cg_ms": _mean(cg_overheads),
                "controller_stc_ms": _mean(stc_overheads),
                "oracle_cost_ms": _mean(oracle_costs),
                "optimality_gap": _mean(optimality_gaps),
                "audit_rate": _mean(audit_rates),
                "switch_count": float(switch_count),
                "budget_hit_rate": budget_hit_count / len(rows) if rows else float("nan"),
                "policy_budget_filter_rate": policy_budget_filter_count / len(rows) if rows else float("nan"),
                "restore_count": float(restore_count),
                "restore_time_ms": _mean(restore_times),
                "restore_time_ms_total": _total(restore_times),
                "recompute_count": float(recompute_count),
                "recompute_time_ms": _mean(recompute_times),
                "recompute_time_ms_total": _total(recompute_times),
                "queue_delay_ms": _mean(queue_delays),
                "queue_event_count": float(queue_event_count),
                "evict_event_count": float(evict_event_count),
                "budget_hit_count": float(budget_hit_count),
                "triggered_restore": bool(restore_count),
                "triggered_recompute": bool(recompute_count),
                "triggered_evict": bool(evict_event_count),
                "triggered_queue": bool(queue_event_count),
                "mean_resident_kv_mib": _mean(resident_kv_after),
                "mean_global_resident_kv_mib": _mean(global_resident_kv),
                "p95_global_resident_kv_mib": _percentile(global_resident_kv, 0.95),
                "mean_cumulative_kv_mib": _mean(cumulative_kv),
                "profile_residence_share": _share(profile_resident_totals),
                "drift_state": drift_states.most_common(1)[0][0] if drift_states else "",
                "oracle": any(row.oracle for row in rows),
                "placeholder": any(row.placeholder for row in rows),
                "backend_events_applicable": not is_quality,
                "backend_semantics_status": "not_applicable" if is_quality else "validated",
                "backend_not_applicable_reason": (
                    "baseline_quality 使用独立质量请求，不验收 session/backend 事件"
                    if is_quality
                    else ""
                ),
            }
            if is_quality:
                for field in (
                    "budget_hit_rate",
                    "budget_hit_count",
                    "restore_count",
                    "restore_time_ms",
                    "restore_time_ms_total",
                    "recompute_count",
                    "recompute_time_ms",
                    "recompute_time_ms_total",
                    "queue_delay_ms",
                    "queue_event_count",
                    "evict_event_count",
                    "triggered_restore",
                    "triggered_recompute",
                    "triggered_evict",
                    "triggered_queue",
                    "mean_resident_kv_mib",
                    "mean_global_resident_kv_mib",
                    "p95_global_resident_kv_mib",
                    "mean_cumulative_kv_mib",
                ):
                    summary[policy][field] = None
            summary[policy].update(
                {
                    "session_reuse_evidence": bool(
                        session_turn_counts and any(count > 1 for count in session_turn_counts.values())
                    ),
                    "global_resident_evolution": len(set(global_resident_kv)) >= 2,
                    "backend_event_evidence": bool(
                        budget_hit_count
                        or evict_event_count
                        or restore_count
                        or recompute_count
                        or queue_event_count
                    ),
                }
            )
        return summary


def _total(values: list[float]) -> float:
    return sum(value for value in values if value is not None and not isnan(value))


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


def _slug_number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _extra_distribution(rows: list[ProfileMeasurement], key: str) -> dict[str, int]:
    counts = Counter(str(row.extra.get(key)) for row in rows if row.extra.get(key) not in {None, ""})
    return dict(sorted(counts.items()))


def _delta_slack(rows: list[PolicyRunRecord], epsilon: float, delta: float) -> float:
    known_rows = [row for row in rows if row.quality_loss is not None]
    if not known_rows:
        return float("nan")
    violation_rate = sum(1 for row in known_rows if row.quality_loss is not None and row.quality_loss > epsilon) / len(known_rows)
    return delta - violation_rate


def _share(counts: Counter[str]) -> dict[str, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}
