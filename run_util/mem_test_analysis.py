from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


NON_ORACLE_POLICIES = {"static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"}


def analyze_mem_test_summary(summary_path: Path, *, kv_drop_threshold: float = 0.5) -> dict[str, Any]:
    rows = _read_rows(summary_path)
    by_key = {
        (row["policy"], _float(row["epsilon"]), _float(row["delta"]), _float(row["memory_budget_mib"])): row
        for row in rows
    }
    passing: list[dict[str, Any]] = []
    near_misses: list[dict[str, Any]] = []

    for row in rows:
        if row["policy"] not in NON_ORACLE_POLICIES:
            continue

        action_distribution = _action_distribution(row)
        lossy_actions = {name: count for name, count in action_distribution.items() if name != "full_gpu"}
        if not lossy_actions:
            continue

        key = ("full_lru", _float(row["epsilon"]), _float(row["delta"]), _float(row["memory_budget_mib"]))
        full = by_key.get(key)
        if full is None:
            continue

        record = _comparison_record(row, full, lossy_actions)
        kv_ok = record["kv_drop_mean"] >= kv_drop_threshold
        mean_ttft_ok = _float(row["mean_ttft_ms"]) < _float(full["mean_ttft_ms"])
        p95_ttft_ok = _float(row["p95_ttft_ms"]) < _float(full["p95_ttft_ms"])
        record["kv_ok"] = kv_ok
        record["mean_ttft_ok"] = mean_ttft_ok
        record["p95_ttft_ok"] = p95_ttft_ok

        if kv_ok and (mean_ttft_ok or p95_ttft_ok):
            record["ttft_win_metric"] = "mean_ttft_ms" if mean_ttft_ok else "p95_ttft_ms"
            passing.append(record)
        else:
            near_misses.append(record)

    passing.sort(key=lambda item: (-item["memory_budget_mib"], item["policy"], item["epsilon"], item["delta"]))
    near_misses.sort(key=lambda item: (-max(item["kv_drop_mean"], item["kv_drop_p95"]), item["mean_ttft_delta_ms"]))

    return {
        "summary_path": str(summary_path),
        "found_passing_budget": bool(passing),
        "passing_points": passing,
        "near_misses": near_misses[:10],
        "policy_names": sorted({row["policy"] for row in rows}),
        "budget_count": len({_float(row["memory_budget_mib"]) for row in rows}),
        "epsilon_delta_count_by_budget": _epsilon_delta_count_by_budget(rows),
    }


def write_mem_test_analysis(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# run_mem_test 显存预算分析", ""]
    lines.append(f"- summary: `{analysis['summary_path']}`")
    lines.append(f"- found_passing_budget: `{str(analysis['found_passing_budget']).lower()}`")

    if analysis["passing_points"]:
        lines.extend(["", "## 通过点", ""])
        for point in analysis["passing_points"]:
            lines.append(
                f"- budget={point['memory_budget_mib']} MiB, policy={point['policy']}, "
                f"epsilon={point['epsilon']}, delta={point['delta']}, "
                f"kv_drop_mean={point['kv_drop_mean']:.3f}, kv_drop_p95={point['kv_drop_p95']:.3f}, "
                f"mean_ttft_delta_ms={point['mean_ttft_delta_ms']:.3f}, "
                f"p95_ttft_delta_ms={point['p95_ttft_delta_ms']:.3f}, "
                f"actions={json.dumps(point['lossy_actions'], ensure_ascii=False, sort_keys=True)}"
            )
    else:
        lines.extend(["", "## 结论", "", "未找到同时满足 KIVI/H2O 显存收益和 TTFT 系统收益的非 oracle budget 点。"])

    if analysis["near_misses"]:
        lines.extend(["", "## 近似点", ""])
        for point in analysis["near_misses"]:
            lines.append(
                f"- budget={point['memory_budget_mib']} MiB, policy={point['policy']}, "
                f"kv_drop_mean={point['kv_drop_mean']:.3f}, kv_drop_p95={point['kv_drop_p95']:.3f}, "
                f"mean_ttft_delta_ms={point['mean_ttft_delta_ms']:.3f}, "
                f"p95_ttft_delta_ms={point['p95_ttft_delta_ms']:.3f}"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value)


def _action_distribution(row: dict[str, str]) -> dict[str, int]:
    raw = row.get("action_distribution") or "{}"
    payload = json.loads(raw)
    return {str(key): int(value) for key, value in payload.items()}


def _comparison_record(row: dict[str, str], full: dict[str, str], lossy_actions: dict[str, int]) -> dict[str, Any]:
    mean_kv = _float(row["mean_kv_cache_memory_mib"])
    p95_kv = _float(row["p95_kv_cache_memory_mib"])
    full_mean_kv = _float(full["mean_kv_cache_memory_mib"])
    full_p95_kv = _float(full["p95_kv_cache_memory_mib"])
    return {
        "policy": row["policy"],
        "memory_budget_mib": _float(row["memory_budget_mib"]),
        "epsilon": _float(row["epsilon"]),
        "delta": _float(row["delta"]),
        "lossy_actions": lossy_actions,
        "mean_kv_cache_memory_mib": mean_kv,
        "full_mean_kv_cache_memory_mib": full_mean_kv,
        "p95_kv_cache_memory_mib": p95_kv,
        "full_p95_kv_cache_memory_mib": full_p95_kv,
        "kv_drop_mean": 1.0 - mean_kv / full_mean_kv if full_mean_kv else 0.0,
        "kv_drop_p95": 1.0 - p95_kv / full_p95_kv if full_p95_kv else 0.0,
        "mean_ttft_delta_ms": _float(row["mean_ttft_ms"]) - _float(full["mean_ttft_ms"]),
        "p95_ttft_delta_ms": _float(row["p95_ttft_ms"]) - _float(full["p95_ttft_ms"]),
        "violation_rate": _float(row["violation_rate"]),
        "candidate_safe_count": row.get("candidate_safe_count") or "",
    }


def _epsilon_delta_count_by_budget(rows: list[dict[str, str]]) -> dict[str, int]:
    grouped: dict[float, set[tuple[float, float]]] = {}
    for row in rows:
        grouped.setdefault(_float(row["memory_budget_mib"]), set()).add((_float(row["epsilon"]), _float(row["delta"])))
    return {str(budget): len(values) for budget, values in sorted(grouped.items())}
