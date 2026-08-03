from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


BASELINE_POLICIES = [
    "full_lru",
    "static_best",
    "static_safe",
    "quality_oracle",
    "utility_dynamic",
    "uncalibrated_dynamic",
]


def build_budget_series(start_mib: float, stop_mib: float, step_mib: float) -> list[float]:
    if step_mib <= 0:
        raise ValueError("--budget-step-mib 必须大于 0")
    if start_mib <= 0:
        raise ValueError("--budget-start-mib 必须大于 0")
    if stop_mib < start_mib:
        raise ValueError("--budget-stop-mib 必须大于等于 --budget-start-mib")

    budgets: list[float] = []
    current = float(start_mib)
    stop = float(stop_mib)
    step = float(step_mib)

    while current <= stop + 1e-9:
        budgets.append(float(current))
        current += step

    return budgets


def build_mem_test_config(
    base_config: dict[str, Any],
    *,
    max_requests: int,
    budgets_mib: list[float],
    include_tailguard: bool = False,
) -> dict[str, Any]:
    if max_requests <= 0:
        raise ValueError("--max-requests 必须大于 0")
    if not budgets_mib:
        raise ValueError("memory budgets 不能为空")

    config = deepcopy(base_config)

    policies = config.setdefault("policies", {})
    policy_names = list(BASELINE_POLICIES)
    if include_tailguard:
        policy_names.insert(3, "tailguard")
    policies["names"] = policy_names
    policies["record_rejected_unsafe"] = True

    data = config.setdefault("data", {})
    data["max_requests"] = int(max_requests)

    pilot = config.setdefault("pilot", {})
    pilot["memory_budgets_mib"] = list(budgets_mib)

    outputs = config.setdefault("outputs", {})
    outputs["smoke_profiles"] = "out/profile_tables/run_mem_test_profiles.csv"
    outputs["smoke_profile_checks"] = "out/profile_tables/run_mem_test_profile_checks.csv"
    outputs["smoke_policy"] = "out/policy_tables/run_mem_test_policy.csv"
    outputs["smoke_summary"] = "out/policy_tables/run_mem_test_summary.csv"

    return config


def write_generated_config(config: dict[str, Any], run_dir: Path) -> Path:
    path = run_dir / "configs" / "run_mem_test.generated.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path
