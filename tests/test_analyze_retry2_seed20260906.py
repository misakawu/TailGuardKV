from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "scripts" / "analyze_retry2_seed20260906.py"
    spec = importlib.util.spec_from_file_location("retry2_seed_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_attempt(root: Path, *, requests_per_policy: int) -> Path:
    seed = "20260906"
    policy_dir = root / f"seed{seed}" / "policy_tables"
    policy_dir.mkdir(parents=True)
    fields = [
        "policy", "request_id", "action_profile", "ok", "measured",
        "diagnostic_only", "backend_name", "session_id", "ttft_ms",
        "kv_cache_memory_mib", "resident_kv_mib_after", "budget_hit",
        "policy_budget_filtered", "restore_ms", "recompute_ms", "queue_delay_ms",
        "evicted_kv_mib", "epsilon", "delta", "config", "run_dir",
        "quality_status", "violation_status",
    ]
    cells = []
    for memory in (120, 469, 1078, 1362):
        for epsilon in ("0p05", "0p1"):
            for delta in ("0p05", "0p1"):
                output = policy_dir / f"session25_policy_eps{epsilon}_delta{delta}_mem{memory}.csv"
                with output.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    for policy in ("full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"):
                        for index in range(requests_per_policy):
                            writer.writerow({
                                "policy": policy, "request_id": f"session-{index:03d}-turn-0",
                                "action_profile": "full_gpu", "ok": "True", "measured": "True",
                                "diagnostic_only": "True", "backend_name": "online_qwen",
                                "session_id": f"session-{index:03d}", "ttft_ms": "10.0",
                                "kv_cache_memory_mib": "2.0", "resident_kv_mib_after": "2.0",
                                "budget_hit": "False", "policy_budget_filtered": "False",
                                "restore_ms": "0.0", "recompute_ms": "0.0", "queue_delay_ms": "0.0",
                                "evicted_kv_mib": "0.0", "epsilon": epsilon.replace("p", "."), "delta": delta.replace("p", "."),
                                "config": "test.yaml", "run_dir": str(root),
                                "quality_status": "risk_evidence_insufficient",
                                "violation_status": "risk_evidence_insufficient",
                            })
                cells.append({
                    "seed": int(seed), "state": "success", "output": str(output),
                    "policies": ["full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"],
                    "evaluation_requests": 65, "expected_requests_per_policy": 65,
                    "diagnostic_only": True,
                })
    (root / "fourth_grid_status.json").write_text(json.dumps({"diagnostic_only": True, "cells": cells}), encoding="utf-8")
    return root


def test_audit_seed_attempt_rejects_missing_policy_request(tmp_path: Path) -> None:
    module = _load_module()
    attempt = _write_attempt(tmp_path / "attempt", requests_per_policy=64)

    with pytest.raises(ValueError, match="65 unique evaluation requests"):
        module.audit_seed_attempt(attempt, "20260906")


def test_generate_analysis_writes_reproducible_artifacts(tmp_path: Path) -> None:
    module = _load_module()
    attempt = _write_attempt(tmp_path / "attempt", requests_per_policy=65)

    report = module.generate_analysis(attempt, tmp_path / "analysis", "20260906")

    assert report["policy_rows"] == 80
    assert (tmp_path / "analysis" / "audit.json").is_file()
    assert (tmp_path / "analysis" / "baseline_smoke.md").is_file()
    assert (tmp_path / "analysis" / "action_distribution.png").is_file()
    assert (tmp_path / "analysis" / "event_rates.png").is_file()


def test_cli_can_start_from_repository_root() -> None:
    root = Path(__file__).parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/analyze_retry2_seed20260906.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--attempt-root" in result.stdout
