#!/usr/bin/env python3
"""Run the diagnostic-only fourth-experiment grid after complete profile measurement."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_session25_fourth import audit_fixture
from scripts.aggregate_session27_baselines import aggregate_session27
from run_util.data_utils import split_measurements
from run_util.experiment_common import read_measurements


SEEDS = (20260906,)
PARAMETERS = (0.05, 0.10)
POLICIES = ("full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic")
POLICY_CSV_REQUIRED_FIELDS = {"request_id", "policy", "diagnostic_only", "backend_name", "ok", "split_seed"}
# Sorted schema emitted by run_util.run_policies for this diagnostic online policy run.
POLICY_OUTPUT_HEADER = (
    "action_profile", "active_session_count", "audit_rate", "audit_selected", "backend_budget_hit",
    "backend_name", "budget_hit", "candidate_safe_count", "config", "controller_cg_ms",
    "controller_overhead_ms", "controller_qrp_ms", "controller_stc_ms", "delta", "diagnostic_only",
    "drift_state", "epsilon", "error", "evicted_kv_mib", "exact", "extra_warm_profile",
    "extra_warm_profile_success", "extra_worker_cuda_visible_devices", "extra_worker_device_strategy",
    "extra_worker_generation", "extra_worker_mode", "extra_worker_model_load_ms", "extra_worker_pid",
    "extra_worker_startup_ms", "fallback_reason", "global_budget_mib", "global_resident_kv_mib",
    "kv_cache_memory_mib", "kv_cumulative_mib", "kv_incremental_mib", "latency_ms", "length_bucket",
    "measured", "observed_quality_loss", "ok", "optimality_gap", "oracle", "oracle_cost_ms",
    "peak_memory_mib", "placeholder", "policy", "policy_budget_filtered", "pred_loss",
    "predicted_quality_loss", "primary_profile", "quality_estimate", "quality_loss", "quality_status",
    "queue_delay_ms", "reason", "recompute_ms", "rejected_pred_loss", "rejected_profile",
    "rejected_risk_upper", "request_id", "resident_kv_mib_after", "resident_kv_mib_before",
    "resident_memory_mib", "restore_ms", "risk_upper", "run_dir", "safe", "safety_reason",
    "session_id", "task", "ttft_ms", "turn_index", "violation_status", "split_seed",
)


def cells(
    budgets: list[float], parameters: tuple[float, ...] = PARAMETERS
) -> list[tuple[int, float, float, float]]:
    if not budgets or len(budgets) != len(set(budgets)):
        raise ValueError("expected one or more distinct budgets")
    return list(product(SEEDS, budgets, parameters, parameters))


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _request_split_audit(
    measurements: list,
    *,
    split_seed: int,
    calibration_fraction: float,
    stratify_session: bool,
) -> dict[str, object]:
    calibration, evaluation = split_measurements(
        measurements,
        split_seed=split_seed,
        stratify_session=stratify_session,
        calibration_fraction=calibration_fraction,
    )
    total_ids = {row.request_id for row in measurements}
    calibration_ids = {row.request_id for row in calibration}
    evaluation_ids = {row.request_id for row in evaluation}
    if calibration_ids & evaluation_ids or total_ids != calibration_ids | evaluation_ids:
        raise ValueError("measurement split does not partition fixture request IDs")
    return {
        "total_request_count": len(total_ids),
        "calibration_request_count": len(calibration_ids),
        "evaluation_request_count": len(evaluation_ids),
        "evaluation_request_ids": sorted(evaluation_ids),
    }


def _read_policy_csv(path: Path) -> tuple[list[str], list[dict[str, str | None]]] | None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, csv.Error):
        return None
    if not fieldnames or len(fieldnames) != len(set(fieldnames)) or any(not field for field in fieldnames):
        return None
    if any(None in row for row in rows):
        return None
    return fieldnames, rows


def _policy_csv_rows_are_complete(rows: list[dict[str, str | None]], seed: int, expected_request_ids: set[str]) -> bool:
    counts = Counter(str(row.get("policy") or "") for row in rows)
    request_ids_by_policy = {
        policy: [str(row.get("request_id") or "") for row in rows if str(row.get("policy") or "") == policy]
        for policy in POLICIES
    }
    return counts == {policy: len(expected_request_ids) for policy in POLICIES} and all(
        set(request_ids) == expected_request_ids and len(request_ids) == len(set(request_ids))
        for request_ids in request_ids_by_policy.values()
    ) and all(
        str(row.get("diagnostic_only") or "").lower() == "true"
        and str(row.get("backend_name") or "") == "online_qwen"
        and str(row.get("split_seed") or "") == str(seed)
        and str(row.get("ok") or "").lower() == "true"
        for row in rows
    )


def _validate_policy_csv(path: Path, seed: int, expected_request_ids: set[str]) -> bool:
    policy_csv = _read_policy_csv(path)
    if policy_csv is None:
        return False
    fieldnames, rows = policy_csv
    return POLICY_CSV_REQUIRED_FIELDS <= set(fieldnames) and _policy_csv_rows_are_complete(rows, seed, expected_request_ids)


def _record_seed(path: Path, seed: int) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames or "split_seed" in fieldnames:
            raise ValueError("policy CSV has unexpected/missing header")
        rows = list(reader)
    temporary = path.with_suffix(".seed.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, "split_seed"])
        writer.writeheader()
        writer.writerows({**row, "split_seed": seed} for row in rows)
    os.replace(temporary, path)


def _recover_complete_legacy_policy_csv(path: Path, seed: int, expected_request_ids: set[str]) -> bool:
    policy_csv = _read_policy_csv(path)
    if policy_csv is None:
        return False
    fieldnames, rows = policy_csv
    legacy_header = [field for field in POLICY_OUTPUT_HEADER if field != "split_seed"]
    if fieldnames != legacy_header:
        return False
    seeded_rows = [{**row, "split_seed": str(seed)} for row in rows]
    if not _policy_csv_rows_are_complete(seeded_rows, seed, expected_request_ids):
        return False
    _record_seed(path, seed)
    return _validate_policy_csv(path, seed, expected_request_ids)


def _command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode


def _assert_attempt_output_available(
    path: Path, seed: int | None = None, expected_request_ids: set[str] | None = None
) -> bool:
    if not path.exists():
        return False
    if seed is not None and expected_request_ids is not None and _validate_policy_csv(path, seed, expected_request_ids):
        return True
    if seed is not None and expected_request_ids is not None and _recover_complete_legacy_policy_csv(path, seed, expected_request_ids):
        return True
    raise ValueError(f"existing incomplete policy CSV; inspect before retry: {path}")


def run(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    attempt_root = (args.attempt_root or run_root).resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    fixture = (ROOT / config["data"]["requests"]).resolve()
    fixture_audit = audit_fixture(fixture)
    if config.get("diagnostic_only") is not True or config.get("policies", {}).get("backend") != "online_qwen":
        raise ValueError("diagnostic online policy configuration required")
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    supervisor = json.loads((run_root / "supervisor_manifest.json").read_text(encoding="utf-8"))
    if not supervisor.get("merged") or manifest.get("expected_sessions") != 25 or manifest.get("expected_requests") != 125 or manifest.get("expected_profile_rows") != 1000:
        raise ValueError("full 25-session, 1000-profile-row measurement gate not passed")
    if manifest.get("diagnostic_only") is not True or manifest.get("filtered_fixture") != str(run_root / "filtered_fixture.jsonl"):
        raise ValueError("profile manifest has unexpected provenance")
    profile_paths = list((run_root / "merged" / "profile_tables").glob("*_profiles*.csv"))
    if len(profile_paths) != 1:
        raise ValueError("expected one complete merged profile CSV")
    measurements = read_measurements(profile_paths[0])
    data_config = config.get("data", {})
    calibration_fraction = float(data_config.get("calibration_fraction", 0.5))
    stratify_session = bool(data_config.get("stratify_session", True))

    budgets_path = run_root / "budgets.json"
    resume_only = getattr(args, "resume_only", False)
    if not budgets_path.exists() and resume_only:
        raise ValueError(f"resume-only requires an existing budgets file: {budgets_path}")
    if not budgets_path.exists() and attempt_root == run_root:
        code = _command([
            "conda", "run", "--no-capture-output", "-n", args.conda_env, "python", "-m",
            "run_util.derive_session_budgets", "--measurements", str(profile_paths[0]),
            "--output", str(budgets_path),
        ], run_root / "logs" / "budgets.log")
        if code:
            raise RuntimeError(f"budget derivation failed: {code}")
    if not budgets_path.exists():
        raise FileNotFoundError(f"profile budget file is required: {budgets_path}")
    budgets_payload = json.loads(budgets_path.read_text(encoding="utf-8"))
    if budgets_payload.get("diagnostic_only") is not True:
        raise ValueError("budgets lack diagnostic-only provenance")
    configured_budgets = config.get("pilot", {}).get("memory_budgets_mib", [])
    budgets = [float(value) for value in (configured_budgets or budgets_payload["memory_budgets_mib"])]
    schedule = cells(budgets)
    if getattr(args, "preflight", False):
        schedule = [cell for cell in schedule if cell[2:] == (PARAMETERS[0], PARAMETERS[0])]
    status_path = attempt_root / "fourth_grid_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    statuses: list[dict[str, object]] = []
    successful_paths_by_seed: dict[int, list[Path]] = {seed: [] for seed in SEEDS}
    for seed, budget, epsilon, delta in schedule:
        seed_root = attempt_root / f"seed{seed}"
        seed_config = seed_root / "config.yaml"
        seed_config.parent.mkdir(parents=True, exist_ok=True)
        seed_config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seed_config_data["data"].update({
            "requests": str(fixture), "split_seed": seed, "stratify_session": True,
        })
        seed_config.write_text(yaml.safe_dump(seed_config_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tag = f"eps{_slug(epsilon)}_delta{_slug(delta)}_mem{_slug(budget)}"
        path = seed_root / "policy_tables" / f"session25_policy_{tag}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        split_audit = _request_split_audit(
            measurements,
            split_seed=seed,
            calibration_fraction=calibration_fraction,
            stratify_session=stratify_session,
        )
        expected_request_ids = set(split_audit["evaluation_request_ids"])
        state: dict[str, object] = {
            "seed": seed, "budget_mib": budget, "epsilon": epsilon, "delta": delta,
            "policies": list(POLICIES),
            "total_fixture_requests": split_audit["total_request_count"],
            "calibration_requests": split_audit["calibration_request_count"],
            "evaluation_requests": split_audit["evaluation_request_count"],
            "expected_requests_per_policy": split_audit["evaluation_request_count"],
            "output": str(path), "diagnostic_only": True,
        }
        try:
            output_available = _assert_attempt_output_available(path, seed, expected_request_ids)
        except ValueError:
            if resume_only:
                raise ValueError(f"resume-only requires an existing valid policy CSV: {path}") from None
            raise
        if output_available:
            state["state"] = "resumed_success"
            state["completed_requests_per_policy"] = split_audit["evaluation_request_count"]
            successful_paths_by_seed[seed].append(path)
        else:
            if resume_only:
                raise ValueError(f"resume-only requires an existing valid policy CSV: {path}")
            log_path = seed_root / "logs" / f"{tag}.log"
            code = _command([
                "conda", "run", "--no-capture-output", "-n", args.conda_env,
                "python", "-m", "run_util.run_policies", "--config", str(seed_config),
                "--measurements", str(profile_paths[0]), "--output", str(path),
                "--epsilon", str(epsilon), "--delta", str(delta),
                "--memory-budget-mib", str(budget),
            ], log_path)
            state.update({"returncode": code, "log": str(log_path)})
            if code == 0:
                _record_seed(path, seed)
            valid = code == 0 and _validate_policy_csv(path, seed, expected_request_ids)
            state["state"] = "success" if valid else "failed"
            state["completed_requests_per_policy"] = split_audit["evaluation_request_count"] if valid else 0
            if valid:
                successful_paths_by_seed[seed].append(path)
            if not valid:
                state["error_type"] = "command_failed" if code else "policy_csv_validation_failed"
        statuses.append(state)
        status_path.write_text(json.dumps({
            "diagnostic_only": True,
            "fixture": fixture_audit,
            "profile_run_root": str(run_root),
            "attempt_root": str(attempt_root),
            "total_request_count": split_audit["total_request_count"],
            "calibration_request_count": split_audit["calibration_request_count"],
            "evaluation_request_count": split_audit["evaluation_request_count"],
            "planned_cell_configurations": len(schedule),
            "planned_cells": len(schedule) * len(POLICIES),
            "finished_cells": len(statuses) * len(POLICIES), "cells": statuses,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed = any(state["state"] == "failed" for state in statuses)
    for seed in SEEDS:
        successful_paths = successful_paths_by_seed[seed]
        if not successful_paths:
            continue
        policy_tables = attempt_root / f"seed{seed}" / "policy_tables"
        aggregate_session27(
            [str(policy_tables)],
            str(policy_tables),
            name="session25_fourth",
            input_paths=successful_paths,
        )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pilot_diagnostic_session25_fourth.yaml")
    parser.add_argument("--run-root", type=Path, required=True)
    # Keep profile inputs isolated from retry-specific policy outputs.
    parser.add_argument(
        "--attempt-root",
        type=Path,
        default=None,
        help="独立 policy attempt 输出根目录；不提供时兼容旧行为并写入 --run-root。",
    )
    parser.add_argument("--conda-env", default="tailguardkv-base")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="仅运行第一个 epsilon/delta 的所有预算受控在线验证，不启动完整网格。",
    )
    parser.add_argument(
        "--resume-only",
        action="store_true",
        help="仅恢复已有且通过校验的 policy CSV，不执行 policy GPU 命令。",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
