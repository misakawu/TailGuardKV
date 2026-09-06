#!/usr/bin/env python3
"""Drive the session27 online baseline: profiles -> budgets -> four-B sweeps -> aggregate/acceptance."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


EPSILON = 0.05
DELTA = 0.05
POLICY_PREFIX = "session27_policy"
MERGE_DIR = "merged"
POLICY_TABLES = "policy_tables"
LOGS = "logs"
BUDGETS_NAME = "budgets.json"
ACCEPTANCE_NAME = "acceptance.json"
PROFILE_GLOB = "*_profiles.csv"


def _slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _config_profile_name(config_path: Path, key: str, default: str) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    outputs = config.get("outputs") or {}
    return str(Path(str(outputs.get(key) or default)).name)


def _run(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT)
    return completed.returncode


def _policy_filename(budget: float) -> str:
    return f"{POLICY_PREFIX}_eps{_slug(EPSILON)}_delta{_slug(DELTA)}_mem{_slug(budget)}.csv"


def _read_merged_profiles(root: Path) -> Path | None:
    merged = root / MERGE_DIR / "profile_tables"
    profiles = sorted(merged.glob(PROFILE_GLOB))
    return profiles[0] if len(profiles) == 1 else None


def _run_profiles(args: argparse.Namespace, root: Path, repo_root: Path) -> int:
    if args.skip_profiles and _read_merged_profiles(root) is not None:
        return 0
    command = [
        "conda", "run", "--no-capture-output", "--cwd", str(repo_root), "-n", args.conda_env,
        "python", "scripts/run_diagnostic_session_batches.py",
        "--fixture", str(Path(args.fixture).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--run-root", str(root),
        "--sessions-per-batch", str(args.sessions_per_batch),
        "--conda-env", args.conda_env,
        "--profile-only",
    ]
    return _run(command, cwd=repo_root, log_path=root / LOGS / "profiles.log")


def _derive_budgets(args: argparse.Namespace, root: Path, repo_root: Path, merged_profiles: Path) -> Path:
    budgets_path = root / BUDGETS_NAME
    command = [
        "conda", "run", "--no-capture-output", "--cwd", str(repo_root), "-n", args.conda_env,
        "python", "-m", "run_util.derive_session_budgets",
        "--measurements", str(merged_profiles),
        "--output", str(budgets_path),
    ]
    code = _run(command, cwd=repo_root, log_path=root / LOGS / "budgets.log")
    if code != 0:
        raise RuntimeError(f"derive_session_budgets exited {code}; see {root / LOGS / 'budgets.log'}")
    return budgets_path


def _load_budgets(budgets_path: Path) -> dict[str, Any]:
    payload = json.loads(budgets_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("diagnostic_only") is not True:
        raise ValueError("budgets.json 必须标记 diagnostic_only=true")
    budgets = payload.get("memory_budgets_mib")
    percentiles = payload.get("percentiles_mib")
    if not isinstance(budgets, list) or len(budgets) != 4:
        raise ValueError("budgets.json 必须提供四档 B（p25/p50/p75/p90）")
    return {"payload": payload, "budgets": [float(value) for value in budgets], "percentiles": percentiles}


def _run_sweeps(args: argparse.Namespace, root: Path, repo_root: Path, merged_profiles: Path, budgets: list[float]) -> list[dict[str, Any]]:
    policy_tables = root / POLICY_TABLES
    policy_tables.mkdir(parents=True, exist_ok=True)
    sweeps: list[dict[str, Any]] = []
    for budget in budgets[: args.budgets_limit]:
        output_path = policy_tables / _policy_filename(budget)
        command = [
            "conda", "run", "--no-capture-output", "--cwd", str(repo_root), "-n", args.conda_env,
            "python", "-m", "run_util.run_policies",
            "--config", str(Path(args.config).resolve()),
            "--measurements", str(merged_profiles),
            "--output", str(output_path),
            "--epsilon", str(EPSILON),
            "--delta", str(DELTA),
            "--memory-budget-mib", str(budget),
        ]
        code = _run(command, cwd=repo_root, log_path=root / LOGS / f"sweep_mem{_slug(budget)}.log")
        sweeps.append(
            {
                "memory_budget_mib": budget,
                "output": str(output_path),
                "returncode": code,
                "ok": code == 0,
            }
        )
        if code != 0:
            break
    return sweeps


def _aggregate(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT_DIR))
    from scripts.aggregate_session27_baselines import aggregate_session27

    return aggregate_session27([str(root / POLICY_TABLES)], str(root / POLICY_TABLES))


def _csv_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or [])


def _check_policy_rows(path: Path, expected_policies: list[str]) -> list[str]:
    errors: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [f"{path.name} 没有 policy 行"]
    observed_policies = {str(row.get("policy") or "") for row in rows}
    for policy in expected_policies:
        if policy not in observed_policies:
            errors.append(f"{path.name} 缺少 policy {policy}")
    event_columns = {"budget_hit", "restore_ms", "recompute_ms", "queue_delay_ms", "global_resident_kv_mib"}
    for row in rows:
        if str(row.get("backend_name") or "").strip().lower() != "online_qwen":
            errors.append(f"{path.name} 存在非 online backend 行")
        if str(row.get("measured") or "").strip().lower() != "true":
            errors.append(f"{path.name} 存在非 measured 行")
        if not str(row.get("ttft_ms") or "").strip():
            errors.append(f"{path.name} 行缺少 ttft_ms")
        if not any(str(row.get(column) or "").strip() for column in event_columns):
            errors.append(f"{path.name} 行缺少 backend 事件字段")
        replay = str(row.get("replay_source") or row.get("extra_replay_source") or "").strip()
        if replay:
            errors.append(f"{path.name} 出现 profile 表 TTFT 回放: {replay}")
    return errors


def _evaluate_artifacts(root: Path, expected_policies: list[str], sweeps: list[dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    merged_profiles = _read_merged_profiles(root)
    merged_status = {"exists": merged_profiles is not None, "path": str(merged_profiles) if merged_profiles else ""}
    if merged_profiles is None:
        failures.append("merged profile 不存在")

    budgets_path = root / BUDGETS_NAME
    budgets_status: dict[str, Any] = {"exists": budgets_path.exists()}
    budgets: list[float] = []
    if budgets_path.exists():
        try:
            loaded = _load_budgets(budgets_path)
            budgets = loaded["budgets"]
            budgets_status["budgets"] = budgets
            budgets_status["percentiles"] = loaded["percentiles"]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            budgets_status["error"] = str(exc)
            failures.append(f"budgets 解析失败: {exc}")

    sweep_statuses: list[dict[str, Any]] = []
    for sweep in sweeps:
        path = Path(sweep["output"])
        status = {"memory_budget_mib": sweep["memory_budget_mib"], "returncode": sweep["returncode"], "exists": path.exists()}
        if not sweep["ok"] or not path.exists():
            status["errors"] = [f"sweep 失败 returncode={sweep['returncode']}"]
            failures.append(f"sweep B={sweep['memory_budget_mib']} 失败")
        else:
            row_errors = _check_policy_rows(path, expected_policies)
            status["row_errors"] = row_errors
            failures.extend(f"B={sweep['memory_budget_mib']}: {error}" for error in row_errors)
            with path.open("r", encoding="utf-8", newline="") as handle:
                status["rows"] = sum(1 for _ in csv.DictReader(handle))
        sweep_statuses.append(status)

    artifact_paths = {
        "summary_csv": root / POLICY_TABLES / "session27_total_summary.csv",
        "events_csv": root / POLICY_TABLES / "session27_events.csv",
        "session_points_csv": root / POLICY_TABLES / "session27_session_points.csv",
        "baseline_smoke_markdown": root / POLICY_TABLES / "baseline_smoke.md",
    }
    for filename in (
        "summary_policy_p95_ttft.png",
        "summary_policy_kv_memory.png",
        "summary_policy_quality_loss.png",
        "summary_policy_violation_rate.png",
    ):
        artifact_paths[f"plot_{filename}"] = root / POLICY_TABLES / filename
    artifact_status = {key: {"exists": path.exists()} for key, path in artifact_paths.items()}
    for key, path in artifact_paths.items():
        if not path.exists():
            failures.append(f"artifact 缺失: {key}")

    sensitivity: dict[str, Any] = {"checked": False, "metrics": {}, "conclusion": ""}
    summary_csv = artifact_paths["summary_csv"]
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        metric_names = ("p95_ttft_ms", "budget_hit_rate", "restore_count", "recompute_count", "mean_kv_cache_memory_mib")
        per_policy: dict[str, dict[str, float]] = {}
        for row in rows:
            policy = str(row.get("policy") or "")
            try:
                memory = float(row.get("memory_budget_mib"))
            except (TypeError, ValueError):
                continue
            entry: dict[str, float] = {}
            for name in metric_names:
                try:
                    value = float(row.get(name))
                except (TypeError, ValueError):
                    value = float("nan")
                entry[name] = value
            per_policy.setdefault(policy, {})[memory] = entry
        metric_changes: dict[str, Any] = {}
        for name in metric_names:
            changed: dict[str, float] = {}
            for policy, by_budget in per_policy.items():
                if len(by_budget) < 2:
                    continue
                values = [entry.get(name) for entry in by_budget.values()]
                finite = [value for value in values if value == value]
                if len(set(finite)) >= 2:
                    changed[policy] = float(max(finite) - min(finite))
            if changed:
                metric_changes[name] = changed
        sensitivity = {
            "checked": True,
            "budgets": sorted({float(row.get("memory_budget_mib")) for row in rows if str(row.get("memory_budget_mib") or "").strip()}),
            "metrics": metric_changes,
            "per_policy": per_policy,
            "event_decomposition": sorted(
                {
                    (str(row.get("policy") or ""), str(row.get("memory_budget_mib") or ""), str(row.get("budget_hit_rate") or ""), str(row.get("restore_count") or ""), str(row.get("recompute_count") or ""))
                    for row in rows
                }
            )[:200],
        }
        if not metric_changes:
            sensitivity["conclusion"] = (
                "B 对本批 baseline 的 TTFT/backend 事件不敏感；按事件分解记录，不调整策略制造预设排序。"
            )
        else:
            sensitivity["conclusion"] = "检测到随 B 变化的 backend 事件指标：" + ", ".join(
                f"{name}[" + ", ".join(f"{policy}={delta:.4g}" for policy, delta in sorted(changes.items())) + "]"
                for name, changes in sorted(metric_changes.items())
            )

    payload = {
        "diagnostic_only": True,
        "quality_status": "risk_evidence_insufficient",
        "violation_status": "risk_evidence_insufficient",
        "merged_profiles": merged_status,
        "budgets": budgets_status,
        "sweeps": sweep_statuses,
        "sensitivity": sensitivity,
        "artifacts": artifact_status,
        "failures": failures,
        "passed": not failures,
    }
    (root / ACCEPTANCE_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_diagnostic_session27.yaml")
    parser.add_argument("--fixture", default="data/fixtures/diagnostic_session27.jsonl")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--sessions-per-batch", type=int, default=3)
    parser.add_argument("--conda-env", default="tailguardkv-base")
    parser.add_argument("--budgets-limit", type=int, default=4)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).resolve()
    base = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expected_policies = [str(name) for name in ((base.get("policies") or {}).get("names") or [])]
    root = Path(args.run_root or f"out/session27_online_{_timestamp()}").resolve()
    repo_root = ROOT_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / LOGS).mkdir(parents=True, exist_ok=True)
    phase_status: dict[str, Any] = {}

    if args.prepare_only:
        manifest = {
            "diagnostic_only": True,
            "config": str(config_path),
            "fixture": str(Path(args.fixture).resolve()),
            "run_root": str(root),
            "sessions_per_batch": args.sessions_per_batch,
            "policies": expected_policies,
        }
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    profiles_code = _run_profiles(args, root, repo_root)
    phase_status["profiles"] = {"returncode": profiles_code, "merged": _read_merged_profiles(root) is not None}
    if profiles_code != 0:
        print(json.dumps({"ok": False, "phase": "profiles", "run_root": str(root), "phase_status": phase_status}, ensure_ascii=False))
        return 1

    merged_profiles = _read_merged_profiles(root)
    if merged_profiles is None:
        print(json.dumps({"ok": False, "phase": "profiles-merged-missing", "run_root": str(root)}, ensure_ascii=False))
        return 1
    budgets_path = _derive_budgets(args, root, repo_root, merged_profiles)
    loaded = _load_budgets(budgets_path)
    phase_status["budgets"] = {"budgets": loaded["budgets"]}

    sweeps = _run_sweeps(args, root, repo_root, merged_profiles, loaded["budgets"])
    phase_status["sweeps"] = sweeps
    if any(not sweep["ok"] for sweep in sweeps):
        payload = _evaluate_artifacts(root, expected_policies, sweeps)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    aggregate_payload = _aggregate(args, root)
    phase_status["aggregate"] = aggregate_payload
    acceptance = _evaluate_artifacts(root, expected_policies, sweeps)
    acceptance["phases"] = phase_status
    (root / ACCEPTANCE_NAME).write_text(json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    return 0 if acceptance["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
