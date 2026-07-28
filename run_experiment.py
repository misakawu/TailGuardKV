from __future__ import annotations

import argparse
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

from experiment_common import config_policies, config_profiles, json_ready, load_config, read_measurements, validate_profile_measurements
from run_build_profile_table import build_profile_table
from run_run_policies import run_policies


PILOT_CONFIG = "configs/pilot.yaml"
PILOT_PROFILE_OUTPUT = "out/profile_tables/pilot_smoke_measured_profiles.csv"
PILOT_POLICY_OUTPUT = "out/policy_tables/pilot_smoke_measured_policy.csv"
PILOT_SUMMARY_OUTPUT = "out/policy_tables/pilot_smoke_measured_summary.csv"

SUMMARY_KEY_COLUMNS = [
    "section",
    "name",
    "ok",
    "error",
    "config",
    "profile_rows",
    "policy_rows",
    "epsilon",
    "delta",
    "memory_budget_mib",
    "count",
    "ok_count",
    "measured_count",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_peak_memory_mib",
    "p95_peak_memory_mib",
    "mean_quality_loss",
    "p95_quality_loss",
    "p99_quality_loss",
    "cvar_quality_loss",
    "violation_rate",
    "delta_slack",
    "worst_group_violation",
    "safe_ratio",
    "fallback_ratio",
    "exact_fallback_ratio",
    "lossy_action_ratio",
    "unique_action_count",
    "identical_to_full_lru",
    "unsafe_action_count",
    "candidate_safe_count",
    "action_distribution",
]


def _run_stage(func: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    stream = io.StringIO()
    with redirect_stdout(stream):
        code = int(func(args))
    raw_output = stream.getvalue().strip()
    if not raw_output:
        return code, {}
    try:
        return code, json.loads(raw_output)
    except json.JSONDecodeError:
        return code, {"raw_stdout": raw_output}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)
    return json_ready(value)


def _summary_error(payload: dict[str, Any]) -> Any:
    if payload.get("error"):
        return payload.get("error")
    for section in ("profile", "policy"):
        nested = payload.get(section)
        if isinstance(nested, dict) and nested.get("error"):
            return nested.get("error")
    return ""


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "section": "experiment",
            "name": "pilot-smoke-measured",
            "ok": payload.get("ok"),
            "error": _summary_error(payload),
            "config": payload.get("config"),
            "profile_rows": (payload.get("rows") or {}).get("profiles") if isinstance(payload.get("rows"), dict) else "",
            "policy_rows": (payload.get("rows") or {}).get("policy") if isinstance(payload.get("rows"), dict) else "",
            "epsilon": payload.get("epsilon"),
            "delta": payload.get("delta"),
            "memory_budget_mib": payload.get("memory_budget_mib"),
        }
    ]
    for section in ("profile", "policy"):
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            continue
        summary = section_payload.get("summary")
        if not isinstance(summary, dict):
            continue
        for name, metrics in summary.items():
            row = {
                "section": section,
                "name": name,
                "ok": payload.get("ok"),
                "config": payload.get("config"),
                "epsilon": payload.get("epsilon"),
                "delta": payload.get("delta"),
                "memory_budget_mib": payload.get("memory_budget_mib"),
            }
            if isinstance(metrics, dict):
                row.update(metrics)
            rows.append(row)
    return rows


def _write_summary(payload: dict[str, Any]) -> None:
    path = Path(str(payload.get("summary_output") or PILOT_SUMMARY_OUTPUT))
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows(payload)
    fieldnames = SUMMARY_KEY_COLUMNS + sorted({key for row in rows for key in row}.difference(SUMMARY_KEY_COLUMNS))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()} for row in rows)


def _print_and_write(payload: dict[str, Any]) -> None:
    _write_summary(payload)


def pilot_smoke_measured(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", PILOT_CONFIG)
    try:
        config = load_config(Path(config_path))
        profiles = config_profiles(config)
        policies = config_policies(config)
        outputs = config.get("outputs", {})
        outputs = outputs if isinstance(outputs, dict) else {}
        profile_output = str(outputs.get("smoke_profiles", PILOT_PROFILE_OUTPUT))
        policy_output = str(outputs.get("smoke_policy", PILOT_POLICY_OUTPUT))
        summary_output = str(outputs.get("smoke_summary", PILOT_SUMMARY_OUTPUT))
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "load_config",
            "error": str(exc),
            "config": config_path,
            "summary_output": PILOT_SUMMARY_OUTPUT,
        }
        _print_and_write(payload)
        return 2

    profile_args = argparse.Namespace(
        config=config_path,
        adapters=None,
        output=profile_output,
        import_measurements="",
        dry_run=False,
    )
    profile_code, profile_payload = _run_stage(build_profile_table, profile_args)
    if profile_code != 0:
        payload = {
            "ok": False,
            "return_code": profile_code,
            "step": "build_profile_table",
            "config": config_path,
            "summary_output": summary_output,
            "profile": profile_payload,
        }
        _print_and_write(payload)
        return profile_code

    try:
        measurements = read_measurements(Path(profile_output))
        validate_profile_measurements(
            measurements,
            profile_output,
            required_profiles=profiles,
            require_measured=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "validate_profile_table",
            "error": str(exc),
            "config": config_path,
            "summary_output": summary_output,
            "profile": profile_payload,
        }
        _print_and_write(payload)
        return 2

    policy_args = argparse.Namespace(
        config=config_path,
        measurements=profile_output,
        output=policy_output,
        profiles=None,
        policies=None,
        policy_config=None,
        epsilon=None,
        delta=None,
        memory_budget_mib=None,
        use_pandas_replay=False,
        allow_dry_run_replay=False,
    )
    policy_code, policy_payload = _run_stage(run_policies, policy_args)
    payload = {
        "ok": policy_code == 0,
        "return_code": policy_code,
        "step": "complete" if policy_code == 0 else "run_policies",
        "config": config_path,
        "summary_output": summary_output,
        "profiles": profiles,
        "policies": policies,
        "rows": {
            "profiles": profile_payload.get("rows", len(measurements)),
            "policy": policy_payload.get("rows"),
        },
        "epsilon": policy_payload.get("epsilon"),
        "delta": policy_payload.get("delta"),
        "memory_budget_mib": policy_payload.get("memory_budget_mib"),
        "profile": profile_payload,
        "policy": policy_payload,
    }
    _print_and_write(payload)
    return policy_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV 正式实验入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot-smoke-measured", help="运行真实 pilot measured smoke 实验。")
    pilot.add_argument("--config", default=PILOT_CONFIG)
    pilot.set_defaults(func=pilot_smoke_measured)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
