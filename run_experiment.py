from __future__ import annotations

# 常用运行命令:
# 1. profile runtime 预检:
#    python3 run_check_profiles.py --config configs/pilot_50.yaml --timeout 180
# 2. 快速 measured gate，50 requests，policy sweep 写参数后缀 CSV:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml
# 3. 正式 pilot measured，200 requests:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 4. 完整实验一键运行，含 profile 构建、profile 校验、policy sweep 和 summary 聚合:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 5. dry-run/CLI 兼容检查:
#    python3 run_build_profile_table.py --config configs/pilot.yaml --dry-run --output /tmp/tailguardkv_profiles.csv
#    python3 run_run_policies.py --config configs/pilot.yaml --measurements /tmp/tailguardkv_profiles.csv --output /tmp/tailguardkv_policy.csv --allow-dry-run-replay
# 6. 单个 policy 组合复跑:
#    python3 run_run_policies.py --config configs/pilot_50.yaml --measurements out/profile_tables/pilot_50_measured_profiles.csv --output /tmp/tailguardkv_policy_eps0p05_delta0p05_mem4900.csv --epsilon 0.05 --delta 0.05 --memory-budget-mib 4900
# 7. nohup + conda 后台跑完整实验:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_experiment.py pilot-smoke-measured --config configs/pilot.yaml > out/logs/pilot_measured.nohup.log 2>&1 < /dev/null & echo $! > out/logs/pilot_measured.pid
# 8. 查看后台实验状态和日志:
#    cat out/logs/pilot_measured.pid && ps -fp "$(cat out/logs/pilot_measured.pid)" && tail -n 120 out/logs/pilot_measured.nohup.log
# 9. 停止后台实验:
#    kill "$(cat out/logs/pilot_measured.pid)"

import argparse
import io
import json
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from typing import Any, Callable

from experiment_common import config_policies, config_profiles, json_ready, load_config, read_measurements, validate_profile_measurements
from experiment_summary import summary_rows, write_summary
from run_build_profile_table import build_profile_table
from run_cli_common import first_number
from run_run_policies import run_policies


PILOT_CONFIG = "configs/pilot.yaml"
PILOT_PROFILE_OUTPUT = "out/profile_tables/pilot_smoke_measured_profiles.csv"
PILOT_POLICY_OUTPUT = "out/policy_tables/pilot_smoke_measured_policy.csv"
PILOT_SUMMARY_OUTPUT = "out/policy_tables/pilot_smoke_measured_summary.csv"

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
def _number_list(value: Any, *, default: float, name: str) -> list[float]:
    if value is None:
        return [default]
    if isinstance(value, str):
        return [first_number(value, None, default=default, name=name)]
    try:
        values = list(value)
    except TypeError:
        values = [value]
    if not values:
        return [default]
    return [first_number(item, None, default=default, name=name) for item in values]


def _policy_sweep_points(config: dict[str, Any]) -> list[dict[str, float]]:
    pilot = config.get("pilot", {})
    pilot = pilot if isinstance(pilot, dict) else {}
    epsilons = _number_list(pilot.get("epsilons"), default=0.2, name="epsilon")
    deltas = _number_list(pilot.get("deltas"), default=0.05, name="delta")
    budgets = _number_list(pilot.get("memory_budgets_mib"), default=float("inf"), name="memory-budget-mib")
    return [
        {"epsilon": epsilon, "delta": delta, "memory_budget_mib": memory_budget_mib}
        for epsilon, delta, memory_budget_mib in product(epsilons, deltas, budgets)
    ]


def _slug_number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    if value.is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _policy_output_for_sweep(base_output: str, sweep: dict[str, float], sweep_count: int) -> str:
    if sweep_count <= 1:
        return base_output
    path = Path(base_output)
    suffix = (
        f"eps{_slug_number(sweep['epsilon'])}"
        f"_delta{_slug_number(sweep['delta'])}"
        f"_mem{_slug_number(sweep['memory_budget_mib'])}"
    )
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))
def _print_and_write(payload: dict[str, Any]) -> None:
    write_summary(payload, str(payload.get("summary_output") or PILOT_SUMMARY_OUTPUT))


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return summary_rows(payload)


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
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
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
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
            "profile": profile_payload,
        }
        _print_and_write(payload)
        return 2

    sweeps = _policy_sweep_points(config)
    policy_runs: list[dict[str, Any]] = []
    policy_rows = 0
    policy_code = 0
    policy_payload: dict[str, Any] = {}
    for sweep in sweeps:
        run_output = _policy_output_for_sweep(policy_output, sweep, len(sweeps))
        policy_args = argparse.Namespace(
            config=config_path,
            measurements=profile_output,
            output=run_output,
            profiles=None,
            policies=None,
            policy_config=None,
            epsilon=sweep["epsilon"],
            delta=sweep["delta"],
            memory_budget_mib=sweep["memory_budget_mib"],
            use_pandas_replay=False,
            allow_dry_run_replay=False,
        )
        policy_code, policy_payload = _run_stage(run_policies, policy_args)
        run = {
            "ok": policy_code == 0,
            "return_code": policy_code,
            "epsilon": sweep["epsilon"],
            "delta": sweep["delta"],
            "memory_budget_mib": sweep["memory_budget_mib"],
            "output": run_output,
            "payload": policy_payload,
        }
        policy_runs.append(run)
        policy_rows += int(policy_payload.get("rows") or 0)
        if policy_code != 0:
            break
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
            "policy": policy_rows,
        },
        "epsilon": policy_payload.get("epsilon"),
        "delta": policy_payload.get("delta"),
        "memory_budget_mib": policy_payload.get("memory_budget_mib"),
        "profile": profile_payload,
        "policy": policy_payload,
        "policy_runs": policy_runs,
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
