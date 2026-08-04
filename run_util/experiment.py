from __future__ import annotations

# 常用运行命令:
# 1. profile runtime 预检:
#    python3 -m run_util.check_profiles --config configs/pilot_50.yaml --timeout 180
# 2. 快速 measured gate，50 requests，policy sweep 写参数后缀 CSV:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot_50.yaml
# 3. 正式 pilot measured，200 requests:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 4. 完整实验一键运行，含 profile 构建、profile 校验、policy sweep 和 summary 聚合:
#    python3 run_experiment.py pilot-smoke-measured --config configs/pilot.yaml
# 5. dry-run/CLI 兼容检查:
#    python3 -m run_util.build_profile_table --config configs/pilot.yaml --dry-run --output /tmp/tailguardkv_profiles.csv
#    python3 -m run_util.run_policies --config configs/pilot.yaml --measurements /tmp/tailguardkv_profiles.csv --output /tmp/tailguardkv_policy.csv --allow-dry-run-replay
# 6. 单个 policy 组合复跑:
#    python3 -m run_util.run_policies --config configs/pilot_50.yaml --measurements out/profile_tables/pilot_50_measured_profiles.csv --output /tmp/tailguardkv_policy_eps0p05_delta0p05_mem4900.csv --epsilon 0.05 --delta 0.05 --memory-budget-mib 4900
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
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

from run_util.experiment_common import config_policies, config_profiles, json_ready, load_config, read_measurements, validate_profile_measurements
from run_util.experiment_summary import summary_rows, write_summary, write_total_policy_summary
from run_util.build_profile_table import build_profile_table
from run_util.cli_common import first_number
from run_util.run_policies import run_policies
from visual.plot_summary import plot_summary


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


def _resolve_run_dir(run_dir: str | None, config_path: str) -> Path:
    if run_dir:
        return Path(run_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"{timestamp}_{Path(config_path).stem}"


def _resolve_run_output(output: str, run_dir: Path) -> Path:
    path = Path(output)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "out":
        parts = parts[1:]
    return run_dir.joinpath(*parts)


def _derive_total_summary_output(summary_output: str) -> str:
    path = Path(summary_output)
    stem = path.stem
    if stem.endswith("_summary"):
        stem = f"{stem.removesuffix('_summary')}_total_summary"
    else:
        stem = f"{stem}_total_summary"
    return str(path.with_name(f"{stem}{path.suffix}"))


def _print_and_write(payload: dict[str, Any]) -> None:
    write_summary(payload, str(payload.get("summary_output") or PILOT_SUMMARY_OUTPUT))
    total_summary_output = payload.get("total_summary_output")
    if total_summary_output:
        write_total_policy_summary(payload, str(total_summary_output))


def _summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return summary_rows(payload)


def pilot_smoke_measured(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", PILOT_CONFIG)
    run_dir = _resolve_run_dir(getattr(args, "run_dir", None), config_path)
    fallback_summary_output = str(_resolve_run_output(PILOT_SUMMARY_OUTPUT, run_dir))
    explicit_total_summary_output = str(getattr(args, "total_summary_output", "") or "")
    fallback_total_summary_output = str(
        _resolve_run_output(explicit_total_summary_output, run_dir)
        if explicit_total_summary_output
        else _derive_total_summary_output(fallback_summary_output)
    )
    try:
        config = load_config(Path(config_path))
        profiles = config_profiles(config)
        policies = config_policies(config)
        outputs = config.get("outputs", {})
        outputs = outputs if isinstance(outputs, dict) else {}
        profile_output = str(_resolve_run_output(str(outputs.get("smoke_profiles", PILOT_PROFILE_OUTPUT)), run_dir))
        policy_output = str(_resolve_run_output(str(outputs.get("smoke_policy", PILOT_POLICY_OUTPUT)), run_dir))
        summary_output = str(_resolve_run_output(str(outputs.get("smoke_summary", PILOT_SUMMARY_OUTPUT)), run_dir))
        configured_total_summary_output = outputs.get("smoke_total_summary")
        if explicit_total_summary_output:
            total_summary_output = str(_resolve_run_output(explicit_total_summary_output, run_dir))
        elif configured_total_summary_output:
            total_summary_output = str(_resolve_run_output(str(configured_total_summary_output), run_dir))
        else:
            total_summary_output = _derive_total_summary_output(summary_output)
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "load_config",
            "error": str(exc),
            "config": config_path,
            "run_dir": str(run_dir),
            "summary_output": fallback_summary_output,
            "total_summary_output": fallback_total_summary_output,
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
            "run_dir": str(run_dir),
            "summary_output": summary_output,
            "total_summary_output": total_summary_output,
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
            "run_dir": str(run_dir),
            "summary_output": summary_output,
            "total_summary_output": total_summary_output,
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
        "run_dir": str(run_dir),
        "summary_output": summary_output,
        "total_summary_output": total_summary_output,
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
    visual_outputs: list[Path] = []
    if policy_code == 0:
        try:
            visual_outputs = plot_summary(total_summary_output)
        except Exception as exc:  # pragma: no cover - visualization must not fail the experiment.
            payload["visual_error"] = str(exc)
    payload["visual_outputs"] = [str(path) for path in visual_outputs]
    _print_and_write(payload)
    return policy_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV 正式实验入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot = subparsers.add_parser("pilot-smoke-measured", help="运行真实 pilot measured smoke 实验。")
    pilot.add_argument("--config", default=PILOT_CONFIG)
    pilot.add_argument("--run-dir")
    pilot.add_argument("--total-summary-output", default="")
    pilot.set_defaults(func=pilot_smoke_measured)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
