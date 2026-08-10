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
import csv
from contextlib import redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

from run_util.experiment_common import (
    config_experiment_type,
    config_policies,
    config_profiles,
    json_ready,
    load_requests,
    load_config,
    read_measurements,
    synthesize_pressure_trace,
    validate_requests_for_experiment_type,
    validate_profile_measurements,
    write_csv,
)
from run_util.experiment_summary import summary_rows, write_summary, write_total_policy_summary
from run_util.build_profile_table import build_profile_table
from run_util.cli_common import first_number
from run_util.run_policies import run_policies
from scripts.validate_trace_quality import validate_trace_quality
from visual.plot_summary import plot_summary


PILOT_CONFIG = "configs/pilot.yaml"
PILOT_PROFILE_OUTPUT = "out/profile_tables/pilot_smoke_measured_profiles.csv"
PILOT_SESSION_TRACE_OUTPUT = "out/session_traces/pilot_smoke_measured_session_trace.csv"
PILOT_POLICY_OUTPUT = "out/policy_tables/pilot_smoke_measured_policy.csv"
PILOT_SUMMARY_OUTPUT = "out/policy_tables/pilot_smoke_measured_summary.csv"
PILOT_TRACE_QUALITY_GATE_OUTPUT = "out/profile_tables/pilot_session_trace_quality_gate.json"

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


def _load_fixture_rows(request_path: str) -> list[dict[str, Any]]:
    path = Path(request_path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _session_trace_settings(config: dict[str, Any]) -> dict[str, float | int]:
    trace_config = config.get("session_trace", {})
    trace_config = trace_config if isinstance(trace_config, dict) else {}
    pilot = config.get("pilot", {})
    pilot = pilot if isinstance(pilot, dict) else {}
    budgets = _number_list(
        trace_config.get("memory_budgets_mib", pilot.get("memory_budgets_mib")),
        default=32.0,
        name="session-trace-memory-budget-mib",
    )
    finite_budgets = [budget for budget in budgets if budget > 0 and budget != float("inf")]
    return {
        "copies": max(1, int(trace_config.get("copies", 2))),
        "repeat_rounds": max(1, int(trace_config.get("repeat_rounds", 1))),
        "memory_budget_mib": min(finite_budgets) if finite_budgets else 32.0,
    }


def _write_session_trace(
    requests: list,
    output: str,
    settings: dict[str, float | int],
) -> dict[str, Any]:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "request_id",
                "session_id",
                "turn_index",
                "arrival_index",
                "task",
                "prompt",
                "reference",
                "history_turns",
                "split",
                "original_request_id",
                "original_session_id",
                "pressure_round",
                "pressure_copy",
                "pressure_budget_mib",
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                "request_id": request.request_id,
                "session_id": request.session_id or "",
                "turn_index": request.turn_index,
                "arrival_index": request.arrival_index,
                "task": request.task,
                "prompt": request.prompt,
                "reference": request.reference or "",
                "history_turns": json.dumps(list(request.history_turns), ensure_ascii=False),
                "split": request.metadata.get("split", ""),
                "original_request_id": request.metadata.get("original_request_id", ""),
                "original_session_id": request.metadata.get("original_session_id", ""),
                "pressure_round": request.metadata.get("pressure_round", ""),
                "pressure_copy": request.metadata.get("pressure_copy", ""),
                "pressure_budget_mib": request.metadata.get("pressure_budget_mib", ""),
            }
            for request in requests
        )
    return {
        "output": output,
        "rows": len(requests),
        "copies": int(settings["copies"]),
        "repeat_rounds": int(settings["repeat_rounds"]),
        "memory_budget_mib": float(settings["memory_budget_mib"]),
        "session_count": len({request.session_id for request in requests}),
    }


def pilot_smoke_measured(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", PILOT_CONFIG)
    experiment_type = ""
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
        experiment_type = config_experiment_type(config)
        outputs = config.get("outputs", {})
        outputs = outputs if isinstance(outputs, dict) else {}
        profile_output = str(_resolve_run_output(str(outputs.get("smoke_profiles", PILOT_PROFILE_OUTPUT)), run_dir))
        session_trace_output = str(
            _resolve_run_output(
                str(outputs.get("smoke_session_trace", PILOT_SESSION_TRACE_OUTPUT)),
                run_dir,
            )
        )
        policy_output = str(_resolve_run_output(str(outputs.get("smoke_policy", PILOT_POLICY_OUTPUT)), run_dir))
        summary_output = str(_resolve_run_output(str(outputs.get("smoke_summary", PILOT_SUMMARY_OUTPUT)), run_dir))
        trace_quality_gate_output = str(
            _resolve_run_output(str(outputs.get("smoke_trace_quality_gate", PILOT_TRACE_QUALITY_GATE_OUTPUT)), run_dir)
        )
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
            "experiment_type": experiment_type,
            "summary_output": fallback_summary_output,
            "total_summary_output": fallback_total_summary_output,
            "session_trace_output": str(_resolve_run_output(PILOT_SESSION_TRACE_OUTPUT, run_dir)),
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
    session_trace_payload: dict[str, Any] = {}
    replay_measurements_path = profile_output
    if experiment_type == "baseline_session":
        try:
            seed_requests, fallback_requests = load_requests(config)
            validate_requests_for_experiment_type(seed_requests, experiment_type)
            pressure_requests = synthesize_pressure_trace(
                seed_requests,
                copies=int(_session_trace_settings(config)["copies"]),
                repeat_rounds=int(_session_trace_settings(config)["repeat_rounds"]),
                memory_budget_mib=float(_session_trace_settings(config)["memory_budget_mib"]),
            )
            session_trace_payload = _write_session_trace(
                pressure_requests,
                session_trace_output,
                _session_trace_settings(config),
            )
            profile_args.preloaded_requests = pressure_requests
            profile_args.fallback_requests = fallback_requests
        except (FileNotFoundError, ValueError) as exc:
            payload = {
                "ok": False,
                "return_code": 2,
                "step": "pressure_trace_synthesize",
                "error": str(exc),
                "config": config_path,
                "run_dir": str(run_dir),
                "experiment_type": experiment_type,
                "summary_output": summary_output,
                "total_summary_output": total_summary_output,
                "session_trace_output": session_trace_output,
            }
            _print_and_write(payload)
            return 2
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
            "session_trace_output": session_trace_output,
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
            experiment_type=experiment_type,
        )
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "validate_profile_table",
            "error": str(exc),
            "config": config_path,
            "run_dir": str(run_dir),
            "experiment_type": experiment_type,
            "summary_output": summary_output,
            "total_summary_output": total_summary_output,
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
            "profile": profile_payload,
        }
        _print_and_write(payload)
        return 2

    sweeps = _policy_sweep_points(config)
    gate_payload: dict[str, Any] = {}
    if experiment_type == "baseline_session":
        try:
            fixture_rows = _load_fixture_rows(str(config.get("data", {}).get("requests")))
            gate_result = validate_trace_quality(measurements, fixture_rows)
            gate_payload = gate_result.to_json()
            Path(trace_quality_gate_output).parent.mkdir(parents=True, exist_ok=True)
            Path(trace_quality_gate_output).write_text(
                json.dumps(gate_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not gate_result.passed:
                payload = {
                    "ok": False,
                    "return_code": 2,
                    "step": "validate_trace_quality",
                    "error": "session trace quality gate failed",
                    "config": config_path,
                    "run_dir": str(run_dir),
                    "experiment_type": experiment_type,
                    "summary_output": summary_output,
                    "total_summary_output": total_summary_output,
                    "session_trace_output": session_trace_output,
                    "trace_quality_gate_output": trace_quality_gate_output,
                    "trace_quality_gate": gate_payload,
                }
                _print_and_write(payload)
                return 2
        except (FileNotFoundError, ValueError, KeyError) as exc:
            payload = {
                "ok": False,
                "return_code": 2,
                "step": "validate_trace_quality",
                "error": str(exc),
                "config": config_path,
                "run_dir": str(run_dir),
                "experiment_type": experiment_type,
                "summary_output": summary_output,
                "total_summary_output": total_summary_output,
                "session_trace_output": session_trace_output,
                "trace_quality_gate_output": trace_quality_gate_output,
            }
            _print_and_write(payload)
            return 2
    policy_runs: list[dict[str, Any]] = []
    policy_rows = 0
    policy_code = 0
    policy_payload: dict[str, Any] = {}
    for sweep in sweeps:
        run_output = _policy_output_for_sweep(policy_output, sweep, len(sweeps))
        policy_args = argparse.Namespace(
            config=config_path,
            measurements=replay_measurements_path,
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
        "experiment_type": experiment_type,
        "summary_output": summary_output,
        "total_summary_output": total_summary_output,
        "session_trace_output": session_trace_output,
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
        "session_trace": session_trace_payload,
        "trace_quality_gate_output": trace_quality_gate_output,
        "trace_quality_gate": gate_payload,
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
