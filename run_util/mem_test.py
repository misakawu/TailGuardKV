from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

from experiment_common import config_policies, config_profiles, json_ready, load_config, read_measurements, validate_profile_measurements
from experiment_summary import write_summary, write_total_policy_summary
from run_util.build_profile_table import build_profile_table
from run_util.cli_common import first_number, run_command
from run_util.mem_test_analysis import analyze_mem_test_summary, write_mem_test_analysis
from run_util.mem_test_config import build_budget_series, build_mem_test_config, write_generated_config
from run_util.run_policies import run_policies
from visual.plot_summary import plot_summary


MEM_TEST_SUMMARY_OUTPUT = "out/policy_tables/run_mem_test_summary.csv"


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


def _resolve_run_dir(run_dir: str | None) -> Path:
    if run_dir:
        return Path(run_dir)
    return Path("out") / f"{datetime.now().strftime('%Y%m%d')}_mem_test"


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
    write_summary(payload, str(payload.get("summary_output") or MEM_TEST_SUMMARY_OUTPUT))
    total_summary_output = payload.get("total_summary_output")
    if total_summary_output:
        write_total_policy_summary(payload, str(total_summary_output))


def _run_generated_config(
    config_path: Path,
    run_dir: Path,
    *,
    explicit_total_summary_output: str,
) -> int:
    fallback_summary_output = str(_resolve_run_output(MEM_TEST_SUMMARY_OUTPUT, run_dir))
    fallback_total_summary_output = str(
        _resolve_run_output(explicit_total_summary_output, run_dir)
        if explicit_total_summary_output
        else _derive_total_summary_output(fallback_summary_output)
    )
    try:
        config = load_config(config_path)
        profiles = config_profiles(config)
        policies = config_policies(config)
        outputs = config.get("outputs", {})
        outputs = outputs if isinstance(outputs, dict) else {}
        profile_output = str(_resolve_run_output(str(outputs.get("smoke_profiles", "out/profile_tables/run_mem_test_profiles.csv")), run_dir))
        policy_output = str(_resolve_run_output(str(outputs.get("smoke_policy", "out/policy_tables/run_mem_test_policy.csv")), run_dir))
        summary_output = str(_resolve_run_output(str(outputs.get("smoke_summary", MEM_TEST_SUMMARY_OUTPUT)), run_dir))
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
            "config": str(config_path),
            "run_dir": str(run_dir),
            "summary_output": fallback_summary_output,
            "total_summary_output": fallback_total_summary_output,
        }
        _print_and_write(payload)
        return 2

    profile_args = argparse.Namespace(
        config=str(config_path),
        adapters=None,
        output=profile_output,
        import_measurements="",
        allow_import_measurements_for_debug=False,
        dry_run=False,
        formal_run=False,
    )
    profile_code, profile_payload = _run_stage(build_profile_table, profile_args)
    if profile_code != 0:
        payload = {
            "ok": False,
            "return_code": profile_code,
            "step": "build_profile_table",
            "config": str(config_path),
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
            "config": str(config_path),
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
            config=str(config_path),
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
        "config": str(config_path),
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
        "generated_config": str(config_path),
    }
    _print_and_write(payload)
    visual_outputs: list[Path] = []
    if policy_code == 0:
        try:
            visual_outputs = plot_summary(total_summary_output)
        except Exception as exc:  # pragma: no cover
            payload["visual_error"] = str(exc)
    payload["visual_outputs"] = [str(path) for path in visual_outputs]
    _print_and_write(payload)

    if policy_code == 0:
        analysis_path = run_dir / "run_mem_test_analysis.json"
        analysis_md_path = run_dir / "run_mem_test_analysis.md"
        analysis = analyze_mem_test_summary(Path(total_summary_output))
        analysis.update(
            {
                "config": str(config_path),
                "generated_config": str(config_path),
                "run_dir": str(run_dir),
            }
        )
        analysis_path.write_text(json.dumps(json_ready(analysis), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_mem_test_analysis(analysis, analysis_md_path)
    return policy_code


def run(args: argparse.Namespace) -> int:
    run_dir = _resolve_run_dir(args.run_dir)
    budgets = build_budget_series(args.budget_start_mib, args.budget_stop_mib, args.budget_step_mib)
    base_config = load_config(Path(args.base_config))
    config = build_mem_test_config(
        base_config,
        max_requests=args.max_requests,
        budgets_mib=budgets,
        include_tailguard=args.include_tailguard,
    )
    config_path = write_generated_config(config, run_dir)
    return _run_generated_config(
        config_path,
        run_dir,
        explicit_total_summary_output=args.total_summary_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TailGuardKV memory-budget baseline sweep.")
    parser.add_argument("--base-config", default="configs/pilot.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--budget-start-mib", type=float, default=100.0)
    parser.add_argument("--budget-stop-mib", type=float, default=5000.0)
    parser.add_argument("--budget-step-mib", type=float, default=100.0)
    parser.add_argument("--include-tailguard", action="store_true")
    parser.add_argument("--total-summary-output", default="")
    return parser


def main() -> int:
    return run_command(run, build_parser().parse_args())
