from __future__ import annotations

# 常用运行命令:
# 1. conda 前台运行 profile test:
#    conda run -n tailguardkv-base python run_profile_test.py --config configs/pilot_50.yaml > out/logs/profile_test.nohup.log 2>&1 < /dev/null & echo $! > out/logs/profile_test.pid
# 2. nohup + conda 后台运行 profile test:
#    mkdir -p out/logs && nohup conda run -n tailguardkv-base python run_profile_test.py --config configs/pilot_50.yaml > out/logs/profile_test.nohup.log 2>&1 < /dev/null & echo $! > out/logs/profile_test.pid

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

from run_util.experiment_common import config_profiles, config_runtime, load_config, read_measurements, validate_profile_measurements
from run_util.profile_summary import write_profile_summary
from run_util.build_profile_table import build_profile_table
from profiles.registry import build_profile_adapters
from run_util.cli_common import run_command


DEFAULT_PROFILE_OUTPUT = "out/profile_tables/smoke_profiles.csv"


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


def _derive_summary_output(profile_output: str) -> str:
    path = Path(profile_output)
    return str(path.with_name(f"{path.stem}_summary.csv"))


def _resolve_outputs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[str, str]:
    outputs = config.get("outputs", {})
    outputs = outputs if isinstance(outputs, dict) else {}
    profile_output = str(args.output or outputs.get("smoke_profiles") or DEFAULT_PROFILE_OUTPUT)
    summary_output = str(
        args.summary_output
        or outputs.get("smoke_profile_summary")
        or _derive_summary_output(profile_output)
    )
    return profile_output, summary_output


def _active_profile_names(config: dict[str, Any], adapters: list[str] | None) -> list[str]:
    configured_profiles = config_profiles(config)
    runtime = config_runtime(config)
    active: list[str] = []
    for adapter in build_profile_adapters(adapters or config.get("profiles", {}).get("adapters", []), runtime):
        for spec in adapter.profiles():
            if spec.name in configured_profiles and spec.name not in active:
                active.append(spec.name)
    return active


def _write_payload(
    payload: dict[str, Any],
    summary_output: str,
    *,
    measurements: list[Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    write_profile_summary(payload, summary_output, measurements=measurements, config=config)


def run_profile_test(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config", "configs/pilot.yaml")
    fallback_profile_output = str(getattr(args, "output", "") or DEFAULT_PROFILE_OUTPUT)
    fallback_summary_output = str(getattr(args, "summary_output", "") or _derive_summary_output(fallback_profile_output))
    try:
        config = load_config(Path(config_path))
        runtime = config_runtime(config)
        profiles = _active_profile_names(config, args.adapters)
        require_quality_loss = "full_gpu" in profiles
        profile_output, summary_output = _resolve_outputs(args, config)
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "load_config",
            "error": str(exc),
            "config": config_path,
        }
        _write_payload(payload, fallback_summary_output)
        return 2

    profile_args = argparse.Namespace(
        config=config_path,
        adapters=args.adapters,
        output=profile_output,
        import_measurements="",
        dry_run=args.dry_run,
    )
    profile_code, profile_payload = _run_stage(build_profile_table, profile_args)
    if profile_code != 0:
        payload = {
            "ok": False,
            "return_code": profile_code,
            "step": "build_profile_table",
            "config": config_path,
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
            "profile": profile_payload,
        }
        _write_payload(payload, summary_output, config=config)
        return profile_code

    try:
        measurements = read_measurements(Path(profile_output), require_quality_loss=require_quality_loss)
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "read_profile_table",
            "error": str(exc),
            "config": config_path,
            "dry_run": args.dry_run,
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
            "profile": profile_payload,
        }
        _write_payload(payload, summary_output, config=config)
        return 2

    try:
        validate_profile_measurements(
            measurements,
            profile_output,
            required_profiles=profiles,
            require_measured=not args.dry_run,
            require_ttft=bool(runtime.get("require_ttft", False)) and not args.dry_run,
            require_quality_loss=require_quality_loss,
        )
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "ok": False,
            "return_code": 2,
            "step": "validate_profile_table",
            "error": str(exc),
            "config": config_path,
            "dry_run": args.dry_run,
            "diagnostic_output": profile_payload.get("diagnostic_output"),
            "failures": profile_payload.get("failures"),
            "profile": profile_payload,
        }
        _write_payload(payload, summary_output, measurements=measurements, config=config)
        return 2

    payload = {
        "ok": True,
        "return_code": 0,
        "step": "complete",
        "config": config_path,
        "dry_run": args.dry_run,
        "rows": {
            "profiles": profile_payload.get("rows", len(measurements)),
        },
        "profile": profile_payload,
    }
    _write_payload(payload, summary_output, measurements=measurements, config=config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV profile-only measured runner.")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--adapters", nargs="+")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> int:
    return run_command(run_profile_test, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
