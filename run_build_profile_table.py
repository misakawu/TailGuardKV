from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiment_common import (
    annotate_measurement,
    config_adapters,
    config_profiles,
    config_runtime,
    exact_profiles,
    expand_repeated_requests,
    failed_measurement_summary,
    json_ready,
    length_bucket,
    load_config,
    load_requests,
    read_measurements,
    validate_profile_measurements,
    with_quality,
    write_csv,
)
from metrics import MetricCollector
from profiles.registry import build_profile_adapters
from run_cli_common import add_profile_table_arguments, print_error, run_command


def build_profile_table(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = args.output or config.get("outputs", {}).get("smoke_profiles", "out/profile_tables/smoke_profiles.csv")
    if args.import_measurements:
        try:
            measurements = read_measurements(Path(args.import_measurements))
            validate_profile_measurements(
                measurements,
                args.import_measurements,
                required_profiles=config_profiles(config),
                require_measured=not args.dry_run,
            )
        except ValueError as exc:
            print_error(exc, output=output)
            return 2
        write_csv(Path(output), [measurement.to_row() for measurement in measurements])
        summary = MetricCollector().summarize_profiles(measurements)
        print(json.dumps(json_ready({
            "output": output,
            "rows": len(measurements),
            "imported_from": args.import_measurements,
            "summary": summary,
        }), indent=2))
        return 0 if all(measurement.ok and measurement.measured for measurement in measurements) else 1
    profiles = config_profiles(config)
    runtime = config_runtime(config)
    adapters = build_profile_adapters(args.adapters or config_adapters(config), runtime)
    requests, fallback_requests = load_requests(config)
    requests = expand_repeated_requests(requests, int(runtime.get("repeat", 1)))
    requests = [
        replace(
            request,
            metadata={**request.metadata, "task": request.task, "length_bucket": length_bucket(request.prompt_chars)},
        )
        for request in requests
    ]
    measurements = []
    for adapter in adapters:
        for spec in adapter.profiles():
            for request in requests:
                if spec.name not in profiles:
                    continue
                measurements.append(annotate_measurement(adapter.profile(request, spec.name, dry_run=args.dry_run), request, fallback_requests))

    measurements = with_quality(measurements, exact_profiles(profiles))
    try:
        validate_profile_measurements(
            measurements,
            output,
            required_profiles=profiles,
            require_measured=not args.dry_run,
        )
    except ValueError as exc:
        write_csv(Path(output), [measurement.to_row() for measurement in measurements])
        print(json.dumps({
            "ok": False,
            "output": output,
            "error": str(exc),
            "failures": failed_measurement_summary(measurements),
        }, ensure_ascii=False, indent=2))
        return 2
    write_csv(Path(output), [measurement.to_row() for measurement in measurements])
    summary = MetricCollector().summarize_profiles(measurements)
    print(json.dumps(json_ready({
        "output": output,
        "rows": len(measurements),
        "builtin_request_fallback": fallback_requests,
        "summary": summary,
    }), indent=2))
    return 0 if all(measurement.ok for measurement in measurements) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 request x profile 统一表。")
    add_profile_table_arguments(parser)
    return parser


def main() -> int:
    return run_command(build_profile_table, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
