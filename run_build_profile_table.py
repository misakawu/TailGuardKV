from __future__ import annotations

import argparse
import csv
import json
import sys
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
    limit_requests_by_split,
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


PROFILE_CHUNK_SIZE = 10
PROFILE_TABLE_FIELDNAMES = [
    "adapter",
    "error",
    "latency_ms",
    "measured",
    "ok",
    "output_text",
    "peak_memory_mib",
    "profile",
    "quality_loss",
    "quality_score",
    "request_id",
    "resident_memory_mib",
    "ttft_ms",
    "length_bucket",
    "split",
    "task",
    "extra_backend",
    "extra_bits",
    "extra_builtin_request_fallback",
    "extra_env",
    "extra_error_type",
    "extra_failure_stage",
    "extra_family",
    "extra_h2o_recent_size",
    "extra_h2o_selected_size",
    "extra_kivi_bits",
    "extra_kivi_effective_mode",
    "extra_kivi_group_size",
    "extra_kivi_quantization_triggered",
    "extra_kivi_residual_length",
    "extra_metric_em",
    "extra_metric_f1",
    "extra_metric_rouge_l",
    "extra_model",
    "extra_stage_startup_ms",
    "extra_stage_model_load_ms",
    "extra_stage_tokenize_ms",
    "extra_stage_transfer_ms",
    "extra_stage_generate_ms",
    "extra_stage_decode_ms",
    "extra_stage_total_ms",
    "extra_original_request_id",
    "extra_profile_note",
    "extra_reference",
    "extra_repeat_index",
    "extra_request_source",
    "extra_returncode",
    "extra_note",
    "extra_strategy",
    "extra_unsupported",
    "extra_worker_mode",
    "extra_vllm_cache_hits",
    "extra_vllm_cache_misses",
    "extra_vllm_cached_blocks",
    "extra_vllm_evictions",
    "extra_vllm_policy_time_ms",
    "extra_vllm_eviction_decision_time_ms",
]


def _request_chunks(requests: list, chunk_size: int = PROFILE_CHUNK_SIZE):
    for start in range(0, len(requests), chunk_size):
        yield start // chunk_size + 1, requests[start : start + chunk_size]


def _append_profile_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_TABLE_FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def _failed_chunks_output(output: str | Path) -> Path:
    path = Path(output)
    return path.with_name(f"{path.stem}_failed_chunks.csv")


def _print_chunk_progress(
    *,
    adapter: str,
    profile: str,
    chunk_index: int,
    completed_requests: int,
    total_requests: int,
    rows: int,
    output: str,
) -> None:
    print(
        json.dumps(
            {
                "event": "profile_chunk_complete",
                "adapter": adapter,
                "profile": profile,
                "chunk_index": chunk_index,
                "completed_requests": completed_requests,
                "total_requests": total_requests,
                "percent": round(completed_requests * 100.0 / total_requests, 2) if total_requests else 100.0,
                "rows": rows,
                "output": output,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def build_profile_table(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.config))
        output = args.output or config.get("outputs", {}).get("smoke_profiles", "out/profile_tables/smoke_profiles.csv")
    except (FileNotFoundError, ValueError) as exc:
        print_error(exc, output=getattr(args, "output", None) or "")
        return 2
    except Exception as exc:
        print_error(exc, output=getattr(args, "output", None) or "")
        return 1
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
    try:
        profiles = config_profiles(config)
        runtime = config_runtime(config)
        adapters = build_profile_adapters(args.adapters or config_adapters(config), runtime)
        requests, fallback_requests = load_requests(config)
        max_requests = int(runtime.get("max_requests", 0) or 0)
        requests = limit_requests_by_split(requests, max_requests)
        requests = expand_repeated_requests(requests, int(runtime.get("repeat", 1)))
        requests = [
            replace(
                request,
                metadata={**request.metadata, "task": request.task, "length_bucket": length_bucket(request.prompt_chars)},
            )
            for request in requests
        ]
        output_path = Path(output)
        if output_path.exists():
            output_path.unlink()
        measurements = []
        exact = exact_profiles(profiles, config)
        for adapter in adapters:
            for spec in adapter.profiles():
                if spec.name not in profiles:
                    continue
                for chunk_index, request_chunk in _request_chunks(requests):
                    chunk_measurements = [
                        annotate_measurement(measurement, request, fallback_requests)
                        for measurement, request in zip(
                            adapter.profile_many(request_chunk, spec.name, dry_run=args.dry_run),
                            request_chunk,
                            strict=True,
                        )
                    ]
                    updated = with_quality(measurements + chunk_measurements, exact)
                    chunk_measurements = updated[-len(chunk_measurements) :]
                    try:
                        validate_profile_measurements(
                            chunk_measurements,
                            output,
                            required_profiles=[spec.name],
                            require_measured=not args.dry_run,
                        )
                    except ValueError as exc:
                        diagnostic_output = _failed_chunks_output(output)
                        write_csv(diagnostic_output, [measurement.to_row() for measurement in chunk_measurements])
                        print(json.dumps(json_ready({
                            "ok": False,
                            "output": output,
                            "diagnostic_output": str(diagnostic_output),
                            "error": str(exc),
                            "failures": failed_measurement_summary(chunk_measurements),
                        }), ensure_ascii=False, indent=2))
                        return 2
                    measurements = updated
                    _append_profile_rows(output_path, [measurement.to_row() for measurement in chunk_measurements])
                    completed_requests = min(chunk_index * PROFILE_CHUNK_SIZE, len(requests))
                    _print_chunk_progress(
                        adapter=adapter.name,
                        profile=spec.name,
                        chunk_index=chunk_index,
                        completed_requests=completed_requests,
                        total_requests=len(requests),
                        rows=len(measurements),
                        output=output,
                    )
    except (FileNotFoundError, ValueError) as exc:
        print_error(exc, output=output)
        return 2
    except Exception as exc:
        print_error(exc, output=output)
        return 1

    measurements = with_quality(measurements, exact_profiles(profiles, config))
    try:
        validate_profile_measurements(
            measurements,
            output,
            required_profiles=profiles,
            require_measured=not args.dry_run,
        )
    except ValueError as exc:
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
