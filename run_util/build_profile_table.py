from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

from run_util.experiment_common import (
    annotate_measurement,
    config_adapters,
    config_profiles,
    config_quality_mode,
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
    validate_requests_for_quality_mode,
    validate_profile_measurements,
    with_quality,
    write_csv,
)
from metrics import MetricCollector
from profiles.base import PersistentWorkerFatalError, create_persistent_profile_worker
from profiles.registry import build_profile_adapters
from run_util.cli_common import add_profile_table_arguments, print_error, run_command


DEFAULT_PROFILE_CHUNK_SIZE = 10
PROFILE_TABLE_FIELDNAMES = [
    "adapter",
    "error",
    "session_id",
    "turn_index",
    "latency_ms",
    "measured",
    "ok",
    "output_text",
    "peak_memory_mib",
    "kv_cache_memory_mib",
    "kv_incremental_mib",
    "kv_cumulative_mib",
    "profile",
    "quality_loss",
    "quality_score",
    "request_id",
    "resident_memory_mib",
    "resident_kv_mib_before",
    "resident_kv_mib_after",
    "restore_ms",
    "recompute_ms",
    "evicted_kv_mib",
    "budget_hit",
    "ttft_ms",
    "length_bucket",
    "split",
    "task",
    "extra_backend",
    "extra_bits",
    "extra_builtin_request_fallback",
    "extra_dry_run",
    "extra_env",
    "extra_error_type",
    "extra_failure_stage",
    "extra_family",
    "extra_h2o_recent_size",
    "extra_h2o_selected_size",
    "extra_h2o_kept_tokens",
    "extra_h2o_cache_budget",
    "extra_h2o_prompt_tokens",
    "extra_kivi_bits",
    "extra_kivi_effective_mode",
    "extra_kivi_group_size",
    "extra_kivi_quantization_triggered",
    "extra_kivi_residual_length",
    "extra_metric_loss_em",
    "extra_metric_loss_f1",
    "extra_metric_loss_rouge_l",
    "extra_model",
    "extra_arrival_index",
    "extra_history_turns",
    "extra_prompt_text",
    "extra_effective_prompt_chars",
    "extra_primary_metric",
    "extra_stage_startup_ms",
    "extra_stage_model_load_ms",
    "extra_stage_tokenize_ms",
    "extra_stage_transfer_ms",
    "extra_stage_generate_ms",
    "extra_stage_decode_ms",
    "extra_stage_prefill_ms",
    "extra_stage_first_token_ms",
    "extra_stage_total_ms",
    "extra_original_request_id",
    "extra_profile_note",
    "extra_reference",
    "extra_repeat_index",
    "extra_request_source",
    "extra_returncode",
    "extra_note",
    "extra_source",
    "extra_strategy",
    "extra_ttft_semantics",
    "extra_unsupported",
    "extra_worker_mode",
    "extra_vllm_cache_hits",
    "extra_vllm_cache_misses",
    "extra_vllm_cached_blocks",
    "extra_vllm_evictions",
    "extra_vllm_policy_time_ms",
    "extra_vllm_eviction_decision_time_ms",
]

PROFILE_TRACE_FIELDNAMES = [
    "session_id",
    "turn_index",
    "request_id",
    "profile",
    "event",
    "kv_incremental_mib",
    "kv_cumulative_mib",
    "resident_kv_mib_before",
    "resident_kv_mib_after",
    "restore_ms",
    "recompute_ms",
    "evicted_kv_mib",
    "budget_hit",
]


def _pilot_epsilons(config: dict) -> list[float]:
    pilot = config.get("pilot", {})
    values = pilot.get("epsilons") if isinstance(pilot, dict) else None
    if values is None:
        return [0.05, 0.10]
    if isinstance(values, str):
        values = [values]
    try:
        items = list(values)
    except TypeError:
        items = [values]
    return [float(item) for item in items] or [0.05, 0.10]


def _request_chunks(requests: list, chunk_size: int = DEFAULT_PROFILE_CHUNK_SIZE):
    for start in range(0, len(requests), chunk_size):
        yield start // chunk_size + 1, requests[start : start + chunk_size]


def _active_profile_names(adapters: list, configured_profiles: list[str]) -> list[str]:
    configured = set(configured_profiles)
    active: list[str] = []
    for adapter in adapters:
        for spec in adapter.profiles():
            if spec.name in configured and spec.name not in active:
                active.append(spec.name)
    return active


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


def _trace_rows(measurements: list) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for measurement in measurements:
        if not measurement.session_id:
            continue
        base = {
            "session_id": measurement.session_id,
            "turn_index": measurement.turn_index,
            "request_id": measurement.request_id,
            "profile": measurement.profile,
            "kv_incremental_mib": measurement.kv_incremental_mib,
            "kv_cumulative_mib": measurement.kv_cumulative_mib,
            "resident_kv_mib_before": measurement.resident_kv_mib_before,
            "resident_kv_mib_after": measurement.resident_kv_mib_after,
            "restore_ms": measurement.restore_ms,
            "recompute_ms": measurement.recompute_ms,
            "evicted_kv_mib": measurement.evicted_kv_mib,
            "budget_hit": measurement.budget_hit,
        }
        rows.append({**base, "event": "resident"})
        if measurement.evicted_kv_mib and measurement.evicted_kv_mib > 0:
            rows.append({**base, "event": "evict"})
        if measurement.restore_ms and measurement.restore_ms > 0:
            rows.append({**base, "event": "restore"})
        if measurement.recompute_ms and measurement.recompute_ms > 0:
            rows.append({**base, "event": "recompute"})
    return rows


def _write_trace_if_configured(config: dict, measurements: list) -> None:
    trace_path = config.get("outputs", {}).get("smoke_profile_trace")
    if not trace_path:
        return
    rows = _trace_rows(measurements)
    if not rows:
        return
    path = Path(str(trace_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_TRACE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def _profile_many_compat(
    adapter: object,
    request_chunk: list,
    profile_name: str,
    *,
    dry_run: bool,
    session_runtime: object | None,
    memory_budget_mib: float | None,
    persistent_worker: object | None,
):
    profile_many = getattr(adapter, "profile_many")
    parameters = inspect.signature(profile_many).parameters
    kwargs: dict[str, object] = {"dry_run": dry_run}
    if "session_runtime" in parameters:
        kwargs["session_runtime"] = session_runtime
    if "memory_budget_mib" in parameters:
        kwargs["memory_budget_mib"] = memory_budget_mib
    if "persistent_worker" in parameters:
        kwargs["persistent_worker"] = persistent_worker
    return profile_many(request_chunk, profile_name, **kwargs)


def create_persistent_worker(adapter: object, runtime_config: dict[str, object]):
    adapter_name = str(getattr(adapter, "name", "") or "")
    env_name = str(getattr(adapter, "env", "") or "")
    if adapter_name not in {"full", "kivi", "h2o"} or not env_name:
        return None
    return create_persistent_profile_worker(
        adapter=adapter_name,
        env_name=env_name,
        runtime_module="profiles.qwen2_kv_runtime",
        runtime_config=runtime_config,
        pythonpath=tuple(getattr(adapter, "pythonpath", ()) or ()),
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
        _write_trace_if_configured(config, measurements)
        summary = MetricCollector().summarize_profiles(measurements, epsilons=_pilot_epsilons(config))
        print(json.dumps(json_ready({
            "output": output,
            "rows": len(measurements),
            "imported_from": args.import_measurements,
            "dry_run": bool(args.dry_run),
            "summary": summary,
        }), indent=2))
        return 0 if all(measurement.ok and measurement.measured for measurement in measurements) else 1
    try:
        profiles = config_profiles(config)
        quality_mode = config_quality_mode(config)
        runtime = config_runtime(config)
        require_ttft = bool(runtime.get("require_ttft", False))
        adapters = build_profile_adapters(args.adapters or config_adapters(config), runtime)
        active_profiles = _active_profile_names(adapters, profiles)
        require_quality_loss = quality_mode == "baseline" and "full_gpu" in active_profiles
        requests, fallback_requests = load_requests(config)
        max_requests = int(runtime.get("max_requests", 0) or 0)
        requests = limit_requests_by_split(requests, max_requests)
        requests = expand_repeated_requests(requests, int(runtime.get("repeat", 1)))
        requests = [
            replace(
                request,
                metadata={
                    **request.metadata,
                    "task": request.task,
                    "length_bucket": str(request.metadata.get("length_bucket") or length_bucket(request.prompt_chars)),
                },
            )
            for request in requests
        ]
        output_path = Path(output)
        diagnostic_output = _failed_chunks_output(output_path)
        if output_path.exists():
            output_path.unlink()
        if diagnostic_output.exists():
            diagnostic_output.unlink()
        measurements = []
        exact = exact_profiles(active_profiles, config)
        validate_requests_for_quality_mode(requests, quality_mode, exact)
        session_runtime_by_profile: dict[str, dict[str, object]] = {}
        memory_budget_mib = runtime.get("memory_budget_mib")
        profile_chunk_size = max(1, int(runtime.get("profile_chunk_size", DEFAULT_PROFILE_CHUNK_SIZE) or DEFAULT_PROFILE_CHUNK_SIZE))
        try:
            for adapter in adapters:
                persistent_worker = None
                try:
                    if not args.dry_run and bool(runtime.get("use_persistent_workers", True)):
                        persistent_worker = create_persistent_worker(adapter, runtime)
                    for spec in adapter.profiles():
                        if spec.name not in profiles:
                            continue
                        session_runtime = session_runtime_by_profile.setdefault(spec.name, {})
                        for chunk_index, request_chunk in _request_chunks(requests, profile_chunk_size):
                            try:
                                raw_measurements = _profile_many_compat(
                                    adapter,
                                    request_chunk,
                                    spec.name,
                                    dry_run=args.dry_run,
                                    session_runtime=session_runtime,
                                    memory_budget_mib=memory_budget_mib,
                                    persistent_worker=persistent_worker,
                                )
                            except PersistentWorkerFatalError as exc:
                                diagnostic_output = _failed_chunks_output(output)
                                write_csv(diagnostic_output, [measurement.to_row() for measurement in exc.measurements])
                                print(json.dumps(json_ready({
                                    "ok": False,
                                    "output": output,
                                    "diagnostic_output": str(diagnostic_output),
                                    "error": str(exc),
                                    "failures": failed_measurement_summary(exc.measurements),
                                }), ensure_ascii=False, indent=2))
                                return 2
                            chunk_measurements = [
                                annotate_measurement(measurement, request, fallback_requests)
                                for measurement, request in zip(raw_measurements, request_chunk, strict=True)
                            ]
                            updated = with_quality(measurements + chunk_measurements, exact, quality_mode=quality_mode)
                            chunk_measurements = updated[-len(chunk_measurements) :]
                            try:
                                validate_profile_measurements(
                                    chunk_measurements,
                                    output,
                                    required_profiles=[spec.name],
                                    require_measured=not args.dry_run,
                                    require_ttft=require_ttft and not args.dry_run,
                                    require_quality_loss=require_quality_loss,
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
                            completed_requests = min(chunk_index * profile_chunk_size, len(requests))
                            _print_chunk_progress(
                                adapter=adapter.name,
                                profile=spec.name,
                                chunk_index=chunk_index,
                                completed_requests=completed_requests,
                                total_requests=len(requests),
                                rows=len(measurements),
                                output=output,
                            )
                finally:
                    if persistent_worker is not None:
                        close = getattr(persistent_worker, "close", None)
                        if callable(close):
                            close()
        finally:
            pass
    except (FileNotFoundError, ValueError) as exc:
        print_error(exc, output=output)
        return 2
    except Exception as exc:
        print_error(exc, output=output)
        return 1

    measurements = with_quality(measurements, exact, quality_mode=quality_mode)
    try:
        validate_profile_measurements(
            measurements,
            output,
            required_profiles=active_profiles,
            require_measured=not args.dry_run,
            require_ttft=bool(config_runtime(config).get("require_ttft", False)) and not args.dry_run,
            require_quality_loss=require_quality_loss,
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
    _write_trace_if_configured(config, measurements)
    summary = MetricCollector().summarize_profiles(measurements, epsilons=_pilot_epsilons(config))
    print(json.dumps(json_ready({
        "output": output,
        "rows": len(measurements),
        "builtin_request_fallback": fallback_requests,
        "dry_run": bool(args.dry_run),
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
