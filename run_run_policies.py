from __future__ import annotations

import argparse
import json
from pathlib import Path

from backends.measured_replay import MeasuredReplayBackend
from core_types import CacheState, DeviceState, PolicyRunRecord, ProfileMeasurement, Request
from experiment_common import (
    config_policies,
    config_profiles,
    exact_profiles,
    json_ready,
    load_config,
    read_measurements,
    requests_from_measurements,
    split_measurements,
    validate_profile_measurements,
    write_csv,
)
from metrics import MetricCollector
from policies import build_policies
from policies.base import Policy
from run_cli_common import add_policy_arguments, first_number, print_error, run_command


def _run_settings(args: argparse.Namespace, config: dict) -> tuple[str, list[str], list[str], float, float, float]:
    output = args.output or config.get("outputs", {}).get("smoke_policy", "out/policy_tables/smoke_policy.csv")
    profiles = args.profiles or config_profiles(config)
    policy_names = args.policies or config_policies(config)
    pilot = config.get("pilot", {})
    epsilon = first_number(args.epsilon, pilot.get("epsilons"), default=0.2, name="epsilon")
    delta = first_number(args.delta, pilot.get("deltas"), default=0.05, name="delta")
    memory_budget_mib = first_number(
        getattr(args, "memory_budget_mib", None),
        pilot.get("memory_budgets_mib"),
        default=float("inf"),
        name="memory-budget-mib",
    )
    return output, profiles, policy_names, epsilon, delta, memory_budget_mib


def _load_replay_inputs(
    args: argparse.Namespace,
    profiles: list[str],
) -> tuple[list, list, list]:
    measurements = read_measurements(Path(args.measurements))
    validate_profile_measurements(
        measurements,
        args.measurements,
        required_profiles=profiles,
        require_measured=not args.allow_dry_run_replay,
    )
    if not args.allow_dry_run_replay and any(not measurement.measured for measurement in measurements):
        raise ValueError("run-policies 默认拒绝 dry-run replay；请提供 measured=True 的 profile 表。")
    calibration_measurements, evaluation_measurements = split_measurements(measurements)
    requests = requests_from_measurements(evaluation_measurements)
    if not requests:
        raise ValueError(f"profile 表没有可评估 request: {args.measurements}")
    return measurements, calibration_measurements, requests


def _build_policy_set(
    policy_names: list[str],
    calibration_measurements: list[ProfileMeasurement],
    measurements: list[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    delta: float,
    exact: set[str],
    memory_budget_mib: float,
) -> list[Policy]:
    return build_policies(
        policy_names,
        calibration_measurements,
        measurements,
        profiles,
        epsilon,
        delta,
        exact,
        memory_budget_mib=memory_budget_mib,
    )


def _failure_record(policy: Policy, request: Request, error: BaseException, *, action=None, exact: set[str]) -> PolicyRunRecord:
    profile = action.profile if action is not None else ""
    return PolicyRunRecord(
        policy=policy.name,
        request_id=request.request_id,
        task=str(request.task or request.metadata.get("task") or "unknown"),
        length_bucket=str(request.metadata.get("length_bucket") or "unknown"),
        action_profile=profile,
        ok=False,
        measured=False,
        placeholder=policy.placeholder,
        reason=action.reason if action is not None else "policy error",
        error=str(error),
        exact=profile in exact,
        oracle=bool(getattr(policy, "oracle", False)),
        pred_loss=action.pred_loss if action is not None else None,
        risk_upper=action.risk_upper if action is not None else None,
        safe=action.safe if action is not None else None,
        epsilon=action.epsilon if action is not None else None,
        delta=action.delta if action is not None else None,
        fallback_reason=action.fallback_reason if action is not None else "",
        controller_overhead_ms=action.controller_overhead_ms if action is not None else None,
    )


def _run_policy_matrix(
    policies: list[Policy],
    requests: list[Request],
    backend: MeasuredReplayBackend,
    exact: set[str],
) -> list[PolicyRunRecord]:
    records: list[PolicyRunRecord] = []
    for policy in policies:
        for request in requests:
            try:
                action = policy.decide(request, CacheState(), DeviceState())
            except Exception as exc:
                records.append(_failure_record(policy, request, exc, exact=exact))
                continue
            try:
                measurement = backend.run([request], [action.profile])[0]
                records.append(
                    PolicyRunRecord(
                        policy=policy.name,
                        request_id=request.request_id,
                        task=str(request.task or request.metadata.get("task") or "unknown"),
                        length_bucket=str(request.metadata.get("length_bucket") or "unknown"),
                        action_profile=action.profile,
                        ok=measurement.ok,
                        measured=measurement.measured,
                        placeholder=policy.placeholder,
                        reason=action.reason,
                        error=measurement.error,
                        latency_ms=measurement.latency_ms,
                        ttft_ms=measurement.ttft_ms,
                        peak_memory_mib=measurement.peak_memory_mib,
                        resident_memory_mib=measurement.resident_memory_mib,
                        quality_loss=measurement.quality_loss,
                        exact=action.profile in exact,
                        oracle=bool(getattr(policy, "oracle", False)),
                        pred_loss=action.pred_loss,
                        risk_upper=action.risk_upper,
                        safe=action.safe,
                        epsilon=action.epsilon,
                        delta=action.delta,
                        fallback_reason=action.fallback_reason,
                        controller_overhead_ms=action.controller_overhead_ms,
                    )
                )
            except Exception as exc:
                records.append(_failure_record(policy, request, exc, action=action, exact=exact))
    return records


def run_policies(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.config))
        output, profiles, policy_names, epsilon, delta, memory_budget_mib = _run_settings(args, config)
        measurements, calibration_measurements, requests = _load_replay_inputs(args, profiles)
        backend = MeasuredReplayBackend(measurements, allow_dry_run=args.allow_dry_run_replay)
        exact = exact_profiles(profiles)
        policies = _build_policy_set(
            policy_names,
            calibration_measurements,
            measurements,
            profiles,
            epsilon,
            delta,
            exact,
            memory_budget_mib,
        )
    except (FileNotFoundError, ValueError, KeyError, IndexError) as exc:
        print_error(exc)
        return 2
    records = _run_policy_matrix(policies, requests, backend, exact)

    write_csv(Path(output), [record.to_row() for record in records])
    summary = MetricCollector().summarize_policy_runs(records, epsilon=epsilon, delta=delta, exact_profiles=exact)
    print(
        json.dumps(
            json_ready({
                "output": output,
                "rows": len(records),
                "epsilon": epsilon,
                "delta": delta,
                "memory_budget_mib": memory_budget_mib,
                "summary": summary,
            }),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(record.ok for record in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用同一 measured-replay backend 运行策略。")
    add_policy_arguments(parser)
    return parser


def main() -> int:
    return run_command(run_policies, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
