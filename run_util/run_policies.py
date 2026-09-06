from __future__ import annotations

import argparse
import json
from pathlib import Path

from backends.measured_replay import MeasuredReplayBackend
from run_util.core_types import BackendResult, CacheState, DeviceState, PolicyRunRecord, ProfileMeasurement, Request
from run_util.experiment_common import (
    config_policies,
    config_experiment_type,
    config_profiles,
    exact_profiles,
    json_ready,
    load_config,
    read_measurements,
    requests_from_measurements,
    split_measurements,
    validate_backend_results,
    validate_experiment_policy_records,
    validate_profile_measurements,
    write_csv,
)
from run_util.derive_session_budgets import configured_memory_budgets
from metrics import MetricCollector
from policies import build_policies
from policies.base import Policy
from run_util.cli_common import add_policy_arguments, first_number, print_error, run_command


def _run_settings(args: argparse.Namespace, config: dict) -> tuple[str, list[str], list[str | dict], float, float, float]:
    output = args.output or config.get("outputs", {}).get("smoke_policy", "out/policy_tables/smoke_policy.csv")
    profiles = args.profiles or config_profiles(config)
    if getattr(args, "policy_config", None):
        policy_names = config_policies(load_config(Path(args.policy_config)))
    else:
        policy_names = args.policies or config_policies(config)
    pilot = config.get("pilot", {})
    epsilon = first_number(args.epsilon, pilot.get("epsilons"), default=0.2, name="epsilon")
    delta = first_number(args.delta, pilot.get("deltas"), default=0.05, name="delta")
    memory_budget_mib = first_number(
        getattr(args, "memory_budget_mib", None),
        configured_memory_budgets(pilot),
        default=float("inf"),
        name="memory-budget-mib",
    )
    return output, profiles, policy_names, epsilon, delta, memory_budget_mib


def _load_replay_inputs(
    args: argparse.Namespace,
    profiles: list[str],
    experiment_type: str,
    config: dict,
) -> tuple[list, list, list[Request], set[tuple[str, int, str]]]:
    measurements = read_measurements(Path(args.measurements))
    validate_profile_measurements(
        measurements,
        args.measurements,
        required_profiles=profiles,
        require_measured=not args.allow_dry_run_replay,
        experiment_type=experiment_type,
    )
    if not args.allow_dry_run_replay and any(not measurement.measured for measurement in measurements):
        raise ValueError("run-policies 默认拒绝 dry-run replay；请提供 measured=True 的 profile 表。")
    data_config = config.get("data", {})
    data_config = data_config if isinstance(data_config, dict) else {}
    calibration_measurements, evaluation_measurements = split_measurements(
        measurements,
        split_seed=int(data_config.get("split_seed", 20260906)),
        stratify_session=bool(data_config.get("stratify_session", False)),
    )
    replay_measurements = measurements if experiment_type == "baseline_session" else evaluation_measurements
    replay_requests = requests_from_measurements(replay_measurements)
    evaluation_requests = requests_from_measurements(evaluation_measurements)
    if not replay_requests:
        raise ValueError(f"profile 表没有可评估 request: {args.measurements}")
    evaluation_request_keys = {
        (request.session_id or "", request.turn_index, request.request_id)
        for request in evaluation_requests
    }
    return measurements, calibration_measurements, replay_requests, evaluation_request_keys


def _build_policy_set(
    policy_names: list[str | dict],
    calibration_measurements: list[ProfileMeasurement],
    measurements: list[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    delta: float,
    exact: set[str],
    memory_budget_mib: float,
    record_rejected_unsafe: bool = False,
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
        record_rejected_unsafe=record_rejected_unsafe,
    )


def _failure_record(policy: Policy, request: Request, error: BaseException, *, action=None, exact: set[str]) -> PolicyRunRecord:
    profile = action.profile if action is not None else ""
    return PolicyRunRecord(
        policy=policy.name,
        request_id=request.request_id,
        session_id=request.session_id,
        turn_index=request.turn_index,
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
        safety_reason=action.safety_reason if action is not None else "",
        rejected_profile=action.rejected_profile if action is not None else "",
        rejected_pred_loss=action.rejected_pred_loss if action is not None else None,
        rejected_risk_upper=action.rejected_risk_upper if action is not None else None,
        candidate_safe_count=action.candidate_safe_count if action is not None else None,
        controller_overhead_ms=action.controller_overhead_ms if action is not None else None,
        controller_qrp_ms=action.controller_qrp_ms if action is not None else None,
        controller_cg_ms=action.controller_cg_ms if action is not None else None,
        controller_stc_ms=action.controller_stc_ms if action is not None else None,
        oracle_cost_ms=action.oracle_cost_ms if action is not None else None,
        optimality_gap=action.optimality_gap if action is not None else None,
        audit_rate=action.audit_rate if action is not None else None,
        drift_state=action.drift_state if action is not None else "",
        active_session_count=None,
        budget_hit=action.budget_hit if action is not None else False,
        policy_budget_filtered=action.policy_budget_filtered if action is not None else False,
    )


def _record_from_backend_result(
    policy: Policy,
    request: Request,
    action,
    backend_result: BackendResult,
    exact: set[str],
) -> PolicyRunRecord:
    return PolicyRunRecord.from_action_and_backend_result(
        policy_name=policy.name,
        request=request,
        action_profile=action.profile,
        action_reason=action.reason,
        placeholder=policy.placeholder,
        exact_profiles=exact,
        oracle=bool(getattr(policy, "oracle", False)),
        backend_result=backend_result,
        pred_loss=action.pred_loss,
        risk_upper=action.risk_upper,
        safe=action.safe,
        epsilon=action.epsilon,
        delta=action.delta,
        fallback_reason=action.fallback_reason,
        safety_reason=action.safety_reason,
        rejected_profile=action.rejected_profile,
        rejected_pred_loss=action.rejected_pred_loss,
        rejected_risk_upper=action.rejected_risk_upper,
        candidate_safe_count=action.candidate_safe_count,
        controller_overhead_ms=action.controller_overhead_ms,
        controller_qrp_ms=action.controller_qrp_ms,
        controller_cg_ms=action.controller_cg_ms,
        controller_stc_ms=action.controller_stc_ms,
        oracle_cost_ms=action.oracle_cost_ms,
        optimality_gap=action.optimality_gap,
        audit_rate=action.audit_rate,
        drift_state=action.drift_state,
        active_session_count=_active_session_count(backend_result.extra),
        budget_hit=action.budget_hit,
        policy_budget_filtered=action.policy_budget_filtered,
    )


def _active_session_count(extra: dict[str, object]) -> float | None:
    active = extra.get("active_sessions")
    if isinstance(active, (list, tuple)):
        return float(len(active))
    return None


def _run_policy_matrix(
    policies: list[Policy],
    replay_requests: list[Request],
    backend: MeasuredReplayBackend,
    exact: set[str],
    evaluation_request_keys: set[tuple[str, int, str]] | None = None,
) -> list[PolicyRunRecord]:
    records: list[PolicyRunRecord] = []
    eval_only = evaluation_request_keys is not None
    for policy in policies:
        if hasattr(backend, "reset"):
            backend.reset()
        cache_state = CacheState()
        for request in sorted(
            replay_requests,
            key=lambda item: (item.arrival_index, item.session_id or item.request_id, item.turn_index),
        ):
            request_key = (request.session_id or "", request.turn_index, request.request_id)
            emit_record = (not eval_only) or (request_key in evaluation_request_keys)
            try:
                action = policy.decide(request, cache_state, DeviceState())
            except Exception as exc:
                if emit_record:
                    records.append(_failure_record(policy, request, exc, exact=exact))
                continue
            try:
                backend_result = backend.run([request], [action.profile])[0]
                validate_backend_results([backend_result], path=f"{policy.name}:{request.request_id}")
                cache_state = backend.cache_state
                if emit_record:
                    records.append(_record_from_backend_result(policy, request, action, backend_result, exact))
            except Exception as exc:
                if emit_record:
                    records.append(_failure_record(policy, request, exc, action=action, exact=exact))
    return records


def run_policies(args: argparse.Namespace) -> int:
    try:
        config = load_config(Path(args.config))
        experiment_type = config_experiment_type(config)
        output, profiles, policy_names, epsilon, delta, memory_budget_mib = _run_settings(args, config)
        policy_config = config.get("policies", {})
        record_rejected_unsafe = bool(policy_config.get("record_rejected_unsafe", False)) if isinstance(policy_config, dict) else False
        measurements, calibration_measurements, replay_requests, evaluation_request_keys = _load_replay_inputs(
            args,
            profiles,
            experiment_type,
            config,
        )
        backend = MeasuredReplayBackend(
            measurements,
            allow_dry_run=args.allow_dry_run_replay,
            use_pandas=getattr(args, "use_pandas_replay", False),
            global_budget_mib=memory_budget_mib,
        )
        exact = exact_profiles(profiles, config)
        policies = _build_policy_set(
            policy_names,
            calibration_measurements,
            measurements,
            profiles,
            epsilon,
            delta,
            exact,
            memory_budget_mib,
            record_rejected_unsafe=record_rejected_unsafe,
        )
    except (FileNotFoundError, ValueError) as exc:
        print_error(exc)
        return 2
    except Exception as exc:
        print_error(exc)
        return 1
    records = _run_policy_matrix(
        policies,
        replay_requests,
        backend,
        exact,
        evaluation_request_keys if experiment_type == "baseline_session" else None,
    )

    try:
        write_csv(Path(output), [record.to_row() for record in records])
        validate_experiment_policy_records(records, experiment_type, output)
        summary = MetricCollector().summarize_policy_runs(
            records,
            epsilon=epsilon,
            delta=delta,
            exact_profiles=exact,
            experiment_type=experiment_type,
        )
        print(
            json.dumps(
                json_ready({
                    "output": output,
                    "rows": len(records),
                    "epsilon": epsilon,
                    "delta": delta,
                    "memory_budget_mib": memory_budget_mib,
                    "experiment_type": experiment_type,
                    "summary": summary,
                }),
                ensure_ascii=False,
                indent=2,
            )
        )
    except Exception as exc:
        print_error(exc, output=output, rows=len(records))
        return 1
    return 0 if all(record.ok for record in records) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用同一 measured-replay backend 运行策略。")
    add_policy_arguments(parser)
    return parser


def main() -> int:
    return run_command(run_policies, build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
