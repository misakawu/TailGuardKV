#!/usr/bin/env python3
"""Run a fail-closed GPU gate before the session25 diagnostic policy grid."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.qwen_session import OnlineQwenSessionBackend
from profiles.registry import build_profile_adapters
from run_util.core_types import Action, BackendResult, ProfileMeasurement, Request
from run_util.data_utils import load_requests, split_measurements
from run_util.experiment_common import config_adapters, config_runtime, load_config, read_measurements


BackendFactory = Callable[[], object]


def _split_request_ids(config: dict, measurements: list[ProfileMeasurement]) -> tuple[set[str], set[str]]:
    data_config = config.get("data", {})
    calibration, evaluation = split_measurements(
        measurements,
        split_seed=int(data_config.get("split_seed", 20260906)),
        calibration_fraction=float(data_config.get("calibration_fraction", 0.5)),
        stratify_session=bool(data_config.get("stratify_session", True)),
    )
    return ({row.request_id for row in calibration}, {row.request_id for row in evaluation})


def select_validation_sessions(
    config: dict,
    measurements: list[ProfileMeasurement],
    requests: list[Request],
) -> list[list[Request]]:
    _, evaluation_ids = _split_request_ids(config, measurements)
    by_session: dict[str, list[Request]] = {}
    for request in requests:
        if request.request_id not in evaluation_ids:
            continue
        session_id = request.session_id or request.request_id
        by_session.setdefault(session_id, []).append(request)
    complete = []
    for session_id in sorted(by_session):
        rows = sorted(by_session[session_id], key=lambda item: (item.turn_index, item.arrival_index, item.request_id))
        if [row.turn_index for row in rows[:3]] == [0, 1, 2]:
            complete.append(rows)
    if len(complete) < 3:
        raise ValueError("validation requires three complete evaluation sessions")
    return complete[:3]


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def validate_scenario(
    name: str,
    rows: list[dict[str, object]],
    requirements: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not _truthy(row.get("ok")):
            errors.append(f"runtime failure at row {index}: {row.get('error') or 'unknown error'}")
        ttft = row.get("ttft_ms")
        if not isinstance(ttft, (int, float)) or not math.isfinite(float(ttft)) or float(ttft) <= 0:
            errors.append(f"invalid TTFT at row {index}: {ttft}")
    if requirements.get("require_reuse_after_first"):
        for index, row in enumerate(rows[1:], start=1):
            if not _truthy(row.get("cache_reused")):
                errors.append(f"cache reuse missing at row {index}")
    if requirements.get("require_transition_last"):
        last = rows[-1] if rows else {}
        transition_ms = float(last.get("runtime_transition_ms") or 0.0)
        recompute_ms = float(last.get("recompute_ms") or 0.0)
        if transition_ms <= 0 and recompute_ms <= 0:
            errors.append("transition/recompute evidence missing on final row")
    return {"name": name, "ok": not errors, "errors": errors, "rows": rows}


def _result_row(result: BackendResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "session_id": result.session_id or "",
        "turn_index": result.turn_index,
        "profile": result.profile,
        "ok": result.ok,
        "measured": result.measured,
        "error": result.error or "",
        "ttft_ms": result.ttft_ms,
        "latency_ms": result.latency_ms,
        "kv_cache_memory_mib": result.kv_cache_memory_mib,
        "resident_kv_mib_after": result.resident_kv_mib_after,
        "recompute_ms": result.recompute_ms,
        "cache_reused": result.extra.get("cache_reused", False),
        "cache_rebuild_reason": result.extra.get("cache_rebuild_reason", ""),
        "runtime_transition_ms": result.extra.get("runtime_transition_ms", 0.0),
        "backend_name": result.backend_name,
    }


def _execute_case(
    backend_factory: BackendFactory,
    requests_and_profiles: list[tuple[Request, str]],
) -> list[dict[str, object]]:
    backend = backend_factory()
    rows: list[dict[str, object]] = []
    try:
        for request, profile in requests_and_profiles:
            cache_state = getattr(backend, "cache_state", None)
            result = backend.execute(request, Action(profile=profile, reason="diagnostic_runtime_gate"), cache_state)
            rows.append(_result_row(result))
            if not result.ok:
                break
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()
    return rows


def run_scenarios(sessions: list[list[Request]], backend_factory: BackendFactory) -> dict[str, object]:
    if len(sessions) < 3:
        raise ValueError("validation requires three sessions")
    definitions = (
        (
            "exact_two_turn",
            [(sessions[0][0], "full_gpu"), (sessions[0][1], "full_gpu")],
            {},
        ),
        (
            "kivi_three_turn",
            [(sessions[1][index], "kivi_4bit_residual64") for index in range(3)],
            {"require_reuse_after_first": True},
        ),
        (
            "runtime_switch",
            [
                (sessions[2][0], "full_gpu"),
                (sessions[1][0], "kivi_4bit_residual64"),
                (sessions[2][1], "full_gpu"),
            ],
            {"require_transition_last": True},
        ),
    )
    scenario_reports: list[dict[str, object]] = []
    for name, sequence, requirements in definitions:
        rows = _execute_case(backend_factory, sequence)
        report = validate_scenario(name, rows, requirements)
        scenario_reports.append(report)
        if not report["ok"]:
            break
    return {
        "diagnostic_only": True,
        "ok": len(scenario_reports) == len(definitions) and all(report["ok"] for report in scenario_reports),
        "scenarios": scenario_reports,
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_validation(config_path: Path, run_root: Path, output_path: Path) -> int:
    report: dict[str, object] = {
        "diagnostic_only": True,
        "ok": False,
        "config": str(config_path.resolve()),
        "run_root": str(run_root.resolve()),
    }
    try:
        config = load_config(config_path)
        if config.get("diagnostic_only") is not True or config.get("policies", {}).get("backend") != "online_qwen":
            raise ValueError("diagnostic online_qwen config required")
        profile_paths = list((run_root / "merged" / "profile_tables").glob("*_profiles*.csv"))
        if len(profile_paths) != 1:
            raise ValueError("expected one merged profile CSV")
        measurements = read_measurements(profile_paths[0])
        data_config = config.setdefault("data", {})
        request_path = Path(str(data_config.get("requests") or ""))
        if not request_path.is_absolute():
            data_config["requests"] = str((ROOT / request_path).resolve())
        requests, _ = load_requests(config)
        calibration_ids, evaluation_ids = _split_request_ids(config, measurements)
        sessions = select_validation_sessions(config, measurements, requests)
        runtime_config = config_runtime(config)
        adapters = config_adapters(config)

        def backend_factory() -> OnlineQwenSessionBackend:
            return OnlineQwenSessionBackend(
                adapters=build_profile_adapters(adapters, runtime_config),
            )

        scenario_report = run_scenarios(sessions, backend_factory)
        report.update(
            {
                "profile_table": str(profile_paths[0].resolve()),
                "total_request_count": len({row.request_id for row in measurements}),
                "calibration_request_count": len(calibration_ids),
                "evaluation_request_count": len(evaluation_ids),
                "selected_sessions": [session[0].session_id or session[0].request_id for session in sessions],
                **scenario_report,
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    _write_json_atomic(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") is True else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run_validation(args.config.resolve(), args.run_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
