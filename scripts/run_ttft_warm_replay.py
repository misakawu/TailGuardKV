#!/usr/bin/env python3
"""Replay the three TTFT warmup gate sessions and fail closed on its audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.qwen_session import OnlineQwenSessionBackend
from profiles.registry import build_profile_adapters
from run_util.config_loader import exact_profiles
from run_util.data_utils import load_requests
from run_util.experiment_common import config_adapters, config_policies, config_profiles, config_runtime, load_config, read_measurements
from run_util.io_utils import write_csv
from run_util.run_policies import _build_policy_set, _policy_rows_with_provenance, _run_policy_matrix
from scripts.ttft_warm_gate import calibration_p99, validate_gate_records


GATE_SESSIONS = ("hybrid-session-002", "hybrid-session-003", "hybrid-session-004")


def select_gate_requests(requests):
    selected = [request for request in requests if request.session_id in GATE_SESSIONS]
    selected.sort(key=lambda item: (item.arrival_index, item.session_id or "", item.turn_index, item.request_id))
    if len(selected) != 15:
        raise ValueError(f"expected 15 requests from {GATE_SESSIONS}, got {len(selected)}")
    for session_id in GATE_SESSIONS:
        turns = [request.turn_index for request in selected if request.session_id == session_id]
        if turns != [0, 1, 2, 3, 4]:
            raise ValueError(f"{session_id} must contain turns 0..4, got {turns}")
    return selected


def _p99_from_measurements(measurements) -> dict[str, float]:
    rows = [{"profile": row.profile, "ttft_ms": row.ttft_ms} for row in measurements]
    return calibration_p99(rows)


def run(config_path: Path, measurements_path: Path, output_root: Path) -> int:
    config = load_config(config_path)
    if config.get("policies", {}).get("backend") != "online_qwen":
        raise ValueError("TTFT warm gate requires online_qwen")
    profiles = config_profiles(config)
    measurements = read_measurements(measurements_path)
    requests, _ = load_requests(config)
    gate_requests = select_gate_requests(requests)
    runtime = config_runtime(config)
    adapters = build_profile_adapters(config_adapters(config), runtime)
    exact = exact_profiles(profiles, config)
    pilot = config.get("pilot", {})
    epsilon = float(pilot.get("epsilons", [0.05])[0])
    delta = float(pilot.get("deltas", [0.05])[0])
    policies = _build_policy_set(
        config_policies(config),
        measurements,
        measurements,
        profiles,
        epsilon,
        delta,
        exact,
        float("inf"),
        record_rejected_unsafe=bool(config.get("policies", {}).get("record_rejected_unsafe", False)),
    )
    if tuple(policy.name for policy in policies) != (
        "full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic"
    ):
        raise ValueError("TTFT warm gate requires exactly the five configured policies")
    backend = OnlineQwenSessionBackend(adapters=adapters)
    try:
        records = _run_policy_matrix(policies, gate_requests, backend, exact)
    finally:
        backend.close()
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "ttft_warm_gate_records.csv"
    rows = _policy_rows_with_provenance(records, config, source_config=str(config_path), run_dir=str(output_root))
    write_csv(records_path, rows)
    gate = validate_gate_records(
        rows,
        calibration_p99_ms=_p99_from_measurements(measurements),
        expected_request_ids={request.request_id for request in gate_requests},
    )
    report = {
        **gate,
        "config": str(config_path.resolve()),
        "measurements": str(measurements_path.resolve()),
        "records": str(records_path.resolve()),
        "input_sha256": hashlib.sha256(measurements_path.read_bytes()).hexdigest(),
    }
    (output_root / "ttft_warm_gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return run(args.config.resolve(), args.measurements.resolve(), args.output_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
