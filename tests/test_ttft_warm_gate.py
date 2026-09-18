from __future__ import annotations

from scripts.ttft_warm_gate import validate_gate_records


def _rows(*, count: int = 75) -> list[dict[str, object]]:
    policies = ("full_lru", "static_best", "static_safe", "utility_dynamic", "uncalibrated_dynamic")
    rows = []
    for policy_index, policy in enumerate(policies):
        for request_index in range(15):
            profile = "full_gpu" if request_index % 2 == 0 else "kivi_4bit_residual64"
            rows.append(
                {
                    "policy": policy,
                    "request_id": f"r{request_index}",
                    "profile": profile,
                    "ok": True,
                    "ttft_ms": 20.0,
                    "recompute_ms": 10.0,
                    "extra_worker_pid": 1000 + policy_index,
                    "extra_worker_generation": 1,
                    "extra_warm_profile_success": True,
                    "extra_worker_startup_ms": 100.0,
                    "extra_worker_model_load_ms": 200.0,
                }
            )
    return rows[:count]


def test_gate_accepts_complete_stable_records_with_audit_only_startup_fields() -> None:
    report = validate_gate_records(
        _rows(),
        calibration_p99_ms={"full_gpu": 50.0, "kivi_4bit_residual64": 50.0},
    )

    assert report["passed"] is True
    assert report["record_count"] == 75


def test_gate_fails_closed_for_missing_record_state_loss_and_threshold_violation() -> None:
    rows = _rows(count=74)
    rows[0]["extra_worker_state_lost"] = True
    rows[1]["ttft_ms"] = 251.0

    report = validate_gate_records(
        rows,
        calibration_p99_ms={"full_gpu": 50.0, "kivi_4bit_residual64": 50.0},
    )

    assert report["passed"] is False
    assert any("expected 75 records" in error for error in report["errors"])
    assert any("worker state loss" in error for error in report["errors"])
    assert any("TTFT threshold" in error for error in report["errors"])


def test_gate_rejects_startup_fields_mixed_into_service_columns() -> None:
    rows = _rows()
    rows[0]["worker_model_load_ms"] = 200.0

    report = validate_gate_records(
        rows,
        calibration_p99_ms={"full_gpu": 50.0, "kivi_4bit_residual64": 50.0},
    )

    assert report["passed"] is False
    assert any("service column" in error for error in report["errors"])


def test_gate_accepts_first_ttft_above_three_seconds_when_within_profile_p99_limit() -> None:
    rows = _rows()
    rows[0]["ttft_ms"] = 3500.0

    report = validate_gate_records(
        rows,
        calibration_p99_ms={"full_gpu": 1000.0, "kivi_4bit_residual64": 1000.0},
    )

    assert report["passed"] is True
