from __future__ import annotations

from run_util.core_types import ProfileMeasurement


BASELINE_PROFILES = (
    "full_gpu",
    "kivi_4bit_residual32",
    "kivi_4bit_residual64",
    "kivi_2bit_residual32",
    "kivi_2bit_residual64",
    "h2o_heavy10_recent10",
    "h2o_heavy15_recent15",
    "h2o_heavy20_recent20",
)
KIVI_PROFILES = set(BASELINE_PROFILES[1:5])
H2O_PROFILES = set(BASELINE_PROFILES[5:])


def _strict_fixture_and_measurements() -> tuple[list[dict[str, object]], list[ProfileMeasurement]]:
    fixture_rows: list[dict[str, object]] = []
    measurements: list[ProfileMeasurement] = []
    for risk_family in ("kivi_sensitive", "h2o_sensitive", "low_risk"):
        for index in range(60):
            request_id = f"{risk_family}-{index:03d}"
            split = "calibration" if index < 30 else "eval"
            fixture_rows.append(
                {
                    "request_id": request_id,
                    "task": "qa" if index % 2 == 0 else "summary",
                    "prompt": f"final-form prompt {request_id}",
                    "reference": f"reference {request_id}",
                    "metadata": {
                        "source": "external_labeling",
                        "source_dataset": "longbench_qasper",
                        "split": split,
                        "risk_family": risk_family,
                    },
                }
            )
            for profile in BASELINE_PROFILES:
                loss = 0.0 if profile == "full_gpu" else 0.005
                if risk_family == "kivi_sensitive" and profile in KIVI_PROFILES:
                    loss = 0.06
                if risk_family == "h2o_sensitive" and profile in H2O_PROFILES:
                    loss = 0.06
                measurements.append(
                    ProfileMeasurement(
                        request_id=request_id,
                        profile=profile,
                        adapter="test",
                        ok=True,
                        measured=True,
                        quality_loss=loss,
                        extra={"original_request_id": request_id, "split": split},
                    )
                )
    return fixture_rows, measurements


def test_baseline_quality_signal_gate_accepts_strict_final_form_evidence() -> None:
    fixture_rows, measurements = _strict_fixture_and_measurements()

    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate

    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)

    assert result.passed is True
    assert result.fixture_group_counts == {"h2o_sensitive": 60, "kivi_sensitive": 60, "low_risk": 60}
    assert result.complete_profiles == list(BASELINE_PROFILES)
    assert KIVI_PROFILES & set(result.qualifying_profiles)
    assert H2O_PROFILES & set(result.qualifying_profiles)


def test_baseline_quality_signal_gate_rejects_incomplete_profile_measurements() -> None:
    fixture_rows, measurements = _strict_fixture_and_measurements()
    measurements.pop()

    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate

    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)

    assert result.passed is False
    assert any("complete measurement" in error for error in result.errors)


def test_baseline_quality_signal_gate_rejects_relabeled_tie() -> None:
    fixture_rows, measurements = _strict_fixture_and_measurements()
    for measurement in measurements:
        if measurement.request_id == "kivi_sensitive-000" and measurement.profile != "full_gpu":
            object.__setattr__(measurement, "quality_loss", 0.06)

    from scripts.validate_trace_quality import validate_baseline_quality_signal_gate

    result = validate_baseline_quality_signal_gate(measurements, fixture_rows)

    assert result.passed is False
    assert any("tie" in error for error in result.errors)


def test_failed_quality_gate_marks_policy_output_risk_evidence_insufficient() -> None:
    from run_util.experiment import apply_signal_gate_policy_status

    policy_payload = {"rows": 5, "summary": {"full_lru": {"count": 5.0}}}

    status = apply_signal_gate_policy_status(policy_payload, gate_passed=False)

    assert status == "risk_evidence_insufficient"
    assert policy_payload["rows"] == 5
    assert policy_payload["summary"]["full_lru"]["policy_comparison_status"] == status
