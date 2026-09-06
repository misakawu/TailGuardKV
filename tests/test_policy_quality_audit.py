from __future__ import annotations

from math import sqrt

import pytest

from backends.measured_replay import MeasuredReplayBackend
from metrics import MetricCollector
from policies.full_lru import FullLRUPolicy
from policies.static_safe import StaticSafePolicy
from policies.utility_dynamic import UtilityDynamicPolicy
from run_util import run_policies
from run_util.core_types import CacheState, DeviceState, ProfileMeasurement, Request
from run_util.experiment_summary import total_policy_summary_rows


def _request(index: int) -> Request:
    return Request(
        request_id=f"r{index:02d}",
        task="qa",
        prompt=f"question {index}",
        session_id=f"s{index:02d}",
        turn_index=0,
        arrival_index=index,
        metadata={"length_bucket": "short"},
    )


def _measurement(
    request: Request,
    profile: str,
    *,
    loss: float,
    ttft_ms: float,
    kv_mib: float,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request.request_id,
        session_id=request.session_id,
        turn_index=request.turn_index,
        profile=profile,
        adapter="test",
        ok=True,
        measured=True,
        output_text=request.request_id,
        latency_ms=ttft_ms + 5.0,
        ttft_ms=ttft_ms,
        peak_memory_mib=kv_mib,
        kv_cache_memory_mib=kv_mib,
        resident_memory_mib=kv_mib,
        kv_incremental_mib=kv_mib,
        kv_cumulative_mib=kv_mib,
        quality_loss=loss,
        extra={"task": "qa", "length_bucket": "short"},
    )


def test_shadow_audit_uses_a_stable_ten_percent_request_set() -> None:
    requests = [_request(index) for index in range(20)]

    first = run_policies._shadow_audit_request_keys(requests)
    second = run_policies._shadow_audit_request_keys(list(reversed(requests)))

    assert first == second
    assert len(first) == 2
    assert first <= {(request.session_id or "", request.turn_index, request.request_id) for request in requests}


def test_shadow_audit_samples_the_emitted_evaluation_population() -> None:
    requests = [_request(index) for index in range(20)]
    evaluation_requests = [request for index, request in enumerate(requests) if index not in {9, 11}][:10]
    evaluation_keys = {
        (request.session_id or "", request.turn_index, request.request_id) for request in evaluation_requests
    }
    measurements = [
        _measurement(request, "full_gpu", loss=0.0, ttft_ms=100.0, kv_mib=100.0)
        for request in requests
    ]

    records = run_policies._run_policy_matrix(
        [FullLRUPolicy("full_gpu")],
        requests,
        MeasuredReplayBackend(measurements),
        {"full_gpu"},
        evaluation_request_keys=evaluation_keys,
    )

    assert len(records) == 10
    assert sum(record.audit_selected for record in records) == 1
    assert {record.request_id for record in records if record.audit_selected} <= {
        request.request_id for request in evaluation_requests
    }


def test_shadow_audit_records_observed_and_inverse_probability_quality_estimates() -> None:
    requests = [_request(index) for index in range(10)]
    audit_keys = run_policies._shadow_audit_request_keys(requests)
    audited_request_id = next(request_id for _, _, request_id in audit_keys)
    calibration_request = _request(99)
    calibration = [
        _measurement(calibration_request, "full_gpu", loss=0.0, ttft_ms=100.0, kv_mib=100.0),
        _measurement(calibration_request, "lossy", loss=0.1, ttft_ms=10.0, kv_mib=10.0),
    ]
    policy = UtilityDynamicPolicy(calibration, ["full_gpu", "lossy"], 0.05, 0.05, {"full_gpu"})
    evaluation = [
        measurement
        for request in requests
        for measurement in (
            _measurement(request, "full_gpu", loss=0.0, ttft_ms=100.0, kv_mib=100.0),
            _measurement(
                request,
                "lossy",
                loss=0.3 if request.request_id == audited_request_id else 0.0,
                ttft_ms=10.0,
                kv_mib=10.0,
            ),
        )
    ]

    records = run_policies._run_policy_matrix(
        [policy], requests, MeasuredReplayBackend(evaluation), {"full_gpu"}
    )

    audited = next(record for record in records if record.audit_selected)
    unaudited = next(record for record in records if not record.audit_selected)
    assert audited.request_id == audited_request_id
    assert audited.predicted_quality_loss == pytest.approx(0.1)
    assert audited.observed_quality_loss == pytest.approx(0.3)
    assert audited.quality_estimate == pytest.approx(2.1)
    assert unaudited.predicted_quality_loss == pytest.approx(0.1)
    assert unaudited.observed_quality_loss is None
    assert unaudited.quality_estimate == pytest.approx(0.1)

    expected_lambda = 0.0
    for index, record in enumerate(records, start=1):
        expected_lambda = max(0.0, expected_lambda + 0.5 / sqrt(index) * (record.quality_estimate - 0.05))
    assert policy.dual_lambda == pytest.approx(expected_lambda)

    summary = run_policies._add_shadow_audit_summary({"utility_dynamic": {}}, records)
    assert summary["utility_dynamic"]["mean_quality_estimate"] == pytest.approx(0.3)
    assert summary["utility_dynamic"]["audit_sample_count"] == 1.0
    [main_table_row] = total_policy_summary_rows({"policy": {"summary": summary}})
    assert main_table_row["mean_quality_estimate"] == pytest.approx(0.3)
    assert main_table_row["audit_sample_count"] == 1.0


def test_static_safe_fallback_keeps_primary_profile_and_final_execution_metrics() -> None:
    request = _request(0)
    calibration = [
        _measurement(request, "full_gpu", loss=0.0, ttft_ms=100.0, kv_mib=100.0),
        _measurement(request, "lossy", loss=0.2, ttft_ms=10.0, kv_mib=10.0),
    ]
    policy = StaticSafePolicy(calibration, ["full_gpu", "lossy"], 0.05, 0.05, {"full_gpu"})
    final_execution = _measurement(request, "full_gpu", loss=0.0, ttft_ms=125.0, kv_mib=120.0)

    [record] = run_policies._run_policy_matrix(
        [policy], [request], MeasuredReplayBackend([final_execution]), {"full_gpu"}
    )

    assert policy.decide(request, CacheState(), DeviceState()).profile == "full_gpu"
    assert record.primary_profile == "lossy"
    assert record.action_profile == "full_gpu"
    assert record.ttft_ms == 125.0
    assert record.kv_cache_memory_mib == 120.0
    assert record.quality_loss == 0.0
    totals = MetricCollector().summarize_policy_runs(
        [record], epsilon=0.05, delta=0.05, exact_profiles={"full_gpu"}
    )["static_safe"]
    assert totals["mean_ttft_ms"] == 125.0
    assert totals["mean_kv_cache_memory_mib"] == 120.0
    assert totals["mean_quality_loss"] == 0.0
