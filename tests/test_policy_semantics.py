from __future__ import annotations

import pytest

from policies.static_best import StaticBestPolicy
from policies.static_safe import StaticSafePolicy
from policies.uncalibrated_dynamic import UncalibratedDynamicPolicy
from policies.utility_dynamic import UtilityDynamicPolicy
from policies.registry import build_policies
from run_util.core_types import CacheState, DeviceState, ProfileMeasurement, Request


def _measurement(
    request_id: str,
    profile: str,
    *,
    loss: float,
    ttft_ms: float,
    kv_mib: float = 10.0,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request_id,
        session_id="calibration",
        profile=profile,
        adapter="test",
        ok=True,
        measured=True,
        output_text="answer",
        ttft_ms=ttft_ms,
        peak_memory_mib=kv_mib,
        kv_cache_memory_mib=kv_mib,
        kv_incremental_mib=kv_mib,
        kv_cumulative_mib=kv_mib,
        quality_loss=loss,
        extra={"task": "qa", "length_bucket": "short"},
    )


def _request() -> Request:
    return Request("evaluation", "qa", "prompt", metadata={"length_bucket": "short"})


def test_static_best_requires_a_five_percent_p95_ttft_benefit_over_full() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_near", loss=0.01, ttft_ms=96.0),
        _measurement("c2", "lossy_near", loss=0.01, ttft_ms=96.0),
        _measurement("c1", "lossy_fast", loss=0.01, ttft_ms=90.0),
        _measurement("c2", "lossy_fast", loss=0.01, ttft_ms=90.0),
    ]

    near_only = StaticBestPolicy(
        calibration[:4], ["full_gpu", "lossy_near"], 0.05, 0.05, {"full_gpu"}
    )
    fastest = StaticBestPolicy(
        calibration, ["full_gpu", "lossy_near", "lossy_fast"], 0.05, 0.05, {"full_gpu"}
    )

    assert near_only.decide(_request(), CacheState(), DeviceState()).profile == "full_gpu"
    assert fastest.decide(_request(), CacheState(), DeviceState()).profile == "lossy_fast"


def test_static_best_uses_merged_calibration_p95_point_estimates() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_spiky", loss=0.01, ttft_ms=20.0),
        _measurement("c2", "lossy_spiky", loss=0.01, ttft_ms=200.0),
        _measurement("c1", "lossy_steady", loss=0.01, ttft_ms=80.0),
        _measurement("c2", "lossy_steady", loss=0.01, ttft_ms=80.0),
    ]
    policy = StaticBestPolicy(
        calibration, ["full_gpu", "lossy_spiky", "lossy_steady"], 0.05, 0.05, {"full_gpu"}
    )

    assert policy.decide(_request(), CacheState(), DeviceState()).profile == "lossy_steady"


def test_static_best_keeps_exact_when_exact_merged_p95_ttft_is_infinite() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=float("inf")),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=float("inf")),
        _measurement("c1", "lossy_fast", loss=0.01, ttft_ms=10.0),
        _measurement("c2", "lossy_fast", loss=0.01, ttft_ms=10.0),
    ]
    policy = StaticBestPolicy(
        calibration, ["full_gpu", "lossy_fast"], 0.05, 0.05, {"full_gpu"}
    )

    assert policy.decide(_request(), CacheState(), DeviceState()).profile == "full_gpu"


def test_static_best_keeps_exact_when_exact_calibration_ttft_contains_nan() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=float("nan")),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_fast", loss=0.01, ttft_ms=10.0),
        _measurement("c2", "lossy_fast", loss=0.01, ttft_ms=10.0),
    ]
    policy = StaticBestPolicy(
        calibration, ["full_gpu", "lossy_fast"], 0.05, 0.05, {"full_gpu"}
    )

    assert policy.decide(_request(), CacheState(), DeviceState()).profile == "full_gpu"


def test_static_best_keeps_exact_when_lossy_calibration_ttft_contains_nan() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_fast", loss=0.01, ttft_ms=float("nan")),
        _measurement("c2", "lossy_fast", loss=0.01, ttft_ms=10.0),
    ]
    policy = StaticBestPolicy(
        calibration, ["full_gpu", "lossy_fast"], 0.05, 0.05, {"full_gpu"}
    )

    assert policy.decide(_request(), CacheState(), DeviceState()).profile == "full_gpu"


def test_static_safe_records_primary_lossy_profile_when_conformal_guard_falls_back() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c2", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_low_loss", loss=0.00, ttft_ms=40.0),
        _measurement("c2", "lossy_low_loss", loss=0.07, ttft_ms=40.0),
        _measurement("c1", "lossy_higher_loss", loss=0.04, ttft_ms=10.0),
        _measurement("c2", "lossy_higher_loss", loss=0.04, ttft_ms=10.0),
    ]
    policy = StaticSafePolicy(
        calibration,
        ["full_gpu", "lossy_low_loss", "lossy_higher_loss"],
        0.06,
        0.9,
        {"full_gpu"},
    )

    action = policy.decide(_request(), CacheState(), DeviceState())

    assert policy.profile == "lossy_low_loss"
    assert action.profile == "full_gpu"
    assert action.rejected_profile == "lossy_low_loss"
    assert action.fallback_reason == "calibrated unsafe"


def test_uncalibrated_dynamic_chooses_lowest_predicted_ttft_among_point_safe_lossy_profiles() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy_low_loss_slow", loss=0.01, ttft_ms=60.0),
        _measurement("c1", "lossy_higher_loss_fast", loss=0.04, ttft_ms=20.0),
    ]
    policy = UncalibratedDynamicPolicy(
        calibration,
        ["full_gpu", "lossy_low_loss_slow", "lossy_higher_loss_fast"],
        0.05,
        0.05,
        {"full_gpu"},
    )

    assert policy.decide(_request(), CacheState(), DeviceState()).profile == "lossy_higher_loss_fast"


def test_utility_dynamic_updates_one_global_dual_variable_between_decisions() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0, kv_mib=100.0),
        _measurement("c1", "lossy_fast_unsafe", loss=1.0, ttft_ms=10.0, kv_mib=10.0),
        _measurement("c1", "lossy_slow_safe", loss=0.0, ttft_ms=20.0, kv_mib=10.0),
    ]
    policy = UtilityDynamicPolicy(
        calibration,
        ["full_gpu", "lossy_fast_unsafe", "lossy_slow_safe"],
        0.05,
        0.05,
        {"full_gpu"},
    )

    first = policy.decide(_request(), CacheState(), DeviceState())
    second = policy.decide(_request(), CacheState(), DeviceState())

    assert first.profile == "lossy_fast_unsafe"
    assert policy.dual_lambda == pytest.approx(0.4573223304703363)
    assert second.profile == "lossy_slow_safe"


def test_registry_rejects_retired_utility_dynamic_weights() -> None:
    calibration = [
        _measurement("c1", "full_gpu", loss=0.0, ttft_ms=100.0),
        _measurement("c1", "lossy", loss=0.01, ttft_ms=10.0),
    ]

    with pytest.raises(ValueError) as error:
        build_policies(
            [{"type": "utility_dynamic", "memory_weight": 0.0, "loss_weight": 0.0}],
            calibration,
            calibration,
            ["full_gpu", "lossy"],
            0.05,
            0.05,
            {"full_gpu"},
        )

    message = str(error.value)
    assert "memory_weight" in message
    assert "loss_weight" in message
    assert "fixed formula" in message
