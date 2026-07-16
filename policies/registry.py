from __future__ import annotations

from core_types import ProfileMeasurement
from policies.base import (
    FullLRUPolicy,
    Policy,
    QualityOraclePolicy,
    StaticBestPolicy,
    StaticProfilePolicy,
    StaticSafePolicy,
    TailGuardPolicy,
    UncalibratedDynamicPolicy,
    UtilityDynamicPolicy,
)


def build_policies(
    names: list[str],
    calibration_measurements: list[ProfileMeasurement],
    oracle_measurements: list[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    delta: float,
    exact_profiles: set[str],
) -> list[Policy]:
    policies: list[Policy] = []
    for name in names:
        if name == "full_lru":
            policies.append(FullLRUPolicy())
        elif name.startswith("static_profile:"):
            policies.append(StaticProfilePolicy(name.split(":", 1)[1], name=name))
        elif name == "static_best":
            policies.append(StaticBestPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "static_safe":
            policies.append(StaticSafePolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "utility_dynamic":
            policies.append(UtilityDynamicPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "uncalibrated_dynamic":
            policies.append(UncalibratedDynamicPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "tailguard":
            policies.append(TailGuardPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "quality_oracle":
            policies.append(QualityOraclePolicy(oracle_measurements, profiles, epsilon, delta, exact_profiles))
        else:
            raise ValueError(f"未知 policy: {name}")
    return policies
