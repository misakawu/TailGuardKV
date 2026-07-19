from __future__ import annotations

from core_types import ProfileMeasurement
from policies.base import Policy, StaticProfilePolicy
from policies.full_lru import FullLRUPolicy
from policies.offline_ilp_oracle import OfflineILPOraclePolicy
from policies.quality_oracle import QualityOraclePolicy
from policies.static_best import StaticBestPolicy
from policies.static_safe import StaticSafePolicy
from policies.tailguard import TailGuardPolicy
from policies.uncalibrated_dynamic import UncalibratedDynamicPolicy
from policies.utility_dynamic import UtilityDynamicPolicy


def build_policies(
    names: list[str | dict],
    calibration_measurements: list[ProfileMeasurement],
    oracle_measurements: list[ProfileMeasurement],
    profiles: list[str],
    epsilon: float,
    delta: float,
    exact_profiles: set[str],
    memory_budget_mib: float = float("inf"),
    tailguard_config: dict | None = None,
) -> list[Policy]:
    policies: list[Policy] = []
    for item in names:
        name, options = _normalize_policy_config(item)
        if name == "full_lru":
            policies.append(FullLRUPolicy())
        elif name.startswith("static_profile:"):
            policies.append(StaticProfilePolicy(name.split(":", 1)[1], name=name))
        elif name == "static_profile":
            profile = str(options.get("profile") or "")
            if not profile:
                raise ValueError("static_profile policy 需要 profile 字段")
            policies.append(StaticProfilePolicy(profile, name=options.get("name") or f"static_profile:{profile}"))
        elif name == "static_best":
            policies.append(StaticBestPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib))
        elif name == "static_safe":
            policies.append(StaticSafePolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib))
        elif name == "utility_dynamic":
            policies.append(UtilityDynamicPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib))
        elif name == "uncalibrated_dynamic":
            policies.append(UncalibratedDynamicPolicy(calibration_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib))
        elif name == "tailguard":
            policies.append(
                TailGuardPolicy(
                    calibration_measurements,
                    profiles,
                    epsilon,
                    delta,
                    exact_profiles,
                    memory_budget_mib,
                    stc_config=(tailguard_config or {}).get("stc"),
                )
            )
        elif name == "quality_oracle":
            policies.append(QualityOraclePolicy(oracle_measurements, profiles, epsilon, delta, exact_profiles))
        elif name == "offline_ilp_oracle":
            policies.append(OfflineILPOraclePolicy(oracle_measurements, profiles, epsilon, delta, exact_profiles, memory_budget_mib))
        else:
            raise ValueError(f"未知 policy: {name}")
    return policies


def _normalize_policy_config(item: str | dict) -> tuple[str, dict]:
    if isinstance(item, str):
        return item, {}
    if not isinstance(item, dict):
        raise ValueError(f"policy 配置必须是字符串或 mapping: {item}")
    policy_type = str(item.get("type") or item.get("name") or "")
    if policy_type == "static":
        policy_type = "static_profile"
    if not policy_type:
        raise ValueError(f"policy 配置缺少 type/name: {item}")
    return policy_type, item
