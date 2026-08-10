from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return payload


def config_adapters(config: dict[str, Any]) -> list[str]:
    profiles_config = _required_mapping(config, "profiles")
    return _required_string_list(profiles_config, "profiles.adapters")


def config_profiles(config: dict[str, Any]) -> list[str]:
    profiles_config = _required_mapping(config, "profiles")
    return _required_string_list(profiles_config, "profiles.names")


def config_policies(config: dict[str, Any]) -> list[str | dict[str, Any]]:
    policies_config = _required_mapping(config, "policies")
    items = policies_config.get("items")
    if items is not None:
        if not isinstance(items, list) or not items:
            raise ValueError("配置缺少必需列表: policies.items")
        return items
    return _required_string_list(policies_config, "policies.names")


def config_runtime(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    pilot = config.get("pilot", {})
    profile = config.get("profile_smoke", {})
    data = config.get("data", {})
    return {
        "pilot_model": model.get("pilot_model") or model.get("path") or model.get("name"),
        "model_cache_dir": model.get("cache_dir"),
        "max_new_tokens": int(profile.get("max_new_tokens", pilot.get("max_new_tokens", 16))),
        "timeout_s": int(profile.get("timeout_s", profile.get("timeout", 180))),
        "repeat": int(profile.get("repeat", pilot.get("repeats", 1))),
        "profile_chunk_size": int(profile.get("profile_chunk_size", 10)),
        "use_persistent_workers": bool(profile.get("use_persistent_workers", True)),
        "max_requests": int(data.get("max_requests", profile.get("max_requests", 0)) or 0),
        "local_files_only": bool(profile.get("local_files_only", True)),
        "require_ttft": bool(profile.get("require_ttft", False)),
        "device_mode": str(profile.get("device_mode", "auto")),
        "device_strategy": str(profile.get("device_strategy", "balanced_two_gpu")),
        "cuda_visible_devices": str(profile.get("cuda_visible_devices", "")),
        "full_device_mode": str(profile.get("full_device_mode", profile.get("device_mode", "auto"))),
        "full_device_strategy": str(profile.get("full_device_strategy", profile.get("device_strategy", "balanced_two_gpu"))),
        "full_cuda_visible_devices": str(profile.get("full_cuda_visible_devices", profile.get("cuda_visible_devices", ""))),
        "kivi_device_strategy": str(profile.get("kivi_device_strategy", profile.get("device_strategy", "balanced_two_gpu"))),
        "kivi_cuda_visible_devices": str(profile.get("kivi_cuda_visible_devices", profile.get("cuda_visible_devices", ""))),
        "h2o_device_strategy": str(profile.get("h2o_device_strategy", profile.get("device_strategy", "balanced_two_gpu"))),
        "h2o_cuda_visible_devices": str(profile.get("h2o_cuda_visible_devices", profile.get("cuda_visible_devices", ""))),
        "vllm_enforce_eager": bool(profile.get("vllm_enforce_eager", True)),
        "vllm_gpu_memory_utilization": float(profile.get("vllm_gpu_memory_utilization", 0.75)),
        "vllm_max_model_len": int(profile.get("vllm_max_model_len", 1024)),
        "vllm_cuda_visible_devices": str(profile.get("vllm_cuda_visible_devices", "")),
        "vllm_tensor_parallel_size": int(profile.get("vllm_tensor_parallel_size", 1)),
        "kivi_group_size": int(profile.get("kivi_group_size", 32)),
        "kivi_residual_length": int(profile.get("kivi_residual_length", 32)),
        "h2o_heavy_ratio": float(profile.get("h2o_heavy_ratio", 0.1)),
        "h2o_recent_ratio": float(profile.get("h2o_recent_ratio", 0.1)),
    }


def config_quality_mode(config: dict[str, Any]) -> str:
    data = config.get("data", {})
    if not isinstance(data, dict):
        return "session_diagnostic"
    mode = str(data.get("quality_mode", "session_diagnostic") or "session_diagnostic").strip().lower()
    if mode not in {"baseline", "session_diagnostic"}:
        raise ValueError(f"data.quality_mode 仅支持 baseline 或 session_diagnostic: {mode}")
    return mode


def config_experiment_type(config: dict[str, Any]) -> str:
    """Return the explicit dual-track experiment type, with legacy compatibility."""
    experiment = config.get("experiment", {})
    if isinstance(experiment, dict) and experiment.get("type"):
        experiment_type = str(experiment["type"]).strip().lower()
    else:
        data = config.get("data", {})
        quality_mode = (
            str(data.get("quality_mode", "baseline")).strip().lower()
            if isinstance(data, dict)
            else "baseline"
        )
        experiment_type = (
            "baseline_quality"
            if quality_mode == "baseline"
            else "baseline_session"
        )
    if experiment_type not in {"baseline_quality", "baseline_session"}:
        raise ValueError(
            "experiment.type 仅支持 baseline_quality 或 baseline_session: "
            f"{experiment_type}"
        )
    return experiment_type


def exact_profiles(profiles: list[str], config: dict[str, Any] | None = None) -> set[str]:
    if config is not None:
        profiles_config = config.get("profiles", {})
        specs = profiles_config.get("specs", {}) if isinstance(profiles_config, dict) else {}
        if isinstance(specs, dict) and specs:
            return {
                profile
                for profile in profiles
                if isinstance(specs.get(profile), dict) and bool(specs[profile].get("exact"))
            }
    return {profile for profile in profiles if profile in {"full_gpu", "engine_full_lru"}}


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置缺少必需 section: {key}")
    return value


def _required_string_list(config: dict[str, Any], dotted_key: str) -> list[str]:
    key = dotted_key.rsplit(".", 1)[-1]
    value = config.get(key)
    if not isinstance(value, list) or not value or any(item is None or not str(item).strip() for item in value):
        raise ValueError(f"配置缺少必需列表: {dotted_key}")
    return [str(item) for item in value]
