from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ModuleNotFoundError:
        return _load_yaml_fallback(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return payload


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    """本地环境缺 PyYAML 时的最小 fallback；生产/实验环境应安装 PyYAML。"""

    config: dict[str, Any] = {}
    section: str | None = None
    list_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if indent == 0 and line.endswith(":"):
            section = line[:-1]
            config[section] = {}
            list_key = None
            continue
        if section is None:
            raise ValueError(f"无法解析配置行: {raw_line}")
        if indent == 2 and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                config[section][key] = _parse_scalar(value)
                list_key = None
            else:
                config[section][key] = []
                list_key = key
            continue
        if indent == 4 and ":" in line and list_key is not None:
            if config[section][list_key] == []:
                config[section][list_key] = {}
            if not isinstance(config[section][list_key], dict):
                raise ValueError(f"无法解析配置行: {raw_line}")
            key, value = line.split(":", 1)
            config[section][list_key][key.strip()] = _parse_scalar(value.strip())
            continue
        if indent == 4 and line.startswith("- ") and list_key is not None:
            config[section][list_key].append(_parse_scalar(line[2:].strip()))
            continue
        raise ValueError(f"无法解析配置行: {raw_line}")
    return config


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


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
    return {
        "pilot_model": model.get("pilot_model") or model.get("path") or model.get("name"),
        "profile_smoke_model": model.get("profile_smoke_model") or model.get("path") or model.get("name"),
        "model_cache_dir": model.get("cache_dir"),
        "max_new_tokens": int(profile.get("max_new_tokens", pilot.get("max_new_tokens", 16))),
        "timeout_s": int(profile.get("timeout_s", profile.get("timeout", 180))),
        "repeat": int(profile.get("repeat", pilot.get("repeats", 1))),
        "local_files_only": bool(profile.get("local_files_only", True)),
        "kivi_group_size": int(profile.get("kivi_group_size", 32)),
        "kivi_residual_length": int(profile.get("kivi_residual_length", 32)),
        "h2o_heavy_ratio": float(profile.get("h2o_heavy_ratio", 0.1)),
        "h2o_recent_ratio": float(profile.get("h2o_recent_ratio", 0.1)),
    }


def exact_profiles(profiles: list[str]) -> set[str]:
    exact_names = {"full_gpu", "full_cpu", "recompute", "exact_offload"}
    return {profile for profile in profiles if profile in exact_names or profile.startswith("full_")}


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
