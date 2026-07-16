from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from backends.measured_replay import MeasuredReplayBackend
from core_types import CacheState, DeviceState, PolicyRunRecord, ProfileMeasurement, Request
from metrics import MetricCollector
from policies import build_policies
from profiles.registry import build_profile_adapters


def default_requests() -> list[Request]:
    """没有正式数据集前，先用固定 smoke 请求验证表结构和执行链路。"""

    return [
        Request(
            request_id="smoke_qa_001",
            task="qa",
            prompt="Question: What is KV cache used for in autoregressive decoding?\nAnswer:",
            reference="KV cache stores past key/value tensors to avoid recomputing attention history.",
            metadata={"source": "builtin_smoke", "split": "calibration"},
        ),
        Request(
            request_id="smoke_sum_001",
            task="summary",
            prompt="Summarize: KV cache compression can reduce memory but may hurt tail quality.",
            reference="KV compression saves memory while risking large quality loss on some requests.",
            metadata={"source": "builtin_smoke", "split": "eval"},
        ),
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_measurements(path: Path) -> list[ProfileMeasurement]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [ProfileMeasurement.from_row(row) for row in csv.DictReader(handle)]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    try:
        import yaml
    except ModuleNotFoundError as exc:
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


def json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def config_adapters(config: dict[str, Any]) -> list[str]:
    return list(config.get("profiles", {}).get("adapters", ["full", "kivi", "h2o"]))


def config_profiles(config: dict[str, Any]) -> list[str]:
    return list(
        config.get("profiles", {}).get(
            "names",
            ["full_gpu", "kivi_4bit", "kivi_2bit", "h2o_heavy_hitter", "full_cpu", "recompute"],
        )
    )


def config_policies(config: dict[str, Any]) -> list[str]:
    return list(
        config.get("policies", {}).get(
            "names",
            [
                "full_lru",
                "static_best",
                "static_safe",
                "utility_dynamic",
                "uncalibrated_dynamic",
                "tailguard",
                "quality_oracle",
            ],
        )
    )


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
    }


def exact_profiles(profiles: list[str]) -> set[str]:
    exact_names = {"full_gpu", "full_cpu", "recompute", "exact_offload"}
    return {profile for profile in profiles if profile in exact_names or profile.startswith("full_")}


def load_requests(config: dict[str, Any]) -> tuple[list[Request], bool]:
    data_config = config.get("data", {})
    request_path = data_config.get("requests") or data_config.get("request_path")
    if not request_path:
        return default_requests(), True
    path = Path(str(request_path))
    if not path.exists():
        raise FileNotFoundError(f"请求输入文件不存在: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"请求输入仅支持 JSONL/CSV: {path}")
    requests: list[Request] = []
    for index, row in enumerate(rows):
        request_id = str(row.get("request_id") or row.get("id") or f"request_{index:06d}")
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"request_id", "id", "task", "prompt", "reference"}
        }
        metadata.setdefault("source", str(path))
        requests.append(
            Request(
                request_id=request_id,
                task=str(row.get("task") or "unknown"),
                prompt=str(row.get("prompt") or ""),
                reference=(None if row.get("reference") in {None, ""} else str(row.get("reference"))),
                metadata=metadata,
            )
        )
    return _ensure_request_splits(requests, float(data_config.get("calibration_fraction", 0.5))), False


def _ensure_request_splits(requests: list[Request], calibration_fraction: float) -> list[Request]:
    if not requests:
        return []
    if any(request.metadata.get("split") for request in requests):
        return requests
    cutoff = max(1, min(len(requests) - 1, int(round(len(requests) * calibration_fraction)))) if len(requests) > 1 else 1
    return [
        replace(
            request,
            metadata={**request.metadata, "split": "calibration" if index < cutoff else "eval"},
        )
        for index, request in enumerate(requests)
    ]


def requests_from_measurements(measurements: list[ProfileMeasurement]) -> list[Request]:
    seen: dict[str, Request] = {}
    for measurement in measurements:
        if measurement.request_id in seen:
            continue
        task = measurement.request_id.split("_", 1)[0] if "_" in measurement.request_id else "unknown"
        seen[measurement.request_id] = Request(
            request_id=measurement.request_id,
            task=task,
            prompt=measurement.output_text,
            metadata={
                "source": "profile_measurement",
                "split": measurement.extra.get("split", ""),
            },
        )
    return list(seen.values())


def split_measurements(measurements: list[ProfileMeasurement]) -> tuple[list[ProfileMeasurement], list[ProfileMeasurement]]:
    calibration = [row for row in measurements if row.extra.get("split") == "calibration"]
    evaluation = [row for row in measurements if row.extra.get("split") != "calibration"]
    if calibration and evaluation:
        return calibration, evaluation
    request_ids = sorted({row.request_id for row in measurements})
    if len(request_ids) <= 1:
        return measurements, measurements
    cutoff = max(1, len(request_ids) // 2)
    calibration_ids = set(request_ids[:cutoff])
    return (
        [row for row in measurements if row.request_id in calibration_ids],
        [row for row in measurements if row.request_id not in calibration_ids],
    )


def annotate_measurement(measurement: ProfileMeasurement, request: Request, fallback_requests: bool) -> ProfileMeasurement:
    return replace(
        measurement,
        extra={
            **measurement.extra,
            "request_source": request.metadata.get("source", "unknown"),
            "split": request.metadata.get("split", ""),
            "builtin_request_fallback": str(fallback_requests).lower(),
        },
    )


def expand_repeated_requests(requests: list[Request], repeat: int) -> list[Request]:
    if repeat <= 1:
        return requests
    repeated: list[Request] = []
    for repeat_index in range(repeat):
        for request in requests:
            repeated.append(
                replace(
                    request,
                    request_id=f"{request.request_id}__r{repeat_index + 1}",
                    metadata={
                        **request.metadata,
                        "original_request_id": request.request_id,
                        "repeat_index": str(repeat_index + 1),
                    },
                )
            )
    return repeated


def with_quality(measurements: list[ProfileMeasurement], exact: set[str]) -> list[ProfileMeasurement]:
    baseline_by_request = {
        row.request_id: row
        for row in measurements
        if row.profile == "full_gpu" and row.ok and row.measured and row.output_text
    }
    updated: list[ProfileMeasurement] = []
    for row in measurements:
        baseline = baseline_by_request.get(row.request_id)
        if not row.ok or not row.measured or not row.output_text or baseline is None:
            updated.append(row)
            continue
        loss = 0.0 if row.profile in exact else (0.0 if row.output_text.strip() == baseline.output_text.strip() else 1.0)
        updated.append(replace(row, quality_loss=loss, quality_score=1.0 - loss))
    return updated


def check_profiles(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = args.output or config.get("outputs", {}).get("smoke_profile_checks", "")
    adapters = build_profile_adapters(args.adapters or config_adapters(config), config_runtime(config))
    rows = [adapter.smoke(timeout_s=args.timeout).to_row() for adapter in adapters]
    print(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True))
    if output:
        write_csv(Path(output), rows)
    return 0 if all(row["ok"] for row in rows) else 1


def build_profile_table(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = args.output or config.get("outputs", {}).get("smoke_profiles", "out/profile_tables/smoke_profiles.csv")
    if args.import_measurements:
        measurements = read_measurements(Path(args.import_measurements))
        write_csv(Path(output), [measurement.to_row() for measurement in measurements])
        summary = MetricCollector().summarize_profiles(measurements)
        print(json.dumps(json_ready({
            "output": output,
            "rows": len(measurements),
            "imported_from": args.import_measurements,
            "summary": summary,
        }), indent=2))
        return 0 if all(measurement.ok and measurement.measured for measurement in measurements) else 1
    profiles = config_profiles(config)
    runtime = config_runtime(config)
    adapters = build_profile_adapters(args.adapters or config_adapters(config), runtime)
    requests, fallback_requests = load_requests(config)
    requests = expand_repeated_requests(requests, int(runtime.get("repeat", 1)))
    measurements: list[ProfileMeasurement] = []
    for adapter in adapters:
        for spec in adapter.profiles():
            for request in requests:
                if spec.name not in profiles:
                    continue
                measurements.append(annotate_measurement(adapter.profile(request, spec.name, dry_run=args.dry_run), request, fallback_requests))

    measurements = with_quality(measurements, exact_profiles(profiles))
    write_csv(Path(output), [measurement.to_row() for measurement in measurements])
    summary = MetricCollector().summarize_profiles(measurements)
    print(json.dumps(json_ready({
        "output": output,
        "rows": len(measurements),
        "builtin_request_fallback": fallback_requests,
        "summary": summary,
    }), indent=2))
    return 0 if all(measurement.ok for measurement in measurements) else 1


def run_policies(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    output = args.output or config.get("outputs", {}).get("smoke_policy", "out/policy_tables/smoke_policy.csv")
    profiles = args.profiles or config_profiles(config)
    policy_names = args.policies or config_policies(config)
    epsilon = float(args.epsilon if args.epsilon is not None else config["pilot"]["epsilons"][0])
    delta = float(args.delta if args.delta is not None else config["pilot"]["deltas"][0])
    measurements = read_measurements(Path(args.measurements))
    try:
        backend = MeasuredReplayBackend(measurements, allow_dry_run=args.allow_dry_run_replay)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    calibration_measurements, evaluation_measurements = split_measurements(measurements)
    requests = requests_from_measurements(evaluation_measurements)
    exact = exact_profiles(profiles)
    policies = build_policies(policy_names, calibration_measurements, measurements, profiles, epsilon, delta, exact)
    records: list[PolicyRunRecord] = []

    for policy in policies:
        for request in requests:
            action = policy.decide(request, CacheState(), DeviceState())
            try:
                measurement = backend.run([request], [action.profile])[0]
                records.append(
                    PolicyRunRecord(
                        policy=policy.name,
                        request_id=request.request_id,
                        action_profile=action.profile,
                        ok=measurement.ok,
                        measured=measurement.measured,
                        placeholder=policy.placeholder,
                        reason=action.reason,
                        error=measurement.error,
                        latency_ms=measurement.latency_ms,
                        ttft_ms=measurement.ttft_ms,
                        peak_memory_mib=measurement.peak_memory_mib,
                        resident_memory_mib=measurement.resident_memory_mib,
                        quality_loss=measurement.quality_loss,
                        exact=action.profile in exact,
                        oracle=bool(getattr(policy, "oracle", False)),
                    )
                )
            except Exception as exc:
                records.append(
                    PolicyRunRecord(
                        policy=policy.name,
                        request_id=request.request_id,
                        action_profile=action.profile,
                        ok=False,
                        measured=False,
                        placeholder=policy.placeholder,
                        reason=action.reason,
                        error=str(exc),
                        exact=action.profile in exact,
                        oracle=bool(getattr(policy, "oracle", False)),
                    )
                )

    write_csv(Path(output), [record.to_row() for record in records])
    summary = MetricCollector().summarize_policy_runs(records, epsilon=epsilon, exact_profiles=exact)
    print(
        json.dumps(
            json_ready({
                "output": output,
                "rows": len(records),
                "epsilon": epsilon,
                "delta": delta,
                "summary": summary,
            }),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(record.ok for record in records) else 1


def reproduce_profiles(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    adapters = args.adapters or config_adapters(config)
    outputs = config.get("outputs", {})
    smoke_args = argparse.Namespace(
        config=args.config,
        adapters=adapters,
        timeout=args.timeout,
        output=args.smoke_output or outputs.get("smoke_profile_checks", "out/profile_tables/profile_smoke.csv"),
    )
    table_args = argparse.Namespace(
        config=args.config,
        adapters=adapters,
        output=args.profile_output or outputs.get("smoke_profiles", "out/profile_tables/smoke_profiles.csv"),
        dry_run=args.dry_run,
        import_measurements="",
    )
    smoke_code = check_profiles(smoke_args)
    table_code = build_profile_table(table_args)
    return 0 if smoke_code == 0 and table_code == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TailGuardKV 统一实验入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-profiles", help="检查 full/KIVI/H2O adapter 连通性。")
    check.add_argument("--config", default="configs/pilot.yaml")
    check.add_argument("--adapters", nargs="+")
    check.add_argument("--timeout", type=int, default=120)
    check.add_argument("--output", default="")
    check.set_defaults(func=check_profiles)

    table = subparsers.add_parser("build-profile-table", help="生成 request x profile 统一表。")
    table.add_argument("--config", default="configs/pilot.yaml")
    table.add_argument("--adapters", nargs="+")
    table.add_argument("--output")
    table.add_argument("--import-measurements", default="")
    table.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    table.set_defaults(func=build_profile_table)

    replay = subparsers.add_parser("run-policies", help="用同一 measured-replay backend 运行策略。")
    replay.add_argument("--config", default="configs/pilot.yaml")
    replay.add_argument("--measurements", default="out/profile_tables/smoke_profiles.csv")
    replay.add_argument("--output")
    replay.add_argument("--profiles", nargs="+")
    replay.add_argument("--policies", nargs="+")
    replay.add_argument("--epsilon", type=float)
    replay.add_argument("--delta", type=float)
    replay.add_argument("--allow-dry-run-replay", action="store_true")
    replay.set_defaults(func=run_policies)

    reproduce = subparsers.add_parser("reproduce-profiles", help="复现 full/KIVI/H2O adapter smoke 和统一 profile 表。")
    reproduce.add_argument("--config", default="configs/pilot.yaml")
    reproduce.add_argument("--adapters", nargs="+")
    reproduce.add_argument("--timeout", type=int, default=120)
    reproduce.add_argument("--smoke-output")
    reproduce.add_argument("--profile-output")
    reproduce.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    reproduce.set_defaults(func=reproduce_profiles)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
