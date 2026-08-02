from __future__ import annotations

import json
import os
import subprocess
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult


class ProfileAdapter(ABC):
    """统一封装 full、量化、剪枝等 profile 的最小接口。"""

    name: str
    env: str

    def __init__(self, runtime_config: dict[str, object] | None = None) -> None:
        self.runtime_config = runtime_config or {}

    @abstractmethod
    def profiles(self) -> tuple[ProfileSpec, ...]:
        ...

    @abstractmethod
    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ...

    @abstractmethod
    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        ...

    def profile_many(self, requests: Sequence[Request], profile_name: str, dry_run: bool = True) -> list[ProfileMeasurement]:
        return [self.profile(request, profile_name, dry_run=dry_run) for request in requests]

    def profile_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.profiles())

    def get_profile(self, profile_name: str) -> ProfileSpec:
        for spec in self.profiles():
            if spec.name == profile_name:
                return spec
        raise KeyError(f"{self.name} 没有 profile: {profile_name}")


def run_conda_probe(
    env_name: str,
    modules: Sequence[str],
    timeout_s: int = 120,
    pythonpath: Sequence[str] = (),
) -> tuple[bool, dict[str, str], str | None]:
    code = """
import importlib
import json
import sys

payload = {"python": sys.version.split()[0], "modules": {}}
for name in __MODULES__:
    try:
        module = importlib.import_module(name)
        payload["modules"][name] = {
            "ok": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
    except Exception as exc:
        payload["modules"][name] = {
            "ok": False,
            "error": type(exc).__name__ + ": " + str(exc)[:300],
        }
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
""".replace("__MODULES__", repr(list(modules)))

    env = os.environ.copy()
    if pythonpath:
        paths = [os.path.abspath(path) for path in pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)

    proc = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        return False, {}, (proc.stderr or proc.stdout).strip()[-1000:]
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return False, {}, f"无法解析探测输出: {exc}; output={proc.stdout[-1000:]}"

    versions = {"python": str(payload["python"])}
    ok = True
    for module_name, info in payload["modules"].items():
        versions[module_name] = str(info.get("version") if info["ok"] else info.get("error"))
        ok = ok and bool(info["ok"])
    return ok, versions, None if ok else "至少一个模块导入失败"


def dry_profile_measurement(
    adapter: str,
    request: Request,
    spec: ProfileSpec,
    latency_ms: float,
    peak_memory_mib: float,
) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=True,
        measured=False,
        output_text=request.prompt,
        latency_ms=latency_ms,
        ttft_ms=None,
        peak_memory_mib=peak_memory_mib,
        kv_cache_memory_mib=peak_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=None,
        extra={
            "family": spec.family,
            "ttft_semantics": "unavailable",
            "note": "dry_run仅验证统一表结构，尚未执行真实模型和profile kernel",
        },
    )


def transformers_profile_many_measurements(
    adapter: str,
    env_name: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    *,
    timeout_s: int | None = None,
    pythonpath: Sequence[str] = (),
    extra: dict[str, object] | None = None,
) -> list[ProfileMeasurement]:
    request_list = list(requests)
    if not request_list:
        return []
    model_name = _runtime_model_name(runtime_config)
    if not model_name:
        return _missing_model_measurements(
            adapter,
            request_list,
            spec,
            "未配置 model.pilot_model，无法执行真实 transformers profile。",
            {"backend": "transformers", **(extra or {})},
        )
    payload = {
        "requests": [_transformers_payload(request, spec, runtime_config, model_name) for request in request_list],
    }
    proc, result, error = _run_runtime_batch(
        env_name,
        "profiles.transformers_runtime",
        "TRANSFORMERS_PROFILE_PAYLOAD",
        payload,
        timeout_s=timeout_s or _batch_timeout_s(runtime_config, len(request_list)),
        pythonpath=pythonpath,
    )
    if error is not None:
        return _worker_failure_measurements(
            adapter,
            request_list,
            spec,
            error,
            {"backend": "transformers", "model": model_name, **(extra or {})},
        )
    return _measurements_from_batch_result(
        adapter,
        request_list,
        spec,
        proc,
        result,
        default_extra={"backend": "transformers", "model": model_name, **(extra or {})},
    )


def qwen2_kv_profile_many_measurements(
    adapter: str,
    env_name: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    *,
    timeout_s: int | None = None,
    pythonpath: Sequence[str] = (),
    extra: dict[str, object] | None = None,
) -> list[ProfileMeasurement]:
    request_list = list(requests)
    if not request_list:
        return []
    model_name = _runtime_model_name(runtime_config)
    if not model_name:
        return _missing_model_measurements(
            adapter,
            request_list,
            spec,
            "未配置 model.pilot_model，无法执行 Qwen2 KV runtime。",
            {"backend": "qwen2_kv_runtime", "env": env_name, **(extra or {})},
        )
    payload = {
        "requests": [_qwen2_payload(request, spec, runtime_config, model_name) for request in request_list],
    }
    proc, result, error = _run_runtime_batch(
        env_name,
        "profiles.qwen2_kv_runtime",
        "QWEN2_KV_PAYLOAD",
        payload,
        timeout_s=timeout_s or _batch_timeout_s(runtime_config, len(request_list)),
        pythonpath=pythonpath,
    )
    if error is not None:
        return _worker_failure_measurements(
            adapter,
            request_list,
            spec,
            error,
            {"backend": "qwen2_kv_runtime", "env": env_name, "model": model_name, **(extra or {})},
        )
    return _measurements_from_batch_result(
        adapter,
        request_list,
        spec,
        proc,
        result,
        default_extra={"backend": "qwen2_kv_runtime", "env": env_name, "model": model_name, **(extra or {})},
    )


def transformers_profile_measurement(
    adapter: str,
    env_name: str,
    request: Request,
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    timeout_s: int | None = None,
    pythonpath: Sequence[str] = (),
    extra: dict[str, object] | None = None,
) -> ProfileMeasurement:
    model_name = _runtime_model_name(runtime_config)
    if not model_name:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error="未配置 model.pilot_model，无法执行真实 transformers profile。",
            extra={"backend": "transformers", "unsupported": "true", **(extra or {})},
        )

    payload = _transformers_payload(request, spec, runtime_config, model_name)
    code = _transformers_profile_code(payload)
    env = os.environ.copy()
    if pythonpath:
        paths = [os.path.abspath(path) for path in pythonpath]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)

    command = ["conda", "run", "-n", env_name, "python", "-c", code]
    try:
        proc = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s or int(runtime_config.get("timeout_s", 180)),
            check=False,
        )
    except Exception as exc:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=f"真实 transformers profile 启动失败: {type(exc).__name__}: {exc}",
            extra={"backend": "transformers", "unsupported": "true", **(extra or {})},
        )

    if proc.returncode != 0:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=(proc.stderr or proc.stdout).strip()[-1200:],
            extra={"backend": "transformers", "unsupported": "true", **(extra or {})},
        )
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=f"无法解析 transformers profile 输出: {exc}; output={proc.stdout[-1200:]}",
            extra={"backend": "transformers", "unsupported": "true", **(extra or {})},
        )
    ok = bool(result.get("ok"))
    result_extra = {
        "backend": "transformers",
        "model": model_name,
        **(extra or {}),
    }
    if not ok:
        result_extra["unsupported"] = "true"
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=ok,
        measured=ok,
        output_text=str(result.get("output_text") or ""),
        error=None if result.get("ok") else str(result.get("error") or ""),
        latency_ms=_optional_float(result.get("latency_ms")),
        ttft_ms=_optional_float(result.get("ttft_ms")),
        peak_memory_mib=_optional_float(result.get("peak_memory_mib")),
        kv_cache_memory_mib=_optional_float(result.get("kv_cache_memory_mib")),
        resident_memory_mib=_optional_float(result.get("resident_memory_mib")),
        extra=result_extra,
    )


def qwen2_kv_profile_measurement(
    adapter: str,
    env_name: str,
    request: Request,
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    timeout_s: int | None = None,
    extra: dict[str, object] | None = None,
) -> ProfileMeasurement:
    model_name = _runtime_model_name(runtime_config)
    if not model_name:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error="未配置 model.pilot_model，无法执行 Qwen2 KV runtime。",
            extra={"backend": "qwen2_kv_runtime", "env": env_name, "unsupported": "true", **(extra or {})},
        )

    payload = _qwen2_payload(request, spec, runtime_config, model_name)
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    python_paths = [repo_root]
    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        python_paths.append(current_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["QWEN2_KV_PAYLOAD"] = json.dumps(payload, ensure_ascii=False)

    command = ["conda", "run", "-n", env_name, "python", "-m", "profiles.qwen2_kv_runtime"]
    try:
        proc = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s or int(runtime_config.get("timeout_s", 180)),
            check=False,
        )
    except Exception as exc:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=f"Qwen2 KV runtime 启动失败: {type(exc).__name__}: {exc}",
            extra={"backend": "qwen2_kv_runtime", "env": env_name, "unsupported": "true", **(extra or {})},
        )

    result: dict[str, object]
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=f"无法解析 Qwen2 KV runtime 输出: {exc}; stderr={(proc.stderr or '')[-800:]}; stdout={proc.stdout[-800:]}",
            extra={"backend": "qwen2_kv_runtime", "env": env_name, "unsupported": "true", **(extra or {})},
        )

    ok = bool(result.get("ok")) and proc.returncode == 0
    measured = ok and bool(result.get("measured"))
    result_extra = {
        "backend": str(result.get("backend") or "qwen2_kv_runtime"),
        "env": env_name,
        "model": model_name,
        **(extra or {}),
    }
    for key, value in result.items():
        if key in {
            "ok",
            "measured",
            "output_text",
            "error",
            "latency_ms",
            "ttft_ms",
            "peak_memory_mib",
            "kv_cache_memory_mib",
            "resident_memory_mib",
        }:
            continue
        result_extra[key] = value
    if not measured:
        result_extra["unsupported"] = "true"
        if proc.returncode != 0:
            result_extra["returncode"] = proc.returncode
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=ok,
        measured=measured,
        output_text=str(result.get("output_text") or ""),
        error=None if ok else str(result.get("error") or (proc.stderr or proc.stdout).strip()[-1200:]),
        latency_ms=_optional_float(result.get("latency_ms")),
        ttft_ms=_optional_float(result.get("ttft_ms")),
        peak_memory_mib=_optional_float(result.get("peak_memory_mib")),
        kv_cache_memory_mib=_optional_float(result.get("kv_cache_memory_mib")),
        resident_memory_mib=_optional_float(result.get("resident_memory_mib")),
        extra=result_extra,
    )


def _transformers_payload(
    request: Request,
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    model_name: str,
) -> dict[str, object]:
    return {
        "profile": spec.name,
        "model_name": model_name,
        "prompt": request.prompt,
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 16)),
        "cache_dir": runtime_config.get("model_cache_dir"),
        "local_files_only": bool(runtime_config.get("local_files_only", True)),
        "device_mode": spec.metadata.get("device_mode", "auto"),
        "use_cache": bool(spec.metadata.get("use_cache", True)),
    }


def _runtime_model_name(runtime_config: dict[str, object]) -> str:
    return str(runtime_config.get("pilot_model") or "")


def _batch_timeout_s(runtime_config: dict[str, object], request_count: int) -> int:
    per_request = int(runtime_config.get("timeout_s", 180))
    return max(per_request, per_request * max(1, request_count))


def _qwen2_payload(
    request: Request,
    spec: ProfileSpec,
    runtime_config: dict[str, object],
    model_name: str,
) -> dict[str, object]:
    return {
        "profile": spec.name,
        "model_name": model_name,
        "prompt": request.prompt,
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 16)),
        "cache_dir": runtime_config.get("model_cache_dir"),
        "local_files_only": bool(runtime_config.get("local_files_only", True)),
        "bits": spec.metadata.get("bits"),
        "kivi_group_size": int(spec.metadata.get("kivi_group_size", runtime_config.get("kivi_group_size", 32))),
        "kivi_residual_length": int(spec.metadata.get("kivi_residual_length", runtime_config.get("kivi_residual_length", 32))),
        "h2o_heavy_ratio": float(spec.metadata.get("h2o_heavy_ratio", runtime_config.get("h2o_heavy_ratio", 0.1))),
        "h2o_recent_ratio": float(spec.metadata.get("h2o_recent_ratio", runtime_config.get("h2o_recent_ratio", 0.1))),
    }


def _run_runtime_batch(
    env_name: str,
    module_name: str,
    payload_env_key: str,
    payload: dict[str, object],
    *,
    timeout_s: int,
    pythonpath: Sequence[str] = (),
) -> tuple[subprocess.CompletedProcess[str] | None, dict[str, object] | None, str | None]:
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[1])
    path_parts = [repo_root, *[os.path.abspath(path) for path in pythonpath]]
    if env.get("PYTHONPATH"):
        path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env[payload_env_key] = json.dumps(payload, ensure_ascii=False)
    command = ["conda", "run", "-n", env_name, "python", "-m", module_name]
    try:
        proc = subprocess.run(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, None, _format_timeout_error(module_name, timeout_s, exc)
    except Exception as exc:
        return None, None, f"{module_name} 启动失败: {type(exc).__name__}: {exc}"
    output = proc.stdout.strip()
    if not output:
        if proc.returncode == 0:
            return proc, None, f"{module_name} 未输出 JSON 结果"
        return proc, None, (proc.stderr or proc.stdout).strip()[-1200:]
    try:
        return proc, json.loads(output.splitlines()[-1]), None
    except Exception as exc:
        return proc, None, f"无法解析 {module_name} 输出: {exc}; stderr={(proc.stderr or '')[-800:]}; stdout={proc.stdout[-800:]}"


def _measurements_from_batch_result(
    adapter: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    proc: subprocess.CompletedProcess[str] | None,
    result: dict[str, object] | None,
    *,
    default_extra: dict[str, object],
) -> list[ProfileMeasurement]:
    if proc is None or result is None:
        return _worker_failure_measurements(adapter, requests, spec, "worker did not return a JSON result", default_extra)
    items = result.get("results")
    if items is None and len(requests) == 1:
        items = [result]
    if not isinstance(items, list) or len(items) != len(requests):
        detail = (proc.stderr or proc.stdout).strip()[-1200:] if proc.returncode != 0 else ""
        error = f"worker result count mismatch: expected {len(requests)}, got {0 if not isinstance(items, list) else len(items)}"
        if detail:
            error = f"{error}; {detail}"
        return _worker_failure_measurements(adapter, requests, spec, error, default_extra)
    worker = result.get("worker") if isinstance(result, dict) else None
    worker_mode = str(worker.get("mode") or "") if isinstance(worker, dict) else ""
    rows = []
    for request, item in zip(requests, items, strict=True):
        if not isinstance(item, dict):
            rows.append(
                ProfileMeasurement(
                    request_id=request.request_id,
                    profile=spec.name,
                    adapter=adapter,
                    ok=False,
                    measured=False,
                    error=f"worker returned invalid item: {item!r}",
                    extra={**default_extra, "worker_mode": worker_mode, "unsupported": "true"},
                )
            )
            continue
        row = _measurement_from_result(
            adapter,
            request,
            spec,
            item,
            default_extra=default_extra,
            worker_mode=worker_mode,
        )
        if proc.returncode != 0 and row.ok:
            row = ProfileMeasurement(
                request_id=row.request_id,
                profile=row.profile,
                adapter=row.adapter,
                ok=False,
                measured=False,
                output_text=row.output_text,
                error=(proc.stderr or proc.stdout).strip()[-1200:] or row.error or "worker returned non-zero status",
                latency_ms=row.latency_ms,
                ttft_ms=row.ttft_ms,
                peak_memory_mib=row.peak_memory_mib,
                kv_cache_memory_mib=row.kv_cache_memory_mib,
                resident_memory_mib=row.resident_memory_mib,
                quality_score=row.quality_score,
                quality_loss=row.quality_loss,
                extra={**row.extra, "returncode": proc.returncode, "unsupported": "true"},
            )
        rows.append(row)
    return rows


def _measurement_from_result(
    adapter: str,
    request: Request,
    spec: ProfileSpec,
    item: dict[str, object],
    *,
    default_extra: dict[str, object],
    worker_mode: str,
) -> ProfileMeasurement:
    ok = bool(item.get("ok"))
    measured = ok and bool(item.get("measured", ok))
    result_extra = dict(default_extra)
    standard_keys = {
        "ok",
        "measured",
        "output_text",
        "error",
        "latency_ms",
        "ttft_ms",
        "peak_memory_mib",
        "kv_cache_memory_mib",
        "resident_memory_mib",
    }
    for key, value in item.items():
        if key in standard_keys:
            continue
        result_extra[key] = value
    if worker_mode and "worker_mode" not in result_extra:
        result_extra["worker_mode"] = worker_mode
    if not measured:
        result_extra.setdefault("unsupported", "true")
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=ok,
        measured=measured,
        output_text=str(item.get("output_text") or ""),
        error=None if ok else str(item.get("error") or ""),
        latency_ms=_optional_float(item.get("latency_ms")),
        ttft_ms=_optional_float(item.get("ttft_ms")),
        peak_memory_mib=_optional_float(item.get("peak_memory_mib")),
        kv_cache_memory_mib=_optional_float(item.get("kv_cache_memory_mib")),
        resident_memory_mib=_optional_float(item.get("resident_memory_mib")),
        extra=result_extra,
    )


def _missing_model_measurements(
    adapter: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    error: str,
    extra: dict[str, object],
) -> list[ProfileMeasurement]:
    return [
        ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=error,
            extra={**extra, "unsupported": "true"},
        )
        for request in requests
    ]


def _worker_failure_measurements(
    adapter: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    error: str,
    extra: dict[str, object],
) -> list[ProfileMeasurement]:
    failure_extra = dict(extra)
    failure_extra.setdefault("unsupported", "true")
    if "TimeoutExpired" in error:
        failure_extra.setdefault("error_type", "timeout")
        failure_extra.setdefault("failure_stage", "worker_startup")
    return [
        ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error=error,
            extra=dict(failure_extra),
        )
        for request in requests
    ]


def _format_timeout_error(module_name: str, timeout_s: int, exc: subprocess.TimeoutExpired) -> str:
    stderr_tail = _trim_timeout_output(exc.stderr)
    stdout_tail = _trim_timeout_output(exc.output)
    detail = f"{module_name} 启动超时: TimeoutExpired after {timeout_s}s"
    if stderr_tail:
        detail = f"{detail}; stderr={stderr_tail}"
    if stdout_tail:
        detail = f"{detail}; stdout={stdout_tail}"
    return detail


def _trim_timeout_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text.strip()[-800:]


def _transformers_profile_code(payload: dict[str, object]) -> str:
    return textwrap.dedent(
        f"""
        import json
        import resource
        import time

        payload = {payload!r}
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = payload["model_name"]
            cache_dir = payload.get("cache_dir") or None
            local_files_only = bool(payload.get("local_files_only"))
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=True,
            )
            device_mode = payload.get("device_mode", "auto")
            has_cuda = torch.cuda.is_available()
            kwargs = {{
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "trust_remote_code": True,
            }}
            if device_mode == "cpu" or not has_cuda:
                kwargs["device_map"] = "cpu"
            else:
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            inputs = tokenizer(payload["prompt"], return_tensors="pt")
            if device_mode != "cpu" and has_cuda:
                inputs = {{key: value.to(model.device) for key, value in inputs.items()}}
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(payload["max_new_tokens"]),
                    do_sample=False,
                    use_cache=bool(payload["use_cache"]),
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            if device_mode != "cpu" and has_cuda:
                torch.cuda.synchronize()
                peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            latency_ms = (time.perf_counter() - start) * 1000
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            output_ids = generated.sequences if hasattr(generated, "sequences") else generated
            generated_ids = output_ids[0][prompt_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if not output_text:
                output_text = " ".join(str(int(token)) for token in generated_ids)
            print(json.dumps({{
                "ok": True,
                "output_text": output_text,
                "latency_ms": latency_ms,
                "ttft_ms": None,
                "peak_memory_mib": peak_memory_mib,
                "kv_cache_memory_mib": 0.0,
                "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "ttft_semantics": "unavailable",
            }}, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({{
                "ok": False,
                "error": type(exc).__name__ + ": " + str(exc)[:1000],
            }}, ensure_ascii=False))
        """
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
