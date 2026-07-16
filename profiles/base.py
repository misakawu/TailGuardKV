from __future__ import annotations

import json
import os
import subprocess
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence

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
        ttft_ms=latency_ms,
        peak_memory_mib=peak_memory_mib,
        resident_memory_mib=peak_memory_mib,
        quality_loss=None,
        extra={
            "family": spec.family,
            "note": "dry_run仅验证统一表结构，尚未执行真实模型和profile kernel",
        },
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
    model_name = str(runtime_config.get("profile_smoke_model") or runtime_config.get("pilot_model") or "")
    if not model_name:
        return ProfileMeasurement(
            request_id=request.request_id,
            profile=spec.name,
            adapter=adapter,
            ok=False,
            measured=False,
            error="未配置 model.profile_smoke_model 或 model.pilot_model，无法执行真实 transformers profile。",
            extra={"backend": "transformers", "unsupported": "true", **(extra or {})},
        )

    payload = {
        "model_name": model_name,
        "prompt": request.prompt,
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 16)),
        "cache_dir": runtime_config.get("model_cache_dir"),
        "local_files_only": bool(runtime_config.get("local_files_only", True)),
        "device_mode": spec.metadata.get("device_mode", "auto"),
    }
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
        resident_memory_mib=_optional_float(result.get("resident_memory_mib")),
        extra=result_extra,
    )


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
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(payload["max_new_tokens"]),
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if device_mode != "cpu" and has_cuda:
                torch.cuda.synchronize()
                peak_memory_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_memory_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            latency_ms = (time.perf_counter() - start) * 1000
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            generated_ids = output_ids[0][prompt_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(json.dumps({{
                "ok": True,
                "output_text": output_text,
                "latency_ms": latency_ms,
                "ttft_ms": latency_ms,
                "peak_memory_mib": peak_memory_mib,
                "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
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
