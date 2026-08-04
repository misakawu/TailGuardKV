from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from run_util.core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult
from profiles.base import ProfileAdapter, dry_profile_measurement, run_conda_probe


class VLLMAdapter(ProfileAdapter):
    name = "vllm"
    env = "edgekv-vllm0110"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (ProfileSpec("engine_full_lru", self.name, self.env, lossy=False, exact=True, metadata={"backend": "vllm"}),)

    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ok, versions, error = run_conda_probe(
            self.env,
            ("vllm", "torch", "transformers"),
            timeout_s=timeout_s,
            pythonpath=(str(Path(__file__).resolve().parents[1]),),
        )
        return SmokeResult(
            adapter=self.name,
            env=self.env,
            ok=ok,
            profiles=self.profile_names(),
            detail="vLLM engine full-cache LRU profile.",
            error=error,
            versions=versions,
        )

    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        return self.profile_many([request], profile_name, dry_run=dry_run)[0]

    def profile_many(self, requests: Sequence[Request], profile_name: str, dry_run: bool = True) -> list[ProfileMeasurement]:
        spec = self.get_profile(profile_name)
        if dry_run:
            return [dry_profile_measurement(self.name, request, spec, max(request.prompt_chars, 1) * 0.1, 0.0) for request in requests]
        return _run_vllm_worker(self.name, self.env, list(requests), spec, self.runtime_config)


def _run_vllm_worker(
    adapter: str,
    env_name: str,
    requests: list[Request],
    spec: ProfileSpec,
    runtime_config: dict[str, object],
) -> list[ProfileMeasurement]:
    model_name = str(runtime_config.get("pilot_model") or "")
    if not model_name:
        return [_failure(adapter, spec, request, "未配置 model.pilot_model，无法执行 vLLM profile。", env_name) for request in requests]

    payload = {
        "model": model_name,
        "requests": [{"request_id": request.request_id, "prompt": request.prompt} for request in requests],
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 16)),
        "cache_dir": runtime_config.get("model_cache_dir"),
        "local_files_only": bool(runtime_config.get("local_files_only", True)),
        "enforce_eager": bool(runtime_config.get("vllm_enforce_eager", True)),
        "gpu_memory_utilization": float(runtime_config.get("vllm_gpu_memory_utilization", 0.75)),
        "max_model_len": int(runtime_config.get("vllm_max_model_len", 1024)),
        "tensor_parallel_size": int(runtime_config.get("vllm_tensor_parallel_size", 1)),
    }
    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    pythonpath = [repo_root]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    cuda_visible_devices = str(runtime_config.get("vllm_cuda_visible_devices") or "")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    payload_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            payload_path = handle.name
        env["TAILGUARDKV_VLLM_PROFILE_PAYLOAD_PATH"] = payload_path
        proc = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-m", "profiles.vllm_worker"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(int(runtime_config.get("timeout_s", 180)), int(runtime_config.get("timeout_s", 180)) * max(1, len(requests))),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return [_failure(adapter, spec, request, f"vLLM profile timeout after {exc.timeout}s", env_name) for request in requests]
    except Exception as exc:
        return [_failure(adapter, spec, request, f"vLLM profile 启动失败: {type(exc).__name__}: {exc}", env_name) for request in requests]
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except OSError:
                pass

    if proc.returncode != 0:
        error = _excerpt(proc.stderr or proc.stdout)
        return [_failure(adapter, spec, request, error or "vLLM worker returned non-zero", env_name) for request in requests]
    try:
        output = json.loads(proc.stdout.strip().splitlines()[-1])
        items = output["results"]
    except Exception as exc:
        error = f"无法解析 vLLM profile 输出: {exc}; stdout={proc.stdout[-800:]}; stderr={(proc.stderr or '')[-800:]}"
        return [_failure(adapter, spec, request, error, env_name) for request in requests]
    by_id = {str(item.get("request_id")): item for item in items if isinstance(item, dict)}
    return [_measurement(adapter, spec, request, by_id.get(request.request_id, {}), env_name, model_name) for request in requests]


def _measurement(adapter: str, spec: ProfileSpec, request: Request, item: dict[str, Any], env_name: str, model_name: str) -> ProfileMeasurement:
    ok = bool(item.get("ok"))
    extra = {"backend": "vllm", "env": env_name, "model": model_name}
    if item.get("ttft_semantics"):
        extra["ttft_semantics"] = item["ttft_semantics"]
    if not ok:
        extra["unsupported"] = "true"
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=ok,
        measured=ok,
        output_text=str(item.get("output_text") or ""),
        error=None if ok else str(item.get("error") or "vLLM worker did not return this request"),
        latency_ms=_optional_float(item.get("latency_ms")),
        ttft_ms=_optional_float(item.get("ttft_ms")),
        peak_memory_mib=_optional_float(item.get("peak_memory_mib")),
        kv_cache_memory_mib=_optional_float(item.get("kv_cache_memory_mib")),
        resident_memory_mib=_optional_float(item.get("resident_memory_mib")),
        extra=extra,
    )


def _failure(adapter: str, spec: ProfileSpec, request: Request, error: str, env_name: str) -> ProfileMeasurement:
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter=adapter,
        ok=False,
        measured=False,
        error=error,
        extra={"backend": "vllm", "env": env_name, "unsupported": "true"},
    )


def _excerpt(text: str, limit: int = 12000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    half = limit // 2
    return f"{cleaned[:half]}\n...\n{cleaned[-half:]}"


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
