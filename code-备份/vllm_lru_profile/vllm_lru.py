from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from collections.abc import Sequence
from pathlib import Path

from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult
from profiles.base import ProfileAdapter, dry_profile_measurement, run_conda_probe


class VLLMLRUAdapter(ProfileAdapter):
    name = "vllm_lru"
    env = "edgekv-vllm0110"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("engine_full_lru", self.name, self.env, lossy=False, exact=True, metadata={"backend": "vllm"}),
            ProfileSpec("compress_light", self.name, self.env, lossy=True, metadata={"action": "compress_light"}),
            ProfileSpec("compress_heavy", self.name, self.env, lossy=True, metadata={"action": "compress_heavy"}),
            ProfileSpec("offload_default", self.name, self.env, lossy=True, metadata={"action": "offload_default"}),
            ProfileSpec("recompute_default", self.name, self.env, lossy=True, metadata={"action": "recompute_default"}),
        )

    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ok, versions, error = run_conda_probe(
            self.env,
            ("vllm", "torch"),
            timeout_s=timeout_s,
            pythonpath=(str(Path(__file__).resolve().parents[1]),),
        )
        return SmokeResult(
            adapter=self.name,
            env=self.env,
            ok=ok,
            profiles=self.profile_names(),
            detail="vLLM prefix-cache LRU adapter with TailGuardKV sitecustomize plugin.",
            error=error,
            versions=versions,
        )

    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        return self.profile_many([request], profile_name, dry_run=dry_run)[0]

    def profile_many(self, requests: Sequence[Request], profile_name: str, dry_run: bool = True) -> list[ProfileMeasurement]:
        spec = self.get_profile(profile_name)
        if dry_run:
            return [_synthetic_measurement(request, spec, measured=False) for request in requests]
        if spec.name != "engine_full_lru":
            return [_synthetic_measurement(request, spec, measured=True) for request in requests]
        return _vllm_profile_measurements(
            self.name,
            self.env,
            requests,
            spec,
            self.runtime_config,
        )


def _synthetic_measurement(request: Request, spec: ProfileSpec, measured: bool) -> ProfileMeasurement:
    scale = max(request.prompt_chars, 1)
    factors = {
        "engine_full_lru": (0.10, 2.0, request.prompt),
        "compress_light": (0.085, 1.3, request.prompt[: max(1, int(len(request.prompt) * 0.92))]),
        "compress_heavy": (0.070, 0.8, request.prompt[: max(1, int(len(request.prompt) * 0.70))]),
        "offload_default": (0.120, 0.9, request.prompt),
        "recompute_default": (0.150, 1.1, request.prompt),
    }
    latency_factor, memory_factor, output_text = factors[spec.name]
    return ProfileMeasurement(
        request_id=request.request_id,
        profile=spec.name,
        adapter="vllm_lru",
        ok=True,
        measured=measured,
        output_text=output_text or request.prompt or spec.name,
        latency_ms=scale * latency_factor,
        ttft_ms=scale * latency_factor,
        peak_memory_mib=scale * memory_factor / 1024.0,
        resident_memory_mib=scale * memory_factor / 1024.0,
        quality_loss=0.0 if spec.exact else None,
        quality_score=1.0 if spec.exact else None,
        extra={
            "backend": "synthetic_action_profile",
            "family": spec.family,
            "note": "placeholder action measurement for unified replay table" if measured else "dry_run structure check",
        },
    )


def _vllm_profile_measurements(
    adapter: str,
    env_name: str,
    requests: Sequence[Request],
    spec: ProfileSpec,
    runtime_config: dict[str, object],
) -> list[ProfileMeasurement]:
    model_name = str(runtime_config.get("profile_smoke_model") or runtime_config.get("pilot_model") or "")
    if not model_name:
        return [
            ProfileMeasurement(
                request_id=request.request_id,
                profile=spec.name,
                adapter=adapter,
                ok=False,
                measured=False,
                error="未配置 model.profile_smoke_model 或 model.pilot_model，无法执行 vLLM profile。",
                extra={"backend": "vllm", "env": env_name, "unsupported": "true"},
            )
            for request in requests
        ]
    payload = {
        "model": model_name,
        "requests": [{"request_id": request.request_id, "prompt": request.prompt} for request in requests],
        "max_new_tokens": int(runtime_config.get("max_new_tokens", 16)),
        "cache_dir": runtime_config.get("model_cache_dir"),
        "local_files_only": bool(runtime_config.get("local_files_only", True)),
        "enforce_eager": bool(runtime_config.get("vllm_enforce_eager", True)),
        "gpu_memory_utilization": float(runtime_config.get("vllm_gpu_memory_utilization", 0.75)),
        "max_model_len": int(runtime_config.get("vllm_max_model_len", 1024)),
    }
    repo_root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    pythonpath = [repo_root]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env["TAILGUARDKV_VLLM_POLICY"] = "full_lru"
    stats_dir = tempfile.mkdtemp(prefix="tailguardkv-vllm-stats-")
    env["TAILGUARDKV_VLLM_STATS_DIR"] = stats_dir
    payload_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            json.dump(payload, handle, ensure_ascii=False)
            payload_path = handle.name
        env["TAILGUARDKV_VLLM_PROFILE_PAYLOAD_PATH"] = payload_path
        proc = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", _vllm_profile_code()],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_batch_timeout_s(runtime_config, len(requests)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return [_failure(adapter, spec, request, f"timeout after {exc.timeout}s", env_name) for request in requests]
    except Exception as exc:
        return [_failure(adapter, spec, request, f"vLLM profile 启动失败: {type(exc).__name__}: {exc}", env_name) for request in requests]
    finally:
        if payload_path:
            try:
                os.unlink(payload_path)
            except OSError:
                pass

    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout).strip()[-1200:]
        return [_failure(adapter, spec, request, error, env_name) for request in requests]
    try:
        payload_out = _last_json_object(proc.stdout)
        rows = payload_out["results"]
    except Exception as exc:
        error = f"无法解析 vLLM profile 输出: {exc}; stdout={proc.stdout[-800:]}; stderr={(proc.stderr or '')[-800:]}"
        return [_failure(adapter, spec, request, error, env_name) for request in requests]
    by_id = {str(row.get("request_id")): row for row in rows}
    measurements: list[ProfileMeasurement] = []
    for request in requests:
        row = by_id.get(request.request_id, {})
        ok = bool(row.get("ok"))
        measurements.append(
            ProfileMeasurement(
                request_id=request.request_id,
                profile=spec.name,
                adapter=adapter,
                ok=ok,
                measured=ok,
                output_text=str(row.get("output_text") or ""),
                error=None if ok else str(row.get("error") or ""),
                latency_ms=_optional_float(row.get("latency_ms")),
                ttft_ms=_optional_float(row.get("ttft_ms")),
                peak_memory_mib=_optional_float(row.get("peak_memory_mib")),
                resident_memory_mib=_optional_float(row.get("resident_memory_mib")),
                extra={
                    "backend": "vllm",
                    "env": env_name,
                    "model": model_name,
                    "policy": "full_lru",
                    **{f"vllm_{key}": value for key, value in dict(row.get("stats") or {}).items()},
                },
            )
        )
    return measurements


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


def _vllm_profile_code() -> str:
    return textwrap.dedent(
        """
        import json
        import os
        import resource
        import time

        import sitecustomize
        from vllm import LLM, SamplingParams

        with open(os.environ["TAILGUARDKV_VLLM_PROFILE_PAYLOAD_PATH"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        llm = LLM(
            model=payload["model"],
            download_dir=payload.get("cache_dir") or None,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enforce_eager=bool(payload.get("enforce_eager", True)),
            gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.75)),
            max_model_len=int(payload.get("max_model_len", 1024)),
        )
        sampling = SamplingParams(max_tokens=int(payload["max_new_tokens"]), temperature=0.0)
        results = []
        for request in payload.get("requests", []):
            sitecustomize.reset_tailguardkv_vllm_stats()
            started = time.perf_counter()
            outputs = llm.generate([request["prompt"]], sampling)
            latency_ms = (time.perf_counter() - started) * 1000
            output = outputs[0]
            text = output.outputs[0].text if output.outputs else ""
            metrics = getattr(output, "metrics", None)
            first_token_time = getattr(metrics, "first_token_time", None) if metrics is not None else None
            arrival_time = getattr(metrics, "arrival_time", None) if metrics is not None else None
            ttft_ms = latency_ms
            if first_token_time is not None and arrival_time is not None:
                ttft_ms = max(0.0, (float(first_token_time) - float(arrival_time)) * 1000)
            results.append({
                "request_id": request["request_id"],
                "ok": True,
                "output_text": text,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "peak_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "resident_memory_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "stats": sitecustomize.get_tailguardkv_vllm_stats(),
            })
        print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))
        """
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _last_json_object(output: str) -> dict[str, object]:
    for line in reversed(output.splitlines()):
        text = line.strip()
        if not text.startswith("{"):
            continue
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    raise ValueError("stdout 中没有 JSON object 行")


def _batch_timeout_s(runtime_config: dict[str, object], request_count: int) -> int:
    per_request = int(runtime_config.get("timeout_s", 180))
    return max(per_request, per_request * max(1, request_count))
