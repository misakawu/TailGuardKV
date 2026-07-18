from __future__ import annotations

from dataclasses import replace

from profiles.base import ProfileAdapter, dry_profile_measurement, qwen2_kv_profile_measurement, run_conda_probe
from core_types import ProfileSpec, Request, SmokeResult


class KIVIAdapter(ProfileAdapter):
    name = "kivi"
    env = "edgekv-kivi"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("kivi_4bit", self.name, self.env, lossy=True, metadata={"bits": 4}),
            ProfileSpec("kivi_2bit", self.name, self.env, lossy=True, metadata={"bits": 2}),
        )

    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ok, versions, error = run_conda_probe(
            self.env,
            ("torch", "transformers", "models", "quant", "kivi_gemv"),
            timeout_s=timeout_s,
        )
        return SmokeResult(
            adapter=self.name,
            env=self.env,
            ok=ok,
            profiles=self.profile_names(),
            detail="KIVI 源码与 CUDA 扩展可导入；Qwen 真实适配需单独 smoke。",
            error=error,
            versions=versions,
        )

    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        spec = self.get_profile(profile_name)
        if not dry_run:
            row = qwen2_kv_profile_measurement(
                self.name,
                self.env,
                request,
                spec,
                self.runtime_config,
                extra={
                    "family": spec.family,
                    "bits": spec.metadata.get("bits", ""),
                },
            )
            if not row.ok:
                return replace(row, error=f"KIVI proof/runtime failed ({profile_name}): {row.error or ''}")
            return row
        bits = int(spec.metadata["bits"])
        scale = max(request.prompt_chars, 1)
        memory_factor = 0.5 if bits == 4 else 0.25
        latency_factor = 0.09 if bits == 4 else 0.095
        return dry_profile_measurement(
            self.name,
            request,
            spec,
            scale * latency_factor,
            scale * memory_factor / 1024.0,
        )
