from __future__ import annotations

from profiles.base import ProfileAdapter, dry_profile_measurement, run_conda_probe, transformers_profile_measurement
from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult


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
            return transformers_profile_measurement(
                self.name,
                self.env,
                request,
                spec,
                self.runtime_config,
                pythonpath=("third_party/KIVI",),
                extra={
                    "family": spec.family,
                    "bits": spec.metadata.get("bits", ""),
                    "profile_note": "KIVI-compatible transformers smoke; not claimed as final KIVI kernel result",
                },
            )
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
