from __future__ import annotations

from profiles.base import ProfileAdapter, dry_profile_measurement, run_conda_probe
from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult


class FullKVAdapter(ProfileAdapter):
    name = "full"
    env = "tailguardkv-base"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("full_gpu", self.name, self.env, lossy=False, exact=True),
            ProfileSpec("full_cpu", self.name, self.env, lossy=False, exact=True),
            ProfileSpec("recompute", self.name, self.env, lossy=False, exact=True),
        )

    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ok, versions, error = run_conda_probe(
            self.env,
            ("torch", "transformers", "numpy", "pandas", "pyarrow"),
            timeout_s=timeout_s,
        )
        return SmokeResult(
            adapter=self.name,
            env=self.env,
            ok=ok,
            profiles=self.profile_names(),
            detail="full/exact profile 先通过 base 环境驱动，后续接 transformers 或 vLLM 实测。",
            error=error,
            versions=versions,
        )

    def profile(self, request: Request, profile_name: str, dry_run: bool = True) -> ProfileMeasurement:
        spec = self.get_profile(profile_name)
        if not dry_run:
            return ProfileMeasurement(
                request_id=request.request_id,
                profile=spec.name,
                adapter=self.name,
                ok=False,
                measured=False,
                error="真实 full-KV 生成尚未接入；下一步应接 TransformersBackend 或 VLLMBackend。",
            )
        scale = max(request.prompt_chars, 1)
        return dry_profile_measurement(self.name, request, spec, scale * 0.08, scale * 2.0 / 1024.0)
