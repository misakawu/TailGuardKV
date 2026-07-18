from __future__ import annotations

from dataclasses import replace

from profiles.base import ProfileAdapter, dry_profile_measurement, run_conda_probe, transformers_profile_measurement
from core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult


class FullKVAdapter(ProfileAdapter):
    name = "full"
    env = "tailguardkv-base"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("full_gpu", self.name, self.env, lossy=False, exact=True),
            ProfileSpec("full_cpu", self.name, self.env, lossy=False, exact=True, metadata={"device_mode": "cpu"}),
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
            row = transformers_profile_measurement(
                self.name,
                self.env,
                request,
                spec,
                self.runtime_config,
                extra={"family": spec.family, "profile_note": "full/exact transformers smoke"},
            )
            if not row.ok:
                return replace(row, error=f"full transformers profile failed ({profile_name}): {row.error or ''}")
            return row
        scale = max(request.prompt_chars, 1)
        return dry_profile_measurement(self.name, request, spec, scale * 0.08, scale * 2.0 / 1024.0)
