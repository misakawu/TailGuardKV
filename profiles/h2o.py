from __future__ import annotations

from dataclasses import replace

from profiles.base import ProfileAdapter, dry_profile_measurement, qwen2_kv_profile_measurement, run_conda_probe
from core_types import ProfileSpec, Request, SmokeResult


class H2OAdapter(ProfileAdapter):
    name = "h2o"
    env = "edgekv-h2o"
    pythonpath = ("third_party/H2O/h2o_hf",)

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec(
                "h2o_heavy_hitter",
                self.name,
                self.env,
                lossy=True,
                metadata={"strategy": "heavy_hitter"},
            ),
        )

    def smoke(self, timeout_s: int = 120) -> SmokeResult:
        ok, versions, error = run_conda_probe(
            self.env,
            ("torch", "transformers", "utils_hh.modify_llama"),
            timeout_s=timeout_s,
            pythonpath=self.pythonpath,
        )
        return SmokeResult(
            adapter=self.name,
            env=self.env,
            ok=ok,
            profiles=self.profile_names(),
            detail="H2O 通过 PYTHONPATH=third_party/H2O/h2o_hf 使用 monkeypatch 入口。",
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
                    "strategy": spec.metadata.get("strategy", ""),
                },
            )
            if not row.ok:
                return replace(row, error=f"H2O proof/runtime failed ({profile_name}): {row.error or ''}")
            return row
        scale = max(request.prompt_chars, 1)
        return dry_profile_measurement(self.name, request, spec, scale * 0.075, scale * 1.0 / 1024.0)
