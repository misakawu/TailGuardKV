from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from run_util.core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult
from profiles.base import (
    ProfileAdapter,
    dry_profile_measurement,
    qwen2_kv_profile_many_measurements,
    qwen2_kv_profile_measurement,
    run_conda_probe,
)


class KIVIAdapter(ProfileAdapter):
    name = "kivi"
    env = "edgekv-kivi"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("kivi_4bit_residual32", self.name, self.env, lossy=True, metadata={"bits": 4, "kivi_residual_length": 32}),
            ProfileSpec("kivi_4bit_residual64", self.name, self.env, lossy=True, metadata={"bits": 4, "kivi_residual_length": 64}),
            ProfileSpec("kivi_2bit_residual32", self.name, self.env, lossy=True, metadata={"bits": 2, "kivi_residual_length": 32}),
            ProfileSpec("kivi_2bit_residual64", self.name, self.env, lossy=True, metadata={"bits": 2, "kivi_residual_length": 64}),
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

    def profile(
        self,
        request: Request,
        profile_name: str,
        dry_run: bool = True,
        session_runtime: object | None = None,
        memory_budget_mib: float | None = None,
    ) -> ProfileMeasurement:
        del session_runtime, memory_budget_mib
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
                    "kivi_residual_length": spec.metadata.get("kivi_residual_length", ""),
                },
            )
            if not row.ok:
                return replace(row, error=f"KIVI proof/runtime failed ({profile_name}): {row.error or ''}")
            return row
        bits = int(spec.metadata["bits"])
        residual = int(spec.metadata["kivi_residual_length"])
        scale = max(request.prompt_chars, 1)
        memory_factor = 0.5 if bits == 4 else 0.25
        latency_factor = (0.09 if bits == 4 else 0.095) + (0.002 if residual == 64 else 0.0)
        return dry_profile_measurement(
            self.name,
            request,
            spec,
            scale * latency_factor,
            scale * memory_factor / 1024.0,
        )

    def profile_many(
        self,
        requests: Sequence[Request],
        profile_name: str,
        dry_run: bool = True,
        session_runtime: object | None = None,
        memory_budget_mib: float | None = None,
    ) -> list[ProfileMeasurement]:
        if dry_run:
            return super().profile_many(
                requests,
                profile_name,
                dry_run=dry_run,
                session_runtime=session_runtime,
                memory_budget_mib=memory_budget_mib,
            )
        spec = self.get_profile(profile_name)
        rows = qwen2_kv_profile_many_measurements(
            self.name,
            self.env,
            requests,
            spec,
            self.runtime_config,
            session_runtime=session_runtime,
            memory_budget_mib=memory_budget_mib,
            extra={
                "family": spec.family,
                "bits": spec.metadata.get("bits", ""),
                "kivi_residual_length": spec.metadata.get("kivi_residual_length", ""),
            },
        )
        repaired = []
        for row in rows:
            if not row.ok:
                repaired.append(replace(row, error=f"KIVI proof/runtime failed ({profile_name}): {row.error or ''}"))
            else:
                repaired.append(row)
        return repaired
