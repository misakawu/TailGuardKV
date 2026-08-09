from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from run_util.core_types import ProfileMeasurement, ProfileSpec, Request, SmokeResult
from profiles.base import (
    ProfileAdapter,
    dry_profile_measurement,
    qwen2_exact_profile_many_measurements,
    run_conda_probe,
)


class FullKVAdapter(ProfileAdapter):
    name = "full"
    env = "tailguardkv-base"

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec("full_gpu", self.name, self.env, lossy=False, exact=True),
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
            row = self.profile_many(
                [request],
                profile_name,
                dry_run=False,
                session_runtime=session_runtime,
                memory_budget_mib=memory_budget_mib,
            )[0]
            if not row.ok:
                return replace(row, error=f"full transformers profile failed ({profile_name}): {row.error or ''}")
            return row
        scale = max(request.prompt_chars, 1)
        return dry_profile_measurement(self.name, request, spec, scale * 0.08, scale * 2.0 / 1024.0)

    def profile_many(
        self,
        requests: Sequence[Request],
        profile_name: str,
        dry_run: bool = True,
        session_runtime: object | None = None,
        memory_budget_mib: float | None = None,
        persistent_worker: object | None = None,
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
        rows = qwen2_exact_profile_many_measurements(
            self.name,
            self.env,
            requests,
            spec,
            self.runtime_config,
            session_runtime=session_runtime,
            memory_budget_mib=memory_budget_mib,
            extra={"family": spec.family, "profile_note": "full/exact qwen2 runtime"},
            persistent_worker=persistent_worker,
        )
        repaired = []
        for row in rows:
            if not row.ok:
                repaired.append(replace(row, error=f"full transformers profile failed ({profile_name}): {row.error or ''}"))
            else:
                repaired.append(row)
        return repaired
