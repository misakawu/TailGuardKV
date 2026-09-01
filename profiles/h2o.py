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


class H2OAdapter(ProfileAdapter):
    name = "h2o"
    env = "edgekv-h2o"
    pythonpath = ("third_party/H2O/h2o_hf",)

    def profiles(self) -> tuple[ProfileSpec, ...]:
        return (
            ProfileSpec(
                "h2o_heavy05_recent05",
                self.name,
                self.env,
                lossy=True,
                metadata={"h2o_heavy_ratio": 0.05, "h2o_recent_ratio": 0.05},
            ),
            ProfileSpec(
                "h2o_heavy08_recent08",
                self.name,
                self.env,
                lossy=True,
                metadata={"h2o_heavy_ratio": 0.08, "h2o_recent_ratio": 0.08},
            ),
            ProfileSpec(
                "h2o_heavy10_recent10",
                self.name,
                self.env,
                lossy=True,
                metadata={"h2o_heavy_ratio": 0.10, "h2o_recent_ratio": 0.10},
            ),
            ProfileSpec(
                "h2o_heavy15_recent15",
                self.name,
                self.env,
                lossy=True,
                metadata={"h2o_heavy_ratio": 0.15, "h2o_recent_ratio": 0.15},
            ),
            ProfileSpec(
                "h2o_heavy20_recent20",
                self.name,
                self.env,
                lossy=True,
                metadata={"h2o_heavy_ratio": 0.20, "h2o_recent_ratio": 0.20},
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
                pythonpath=self.pythonpath,
                extra={
                    "family": spec.family,
                    "h2o_heavy_ratio": spec.metadata.get("h2o_heavy_ratio", ""),
                    "h2o_recent_ratio": spec.metadata.get("h2o_recent_ratio", ""),
                },
            )
            if not row.ok:
                return replace(row, error=f"H2O proof/runtime failed ({profile_name}): {row.error or ''}")
            return row
        scale = max(request.prompt_chars, 1)
        heavy_ratio = float(spec.metadata["h2o_heavy_ratio"])
        return dry_profile_measurement(self.name, request, spec, scale * (0.07 + heavy_ratio * 0.05), scale * (0.7 + heavy_ratio) / 1024.0)

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
        rows = qwen2_kv_profile_many_measurements(
            self.name,
            self.env,
            requests,
            spec,
            self.runtime_config,
            pythonpath=self.pythonpath,
            session_runtime=session_runtime,
            memory_budget_mib=memory_budget_mib,
            extra={
                "family": spec.family,
                "h2o_heavy_ratio": spec.metadata.get("h2o_heavy_ratio", ""),
                "h2o_recent_ratio": spec.metadata.get("h2o_recent_ratio", ""),
            },
            persistent_worker=persistent_worker,
        )
        repaired = []
        for row in rows:
            if not row.ok:
                repaired.append(replace(row, error=f"H2O proof/runtime failed ({profile_name}): {row.error or ''}"))
            else:
                repaired.append(row)
        return repaired
