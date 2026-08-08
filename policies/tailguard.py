from __future__ import annotations

import time
from collections.abc import Iterable
from math import inf

from run_util.core_types import Action, CacheState, DeviceState, ProfileMeasurement, Request
from policies.base import StatsPolicy


class TailGuardPolicy(StatsPolicy):
    def __init__(
        self,
        calibration_measurements: Iterable[ProfileMeasurement],
        profiles: list[str],
        epsilon: float,
        delta: float,
        exact_profiles: set[str],
        memory_budget_mib: float = float("inf"),
        record_rejected_unsafe: bool = True,
    ) -> None:
        super().__init__(
            "tailguard",
            calibration_measurements,
            profiles,
            epsilon,
            delta,
            exact_profiles,
            placeholder=False,
            memory_budget_mib=memory_budget_mib,
        )
        self.record_rejected_unsafe = record_rejected_unsafe

    def decide(self, request: Request, cache_state: CacheState, device_state: DeviceState) -> Action:
        start = time.perf_counter()
        safe_candidates: list[tuple[float, str, float, float]] = []
        rejected: tuple[str, float, float] | None = None
        budget_filtered = False
        qrp_ms = 0.0
        cg_ms = 0.0
        stc_start = time.perf_counter()
        for profile in [profile for profile in self.profiles if profile not in self.exact_profiles]:
            if not self._within_memory_budget(profile, request, cache_state):
                budget_filtered = True
                continue
            qrp_start = time.perf_counter()
            pred_loss = self.predictor.predict_loss(request, profile)
            qrp_ms += (time.perf_counter() - qrp_start) * 1000
            cg_start = time.perf_counter()
            risk_upper = self.guard.risk_upper(request, profile, pred_loss)
            cg_ms += (time.perf_counter() - cg_start) * 1000
            if risk_upper <= self.epsilon:
                safe_candidates.append((self._ttft_or_inf(profile), profile, pred_loss, risk_upper))
            elif rejected is None or risk_upper < rejected[2]:
                rejected = (profile, pred_loss, risk_upper)
        safe_candidates.sort(key=lambda item: item[0])
        stc_ms = (time.perf_counter() - stc_start) * 1000
        if safe_candidates:
            _, profile, pred_loss, risk_upper = safe_candidates[0]
            return Action(
                profile=profile,
                reason="tailguard calibrated safe",
                pred_loss=pred_loss,
                risk_upper=risk_upper,
                safe=True,
                budget_hit=budget_filtered,
                epsilon=self.epsilon,
                delta=self.delta,
                candidate_safe_count=float(len(safe_candidates)),
                controller_overhead_ms=(time.perf_counter() - start) * 1000,
                controller_qrp_ms=qrp_ms,
                controller_cg_ms=cg_ms,
                controller_stc_ms=stc_ms,
            )

        fallback = self._fastest_exact_profile()
        pred_loss, risk_upper, safe, _ = self._predict_and_guard(request, fallback)
        rejected_profile = rejected[0] if rejected and self.record_rejected_unsafe else ""
        return Action(
            profile=fallback,
            reason="tailguard exact fallback",
            pred_loss=pred_loss,
            risk_upper=risk_upper,
            safe=safe,
            budget_hit=budget_filtered,
            epsilon=self.epsilon,
            delta=self.delta,
            fallback_reason="no calibrated safe lossy profile within memory budget",
            rejected_profile=rejected_profile,
            rejected_pred_loss=(rejected[1] if rejected_profile else None),
            rejected_risk_upper=(rejected[2] if rejected_profile else None),
            candidate_safe_count=0.0,
            controller_overhead_ms=(time.perf_counter() - start) * 1000,
            controller_qrp_ms=qrp_ms,
            controller_cg_ms=cg_ms,
            controller_stc_ms=stc_ms,
        )
