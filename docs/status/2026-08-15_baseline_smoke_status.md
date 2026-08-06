# 2026-08-15 Baseline Smoke Status

## Summary

The 5-baseline smoke table has been generated and validated with `configs/pilot.yaml`.

Outputs:

- Profile table: `out/20260806_122619_pilot/profile_tables/pilot_smoke_measured_profiles.csv`
- Policy tables and summary: `out/20260806_122619_pilot_repolicy/policy_tables/`
- Diagnostic: `out/20260806_122619_pilot_repolicy/policy_tables/SMOKE_DIAGNOSTIC.txt`

All required policies completed:

- `full_lru`
- `static_best`
- `static_safe`
- `utility_dynamic`
- `uncalibrated_dynamic`

Each policy produced 100 valid `ok=True` evaluation records per sweep. The policy outputs no longer contain `tailguard` or `quality_oracle`.

## Diagnostic Findings

Reference cell: `epsilon=0.05`, `delta=0.05`, `memory_budget_mib=4900`.

H0 is supported. `static_best` has `mean_quality_loss=0.029552`, but `p95_quality_loss=0.128696` and `violation_rate=0.24`, showing that average quality hides tail loss.

H1 is supported as a baseline failure case. `uncalibrated_dynamic` has `violation_rate=0.21`, which is substantially above `delta=0.05`.

H2-lite is not supported by TTFT in this smoke cell. `utility_dynamic` chooses lossy profiles and reduces mean KV cache memory from `36.161563 MiB` to `12.927018 MiB`, but its `p95_ttft_ms=2985.559274` is higher than `full_lru` at `2790.079042`.

## Implementation Notes

`utility_dynamic` and `uncalibrated_dynamic` were corrected to behave as uncalibrated baselines. They now choose from lossy candidates first and no longer use conformal risk upper to force exact fallback. `static_safe` remains the calibrated fixed-profile baseline.

Smoke 表获取任务已完成，可进入下一阶段（实现 TailGuard / Oracle）。H2-lite 的 TTFT 收益仍需在 TailGuard 阶段继续追踪。
