# retry2 seed20260906 policy analysis

`diagnostic_only=true`。本报告仅覆盖 retry2 的单个 seed 和 16 个成功 cell，不能用作正式 baseline、TailGuard 或论文结论。

## Coverage

- Seed: `20260906`
- Successful cells: `16`
- Raw records: `5200`
- Summary rows: `80`
- Policies: `full_lru, static_best, static_safe, uncalibrated_dynamic, utility_dynamic`

## Reading the outputs

- `seed20260906_total_summary.csv` is the recomputable per-cell policy table; it retains epsilon, delta, and memory budget provenance.
- `seed20260906_events.csv` and `seed20260906_session_points.csv` retain event and session-level evidence.
- PNG curves are separated by epsilon/delta panels. They only contain accepted cells; no failed cell is converted to zero or connected into a line.
- Quality and violation fields remain `risk_evidence_insufficient`; interpret TTFT, KV, budget, events, and actions as runtime diagnostics only.
