# Session25 Fourth online baseline smoke（diagnostic_only）

| memory_budget_mib | policy | p95_ttft_ms | mean_ttft_ms | mean_kv_cache_memory_mib | budget_hit_rate | restore_count | recompute_count | mean_quality_loss | quality_status |
|---|---|---|---|---|---|---|---|---|---|
| 120.258 | full_lru | 193.537 | 169.332 | 8.6381 | 0.446154 | 0 | 28 |  | risk_evidence_insufficient |
| 120.258 | static_best | 237.578 | 232.25 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 120.258 | static_safe | 242.318 | 232.268 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 120.258 | uncalibrated_dynamic | 234.818 | 224.784 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 120.258 | utility_dynamic | 238.378 | 226.83 | 3.69827 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 469.602 | full_lru | 100.528 | 143.809 | 8.97885 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 469.602 | static_best | 201.47 | 198.328 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 469.602 | static_safe | 209.96 | 204.464 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 469.602 | uncalibrated_dynamic | 231.959 | 227.544 | 4.48737 | 0 | 0 | 0 |  | risk_evidence_insufficient |
| 469.602 | utility_dynamic | 233.373 | 229.302 | 3.69827 | 0 | 0 | 0 |  | risk_evidence_insufficient |

> 全部输出为 `diagnostic_only=true`；quality/violation 均为 `risk_evidence_insufficient`，不作为 tail SLO 结论。
