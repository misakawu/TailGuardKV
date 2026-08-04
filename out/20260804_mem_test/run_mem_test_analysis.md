# run_mem_test 显存预算分析

- summary: `out/20260804_mem_test/policy_tables/run_mem_test_total_summary.csv`
- found_passing_budget: `false`

## 结论

未找到同时满足 KIVI/H2O 显存收益和 TTFT 系统收益的非 oracle budget 点。

## 近似点

- budget=20.0 MiB, policy=static_best, kv_drop_mean=0.806, kv_drop_p95=0.802, mean_ttft_delta_ms=86.016, p95_ttft_delta_ms=33.319
- budget=20.0 MiB, policy=static_best, kv_drop_mean=0.806, kv_drop_p95=0.802, mean_ttft_delta_ms=86.016, p95_ttft_delta_ms=33.319
- budget=20.0 MiB, policy=static_best, kv_drop_mean=0.806, kv_drop_p95=0.802, mean_ttft_delta_ms=86.016, p95_ttft_delta_ms=33.319
- budget=20.0 MiB, policy=static_best, kv_drop_mean=0.806, kv_drop_p95=0.802, mean_ttft_delta_ms=86.016, p95_ttft_delta_ms=33.319
- budget=20.0 MiB, policy=static_safe, kv_drop_mean=0.778, kv_drop_p95=0.797, mean_ttft_delta_ms=243.275, p95_ttft_delta_ms=1641.545
- budget=30.0 MiB, policy=static_safe, kv_drop_mean=0.778, kv_drop_p95=0.797, mean_ttft_delta_ms=243.275, p95_ttft_delta_ms=1641.545
- budget=30.0 MiB, policy=static_best, kv_drop_mean=0.708, kv_drop_p95=0.703, mean_ttft_delta_ms=86.258, p95_ttft_delta_ms=59.395
- budget=40.0 MiB, policy=static_best, kv_drop_mean=0.708, kv_drop_p95=0.703, mean_ttft_delta_ms=86.258, p95_ttft_delta_ms=59.395
- budget=50.0 MiB, policy=static_best, kv_drop_mean=0.708, kv_drop_p95=0.703, mean_ttft_delta_ms=86.258, p95_ttft_delta_ms=59.395
- budget=60.0 MiB, policy=static_best, kv_drop_mean=0.708, kv_drop_p95=0.703, mean_ttft_delta_ms=86.258, p95_ttft_delta_ms=59.395
