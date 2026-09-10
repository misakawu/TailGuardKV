# TailGuardKV 排除 Batch7 后的诊断汇总

## 数据范围

- 工作仓库：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV`。
- 数据目录：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV/out/session32_request_id_batch7_recovery_20260908`。
- Batch7 已完全排除，不参与统计、排序或结论。
- 纳入批次：batch000, batch001, batch002, batch003, batch004, batch005, batch006, batch008, batch009, batch010, batch011, batch012。
- 合并记录：923 行，102 个字段，8 个 profile。
- 各批次字段不完全一致，主表采用字段并集，缺失字段留空。

## Profile 统计

- **full_gpu**：记录 124；会话 25；请求 124；平均 TTFT 711.13 ms；P95 TTFT 3997.67 ms；平均峰值显存 15860.27 MiB。
- **h2o_heavy10_recent10**：记录 115；会话 23；请求 115；平均 TTFT 803.88 ms；P95 TTFT 4105.90 ms；平均峰值显存 15844.80 MiB。
- **h2o_heavy15_recent15**：记录 115；会话 23；请求 115；平均 TTFT 423.28 ms；P95 TTFT 1301.03 ms；平均峰值显存 15849.55 MiB。
- **h2o_heavy20_recent20**：记录 109；会话 23；请求 109；平均 TTFT 415.23 ms；P95 TTFT 1304.43 ms；平均峰值显存 15824.52 MiB。
- **kivi_2bit_residual32**：记录 115；会话 23；请求 115；平均 TTFT 425.39 ms；P95 TTFT 1282.33 ms；平均峰值显存 15848.92 MiB。
- **kivi_2bit_residual64**：记录 115；会话 23；请求 115；平均 TTFT 418.56 ms；P95 TTFT 1279.58 ms；平均峰值显存 15849.93 MiB。
- **kivi_4bit_residual32**：记录 115；会话 23；请求 115；平均 TTFT 868.84 ms；P95 TTFT 3765.62 ms；平均峰值显存 15854.05 MiB。
- **kivi_4bit_residual64**：记录 115；会话 23；请求 115；平均 TTFT 431.25 ms；P95 TTFT 1300.94 ms；平均峰值显存 15855.58 MiB。

## 限制

- 当前结果是描述性诊断汇总。
- 若缺少完整可合并的 policy/session trace，不据此宣称策略效果或会话复用已完成整体验证。
- 检测到 1 个 failed-chunk sidecar，已记录但未混入主数据。
