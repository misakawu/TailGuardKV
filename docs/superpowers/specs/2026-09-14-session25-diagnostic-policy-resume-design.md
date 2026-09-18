# 25-session 诊断性 baseline smoke 恢复方案

## 目标

复用 `out/full_25session_baseline_9_14/full` 已测量的 profile，得到可回溯的诊断性 policy 表、TTFT 等图表。所有结果保留 `diagnostic_only=true`，不作为正式 baseline 结论。

## 流程

1. 根据 manifest 校验 25 个 session、125 条请求、8 个 profile 的 1000 条成功测量；拒绝缺失、重复、失败或多份不明确的批次文件。
2. 仅在校验全部通过时合并 profile，并将 `supervisor_manifest.json` 标记为 `merged=true`；原始批次和备份保持不变。
3. 使用现有 `run_session25_fourth_sweeps.py` 的 3 个 seed、4 个派生预算、4 组 ε/δ 和 5 个 baseline 进行 policy 实验；失败即停止并保留日志及状态。
4. 按 seed 输出逐请求记录、汇总表、`baseline_smoke.md`、TTFT/KV/事件图，核验表图与逐请求数据一致。备份 `policy_tables` 只供解释结果，不作为本次输入。

## 验收

- 合并 profile 覆盖完整且来源明确；缺失批次时不启动 policy。
- 每个 policy cell 为 125 条请求，所有输出保留诊断性 provenance。
- 三个 seed 各自有统计表和可视化；不将 seed 混合后掩盖异常，不将诊断性数字用于正式论文结论。
- 所有程序使用 `nohup` + `setsid` 异步启动，记录 PID、日志与退出码；运行期间不轮询，用户通知结束后再检查结果。
