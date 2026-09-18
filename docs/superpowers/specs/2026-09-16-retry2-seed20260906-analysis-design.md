# retry2 seed20260906 分析设计

## 目标

针对 `policy_attempts/retry2_20260915/seed20260906` 的 16 个成功 policy cell，生成可复现的诊断性分析报告、汇总数据和静态图表。所有产物写入 `out/full_25session_baseline_9_14/full/seed20260906_analysis/`。

## 输入与完整性门禁

分析器只读取该 seed 的 16 份逐请求 policy CSV 和 attempt 的 `fourth_grid_status.json`。它要求每个 cell 为 `success`、每个 CSV 含五个 policy、每个 policy 恰有 65 个唯一 evaluation request，并且所有记录具有 `diagnostic_only=true`、`backend_name=online_qwen` 与 `ok=true`。任一条件失败即不写最终结果。

## 数据与图表

按 `budget_mib`、`epsilon`、`delta`、`policy` 汇总逐请求 TTFT 的 p50/p95、resident KV 的均值/p95、预算事件率、restore/recompute/evict/queue 事件率及 action profile 分布。输出审计 JSON、总汇总 CSV、事件 CSV、session points CSV、TTFT、KV、预算事件、动作分布、参数敏感性和 session 趋势 PNG。

## 报告与限制

Markdown 报告记录数据覆盖、关键发现、生成文件及方法。所有表格和图表保留 `diagnostic_only=true`；报告明确该分析限于一个 seed、retry2 的历史成功子集，不能作为正式 baseline 或论文结论。

## 验证

测试使用最小合成 CSV 覆盖完整性拒绝与指标复算。实际生成后验证审计的覆盖数、汇总分组数、图表存在性和 Markdown 的诊断性声明。
