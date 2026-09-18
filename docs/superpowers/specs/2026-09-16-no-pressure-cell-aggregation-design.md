# 无压力 Cell 聚合与简化图表设计

## 目标

使已完成且无后端压力事件的 baseline session policy cell 可被认证和聚合；最终图仅保留策略汇总曲线。

## 行为

- baseline session 记录仍须完整、全部成功，并保留 session 历史和预算上限验证。
- 缺少 `budget_hit`、evict、restore、recompute 或 queue 事件不再使单个 policy CSV 失败。
- 图表保留策略曲线和置信区间，不绘制 session-level 灰色散点。
- 现有 40-cell attempt 中完整的无压力 CSV 可以补齐 `split_seed` 后进入本地聚合；不得重跑 GPU cell。

## 验证

- 单元测试验证无压力的有效 baseline session records 被接受。
- 图表测试验证提供 session points 不会创建 scatter collection。
- 使用已有 8 个 CSV 重建汇总 CSV 和全部 PNG。
