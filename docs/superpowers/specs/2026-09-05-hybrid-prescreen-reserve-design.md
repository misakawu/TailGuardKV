# 混合会话预筛 Reserve 门禁设计

**目标：** 在 LongBench 已耗尽时，仅允许 KIVI-sensitive QA 和 Summary 的直接预筛 reserve 分别从 12 降至 7 和 10，以继续构造并实测最终混合 session。

## 变更范围

- `kivi_sensitive/qa` 的 direct prescreen `required_per_cell` 为 7。
- `kivi_sensitive/summary` 的 direct prescreen `required_per_cell` 为 10。
- 其余四个 `risk/task` direct prescreen cell 仍为 12。
- 该规则只影响“是否允许进入 final-form hybrid 测量”；不能把 direct prescreen 标签写入最终 fixture。

## 不变约束

- RAGhot 仍只允许 QA，不能用于 Summary。
- 每个 final-form hybrid batch 仍为 12 个五轮 session，所有请求必须具有 8-profile 完整、`ok=true`、`measured=true` 测量。
- 最终 session fixture 仍必须为 48 个 session、240 条记录，且每个 `risk x task x split` cell 恰有 4 个 session。
- provenance、payload/source/skeleton 唯一性、tie 拒绝、arrival 交错与 quality/trace/risk smoke gate 均不放宽。

## 判定流程

1. 完整合并 32 个 LongBench prescreen manifest。
2. 对四个未放宽 cell 要求至少 12 条；对 `kivi_sensitive/qa` 和 `kivi_sensitive/summary` 分别要求至少 7 和 10 条。
3. 满足后，按现有确定性排序选择内容与 ShareGPT skeleton，构造一个 12-session final-form hybrid batch。
4. 仅根据 final-form 8-profile 测量重新分类；如果最终严格 reserve 或 48-session fixture 不足，导出精确不足报告并停止，不回填 tie 或伪造分类。

## 验证

- 默认 gate 在现有计数 `kivi_sensitive/qa=7`、`kivi_sensitive/summary=10` 时返回可继续；任一分别降至 6 或 9 时仍返回不足。
- 其他 cell 即使降至 11 也必须返回不足。
- 现有 final fixture 严格分布测试保持通过。
