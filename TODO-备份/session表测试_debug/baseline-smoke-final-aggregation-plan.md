# Baseline Smoke 最终聚合实施计划

> **目标：** 汇总排除原 batch7 对应会话后的 12 个诊断 batch，生成可审计的统一结果；与原始 session27 结果进行同口径对比，并严格保留 `diagnostic_only=true` 边界，避免诊断结果被误用为正式 baseline 或论文证据。

## 一、当前上下文

当前工作分支为 `baseline-smoke-final-aggregation`，基于提交 `b7f8d4e`。该提交已移除原 batch7，下一步是汇总输出。

已完成的重跑位于：

```text
out/diagnostic_session25_excluding_original_batch7_v2/
```

该运行的已知边界：

- 原始数据包含 27 个 session、135 个 request。
- 排除 `hybrid-session-014` 和 `hybrid-session-015`。
- 排除后包含 25 个 session、125 个 request。
- 重新划分为 12 个 batch，通常每批 2 个 session，最后一批 3 个 session。
- 覆盖 8 个 profile：`full_gpu`、4 个 KIVI profile、3 个 H2O profile。
- 所有结果均为诊断性质，必须保留 `diagnostic_only=true` provenance。

现有相关实现包括：

- `scripts/run_diagnostic_session_batches.py`
- `scripts/aggregate_session27_baselines.py`
- `scripts/aggregate_baseline_wide_sweep.py`
- `scripts/visualize_diagnostic_smoke.py`
- `tests/test_diagnostic_batch_runner.py`
- `tests/test_session_aggregation.py`
- `tests/test_baseline_wide_sweep.py`

## 二、实施原则

1. **先审计输入，再聚合数值。** 聚合前必须验证 12 个 batch 是否齐全、manifest 与 fixture 是否一致、session 是否无重复或遗漏。
2. **复用成熟聚合逻辑。** 优先扩展或调用已有 `aggregate_session27_baselines.py`，不另造一套不兼容的指标定义。
3. **同口径比较。** 排除前后比较只能使用两侧共同存在、语义一致的字段和 profile/policy 组合。
4. **禁止伪造缺失值。** 缺失、失败、超时、拒绝或不可比记录必须显式标注��，不能用零值或均值填补。
5. **保留诊断边界。** 所有 CSV、JSON、Markdown 和图表都必须标明结果不可替代正式 baseline、质量 gate 或 session gate。
6. **不改动原始输出。** 聚合产物写入独立目录，原 batch 输出保持只读，便于回滚和复核。

## 三、任务分解

### 任务 1：建立输入清单与完整性审计

检查：

```text
out/diagnostic_session25_excluding_original_batch7_v2/
```

具体步骤：

1. 枚举 12 个 batch 的配置、fixture、manifest、profile 表、policy 表、summary 和 trace 文件。
2. 验证 batch 编号连续且数量为 12。
3. 从 manifest 和 fixture 交叉核对：
   - session 总数为 25；
   - request 总数为 125；
   - 被排除的两个 session 不存在；
   - session/request ID 在 batch 间不重复；
   - 最后一批的 3-session 特例与 manifest 一致。
4. 验证每个输出都保留 `diagnostic_only=true` 或等价 provenance。
5. 生成机器可读的��审计结果；发现任何不一致时终止后续聚合。

建议产物：

```text
out/diagnostic_session25_excluding_original_batch7_v2/aggregate/input_audit.json
```

### 任务 2：为聚合行为补充失败测试

修改或新增：

```text
tests/test_session_aggregation.py
```

测试至少覆盖：

1. 正常聚合 12 个 batch，保持 profile、policy 和 provenance 字段。
2. batch 缺失时明确失败。
3. session/request 重复时明确失败。
4. manifest 与 fixture 计数不一致时明确失败。
5. 混入非诊断输出或缺失 provenance 时明确失败。
6. failed、timeout、rejected、unsafe 等状态不会被静默当作成功样本。
7. 加权总体值基于底层有效记录或明确分母，不能简单平均 batch 均值。
8. 输出排序稳定，以保证重复运行产生一致文件。

先运行测试并确认新增测试按预期失败，再进入实现。

### 任务 3：实现 session25 最终聚合

优先扩展：

```text
scripts/aggregate_session27_baselines.py
```

若现有脚本与 session27 命名强耦合，则进行最小范围重构，使其接受：

```text
--input-root
--output-root
--expected-batches
--expected-sessions
--expected-requests
--diagnostic-only
```

实现要求：

1. 对每个 batch 读取原始 profile、policy、summary 和 trace 数据。
2. 统一 schema，并保留来源 batch、session、request、profile、policy、seed 和运行状态。
3. 基于底层记录重新计算总体指标及分母。
4. 同时输出：
   - 明细合并表；
   - profile 聚合表；
   - policy 聚合表；
   - session 聚合表；
   - 失败与缺失审计表；
   - provenance 清单；
   - JSON 摘要。
5. 对非有限数值、字段缺失和 schema 漂移执行显式检查。
6. 聚合产物统一写入：

```text
out/diagnostic_session25_excluding_original_batch7_v2/aggregate/
```

### 任务 4：生成排除前后同口径对比

比较对象：

- 排除前的 session27 诊断结果；
- 排除原 batch7 后的 session25 聚合结果。

具体步骤：

1. 自动发现或显式指定排除前结果目录，不允许根据文件名猜测后直接使用。
2. 验证两侧指标定义、profile、policy、状态编码和单位一致。
3. 只比较两侧共同的 profile/policy 组合。
4. 输出绝对差、相对差、样本数变化和失败率变化。
5. 分离展示：
   - 性能指标；
   - 质量或风险指标；
   - 成功率、超时率和失败率；
   - session 级离散程度。
6. 明确说明差异是“排除两个异常会话后的诊断变化”，不推断正式实验结论。

建议产物：

```text
out/diagnostic_session25_excluding_original_batch7_v2/aggregate/session27_vs_session25.csv
out/diagnostic_session25_excluding_original_batch7_v2/aggregate/session27_vs_session25.json
```

### 任务 5：生成最终诊断报告

生成：

```text
out/diagnostic_session25_excluding_original_batch7_v2/aggregate/README.md
```

报告包括：

1. 数据范围与排除规则。
2. 输入审计结论。
3. 12 个 batch 的运行完整性。
4. profile 和 policy 聚合结果。
5. 排除前后差异。
6. 失败、超时、拒绝和缺失记录。
7. 明确的诊断用途限制。
8. 可复现命令和输入/输出文件清单。

如生成图表，应同时保留对应源 CSV，图中必须标注 `diagnostic_only`。

### 任务 6：端到端验证

依次执行：

```bash
pytest -q tests/test_session_aggregation.py
pytest -q tests/test_diagnostic_batch_runner.py
pytest -q tests/test_baseline_wide_sweep.py
```

随后运行真实聚合命令，并验证：

1. 命令退出码为 0。
2. 输入审计通过。
3. 12 个 batch、25 个 session、125 个 request 均被正确识别。
4. 两个被排除 session 不出现在任何最终产物中。
5. 所有输出均保留诊断 provenance。
6. CSV/JSON 可重新读取，数值字段不存在意外的 NaN 或 infinity。
7. 使用相同输入重复运行后，产物内容和排序稳定。
8. `git status` 中不出现非预期源码改动，也不覆盖原始实验输出。

## 四、验收标准

仅当以下条件全部满足，才能宣称聚合完成：

- 输入审计确认 12 个 batch、25 个 session、125 个 request。
- 被排除 session 为且仅为 `hybrid-session-014`、`hybrid-session-015`。
- 聚合逻辑具有异常输入和加权分母测试。
- 所有相关测试通过。
- 最终聚合命令真实执行成功。
- 排�除前后比较经过 schema 和口径一致性检查。
- 聚合产物包含失败审计、provenance 和诊断用途声明。
- 未将诊断结果描述为正式 baseline、质量 gate、session gate 或论文结论。

## 五、风险与处理

### 原 session27 输出位置或 schema 不一致

停止自动比较，在报告中列出缺失信息；不得猜测目录或手工拼接不可比字段。

### 某些 batch 输出不完整

审计失败并列出缺失文件，不生成貌似完整的��总体表。

### batch 间 schema 漂移

显式生成 schema 差异并终止；只允许通过有测试覆盖的兼容映射修复。

### 简单平均造成统计偏差

从底层记录重新聚合，或依据每项指标的有效样本数进行加权，并在输出中保留分子、分母。

### 工作树已有无关改动

只修改聚合脚本、对应测试和新聚合产物；不恢复、删除、覆盖或提交当前已有的其他改动。

## 六、回滚方式

源码回滚仅涉及本计划后续明确修改的聚合脚本和测试，可通过 Git 恢复这些单独文件。聚合输出全部位于独立的 `aggregate/` 子目录，回滚时删除该子目录即可，原始 batch 输出不受影响。
