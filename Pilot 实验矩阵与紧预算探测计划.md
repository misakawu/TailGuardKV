# Pilot 实验矩阵与紧预算探测计划

## 1. 摘要

将当前“单预算 4900 MiB、单组 epsilon=0.05、delta=0.05”调整为：

- 显存预算：通过预探测确定 2 个能真实触发 profile 选择的紧预算
- epsilon：`[0.05, 0.10]`
- delta：`[0.05, 0.10]`
- 重复次数：每个完整实验 cell 独立回放 3 次

每个 batch 的正式矩阵为：

```text
2 个预算 × 2 个 epsilon × 2 个 delta × 3 次重复 = 24 个运行 cell
```

完整实验数量还需乘以策略和任务维度。当前阶段只运行一个代表性 batch 进行紧预算探测，不启动完整矩阵。

## 2. 已锁定条件

- epsilon 两档确定为 `0.05` 和 `0.10`
- delta 两档确定为 `0.05` 和 `0.10`
- 探测阶段使用 `epsilon=0.05`、`delta=0.05`
- 探测阶段使用 `repeat_rounds: 1`
- 正式实验使用 `repeat_rounds: 3`
- 预算采用“500 MiB 粗扫，再用 100 MiB 细化”的两阶段搜索
- 三次重复是每个预算、epsilon、delta、策略和任务组合分别回放三次
- 两个最终紧预算尚未通过实验确认，不得提前写死为 4900 MiB 或其他推测值

## 3. 原 batch7 OOM 与数据处理决策

### 3.1 OOM 根因

原始 `batch007` 在 profile 生成阶段发生真实 GPU OOM，失败证据位于：

```text
.worktrees/baseline-smoke-final-aggregation/out/
baseline_smoke_final_aggregation_20260910/
batch007_local_smoke/profile_tables/
diagnostic_session27_profiles_batch007_failed_chunks.csv
```

生成阶段还需分配约 `1.68 GiB`，两张 GPU 当时仅剩约 `1.0–1.2 GiB`。虽然已设置：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

但仍无法完成，因此根因是实际显存容量不足，而不是单纯的显存碎片。

处理决策：

- 放弃原始 batch7
- 不恢复、不补跑原始 batch7
- 不将其部分产物纳入正式汇总
- 不采用运行时发生 OOM 后临时跳过的方式

### 3.2 可复现的数据排除方式

应保留原始 fixture 不变，并从输入数据层按固定 session ID 排除原 batch7 对应的数据：

```text
hybrid-session-014
hybrid-session-015
```

源 fixture：

```text
data/fixtures/diagnostic_session27.jsonl
```

过滤审计预期为：

```text
25 sessions
125 requests
```

建议记录：

```yaml
dataset_source: data/fixtures/diagnostic_session27.jsonl
dataset_version: diagnostic_session27_exclude_original_batch7_v1
excluded_sessions:
  - hybrid-session-014
  - hybrid-session-015
expected_sessions: 25
expected_requests: 125
```

同时保存：

- 源 fixture 的 SHA-256
- 过滤后 fixture 或 manifest 的 SHA-256
- runner 版本及 Git commit
- 配置文件快照
- `sessions_per_batch`
- batch 物化顺序
- 随机种子和重复编号
- 实际启动命令

过滤后 runner 会重新物化并编号 batch。因此，过滤后的 `batch007` 不等于发生 OOM 的原始 `batch007`。是否正确排除原 batch7，必须依据固定 session ID 排除清单和数据校验值判断，不能只依赖动态 batch 编号。

## 4. 代表性 batch 选择

探测阶段只选择一个代表性 batch，并在所有预算之间保持以下内容不变：

- trace
- 并发压力
- profile 实测数据
- 策略集合
- epsilon 和 delta
- 重复编号

代表性 batch 应满足：

- 不包含 `hybrid-session-014` 和 `hybrid-session-015`
- 能形成明显的并发驻留压力
- profile 数据完整
- 不会像原始 batch7 一样必然 OOM
- 能在不同预算之间公平复用同一输入

选择前只读检查过滤后 manifest，比较各 batch 的 session 数、request 数、长度分布、时间重叠、并发度和估算 KV 驻留压力，不得盲选。

## 5. 两阶段紧预算探测

### 5.1 探测配置

探测阶段只使用一组代表性质量约束和一次回放：

```yaml
pilot:
  epsilons: [0.05]
  deltas: [0.05]

session_trace:
  memory_budgets_mib: [4900, 4400, 3900, 3400]
  repeat_rounds: 1
```

使用 runner：

```text
scripts/run_diagnostic_session_batches.py
```

通过以下参数限制只运行一个代表性 batch：

```text
--only-batches batchXXX
```

输出写入新的目录，例如：

```text
out/tight_budget_probe_batchXXX_20260910
```

不得覆盖已有输出。

### 5.2 第一阶段：500 MiB 粗扫

依次扫描：

```text
4900 MiB
4400 MiB
3900 MiB
3400 MiB
```

必要时继续降低。每个候选预算记录：

- `global_resident_kv_mib` 峰值和时间序列
- `budget_hit`
- `evict`
- `restore`
- `recompute`
- 各 profile 的选择次数和占比
- exact fallback 是否出现
- 是否存在无法满足预算的情况
- 回放是否完整完成

如果降至当前 profile 集合的最低可行驻留需求仍未出现选择变化，则停止盲目降低预算，检查 trace 的并发驻留压力是否不足。

### 5.3 第二阶段：100 MiB 细扫

当相邻两个粗粒度预算之间出现以下任一明显变化时，在该区间按 100 MiB 步长细化：

- profile 选择分布发生变化
- `budget_hit` 从零变为非零或频率明显增加
- evict、restore、recompute 开始出现或显著增加
- exact fallback 开始出现
- `global_resident_kv_mib` 开始接近预算线

细扫继续使用相同代表性 batch、trace、策略、`epsilon=0.05`、`delta=0.05` 和 `repeat_rounds=1`。

## 6. 紧预算 A 和 B 的选择标准

### 6.1 紧预算 A

- 稳定触发预算约束
- 存在非零 `budget_hit`，或预算压力直接导致 evict、restore、recompute
- 驻留显存在回放期间接近预算线
- 至少两个可行 profile 出现非零选择或切换
- 回放能够完整完成

### 6.2 紧预算 B

- 比 A 更紧
- budget-hit 或降级动作频率高于 A
- profile 分布体现更高压力
- 回放仍能完整完成
- 不能几乎所有请求都采用同一个最低成本 profile
- 不能以大量 exact fallback 或无法满足预算为主要行为

A 和 B 必须代表不同压力等级，不能只是行为基本相同的两个相邻值。

## 7. 正式实验配置

紧预算 A、B 经真实探测确认后，正式配置更新为：

```yaml
pilot:
  epsilons: [0.05, 0.10]
  deltas: [0.05, 0.10]

session_trace:
  memory_budgets_mib: [tight_budget_a, tight_budget_b]
  repeat_rounds: 3
```

`tight_budget_a` 和 `tight_budget_b` 必须替换为实测 MiB 数值。

三次重复通过 `repeat_rounds: 3` 表达，不与 trace 压力复制参数混用。所有策略必须使用相同 trace、预算、epsilon、delta、重复编号和过滤后数据版本，以保证配对比较。

三轮结果必须独立保存，汇总时报告均值和离散程度，不得将三轮请求简单拼接为一次回放。

## 8. 验证与验收

### 8.1 配置展开

- 每个正式 batch 有 2 个预算、2 个 epsilon、2 个 delta
- 每个组合包含 3 个独立回放
- 每个 batch 共 24 个运行 cell，再乘以策略和任务维度
- 不再残留仅使用 `4900 MiB × epsilon=0.05 × delta=0.05` 的正式配置
- 单值旧配置在解析层面仍可读取

### 8.2 紧预算有效性

- 两个预算均有非零 budget-hit，或有预算压力导致的 evict、restore、recompute
- 驻留显存实际接近预算线
- profile 分布相对宽松预算发生变化
- A、B 表示不同压力等级
- 两个预算均能完成回放
- B 不退化为几乎所有请求使用同一最低成本 profile

### 8.3 重复正确性

- 每个正式 cell 恰有 3 个独立结果
- 日志可区分重复编号
- 三轮结果独立保存
- 汇总报告均值和离散程度
- 不把三轮请求拼接成一轮

### 8.4 数据排除正确性

- 原始 fixture 保持不变
- 过滤后固定排除 `hybrid-session-014` 和 `hybrid-session-015`
- 审计结果为 25 sessions、125 requests
- 两次实验使用相同过滤后数据版本和 SHA-256
- manifest、profile 和批次索引由过滤后数据重新生成
- 不通过重新编号后的 `batch007` 判断是否已排除原 batch7

### 8.5 回归检查

- profile 汇总仍能处理 epsilon 和 delta 列表
- session 实验笛卡尔积数量符合预期
- 单值旧配置保持兼容
- 至少完成一条真实 vLLM/LMCache 小 trace smoke test
- 确认预算动作接口不只在 measured-replay backend 中成立

## 9. 当前进度

### 9.1 已完成

- 已创建工作树：`.worktrees/baseline-smoke-final-aggregation`
- 已使用分支：`baseline-smoke-final-aggregation`
- 已定位原始 batch007 的 OOM，并确认属于真实 GPU 容量不足
- 已决定放弃原始 batch7
- 已确认参考流程 `out/session_exlude_batch7` 排除 `hybrid-session-014` 和 `hybrid-session-015`
- 曾启动排除 session 后的全批次重建
- 用户指出偏离当前计划后，已停止该运行
- 原主进程为 `46318`，停止后未发现相关 runner 或 worker 残留
- 已恢复并暂存：

```text
configs/baseline_wide_sweep.yaml
configs/pilot_external_baseline_quality.yaml
configs/pilot_external_baseline_session.yaml
```

- 相关测试曾分别获得：

```text
55 passed
51 passed
```

上述恢复内容不得在后续工作中无意撤销。

### 9.2 已停止且不能作为完整结果使用

```text
out/baseline_smoke_excluding_original_batch7_20260910
out/baseline_smoke_excluding_original_batch7_20260910.log
out/baseline_smoke_excluding_original_batch7_20260910.pid
```

该输出可能包含已完成的 `batch000` 和部分 `batch001`，但不能视为完整实验结果，不能用于正式汇总，也不能被覆盖。

### 9.3 尚未完成

- 尚未分析过滤后 manifest 的逐 batch 压力
- 尚未选定代表性 batch
- 尚未启动单 batch 四档预算粗扫
- 尚未执行 100 MiB 细扫
- 尚未确定紧预算 A 和 B
- 尚未将最终预算写入正式配置
- 尚未更新论文规划的最终 Pilot 参数
- 尚未启动正式矩阵
- 尚未完成真实 vLLM/LMCache 小 trace smoke test

## 10. 下一步执行顺序

1. 在隔离工作树中只读检查过滤后 manifest。
2. 验证过滤审计为 25 sessions、125 requests。
3. 比较各 batch 的长度、并发度和 KV 驻留压力。
4. 选择一个不包含已排除 session 的代表性 batch。
5. 创建单 batch 探测配置或 override。
6. 使用 `--only-batches batchXXX` 限制运行范围。
7. 使用 `setsid + nohup` 异步启动到新的输出目录。
8. 报告 PID、日志和输出路径，不主动轮询。
9. 用户通知结束后分析预算事件、驻留曲线和 profile 分布。
10. 在行为变化区间执行 100 MiB 细扫。
11. 确定紧预算 A 和 B。
12. 更新正式配置和 `论文规划/王祯祥_论文工作规划.md`。
13. 验证正式矩阵、独立重复、旧配置兼容性和真实后端 smoke test。

## 11. 执行约束

- 当前阶段只启动一个代表性 batch
- 不再次启动完整批次矩阵
- 所有实验脚本使用 `setsid + nohup` 异步运行
- 进程结束由用户提醒，不主动轮询
- 不覆盖已有输出目录
- 不提前写死最终预算
- 不撤销已恢复并暂存的三个配置文件
- 不访问其他用户目录
- 不进行超出 `/DATACENTER3/zhenxiang.wang` 范围的扫描
- 若需修改代码，先读取 TDD 指引，先写失败测试，再实现

## 12. 论文规划同步

预算探测完成后，同步更新：

```text
论文规划/王祯祥_论文工作规划.md
```

更新内容包括：

- 两档 epsilon 为 `0.05` 和 `0.10`
- 两档 delta 为 `0.05` 和 `0.10`
- 两个紧预算来自 500 MiB 粗扫和 100 MiB 细化
- 每个完整 cell 独立回放三次
- 原 batch7 对应 session 已从统一实验输入版本中排除
- 两个紧预算的实际数值和选择依据

## 13. 当前结论

OOM 根因和原 batch7 的排除原则已经明确，全批次误启动已经停止。当前处于选择单个代表性 batch，并准备 `4900/4400/3900/3400 MiB` 粗扫的阶段。

两个最终紧预算尚无实验依据，正式矩阵暂不能启动，也不 能提前填写推测预算值。
