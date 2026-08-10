# 2026-08-10 Session Trace 质量曲线异常修复指引

## 文档目的

这份文档服务于当前 `configs/pilot_session_trace.yaml` 这一条 `baseline_session` 路径，目标不是总结论文结论，而是把“为什么质量曲线没有张开”拆成可执行的代码修复入口。

本文只回答四个问题：

1. 现象到底是什么。
2. 已确认的根因分别落在哪些模块。
3. 每个模块应该修到什么状态。
4. 修完后应该看到什么验证信号。

## 适用范围

- 配置：`configs/pilot_session_trace.yaml`
- 输出：
  - `out-备份/部分曲线符合预期/profile_tables/pilot_session_trace_measured_profiles.csv`
  - `out-备份/部分曲线符合预期/policy_tables/pilot_session_trace_measured_total_summary.csv`
  - `out-备份/部分曲线符合预期/session_traces/pilot_session_trace_pressure_trace.csv`
- 相关代码：
  - `run_util/data_utils.py`
  - `run_util/run_policies.py`
  - `backends/measured_replay.py`
  - `metrics/collector.py`
  - `policies/`
  - `data/fixtures/` 下 session trace 请求构造逻辑

## 一句话结论

当前问题的主根因不是 policy summary 聚合 bug，而是：

- 旧版 `baseline_session` trace 只有 12 条静态请求，风险样本稀疏且 split 不稳；
- policy replay 正确地只评估 `eval`，但旧数据没有保证 `eval` 同时保留 `KIVI` 与 `H2O` 主流风险样本；
- 因此 runtime 曲线能张开，quality 曲线却经常塌缩；
- 修复方式应落在“session trace 数据重构 + 独立 gate”，而不是改 policy replay 逻辑。

## 已确认症状

### 症状 1：profile 级质量信号存在，但 policy 级几乎消失

已确认：

- `profile_tables/pilot_session_trace_measured_profiles.csv` 中，激进 lossy profile 存在非零 `quality_loss`
- 但 `policy_tables/pilot_session_trace_measured_total_summary.csv` 中，多数 policy 的 `mean_quality_loss / p95_quality_loss / violation_rate` 近似 0

这说明问题不是“完全没有质量标签”，而是“策略真正评估到的子集没有风险”。

### 症状 2：runtime 链路有效，quality 链路无效

已确认：

- `budget_hit_rate`、`mean_global_resident_kv_mib`、`p95_ttft_ms`、`restore_count` 之间能形成闭环
- `full_lru / static_safe` 和 `static_best / utility_dynamic / uncalibrated_dynamic` 的内存与 TTFT 能拉开
- 但质量曲线没有对应展开

这说明 backend replay 与 session pressure 机制本身不是主故障点。

### 症状 3：同一 profile 在 calibration 和 eval 上风险分布不一致

已确认的典型例子：

- `h2o_heavy20_recent20`
  - calibration 平均 loss 非零
  - eval 平均 loss 为 0
- `kivi_2bit_residual32`
  - calibration 与 eval 都有风险，但 policy 实际未稳定选中它

这说明 split 与 profile 覆盖一起导致了“校准集有风险、评估集没风险”的结构性偏差。

## 根因树

### 根因 A：session trace 数据集设计问题

#### 现象

- 旧 fixture 规模过小，无法同时承载内存压力和质量压力
- 高风险样本主要集中在少数 request，且 `eval` 缺少稳定覆盖
- 同一 request 被 `pressure copy` 放大，但复制不等于新增独立尾部样本

#### 影响模块

- `data/fixtures/pilot_session_trace_requests.jsonl`
- `scripts/generate_pilot_session_trace_requests.py`
- `scripts/validate_trace_quality.py`

#### 当前判断

修复后的 `baseline_session` 不再承担 H0/H1 主证明，而是收缩为：

- 验证多 session 并发压力
- 验证 resident/global resident 演化
- 验证 budget hit / evict / restore / recompute / queue 语义
- 保留可观测但非主证据级的质量差异

#### 修复目标

把 `baseline_session` 的请求构造从“12 条手工静态 trace”改成“脚本生成、固定落盘的 QA + Summary session fixture”，并在 policy sweep 前增加独立 gate。

#### 代码修复入口

- `scripts/generate_pilot_session_trace_requests.py` 生成并落盘固定 fixture
- `data/fixtures/pilot_session_trace_requests.jsonl` 只作为生成产物，不再手工维护
- `scripts/validate_trace_quality.py` 读取 measured profile CSV 与 fixture，独立检查 gate

#### 推荐修改方向

当前固定规则：

1. 数据只保留 `QA + Summary`
2. 每个 session 固定 5 turn，其中前 `60%` 请求为 `pressure_phase=memory`，后 `40%` 为 `pressure_phase=quality`
3. `QA` follow-up 固定使用 `constraint_recall`，优先制造 `KIVI` 风险
4. `Summary` follow-up 固定使用 `detail_recall`，优先制造 `H2O` 风险
5. fixture metadata 固定包含 `split / length_bucket / risk_family / risk_profiles / pressure_phase / followup_kind`

#### 验证信号

- `eval` 上至少同时保留一组 `KIVI` 主流档位风险样本与一组 `H2O` 主流档位风险样本
- 非零样本同时覆盖 `QA` 与 `Summary`
- policy sweep 前先通过 trace quality gate，失败则直接阻断后续策略比较

### 根因 B：calibration / eval split 设计问题

#### 现象

旧 split 让高风险样本偏向 calibration，导致：

- profile summary 看起来有风险
- policy eval 却读不到风险

#### 影响模块

- session trace 请求的 `split` 字段
- `run_util/run_policies.py` 中 `split_measurements()` 后 replay 只保留 eval 的逻辑

#### 当前判断

`run_util/run_policies.py` 这里的行为本身是合理的：

- `baseline_session` 评估只取 `evaluation_measurements`

所以问题不在 replay 代码，而在 split 设计没有满足 H0/H1 的评估需要。

#### 修复目标

让 calibration 和 eval 在任务类型、长度桶、风险难度上至少粗略匹配，避免把尾部风险全部切到 calibration。

#### 代码修复入口

- 优先改数据，不优先改 `split_measurements()` 逻辑
- 只在确认数据设计无法承载时，才考虑增加“risk-stratified split”生成器

#### 推荐修改方向

1. 高风险样本固定采用 `40% calibration / 60% eval`
2. `eval` 必须同时保留 `summary` 和 `qa` 两类 lossy 风险点
3. 覆盖目标固定为 `KIVI + H2O` 主流档位，而不是所有 profile 全覆盖

#### 验证信号

- 对主选 profile，`eval` 不再是全 0 尾部
- `scripts/validate_trace_quality.py` 返回通过后才运行 policy sweep
- `policy_tables/*mem18.csv` 中 lossy policy 不再稳定全 0

### 根因 C：profile 动作空间与风险区间错位

#### 现象

当前策略常选：

- `h2o_heavy20_recent20`

但这个 profile 在当前 `eval` 子集上几乎无损，因此：

- 策略切换是真实的
- 质量代价却没被激活

#### 影响模块

- `configs/pilot_session_trace.yaml` 中 active profiles
- profile registry 与 profile sweep 设计
- `policies/` 中静态与动态 baseline 的候选空间

#### 当前判断

不是所有 profile 都没风险，而是“最常被选中的 profile 没踩到风险区”。

#### 修复目标

让 session trace 的候选空间包含：

- 一档“轻度 lossy，但大多安全”的 profile
- 一档“中度 lossy，能在 eval 上暴露 tail risk”的 profile
- 一档“激进 lossy，能稳定制造明显风险”的 profile

#### 代码修复入口

- `configs/pilot_session_trace.yaml`
- profile 注册与 profile 过滤逻辑
- `policies/base.py` 中静态策略选 profile 的统计依据

#### 推荐修改方向

1. 保留 `h2o_heavy20_recent20` 作为轻中度档
2. 强制纳入能在 eval 上稳定产生风险的 profile，如更激进的 H2O 或更激进的 KIVI
3. 检查 `static_best`、`utility_dynamic`、`uncalibrated_dynamic` 是否因为排序规则稳定绕开了高风险档

#### 验证信号

- `action_distribution` 不再过度集中到单一温和 lossy profile
- 至少一个 deployable baseline 在 eval 上出现非零 `violation_rate`
- `static_best` 与 `static_safe` 的质量尾部出现可解释差异

### 根因 D：实验职责混用

#### 现象

当前你实际上在用 `baseline_session` 路径，期待 `baseline_quality` 级别的 H0/H1 证据。

#### 影响模块

- `configs/pilot_session_trace.yaml`
- `docs/status/2026-08-15_baseline_smoke_status.md`
- 结果解读文档

#### 当前判断

这不是代码 bug，而是实验接口定义问题。

修复后，`baseline_session` 的首要职责固定为：

- 验证 session reuse
- 验证 resident/global resident 演化
- 验证 budget hit / restore / recompute / queue 语义

它可以带质量字段，但不应该独自承担主质量结论。

#### 修复目标

把两个实验轨道明确分开：

- `baseline_quality`：负责 H0/H1 主证据
- `baseline_session`：负责 backend/runtime 语义闭环

#### 代码修复入口

- 配置命名
- summary 输出字段说明
- 画图与报告脚本

#### 推荐修改方向

1. 在 summary/plot 标题里标出 `baseline_session`
2. 避免把这条 trace 的 quality 图当主图
3. 如果需要在这条线上也看质量，只把它当辅证
4. H0/H1 主质量结论交给后续独立 `baseline_quality` 配置

#### 验证信号

- 报表不会再误导为“session trace 已证明 H0/H1”
- `baseline_session` 会先执行 trace quality gate，再进入策略比较
- 代码层不会把这条配置当 baseline_quality 主入口

## 修复优先级

### P0：先修数据与 split，不先动聚合器

先做：

1. 修 session trace 请求内容
2. 重排 calibration / eval
3. 重新导出 measured profile 表

原因：

- 当前 `metrics/collector.py` 和 `run_util/run_policies.py` 的核心行为与现象是对得上的
- 如果先改 summary 逻辑，大概率只是掩盖问题

### P1：再修 profile 网格与策略候选空间

在 P0 后，如果质量曲线仍然不张开，再做：

1. 扩大激进 profile 档位
2. 检查 `static_best / utility_dynamic / uncalibrated_dynamic` 的候选排序
3. 观察 action distribution 是否仍然锁死在温和 lossy profile

### P2：最后修文档和实验接口命名

当数据与策略信号恢复后，再统一：

1. 配置职责说明
2. 输出表头说明
3. 组会/论文里的口径

## 建议的代码修复顺序

### 第一步：定位并修改 session trace 数据源

目标：

- 找到 `pilot_session_trace` 请求是静态写死还是脚本生成
- 为 `eval` 增加真正有质量压力的 request

完成后必须重跑：

- profile measured table

暂时不必重跑：

- 全部 policy sweep

### 第二步：检查 split 生成逻辑

目标：

- 确认高风险样本不会全部进 calibration
- 保证 eval 至少覆盖 2 类 task 的 lossy 风险

完成后必须重跑：

- profile measured table
- 单个参考 cell 的 policy replay，例如 `eps0p05_delta0p05_mem18`

### 第三步：检查候选 profile 设计

目标：

- 确认 eval 上真正有风险的 profile 在配置中可选
- 确认 baseline 并没有系统性绕开风险档

完成后必须重跑：

- 单个参考 cell
- 再决定是否扩到全 sweep

### 第四步：恢复 summary 与图的判读价值

目标：

- `summary_policy_quality_loss.png`
- `summary_policy_violation_rate.png`

至少要从“全平”变成“可区分”

## 每一步的最低验证标准

### 验证标准 A：profile 表层

在 `profile_tables/pilot_session_trace_measured_profiles.csv` 中检查：

- lossy profile 在 `eval` 上存在非零 `quality_loss`
- 非零值至少来自两个不同 request family
- 不同 profile 的 `eval` 风险强度有分层

### 验证标准 B：policy 单 cell 层

在单个参考 cell 中检查：

- `static_best` 不再全 0
- `utility_dynamic` 或 `uncalibrated_dynamic` 至少一者出现非零 `violation_rate`
- `static_safe` 和 `static_best` 的 tail 指标不再重合

### 验证标准 C：总表层

在 `pilot_session_trace_measured_total_summary.csv` 中检查：

- `mean_quality_loss`
- `p95_quality_loss`
- `violation_rate`

至少有一列能把三类 baseline 拉开：

- 保守 exact / safe
- 固定 lossy
- 动态 lossy

## 本轮不建议优先修改的模块

以下模块当前没有证据表明是主根因：

- `metrics/collector.py`
  - 目前聚合逻辑与输入数据一致
- `run_util/experiment_summary.py`
  - 只是把 summary 展平输出
- `backends/measured_replay.py`
  - 当前主要负责 runtime 语义回放，不负责质量切分

它们可以在后续增强，但不该作为这轮 P0 的首改对象。

## 修复完成后的预期结果

修复成功后，`pilot_session_trace` 这条线应该变成：

- runtime 图仍然成立
- quality 图不再全平
- 但质量证据仍然只是辅证，不替代 `baseline_quality`

更具体地说：

- `TTFT`、`KV memory`、backend 事件继续解释系统行为
- `quality_loss`、`violation_rate` 至少能显示“激进策略有代价，保守策略更稳”
- 然后再决定是否值得把这条 trace 扩成正式 session-aware 质量附图

## 下一步建议

如果按最小代价推进，下一轮工作应当是：

1. 找到 `pilot_session_trace` 请求源文件或生成脚本。
2. 改 `eval` 子集，让至少两个 lossy profile 在 eval 上出现稳定非零 loss。
3. 只重跑一个参考 cell 验证质量曲线是否恢复。
4. 确认恢复后再扩到全 sweep。
