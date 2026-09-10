# Baseline Smoke 实验数据构造执行计划（2026-08-12）

## 1. 目标

目标是在外部工作区 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling` 中重构两份正式实验夹具，并导入主仓库供 `2026-08-12` 之后的 baseline smoke 使用。

固定结论：

- `baseline_quality`：纯 LongBench 单轮夹具，只承担质量参照。
- `baseline_session`：ShareGPT 会话骨架 + LongBench 风险内容注入，只承担 session / backend 语义。
- 两条线共享同一套 LongBench measured 风险池。
- 样本规模按“请求数”控制，不按 session 数扩张；本轮目标是生成 `200+` 条请求。
- 主仓库不新增筛选主流程，只消费外部产物。

## 2. 正式产物与导入目标

正式产物固定为：

- `artifacts/fixtures/baseline_quality_external.jsonl`
- `artifacts/manifests/baseline_quality_external_manifest.json`
- `artifacts/candidates/hybrid_session_candidates.jsonl`
- `artifacts/manifests/hybrid_session_candidates_manifest.json`
- `artifacts/measurements/hybrid_session_candidates_profiles.csv`
- `artifacts/measurements/hybrid_session_candidates_trace.csv`
- `artifacts/fixtures/baseline_session_external.jsonl`
- `artifacts/manifests/baseline_session_external_manifest.json`

导入目标固定为主仓库：

- `data/fixtures/baseline_quality_external.jsonl`
- `data/fixtures/baseline_session_external.jsonl`

两份 JSONL 的必备字段固定遵循主仓库当前校验器：

- `baseline_quality`：`request_id/task/prompt/reference/metadata`
- `baseline_session`：`request_id/task/prompt/reference/session_id/turn_index/arrival_index/metadata`

## 3. 实施方案

### 3.1 共享 LongBench 风险池

先重建一套唯一风险池，来源固定为：

- `artifacts/candidates/longbench_candidates.jsonl`
- `artifacts/measurements/longbench_candidates_profiles.csv`

主流 profile 固定写死为：

- KIVI：`kivi_4bit_residual32`、`kivi_4bit_residual64`、`kivi_2bit_residual32`、`kivi_2bit_residual64`
- H2O：`h2o_heavy10_recent10`、`h2o_heavy15_recent15`、`h2o_heavy20_recent20`

风险判定固定只使用 `max loss` 口径：

- `kivi_sensitive`
  条件：`max_kivi_loss >= 0.05` 且 `max_kivi_loss - max_h2o_loss >= 0.02`
- `h2o_sensitive`
  条件：`max_h2o_loss >= 0.05` 且 `max_h2o_loss - max_kivi_loss >= 0.02`
- `low_risk`
  条件：所有主流 KIVI/H2O profile 的 `quality_loss <= 0.01`
- `tie_sensitive`
  条件：`max(max_kivi_loss, max_h2o_loss) >= 0.05` 且 `abs(max_kivi_loss - max_h2o_loss) < 0.02`

共享风险池每条记录固定带：

- `request_id`
- `task`
- `prompt`
- `reference`
- `source_dataset`
- `risk_family`
- `max_kivi_loss`
- `max_h2o_loss`

LongBench 文本控制固定为：

- 单条 prompt 最大约 `3000` 字符
- 超限样本直接不入池

### 3.2 任务选择与候选规模

按实验设计要求，采用“所有任务混合”方式，但任务集合固定为：

- 必选 `qa`
- 二选一固定选 `summary`
- 不纳入 `code`

也就是本轮统一使用两类任务混合：

- `qa`
- `summary`

候选扫描规模固定为“每类先 `200-300` 请求”，本轮定死为：

- `qa` 候选 `250`
- `summary` 候选 `250`

共享风险池目标输入总量固定为：

- `500` 条 LongBench 候选请求

如果某类在过滤后不足 `250`，允许补到不少于 `200`；低于 `200` 视为风险池不合格，需要回到候选生成阶段补样本。

### 3.3 `baseline_quality` 导出

`baseline_quality` 从共享风险池直接导出，不混入 ShareGPT 内容。

正式总量固定为 `216` 条请求，满足“`200+` 请求”要求。固定分布为：

- `low_risk = 120`
- `kivi_sensitive = 48`
- `h2o_sensitive = 48`

`tie_sensitive` 不允许进入正式 `baseline_quality` 夹具，只能作为候补池参与上游重筛统计。

任务分布固定为：

- `qa = 108`
- `summary = 108`

split 分布固定为：

- `calibration = 108`
- `eval = 108`

切分规则固定为分层切分：

- 先按 `task × risk_family` 分桶
- 每个桶内部按固定顺序切成 `calibration/eval`
- 目标是每个桶尽量 `1:1`
- 若某桶为奇数，则多出的 `1` 条优先放入 `eval`

如果真实样本不足以满足 `216` 条，收缩顺序固定为：

1. 先减少 `low_risk`
2. 保持 `kivi_sensitive` 与 `h2o_sensitive` 数量相等
3. 保持 `qa` 与 `summary` 数量相等
4. 最终总量最低不得低于 `200`

输出 metadata 固定为：

- `source = "external_labeling"`
- `source_dataset = "<具体 longbench 子集名>"`
- `split = calibration|eval`
- `risk_family = kivi_sensitive|h2o_sensitive|low_risk`

### 3.4 `baseline_session` 骨架与规模

ShareGPT 骨架固定来源于：

- `artifacts/candidates/sharegpt_session_candidates.jsonl`

骨架过滤规则固定为：

- `min_turns = 4`
- `max_turns = 5`
- `max_prompt_chars = 1024`
- `max_effective_prompt_chars = 5000`

正式 `baseline_session` 固定为 `216` 条请求，不按 session 数做目标。session 规模由请求数反推：

- 固定使用 `48` 个 session
- 每个 session 固定 `4` 或 `5` 个 turn
- 优先分配为 `24` 个 5-turn session 与 `24` 个 4-turn session
- 总请求数固定为 `24*5 + 24*4 = 216`

split 固定按 session 整体切分，禁止同一 session 跨 split：

- `24` 个 session 放 `calibration`
- `24` 个 session 放 `eval`

风险主导 session 分布固定为：

- `16` 个 `kivi_sensitive` 主导 session
- `16` 个 `h2o_sensitive` 主导 session
- `16` 个 `low_risk` 对照 session

`tie_sensitive` 不进入正式 session 集。

任务固定只允许：

- `qa`
- `summary`

### 3.5 `baseline_session` 注入模板

只允许两种模板，分配比例固定为 `1:1`：

- 模板 A：`24` 个 session
- 模板 C：`24` 个 session

模板 A 固定结构：

- `turn0`：保留 ShareGPT 原始开场
- `turn1`：保留 ShareGPT 原始延续
- `turn2`：注入主导风险内容
- `turn3`：对 `turn2` 做显式依赖追问
- `turn4`：对 `turn2 + turn3` 做总结或改写
- 若该 session 为 4-turn 版本，则删除 `turn4`

模板 C 固定结构：

- `turn0`：保留 ShareGPT 原始开场
- `turn1`：注入 `low_risk`
- `turn2`：注入主导风险内容
- `turn3`：要求同时引用 `turn1` 与 `turn2`
- `turn4`：比较、复述或综合前两轮约束
- 若该 session 为 4-turn 版本，则删除 `turn4`

模板分配固定为：

- `kivi_sensitive` session：`8` 个模板 A，`8` 个模板 C
- `h2o_sensitive` session：`8` 个模板 A，`8` 个模板 C
- `low_risk` session：`8` 个模板 A，`8` 个模板 C

### 3.6 `baseline_session` 风险标注与依赖规则

`baseline_session` 的 `risk_family` 固定按 session 主导风险写入，同一 session 的所有 turn 统一使用同一个 `risk_family`。

具体规则：

- `kivi_sensitive` 主导 session：该 session 全部 turn 标 `kivi_sensitive`
- `h2o_sensitive` 主导 session：该 session 全部 turn 标 `h2o_sensitive`
- `low_risk` 对照 session：该 session 全部 turn 标 `low_risk`

later-turn 显式依赖 earlier-turn 的判定标准固定为至少满足以下之一：

- prompt 中明确要求“基于上文/前面内容/之前答案”
- prompt 中要求比较前两轮信息
- prompt 中要求总结、改写或复述前一轮内容
- prompt 中引用 earlier-turn 的实体、约束或结论

不满足上述任一条件的混合 session 不允许导出。

### 3.7 `baseline_session` arrival 与交错规则

`arrival_index` 生成规则固定为全局交错编排，不按 session 整块输出。

固定编排方式：

- 将 `48` 个 session 划分为 `16` 个三元组
- 每个三元组固定包含：
  - `1` 个 `kivi_sensitive` session
  - `1` 个 `h2o_sensitive` session
  - `1` 个 `low_risk` session
- 在每个三元组内按轮次交错输出
- 所有 5-turn session 先完成 `turn0-3` 交错，含 `turn4` 的三元组最后再补第五轮
- 所有 4-turn session 不生成 `turn4`

这样固定保证：

- `arrival_index` 全局严格递增
- 至少两个可交错 session 存在
- backend 压力能在 session trace 中读出来

输出 metadata 固定补充：

- `source = "hybrid_session_builder"`
- `source_dataset = "sharegpt_longbench_hybrid_session"`
- `source_session_dataset = "sharegpt_longbench_multiturn_pilot"`
- `content_source_dataset = "longbench"`
- `content_source_request_id`
- `original_session_id`
- `injection_template`
- `split`
- `risk_family`

## 4. 测试与验收

### 4.1 风险池验证

- 统计四类风险池样本数
- 复核 `max_kivi_loss/max_h2o_loss` 与标签是否一致
- 确认 `qa/summary` 原始覆盖存在
- 确认两类任务候选数均不少于 `200`

### 4.2 `baseline_quality` 校验

运行：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_quality \
  --input /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl \
  --validate-only
```

验收固定为：

- `row_count = 216`
- `tasks = {qa, summary}`
- `splits = {calibration, eval}`
- `risk_families = {kivi_sensitive, h2o_sensitive, low_risk}`

说明：
主仓库当前校验器允许 `code` 存在，但不要求必须有 `code`；本轮按实验设计只使用 `qa + summary`。

### 4.3 `baseline_session` 校验

运行：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_session \
  --input /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_session_external.jsonl \
  --validate-only
```

验收固定为：

- `row_count = 216`
- `session_count = 48`
- `multi_turn_session_count = 48`
- `interleaved_session_count >= 2`
- `splits = {calibration, eval}`
- `risk_families = {kivi_sensitive, h2o_sensitive, low_risk}`

### 4.4 Smoke 验收

运行主仓库 external smoke：

- `configs/pilot_external_baseline_quality.yaml`
- `configs/pilot_external_baseline_session.yaml`

最低通过标准固定为：

- `baseline_quality`
  - 至少一个 cell 中 `static_safe` 与 `static_best` 的 `p95_quality_loss` 不完全相等
  - 至少一个 cell 中 `utility_dynamic.mean_kv_cache_memory_mib < full_lru.mean_kv_cache_memory_mib`
  - 至少一个 cell 中五条 baseline 不全部走 exact 单一路径
- `baseline_session`
  - 至少一个 cell 中出现非零 `budget_hit_rate`
  - 至少一个 cell 中 `restore_count`、`recompute_count`、`queue_delay_ms` 三者之一非零
  - session trace 中存在 `resident_kv_mib_before/after` 与 `global_resident_kv_mib`
  - 至少一个 cell 中 `kivi` 与 `h2o` 主流档位分别各自表现出一侧可见质量信号

若任一条件不满足，回退到外部工作区重新筛选风险池或重排 session，不接受当前夹具为正式版本。

## 5. 假设

- 当前日期为 `2026-08-12`。
- 已存在的 `baseline_quality_external.jsonl` 和纯 ShareGPT session 结果只作参考，不直接复用。
- 主仓库当前 `baseline_session` 校验器不限制 `task` 枚举，因此 session 线保留 `qa/summary` 即可。
- 本轮严格按“`200+` 请求”设计，不把“`200+` session”作为目标。

## 6. 实现日志

### 2026-08-12

已实现：

- 已新建本执行文档，完整写入 Baseline Smoke 实验数据构造方案。
- 已确定后续统一在本文档末尾追加“已实现-未实现”日志，而不是分散记录到其他 TODO 文档。

未实现：

- 尚未在外部工作区重建共享 LongBench 风险池。
- 尚未导出新的 `baseline_quality_external.jsonl` 与 `baseline_session_external.jsonl`。
- 尚未执行 `scripts/import_external_fixtures.py --validate-only` 校验。
- 尚未运行 `configs/pilot_external_baseline_quality.yaml` 与 `configs/pilot_external_baseline_session.yaml` 的 smoke 验收。

### 2026-08-12（执行检查）

已实现：

- 已创建隔离 worktree：`.worktrees/baseline-smoke-2026-08-12`，避免直接在 `main` 分支上执行实现。
- 已核对主仓库与外部工作区现状，确认主仓库已有 external fixture 校验脚本与 smoke 配置草稿，外部工作区已有 LongBench / ShareGPT 候选与 LongBench measurement 产物。
- 已按正式方案口径对 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/longbench_candidates.jsonl` 与 `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/longbench_candidates_profiles.csv` 做风险池可行性盘点。
- 已确认当前 `qa/summary` 且 `prompt <= 3000` 的 LongBench measured 风险池统计结果为：
  - `low_risk = 366`
  - `kivi_sensitive = 5`
  - `h2o_sensitive = 4`
  - `tie_sensitive = 14`
- 已确认任务覆盖结果为：
  - `qa = 189`
  - `summary = 200`
- 已确认按 `task × risk_family` 分布时：
  - `qa × kivi_sensitive = 5`
  - `qa × h2o_sensitive = 4`
  - `qa × low_risk = 166`
  - `qa × tie_sensitive = 14`
  - `summary × low_risk = 200`

未实现：

- 尚未开始新脚本实现，因为当前共享 LongBench 风险池真实分布与固定配额存在硬冲突。
- 尚未导出 `216` 条正式 `baseline_quality`，因为当前严格口径下 `kivi_sensitive` 与 `h2o_sensitive` 样本远低于计划要求的 `48 + 48`。
- 尚未构造 `48` 个正式 `baseline_session`，因为 session 主导风险内容依赖同一套 LongBench 风险池，而上游高风险样本不足。
- 尚未进入 `validate-only` 与 smoke 验收阶段。

阻塞结论：

- 按 `2026-08-12` 当前外部工作区实测数据，现有 LongBench measured 风险池不足以支撑文档中固定的正式导出规模与分布。
- 若不补候选、补测量，或放宽固定配额 / 风险口径，本轮 plan 不能继续落地到正式夹具生成步骤。

### 2026-08-12（后台测量重启）

已实现：

- 已停止前台 LongBench measurement 进程，停止的进程为：
  - 外层 `conda run`：`PID 57874`
  - 内层 `python -m run_util.build_profile_table`：`PID 57910`
- 已按 `setsid + nohup` 方式后台重启 LongBench measurement，避免继续占用交互会话。
- 已写入后台 PID 文件：
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/tmp/2026-08-12-longbench-measurement.pid`
- 已写入后台日志文件：
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/tmp/2026-08-12-longbench-measurement.log`
- 已确认后台进程状态满足脱离会话运行：
  - `PID 24783`
  - `PPID = 1`
  - `PGID = SID = 24783`

未实现：

- 后台重启后的 LongBench measurement 仍在运行，尚未完成全部 profiles 的测量。
- 尚未基于后台重启后的完整 measurement 重算共享风险池分布。
- 尚未导出正式 `baseline_quality_external.jsonl` 与 `baseline_session_external.jsonl`。
- 尚未进入 `validate-only` 与 external smoke 验收。

### 2026-08-13（放宽配额并启动 smoke）

已实现：

- 已按“优先产出一版 smoke 表”的要求放宽外部夹具构造口径，不再坚持原方案中的高风险固定配额。
- 已重新导出外部质量夹具：
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl`
  - 行数：`84`
  - 风险组：`kivi_sensitive / h2o_sensitive / low_risk`
- 已重新导出外部 session 夹具：
  - `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_session_external.jsonl`
  - 行数：`164`
  - session 数：`36`
  - 风险组：当前为 `low_risk`
  - `interleaved_session_count = 36`
- 已将两份外部夹具导入主仓库：
  - `data/fixtures/baseline_quality_external.jsonl`
  - `data/fixtures/baseline_session_external.jsonl`
- 已完成两份夹具的 `validate-only` 校验：
  - `baseline_quality`：通过
  - `baseline_session`：通过
- 已启动 quality 线 external smoke：
  - 配置：`configs/pilot_external_baseline_quality.yaml`
  - 启动时间：`2026-08-13 09:10:26`
  - 日志：`out/logs/pilot_external_baseline_quality_20260813_091026.nohup.log`
- 已启动 session 线 external smoke：
  - 配置：`configs/pilot_external_baseline_session.yaml`
  - 启动时间：`2026-08-13 09:10:36`
  - 日志：`out/logs/pilot_external_baseline_session_20260813_091036.nohup.log`

未实现：

- 两条 external smoke 仍在后台运行，尚未读取最终 `summary` / `policy` 表结果。
- 尚未确认 quality 线是否已经表现出可见的 profile / policy 差异。
- 尚未确认 session 线在 low-risk-only 夹具上是否还能产生足够的 backend 压力信号。
