# 2026-08-12 baseline smoke 执行记录

## 目标

尽快为 `2026-08-01` 到 `2026-08-15` 这一阶段补出可用的 `baseline_quality` 和 `baseline_session` smoke 表，同时避免把数据准备和标注逻辑塞进当前仓库。

这次执行分成两条线：

- `baseline_quality`：使用外部标注后的 LongBench 子集，覆盖 `QA / Summary / Code`
- `baseline_session`：使用外部标注后的 ShareGPT 多轮会话子集

仓库内只负责三件事：

- 接收外部生成的夹具
- 校验夹具是否符合当前 loader 和 smoke 流程要求
- 用现有 `pilot-smoke-measured` 链路消费这些夹具

## 本次执行边界

不在当前仓库中新增 LongBench 或 ShareGPT 的筛选、打标、扫描主流程代码。外部标注工作区可以单独创建，但与主仓库解耦。

如果出现 schema 不兼容，优先改外部导出格式或增加仓库侧导入转换，不直接改主实验语义。

## 执行计划

### 1. 记录计划并建立持续更新机制

先把已确认方案写进文档。之后每完成一个完整任务，都在文档末尾追加一条更新日志，记录时间、动作和产物。

### 2. 补仓库侧导入与校验能力

为两类外部夹具增加统一入口：

- `baseline_quality` JSONL 夹具
- `baseline_session` 会话夹具

需要覆盖的校验点：

- `baseline_quality`
  - 必填字段：`request_id`、`task`、`prompt`、`reference`、`metadata`
  - `metadata` 至少包含 `source`、`source_dataset`、`split`
  - `task` 允许覆盖 `qa`、`summary`、`code`
  - `split` 至少覆盖 `calibration` 和 `eval`
  - 三个风险组都要出现：`kivi_sensitive`、`h2o_sensitive`、`low_risk`

- `baseline_session`
  - 允许输入 ShareGPT 风格 `json/jsonl`
  - 会话必须带连续 turn
  - 导出后要保留 `session_id`、`turn_index`、`arrival_index`
  - 至少存在一个多轮会话和两个可交错会话
  - 三个风险组都要出现：`kivi_sensitive`、`h2o_sensitive`、`low_risk`

### 3. 建立外部工作区落地方式

在仓库外创建单独工作区，用来放标注脚本、运行记录和中间产物。当前仓库只保留消费入口和使用说明。

### 4. 形成可执行命令

补齐以下命令的说明和脚本入口：

- 外部夹具校验
- 外部夹具导入到仓库固定位置
- `baseline_quality` smoke 运行
- `baseline_session` smoke 运行

### 5. 验证

至少完成：

- 单元测试
- 脚本级命令自检
- 文档记录回写

如果环境允许，再做小规模 smoke 预检；若当前回合内不适合跑全量模型实验，需要明确写明阻塞点。

## 更新日志规则

每完成一个完整任务，在文档末尾追加：

- 时间
- 完成的任务
- 主要产物
- 还剩什么

## 更新日志

### 2026-08-12 任务 1 完成

已新增这份执行记录文档，固定了本轮边界、实施顺序和更新日志规则。

当前下一步：先写测试，明确外部夹具导入和校验的仓库接口，再补脚本实现。

### 2026-08-12 任务 2 完成

已新增 `tests/test_external_fixture_import.py`，先用失败测试把外部夹具导入接口收紧。

这组测试覆盖了两类场景：

- `baseline_quality` 的必填字段、元数据和风险组覆盖
- `baseline_session` 的会话连续性、交错性和风险组覆盖

当前下一步：补仓库侧导入脚本，让这些测试转绿。

### 2026-08-12 任务 3 完成

已新增 `scripts/import_external_fixtures.py`，提供两个能力：

- 校验外部 `baseline_quality` / `baseline_session` 夹具
- 在校验通过后导入到仓库固定位置

同时更新了 `README.md`，补充了脚本用法、默认导入位置和当前校验规则。

当前下一步：跑测试和命令级自检，确认这批实现可以直接使用。

### 2026-08-12 任务 4 完成

已完成两类验证：

- 针对外部夹具导入的新增测试
- 现有 pilot 数据集相关测试回归

验证结果：

- `pytest tests/test_external_fixture_import.py tests/test_pilot_dataset.py tests/test_pilot_session_trace_dataset.py -q`
- `25 passed`

命令入口也已自检通过：

- `python scripts/import_external_fixtures.py --help`

当前状态：仓库已经具备外部夹具的校验和导入入口，可以开始在仓库外准备 LongBench / ShareGPT 标注工作区，并把产物接回当前仓库。

### 2026-08-12 任务 5 完成

已补主仓库对“外部 baseline 夹具”的正式接入入口。

新增内容：

- `configs/pilot_external_baseline_quality.yaml`
- `configs/pilot_external_baseline_session.yaml`
- `scripts/run_external_baseline_smoke.sh`
- `metrics/quality.py` 增加 `code` 任务主指标支持

这一步完成后，外部 LongBench `QA / Summary / Code` 夹具可以直接走主仓库 baseline quality 轨道，不再被 `task=code` 拦住。

### 2026-08-12 任务 6 完成

已在仓库外创建独立标注工作区：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling`

工作区已落地：

- 候选生成脚本
- LongBench / ShareGPT 标注聚合脚本
- 独立配置
- 一键运行脚本
- 导入主仓库脚本
- README 使用说明

实际候选生成结果：

- LongBench 候选 `600` 条
- ShareGPT 候选 `48` 个 session，`232` 个 turn

同时根据真实数据量调整了默认扫描规模：

- LongBench 每类从 `240` 调整为 `200`
- ShareGPT session 从 `72` 调整为 `48`

当前下一步：运行 measured profile 标注，生成正式 `baseline_quality_external.jsonl` 和 `baseline_session_external.jsonl`，然后导入主仓库并跑 smoke。

### 2026-08-12 12:41

LongBench 第一次真测量中断。问题不在脚本，而在显存。

当时 `full_gpu` 跑到 `qa_qasper_00018` 时触发 OOM。检查候选后发现，LongBench 三个任务的 prompt 几乎都被截在 `7000` 字符附近，超出了这台机器上当前配置能稳定承受的范围。

处理动作：

- 把 LongBench 候选默认截断从 `7000` 改到 `3000`
- 先跑 `20` 条真测量预检，再决定是否恢复全量

### 2026-08-12 13:18

LongBench `20` 条真测量预检通过。

结论很直接：把 prompt 上限收紧到 `3000` 之后，`full_gpu`、KIVI、H2O 三组 profile 都能跑完，不再在前几个 chunk 直接爆显存。

处理动作：

- 按同样参数重启 LongBench 全量测量

### 2026-08-12 14:46

LongBench 全量测量跑完，测量表完整落地。

检查结果：

- profile 表共 `4800` 行
- `8` 个 profile 各 `600` 行
- 全部 `measured=true`
- `quality_loss` 字段完整

这一步过后，问题从“跑不完”变成了“导不出来”。

### 2026-08-12 15:06

LongBench 导出阶段第一次失败。

失败原因不是测量表坏了，而是导出策略里写死了 `60 / 60 / 60` 三个风险组目标。实际统计结果不是这个分布：

- `low_risk = 542`
- `kivi_sensitive = 11`
- `h2o_sensitive = 11`
- 另有 `36` 条样本两家族损失几乎打平，原先被丢掉

处理动作：

- 先做真实统计，再改导出策略
- 把 `tie_sensitive` 样本纳入回填池
- `low_risk` 保留 `60`
- 敏感组按真实可用数量收缩，不再硬卡 `60`
- 低风险采样时保住 `summary` 覆盖

导出结果：

- `artifacts/fixtures/baseline_quality_external.jsonl`
- 总数 `100`
- 风险分布：`kivi_sensitive=20`、`h2o_sensitive=20`、`low_risk=60`
- split 分布：`calibration=50`、`eval=50`

### 2026-08-12 15:07

ShareGPT 第一次全量真测量失败。

最开始以为是单条 prompt 太长。回头查失败样本才看清楚，真正的问题是多轮 history 累加后的 `effective prompt` 太长。

当时看到的几个失败点：

- 单条 longbench prompt 已被截到约 `2048`
- 但累加 history 后，`effective_len` 仍然冲到 `4608`、`6839`、`9021`
- `full_gpu` 因此再次 OOM

处理动作：

- 先加 ShareGPT 候选截断测试
- 再加按 `effective prompt` 过滤 session 的测试
- 先后确认两层逻辑都按预期生效

### 2026-08-12 15:21

ShareGPT 第二轮候选收紧完成，策略改成优先保住能稳定跑完的 session。

新的默认策略：

- `max_prompt_chars = 1024`
- `max_turns = 5`
- `max_effective_prompt_chars = 5000`
- 不再盲取前 `48` 个 session，而是先筛掉超预算 session，再按最坏 `effective prompt` 从短到长取前 `36` 个

新的候选结果：

- `36` 个 session
- `164` 个 turn
- 最坏 `effective prompt = 4751`

### 2026-08-12 15:34

ShareGPT `20` 条真测量预检通过。

这一步说明新的 session 过滤策略是有效的：在当前机器和当前 profile 组合下，`full_gpu` 不再在前几个 chunk 因为会话累积 prompt 直接 OOM。

处理动作：

- 立刻重启 ShareGPT 全量真测量

### 2026-08-12 15:47

截至当前分钟，ShareGPT 全量真测量仍在运行。

当前观察到的状态：

- 进程仍在
- 还没有新的 OOM 回报
- LongBench 这条线已经有正式外部 quality 夹具
- ShareGPT 这条线还在等全量测量和 session 夹具导出完成

当前下一步：等 ShareGPT 全量测量结束，检查 `baseline_session_external.jsonl` 是否生成，再决定是否导入主仓库。

### 2026-08-12 15:57

继续检查 ShareGPT 全量真测量状态，并据此准备改 README 和后续导出策略。

这一步确认了三件事：

- 当前工作目录不是独立 worktree，而是主 checkout，本轮先继续原地推进，不额外切工作区
- ShareGPT 测量进程还在，`run_sharegpt_labeling.sh` 和 `run_util.build_profile_table` 都没有退出
- 测量文件已经写到 `696` 行，说明流程不是直接崩掉，而是在 `kivi_2bit_residual64` 这一档推进较慢

同步看到的产物状态：

- LongBench 外部 quality 夹具已经稳定落地，统计仍是 `100` 条、`20 / 20 / 60`
- ShareGPT 还没有导出 `baseline_session_external.jsonl`
- ShareGPT 当前测量表里已经有 `696` 行，较上一轮检查继续增长，但最近一个 `5` 秒窗口内没有继续写入，速度明显变慢

当前下一步：

- 继续盯 ShareGPT 全量测量是否结束
- 如果测量正常结束，立刻检查 session 风险分布和导出结果
- 如果长时间停在同一 profile，再按这次检查结果继续排障
