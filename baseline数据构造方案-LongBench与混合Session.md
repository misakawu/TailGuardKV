# baseline 数据构造方案：LongBench 与混合 Session

## 1. 目标

这份文档定义两条外部 baseline 夹具的构造方式：

- `baseline_quality`：继续使用纯 LongBench，不混入 ShareGPT。
- `baseline_session`：使用混合方案，用 ShareGPT 提供会话骨架，用 LongBench 高风险样本补强 KIVI / H2O 敏感性。

这份文档的用途不是直接改主仓库代码，而是给后续独立会话一个可执行的数据构造口径。后续如果启动新的数据构造任务，应以这份文档为准。

## 2. 职责边界

### 2.1 `baseline_quality`

`baseline_quality` 是质量参照表，目标是稳定拉开：

- `full_lru`
- `static_best`
- `static_safe`
- `utility_dynamic`
- `uncalibrated_dynamic`

它主要回答：

- 不同 profile / policy 的质量损失是否有可解释差异
- 内存收益和质量代价是否能一起读出来
- `QA / Summary / Code` 三类任务是否都被覆盖

它不负责承诺 session/backend 语义，因此不要求：

- `restore`
- `recompute`
- `queue`
- `global_resident`

### 2.2 `baseline_session`

`baseline_session` 是 backend 语义表，目标是验证：

- session 复用
- history 累积
- `resident_kv_mib_before/after`
- `global_resident_kv_mib`
- `budget_hit`
- `restore/recompute/evict/queue`

它需要的是“会话结构 + 足够的 profile 风险信号”，不是必须保持纯 ShareGPT 数据分布。

因此这里允许使用混合数据集，只要满足：

- 会话结构真实可回放
- `session_id / turn_index / arrival_index` 完整
- history 累积逻辑不被破坏
- 风险标签和质量信号有明确来源

## 3. 为什么采用这套分工

## 3.1 为什么 `baseline_quality` 继续用 LongBench

原因很直接：

- LongBench 自带参考答案，适合直接计算 `quality_loss`
- `QA / Summary / Code` 三类任务覆盖清楚
- 单轮样本更容易解释 profile 差异
- 不会把 session/history 变量混进质量参照

当前外部工作区已经有成熟链路：

- 候选集生成
- measured profile 测量
- 风险聚合
- `baseline_quality_external.jsonl` 导出

因此这条线不建议再混入 ShareGPT。

## 3.2 为什么原始 ShareGPT 不足以单独承担 `baseline_session`

到 `2026-08-12` 为止，现有结果已经暴露出问题：

- ShareGPT 会话结构是合格的
- history 累积也确实发生了
- 但当前筛出的 ShareGPT 样本对 KIVI 不够敏感

这意味着：

- 它适合作为 session 骨架
- 但不适合作为唯一的风险信号来源

如果继续纯 ShareGPT 路线，`baseline_session` 很容易再次卡在：

- KIVI 主流档位 `mean_quality_loss` 太低
- `quality gate` 过不去
- policy 曲线拉不开

## 3.3 为什么推荐“ShareGPT 骨架 + LongBench 风险内容”的混合方案

这个方案同时保住两件事：

- ShareGPT 提供真实多轮结构、turn 顺序、交错和 history 累积
- LongBench 提供更强、更可控的 KIVI / H2O 风险样本

因此它最适合 `baseline_session`：

- 仍然是 session-aware 数据
- 但不再把风险可分性完全赌在 ShareGPT 原始语料上

## 4. 源数据与工作区位置

## 4.1 主仓库

- 主仓库路径：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV`

主仓库中与导入、校验和实验消费有关的关键文件：

- `scripts/import_external_fixtures.py`
- `configs/pilot_external_baseline_quality.yaml`
- `configs/pilot_external_baseline_session.yaml`
- `scripts/run_external_baseline_smoke.sh`

## 4.2 外部标注工作区

- 外部工作区路径：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling`

关键配置：

- `configs/longbench_labeling.yaml`
- `configs/sharegpt_labeling.yaml`

关键脚本：

- `scripts/prepare_longbench_candidates.py`
- `scripts/prepare_sharegpt_session_candidates.py`
- `scripts/label_longbench_quality.py`
- `scripts/label_sharegpt_sessions.py`

当前已有产物：

- LongBench 候选：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/longbench_candidates.jsonl`
- ShareGPT 候选：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/sharegpt_session_candidates.jsonl`
- LongBench 测量表：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/longbench_candidates_profiles.csv`
- ShareGPT 测量表：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/sharegpt_session_candidates_profiles.csv`
- ShareGPT trace：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/sharegpt_session_candidates_trace.csv`
- 已导出 quality 夹具：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl`

## 4.3 原始数据源

### LongBench

- 原始压缩包：`/DATACENTER3/zhenxiang.wang/resource/LongBench/data.zip`

### ShareGPT 多轮池

- 原始多轮池：`/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/requests/sharegpt_longbench_multiturn_pilot.jsonl`

### 模型与 tokenizer

- 模型路径：`/DATACENTER3/zhenxiang.wang/resource/Qwen2.5-7B-Instruct`
- HuggingFace cache：`/DATACENTER3/zhenxiang.wang/resource/huggingface`

## 5. `baseline_quality` 的构造方式

## 5.1 输入

使用纯 LongBench 路线：

- 原始数据：`/DATACENTER3/zhenxiang.wang/resource/LongBench/data.zip`
- 候选文件：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/longbench_candidates.jsonl`
- 测量配置：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/configs/longbench_labeling.yaml`

## 5.2 当前构造口径

按当前外部工作区 README 和配置，`baseline_quality` 应满足：

- 任务覆盖：`qa / summary / code`
- 输出格式：JSONL
- 每条记录带：
  - `request_id`
  - `task`
  - `prompt`
  - `reference`
  - `metadata`
- `metadata` 至少带：
  - `source`
  - `source_dataset`
  - `split`
  - `risk_family`

## 5.3 现有筛选与测量约束

当前已知约束：

- LongBench 候选按每类约 `200` 条扫描
- 为避免 `full_gpu` OOM，默认 prompt 截断已收紧到约 `3000` 字符
- 实测 profile 使用主仓库 measured smoke 链路完成

## 5.4 风险分组

`baseline_quality` 继续维持三类风险标签：

- `kivi_sensitive`
- `h2o_sensitive`
- `low_risk`

当前已确认的正式导出结果是：

- 文件：`/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl`
- 总数：`100`
- 风险分布：`20 / 20 / 60`
- split 分布：`50 / 50`

## 5.5 输出文件

正式输出：

- `baseline_quality_external.jsonl`
- 可选 manifest：`baseline_quality_external_manifest.json`

建议固定路径：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/manifests/baseline_quality_external_manifest.json`

## 6. `baseline_session` 的混合构造方式

## 6.1 总体思路

`baseline_session` 不再要求纯 ShareGPT。

推荐构造方式：

1. 从 ShareGPT 多轮池中筛出“结构合格”的 session，作为会话骨架。
2. 从 LongBench measured 候选中筛出“风险足够强”的单轮内容，作为内容注入池。
3. 把 LongBench 内容注入 ShareGPT session 的后续 turn，形成混合 session。
4. 重新跑 measured profile。
5. 按 session 级质量表现和 backend 语义导出 `baseline_session_external.jsonl`。

## 6.2 为什么只注入后续 turn

不建议替换所有 turn。

推荐策略：

- `turn0` 尽量保留 ShareGPT 原始开场
- 从 `turn1` 或 `turn2` 开始注入 LongBench 风险内容

原因：

- 保住 session 开场自然性
- 保住第一轮 context 建立逻辑
- 将风险集中到 history 已累积的 turn，更容易暴露 KIVI / H2O 差异

## 6.3 混合 session 的输入来源

### 会话骨架来源

使用：

- `/DATACENTER3/zhenxiang.wang/resource/tailguardkv_pilot/requests/sharegpt_longbench_multiturn_pilot.jsonl`

当前工作区已有筛选结果：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/sharegpt_session_candidates.jsonl`

当前骨架限制：

- `36` 个 session
- `164` 个 turn
- `max_turns = 5`
- `min_turns = 4`
- `max_prompt_chars = 1024`
- `max_effective_prompt_chars = 5000`

这些约束来自：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/configs/sharegpt_labeling.yaml`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/manifests/baseline_session_external_manifest.json`

### 风险内容来源

使用 LongBench measured 候选：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/longbench_candidates.jsonl`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/longbench_candidates_profiles.csv`

风险内容只从已经完成 measured profile 的 LongBench 候选中选，不从未测量原始样本直接选。

## 6.4 LongBench 注入样本的筛选规则

LongBench 注入池建议分三类：

- `kivi_sensitive`
- `h2o_sensitive`
- `low_risk`

推荐筛选规则：

### KIVI 敏感样本

满足以下任一条件：

- `max(KIVI mainstream quality_loss) >= 0.05`
- 且 `max(KIVI mainstream quality_loss) - max(H2O mainstream quality_loss) >= 0.02`

### H2O 敏感样本

满足以下任一条件：

- `max(H2O mainstream quality_loss) >= 0.05`
- 且 `max(H2O mainstream quality_loss) - max(KIVI mainstream quality_loss) >= 0.02`

### 低风险样本

满足：

- 所有主流 KIVI / H2O profile 的 `quality_loss <= 0.01`

如果一条样本同时对两家族都高，但差值不足：

- 先标成 `tie_sensitive`
- 不直接丢弃
- 可在 session 混合阶段作为补位池使用

## 6.5 Session 注入模板

每个 ShareGPT session 不建议全部 turn 都注入 LongBench 内容。

推荐三种模板：

### 模板 A：单风险注入

- `turn0`：保留 ShareGPT 原始 turn
- `turn1`：保留 ShareGPT 原始 turn
- `turn2`：注入 `kivi_sensitive` 或 `h2o_sensitive`
- `turn3`：基于注入内容提出追问
- `turn4`：总结或改写追问

适合构造清晰的单家族风险信号。

### 模板 B：双阶段注入

- `turn0`：原始 ShareGPT
- `turn1`：原始 ShareGPT
- `turn2`：注入 `low_risk`
- `turn3`：注入 `kivi_sensitive` 或 `h2o_sensitive`
- `turn4`：对 `turn3` 内容做依赖历史的追问

适合让风险在后半段才真正暴露。

### 模板 C：对照注入

- `turn0`：原始 ShareGPT
- `turn1`：注入 `low_risk`
- `turn2`：注入 `sensitive`
- `turn3`：要求同时引用前两轮信息
- `turn4`：复述或比较前序约束

适合把“history 依赖”和“lossy 风险”绑在一起。

推荐优先使用模板 A 和模板 C。

## 6.6 注入时必须保持的结构字段

混合后每条 request 必须继续保留：

- `request_id`
- `task`
- `prompt`
- `reference`
- `session_id`
- `turn_index`
- `arrival_index`
- `metadata`

其中 `metadata` 建议至少补充：

- `source = "hybrid_session_builder"`
- `source_dataset = "sharegpt_longbench_hybrid_session"`
- `split`
- `risk_family`
- `source_session_dataset = "sharegpt_longbench_multiturn_pilot"`
- `content_source_dataset = "longbench"`
- `content_source_request_id`
- `injection_template`
- `original_session_id`

## 6.7 history 与 arrival 规则

混合后必须满足：

- 同一 `session_id` 内 `turn_index` 从 `0` 连续递增
- 全局 `arrival_index` 严格递增且不重复
- 至少两个 session 可交错
- 至少一个多轮 session
- 后续 turn 的语义依赖前序 turn，而不是彼此独立

不要把 session 变成“只是把单轮样本塞进同一个 session_id”。

## 6.8 建议的数量配比

建议目标不是先追求大，而是先追求可解释。

第一版混合 `baseline_session` 建议：

- session 数：`24` 到 `36`
- 每个 session turn 数：`4` 到 `5`
- eval 中至少保留：
  - `8` 个以 `kivi_sensitive` 为主的 session
  - `8` 个以 `h2o_sensitive` 为主的 session
  - `8` 个 `low_risk` 或 `tie_sensitive` 对照 session

如果总量受限，优先保证：

- eval 的风险可分性
- 多 session 交错
- 历史依赖

而不是优先保大样本数。

## 6.9 任务类型建议

混合 `baseline_session` 不建议继续追求 `code` 覆盖。

建议只保留：

- `qa`
- `summary`

原因：

- 当前 `baseline_session` 质量门本来就只要求 `QA + Summary`
- `code` 注入到多轮 session 后更容易带来格式噪声
- 先把 KIVI / H2O 风险和 backend 语义站稳更重要

## 7. 输出文件与目录约定

## 7.1 混合候选中间文件

建议新增中间文件：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/candidates/hybrid_session_candidates.jsonl`

建议新增 manifest：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/manifests/hybrid_session_candidates_manifest.json`

## 7.2 混合 measured profile 表

建议输出：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_profiles.csv`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/measurements/hybrid_session_candidates_trace.csv`

## 7.3 正式 external 夹具

正式输出：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_session_external.jsonl`
- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/manifests/baseline_session_external_manifest.json`

## 8. 推荐执行顺序

## 8.1 `baseline_quality`

1. 使用 LongBench 原始数据生成候选
2. 跑 measured profile
3. 做风险分组
4. 导出 `baseline_quality_external.jsonl`
5. 用主仓库导入校验
6. 跑 external baseline quality smoke

## 8.2 `baseline_session`

1. 生成或复用 ShareGPT session 骨架
2. 从 LongBench measured 表生成注入池
3. 选择注入模板并构造混合 session
4. 导出 `hybrid_session_candidates.jsonl`
5. 跑 measured profile
6. 检查 quality gate 是否已有 KIVI / H2O 双侧信号
7. 导出 `baseline_session_external.jsonl`
8. 用主仓库导入校验
9. 跑 external baseline session smoke

## 9. 主仓库导入与校验

导入脚本：

- `/DATACENTER3/zhenxiang.wang/work/TailGuardKV/scripts/import_external_fixtures.py`

只校验 `baseline_quality`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_quality \
  --input /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_quality_external.jsonl \
  --validate-only
```

只校验 `baseline_session`：

```bash
python scripts/import_external_fixtures.py \
  --kind baseline_session \
  --input /DATACENTER3/zhenxiang.wang/work/TailGuardKV-labeling/artifacts/fixtures/baseline_session_external.jsonl \
  --validate-only
```

导入默认目标：

- `data/fixtures/baseline_quality_external.jsonl`
- `data/fixtures/baseline_session_external.jsonl`

## 10. 验收标准

## 10.1 `baseline_quality`

至少满足：

- `qa / summary / code` 都有覆盖
- `calibration / eval` 都存在
- `kivi_sensitive / h2o_sensitive / low_risk` 都存在
- 五条 baseline 有可解释分化

## 10.2 `baseline_session`

至少满足：

- `session_id / turn_index / arrival_index` 合法
- 至少一个多轮 session
- 至少两个可交错 session
- measured profile 中有完整 history 相关字段
- KIVI 与 H2O 至少各有一侧主流档位在 eval 上形成可见质量信号
- 后续 smoke 能读出 backend 事件链

## 11. 风险点与回退方案

### 风险 1：混合后 prompt 过长再次 OOM

处理：

- 继续保留 `max_prompt_chars`
- 继续保留 `max_effective_prompt_chars`
- 优先压缩注入后的 later-turn 文本

### 风险 2：LongBench 注入后 session 看起来像伪多轮

处理：

- 保留 ShareGPT 开场 turn
- 让后续追问显式依赖前序信息
- 不要把每一轮都改成独立 QA

### 风险 3：KIVI 仍然不敏感

处理：

- 提高 LongBench 注入池中 `kivi_sensitive` 的占比
- 优先把高风险样本放到 history 已累积的后续 turn
- 必要时降低 `low_risk` session 占比

### 风险 4：解释口径混乱

处理：

- 在 manifest 中明确写明这是混合 session 数据集
- 不再把它描述成“纯 ShareGPT”
- `baseline_quality` 保持纯 LongBench，避免两条线一起混

## 12. 结论

后续数据构造任务应采用以下固定口径：

- `baseline_quality`：纯 LongBench，不混。
- `baseline_session`：使用 ShareGPT 会话骨架 + LongBench 风险内容注入的混合方案。

这样做的目的不是追求“数据集名字统一”，而是让两张 baseline 表各自完成自己的职责：

- `baseline_quality` 负责质量参照
- `baseline_session` 负责 session/backend 语义，并且真的能拉开 KIVI / H2O 风险
